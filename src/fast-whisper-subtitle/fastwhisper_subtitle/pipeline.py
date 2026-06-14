from __future__ import annotations

import contextlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

from fastwhisper_subtitle.config import DEFAULT_CONFIG
from fastwhisper_subtitle.services.audio import (
    cut_audio_file,
    cut_audio_segment_memory,
    extract_audio,
    read_mono_audio,
)
from fastwhisper_subtitle.services.srt import seconds_to_srt_time, segments_to_srt
from fastwhisper_subtitle.storage.cache import (
    check_segment_completed,
    load_existing_result,
    load_segments_cache,
    save_segments_cache,
)


T = TypeVar("T")
QUALITY_PROFILES_FILE = Path(__file__).resolve().parent / "data" / "quality_profiles.json"
HALLUCINATION_PHRASES_FILE = Path(__file__).resolve().parent / "data" / "hallucination_phrases.json"
_QUALITY_PROFILES_CONFIG: dict[str, Any] | None = None
_HALLUCINATION_PHRASES_CONFIG: dict[str, Any] | None = None


@dataclass
class TranscriptionConfig:
    input_file: str
    project_id: str = ""
    language: str | None = None
    output_languages: list[str] | None = None
    model: str = DEFAULT_CONFIG["model"]
    task: str = DEFAULT_CONFIG["task"]
    beam_size: int = DEFAULT_CONFIG["beam_size"]
    compute_type: str = DEFAULT_CONFIG["compute_type"]
    strict: bool = DEFAULT_CONFIG["strict"]
    silence_threshold: float = DEFAULT_CONFIG["silence_threshold"]
    speech_pad: int = DEFAULT_CONFIG["speech_pad"]
    force_redetect: bool = DEFAULT_CONFIG["force_redetect"]
    temp_dir: str | None = None
    vad_model_path: str | None = None
    language_id_model_path: str | None = None
    allow_model_download: bool = DEFAULT_CONFIG["allow_model_download"]
    vad_backend: str = DEFAULT_CONFIG["vad_backend"]


MEDIA_VIDEO_EXTENSIONS = {
    ".3gp", ".avi", ".flv", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm", ".wmv"
}


def run_transcription_pipeline(params: dict[str, Any]) -> Iterable[dict[str, Any]]:
    mode = read_string(params, "mode") or "transcribe"
    if mode != "transcribe":
        raise ValueError(f"Unsupported subtitle media import mode: {mode}")

    config = build_config(params)
    if config.temp_dir:
        os.makedirs(config.temp_dir, exist_ok=True)
        yield from transcribe_with_temp_dir(config)
        return

    with tempfile.TemporaryDirectory(prefix="lator-fast-whisper-") as temp_dir:
        yield from transcribe_with_temp_dir(replace(config, temp_dir=temp_dir))


def retry_subtitle_segment(params: dict[str, Any]) -> Iterable[dict[str, Any]]:
    mode = read_string(params, "mode") or "retry-segment"
    if mode != "retry-segment":
        raise ValueError(f"Unsupported subtitle media retry mode: {mode}")

    config = build_config({**params, "mode": "transcribe"})
    segment_id = read_string(params, "segmentId") or read_string(params, "segment_id")
    start_seconds = read_required_float(params, "startSeconds")
    end_seconds = read_required_float(params, "endSeconds")
    start_time = read_string(params, "startTime") or seconds_to_srt_time(start_seconds)
    end_time = read_string(params, "endTime") or seconds_to_srt_time(end_seconds)
    if end_seconds <= start_seconds:
        raise ValueError("Segment endSeconds must be greater than startSeconds.")

    with tempfile.TemporaryDirectory(prefix="lator-fast-whisper-retry-") as temp_dir:
        segment_file = os.path.join(temp_dir, "retry_segment.wav")
        yield progress("audio", "Cutting subtitle segment with ffmpeg", 0, 0)
        call_with_plugin_logs(cut_audio_file, config.input_file, start_seconds, end_seconds, segment_file)

        yield progress("model", f"Loading Whisper model: {config.model}", 0, 0)
        from fastwhisper_subtitle.services.whisper import load_whisper_model, transcribe_with_whisper

        whisper_model = call_with_plugin_logs(
            load_whisper_model,
            config.model,
            compute_type=config.compute_type,
            allow_model_download=config.allow_model_download,
        )

        yield progress("transcribe", "Recognizing subtitle segment", 0, 1)
        whisper_result = call_with_plugin_logs(
            transcribe_with_whisper,
            segment_file,
            config.language,
            config.model,
            temp_dir,
            whisper_model_instance=whisper_model,
            beam_size=config.beam_size,
            task=config.task,
            compute_type=config.compute_type,
            allow_model_download=config.allow_model_download,
        )
        text = normalize_retry_text(whisper_result)
        if not text:
            raise RuntimeError("Whisper returned no text for this subtitle segment.")
        if is_hallucination_text(text):
            raise RuntimeError("Whisper output matched the hallucination phrase filter.")

        segment: dict[str, Any] = {
            "segmentId": segment_id,
            "startSeconds": start_seconds,
            "endSeconds": end_seconds,
            "startTime": start_time,
            "endTime": end_time,
            "text": text,
        }
        confidence = estimate_retry_confidence(whisper_result)
        if confidence is not None:
            segment["confidence"] = confidence

        yield progress("transcribe", "Recognized subtitle segment", 1, 1)
        yield {
            "type": "done",
            "data": {
                "projectId": config.project_id,
                "media": {
                    "sourcePath": config.input_file,
                    "kind": infer_media_kind(config.input_file),
                },
                "segment": segment,
                "detectedLanguage": normalize_detected_language(whisper_result.get("language")) or config.language,
                "model": config.model,
                "task": config.task,
            },
        }


def transcribe_with_temp_dir(config: TranscriptionConfig) -> Iterable[dict[str, Any]]:
    if not config.temp_dir:
        raise ValueError("temp_dir is required.")

    yield progress("prepare", f"Preparing {Path(config.input_file).name}", 0, 0)

    audio_file = os.path.join(config.temp_dir, "extracted_audio.wav")
    if not os.path.exists(audio_file):
        yield progress("audio", "Extracting audio with ffmpeg", 0, 0)
        call_with_plugin_logs(extract_audio, config.input_file, audio_file)

    continuous_segments = None
    if not config.force_redetect:
        continuous_segments = call_with_plugin_logs(load_segments_cache, config.temp_dir)
        if continuous_segments:
            yield progress("vad", f"Loaded {len(continuous_segments)} cached speech segments", len(continuous_segments), len(continuous_segments))

    if continuous_segments is None:
        if config.vad_backend == "pyannote":
            yield progress("vad", "Loading pyannote VAD backend", 0, 0)
            from fastwhisper_subtitle.services.vad import detect_continuous_speech_segments
            vad_message = "Detecting speech segments with pyannote VAD"
        else:
            yield progress("vad", "Loading Silero VAD backend", 0, 0)
            from fastwhisper_subtitle.services.vad_silero import detect_continuous_speech_segments
            vad_message = "Detecting speech segments with Silero VAD"

        yield progress("vad", vad_message, 0, 0)
        continuous_segments = call_with_plugin_logs(
            detect_continuous_speech_segments,
            audio_file,
            config.silence_threshold,
            config.speech_pad,
            config.vad_model_path,
        )
        call_with_plugin_logs(save_segments_cache, config.temp_dir, continuous_segments)

    if not continuous_segments:
        raise RuntimeError("No speech segments were detected.")

    language_info_map: dict[tuple[float, float], dict[str, Any]] = {}
    if config.language:
        segments_to_process = continuous_segments
        yield progress("language", f"Using specified language: {config.language}", len(segments_to_process), len(segments_to_process))
    else:
        from fastwhisper_subtitle.services.language_id import detect_language_for_segments

        yield progress("language", "Detecting segment languages with SpeechBrain", 0, len(continuous_segments))
        segments_with_language = call_with_plugin_logs(
            detect_language_for_segments,
            audio_file,
            continuous_segments,
            config.temp_dir,
            model_dir=config.language_id_model_path,
        )
        language_info_map = build_language_info_map(segments_with_language)
        segments_to_process = filter_segments_by_output(
            continuous_segments,
            language_info_map,
            config.output_languages,
        )
        yield progress("language", f"Selected {len(segments_to_process)} speech segments", len(segments_to_process), len(continuous_segments))

    yield progress("model", f"Loading Whisper model: {config.model}", 0, 0)
    from fastwhisper_subtitle.services.whisper import load_whisper_model, transcribe_with_whisper

    whisper_model = call_with_plugin_logs(
        load_whisper_model,
        config.model,
        compute_type=config.compute_type,
        allow_model_download=config.allow_model_download,
    )

    yield progress("audio", "Reading normalized audio", 0, 0)
    full_waveform, sample_rate = call_with_plugin_logs(read_mono_audio, audio_file)

    host_segments: list[dict[str, Any]] = []
    language_stats: dict[str, int] = {}
    total_segments = len(segments_to_process)

    for index, (start_ms, end_ms) in enumerate(segments_to_process):
        segment_language = resolve_segment_language(config, language_info_map, start_ms, end_ms)
        yield progress(
            "transcribe",
            f"Recognizing speech segment {index + 1}/{total_segments}",
            index,
            total_segments,
        )

        if check_segment_completed(index, config.temp_dir):
            whisper_result = call_with_plugin_logs(load_existing_result, index, config.temp_dir)
        else:
            segment_file = os.path.join(config.temp_dir, f"segment_{index:04d}.wav")
            segment_array = cut_audio_segment_memory(
                full_waveform,
                sample_rate,
                start_ms,
                end_ms,
                segment_file,
            )
            maybe_save_low_confidence_segment(
                config.temp_dir,
                index,
                segment_file,
                language_info_map.get((start_ms, end_ms)),
            )
            whisper_result = call_with_plugin_logs(
                transcribe_with_whisper,
                segment_file,
                segment_language,
                config.model,
                config.temp_dir,
                whisper_model_instance=whisper_model,
                audio_array=segment_array,
                beam_size=config.beam_size,
                task=config.task,
                compute_type=config.compute_type,
                allow_model_download=config.allow_model_download,
            )

        detected_language = normalize_detected_language(whisper_result.get("language"))
        if detected_language:
            language_stats[detected_language] = language_stats.get(detected_language, 0) + 1

        new_segments = build_host_segments(
            whisper_result,
            start_ms,
            segment_id_offset=len(host_segments),
            enable_quality_filter=config.strict,
            model=config.model,
        )
        for segment in new_segments:
            host_segments.append(segment)
            yield {"type": "segment", "segment": segment}

        yield progress(
            "transcribe",
            f"Recognized {len(host_segments)} subtitle segments",
            index + 1,
            total_segments,
        )

    result = build_result(
        config=config,
        segments=host_segments,
        detected_language=majority_language(language_stats) or config.language,
        language_stats=language_stats,
        detected_speech_segments=len(continuous_segments),
        processed_speech_segments=len(segments_to_process),
    )
    yield {"type": "done", "data": result}


def build_config(params: dict[str, Any]) -> TranscriptionConfig:
    file_path = read_string(params, "filePath") or read_string(params, "file_path")
    if not file_path:
        raise ValueError("filePath is required.")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Media file does not exist: {file_path}")

    parameter_values = collect_parameters(params)
    language = normalize_language(read_parameter(parameter_values, "language", DEFAULT_CONFIG["language"]))
    return TranscriptionConfig(
        input_file=file_path,
        project_id=read_string(params, "projectId") or read_string(params, "project_id") or "",
        language=language,
        output_languages=normalize_output_languages(read_parameter(parameter_values, "outputLanguages", DEFAULT_CONFIG["output"])),
        model=read_clean_string(read_parameter(parameter_values, "model", DEFAULT_CONFIG["model"])) or DEFAULT_CONFIG["model"],
        task=normalize_choice(
            read_parameter(parameter_values, "task", DEFAULT_CONFIG["task"]),
            {"transcribe", "translate"},
            DEFAULT_CONFIG["task"],
        ),
        beam_size=read_positive_int(read_parameter(parameter_values, "beamSize", DEFAULT_CONFIG["beam_size"]), DEFAULT_CONFIG["beam_size"]),
        compute_type=normalize_choice(
            read_parameter(parameter_values, "computeType", DEFAULT_CONFIG["compute_type"]),
            {"default", "int8", "float16", "float32"},
            DEFAULT_CONFIG["compute_type"],
        ),
        strict=read_bool(read_parameter(parameter_values, "strict", DEFAULT_CONFIG["strict"])),
        silence_threshold=read_positive_float(
            read_parameter(parameter_values, "silenceThreshold", DEFAULT_CONFIG["silence_threshold"]),
            DEFAULT_CONFIG["silence_threshold"],
        ),
        speech_pad=read_positive_int(read_parameter(parameter_values, "speechPad", DEFAULT_CONFIG["speech_pad"]), DEFAULT_CONFIG["speech_pad"]),
        force_redetect=read_bool(read_parameter(parameter_values, "forceRedetect", DEFAULT_CONFIG["force_redetect"])),
        temp_dir=read_clean_string(params.get("tempDir") or params.get("cacheDir")),
        vad_model_path=read_clean_string(read_parameter(parameter_values, "vadModelPath", "")) or None,
        language_id_model_path=read_clean_string(read_parameter(parameter_values, "languageIdModelPath", "")) or None,
        allow_model_download=read_bool(read_parameter(
            parameter_values,
            "allowModelDownload",
            DEFAULT_CONFIG["allow_model_download"],
        )),
        vad_backend=normalize_choice(
            read_parameter(parameter_values, "vadBackend", DEFAULT_CONFIG["vad_backend"]),
            {"silero", "pyannote"},
            DEFAULT_CONFIG["vad_backend"],
        ),
    )


def build_language_info_map(segments_with_lang: list[dict[str, Any]]) -> dict[tuple[float, float], dict[str, Any]]:
    language_info_map: dict[tuple[float, float], dict[str, Any]] = {}
    for info in segments_with_lang:
        segment = info.get("segment")
        if not isinstance(segment, tuple) or len(segment) != 2:
            continue
        language_info_map[segment] = {
            "language": info.get("language"),
            "languageFull": info.get("language_full"),
            "confidence": info.get("confidence"),
            "lowConfidence": info.get("low_confidence"),
        }
    return language_info_map


def filter_segments_by_output(
    continuous_segments: list[tuple[float, float]],
    language_info_map: dict[tuple[float, float], dict[str, Any]],
    output_languages: list[str] | None,
) -> list[tuple[float, float]]:
    if not output_languages:
        return continuous_segments

    filtered_segments: list[tuple[float, float]] = []
    for segment in continuous_segments:
        lang_info = language_info_map.get(segment)
        if not lang_info:
            filtered_segments.append(segment)
            continue
        if lang_info.get("lowConfidence"):
            filtered_segments.append(segment)
            continue
        if lang_info.get("language") in output_languages:
            filtered_segments.append(segment)
    return filtered_segments


def resolve_segment_language(
    config: TranscriptionConfig,
    language_info_map: dict[tuple[float, float], dict[str, Any]],
    start_ms: float,
    end_ms: float,
) -> str | None:
    if config.language:
        return config.language
    lang_info = language_info_map.get((start_ms, end_ms))
    if not lang_info or lang_info.get("lowConfidence"):
        return None
    language = lang_info.get("language")
    return language if isinstance(language, str) and language else None


def maybe_save_low_confidence_segment(
    temp_dir: str,
    index: int,
    segment_file: str,
    lang_info: dict[str, Any] | None,
) -> None:
    if not lang_info or not lang_info.get("lowConfidence"):
        return
    low_conf_dir = os.path.join(temp_dir, "low_confidence_segments")
    os.makedirs(low_conf_dir, exist_ok=True)
    low_conf_file = os.path.join(low_conf_dir, f"segment_{index:04d}.wav")
    if not os.path.exists(low_conf_file):
        shutil.copy2(segment_file, low_conf_file)


def build_host_segments(
    whisper_result: dict[str, Any],
    segment_start_ms: float,
    segment_id_offset: int,
    enable_quality_filter: bool,
    model: str,
) -> list[dict[str, Any]]:
    quality_profile = resolve_quality_profile(model)
    host_segments: list[dict[str, Any]] = []
    for segment in whisper_result.get("segments", []):
        if not isinstance(segment, dict):
            continue

        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        if is_hallucination_text(text):
            continue
        if enable_quality_filter and not is_quality_segment(segment, quality_profile):
            text = wrap_deleted_text(text)

        start_seconds = (segment_start_ms / 1000) + float(segment.get("start", 0))
        end_seconds = (segment_start_ms / 1000) + float(segment.get("end", 0))
        host_segment: dict[str, Any] = {
            "segmentId": str(segment_id_offset + len(host_segments) + 1),
            "startSeconds": start_seconds,
            "endSeconds": end_seconds,
            "startTime": seconds_to_srt_time(start_seconds),
            "endTime": seconds_to_srt_time(end_seconds),
            "text": text,
        }
        confidence = estimate_confidence(segment.get("avg_logprob"))
        if confidence is not None:
            host_segment["confidence"] = confidence
        host_segments.append(host_segment)
    return host_segments


def is_quality_segment(segment: dict[str, Any], profile: dict[str, Any]) -> bool:
    avg_logprob_min = read_optional_float(profile.get("avgLogprobMin"))
    compression_ratio_min = read_optional_float(profile.get("compressionRatioMin"))
    no_speech_prob_max = read_optional_float(profile.get("noSpeechProbMax"))

    avg_logprob = read_optional_float(segment.get("avg_logprob"))
    if avg_logprob_min is not None and avg_logprob is not None and avg_logprob < avg_logprob_min:
        return False

    compression_ratio = read_optional_float(segment.get("compression_ratio"))
    if (
        compression_ratio_min is not None
        and compression_ratio is not None
        and compression_ratio < compression_ratio_min
    ):
        return False

    no_speech_prob = read_optional_float(segment.get("no_speech_prob"))
    if no_speech_prob_max is not None and no_speech_prob is not None and no_speech_prob > no_speech_prob_max:
        return False

    return True


def resolve_quality_profile(model: str) -> dict[str, Any]:
    config = load_quality_profiles_config()
    profiles = config.get("profiles")
    if not isinstance(profiles, dict):
        return {}

    model_key = normalize_model_profile_key(model)
    profile = profiles.get(model_key)
    if isinstance(profile, dict):
        return profile

    default_profile = profiles.get("default")
    return default_profile if isinstance(default_profile, dict) else {}


def normalize_model_profile_key(model: str) -> str:
    model_text = read_clean_string(model).lower()
    if "turbo" in model_text:
        return "turbo"
    if "large-v3" in model_text:
        return "large-v3"
    if "large" in model_text:
        return "large"
    if "medium" in model_text:
        return "medium"
    if "small" in model_text:
        return "small"
    return "base" if model_text in {"", "base"} or "base" in model_text else model_text


def is_hallucination_text(text: str) -> bool:
    normalized_text = normalize_filter_text(text)
    if not normalized_text:
        return False
    return normalized_text in load_hallucination_phrase_set()


def load_hallucination_phrase_set() -> set[str]:
    phrases = load_hallucination_phrases_config().get("phrases")
    if not isinstance(phrases, dict):
        return set()

    normalized: set[str] = set()
    for values in phrases.values():
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, str):
                phrase = normalize_filter_text(value)
                if phrase:
                    normalized.add(phrase)
    return normalized


def normalize_filter_text(text: str) -> str:
    normalized = re.sub(r"\s+", "", text.strip().casefold())
    return normalized.strip("。．.！!？?、，,;；:：\"'“”‘’（）()[]【】")


def load_quality_profiles_config() -> dict[str, Any]:
    global _QUALITY_PROFILES_CONFIG
    if _QUALITY_PROFILES_CONFIG is not None:
        return _QUALITY_PROFILES_CONFIG

    try:
        with QUALITY_PROFILES_FILE.open("r", encoding="utf-8") as file:
            config = json.load(file)
    except Exception as error:
        print(f"⚠️ 加载质量配置失败: {error}", file=sys.stderr)
        config = {}

    _QUALITY_PROFILES_CONFIG = config if isinstance(config, dict) else {}
    return _QUALITY_PROFILES_CONFIG


def load_hallucination_phrases_config() -> dict[str, Any]:
    global _HALLUCINATION_PHRASES_CONFIG
    if _HALLUCINATION_PHRASES_CONFIG is not None:
        return _HALLUCINATION_PHRASES_CONFIG

    try:
        with HALLUCINATION_PHRASES_FILE.open("r", encoding="utf-8") as file:
            config = json.load(file)
    except Exception as error:
        print(f"⚠️ 加载幻觉短语配置失败: {error}", file=sys.stderr)
        config = {}

    _HALLUCINATION_PHRASES_CONFIG = config if isinstance(config, dict) else {}
    return _HALLUCINATION_PHRASES_CONFIG


def wrap_deleted_text(text: str) -> str:
    return f"<del>{text}</del>"


def estimate_confidence(avg_logprob: Any) -> float | None:
    if not isinstance(avg_logprob, (int, float)):
        return None
    return max(0.0, min(1.0, 1.0 + float(avg_logprob)))


def normalize_retry_text(whisper_result: dict[str, Any]) -> str:
    text = str(whisper_result.get("text", "")).strip()
    if text:
        return text

    parts: list[str] = []
    for segment in whisper_result.get("segments", []):
        if isinstance(segment, dict):
            part = str(segment.get("text", "")).strip()
            if part:
                parts.append(part)
    return " ".join(parts).strip()


def estimate_retry_confidence(whisper_result: dict[str, Any]) -> float | None:
    values = [
        segment.get("avg_logprob")
        for segment in whisper_result.get("segments", [])
        if isinstance(segment, dict) and isinstance(segment.get("avg_logprob"), (int, float))
    ]
    if not values:
        return None
    return estimate_confidence(sum(float(value) for value in values) / len(values))


def build_result(
    config: TranscriptionConfig,
    segments: list[dict[str, Any]],
    detected_language: str | None,
    language_stats: dict[str, int],
    detected_speech_segments: int,
    processed_speech_segments: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "projectId": config.project_id,
        "media": {
            "sourcePath": config.input_file,
            "kind": infer_media_kind(config.input_file),
        },
        "segments": segments,
        "srt": segments_to_srt(segments),
        "detectedLanguage": detected_language,
        "languageStats": language_stats,
        "model": config.model,
        "task": config.task,
        "detectedSpeechSegments": detected_speech_segments,
        "processedSpeechSegments": processed_speech_segments,
    }
    if extra:
        result.update(extra)
    return result


def call_with_plugin_logs(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    with contextlib.redirect_stdout(sys.stderr):
        return fn(*args, **kwargs)


def collect_parameters(params: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    settings_raw = os.environ.get("LATOR_PLUGIN_SETTINGS")
    if settings_raw:
        try:
            settings = json.loads(settings_raw)
            if isinstance(settings, dict):
                values.update(settings)
        except json.JSONDecodeError:
            pass

    payload_parameters = params.get("parameters")
    if isinstance(payload_parameters, dict):
        values.update(payload_parameters)
    return values


def read_parameter(values: dict[str, Any], key: str, default: Any) -> Any:
    if key in values:
        return values[key]
    return default


def infer_media_kind(file_path: str) -> str:
    return "video" if Path(file_path).suffix.lower() in MEDIA_VIDEO_EXTENSIONS else "audio"


def majority_language(language_stats: dict[str, int]) -> str | None:
    if not language_stats:
        return None
    return max(language_stats.items(), key=lambda item: item[1])[0]


def progress(stage: str, message: str, current: int, total: int) -> dict[str, Any]:
    return {
        "type": "progress",
        "stage": stage,
        "message": message,
        "current": current,
        "total": total,
    }


def read_string(value: dict[str, Any], key: str) -> str:
    raw = value.get(key)
    return raw.strip() if isinstance(raw, str) else ""


def read_clean_string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def read_positive_int(value: Any, default: int) -> int:
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str):
        try:
            parsed = int(value)
            if parsed > 0:
                return parsed
        except ValueError:
            pass
    return default


def read_positive_float(value: Any, default: float) -> float:
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    if isinstance(value, str):
        try:
            parsed = float(value)
            if parsed > 0:
                return parsed
        except ValueError:
            pass
    return default


def read_optional_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, str):
        try:
            parsed = float(value)
            if math.isfinite(parsed):
                return parsed
        except ValueError:
            pass
    return None


def read_required_float(value: dict[str, Any], key: str) -> float:
    raw = value.get(key)
    if isinstance(raw, (int, float)) and math.isfinite(float(raw)):
        return float(raw)
    if isinstance(raw, str):
        try:
            parsed = float(raw)
            if math.isfinite(parsed):
                return parsed
        except ValueError:
            pass
    raise ValueError(f"{key} is required.")


def read_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def normalize_language(value: Any) -> str | None:
    language = read_clean_string(value).lower()
    if not language or language in {"auto", "detect", "none", "null"}:
        return None
    return language


def normalize_output_languages(value: Any) -> list[str] | None:
    if isinstance(value, str):
        if value.strip().lower() in {"", "all", "none"}:
            return None
        return [item.strip().lower() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        values = [item.strip().lower() for item in value if isinstance(item, str) and item.strip()]
        return values or None
    return None


def normalize_choice(value: Any, allowed: set[str], default: str) -> str:
    text = read_clean_string(value).lower()
    return text if text in allowed else default


def normalize_detected_language(value: Any) -> str | None:
    language = read_clean_string(value).lower()
    return language or None
