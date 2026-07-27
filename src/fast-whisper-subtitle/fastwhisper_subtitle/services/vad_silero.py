from typing import List, Tuple

import torch
from silero_vad import get_speech_timestamps, load_silero_vad

from fastwhisper_subtitle.services.audio import read_mono_audio
from fastwhisper_subtitle.services.speech_segments import merge_and_pad_speech_segments


SILERO_THRESHOLD = 0.2
SILERO_MIN_SILENCE_DURATION_MS = 100
SILERO_SPEECH_PAD_MS = 0


def detect_continuous_speech_segments(
    audio_file: str,
    silence_threshold_sec: float = 2.0,
    speech_pad_ms: int = 300,
    model_dir: str | None = None,
) -> List[Tuple[float, float]]:
    """Detect speech segments using Silero VAD.

    The signature matches the pyannote backend so the pipeline can switch VAD
    implementations without special casing call sites. The VAD model keeps a
    short raw silence threshold; plugin-level silence merging and padding are
    applied afterwards so both backends use the same user-facing semantics.
    """
    print("正在使用 Silero VAD 检测语音片段...")
    print(
        "Silero 参数: "
        f"threshold={SILERO_THRESHOLD}, "
        f"min_silence_duration_ms={SILERO_MIN_SILENCE_DURATION_MS}, "
        f"raw_speech_pad_ms={SILERO_SPEECH_PAD_MS}, "
        f"merge_gap={silence_threshold_sec}s, "
        f"recognition_pad_ms={speech_pad_ms}"
    )

    waveform, sample_rate = read_mono_audio(audio_file)
    if sample_rate != 16000:
        raise ValueError(f"Silero VAD expects 16kHz audio, got {sample_rate}Hz.")

    model = load_silero_vad()
    audio_tensor = torch.from_numpy(waveform).float()
    timestamps = get_speech_timestamps(
        audio_tensor,
        model,
        sampling_rate=sample_rate,
        threshold=SILERO_THRESHOLD,
        min_silence_duration_ms=SILERO_MIN_SILENCE_DURATION_MS,
        speech_pad_ms=SILERO_SPEECH_PAD_MS,
        return_seconds=False,
    )

    raw_speech_segments: List[Tuple[float, float]] = []
    for timestamp in timestamps:
        start_sample = float(timestamp.get("start", 0))
        end_sample = float(timestamp.get("end", 0))
        if end_sample <= start_sample:
            continue
        raw_speech_segments.append((
            start_sample / sample_rate * 1000,
            end_sample / sample_rate * 1000,
        ))

    if not raw_speech_segments:
        print("未检测到任何语音")
        return []

    audio_length_ms = len(waveform) / sample_rate * 1000
    speech_segments = merge_and_pad_speech_segments(
        raw_speech_segments,
        silence_threshold_sec,
        speech_pad_ms,
        audio_length_ms,
    )

    print(f"检测到 {len(raw_speech_segments)} 个原始语音片段，合并后 {len(speech_segments)} 个连续语音片段")
    for index, (start_ms, end_ms) in enumerate(speech_segments):
        print(
            f"  片段{index + 1}: {start_ms/1000:.2f}s - {end_ms/1000:.2f}s "
            f"(时长: {(end_ms-start_ms)/1000:.2f}s)"
        )

    print(f"Silero VAD 检测完成，总计: {len(speech_segments)} 个语音片段")
    return speech_segments
