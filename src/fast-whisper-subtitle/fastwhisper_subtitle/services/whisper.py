import json
import os
import time
from typing import Dict, Optional

import torch
from faster_whisper import WhisperModel

from fastwhisper_subtitle.model_paths import resolve_whisper_model


def resolve_device_compute(compute_type: Optional[str] = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    default_compute_type = "float16" if device == "cuda" else "int8"
    normalized_compute_type = (compute_type or "default").strip().lower()
    if normalized_compute_type in ("", "default", "auto"):
        normalized_compute_type = default_compute_type
    return device, normalized_compute_type


def load_whisper_model(
    model: str,
    compute_type: Optional[str] = None,
    allow_model_download: bool = False,
) -> WhisperModel:
    device, resolved_compute_type = resolve_device_compute(compute_type)
    resolved_model = resolve_whisper_model(model, allow_download=allow_model_download)
    print(f"使用设备: {device}, 计算类型: {resolved_compute_type}")
    print(f"使用 Whisper 模型: {resolved_model.model}")
    return WhisperModel(
        resolved_model.model,
        device=device,
        compute_type=resolved_compute_type,
        local_files_only=resolved_model.local_files_only,
    )


def transcribe_with_whisper(
    audio_file: str,
    language: Optional[str],
    model: str,
    output_dir: Optional[str] = None,
    whisper_model_instance: Optional[WhisperModel] = None,
    max_retries: int = 3,
    audio_array=None,
    output_file: Optional[str] = None,
    beam_size: int = 5,
    task: str = "transcribe",
    compute_type: Optional[str] = None,
    allow_model_download: bool = False,
) -> Dict:
    """Transcribe audio with faster-whisper and return the common JSON shape."""
    if audio_array is not None:
        print(f"    使用 faster-whisper 识别: {os.path.basename(audio_file)} (内存模式)")
        print(f"    音频时长: {len(audio_array)/16000:.1f}秒")
    else:
        print(f"    使用 faster-whisper 识别: {os.path.basename(audio_file)}")
        try:
            import wave
            with wave.open(audio_file, 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / float(rate)
            print(f"    音频时长: {duration:.1f}秒")
        except Exception:
            pass

    for attempt in range(max_retries):
        if attempt > 0:
            print(f"    重试 {attempt}/{max_retries}...")

        try:
            whisper_model = whisper_model_instance or load_whisper_model(
                model,
                compute_type=compute_type,
                allow_model_download=allow_model_download,
            )

            if language:
                print(f"    使用指定语言: {language}")
            else:
                print("    使用自动语言检测")

            transcribe_kwargs = {
                'audio': audio_array if audio_array is not None else audio_file,
                'language': language.lower() if language else None,
                'beam_size': beam_size,
                'task': task,
                'vad_filter': False,
            }

            segments_generator, info = whisper_model.transcribe(**transcribe_kwargs)
            segments_list = []
            full_text_parts = []

            for segment in segments_generator:
                segment_dict = {
                    'id': len(segments_list),
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text,
                    'avg_logprob': segment.avg_logprob,
                    'compression_ratio': segment.compression_ratio,
                    'no_speech_prob': segment.no_speech_prob,
                }

                segments_list.append(segment_dict)
                full_text_parts.append(segment.text)

            transcription_result = {
                'text': ''.join(full_text_parts),
                'segments': segments_list,
                'language': info.language,
            }

            target_file = output_file
            if target_file is None and output_dir is not None:
                audio_basename = os.path.basename(audio_file).replace('.wav', '')
                target_file = os.path.join(output_dir, f"{audio_basename}.json")

            if target_file:
                with open(target_file, 'w', encoding='utf-8') as f:
                    json.dump(transcription_result, f, ensure_ascii=False, indent=2)

            segment_count = len(segments_list)
            print(f"    ✓ 识别完成: {segment_count} 个字幕片段 | 检测语言: {info.language}")
            return transcription_result

        except Exception as e:
            print(f"    ⚠️ faster-whisper 识别失败: {e}")
            import traceback
            traceback.print_exc()
            if attempt < max_retries - 1:
                time.sleep(5)
            continue

    print(f"    ✗ 识别失败，已重试{max_retries}次")
    return {'segments': [], 'language': 'unknown', 'text': ''}
