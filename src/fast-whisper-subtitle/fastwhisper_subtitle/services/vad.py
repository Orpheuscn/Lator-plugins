from typing import List, Tuple

import torch
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection

from fastwhisper_subtitle.model_paths import resolve_vad_model_path
from fastwhisper_subtitle.services.audio import read_mono_audio


def merge_speech_segments_from_tuples(
    segments: List[Tuple[float, float]],
    silence_threshold_sec: float,
) -> List[Tuple[float, float]]:
    if not segments:
        return []

    segments = sorted(segments, key=lambda x: x[0])
    silence_threshold_ms = silence_threshold_sec * 1000
    merged_segments = []
    current_start, current_end = segments[0]

    for i in range(1, len(segments)):
        next_start, next_end = segments[i]
        if (next_start - current_end) < silence_threshold_ms:
            current_end = max(current_end, next_end)
        else:
            merged_segments.append((current_start, current_end))
            current_start, current_end = next_start, next_end

    merged_segments.append((current_start, current_end))
    return merged_segments


def detect_continuous_speech_segments(
    audio_file: str,
    silence_threshold_sec: float = 2.0,
    speech_pad_ms: int = 300,
    model_dir: str | None = None,
) -> List[Tuple[float, float]]:
    """Detect and merge continuous speech segments using pyannote VAD."""
    print(f"正在检测连续语音片段（断句间隔: {silence_threshold_sec}秒）...")
    print("正在加载本地 pyannote segmentation-3.0 VAD 模型...")

    try:
        model_path = resolve_vad_model_path(model_dir)
        model = Model.from_pretrained(str(model_path))
        vad_pipeline = VoiceActivityDetection(segmentation=model)
        vad_pipeline.instantiate({
            "min_duration_on": 0.0,
            "min_duration_off": 0.0,
        })
        print(f"模型加载完成: {model_path}\n")
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("请确认本地 VAD 模型已放在 models/pyannote-segmentation-3.0/")
        raise

    print("正在分析音频文件...")
    import numpy as np

    waveform, sample_rate = read_mono_audio(audio_file)
    audio_length = len(waveform) / sample_rate
    audio_length_ms = audio_length * 1000
    print(f"音频读取完成，总长度: {audio_length:.2f}秒\n")

    print("=" * 60)
    print("使用 pyannote segmentation-3.0 检测语音片段")
    print("=" * 60)

    audio_data = {
        "waveform": torch.from_numpy(waveform[np.newaxis, :]).float(),
        "sample_rate": sample_rate,
    }
    vad_result = vad_pipeline(audio_data)

    speech_segments = []
    for segment, _, label in vad_result.itertracks(yield_label=True):
        start_ms = max(0, segment.start * 1000 - speech_pad_ms)
        end_ms = min(audio_length_ms, segment.end * 1000 + speech_pad_ms)
        speech_segments.append((start_ms, end_ms))

    if not speech_segments:
        print("未检测到任何语音")
        return []

    print(f"检测到 {len(speech_segments)} 个语音片段")
    merged_segments = merge_speech_segments_from_tuples(
        speech_segments,
        silence_threshold_sec,
    )

    print(f"\n合并后: {len(merged_segments)} 个连续语音片段")
    for i, (start_ms, end_ms) in enumerate(merged_segments):
        print(
            f"  片段{i+1}: {start_ms/1000:.2f}s - {end_ms/1000:.2f}s "
            f"(时长: {(end_ms-start_ms)/1000:.2f}s)"
        )

    print("\n" + "=" * 60)
    print(f"检测完成！总计: {len(merged_segments)} 个连续语音片段")
    print("=" * 60)
    return merged_segments
