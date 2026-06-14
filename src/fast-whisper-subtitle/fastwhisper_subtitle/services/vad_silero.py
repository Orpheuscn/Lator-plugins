from typing import List, Tuple

import torch
from silero_vad import get_speech_timestamps, load_silero_vad

from fastwhisper_subtitle.services.audio import read_mono_audio


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
    implementations without special casing call sites. Silero uses fixed
    parameters chosen for subtitle segmentation.
    """
    print("正在使用 Silero VAD 检测语音片段...")
    print(
        "Silero 参数: "
        f"threshold={SILERO_THRESHOLD}, "
        f"min_silence_duration_ms={SILERO_MIN_SILENCE_DURATION_MS}, "
        f"speech_pad_ms={SILERO_SPEECH_PAD_MS}"
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

    speech_segments: List[Tuple[float, float]] = []
    for timestamp in timestamps:
        start_sample = float(timestamp.get("start", 0))
        end_sample = float(timestamp.get("end", 0))
        if end_sample <= start_sample:
            continue
        speech_segments.append((
            start_sample / sample_rate * 1000,
            end_sample / sample_rate * 1000,
        ))

    if not speech_segments:
        print("未检测到任何语音")
        return []

    print(f"检测到 {len(speech_segments)} 个语音片段")
    for index, (start_ms, end_ms) in enumerate(speech_segments):
        print(
            f"  片段{index + 1}: {start_ms/1000:.2f}s - {end_ms/1000:.2f}s "
            f"(时长: {(end_ms-start_ms)/1000:.2f}s)"
        )

    print(f"Silero VAD 检测完成，总计: {len(speech_segments)} 个语音片段")
    return speech_segments
