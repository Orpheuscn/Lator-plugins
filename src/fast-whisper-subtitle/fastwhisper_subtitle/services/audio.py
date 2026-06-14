import subprocess
from pathlib import Path
from typing import Tuple

from fastwhisper_subtitle.services.ffmpeg import resolve_ffmpeg_binary


def extract_audio(input_file: str, output_audio: str) -> None:
    """Extract mono 16kHz PCM WAV audio from any ffmpeg-supported input."""
    print("正在提取音频（ffmpeg自动检测格式）...")
    cmd = [
        resolve_ffmpeg_binary(), '-i', input_file,
        '-vn', '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le',
        '-y', output_audio,
    ]
    subprocess.run(cmd, check=True)


def read_mono_audio(audio_file: str):
    """Read audio as mono waveform and return (waveform, sample_rate)."""
    import numpy as np
    import soundfile as sf

    waveform, sample_rate = sf.read(audio_file)
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    return waveform, sample_rate


def cut_audio_segment_memory(
    waveform,
    sample_rate: int,
    start_ms: float,
    end_ms: float,
    output_file: str,
):
    """Cut an audio segment from a waveform and persist it as WAV."""
    import soundfile as sf

    start_sample = int(start_ms * sample_rate / 1000)
    end_sample = int(end_ms * sample_rate / 1000)
    start_sample = max(0, start_sample)
    end_sample = min(len(waveform), end_sample)

    segment = waveform[start_sample:end_sample]
    sf.write(output_file, segment, sample_rate)
    return segment


def cut_audio_file(source_audio: str, start: float, end: float, output_file: str) -> None:
    """Cut a time range from an audio/video file to a normalized WAV."""
    cmd = [
        resolve_ffmpeg_binary(), '-y',
        '-i', source_audio,
        '-ss', str(start),
        '-to', str(end),
        '-c:a', 'pcm_s16le', '-ar', '16000', '-ac', '1',
        output_file,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def default_temp_dir_for(input_file: str) -> str:
    return str(Path(input_file).parent / 'temp')
