from __future__ import annotations

import os
import subprocess
from typing import Any

from fastwhisper_subtitle.services.ffmpeg import resolve_ffmpeg_binary


def play_subtitle_segment(params: dict[str, Any]) -> dict[str, str]:
    host_media = params.get("hostMedia")
    if not isinstance(host_media, dict):
        raise ValueError("hostMedia is required for subtitle playback.")

    start_seconds = read_number(params, "startSeconds")
    end_seconds = read_number(params, "endSeconds")
    if end_seconds <= start_seconds:
        raise ValueError("Invalid subtitle playback time range.")

    audio_url = ""
    video_url = ""
    source_audio = read_string(host_media, "sourceAudioPath") or read_string(host_media, "sourceVideoPath")
    source_video = read_string(host_media, "sourceVideoPath")
    audio_output = read_string(host_media, "inlineAudioOutputPath")
    video_output = read_string(host_media, "inlineVideoOutputPath")

    if source_audio and audio_output:
        ensure_parent_dir(audio_output)
        cut_audio(source_audio, start_seconds, end_seconds, audio_output)
        audio_url = read_string(host_media, "inlineAudioUrl")

    if source_video and video_output:
        ensure_parent_dir(video_output)
        cut_video(source_video, start_seconds, end_seconds, video_output)
        video_url = read_string(host_media, "inlineVideoUrl")

    if not audio_url and not video_url:
        raise ValueError("No playable media output was prepared by the host.")

    return {
        **({"audioUrl": audio_url} if audio_url else {}),
        **({"videoUrl": video_url} if video_url else {}),
    }


def cut_audio(source_file: str, start_seconds: float, end_seconds: float, output_file: str) -> None:
    cmd = [
        resolve_ffmpeg_binary(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(start_seconds),
        "-to",
        str(end_seconds),
        "-i",
        source_file,
        "-vn",
        "-ar",
        "44100",
        "-ac",
        "2",
        "-c:a",
        "aac",
        output_file,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def cut_video(source_file: str, start_seconds: float, end_seconds: float, output_file: str) -> None:
    cmd = [
        resolve_ffmpeg_binary(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(start_seconds),
        "-to",
        str(end_seconds),
        "-i",
        source_file,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        output_file,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def ensure_parent_dir(file_path: str) -> None:
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def read_string(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    return item.strip() if isinstance(item, str) else ""


def read_number(value: dict[str, Any], key: str) -> float:
    item = value.get(key)
    if isinstance(item, (int, float)):
        return float(item)
    raise ValueError(f"Missing numeric playback field: {key}")
