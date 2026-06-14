from __future__ import annotations

from typing import Any


def milliseconds_to_srt_time(ms: float) -> str:
    hours = int(ms // 3600000)
    ms %= 3600000
    minutes = int(ms // 60000)
    ms %= 60000
    seconds = int(ms // 1000)
    milliseconds = int(ms % 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def seconds_to_srt_time(seconds: float) -> str:
    return milliseconds_to_srt_time(seconds * 1000)


def segments_to_srt(segments: list[dict[str, Any]]) -> str:
    subtitle_index = 1
    blocks: list[str] = []

    for segment in segments:
        text = str(segment.get("text", "")).strip()
        start_time = str(segment.get("startTime", "")).strip()
        end_time = str(segment.get("endTime", "")).strip()
        if not text or not start_time or not end_time:
            continue
        blocks.append(f"{subtitle_index}\n{start_time} --> {end_time}\n{text}")
        subtitle_index += 1

    return "\n\n".join(blocks)
