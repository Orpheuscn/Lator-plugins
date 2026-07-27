from __future__ import annotations

from typing import Iterable


SpeechSegment = tuple[float, float]


def merge_speech_segments(
    segments: Iterable[SpeechSegment],
    silence_threshold_sec: float,
) -> list[SpeechSegment]:
    """Merge raw VAD intervals separated by less than the configured silence."""
    ordered = sorted(
        ((float(start_ms), float(end_ms)) for start_ms, end_ms in segments if end_ms > start_ms),
        key=lambda segment: segment[0],
    )
    if not ordered:
        return []

    silence_threshold_ms = max(0.0, silence_threshold_sec * 1000)
    merged: list[SpeechSegment] = []
    current_start, current_end = ordered[0]

    for next_start, next_end in ordered[1:]:
        if next_start - current_end < silence_threshold_ms:
            current_end = max(current_end, next_end)
            continue
        merged.append((current_start, current_end))
        current_start, current_end = next_start, next_end

    merged.append((current_start, current_end))
    return merged


def pad_speech_segments(
    segments: Iterable[SpeechSegment],
    speech_pad_ms: int,
    audio_length_ms: float,
) -> list[SpeechSegment]:
    """Apply recognition-context padding after silence-based merging."""
    pad_ms = max(0, speech_pad_ms)
    bounded_length_ms = max(0.0, audio_length_ms)
    return [
        (
            max(0.0, start_ms - pad_ms),
            min(bounded_length_ms, end_ms + pad_ms),
        )
        for start_ms, end_ms in segments
    ]


def merge_and_pad_speech_segments(
    segments: Iterable[SpeechSegment],
    silence_threshold_sec: float,
    speech_pad_ms: int,
    audio_length_ms: float,
) -> list[SpeechSegment]:
    return pad_speech_segments(
        merge_speech_segments(segments, silence_threshold_sec),
        speech_pad_ms,
        audio_length_ms,
    )
