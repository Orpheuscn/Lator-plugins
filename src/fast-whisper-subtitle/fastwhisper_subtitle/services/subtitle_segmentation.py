from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


SENTENCE_ENDINGS = frozenset(".?!。！？")
WEAK_BREAK_PUNCTUATION = frozenset(",;:，、；：")


@dataclass(frozen=True)
class SubtitleSegmentationOptions:
    mode: str = "auto"
    max_duration_seconds: float = 7.0
    target_duration_seconds: float = 4.5
    min_duration_seconds: float = 1.2
    max_words: int = 16
    max_characters: int = 50
    min_pause_seconds: float = 0.3


def should_request_word_timestamps(
    mode: str,
    speech_duration_seconds: float,
    max_duration_seconds: float,
) -> bool:
    """Request alignment only when the result can require readable cue splitting."""
    if mode == "pause":
        return False
    if mode == "readable":
        return speech_duration_seconds > 0
    return speech_duration_seconds > max_duration_seconds


def split_whisper_segments(
    segments: Iterable[dict[str, Any]],
    options: SubtitleSegmentationOptions,
) -> list[dict[str, Any]]:
    """Split long Whisper segments into readable subtitle cues without re-decoding."""
    cues: list[dict[str, Any]] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        cues.extend(split_whisper_segment(segment, options))
    return cues


def split_whisper_segment(
    segment: dict[str, Any],
    options: SubtitleSegmentationOptions,
) -> list[dict[str, Any]]:
    text = str(segment.get("text", "")).strip()
    if not text:
        return []

    if options.mode == "pause" or not cue_exceeds_limits(segment, text, options):
        return [copy_segment_with_text(segment, text)]

    words = normalize_words(segment.get("words"))
    if not words:
        # Auto mode requests words for every VAD block that can exceed the hard
        # duration. Keeping this fallback preserves compatibility with cached or
        # older results without inventing timestamps from character positions.
        return [copy_segment_with_text(segment, text)]

    return build_cues_from_words(segment, words, options)


def cue_exceeds_limits(
    segment: dict[str, Any],
    text: str,
    options: SubtitleSegmentationOptions,
) -> bool:
    duration = max(0.0, as_float(segment.get("end")) - as_float(segment.get("start")))
    return (
        duration > options.max_duration_seconds
        or count_visible_characters(text) > options.max_characters
        or count_words(text) > options.max_words
    )


def build_cues_from_words(
    segment: dict[str, Any],
    words: list[dict[str, Any]],
    options: SubtitleSegmentationOptions,
) -> list[dict[str, Any]]:
    cues: list[dict[str, Any]] = []
    start_index = 0
    while start_index < len(words):
        end_index = select_cue_end(words, start_index, options)
        cue_words = words[start_index:end_index + 1]
        cue_text = join_word_text(cue_words)
        if cue_text:
            cue = copy_segment_with_text(segment, cue_text)
            cue["start"] = cue_words[0]["start"]
            cue["end"] = cue_words[-1]["end"]
            cue["words"] = cue_words
            cues.append(cue)
        start_index = end_index + 1
    return cues


def select_cue_end(
    words: list[dict[str, Any]],
    start_index: int,
    options: SubtitleSegmentationOptions,
) -> int:
    start_time = words[start_index]["start"]
    candidate_ends: list[tuple[float, int]] = []
    last_allowed_index = start_index

    for index in range(start_index, len(words)):
        cue_words = words[start_index:index + 1]
        duration = words[index]["end"] - start_time
        text = join_word_text(cue_words)
        exceeds_limits = (
            duration > options.max_duration_seconds
            or count_words(text) > options.max_words
            or count_visible_characters(text) > options.max_characters
        )
        if exceeds_limits and index > start_index:
            break

        last_allowed_index = index
        if duration < options.min_duration_seconds:
            continue

        boundary_strength = boundary_strength_after(words, index, options)
        if boundary_strength <= 0:
            continue
        duration_penalty = abs(duration - options.target_duration_seconds)
        candidate_ends.append((boundary_strength * 10 - duration_penalty, index))

    if candidate_ends:
        return max(candidate_ends, key=lambda candidate: candidate[0])[1]
    return last_allowed_index


def boundary_strength_after(
    words: list[dict[str, Any]],
    index: int,
    options: SubtitleSegmentationOptions,
) -> int:
    word_text = words[index]["word"].rstrip()
    if word_text and word_text[-1] in SENTENCE_ENDINGS:
        return 4
    if index + 1 < len(words):
        pause = words[index + 1]["start"] - words[index]["end"]
        if pause >= options.min_pause_seconds:
            return 3
    if word_text and word_text[-1] in WEAK_BREAK_PUNCTUATION:
        return 2
    return 0


def normalize_words(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    words: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        word = item.get("word")
        if not isinstance(word, str) or not word:
            continue
        start = as_float(item.get("start"))
        end = as_float(item.get("end"))
        if end < start:
            continue
        words.append({"word": word, "start": start, "end": end})
    return words


def copy_segment_with_text(segment: dict[str, Any], text: str) -> dict[str, Any]:
    copied = dict(segment)
    copied["text"] = text
    return copied


def join_word_text(words: Iterable[dict[str, Any]]) -> str:
    values = [str(word["word"]) for word in words]
    return "".join(values).strip()


def count_visible_characters(text: str) -> int:
    return len("".join(text.split()))


def count_words(text: str) -> int:
    return len([part for part in text.split() if part])


def as_float(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0
