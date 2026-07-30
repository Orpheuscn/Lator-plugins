from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Segment:
    segment_id: str
    segment_index: int
    display_label: str
    source_text: str
    translated_text: str


def collect_segments(params: dict[str, Any]) -> list[Segment]:
    raw_segments = params.get("segments")
    if isinstance(raw_segments, list):
        segments = [normalize_segment(item, index) for index, item in enumerate(raw_segments)]
        return [segment for segment in segments if segment is not None]

    segment = normalize_segment(params, 0)
    return [segment] if segment is not None else []


def normalize_segment(value: Any, fallback_index: int) -> Segment | None:
    if not isinstance(value, dict):
        return None

    segment_ref = value.get("segmentRef")
    stable_segment_id = (
        _read_string(segment_ref, "segmentId")
        if isinstance(segment_ref, dict)
        else ""
    )
    segment_id = (
        stable_segment_id or
        _read_string(value, "segmentId") or
        _read_string(value, "segment_id")
    )
    if not segment_id:
        return None

    source_text = _read_string(value, "sourceText") or _read_string(value, "source_text")
    translated_text = (
        _read_string(value, "translatedText") or
        _read_string(value, "targetText") or
        _read_string(value, "translated_text")
    )
    if not source_text.strip() or not translated_text.strip():
        return None

    display_ordinal = value.get("displayOrdinal")
    segment_index = value.get(
        "segmentIndex",
        value.get(
            "segment_index",
            display_ordinal - 1 if isinstance(display_ordinal, int) else fallback_index,
        ),
    )
    if not isinstance(segment_index, int):
        segment_index = fallback_index
    display_label = (
        _read_string(value, "displayLabel") or
        str(display_ordinal if isinstance(display_ordinal, int) else segment_index + 1)
    )

    return Segment(
        segment_id=segment_id,
        segment_index=segment_index,
        display_label=display_label,
        source_text=source_text,
        translated_text=translated_text,
    )


def _read_string(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    return item.strip() if isinstance(item, str) else ""
