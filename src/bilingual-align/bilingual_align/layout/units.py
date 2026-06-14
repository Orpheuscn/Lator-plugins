"""Alignment source units derived from layout lines."""

from __future__ import annotations

from dataclasses import dataclass

from . import text_processing as text
from .types import SourceLine


@dataclass(frozen=True)
class AlignmentUnit:
    """One model-facing source unit mapped back to an output row."""

    index: int
    source_pos: int
    text: str


def build_alignment_units(
    source_lines: list[SourceLine],
    long_line_chars: int,
    unit_char_budget: int,
    source_pos_offset: int = 0,
):
    units = []
    for offset, line in enumerate(source_lines):
        source_pos = source_pos_offset + offset
        for unit_text in split_source_line(line.text, long_line_chars, unit_char_budget):
            units.append(AlignmentUnit(index=line.index, source_pos=source_pos, text=unit_text))
    return units


def split_source_line(value: str, long_line_chars: int, unit_char_budget: int):
    value = str(value).strip()
    if not value:
        return []
    if long_line_chars <= 0 or len(value) <= long_line_chars:
        return [value]

    pieces = text.split_strong(value)
    if len(pieces) <= 1:
        pieces = text.split_by_delimiters(value)
    if len(pieces) <= 1:
        return [value]

    budget = max(1, int(unit_char_budget))
    units = []
    current: list[str] = []
    for piece in pieces:
        candidate = text.join_fragments([*current, piece])
        if current and len(candidate) > budget:
            units.append(text.join_fragments(current))
            current = [piece]
        else:
            current.append(piece)

    if current:
        units.append(text.join_fragments(current))
    return units or [value]
