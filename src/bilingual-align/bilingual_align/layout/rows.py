#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Output row helpers."""

from __future__ import annotations

from . import text_processing as text
from .types import OutputRow, SourceLine


def build_initial_rows(source_lines: list[SourceLine]):
    return [
        OutputRow(
            line_number=line.index + 1,
            source=line.text,
            target="" if line.text.strip() else "",
            status="pending" if line.text.strip() else "blank",
        )
        for line in source_lines
    ]


def build_response(rows: list[OutputRow], source_lines: list[SourceLine], target_segments: list[str]):
    active_rows = [row for row in rows if row.status != "blank"]
    missing_count = sum(row.status.startswith("missing") for row in active_rows)
    addition_count = sum("addition" in row.status for row in active_rows)
    mistranslation_count = sum("mistranslation" in row.status for row in active_rows)
    return {
        "rows": [row.to_dict() for row in rows],
        "summary": {
            "source_line_count": len(source_lines),
            "target_segment_count": len(target_segments),
            "missing_count": missing_count,
            "addition_count": addition_count,
            "mistranslation_count": mistranslation_count,
        },
    }


def completed_row_count(rows: list[OutputRow]):
    return sum(row.status != "pending" for row in rows)


def snapshot_rows(rows: list[OutputRow]):
    return [row.to_dict() for row in rows]


def mark_missing(rows: list[OutputRow], source_row_indices: list[int]):
    for index in source_row_indices:
        rows[index].target = text.missing_marker(rows[index].source)
        rows[index].status = "missing"
        rows[index].similarity = None
        rows[index].target_fragments = None


def mark_remaining_missing(
    rows: list[OutputRow],
    active_lines: list[SourceLine],
    source_pos: int,
):
    for line in active_lines[source_pos:]:
        rows[line.index].target = text.missing_marker(line.text)
        rows[line.index].status = "missing"
        rows[line.index].similarity = None
        rows[line.index].target_fragments = None


def attach_addition(
    rows: list[OutputRow],
    anchor_index: int | None,
    addition_text: str,
):
    addition_text = addition_text.strip()
    if not addition_text:
        return

    if anchor_index is None:
        anchor_index = first_nonblank_row_index(rows)
    if anchor_index is None:
        return

    marker = text.addition_marker(addition_text)
    current = rows[anchor_index].target.strip()
    rows[anchor_index].target = text.join_fragments([current, marker]) if current else marker
    rows[anchor_index].target_fragments = [
        *(rows[anchor_index].target_fragments or []),
        addition_text,
    ]
    if rows[anchor_index].status in {"pending", "blank"}:
        rows[anchor_index].status = "addition"
    elif "addition" not in rows[anchor_index].status:
        rows[anchor_index].status = f"{rows[anchor_index].status}_with_addition"


def first_nonblank_row_index(rows: list[OutputRow]):
    for index, row in enumerate(rows):
        if row.source.strip():
            return index
    return None
