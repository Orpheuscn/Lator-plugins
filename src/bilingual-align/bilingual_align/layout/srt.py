#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SRT parsing helpers for source subtitle layout."""

from __future__ import annotations

from dataclasses import dataclass
import re

from . import text_processing as text


TIMING_PATTERN = re.compile(
    r"^\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}\s+-->\s+"
    r"\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}(?:\s+.*)?$",
    re.MULTILINE,
)
TAG_PATTERN = re.compile(r"</?[^>]+>")
ASS_OVERRIDE_PATTERN = re.compile(r"\{\\[^}]+\}")


@dataclass(frozen=True)
class SrtCue:
    """One parsed SRT cue."""

    sequence: str
    timing: str
    text_lines: list[str]
    plain_text: str


def looks_like_srt(content: str):
    """Return true when the text contains SRT timing blocks."""
    return bool(TIMING_PATTERN.search(_normalize_newlines(content)))


def parse_srt(content: str):
    """Parse an SRT document into cues, skipping malformed blocks."""
    normalized = _normalize_newlines(content).lstrip("\ufeff")
    blocks = [
        block
        for block in re.split(r"\n[ \t]*\n+", normalized)
        if block.strip()
    ]

    cues = []
    for block in blocks:
        lines = [line.rstrip() for line in block.split("\n")]
        timing_index = next(
            (
                index
                for index, line in enumerate(lines)
                if TIMING_PATTERN.match(line.strip())
            ),
            -1,
        )
        if timing_index == -1:
            continue

        sequence = lines[0].strip() if timing_index > 0 else str(len(cues) + 1)
        timing = lines[timing_index].strip()
        text_lines = [line for line in lines[timing_index + 1:] if line.strip()]
        plain_text = cue_text_to_line(text_lines)
        if not plain_text:
            continue

        cues.append(SrtCue(
            sequence=sequence or str(len(cues) + 1),
            timing=timing,
            text_lines=text_lines,
            plain_text=plain_text,
        ))

    return cues


def cues_to_source_text(cues: list[SrtCue]):
    """Convert SRT cues into one source line per subtitle cue."""
    return "\n".join(cue.plain_text for cue in cues)


def cue_text_to_line(text_lines: list[str]):
    """Collapse one cue's text lines into the source line used for alignment."""
    value = " ".join(text_lines)
    value = TAG_PATTERN.sub("", value)
    value = ASS_OVERRIDE_PATTERN.sub("", value)
    return text.collapse_spaces(value)


def _normalize_newlines(content: str):
    return str(content or "").replace("\r\n", "\n").replace("\r", "\n")
