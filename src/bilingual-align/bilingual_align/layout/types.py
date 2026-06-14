#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data types used by layout-preserving alignment."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceLine:
    """One source layout line."""

    index: int
    text: str


@dataclass
class OutputRow:
    """One layout-preserving aligned output row."""

    line_number: int
    source: str
    target: str
    status: str
    similarity: float | None = None
    target_fragments: list[str] | None = None

    def to_dict(self):
        return {
            "line_number": self.line_number,
            "source": self.source,
            "target": self.target,
            "status": self.status,
            "similarity": self.similarity,
        }


@dataclass(frozen=True)
class MatchCandidate:
    """A local source/target match candidate in the sliding window."""

    source_take: int
    target_skip: int
    target_take: int
    similarity: float
    score: float
