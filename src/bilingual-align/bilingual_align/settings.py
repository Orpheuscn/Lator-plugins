#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Load layout alignment settings from JSON."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EncoderSettings:
    """Embedding encoder runtime settings."""

    model_path: str | None
    max_length: int
    pooling: str | None
    batch_size: int | None
    thread_count: int | None


@dataclass(frozen=True)
class CandidateScoreSettings:
    """Local candidate scoring penalties."""

    target_skip_penalty: float
    extra_source_penalty: float
    extra_target_penalty: float


@dataclass(frozen=True)
class AssignmentSettings:
    """Within-group target fragment assignment settings."""

    empty_score: float
    extra_fragment_penalty: float
    good_fragment_threshold: float
    addition_fragment_threshold: float


@dataclass(frozen=True)
class NMSettings:
    """N:M dynamic-programming aligner settings."""

    max_group_size: int
    top_k: int
    window: int
    skip: float
    margin: bool
    length_penalty: bool


@dataclass(frozen=True)
class AlignmentSettings:
    """All tunable settings for layout-preserving alignment."""

    encoder: EncoderSettings
    candidate_score: CandidateScoreSettings
    assignment: AssignmentSettings
    nm: NMSettings
    mistranslation_threshold: float
    future_match_margin: float
    target_window: int
    source_lookahead: int
    max_target_take: int
    max_addition_skip: int
    single_nm_source_limit: int
    single_nm_target_limit: int
    block_source_lines: int
    block_source_overlap: int
    block_target_slack: int
    source_chunk_char_budget: int
    target_chunk_char_budget: int
    long_source_line_chars: int
    source_unit_char_budget: int
    max_assign_fragments: int
    max_atomic_fragment_count: int
    fine_whitespace_max_tokens: int


def load_alignment_settings(config_path: str | Path | None = None) -> AlignmentSettings:
    """Load and validate the JSON settings file."""
    path = _resolve_config_path(config_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    alignment = _required_mapping(data, "alignment")

    return AlignmentSettings(
        encoder=_load_encoder(_required_mapping(data, "encoder")),
        candidate_score=_load_candidate_score(_required_mapping(alignment, "candidate_score")),
        assignment=_load_assignment(_required_mapping(alignment, "assignment")),
        nm=_load_nm(_required_mapping(alignment, "nm")),
        mistranslation_threshold=_float(alignment, "mistranslation_threshold"),
        future_match_margin=_float(alignment, "future_match_margin"),
        target_window=_int(alignment, "target_window"),
        source_lookahead=_int(alignment, "source_lookahead"),
        max_target_take=_int(alignment, "max_target_take"),
        max_addition_skip=_int(alignment, "max_addition_skip"),
        single_nm_source_limit=_int(alignment, "single_nm_source_limit"),
        single_nm_target_limit=_int(alignment, "single_nm_target_limit"),
        block_source_lines=_int(alignment, "block_source_lines"),
        block_source_overlap=_optional_int(alignment, "block_source_overlap", default=0),
        block_target_slack=_int(alignment, "block_target_slack"),
        source_chunk_char_budget=_optional_int(alignment, "source_chunk_char_budget", default=24000),
        target_chunk_char_budget=_optional_int(alignment, "target_chunk_char_budget", default=28000),
        long_source_line_chars=_optional_int(alignment, "long_source_line_chars", default=1200),
        source_unit_char_budget=_optional_int(alignment, "source_unit_char_budget", default=700),
        max_assign_fragments=_int(alignment, "max_assign_fragments"),
        max_atomic_fragment_count=_int(alignment, "max_atomic_fragment_count"),
        fine_whitespace_max_tokens=_int(alignment, "fine_whitespace_max_tokens"),
    )


def _resolve_config_path(config_path: str | Path | None) -> Path:
    raw_path = config_path or os.environ.get("ALIGNMENT_CONFIG") or Path("config/alignment.json")
    return Path(raw_path).expanduser()


def _load_encoder(data: dict[str, Any]) -> EncoderSettings:
    return EncoderSettings(
        model_path=_optional_str(data, "model_path"),
        max_length=_int(data, "max_length"),
        pooling=_optional_str(data, "pooling"),
        batch_size=_optional_int(data, "batch_size"),
        thread_count=_optional_int(data, "thread_count"),
    )


def _load_candidate_score(data: dict[str, Any]) -> CandidateScoreSettings:
    return CandidateScoreSettings(
        target_skip_penalty=_float(data, "target_skip_penalty"),
        extra_source_penalty=_float(data, "extra_source_penalty"),
        extra_target_penalty=_float(data, "extra_target_penalty"),
    )


def _load_assignment(data: dict[str, Any]) -> AssignmentSettings:
    return AssignmentSettings(
        empty_score=_float(data, "empty_score"),
        extra_fragment_penalty=_float(data, "extra_fragment_penalty"),
        good_fragment_threshold=_float(data, "good_fragment_threshold"),
        addition_fragment_threshold=_float(data, "addition_fragment_threshold"),
    )


def _load_nm(data: dict[str, Any]) -> NMSettings:
    return NMSettings(
        max_group_size=_int(data, "max_group_size"),
        top_k=_int(data, "top_k"),
        window=_int(data, "window"),
        skip=_float(data, "skip"),
        margin=_bool(data, "margin"),
        length_penalty=_bool(data, "length_penalty"),
    )


def _required_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = _required(data, key)
    if not isinstance(value, dict):
        raise TypeError(f"配置项 {key} 必须是对象")
    return value


def _required(data: dict[str, Any], key: str):
    if key not in data:
        raise KeyError(f"配置缺少必填项: {key}")
    return data[key]


def _int(data: dict[str, Any], key: str) -> int:
    return int(_required(data, key))


def _optional_int(data: dict[str, Any], key: str, default: int | None = None) -> int | None:
    if key not in data:
        return default
    value = data[key]
    return None if value is None else int(value)


def _float(data: dict[str, Any], key: str) -> float:
    return float(_required(data, key))


def _bool(data: dict[str, Any], key: str) -> bool:
    value = _required(data, key)
    if not isinstance(value, bool):
        raise TypeError(f"配置项 {key} 必须是布尔值")
    return value


def _optional_str(data: dict[str, Any], key: str) -> str | None:
    value = _required(data, key)
    return None if value is None else str(value)
