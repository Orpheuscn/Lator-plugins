"""Candidate split helpers for group-level fragment reassignment."""

from __future__ import annotations

from . import text_processing as text

MAX_TOKEN_SPLIT_CANDIDATES = 96


def binary_split_candidates(fragments: list[str]):
    values = [str(fragment).strip() for fragment in fragments if str(fragment).strip()]
    candidates: list[tuple[list[str], list[str]]] = []
    seen: set[tuple[str, str]] = set()

    for index in range(1, len(values)):
        _add_candidate(candidates, seen, values[:index], values[index:])

    combined = text.join_fragments(values)
    for left, right in _token_split_candidates(combined):
        _add_candidate(candidates, seen, [left], [right])

    return candidates


def _token_split_candidates(value: str):
    tokens, separator = text.proportional_tokens(value)
    if len(tokens) < 2:
        return []

    candidates = []
    for split_index in _bounded_split_indices(len(tokens)):
        left = separator.join(tokens[:split_index]).strip()
        right = separator.join(tokens[split_index:]).strip()
        if left and right and not _starts_with_closing_punctuation(right):
            candidates.append((left, right))
    return candidates


def _bounded_split_indices(token_count: int):
    indices = list(range(1, token_count))
    if len(indices) <= MAX_TOKEN_SPLIT_CANDIDATES:
        return indices

    step = len(indices) / MAX_TOKEN_SPLIT_CANDIDATES
    sampled = {
        indices[int(index * step)]
        for index in range(MAX_TOKEN_SPLIT_CANDIDATES)
    }
    sampled.add(token_count // 2)
    return sorted(sampled)


def _add_candidate(
    candidates: list[tuple[list[str], list[str]]],
    seen: set[tuple[str, str]],
    left: list[str],
    right: list[str],
):
    left_text = text.join_fragments(left)
    right_text = text.join_fragments(right)
    if not _has_semantic_content(left_text) or not _has_semantic_content(right_text):
        return

    key = (left_text, right_text)
    if key in seen:
        return

    seen.add(key)
    candidates.append((left, right))


def _has_semantic_content(value: str):
    return any(char.isalnum() for char in value)


def _starts_with_closing_punctuation(value: str):
    return value[:1] in ")]}】》。，、；：！？!?.,;:%”’"
