from __future__ import annotations

from typing import Any, Protocol, Sequence


class TokenLike(Protocol):
    text: str
    start: int
    end: int


class LinkLike(Protocol):
    src_index: int
    tgt_index: int


def build_unaligned_token_entries(
    source_tokens: Sequence[TokenLike],
    target_tokens: Sequence[TokenLike],
    links: Sequence[LinkLike],
) -> list[dict[str, Any]]:
    """Return searchable one-sided entries for tokens with no retained alignment link."""
    aligned_source_indexes = {link.src_index for link in links}
    aligned_target_indexes = {link.tgt_index for link in links}
    entries: list[dict[str, Any]] = []

    for index, token in enumerate(source_tokens):
        if index in aligned_source_indexes or not token.text.strip():
            continue
        entries.append({
            "src_text": token.text,
            "src_span": [token.start, token.end],
            "tgt_text": None,
            "tgt_span": None,
            "score": 0.0,
            "alignment_type": "source_unaligned",
        })

    for index, token in enumerate(target_tokens):
        if index in aligned_target_indexes or not token.text.strip():
            continue
        entries.append({
            "src_text": None,
            "src_span": None,
            "tgt_text": token.text,
            "tgt_span": [token.start, token.end],
            "score": 0.0,
            "alignment_type": "target_unaligned",
        })

    return entries
