from __future__ import annotations

import unittest
from dataclasses import dataclass

from unaligned_entries import build_unaligned_token_entries


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Link:
    src_index: int
    tgt_index: int


class BuildUnalignedTokenEntriesTests(unittest.TestCase):
    def test_emits_searchable_entries_for_both_unaligned_sides(self) -> None:
        entries = build_unaligned_token_entries(
            [Token("known", 0, 5), Token("orphan", 6, 12)],
            [Token("已知", 0, 2), Token("遗词", 3, 5)],
            [Link(0, 0)],
        )

        self.assertEqual(entries, [
            {
                "src_text": "orphan",
                "src_span": [6, 12],
                "tgt_text": None,
                "tgt_span": None,
                "score": 0.0,
                "alignment_type": "source_unaligned",
            },
            {
                "src_text": None,
                "src_span": None,
                "tgt_text": "遗词",
                "tgt_span": [3, 5],
                "score": 0.0,
                "alignment_type": "target_unaligned",
            },
        ])

    def test_emits_every_token_when_no_links_survive(self) -> None:
        entries = build_unaligned_token_entries(
            [Token("source", 0, 6)],
            [Token("译文", 0, 2)],
            [],
        )

        self.assertEqual(
            [entry["alignment_type"] for entry in entries],
            ["source_unaligned", "target_unaligned"],
        )


if __name__ == "__main__":
    unittest.main()
