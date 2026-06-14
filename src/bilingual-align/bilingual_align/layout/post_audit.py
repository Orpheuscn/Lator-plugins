#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-alignment audit for low-confidence bilingual rows."""

from __future__ import annotations

from collections.abc import Callable

from . import text_processing as text
from .types import OutputRow

LOW_CONFIDENCE_SCORE_THRESHOLD = 0.60
LOW_CONFIDENCE_AUDIT_CONTEXT_ROWS = 2
LOW_CONFIDENCE_AUDIT_MAX_ROWS = 10
LOW_CONFIDENCE_AUDIT_MIN_GAIN = 0.12
LOW_CONFIDENCE_NEIGHBOR_RADIUS = 2
LOW_CONFIDENCE_NEIGHBOR_MARGIN = 0.08


class LowConfidenceAlignmentAuditor:
    """Conservatively repair or flag low-confidence rows after alignment."""

    def __init__(
        self,
        assigner,
        assignment_score: Callable[[str, list[str]], float],
        similarity: Callable[[str, str], float],
    ):
        self.assigner = assigner
        self.assignment_score = assignment_score
        self.similarity = similarity

    def audit(self, output_rows: list[OutputRow]):
        changed_count = self._recover_low_confidence_windows(output_rows)
        changed_count += self._mark_neighbor_stolen_low_confidence_rows(output_rows)
        return changed_count

    def _recover_low_confidence_windows(self, output_rows: list[OutputRow]):
        recovered_count = 0
        index = 0
        while index < len(output_rows):
            if not self._is_low_confidence_row(output_rows[index]):
                index += 1
                continue

            run_start = index
            while index < len(output_rows) and self._is_low_confidence_row(output_rows[index]):
                index += 1
            run_end = index
            window_start = max(0, run_start - LOW_CONFIDENCE_AUDIT_CONTEXT_ROWS)
            window_end = min(len(output_rows), run_end + LOW_CONFIDENCE_AUDIT_CONTEXT_ROWS)
            if window_end - window_start > LOW_CONFIDENCE_AUDIT_MAX_ROWS:
                continue

            if self._recover_low_confidence_window(output_rows, window_start, window_end):
                recovered_count += 1
                index = window_end

        return recovered_count

    def _recover_low_confidence_window(
        self,
        output_rows: list[OutputRow],
        window_start: int,
        window_end: int,
    ):
        window_rows = output_rows[window_start:window_end]
        if len(window_rows) < 2:
            return False
        if any(not self._is_auditable_row(row) for row in window_rows):
            return False

        current_fragments = [list(row.target_fragments or []) for row in window_rows]
        if any(not fragments for fragments in current_fragments):
            return False

        target_fragments = [
            fragment
            for fragments in current_fragments
            for fragment in fragments
        ]
        source_texts = [row.source for row in window_rows]
        assigned = self.assigner.assign_group(source_texts, target_fragments)
        if any(not fragments for fragments in assigned):
            return False
        if assigned == current_fragments:
            return False

        current_scores = [
            self.assignment_score(row.source, fragments)
            for row, fragments in zip(window_rows, current_fragments)
        ]
        candidate_scores = [
            self.assignment_score(source_text, fragments)
            for source_text, fragments in zip(source_texts, assigned)
        ]
        if min(candidate_scores, default=0.0) < LOW_CONFIDENCE_SCORE_THRESHOLD:
            return False
        if sum(candidate_scores) <= sum(current_scores) + LOW_CONFIDENCE_AUDIT_MIN_GAIN:
            return False

        for row, assigned_fragments in zip(window_rows, assigned):
            target, status, similarity = self.assigner.format_assignment(
                row.source,
                assigned_fragments,
            )
            row.target = target
            row.status = status
            row.similarity = similarity
            row.target_fragments = list(assigned_fragments)

        return True

    def _mark_neighbor_stolen_low_confidence_rows(self, output_rows: list[OutputRow]):
        suspicious_indices = []
        index = 0
        while index < len(output_rows):
            if not self._is_low_confidence_row(output_rows[index]):
                index += 1
                continue

            run_start = index
            while index < len(output_rows) and self._is_low_confidence_row(output_rows[index]):
                index += 1
            run_indices = list(range(run_start, index))
            if any(self._has_neighbor_stolen_target(output_rows, row_index) for row_index in run_indices):
                suspicious_indices.extend(run_indices)

        for index in suspicious_indices:
            self._mark_low_confidence_mistranslation(output_rows[index])
        return len(suspicious_indices)

    def _has_neighbor_stolen_target(self, output_rows: list[OutputRow], row_index: int):
        row = output_rows[row_index]
        row_fragments = list(row.target_fragments or [])
        if not row_fragments:
            return False

        current_score = self.assignment_score(row.source, row_fragments)
        for neighbor_index in self._low_confidence_neighbor_indices(output_rows, row_index):
            neighbor = output_rows[neighbor_index]
            neighbor_fragments = list(neighbor.target_fragments or [])
            if not neighbor_fragments:
                continue

            if neighbor_index < row_index:
                candidate_fragments = neighbor_fragments + row_fragments
            else:
                candidate_fragments = row_fragments + neighbor_fragments
            neighbor_score = self.assignment_score(neighbor.source, neighbor_fragments)
            candidate_score = self.assignment_score(neighbor.source, candidate_fragments)
            if (
                candidate_score >= LOW_CONFIDENCE_SCORE_THRESHOLD
                and candidate_score >= current_score + LOW_CONFIDENCE_NEIGHBOR_MARGIN
                and candidate_score >= neighbor_score + LOW_CONFIDENCE_NEIGHBOR_MARGIN
            ):
                return True

        return False

    def _low_confidence_neighbor_indices(self, output_rows: list[OutputRow], row_index: int):
        start = max(0, row_index - LOW_CONFIDENCE_NEIGHBOR_RADIUS)
        end = min(len(output_rows), row_index + LOW_CONFIDENCE_NEIGHBOR_RADIUS + 1)
        return [
            index
            for index in range(start, end)
            if index != row_index and self._is_auditable_row(output_rows[index])
        ]

    def _mark_low_confidence_mistranslation(self, row: OutputRow):
        target_text = text.join_fragments(row.target_fragments or [])
        if not target_text:
            return
        row.target = text.mistranslation_marker(target_text)
        row.status = "mistranslation"
        row.similarity = self.similarity(row.source, target_text)

    def _is_low_confidence_row(self, row: OutputRow):
        return (
            row.status in {"ok", "ok_with_addition"}
            and bool(row.source.strip())
            and bool(row.target_fragments)
            and row.similarity is not None
            and row.similarity < LOW_CONFIDENCE_SCORE_THRESHOLD
        )

    def _is_auditable_row(self, row: OutputRow):
        return (
            row.status in {"ok", "ok_with_addition", "mistranslation"}
            and bool(row.source.strip())
            and bool(row.target_fragments)
        )
