#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Layout-preserving bilingual alignment shared by all model variants."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .layout import rows
from .layout import srt
from .layout import text_processing as text
from .layout import units
from .layout.assignment import FragmentAssigner
from .layout.post_audit import LowConfidenceAlignmentAuditor
from .layout.types import OutputRow, SourceLine
from .model_registry import get_model_spec
from .nm_alignment import AlignmentConfig, EmbeddingNMAligner
from .settings import AlignmentSettings, load_alignment_settings

MISSING_RECOVERY_MAX_MISSING_ROWS = 8
MISSING_RECOVERY_MIN_GAIN = 0.05
MISSING_RECOVERY_MIN_SCORE = 0.50
WEAK_WINDOW_SCORE_THRESHOLD = 0.60
SWALLOWED_MISSING_MIN_SCORE = WEAK_WINDOW_SCORE_THRESHOLD
WEAK_WINDOW_CONTEXT_ROWS = 1
WEAK_WINDOW_MAX_ROWS = 8
WEAK_WINDOW_MIN_GAIN = 0.08
STAGE_MESSAGES = {
    "prepare": "正在准备对齐",
    "embedding": "正在编码文本",
    "aligning": "正在对齐文本",
    "chunk": "正在对齐文本",
    "missing_tail": "正在处理尾部缺失对齐",
    "missing_recovery": "正在恢复缺失对齐",
    "swallowed_missing_recovery": "正在校验续行",
    "weak_window_recovery": "正在校验弱匹配",
    "low_confidence_audit": "正在复核低置信对齐",
    "complete": "对齐完成",
}


class LayoutPreservingAligner:
    """Align target translation text to the source text's line layout."""

    def __init__(
        self,
        model_key: str = "bert",
        config_path: str | Path | None = None,
        settings: AlignmentSettings | None = None,
        encoder=None,
    ):
        self.model_spec = get_model_spec(model_key)
        self.model_key = self.model_spec.key
        self.settings = settings or load_alignment_settings(config_path or self.model_spec.config_path)
        self.encoder = encoder or self.model_spec.create_encoder(self.settings.encoder)

        nm_settings = self.settings.nm
        self._nm_aligner = EmbeddingNMAligner(
            self.encoder,
            AlignmentConfig(
                max_group_size=nm_settings.max_group_size,
                top_k=nm_settings.top_k,
                window=nm_settings.window,
                skip=nm_settings.skip,
                margin=nm_settings.margin,
                length_penalty=nm_settings.length_penalty,
            ),
        )
        self._assigner = FragmentAssigner(
            self.settings,
            similarity=self._similarity,
            warm_embeddings=self._warm_embeddings,
        )
        self._similarity_cache: dict[tuple[str, str], float] = {}
        self._embedding_cache: dict[str, np.ndarray] = {}
        self._low_confidence_auditor = LowConfidenceAlignmentAuditor(
            assigner=self._assigner,
            assignment_score=self._assignment_score,
            similarity=self._similarity,
        )

    def align(self, source_text: str, target_text: str):
        """Return source-line-preserving alignment results."""
        result = None
        for event in self.align_events(source_text, target_text):
            if event["type"] == "done":
                result = event["data"]
        return result

    def align_events(self, source_text: str, target_text: str):
        """Yield alignment progress events and chunk-level row updates."""
        self._similarity_cache.clear()
        self._embedding_cache.clear()
        source_text = self._normalize_source_text(source_text)
        target_text = self._normalize_target_text(target_text)
        source_lines = text.build_source_lines(source_text)
        active_lines = [line for line in source_lines if line.text.strip()]
        active_units = self._build_source_units(active_lines)
        target_segments = text.build_target_segments(target_text)
        output_rows = rows.build_initial_rows(source_lines)

        yield {
            "type": "meta",
            "model": self.model_key,
            "message": self._stage_message("prepare"),
            "source_line_count": len(source_lines),
            "target_segment_count": len(target_segments),
            "rows": [row.to_dict() for row in output_rows],
        }

        if not active_lines:
            response = rows.build_response(output_rows, source_lines, target_segments)
            yield self._row_update_event(output_rows, output_rows, "complete")
            yield {"type": "done", "data": response}
            return

        if not target_segments:
            rows.mark_remaining_missing(output_rows, active_lines, 0)
            response = rows.build_response(output_rows, source_lines, target_segments)
            yield self._row_update_event(output_rows, output_rows, "complete")
            yield {"type": "done", "data": response}
            return

        yield {
            "type": "progress",
            "stage": "embedding",
            "message": self._stage_message("embedding"),
            "completed_count": rows.completed_row_count(output_rows),
            "source_line_count": len(source_lines),
        }
        self._warm_embeddings([unit.text for unit in active_units] + target_segments)

        yield {
            "type": "progress",
            "stage": "aligning",
            "message": self._stage_message("aligning"),
            "completed_count": rows.completed_row_count(output_rows),
            "source_line_count": len(source_lines),
        }
        yield from self._align_with_nm_events(output_rows, active_lines, active_units, target_segments)

        yield self._progress_event(output_rows, "missing_recovery")
        before = rows.snapshot_rows(output_rows)
        if self._recover_missing_runs(output_rows):
            yield self._changed_row_update_event(output_rows, before, "missing_recovery")

        yield self._progress_event(output_rows, "swallowed_missing_recovery")
        before = rows.snapshot_rows(output_rows)
        if self._recover_swallowed_missing_runs(output_rows):
            yield self._changed_row_update_event(output_rows, before, "swallowed_missing_recovery")

        yield self._progress_event(output_rows, "weak_window_recovery")
        before = rows.snapshot_rows(output_rows)
        if self._recover_weak_windows(output_rows):
            yield self._changed_row_update_event(output_rows, before, "weak_window_recovery")

        yield self._progress_event(output_rows, "low_confidence_audit")
        before = rows.snapshot_rows(output_rows)
        if self._low_confidence_auditor.audit(output_rows):
            yield self._changed_row_update_event(output_rows, before, "low_confidence_audit")

        response = rows.build_response(output_rows, source_lines, target_segments)
        yield {"type": "done", "data": response}

    def _align_with_nm_events(
        self,
        output_rows: list[OutputRow],
        active_lines: list[SourceLine],
        active_units: list[units.AlignmentUnit],
        target_segments: list[str],
    ):
        """Run N:M alignment and yield chunk-level row updates."""
        if self._fits_single_nm_block(active_units, target_segments):
            groups = self._nm_aligner.align(
                [unit.text for unit in active_units],
                target_segments,
            )
            self._apply_nm_groups(output_rows, active_units, target_segments, groups)
            yield self._row_update_event(output_rows, output_rows, "complete")
            return

        yield from self._align_chunked_nm_events(output_rows, active_lines, target_segments)

    def _fits_single_nm_block(
        self,
        active_units: list[units.AlignmentUnit],
        target_segments: list[str],
    ):
        return (
            len(active_units) <= self.settings.single_nm_source_limit
            and len(target_segments) <= self.settings.single_nm_target_limit
            and self._text_char_count(unit.text for unit in active_units) <= self.settings.source_chunk_char_budget
            and self._text_char_count(target_segments) <= self.settings.target_chunk_char_budget
        )

    def _align_chunked_nm_events(
        self,
        output_rows: list[OutputRow],
        active_lines: list[SourceLine],
        target_segments: list[str],
    ):
        source_pos = 0
        target_cursor = 0
        chunk_index = 0

        while source_pos < len(active_lines):
            if target_cursor >= len(target_segments):
                before = rows.snapshot_rows(output_rows)
                rows.mark_remaining_missing(output_rows, active_lines, source_pos)
                yield self._changed_row_update_event(output_rows, before, "missing_tail", chunk_index)
                return

            before = rows.snapshot_rows(output_rows)
            stable_source_take = self._source_block_take(active_lines, source_pos)
            overlap_take = self._source_overlap_take(active_lines, source_pos, stable_source_take)
            source_take = stable_source_take + overlap_take
            source_block = active_lines[source_pos:source_pos + source_take]
            source_units = self._build_source_units(source_block, source_pos_offset=source_pos)
            is_final_block = source_pos + source_take >= len(active_lines)

            target_end = len(target_segments) if is_final_block else self._target_block_end(
                active_lines,
                target_segments,
                source_pos,
                source_take,
                target_cursor,
            )
            target_block = target_segments[target_cursor:target_end]
            groups = self._nm_aligner.align(
                [unit.text for unit in source_units],
                target_block,
            )
            if not is_final_block:
                groups, finalized_source_end = self._finalized_boundary_groups(
                    groups,
                    source_units,
                    source_pos + stable_source_take,
                )
                finalized_source_count = finalized_source_end - source_pos
            else:
                finalized_source_count = source_take

            consumed_targets = self._apply_nm_groups(output_rows, source_units, target_block, groups)
            target_cursor += consumed_targets
            source_pos += finalized_source_count
            chunk_index += 1
            yield self._changed_row_update_event(output_rows, before, "chunk", chunk_index)

        if target_cursor < len(target_segments):
            before = rows.snapshot_rows(output_rows)
            anchor_index = active_lines[-1].index if active_lines else None
            rows.attach_addition(
                output_rows,
                anchor_index,
                text.join_fragments(target_segments[target_cursor:]),
            )
            yield self._changed_row_update_event(
                output_rows,
                before,
                "trailing_addition",
                chunk_index,
            )

    def _target_block_end(
        self,
        active_lines: list[SourceLine],
        target_segments: list[str],
        source_pos: int,
        source_take: int,
        target_cursor: int,
    ):
        remaining_sources = len(active_lines) - source_pos
        remaining_targets = len(target_segments) - target_cursor
        expected_targets = round(source_take * remaining_targets / remaining_sources)
        target_take = max(1, expected_targets + self.settings.block_target_slack)
        target_end = min(len(target_segments), target_cursor + target_take)
        return self._bounded_target_end(target_segments, target_cursor, target_end)

    def _source_block_take(self, active_lines: list[SourceLine], source_pos: int):
        remaining_sources = len(active_lines) - source_pos
        max_take = min(self.settings.block_source_lines, remaining_sources)
        budget = max(0, int(self.settings.source_chunk_char_budget))
        if budget <= 0:
            return max(1, max_take)

        take = 0
        total_chars = 0
        while take < max_take:
            next_chars = len(active_lines[source_pos + take].text)
            if take > 0 and total_chars + next_chars > budget:
                break
            take += 1
            total_chars += next_chars
        return max(1, take)

    def _source_overlap_take(
        self,
        active_lines: list[SourceLine],
        source_pos: int,
        stable_source_take: int,
    ):
        overlap_limit = min(
            self._block_source_overlap(),
            max(0, len(active_lines) - source_pos - stable_source_take),
        )
        budget = max(0, int(self.settings.source_chunk_char_budget))
        if overlap_limit <= 0:
            return 0
        if budget <= 0:
            return overlap_limit

        stable_chars = self._text_char_count(
            line.text for line in active_lines[source_pos:source_pos + stable_source_take]
        )
        if stable_chars >= budget:
            return 0

        overlap_take = 0
        total_chars = stable_chars
        while overlap_take < overlap_limit:
            line = active_lines[source_pos + stable_source_take + overlap_take]
            next_chars = len(line.text)
            if total_chars + next_chars > budget:
                break
            overlap_take += 1
            total_chars += next_chars
        return overlap_take

    def _bounded_target_end(self, target_segments: list[str], target_cursor: int, target_end: int):
        budget = max(0, int(self.settings.target_chunk_char_budget))
        if budget <= 0:
            return target_end

        total_chars = 0
        bounded_end = target_cursor
        while bounded_end < target_end:
            next_chars = len(target_segments[bounded_end])
            if bounded_end > target_cursor and total_chars + next_chars > budget:
                break
            bounded_end += 1
            total_chars += next_chars
        return max(target_cursor + 1, bounded_end)

    def _block_source_overlap(self):
        overlap = max(0, int(self.settings.block_source_overlap))
        if self.settings.block_source_lines <= 1:
            return 0
        return min(overlap, self.settings.block_source_lines - 1)

    def _finalized_boundary_groups(
        self,
        groups,
        active_units: list[units.AlignmentUnit],
        stable_source_end: int,
    ):
        finalized_source_end = stable_source_end
        for group in groups:
            group_source_positions = [
                active_units[index].source_pos
                for index in group.src_indices
            ]
            if group_source_positions and max(group_source_positions) >= stable_source_end:
                finalized_source_end = min(group_source_positions)
                break

        first_source_pos = active_units[0].source_pos if active_units else 0
        if finalized_source_end <= first_source_pos:
            return self._drop_trailing_target_only_groups(groups), stable_source_end

        committed_groups = []
        for group in groups:
            group_source_positions = [
                active_units[index].source_pos
                for index in group.src_indices
            ]
            if group_source_positions and max(group_source_positions) >= finalized_source_end:
                break
            committed_groups.append(group)

        return self._drop_trailing_target_only_groups(committed_groups), finalized_source_end

    def _drop_trailing_target_only_groups(self, groups):
        groups = list(groups)
        while groups and not groups[-1].src_indices and groups[-1].tgt_indices:
            groups.pop()
        return groups

    def _apply_nm_groups(
        self,
        output_rows: list[OutputRow],
        active_units: list[units.AlignmentUnit],
        target_segments: list[str],
        groups,
    ):
        last_source_index: int | None = None
        consumed_targets = 0
        row_targets: dict[int, list[str]] = {}
        row_additions: dict[int, list[str]] = {}
        touched_rows: set[int] = set()

        for group in groups:
            if group.tgt_indices:
                consumed_targets = max(consumed_targets, max(group.tgt_indices) + 1)

            target_group = [target_segments[index] for index in group.tgt_indices]
            source_units = [active_units[index] for index in group.src_indices]
            if not group.src_indices:
                addition_index = last_source_index
                if addition_index is None:
                    addition_index = rows.first_nonblank_row_index(output_rows)
                if addition_index is not None:
                    row_additions.setdefault(addition_index, []).extend(target_group)
                continue

            source_row_indices = self._unit_row_indices(source_units)
            touched_rows.update(source_row_indices)
            if not group.tgt_indices:
                last_source_index = source_row_indices[-1]
                continue

            if len(source_row_indices) == 1:
                row_targets.setdefault(source_row_indices[0], []).extend(target_group)
            else:
                source_group = [
                    text.join_fragments(unit.text for unit in source_units if unit.index == row_index)
                    for row_index in source_row_indices
                ]
                assigned = self._assigner.assign_group(source_group, target_group)
                for row_index, assigned_fragments in zip(source_row_indices, assigned):
                    row_targets.setdefault(row_index, []).extend(assigned_fragments)
            last_source_index = source_row_indices[-1]

        self._fill_aggregated_rows(
            output_rows,
            sorted(touched_rows | set(row_additions)),
            row_targets,
            row_additions,
        )
        return consumed_targets

    def _fill_aggregated_rows(
        self,
        output_rows: list[OutputRow],
        touched_rows: list[int],
        row_targets: dict[int, list[str]],
        row_additions: dict[int, list[str]],
    ):
        for row_index in touched_rows:
            target_fragments = row_targets.get(row_index, [])
            if target_fragments:
                target, status, similarity = self._assigner.format_assignment(
                    output_rows[row_index].source,
                    target_fragments,
                )
                output_rows[row_index].target = target
                output_rows[row_index].status = status
                output_rows[row_index].similarity = similarity
                output_rows[row_index].target_fragments = list(target_fragments)
            else:
                rows.mark_missing(output_rows, [row_index])

            additions = row_additions.get(row_index, [])
            if additions:
                rows.attach_addition(output_rows, row_index, text.join_fragments(additions))

    def _unit_row_indices(self, source_units: list[units.AlignmentUnit]):
        row_indices = []
        seen = set()
        for unit in source_units:
            if unit.index in seen:
                continue
            seen.add(unit.index)
            row_indices.append(unit.index)
        return row_indices

    def _recover_missing_runs(self, output_rows: list[OutputRow]):
        recovered_count = 0
        index = 0
        while index < len(output_rows):
            if output_rows[index].status != "missing":
                index += 1
                continue

            run_start = index
            while index < len(output_rows) and output_rows[index].status == "missing":
                index += 1
            run_end = index
            missing_count = run_end - run_start
            if missing_count > MISSING_RECOVERY_MAX_MISSING_ROWS:
                continue
            if run_end >= len(output_rows):
                continue

            anchor_index = run_end
            if self._recover_missing_run(output_rows, run_start, anchor_index):
                recovered_count += missing_count
                index = anchor_index + 1

        return recovered_count

    def _recover_missing_run(
        self,
        output_rows: list[OutputRow],
        run_start: int,
        anchor_index: int,
    ):
        anchor_row = output_rows[anchor_index]
        target_fragments = list(anchor_row.target_fragments or [])
        if not target_fragments or not self._is_missing_recovery_anchor(anchor_row):
            return False

        row_indices = list(range(run_start, anchor_index + 1))
        source_texts = [output_rows[index].source for index in row_indices]
        assigned = self._assigner.assign_group(source_texts, target_fragments)
        if any(not fragments for fragments in assigned[:-1]):
            return False

        current_score = (
            (len(row_indices) - 1) * self.settings.assignment.empty_score
            + self._assignment_score(anchor_row.source, target_fragments)
        )
        candidate_scores = [
            self._assignment_score(source_text, fragments)
            for source_text, fragments in zip(source_texts, assigned)
        ]
        minimum_score = min(candidate_scores, default=0.0)
        minimum_required_score = max(
            self.settings.mistranslation_threshold,
            MISSING_RECOVERY_MIN_SCORE,
        )
        if minimum_score < minimum_required_score:
            return False
        if sum(candidate_scores) <= current_score + MISSING_RECOVERY_MIN_GAIN:
            return False

        for row_index, assigned_fragments in zip(row_indices, assigned):
            target, status, similarity = self._assigner.format_assignment(
                output_rows[row_index].source,
                assigned_fragments,
            )
            output_rows[row_index].target = target
            output_rows[row_index].status = status
            output_rows[row_index].similarity = similarity
            output_rows[row_index].target_fragments = list(assigned_fragments)

        return True

    def _is_missing_recovery_anchor(self, row: OutputRow):
        if row.status not in {"ok", "ok_with_addition", "mistranslation"}:
            return False
        return bool(row.source.strip() and row.target_fragments)

    def _recover_swallowed_missing_runs(self, output_rows: list[OutputRow]):
        recovered_count = 0
        index = 0
        while index < len(output_rows) - 1:
            if not self._is_missing_recovery_anchor(output_rows[index]):
                index += 1
                continue

            run_start = index + 1
            if output_rows[run_start].status != "missing":
                index += 1
                continue

            run_end = run_start
            while run_end < len(output_rows) and output_rows[run_end].status == "missing":
                run_end += 1

            missing_count = run_end - run_start
            if missing_count <= MISSING_RECOVERY_MAX_MISSING_ROWS and self._recover_swallowed_missing_run(
                output_rows,
                index,
                run_end,
            ):
                recovered_count += missing_count

            index = run_end

        return recovered_count

    def _recover_swallowed_missing_run(
        self,
        output_rows: list[OutputRow],
        anchor_index: int,
        run_end: int,
    ):
        anchor_row = output_rows[anchor_index]
        target_fragments = list(anchor_row.target_fragments or [])
        if not target_fragments or not self._is_missing_recovery_anchor(anchor_row):
            return False

        row_indices = list(range(anchor_index, run_end))
        source_texts = [output_rows[index].source for index in row_indices]
        assigned = self._assigner.assign_group(source_texts, target_fragments)
        if any(not fragments for fragments in assigned):
            return False

        current_score = (
            self._assignment_score(anchor_row.source, target_fragments)
            + (len(row_indices) - 1) * self.settings.assignment.empty_score
        )
        candidate_scores = [
            self._assignment_score(source_text, fragments)
            for source_text, fragments in zip(source_texts, assigned)
        ]
        minimum_score = min(candidate_scores, default=0.0)
        minimum_required_score = max(
            self.settings.mistranslation_threshold,
            MISSING_RECOVERY_MIN_SCORE,
        )
        if minimum_score < minimum_required_score:
            return False
        recovered_scores = candidate_scores[1:]
        recovered_required_score = max(
            minimum_required_score,
            SWALLOWED_MISSING_MIN_SCORE,
        )
        if min(recovered_scores, default=0.0) < recovered_required_score:
            return False
        if sum(candidate_scores) <= current_score + MISSING_RECOVERY_MIN_GAIN:
            return False

        for row_index, assigned_fragments in zip(row_indices, assigned):
            target, status, similarity = self._assigner.format_assignment(
                output_rows[row_index].source,
                assigned_fragments,
            )
            output_rows[row_index].target = target
            output_rows[row_index].status = status
            output_rows[row_index].similarity = similarity
            output_rows[row_index].target_fragments = list(assigned_fragments)

        return True

    def _recover_weak_windows(self, output_rows: list[OutputRow]):
        recovered_count = 0
        index = 0
        while index < len(output_rows):
            if not self._is_weak_window_seed(output_rows[index]):
                index += 1
                continue

            run_start = index
            while index < len(output_rows) and self._is_weak_window_seed(output_rows[index]):
                index += 1
            run_end = index
            window_start = max(0, run_start - WEAK_WINDOW_CONTEXT_ROWS)
            window_end = min(len(output_rows), run_end + WEAK_WINDOW_CONTEXT_ROWS)
            if window_end - window_start > WEAK_WINDOW_MAX_ROWS:
                continue

            if self._recover_weak_window(output_rows, window_start, window_end):
                recovered_count += 1
                index = window_end

        return recovered_count

    def _recover_weak_window(
        self,
        output_rows: list[OutputRow],
        window_start: int,
        window_end: int,
    ):
        window_rows = output_rows[window_start:window_end]
        if len(window_rows) < 2:
            return False
        if any(not self._is_weak_window_row(row) for row in window_rows):
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
        assigned = self._assigner.assign_group(source_texts, target_fragments)
        if any(not fragments for fragments in assigned):
            return False
        if assigned == current_fragments:
            return False

        current_scores = [
            self._assignment_score(row.source, fragments)
            for row, fragments in zip(window_rows, current_fragments)
        ]
        candidate_scores = [
            self._assignment_score(source_text, fragments)
            for source_text, fragments in zip(source_texts, assigned)
        ]

        current_bad_count = self._weak_score_count(current_scores)
        candidate_bad_count = self._weak_score_count(candidate_scores)
        current_total = sum(current_scores)
        candidate_total = sum(candidate_scores)
        if candidate_bad_count > current_bad_count:
            return False
        if (
            candidate_bad_count == current_bad_count
            and candidate_total <= current_total + WEAK_WINDOW_MIN_GAIN
        ):
            return False
        if min(candidate_scores, default=0.0) < self.settings.mistranslation_threshold:
            return False

        for row, assigned_fragments in zip(window_rows, assigned):
            target, status, similarity = self._assigner.format_assignment(
                row.source,
                assigned_fragments,
            )
            row.target = target
            row.status = status
            row.similarity = similarity
            row.target_fragments = list(assigned_fragments)

        return True

    def _is_weak_window_seed(self, row: OutputRow):
        if row.status == "mistranslation":
            return True
        if row.status not in {"ok", "ok_with_addition"}:
            return False
        return row.similarity is not None and row.similarity < WEAK_WINDOW_SCORE_THRESHOLD

    def _is_weak_window_row(self, row: OutputRow):
        return (
            row.status in {"ok", "ok_with_addition", "mistranslation"}
            and bool(row.source.strip())
            and bool(row.target_fragments)
        )

    def _weak_score_count(self, scores: list[float]):
        return sum(score < WEAK_WINDOW_SCORE_THRESHOLD for score in scores)

    def _assignment_score(self, source_text: str, fragments: list[str]):
        if not fragments:
            return self.settings.assignment.empty_score
        target_text = text.join_fragments(fragments)
        score = self._similarity(source_text, target_text)
        return score - max(0, len(fragments) - 1) * self.settings.assignment.extra_fragment_penalty

    def _row_update_event(self, output_rows, changed_rows, stage, chunk_index=None):
        return {
            "type": "chunk",
            "stage": stage,
            "message": self._stage_message(stage),
            "chunk_index": chunk_index,
            "completed_count": rows.completed_row_count(output_rows),
            "source_line_count": len(output_rows),
            "rows": [row.to_dict() for row in changed_rows],
        }

    def _progress_event(self, output_rows, stage):
        return {
            "type": "progress",
            "stage": stage,
            "message": self._stage_message(stage),
            "completed_count": rows.completed_row_count(output_rows),
            "source_line_count": len(output_rows),
        }

    def _changed_row_update_event(self, output_rows, before, stage, chunk_index=None):
        changed_rows = [
            row
            for index, row in enumerate(output_rows)
            if row.to_dict() != before[index]
        ]
        return self._row_update_event(output_rows, changed_rows, stage, chunk_index)

    def _build_source_units(self, source_lines: list[SourceLine], source_pos_offset: int = 0):
        return units.build_alignment_units(
            source_lines,
            long_line_chars=self.settings.long_source_line_chars,
            unit_char_budget=self.settings.source_unit_char_budget,
            source_pos_offset=source_pos_offset,
        )

    def _text_char_count(self, values):
        return sum(len(str(value)) for value in values)

    def _stage_message(self, stage):
        return STAGE_MESSAGES.get(stage, stage.replace("_", " "))

    def _similarity(self, source_text: str, target_text: str):
        source_text = source_text.strip()
        target_text = target_text.strip()
        if not source_text or not target_text:
            return 0.0

        key = (source_text, target_text)
        if key in self._similarity_cache:
            return self._similarity_cache[key]

        source_embedding = self._embedding(source_text)
        target_embedding = self._embedding(target_text)
        similarity = float(np.dot(source_embedding, target_embedding))
        self._similarity_cache[key] = similarity
        return similarity

    def _embedding(self, value: str):
        if value not in self._embedding_cache:
            self._embedding_cache[value] = self.encoder.encode_sentences([value])[0]
        return self._embedding_cache[value]

    def _warm_embeddings(self, values):
        unique_values = []
        seen = set()
        for value in values:
            value = str(value).strip()
            if not value or value in self._embedding_cache or value in seen:
                continue
            seen.add(value)
            unique_values.append(value)

        if not unique_values:
            return

        embeddings = self.encoder.encode_sentences(unique_values)
        for value, embedding in zip(unique_values, embeddings):
            self._embedding_cache[value] = embedding

    def _normalize_source_text(self, source_text: str):
        return self._normalize_srt_text(source_text)

    def _normalize_target_text(self, target_text: str):
        return self._normalize_srt_text(target_text)

    def _normalize_srt_text(self, value: str):
        if not srt.looks_like_srt(value):
            return value

        cues = srt.parse_srt(value)
        if not cues:
            return value
        return srt.cues_to_source_text(cues)
