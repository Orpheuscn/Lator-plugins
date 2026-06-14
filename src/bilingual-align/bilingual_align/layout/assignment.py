#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assign target fragments inside one N:M source group."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..settings import AlignmentSettings
from .assignment_refinement import binary_split_candidates
from . import text_processing as text

PAIR_REFINEMENT_LOW_SCORE = 0.82
PAIR_REFINEMENT_MIN_GAIN = 0.025
PAIR_REFINEMENT_MIN_SIDE_GAIN = 0.005
PAIR_REFINEMENT_EMPTY_SIDE_MIN_SCORE = 0.80
NATURAL_BOUNDARY_BONUS = 0.03
ADDITION_CORE_DROP_MIN = 0.12
ADDITION_CORE_FRAGMENT_GAP = 0.25


class FragmentAssigner:
    """Split and assign target fragments back to source layout rows."""

    def __init__(
        self,
        settings: AlignmentSettings,
        similarity: Callable[[str, str], float],
        warm_embeddings: Callable[[list[str]], None],
    ):
        self.settings = settings
        self._similarity = similarity
        self._warm_embeddings = warm_embeddings

    def assign_group(self, source_texts: list[str], target_texts: list[str]):
        target_fragments = self._expand_target_fragments(target_texts, len(source_texts))
        source_count = len(source_texts)
        target_count = len(target_fragments)
        scores: dict[tuple[int, int, int], float] = {}

        def slice_score(source_index: int, start: int, end: int):
            key = (source_index, start, end)
            if key in scores:
                return scores[key]

            if start == end:
                score = self.settings.assignment.empty_score
            else:
                target_text = text.join_fragments(target_fragments[start:end])
                score = self._similarity(source_texts[source_index], target_text)
                score -= max(0, end - start - 1) * self.settings.assignment.extra_fragment_penalty

            scores[key] = score
            return score

        self._warm_assignment_embeddings(source_texts, target_fragments)

        dp = np.full((source_count + 1, target_count + 1), -np.inf, dtype=np.float32)
        back: list[list[int | None]] = [
            [None for _ in range(target_count + 1)]
            for _ in range(source_count + 1)
        ]
        dp[0][0] = 0.0

        for source_index in range(source_count):
            for used_count in range(target_count + 1):
                if not np.isfinite(dp[source_index][used_count]):
                    continue
                for next_count in range(used_count, target_count + 1):
                    score = dp[source_index][used_count] + slice_score(
                        source_index,
                        used_count,
                        next_count,
                    )
                    if score > dp[source_index + 1][next_count]:
                        dp[source_index + 1][next_count] = score
                        back[source_index + 1][next_count] = used_count

        assignments: list[list[str]] = [[] for _ in range(source_count)]
        used_count = target_count
        for source_index in range(source_count, 0, -1):
            previous_count = back[source_index][used_count]
            if previous_count is None:
                previous_count = used_count
            assignments[source_index - 1] = target_fragments[previous_count:used_count]
            used_count = previous_count

        return self._refine_assignments(source_texts, assignments)

    def format_assignment(self, source_text: str, fragments: list[str]):
        if not fragments:
            return text.missing_marker(source_text), "missing", None

        whole_target = text.join_fragments(fragments)
        whole_similarity = self._similarity(source_text, whole_target)
        if whole_similarity < self.settings.mistranslation_threshold:
            return text.mistranslation_marker(whole_target), "mistranslation", whole_similarity

        format_fragments = self._format_fragments(fragments)
        if len(format_fragments) == 1:
            return text.html_escape(format_fragments[0]), "ok", whole_similarity

        fragment_scores = [
            self._similarity(source_text, fragment)
            for fragment in format_fragments
        ]
        addition_flags = self._addition_fragment_flags(fragment_scores, whole_similarity)

        formatted = []
        has_addition = False
        for fragment, is_addition in zip(format_fragments, addition_flags):
            if is_addition:
                formatted.append(text.addition_marker(fragment))
                has_addition = True
            else:
                formatted.append(text.html_escape(fragment))

        status = "ok_with_addition" if has_addition else "ok"
        return text.join_fragments(formatted), status, whole_similarity

    def _addition_fragment_flags(self, fragment_scores: list[float], whole_similarity: float):
        flags = [False for _ in fragment_scores]
        best_score = max(fragment_scores, default=0.0)
        if best_score < self.settings.assignment.good_fragment_threshold:
            return flags

        for index, score in enumerate(fragment_scores):
            if score < self.settings.assignment.addition_fragment_threshold:
                flags[index] = True

        if whole_similarity <= best_score - ADDITION_CORE_DROP_MIN:
            for index, score in enumerate(fragment_scores):
                if score <= best_score - ADDITION_CORE_FRAGMENT_GAP:
                    flags[index] = True

        return flags

    def _format_fragments(self, fragments: list[str]):
        format_fragments = []
        for fragment in fragments:
            pieces = text.split_by_delimiters(fragment)
            format_fragments.extend(pieces if len(pieces) > 1 else [fragment])
        return format_fragments or list(fragments)

    def _expand_target_fragments(self, target_texts: list[str], source_count: int):
        fragments = []
        for target_text in target_texts:
            fragments.extend(
                text.split_fine(target_text, self.settings.fine_whitespace_max_tokens)
            )

        if len(fragments) > self.settings.max_assign_fragments:
            combined = text.join_fragments(target_texts)
            if source_count > 1:
                return text.split_proportionally(combined, source_count)
            return [combined]

        if len(fragments) >= source_count:
            return fragments

        combined = text.join_fragments(target_texts)
        atomic = text.split_atomic(combined)
        if (
            text.should_split_atomic(combined)
            and len(fragments) < len(atomic) <= self.settings.max_atomic_fragment_count
        ):
            return atomic

        proportional = text.split_proportionally(combined, source_count)
        return proportional if len(proportional) > len(fragments) else fragments

    def _warm_assignment_embeddings(self, source_texts: list[str], target_fragments: list[str]):
        candidate_texts = list(source_texts) + list(target_fragments)
        for start in range(len(target_fragments)):
            for end in range(start + 1, len(target_fragments) + 1):
                candidate_texts.append(text.join_fragments(target_fragments[start:end]))
        self._warm_embeddings(candidate_texts)

    def _refine_assignments(self, source_texts: list[str], assignments: list[list[str]]):
        refined = [list(fragments) for fragments in assignments]
        for _ in range(len(source_texts)):
            changed = False
            for index in range(len(source_texts) - 1):
                replacement = self._best_pair_replacement(source_texts, refined, index)
                if replacement:
                    refined[index], refined[index + 1] = replacement
                    changed = True
            if not changed:
                break
        return refined

    def _best_pair_replacement(
        self,
        source_texts: list[str],
        assignments: list[list[str]],
        left_index: int,
    ):
        right_index = left_index + 1
        left = assignments[left_index]
        right = assignments[right_index]
        combined = list(left) + list(right)
        if not combined:
            return None

        left_score = self._assignment_score(source_texts[left_index], left)
        right_score = self._assignment_score(source_texts[right_index], right)
        has_empty_side = not left or not right
        current_minimum = min(left_score, right_score)
        if not has_empty_side and current_minimum >= PAIR_REFINEMENT_LOW_SCORE:
            return None

        candidates = binary_split_candidates(combined)
        if not candidates:
            return None

        self._warm_pair_candidates(source_texts[left_index], source_texts[right_index], candidates)

        current_total = left_score + right_score
        best = None
        for candidate_left, candidate_right in candidates:
            candidate_left_score = self._assignment_score(source_texts[left_index], candidate_left)
            candidate_right_score = self._assignment_score(source_texts[right_index], candidate_right)
            candidate_minimum = min(candidate_left_score, candidate_right_score)
            if (
                (not left and candidate_left_score < PAIR_REFINEMENT_EMPTY_SIDE_MIN_SCORE)
                or (not right and candidate_right_score < PAIR_REFINEMENT_EMPTY_SIDE_MIN_SCORE)
            ):
                continue
            if candidate_minimum < self.settings.mistranslation_threshold:
                continue
            if (
                not has_empty_side
                and candidate_minimum + PAIR_REFINEMENT_MIN_SIDE_GAIN < current_minimum
            ):
                continue

            candidate_total = candidate_left_score + candidate_right_score
            candidate_ranking_score = candidate_total + _split_boundary_bonus(candidate_left)
            if (
                candidate_total > current_total + PAIR_REFINEMENT_MIN_GAIN
                and (best is None or candidate_ranking_score > best[2])
            ):
                best = (candidate_left, candidate_right, candidate_ranking_score)

        if best is None:
            return None
        return best[0], best[1]

    def _assignment_score(self, source_text: str, fragments: list[str]):
        if not fragments:
            return self.settings.assignment.empty_score
        target_text = text.join_fragments(fragments)
        score = self._similarity(source_text, target_text)
        return score - max(0, len(fragments) - 1) * self.settings.assignment.extra_fragment_penalty

    def _warm_pair_candidates(
        self,
        left_source: str,
        right_source: str,
        candidates: list[tuple[list[str], list[str]]],
    ):
        values = [left_source, right_source]
        for candidate_left, candidate_right in candidates:
            values.extend([
                text.join_fragments(candidate_left),
                text.join_fragments(candidate_right),
            ])
        self._warm_embeddings(values)


def _split_boundary_bonus(left_fragments: list[str]):
    left_text = text.join_fragments(left_fragments).strip()
    return NATURAL_BOUNDARY_BONUS if left_text.endswith(("，", ",", ";", "；", ":", "：")) else 0.0
