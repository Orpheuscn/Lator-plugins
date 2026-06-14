"""Two-pass embedding N:M sentence alignment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AlignmentConfig:
    """Parameters for Bertalign-style two-pass dynamic programming."""

    max_group_size: int = 6
    top_k: int = 3
    window: int = 5
    skip: float = -1.0
    margin: bool = True
    length_penalty: bool = True


@dataclass(frozen=True)
class AlignmentGroup:
    """One contiguous source-target alignment group."""

    src_indices: list[int]
    tgt_indices: list[int]
    similarity: float | None


class EmbeddingNMAligner:
    """Run two-pass N:M alignment using an embedding encoder."""

    def __init__(self, encoder, config: AlignmentConfig | None = None):
        self.encoder = encoder
        self.config = config or AlignmentConfig()

    def align(self, source_sentences: list[str], target_sentences: list[str]):
        src_sents = _normalize_sentences(source_sentences)
        tgt_sents = _normalize_sentences(target_sentences)
        if not src_sents or not tgt_sents:
            return _empty_alignment(src_sents, tgt_sents)

        max_group_size = max(2, int(self.config.max_group_size))
        src_vecs, src_lens = _transform(self.encoder, src_sents, max_group_size - 1)
        tgt_vecs, tgt_lens = _transform(self.encoder, tgt_sents, max_group_size - 1)

        tgt_total_len = float(np.sum(tgt_lens[0]))
        char_ratio = float(np.sum(src_lens[0])) / tgt_total_len if tgt_total_len else 1.0

        first_alignment = _first_pass(
            src_vecs[0],
            tgt_vecs[0],
            top_k=max(1, int(self.config.top_k)),
        )
        raw_groups = _second_pass(
            src_vecs=src_vecs,
            tgt_vecs=tgt_vecs,
            src_lens=src_lens,
            tgt_lens=tgt_lens,
            first_alignment=first_alignment,
            max_group_size=max_group_size,
            window=max(1, int(self.config.window)),
            skip=float(self.config.skip),
            char_ratio=char_ratio,
            margin=bool(self.config.margin),
            length_penalty=bool(self.config.length_penalty),
        )

        return [
            _build_group(src_vecs, tgt_vecs, src_indices, tgt_indices)
            for src_indices, tgt_indices in raw_groups
        ]


def _normalize_sentences(sentences: list[str]):
    return [str(sentence).strip() for sentence in sentences if str(sentence).strip()]


def _empty_alignment(src_sents: list[str], tgt_sents: list[str]):
    groups = []
    groups.extend(AlignmentGroup([index], [], None) for index in range(len(src_sents)))
    groups.extend(AlignmentGroup([], [index], None) for index in range(len(tgt_sents)))
    return groups


def _transform(encoder, sentences: list[str], num_overlaps: int):
    overlaps = _yield_overlaps(sentences, num_overlaps)
    vectors = encoder.encode(overlaps)
    vectors = vectors.reshape(num_overlaps, len(sentences), vectors.shape[1])
    lengths = np.array([len(text.encode("utf-8")) for text in overlaps], dtype=np.float32)
    lengths = lengths.reshape(num_overlaps, len(sentences))
    return vectors.astype(np.float32), lengths


def _yield_overlaps(sentences: list[str], num_overlaps: int):
    lines = [sentence.strip() or "BLANK_LINE" for sentence in sentences]
    overlaps = []
    for overlap in range(1, num_overlaps + 1):
        overlaps.extend(_window_layer(lines, overlap))
    return overlaps


def _window_layer(lines: list[str], overlap: int):
    out = ["PAD"] * min(overlap - 1, len(lines))
    for index in range(len(lines) - overlap + 1):
        out.append(" ".join(lines[index:index + overlap]))
    return [line[:10000] for line in out]


def _first_pass(src_vecs: np.ndarray, tgt_vecs: np.ndarray, top_k: int):
    src_len = src_vecs.shape[0]
    tgt_len = tgt_vecs.shape[0]
    distances, indices = _find_top_k(src_vecs, tgt_vecs, top_k)
    align_types = _alignment_types(2)
    search_path = _first_search_path(src_len, tgt_len)

    cost = {(0, 0): 0.0}
    pointers: dict[tuple[int, int], tuple[int, int]] = {}
    for source_pos in range(src_len + 1):
        start, end = search_path[source_pos]
        for target_pos in range(start, end + 1):
            if (source_pos, target_pos) == (0, 0):
                continue
            best_score = -np.inf
            best_step = None
            for source_step, target_step in align_types:
                previous = (source_pos - source_step, target_pos - target_step)
                if previous not in cost:
                    continue
                score = cost[previous]
                if source_step > 0 and target_step > 0:
                    score += _top_k_score(source_pos - 1, target_pos - 1, distances, indices)
                if score > best_score:
                    best_score = score
                    best_step = (source_step, target_step)
            if best_step is not None:
                cost[(source_pos, target_pos)] = float(best_score)
                pointers[(source_pos, target_pos)] = best_step

    anchors = []
    source_pos, target_pos = src_len, tgt_len
    while (source_pos, target_pos) != (0, 0):
        source_step, target_step = pointers[(source_pos, target_pos)]
        if (source_step, target_step) == (1, 1):
            anchors.append((source_pos, target_pos))
        source_pos -= source_step
        target_pos -= target_step
    return anchors[::-1]


def _second_pass(
    src_vecs: np.ndarray,
    tgt_vecs: np.ndarray,
    src_lens: np.ndarray,
    tgt_lens: np.ndarray,
    first_alignment: list[tuple[int, int]],
    max_group_size: int,
    window: int,
    skip: float,
    char_ratio: float,
    margin: bool,
    length_penalty: bool,
):
    src_len = src_vecs.shape[1]
    tgt_len = tgt_vecs.shape[1]
    align_types = _alignment_types(max_group_size)
    search_path = _second_search_path(first_alignment, window, src_len, tgt_len)

    cost = {(0, 0): 0.0}
    pointers: dict[tuple[int, int], tuple[int, int]] = {}
    for source_pos in range(src_len + 1):
        start, end = search_path[source_pos]
        for target_pos in range(start, end + 1):
            if (source_pos, target_pos) == (0, 0):
                continue
            best_score = -np.inf
            best_step = None
            for source_step, target_step in align_types:
                previous = (source_pos - source_step, target_pos - target_step)
                if previous not in cost or not _within_path(previous[0], previous[1], search_path):
                    continue
                step_score = _step_score(
                    src_vecs,
                    tgt_vecs,
                    src_lens,
                    tgt_lens,
                    source_pos,
                    target_pos,
                    source_step,
                    target_step,
                    char_ratio,
                    skip,
                    margin,
                    length_penalty,
                )
                score = cost[previous] + step_score
                if score > best_score:
                    best_score = score
                    best_step = (source_step, target_step)
            if best_step is not None:
                cost[(source_pos, target_pos)] = float(best_score)
                pointers[(source_pos, target_pos)] = best_step

    groups = []
    source_pos, target_pos = src_len, tgt_len
    while (source_pos, target_pos) != (0, 0):
        source_step, target_step = pointers[(source_pos, target_pos)]
        groups.append((
            list(range(source_pos - source_step, source_pos)),
            list(range(target_pos - target_step, target_pos)),
        ))
        source_pos -= source_step
        target_pos -= target_step
    return groups[::-1]


def _alignment_types(max_group_size: int):
    alignment_types = [(0, 1), (1, 0)]
    for source_count in range(1, max_group_size):
        for target_count in range(1, max_group_size):
            if source_count + target_count <= max_group_size:
                alignment_types.append((source_count, target_count))
    return alignment_types


def _find_top_k(src_vecs: np.ndarray, tgt_vecs: np.ndarray, top_k: int):
    scores = np.matmul(src_vecs, tgt_vecs.T)
    top_k = min(top_k, scores.shape[1])
    if top_k == scores.shape[1]:
        indices = np.argsort(-scores, axis=1)
    else:
        indices = np.argpartition(-scores, kth=top_k - 1, axis=1)[:, :top_k]
        top_scores = np.take_along_axis(scores, indices, axis=1)
        order = np.argsort(-top_scores, axis=1)
        indices = np.take_along_axis(indices, order, axis=1)
    distances = np.take_along_axis(scores, indices, axis=1)
    return distances.astype(np.float32), indices.astype(np.int64)


def _top_k_score(source_index: int, target_index: int, distances: np.ndarray, indices: np.ndarray):
    matches = np.where(indices[source_index] == target_index)[0]
    if matches.size == 0:
        return 0.0
    return float(distances[source_index, matches[0]])


def _first_search_path(src_len: int, tgt_len: int, min_window_size: int = 250, percent: float = 0.06):
    if src_len == 0:
        return [(0, tgt_len)]
    window_size = max(min_window_size, int(max(src_len, tgt_len) * percent))
    ratio = tgt_len / src_len
    path = []
    for source_pos in range(src_len + 1):
        center = int(ratio * source_pos)
        path.append((max(0, center - window_size), min(center + window_size, tgt_len)))
    return path


def _second_search_path(
    alignments: list[tuple[int, int]],
    window: int,
    src_len: int,
    tgt_len: int,
):
    alignments = list(alignments)
    if not alignments:
        alignments.append((src_len, tgt_len))
    else:
        last_src, last_tgt = alignments[-1]
        if last_src != src_len:
            if last_tgt == tgt_len:
                alignments.pop()
            alignments.append((src_len, tgt_len))
        elif last_tgt != tgt_len:
            alignments.pop()
            alignments.append((src_len, tgt_len))

    previous_source = 0
    previous_target = 0
    path = []
    for source_pos, target_pos in alignments:
        lower = max(0, previous_target - window)
        upper = min(tgt_len, target_pos + window)
        path.extend((lower, upper) for _ in range(previous_source + 1, source_pos + 1))
        previous_source = source_pos
        previous_target = target_pos
    return [path[0]] + path if path else [(0, tgt_len)]


def _within_path(source_pos: int, target_pos: int, search_path: list[tuple[int, int]]):
    return 0 <= source_pos < len(search_path) and search_path[source_pos][0] <= target_pos <= search_path[source_pos][1]


def _step_score(
    src_vecs: np.ndarray,
    tgt_vecs: np.ndarray,
    src_lens: np.ndarray,
    tgt_lens: np.ndarray,
    source_pos: int,
    target_pos: int,
    source_step: int,
    target_step: int,
    char_ratio: float,
    skip: float,
    margin: bool,
    length_penalty: bool,
):
    if source_step == 0 or target_step == 0:
        return skip

    score = _similarity_score(src_vecs, tgt_vecs, source_pos, target_pos, source_step, target_step, margin)
    if length_penalty:
        score *= _length_penalty(
            src_lens,
            tgt_lens,
            source_pos,
            target_pos,
            source_step,
            target_step,
            char_ratio,
        )
    return float(score)


def _similarity_score(
    src_vecs: np.ndarray,
    tgt_vecs: np.ndarray,
    source_pos: int,
    target_pos: int,
    source_step: int,
    target_step: int,
    margin: bool,
):
    src_v = src_vecs[source_step - 1, source_pos - 1]
    tgt_v = tgt_vecs[target_step - 1, target_pos - 1]
    similarity = float(np.dot(src_v, tgt_v))
    if margin:
        target_neighbor = _neighbor_similarity(src_v, target_step, target_pos, tgt_vecs)
        source_neighbor = _neighbor_similarity(tgt_v, source_step, source_pos, src_vecs)
        similarity -= (target_neighbor + source_neighbor) / 2
    return similarity


def _neighbor_similarity(vector: np.ndarray, overlap: int, sentence_pos: int, db: np.ndarray):
    sentence_count = db.shape[1]
    left_index = sentence_pos - overlap
    right_index = sentence_pos + 1
    scores = []
    if left_index > 0:
        scores.append(float(np.dot(vector, db[0, left_index - 1])))
    if right_index <= sentence_count:
        scores.append(float(np.dot(vector, db[0, right_index - 1])))
    return float(np.mean(scores)) if scores else 0.0


def _length_penalty(
    src_lens: np.ndarray,
    tgt_lens: np.ndarray,
    source_pos: int,
    target_pos: int,
    source_step: int,
    target_step: int,
    char_ratio: float,
):
    source_len = float(src_lens[source_step - 1, source_pos - 1])
    target_len = float(tgt_lens[target_step - 1, target_pos - 1]) * char_ratio
    max_len = max(source_len, target_len)
    if max_len == 0:
        return 1.0
    return float(np.log2(1 + min(source_len, target_len) / max_len))


def _build_group(
    src_vecs: np.ndarray,
    tgt_vecs: np.ndarray,
    src_indices: list[int],
    tgt_indices: list[int],
):
    if not src_indices or not tgt_indices:
        return AlignmentGroup(src_indices, tgt_indices, None)

    source_step = len(src_indices)
    target_step = len(tgt_indices)
    source_end = src_indices[-1] + 1
    target_end = tgt_indices[-1] + 1
    similarity = float(np.dot(
        src_vecs[source_step - 1, source_end - 1],
        tgt_vecs[target_step - 1, target_end - 1],
    ))
    return AlignmentGroup(src_indices, tgt_indices, similarity)
