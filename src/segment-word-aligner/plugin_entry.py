from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import transformers
import stopwordsiso  # type: ignore
from transformers import AutoModel, AutoTokenizer

try:
    import hanlp  # type: ignore
except Exception:  # pragma: no cover - optional fallback.
    hanlp = None

try:
    from sudachipy import dictionary as sudachi_dictionary  # type: ignore
    from sudachipy import tokenizer as sudachi_tokenizer  # type: ignore
except Exception:  # pragma: no cover - optional fallback.
    sudachi_dictionary = None
    sudachi_tokenizer = None


CAPABILITY_BUILD_WORD_ALIGNMENTS = "build-word-alignments"
MODEL_ASSET_ENV = "LATOR_PLUGIN_ASSET_AWESOME_ALIGN_WITH_CO"
INTERNAL_SETTINGS_ENV = "LATOR_SEGMENT_WORD_ALIGNER_SETTINGS"
MODEL_ID = "aneuraz/awesome-align-with-co"
DEFAULT_ALIGN_LAYER = 8
DEFAULT_MAX_WORDPIECES = 510
DEFAULT_SOFTMAX_THRESHOLD = 0.001
DEFAULT_MIN_WORD_SCORE = 0.15
DEFAULT_MIN_PHRASE_SCORE = 0.15
DEFAULT_MAX_PHRASE_SOURCE_WORDS = 6
DEFAULT_MAX_PHRASE_TARGET_WORDS = 10
HANLP_ZH_TOK_ENV = "LATOR_HANLP_ZH_TOKENIZER"
HANLP_ZH_TOK_MODEL = "https://file.hankcs.com/hanlp/tok/coarse_electra_small_20220616_012050.zip"
# Plugin assets downloaded at install time so the tokenizer never fetches on demand.
HANLP_ZH_TOK_ASSET_ENV = "LATOR_PLUGIN_ASSET_HANLP_COARSE_ELECTRA_SMALL_ZH"
HANLP_ZH_TOK_ASSET_SUBDIR = "coarse_electra_small_20220616_012050"
HANLP_ELECTRA_ASSET_ENV = "LATOR_PLUGIN_ASSET_CHINESE_ELECTRA_180G_SMALL_DISCRIMINATOR"
HANLP_CHAR_TABLE_ASSET_ENV = "LATOR_PLUGIN_ASSET_HANLP_CHAR_TABLE"
HANLP_CHAR_TABLE_FILENAME = "char_table_20210602_202632.json"
SUDACHI_JA_CONFIG_ENV = "LATOR_SUDACHI_JA_CONFIG"
SUDACHI_JA_SPLIT_MODE_ENV = "LATOR_SUDACHI_JA_SPLIT_MODE"

hanlp_zh_tokenizer: Any | None = None
sudachi_ja_tokenizer: Any | None = None


@dataclass(frozen=True)
class Segment:
    segment_id: str
    segment_index: int
    source_text: str
    translated_text: str


@dataclass(frozen=True)
class LexToken:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class EncodedSide:
    tokens: list[LexToken]
    input_ids: torch.Tensor
    word_ids: list[int]


@dataclass(frozen=True)
class WordLink:
    src_index: int
    tgt_index: int
    score: float


@dataclass(frozen=True)
class TranslationUnit:
    source_indexes: frozenset[int]
    target_indexes: frozenset[int]
    score: float


class AwesomeAligner:
    def __init__(
        self,
        model_dir: Path,
        align_layer: int = DEFAULT_ALIGN_LAYER,
        max_wordpieces: int = DEFAULT_MAX_WORDPIECES,
        softmax_threshold: float = DEFAULT_SOFTMAX_THRESHOLD,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.align_layer = align_layer
        self.max_wordpieces = max_wordpieces
        self.softmax_threshold = softmax_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            local_files_only=True,
            use_fast=True,
        )
        self.model = AutoModel.from_pretrained(
            self.model_dir,
            local_files_only=True,
            output_hidden_states=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def align(
        self,
        source_text: str,
        target_text: str,
        source_language: str,
        target_language: str,
    ) -> tuple[list[LexToken], list[LexToken], list[WordLink]]:
        source = encode_side(self.tokenizer, source_text, source_language, self.max_wordpieces)
        target = encode_side(self.tokenizer, target_text, target_language, self.max_wordpieces)
        if not source.word_ids or not target.word_ids:
            return source.tokens, target.tokens, []

        with torch.no_grad():
            src_hidden = self._hidden_states(source.input_ids)
            tgt_hidden = self._hidden_states(target.input_ids)

        scores = torch.matmul(src_hidden, tgt_hidden.transpose(0, 1))
        src_probs = torch.softmax(scores, dim=1)
        tgt_probs = torch.softmax(scores, dim=0)
        # awesome-align "softmax" extraction: keep wordpiece pairs that clear
        # the threshold in both directions. Unlike mutual argmax this allows
        # many-to-many links, which one-to-many translations (one French word
        # to a multi-character Chinese word and vice versa) require.
        mutual = (src_probs > self.softmax_threshold) & (tgt_probs > self.softmax_threshold)

        word_scores: dict[tuple[int, int], list[float]] = {}
        src_word_ids = source.word_ids
        tgt_word_ids = target.word_ids
        for src_piece_index, tgt_piece_index in mutual.nonzero().tolist():
            src_score = float(src_probs[src_piece_index, tgt_piece_index].item())
            tgt_score = float(tgt_probs[src_piece_index, tgt_piece_index].item())
            src_word = src_word_ids[src_piece_index]
            tgt_word = tgt_word_ids[tgt_piece_index]
            score = harmonic_mean(src_score, tgt_score)
            word_scores.setdefault((src_word, tgt_word), []).append(score)

        links = [
            WordLink(src_index=src, tgt_index=tgt, score=max(scores))
            for (src, tgt), scores in word_scores.items()
        ]
        links.sort(key=lambda item: (item.src_index, item.tgt_index))
        return source.tokens, target.tokens, links

    def _hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        ids = input_ids.unsqueeze(0).to(self.device)
        attention_mask = torch.ones_like(ids)
        outputs = self.model(input_ids=ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states
        layer_index = min(max(1, self.align_layer), len(hidden_states) - 1)
        # Drop [CLS] and [SEP] to keep indexes aligned with EncodedSide.word_ids.
        return hidden_states[layer_index][0, 1:-1].detach().cpu()


def handle(capability_id: str, params: dict[str, Any]) -> Iterable[dict[str, Any]]:
    if capability_id != CAPABILITY_BUILD_WORD_ALIGNMENTS:
        raise ValueError(f"Unknown capability: {capability_id}")
    return build_word_alignments(params)


def build_word_alignments(params: dict[str, Any]) -> Iterable[dict[str, Any]]:
    segments = collect_segments(params)
    if not segments:
        raise ValueError("No segments were provided by the host.")

    settings = read_settings()
    source_language = normalize_language_code(read_string(params, "sourceLanguageCode"))
    target_language = normalize_language_code(read_string(params, "targetLanguage"))
    include_word_alignments = bool(settings.get("includeWordAlignments", True))
    max_phrase_source_words = read_positive_int(settings.get("maxPhraseSourceWords"), DEFAULT_MAX_PHRASE_SOURCE_WORDS)
    max_phrase_target_words = DEFAULT_MAX_PHRASE_TARGET_WORDS

    yield {
        "stage": "prepare",
        "message": f"Preparing {len(segments)} segments for lexicon QA",
        "total": len(segments),
    }

    aligner = AwesomeAligner(read_model_dir())
    source_stopwords = load_stopwords(source_language)
    target_stopwords = load_stopwords(target_language)
    aligned_segments = []
    total_entries = 0

    for index, segment in enumerate(segments):
        entries = build_segment_entries(
            aligner,
            segment,
            source_stopwords,
            target_stopwords,
            source_language=source_language,
            target_language=target_language,
            include_word_alignments=include_word_alignments,
            max_phrase_source_words=max_phrase_source_words,
            max_phrase_target_words=max_phrase_target_words,
        )
        total_entries += len(entries)
        aligned_segments.append({
            "segmentId": segment.segment_id,
            "segmentIndex": segment.segment_index,
            "sourceText": segment.source_text,
            "targetText": segment.translated_text,
            "entries": entries,
        })
        yield {
            "stage": "alignment",
            "message": f"Checked {index + 1}/{len(segments)} segments",
            "current": index + 1,
            "total": len(segments),
        }

    result = {
        "projectId": read_string(params, "projectId"),
        "projectName": read_string(params, "projectName"),
        "mode": read_string(params, "mode") or "unknown",
        "model": MODEL_ID,
        "method": "awesome-align-softmax-translation-units-hanlp-zh",
        "alignLayer": DEFAULT_ALIGN_LAYER,
        "softmaxThreshold": DEFAULT_SOFTMAX_THRESHOLD,
        "minWordScore": DEFAULT_MIN_WORD_SCORE,
        "minPhraseScore": DEFAULT_MIN_PHRASE_SCORE,
        "sourceLanguageCode": source_language,
        "targetLanguage": target_language,
        "segments": aligned_segments,
        "summary": {
            "segmentCount": len(aligned_segments),
            "entryCount": total_entries,
        },
    }

    yield {
        "type": "done",
        "data": {
            "filename": build_output_filename(params),
            "content": result,
        },
    }


def build_segment_entries(
    aligner: AwesomeAligner,
    segment: Segment,
    source_stopwords: set[str],
    target_stopwords: set[str],
    source_language: str,
    target_language: str,
    include_word_alignments: bool,
    max_phrase_source_words: int,
    max_phrase_target_words: int,
) -> list[dict[str, Any]]:
    if not segment.source_text.strip() or not segment.translated_text.strip():
        return []

    source_tokens, target_tokens, links = aligner.align(
        segment.source_text,
        segment.translated_text,
        source_language,
        target_language,
    )
    links = [link for link in links if link.score >= DEFAULT_MIN_WORD_SCORE]
    if not links:
        return []

    units = build_translation_units(
        links,
        max_phrase_source_words=max_phrase_source_words,
        max_phrase_target_words=max_phrase_target_words,
    )

    entries: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int, str]] = set()
    for unit in units:
        entry = build_unit_entry(segment, source_tokens, target_tokens, unit)
        if entry is None:
            continue
        if all_stopwords(unit.source_indexes, source_tokens, source_stopwords):
            continue
        if all_stopwords(unit.target_indexes, target_tokens, target_stopwords):
            continue
        if entry["alignment_type"] == "word":
            if not include_word_alignments:
                continue
            if entry["score"] < DEFAULT_MIN_WORD_SCORE:
                continue
            if is_single_stopword(entry["src_text"], source_stopwords) or is_single_stopword(entry["tgt_text"], target_stopwords):
                continue
            if is_low_value_word_entry(entry["src_text"], entry["tgt_text"]):
                continue
        elif entry["score"] < DEFAULT_MIN_PHRASE_SCORE:
            continue
        add_unique_entry(entries, seen, entry)

    entries.sort(key=lambda item: (item["src_span"][0], item["src_span"][1], item["tgt_span"][0], item["alignment_type"]))
    return entries


def build_translation_units(
    links: list[WordLink],
    max_phrase_source_words: int,
    max_phrase_target_words: int,
) -> list[TranslationUnit]:
    """Group word links into minimal translation units.

    Links sharing a source or target word form one bipartite component. A
    component becomes a phrase unit only when it stays within the size limits
    and both spans are consistent (every word skipped inside a span is
    unaligned, so the span text never steals words from another unit).
    Oversized or inconsistent components fall back to per-link word units, so
    the dictionary never contains overlapping or interleaved spans.
    """
    components = connected_components(links)
    aligned_sources = {link.src_index for link in links}
    aligned_targets = {link.tgt_index for link in links}

    units: list[TranslationUnit] = []
    for component in components:
        source_indexes = frozenset(link.src_index for link in component)
        target_indexes = frozenset(link.tgt_index for link in component)
        score = sum(link.score for link in component) / len(component)
        if len(source_indexes) == 1 and len(target_indexes) == 1:
            units.append(TranslationUnit(source_indexes, target_indexes, score))
            continue

        consistent = (
            len(source_indexes) <= max_phrase_source_words and
            len(target_indexes) <= max_phrase_target_words and
            span_is_consistent(source_indexes, aligned_sources) and
            span_is_consistent(target_indexes, aligned_targets)
        )
        if consistent:
            units.append(TranslationUnit(source_indexes, target_indexes, score))
            continue

        for link in component:
            units.append(TranslationUnit(
                frozenset({link.src_index}),
                frozenset({link.tgt_index}),
                link.score,
            ))
    return units


def connected_components(links: list[WordLink]) -> list[list[WordLink]]:
    parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(node: tuple[str, int]) -> tuple[str, int]:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: tuple[str, int], right: tuple[str, int]) -> None:
        parent[find(left)] = find(right)

    for link in links:
        union(("s", link.src_index), ("t", link.tgt_index))

    grouped: dict[tuple[str, int], list[WordLink]] = {}
    for link in links:
        grouped.setdefault(find(("s", link.src_index)), []).append(link)
    return list(grouped.values())


def span_is_consistent(indexes: frozenset[int], aligned_indexes: set[int]) -> bool:
    lower = min(indexes)
    upper = max(indexes)
    return all(
        index in indexes or index not in aligned_indexes
        for index in range(lower, upper + 1)
    )


def build_unit_entry(
    segment: Segment,
    source_tokens: list[LexToken],
    target_tokens: list[LexToken],
    unit: TranslationUnit,
) -> dict[str, Any] | None:
    src_start = min(source_tokens[item].start for item in unit.source_indexes)
    src_end = max(source_tokens[item].end for item in unit.source_indexes)
    tgt_start = min(target_tokens[item].start for item in unit.target_indexes)
    tgt_end = max(target_tokens[item].end for item in unit.target_indexes)
    src_text = segment.source_text[src_start:src_end]
    tgt_text = segment.translated_text[tgt_start:tgt_end]
    if not src_text.strip() or not tgt_text.strip():
        return None
    if contains_phrase_break(src_text) or contains_phrase_break(tgt_text):
        return None

    is_word = len(unit.source_indexes) == 1 and len(unit.target_indexes) == 1
    return {
        "src_text": src_text,
        "src_span": [src_start, src_end],
        "tgt_text": tgt_text,
        "tgt_span": [tgt_start, tgt_end],
        "score": round_score(unit.score),
        "alignment_type": "word" if is_word else "phrase",
    }


def encode_side(tokenizer: Any, text: str, language_code: str, max_wordpieces: int) -> EncodedSide:
    tokens = tokenize_with_spans(text, language_code)
    pieces: list[int] = []
    word_ids: list[int] = []
    kept_tokens: list[LexToken] = []
    for token in tokens:
        token_pieces = tokenizer.encode(token.text, add_special_tokens=False)
        if not token_pieces:
            continue
        if len(pieces) + len(token_pieces) > max_wordpieces:
            break
        next_word_id = len(kept_tokens)
        pieces.extend(token_pieces)
        word_ids.extend([next_word_id] * len(token_pieces))
        kept_tokens.append(token)

    input_ids = torch.tensor(
        [tokenizer.cls_token_id, *pieces, tokenizer.sep_token_id],
        dtype=torch.long,
    )
    return EncodedSide(tokens=kept_tokens, input_ids=input_ids, word_ids=word_ids)


def tokenize_with_spans(text: str, language_code: str = "") -> list[LexToken]:
    if language_code == "zh":
        return tokenize_zh_with_hanlp(text)
    if language_code == "ja":
        return tokenize_ja_with_sudachi(text)

    tokens: list[LexToken] = []
    start: int | None = None
    chars: list[str] = []

    def flush(end: int) -> None:
        nonlocal start, chars
        if start is not None and chars:
            tokens.append(LexToken("".join(chars), start, end))
        start = None
        chars = []

    for index, char in enumerate(text):
        if is_cjk_char(char):
            flush(index)
            tokens.append(LexToken(char, index, index + 1))
            continue
        if is_word_char(char):
            if start is None:
                start = index
            chars.append(char)
            continue
        if is_internal_word_separator(char, text, index) and chars:
            chars.append(char)
            continue
        flush(index)
    flush(len(text))
    return tokens


def tokenize_zh_with_hanlp(text: str) -> list[LexToken]:
    tokenizer = get_hanlp_zh_tokenizer()
    words = tokenizer(text)
    tokens: list[LexToken] = []
    cursor = 0
    for word in words:
        if not isinstance(word, str) or not word.strip():
            continue
        start = text.find(word, cursor)
        if start < 0:
            start = text.find(word)
        if start < 0:
            continue
        end = start + len(word)
        cursor = end
        if not any(is_word_char(char) or is_cjk_char(char) for char in word):
            continue
        tokens.append(LexToken(word, start, end))
    return tokens


def tokenize_ja_with_sudachi(text: str) -> list[LexToken]:
    tokenizer = get_sudachi_ja_tokenizer()
    mode = get_sudachi_split_mode()
    tokens: list[LexToken] = []
    cursor = 0
    for morpheme in tokenizer.tokenize(text, mode):
        word = morpheme.surface()
        if not isinstance(word, str) or not word.strip():
            continue
        start = text.find(word, cursor)
        if start < 0:
            start = text.find(word)
        if start < 0:
            continue
        end = start + len(word)
        cursor = end
        if not any(is_word_char(char) or is_cjk_char(char) for char in word):
            continue
        tokens.append(LexToken(word, start, end))
    return tokens


def get_hanlp_zh_tokenizer() -> Any:
    global hanlp_zh_tokenizer
    if hanlp_zh_tokenizer is not None:
        return hanlp_zh_tokenizer
    if hanlp is None:
        raise RuntimeError("HanLP is required for Chinese tokenization but could not be imported.")
    ensure_transformers_encode_plus()
    model, overrides = resolve_hanlp_zh_model()
    try:
        hanlp_zh_tokenizer = hanlp.load(model, verbose=False, **overrides)
    except Exception as error:
        raise RuntimeError(
            "Unable to load the HanLP Chinese tokenizer. Install the plugin assets or "
            f"set {HANLP_ZH_TOK_ENV} to a local HanLP tokenizer path. "
            "This plugin intentionally does not use hanlp[full] because TensorFlow wheels are "
            "not available for the app's Python 3.12 runtime. "
            f"Original error: {error}"
        ) from error
    return hanlp_zh_tokenizer


def get_sudachi_ja_tokenizer() -> Any:
    global sudachi_ja_tokenizer
    if sudachi_ja_tokenizer is not None:
        return sudachi_ja_tokenizer
    if sudachi_dictionary is None:
        raise RuntimeError("SudachiPy is required for Japanese tokenization but could not be imported.")
    config_path = os.environ.get(SUDACHI_JA_CONFIG_ENV, "").strip()
    try:
        if config_path:
            sudachi_ja_tokenizer = sudachi_dictionary.Dictionary(config_path=config_path).create()
        else:
            sudachi_ja_tokenizer = sudachi_dictionary.Dictionary().create()
    except Exception as error:
        raise RuntimeError(
            "Unable to load the SudachiPy Japanese tokenizer. Install the plugin dependencies "
            f"or set {SUDACHI_JA_CONFIG_ENV} to a local Sudachi config path. "
            f"Original error: {error}"
        ) from error
    return sudachi_ja_tokenizer


def get_sudachi_split_mode() -> Any:
    if sudachi_tokenizer is None:
        raise RuntimeError("SudachiPy tokenizer module is required for Japanese tokenization.")
    raw_mode = os.environ.get(SUDACHI_JA_SPLIT_MODE_ENV, "C").strip().upper() or "C"
    if raw_mode == "A":
        return sudachi_tokenizer.Tokenizer.SplitMode.A
    if raw_mode == "B":
        return sudachi_tokenizer.Tokenizer.SplitMode.B
    return sudachi_tokenizer.Tokenizer.SplitMode.C


def resolve_hanlp_zh_model() -> tuple[str, dict[str, Any]]:
    """Resolve the tokenizer model path and any config overrides.

    Preference order:
    1. ``LATOR_HANLP_ZH_TOKENIZER`` — an explicit local tokenizer path set by the user.
    2. The bundled plugin assets, downloaded at install time. The tokenizer model, its
       transformer encoder (``hfl/chinese-electra-180g-small-discriminator``) and the
       char-normalization table are all served from local asset dirs, so nothing is
       fetched on demand.
    3. The original remote URL (legacy on-demand download) as a last resort.
    """
    explicit = os.environ.get(HANLP_ZH_TOK_ENV, "").strip()
    if explicit:
        return explicit, {}

    asset_root = os.environ.get(HANLP_ZH_TOK_ASSET_ENV, "").strip()
    if asset_root:
        model_dir = Path(asset_root)
        nested = model_dir / HANLP_ZH_TOK_ASSET_SUBDIR
        if nested.is_dir():
            model_dir = nested
        overrides: dict[str, Any] = {}
        electra = os.environ.get(HANLP_ELECTRA_ASSET_ENV, "").strip()
        if electra:
            overrides["transformer"] = electra
        char_table = resolve_hanlp_char_table_path()
        if char_table:
            # HanLP expects a callable transform object at prediction time; passing
            # the config dict through load overrides leaves a plain dict in the runtime config.
            from hanlp.common.transform import NormalizeCharacter  # type: ignore
            overrides["transform"] = NormalizeCharacter(char_table, src="token", dst="token")
        return str(model_dir), overrides

    return HANLP_ZH_TOK_MODEL, {}


def resolve_hanlp_char_table_path() -> str | None:
    root = os.environ.get(HANLP_CHAR_TABLE_ASSET_ENV, "").strip()
    if not root:
        return None
    path = Path(root)
    if path.is_file():
        return str(path)
    candidate = path / HANLP_CHAR_TABLE_FILENAME
    if candidate.is_file():
        return str(candidate)
    return None


def ensure_transformers_encode_plus() -> None:
    for class_name in (
        "PreTrainedTokenizerBase",
        "PreTrainedTokenizer",
        "BertTokenizer",
        "BertTokenizerFast",
    ):
        tokenizer_class = getattr(transformers, class_name, None)
        if tokenizer_class is None or hasattr(tokenizer_class, "encode_plus"):
            continue

        def encode_plus(self: Any, text: Any, text_pair: Any = None, **kwargs: Any) -> Any:
            return self(text, text_pair=text_pair, **kwargs)

        setattr(tokenizer_class, "encode_plus", encode_plus)


def is_word_char(char: str) -> bool:
    category = unicodedata.category(char)
    return category[0] in {"L", "M", "N"}


def is_internal_word_separator(char: str, text: str, index: int) -> bool:
    if char not in {"'", "-", "_", "’"}:
        return False
    previous_char = text[index - 1] if index > 0 else ""
    next_char = text[index + 1] if index + 1 < len(text) else ""
    return is_word_char(previous_char) and is_word_char(next_char)


def is_cjk_char(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x3400 <= codepoint <= 0x4DBF or
        0x4E00 <= codepoint <= 0x9FFF or
        0xF900 <= codepoint <= 0xFAFF or
        0x3040 <= codepoint <= 0x30FF
    )


def contains_phrase_break(text: str) -> bool:
    return any(char in ",，;；:：!?！？()（）[]【】{}《》<>—" for char in text)


def collect_segments(params: dict[str, Any]) -> list[Segment]:
    raw_segments = params.get("segments")
    if isinstance(raw_segments, list):
        segments = [normalize_segment(item, index) for index, item in enumerate(raw_segments)]
        return [segment for segment in segments if segment is not None]

    segment = normalize_segment(params, 0)
    return [segment] if segment is not None else []


def normalize_segment(value: Any, fallback_index: int) -> Segment | None:
    if not isinstance(value, dict):
        return None

    segment_id = read_string(value, "segmentId") or read_string(value, "segment_id") or str(fallback_index + 1)
    source_text = read_string(value, "sourceText") or read_string(value, "source_text") or ""
    translated_text = (
        read_string(value, "translatedText") or
        read_string(value, "targetText") or
        read_string(value, "translated_text") or
        ""
    )
    if not source_text.strip() or not translated_text.strip():
        return None

    segment_index = value.get("segmentIndex", value.get("segment_index", fallback_index))
    if not isinstance(segment_index, int):
        segment_index = fallback_index

    return Segment(
        segment_id=segment_id,
        segment_index=segment_index,
        source_text=source_text,
        translated_text=translated_text,
    )


def add_unique_entry(entries: list[dict[str, Any]], seen: set[tuple[int, int, int, int, str]], entry: dict[str, Any]) -> None:
    src_span = entry["src_span"]
    tgt_span = entry["tgt_span"]
    key = (src_span[0], src_span[1], tgt_span[0], tgt_span[1], entry["alignment_type"])
    if key in seen:
        return
    seen.add(key)
    entries.append(entry)


def all_stopwords(indexes: frozenset[int], tokens: list[LexToken], stopwords: set[str]) -> bool:
    return all(is_single_stopword(tokens[index].text, stopwords) for index in indexes)


def is_single_stopword(text: str, stopwords: set[str]) -> bool:
    normalized = normalize_stopword_text(text)
    return bool(normalized) and normalized in stopwords


def is_low_value_word_entry(source_text: str, target_text: str) -> bool:
    normalized_source = normalize_stopword_text(source_text)
    if len(normalized_source) <= 1 and not source_text.isnumeric():
        return True
    return is_single_cjk_token(target_text) and not source_text.isnumeric()


def is_single_cjk_token(text: str) -> bool:
    stripped = text.strip()
    return len(stripped) == 1 and is_cjk_char(stripped)


def load_stopwords(language_code: str) -> set[str]:
    if not language_code:
        return set()
    return {
        normalized
        for item in stopwordsiso.stopwords(language_code)
        if (normalized := normalize_stopword_text(item))
    }


def normalize_stopword_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold().strip()
    return re.sub(r"\s+", " ", text)


def normalize_language_code(value: str) -> str:
    value = value.strip().lower().replace("_", "-")
    if not value:
        return ""
    if "中文" in value or "简体" in value or "繁體" in value or "chinese" in value:
        return "zh"
    if "日语" in value or "日語" in value or "日本語" in value or "japanese" in value:
        return "ja"
    return value.split("-", 1)[0]


def harmonic_mean(left: float, right: float) -> float:
    if left <= 0 or right <= 0:
        return 0.0
    return (2 * left * right) / (left + right)


def round_score(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return round(min(1.0, max(0.0, value)), 4)


def build_output_filename(params: dict[str, Any]) -> str:
    mode = read_string(params, "mode") or "project"
    project_name = read_string(params, "projectName") or read_string(params, "projectId") or "project"
    safe_project_name = "".join(char if char.isalnum() or char in ("-", "_") else "-" for char in project_name)
    safe_project_name = safe_project_name.strip("-") or "project"
    return f"{safe_project_name}.{mode}.word-alignments.json"


def read_model_dir() -> Path:
    raw = os.environ.get(MODEL_ASSET_ENV, "").strip()
    if not raw:
        raise RuntimeError(
            f"Missing {MODEL_ASSET_ENV}. Install the plugin assets before running this capability."
        )
    return Path(raw)


def read_settings() -> dict[str, Any]:
    settings = parse_json_object(os.environ.get("LATOR_PLUGIN_SETTINGS", "{}"))
    settings.update(parse_json_object(os.environ.get(INTERNAL_SETTINGS_ENV, "{}")))
    return settings


def parse_json_object(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def read_positive_int(value: Any, fallback: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback


def read_string(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    return item.strip() if isinstance(item, str) else ""
