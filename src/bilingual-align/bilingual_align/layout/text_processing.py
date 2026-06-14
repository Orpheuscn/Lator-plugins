#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Text splitting, joining, and HTML marker helpers."""

from __future__ import annotations

import html
import re

from .types import SourceLine


def build_source_lines(text: str):
    lines = str(text).splitlines()
    if not lines and str(text).strip():
        lines = [str(text)]
    return [SourceLine(index=index, text=line.rstrip()) for index, line in enumerate(lines)]


def build_target_segments(text: str):
    text = strip_markdown_front_matter(text)
    segments = []
    for line in str(text).splitlines():
        line = collapse_spaces(line)
        if not line:
            continue
        segments.extend(split_strong(line))

    if not segments and str(text).strip():
        segments = split_strong(collapse_spaces(text))

    return [segment for segment in segments if segment]


def split_strong(text: str):
    text = collapse_spaces(text)
    if not text:
        return []

    pattern = r".+?[。！？!?]+[”’\"')）】》〉]*|.+?(?:\.\s+|$)|.+$"
    pieces = [match.group(0).strip() for match in re.finditer(pattern, text)]
    return [piece for piece in pieces if piece]


def split_fine(text: str, fine_whitespace_max_tokens: int):
    text = collapse_spaces(text)
    if not text:
        return []

    pieces = split_by_delimiters(text)
    if len(pieces) > 1:
        return pieces

    whitespace_pieces = [piece.strip() for piece in re.split(r"\s+", text) if piece.strip()]
    if 1 < len(whitespace_pieces) <= fine_whitespace_max_tokens:
        return whitespace_pieces

    return [text]


def split_by_delimiters(text: str):
    delimiters = set("。！？!?；;，,、：:")
    pieces = []
    buffer = []
    for char in text:
        buffer.append(char)
        if char in delimiters:
            piece = "".join(buffer).strip()
            if piece:
                pieces.append(piece)
            buffer = []

    tail = "".join(buffer).strip()
    if tail:
        pieces.append(tail)

    return pieces


def split_proportionally(text: str, count: int):
    text = text.strip()
    if count <= 1 or not text:
        return [text] if text else []

    tokens, separator = proportional_tokens(text)
    if len(tokens) < count:
        return [text]

    pieces = []
    for index in range(count):
        start = round(index * len(tokens) / count)
        end = round((index + 1) * len(tokens) / count)
        piece = separator.join(tokens[start:end]).strip()
        if piece:
            pieces.append(piece)

    return pieces if len(pieces) == count else [text]


def proportional_tokens(text: str):
    if should_split_words(text):
        return re.findall(r"\S+", text), " "
    return split_atomic(text), ""


def split_atomic(text: str):
    return re.findall(
        r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]|[^\W_]+(?:[’'][^\W_]+)*|[^\w\s]",
        text,
    )


def should_split_atomic(text: str):
    return has_cjk(text) or not re.search(r"\s", text)


def should_split_words(text: str):
    return bool(re.search(r"\s", text)) and not has_cjk(text)


def has_cjk(text: str):
    return bool(re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text))


def strip_markdown_front_matter(text: str):
    value = str(text)
    if value.startswith("\ufeff"):
        value = value[1:]
    match = re.match(r"\A---[ \t]*\n.*?\n---[ \t]*(?:\n|\Z)", value, flags=re.DOTALL)
    if not match:
        return value
    return value[match.end():].lstrip("\n")


def addition_marker(text: str):
    return f'<ins>{html_escape(text.strip())}</ins>'


def missing_marker(source_text: str):
    return f'<del>{html_escape(source_text.strip())}</del>'


def mistranslation_marker(target_text: str):
    return f'<mark>{html_escape(target_text.strip())}</mark>'


def html_escape(text: str):
    return html.escape(str(text), quote=True)


def join_fragments(parts):
    result = ""
    for part in parts:
        part = str(part).strip()
        if not part:
            continue
        if not result:
            result = part
        elif needs_space(result[-1], part[0]):
            result += " " + part
        else:
            result += part
    return result


def needs_space(left: str, right: str):
    if left in "([{【《“‘。，、；：！？":
        return False
    if right in ")]}】》。，、；：！？!?.,;:%”’":
        return False
    if is_cjk(left) or is_cjk(right):
        return False
    return True


def is_cjk(char: str):
    return bool(re.match(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", char))


def collapse_spaces(text: str):
    return re.sub(r"\s+", " ", str(text)).strip()
