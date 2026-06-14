"""Bilingual layout alignment plugin entry."""

import os
import re

from .layout_aligner import LayoutPreservingAligner

ALIGNERS = {}
BILINGUAL_ALIGN_CAPABILITY_ID = "bilingual-align.align"
LABSE_ASSET_ENV = "LATOR_PLUGIN_ASSET_LABSE_ONNX"

def handle(capability_id, params):
    """Dispatch plugin capabilities from the Lator Python worker."""
    if capability_id != BILINGUAL_ALIGN_CAPABILITY_ID:
        raise ValueError(f"Unknown bilingual alignment capability: {capability_id}")

    payload = params if isinstance(params, dict) else {}
    return build_bilingual_align_events(
        source_text=str(payload.get("sourceText", "")),
        target_text=str(payload.get("targetText", "")),
    )


def align_bilingual_text(source_text, target_text):
    """Return the final alignment response for non-streaming callers."""
    result = None
    for event in build_bilingual_align_events(source_text, target_text):
        if event.get("type") == "done":
            result = event.get("data")
    return result


def build_bilingual_align_events(source_text, target_text):
    """Yield alignment progress events and the final row data."""
    if not source_text.strip():
        raise ValueError("原文不能为空")
    if not target_text.strip():
        raise ValueError("译文不能为空")

    yield _notification_event(
        "info",
        "模型正在运行，可离开本页面，对齐成功会通知您。",
    )
    aligner = _get_aligner()

    yield from aligner.align_events(source_text, target_text)
    yield _notification_event("success", "对齐完成。")


def _collapse_line(value):
    return re.sub(r"\s+", " ", str(value)).strip()


def _get_aligner():
    _ensure_labse_model_env()
    aligner = ALIGNERS.get("bert")
    if aligner is None:
        aligner = LayoutPreservingAligner(model_key="bert")
        ALIGNERS["bert"] = aligner
    return aligner


def _notification_event(level, message):
    return {
        "type": "notification",
        "level": level,
        "message": message,
    }


def _ensure_labse_model_env():
    if os.environ.get("LABSE_ONNX_DIR"):
        return

    asset_path = os.environ.get(LABSE_ASSET_ENV)
    if asset_path:
        os.environ["LABSE_ONNX_DIR"] = asset_path


__all__ = [
    "LayoutPreservingAligner",
    "align_bilingual_text",
    "build_bilingual_align_events",
    "handle",
]
