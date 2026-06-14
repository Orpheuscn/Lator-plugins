#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model registry for shared alignment code."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .settings import EncoderSettings


APP_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ModelSpec:
    """Configuration and construction details for one aligner variant."""

    key: str
    label: str
    description: str
    config_path: Path
    create_encoder: Callable[[EncoderSettings], object]


def get_model_specs() -> dict[str, ModelSpec]:
    """Return supported alignment model variants."""
    return {
        "bert": ModelSpec(
            key="bert",
            label="Bert / LaBSE",
            description="LaBSE ONNX 版",
            config_path=APP_DIR / "config" / "bert" / "alignment.json",
            create_encoder=_create_bert_encoder,
        )
    }


def get_model_spec(model_key: str) -> ModelSpec:
    """Return one supported model spec or raise a readable error."""
    key = normalize_model_key(model_key)
    specs = get_model_specs()
    if key not in specs:
        available = ", ".join(specs)
        raise KeyError(f"未知模型版本: {model_key}. 可选: {available}")
    return specs[key]


def normalize_model_key(model_key: str | None) -> str:
    """Normalize public model aliases used by routes and the UI."""
    key = (model_key or "bert").strip().lower()
    aliases = {
        "labse": "bert",
        "bert": "bert",
    }
    return aliases.get(key, key)


def _create_bert_encoder(settings: EncoderSettings):
    from .encoders.labse_onnx_encoder import LaBSEOnnxEncoder

    return LaBSEOnnxEncoder(
        model_path=settings.model_path,
        max_length=settings.max_length,
        batch_size=settings.batch_size,
        thread_count=settings.thread_count,
    )
