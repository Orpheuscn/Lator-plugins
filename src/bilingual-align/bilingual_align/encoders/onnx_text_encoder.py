#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared ONNX text encoder utilities."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


class OnnxTextEncoder:
    """Base class for local ONNX sentence encoders."""

    model_name = "ONNX"
    env_dir_var = ""
    env_batch_var = ""
    env_thread_var = ""
    default_batch_size = 32
    model_file = "model.onnx"
    required_files = ("model.onnx", "tokenizer.json", "tokenizer_config.json")

    def __init__(
        self,
        model_path=None,
        max_length: int = 512,
        batch_size: int | None = None,
        thread_count: int | None = None,
    ):
        self.model_path = str(self._resolve_model_path(model_path))
        self.max_length = int(max_length)
        self._configured_batch_size = batch_size
        self._configured_thread_count = thread_count

        self.tokenizer = self._load_tokenizer()
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = self._thread_count()
        session_options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(Path(self.model_path) / self.model_file),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {input_meta.name for input_meta in self.session.get_inputs()}

    def encode(self, sentences):
        """Compatibility alias for the N:M alignment module."""
        return self.encode_sentences(sentences)

    def encode_sentences(self, sentences, batch_size=None):
        """Return L2-normalized embeddings for a list of texts."""
        sentences = [str(sentence) for sentence in sentences]
        if not sentences:
            return np.zeros((0, 0), dtype=np.float32)

        batch_size = batch_size or self._batch_size()
        batches = []
        for start in range(0, len(sentences), batch_size):
            batches.append(self._encode_batch(sentences[start:start + batch_size]))
        return np.vstack(batches).astype(np.float32)

    def _load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)

    def _encode_batch(self, sentences):
        raise NotImplementedError

    def _thread_count(self):
        if self._configured_thread_count is not None:
            return max(1, min(int(self._configured_thread_count), 4))

        raw_value = os.environ.get(self.env_thread_var, "2")
        try:
            thread_count = int(raw_value)
        except ValueError:
            thread_count = 2
        return max(1, min(thread_count, 4))

    def _batch_size(self):
        if self._configured_batch_size is not None:
            return max(1, min(int(self._configured_batch_size), 128))

        raw_value = os.environ.get(self.env_batch_var, str(self.default_batch_size))
        try:
            batch_size = int(raw_value)
        except ValueError:
            batch_size = self.default_batch_size
        return max(1, min(batch_size, 128))

    def _resolve_model_path(self, model_path):
        if model_path:
            return self._validate_model_path(Path(model_path).expanduser())

        env_path = os.environ.get(self.env_dir_var) if self.env_dir_var else None
        if env_path:
            return self._validate_model_path(Path(env_path).expanduser())

        raise FileNotFoundError(f"未配置 {self.model_name} 模型目录。")

    def _validate_model_path(self, model_path: Path):
        missing = [name for name in self.required_files if not (model_path / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"{self.model_name} 模型目录不完整: {model_path}. 缺少: {', '.join(missing)}"
            )
        return model_path

    @staticmethod
    def _normalize(embeddings):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return embeddings / norms
