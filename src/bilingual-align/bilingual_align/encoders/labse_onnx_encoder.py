#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LaBSE ONNX encoder used by the Bert layout aligner."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .onnx_text_encoder import OnnxTextEncoder


class LaBSEOnnxEncoder(OnnxTextEncoder):
    """Encode text with a local ONNX export of sentence-transformers/LaBSE."""

    model_name = "LaBSE ONNX"
    env_dir_var = "LABSE_ONNX_DIR"
    env_batch_var = "LABSE_ONNX_BATCH_SIZE"
    env_thread_var = "LABSE_ONNX_THREADS"
    default_batch_size = 64
    required_files = ("model.onnx", "tokenizer.json", "tokenizer_config.json", "vocab.txt")

    def __init__(
        self,
        model_path=None,
        max_length: int = 512,
        batch_size: int | None = None,
        thread_count: int | None = None,
    ):
        super().__init__(
            model_path=model_path,
            max_length=max_length,
            batch_size=batch_size,
            thread_count=thread_count,
        )
        print(f"[OK] LaBSE ONNX编码器初始化成功 (模型路径: {self.model_path})")

    def _load_tokenizer(self):
        return self._tokenizer_cls().from_pretrained(self.model_path, local_files_only=True)

    def _tokenizer_cls(self):
        from transformers import AutoTokenizer

        return AutoTokenizer

    def _resolve_model_path(self, model_path):
        if model_path:
            return self._validate_model_path(Path(model_path).expanduser())

        env_path = os.environ.get(self.env_dir_var)
        if env_path:
            return self._validate_model_path(Path(env_path).expanduser())

        raise FileNotFoundError(
            "未找到 LaBSE ONNX 模型目录。请设置环境变量 "
            f"{self.env_dir_var} 指向 Lator 管理的模型目录。"
        )

    def _encode_batch(self, sentences):
        inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )

        onnx_inputs = {
            name: inputs[name].astype(np.int64)
            for name in ("input_ids", "attention_mask", "token_type_ids")
            if name in inputs and name in self.input_names
        }
        outputs = self.session.run(None, onnx_inputs)
        return self._normalize(outputs[0][:, 0, :].astype(np.float32))
