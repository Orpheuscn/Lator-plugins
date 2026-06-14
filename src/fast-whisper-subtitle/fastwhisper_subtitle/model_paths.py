from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PLUGIN_ROOT = PACKAGE_DIR.parent
MODELS_DIR = PLUGIN_ROOT / "models"
DEFAULT_WHISPER_MODEL_NAME = "base"
DEFAULT_WHISPER_MODEL_ARG = DEFAULT_WHISPER_MODEL_NAME
WHISPER_BASE_ASSET_ENV = "LATOR_PLUGIN_ASSET_FASTER_WHISPER_BASE"
WHISPER_SMALL_ASSET_ENV = "LATOR_PLUGIN_ASSET_FASTER_WHISPER_SMALL"
WHISPER_MEDIUM_ASSET_ENV = "LATOR_PLUGIN_ASSET_FASTER_WHISPER_MEDIUM"
WHISPER_LARGE_V3_ASSET_ENV = "LATOR_PLUGIN_ASSET_FASTER_WHISPER_LARGE_V3"
WHISPER_TURBO_ASSET_ENV = "LATOR_PLUGIN_ASSET_FASTER_WHISPER_LARGE_V3_TURBO"
VAD_MODEL_ASSET_ENV = "LATOR_PLUGIN_ASSET_PYANNOTE_SEGMENTATION_3_0"
LANGUAGE_ID_MODEL_ASSET_ENV = "LATOR_PLUGIN_ASSET_LANGUAGE_ID_VOXLINGUA107_ECAPA"
VAD_MODEL_DIR = MODELS_DIR / "pyannote-segmentation-3.0"
LANGUAGE_ID_MODEL_DIR = MODELS_DIR / "language-id-voxlingua107-ecapa"


@dataclass(frozen=True)
class ResolvedWhisperModel:
    model: str
    local_files_only: bool


def require_model_dir(path: Path, required_files: tuple[str, ...], label: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} model directory does not exist: {path}")

    missing = [name for name in required_files if not (path / name).is_file()]
    if missing:
        missing_files = ", ".join(missing)
        raise FileNotFoundError(f"{label} model files are missing: {missing_files} ({path})")

    return path


def _resolve_relative_model_path(path: Path) -> Path:
    if path.is_absolute():
        return path

    if path.parts and path.parts[0] == "models":
        return PLUGIN_ROOT / path

    return path


def resolve_whisper_model(model: str | Path | None, allow_download: bool = False) -> ResolvedWhisperModel:
    model_text = str(model or DEFAULT_WHISPER_MODEL_ARG).strip() or DEFAULT_WHISPER_MODEL_ARG

    aliases: dict[str, Path | None] = {
        "base": _asset_or_local_path(WHISPER_BASE_ASSET_ENV, MODELS_DIR / "base"),
        "small": _asset_or_local_path(WHISPER_SMALL_ASSET_ENV, MODELS_DIR / "small"),
        "medium": _asset_or_local_path(WHISPER_MEDIUM_ASSET_ENV, MODELS_DIR / "medium"),
        "large": _asset_or_local_path(WHISPER_LARGE_V3_ASSET_ENV, MODELS_DIR / "large-v3"),
        "large-v3": _asset_or_local_path(WHISPER_LARGE_V3_ASSET_ENV, MODELS_DIR / "large-v3"),
        "turbo": _asset_or_local_path(WHISPER_TURBO_ASSET_ENV, MODELS_DIR / "turbo"),
    }

    alias_path = aliases.get(model_text)
    if alias_path is not None:
        return ResolvedWhisperModel(
            model=str(_require_whisper_model_path(alias_path)),
            local_files_only=True,
        )

    maybe_path = _resolve_relative_model_path(Path(model_text).expanduser())
    if maybe_path.exists() or maybe_path.is_absolute() or len(maybe_path.parts) > 1:
        return ResolvedWhisperModel(
            model=str(_require_whisper_model_path(maybe_path)),
            local_files_only=True,
        )

    return ResolvedWhisperModel(
        model=model_text,
        local_files_only=not allow_download,
    )


def resolve_whisper_model_path(model: str | Path | None) -> Path:
    resolved = resolve_whisper_model(model)
    if not resolved.local_files_only:
        raise FileNotFoundError(f"Whisper model is not available locally: {model}")
    return Path(resolved.model)


def resolve_vad_model_path(model_dir: str | Path | None = None) -> Path:
    path = _resolve_configured_model_path(
        model_dir,
        VAD_MODEL_ASSET_ENV,
        VAD_MODEL_DIR,
    )
    return require_model_dir(path.resolve(), ("pytorch_model.bin",), "VAD")


def resolve_language_id_model_path(model_dir: str | Path | None = None) -> Path:
    path = _resolve_configured_model_path(
        model_dir,
        LANGUAGE_ID_MODEL_ASSET_ENV,
        LANGUAGE_ID_MODEL_DIR,
    )
    return require_model_dir(
        path.resolve(),
        (
            "hyperparams.yaml",
            "embedding_model.ckpt",
            "classifier.ckpt",
            "label_encoder.txt",
        ),
        "Language identification",
    )


def _resolve_configured_model_path(
    configured_path: str | Path | None,
    asset_env_name: str,
    fallback: Path,
) -> Path:
    if configured_path:
        return _resolve_relative_model_path(Path(configured_path).expanduser())

    asset_path = os.environ.get(asset_env_name)
    if asset_path:
        return Path(asset_path).expanduser()

    return fallback


def _asset_or_local_path(asset_env_name: str, fallback: Path) -> Path | None:
    asset_path = os.environ.get(asset_env_name)
    if asset_path:
        return Path(asset_path).expanduser()
    if fallback.exists():
        return fallback
    return None


def _require_whisper_model_path(path: Path) -> Path:
    return require_model_dir(
        path.resolve(),
        ("model.bin", "config.json", "tokenizer.json"),
        "Whisper",
    )
