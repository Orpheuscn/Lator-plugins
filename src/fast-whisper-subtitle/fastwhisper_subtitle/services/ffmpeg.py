from __future__ import annotations

import os
from pathlib import Path


def resolve_ffmpeg_binary() -> str:
    configured = (
        os.environ.get("LATOR_PLUGIN_ASSET_FFMPEG")
        or os.environ.get("FFMPEG_BINARY")
        or ""
    ).strip()
    if not configured:
        return "ffmpeg"

    path = Path(configured).expanduser()
    if path.is_dir():
        return str(path / "ffmpeg")
    return str(path)
