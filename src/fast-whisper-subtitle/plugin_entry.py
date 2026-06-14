from __future__ import annotations

from typing import Any, Iterable

from fastwhisper_subtitle.pipeline import retry_subtitle_segment, run_transcription_pipeline
from fastwhisper_subtitle.services.playback import play_subtitle_segment


CAPABILITY_TRANSCRIBE_MEDIA = "transcribe-media"
CAPABILITY_RETRY_SUBTITLE_SEGMENT = "retry-subtitle-segment"
CAPABILITY_PLAY_SUBTITLE_SEGMENT = "play-subtitle-segment"


def handle(capability_id: str, params: dict[str, Any]) -> Iterable[dict[str, Any]] | dict[str, str]:
    if capability_id == CAPABILITY_TRANSCRIBE_MEDIA:
        return run_transcription_pipeline(params)
    if capability_id == CAPABILITY_RETRY_SUBTITLE_SEGMENT:
        return retry_subtitle_segment(params)
    if capability_id == CAPABILITY_PLAY_SUBTITLE_SEGMENT:
        return play_subtitle_segment(params)
    raise ValueError(f"Unknown capability: {capability_id}")
