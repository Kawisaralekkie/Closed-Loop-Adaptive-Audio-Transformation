"""VAD (Voice Activity Detection) data contracts.

Defines Pydantic models for speech segments and VAD results produced by
SpeechScanTool.

Requirements: 2.2, 16.1
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SpeechSegment(BaseModel):
    """A detected speech region within an audio chunk."""

    start: float  # seconds
    end: float  # seconds
    confidence: float = Field(ge=0.0, le=1.0)


class VADResult(BaseModel):
    """Result of voice activity detection on a single AudioChunk."""

    chunk_id: str
    segments: list[SpeechSegment]
    speech_ratio: float = Field(ge=0.0, le=1.0)
    has_speech: bool  # convenience: len(segments) > 0
