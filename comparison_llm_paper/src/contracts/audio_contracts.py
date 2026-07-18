"""Audio data contracts for the Privacy-Preserving Urban Soundscape system.

Defines Pydantic models for audio ingestion request/response, canonical audio
representation, and audio chunks used throughout the pipeline.

Requirements: 16.1, 16.2
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field


class AudioIngestRequest(BaseModel):
    """Incoming audio ingestion request from IoT/sensor API gateway."""

    source_id: str
    raw_audio_path: str
    format_hint: str = "wav"
    metadata: dict = Field(default_factory=dict)


class CanonicalAudio(BaseModel):
    """Canonicalized audio representation (16 kHz / 16-bit / mono)."""

    wav_path: str
    sample_rate: int = 16000
    bit_depth: int = 16
    channels: int = 1
    duration_seconds: float


class AudioChunk(BaseModel):
    """A single audio segment produced by PrepareDataTool."""

    chunk_id: str
    run_id: UUID
    wav_path: str
    start_time: float
    end_time: float
    duration: float
    metadata: dict = Field(default_factory=dict)


class AudioIngestResponse(BaseModel):
    """Response from PrepareDataTool after ingestion and chunking."""

    run_id: UUID
    source_id: str
    canonical_audio: CanonicalAudio
    chunks: list[AudioChunk]
