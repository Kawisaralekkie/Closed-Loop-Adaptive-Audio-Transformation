"""Source separation data contracts.

Defines the Pydantic model for nussl-based source separation results,
including speech/residual track paths, quality metrics, and fallback status.

Requirements: 2.1, 2.2, 2.3
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SourceSeparationResult(BaseModel):
    """Result of a nussl source separation process."""

    chunk_id: str
    speech_wav_path: str
    residual_wav_path: str
    remix_wav_path: str
    separation_quality_score: float = Field(ge=0.0, le=1.0)
    processing_time_ms: float = Field(ge=0.0)
    nussl_model_name: str
    fallback_used: bool = False
