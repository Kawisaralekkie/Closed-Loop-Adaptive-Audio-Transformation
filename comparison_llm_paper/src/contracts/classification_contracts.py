"""Classification data contracts for YAMNet outputs.

Defines Pydantic models for class predictions, embeddings, and the
overall YAMNet output produced by ClassificationTool.

Requirements: 5.2, 5.3, 16.1
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClassPrediction(BaseModel):
    """A single predicted sound class label with confidence."""

    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class YamnetOutput(BaseModel):
    """Full output from YAMNet classification on an AudioChunk."""

    chunk_id: str
    embeddings: list[list[float]]  # shape: (N_frames, 1024)
    predictions: list[ClassPrediction]


class EmbeddingResult(BaseModel):
    """Aggregated embedding vector for a single AudioChunk."""

    chunk_id: str
    embedding_vector: list[float]
