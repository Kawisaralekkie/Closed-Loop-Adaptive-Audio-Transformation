"""Report data contracts for per-chunk and per-run reporting.

Defines Pydantic models for ChunkReport and RunReport used to capture
processing details, quality scores, and failure information.

Requirements: 15.1, 15.2, 16.1
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field

from src.contracts.metrics_contracts import MetricsResult
from src.contracts.transform_contracts import TransformParams, TransformRecipeRef


class ChunkReport(BaseModel):
    """Per-chunk processing report."""

    chunk_id: str
    run_id: UUID
    had_speech: bool
    speech_ratio: float = 0.0
    vad_confidence: float = 0.0
    recipe_applied: TransformRecipeRef | None = None
    params_applied: TransformParams | None = None
    trials: int
    metrics: MetricsResult | None = None
    routing_decision: str  # "blurred" | "bypass"
    failure: str | None = None
    llm_token_usage: dict | None = None  # {"input_tokens", "output_tokens", "total_tokens"}
    trial_details: list[dict] | None = None  # per-trial: recipe, params, privacy_score, preserve_score
    llm_responses: list[dict] | None = None  # per-trial: raw LLM response / reasoning
    memory_snapshot: dict | None = None  # ExperienceMemory state after REFLECT
    classification_top3: list[dict] | None = None  # Top-3 YAMNet predictions on processed audio
    classification_top3_original: list[dict] | None = None  # Top-3 YAMNet predictions on original audio
    used_source_separation: bool = False  # Whether source separation was applied as pre-processing
    amplitude_stats: dict | None = None  # min/max/mean/rms of amplitude arrays before & after transform
    ground_truth_label: str | None = None  # File-level GT class (label1_audioset) — set during transfer-learning eval


class RunReport(BaseModel):
    """Comprehensive report for an entire processing run."""

    run_id: UUID
    source_id: str
    kb_version: str
    model_versions: dict  # {"silero_vad": "x.y", "yamnet": "x.y"}
    config_params: dict
    chunks: list[ChunkReport]
    total_chunks: int
    succeeded_chunks: int
    failed_chunks: int
    started_at: str = ""  # ISO 8601
    created_at: str  # ISO 8601 (finished)
    total_runtime_seconds: float = 0.0
    persisted_paths: list[str] = Field(default_factory=list)
    total_llm_token_usage: dict | None = None  # {"input_tokens": N, "output_tokens": N, "total_tokens": N}
