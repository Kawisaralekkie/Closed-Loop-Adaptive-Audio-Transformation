"""Metrics data contracts for privacy, utility, and quality evaluation.

Defines Pydantic models for privacy scores, psychoacoustic features,
preservation sub-scores, utility metrics, quality decisions, and the
composite metrics result.

Requirements: 6.1, 6.2, 6.3, 6.5, 6.6, 16.1, 17.5
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────────────
# privacy_target naming (renamed 2026):
#   OLD VOCAB     →   NEW VOCAB         threshold (privacy_score_min)
#   "high"        →   "moderate"        0.65
#   "very_high"   →   "high"            0.80
#
# Throughout the code we now use the NEW vocab. Legacy callers that still pass
# "very_high" are silently mapped to the new "high" via
# normalize_privacy_target() (see below). The string "high" is now interpreted
# as the NEW strict target (0.80) — old Step Function payloads / scripts must
# be updated to "moderate" to keep the previous lighter behaviour.
# ─────────────────────────────────────────────────────────────────────────────

# Canonical thresholds (NEW vocabulary)
PRIVACY_SCORE_MIN = {
    "moderate": 0.65,  # was "high" in old vocab
    "high": 0.80,      # was "very_high" in old vocab
}
PRESERVE_SCORE_MIN = 0.80
# Minimum speaker_privacy guard — prevents passing when identity leaks
SPEAKER_PRIVACY_MIN = 0.50


def normalize_privacy_target(privacy_target: str) -> str:
    """Map any legacy or new ``privacy_target`` string to the new vocabulary.

    Mapping:
        "very_high" → "high"        (legacy alias for the strict target)
        "high"      → "high"        (already new vocab)
        "moderate"  → "moderate"    (already new vocab)

    NOTE: The legacy lighter target ("high" with threshold 0.65 in OLD vocab)
    is NOT auto-mapped — callers must explicitly rename it to "moderate".
    Otherwise the same string "high" would silently swap meanings.
    """
    if privacy_target == "very_high":
        return "high"
    return privacy_target  # already new vocab or unknown — pass through


class PrivacyMetrics(BaseModel):
    """Privacy evaluation metrics.

    All fields are Optional: for chunks with NO speech (had_speech=False,
    routed to bypass) privacy metrics are UNDEFINED — there is no human voice
    to obscure — so they are reported as ``None`` rather than misleading
    zeros or ASR-hallucination garbage. They are populated (floats) only for
    chunks that actually contain speech.
    """

    wer: float | None = None
    cer: float | None = None
    speaker_privacy: float | None = None
    content_privacy: float | None = None  # 0.6 * WER + 0.4 * CER
    privacy_score: float | None = None  # 0.7 * content_privacy + 0.3 * speaker_privacy


class PsychoacousticFeatures(BaseModel):
    """Psychoacoustic feature measurements for a processed chunk."""

    short_term_loudness: float
    sharpness_proxy: float
    roughness_proxy: float
    fluctuation_proxy: float


class PreserveSubScores(BaseModel):
    """Individual sub-scores composing the preserve_score.

    NOTE: Field range constraints (ge/le) were removed so the true value
    range of each sub-score is observable in reports, rather than being
    silently clamped to [0, 1].
    """

    s_loud: float
    s_hf: float
    s_sc: float
    s_con: float
    s_psy: float  # derived from PsychoacousticFeatures


class UtilityMetrics(BaseModel):
    """Utility and preservation metrics.

    When a ground-truth label is available (transfer-learning evaluation),
    ``mAP``/``f1``/``accuracy`` are computed against it and the GT-related
    fields below are populated. Otherwise they fall back to a YAMNet
    confidence proxy and the GT fields stay ``None``.

    NOTE on aggregation: ``accuracy`` is real top-1 correctness per chunk and
    aggregates correctly by averaging. ``mAP`` is the per-clip average
    precision (reciprocal rank of the GT label). True dataset-level macro-F1
    and mAP should be computed post-hoc from ``ground_truth_label`` +
    ``predicted_label`` / ``classification_top3`` across the test set.
    """

    mAP: float
    f1: float
    accuracy: float
    preserve_score: float  # composite of sub-scores
    sub_scores: PreserveSubScores
    # Ground-truth based fields (populated only when GT label is provided)
    ground_truth_label: str | None = None
    predicted_label: str | None = None
    top1_correct: bool | None = None
    top3_correct: bool | None = None
    metrics_source: str = "proxy"  # "ground_truth" | "proxy"


class QualityDecision(BaseModel):
    """Pass/fail decision based on privacy and preservation thresholds."""

    privacy_pass: bool
    preserve_pass: bool
    overall_pass: bool  # privacy_pass AND preserve_pass
    privacy_target: str
    privacy_score_min: float
    preserve_score_min: float = PRESERVE_SCORE_MIN


class MetricsResult(BaseModel):
    """Composite metrics result for a single processed chunk."""

    chunk_id: str
    privacy: PrivacyMetrics
    utility: UtilityMetrics
    psychoacoustic: PsychoacousticFeatures
    decision: QualityDecision
