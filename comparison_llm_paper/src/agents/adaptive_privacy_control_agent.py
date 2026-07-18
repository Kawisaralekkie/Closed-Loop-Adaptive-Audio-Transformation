"""AdaptivePrivacyControlAgent — LLM-driven adaptive privacy control.

Implements a PLAN → ACT → GATE → (RETRY|DONE) → REFLECT state machine
that selects and tunes voice blurring recipes based on audio characteristics,
privacy_target level, and PolicyTransformationRules from the Knowledge Base.

NOTE (renamed 2026): privacy_target vocabulary
    OLD VOCAB        NEW VOCAB     privacy_score_min
    "high"      →    "moderate"    0.65
    "very_high" →    "high"        0.80
Legacy callers may still pass "high"/"very_high" — they are normalized via
``normalize_privacy_target()`` from src.contracts.metrics_contracts.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 10.2, 10.4
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from src.config import config
from src.contracts.audio_contracts import AudioChunk
from src.contracts.metrics_contracts import MetricsResult
from src.contracts.report_contracts import ChunkReport
from src.contracts.transform_contracts import (
    TransformParams,
    TransformRecipeRef,
    TransformResult,
)
from src.contracts.vad_contracts import VADResult
from src.knowledge_base.kb_loader import KnowledgeBase
from src.tools.mid_band_attenuation_tool import MidBandAttenuationTool
from src.tools.quality_evaluation_tool import QualityEvaluationTool
from src.tools.strong_blurring_tool import StrongBlurringTool

logger = logging.getLogger(__name__)

RECIPE_MID_BAND_ATTEN = "RECIPE_MID_BAND_ATTEN"
RECIPE_LOWPASS_HIGHPASS_MIX = "RECIPE_LOWPASS_HIGHPASS_MIX"
RECIPE_SOURCE_SEPARATION = "RECIPE_SOURCE_SEPARATION"


class AgentState(str, Enum):
    """States in the adaptive privacy control state machine."""

    PLAN = "PLAN"
    ACT = "ACT"
    GATE = "GATE"
    RETRY = "RETRY"
    DONE = "DONE"
    REFLECT = "REFLECT"


# ---------------------------------------------------------------------------
# Action space — 6-level retry ladder (preserve-friendly → privacy-heavy)
# ---------------------------------------------------------------------------
# Each level specifies the recipe, its parameters, and optional pitch shift.
# The agent climbs the ladder when the GATE check fails.
# ---------------------------------------------------------------------------

RETRY_LADDER: list[dict[str, Any]] = [
    # Level 0 — MID_LIGHT + lowpass + pitch
    {
        "recipe": RECIPE_MID_BAND_ATTEN,
        "band_hz": (700, 2800),
        "atten_db": 18.0,
        "lowpass_cutoff": 1500,
        "pitch_shift_semitones": -2.0,
    },
    # Level 1 — MID_BALANCED + lowpass + pitch
    {
        "recipe": RECIPE_MID_BAND_ATTEN,
        "band_hz": (700, 2750),
        "atten_db": 24.0,
        "lowpass_cutoff": 1200,
        "pitch_shift_semitones": -3.0,
    },
    # Level 2 — MID_STRONG + lowpass + pitch
    {
        "recipe": RECIPE_MID_BAND_ATTEN,
        "band_hz": (700, 2700),
        "atten_db": 27.0,
        "lowpass_cutoff": 1100,
        "pitch_shift_semitones": -3.5,
    },
    # Level 3 — MID_BASELINE + lowpass + pitch
    {
        "recipe": RECIPE_MID_BAND_ATTEN,
        "band_hz": (700, 2700),
        "atten_db": 30.0,
        "lowpass_cutoff": 950,
        "pitch_shift_semitones": -4.0,
    },
    # Level 4 — STRONG_BLUR + band attenuation + pitch shift
    {
        "recipe": RECIPE_LOWPASS_HIGHPASS_MIX,
        "lowpass_cutoff": 600,
        "lowpass_mix": 0.55,
        "noise_snr_db": 3.0,
        "band_hz": (600, 2800),
        "atten_db": 35.0,
        "pitch_shift_semitones": -5.0,
    },
    # Level 5 — STRONG_BLUR (hybrid, aggressive) + pitch shift
    {
        "recipe": RECIPE_LOWPASS_HIGHPASS_MIX,
        "lowpass_cutoff": 500,
        "lowpass_mix": 0.55,
        "noise_snr_db": 0.0,
        "band_hz": (700, 2700),
        "atten_db": 40.0,
        "pitch_shift_semitones": -5.0,
    },
]


def select_recipe(
    vad_result: VADResult,
    privacy_target: str,
    kb: KnowledgeBase,
) -> list[int]:
    """Select retry level order based on audio features.

    High speech confidence → start with StrongBlur (4,5) then fall back to MidBand (3,2,1,0)
    Low speech confidence  → start with MidBand (0,1,2,3) then escalate to StrongBlur (4,5)

    Returns
    -------
    list[int]
        Ordered list of RETRY_LADDER level indices to try.
    """
    vad_conf = _vad_conf_mean(vad_result)
    speech_ratio = vad_result.speech_ratio

    if speech_ratio >= 0.5 and vad_conf >= 0.6:
        # High speech → strong first, then fall back to lighter
        order = [5, 4, 3, 2, 1, 0]
    elif speech_ratio >= 0.3 or vad_conf >= 0.5:
        # Medium speech → start mid-strong, escalate
        order = [2, 3, 4, 5, 1, 0]
    else:
        # Low speech → start light, escalate
        order = [0, 1, 2, 3, 4, 5]

    # ─────────────────────────────────────────────────────────────────────
    # privacy_target naming (renamed 2026):
    #     OLD VOCAB        NEW VOCAB     privacy_score_min
    #     "high"      →    "moderate"    0.65
    #     "very_high" →    "high"        0.80
    # The "strict" target is now called "high" (was "very_high").
    # Bump only when the caller asks for the strict target.
    # We normalize first so legacy "very_high" maps to new "high" (strict)
    # and legacy "high" maps to new "moderate" (light, no bump).
    # ─────────────────────────────────────────────────────────────────────
    from src.contracts.metrics_contracts import normalize_privacy_target
    if normalize_privacy_target(privacy_target) == "high" and order[0] < 2:
        order = [2, 3, 4, 5, 1, 0]

    return order


def _vad_conf_mean(vad_result: VADResult) -> float:
    """Mean confidence across all speech segments."""
    if not vad_result.segments:
        return 0.0
    return sum(s.confidence for s in vad_result.segments) / len(vad_result.segments)


class AdaptivePrivacyControlAgent:
    """Adaptive privacy control agent with PLAN→ACT→GATE→(RETRY|DONE)→REFLECT.

    Parameters
    ----------
    mid_band_tool : MidBandAttenuationTool
        Tool for mid-band attenuation blurring.
    strong_blur_tool : StrongBlurringTool
        Tool for strong composite blurring.
    quality_tool : QualityEvaluationTool
        Tool for evaluating privacy and preservation quality.
    source_separation_tool : SourceSeparationTool | None
        Optional source separation tool.
    classification_tool : ClassificationTool | None
        Optional YAMNet classification tool. When provided, the GATE uses
        real classification output for s_con instead of a stub.
    max_trials : int | None
        Maximum ACT→GATE cycles. Defaults to ``config.agent.max_trials``.
    """

    def __init__(
        self,
        mid_band_tool: MidBandAttenuationTool,
        strong_blur_tool: StrongBlurringTool,
        quality_tool: QualityEvaluationTool,
        source_separation_tool=None,
        classification_tool=None,
        max_trials: int | None = None,
    ) -> None:
        self._mid_band_tool = mid_band_tool
        self._strong_blur_tool = strong_blur_tool
        self._quality_tool = quality_tool
        # ── SOURCE SEPARATION GLOBALLY DISABLED ──
        # SS is hard-disabled across the whole system. We ignore any tool
        # passed in so no code path can run source separation.
        self._source_separation_tool = None
        self._classification_tool = classification_tool
        self._max_trials = max_trials if max_trials is not None else config.agent.max_trials

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        chunk: AudioChunk,
        vad_result: VADResult,
        kb: KnowledgeBase,
        privacy_target: str,
    ) -> tuple[TransformResult, ChunkReport]:
        """Execute the adaptive privacy control state machine.

        Parameters
        ----------
        chunk : AudioChunk
            The audio chunk to process.
        vad_result : VADResult
            Speech detection result for the chunk.
        kb : KnowledgeBase
            Knowledge Base with policies and playbook.
        privacy_target : str
            ``"moderate"`` or ``"high"`` (new vocabulary).
            Legacy values ``"high"`` and ``"very_high"`` are also accepted
            and mapped to ``"moderate"`` / ``"high"`` respectively.

        Returns
        -------
        tuple[TransformResult, ChunkReport]
            The best blurring result and the chunk report.
        """
        state_trace: list[AgentState] = []
        trials: list[tuple[TransformResult, MetricsResult]] = []

        # --- PLAN (Req 8.2) ---
        state_trace.append(AgentState.PLAN)

        vad_conf = _vad_conf_mean(vad_result)

        # --- Source Separation pre-processing (if enabled and conditions met) ---
        source_sep_chunk = chunk  # default: use original chunk
        used_source_sep = False
        if (self._source_separation_tool is not None
                and vad_result.speech_ratio >= 0.3
                and vad_conf >= 0.6):
            logger.info(
                "SOURCE_SEP: speech_ratio=%.3f vad_conf=%.3f → applying source separation for chunk=%s",
                vad_result.speech_ratio, vad_conf, chunk.chunk_id,
            )
            try:
                sep_result = self._source_separation_tool.run(
                    wav_path=chunk.wav_path,
                    segments=vad_result.segments,
                    blur_method="strong_blur",
                    blur_params={"atten_db": 30.0, "noise_snr_db": 5.0, "pitch_shift_semitones": -4.0},
                    chunk_id=chunk.chunk_id,
                    trial=0,
                )
                # Use separated audio as input for retry ladder
                source_sep_chunk = AudioChunk(
                    chunk_id=chunk.chunk_id,
                    run_id=chunk.run_id,
                    wav_path=sep_result.blurred_wav_path,
                    start_time=chunk.start_time,
                    end_time=chunk.end_time,
                    duration=chunk.duration,
                    metadata=chunk.metadata,
                )
                used_source_sep = True
                logger.info("SOURCE_SEP: success for chunk=%s", chunk.chunk_id)
            except Exception as e:
                logger.warning("SOURCE_SEP: failed for chunk=%s: %s — continuing without", chunk.chunk_id, e)

        level_order = select_recipe(vad_result, privacy_target, kb)
        logger.info(
            "PLAN: level_order=%s for chunk=%s (target=%s, sr=%.3f, vad=%.3f)",
            level_order[:self._max_trials], chunk.chunk_id, privacy_target,
            vad_result.speech_ratio, vad_conf,
        )

        best_result: TransformResult | None = None
        best_metrics: MetricsResult | None = None
        best_privacy_score = -1.0

        for trial_idx in range(min(self._max_trials, len(level_order))):
            level = level_order[trial_idx]
            level_params = RETRY_LADDER[level]
            recipe_name = level_params["recipe"]

            # --- ACT (Req 8.3) ---
            state_trace.append(AgentState.ACT)
            input_chunk = source_sep_chunk if used_source_sep else chunk
            transform_result = self._execute_blurring(
                recipe_name, input_chunk, vad_result, level_params, trial_idx,
            )

            # --- GATE (Req 8.4) ---
            state_trace.append(AgentState.GATE)
            processed_chunk = AudioChunk(
                chunk_id=chunk.chunk_id,
                run_id=chunk.run_id,
                wav_path=transform_result.blurred_wav_path,
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                duration=chunk.duration,
                metadata=chunk.metadata,
            )

            # Use real YAMNet classification if tool is available;
            # otherwise fall back to stub (backward-compat).
            from src.contracts.classification_contracts import (
                ClassPrediction,
                YamnetOutput,
            )

            if self._classification_tool is not None:
                classification_result = self._classification_tool.run(processed_chunk)
            else:
                classification_result = YamnetOutput(
                    chunk_id=chunk.chunk_id,
                    embeddings=[[0.0]],
                    predictions=[ClassPrediction(label="unknown", confidence=0.5)],
                )

            metrics_result = self._quality_tool.run(
                original_chunk=chunk,
                processed_chunk=processed_chunk,
                classification_result=classification_result,
                privacy_target=privacy_target,
            )

            trials.append((transform_result, metrics_result))

            # Track best result
            if metrics_result.privacy.privacy_score > best_privacy_score:
                best_privacy_score = metrics_result.privacy.privacy_score
                best_result = transform_result
                best_metrics = metrics_result

            # GATE decision (Req 8.5, 8.6)
            if metrics_result.decision.overall_pass:
                # Pass — go to DONE
                break

            if trial_idx < min(self._max_trials, len(level_order)) - 1:
                # RETRY — try next level in order
                state_trace.append(AgentState.RETRY)
                next_level = level_order[trial_idx + 1]
                logger.info(
                    "RETRY: trial=%d level=%d→%d for chunk=%s",
                    trial_idx, level, next_level, chunk.chunk_id,
                )
            # else: max trials reached, fall through to DONE

        # --- DONE (Req 8.6) ---
        state_trace.append(AgentState.DONE)
        assert best_result is not None  # at least one trial always runs

        # --- REFLECT (Req 8.7) ---
        state_trace.append(AgentState.REFLECT)
        chunk_report = ChunkReport(
            chunk_id=chunk.chunk_id,
            run_id=chunk.run_id,
            had_speech=vad_result.has_speech,
            recipe_applied=best_result.recipe_ref,
            params_applied=best_result.params,
            trials=len(trials),
            metrics=best_metrics,
            routing_decision="blurred",
            used_source_separation=used_source_sep,
        )

        logger.info(
            "REFLECT: chunk=%s trials=%d privacy_score=%.3f best_recipe=%s",
            chunk.chunk_id,
            len(trials),
            best_privacy_score,
            best_result.recipe_ref.recipe_name if best_result else "none",
        )

        return best_result, chunk_report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_params(level_params: dict[str, Any]) -> dict[str, Any]:
        """Extract tool parameters from a ladder level definition."""
        return {k: v for k, v in level_params.items() if k != "recipe"}

    def _execute_blurring(
        self,
        recipe_name: str,
        chunk: AudioChunk,
        vad_result: VADResult,
        level_params: dict[str, Any],
        trial: int,
    ) -> TransformResult:
        """Execute the selected blurring tool on speech segments.

        Non-speech segments are preserved unmodified by the underlying tools.
        """
        segments = vad_result.segments
        params = self._get_params(level_params)

        if recipe_name == RECIPE_MID_BAND_ATTEN:
            return self._mid_band_tool.run(
                wav_path=chunk.wav_path,
                segments=segments,
                band_hz=tuple(params.get("band_hz", (500, 3000))),
                atten_db=params.get("atten_db", 20.0),
                lowpass_cutoff=params.get("lowpass_cutoff", 0),
                pitch_shift_semitones=params.get("pitch_shift_semitones", 0.0),
                scope="speech_only",
                chunk_id=chunk.chunk_id,
                trial=trial,
            )
        elif recipe_name == RECIPE_SOURCE_SEPARATION and self._source_separation_tool is not None:
            blur_params = {}
            if "atten_db" in params:
                blur_params["atten_db"] = params["atten_db"]
            if "noise_snr_db" in params:
                blur_params["noise_snr_db"] = params["noise_snr_db"]
            if "pitch_shift_semitones" in params:
                blur_params["pitch_shift_semitones"] = params["pitch_shift_semitones"]
            return self._source_separation_tool.run(
                wav_path=chunk.wav_path,
                segments=segments,
                blur_method=params.get("blur_method", "strong_blur"),
                blur_params=blur_params,
                chunk_id=chunk.chunk_id,
                trial=trial,
            )
        else:
            # RECIPE_LOWPASS_HIGHPASS_MIX
            return self._strong_blur_tool.run(
                wav_path=chunk.wav_path,
                segments=segments,
                lowpass_cutoff=params.get("lowpass_cutoff", 1000),
                lowpass_mix=params.get("lowpass_mix", 0.55),
                noise_snr_db=params.get("noise_snr_db", 18.0),
                pitch_shift_semitones=params.get("pitch_shift_semitones", 0.0),
                chunk_id=chunk.chunk_id,
                trial=trial,
            )
