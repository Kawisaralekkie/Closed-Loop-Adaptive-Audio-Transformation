"""LLM-based Adaptive Privacy Control Agent using Claude 4.5 Haiku via Bedrock.

Upgraded from Reactive → Adaptive Agent with:
- Cross-chunk memory: learns from previous chunks to improve decisions
- Strategy adaptation: dynamically adjusts approach based on accumulated experience
- Knowledge Base integration: uses playbook guidelines in LLM prompts

Falls back to the rule-based agent if LLM is unavailable.

NOTE (renamed 2026): privacy_target vocabulary
    OLD VOCAB        NEW VOCAB     privacy_score_min
    "high"      →    "moderate"    0.65
    "very_high" →    "high"        0.80
Legacy values are accepted via ``normalize_privacy_target()``.

Requirements: 8.1–8.8
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from src.config import config
from src.contracts.audio_contracts import AudioChunk
from src.contracts.classification_contracts import ClassPrediction, YamnetOutput
from src.contracts.metrics_contracts import MetricsResult
from src.contracts.report_contracts import ChunkReport
from src.contracts.transform_contracts import TransformResult
from src.contracts.vad_contracts import VADResult
from src.knowledge_base.kb_loader import KnowledgeBase
from src.tools.mid_band_attenuation_tool import MidBandAttenuationTool
from src.tools.quality_evaluation_tool import QualityEvaluationTool
from src.tools.source_separation_tool import SourceSeparationTool
from src.tools.strong_blurring_tool import StrongBlurringTool

logger = logging.getLogger(__name__)

RECIPE_MID_BAND_ATTEN = "RECIPE_MID_BAND_ATTEN"
RECIPE_LOWPASS_HIGHPASS_MIX = "RECIPE_LOWPASS_HIGHPASS_MIX"
RECIPE_SOURCE_SEPARATION = "RECIPE_SOURCE_SEPARATION"


class AgentState(str, Enum):
    PLAN = "PLAN"
    ACT = "ACT"
    GATE = "GATE"
    RETRY = "RETRY"
    DONE = "DONE"
    REFLECT = "REFLECT"


# ── Experience Memory (Cross-chunk learning) ──────────────────────────

@dataclass
class ChunkExperience:
    """Summary of one chunk's processing outcome for cross-chunk learning."""
    chunk_id: str
    speech_ratio: float
    vad_confidence: float
    privacy_target: str
    winning_recipe: str
    winning_params: dict[str, Any]
    trials_needed: int
    final_privacy_score: float
    final_preserve_score: float
    final_speaker_privacy: float
    overall_pass: bool


class ExperienceMemory:
    """Accumulates chunk processing experiences for adaptive strategy.

    Tracks which recipes/params worked for different audio profiles,
    enabling the LLM to make better first-trial choices over time.
    """

    def __init__(self, max_entries: int = 50) -> None:
        self._experiences: list[ChunkExperience] = []
        self._max_entries = max_entries
        # Aggregated stats for quick strategy lookup
        self._recipe_success: dict[str, int] = defaultdict(int)
        self._recipe_total: dict[str, int] = defaultdict(int)

    def add(self, exp: ChunkExperience) -> None:
        """Record a chunk experience."""
        self._experiences.append(exp)
        if len(self._experiences) > self._max_entries:
            self._experiences.pop(0)
        self._recipe_total[exp.winning_recipe] += 1
        if exp.overall_pass:
            self._recipe_success[exp.winning_recipe] += 1

    @property
    def size(self) -> int:
        return len(self._experiences)

    def get_strategy_summary(self) -> str:
        """Build a concise summary of accumulated experience for the LLM prompt."""
        if not self._experiences:
            return ""

        lines = [f"Experience from {len(self._experiences)} previous chunk(s):"]

        # Recipe success rates
        for recipe in sorted(self._recipe_total.keys()):
            total = self._recipe_total[recipe]
            success = self._recipe_success[recipe]
            rate = success / total if total > 0 else 0
            lines.append(f"  - {recipe}: {success}/{total} passed ({rate:.0%})")

        # Average trials needed
        avg_trials = sum(e.trials_needed for e in self._experiences) / len(self._experiences)
        lines.append(f"  - avg trials needed: {avg_trials:.1f}")

        # Recent pattern: last 5 chunks
        recent = self._experiences[-5:]
        if recent:
            lines.append("  Recent chunks:")
            for e in recent:
                status = "PASS" if e.overall_pass else "FAIL"
                lines.append(
                    f"    [{status}] speech={e.speech_ratio:.2f} → {e.winning_recipe} "
                    f"(trials={e.trials_needed}, priv={e.final_privacy_score:.3f}, "
                    f"pres={e.final_preserve_score:.3f})"
                )

        # Learned patterns
        patterns = self._detect_patterns()
        if patterns:
            lines.append("  Learned patterns:")
            for p in patterns:
                lines.append(f"    - {p}")

        return "\n".join(lines)

    def get_recommended_start(self, speech_ratio: float, privacy_target: str) -> dict[str, Any] | None:
        """Suggest a starting recipe based on similar past chunks.

        Returns a hint dict or None if insufficient experience.
        """
        if len(self._experiences) < 3:
            return None

        # Find chunks with similar speech_ratio (±0.15) and same privacy_target
        similar = [
            e for e in self._experiences
            if abs(e.speech_ratio - speech_ratio) < 0.15
            and e.privacy_target == privacy_target
            and e.overall_pass
        ]
        if not similar:
            return None

        # Pick the most common winning recipe among similar chunks
        recipe_counts: dict[str, int] = defaultdict(int)
        for e in similar:
            recipe_counts[e.winning_recipe] += 1

        best_recipe = max(recipe_counts, key=recipe_counts.get)  # type: ignore[arg-type]
        # Average the winning params of that recipe
        matching = [e for e in similar if e.winning_recipe == best_recipe]
        return {
            "recommended_recipe": best_recipe,
            "confidence": len(matching) / len(similar),
            "based_on": len(matching),
        }

    def _detect_patterns(self) -> list[str]:
        """Detect simple patterns from experience."""
        patterns: list[str] = []
        if len(self._experiences) < 5:
            return patterns

        # Pattern: high speech ratio chunks tend to need strong blur
        high_speech = [e for e in self._experiences if e.speech_ratio > 0.5]
        if high_speech:
            strong_count = sum(1 for e in high_speech if e.winning_recipe == RECIPE_LOWPASS_HIGHPASS_MIX)
            if strong_count / len(high_speech) > 0.7:
                patterns.append("High speech ratio (>0.5) → Strong Blur works best")

        # Pattern: low speech ratio can use lighter recipe
        low_speech = [e for e in self._experiences if e.speech_ratio <= 0.3]
        if low_speech:
            mid_count = sum(1 for e in low_speech if e.winning_recipe == RECIPE_MID_BAND_ATTEN)
            if mid_count / len(low_speech) > 0.5:
                patterns.append("Low speech ratio (<=0.3) → MidBand Attenuation often sufficient")

        # Pattern: average trials trending down = agent is learning
        if len(self._experiences) >= 10:
            first_half = self._experiences[:len(self._experiences) // 2]
            second_half = self._experiences[len(self._experiences) // 2:]
            avg_first = sum(e.trials_needed for e in first_half) / len(first_half)
            avg_second = sum(e.trials_needed for e in second_half) / len(second_half)
            if avg_second < avg_first - 0.3:
                patterns.append(
                    f"Improving: avg trials dropped from {avg_first:.1f} to {avg_second:.1f}"
                )

        # Pattern: source separation produces higher preserve_score than
        # frequency blurring for high-overlap audio (speech_ratio > 0.3)
        high_overlap = [e for e in self._experiences if e.speech_ratio > 0.3]
        if high_overlap:
            ss_exps = [e for e in high_overlap if e.winning_recipe == RECIPE_SOURCE_SEPARATION]
            freq_exps = [
                e for e in high_overlap
                if e.winning_recipe in (RECIPE_MID_BAND_ATTEN, RECIPE_LOWPASS_HIGHPASS_MIX)
            ]
            if ss_exps and freq_exps:
                avg_ss_preserve = sum(e.final_preserve_score for e in ss_exps) / len(ss_exps)
                avg_freq_preserve = sum(e.final_preserve_score for e in freq_exps) / len(freq_exps)
                if avg_ss_preserve > avg_freq_preserve + 0.02:
                    patterns.append(
                        f"Source separation preserves better for high-overlap audio "
                        f"(avg preserve {avg_ss_preserve:.3f} vs {avg_freq_preserve:.3f})"
                    )

        return patterns


# ── Bedrock Tool Definitions ─────────────────────────────────────────
TOOL_DEFINITIONS = [
    {
        "toolSpec": {
            "name": "apply_midband_attenuation",
            "description": (
                "Apply mid-band attenuation blurring on speech segments. "
                "Attenuates the core speech frequency band to obscure voice content. "
                "Use for light-to-moderate privacy needs."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "band_hz_low": {
                            "type": "integer",
                            "description": "Lower bound of attenuation band in Hz (400-800)",
                            "minimum": 400,
                            "maximum": 800,
                        },
                        "band_hz_high": {
                            "type": "integer",
                            "description": "Upper bound of attenuation band in Hz (2500-3500)",
                            "minimum": 2500,
                            "maximum": 3500,
                        },
                        "atten_db": {
                            "type": "number",
                            "description": "Attenuation in dB (15-35). Higher = more privacy, less preserve.",
                            "minimum": 15,
                            "maximum": 35,
                        },
                        "lowpass_cutoff": {
                            "type": "integer",
                            "description": "Lowpass filter cutoff in Hz (800-1500). 0 = disabled. Lower = more privacy.",
                            "minimum": 0,
                            "maximum": 1500,
                        },
                        "pitch_shift_semitones": {
                            "type": "number",
                            "description": "Pitch shift in semitones (-5 to +5). 0 = no shift. Helps degrade speaker identity.",
                            "minimum": -5,
                            "maximum": 5,
                        },
                    },
                    "required": ["band_hz_low", "band_hz_high", "atten_db", "lowpass_cutoff", "pitch_shift_semitones"],
                },
            },
        },
    },
    {
        "toolSpec": {
            "name": "apply_strong_blur",
            "description": (
                "Apply strong multi-technique blurring: lowpass + highband mix + "
                "midband preservation + noise injection + optional pitch shift. "
                "Use when mid-band attenuation alone is insufficient for privacy."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "noise_snr_db": {
                            "type": "number",
                            "description": "Noise injection SNR in dB (0-25). Lower = more noise = more privacy.",
                            "minimum": 0,
                            "maximum": 25,
                        },
                        "lowpass_cutoff": {
                            "type": "integer",
                            "description": "Lowpass filter cutoff in Hz (400-1200)",
                            "minimum": 400,
                            "maximum": 1200,
                        },
                        "lowpass_mix": {
                            "type": "number",
                            "description": "Lowpass mix ratio (0.4-0.8). Higher = more lowpass effect.",
                            "minimum": 0.4,
                            "maximum": 0.8,
                        },
                        "pitch_shift_semitones": {
                            "type": "number",
                            "description": "Pitch shift in semitones (-5 to +5). 0 = no shift.",
                            "minimum": -5,
                            "maximum": 5,
                        },
                    },
                    "required": ["noise_snr_db", "lowpass_cutoff", "lowpass_mix", "pitch_shift_semitones"],
                },
            },
        },
    },
    {
        "toolSpec": {
            "name": "apply_source_separation_blur",
            "description": (
                "Apply source separation using nussl to isolate speech from "
                "environmental sounds, then blur only the speech track and remix. "
                "Use when speech overlaps significantly with environmental sounds "
                "(speech_ratio > 0.3 AND vad_confidence > 0.6) to preserve "
                "environmental audio quality. Produces higher preserve_score than "
                "frequency blurring for high-overlap audio."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "blur_method": {
                            "type": "string",
                            "enum": ["midband_attenuation", "strong_blur"],
                            "description": "Blurring method to apply on the separated speech track.",
                        },
                        "atten_db": {
                            "type": "number",
                            "description": "Attenuation in dB for midband_attenuation (15-30). Ignored for strong_blur.",
                            "minimum": 15,
                            "maximum": 30,
                        },
                        "noise_snr_db": {
                            "type": "number",
                            "description": "Noise SNR in dB for strong_blur (10-25). Ignored for midband_attenuation.",
                            "minimum": 10,
                            "maximum": 25,
                        },
                        "pitch_shift_semitones": {
                            "type": "number",
                            "description": "Pitch shift in semitones (-5 to +5). 0 = no shift.",
                            "minimum": -5,
                            "maximum": 5,
                        },
                    },
                    "required": ["blur_method", "pitch_shift_semitones"],
                },
            },
        },
    },
]


# ── System Prompt ─────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an Adaptive Privacy Control Agent for an urban soundscape processing system.

Your job: choose the right audio blurring recipe and parameters to protect human voice privacy while preserving environmental sounds (birds, traffic, rain, etc.).

## Available Tools
- apply_midband_attenuation: Light-to-moderate blurring. Good for preserving audio quality.
- apply_strong_blur: Heavy multi-technique blurring. Use when midband alone isn't enough.
- apply_source_separation_blur: Separate speech from environment using nussl, blur only speech, then remix. Best for preserving environmental sounds when speech overlaps significantly.

## Decision Guidelines
- Pick the recipe that best fits the audio profile from the FIRST trial. When speech overlaps
  significantly (speech_ratio > 0.3 AND vad_confidence > 0.6), START with apply_source_separation_blur
  instead of defaulting to midband; otherwise start with midband and escalate only if needed.
- If speaker_privacy is low (<0.50), add pitch shifting (±2 to ±4 semitones).
- Alternate pitch shift direction between retries (speaker models react differently to +/- shifts).
- If privacy_score is close to threshold but not passing, increase atten_db or lower noise_snr_db slightly.
- If preserve_score is dropping too much, reduce atten_db or increase noise_snr_db.
- Balance privacy vs preservation — both must pass their thresholds.

## Source Separation Guidelines
- Use source separation WHEN speech_ratio > 0.3 AND vad_confidence > 0.6 (significant speech-environment overlap).
  In this regime source separation is the PREFERRED first choice — select it up front, do not wait to escalate to it.
- Prefer traditional frequency blurring WHEN speech_ratio <= 0.3 OR vad_confidence <= 0.6 (clearly separated speech, less overlap).
- If ExperienceMemory shows frequency blurring produced low preserve_score (<0.80) for similar audio profiles, prefer source separation.
- Source separation and frequency blurring are equally acceptable; choose based on the audio profile and the
  privacy/preservation goals, NOT on processing time or latency. Do not avoid source separation to save time.
- Source separation with strong_blur provides maximum privacy; with midband_attenuation provides better preservation.

## Adaptive Behavior
- You will receive experience summaries from previously processed chunks.
- USE this experience to make better first-trial choices:
  - If similar chunks (same speech_ratio range) succeeded with a specific recipe, start there.
  - If a recommendation is provided, follow it unless you have a strong reason not to.
  - Pay attention to learned patterns — they reflect what actually works.
- Your goal is to REDUCE the number of trials needed over time by learning from experience.

## Privacy Thresholds
# NOTE (renamed 2026): privacy_target vocabulary
#   OLD VOCAB        NEW VOCAB     privacy_score_min
#   "high"      →    "moderate"    0.65
#   "very_high" →    "high"        0.80
- privacy_target "moderate": privacy_score >= 0.65, preserve_score >= 0.80, speaker_privacy >= 0.50
- privacy_target "high":     privacy_score >= 0.80, preserve_score >= 0.80, speaker_privacy >= 0.50
# Legacy values "high" (= moderate) and "very_high" (= high) are still
# accepted by the agent and normalized internally.

## Important
- You MUST call exactly one tool per response.
- Respond with ONLY the tool call, no extra text.
- Parameters must be within the specified ranges."""


# ── Bedrock Client ────────────────────────────────────────────────────
def _get_bedrock_client():
    """Lazy-load boto3 Bedrock Runtime client."""
    import boto3
    return boto3.client("bedrock-runtime", region_name="ap-xxxxxxxx-1")


def _build_kb_context(kb: KnowledgeBase | None) -> str:
    """Extract relevant guidelines from KnowledgeBase for the LLM prompt."""
    if kb is None:
        return ""
    try:
        playbook = kb.playbook
        lines = ["Knowledge Base guidelines:"]
        if playbook.selection_strategy:
            lines.append(f"  - selection_strategy: {playbook.selection_strategy}")
        if playbook.max_trials:
            lines.append(f"  - kb_max_trials: {playbook.max_trials}")
        if playbook.utility_preserve_target:
            lines.append(f"  - preserve_target: {playbook.utility_preserve_target}")
        if playbook.pass_criteria:
            lines.append(f"  - pass_criteria: {json.dumps(playbook.pass_criteria, default=str)}")
        if playbook.recipes:
            lines.append("  - available recipes:")
            for r in playbook.recipes:
                conditions = json.dumps(r.use_when, default=str) if r.use_when else "any"
                lines.append(f"    * {r.name}: use_when={conditions}")
        return "\n".join(lines)
    except Exception as e:
        logger.debug("Could not extract KB context: %s", e)
        return ""


def _build_user_message(
    vad_result: VADResult,
    privacy_target: str,
    trial: int,
    previous_metrics: MetricsResult | None = None,
    kb: KnowledgeBase | None = None,
    experience_summary: str = "",
    recommendation: dict[str, Any] | None = None,
) -> str:
    """Build the user message with audio context, experience, and KB for the LLM."""
    vad_conf = sum(s.confidence for s in vad_result.segments) / max(len(vad_result.segments), 1)

    msg = f"""Audio chunk analysis:
- speech_ratio: {vad_result.speech_ratio:.3f}
- vad_confidence_mean: {vad_conf:.3f}
- num_speech_segments: {len(vad_result.segments)}
- privacy_target: {privacy_target}
- trial: {trial} (0-indexed)
- max_trials: {config.agent.max_trials}"""

    # Add KB context
    kb_ctx = _build_kb_context(kb)
    if kb_ctx:
        msg += f"\n\n{kb_ctx}"

    # Add experience from previous chunks (cross-chunk memory)
    if experience_summary:
        msg += f"\n\n{experience_summary}"

    # Add recommendation from experience memory
    if recommendation:
        msg += f"""

Recommendation based on similar past chunks:
- suggested_recipe: {recommendation['recommended_recipe']}
- confidence: {recommendation['confidence']:.0%} (based on {recommendation['based_on']} similar chunks)
- You should follow this recommendation unless previous trial metrics suggest otherwise."""

    if previous_metrics:
        pm = previous_metrics
        msg += f"""

Previous trial results (did NOT pass):
- privacy_score: {pm.privacy.privacy_score:.4f} (need >= {pm.decision.privacy_score_min})
- preserve_score: {pm.utility.preserve_score:.4f} (need >= 0.80)
- speaker_privacy: {pm.privacy.speaker_privacy:.4f} (need >= 0.50)
- WER: {pm.privacy.wer:.4f}
- CER: {pm.privacy.cer:.4f}
- content_privacy: {pm.privacy.content_privacy:.4f}
- privacy_pass: {pm.decision.privacy_pass}
- preserve_pass: {pm.decision.preserve_pass}
- s_loud: {pm.utility.sub_scores.s_loud:.4f}
- s_con: {pm.utility.sub_scores.s_con:.4f}
- s_psy: {pm.utility.sub_scores.s_psy:.4f}

Analyze what failed and choose a better recipe/params for this retry."""
    else:
        msg += "\n\nThis is the first trial. Choose an appropriate starting recipe."

    return msg


def _call_bedrock(
    messages: list[dict],
    max_retries: int = 2,
) -> tuple[dict | None, dict | None]:
    """Call Bedrock Converse API with tool use and optional Guardrail.

    Returns ``(tool_use_block, raw_response)`` where *raw_response* is the
    full Bedrock response dict (for logging/debugging).  Includes token
    usage and latency_ms for cost/performance tracking.

    When guardrail is enabled (via config.guardrail.enabled), the call
    includes guardrailConfig for PII protection and topic enforcement.
    """
    try:
        client = _get_bedrock_client()

        # Build base request parameters
        request_params = {
            "modelId": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
            "system": [{"text": SYSTEM_PROMPT}],
            "messages": messages,
            "toolConfig": {"tools": TOOL_DEFINITIONS},
            "inferenceConfig": {
                "maxTokens": 512,
                "temperature": 0.1,
            },
        }

        # Add guardrail config if enabled
        if config.guardrail.enabled:
            request_params["guardrailConfig"] = {
                "guardrailIdentifier": config.guardrail.guardrail_identifier,
                "guardrailVersion": config.guardrail.guardrail_version,
            }
            logger.debug(
                "Guardrail enabled: %s (version=%s)",
                config.guardrail.guardrail_identifier,
                config.guardrail.guardrail_version,
            )

        t0 = time.time()
        response = client.converse(**request_params)
        latency_ms = round((time.time() - t0) * 1000, 1)

        # Build a JSON-safe snapshot of the raw response
        usage = response.get("usage", {})
        raw_response = {
            "stopReason": response.get("stopReason"),
            "usage": {
                "inputTokens": usage.get("inputTokens", 0),
                "outputTokens": usage.get("outputTokens", 0),
                "totalTokens": usage.get("inputTokens", 0) + usage.get("outputTokens", 0),
            },
            "latency_ms": latency_ms,
            "content": response.get("output", {}).get("message", {}).get("content", []),
            "guardrail_enabled": config.guardrail.enabled,
        }

        # Check if guardrail blocked the request
        stop_reason = response.get("stopReason")
        if stop_reason == "guardrail_intervened":
            logger.warning(
                "Guardrail intervened - request blocked. Trace: %s",
                response.get("trace", {}).get("guardrail", {}),
            )
            raw_response["guardrail_intervened"] = True
            raw_response["guardrail_trace"] = response.get("trace", {}).get("guardrail", {})
            return None, raw_response

        logger.info(
            "Bedrock call: %d input + %d output tokens, %.0f ms (guardrail=%s)",
            raw_response["usage"]["inputTokens"],
            raw_response["usage"]["outputTokens"],
            latency_ms,
            "on" if config.guardrail.enabled else "off",
        )

        # Extract tool use from response
        for block in raw_response["content"]:
            if "toolUse" in block:
                return block["toolUse"], raw_response

        logger.warning("LLM response did not contain tool use")
        return None, raw_response

    except Exception as e:
        logger.error("Bedrock call failed: %s", e)
        return None, {"error": str(e)}


# ── Guardrail: Safe Parameter Ranges ──────────────────────────────────
SAFE_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    RECIPE_MID_BAND_ATTEN: {
        "band_hz_low":           (400, 800),
        "band_hz_high":          (2500, 3500),
        "atten_db":              (15.0, 35.0),
        "lowpass_cutoff":        (800, 1500),
        "pitch_shift_semitones": (-5.0, 5.0),
    },
    RECIPE_LOWPASS_HIGHPASS_MIX: {
        "noise_snr_db":          (0.0, 25.0),
        "lowpass_cutoff":        (400, 1200),
        "lowpass_mix":           (0.4, 0.8),
        "atten_db":              (15.0, 45.0),
        "pitch_shift_semitones": (-5.0, 5.0),
    },
    RECIPE_SOURCE_SEPARATION: {
        "atten_db":              (15.0, 30.0),
        "noise_snr_db":          (10.0, 25.0),
        "pitch_shift_semitones": (-5.0, 5.0),
    },
}


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


def _validate_and_clamp_params(
    recipe_name: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Validate LLM-chosen params against safe ranges.

    Clamps out-of-range values and logs warnings. This is the guardrail
    that prevents the LLM from choosing dangerous parameter combinations.
    """
    safe = SAFE_RANGES.get(recipe_name, {})
    validated = dict(params)
    violations: list[str] = []

    for key, (lo, hi) in safe.items():
        if key in validated:
            original = validated[key]
            clamped = _clamp(float(original), lo, hi)
            if clamped != float(original):
                violations.append(
                    f"{key}: {original} → clamped to {clamped} (range [{lo}, {hi}])"
                )
                validated[key] = type(original)(clamped) if isinstance(original, int) else clamped

    # Special guardrail for band_hz tuple
    if "band_hz" in validated:
        lo_val, hi_val = validated["band_hz"]
        lo_val = _clamp(float(lo_val), 400, 800)
        hi_val = _clamp(float(hi_val), 2500, 3500)
        if hi_val <= lo_val:
            violations.append(f"band_hz: high ({hi_val}) <= low ({lo_val}), resetting to defaults")
            lo_val, hi_val = 500, 3000
        validated["band_hz"] = (int(lo_val), int(hi_val))

    if violations:
        logger.warning(
            "Guardrail clamped %d param(s) for %s:\n  %s",
            len(violations), recipe_name, "\n  ".join(violations),
        )

    return validated


def _parse_tool_response(tool_use: dict) -> tuple[str, dict[str, Any]]:
    """Parse LLM tool use response into recipe name and params, with guardrail."""
    tool_name = tool_use["name"]
    tool_input = tool_use["input"]

    if tool_name == "apply_midband_attenuation":
        recipe = RECIPE_MID_BAND_ATTEN
        params = {
            "band_hz": (tool_input["band_hz_low"], tool_input["band_hz_high"]),
            "atten_db": tool_input["atten_db"],
            "pitch_shift_semitones": tool_input["pitch_shift_semitones"],
        }
    elif tool_name == "apply_strong_blur":
        recipe = RECIPE_LOWPASS_HIGHPASS_MIX
        params = {
            "noise_snr_db": tool_input["noise_snr_db"],
            "lowpass_cutoff": tool_input["lowpass_cutoff"],
            "lowpass_mix": tool_input["lowpass_mix"],
            "pitch_shift_semitones": tool_input["pitch_shift_semitones"],
        }
    elif tool_name == "apply_source_separation_blur":
        recipe = RECIPE_SOURCE_SEPARATION
        params: dict[str, Any] = {
            "blur_method": tool_input["blur_method"],
            "pitch_shift_semitones": tool_input["pitch_shift_semitones"],
        }
        if "atten_db" in tool_input:
            params["atten_db"] = tool_input["atten_db"]
        if "noise_snr_db" in tool_input:
            params["noise_snr_db"] = tool_input["noise_snr_db"]
    else:
        raise ValueError(f"Unknown tool: {tool_name}")

    # Apply guardrail
    params = _validate_and_clamp_params(recipe, params)
    return recipe, params


# ── Serialization helpers ─────────────────────────────────────────────
def _serialize_params(params: dict[str, Any]) -> dict[str, Any]:
    """Convert params dict to JSON-serializable format (tuples → lists, etc.)."""
    out: dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, tuple):
            out[k] = list(v)
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)
    return out


def save_decision_log(
    decision_log: list[dict[str, Any]],
    out_dir: str = "logs",
    tag: str = "llm",
) -> str:
    """Write decision_log list to a timestamped JSON file in *out_dir*.

    Returns the path of the saved file.
    """
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    filename = f"decisions_{tag}_{ts}.json"
    path = os.path.join(out_dir, filename)
    with open(path, "w") as f:
        json.dump(decision_log, f, indent=2, default=str)
    logger.info("Decision log saved → %s (%d entries)", path, len(decision_log))
    return path


# ── Fallback: rule-based selection ────────────────────────────────────
def _fallback_select(
    vad_result: VADResult,
    privacy_target: str,
    trial: int,
) -> tuple[str, dict[str, Any]]:
    """Rule-based fallback matching the adaptive level-order ladder."""
    from src.agents.adaptive_privacy_control_agent import (
        RETRY_LADDER,
        select_recipe,
    )

    kb_stub = type("KB", (), {"playbook": {}})()
    level_order = select_recipe(vad_result, privacy_target, kb_stub)
    # Pick the level for this trial from the ordered list
    idx = min(trial, len(level_order) - 1)
    level = level_order[idx]
    level_params = RETRY_LADDER[level]
    recipe_name = level_params["recipe"]
    params = {k: v for k, v in level_params.items() if k != "recipe"}
    return recipe_name, params


# ── Main Agent Class ──────────────────────────────────────────────────
class LLMPrivacyControlAgent:
    """Adaptive Privacy Control Agent powered by Claude 4.5 Haiku via Bedrock.

    Upgraded to Adaptive Agent level with cross-chunk learning:
    - Maintains ExperienceMemory across chunks within a run
    - Provides experience summaries and recommendations to the LLM
    - Integrates Knowledge Base guidelines into prompts
    - Tracks strategy effectiveness and adapts over time

    Parameters
    ----------
    mid_band_tool : MidBandAttenuationTool
    strong_blur_tool : StrongBlurringTool
    quality_tool : QualityEvaluationTool
    source_separation_tool : SourceSeparationTool | None
        Optional nussl-based source separation tool for the agentic pipeline.
    max_trials : int | None
        Max ACT→GATE cycles. Defaults to config.agent.max_trials.
    use_llm : bool
        If False, skip LLM and use rule-based fallback (for testing).
    """

    def __init__(
        self,
        mid_band_tool: MidBandAttenuationTool,
        strong_blur_tool: StrongBlurringTool,
        quality_tool: QualityEvaluationTool,
        source_separation_tool: SourceSeparationTool | None = None,
        classification_tool=None,
        max_trials: int | None = None,
        use_llm: bool = True,
    ) -> None:
        self._mid_band_tool = mid_band_tool
        self._strong_blur_tool = strong_blur_tool
        self._quality_tool = quality_tool
        self._source_separation_tool = source_separation_tool
        self._classification_tool = classification_tool
        self._max_trials = max_trials if max_trials is not None else config.agent.max_trials
        self._use_llm = use_llm
        # Adaptive: cross-chunk experience memory (persists across chunks)
        self.memory = ExperienceMemory()

    def run(
        self,
        chunk: AudioChunk,
        vad_result: VADResult,
        kb: KnowledgeBase,
        privacy_target: str,
    ) -> tuple[TransformResult, ChunkReport]:
        """Execute the adaptive LLM-driven privacy control state machine.

        PLAN → (ACT → GATE → RETRY)* → DONE → REFLECT

        The PLAN phase now consults cross-chunk memory for strategy.
        The REFLECT phase records experience for future chunks.
        Decision log is stored in ``self.decision_log`` after each run.
        """
        state_trace: list[AgentState] = []
        trials: list[tuple[TransformResult, MetricsResult]] = []
        decision_log: list[dict[str, Any]] = []

        best_result: TransformResult | None = None
        best_metrics: MetricsResult | None = None
        best_privacy_score = -1.0
        previous_metrics: MetricsResult | None = None

        vad_conf = sum(s.confidence for s in vad_result.segments) / max(len(vad_result.segments), 1)

        # --- PLAN (adaptive: consult memory) ---
        state_trace.append(AgentState.PLAN)
        experience_summary = self.memory.get_strategy_summary()
        recommendation = self.memory.get_recommended_start(
            vad_result.speech_ratio, privacy_target,
        )

        if recommendation:
            logger.info(
                "PLAN: Memory recommends %s (confidence=%.0f%%, based_on=%d) "
                "for chunk=%s speech_ratio=%.3f target=%s",
                recommendation["recommended_recipe"],
                recommendation["confidence"] * 100,
                recommendation["based_on"],
                chunk.chunk_id, vad_result.speech_ratio, privacy_target,
            )
        else:
            logger.info(
                "PLAN: No recommendation yet (memory=%d chunks) for chunk=%s target=%s",
                self.memory.size, chunk.chunk_id, privacy_target,
            )

        for trial_idx in range(self._max_trials):
            # --- Select recipe via LLM or fallback ---
            recipe_name, params, raw_llm_response = self._select_recipe_llm(
                vad_result, privacy_target, trial_idx, previous_metrics, kb,
                experience_summary=experience_summary,
                recommendation=recommendation if trial_idx == 0 else None,
            )
            logger.info(
                "Trial %d: recipe=%s params=%s",
                trial_idx, recipe_name, params,
            )

            # --- ACT ---
            state_trace.append(AgentState.ACT)
            transform_result = self._execute_blurring(
                recipe_name, chunk, vad_result, params, trial_idx,
            )

            # --- GATE ---
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

            stub_classification = YamnetOutput(
                chunk_id=chunk.chunk_id,
                embeddings=[[0.0]],
                predictions=[ClassPrediction(label="unknown", confidence=0.5)],
            )

            # Use real YAMNet classification if tool is available;
            # otherwise fall back to stub (backward-compat).
            if self._classification_tool is not None:
                gate_classification = self._classification_tool.run(processed_chunk)
            else:
                gate_classification = stub_classification

            metrics_result = self._quality_tool.run(
                original_chunk=chunk,
                processed_chunk=processed_chunk,
                classification_result=gate_classification,
                privacy_target=privacy_target,
            )

            trials.append((transform_result, metrics_result))

            # --- Log this decision ---
            # Extract token usage from raw LLM response before storing
            llm_usage = None
            llm_latency = None
            llm_stop_reason = None
            if isinstance(raw_llm_response, dict):
                llm_usage = raw_llm_response.get("usage")
                llm_latency = raw_llm_response.get("latency_ms")
                llm_stop_reason = raw_llm_response.get("stopReason")
                logger.info(
                    "Token usage trial=%d: input=%s output=%s latency=%s",
                    trial_idx,
                    llm_usage.get("inputTokens") if llm_usage else "N/A",
                    llm_usage.get("outputTokens") if llm_usage else "N/A",
                    llm_latency,
                )
            else:
                logger.warning(
                    "No raw_llm_response for trial=%d (type=%s)",
                    trial_idx, type(raw_llm_response).__name__,
                )

            decision_entry: dict[str, Any] = {
                "chunk_id": chunk.chunk_id,
                "trial": trial_idx,
                "recipe": recipe_name,
                "params": _serialize_params(params),
                "source": "llm" if self._use_llm else "fallback",
                "llm_usage": llm_usage,
                "llm_latency_ms": llm_latency,
                "llm_stop_reason": llm_stop_reason,
                "input_context": {
                    "speech_ratio": vad_result.speech_ratio,
                    "vad_confidence": round(vad_conf, 4),
                    "privacy_target": privacy_target,
                },
                "adaptive_context": {
                    "memory_size": self.memory.size,
                    "had_recommendation": recommendation is not None and trial_idx == 0,
                    "experience_provided": bool(experience_summary),
                },
                "metrics": {
                    "privacy_score": metrics_result.privacy.privacy_score,
                    "preserve_score": metrics_result.utility.preserve_score,
                    "speaker_privacy": metrics_result.privacy.speaker_privacy,
                    "wer": metrics_result.privacy.wer,
                    "cer": metrics_result.privacy.cer,
                    "content_privacy": metrics_result.privacy.content_privacy,
                    "s_loud": metrics_result.utility.sub_scores.s_loud,
                    "s_con": metrics_result.utility.sub_scores.s_con,
                    "s_psy": metrics_result.utility.sub_scores.s_psy,
                },
                "decision": {
                    "privacy_pass": metrics_result.decision.privacy_pass,
                    "preserve_pass": metrics_result.decision.preserve_pass,
                    "overall_pass": metrics_result.decision.overall_pass,
                },
            }

            # Enrich audit entry for source separation decisions (Req 7.1, 7.2, 7.3)
            if recipe_name == RECIPE_SOURCE_SEPARATION:
                sep_result = (
                    self._source_separation_tool.last_separation_result
                    if self._source_separation_tool is not None
                    else None
                )
                decision_entry["tool_name"] = "SourceSeparationTool"
                decision_entry["decision_parameters"] = {
                    "blur_method": params.get("blur_method", "strong_blur"),
                    "blur_params": {
                        k: v for k, v in params.items() if k != "blur_method"
                    },
                    "speech_ratio": vad_result.speech_ratio,
                    "vad_confidence": round(vad_conf, 4),
                    "separation_quality_score": (
                        sep_result.separation_quality_score if sep_result else 0.0
                    ),
                    "processing_time_ms": (
                        sep_result.processing_time_ms if sep_result else 0.0
                    ),
                    "is_retry_escalation": trial_idx > 0,
                }

            decision_log.append(decision_entry)

            previous_metrics = metrics_result

            # Track best
            if metrics_result.privacy.privacy_score > best_privacy_score:
                best_privacy_score = metrics_result.privacy.privacy_score
                best_result = transform_result
                best_metrics = metrics_result

            if metrics_result.decision.overall_pass:
                break

            if trial_idx < self._max_trials - 1:
                state_trace.append(AgentState.RETRY)
                logger.info("RETRY: trial=%d for chunk=%s", trial_idx, chunk.chunk_id)

        # --- DONE ---
        state_trace.append(AgentState.DONE)
        assert best_result is not None

        # --- REFLECT (adaptive: record experience) ---
        state_trace.append(AgentState.REFLECT)

        experience = ChunkExperience(
            chunk_id=chunk.chunk_id,
            speech_ratio=vad_result.speech_ratio,
            vad_confidence=vad_conf,
            privacy_target=privacy_target,
            winning_recipe=best_result.recipe_ref.recipe_name,
            winning_params=_serialize_params(best_result.params.params),
            trials_needed=len(trials),
            final_privacy_score=best_metrics.privacy.privacy_score,
            final_preserve_score=best_metrics.utility.preserve_score,
            final_speaker_privacy=best_metrics.privacy.speaker_privacy,
            overall_pass=best_metrics.decision.overall_pass,
        )
        self.memory.add(experience)

        chunk_report = ChunkReport(
            chunk_id=chunk.chunk_id,
            run_id=chunk.run_id,
            had_speech=vad_result.has_speech,
            recipe_applied=best_result.recipe_ref,
            params_applied=best_result.params,
            trials=len(trials),
            metrics=best_metrics,
            routing_decision="blurred",
            llm_token_usage=self._aggregate_token_usage(decision_log),
            trial_details=self._extract_trial_details(decision_log, best_result.trial),
            llm_responses=self._extract_llm_responses(decision_log),
            memory_snapshot=self._snapshot_memory(),
        )

        logger.info(
            "REFLECT: chunk=%s trials=%d privacy=%.3f recipe=%s llm=%s memory=%d",
            chunk.chunk_id, len(trials), best_privacy_score,
            best_result.recipe_ref.recipe_name, self._use_llm,
            self.memory.size,
        )

        # Store decision log for external access
        self.decision_log = decision_log

        return best_result, chunk_report

    # ------------------------------------------------------------------
    @staticmethod
    def _aggregate_token_usage(decision_log: list[dict[str, Any]]) -> dict:
        """Sum token usage across all LLM calls in a chunk's decision log."""
        total_input = 0
        total_output = 0
        for entry in decision_log:
            usage = entry.get("llm_usage")
            if isinstance(usage, dict):
                total_input += usage.get("inputTokens", 0)
                total_output += usage.get("outputTokens", 0)
        return {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_input + total_output,
        }

    @staticmethod
    def _extract_trial_details(decision_log: list[dict[str, Any]], best_trial_idx: int = -1) -> list[dict]:
        """Extract per-trial summary: recipe, params, scores, is_selected."""
        details = []
        for entry in decision_log:
            metrics = entry.get("metrics", {})
            trial_idx = entry.get("trial", -1)
            details.append({
                "trial": trial_idx,
                "recipe": entry.get("recipe"),
                "params": entry.get("params"),
                "source": entry.get("source"),
                "privacy_score": metrics.get("privacy_score"),
                "preserve_score": metrics.get("preserve_score"),
                "wer": metrics.get("wer"),
                "cer": metrics.get("cer"),
                "speaker_privacy": metrics.get("speaker_privacy"),
                "overall_pass": entry.get("decision", {}).get("overall_pass"),
                "is_selected": trial_idx == best_trial_idx,
            })
        return details

    @staticmethod
    def _extract_llm_responses(decision_log: list[dict[str, Any]]) -> list[dict]:
        """Extract LLM raw response per trial (token usage, latency, stop reason)."""
        responses = []
        for entry in decision_log:
            responses.append({
                "trial": entry.get("trial"),
                "stop_reason": entry.get("llm_stop_reason"),
                "usage": entry.get("llm_usage"),
                "latency_ms": entry.get("llm_latency_ms"),
            })
        return responses

    def _snapshot_memory(self) -> dict:
        """Snapshot current ExperienceMemory state."""
        return {
            "size": self.memory.size,
            "entries": [
                {
                    "chunk_id": exp.chunk_id,
                    "speech_ratio": exp.speech_ratio,
                    "winning_recipe": exp.winning_recipe,
                    "trials_needed": exp.trials_needed,
                    "final_privacy_score": exp.final_privacy_score,
                    "final_preserve_score": exp.final_preserve_score,
                    "overall_pass": exp.overall_pass,
                }
                for exp in self.memory._experiences
            ],
        }

    def _select_recipe_llm(
        self,
        vad_result: VADResult,
        privacy_target: str,
        trial: int,
        previous_metrics: MetricsResult | None,
        kb: KnowledgeBase,
        experience_summary: str = "",
        recommendation: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any], dict | None]:
        """Ask LLM to select recipe, with rule-based fallback.

        Now includes experience summary and recommendation in the prompt
        for adaptive decision-making.

        Returns ``(recipe_name, params, raw_llm_response)``.
        *raw_llm_response* is ``None`` when the fallback path is used.
        """
        if not self._use_llm:
            r, p = _fallback_select(vad_result, privacy_target, trial)
            return r, p, None

        try:
            user_msg = _build_user_message(
                vad_result, privacy_target, trial, previous_metrics, kb,
                experience_summary=experience_summary,
                recommendation=recommendation,
            )

            messages = [{"role": "user", "content": [{"text": user_msg}]}]
            tool_use, raw_response = _call_bedrock(messages)

            if tool_use is None:
                logger.warning("LLM returned no tool use, falling back to rule-based")
                r, p = _fallback_select(vad_result, privacy_target, trial)
                return r, p, raw_response

            recipe_name, params = _parse_tool_response(tool_use)
            logger.info("LLM selected: %s with %s", recipe_name, params)
            return recipe_name, params, raw_response

        except Exception as e:
            logger.error("LLM selection failed: %s — falling back", e)
            r, p = _fallback_select(vad_result, privacy_target, trial)
            return r, p, None

    def _execute_blurring(
        self,
        recipe_name: str,
        chunk: AudioChunk,
        vad_result: VADResult,
        params: dict[str, Any],
        trial: int,
    ) -> TransformResult:
        """Execute the selected blurring tool."""
        segments = vad_result.segments

        if recipe_name == RECIPE_MID_BAND_ATTEN:
            band_hz = params.get("band_hz", (500, 3000))
            if isinstance(band_hz, list):
                band_hz = tuple(band_hz)
            return self._mid_band_tool.run(
                wav_path=chunk.wav_path,
                segments=segments,
                band_hz=band_hz,
                atten_db=params.get("atten_db", 20.0),
                lowpass_cutoff=params.get("lowpass_cutoff", 0),
                pitch_shift_semitones=params.get("pitch_shift_semitones", 0.0),
                scope="speech_only",
                chunk_id=chunk.chunk_id,
                trial=trial,
            )
        elif recipe_name == RECIPE_SOURCE_SEPARATION:
            if self._source_separation_tool is None:
                logger.warning(
                    "SourceSeparationTool not available, falling back to StrongBlurringTool"
                )
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
            blur_method = params.get("blur_method", "strong_blur")
            blur_params: dict[str, Any] = {}
            if "atten_db" in params:
                blur_params["atten_db"] = params["atten_db"]
            if "noise_snr_db" in params:
                blur_params["noise_snr_db"] = params["noise_snr_db"]
            if "pitch_shift_semitones" in params:
                blur_params["pitch_shift_semitones"] = params["pitch_shift_semitones"]
            return self._source_separation_tool.run(
                wav_path=chunk.wav_path,
                segments=segments,
                blur_method=blur_method,
                blur_params=blur_params,
                chunk_id=chunk.chunk_id,
                trial=trial,
            )
        else:
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
