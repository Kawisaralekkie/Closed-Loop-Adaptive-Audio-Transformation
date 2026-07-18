"""Global configuration for the Privacy-Preserving Urban Soundscape system.

Centralizes audio processing parameters, privacy thresholds, retry policy,
and S3/Knowledge Base settings. Values are configurable (not hardcoded)
via environment variables or direct override.

Requirements: 1.3, 6.5, 6.6, 14.1
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AudioConfig:
    """Canonical audio format settings (Req 1.2, 1.3)."""

    sample_rate: int = 16_000  # Hz
    bit_depth: int = 16
    channels: int = 1  # mono

    # Chunking — configurable, not hardcoded (Req 1.3)
    window_size: float = 4.0  # seconds
    overlap: float = 1.0  # seconds


@dataclass(frozen=True)
class PrivacyThresholds:
    """Quality-gate thresholds keyed by privacy_target (Req 6.5, 6.6).

    NOTE (renamed 2026):
        OLD VOCAB        NEW VOCAB     privacy_score_min
        "high"      →    "moderate"    0.65
        "very_high" →    "high"        0.80
    """

    moderate_privacy_score_min: float = 0.65  # was "high" in old vocab
    high_privacy_score_min: float = 0.80      # was "very_high" in old vocab
    preserve_score_min: float = 0.80

    def privacy_score_min(self, privacy_target: str) -> float:
        """Return the minimum privacy_score for the given target level.

        Uses the NEW vocabulary ("moderate", "high"). Legacy "very_high" is
        aliased to "high" (strict). Legacy "high" (lighter target with
        threshold 0.65) is NOT auto-mapped — old callers must rename it
        to "moderate" explicitly to avoid the same string having two
        meanings across versions.
        """
        if privacy_target == "very_high":
            return self.high_privacy_score_min
        if privacy_target == "high":
            return self.high_privacy_score_min  # NEW strict (0.80)
        if privacy_target == "moderate":
            return self.moderate_privacy_score_min
        # Unknown → safest default = stricter threshold
        return self.high_privacy_score_min


@dataclass(frozen=True)
class RetryPolicy:
    """Configurable retry policy for transient failures (Req 14.1)."""

    max_retries: int = 3
    backoff_base: float = 1.0  # seconds
    backoff_multiplier: float = 2.0
    retryable_exceptions: tuple[str, ...] = (
        "TimeoutError",
        "ConnectionError",
        "TransientError",
    )


@dataclass(frozen=True)
class S3Config:
    """S3 / Knowledge Base bucket configuration."""

    bucket_name: str = os.environ.get(
        "KB_S3_BUCKET", "urban-soundscape-kb"
    )
    prefix: str = os.environ.get("KB_S3_PREFIX", "kb/")
    region: str = os.environ.get("KB_S3_REGION", "ap-southeast-1")


@dataclass(frozen=True)
class AgentConfig:
    """Settings for the AdaptivePrivacyControlAgent."""

    max_trials: int = 4  # max ACT→GATE cycles before DONE


@dataclass(frozen=True)
class GuardrailConfig:
    """Bedrock Guardrail configuration for LLM safety."""

    # Guardrail identifier - can be name or ID
    guardrail_identifier: str = os.environ.get(
        "BEDROCK_GUARDRAIL_ID", "urban-soundscape-guardrail"
    )
    # Guardrail version - "DRAFT" or specific version number
    guardrail_version: str = os.environ.get(
        "BEDROCK_GUARDRAIL_VERSION", "DRAFT"
    )
    # Enable/disable guardrail (useful for testing)
    enabled: bool = os.environ.get("BEDROCK_GUARDRAIL_ENABLED", "true").lower() == "true"


@dataclass(frozen=True)
class SourceSeparationConfig:
    """Configuration for SourceSeparationTool (Req 10.1, 10.2, 10.3)."""

    timeout_seconds: float = 30.0
    max_memory_gb: float = 4.0
    min_separation_quality: float = 0.3
    nussl_model: str = "SeparationModel"


@dataclass
class AppConfig:
    """Top-level application configuration aggregating all sub-configs."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    privacy: PrivacyThresholds = field(default_factory=PrivacyThresholds)
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    s3: S3Config = field(default_factory=S3Config)
    agent: AgentConfig = field(default_factory=AgentConfig)
    guardrail: GuardrailConfig = field(default_factory=GuardrailConfig)
    source_separation: SourceSeparationConfig = field(default_factory=SourceSeparationConfig)


# Module-level singleton — importable as `from src.config import config`
config = AppConfig()
