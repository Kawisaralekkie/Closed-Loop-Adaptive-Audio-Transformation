"""LLM-based Privacy Control Agent WITHOUT cross-chunk Experience Memory.

Identical to LLMPrivacyControlAgent except:
- PLAN phase does NOT consult ExperienceMemory
- REFLECT phase does NOT record ChunkExperience
- Each chunk is treated independently (no cross-chunk learning)

Used for ablation study to measure the contribution of memory.
"""

from __future__ import annotations

import logging
from typing import Any

from src.agents.llm_privacy_control_agent import (
    LLMPrivacyControlAgent,
    ExperienceMemory,
)
from src.contracts.audio_contracts import AudioChunk
from src.contracts.vad_contracts import VADResult
from src.knowledge_base.kb_loader import KnowledgeBase
from src.contracts.report_contracts import ChunkReport
from src.contracts.transform_contracts import TransformResult
from src.tools.mid_band_attenuation_tool import MidBandAttenuationTool
from src.tools.strong_blurring_tool import StrongBlurringTool
from src.tools.quality_evaluation_tool import QualityEvaluationTool
from src.tools.source_separation_tool import SourceSeparationTool
from src.config import config

logger = logging.getLogger(__name__)


class LLMNoMemoryAgent(LLMPrivacyControlAgent):
    """LLM agent that resets memory before every chunk — no cross-chunk learning."""

    def run(
        self,
        chunk: AudioChunk,
        vad_result: VADResult,
        kb: KnowledgeBase,
        privacy_target: str,
    ) -> tuple[TransformResult, ChunkReport]:
        """Reset memory before each chunk, then delegate to parent."""
        # Clear memory so each chunk starts fresh
        self.memory = ExperienceMemory()
        return super().run(chunk, vad_result, kb, privacy_target)
