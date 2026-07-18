#!/usr/bin/env python3
"""Ablation study: Compare 4 pipeline variants on the same audio file.

Variants:
  1. Fixed Baseline — rule-based, no retry, no memory
  2. Adaptive Rule-Based — rule-based with retry ladder + GATE
  3. Adaptive LLM (no memory) — LLM selects recipe, retry + GATE, no cross-chunk memory
  4. Adaptive LLM (with memory) — LLM selects recipe, retry + GATE, cross-chunk memory

Usage:
    python3 scripts/run_ablation_study.py <audio_file> [--max-chunks N]
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.contracts.audio_contracts import AudioIngestRequest
from src.pipelines.agentic_pipeline import AgenticPipeline
from src.pipelines.fixed_baseline_pipeline import FixedBaselinePipeline
from src.tools.prepare_data_tool import PrepareDataTool
from src.tools.speech_scan_tool import SpeechScanTool
from src.tools.mid_band_attenuation_tool import MidBandAttenuationTool
from src.tools.strong_blurring_tool import StrongBlurringTool
from src.tools.source_separation_tool import SourceSeparationTool
from src.tools.classification_tool import ClassificationTool
from src.tools.quality_evaluation_tool import QualityEvaluationTool
from src.tools.data_lake_writer import DataLakeWriter
from src.agents.adaptive_privacy_control_agent import AdaptivePrivacyControlAgent
from src.agents.llm_privacy_control_agent import LLMPrivacyControlAgent
from src.agents.llm_no_memory_agent import LLMNoMemoryAgent
from src.knowledge_base.kb_loader import (
    KnowledgeBase, PolicyTransformationRules, PrivacyPlaybook,
    RecipeDefinition, SoundLabelTaxonomy,
)

import numpy as np

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")


def convert_to_wav(input_path: str, output_dir: str) -> str:
    stem = Path(input_path).stem
    wav_path = os.path.join(output_dir, f"{stem}.wav")
    cmd = ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1",
           "-sample_fmt", "s16", "-f", "wav", wav_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    return wav_path


def build_kb(privacy_target: str) -> KnowledgeBase:
    return KnowledgeBase(
        version="ablation-study-1.0",
        manifest_hash="ablation-hash",
        policies=PolicyTransformationRules(
            privacy_target=privacy_target,
            allow_store_raw_audio=True,
            max_retention_days=90,
        ),
        playbook=PrivacyPlaybook(
            analyzer_required_features=["speech_ratio", "vad_conf_mean", "hf_energy_ratio", "mid_energy_ratio"],
            evaluator_metrics=["wer", "cer", "speaker_privacy"],
            privacy_score_formula={"wer_weight": 0.6, "cer_weight": 0.4, "content_weight": 0.7, "speaker_weight": 0.3},
            preserve_score_formula={"weights": {"s_loud": 0.2, "s_hf": 0.2, "s_sc": 0.2, "s_con": 0.2, "s_psy": 0.2}},
            pass_criteria={
                "high": {"privacy_score_min": 0.65, "preserve_score_min": 0.80},
                "very_high": {"privacy_score_min": 0.80, "preserve_score_min": 0.80},
            },
            recipes=[
                RecipeDefinition(
                    name="RECIPE_MID_BAND_ATTEN",
                    params={"band_hz": [500, 3000], "atten_db": 25.0},
                    use_when={"speech_ratio": {"min": 0.0, "max": 1.0}},
                    risks=["Insufficient privacy"], mitigations=["Escalate atten_db"],
                    auto_tune_rules={"atten_db": {"step": 10.0, "max": 40.0}},
                ),
                RecipeDefinition(
                    name="RECIPE_LOWPASS_HIGHPASS_MIX",
                    params={"lowpass_cutoff": 800, "lowpass_mix": 0.7, "noise_snr_db": 10.0},
                    use_when={"speech_ratio": {"min": 0.0, "max": 1.0}},
                    risks=["May degrade quality"], mitigations=["Monitor preserve_score"],
                    auto_tune_rules={"lowpass_cutoff": {"step": -100, "min": 400}},
                ),
            ],
            selection_strategy="score_then_try", max_trials=4,
            fallback_rules={}, utility_preserve_target=0.80,
        ),
        taxonomy=SoundLabelTaxonomy(
            classes=["Speech", "Music", "Environmental"],
            mappings={"Speech": "human_voice", "Music": "music", "Environmental": "environment"},
            confidence_thresholds={"default": 0.3},
        ),
    )


def report_to_dict(report, label: str) -> dict:
    chunks = []
    for cr in report.chunks:
        chunk_data = {
            "chunk_id": cr.chunk_id, "had_speech": cr.had_speech,
            "speech_ratio": round(cr.speech_ratio, 4),
            "vad_confidence": round(cr.vad_confidence, 4),
            "routing": cr.routing_decision, "trials": cr.trials,
            "recipe": cr.recipe_applied.recipe_name if cr.recipe_applied else None,
        }
        if cr.metrics:
            pm, um, d = cr.metrics.privacy, cr.metrics.utility, cr.metrics.decision
            chunk_data["privacy"] = {
                "wer": round(pm.wer, 4), "cer": round(pm.cer, 4),
                "speaker_privacy": round(pm.speaker_privacy, 4),
                "content_privacy": round(pm.content_privacy, 4),
                "privacy_score": round(pm.privacy_score, 4),
            }
            chunk_data["preserve"] = {
                "preserve_score": round(um.preserve_score, 4),
                "s_loud": round(um.sub_scores.s_loud, 4),
                "s_hf": round(um.sub_scores.s_hf, 4),
                "s_sc": round(um.sub_scores.s_sc, 4),
                "s_con": round(um.sub_scores.s_con, 4),
                "s_psy": round(um.sub_scores.s_psy, 4),
            }
            chunk_data["decision"] = {
                "privacy_pass": d.privacy_pass, "preserve_pass": d.preserve_pass,
                "overall_pass": d.overall_pass,
            }
        chunks.append(chunk_data)

    scored = [c for c in chunks if "privacy" in c]
    n = len(scored) or 1
    return {
        "label": label, "run_id": str(report.run_id),
        "total_chunks": report.total_chunks,
        "succeeded_chunks": report.succeeded_chunks,
        "total_runtime_seconds": report.total_runtime_seconds,
        "summary": {
            "avg_privacy_score": round(sum(c["privacy"]["privacy_score"] for c in scored) / n, 4),
            "avg_preserve_score": round(sum(c["preserve"]["preserve_score"] for c in scored) / n, 4),
        },
        "chunks": chunks,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/run_ablation_study.py <audio_file> [--max-chunks N]")
        sys.exit(1)

    audio_path = sys.argv[1]
    max_chunks = None
    if "--max-chunks" in sys.argv:
        idx = sys.argv.index("--max-chunks")
        if idx + 1 < len(sys.argv):
            max_chunks = int(sys.argv[idx + 1])

    output_dir = tempfile.mkdtemp(prefix="ablation_")
    print(f"Output: {output_dir}\n")

    ext = Path(audio_path).suffix.lower()
    wav_path = convert_to_wav(audio_path, output_dir) if ext != ".wav" else audio_path

    if max_chunks:
        from src.config import config as app_cfg
        trim_seconds = max_chunks * app_cfg.audio.window_size
        trimmed = os.path.join(output_dir, "trimmed.wav")
        subprocess.run(["ffmpeg", "-y", "-i", wav_path, "-t", str(trim_seconds),
                        "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", "-f", "wav", trimmed],
                       capture_output=True)
        wav_path = trimmed
        print(f"Trimmed to {trim_seconds}s\n")

    # Shared tools
    print("Loading models...")
    speech_scan = SpeechScanTool()
    classification = ClassificationTool()
    quality = QualityEvaluationTool(use_real_models=True)
    print("Models loaded.\n")

    kb = build_kb("high")
    request = AudioIngestRequest(source_id="ablation-test", raw_audio_path=wav_path)
    all_runs = []

    # ── 1. Fixed Baseline ──
    print(">>> 1. Fixed Baseline")
    d = os.path.join(output_dir, "fixed")
    os.makedirs(d, exist_ok=True)
    fixed = FixedBaselinePipeline(
        privacy_target="high", prepare_data_tool=PrepareDataTool(output_dir=d),
        speech_scan_tool=speech_scan,
        mid_band_tool=MidBandAttenuationTool(output_dir=d),
        strong_blur_tool=StrongBlurringTool(output_dir=d),
        classification_tool=classification, quality_tool=quality,
        data_lake_writer=DataLakeWriter(base_path=d),
    )
    r = fixed.run(request)
    all_runs.append(report_to_dict(r, "Fixed Baseline"))
    print(f"    Done: {r.succeeded_chunks}/{r.total_chunks} chunks\n")

    # ── 2. Adaptive Rule-Based ──
    print(">>> 2. Adaptive Rule-Based")
    d = os.path.join(output_dir, "rule_based")
    os.makedirs(d, exist_ok=True)
    mb = MidBandAttenuationTool(output_dir=d)
    sb = StrongBlurringTool(output_dir=d)
    agent_rb = AdaptivePrivacyControlAgent(mid_band_tool=mb, strong_blur_tool=sb, quality_tool=quality)
    pipe_rb = AgenticPipeline(
        privacy_target="high", kb=kb,
        prepare_data_tool=PrepareDataTool(output_dir=d),
        speech_scan_tool=speech_scan, agent=agent_rb,
        classification_tool=classification, quality_tool=quality,
        data_lake_writer=DataLakeWriter(base_path=d),
    )
    r = pipe_rb.run(request)
    all_runs.append(report_to_dict(r, "Adaptive Rule-Based"))
    print(f"    Done: {r.succeeded_chunks}/{r.total_chunks} chunks\n")

    # ── 3. Adaptive LLM (no memory) ──
    print(">>> 3. Adaptive LLM (no memory)")
    d = os.path.join(output_dir, "llm_no_mem")
    os.makedirs(d, exist_ok=True)
    mb = MidBandAttenuationTool(output_dir=d)
    sb = StrongBlurringTool(output_dir=d)
    ss = SourceSeparationTool(output_dir=d, mid_band_tool=mb, strong_blur_tool=sb)
    agent_nm = LLMNoMemoryAgent(mid_band_tool=mb, strong_blur_tool=sb, quality_tool=quality, source_separation_tool=ss)
    pipe_nm = AgenticPipeline(
        privacy_target="high", kb=kb,
        prepare_data_tool=PrepareDataTool(output_dir=d),
        speech_scan_tool=speech_scan, agent=agent_nm,
        classification_tool=classification, quality_tool=quality,
        data_lake_writer=DataLakeWriter(base_path=d),
    )
    r = pipe_nm.run(request)
    all_runs.append(report_to_dict(r, "Adaptive LLM (no memory)"))
    print(f"    Done: {r.succeeded_chunks}/{r.total_chunks} chunks\n")

    # ── 4. Adaptive LLM (with memory) ──
    print(">>> 4. Adaptive LLM (with memory)")
    d = os.path.join(output_dir, "llm_with_mem")
    os.makedirs(d, exist_ok=True)
    mb = MidBandAttenuationTool(output_dir=d)
    sb = StrongBlurringTool(output_dir=d)
    ss = SourceSeparationTool(output_dir=d, mid_band_tool=mb, strong_blur_tool=sb)
    agent_wm = LLMPrivacyControlAgent(mid_band_tool=mb, strong_blur_tool=sb, quality_tool=quality, source_separation_tool=ss)
    pipe_wm = AgenticPipeline(
        privacy_target="high", kb=kb,
        prepare_data_tool=PrepareDataTool(output_dir=d),
        speech_scan_tool=speech_scan, agent=agent_wm,
        classification_tool=classification, quality_tool=quality,
        data_lake_writer=DataLakeWriter(base_path=d),
    )
    r = pipe_wm.run(request)
    all_runs.append(report_to_dict(r, "Adaptive LLM (with memory)"))
    print(f"    Done: {r.succeeded_chunks}/{r.total_chunks} chunks\n")

    # ── Save results ──
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    log_path = os.path.join(LOG_DIR, f"ablation_{ts}.json")
    log_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "audio_file": os.path.basename(audio_path),
        "max_chunks": max_chunks,
        "pipelines": all_runs,
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    # ── Print summary ──
    print("=" * 70)
    print(f"{'Pipeline':<35s} {'Privacy':>10s} {'Preserve':>10s} {'Runtime':>10s}")
    print("=" * 70)
    for run in all_runs:
        s = run["summary"]
        rt = run.get("total_runtime_seconds", 0)
        print(f"{run['label']:<35s} {s['avg_privacy_score']:>10.4f} {s['avg_preserve_score']:>10.4f} {rt:>9.1f}s")

    print(f"\nLog saved to: {log_path}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
