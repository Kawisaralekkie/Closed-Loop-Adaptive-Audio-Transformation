#!/usr/bin/env python3
"""Integration test for the LLM Privacy Control Agent (Claude 3.5 Haiku via Bedrock).

Usage:
    python scripts/run_llm_integration_test.py <audio_file> [--max-chunks N] [--no-llm]

Runs 2 pipeline configurations using LLMPrivacyControlAgent:
  1. LLM Agentic — privacy_target="high"
  2. LLM Agentic — privacy_target="very_high"

Options:
  --max-chunks N   Limit to first N chunks (faster testing)
  --no-llm         Use rule-based fallback instead of Bedrock (offline testing)

Outputs:
  - Per-chunk report printed to console
  - JSON run log saved to logs/
  - LLM decision log saved to logs/ (for use with plot_llm_decisions.py)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.contracts.audio_contracts import AudioIngestRequest
from src.pipelines.agentic_pipeline import AgenticPipeline
from src.tools.prepare_data_tool import PrepareDataTool
from src.tools.speech_scan_tool import SpeechScanTool
from src.tools.mid_band_attenuation_tool import MidBandAttenuationTool
from src.tools.strong_blurring_tool import StrongBlurringTool
from src.tools.classification_tool import ClassificationTool
from src.tools.quality_evaluation_tool import QualityEvaluationTool
from src.tools.data_lake_writer import DataLakeWriter
from src.agents.llm_privacy_control_agent import LLMPrivacyControlAgent, save_decision_log
from src.tools.source_separation_tool import SourceSeparationTool
from src.knowledge_base.kb_loader import (
    KnowledgeBase,
    PolicyTransformationRules,
    PrivacyPlaybook,
    RecipeDefinition,
    SoundLabelTaxonomy,
)

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")


# ── Helpers ──────────────────────────────────────────────────────────────

def convert_to_wav(input_path: str, output_dir: str) -> str:
    """Convert any audio file to WAV PCM mono 16 kHz using ffmpeg."""
    stem = Path(input_path).stem
    wav_path = os.path.join(output_dir, f"{stem}.wav")
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
        "-f", "wav", wav_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    return wav_path


def build_mock_kb(privacy_target: str) -> KnowledgeBase:
    """Build a local KnowledgeBase (no S3 needed)."""
    return KnowledgeBase(
        version="llm-integration-test-1.0",
        manifest_hash="test-hash",
        policies=PolicyTransformationRules(
            privacy_target=privacy_target,
            allow_store_raw_audio=True,
            max_retention_days=90,
        ),
        playbook=PrivacyPlaybook(
            analyzer_required_features=[
                "speech_ratio", "vad_conf_mean",
                "hf_energy_ratio", "mid_energy_ratio",
            ],
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
                    risks=["Insufficient privacy for high speech ratio"],
                    mitigations=["Escalate atten_db on retry"],
                    auto_tune_rules={"atten_db": {"step": 10.0, "max": 40.0}},
                ),
                RecipeDefinition(
                    name="RECIPE_LOWPASS_HIGHPASS_MIX",
                    params={"lowpass_cutoff": 800, "lowpass_mix": 0.7, "noise_snr_db": 10.0},
                    use_when={"speech_ratio": {"min": 0.0, "max": 1.0}},
                    risks=["May degrade environmental sound quality"],
                    mitigations=["Monitor preserve_score"],
                    auto_tune_rules={"lowpass_cutoff": {"step": -100, "min": 400}},
                ),
            ],
            selection_strategy="score_then_try",
            max_trials=2,
            fallback_rules={},
            utility_preserve_target=0.80,
        ),
        taxonomy=SoundLabelTaxonomy(
            classes=["Speech", "Music", "Environmental"],
            mappings={"Speech": "human_voice", "Music": "music", "Environmental": "environment"},
            confidence_thresholds={"default": 0.3},
        ),
    )


def print_separator(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_run_report(report, label: str) -> None:
    """Pretty-print a RunReport."""
    print_separator(label)
    print(f"  run_id:           {report.run_id}")
    print(f"  total_chunks:     {report.total_chunks}")
    print(f"  succeeded_chunks: {report.succeeded_chunks}")
    print(f"  failed_chunks:    {report.failed_chunks}")
    print()

    for cr in report.chunks:
        print(f"  ── Chunk: {cr.chunk_id}")
        print(f"     had_speech:  {cr.had_speech}")
        print(f"     routing:     {cr.routing_decision}")
        print(f"     trials:      {cr.trials}")

        if cr.recipe_applied:
            print(f"     recipe:      {cr.recipe_applied.recipe_name}")

        if cr.metrics:
            pm = cr.metrics.privacy
            um = cr.metrics.utility
            d = cr.metrics.decision
            print(f"     privacy_score:  {pm.privacy_score:.4f}  (WER={pm.wer:.4f}, CER={pm.cer:.4f}, speaker={pm.speaker_privacy:.4f})")
            print(f"     preserve_score: {um.preserve_score:.4f}  (s_loud={um.sub_scores.s_loud:.3f}, s_hf={um.sub_scores.s_hf:.3f}, s_sc={um.sub_scores.s_sc:.3f}, s_con={um.sub_scores.s_con:.3f}, s_psy={um.sub_scores.s_psy:.3f})")
            print(f"     overall_pass:   {d.overall_pass}  (privacy_pass={d.privacy_pass}, preserve_pass={d.preserve_pass})")

        if cr.failure:
            print(f"     FAILURE: {cr.failure[:200]}")
        print()


# ── Logging ──────────────────────────────────────────────────────────────

def report_to_dict(report, label: str) -> dict:
    """Convert a RunReport to a JSON-serialisable dict."""
    chunks = []
    for cr in report.chunks:
        chunk_data: dict = {
            "chunk_id": cr.chunk_id,
            "had_speech": cr.had_speech,
            "speech_ratio": round(cr.speech_ratio, 4),
            "vad_confidence": round(cr.vad_confidence, 4),
            "routing": cr.routing_decision,
            "trials": cr.trials,
            "recipe": cr.recipe_applied.recipe_name if cr.recipe_applied else None,
        }
        if cr.metrics:
            pm = cr.metrics.privacy
            um = cr.metrics.utility
            d = cr.metrics.decision
            chunk_data["privacy"] = {
                "wer": round(pm.wer, 4),
                "cer": round(pm.cer, 4),
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
            chunk_data["utility"] = {
                "mAP": round(um.mAP, 4),
                "f1": round(um.f1, 4),
                "accuracy": round(um.accuracy, 4),
            }
            chunk_data["decision"] = {
                "privacy_pass": d.privacy_pass,
                "preserve_pass": d.preserve_pass,
                "overall_pass": d.overall_pass,
                "privacy_target": d.privacy_target,
                "privacy_score_min": d.privacy_score_min,
                "preserve_score_min": d.preserve_score_min,
            }
        if cr.failure:
            chunk_data["failure"] = cr.failure[:500]
        chunks.append(chunk_data)

    scored = [c for c in chunks if "privacy" in c]
    n = len(scored) or 1
    summary = {
        "total_chunks": report.total_chunks,
        "succeeded_chunks": report.succeeded_chunks,
        "failed_chunks": report.failed_chunks,
        "pass_rate": sum(1 for c in scored if c.get("decision", {}).get("overall_pass")) / n,
        "avg_privacy_score": round(sum(c["privacy"]["privacy_score"] for c in scored) / n, 4),
        "avg_preserve_score": round(sum(c["preserve"]["preserve_score"] for c in scored) / n, 4),
        "avg_wer": round(sum(c["privacy"]["wer"] for c in scored) / n, 4),
        "avg_cer": round(sum(c["privacy"]["cer"] for c in scored) / n, 4),
        "avg_speaker_privacy": round(sum(c["privacy"]["speaker_privacy"] for c in scored) / n, 4),
    }

    return {
        "label": label,
        "run_id": str(report.run_id),
        "summary": summary,
        "chunks": chunks,
    }


def save_run_log(all_runs: list[dict], audio_path: str, max_chunks: int | None) -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    log_path = os.path.join(LOG_DIR, f"run_llm_{ts}.json")
    log_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "audio_file": os.path.basename(audio_path),
        "max_chunks": max_chunks,
        "pipelines": all_runs,
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    return log_path


# ── Shared tool instances ────────────────────────────────────────────────

_tools_cache: dict = {}


def get_shared_tools(output_dir: str) -> dict:
    if _tools_cache:
        return _tools_cache

    print("Loading models (Silero VAD + YAMNet + Whisper + WavLM) — this may take a moment...")
    _tools_cache["prepare"] = PrepareDataTool(output_dir=output_dir)
    _tools_cache["speech_scan"] = SpeechScanTool()
    _tools_cache["mid_band"] = MidBandAttenuationTool(output_dir=output_dir)
    _tools_cache["strong_blur"] = StrongBlurringTool(output_dir=output_dir)
    _tools_cache["source_separation"] = SourceSeparationTool(
        output_dir=output_dir,
        mid_band_tool=_tools_cache["mid_band"],
        strong_blur_tool=_tools_cache["strong_blur"],
    )
    _tools_cache["classification"] = ClassificationTool()
    _tools_cache["quality"] = QualityEvaluationTool(use_real_models=True)
    _tools_cache["data_lake_writer"] = DataLakeWriter(base_path=os.path.join(output_dir, "data_lake"))
    print("Models loaded.\n")
    return _tools_cache


# ── Pipeline runner ──────────────────────────────────────────────────────

def run_llm_agentic(
    wav_path: str,
    privacy_target: str,
    output_dir: str,
    use_llm: bool = True,
) -> tuple:
    """Run the LLM Agentic Pipeline and return (report, label, decision_log)."""
    tools = get_shared_tools(output_dir)
    kb = build_mock_kb(privacy_target)

    agent = LLMPrivacyControlAgent(
        mid_band_tool=tools["mid_band"],
        strong_blur_tool=tools["strong_blur"],
        quality_tool=tools["quality"],
        source_separation_tool=tools["source_separation"],
        use_llm=use_llm,
    )

    pipeline = AgenticPipeline(
        privacy_target=privacy_target,
        kb=kb,
        prepare_data_tool=tools["prepare"],
        speech_scan_tool=tools["speech_scan"],
        agent=agent,
        classification_tool=tools["classification"],
        quality_tool=tools["quality"],
        data_lake_writer=tools["data_lake_writer"],
    )

    mode = "LLM" if use_llm else "Fallback"
    label = f"LLM Agentic ({mode}) — privacy_target={privacy_target!r}"
    print(f"\n>>> Starting {label} ...", flush=True)

    request = AudioIngestRequest(
        source_id="llm-integration-test",
        raw_audio_path=wav_path,
    )
    report = pipeline.run(request)
    print_run_report(report, label)

    # Collect decision log from agent
    decision_log = getattr(agent, "decision_log", [])
    return report, label, decision_log


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_llm_integration_test.py <audio_file> [--max-chunks N] [--no-llm]")
        sys.exit(1)

    audio_path = sys.argv[1]
    if not os.path.isfile(audio_path):
        print(f"File not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    # Parse options
    max_chunks = None
    use_llm = True
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "--max-chunks" and i + 1 < len(args):
            max_chunks = int(args[i + 1])
            i += 2
        elif args[i] == "--no-llm":
            use_llm = False
            i += 1
        else:
            i += 1

    if max_chunks:
        print(f"Limiting to first {max_chunks} chunks per pipeline run.")
    if not use_llm:
        print("Running in FALLBACK mode (no Bedrock calls).")
    print()

    output_dir = tempfile.mkdtemp(prefix="llm_integration_test_")
    print(f"Output directory: {output_dir}\n")

    # Convert to WAV if needed
    ext = Path(audio_path).suffix.lower()
    if ext != ".wav":
        print(f"Converting {ext} → WAV PCM mono 16 kHz ...")
        wav_path = convert_to_wav(audio_path, output_dir)
        print(f"Converted: {wav_path}\n")
    else:
        wav_path = audio_path

    # Trim if --max-chunks
    if max_chunks is not None:
        from src.config import config as app_cfg
        trim_seconds = max_chunks * app_cfg.audio.window_size
        trimmed_path = os.path.join(output_dir, "trimmed.wav")
        trim_cmd = [
            "ffmpeg", "-y", "-i", wav_path,
            "-t", str(trim_seconds),
            "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
            "-f", "wav", trimmed_path,
        ]
        subprocess.run(trim_cmd, capture_output=True, text=True)
        wav_path = trimmed_path
        print(f"Trimmed to {trim_seconds}s → ~{max_chunks} chunks\n")

    # ── Run LLM Agentic pipelines ──
    all_runs: list[dict] = []
    all_decisions: list[dict] = []

    # 1. LLM Agentic — high
    report, label, decisions = run_llm_agentic(wav_path, "high", output_dir, use_llm)
    all_runs.append(report_to_dict(report, label))
    all_decisions.extend(decisions)

    # 2. LLM Agentic — very_high
    report, label, decisions = run_llm_agentic(wav_path, "very_high", output_dir, use_llm)
    all_runs.append(report_to_dict(report, label))
    all_decisions.extend(decisions)

    # ── Save logs ──
    log_path = save_run_log(all_runs, audio_path, max_chunks)

    tag = "llm" if use_llm else "fallback"
    decision_log_path = save_decision_log(all_decisions, out_dir=LOG_DIR, tag=tag)

    print_separator("Done")
    print(f"  Run log saved to:      {log_path}")
    print(f"  Decision log saved to: {decision_log_path}")
    print(f"  Output directory:      {output_dir}")
    print()
    print(f"  To visualize decisions:")
    print(f"    python scripts/plot_llm_decisions.py {decision_log_path}")
    print()


if __name__ == "__main__":
    main()
