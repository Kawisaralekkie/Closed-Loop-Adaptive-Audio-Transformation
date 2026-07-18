#!/usr/bin/env python3
"""Compare v1 (original) vs v2 (revised) pipeline parameters.

Runs Fixed Baseline with both v1 and v2 parameter sets on the same audio,
then saves a combined JSON log for plotting.

Usage:
    python scripts/run_v1_vs_v2_comparison.py <audio_file> [--max-chunks N]

v1 params (original):
  MidBand:    band_hz=(500,3000), atten_db=25.0, no pitch shift
  StrongBlur: lowpass_cutoff=800, lowpass_mix=0.70, highband_start=2000,
              highband_mix=0.30, midband_range=(1000,2000), midband_gain_db=3.0,
              noise_band=(500,3000), noise_snr_db=10.0

v2 params (revised — current):
  MidBand:    band_hz=(500,3000), atten_db=20.0, no pitch shift
  StrongBlur: lowpass_cutoff=1000, lowpass_mix=0.55, highband_start=2500,
              highband_mix=0.15, midband_range=(1200,2200), midband_gain_db=2.0,
              noise_band=(700,2800), noise_snr_db=18.0
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

from src.contracts.audio_contracts import AudioChunk, AudioIngestRequest
from src.contracts.report_contracts import ChunkReport, RunReport
from src.tools.prepare_data_tool import PrepareDataTool
from src.tools.speech_scan_tool import SpeechScanTool
from src.tools.mid_band_attenuation_tool import MidBandAttenuationTool
from src.tools.strong_blurring_tool import StrongBlurringTool
from src.tools.classification_tool import ClassificationTool
from src.tools.quality_evaluation_tool import QualityEvaluationTool

import numpy as np

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")


# ── Parameter sets ────────────────────────────────────────────────────

V1_MIDBAND = {"band_hz": (500, 3000), "atten_db": 25.0}
V2_MIDBAND = {"band_hz": (500, 3000), "atten_db": 20.0}

V1_STRONG = {
    "lowpass_cutoff": 800, "lowpass_mix": 0.70,
    "highband_start": 2000, "highband_mix": 0.30,
    "midband_range": (1000, 2000), "midband_gain_db": 3.0,
    "noise_band": (500, 3000), "noise_snr_db": 10.0,
}
V2_STRONG = {
    "lowpass_cutoff": 1000, "lowpass_mix": 0.55,
    "highband_start": 2500, "highband_mix": 0.15,
    "midband_range": (1200, 2200), "midband_gain_db": 2.0,
    "noise_band": (700, 2800), "noise_snr_db": 18.0,
}

CONFIGS = [
    ("v1 MidBand (high)", "high", "midband", V1_MIDBAND),
    ("v2 MidBand (high)", "high", "midband", V2_MIDBAND),
    ("v1 StrongBlur (very_high)", "very_high", "strong", V1_STRONG),
    ("v2 StrongBlur (very_high)", "very_high", "strong", V2_STRONG),
]


# ── Helpers ───────────────────────────────────────────────────────────

def convert_to_wav(input_path: str, output_dir: str) -> str:
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


_tools: dict = {}


def get_tools(output_dir: str) -> dict:
    if _tools:
        return _tools
    print("Loading models (Silero VAD + YAMNet + Whisper + ECAPA-TDNN)...")
    _tools["prepare"] = PrepareDataTool(output_dir=output_dir)
    _tools["speech_scan"] = SpeechScanTool()
    _tools["mid_band"] = MidBandAttenuationTool(output_dir=output_dir)
    _tools["strong_blur"] = StrongBlurringTool(output_dir=output_dir)
    _tools["classification"] = ClassificationTool()
    _tools["quality"] = QualityEvaluationTool(use_real_models=True)
    print("Models loaded.\n")
    return _tools


def report_to_dict(report: RunReport, label: str) -> dict:
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
            chunk_data["utility"] = {
                "mAP": round(um.mAP, 4), "f1": round(um.f1, 4),
                "accuracy": round(um.accuracy, 4),
            }
            chunk_data["decision"] = {
                "privacy_pass": d.privacy_pass, "preserve_pass": d.preserve_pass,
                "overall_pass": d.overall_pass, "privacy_target": d.privacy_target,
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
    }
    return {"label": label, "run_id": str(report.run_id), "summary": summary, "chunks": chunks}


# ── Pipeline runner ───────────────────────────────────────────────────

def run_config(
    wav_path: str, label: str, privacy_target: str,
    tool_type: str, params: dict, output_dir: str,
) -> RunReport:
    """Run a single Fixed Baseline config with explicit params."""
    tools = get_tools(output_dir)
    timestamp = datetime.now(timezone.utc).isoformat()
    ingest = tools["prepare"].run(
        AudioIngestRequest(source_id="v1v2-compare", raw_audio_path=wav_path),
        timestamp=timestamp,
    )
    run_id = ingest.run_id
    chunk_reports: list[ChunkReport] = []

    for chunk in ingest.chunks:
        try:
            vad = tools["speech_scan"].run(chunk)
            vad_conf = (
                sum(s.confidence for s in vad.segments) / len(vad.segments)
                if vad.segments else 0.0
            )

            if vad.has_speech:
                if tool_type == "midband":
                    tr = tools["mid_band"].run(
                        wav_path=chunk.wav_path, segments=vad.segments,
                        chunk_id=chunk.chunk_id, **params,
                    )
                else:
                    tr = tools["strong_blur"].run(
                        wav_path=chunk.wav_path, segments=vad.segments,
                        chunk_id=chunk.chunk_id, **params,
                    )
                processed = AudioChunk(
                    chunk_id=chunk.chunk_id, run_id=run_id,
                    wav_path=tr.blurred_wav_path,
                    start_time=chunk.start_time, end_time=chunk.end_time,
                    duration=chunk.duration,
                )
                recipe_ref = tr.recipe_ref
                params_applied = tr.params
            else:
                processed = chunk
                recipe_ref = None
                params_applied = None

            cls = tools["classification"].run(processed)
            metrics = tools["quality"].run(
                original_chunk=chunk, processed_chunk=processed,
                classification_result=cls, privacy_target=privacy_target,
            )

            cr = ChunkReport(
                chunk_id=chunk.chunk_id, run_id=run_id,
                had_speech=vad.has_speech,
                speech_ratio=vad.speech_ratio, vad_confidence=vad_conf,
                recipe_applied=recipe_ref, params_applied=params_applied,
                trials=1 if vad.has_speech else 0,
                metrics=metrics,
                routing_decision="blurred" if vad.has_speech else "bypass",
            )
        except Exception as exc:
            import traceback
            cr = ChunkReport(
                chunk_id=chunk.chunk_id, run_id=run_id,
                had_speech=False, trials=0,
                routing_decision="error", failure=traceback.format_exc(),
            )
        chunk_reports.append(cr)

    succeeded = sum(1 for r in chunk_reports if r.failure is None)
    failed = sum(1 for r in chunk_reports if r.failure is not None)

    return RunReport(
        run_id=run_id, source_id="v1v2-compare",
        kb_version="N/A", model_versions={"silero_vad": "5.1", "yamnet": "1"},
        config_params={"label": label, "privacy_target": privacy_target, **params},
        chunks=chunk_reports,
        total_chunks=len(chunk_reports),
        succeeded_chunks=succeeded, failed_chunks=failed,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_v1_vs_v2_comparison.py <audio_file> [--max-chunks N]")
        sys.exit(1)

    audio_path = sys.argv[1]
    if not os.path.isfile(audio_path):
        print(f"File not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    max_chunks = None
    if "--max-chunks" in sys.argv:
        idx = sys.argv.index("--max-chunks")
        if idx + 1 < len(sys.argv):
            max_chunks = int(sys.argv[idx + 1])

    output_dir = tempfile.mkdtemp(prefix="v1v2_compare_")
    print(f"Output directory: {output_dir}\n")

    ext = Path(audio_path).suffix.lower()
    if ext != ".wav":
        print(f"Converting {ext} → WAV...")
        wav_path = convert_to_wav(audio_path, output_dir)
    else:
        wav_path = audio_path

    if max_chunks is not None:
        from src.config import config as app_cfg
        trim_seconds = max_chunks * app_cfg.audio.window_size
        trimmed = os.path.join(output_dir, "trimmed.wav")
        subprocess.run([
            "ffmpeg", "-y", "-i", wav_path,
            "-t", str(trim_seconds), "-ar", "16000", "-ac", "1",
            "-sample_fmt", "s16", "-f", "wav", trimmed,
        ], capture_output=True, text=True)
        wav_path = trimmed
        print(f"Trimmed to {trim_seconds}s → ~{max_chunks} chunks\n")

    all_runs: list[dict] = []
    for label, target, tool_type, params in CONFIGS:
        print(f"\n>>> Running: {label} ...", flush=True)
        report = run_config(wav_path, label, target, tool_type, params, output_dir)
        run_dict = report_to_dict(report, label)

        scored = [c for c in run_dict["chunks"] if "privacy" in c]
        n = len(scored) or 1
        avg_priv = sum(c["privacy"]["privacy_score"] for c in scored) / n
        avg_pres = sum(c["preserve"]["preserve_score"] for c in scored) / n
        pass_rate = sum(1 for c in scored if c.get("decision", {}).get("overall_pass")) / n
        print(f"    avg_privacy={avg_priv:.4f}  avg_preserve={avg_pres:.4f}  pass_rate={pass_rate:.0%}")
        all_runs.append(run_dict)

    # Save log
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    log_path = os.path.join(LOG_DIR, f"run_v1v2_{ts}.json")
    log_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "audio_file": os.path.basename(audio_path),
        "max_chunks": max_chunks,
        "comparison": "v1_vs_v2_parameters",
        "pipelines": all_runs,
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  Log saved to: {log_path}")
    print(f"  Output dir:   {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
