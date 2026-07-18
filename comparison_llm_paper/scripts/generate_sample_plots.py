#!/usr/bin/env python3
"""Generate sample .png plots with mock data to illustrate all chart outputs.

Creates mock JSON log, then calls plot_fixed_vs_agentic_q1q2q3.py to produce
example images for Q1 (Privacy), Q2 (Speech→Transform), Q3 (Preservation).

Output: plots/samples/
"""

import json
import os
import random
import uuid
import importlib.util
import sys

import numpy as np

random.seed(42)
np.random.seed(42)

OUT_DIR = "plots/samples"
MOCK_DIR = "plots/samples/_mock"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MOCK_DIR, exist_ok=True)

RECIPES = ["RECIPE_MID_BAND_ATTEN", "RECIPE_LOWPASS_HIGHPASS_MIX"]


def rand(lo, hi):
    return round(random.uniform(lo, hi), 4)


def make_chunk(idx, target, pipeline_type):
    """Generate a realistic mock chunk."""
    cid = str(uuid.uuid4())
    had_speech = random.random() > 0.15  # 85% have speech

    if not had_speech:
        return {
            "chunk_id": cid, "chunk_index": idx, "had_speech": False,
            "speech_ratio": 0.0, "vad_confidence": 0.0,
            "recipe": None, "trials": 1,
            "privacy": {"privacy_score": 0, "speaker_privacy": 0, "content_privacy": 0, "cer": 0, "wer": 0},
            "preserve": {"preserve_score": 1.0, "s_loud": 1.0, "s_hf": 1.0, "s_sc": 1.0, "s_con": 1.0, "s_psy": 1.0},
            "utility": {"mAP": rand(0.7, 0.95), "f1": rand(0.7, 0.95), "accuracy": rand(0.7, 0.95)},
            "decision": {},
        }

    speech_ratio = rand(0.1, 0.95)
    vad_conf = rand(0.5, 0.99)
    recipe = random.choice(RECIPES)

    # Fixed pipelines: consistent but less adaptive
    if "Fixed" in pipeline_type:
        if target == "very_high":
            priv = rand(0.75, 0.95)
            pres = rand(0.55, 0.75)
            spk = rand(0.5, 0.85)
        else:
            priv = rand(0.6, 0.8)
            pres = rand(0.7, 0.88)
            spk = rand(0.35, 0.65)
        trials = 1
    else:
        # Agentic: better trade-off
        if target == "very_high":
            priv = rand(0.8, 0.97)
            pres = rand(0.65, 0.85)
            spk = rand(0.55, 0.9)
        else:
            priv = rand(0.65, 0.88)
            pres = rand(0.78, 0.95)
            spk = rand(0.4, 0.7)
        trials = random.randint(1, 4)

    cer = rand(0.3, 0.9)
    wer = rand(0.3, 0.9)
    content_priv = round((cer + wer) / 2, 4)
    s_loud = rand(0.6, 1.0)
    s_hf = rand(0.5, 1.0)
    s_sc = rand(0.6, 1.0)
    s_con = rand(0.5, 0.95)
    s_psy = rand(0.55, 0.95)

    return {
        "chunk_id": cid, "chunk_index": idx, "had_speech": True,
        "speech_ratio": speech_ratio, "vad_confidence": vad_conf,
        "recipe": recipe, "trials": trials,
        "privacy": {
            "privacy_score": priv, "speaker_privacy": spk,
            "content_privacy": content_priv, "cer": cer, "wer": wer,
        },
        "preserve": {
            "preserve_score": pres,
            "s_loud": s_loud, "s_hf": s_hf, "s_sc": s_sc, "s_con": s_con, "s_psy": s_psy,
        },
        "utility": {
            "mAP": rand(0.5, 0.95), "f1": rand(0.5, 0.95), "accuracy": rand(0.55, 0.95),
        },
        "decision": {"privacy_score_min": 0.65 if target == "high" else 0.80},
    }


def make_pipeline(label, target, pipeline_type, n_chunks=6):
    chunks = [make_chunk(i, target, pipeline_type) for i in range(n_chunks)]
    speech = [c for c in chunks if c["had_speech"]]
    n = max(len(speech), 1)
    return {
        "label": label,
        "chunks": chunks,
        "summary": {
            "avg_privacy_score": round(sum(c["privacy"]["privacy_score"] for c in speech) / n, 4),
            "avg_preserve_score": round(sum(c["preserve"]["preserve_score"] for c in speech) / n, 4),
            "avg_speaker_privacy": round(sum(c["privacy"]["speaker_privacy"] for c in speech) / n, 4),
            "total_chunks": len(chunks),
            "speech_chunks": len(speech),
        },
    }


def make_run_log():
    return {
        "audio_file": "sample_urban_audio.wav",
        "timestamp": "2026-03-24T10:00:00",
        "pipelines": [
            make_pipeline("Fixed Baseline — privacy_target='high'", "high", "Fixed"),
            make_pipeline("Fixed Baseline — privacy_target='very_high'", "very_high", "Fixed"),
            make_pipeline("Agentic Pipeline — privacy_target='high'", "high", "Agentic"),
            make_pipeline("Agentic Pipeline — privacy_target='very_high'", "very_high", "Agentic"),
        ],
    }


def run_script_main(script_path, argv_override):
    """Import a script and call its main() with overridden sys.argv."""
    spec = importlib.util.spec_from_file_location("mod", script_path)
    mod = importlib.util.module_from_spec(spec)
    old_argv = sys.argv
    sys.argv = argv_override
    try:
        spec.loader.exec_module(mod)
        if hasattr(mod, "main"):
            mod.main()
    except SystemExit:
        pass
    finally:
        sys.argv = old_argv


# ── Generate mock data ────────────────────────────────────────────────
print("Generating mock data...")
run_data = make_run_log()
run_path = os.path.join(MOCK_DIR, "run_log.json")
with open(run_path, "w") as f:
    json.dump(run_data, f, indent=2)
print(f"  Mock run log: {run_path}")
print()

# ── Run combined Q1+Q2+Q3 plot script ────────────────────────────────
print("=" * 60)
print("Fixed vs Agentic — Q1 + Q2 + Q3 (22 charts)")
print("=" * 60)
run_script_main("scripts/plot_fixed_vs_agentic_q1q2q3.py", [
    "plot_fixed_vs_agentic_q1q2q3.py", run_path,
    "--out-dir", os.path.join(OUT_DIR, "fixed_vs_agentic_q1q2q3"),
])

# ── Summary ───────────────────────────────────────────────────────────
print()
print("=" * 60)
total = 0
for root, dirs, files in os.walk(OUT_DIR):
    pngs = [f for f in files if f.endswith(".png")]
    if pngs:
        rel = os.path.relpath(root, OUT_DIR)
        print(f"  {rel}/  ({len(pngs)} charts)")
        for p in sorted(pngs):
            print(f"    - {p}")
        total += len(pngs)
print(f"\nTotal: {total} sample charts in {OUT_DIR}/")
