#!/usr/bin/env python3
"""Verify our understanding of signal ENERGY from first principles.

Energy of a discrete signal x[n] is E = Σ x[n]²  (sum of squared amplitudes).
This script computes E several independent ways and checks they all agree, to
prove we understand the basics rather than trusting one library call blindly:

  1. manual_loop   : a plain Python for-loop  Σ x[i]*x[i]     (ground truth)
  2. numpy_sumsq   : np.sum(x**2)             (vectorised library call)
  3. numpy_dot     : np.dot(x, x)             (BLAS inner product)
  4. numpy_norm    : np.linalg.norm(x)**2     (L2 norm squared)
  5. from_rms      : N * mean(x**2)           (energy ↔ RMS relationship)
  6. parseval_fft  : Σ|FFT(x)|² / N           (Parseval's theorem: time↔freq)

Sampling mode: give an npz ROOT and the run_metrics_per_chunk.csv, and the
script randomly samples chunks — balanced across ENV-only and SPEECH chunks —
so you can see the check hold for both kinds of audio.

Usage:
    # sample 10 chunks (≈5 env + 5 speech) from the fixed npz tree
    python3 scripts/verify_energy.py logs/amplitude_npz/fixed \
        --run-metrics-csv logs/s3/20260701_171530/run_metrics_per_chunk.csv \
        --sample 10 --seed 42

    # a single npz file
    python3 scripts/verify_energy.py "<path>_amplitude.npz"
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import random

import numpy as np


# ---------------------------------------------------------------------------
# npz path parsing + env/speech classification
# ---------------------------------------------------------------------------
def _parse_npz(npz_path: str):
    parts = npz_path.replace("\\", "/").split("/")
    source_file = next((p for p in parts if p.endswith(".wav")), "")
    base = os.path.basename(npz_path)
    run_id = base.split("_", 1)[0]
    stem = base.replace("_amplitude.npz", "")
    chunk_index = stem.rsplit("_", 1)[-1] if "_" in stem else ""
    return source_file, chunk_index, run_id


def _build_lookup(csv_path: str) -> dict:
    """(source_file, chunk_index) -> {had_speech, run_id} for mode=fixed."""
    lut = {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("mode") != "fixed":
                continue
            key = (r.get("source_file", ""), str(r.get("chunk_index", "")))
            lut[key] = {
                "had_speech": str(r.get("had_speech", "")).lower() == "true",
                "run_id": (r.get("run_id", "") or "").strip(),
            }
    return lut


def _sample_balanced(npz_root: str, lut: dict, n: int, seed: int):
    """Return a list of (npz_path, type) balanced between env and speech."""
    files = sorted(glob.glob(os.path.join(npz_root, "**", "*.npz"), recursive=True))
    env, speech = [], []
    for f in files:
        src, cidx, run_id = _parse_npz(f)
        info = lut.get((src, cidx))
        if info is None or (info["run_id"] and run_id != info["run_id"]):
            continue  # unmatched or stale-duplicate npz
        (speech if info["had_speech"] else env).append(f)
    rng = random.Random(seed)
    rng.shuffle(env)
    rng.shuffle(speech)
    half = n // 2
    picked = [(p, "env") for p in env[:half]] + [(p, "speech") for p in speech[:n - half]]
    rng.shuffle(picked)
    return picked


# ---------------------------------------------------------------------------
# Energy computations
# ---------------------------------------------------------------------------
def _energy_manual_loop(x: np.ndarray) -> float:
    total = 0.0
    for v in x.tolist():
        total += v * v
    return total


def _energies(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    Xf = np.fft.fft(x)
    return {
        "manual_loop": _energy_manual_loop(x),
        "numpy_sumsq": float(np.sum(x ** 2)),
        "numpy_dot": float(np.dot(x, x)),
        "numpy_norm": float(np.linalg.norm(x) ** 2),
        "from_rms": float(n * np.mean(x ** 2)),
        "parseval_fft": float(np.sum(np.abs(Xf) ** 2) / n),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify signal energy from first principles")
    ap.add_argument("path", help="A .npz file OR an npz root directory to sample from")
    ap.add_argument("--run-metrics-csv", default=None,
                    help="run_metrics_per_chunk.csv — enables env/speech balanced sampling")
    ap.add_argument("--sample", type=int, default=10, help="How many chunks to sample (default 10)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--which", choices=["original", "processed"], default="original")
    args = ap.parse_args()

    # Build the list of (npz_path, type) to check.
    if os.path.isfile(args.path) and args.path.endswith(".npz"):
        items = [(args.path, "?")]
    elif os.path.isdir(args.path) and args.run_metrics_csv:
        lut = _build_lookup(args.run_metrics_csv)
        items = _sample_balanced(args.path, lut, args.sample, args.seed)
    elif os.path.isdir(args.path):
        files = sorted(glob.glob(os.path.join(args.path, "**", "*.npz"), recursive=True))
        random.Random(args.seed).shuffle(files)
        items = [(f, "?") for f in files[:args.sample]]
    else:
        raise SystemExit(f"No .npz found for {args.path}")
    if not items:
        raise SystemExit("No chunks selected (check --run-metrics-csv / npz root).")

    key = "amp_original" if args.which == "original" else "amp_processed"
    worst = 0.0
    n_env = n_speech = 0
    print(f"Verifying energy on {len(items)} sampled chunks ({key})\n")
    for npz_path, typ in items:
        d = np.load(npz_path)
        if key not in d.files:
            print(f"SKIP {os.path.basename(npz_path)}: no {key}")
            continue
        if typ == "env":
            n_env += 1
        elif typ == "speech":
            n_speech += 1
        x = np.asarray(d[key], dtype=np.float64)
        e = _energies(x)
        ref = e["manual_loop"]
        rels = {k: abs(v - ref) / (abs(ref) + 1e-30) for k, v in e.items()}
        chunk_worst = max(rels.values())
        worst = max(worst, chunk_worst)
        rms = float(np.sqrt(np.mean(x ** 2)))
        src, cidx, _ = _parse_npz(npz_path)
        print("=" * 78)
        print(f"[{typ.upper():6s}] {src} chunk {cidx}   (N={len(x)}, RMS={rms:.5f})")
        print("-" * 78)
        for k, v in e.items():
            print(f"  {k:14s} = {v:16.8f}   rel_diff = {rels[k]:.2e}")
        print(f"  max rel_diff this chunk = {chunk_worst:.2e}")
        print()

    print("#" * 78)
    print(f"Sampled: {n_env} env-only + {n_speech} speech")
    print(f"Max relative difference across ALL methods & chunks: {worst:.2e}")
    if worst < 1e-6:
        print("✓ VERIFIED: manual formula == library functions == Parseval, "
              "for BOTH env and speech chunks (diffs are float round-off only).")
    else:
        print("⚠ Methods disagree by > 1e-6 — investigate.")


if __name__ == "__main__":
    main()
