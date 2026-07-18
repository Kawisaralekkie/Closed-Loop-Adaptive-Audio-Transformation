#!/usr/bin/env python3
"""Compute REAL amplitude stats per chunk, straight from the .npz sample arrays.

Unlike the numbers in the *_report.json (which the pipeline pre-computed),
this reads the full 64000-sample amplitude arrays inside each .npz and
computes the statistics itself — no approximation, no re-aggregation:

    for amp_original AND amp_processed, per chunk:
        min      real minimum sample value
        max      real maximum sample value
        mean     real arithmetic mean of all samples
        peak     real max(|sample|)   (absolute-value peak)
        rms      real sqrt(mean(x^2))
        n        number of samples

The npz tree is expected to look like:
    logs/amplitude_npz/<mode>/cityspeechmix/cityspeechmixed/<source>.wav/<run>_<idx>_amplitude.npz

so `mode`, `source_file`, and `chunk_index` are parsed from the path.

Output: one CSV row per npz (per chunk).
    npz_amplitude_stats_per_chunk.csv   (or --out)

Usage:
    # everything under logs/amplitude_npz  (WARNING: thousands of files)
    python3 scripts/npz_amplitude_stats.py logs/amplitude_npz

    # just one mode
    python3 scripts/npz_amplitude_stats.py logs/amplitude_npz/fixed --out fixed_amp.csv

    # quick test on the first 20 npz
    python3 scripts/npz_amplitude_stats.py logs/amplitude_npz --limit 20
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np

KNOWN_MODES = {
    "fixed", "fixed_strong", "rule_based", "rule_based_strong",
    "rule_based_ss", "llm_no_memory", "llm_with_memory",
    "llm_with_memory_no_ss", "llm_no_memory_no_ss",
}

FIELDNAMES = [
    "mode", "source_file", "chunk_index", "npz_file",
    "n_samples",
    "orig_min", "orig_max", "orig_mean", "orig_peak", "orig_rms",
    "proc_min", "proc_max", "proc_mean", "proc_peak", "proc_rms",
    "diff_max_abs", "diff_rms",
]


def _parse_path(npz_path: str) -> tuple[str, str, str]:
    """Extract (mode, source_file, chunk_index) from the npz path."""
    parts = npz_path.replace("\\", "/").split("/")
    mode = next((p for p in parts if p in KNOWN_MODES), "")
    # source_file = the '<something>.wav' directory that holds the npz
    source_file = ""
    for p in parts:
        if p.endswith(".wav"):
            source_file = p
            break
    # chunk index = trailing _<n>_amplitude in the filename
    base = os.path.basename(npz_path)
    chunk_index = ""
    stem = base.replace("_amplitude.npz", "")
    if "_" in stem:
        tail = stem.rsplit("_", 1)[-1]
        chunk_index = tail if tail.isdigit() else ""
    return mode, source_file, chunk_index


def _stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "peak": float(np.max(np.abs(x))),
        "rms": float(np.sqrt(np.mean(x * x))),
    }


def _row(npz_path: str) -> dict | None:
    try:
        d = np.load(npz_path)
    except Exception as exc:
        print(f"  SKIP {npz_path}: {exc}")
        return None

    if "amp_original" not in d.files:
        print(f"  SKIP {npz_path}: no amp_original")
        return None

    orig = d["amp_original"]
    proc = d["amp_processed"] if "amp_processed" in d.files else orig
    o = _stats(orig)
    p = _stats(proc)

    n = min(len(orig), len(proc))
    diff = np.asarray(orig[:n], dtype=np.float64) - np.asarray(proc[:n], dtype=np.float64)

    mode, source_file, chunk_index = _parse_path(npz_path)
    return {
        "mode": mode,
        "source_file": source_file,
        "chunk_index": chunk_index,
        "npz_file": os.path.basename(npz_path),
        "n_samples": int(len(orig)),
        "orig_min": round(o["min"], 8),
        "orig_max": round(o["max"], 8),
        "orig_mean": round(o["mean"], 8),
        "orig_peak": round(o["peak"], 8),
        "orig_rms": round(o["rms"], 8),
        "proc_min": round(p["min"], 8),
        "proc_max": round(p["max"], 8),
        "proc_mean": round(p["mean"], 8),
        "proc_peak": round(p["peak"], 8),
        "proc_rms": round(p["rms"], 8),
        "diff_max_abs": round(float(np.max(np.abs(diff))), 8),
        "diff_rms": round(float(np.sqrt(np.mean(diff * diff))), 8),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Real per-chunk amplitude stats from npz")
    ap.add_argument("root", help="Folder to scan for *.npz (searched recursively)")
    ap.add_argument("--out", default=None,
                    help="Output CSV path (default: npz_amplitude_stats_per_chunk.csv beside root)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N npz files (quick test)")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.exists(root):
        raise SystemExit(f"Not found: {root}")

    if os.path.isfile(root) and root.endswith(".npz"):
        npz_files = [root]
        base_dir = os.path.dirname(root)
    else:
        npz_files = sorted(glob.glob(os.path.join(root, "**", "*.npz"), recursive=True))
        base_dir = root

    if not npz_files:
        raise SystemExit(f"No .npz files under {root}")
    if args.limit:
        npz_files = npz_files[:args.limit]

    out_path = os.path.abspath(args.out) if args.out \
        else os.path.join(base_dir, "npz_amplitude_stats_per_chunk.csv")

    print(f"Processing {len(npz_files)} npz file(s) -> {out_path}\n")
    n_ok = 0
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for i, npz_path in enumerate(npz_files, 1):
            row = _row(npz_path)
            if row:
                w.writerow(row)
                n_ok += 1
            if i % 500 == 0:
                print(f"  [{i}/{len(npz_files)}] processed")

    print(f"\nDone. Wrote {n_ok} rows to {out_path}")


if __name__ == "__main__":
    main()
