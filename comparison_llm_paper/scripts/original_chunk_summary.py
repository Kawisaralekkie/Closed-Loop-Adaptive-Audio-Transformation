#!/usr/bin/env python3
"""Per-chunk summary of the ORIGINAL audio: amplitude + spectrum stats.

The original (pre-blur) audio is identical across every pipeline mode, so we
read ONE mode's npz tree (default: fixed) and de-duplicate against the report
run_id — leaving exactly one row per unique chunk (e.g. 2226 chunks).

For each chunk it computes, from the REAL sample arrays (no approximation):

  amplitude (amp_original):
      amp_min, amp_max, amp_mean, amp_peak (=max|x|), amp_rms
  spectrum (spectrum_original + freqs_hz):
      n_freq_bins, freq_min_hz, freq_max_hz,
      spec_min, spec_max, spec_mean, spec_peak, spec_rms

Output: one CSV row per chunk.

Usage:
    python3 scripts/original_chunk_summary.py \
        logs/amplitude_npz/fixed \
        logs/s3/20260701_171530/run_metrics_per_chunk.csv \
        --out logs/s3/20260701_171530/original_chunk_summary.csv
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
    "source_file", "chunk_index", "run_id", "n_samples",
    "amp_min", "amp_max", "amp_mean", "amp_peak", "amp_rms",
    "n_freq_bins", "freq_min_hz", "freq_max_hz",
    "spec_min", "spec_max", "spec_mean", "spec_peak", "spec_rms",
]


def _parse_path(npz_path: str):
    parts = npz_path.replace("\\", "/").split("/")
    mode = next((p for p in parts if p in KNOWN_MODES), "")
    source_file = next((p for p in parts if p.endswith(".wav")), "")
    base = os.path.basename(npz_path)
    run_id = base.split("_", 1)[0]
    stem = base.replace("_amplitude.npz", "")
    chunk_index = stem.rsplit("_", 1)[-1] if "_" in stem else ""
    chunk_index = chunk_index if chunk_index.isdigit() else ""
    return mode, source_file, chunk_index, run_id


def _build_run_id_lookup(per_chunk_csv: str, mode_filter: str) -> dict:
    """(source_file, chunk_index) -> report run_id, for the chosen mode."""
    lut = {}
    with open(per_chunk_csv, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("mode", "") != mode_filter:
                continue
            lut[(r.get("source_file", ""), str(r.get("chunk_index", "")))] = \
                (r.get("run_id", "") or "").strip()
    return lut


def _stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {
        "min": float(x.min()), "max": float(x.max()),
        "mean": float(x.mean()), "peak": float(np.max(np.abs(x))),
        "rms": float(np.sqrt(np.mean(x * x))),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-chunk ORIGINAL amplitude + spectrum summary")
    ap.add_argument("npz_root", help="Root of ONE mode's npz tree (e.g. logs/amplitude_npz/fixed)")
    ap.add_argument("per_chunk_csv", help="run_metrics_per_chunk.csv (for run_id dedup)")
    ap.add_argument("--mode", default=None,
                    help="Mode name to match in the report (default: inferred from npz_root)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    # infer mode from the root path if not given
    mode = args.mode
    if not mode:
        for p in args.npz_root.replace("\\", "/").split("/"):
            if p in KNOWN_MODES:
                mode = p
                break
    if not mode:
        raise SystemExit("Could not infer mode; pass --mode explicitly.")

    run_lut = _build_run_id_lookup(args.per_chunk_csv, mode)

    npz_files = sorted(glob.glob(os.path.join(args.npz_root, "**", "*.npz"), recursive=True))
    if not npz_files:
        raise SystemExit(f"No npz under {args.npz_root}")
    if args.limit:
        npz_files = npz_files[:args.limit]

    out_path = os.path.abspath(args.out) if args.out \
        else os.path.join(os.path.dirname(os.path.abspath(args.per_chunk_csv)),
                          "original_chunk_summary.csv")

    kept = dropped = unmatched = 0
    seen = set()
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for i, npz_path in enumerate(npz_files, 1):
            _m, source_file, chunk_index, run_id = _parse_path(npz_path)
            key = (source_file, chunk_index)
            report_run = run_lut.get(key)
            if report_run is None:
                unmatched += 1
                continue
            if report_run and run_id != report_run:
                dropped += 1
                continue
            if key in seen:
                continue
            seen.add(key)
            try:
                d = np.load(npz_path)
                amp = d["amp_original"]
                spec = d["spectrum_original"]
                freqs = np.asarray(d["freqs_hz"], dtype=np.float64)
            except Exception as exc:
                print(f"  SKIP {npz_path}: {exc}")
                continue
            a = _stats(amp)
            s = _stats(spec)
            w.writerow({
                "source_file": source_file, "chunk_index": chunk_index,
                "run_id": run_id, "n_samples": int(len(amp)),
                "amp_min": round(a["min"], 8), "amp_max": round(a["max"], 8),
                "amp_mean": round(a["mean"], 8), "amp_peak": round(a["peak"], 8),
                "amp_rms": round(a["rms"], 8),
                "n_freq_bins": int(len(freqs)),
                "freq_min_hz": round(float(freqs.min()), 4),
                "freq_max_hz": round(float(freqs.max()), 4),
                "spec_min": round(s["min"], 6), "spec_max": round(s["max"], 6),
                "spec_mean": round(s["mean"], 6), "spec_peak": round(s["peak"], 6),
                "spec_rms": round(s["rms"], 6),
            })
            kept += 1
            if i % 1000 == 0:
                print(f"  [{i}/{len(npz_files)}] kept={kept}")

    print(f"\nmode={mode}   kept(unique chunks)={kept}   "
          f"dropped_stale={dropped}   unmatched={unmatched}")
    print(f"  \u2713 {out_path}")


if __name__ == "__main__":
    main()
