#!/usr/bin/env python3
"""Per-band spectral reduction (dB / %) for EVERY mode, split speech vs env.

Walks the amplitude npz tree, and for each chunk:
  * parses (mode, source_file, chunk_index, run_id) from the path/filename,
  * keeps only npz whose run_id matches the report (drops stale duplicates),
  * classifies the chunk as SPEECH or ENV using had_speech from the report,
  * sums the original/processed magnitude spectrum energy per frequency band.

Then, per (mode x subset x band), reports:
    reduction_dB  = 20*log10( sum_energy_orig / sum_energy_proc )
    reduction_pct = (1 - energy_proc / energy_orig) * 100

Subsets: overall, speech (had_speech=True), env (had_speech=False).
Bands (Hz): 0-500, 500-3000 (core speech), 3000-8000, and 0-8000 (ALL).

Outputs one tidy CSV: band_reduction_by_mode.csv
    columns: subset, mode, band_hz, n_chunks, energy_original,
             energy_processed, reduction_dB, reduction_pct

Usage:
    python3 scripts/band_reduction_by_mode.py \
        logs/amplitude_npz \
        logs/s3/20260701_171530/run_metrics_per_chunk.csv \
        --out logs/s3/20260701_171530/band_reduction_by_mode.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
from collections import defaultdict

import numpy as np

DEFAULT_EDGES = [0, 500, 3000, 8000]
KNOWN_MODES = {
    "fixed", "fixed_strong", "rule_based", "rule_based_strong",
    "rule_based_ss", "llm_no_memory", "llm_with_memory",
    "llm_with_memory_no_ss", "llm_no_memory_no_ss",
}


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


def _build_report_lookup(per_chunk_csv: str) -> dict:
    lut = {}
    with open(per_chunk_csv, newline="") as f:
        for r in csv.DictReader(f):
            key = (r.get("mode", ""), r.get("source_file", ""), str(r.get("chunk_index", "")))
            lut[key] = {
                "had_speech": str(r.get("had_speech", "")).strip().lower() in ("true", "1"),
                "run_id": (r.get("run_id", "") or "").strip(),
            }
    return lut


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-band spectral reduction by mode (speech/env)")
    ap.add_argument("npz_root", help="Root of the amplitude npz tree (e.g. logs/amplitude_npz)")
    ap.add_argument("per_chunk_csv", help="run_metrics_per_chunk.csv (had_speech + run_id)")
    ap.add_argument("--bands", default=None, help="Hz edges, comma separated (default 0,500,3000,8000)")
    ap.add_argument("--out", default=None, help="Output CSV path")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    edges = sorted(set(int(x) for x in args.bands.split(","))) if args.bands else DEFAULT_EDGES
    nb = len(edges) - 1
    band_labels = [f"{edges[b]}-{edges[b+1]}" for b in range(nb)]

    report = _build_report_lookup(args.per_chunk_csv)

    npz_files = sorted(glob.glob(os.path.join(args.npz_root, "**", "*.npz"), recursive=True))
    if not npz_files:
        raise SystemExit(f"No npz under {args.npz_root}")
    if args.limit:
        npz_files = npz_files[:args.limit]

    # energy[(subset, mode, band)] = [sum_orig, sum_proc]
    energy = defaultdict(lambda: [0.0, 0.0])
    chunks = defaultdict(set)   # (subset, mode) -> set of (source, chunk)
    kept = dropped = unmatched = 0

    for i, npz_path in enumerate(npz_files, 1):
        mode, source_file, chunk_index, run_id = _parse_path(npz_path)
        info = report.get((mode, source_file, chunk_index))
        if info is None:
            unmatched += 1
            continue
        if info["run_id"] and run_id != info["run_id"]:
            dropped += 1
            continue
        try:
            d = np.load(npz_path)
            freqs = np.asarray(d["freqs_hz"], dtype=np.float64)
            so = np.asarray(d["spectrum_original"], dtype=np.float64)
            sp = np.asarray(d["spectrum_processed"], dtype=np.float64) \
                if "spectrum_processed" in d.files else np.zeros_like(so)
        except Exception as exc:
            print(f"  SKIP {npz_path}: {exc}")
            continue
        kept += 1
        subset = "speech" if info["had_speech"] else "env"
        for b in range(nb):
            mask = (freqs >= edges[b]) & (freqs < edges[b + 1])
            eo = float(so[mask].sum())
            ep = float(sp[mask].sum())
            for sub in (subset, "overall"):
                energy[(sub, mode, band_labels[b])][0] += eo
                energy[(sub, mode, band_labels[b])][1] += ep
        for sub in (subset, "overall"):
            chunks[(sub, mode)].add((source_file, chunk_index))
        if i % 1000 == 0:
            print(f"  [{i}/{len(npz_files)}] processed  (kept={kept})")

    # Assemble rows (also an ALL_MODES aggregate per subset+band).
    modes = sorted({m for (_, m, _) in energy})
    subsets = ["overall", "speech", "env"]
    rows = []

    def _emit(subset, mode, band, eo, ep, nchunks):
        red_db = 20.0 * math.log10(eo / ep) if (eo > 0 and ep > 0) else ""
        red_pct = (1.0 - ep / eo) * 100.0 if eo > 0 else ""
        rows.append({
            "subset": subset, "mode": mode, "band_hz": band,
            "n_chunks": nchunks,
            "energy_original": round(eo, 4), "energy_processed": round(ep, 4),
            "reduction_dB": round(red_db, 4) if red_db != "" else "",
            "reduction_pct": round(red_pct, 4) if red_pct != "" else "",
        })

    for subset in subsets:
        for mode in modes:
            nch = len(chunks.get((subset, mode), ()))
            for band in band_labels:
                eo, ep = energy.get((subset, mode, band), [0.0, 0.0])
                _emit(subset, mode, band, eo, ep, nch)
        # ALL_MODES row per band
        for band in band_labels:
            eo = sum(energy.get((subset, m, band), [0.0, 0.0])[0] for m in modes)
            ep = sum(energy.get((subset, m, band), [0.0, 0.0])[1] for m in modes)
            nch = len(set().union(*[chunks.get((subset, m), set()) for m in modes])) if modes else 0
            _emit(subset, "ALL_MODES", band, eo, ep, nch)

    out_path = os.path.abspath(args.out) if args.out \
        else os.path.join(os.path.dirname(os.path.abspath(args.per_chunk_csv)),
                          "band_reduction_by_mode.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subset", "mode", "band_hz", "n_chunks",
                                          "energy_original", "energy_processed",
                                          "reduction_dB", "reduction_pct"])
        w.writeheader()
        w.writerows(rows)

    # Console: focus on the core speech band per mode/subset.
    print("\n" + "=" * 80)
    print("CORE SPEECH BAND 500-3000 Hz — reduction by mode")
    print("=" * 80)
    print(f"{'subset':>8}  {'mode':>22}  {'reduce_dB':>10}  {'reduce_%':>9}")
    print("-" * 80)
    for subset in subsets:
        for r in rows:
            if r["subset"] == subset and r["band_hz"] == "500-3000":
                print(f"{r['subset']:>8}  {r['mode']:>22}  "
                      f"{str(r['reduction_dB']):>10}  {str(r['reduction_pct']):>9}")
        print("-" * 80)

    print(f"\nkept={kept}  dropped_stale={dropped}  unmatched={unmatched}")
    print(f"  \u2713 {out_path}")


if __name__ == "__main__":
    main()
