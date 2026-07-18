#!/usr/bin/env python3
"""How much did blur reduce each FREQUENCY BAND? (dB and %) — from real npz.

For one or many amplitude .npz files this compares the original vs processed
magnitude spectra and reports, per frequency band, the reduction in energy:

    reduction_dB  = 20*log10( sum|orig| / sum|proc| )   over the band
    reduction_pct = (1 - energy_proc / energy_orig) * 100

Default bands are tuned to the project's speech-blur design:
    0-500      sub / low          (usually untouched)
    500-3000   CORE SPEECH BAND   (MidBandAttenuation target)
    3000-8000  high               (environment / consonants)
plus the whole 0-8000 range.

Energy per band = sum of magnitude over the FFT bins in that band (summed
across all files given), so the ratio is a true aggregate — not an average
of per-file ratios.

Usage:
    # one npz
    python3 scripts/spectrum_band_reduction.py "<path>_amplitude.npz"

    # a whole tree (e.g. one mode) — aggregates energy across every npz
    python3 scripts/spectrum_band_reduction.py logs/amplitude_npz/fixed --recursive \
        --out logs/s3/20260701_171530/band_reduction_fixed.csv

    # custom bands (Hz edges, comma separated)
    python3 scripts/spectrum_band_reduction.py <path> --bands 0,300,500,3000,5000,8000
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os

import numpy as np

DEFAULT_EDGES = [0, 500, 3000, 8000]


def _find_npz(path: str, recursive: bool) -> list[str]:
    if os.path.isfile(path):
        return [path] if path.endswith(".npz") else []
    if os.path.isdir(path):
        pat = os.path.join(path, "**", "*.npz") if recursive else os.path.join(path, "*.npz")
        return sorted(glob.glob(pat, recursive=recursive))
    return []


def _accumulate(npz_path: str, edges: list[int], acc_o, acc_p):
    """Add this file's per-band summed magnitude to acc_o / acc_p."""
    try:
        d = np.load(npz_path)
    except Exception as exc:
        print(f"  SKIP {npz_path}: {exc}")
        return 0
    if "spectrum_original" not in d.files:
        return 0
    freqs = np.asarray(d["freqs_hz"], dtype=np.float64)
    so = np.asarray(d["spectrum_original"], dtype=np.float64)
    sp = np.asarray(d["spectrum_processed"], dtype=np.float64) \
        if "spectrum_processed" in d.files else np.zeros_like(so)
    for b in range(len(edges) - 1):
        lo, hi = edges[b], edges[b + 1]
        mask = (freqs >= lo) & (freqs < hi)
        acc_o[b] += float(so[mask].sum())
        acc_p[b] += float(sp[mask].sum())
    return 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-band spectral reduction (dB and %) from npz")
    ap.add_argument("path", help="A .npz file OR a directory of them")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--bands", default=None,
                    help="Comma-separated Hz band edges (default: 0,500,3000,8000)")
    ap.add_argument("--out", default=None, help="Optional CSV output path")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    edges = ([int(x) for x in args.bands.split(",")] if args.bands else DEFAULT_EDGES)
    edges = sorted(set(edges))
    if len(edges) < 2:
        raise SystemExit("Need at least two band edges.")

    npz_files = _find_npz(args.path, args.recursive)
    if not npz_files:
        raise SystemExit(f"No .npz found for {args.path}"
                         + ("" if args.recursive else "  (try --recursive)"))
    if args.limit:
        npz_files = npz_files[:args.limit]

    nb = len(edges) - 1
    acc_o = [0.0] * nb
    acc_p = [0.0] * nb
    n_ok = 0
    for i, npz_path in enumerate(npz_files, 1):
        n_ok += _accumulate(npz_path, edges, acc_o, acc_p)
        if i % 500 == 0:
            print(f"  [{i}/{len(npz_files)}] processed")

    # Build rows: one per band + a whole-range row.
    rows = []
    total_o = sum(acc_o)
    total_p = sum(acc_p)
    for b in range(nb):
        o, p = acc_o[b], acc_p[b]
        red_db = 20.0 * math.log10(o / p) if (o > 0 and p > 0) else float("nan")
        red_pct = (1.0 - p / o) * 100.0 if o > 0 else float("nan")
        rows.append({
            "band_hz": f"{edges[b]}-{edges[b+1]}",
            "energy_original": round(o, 4),
            "energy_processed": round(p, 4),
            "reduction_dB": round(red_db, 4),
            "reduction_pct": round(red_pct, 4),
        })
    red_db_all = 20.0 * math.log10(total_o / total_p) if (total_o > 0 and total_p > 0) else float("nan")
    red_pct_all = (1.0 - total_p / total_o) * 100.0 if total_o > 0 else float("nan")
    rows.append({
        "band_hz": f"{edges[0]}-{edges[-1]} (ALL)",
        "energy_original": round(total_o, 4),
        "energy_processed": round(total_p, 4),
        "reduction_dB": round(red_db_all, 4),
        "reduction_pct": round(red_pct_all, 4),
    })

    # Console
    print("\n" + "=" * 74)
    print(f"Per-band spectral reduction   (files aggregated: {n_ok})")
    print("=" * 74)
    print(f"{'band_hz':>16}  {'energy_orig':>14}  {'energy_proc':>14}  "
          f"{'reduce_dB':>10}  {'reduce_%':>9}")
    print("-" * 74)
    for r in rows:
        print(f"{r['band_hz']:>16}  {r['energy_original']:>14}  "
              f"{r['energy_processed']:>14}  {r['reduction_dB']:>10}  {r['reduction_pct']:>9}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n  \u2713 {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
