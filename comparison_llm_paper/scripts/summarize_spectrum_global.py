#!/usr/bin/env python3
"""GLOBAL spectrum-magnitude summary tables (like amp_global_*.csv, for spectra).

Walks the amplitude npz tree, de-duplicates against the report run_id, splits
chunks into speech / env using had_speech, and aggregates the REAL magnitude
spectra (spectrum_original / spectrum_processed) into TRUE GLOBAL statistics
per mode (+ an ALL_MODES row):

    *_min   -> global minimum magnitude   (min over chunks)
    *_max   -> global maximum magnitude   (max over chunks)
    *_peak  -> global peak magnitude      (max over chunks; spectra are >= 0)
    *_mean  -> global mean magnitude      = sum(mean_i * bins_i) / sum(bins_i)
    *_rms   -> global rms magnitude       = sqrt(sum(rms_i^2 * bins_i)/sum(bins_i))

mean/rms are sample(bin)-count weighted -> exact, not an average of averages.

Outputs three CSVs (same shape as amp_global_*.csv):
    spec_global_overall.csv
    spec_global_speech.csv
    spec_global_env_only.csv

Usage:
    python3 scripts/summarize_spectrum_global.py \
        logs/amplitude_npz \
        logs/s3/20260701_171530/run_metrics_per_chunk.csv \
        --out-dir logs/s3/20260701_171530
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
from collections import defaultdict

import numpy as np

KNOWN_MODES = {
    "fixed", "fixed_strong", "rule_based", "rule_based_strong",
    "rule_based_ss", "llm_no_memory", "llm_with_memory",
    "llm_with_memory_no_ss", "llm_no_memory_no_ss",
}

OUT_COLS = ["orig_min", "orig_max", "orig_mean", "orig_peak", "orig_rms",
            "proc_min", "proc_max", "proc_mean", "proc_peak", "proc_rms"]
HEADLINE = ["orig_mean", "orig_peak", "orig_rms", "proc_mean", "proc_peak", "proc_rms"]


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


class _Acc:
    """Running global accumulator for one (subset, mode)."""
    __slots__ = ("n", "bins", "o_min", "o_max", "o_msum", "o_sqsum",
                 "p_min", "p_max", "p_msum", "p_sqsum")

    def __init__(self):
        self.n = 0
        self.bins = 0.0
        self.o_min = math.inf; self.o_max = -math.inf; self.o_msum = 0.0; self.o_sqsum = 0.0
        self.p_min = math.inf; self.p_max = -math.inf; self.p_msum = 0.0; self.p_sqsum = 0.0

    def add(self, so: np.ndarray, sp: np.ndarray):
        m = len(so)
        self.n += 1
        self.bins += m
        self.o_min = min(self.o_min, float(so.min())); self.o_max = max(self.o_max, float(so.max()))
        self.o_msum += float(so.mean()) * m; self.o_sqsum += float((so * so).mean()) * m
        self.p_min = min(self.p_min, float(sp.min())); self.p_max = max(self.p_max, float(sp.max()))
        self.p_msum += float(sp.mean()) * m; self.p_sqsum += float((sp * sp).mean()) * m

    def row(self, mode: str) -> dict:
        b = self.bins or 1.0
        return {
            "mode": mode, "n_chunks": self.n,
            "orig_min": round(self.o_min, 6), "orig_max": round(self.o_max, 6),
            "orig_mean": round(self.o_msum / b, 6), "orig_peak": round(self.o_max, 6),
            "orig_rms": round((self.o_sqsum / b) ** 0.5, 6),
            "proc_min": round(self.p_min, 6), "proc_max": round(self.p_max, 6),
            "proc_mean": round(self.p_msum / b, 6), "proc_peak": round(self.p_max, 6),
            "proc_rms": round((self.p_sqsum / b) ** 0.5, 6),
        }


def _print_table(title: str, rows: list[dict]) -> None:
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)
    cols = ["mode", "n_chunks"] + HEADLINE
    w = {c: max(len(c), 11) for c in cols}
    print("  ".join(f"{c:>{w[c]}}" for c in cols))
    print("-" * (sum(w.values()) + 2 * len(cols)))
    for r in rows:
        print("  ".join(f"{str(r.get(c, '')):>{w[c]}}" for c in cols))


def main() -> None:
    ap = argparse.ArgumentParser(description="Global spectrum-magnitude summary (overall/speech/env)")
    ap.add_argument("npz_root")
    ap.add_argument("per_chunk_csv")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir) if args.out_dir \
        else os.path.dirname(os.path.abspath(args.per_chunk_csv))
    os.makedirs(out_dir, exist_ok=True)

    report = _build_report_lookup(args.per_chunk_csv)
    npz_files = sorted(glob.glob(os.path.join(args.npz_root, "**", "*.npz"), recursive=True))
    if not npz_files:
        raise SystemExit(f"No npz under {args.npz_root}")
    if args.limit:
        npz_files = npz_files[:args.limit]

    # acc[subset][mode] = _Acc
    acc = {s: defaultdict(_Acc) for s in ("overall", "speech", "env")}
    kept = dropped = unmatched = 0

    for i, npz_path in enumerate(npz_files, 1):
        mode, source_file, chunk_index, run_id = _parse_path(npz_path)
        info = report.get((mode, source_file, chunk_index))
        if info is None:
            unmatched += 1; continue
        if info["run_id"] and run_id != info["run_id"]:
            dropped += 1; continue
        try:
            d = np.load(npz_path)
            so = np.asarray(d["spectrum_original"], dtype=np.float64)
            sp = np.asarray(d["spectrum_processed"], dtype=np.float64) \
                if "spectrum_processed" in d.files else np.zeros_like(so)
        except Exception as exc:
            print(f"  SKIP {npz_path}: {exc}"); continue
        kept += 1
        subset = "speech" if info["had_speech"] else "env"
        acc["overall"][mode].add(so, sp)
        acc[subset][mode].add(so, sp)
        if i % 1000 == 0:
            print(f"  [{i}/{len(npz_files)}] kept={kept}")

    names = {"overall": "spec_global_overall.csv",
             "speech": "spec_global_speech.csv",
             "env": "spec_global_env_only.csv"}
    titles = {"overall": "1) OVERALL — spectrum magnitude, global",
              "speech": "2) SPEECH — spectrum magnitude, global",
              "env": "3) ENVIRONMENT-ONLY — spectrum magnitude, global"}

    for subset in ("overall", "speech", "env"):
        modes = sorted(acc[subset])
        rows = [acc[subset][m].row(m) for m in modes]
        # ALL_MODES aggregate
        allacc = _Acc()
        for m in modes:
            a = acc[subset][m]
            allacc.n += a.n; allacc.bins += a.bins
            allacc.o_min = min(allacc.o_min, a.o_min); allacc.o_max = max(allacc.o_max, a.o_max)
            allacc.o_msum += a.o_msum; allacc.o_sqsum += a.o_sqsum
            allacc.p_min = min(allacc.p_min, a.p_min); allacc.p_max = max(allacc.p_max, a.p_max)
            allacc.p_msum += a.p_msum; allacc.p_sqsum += a.p_sqsum
        rows.append(allacc.row("ALL_MODES"))

        out_path = os.path.join(out_dir, names[subset])
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["mode", "n_chunks"] + OUT_COLS)
            w.writeheader(); w.writerows(rows)
        _print_table(titles[subset], rows)
        print(f"\n  \u2713 {out_path}")

    print(f"\nkept={kept}  dropped_stale={dropped}  unmatched={unmatched}")


if __name__ == "__main__":
    main()
