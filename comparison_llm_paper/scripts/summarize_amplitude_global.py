#!/usr/bin/env python3
"""Build GLOBAL amplitude summary tables from real per-chunk npz stats.

Input:
    npz_amp_all.csv            (from npz_amplitude_stats.py — REAL per-chunk
                                min/max/mean/peak/rms computed from the sample
                                arrays, no approximation)
    run_metrics_per_chunk.csv  (used ONLY to look up had_speech per chunk,
                                keyed by mode + source_file + chunk_index)

Produces three tables, each grouped by `mode` (+ an ALL_MODES row):
    1. amp_global_overall.csv    — every chunk
    2. amp_global_speech.csv     — chunks WITH speech      (had_speech=True)
    3. amp_global_env_only.csv   — chunks with NO speech   (had_speech=False)

Aggregation is TRUE GLOBAL (never an average of per-chunk extremes):
    *_min                 -> global minimum  = min over chunks
    *_max/_peak/_max_abs  -> global maximum  = max over chunks
    *_mean                -> global mean      = sum(mean_i * n_i) / sum(n_i)   (exact)
    *_rms                 -> global rms       = sqrt(sum(rms_i^2 * n_i)/sum(n_i)) (exact)

The mean/rms formulas are EXACT (not approximations) because they are
sample-count-weighted; with equal 64000-sample chunks they equal the value
you would get from concatenating every sample.

Usage:
    python3 scripts/summarize_amplitude_global.py \
        logs/s3/20260701_171530/npz_amp_all.csv \
        logs/s3/20260701_171530/run_metrics_per_chunk.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

# Columns to aggregate + how.
MIN_COLS = ["orig_min", "proc_min"]
MAX_COLS = ["orig_max", "proc_max", "orig_peak", "proc_peak", "diff_max_abs"]
MEAN_COLS = ["orig_mean", "proc_mean"]
RMS_COLS = ["orig_rms", "proc_rms", "diff_rms"]

OUT_COLS = (["orig_min", "orig_max", "orig_mean", "orig_peak", "orig_rms"]
            + ["proc_min", "proc_max", "proc_mean", "proc_peak", "proc_rms"]
            + ["diff_max_abs", "diff_rms"])

HEADLINE = ["orig_min", "orig_max", "orig_mean", "orig_peak",
            "proc_min", "proc_max", "proc_mean", "proc_peak"]


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _build_report_lookup(per_chunk_csv: str) -> dict:
    """Map (mode, source_file, chunk_index) -> {had_speech, run_id}.

    run_id is the AUTHORITATIVE run recorded in the report; it is used to
    discard stale duplicate npz files left over from earlier runs so the
    counts match the report exactly.
    """
    lut = {}
    with open(per_chunk_csv, newline="") as f:
        for r in csv.DictReader(f):
            key = (r.get("mode", ""), r.get("source_file", ""),
                   str(r.get("chunk_index", "")))
            lut[key] = {
                "had_speech": str(r.get("had_speech", "")).strip().lower() in ("true", "1"),
                "run_id": (r.get("run_id", "") or "").strip(),
            }
    return lut


def _npz_run_id(npz_file: str) -> str:
    """run_id is the prefix of '<run_id>_<chunk>_amplitude.npz' (UUID, no '_')."""
    return npz_file.split("_", 1)[0] if npz_file else ""


def _aggregate(rows: list[dict]) -> dict:
    """Aggregate a list of npz-stat rows into one GLOBAL stat row."""
    out = {"n_chunks": len(rows)}
    ntot = 0.0
    wsum_mean = defaultdict(float)   # sum(mean_i * n_i)
    wsum_sq = defaultdict(float)     # sum(rms_i^2 * n_i)
    mins = defaultdict(lambda: None)
    maxs = defaultdict(lambda: None)

    for r in rows:
        n = _f(r.get("n_samples")) or 0.0
        ntot += n
        for c in MIN_COLS:
            v = _f(r.get(c))
            if v is not None:
                mins[c] = v if mins[c] is None else min(mins[c], v)
        for c in MAX_COLS:
            v = _f(r.get(c))
            if v is not None:
                maxs[c] = v if maxs[c] is None else max(maxs[c], v)
        for c in MEAN_COLS:
            v = _f(r.get(c))
            if v is not None:
                wsum_mean[c] += v * n
        for c in RMS_COLS:
            v = _f(r.get(c))
            if v is not None:
                wsum_sq[c] += v * v * n

    for c in MIN_COLS:
        out[c] = round(mins[c], 8) if mins[c] is not None else ""
    for c in MAX_COLS:
        out[c] = round(maxs[c], 8) if maxs[c] is not None else ""
    for c in MEAN_COLS:
        out[c] = round(wsum_mean[c] / ntot, 8) if ntot else ""
    for c in RMS_COLS:
        out[c] = round((wsum_sq[c] / ntot) ** 0.5, 8) if ntot else ""
    return out


def _summarise(rows: list[dict]) -> list[dict]:
    by_mode = defaultdict(list)
    for r in rows:
        by_mode[r.get("mode", "?")].append(r)
    result = []
    for mode in sorted(by_mode):
        agg = _aggregate(by_mode[mode])
        agg = {"mode": mode, **agg}
        result.append(agg)
    allm = {"mode": "ALL_MODES", **_aggregate(rows)}
    result.append(allm)
    return result


def _print_table(title: str, summary: list[dict]) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)
    cols = ["mode", "n_chunks"] + HEADLINE
    widths = {c: max(len(c), 11) for c in cols}
    print("  ".join(f"{c:>{widths[c]}}" for c in cols))
    print("-" * (sum(widths.values()) + 2 * len(cols)))
    for row in summary:
        print("  ".join(f"{str(row.get(c, '')):>{widths[c]}}" for c in cols))


def main() -> None:
    ap = argparse.ArgumentParser(description="Global amplitude summary tables from npz stats")
    ap.add_argument("npz_stats_csv", help="npz_amp_all.csv (per-chunk real stats)")
    ap.add_argument("per_chunk_csv", help="run_metrics_per_chunk.csv (for had_speech)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    npz_csv = os.path.abspath(args.npz_stats_csv)
    pc_csv = os.path.abspath(args.per_chunk_csv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(npz_csv)
    os.makedirs(out_dir, exist_ok=True)

    report_lut = _build_report_lookup(pc_csv)

    with open(npz_csv, newline="") as f:
        all_rows = list(csv.DictReader(f))

    # Keep only npz whose run_id matches the report (drops stale duplicates),
    # and attach had_speech. Anything not in the report is set aside.
    rows, speech, env, unknown, stale_dupes = [], [], [], [], 0
    for r in all_rows:
        key = (r.get("mode", ""), r.get("source_file", ""), str(r.get("chunk_index", "")))
        info = report_lut.get(key)
        if info is None:
            unknown.append(r)
            continue
        if info["run_id"] and _npz_run_id(r.get("npz_file", "")) != info["run_id"]:
            stale_dupes += 1          # duplicate npz from an earlier run -> drop
            continue
        rows.append(r)
        (speech if info["had_speech"] else env).append(r)

    subsets = [
        ("overall", "1) OVERALL — all chunks (real global from npz, deduped by report run_id)", rows),
        ("speech", "2) SPEECH — had_speech=True (real global from npz)", speech),
        ("env_only", "3) ENVIRONMENT-ONLY — had_speech=False (real global from npz)", env),
    ]
    fieldnames = ["mode", "n_chunks"] + OUT_COLS
    for name, title, subset in subsets:
        if not subset:
            print(f"\n(skip {name}: no rows)")
            continue
        summary = _summarise(subset)
        out_path = os.path.join(out_dir, f"amp_global_{name}.csv")
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(summary)
        _print_table(title, summary)
        print(f"\n  \u2713 {out_path}")

    print("\n" + "-" * 96)
    print(f"npz rows read: {len(all_rows)}   ->   kept (matched report): {len(rows)}")
    print(f"  speech: {len(speech)}   |   env-only: {len(env)}")
    print(f"  dropped stale duplicate npz (run_id != report): {stale_dupes}")
    if unknown:
        print(f"  unmatched npz (no report row): {len(unknown)}  (excluded)")


if __name__ == "__main__":
    main()
