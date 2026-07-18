#!/usr/bin/env python3
"""Summarise amplitude columns from run_metrics_per_chunk.csv into 3 tables.

Reads the per-chunk metrics CSV (produced by export_run_metrics_csv.py) and
builds three summary tables, each grouped by pipeline `mode`:

  1. OVERALL          — every chunk (all audio)
  2. SPEECH           — chunks WITH speech        (had_speech == True)
  3. ENVIRONMENT-ONLY — chunks with NO speech     (had_speech == False)

For every amplitude column it reports a TRUE GLOBAL statistic across the
chunks in that subset:  *_min -> global min, *_max/_peak -> global max,
*_mean -> global mean, *_rms -> global rms. A `n_chunks` column shows how
many chunks fell into each group, and an ALL_MODES row aggregates across
every mode.

Outputs (next to the input CSV, or under --out-dir):
    amp_summary_overall.csv
    amp_summary_speech.csv
    amp_summary_env_only.csv

Pure standard library (csv) — no pandas/numpy required.

Usage:
    python3 scripts/summarize_amplitude.py logs/s3/20260701_171530/run_metrics_per_chunk.csv
    python3 scripts/summarize_amplitude.py <csv> --out-dir plots/amp_summary
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

# Amplitude columns to summarise (mean across chunks).
AMP_COLS = [
    "amp_orig_min", "amp_orig_max", "amp_orig_mean", "amp_orig_std",
    "amp_orig_rms", "amp_orig_peak", "amp_orig_abs_mean",
    "amp_proc_min", "amp_proc_max", "amp_proc_mean", "amp_proc_std",
    "amp_proc_rms", "amp_proc_peak", "amp_proc_abs_mean",
    "amp_diff_rms", "amp_diff_abs_mean", "amp_diff_max_abs",
]

# Compact set shown in the console (full set still written to CSV).
# Focused on min / max / mean of original vs processed amplitude.
HEADLINE = [
    "amp_orig_min", "amp_orig_max", "amp_orig_mean",
    "amp_proc_min", "amp_proc_max", "amp_proc_mean",
]


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _aggregate(col: str, vals: list[float]) -> float | str:
    """Aggregate per-chunk values into a TRUE GLOBAL statistic.

    The aggregation rule depends on what the column represents:
      *_min                -> global minimum  (min of per-chunk mins)
      *_max/_peak/_max_abs -> global maximum  (max of per-chunk maxes)
      *_mean/_abs_mean     -> global mean     (exact: every chunk = 64000 samples)
      *_rms                -> global RMS       = sqrt(mean(rms_i^2))  (exact, equal sizes)
      *_std                -> mean of stds     (approximation; true std needs raw samples)
    """
    clean = [v for v in vals if v is not None]
    if not clean:
        return ""
    if col.endswith("_min"):
        return round(min(clean), 6)
    if col.endswith("_max") or col.endswith("_peak") or col.endswith("_max_abs"):
        return round(max(clean), 6)
    if col.endswith("_rms"):
        return round((sum(v * v for v in clean) / len(clean)) ** 0.5, 6)
    # *_mean, *_abs_mean, *_std  -> arithmetic mean
    return round(sum(clean) / len(clean), 6)


def _summarise(rows: list[dict], amp_cols: list[str]) -> list[dict]:
    """Return a list of summary rows: one per mode + an ALL_MODES row."""
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_mode[r.get("mode", "?")].append(r)

    def _row(label: str, subset: list[dict]) -> dict:
        out = {"mode": label, "n_chunks": len(subset)}
        for c in amp_cols:
            out[c] = _aggregate(c, [_to_float(r.get(c)) for r in subset])
        return out

    summary = [_row(mode, by_mode[mode]) for mode in sorted(by_mode)]
    summary.append(_row("ALL_MODES", rows))
    return summary


def _print_table(title: str, summary: list[dict], amp_cols: list[str]) -> None:
    print("\n" + "=" * 84)
    print(title)
    print("=" * 84)
    cols = ["mode", "n_chunks"] + [c for c in HEADLINE if c in amp_cols]
    widths = {c: max(len(c), 12) for c in cols}
    header = "  ".join(f"{c:>{widths[c]}}" for c in cols)
    print(header)
    print("-" * len(header))
    for row in summary:
        line = "  ".join(f"{str(row.get(c, '')):>{widths[c]}}" for c in cols)
        print(line)


def _is_true(v) -> bool:
    return str(v).strip().lower() in ("true", "1")


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarise amplitude columns into 3 tables")
    ap.add_argument("csv", help="Path to run_metrics_per_chunk.csv")
    ap.add_argument("--out-dir", default=None,
                    help="Where to write the summary CSVs (default: beside input)")
    args = ap.parse_args()

    csv_path = os.path.abspath(args.csv)
    if not os.path.isfile(csv_path):
        raise SystemExit(f"Not a file: {csv_path}")
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(csv_path)
    os.makedirs(out_dir, exist_ok=True)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    amp_cols = [c for c in AMP_COLS if c in (reader.fieldnames or [])]
    if not amp_cols:
        raise SystemExit("No amplitude columns found in the CSV.")

    speech_rows = [r for r in rows if _is_true(r.get("had_speech"))]
    env_rows = [r for r in rows if not _is_true(r.get("had_speech"))]

    subsets = [
        ("overall", "1) OVERALL — all audio chunks", rows),
        ("speech", "2) SPEECH — chunks WITH human speech (had_speech=True)", speech_rows),
        ("env_only", "3) ENVIRONMENT-ONLY — chunks with NO speech (had_speech=False)", env_rows),
    ]

    fieldnames = ["mode", "n_chunks"] + amp_cols
    for name, title, subset in subsets:
        if not subset:
            print(f"\n(skip {name}: no rows)")
            continue
        summary = _summarise(subset, amp_cols)
        out_path = os.path.join(out_dir, f"amp_summary_{name}.csv")
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(summary)
        _print_table(title, summary, amp_cols)
        print(f"\n  \u2713 {out_path}")

    print("\n" + "-" * 84)
    print(f"Total chunks: {len(rows)}   |   speech: {len(speech_rows)}"
          f"   |   env-only: {len(env_rows)}")


if __name__ == "__main__":
    main()
