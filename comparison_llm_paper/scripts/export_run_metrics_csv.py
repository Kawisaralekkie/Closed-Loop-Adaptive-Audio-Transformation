#!/usr/bin/env python3
"""Export privacy/preserve scores + amplitude stats to CSV for a run folder.

Walks a run directory such as ``logs/s3/20260701_171530`` which contains one
sub-folder per pipeline mode (fixed, fixed_strong, rule_based,
rule_based_strong, llm_no_memory_no_ss, ...). Each mode folder holds one
``*_report.json`` per source audio file, and each report contains a list of
chunks with privacy metrics, utility/preserve metrics, and amplitude stats.

Outputs (written next to the run folder, or to --out-dir):

  run_metrics_per_chunk.csv  — one row per (mode, source_file, chunk)
      all privacy scores, preserve score + sub-scores, psychoacoustic
      features, pass/fail decision, and amplitude stats (orig/proc/diff).

  run_metrics_by_mode.csv    — one row per mode: mean of the key metrics
      across every chunk (quick comparison table).

Usage:
    python3 scripts/export_run_metrics_csv.py logs/s3/20260701_171530
    python3 scripts/export_run_metrics_csv.py logs/s3/20260701_171530 --out-dir plots/run_20260701
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict


# Column order for the per-chunk CSV.
FIELDNAMES = [
    "mode",
    "source_file",
    "run_id",
    "chunk_id",
    "chunk_index",
    "had_speech",
    "speech_ratio",
    "vad_confidence",
    "recipe_applied",
    "trials",
    "routing_decision",
    "used_source_separation",
    # privacy
    "wer",
    "cer",
    "speaker_privacy",
    "content_privacy",
    "privacy_score",
    # utility / preserve
    "mAP",
    "f1",
    "accuracy",
    "preserve_score",
    "s_loud",
    "s_hf",
    "s_sc",
    "s_con",
    "s_psy",
    # psychoacoustic
    "short_term_loudness",
    "sharpness_proxy",
    "roughness_proxy",
    "fluctuation_proxy",
    # decision
    "privacy_pass",
    "preserve_pass",
    "overall_pass",
    "privacy_score_min",
    "preserve_score_min",
    # amplitude stats — original
    "amp_orig_min",
    "amp_orig_max",
    "amp_orig_mean",
    "amp_orig_std",
    "amp_orig_rms",
    "amp_orig_peak",
    "amp_orig_abs_mean",
    # amplitude stats — processed
    "amp_proc_min",
    "amp_proc_max",
    "amp_proc_mean",
    "amp_proc_std",
    "amp_proc_rms",
    "amp_proc_peak",
    "amp_proc_abs_mean",
    # amplitude stats — difference (orig - proc)
    "amp_diff_rms",
    "amp_diff_abs_mean",
    "amp_diff_max_abs",
]

# Metrics averaged in the by-mode summary.
SUMMARY_METRICS = [
    "privacy_score",
    "preserve_score",
    "wer",
    "cer",
    "content_privacy",
    "speaker_privacy",
    "mAP",
    "f1",
    "accuracy",
    "amp_diff_rms",
    "amp_diff_max_abs",
]


def _get(d: dict, *keys, default=None):
    """Safely walk nested dicts."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _row_for_chunk(mode: str, source_file: str, run_id: str,
                   idx: int, chunk: dict) -> dict:
    m = chunk.get("metrics") or {}
    priv = m.get("privacy") or {}
    util = m.get("utility") or {}
    subs = util.get("sub_scores") or {}
    psy = m.get("psychoacoustic") or {}
    dec = m.get("decision") or {}
    amp = chunk.get("amplitude_stats") or {}
    a_o = amp.get("original") or {}
    a_p = amp.get("processed") or {}
    a_d = amp.get("difference") or {}

    return {
        "mode": mode,
        "source_file": source_file,
        "run_id": run_id,
        "chunk_id": chunk.get("chunk_id"),
        "chunk_index": idx,
        "had_speech": chunk.get("had_speech"),
        "speech_ratio": chunk.get("speech_ratio"),
        "vad_confidence": chunk.get("vad_confidence"),
        "recipe_applied": chunk.get("recipe_applied"),
        "trials": chunk.get("trials"),
        "routing_decision": chunk.get("routing_decision"),
        "used_source_separation": chunk.get("used_source_separation"),
        "wer": priv.get("wer"),
        "cer": priv.get("cer"),
        "speaker_privacy": priv.get("speaker_privacy"),
        "content_privacy": priv.get("content_privacy"),
        "privacy_score": priv.get("privacy_score"),
        "mAP": util.get("mAP"),
        "f1": util.get("f1"),
        "accuracy": util.get("accuracy"),
        "preserve_score": util.get("preserve_score"),
        "s_loud": subs.get("s_loud"),
        "s_hf": subs.get("s_hf"),
        "s_sc": subs.get("s_sc"),
        "s_con": subs.get("s_con"),
        "s_psy": subs.get("s_psy"),
        "short_term_loudness": psy.get("short_term_loudness"),
        "sharpness_proxy": psy.get("sharpness_proxy"),
        "roughness_proxy": psy.get("roughness_proxy"),
        "fluctuation_proxy": psy.get("fluctuation_proxy"),
        "privacy_pass": dec.get("privacy_pass"),
        "preserve_pass": dec.get("preserve_pass"),
        "overall_pass": dec.get("overall_pass"),
        "privacy_score_min": dec.get("privacy_score_min"),
        "preserve_score_min": dec.get("preserve_score_min"),
        "amp_orig_min": a_o.get("min"),
        "amp_orig_max": a_o.get("max"),
        "amp_orig_mean": a_o.get("mean"),
        "amp_orig_std": a_o.get("std"),
        "amp_orig_rms": a_o.get("rms"),
        "amp_orig_peak": a_o.get("peak"),
        "amp_orig_abs_mean": a_o.get("abs_mean"),
        "amp_proc_min": a_p.get("min"),
        "amp_proc_max": a_p.get("max"),
        "amp_proc_mean": a_p.get("mean"),
        "amp_proc_std": a_p.get("std"),
        "amp_proc_rms": a_p.get("rms"),
        "amp_proc_peak": a_p.get("peak"),
        "amp_proc_abs_mean": a_p.get("abs_mean"),
        "amp_diff_rms": a_d.get("rms"),
        "amp_diff_abs_mean": a_d.get("abs_mean"),
        "amp_diff_max_abs": a_d.get("max_abs"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Export run metrics to CSV")
    ap.add_argument("run_dir", help="Run folder, e.g. logs/s3/20260701_171530")
    ap.add_argument("--out-dir", default=None,
                    help="Where to write CSVs (default: the run folder itself)")
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"Not a directory: {run_dir}")
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else run_dir
    os.makedirs(out_dir, exist_ok=True)

    # Discover mode sub-folders (any dir that holds *_report.json).
    modes = sorted(
        d for d in os.listdir(run_dir)
        if os.path.isdir(os.path.join(run_dir, d))
        and glob.glob(os.path.join(run_dir, d, "*_report.json"))
    )
    if not modes:
        raise SystemExit(f"No mode folders with *_report.json under {run_dir}")

    rows: list[dict] = []
    per_mode_files = defaultdict(int)
    per_mode_chunks = defaultdict(int)

    for mode in modes:
        reports = sorted(glob.glob(os.path.join(run_dir, mode, "*_report.json")))
        for rp in reports:
            per_mode_files[mode] += 1
            try:
                with open(rp) as f:
                    report = json.load(f)
            except Exception as exc:
                print(f"  SKIP {rp}: {exc}")
                continue
            source_file = os.path.basename(report.get("source_id", rp))
            run_id = report.get("run_id", "")
            for idx, chunk in enumerate(report.get("chunks", [])):
                rows.append(_row_for_chunk(mode, source_file, run_id, idx, chunk))
                per_mode_chunks[mode] += 1

    # ── Write per-chunk CSV ──
    per_chunk_path = os.path.join(out_dir, "run_metrics_per_chunk.csv")
    with open(per_chunk_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)

    # ── Build & write by-mode summary CSV ──
    summary_rows = []
    for mode in modes:
        mode_rows = [r for r in rows if r["mode"] == mode]
        n = len(mode_rows)
        summary = {"mode": mode, "n_files": per_mode_files[mode], "n_chunks": n}
        for metric in SUMMARY_METRICS:
            vals = [r[metric] for r in mode_rows if isinstance(r[metric], (int, float))]
            summary[f"mean_{metric}"] = round(sum(vals) / len(vals), 6) if vals else ""
        # pass rates
        for flag in ("privacy_pass", "preserve_pass", "overall_pass"):
            vals = [1 for r in mode_rows if r[flag] is True]
            summary[f"rate_{flag}"] = round(len(vals) / n, 4) if n else ""
        summary_rows.append(summary)

    summary_fields = (
        ["mode", "n_files", "n_chunks"]
        + [f"mean_{m}" for m in SUMMARY_METRICS]
        + ["rate_privacy_pass", "rate_preserve_pass", "rate_overall_pass"]
    )
    by_mode_path = os.path.join(out_dir, "run_metrics_by_mode.csv")
    with open(by_mode_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(summary_rows)

    # ── Console summary ──
    print("=" * 70)
    print(f"Run: {run_dir}")
    print(f"Modes ({len(modes)}): {', '.join(modes)}")
    print(f"Total chunks: {len(rows)}")
    print("=" * 70)
    print(f"{'mode':<24} {'files':>6} {'chunks':>7} "
          f"{'privacy':>8} {'preserve':>9} {'overall_pass':>13}")
    print("-" * 70)
    for s in summary_rows:
        print(f"{s['mode']:<24} {s['n_files']:>6} {s['n_chunks']:>7} "
              f"{s.get('mean_privacy_score', ''):>8} "
              f"{s.get('mean_preserve_score', ''):>9} "
              f"{s.get('rate_overall_pass', ''):>13}")
    print()
    print(f"  ✓ {per_chunk_path}")
    print(f"  ✓ {by_mode_path}")


if __name__ == "__main__":
    main()
