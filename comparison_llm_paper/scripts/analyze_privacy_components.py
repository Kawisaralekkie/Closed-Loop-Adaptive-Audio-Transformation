#!/usr/bin/env python3
"""Analyze the raw privacy-score components before fixing any threshold.

Answers two methodology questions:

  (1) What do the RAW privacy values actually look like?
      → full distribution of WER, CER, content_privacy, speaker_privacy,
        privacy_score (min / max / mean / std / percentiles + histograms).

  (2) Where does the 0.7 / 0.3 (content / speaker) weighting come from, and
      does it correlate with the "true" privacy signal?
      → correlation of each component with privacy_score, plus a weight
        sweep showing how the content/speaker split changes the ranking and
        the correlation with WER (the most interpretable privacy proxy).

Input: per_chunk_metrics_paper.csv produced by plot_comparison_llm_v2.py
(columns: wer, cer, content_privacy, speaker_privacy, privacy_score, ...).

Usage:
    python3 scripts/analyze_privacy_components.py plots/comparison_llm/per_chunk_metrics_paper.csv
"""

from __future__ import annotations

import sys
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COMPONENTS = ["wer", "cer", "content_privacy", "speaker_privacy", "privacy_score"]


def _describe(series: pd.Series) -> dict:
    s = series.dropna().astype(float)
    return {
        "n": int(s.size),
        "min": float(s.min()),
        "p05": float(s.quantile(0.05)),
        "p25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "mean": float(s.mean()),
        "p75": float(s.quantile(0.75)),
        "p95": float(s.quantile(0.95)),
        "max": float(s.max()),
        "std": float(s.std()),
    }


def main(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    out_dir = os.path.dirname(csv_path) or "."

    missing = [c for c in COMPONENTS if c not in df.columns]
    if missing:
        print(f"ERROR: CSV missing columns: {missing}")
        sys.exit(1)

    # ── (1) Raw value distribution ──
    print("=" * 78)
    print("RAW PRIVACY COMPONENT DISTRIBUTION (no thresholds applied)")
    print("=" * 78)
    print(f"{'component':<18} {'n':>5} {'min':>7} {'p05':>7} {'p25':>7} "
          f"{'med':>7} {'mean':>7} {'p75':>7} {'p95':>7} {'max':>7} {'std':>7}")
    print("-" * 78)
    stats_rows = []
    for c in COMPONENTS:
        d = _describe(df[c])
        stats_rows.append({"component": c, **d})
        print(f"{c:<18} {d['n']:>5} {d['min']:>7.3f} {d['p05']:>7.3f} "
              f"{d['p25']:>7.3f} {d['median']:>7.3f} {d['mean']:>7.3f} "
              f"{d['p75']:>7.3f} {d['p95']:>7.3f} {d['max']:>7.3f} {d['std']:>7.3f}")
    pd.DataFrame(stats_rows).to_csv(os.path.join(out_dir, "privacy_components_stats.csv"), index=False)

    # Histograms
    fig, axes = plt.subplots(1, len(COMPONENTS), figsize=(4 * len(COMPONENTS), 4))
    for ax, c in zip(axes, COMPONENTS):
        ax.hist(df[c].dropna().astype(float), bins=30, color="#42A5F5", alpha=0.8, edgecolor="white")
        ax.set_title(c, fontsize=11)
        ax.axvline(df[c].mean(), color="red", ls="--", lw=1)
    fig.suptitle("Raw privacy component distributions", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "privacy_components_hist.png"), dpi=150)
    plt.close(fig)
    print(f"\n  ✓ privacy_components_stats.csv")
    print(f"  ✓ privacy_components_hist.png")

    # ── (2) Correlation of each component with privacy_score & with WER ──
    print("\n" + "=" * 78)
    print("CORRELATION with privacy_score and with WER (Pearson)")
    print("=" * 78)
    base = df[COMPONENTS].dropna().astype(float)
    for c in ["wer", "cer", "content_privacy", "speaker_privacy"]:
        r_ps = np.corrcoef(base[c], base["privacy_score"])[0, 1]
        r_wer = np.corrcoef(base[c], base["wer"])[0, 1]
        print(f"  {c:<18} corr(.,privacy_score)={r_ps:+.3f}   corr(.,WER)={r_wer:+.3f}")

    # ── (2b) Weight sweep: content/speaker split ──
    # privacy_score' = w*content_privacy + (1-w)*speaker_privacy
    # Show how mean privacy and correlation with WER change as w varies.
    print("\n" + "=" * 78)
    print("CONTENT/SPEAKER WEIGHT SWEEP")
    print("  privacy' = w*content_privacy + (1-w)*speaker_privacy")
    print("  (current system uses w=0.7)")
    print("=" * 78)
    print(f"{'w_content':>10} {'w_speaker':>10} {'mean_privacy':>14} {'corr_with_WER':>15}")
    print("-" * 52)
    cp = base["content_privacy"].values
    sp = base["speaker_privacy"].values
    wer = base["wer"].values
    sweep_rows = []
    for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        ps = w * cp + (1 - w) * sp
        mean_ps = float(ps.mean())
        corr_wer = float(np.corrcoef(ps, wer)[0, 1]) if ps.std() > 0 else float("nan")
        marker = "  ← current" if abs(w - 0.7) < 1e-9 else ""
        print(f"{w:>10.1f} {1-w:>10.1f} {mean_ps:>14.4f} {corr_wer:>15.3f}{marker}")
        sweep_rows.append({"w_content": w, "w_speaker": 1 - w,
                           "mean_privacy": mean_ps, "corr_with_wer": corr_wer})
    pd.DataFrame(sweep_rows).to_csv(os.path.join(out_dir, "privacy_weight_sweep.csv"), index=False)

    # Plot: how mean privacy & WER-correlation vary with w_content
    ws = [r["w_content"] for r in sweep_rows]
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(ws, [r["mean_privacy"] for r in sweep_rows], "o-", color="#1976D2", label="mean privacy")
    ax1.set_xlabel("w_content (weight on content_privacy)")
    ax1.set_ylabel("mean privacy_score", color="#1976D2")
    ax1.axvline(0.7, color="red", ls="--", lw=1, alpha=0.7, label="current (0.7)")
    ax2 = ax1.twinx()
    ax2.plot(ws, [r["corr_with_wer"] for r in sweep_rows], "s--", color="#388E3C", label="corr with WER")
    ax2.set_ylabel("corr(privacy', WER)", color="#388E3C")
    ax1.set_title("Effect of content/speaker weighting on privacy_score", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "privacy_weight_sweep.png"), dpi=150)
    plt.close(fig)
    print(f"\n  ✓ privacy_weight_sweep.csv")
    print(f"  ✓ privacy_weight_sweep.png")
    print("\nNOTE: The 0.7/0.3 split is a DESIGN CHOICE (content privacy prioritized")
    print("      over speaker privacy). Use the sweep above to justify or revise it.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/analyze_privacy_components.py <per_chunk_metrics_paper.csv>")
        sys.exit(1)
    main(sys.argv[1])
