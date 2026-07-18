#!/usr/bin/env python3
"""Plot comparison across **8 configurations** — full method-level evaluation.

Mirrors plot_comparison_llm_v2.py but expanded to compare:

  Level 1 (Fixed baselines)
    1. Fixed-MidBand          — Mid-band attenuation, fixed parameters
    2. Fixed-StrongBlur       — Strong blurring, fixed parameters
  Level 2 (Rule-based adaptive)
    3. Rule-NoMem-NoSS        — adaptive retry ladder, no source separation
    4. Rule-NoMem-SS          — adaptive retry ladder + source separation
  Level 3 (LLM-driven adaptive — Claude Haiku 4.5)
    5. LLM-H4.5-Mem           — memory ON,  source-sep OFF
    6. LLM-H4.5-Mem-SS        — memory ON,  source-sep ON   (proposed)
    7. LLM-H4.5-NoMem-SS      — memory OFF, source-sep ON
    8. LLM-H4.5-NoMem-NoSS    — memory OFF, source-sep OFF

Each configuration must have its own log folder containing ``*_report.json``
files. Update the VERSIONS dict below with the actual paths before running.

Usage:
    python3 scripts/plot_comparison_8configs.py
"""

from __future__ import annotations
import json, glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 24,
    "axes.titlesize": 24,
    "axes.labelsize": 22,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 24,
})

# ── Config ──
# 8 configurations — point each "path" to the folder of *_report.json
# files for that configuration. Replace placeholders with real folders.
#
# Color palette grouped by level:
#   Level 1 (Fixed)        — warm reds/oranges
#   Level 2 (Rule-based)   — pinks/magentas
#   Level 3 (LLM-driven)   — blues/greens
VERSIONS = {
    # ── Level 1: Fixed baselines ──
    "Fixed-MidBand": {
        "path": "logs/s3/PLACEHOLDER/fixed",
        "color": "#E57373",  # light red
    },
    "Fixed-StrongBlur": {
        "path": "logs/s3/PLACEHOLDER/fixed_strong",
        "color": "#C62828",  # deep red
    },
    # ── Level 2: Rule-based adaptive ──
    "Rule-NoMem-NoSS": {
        "path": "logs/s3/PLACEHOLDER/rule_based",
        "color": "#F8BBD0",  # pastel pink
    },
    "Rule-NoMem-SS": {
        "path": "logs/s3/20260429_093019/rule_based_ss",
        "color": "#EC407A",  # bright pink
    },
    # ── Level 3: LLM-driven adaptive (Claude Haiku 4.5) ──
    "LLM-H4.5-Mem": {
        "path": "logs/s3/20260429_093223/llm_with_memory_no_ss",
        "color": "#2196F3",  # blue
    },
    "LLM-H4.5-Mem-SS": {
        "path": "logs/s3/20260429_093321/llm_with_memory",
        "color": "#4CAF50",  # green (proposed)
    },
    "LLM-H4.5-NoMem-SS": {
        "path": "logs/s3/20260429_093355/llm_no_memory",
        "color": "#0D47A1",  # dark blue
    },
    "LLM-H4.5-NoMem-NoSS": {
        "path": "logs/s3/PLACEHOLDER/llm_no_memory_no_ss",
        "color": "#7E57C2",  # purple
    },
}

OUT_DIR = "plots/comparison_8configs"


def load(folder):
    if not folder or not os.path.exists(folder):
        return []
    r = []
    for f in sorted(glob.glob(os.path.join(folder, "*_report.json"))):
        with open(f) as fh:
            r.append(json.load(fh))
    return r


def sc(reports):
    return [c for r in reports for c in r.get("chunks", [])
            if c.get("had_speech") and c.get("metrics")]


def no_sc(reports):
    return [c for r in reports for c in r.get("chunks", [])
            if not c.get("had_speech") and c.get("metrics")]


# ─────────────────────────────────────────────────────────────────────────────
# [v2 PATCH] Recompute PreserveScore as (TC@3 + TA@1) / 2
#   • TC@3 = |top3_orig ∩ top3_proc| / 3      (Top-3 Consistency)
#   • TA@1 = 1 if top-1 labels match else 0   (Top-1 Agreement)
# This replaces the old confidence-based preserve_score embedded in JSON.
# Use semantic_preserve_score(c) everywhere instead of c["metrics"]["utility"]["preserve_score"].
# ─────────────────────────────────────────────────────────────────────────────

def semantic_preserve_score(c):
    """(TC@3 + TA@1) / 2 from a chunk's classification top3 vs original top3.
    Returns None if classification info is missing."""
    orig = c.get("classification_top3_original", [])
    proc = c.get("classification_top3", [])
    if not orig or not proc:
        return None
    set_a = set(p["label"] for p in orig[:3])
    set_b = set(p["label"] for p in proc[:3])
    tc3 = len(set_a & set_b) / 3.0
    ta1 = 1.0 if orig[0]["label"] == proc[0]["label"] else 0.0
    return (tc3 + ta1) / 2.0


def preserve_list(chunks):
    """Filter and return non-None preserve scores."""
    out = [semantic_preserve_score(c) for c in chunks]
    return [v for v in out if v is not None]


def annotate_bar(ax, bar, text, ylim_top):
    """Put text above bar, clamped below title."""
    y = min(bar.get_height() + 0.03, ylim_top - 0.12)
    ax.text(bar.get_x() + bar.get_width()/2, y, text,
            ha="center", va="bottom", fontweight="bold", fontsize=14)


def _set_xt(ax, ticks, labels):
    """Set x-tick positions and labels with rotation for many configs.

    With 8 configurations the labels overlap unless rotated; we use a
    moderate 25° rotation that keeps them readable yet aligned right.
    """
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels,
                       rotation=25 if len(labels) > 4 else 0,
                       ha="right" if len(labels) > 4 else "center",
                       fontsize=18 if len(labels) > 4 else 22)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load data — skip configurations whose folder doesn't exist (e.g.
    # placeholder paths that haven't been filled in yet).
    data = {}
    for label, cfg in VERSIONS.items():
        reports = load(cfg["path"])
        if reports:
            data[label] = {"reports": reports, "color": cfg["color"]}
            print(f"  {label}: {len(reports)} reports, {len(sc(reports))} speech chunks")
        else:
            print(f"  WARNING: {label} — no data at {cfg['path']} (skipped)")

    if not data:
        print("ERROR: No data"); sys.exit(1)

    labels = list(data.keys())
    colors = [data[l]["color"] for l in labels]
    x = np.arange(len(labels))
    # Bar width scaled for the configuration count (8 vs 4)
    w = 0.6 if len(labels) <= 4 else 0.55

    print(f"\nPlotting {len(labels)} configurations to {OUT_DIR}/\n")

    # ── 1. Average Privacy Score ──
    means = [np.mean([c["metrics"]["privacy"]["privacy_score"] for c in sc(data[l]["reports"])]) for l in labels]
    stds = [np.std([c["metrics"]["privacy"]["privacy_score"] for c in sc(data[l]["reports"])]) for l in labels]
    fig, ax = plt.subplots(figsize=(18, 7))
    bars = ax.bar(x, means, w, yerr=stds, capsize=5, color=colors, alpha=0.85)
    for bar, m, s in zip(bars, means, stds):
        annotate_bar(ax, bar, f"{m:.3f}±{s:.3f}", 1.3)
    ax.axhline(0.65, color="orange", ls="--", lw=1, alpha=0.7, label="High (0.65)")
    _set_xt(ax, x, labels)
    ax.set_ylabel("Privacy Score"); ax.set_ylim(0, 1.3)
    ax.set_title("Average Privacy Score (±σ)", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, borderaxespad=0); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "avg_privacy_score.png"), dpi=150); plt.close(fig)
    print("  ✓ avg_privacy_score.png")

    # ── 2. Average Utility Score [v2: semantic preserve score] ──
    means = [np.mean(preserve_list(sc(data[l]["reports"]))) for l in labels]
    stds = [np.std(preserve_list(sc(data[l]["reports"]))) for l in labels]
    fig, ax = plt.subplots(figsize=(18, 7))
    bars = ax.bar(x, means, w, yerr=stds, capsize=5, color=colors, alpha=0.85)
    for bar, m, s in zip(bars, means, stds):
        annotate_bar(ax, bar, f"{m:.4f}±{s:.4f}", 1.1)
    ax.axhline(0.80, color="red", ls="--", lw=1, alpha=0.7, label="Threshold (0.80)")

    # ── No-speech BASELINE (best-case ceiling) ──
    # Environmental chunks with NO speech are bypassed (not blurred) — natural
    # upper bound for utility. Pooled across all configs.
    ns_scores_all = []
    for l in labels:
        ns_scores_all.extend(preserve_list(no_sc(data[l]["reports"])))
    if ns_scores_all:
        ns_baseline = float(np.mean(ns_scores_all))
        ax.axhline(ns_baseline, color="green", ls="-.", lw=2, alpha=0.85,
                   label=f"No-speech baseline ({ns_baseline:.3f}, n={len(ns_scores_all)})")

    _set_xt(ax, x, labels)
    ax.set_ylabel("Utility Score (TC@3+TA@1)/2"); ax.set_ylim(0, 1.1)
    ax.set_title("Average Utility Score (±σ)\n[Semantic Preservation]", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, borderaxespad=0); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "avg_utility_score.png"), dpi=150); plt.close(fig)
    print("  ✓ avg_utility_score.png")

    # ── 3. Privacy vs Utility Trade-off [v2: semantic preserve] ──
    fig, ax = plt.subplots(figsize=(10, 8))
    for l in labels:
        s = sc(data[l]["reports"])
        # Pair (privacy, semantic preserve) — drop chunks lacking top3
        pairs = []
        for c in s:
            ps = semantic_preserve_score(c)
            if ps is None:
                continue
            pairs.append((c["metrics"]["privacy"]["privacy_score"], ps))
        if pairs:
            priv, util = zip(*pairs)
            ax.scatter(priv, util, alpha=0.4, s=25, label=l, color=data[l]["color"])
    ax.axvline(0.65, color="orange", ls="--", lw=1, alpha=0.5)
    ax.axhline(0.80, color="red", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("Privacy Score"); ax.set_ylabel("Utility Score (TC@3+TA@1)/2")
    ax.set_title("Privacy vs Utility Trade-off\n[Semantic Preservation]", fontweight="bold")
    ax.legend(loc="lower left"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "privacy_utility_tradeoff.png"), dpi=150); plt.close(fig)
    print("  ✓ privacy_utility_tradeoff.png")

    # ── 4. Classification Metrics ──
    metric_names = ["mAP", "accuracy", "f1"]
    metric_labels_display = ["mAP", "Accuracy", "F1-Score"]
    metric_colors = ["#1976D2", "#F57C00", "#388E3C"]
    bar_w = 0.8 / len(metric_names)
    fig, ax = plt.subplots(figsize=(18, 7))
    for j, (mname, mlabel, mcolor) in enumerate(zip(metric_names, metric_labels_display, metric_colors)):
        vals = []
        for l in labels:
            s = sc(data[l]["reports"])
            scores = [c["metrics"]["utility"][mname] for c in s if mname in c["metrics"]["utility"]]
            vals.append(np.mean(scores) if scores else 0)
        ax.bar(x + j * bar_w - (len(metric_names)-1)*bar_w/2, vals, bar_w,
               label=mlabel, color=mcolor, alpha=0.85)
    _set_xt(ax, x, labels)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.2)
    ax.set_title("Classification Metrics (mAP, Accuracy, F1)", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, borderaxespad=0); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "classification_metrics.png"), dpi=150); plt.close(fig)
    print("  ✓ classification_metrics.png")

    # ── 5. Recipe Usage (including Source Separation pre-processing) ──
    all_recipes = set()
    ver_recipes = {}
    for l in labels:
        s = sc(data[l]["reports"])
        counts = {}
        for c in s:
            # Check if source separation was used as pre-processing
            if c.get("used_source_separation"):
                counts["SOURCE_SEPARATION (pre)"] = counts.get("SOURCE_SEPARATION (pre)", 0) + 1
                all_recipes.add("SOURCE_SEPARATION (pre)")
            r = c.get("recipe_applied", {})
            if r:
                name = r.get("recipe_name", "unknown").replace("RECIPE_", "")
                counts[name] = counts.get(name, 0) + 1
                all_recipes.add(name)
        ver_recipes[l] = counts
    recipe_colors = {"MID_BAND_ATTEN": "#2196F3", "LOWPASS_HIGHPASS_MIX": "#FF9800", "SOURCE_SEPARATION": "#4CAF50", "SOURCE_SEPARATION (pre)": "#81C784"}
    recipes = sorted(all_recipes)
    fig, ax = plt.subplots(figsize=(18, 7))
    bottom = np.zeros(len(labels))
    for recipe in recipes:
        vals = [ver_recipes[l].get(recipe, 0) for l in labels]
        ax.bar(x, vals, 0.5, bottom=bottom, label=recipe, color=recipe_colors.get(recipe, "#999"), alpha=0.85)
        bottom += np.array(vals)
    _set_xt(ax, x, labels)
    ax.set_ylabel("Chunks"); ax.set_title("Recipe Usage (incl. Source Separation pre-processing)", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, borderaxespad=0); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "recipe_usage.png"), dpi=150); plt.close(fig)
    print("  ✓ recipe_usage.png")

    # ── 6. Trials Distribution ──
    fig, ax = plt.subplots(figsize=(18, 7))
    for i, l in enumerate(labels):
        s = sc(data[l]["reports"])
        trials = [c.get("trials", 0) for c in s if c.get("trials", 0) > 0]
        avg = np.mean(trials) if trials else 0
        bar = ax.bar(i, avg, color=colors[i], alpha=0.85)
        ax.text(i, avg + 0.05, f"{avg:.2f}", ha="center", fontweight="bold")
    _set_xt(ax, list(range(len(labels))), labels)
    ax.set_ylabel("Avg Trials"); ax.set_title("Average Trials per Chunk", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "trials_distribution.png"), dpi=150); plt.close(fig)
    print("  ✓ trials_distribution.png")

    # ── 7. Speech Ratio vs Privacy (binned-mean trend + annotations) ──
    fig, ax = plt.subplots(figsize=(20, 14))

    # Helper: per-method binned mean (8 bins along SR axis, skip sparse bins)
    def _binned_trend(ratios, priv, n_bins=8, min_n=3):
        if not ratios:
            return [], []
        r = np.asarray(ratios); p = np.asarray(priv)
        edges = np.linspace(0, 1, n_bins + 1)
        cx, cy = [], []
        for i in range(n_bins):
            lo, hi = edges[i], edges[i+1]
            mask = (r >= lo) & (r < hi if i < n_bins - 1 else r <= hi)
            if mask.sum() >= min_n:
                cx.append((lo + hi) / 2.0)
                cy.append(float(p[mask].mean()))
        return cx, cy

    # (a) Light scatter in background
    for l in labels:
        s = sc(data[l]["reports"])
        non_ss  = [c for c in s if not c.get("used_source_separation")]
        ss_only = [c for c in s if c.get("used_source_separation")]
        if non_ss:
            r = [c.get("speech_ratio", 0) for c in non_ss]
            p = [c["metrics"]["privacy"]["privacy_score"] for c in non_ss]
            ax.scatter(r, p, alpha=0.10, s=16, color=data[l]["color"], marker="o", zorder=1)
        if ss_only:
            r = [c.get("speech_ratio", 0) for c in ss_only]
            p = [c["metrics"]["privacy"]["privacy_score"] for c in ss_only]
            ax.scatter(r, p, alpha=0.85, s=120, color=data[l]["color"],
                       marker="*", edgecolors="black", linewidths=0.7, zorder=4,
                       label=f"{l} (SS applied)")

    # (b) Bold trend lines per method
    for l in labels:
        s = sc(data[l]["reports"])
        r_all = [c.get("speech_ratio", 0) for c in s]
        p_all = [c["metrics"]["privacy"]["privacy_score"] for c in s]
        cx, cy = _binned_trend(r_all, p_all, n_bins=8)
        if cx:
            ax.plot(cx, cy, color=data[l]["color"], lw=3.0, marker="o",
                    markersize=9, markeredgecolor="black", markeredgewidth=0.7,
                    label=f"{l} (binned mean)", zorder=3)

    # Reference lines
    ax.axhline(0.65, color="orange", ls="--", lw=1.2, alpha=0.7, zorder=2)
    ax.text(0.005, 0.665, "T_p = 0.65", color="orange", fontsize=30, va="bottom")
    ax.axvline(0.3, color="red", ls=":", lw=1.2, alpha=0.5, zorder=2)
    ax.text(0.32, 0.55, "SR ≥ 0.3\n(SS pre-proc\nthreshold)",
            color="red", fontsize=30, va="center", ha="left")

    # ── Annotations ──
    # (1) Pink-star cluster (Rule-NoMem-SS) stays near top across all SR
    ax.annotate(
        "SS maintains high privacy\nacross all speech ratios",
        xy=(0.86, 0.88), xytext=(0.30, 1.07),
        fontsize=30, fontweight="bold", color="#880E4F", ha="center",
        arrowprops=dict(arrowstyle="->", color="#880E4F", lw=1.8,
                        connectionstyle="arc3,rad=-0.20"),
        bbox=dict(boxstyle="round,pad=0.35", fc="#FCE4EC", ec="#880E4F", lw=1),
        zorder=6,
    )
    # (2) Bottom-right LLM cluster
    ax.annotate(
        "Adaptive intentionally trades privacy\nfor utility at high speech ratios",
        xy=(0.92, 0.40), xytext=(0.65, 0.18),
        fontsize=30, fontweight="bold", color="#0D47A1", ha="center",
        arrowprops=dict(arrowstyle="->", color="#0D47A1", lw=1.8,
                        connectionstyle="arc3,rad=0.25"),
        bbox=dict(boxstyle="round,pad=0.35", fc="#E3F2FD", ec="#0D47A1", lw=1),
        zorder=6,
    )

    ax.set_xlabel("Speech Ratio", fontsize=30)
    ax.set_ylabel("Privacy Score", fontsize=30)
    ax.tick_params(axis="both", labelsize=30)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.20)
    ax.set_title(
        "Speech Ratio vs Privacy Score\n"
        "(Bold lines = per-method binned mean; stars = chunks where SS was applied)",
        fontweight="bold", fontsize=30)
    ax.legend(loc="lower left", fontsize=30, framealpha=0.92, ncol=1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "speech_ratio_vs_privacy.png"), dpi=150)
    plt.close(fig)
    print("  ✓ speech_ratio_vs_privacy.png")

    # ── 8. Top-K Class Consistency ──
    topk_data = {}
    for l in labels:
        s = sc(data[l]["reports"])
        consistency = []
        for c in s:
            orig = c.get("classification_top3_original", [])
            proc = c.get("classification_top3", [])
            if orig and proc:
                orig_labels = set(p["label"] for p in orig[:3])
                proc_labels = set(p["label"] for p in proc[:3])
                overlap = len(orig_labels & proc_labels)
                consistency.append(overlap / max(len(orig_labels), 1))
        if consistency:
            topk_data[l] = consistency

    if topk_data:
        tk_labels = [l for l in labels if l in topk_data]
        tk_x = np.arange(len(tk_labels))
        means = [np.mean(topk_data[l]) for l in tk_labels]
        stds = [np.std(topk_data[l]) for l in tk_labels]
        tk_colors = [data[l]["color"] for l in tk_labels]
        fig, ax = plt.subplots(figsize=(18, 7))
        bars = ax.bar(tk_x, means, 0.5, yerr=stds, capsize=5, color=tk_colors, alpha=0.85)
        for bar, m, s in zip(bars, means, stds):
            annotate_bar(ax, bar, f"{m:.3f}±{s:.3f}", 1.4)
        ax.set_xticks(tk_x); ax.set_xticklabels(tk_labels)
        ax.set_ylabel("Consistency (0-1)"); ax.set_ylim(0, 1.4)
        ax.set_title("Top-3 Class Consistency\n(Environmental labels preserved after processing)", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "topk_class_consistency.png"), dpi=150); plt.close(fig)
        print("  ✓ topk_class_consistency.png")

    # ── 9. Classification Label Change ──
    changed_counts = []
    for l in labels:
        s = sc(data[l]["reports"])
        changed, total = 0, 0
        for c in s:
            orig = c.get("classification_top3_original", [])
            proc = c.get("classification_top3", [])
            if orig and proc:
                total += 1
                if orig[0]["label"] != proc[0]["label"]:
                    changed += 1
        changed_counts.append((l, changed, total))

    if any(t > 0 for _, _, t in changed_counts):
        fig, ax = plt.subplots(figsize=(18, 7))
        pct = [100 * ch / max(t, 1) for _, ch, t in changed_counts]
        bars = ax.bar(x, pct, w, color=colors, alpha=0.85)
        for bar, (lbl, ch, t) in zip(bars, changed_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{ch}/{t} ({bar.get_height():.1f}%)", ha="center", fontweight="bold", fontsize=11)
        _set_xt(ax, x, labels)
        ax.set_ylabel("% Chunks with Changed Top-1 Label")
        ax.set_title("Classification Label Change After Processing", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "classification_label_change.png"), dpi=150); plt.close(fig)
        print("  ✓ classification_label_change.png")

    # ── 10. Classification Top Labels ──
    fig, axes = plt.subplots(1, len(labels), figsize=(6*len(labels), 6), squeeze=False)
    axes = axes[0]
    for i, l in enumerate(labels):
        ax = axes[i]
        s = sc(data[l]["reports"])
        label_counts = Counter()
        for c in s:
            top3 = c.get("classification_top3") or []
            if top3:
                label_counts[top3[0]["label"]] += 1
        if label_counts:
            top_items = label_counts.most_common(10)
            lnames = [item[0][:20] for item in top_items]
            lcounts = [item[1] for item in top_items]
            ax.barh(range(len(lnames)), lcounts, color=data[l]["color"], alpha=0.85)
            ax.set_yticks(range(len(lnames))); ax.set_yticklabels(lnames, fontsize=10)
            ax.invert_yaxis()
        ax.set_xlabel("Count"); ax.set_title(l, fontsize=13)
    fig.suptitle("Top-1 Classification Labels (Processed Audio)", fontweight="bold")
    fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "classification_top_labels.png"), dpi=150); plt.close(fig)
    print("  ✓ classification_top_labels.png")

    # ── 11. Speech Confidence Gap ──
    conf_data = {}
    for l in labels:
        s = sc(data[l]["reports"])
        gaps = []
        for c in s:
            orig = c.get("classification_top3_original", [])
            proc = c.get("classification_top3", [])
            if orig and proc:
                def find_speech_conf(preds):
                    for p in preds:
                        if "speech" in p["label"].lower():
                            return p["confidence"]
                    return 0.0
                gaps.append(find_speech_conf(orig) - find_speech_conf(proc))
        if gaps:
            conf_data[l] = gaps

    if conf_data:
        cd_labels = [l for l in labels if l in conf_data]
        cd_x = np.arange(len(cd_labels))
        means = [np.mean(conf_data[l]) for l in cd_labels]
        stds = [np.std(conf_data[l]) for l in cd_labels]
        cd_colors = [data[l]["color"] for l in cd_labels]
        fig, ax = plt.subplots(figsize=(18, 7))
        bars = ax.bar(cd_x, means, 0.5, yerr=stds, capsize=5, color=cd_colors, alpha=0.85)
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.02,
                    f"{m:+.3f}±{s:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(cd_x); ax.set_xticklabels(cd_labels)
        ax.set_ylabel("Confidence Drop (positive = reduced)")
        ax.set_title("Speech Confidence Gap (Original − Processed)\n(Higher = better privacy)", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "speech_confidence_gap.png"), dpi=150); plt.close(fig)
        print("  ✓ speech_confidence_gap.png")

    # ── 12. Speech Rank Drop ──
    rank_data = {}
    for l in labels:
        s = sc(data[l]["reports"])
        drops = []
        for c in s:
            orig = c.get("classification_top3_original", [])
            proc = c.get("classification_top3", [])
            if orig and proc:
                def find_speech_rank(preds):
                    for i, p in enumerate(preds):
                        if "speech" in p["label"].lower():
                            return i + 1
                    return len(preds) + 1
                drops.append(find_speech_rank(proc) - find_speech_rank(orig))
        if drops:
            rank_data[l] = drops

    if rank_data:
        rd_labels = [l for l in labels if l in rank_data]
        rd_x = np.arange(len(rd_labels))
        means = [np.mean(rank_data[l]) for l in rd_labels]
        stds = [np.std(rank_data[l]) for l in rd_labels]
        rd_colors = [data[l]["color"] for l in rd_labels]
        fig, ax = plt.subplots(figsize=(18, 7))
        bars = ax.bar(rd_x, means, 0.5, yerr=stds, capsize=5, color=rd_colors, alpha=0.85)
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, max(bar.get_height(), 0) + s + 0.1,
                    f"{m:+.2f}±{s:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(rd_x); ax.set_xticklabels(rd_labels)
        ax.set_ylabel("Rank Drop (positive = dropped)")
        ax.set_title("Speech Rank Drop After Processing\n(Higher = better privacy)", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "speech_rank_drop.png"), dpi=150); plt.close(fig)
        print("  ✓ speech_rank_drop.png")

    # ── 13. No-Speech Analysis [v2: semantic preserve] ──
    ns_labels, ns_means, ns_stds, ns_counts = [], [], [], []
    for l in labels:
        ns = no_sc(data[l]["reports"])
        if ns:
            scores = preserve_list(ns)
            if scores:
                ns_labels.append(l)
                ns_means.append(np.mean(scores))
                ns_stds.append(np.std(scores))
                ns_counts.append(len(scores))

    if ns_labels:
        ns_x = np.arange(len(ns_labels))
        ns_colors = [data[l]["color"] for l in ns_labels]
        fig, ax = plt.subplots(figsize=(18, 7))
        bars = ax.bar(ns_x, ns_means, 0.5, yerr=ns_stds, capsize=5, color=ns_colors, alpha=0.85)
        for bar, m, s, n in zip(bars, ns_means, ns_stds, ns_counts):
            ax.text(bar.get_x() + bar.get_width()/2, min(bar.get_height() + s + 0.005, 1.05),
                    f"{m:.4f}±{s:.4f} (n={n})", ha="center", va="bottom", fontweight="bold", fontsize=10)
        ax.axhline(0.80, color="red", ls="--", lw=1, alpha=0.7)
        ax.set_xticks(ns_x); ax.set_xticklabels(ns_labels)
        ax.set_ylabel("Utility Score"); ax.set_ylim(0.6, 1.1)
        ax.set_title("Utility Score — No-Speech Chunks (Bypass)", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "no_speech_utility.png"), dpi=150); plt.close(fig)
        print("  ✓ no_speech_utility.png")

    # ── 14. Source Separation Analysis ──
    # Identify chunks that used source separation
    ss_analysis = {}
    for l in labels:
        s = sc(data[l]["reports"])
        ss_chunks = [c for c in s if c.get("used_source_separation")]
        non_ss_chunks = [c for c in s if not c.get("used_source_separation")]
        ss_analysis[l] = {"ss": ss_chunks, "non_ss": non_ss_chunks, "total": len(s)}

    # 14a. SS usage count
    fig, ax = plt.subplots(figsize=(18, 7))
    ss_counts = [len(ss_analysis[l]["ss"]) for l in labels]
    non_ss_counts = [len(ss_analysis[l]["non_ss"]) for l in labels]
    bar_w2 = 0.35
    b1 = ax.bar(x - bar_w2/2, non_ss_counts, bar_w2, label="Without SS", color="#90CAF9", alpha=0.85)
    b2 = ax.bar(x + bar_w2/2, ss_counts, bar_w2, label="With SS", color="#4CAF50", alpha=0.85)
    for bar, val in zip(b1, non_ss_counts):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(val), ha="center", fontweight="bold", fontsize=11)
    for bar, val in zip(b2, ss_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(val), ha="center", fontweight="bold", fontsize=11)
    _set_xt(ax, x, labels)
    ax.set_ylabel("Number of Chunks")
    ax.set_title("Source Separation Usage (Speech Chunks)", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, borderaxespad=0); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "source_sep_usage.png"), dpi=150); plt.close(fig)
    print("  ✓ source_sep_usage.png")

    # 14b. Privacy score: SS vs non-SS
    fig, ax = plt.subplots(figsize=(18, 7))
    ss_priv = []
    non_ss_priv = []
    plot_labels = []
    for l in labels:
        ss = ss_analysis[l]["ss"]
        non_ss = ss_analysis[l]["non_ss"]
        if ss or non_ss:
            plot_labels.append(l)
            ss_priv.append(np.mean([c["metrics"]["privacy"]["privacy_score"] for c in ss]) if ss else 0)
            non_ss_priv.append(np.mean([c["metrics"]["privacy"]["privacy_score"] for c in non_ss]) if non_ss else 0)
    if plot_labels:
        px = np.arange(len(plot_labels))
        b1 = ax.bar(px - bar_w2/2, non_ss_priv, bar_w2, label="Without SS", color="#90CAF9", alpha=0.85)
        b2 = ax.bar(px + bar_w2/2, ss_priv, bar_w2, label="With SS", color="#4CAF50", alpha=0.85)
        for bar, m in zip(b1, non_ss_priv):
            if m > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{m:.3f}", ha="center", fontweight="bold", fontsize=11)
        for bar, m in zip(b2, ss_priv):
            if m > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{m:.3f}", ha="center", fontweight="bold", fontsize=11)
        ax.axhline(0.65, color="orange", ls="--", lw=1, alpha=0.7)
        ax.set_xticks(px); ax.set_xticklabels(plot_labels)
        ax.set_ylabel("Privacy Score"); ax.set_ylim(0, 1.2)
        ax.set_title("Privacy Score: With vs Without Source Separation", fontweight="bold")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, borderaxespad=0); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "source_sep_privacy.png"), dpi=150); plt.close(fig)
        print("  ✓ source_sep_privacy.png")

    # 14c. All metrics comparison: SS vs non-SS  [v2: semantic preserve for "preserve_score"]
    metric_keys = [("privacy_score", "Privacy"), ("preserve_score", "Utility"), ("mAP", "mAP"), ("accuracy", "Accuracy"), ("f1", "F1")]
    rows_ss = []
    for l in labels:
        ss = ss_analysis[l]["ss"]
        non_ss = ss_analysis[l]["non_ss"]
        row = [l, str(len(ss)), str(len(non_ss))]
        for key, _ in metric_keys:
            if key == "privacy_score":
                ss_val = np.mean([c["metrics"]["privacy"][key] for c in ss]) if ss else 0
                non_val = np.mean([c["metrics"]["privacy"][key] for c in non_ss]) if non_ss else 0
            elif key == "preserve_score":
                ss_scores = preserve_list(ss)
                non_scores = preserve_list(non_ss)
                ss_val = np.mean(ss_scores) if ss_scores else 0
                non_val = np.mean(non_scores) if non_scores else 0
            else:
                ss_val = np.mean([c["metrics"]["utility"][key] for c in ss if key in c["metrics"]["utility"]]) if ss else 0
                non_val = np.mean([c["metrics"]["utility"][key] for c in non_ss if key in c["metrics"]["utility"]]) if non_ss else 0
            row.append(f"{ss_val:.4f}" if ss else "N/A")
            row.append(f"{non_val:.4f}" if non_ss else "N/A")
        rows_ss.append(row)

    cols_ss = ["Version", "SS\nChunks", "Non-SS\nChunks"]
    for _, name in metric_keys:
        cols_ss.extend([f"{name}\n(SS)", f"{name}\n(No SS)"])

    fig, ax = plt.subplots(figsize=(24, 2 + 0.6 * len(rows_ss)))
    ax.axis("off")
    table = ax.table(cellText=rows_ss, colLabels=cols_ss, loc="center", cellLoc="center", colColours=["#E8F5E9"] * len(cols_ss))
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.8)
    ax.set_title("Source Separation Impact Summary", fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "source_sep_summary.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
    print("  ✓ source_sep_summary.png")

    # ── 15. Speech Density Filtered Analysis [v2: semantic preserve] ──
    fig, ax = plt.subplots(figsize=(18, 7))
    metric_keys_density = [("privacy_score", "privacy", "Privacy Score"), ("preserve_score", "utility", "Utility Score"), ("accuracy", "utility", "Accuracy")]
    bar_wd = 0.8 / (len(labels) * 2)
    group_w = 0.8 / len(metric_keys_density)
    for j, (key, section, mlabel) in enumerate(metric_keys_density):
        all_vals, high_vals = [], []
        for l in labels:
            s = sc(data[l]["reports"])
            high = [c for c in s if c.get("speech_ratio", 0) >= 0.5]
            if section == "privacy":
                all_vals.append(np.mean([c["metrics"][section][key] for c in s]) if s else 0)
                high_vals.append(np.mean([c["metrics"][section][key] for c in high]) if high else 0)
            elif key == "preserve_score":
                all_scores = preserve_list(s)
                high_scores = preserve_list(high)
                all_vals.append(np.mean(all_scores) if all_scores else 0)
                high_vals.append(np.mean(high_scores) if high_scores else 0)
            else:
                all_vals.append(np.mean([c["metrics"][section][key] for c in s if key in c["metrics"][section]]) if s else 0)
                high_vals.append(np.mean([c["metrics"][section][key] for c in high if key in c["metrics"][section]]) if high else 0)
        gx = j * (len(labels) + 1)
        for i, (av, hv, l) in enumerate(zip(all_vals, high_vals, labels)):
            ax.bar(gx + i*0.4, av, 0.18, color=data[l]["color"], alpha=0.5)
            ax.bar(gx + i*0.4 + 0.18, hv, 0.18, color=data[l]["color"], alpha=1.0)
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="grey", alpha=0.5, label="All Chunks"),
                       Patch(facecolor="grey", alpha=1.0, label="High Density (SR≥0.5)")]
    ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, borderaxespad=0)
    xtick_pos = [j * (len(labels) + 1) + len(labels)*0.2 for j in range(len(metric_keys_density))]
    ax.set_xticks(xtick_pos); ax.set_xticklabels([m[2] for m in metric_keys_density])
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.2)
    ax.set_title("Impact of Speech Density: All vs High-Density Chunks (SR≥0.5)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "speech_density_analysis.png"), dpi=150); plt.close(fig)
    print("  ✓ speech_density_analysis.png")

    # ── 16. WER vs Privacy Score (Scatter + Regression) ──
    fig, ax = plt.subplots(figsize=(12, 8))
    for l in labels:
        s = sc(data[l]["reports"])
        wer = [c["metrics"]["privacy"]["wer"] for c in s]
        priv = [c["metrics"]["privacy"]["privacy_score"] for c in s]
        ax.scatter(wer, priv, alpha=0.4, s=25, label=l, color=data[l]["color"])
    # Overall regression line
    all_wer, all_priv = [], []
    for l in labels:
        s = sc(data[l]["reports"])
        all_wer.extend([c["metrics"]["privacy"]["wer"] for c in s])
        all_priv.extend([c["metrics"]["privacy"]["privacy_score"] for c in s])
    if all_wer:
        z = np.polyfit(all_wer, all_priv, 1)
        p = np.poly1d(z)
        wx = np.linspace(0, 1, 100)
        ax.plot(wx, p(wx), "r--", lw=2, alpha=0.7, label=f"Regression (r={np.corrcoef(all_wer, all_priv)[0,1]:.3f})")
    ax.set_xlabel("WER (Word Error Rate)"); ax.set_ylabel("Privacy Score")
    ax.set_title("WER vs Privacy Score\n(Higher WER = speech more obscured)", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, borderaxespad=0); ax.grid(alpha=0.3)
    fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "wer_vs_privacy.png"), dpi=150); plt.close(fig)
    print("  ✓ wer_vs_privacy.png")

    # ── 17. WER Distribution by Recipe Level ──
    fig, ax = plt.subplots(figsize=(18, 7))
    for l in labels:
        s = sc(data[l]["reports"])
        wer_vals = [c["metrics"]["privacy"]["wer"] for c in s]
        if wer_vals:
            ax.hist(wer_vals, bins=20, alpha=0.4, label=l, color=data[l]["color"], edgecolor="white")
    ax.set_xlabel("WER"); ax.set_ylabel("Count")
    ax.set_title("WER Distribution (Content Privacy)\n(WER=1.0 means completely unintelligible)", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, borderaxespad=0); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "wer_distribution.png"), dpi=150); plt.close(fig)
    print("  ✓ wer_distribution.png")

    # ── 18. Utility by Sound Class (Box Plot) [v2: semantic preserve] ──
    # Use classification_top3_original to group by sound type
    class_utility = {}
    for l in labels:
        s = sc(data[l]["reports"])
        for c in s:
            top3_orig = c.get("classification_top3_original", [])
            ps = semantic_preserve_score(c)
            if top3_orig and ps is not None:
                sound_class = top3_orig[0]["label"]
                class_utility.setdefault(sound_class, []).append(ps)

    if class_utility:
        # Top 10 most common classes
        sorted_classes = sorted(class_utility.items(), key=lambda x: -len(x[1]))[:10]
        class_names = [c[0][:20] for c in sorted_classes]
        class_scores = [c[1] for c in sorted_classes]
        fig, ax = plt.subplots(figsize=(20, 7))
        bp = ax.boxplot(class_scores, labels=class_names, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#90CAF9"); patch.set_alpha(0.7)
        ax.axhline(0.80, color="red", ls="--", lw=1, alpha=0.7, label="Threshold (0.80)")
        ax.set_ylabel("Utility Score")
        ax.set_title("Utility Score by Original Sound Class\n(Environmental sound preservation across categories)", fontweight="bold")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, borderaxespad=0); ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "utility_by_sound_class.png"), dpi=150); plt.close(fig)
        print("  ✓ utility_by_sound_class.png")

    # ── 19. Per-Class Analysis Table [v2: semantic preserve] ──
    # Privacy and Utility broken down by original sound class
    class_metrics = {}
    for l in labels:
        s = sc(data[l]["reports"])
        for c in s:
            top3_orig = c.get("classification_top3_original", [])
            ps = semantic_preserve_score(c)
            if top3_orig and ps is not None:
                cls = top3_orig[0]["label"]
                class_metrics.setdefault(cls, {}).setdefault(l, []).append({
                    "privacy": c["metrics"]["privacy"]["privacy_score"],
                    "utility": ps,
                    "wer": c["metrics"]["privacy"]["wer"],
                })

    if class_metrics:
        # Top 8 classes by count
        top_classes = sorted(class_metrics.items(), key=lambda x: -sum(len(v) for v in x[1].values()))[:8]
        cols_pc = ["Class", "n"] + [f"{l}\nPrivacy" for l in labels] + [f"{l}\nUtility" for l in labels]
        rows_pc = []
        for cls, ver_data in top_classes:
            row = [cls[:18], str(sum(len(v) for v in ver_data.values()))]
            for l in labels:
                chunks = ver_data.get(l, [])
                row.append(f"{np.mean([c['privacy'] for c in chunks]):.3f}" if chunks else "—")
            for l in labels:
                chunks = ver_data.get(l, [])
                row.append(f"{np.mean([c['utility'] for c in chunks]):.3f}" if chunks else "—")
            rows_pc.append(row)

        fig, ax = plt.subplots(figsize=(24, 2 + 0.5 * len(rows_pc)))
        ax.axis("off")
        table = ax.table(cellText=rows_pc, colLabels=cols_pc, loc="center", cellLoc="center", colColours=["#E8F5E9"] * len(cols_pc))
        table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.7)
        ax.set_title("Per-Class Privacy & Utility Analysis", fontweight="bold", pad=20)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "per_class_analysis.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
        print("  ✓ per_class_analysis.png")

    # ── 20. Chunk-Level Paired Comparison (Rule vs LLM-Mem-SS) ──
    rule_key = "Rule-NoMem-SS"
    llm_key = "LLM-H4.5-Mem-SS"
    if rule_key in data and llm_key in data:
        rule_reports = data[rule_key]["reports"]
        llm_reports = data[llm_key]["reports"]
        # Match by source_id
        rule_by_src = {r["source_id"]: r for r in rule_reports}
        llm_by_src = {r["source_id"]: r for r in llm_reports}
        common_srcs = set(rule_by_src.keys()) & set(llm_by_src.keys())

        rule_priv, llm_priv = [], []
        for src in common_srcs:
            rc = sc([rule_by_src[src]])
            lc = sc([llm_by_src[src]])
            if rc and lc:
                rule_priv.append(np.mean([c["metrics"]["privacy"]["privacy_score"] for c in rc]))
                llm_priv.append(np.mean([c["metrics"]["privacy"]["privacy_score"] for c in lc]))

        if rule_priv:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.scatter(rule_priv, llm_priv, alpha=0.4, s=30, color="#4CAF50")
            ax.plot([0, 1], [0, 1], "r--", lw=1, alpha=0.7, label="Equal line")
            ax.set_xlabel(f"Privacy Score ({rule_key})"); ax.set_ylabel(f"Privacy Score ({llm_key})")
            ax.set_title(f"Chunk-Level Paired Comparison\n{llm_key} vs {rule_key}", fontweight="bold")
            # Count wins
            llm_wins = sum(1 for r, l in zip(rule_priv, llm_priv) if l > r)
            rule_wins = sum(1 for r, l in zip(rule_priv, llm_priv) if r > l)
            ties = len(rule_priv) - llm_wins - rule_wins
            ax.text(0.05, 0.95, f"LLM wins: {llm_wins}\nRule wins: {rule_wins}\nTies: {ties}\nn={len(rule_priv)}",
                    transform=ax.transAxes, va="top", fontsize=12, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, borderaxespad=0); ax.grid(alpha=0.3)
            fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "paired_comparison_rule_vs_llm.png"), dpi=150); plt.close(fig)
            print("  ✓ paired_comparison_rule_vs_llm.png")

    # ── 21. WER-Only Privacy Ranking ──
    # Compare methods using WER alone as privacy metric
    fig, ax = plt.subplots(figsize=(18, 7))
    wer_means = []
    for l in labels:
        s = sc(data[l]["reports"])
        wer_vals = [c["metrics"]["privacy"]["wer"] for c in s]
        wer_means.append(np.mean(wer_vals) if wer_vals else 0)
    bars = ax.bar(x, wer_means, w, color=colors, alpha=0.85)
    for bar, m in zip(bars, wer_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{m:.3f}", ha="center", fontweight="bold", fontsize=12)
    _set_xt(ax, x, labels)
    ax.set_ylabel("Average WER"); ax.set_ylim(0, 1.2)
    ax.set_title("WER-Only Privacy Ranking\n(WER=1.0 = completely unintelligible speech)", fontweight="bold")
    ax.axhline(0.65, color="orange", ls="--", lw=1, alpha=0.7, label="Privacy threshold (WER≥0.65)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, borderaxespad=0); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "wer_only_ranking.png"), dpi=150); plt.close(fig)
    print("  ✓ wer_only_ranking.png")

    # ── 22. Semantic Preservation Metrics (TC@3, TA@1, PreserveScore) ──
    sem_data = {}
    for l in labels:
        s = sc(data[l]["reports"])
        tc3_scores, ta1_scores = [], []
        for c in s:
            orig = c.get("classification_top3_original", [])
            proc = c.get("classification_top3", [])
            if not orig or not proc:
                continue
            # TC@3: Top-3 Consistency
            set_a = set(p["label"] for p in orig[:3])
            set_b = set(p["label"] for p in proc[:3])
            tc3 = len(set_a & set_b) / 3.0
            tc3_scores.append(tc3)
            # TA@1: Top-1 Agreement
            ta1 = 1.0 if orig[0]["label"] == proc[0]["label"] else 0.0
            ta1_scores.append(ta1)
        if tc3_scores:
            preserve_scores = [(tc + ta) / 2.0 for tc, ta in zip(tc3_scores, ta1_scores)]
            sem_data[l] = {"tc3": tc3_scores, "ta1": ta1_scores, "preserve": preserve_scores}

    if sem_data:
        sem_labels = [l for l in labels if l in sem_data]
        sem_x = np.arange(len(sem_labels))
        sem_colors = [data[l]["color"] for l in sem_labels]

        # 22a. Grouped bar: TC@3, TA@1, PreserveScore
        metrics_sem = [("tc3", "TC@3"), ("ta1", "TA@1"), ("preserve", "Semantic\nPreserveScore")]
        bar_ws = 0.8 / len(metrics_sem)
        fig, ax = plt.subplots(figsize=(20, 7))
        for j, (key, mlabel) in enumerate(metrics_sem):
            means = [np.mean(sem_data[l][key]) for l in sem_labels]
            stds = [np.std(sem_data[l][key]) for l in sem_labels]
            bars = ax.bar(sem_x + j * bar_ws - (len(metrics_sem)-1)*bar_ws/2, means, bar_ws,
                          yerr=stds, capsize=3, label=mlabel, alpha=0.85)
            for bar, m, s in zip(bars, means, stds):
                ax.text(bar.get_x() + bar.get_width()/2,
                        min(bar.get_height() + 0.03, 1.18),
                        f"{m:.3f}±{s:.3f}", ha="center", va="bottom",
                        fontweight="bold", fontsize=24)
        ax.set_xticks(sem_x); ax.set_xticklabels(sem_labels)
        ax.set_ylabel("Score"); ax.set_ylim(0, 1.3)
        ax.set_title("Semantic Preservation Metrics\nTC@3 (Top-3 Consistency) | TA@1 (Top-1 Agreement) | Utility Score", fontweight="bold")
        ax.legend(loc="upper right"); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "semantic_preservation.png"), dpi=150); plt.close(fig)
        print("  ✓ semantic_preservation.png")

        # 22b. TC@3 distribution (histogram per version)
        fig, ax = plt.subplots(figsize=(18, 7))
        for l in sem_labels:
            ax.hist(sem_data[l]["tc3"], bins=[0, 0.33, 0.67, 1.01], alpha=0.5,
                    label=f"{l} (μ={np.mean(sem_data[l]['tc3']):.3f})", color=data[l]["color"], edgecolor="white")
        ax.set_xlabel("TC@3 Score"); ax.set_ylabel("Count")
        ax.set_xticks([0, 0.33, 0.67, 1.0]); ax.set_xticklabels(["0/3", "1/3", "2/3", "3/3"])
        ax.set_title("Top-3 Consistency Distribution\n(How many environmental labels preserved)", fontweight="bold")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, borderaxespad=0); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "tc3_distribution.png"), dpi=150); plt.close(fig)
        print("  ✓ tc3_distribution.png")

        # 22c. TA@1 rate (bar chart — % of chunks where top-1 class unchanged)
        fig, ax = plt.subplots(figsize=(18, 7))
        ta1_rates = [100 * np.mean(sem_data[l]["ta1"]) for l in sem_labels]
        bars = ax.bar(sem_x, ta1_rates, w, color=sem_colors, alpha=0.85)
        for bar, rate in zip(bars, ta1_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{rate:.1f}%", ha="center", fontweight="bold", fontsize=12)
        ax.set_xticks(sem_x); ax.set_xticklabels(sem_labels)
        ax.set_ylabel("TA@1 Rate (%)"); ax.set_ylim(0, 110)
        ax.set_title("Top-1 Agreement Rate\n(% chunks where dominant sound class unchanged after processing)", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "ta1_rate.png"), dpi=150); plt.close(fig)
        print("  ✓ ta1_rate.png")

        # 22d. Summary table for semantic metrics
        cols_sem = ["Version", "n", "TC@3\nMean±σ", "TA@1\nRate", "PreserveScore\nMean±σ"]
        rows_sem = []
        for l in sem_labels:
            n = len(sem_data[l]["tc3"])
            rows_sem.append([
                l, str(n),
                f"{np.mean(sem_data[l]['tc3']):.3f}±{np.std(sem_data[l]['tc3']):.3f}",
                f"{100*np.mean(sem_data[l]['ta1']):.1f}%",
                f"{np.mean(sem_data[l]['preserve']):.3f}±{np.std(sem_data[l]['preserve']):.3f}",
            ])
        fig, ax = plt.subplots(figsize=(16, 2 + 0.5 * len(rows_sem)))
        ax.axis("off")
        table = ax.table(cellText=rows_sem, colLabels=cols_sem, loc="center", cellLoc="center", colColours=["#E3F2FD"] * len(cols_sem))
        table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1, 1.8)
        ax.set_title("Semantic Preservation Summary", fontweight="bold", pad=20)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "semantic_preservation_summary.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
        print("  ✓ semantic_preservation_summary.png")

    # ── 23. Average Speaker Privacy ──
    means_sp = [np.mean([c["metrics"]["privacy"]["speaker_privacy"] for c in sc(data[l]["reports"])]) for l in labels]
    stds_sp = [np.std([c["metrics"]["privacy"]["speaker_privacy"] for c in sc(data[l]["reports"])]) for l in labels]
    fig, ax = plt.subplots(figsize=(18, 7))
    bars = ax.bar(x, means_sp, w, yerr=stds_sp, capsize=5, color=colors, alpha=0.85)
    for bar, m, s in zip(bars, means_sp, stds_sp):
        annotate_bar(ax, bar, f"{m:.3f}±{s:.3f}", 1.3)
    _set_xt(ax, x, labels)
    ax.set_ylabel("Speaker Privacy Score"); ax.set_ylim(0, 1.3)
    ax.set_title("Average Speaker Privacy (±σ)\n(1 - cosine similarity of speaker embeddings)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "avg_speaker_privacy.png"), dpi=150); plt.close(fig)
    print("  ✓ avg_speaker_privacy.png")

    # ── 24. Summary Table [v2: semantic preserve + TC@3 + TA@1] ──
    cols = ["Version", "Speech\nChunks", "Avg Privacy", "Avg Utility\n(TC@3+TA@1)/2", "TC@3\nMean", "TA@1\nRate", "Avg Trials", "Recipes"]
    rows = []
    for l in labels:
        s = sc(data[l]["reports"])
        recipes = {}
        for c in s:
            r = c.get("recipe_applied", {})
            if r:
                name = r.get("recipe_name", "?").replace("RECIPE_", "")
                recipes[name] = recipes.get(name, 0) + 1
        # Compute semantic metrics
        tc3_vals, ta1_vals = [], []
        for c in s:
            orig = c.get("classification_top3_original", [])
            proc = c.get("classification_top3", [])
            if orig and proc:
                set_a = set(p["label"] for p in orig[:3])
                set_b = set(p["label"] for p in proc[:3])
                tc3_vals.append(len(set_a & set_b) / 3.0)
                ta1_vals.append(1.0 if orig[0]["label"] == proc[0]["label"] else 0.0)
        ps_list = preserve_list(s)
        rows.append([
            l,
            str(len(s)),
            f"{np.mean([c['metrics']['privacy']['privacy_score'] for c in s]):.4f}" if s else "N/A",
            f"{np.mean(ps_list):.4f}" if ps_list else "N/A",
            f"{np.mean(tc3_vals):.4f}" if tc3_vals else "N/A",
            f"{100*np.mean(ta1_vals):.1f}%" if ta1_vals else "N/A",
            f"{np.mean([c.get('trials', 0) for c in s]):.2f}" if s else "N/A",
            ", ".join(f"{k}:{v}" for k, v in sorted(recipes.items())),
        ])
    fig, ax = plt.subplots(figsize=(20, 2 + 0.6 * len(rows)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center", colColours=["#E3F2FD"] * len(cols))
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.8)
    ax.set_title("Summary [Semantic Preserve Metric]", fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "summary_table.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
    print("  ✓ summary_table.png")

    # ─────────────────────────────────────────────────────────────────────
    # [v2 PATCH] PAPER-READY OUTPUTS
    #   1. Multi-threshold acceptance grid (Table III in paper)
    #   2. CSV exports of per-chunk metrics + summary
    #   3. Console print of all paper tables in copy-paste form
    # ─────────────────────────────────────────────────────────────────────

    import csv

    # Threshold grids for paper
    T_P_GRID = [0.50, 0.55, 0.60, 0.65, 0.70]
    T_U_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    T_S_DEFAULT = 0.50

    # Compute per-chunk paper metrics
    paper_chunks = {}  # cfg -> list of dicts
    for l in labels:
        s = sc(data[l]["reports"])
        rows_pchunk = []
        for c in s:
            ps = semantic_preserve_score(c)
            if ps is None:
                continue
            priv = c["metrics"]["privacy"].get("privacy_score")
            spk = c["metrics"]["privacy"].get("speaker_privacy")
            wer = c["metrics"]["privacy"].get("wer")
            cer = c["metrics"]["privacy"].get("cer")
            cp = c["metrics"]["privacy"].get("content_privacy")
            orig = c.get("classification_top3_original", [])
            proc = c.get("classification_top3", [])
            set_a = set(p["label"] for p in orig[:3])
            set_b = set(p["label"] for p in proc[:3])
            tc3 = len(set_a & set_b) / 3.0
            ta1 = 1.0 if orig[0]["label"] == proc[0]["label"] else 0.0
            rows_pchunk.append({
                "configuration": l,
                "source_id": c.get("source_id", ""),
                "chunk_id": c.get("chunk_id", ""),
                "speech_ratio": c.get("speech_ratio"),
                "vad_confidence": c.get("vad_confidence"),
                "wer": wer, "cer": cer, "content_privacy": cp,
                "speaker_privacy": spk, "privacy_score": priv,
                "tc3": tc3, "ta1": ta1, "preserve_new": ps,
                "trials_used": c.get("trials"),
                "used_ss": c.get("used_source_separation", False),
                "recipe": (c.get("recipe_applied") or {}).get("recipe_name"),
            })
        paper_chunks[l] = rows_pchunk

    # ── A. Per-chunk CSV ──
    pchunk_path = os.path.join(OUT_DIR, "per_chunk_metrics_paper.csv")
    with open(pchunk_path, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["configuration","source_id","chunk_id","speech_ratio","vad_confidence",
                       "wer","cer","content_privacy","speaker_privacy","privacy_score",
                       "tc3","ta1","preserve_new","trials_used","used_ss","recipe"])
        for l in labels:
            for r in paper_chunks[l]:
                wcsv.writerow([r["configuration"], r["source_id"], r["chunk_id"],
                               r["speech_ratio"], r["vad_confidence"],
                               r["wer"], r["cer"], r["content_privacy"],
                               r["speaker_privacy"], r["privacy_score"],
                               r["tc3"], r["ta1"], r["preserve_new"],
                               r["trials_used"], r["used_ss"], r["recipe"]])
    print(f"  ✓ per_chunk_metrics_paper.csv ({sum(len(paper_chunks[l]) for l in labels)} rows)")

    # ── B. Summary CSV ──
    summary_path = os.path.join(OUT_DIR, "summary_table_paper.csv")
    with open(summary_path, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["Configuration","n_chunks",
                       "PrivacyScore_mean","PrivacyScore_std",
                       "PreserveScore_mean","PreserveScore_std",
                       "TC@3_mean","TC@3_std",
                       "TA@1_rate",
                       "SpeakerPrivacy_mean","SpeakerPrivacy_std",
                       "WER_mean","CER_mean",
                       "AvgTrials"])
        for l in labels:
            rs = paper_chunks[l]
            if not rs:
                continue
            def _mean(xs):
                xs = [x for x in xs if x is not None]
                return sum(xs)/len(xs) if xs else 0.0
            def _std(xs):
                xs = [x for x in xs if x is not None]
                if len(xs) < 2: return 0.0
                m = _mean(xs)
                return (sum((x-m)**2 for x in xs)/(len(xs)-1))**0.5
            wcsv.writerow([
                l, len(rs),
                f"{_mean([r['privacy_score'] for r in rs]):.4f}", f"{_std([r['privacy_score'] for r in rs]):.4f}",
                f"{_mean([r['preserve_new'] for r in rs]):.4f}", f"{_std([r['preserve_new'] for r in rs]):.4f}",
                f"{_mean([r['tc3'] for r in rs]):.4f}", f"{_std([r['tc3'] for r in rs]):.4f}",
                f"{_mean([r['ta1'] for r in rs]):.4f}",
                f"{_mean([r['speaker_privacy'] for r in rs]):.4f}", f"{_std([r['speaker_privacy'] for r in rs]):.4f}",
                f"{_mean([r['wer'] for r in rs]):.4f}",
                f"{_mean([r['cer'] for r in rs]):.4f}",
                f"{_mean([r['trials_used'] for r in rs]):.2f}",
            ])
    print(f"  ✓ summary_table_paper.csv")

    # ── C. Multi-threshold acceptance grid CSV ──
    grid_path = os.path.join(OUT_DIR, "multi_threshold_acceptance_paper.csv")
    with open(grid_path, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["Configuration","T_p","T_u","T_s",
                       "AcceptanceRate","PrivacyPass","UtilityPass","SpeakerPass","n_chunks"])
        for l in labels:
            rs = paper_chunks[l]
            n = len(rs)
            if n == 0:
                continue
            for tp in T_P_GRID:
                for tu in T_U_GRID:
                    p_pass = sum(1 for r in rs if r["privacy_score"] is not None and r["privacy_score"] >= tp) / n
                    u_pass = sum(1 for r in rs if r["preserve_new"] is not None and r["preserve_new"] >= tu) / n
                    s_pass = sum(1 for r in rs if r["speaker_privacy"] is not None and r["speaker_privacy"] >= T_S_DEFAULT) / n
                    accept = sum(1 for r in rs
                                 if r["privacy_score"] is not None and r["privacy_score"] >= tp
                                 and r["preserve_new"] is not None and r["preserve_new"] >= tu
                                 and r["speaker_privacy"] is not None and r["speaker_privacy"] >= T_S_DEFAULT) / n
                    wcsv.writerow([l, f"{tp:.2f}", f"{tu:.2f}", f"{T_S_DEFAULT:.2f}",
                                   f"{accept:.4f}", f"{p_pass:.4f}", f"{u_pass:.4f}", f"{s_pass:.4f}", n])
    print(f"  ✓ multi_threshold_acceptance_paper.csv")

    # ── C2. Acceptance Rate Bar Chart at design spec (T_p=0.65, T_u=0.80, T_s=0.50) ──
    T_P_FIXED, T_U_FIXED, T_S_FIXED = 0.65, 0.80, 0.50
    ar_p, ar_u, ar_s, ar_joint = [], [], [], []
    ar_labels = []
    for l in labels:
        rs = paper_chunks[l]
        n = len(rs)
        if n == 0:
            continue
        p_pass = sum(1 for r in rs if r["privacy_score"] is not None and r["privacy_score"] >= T_P_FIXED) / n
        u_pass = sum(1 for r in rs if r["preserve_new"] is not None and r["preserve_new"] >= T_U_FIXED) / n
        s_pass = sum(1 for r in rs if r["speaker_privacy"] is not None and r["speaker_privacy"] >= T_S_FIXED) / n
        joint = sum(1 for r in rs
                    if r["privacy_score"] is not None and r["privacy_score"] >= T_P_FIXED
                    and r["preserve_new"] is not None and r["preserve_new"] >= T_U_FIXED
                    and r["speaker_privacy"] is not None and r["speaker_privacy"] >= T_S_FIXED) / n
        ar_labels.append(l)
        ar_p.append(p_pass); ar_u.append(u_pass); ar_s.append(s_pass); ar_joint.append(joint)

    if ar_labels:
        fig, ax = plt.subplots(figsize=(16, 8))
        ar_x = np.arange(len(ar_labels))
        bw = 0.20
        b1 = ax.bar(ar_x - 1.5*bw, ar_p,     bw, label=f"P-pass (≥{T_P_FIXED})", color="#42A5F5", alpha=0.9)
        b2 = ax.bar(ar_x - 0.5*bw, ar_u,     bw, label=f"U-pass (≥{T_U_FIXED})", color="#66BB6A", alpha=0.9)
        b3 = ax.bar(ar_x + 0.5*bw, ar_s,     bw, label=f"S-pass (≥{T_S_FIXED})", color="#FFA726", alpha=0.9)
        b4 = ax.bar(ar_x + 1.5*bw, ar_joint, bw, label="Joint Acceptance",       color="#AB47BC", alpha=0.95)
        for bars in (b1, b2, b3, b4):
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                        f"{h*100:.1f}%", ha="center", va="bottom", fontsize=18)
        ax.set_xticks(ar_x)
        ax.set_xticklabels(ar_labels, rotation=15, ha="right")
        ax.set_ylabel("Pass Rate")
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
        ax.set_title(f"Acceptance Rate  "
                     f"(T_p={T_P_FIXED}, T_u={T_U_FIXED}, T_s={T_S_FIXED})",
                     fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper right", framealpha=0.95)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "acceptance_rate.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  ✓ acceptance_rate.png")

    # ── D. Console summary (paper-ready text tables) ──
    print("\n" + "=" * 90)
    print("PAPER TABLE II — Summary (semantic preserve metric)")
    print("=" * 90)
    print(f"{'Configuration':<22} {'Privacy':>10} {'Preserve':>10} {'TC@3':>8} {'TA@1':>8} {'SpkPriv':>8} {'WER':>8}")
    print("-" * 90)
    for l in labels:
        rs = paper_chunks[l]
        if not rs:
            continue
        def _mean(xs):
            xs = [x for x in xs if x is not None]
            return sum(xs)/len(xs) if xs else 0.0
        print(f"{l:<22} "
              f"{_mean([r['privacy_score'] for r in rs]):>10.4f} "
              f"{_mean([r['preserve_new'] for r in rs]):>10.4f} "
              f"{_mean([r['tc3'] for r in rs]):>8.4f} "
              f"{_mean([r['ta1'] for r in rs]):>8.2%} "
              f"{_mean([r['speaker_privacy'] for r in rs]):>8.4f} "
              f"{_mean([r['wer'] for r in rs]):>8.4f}")

    print("\n" + "=" * 90)
    print("PAPER TABLE III — Acceptance rates (T_s = 0.50)")
    print("=" * 90)
    key_cells = [(0.50,0.55),(0.55,0.65),(0.60,0.55),(0.60,0.70),(0.65,0.55),(0.65,0.70),(0.65,0.80),(0.70,0.80)]
    print(f"{'Configuration':<22} | " + " | ".join(f"({tp:.2f},{tu:.2f})" for tp,tu in key_cells))
    print("-" * (22 + len(key_cells) * 13))
    for l in labels:
        rs = paper_chunks[l]
        n = len(rs)
        if n == 0:
            continue
        line = f"{l:<22} | "
        for tp, tu in key_cells:
            accept = sum(1 for r in rs
                         if r["privacy_score"] is not None and r["privacy_score"] >= tp
                         and r["preserve_new"] is not None and r["preserve_new"] >= tu
                         and r["speaker_privacy"] is not None and r["speaker_privacy"] >= T_S_DEFAULT) / n
            line += f"  {accept:>5.1%}    | "
        print(line)

    print("\n" + "=" * 90)
    print("MARGINAL PASS RATES at design spec (T_p=0.65, T_u=0.80, T_s=0.50)")
    print("=" * 90)
    for l in labels:
        rs = paper_chunks[l]
        n = len(rs)
        if n == 0:
            continue
        p_pass = sum(1 for r in rs if r["privacy_score"] is not None and r["privacy_score"] >= 0.65) / n
        u_pass = sum(1 for r in rs if r["preserve_new"] is not None and r["preserve_new"] >= 0.80) / n
        s_pass = sum(1 for r in rs if r["speaker_privacy"] is not None and r["speaker_privacy"] >= 0.50) / n
        print(f"{l:<22}  P_pass={p_pass:>6.2%}  U_pass={u_pass:>6.2%}  S_pass={s_pass:>6.2%}")

    print("\n" + "=" * 90)
    print("FILES TO HAND TO PAPER")
    print("=" * 90)
    print(f"  Numbers : {summary_path}")
    print(f"            {grid_path}")
    print(f"            {pchunk_path}")
    print(f"  Figures : {OUT_DIR}/privacy_utility_tradeoff.png")
    print(f"            {OUT_DIR}/semantic_preservation.png")
    print(f"            {OUT_DIR}/classification_label_change.png")
    print(f"            {OUT_DIR}/speech_ratio_vs_privacy.png")
    print(f"            {OUT_DIR}/summary_table.png")
    print()

    total = sum(1 for f in os.listdir(OUT_DIR) if f.endswith(".png"))
    print(f"\nDone — {total} charts saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
