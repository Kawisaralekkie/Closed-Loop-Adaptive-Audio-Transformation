#!/usr/bin/env python3
"""Plot comparison across LLM pipeline variants.

Usage:
    python3 scripts/plot_comparison_llm.py
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
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

# ── Config ──
VERSIONS = {
    "LLM-H4.5-Mem": {
        "path": "logs/s3/20260429_093223/llm_with_memory_no_ss",
        "color": "#2196F3",
    },
    "LLM-H4.5-NoMem-SS": {
        "path": "logs/s3/20260429_093355/llm_no_memory",
        "color": "#0D47A1",
    },
    "LLM-H4.5-Mem-SS": {
        "path": "logs/s3/20260429_093321/llm_with_memory",
        "color": "#4CAF50",
    },
    "Rule-NoMem-SS": {
        "path": "logs/s3/20260429_093019/rule_based_ss",
        "color": "#F8BBD0",  # pastel pink
    },
}

OUT_DIR = "plots/comparison_llm"


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


def annotate_bar(ax, bar, text, ylim_top):
    """Put text above bar, clamped below title."""
    y = min(bar.get_height() + 0.03, ylim_top - 0.12)
    ax.text(bar.get_x() + bar.get_width()/2, y, text,
            ha="center", va="bottom", fontweight="bold", fontsize=11)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load data
    data = {}
    for label, cfg in VERSIONS.items():
        reports = load(cfg["path"])
        if reports:
            data[label] = {"reports": reports, "color": cfg["color"]}
            print(f"  {label}: {len(reports)} reports, {len(sc(reports))} speech chunks")
        else:
            print(f"  WARNING: {label} — no data at {cfg['path']}")

    if not data:
        print("ERROR: No data"); sys.exit(1)

    labels = list(data.keys())
    colors = [data[l]["color"] for l in labels]
    x = np.arange(len(labels))
    w = 0.6

    print(f"\nPlotting to {OUT_DIR}/\n")

    # ── 1. Average Privacy Score ──
    means = [np.mean([c["metrics"]["privacy"]["privacy_score"] for c in sc(data[l]["reports"])]) for l in labels]
    stds = [np.std([c["metrics"]["privacy"]["privacy_score"] for c in sc(data[l]["reports"])]) for l in labels]
    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.bar(x, means, w, yerr=stds, capsize=5, color=colors, alpha=0.85)
    for bar, m, s in zip(bars, means, stds):
        annotate_bar(ax, bar, f"{m:.3f}±{s:.3f}", 1.3)
    ax.axhline(0.65, color="orange", ls="--", lw=1, alpha=0.7, label="High (0.65)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Privacy Score"); ax.set_ylim(0, 1.3)
    ax.set_title("Average Privacy Score (±σ)", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "avg_privacy_score.png"), dpi=150); plt.close(fig)
    print("  ✓ avg_privacy_score.png")

    # ── 2. Average Utility Score ──
    means = [np.mean([c["metrics"]["utility"]["preserve_score"] for c in sc(data[l]["reports"])]) for l in labels]
    stds = [np.std([c["metrics"]["utility"]["preserve_score"] for c in sc(data[l]["reports"])]) for l in labels]
    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.bar(x, means, w, yerr=stds, capsize=5, color=colors, alpha=0.85)
    for bar, m, s in zip(bars, means, stds):
        annotate_bar(ax, bar, f"{m:.4f}±{s:.4f}", 1.1)
    ax.axhline(0.80, color="red", ls="--", lw=1, alpha=0.7, label="Threshold (0.80)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Utility Score"); ax.set_ylim(0.6, 1.1)
    ax.set_title("Average Utility Score (±σ)", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "avg_utility_score.png"), dpi=150); plt.close(fig)
    print("  ✓ avg_utility_score.png")

    # ── 3. Privacy vs Utility Trade-off ──
    fig, ax = plt.subplots(figsize=(10, 8))
    for l in labels:
        s = sc(data[l]["reports"])
        priv = [c["metrics"]["privacy"]["privacy_score"] for c in s]
        util = [c["metrics"]["utility"]["preserve_score"] for c in s]
        ax.scatter(priv, util, alpha=0.4, s=25, label=l, color=data[l]["color"])
    ax.axvline(0.65, color="orange", ls="--", lw=1, alpha=0.5)
    ax.axhline(0.80, color="red", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("Privacy Score"); ax.set_ylabel("Utility Score")
    ax.set_title("Privacy vs Utility Trade-off", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "privacy_utility_tradeoff.png"), dpi=150); plt.close(fig)
    print("  ✓ privacy_utility_tradeoff.png")

    # ── 4. Classification Metrics ──
    metric_names = ["mAP", "accuracy", "f1"]
    metric_labels_display = ["mAP", "Accuracy", "F1-Score"]
    metric_colors = ["#1976D2", "#F57C00", "#388E3C"]
    bar_w = 0.8 / len(metric_names)
    fig, ax = plt.subplots(figsize=(14, 7))
    for j, (mname, mlabel, mcolor) in enumerate(zip(metric_names, metric_labels_display, metric_colors)):
        vals = []
        for l in labels:
            s = sc(data[l]["reports"])
            scores = [c["metrics"]["utility"][mname] for c in s if mname in c["metrics"]["utility"]]
            vals.append(np.mean(scores) if scores else 0)
        ax.bar(x + j * bar_w - (len(metric_names)-1)*bar_w/2, vals, bar_w,
               label=mlabel, color=mcolor, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.2)
    ax.set_title("Classification Metrics (mAP, Accuracy, F1)", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "classification_metrics.png"), dpi=150); plt.close(fig)
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
    fig, ax = plt.subplots(figsize=(14, 7))
    bottom = np.zeros(len(labels))
    for recipe in recipes:
        vals = [ver_recipes[l].get(recipe, 0) for l in labels]
        ax.bar(x, vals, 0.5, bottom=bottom, label=recipe, color=recipe_colors.get(recipe, "#999"), alpha=0.85)
        bottom += np.array(vals)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Chunks"); ax.set_title("Recipe Usage (incl. Source Separation pre-processing)", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "recipe_usage.png"), dpi=150); plt.close(fig)
    print("  ✓ recipe_usage.png")

    # ── 6. Trials Distribution ──
    fig, ax = plt.subplots(figsize=(14, 7))
    for i, l in enumerate(labels):
        s = sc(data[l]["reports"])
        trials = [c.get("trials", 0) for c in s if c.get("trials", 0) > 0]
        avg = np.mean(trials) if trials else 0
        bar = ax.bar(i, avg, color=colors[i], alpha=0.85)
        ax.text(i, avg + 0.05, f"{avg:.2f}", ha="center", fontweight="bold")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_ylabel("Avg Trials"); ax.set_title("Average Trials per Chunk", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "trials_distribution.png"), dpi=150); plt.close(fig)
    print("  ✓ trials_distribution.png")

    # ── 7. Speech Ratio vs Privacy (colored by SS usage) ──
    fig, ax = plt.subplots(figsize=(12, 7))
    for l in labels:
        s = sc(data[l]["reports"])
        # Split by source separation usage
        ss_chunks = [c for c in s if c.get("used_source_separation")]
        non_ss_chunks = [c for c in s if not c.get("used_source_separation")]
        # Plot non-SS
        if non_ss_chunks:
            ratios = [c.get("speech_ratio", 0) for c in non_ss_chunks]
            priv = [c["metrics"]["privacy"]["privacy_score"] for c in non_ss_chunks]
            ax.scatter(ratios, priv, alpha=0.4, s=25, label=l, color=data[l]["color"], marker="o")
        # Plot SS with different marker
        if ss_chunks:
            ratios = [c.get("speech_ratio", 0) for c in ss_chunks]
            priv = [c["metrics"]["privacy"]["privacy_score"] for c in ss_chunks]
            ax.scatter(ratios, priv, alpha=0.7, s=60, label=f"{l} (SS)", color=data[l]["color"], marker="*", edgecolors="black", linewidths=0.5)
    ax.axhline(0.65, color="orange", ls="--", lw=1, alpha=0.7)
    ax.axvline(0.3, color="green", ls=":", lw=1, alpha=0.5, label="SS threshold (SR≥0.3)")
    ax.set_xlabel("Speech Ratio"); ax.set_ylabel("Privacy Score")
    ax.set_title("Speech Ratio vs Privacy Score\n(★ = Source Separation applied)", fontweight="bold")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "speech_ratio_vs_privacy.png"), dpi=150); plt.close(fig)
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
        fig, ax = plt.subplots(figsize=(14, 7))
        bars = ax.bar(tk_x, means, 0.5, yerr=stds, capsize=5, color=tk_colors, alpha=0.85)
        for bar, m, s in zip(bars, means, stds):
            annotate_bar(ax, bar, f"{m:.3f}±{s:.3f}", 1.4)
        ax.set_xticks(tk_x); ax.set_xticklabels(tk_labels)
        ax.set_ylabel("Consistency (0-1)"); ax.set_ylim(0, 1.4)
        ax.set_title("Top-3 Class Consistency\n(Environmental labels preserved after processing)", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "topk_class_consistency.png"), dpi=150); plt.close(fig)
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
        fig, ax = plt.subplots(figsize=(14, 7))
        pct = [100 * ch / max(t, 1) for _, ch, t in changed_counts]
        bars = ax.bar(x, pct, w, color=colors, alpha=0.85)
        for bar, (lbl, ch, t) in zip(bars, changed_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{ch}/{t} ({bar.get_height():.1f}%)", ha="center", fontweight="bold", fontsize=11)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("% Chunks with Changed Top-1 Label")
        ax.set_title("Classification Label Change After Processing", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "classification_label_change.png"), dpi=150); plt.close(fig)
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
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "classification_top_labels.png"), dpi=150); plt.close(fig)
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
        fig, ax = plt.subplots(figsize=(14, 7))
        bars = ax.bar(cd_x, means, 0.5, yerr=stds, capsize=5, color=cd_colors, alpha=0.85)
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.02,
                    f"{m:+.3f}±{s:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(cd_x); ax.set_xticklabels(cd_labels)
        ax.set_ylabel("Confidence Drop (positive = reduced)")
        ax.set_title("Speech Confidence Gap (Original − Processed)\n(Higher = better privacy)", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "speech_confidence_gap.png"), dpi=150); plt.close(fig)
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
        fig, ax = plt.subplots(figsize=(14, 7))
        bars = ax.bar(rd_x, means, 0.5, yerr=stds, capsize=5, color=rd_colors, alpha=0.85)
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, max(bar.get_height(), 0) + s + 0.1,
                    f"{m:+.2f}±{s:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(rd_x); ax.set_xticklabels(rd_labels)
        ax.set_ylabel("Rank Drop (positive = dropped)")
        ax.set_title("Speech Rank Drop After Processing\n(Higher = better privacy)", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "speech_rank_drop.png"), dpi=150); plt.close(fig)
        print("  ✓ speech_rank_drop.png")

    # ── 13. No-Speech Analysis ──
    ns_labels, ns_means, ns_stds, ns_counts = [], [], [], []
    for l in labels:
        ns = no_sc(data[l]["reports"])
        if ns:
            ns_labels.append(l)
            scores = [c["metrics"]["utility"]["preserve_score"] for c in ns]
            ns_means.append(np.mean(scores))
            ns_stds.append(np.std(scores))
            ns_counts.append(len(ns))

    if ns_labels:
        ns_x = np.arange(len(ns_labels))
        ns_colors = [data[l]["color"] for l in ns_labels]
        fig, ax = plt.subplots(figsize=(14, 7))
        bars = ax.bar(ns_x, ns_means, 0.5, yerr=ns_stds, capsize=5, color=ns_colors, alpha=0.85)
        for bar, m, s, n in zip(bars, ns_means, ns_stds, ns_counts):
            ax.text(bar.get_x() + bar.get_width()/2, min(bar.get_height() + s + 0.005, 1.05),
                    f"{m:.4f}±{s:.4f} (n={n})", ha="center", va="bottom", fontweight="bold", fontsize=10)
        ax.axhline(0.80, color="red", ls="--", lw=1, alpha=0.7)
        ax.set_xticks(ns_x); ax.set_xticklabels(ns_labels)
        ax.set_ylabel("Utility Score"); ax.set_ylim(0.6, 1.1)
        ax.set_title("Utility Score — No-Speech Chunks (Bypass)", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "no_speech_utility.png"), dpi=150); plt.close(fig)
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
    fig, ax = plt.subplots(figsize=(14, 7))
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
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Number of Chunks")
    ax.set_title("Source Separation Usage (Speech Chunks)", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "source_sep_usage.png"), dpi=150); plt.close(fig)
    print("  ✓ source_sep_usage.png")

    # 14b. Privacy score: SS vs non-SS
    fig, ax = plt.subplots(figsize=(14, 7))
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
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "source_sep_privacy.png"), dpi=150); plt.close(fig)
        print("  ✓ source_sep_privacy.png")

    # 14c. All metrics comparison: SS vs non-SS
    metric_keys = [("privacy_score", "Privacy"), ("preserve_score", "Utility"), ("mAP", "mAP"), ("accuracy", "Accuracy"), ("f1", "F1")]
    rows_ss = []
    for l in labels:
        ss = ss_analysis[l]["ss"]
        non_ss = ss_analysis[l]["non_ss"]
        row = [l, str(len(ss)), str(len(non_ss))]
        for key, _ in metric_keys:
            if key in ("privacy_score",):
                ss_val = np.mean([c["metrics"]["privacy"][key] for c in ss]) if ss else 0
                non_val = np.mean([c["metrics"]["privacy"][key] for c in non_ss]) if non_ss else 0
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

    # ── 15. Speech Density Filtered Analysis ──
    fig, ax = plt.subplots(figsize=(14, 7))
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
    ax.legend(handles=legend_elements)
    xtick_pos = [j * (len(labels) + 1) + len(labels)*0.2 for j in range(len(metric_keys_density))]
    ax.set_xticks(xtick_pos); ax.set_xticklabels([m[2] for m in metric_keys_density])
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.2)
    ax.set_title("Impact of Speech Density: All vs High-Density Chunks (SR≥0.5)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "speech_density_analysis.png"), dpi=150); plt.close(fig)
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
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "wer_vs_privacy.png"), dpi=150); plt.close(fig)
    print("  ✓ wer_vs_privacy.png")

    # ── 17. WER Distribution by Recipe Level ──
    fig, ax = plt.subplots(figsize=(14, 7))
    for l in labels:
        s = sc(data[l]["reports"])
        wer_vals = [c["metrics"]["privacy"]["wer"] for c in s]
        if wer_vals:
            ax.hist(wer_vals, bins=20, alpha=0.4, label=l, color=data[l]["color"], edgecolor="white")
    ax.set_xlabel("WER"); ax.set_ylabel("Count")
    ax.set_title("WER Distribution (Content Privacy)\n(WER=1.0 means completely unintelligible)", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "wer_distribution.png"), dpi=150); plt.close(fig)
    print("  ✓ wer_distribution.png")

    # ── 18. Utility by Sound Class (Box Plot) ──
    # Use classification_top3_original to group by sound type
    class_utility = {}
    for l in labels:
        s = sc(data[l]["reports"])
        for c in s:
            top3_orig = c.get("classification_top3_original", [])
            if top3_orig:
                sound_class = top3_orig[0]["label"]
                class_utility.setdefault(sound_class, []).append(c["metrics"]["utility"]["preserve_score"])

    if class_utility:
        # Top 10 most common classes
        sorted_classes = sorted(class_utility.items(), key=lambda x: -len(x[1]))[:10]
        class_names = [c[0][:20] for c in sorted_classes]
        class_scores = [c[1] for c in sorted_classes]
        fig, ax = plt.subplots(figsize=(16, 7))
        bp = ax.boxplot(class_scores, labels=class_names, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#90CAF9"); patch.set_alpha(0.7)
        ax.axhline(0.80, color="red", ls="--", lw=1, alpha=0.7, label="Threshold (0.80)")
        ax.set_ylabel("Utility Score")
        ax.set_title("Utility Score by Original Sound Class\n(Environmental sound preservation across categories)", fontweight="bold")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "utility_by_sound_class.png"), dpi=150); plt.close(fig)
        print("  ✓ utility_by_sound_class.png")

    # ── 19. Per-Class Analysis Table ──
    # Privacy and Utility broken down by original sound class
    class_metrics = {}
    for l in labels:
        s = sc(data[l]["reports"])
        for c in s:
            top3_orig = c.get("classification_top3_original", [])
            if top3_orig:
                cls = top3_orig[0]["label"]
                class_metrics.setdefault(cls, {}).setdefault(l, []).append({
                    "privacy": c["metrics"]["privacy"]["privacy_score"],
                    "utility": c["metrics"]["utility"]["preserve_score"],
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
            ax.legend(); ax.grid(alpha=0.3)
            fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "paired_comparison_rule_vs_llm.png"), dpi=150); plt.close(fig)
            print("  ✓ paired_comparison_rule_vs_llm.png")

    # ── 21. WER-Only Privacy Ranking ──
    # Compare methods using WER alone as privacy metric
    fig, ax = plt.subplots(figsize=(14, 7))
    wer_means = []
    for l in labels:
        s = sc(data[l]["reports"])
        wer_vals = [c["metrics"]["privacy"]["wer"] for c in s]
        wer_means.append(np.mean(wer_vals) if wer_vals else 0)
    bars = ax.bar(x, wer_means, w, color=colors, alpha=0.85)
    for bar, m in zip(bars, wer_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{m:.3f}", ha="center", fontweight="bold", fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Average WER"); ax.set_ylim(0, 1.2)
    ax.set_title("WER-Only Privacy Ranking\n(WER=1.0 = completely unintelligible speech)", fontweight="bold")
    ax.axhline(0.65, color="orange", ls="--", lw=1, alpha=0.7, label="Privacy threshold (WER≥0.65)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "wer_only_ranking.png"), dpi=150); plt.close(fig)
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
        fig, ax = plt.subplots(figsize=(16, 7))
        for j, (key, mlabel) in enumerate(metrics_sem):
            means = [np.mean(sem_data[l][key]) for l in sem_labels]
            stds = [np.std(sem_data[l][key]) for l in sem_labels]
            bars = ax.bar(sem_x + j * bar_ws - (len(metrics_sem)-1)*bar_ws/2, means, bar_ws,
                          yerr=stds, capsize=3, label=mlabel, alpha=0.85)
            for bar, m, s in zip(bars, means, stds):
                annotate_bar(ax, bar, f"{m:.3f}±{s:.3f}", 1.3)
        ax.set_xticks(sem_x); ax.set_xticklabels(sem_labels)
        ax.set_ylabel("Score"); ax.set_ylim(0, 1.3)
        ax.set_title("Semantic Preservation Metrics\nTC@3 (Top-3 Consistency) | TA@1 (Top-1 Agreement) | Utility Score", fontweight="bold")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "semantic_preservation.png"), dpi=150); plt.close(fig)
        print("  ✓ semantic_preservation.png")

        # 22b. TC@3 distribution (histogram per version)
        fig, ax = plt.subplots(figsize=(14, 7))
        for l in sem_labels:
            ax.hist(sem_data[l]["tc3"], bins=[0, 0.33, 0.67, 1.01], alpha=0.5,
                    label=f"{l} (μ={np.mean(sem_data[l]['tc3']):.3f})", color=data[l]["color"], edgecolor="white")
        ax.set_xlabel("TC@3 Score"); ax.set_ylabel("Count")
        ax.set_xticks([0, 0.33, 0.67, 1.0]); ax.set_xticklabels(["0/3", "1/3", "2/3", "3/3"])
        ax.set_title("Top-3 Consistency Distribution\n(How many environmental labels preserved)", fontweight="bold")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "tc3_distribution.png"), dpi=150); plt.close(fig)
        print("  ✓ tc3_distribution.png")

        # 22c. TA@1 rate (bar chart — % of chunks where top-1 class unchanged)
        fig, ax = plt.subplots(figsize=(14, 7))
        ta1_rates = [100 * np.mean(sem_data[l]["ta1"]) for l in sem_labels]
        bars = ax.bar(sem_x, ta1_rates, w, color=sem_colors, alpha=0.85)
        for bar, rate in zip(bars, ta1_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{rate:.1f}%", ha="center", fontweight="bold", fontsize=12)
        ax.set_xticks(sem_x); ax.set_xticklabels(sem_labels)
        ax.set_ylabel("TA@1 Rate (%)"); ax.set_ylim(0, 110)
        ax.set_title("Top-1 Agreement Rate\n(% chunks where dominant sound class unchanged after processing)", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "ta1_rate.png"), dpi=150); plt.close(fig)
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
    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.bar(x, means_sp, w, yerr=stds_sp, capsize=5, color=colors, alpha=0.85)
    for bar, m, s in zip(bars, means_sp, stds_sp):
        annotate_bar(ax, bar, f"{m:.3f}±{s:.3f}", 1.3)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Speaker Privacy Score"); ax.set_ylim(0, 1.3)
    ax.set_title("Average Speaker Privacy (±σ)\n(1 - cosine similarity of speaker embeddings)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "avg_speaker_privacy.png"), dpi=150); plt.close(fig)
    print("  ✓ avg_speaker_privacy.png")

    # ── 24. Summary Table ──
    cols = ["Version", "Speech\nChunks", "Avg Privacy", "Avg Utility", "Avg Trials", "Recipes"]
    rows = []
    for l in labels:
        s = sc(data[l]["reports"])
        recipes = {}
        for c in s:
            r = c.get("recipe_applied", {})
            if r:
                name = r.get("recipe_name", "?").replace("RECIPE_", "")
                recipes[name] = recipes.get(name, 0) + 1
        rows.append([
            l,
            str(len(s)),
            f"{np.mean([c['metrics']['privacy']['privacy_score'] for c in s]):.4f}" if s else "N/A",
            f"{np.mean([c['metrics']['utility']['preserve_score'] for c in s]):.4f}" if s else "N/A",
            f"{np.mean([c.get('trials', 0) for c in s]):.2f}" if s else "N/A",
            ", ".join(f"{k}:{v}" for k, v in sorted(recipes.items())),
        ])
    fig, ax = plt.subplots(figsize=(18, 2 + 0.6 * len(rows)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center", colColours=["#E3F2FD"] * len(cols))
    table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1, 1.8)
    ax.set_title("Summary", fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "summary_table.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
    print("  ✓ summary_table.png")

    total = sum(1 for f in os.listdir(OUT_DIR) if f.endswith(".png"))
    print(f"\nDone — {total} charts saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
