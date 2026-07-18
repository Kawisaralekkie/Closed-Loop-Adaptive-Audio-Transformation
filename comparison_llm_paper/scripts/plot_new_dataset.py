#!/usr/bin/env python3
"""Plot charts for the new dataset (cityspeechmixed).

Usage:
    # After downloading logs:
    python3 scripts/download_logs.py --source s3
    
    # Then plot (auto-detects latest download folder):
    python3 scripts/plot_new_dataset.py
    
    # Or specify folder:
    python3 scripts/plot_new_dataset.py --log-dir logs/s3/20260419_164025
    
    # Custom output:
    python3 scripts/plot_new_dataset.py --out-dir plots/cityspeechmixed
"""

from __future__ import annotations
import json, glob, os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

PIPE_ADAPTIVE = "Adaptive Policy"
PIPE_FIXED = "Fixed Policy"


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
    """No-speech chunks with metrics (bypass)."""
    return [c for r in reports for c in r.get("chunks", [])
            if not c.get("had_speech") and c.get("metrics")]


def find_latest_log_dir():
    dirs = sorted(glob.glob("logs/s3/2026*"))
    return dirs[-1] if dirs else None


def main():
    parser = argparse.ArgumentParser(description="Plot charts for new dataset")
    parser.add_argument("--log-dir", default=None, help="Log directory (default: latest in logs/s3/)")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: plots/<log_dir_name>)")
    args = parser.parse_args()

    log_dir = args.log_dir or find_latest_log_dir()
    if not log_dir:
        print("ERROR: No log directory found"); sys.exit(1)

    out = args.out_dir or f"plots/{os.path.basename(log_dir)}"
    os.makedirs(out, exist_ok=True)

    # Auto-detect available prefixes
    versions = {}
    prefix_labels = {
        "fixed": "Fixed Policy",
        "llm_with_memory": "LLM-Mem\n(SS-enabled)",
        "llm_with_memory_no_ss": "LLM-Mem\n(no SS)",
        "llm_no_memory_no_ss": "LLM-NoMem\n(no SS)",
        "llm_no_memory": "LLM-NoMem\n(SS-enabled)",
        "rule_based": "Rule-Based\n(SS-enabled)",
        "agentic": "Agentic\n(legacy)",
    }

    fixed_reports = []
    for prefix in sorted(os.listdir(log_dir)):
        path = os.path.join(log_dir, prefix)
        if not os.path.isdir(path):
            continue
        reports = load(path)
        if not reports:
            continue
        label = prefix_labels.get(prefix, prefix)
        if prefix == "fixed":
            fixed_reports = reports
            # Also add fixed as its own version for standalone plotting
            versions[label] = {"adaptive": reports, "fixed": []}
        else:
            versions[label] = {"adaptive": reports, "fixed": fixed_reports}
        print(f"  {label}: {len(reports)} reports, {len(sc(reports))} speech chunks")

    if not versions:
        print("ERROR: No data found"); sys.exit(1)

    print(f"\nPlotting to {out}/\n")

    # ── 1. Average Privacy Score ──
    labels = list(versions.keys())
    a_means = [np.mean([c["metrics"]["privacy"]["privacy_score"] for c in sc(v["adaptive"])]) if sc(v["adaptive"]) else 0 for v in versions.values()]
    a_stds = [np.std([c["metrics"]["privacy"]["privacy_score"] for c in sc(v["adaptive"])]) if sc(v["adaptive"]) else 0 for v in versions.values()]
    f_means = [np.mean([c["metrics"]["privacy"]["privacy_score"] for c in sc(v["fixed"])]) if sc(v["fixed"]) else 0 for v in versions.values()]
    f_stds = [np.std([c["metrics"]["privacy"]["privacy_score"] for c in sc(v["fixed"])]) if sc(v["fixed"]) else 0 for v in versions.values()]

    x = np.arange(len(labels)); w = 0.35
    fig, ax = plt.subplots(figsize=(15, 7))
    b1 = ax.bar(x - w/2, f_means, w, yerr=f_stds, capsize=4, label=PIPE_FIXED, color="#FF9800", alpha=0.85)
    b2 = ax.bar(x + w/2, a_means, w, yerr=a_stds, capsize=4, label=PIPE_ADAPTIVE, color="#2196F3", alpha=0.85)
    for bars, means, stds in [(b1, f_means, f_stds), (b2, a_means, a_stds)]:
        for bar, m, s in zip(bars, means, stds):
            if m > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() / 2,
                        f"{m:.3f}\n±{s:.3f}", ha="center", va="center", fontweight="bold", fontsize=12)
    ax.axhline(0.65, color="orange", ls="--", lw=1, alpha=0.7, label="High (0.65)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Privacy Score"); ax.set_ylim(0, 1.35)
    ax.set_title("Average Privacy Score (±σ)", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "avg_privacy_score.png"), dpi=150); plt.close(fig)
    print("  ✓ avg_privacy_score.png")

    # ── 2. Average Preserve Score ──
    a_pres = [np.mean([c["metrics"]["utility"]["preserve_score"] for c in sc(v["adaptive"])]) if sc(v["adaptive"]) else 0 for v in versions.values()]
    a_pres_std = [np.std([c["metrics"]["utility"]["preserve_score"] for c in sc(v["adaptive"])]) if sc(v["adaptive"]) else 0 for v in versions.values()]
    f_pres = [np.mean([c["metrics"]["utility"]["preserve_score"] for c in sc(v["fixed"])]) if sc(v["fixed"]) else 0 for v in versions.values()]
    f_pres_std = [np.std([c["metrics"]["utility"]["preserve_score"] for c in sc(v["fixed"])]) if sc(v["fixed"]) else 0 for v in versions.values()]
    fig, ax = plt.subplots(figsize=(15, 7))
    b1 = ax.bar(x - w/2, f_pres, w, yerr=f_pres_std, capsize=4, label=PIPE_FIXED, color="#FF9800", alpha=0.85)
    b2 = ax.bar(x + w/2, a_pres, w, yerr=a_pres_std, capsize=4, label=PIPE_ADAPTIVE, color="#2196F3", alpha=0.85)
    for bars, means, stds in [(b1, f_pres, f_pres_std), (b2, a_pres, a_pres_std)]:
        for bar, m, s in zip(bars, means, stds):
            if m > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() / 2,
                        f"{m:.4f}\n±{s:.4f}", ha="center", va="center", fontweight="bold", fontsize=12)
    ax.axhline(0.80, color="red", ls="--", lw=1, alpha=0.7, label="Threshold (0.80)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Preserve Score"); ax.set_ylim(0.6, 1.1)
    ax.set_title("Average Preserve Score", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "avg_preserve_score.png"), dpi=150); plt.close(fig)
    print("  ✓ avg_preserve_score.png")

    # ── 3. Recipe Usage ──
    all_recipes = set()
    ver_recipes = {}
    for label, v in versions.items():
        counts = {}
        for c in sc(v["adaptive"]):
            r = c.get("recipe_applied", {})
            if r:
                name = r.get("recipe_name", "unknown")
                counts[name] = counts.get(name, 0) + 1
                all_recipes.add(name)
        ver_recipes[label] = counts

    recipe_colors = {"RECIPE_MID_BAND_ATTEN": "#2196F3", "RECIPE_LOWPASS_HIGHPASS_MIX": "#FF9800", "RECIPE_SOURCE_SEPARATION": "#4CAF50"}
    recipes = sorted(all_recipes)
    fig, ax = plt.subplots(figsize=(15, 7))
    bottom = np.zeros(len(versions))
    for recipe in recipes:
        vals = [ver_recipes[l].get(recipe, 0) for l in versions]
        ax.bar(x, vals, 0.5, bottom=bottom, label=recipe.replace("RECIPE_", ""), color=recipe_colors.get(recipe, "#999"), alpha=0.85)
        bottom += np.array(vals)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Chunks"); ax.set_title("Recipe Usage (Adaptive)", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "recipe_usage.png"), dpi=150); plt.close(fig)
    print("  ✓ recipe_usage.png")

    # ── 4. Trials Distribution ──
    fig, ax = plt.subplots(figsize=(15, 7))
    colors = ["#2196F3", "#0D47A1", "#4CAF50", "#FF5722", "#FF9800"]
    for i, (label, v) in enumerate(versions.items()):
        s = sc(v["adaptive"])
        trials = [c.get("trials", 0) for c in s if c.get("trials", 0) > 0]
        avg = np.mean(trials) if trials else 0
        ax.bar(i, avg, color=colors[i % len(colors)], alpha=0.85)
        ax.text(i, avg + 0.05, f"{avg:.2f}", ha="center", fontweight="bold")
    ax.set_xticks(range(len(versions))); ax.set_xticklabels(labels)
    ax.set_ylabel("Avg Trials"); ax.set_title("Average Trials per Chunk (Adaptive)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "trials_distribution.png"), dpi=150); plt.close(fig)
    print("  ✓ trials_distribution.png")

    # ── 5. Privacy vs Preserve Trade-off ──
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, (label, v) in enumerate(versions.items()):
        s = sc(v["adaptive"])
        priv = [c["metrics"]["privacy"]["privacy_score"] for c in s]
        pres = [c["metrics"]["utility"]["preserve_score"] for c in s]
        ax.scatter(priv, pres, alpha=0.4, s=25, label=label, color=colors[i % len(colors)])
    ax.axvline(0.65, color="orange", ls="--", lw=1, alpha=0.5)
    ax.axhline(0.80, color="red", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("Privacy Score"); ax.set_ylabel("Preserve Score")
    ax.set_title("Privacy vs Preserve Trade-off", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "privacy_preserve_tradeoff.png"), dpi=150); plt.close(fig)
    print("  ✓ privacy_preserve_tradeoff.png")

    # ── 6. Speech Ratio vs Privacy ──
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, (label, v) in enumerate(versions.items()):
        s = sc(v["adaptive"])
        ratios = [c.get("speech_ratio", 0) for c in s]
        priv = [c["metrics"]["privacy"]["privacy_score"] for c in s]
        ax.scatter(ratios, priv, alpha=0.4, s=25, label=label, color=colors[i % len(colors)])
    ax.axhline(0.65, color="orange", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("Speech Ratio"); ax.set_ylabel("Privacy Score")
    ax.set_title("Speech Ratio vs Privacy Score", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "speech_ratio_vs_privacy.png"), dpi=150); plt.close(fig)
    print("  ✓ speech_ratio_vs_privacy.png")

    # ── 7. Classification Metrics (mAP, Accuracy, F1) ──
    fig, ax = plt.subplots(figsize=(15, 7))
    metric_names = ["mAP", "accuracy", "f1"]
    metric_labels = ["mAP", "Accuracy", "F1-Score"]
    metric_colors = ["#1976D2", "#F57C00", "#388E3C"]
    bar_w = 0.8 / len(metric_names)
    for j, (mname, mlabel, mcolor) in enumerate(zip(metric_names, metric_labels, metric_colors)):
        vals, stds = [], []
        for v in versions.values():
            s = sc(v["adaptive"])
            scores = [c["metrics"]["utility"][mname] for c in s if mname in c["metrics"]["utility"]]
            vals.append(np.mean(scores) if scores else 0)
            stds.append(np.std(scores) if scores else 0)
        bars = ax.bar(x + j * bar_w - (len(metric_names)-1)*bar_w/2, vals, bar_w,
                       yerr=stds, capsize=3, label=mlabel, color=mcolor, alpha=0.85)
        for bar, m, s in zip(bars, vals, stds):
            if m > 0:
                ax.text(bar.get_x() + bar.get_width()/2, min(bar.get_height() + s + 0.03, ax.get_ylim()[1] - 0.05),
                        f"{m:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.35)
    ax.set_title("Classification Metrics (mAP, Accuracy, F1) ±σ", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "classification_metrics.png"), dpi=150); plt.close(fig)
    print("  ✓ classification_metrics.png")

    # ── 8. Top-3 Classification Labels ──
    from collections import Counter
    fig, axes = plt.subplots(1, len(versions), figsize=(6*len(versions), 6), squeeze=False)
    axes = axes[0]
    for i, (label, v) in enumerate(versions.items()):
        ax = axes[i]
        s = sc(v["adaptive"])
        # Count top-1 labels from classification_top3
        label_counts = Counter()
        for c in s:
            top3 = c.get("classification_top3") or c.get("classification_top3_original") or []
            if top3:
                label_counts[top3[0]["label"]] += 1
            else:
                # Fallback: use predictions from metrics if available
                preds = c.get("metrics", {}).get("utility", {})
                pass
        if label_counts:
            top_labels = label_counts.most_common(10)
            lnames = [l[0][:20] for l in top_labels]
            lcounts = [l[1] for l in top_labels]
            ax.barh(range(len(lnames)), lcounts, color=colors[i % len(colors)], alpha=0.85)
            ax.set_yticks(range(len(lnames))); ax.set_yticklabels(lnames, fontsize=9)
            ax.invert_yaxis()
        ax.set_xlabel("Count"); ax.set_title(label.replace("\n", " "), fontsize=12)
    fig.suptitle("Top-1 Classification Labels (Processed Audio)", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(out, "classification_top_labels.png"), dpi=150); plt.close(fig)
    print("  ✓ classification_top_labels.png")

    # ── 9. Original vs Processed Classification Comparison ──
    # Show how top-1 label changes after processing
    changed_counts = []
    for label, v in versions.items():
        s = sc(v["adaptive"])
        changed = 0
        total = 0
        for c in s:
            orig = c.get("classification_top3_original", [])
            proc = c.get("classification_top3", [])
            if orig and proc:
                total += 1
                if orig[0]["label"] != proc[0]["label"]:
                    changed += 1
        changed_counts.append((label, changed, total))

    if any(t > 0 for _, _, t in changed_counts):
        fig, ax = plt.subplots(figsize=(15, 7))
        pct = [100 * ch / max(t, 1) for _, ch, t in changed_counts]
        bars = ax.bar(range(len(changed_counts)), pct, color=colors[:len(changed_counts)], alpha=0.85)
        for bar, (lbl, ch, t) in zip(bars, changed_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{ch}/{t}\n({bar.get_height():.1f}%)", ha="center", fontweight="bold", fontsize=12)
        ax.set_xticks(range(len(changed_counts)))
        ax.set_xticklabels([l.replace("\n", " ") for l, _, _ in changed_counts])
        ax.set_ylabel("% Chunks with Changed Top-1 Label")
        ax.set_title("Classification Label Change After Processing", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(out, "classification_label_change.png"), dpi=150); plt.close(fig)
        print("  ✓ classification_label_change.png")

    # ── 10. Top-K Class Consistency, Speech Rank Drop, Confidence Gap ──
    # Only for chunks that have both classification_top3_original and classification_top3
    topk_data = {}
    for label, v in versions.items():
        s = sc(v["adaptive"])
        consistency_scores = []
        speech_rank_drops = []
        confidence_gaps = []
        for c in s:
            orig = c.get("classification_top3_original", [])
            proc = c.get("classification_top3", [])
            if not orig or not proc:
                continue

            # Top-K Class Consistency: how many of original top-3 labels remain in processed top-3
            orig_labels = set(p["label"] for p in orig[:3])
            proc_labels = set(p["label"] for p in proc[:3])
            overlap = len(orig_labels & proc_labels)
            consistency_scores.append(overlap / max(len(orig_labels), 1))

            # Speech Rank Drop: rank of "Speech" in original vs processed
            def find_speech_rank(preds):
                for i, p in enumerate(preds):
                    if "speech" in p["label"].lower():
                        return i + 1  # 1-indexed
                return len(preds) + 1  # not found = beyond list

            orig_rank = find_speech_rank(orig)
            proc_rank = find_speech_rank(proc)
            speech_rank_drops.append(proc_rank - orig_rank)

            # Confidence Gap: drop in Speech confidence
            def find_speech_conf(preds):
                for p in preds:
                    if "speech" in p["label"].lower():
                        return p["confidence"]
                return 0.0

            orig_conf = find_speech_conf(orig)
            proc_conf = find_speech_conf(proc)
            confidence_gaps.append(orig_conf - proc_conf)

        if consistency_scores:
            topk_data[label] = {
                "consistency": consistency_scores,
                "rank_drop": speech_rank_drops,
                "conf_gap": confidence_gaps,
            }

    if topk_data:
        tk_labels = list(topk_data.keys())
        tk_x = np.arange(len(tk_labels))

        # 10a. Top-K Class Consistency
        fig, ax = plt.subplots(figsize=(15, 7))
        means = [np.mean(topk_data[l]["consistency"]) for l in tk_labels]
        stds = [np.std(topk_data[l]["consistency"]) for l in tk_labels]
        bars = ax.bar(tk_x, means, 0.5, yerr=stds, capsize=4, color="#4CAF50", alpha=0.85)
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() / 2,
                    f"{m:.3f}\n±{s:.3f}", ha="center", va="center", fontweight="bold", fontsize=12)
        ax.set_xticks(tk_x); ax.set_xticklabels(tk_labels)
        ax.set_ylabel("Consistency (0-1)"); ax.set_ylim(0, 1.5)
        ax.set_title("Top-3 Class Consistency\n(Fraction of original top-3 labels preserved after processing)", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(out, "topk_class_consistency.png"), dpi=150); plt.close(fig)
        print("  ✓ topk_class_consistency.png")

        # 10b. Speech Rank Drop
        fig, ax = plt.subplots(figsize=(15, 7))
        means = [np.mean(topk_data[l]["rank_drop"]) for l in tk_labels]
        stds = [np.std(topk_data[l]["rank_drop"]) for l in tk_labels]
        bars = ax.bar(tk_x, means, 0.5, yerr=stds, capsize=4, color="#E53935", alpha=0.85)
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, max(bar.get_height(), 0) + 0.15,
                    f"{m:+.2f}\n±{s:.2f}", ha="center", va="center", fontweight="bold", fontsize=12)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(tk_x); ax.set_xticklabels(tk_labels)
        ax.set_ylabel("Rank Drop (positive = dropped)")
        ax.set_title("Speech Rank Drop After Processing\n(Higher = Speech class dropped more ranks = better privacy)", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(out, "speech_rank_drop.png"), dpi=150); plt.close(fig)
        print("  ✓ speech_rank_drop.png")

        # 10c. Confidence Gap
        fig, ax = plt.subplots(figsize=(15, 7))
        means = [np.mean(topk_data[l]["conf_gap"]) for l in tk_labels]
        stds = [np.std(topk_data[l]["conf_gap"]) for l in tk_labels]
        bars = ax.bar(tk_x, means, 0.5, yerr=stds, capsize=4, color="#7B1FA2", alpha=0.85)
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() / 2,
                    f"{m:+.3f}\n±{s:.3f}", ha="center", va="center", fontweight="bold", fontsize=12)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(tk_x); ax.set_xticklabels(tk_labels)
        ax.set_ylabel("Confidence Drop (positive = reduced)")
        ax.set_title("Speech Confidence Gap (Original − Processed)\n(Higher = Speech confidence reduced more = better privacy)", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(out, "speech_confidence_gap.png"), dpi=150); plt.close(fig)
        print("  ✓ speech_confidence_gap.png")

    # ── 10d. No-Speech Chunks Analysis ──
    # Preserve score for no-speech chunks (should be ~1.0 since no blurring)
    ns_labels, ns_pres_means, ns_pres_stds, ns_counts = [], [], [], []
    for label, v in versions.items():
        ns = no_sc(v["adaptive"])
        if ns:
            ns_labels.append(label)
            scores = [c["metrics"]["utility"]["preserve_score"] for c in ns]
            ns_pres_means.append(np.mean(scores))
            ns_pres_stds.append(np.std(scores))
            ns_counts.append(len(ns))

    if ns_labels:
        fig, ax = plt.subplots(figsize=(15, 7))
        xn = np.arange(len(ns_labels))
        bars = ax.bar(xn, ns_pres_means, 0.5, yerr=ns_pres_stds, capsize=4, color="#9E9E9E", alpha=0.85)
        for bar, m, s, n in zip(bars, ns_pres_means, ns_pres_stds, ns_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() / 2,
                    f"{m:.4f}±{s:.4f}\nn={n}", ha="center", va="center", fontweight="bold", fontsize=12)
        ax.axhline(0.80, color="red", ls="--", lw=1, alpha=0.7)
        ax.set_xticks(xn); ax.set_xticklabels(ns_labels)
        ax.set_ylabel("Preserve Score"); ax.set_ylim(0.6, 1.1)
        ax.set_title("Preserve Score — No-Speech Chunks (Bypass, No Blurring)", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(out, "no_speech_preserve.png"), dpi=150); plt.close(fig)
        print("  ✓ no_speech_preserve.png")

    # Top-1 classification for no-speech chunks
    if ns_labels:
        fig, axes = plt.subplots(1, len(ns_labels), figsize=(6*len(ns_labels), 6), squeeze=False)
        axes = axes[0]
        for i, (label, v) in enumerate([(l, versions[l]) for l in ns_labels]):
            ax = axes[i]
            ns = no_sc(v["adaptive"])
            label_counts = Counter()
            for c in ns:
                top3 = c.get("classification_top3") or c.get("classification_top3_original") or []
                if top3:
                    label_counts[top3[0]["label"]] += 1
            if label_counts:
                top_items = label_counts.most_common(10)
                lnames = [l[0][:25] for l in top_items]
                lcounts = [l[1] for l in top_items]
                ax.barh(range(len(lnames)), lcounts, color="#9E9E9E", alpha=0.85)
                ax.set_yticks(range(len(lnames))); ax.set_yticklabels(lnames, fontsize=9)
                ax.invert_yaxis()
            ax.set_xlabel("Count"); ax.set_title(label.replace("\n", " "), fontsize=12)
        fig.suptitle("Top-1 Classification — No-Speech Chunks", fontweight="bold")
        fig.tight_layout(); fig.savefig(os.path.join(out, "no_speech_classification.png"), dpi=150); plt.close(fig)
        print("  ✓ no_speech_classification.png")

    # No-speech summary stats
    if ns_labels:
        cols_ns = ["Version", "No-Speech\nChunks", "Avg Preserve", "Avg mAP", "Avg F1"]
        rows_ns = []
        for label in ns_labels:
            ns = no_sc(versions[label]["adaptive"])
            rows_ns.append([
                label.replace("\n", " "),
                str(len(ns)),
                f"{np.mean([c['metrics']['utility']['preserve_score'] for c in ns]):.4f}",
                f"{np.mean([c['metrics']['utility']['mAP'] for c in ns if 'mAP' in c['metrics']['utility']]):.4f}",
                f"{np.mean([c['metrics']['utility']['f1'] for c in ns if 'f1' in c['metrics']['utility']]):.4f}",
            ])
        fig, ax = plt.subplots(figsize=(16, 2 + 0.5 * len(rows_ns)))
        ax.axis("off")
        table = ax.table(cellText=rows_ns, colLabels=cols_ns, loc="center", cellLoc="center", colColours=["#E0E0E0"] * len(cols_ns))
        table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.8)
        ax.set_title("No-Speech Chunks Summary", fontweight="bold", pad=20)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "no_speech_summary.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
        print("  ✓ no_speech_summary.png")

    # ── 11. Summary Table ──
    cols = ["Version", "Files", "Speech\nChunks", "Avg Privacy", "Avg Preserve", "Avg Trials", "Recipes"]
    rows = []
    for label, v in versions.items():
        s = sc(v["adaptive"])
        recipes = {}
        for c in s:
            r = c.get("recipe_applied", {})
            if r:
                name = r.get("recipe_name", "?").replace("RECIPE_", "")
                recipes[name] = recipes.get(name, 0) + 1
        rows.append([
            label.replace("\n", " "),
            str(len(v["adaptive"])),
            str(len(s)),
            f"{np.mean([c['metrics']['privacy']['privacy_score'] for c in s]):.4f}" if s else "N/A",
            f"{np.mean([c['metrics']['utility']['preserve_score'] for c in s]):.4f}" if s else "N/A",
            f"{np.mean([c.get('trials', 0) for c in s]):.2f}" if s else "N/A",
            ", ".join(f"{k}:{v}" for k, v in sorted(recipes.items())),
        ])

    fig, ax = plt.subplots(figsize=(20, 2 + 0.6 * len(rows)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center", colColours=["#E3F2FD"] * len(cols))
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.8)
    ax.set_title("Summary", fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "summary_table.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
    print("  ✓ summary_table.png")

    total = sum(1 for f in os.listdir(out) if f.endswith(".png"))
    print(f"\nDone — {total} charts saved to {out}/")


if __name__ == "__main__":
    main()
