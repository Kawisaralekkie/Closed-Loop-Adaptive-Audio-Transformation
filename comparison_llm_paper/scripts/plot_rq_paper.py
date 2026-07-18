#!/usr/bin/env python3
"""Plot charts for paper Research Questions (RQ1-RQ3).

Paper: "A Comparative Study of Adaptive Agent-Based Privacy Control
        and Fixed Transformation Pipelines for Urban Soundscape Analysis"

Versions:
  V1: Claude Haiku 3.5                    (LLM with memory)
  V2: Claude Haiku 4.5                    (LLM with memory)
  V3: Claude Haiku 4.5 + Source Separation (LLM with memory)
  V4: Claude Haiku 4.5 + SS + Rule-Based  (no LLM)
  V5: Claude Haiku 4.5 + SS + No Memory   (LLM without memory)

Sections:
  V.A  Privacy Performance Comparison (RQ1)
  V.B  Effect of Speech Characteristics on Privacy (RQ2)
  V.C  Transformation Selection Behavior (RQ2)
  V.D  Impact of Source Separation (RQ2+RQ3)
  V.E  Utility Preservation (RQ3)
  V.F  Metric Variability
  V.G  Summary Table

Usage:
    python3 scripts/plot_rq_paper.py [--out-dir plots/paper_rq]
"""

from __future__ import annotations
import json, glob, os, sys, argparse, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Global font: Times New Roman, size 8 ───────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

# ── Version config ──────────────────────────────────────────────────────
VERSIONS = {
    "LLM-H3.5-Mem": {
        "adaptive": "logs/agentic",
        "fixed": "logs/fixed",
    },
    "LLM-H4.5-Mem": {
        "adaptive": "logs/s3/agentic",
        "fixed": "logs/s3/fixed",
    },
    "LLM-H4.5-Mem\n(SS-enabled)": {
        "adaptive": "logs/s3/20260406_171533/agentic",
        "fixed": "logs/s3/20260406_171533/fixed",
    },
    "Rule-NoMem\n(SS-enabled)": {
        "adaptive": "logs/s3/20260414_132841/rule_based",
        "fixed": None,
    },
    "LLM-H4.5-NoMem\n(SS-enabled)": {
        "adaptive": "logs/s3/20260414_132841/llm_no_memory",
        "fixed": None,
    },
}

# Colors
C_ADAPTIVE = ["#2196F3", "#1565C0", "#0D47A1", "#4CAF50", "#FF5722"]
C_FIXED    = ["#FF9800", "#EF6C00", "#E65100", "#999999", "#999999"]

# Paper-friendly labels
PIPE_ADAPTIVE = "Adaptive Policy"
PIPE_FIXED    = "Fixed Policy"


# ── Data loading ────────────────────────────────────────────────────────

def load(folder):
    if not folder or not os.path.exists(folder):
        return []
    r = []
    files = sorted(glob.glob(os.path.join(folder, "*_report.json")))
    if not files:
        # Try one level deeper (e.g. logs/agentic/6-3-2025/*.json)
        files = sorted(glob.glob(os.path.join(folder, "*/*_report.json")))
    for f in files:
        with open(f) as fh:
            r.append(json.load(fh))
    return r


def sc(reports):
    """Speech chunks with metrics."""
    return [c for r in reports for c in r.get("chunks", [])
            if c.get("had_speech") and c.get("metrics")]


def all_chunks(reports):
    """All chunks with metrics (including bypass)."""
    return [c for r in reports for c in r.get("chunks", []) if c.get("metrics")]


def load_all():
    data = {}
    for ver, paths in VERSIONS.items():
        a = load(paths["adaptive"])
        f = load(paths["fixed"]) if paths["fixed"] else []
        if a:
            data[ver] = {"adaptive": a, "fixed": f}
            print(f"  {ver}: {len(a)} adaptive, {len(f)} fixed reports")
    return data



# ═══════════════════════════════════════════════════════════════════════
# V.A — Privacy Performance Comparison (RQ1)
# ═══════════════════════════════════════════════════════════════════════

def va_avg_privacy_score(data, out):
    """Bar chart: Avg privacy_score — Adaptive vs Fixed per version."""
    labels, a_vals, f_vals = [], [], []
    for ver, info in data.items():
        labels.append(ver)
        sa = sc(info["adaptive"])
        sf = sc(info["fixed"])
        a_vals.append(np.mean([c["metrics"]["privacy"]["privacy_score"] for c in sa]) if sa else 0)
        f_vals.append(np.mean([c["metrics"]["privacy"]["privacy_score"] for c in sf]) if sf else 0)

    x = np.arange(len(labels)); w = 0.35
    fig, ax = plt.subplots(figsize=(13, 6))
    # Uniform color: Fixed = orange, Adaptive = blue
    b1 = ax.bar(x - w/2, f_vals, w, label=PIPE_FIXED, color="#FF9800", alpha=0.85)
    b2 = ax.bar(x + w/2, a_vals, w, label=PIPE_ADAPTIVE, color="#2196F3", alpha=0.85)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                        f"{h:.3f}", ha="center", va="bottom", fontweight="bold")
    ax.axhline(0.65, color="orange", ls="--", lw=1, alpha=0.7, label="High threshold (0.65)")
    ax.axhline(0.80, color="red",    ls="--", lw=1, alpha=0.7, label="Very-high threshold (0.80)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Average Privacy Score"); ax.set_ylim(0, 1.1)
    ax.set_title("Average Privacy Score Across Pipeline Versions", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "va_avg_privacy_score.png"), dpi=150); plt.close(fig)
    print("  ✓ va_avg_privacy_score.png")


def va_privacy_boxplot(data, out):
    """Boxplot: Privacy score distribution per version."""
    fig, ax = plt.subplots(figsize=(14, 6))
    box_data, box_labels, colors = [], [], []
    for i, (ver, info) in enumerate(data.items()):
        for pname, pkey, uni_color in [(PIPE_FIXED, "fixed", "#FF9800"), (PIPE_ADAPTIVE, "adaptive", "#2196F3")]:
            s = sc(info[pkey])
            scores = [c["metrics"]["privacy"]["privacy_score"] for c in s]
            if scores:
                box_data.append(scores)
                box_labels.append(f"{ver}\n{pname}")
                colors.append(uni_color)
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    ax.axhline(0.65, color="orange", ls="--", lw=1, alpha=0.7)
    ax.axhline(0.80, color="red",    ls="--", lw=1, alpha=0.7)
    ax.set_ylabel("Privacy Score")
    ax.set_title("Privacy Score Distribution", fontweight="bold")
    ax.grid(axis="y", alpha=0.3); plt.xticks(rotation=15)
    fig.tight_layout(); fig.savefig(os.path.join(out, "va_privacy_boxplot.png"), dpi=150); plt.close(fig)
    print("  ✓ va_privacy_boxplot.png")


def va_privacy_sub_metrics(data, out):
    """Grouped bar: WER, CER, Speaker Privacy per version (Adaptive only)."""
    labels, wers, cers, speakers = [], [], [], []
    for ver, info in data.items():
        s = sc(info["adaptive"])
        if not s: continue
        labels.append(ver)
        wers.append(np.mean([c["metrics"]["privacy"]["wer"] for c in s]))
        cers.append(np.mean([c["metrics"]["privacy"]["cer"] for c in s]))
        speakers.append(np.mean([c["metrics"]["privacy"]["speaker_privacy"] for c in s]))

    x = np.arange(len(labels)); w = 0.25
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - w, wers, w, label="WER", color="#E53935", alpha=0.8)
    ax.bar(x,     cers, w, label="CER", color="#FB8C00", alpha=0.8)
    ax.bar(x + w, speakers, w, label="Speaker Privacy", color="#7B1FA2", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.1)
    ax.set_title("Privacy Sub-Metrics (Adaptive Policy)", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "va_privacy_sub_metrics.png"), dpi=150); plt.close(fig)
    print("  ✓ va_privacy_sub_metrics.png")


def va_pass_rate(data, out):
    """Bar: % of speech chunks passing privacy threshold per version."""
    labels, a_rates, f_rates = [], [], []
    for ver, info in data.items():
        labels.append(ver)
        sa = sc(info["adaptive"])
        sf = sc(info["fixed"])
        a_rates.append(100 * np.mean([c["metrics"]["decision"]["privacy_pass"] for c in sa]) if sa else 0)
        f_rates.append(100 * np.mean([c["metrics"]["decision"]["privacy_pass"] for c in sf]) if sf else 0)

    x = np.arange(len(labels)); w = 0.35
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - w/2, f_rates, w, label=PIPE_FIXED, color="#FF9800", alpha=0.85)
    ax.bar(x + w/2, a_rates, w, label=PIPE_ADAPTIVE, color="#2196F3", alpha=0.85)
    for bars in [ax.containers[0], ax.containers[1]]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                        f"{h:.1f}%", ha="center", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Privacy Pass Rate (%)"); ax.set_ylim(0, 110)
    ax.set_title("Privacy Threshold Pass Rate", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "va_privacy_pass_rate.png"), dpi=150); plt.close(fig)
    print("  ✓ va_privacy_pass_rate.png")



# ═══════════════════════════════════════════════════════════════════════
# V.B — Effect of Speech Characteristics on Privacy (RQ2)
# ═══════════════════════════════════════════════════════════════════════

def vb_speech_ratio_vs_privacy(data, out):
    """Scatter: speech_ratio vs privacy_score per version (Adaptive)."""
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, (ver, info) in enumerate(data.items()):
        s = sc(info["adaptive"])
        ratios = [c.get("speech_ratio", 0) for c in s]
        scores = [c["metrics"]["privacy"]["privacy_score"] for c in s]
        ax.scatter(ratios, scores, alpha=0.5, s=30, label=ver, color=C_ADAPTIVE[i])
    ax.axhline(0.65, color="orange", ls="--", lw=1, alpha=0.7, label="High (0.65)")
    ax.axhline(0.80, color="red",    ls="--", lw=1, alpha=0.7, label="Very-high (0.80)")
    ax.set_xlabel("Speech Ratio"); ax.set_ylabel("Privacy Score")
    ax.set_title("Speech Ratio vs Privacy Score (Adaptive Policy)", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "vb_speech_ratio_vs_privacy.png"), dpi=150); plt.close(fig)
    print("  ✓ vb_speech_ratio_vs_privacy.png")


def vb_vad_confidence_vs_privacy(data, out):
    """Scatter: vad_confidence vs privacy_score per version (Adaptive)."""
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, (ver, info) in enumerate(data.items()):
        s = sc(info["adaptive"])
        confs = [c.get("vad_confidence", 0) for c in s]
        scores = [c["metrics"]["privacy"]["privacy_score"] for c in s]
        ax.scatter(confs, scores, alpha=0.5, s=30, label=ver, color=C_ADAPTIVE[i])
    ax.axhline(0.65, color="orange", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("VAD Confidence"); ax.set_ylabel("Privacy Score")
    ax.set_title("VAD Confidence vs Privacy Score (Adaptive Policy)", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "vb_vad_confidence_vs_privacy.png"), dpi=150); plt.close(fig)
    print("  ✓ vb_vad_confidence_vs_privacy.png")


def vb_speech_ratio_histogram(data, out):
    """Histogram: speech_ratio distribution across versions."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (ver, info) in enumerate(data.items()):
        s = sc(info["adaptive"])
        ratios = [c.get("speech_ratio", 0) for c in s]
        if ratios:
            ax.hist(ratios, bins=20, alpha=0.4, label=ver, color=C_ADAPTIVE[i], edgecolor="white")
    ax.set_xlabel("Speech Ratio"); ax.set_ylabel("Count")
    ax.set_title("Speech Ratio Distribution (Adaptive Policy)", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "vb_speech_ratio_histogram.png"), dpi=150); plt.close(fig)
    print("  ✓ vb_speech_ratio_histogram.png")


# ═══════════════════════════════════════════════════════════════════════
# V.C — Transformation Selection Behavior (RQ2)
# ═══════════════════════════════════════════════════════════════════════

def vc_recipe_usage(data, out):
    """Stacked bar: Recipe usage per version (Adaptive only)."""
    all_recipes = set()
    version_recipes = {}
    for ver, info in data.items():
        s = sc(info["adaptive"])
        counts = {}
        for c in s:
            r = c.get("recipe_applied", {})
            if r:
                name = r.get("recipe_name", "unknown")
                counts[name] = counts.get(name, 0) + 1
                all_recipes.add(name)
        version_recipes[ver] = counts

    recipes = sorted(all_recipes)
    recipe_colors = {
        "RECIPE_MID_BAND_ATTEN": "#2196F3",
        "RECIPE_LOWPASS_HIGHPASS_MIX": "#FF9800",
        "RECIPE_SOURCE_SEPARATION": "#4CAF50",
    }
    recipe_labels = {
        "RECIPE_MID_BAND_ATTEN": "Mid-Band Attenuation",
        "RECIPE_LOWPASS_HIGHPASS_MIX": "Lowpass-Highpass Mix",
        "RECIPE_SOURCE_SEPARATION": "Source Separation",
    }

    x = np.arange(len(data)); w = 0.5
    fig, ax = plt.subplots(figsize=(13, 6))
    bottom = np.zeros(len(data))
    for recipe in recipes:
        vals = [version_recipes[ver].get(recipe, 0) for ver in data.keys()]
        color = recipe_colors.get(recipe, "#999")
        label = recipe_labels.get(recipe, recipe.replace("RECIPE_", ""))
        ax.bar(x, vals, w, bottom=bottom, label=label, color=color, alpha=0.85)
        bottom += np.array(vals)

    ax.set_xticks(x); ax.set_xticklabels(list(data.keys()))
    ax.set_ylabel("Number of Chunks")
    ax.set_title("Recipe Usage per Version (Adaptive Policy)", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "vc_recipe_usage.png"), dpi=150); plt.close(fig)
    print("  ✓ vc_recipe_usage.png")


def vc_trials_distribution(data, out):
    """Bar: Average trials per version (Adaptive)."""
    labels, avg_trials = [], []
    for ver, info in data.items():
        s = sc(info["adaptive"])
        trials = [c.get("trials", 0) for c in s if c.get("trials", 0) > 0]
        labels.append(ver)
        avg_trials.append(np.mean(trials) if trials else 0)

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(labels, avg_trials, color=C_ADAPTIVE[:len(labels)], alpha=0.85)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{bar.get_height():.2f}", ha="center", fontweight="bold")
    ax.set_ylabel("Average Trials per Chunk")
    ax.set_title("Average Retry Trials (Adaptive Policy)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "vc_trials_distribution.png"), dpi=150); plt.close(fig)
    print("  ✓ vc_trials_distribution.png")


def vc_trials_histogram(data, out):
    """Histogram: trials count distribution per version."""
    fig, axes = plt.subplots(1, len(data), figsize=(4*len(data), 5), sharey=True)
    if len(data) == 1:
        axes = [axes]
    for i, (ver, info) in enumerate(data.items()):
        s = sc(info["adaptive"])
        trials = [c.get("trials", 0) for c in s]
        axes[i].hist(trials, bins=range(0, max(trials)+2) if trials else [0,1],
                     color=C_ADAPTIVE[i], alpha=0.8, edgecolor="white")
        axes[i].set_title(ver)
        axes[i].set_xlabel("Trials")
        if i == 0:
            axes[i].set_ylabel("Count")
    fig.suptitle("Trial Count Distribution (Adaptive Policy)", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(out, "vc_trials_histogram.png"), dpi=150); plt.close(fig)
    print("  ✓ vc_trials_histogram.png")



# ═══════════════════════════════════════════════════════════════════════
# V.D — Impact of Source Separation (RQ2 + RQ3)
# ═══════════════════════════════════════════════════════════════════════

def vd_source_sep_comparison(data, out):
    """Grouped bar: V2 (no SS) vs V3/V4/V5 (SS-enabled) — privacy & preserve."""
    no_ss_key = "LLM-H4.5-Mem"
    ss_keys = [
        ("LLM-H4.5-Mem\n(SS-enabled)", "LLM-Mem (SS)"),
        ("Rule-NoMem\n(SS-enabled)", "Rule-Based (SS)"),
        ("LLM-H4.5-NoMem\n(SS-enabled)", "LLM-NoMem (SS)"),
    ]

    if no_ss_key not in data:
        print("  ⚠ Skipping vd_source_sep_comparison (missing baseline)")
        return

    metrics_labels = ["Privacy Score", "Preserve Score", "WER", "CER", "Speaker Privacy"]

    def get_vals(chunks):
        if not chunks:
            return [0] * 5
        return [
            np.mean([c["metrics"]["privacy"]["privacy_score"] for c in chunks]),
            np.mean([c["metrics"]["utility"]["preserve_score"] for c in chunks]),
            np.mean([c["metrics"]["privacy"]["wer"] for c in chunks]),
            np.mean([c["metrics"]["privacy"]["cer"] for c in chunks]),
            np.mean([c["metrics"]["privacy"]["speaker_privacy"] for c in chunks]),
        ]

    # Collect all versions to plot
    plot_data = []
    plot_data.append(("LLM-Mem (no SS)", get_vals(sc(data[no_ss_key]["adaptive"])), "#1565C0"))
    for key, label in ss_keys:
        if key in data:
            plot_data.append((label, get_vals(sc(data[key]["adaptive"])), None))

    colors = ["#1565C0", "#0D47A1", "#4CAF50", "#FF5722"]
    n = len(plot_data)
    x = np.arange(len(metrics_labels)); w = 0.8 / n
    fig, ax = plt.subplots(figsize=(15, 6))
    for i, (label, vals, _) in enumerate(plot_data):
        bars = ax.bar(x + i * w - (n-1)*w/2, vals, w, label=label, color=colors[i], alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(metrics_labels)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.15)
    ax.set_title("Impact of Source Separation on Metrics", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "vd_source_sep_comparison.png"), dpi=150); plt.close(fig)
    print("  ✓ vd_source_sep_comparison.png")


# ═══════════════════════════════════════════════════════════════════════
# V.E — Utility Preservation (RQ3)
# ═══════════════════════════════════════════════════════════════════════

def ve_avg_preserve_score(data, out):
    """Bar chart: Avg preserve_score — Adaptive vs Fixed per version."""
    labels, a_vals, f_vals = [], [], []
    for ver, info in data.items():
        labels.append(ver)
        sa = sc(info["adaptive"])
        sf = sc(info["fixed"])
        a_vals.append(np.mean([c["metrics"]["utility"]["preserve_score"] for c in sa]) if sa else 0)
        f_vals.append(np.mean([c["metrics"]["utility"]["preserve_score"] for c in sf]) if sf else 0)

    x = np.arange(len(labels)); w = 0.35
    fig, ax = plt.subplots(figsize=(13, 6))
    b1 = ax.bar(x - w/2, f_vals, w, label=PIPE_FIXED, color="#FF9800", alpha=0.85)
    b2 = ax.bar(x + w/2, a_vals, w, label=PIPE_ADAPTIVE, color="#2196F3", alpha=0.85)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                        f"{h:.4f}", ha="center", va="bottom", fontweight="bold")
    ax.axhline(0.80, color="red", ls="--", lw=1, alpha=0.7, label="Threshold (0.80)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Average Preserve Score"); ax.set_ylim(0.7, 1.0)
    ax.set_title("Average Preserve Score Across Pipeline Versions", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "ve_avg_preserve_score.png"), dpi=150); plt.close(fig)
    print("  ✓ ve_avg_preserve_score.png")


def ve_preserve_boxplot(data, out):
    """Boxplot: Preserve score distribution per version."""
    fig, ax = plt.subplots(figsize=(14, 6))
    box_data, box_labels, colors = [], [], []
    for i, (ver, info) in enumerate(data.items()):
        for pname, pkey, uni_color in [(PIPE_FIXED, "fixed", "#FF9800"), (PIPE_ADAPTIVE, "adaptive", "#2196F3")]:
            s = sc(info[pkey])
            scores = [c["metrics"]["utility"]["preserve_score"] for c in s]
            if scores:
                box_data.append(scores)
                box_labels.append(f"{ver}\n{pname}")
                colors.append(uni_color)
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    ax.axhline(0.80, color="red", ls="--", lw=1, alpha=0.7)
    ax.set_ylabel("Preserve Score")
    ax.set_title("Preserve Score Distribution", fontweight="bold")
    ax.grid(axis="y", alpha=0.3); plt.xticks(rotation=15)
    fig.tight_layout(); fig.savefig(os.path.join(out, "ve_preserve_boxplot.png"), dpi=150); plt.close(fig)
    print("  ✓ ve_preserve_boxplot.png")


def ve_preserve_sub_scores(data, out):
    """Grouped bar: Preserve sub-scores per version (Adaptive only)."""
    sub_keys   = ["s_loud", "s_hf", "s_sc", "s_con", "s_psy"]
    sub_labels = ["Loudness", "High Freq", "Spectral Centroid", "mAP", "Psychoacoustic"]

    versions_with_data = [(ver, info) for ver, info in data.items() if sc(info["adaptive"])]
    n = len(versions_with_data)
    if n == 0: return

    x = np.arange(len(sub_keys)); w = 0.8 / n
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (ver, info) in enumerate(versions_with_data):
        s = sc(info["adaptive"])
        vals = []
        for key in sub_keys:
            scores = [c["metrics"]["utility"]["sub_scores"][key] for c in s
                      if "sub_scores" in c["metrics"]["utility"]]
            vals.append(np.mean(scores) if scores else 0)
        ax.bar(x + i * w - (n-1)*w/2, vals, w, label=ver, color=C_ADAPTIVE[i], alpha=0.85)

    ax.set_xticks(x); ax.set_xticklabels(sub_labels)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.1)
    ax.set_title("Preserve Sub-Scores (Adaptive Policy)", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "ve_preserve_sub_scores.png"), dpi=150); plt.close(fig)
    print("  ✓ ve_preserve_sub_scores.png")


def ve_classification_metrics(data, out):
    """Grouped bar: mAP, F1, Accuracy per version (Adaptive only)."""
    labels, maps, f1s, accs = [], [], [], []
    for ver, info in data.items():
        s = sc(info["adaptive"])
        if not s: continue
        labels.append(ver)
        maps.append(np.mean([c["metrics"]["utility"]["mAP"] for c in s]))
        f1s.append(np.mean([c["metrics"]["utility"]["f1"] for c in s]))
        accs.append(np.mean([c["metrics"]["utility"]["accuracy"] for c in s]))

    x = np.arange(len(labels)); w = 0.25
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - w, maps, w, label="mAP",      color="#1976D2", alpha=0.8)
    ax.bar(x,     f1s,  w, label="F1-Score",  color="#388E3C", alpha=0.8)
    ax.bar(x + w, accs, w, label="Accuracy",  color="#F57C00", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.1)
    ax.set_title("Classification Metrics After Processing (Adaptive Policy)", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "ve_classification_metrics.png"), dpi=150); plt.close(fig)
    print("  ✓ ve_classification_metrics.png")


def ve_privacy_preserve_tradeoff(data, out):
    """Scatter: Privacy vs Preserve trade-off per version (Adaptive)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, (ver, info) in enumerate(data.items()):
        s = sc(info["adaptive"])
        privacy  = [c["metrics"]["privacy"]["privacy_score"]  for c in s]
        preserve = [c["metrics"]["utility"]["preserve_score"] for c in s]
        ax.scatter(privacy, preserve, alpha=0.4, s=30, label=ver, color=C_ADAPTIVE[i])
    ax.axvline(0.65, color="orange", ls="--", lw=1, alpha=0.5, label="Privacy high (0.65)")
    ax.axhline(0.80, color="red",    ls="--", lw=1, alpha=0.5, label="Preserve min (0.80)")
    # Shade the "ideal" quadrant
    ax.axvspan(0.65, 1.05, ymin=(0.80-ax.get_ylim()[0])/(ax.get_ylim()[1]-ax.get_ylim()[0]) if ax.get_ylim()[1] > ax.get_ylim()[0] else 0,
               ymax=1, alpha=0.05, color="green")
    ax.set_xlabel("Privacy Score"); ax.set_ylabel("Preserve Score")
    ax.set_title("Privacy vs Preserve Trade-off (Adaptive Policy)", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "ve_privacy_preserve_tradeoff.png"), dpi=150); plt.close(fig)
    print("  ✓ ve_privacy_preserve_tradeoff.png")



# ═══════════════════════════════════════════════════════════════════════
# V.F — Metric Variability & Runtime
# ═══════════════════════════════════════════════════════════════════════

def vf_fixed_variance(data, out):
    """Bar: Std-dev of privacy & preserve scores for Fixed pipeline across versions."""
    labels, priv_std, pres_std = [], [], []
    for ver, info in data.items():
        sf = sc(info["fixed"])
        if not sf: continue
        labels.append(ver)
        priv_std.append(np.std([c["metrics"]["privacy"]["privacy_score"] for c in sf]))
        pres_std.append(np.std([c["metrics"]["utility"]["preserve_score"] for c in sf]))

    if not labels:
        print("  ⚠ Skipping vf_fixed_variance (no fixed data)")
        return

    x = np.arange(len(labels)); w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, priv_std, w, label="Privacy Score σ", color="#E53935", alpha=0.8)
    ax.bar(x + w/2, pres_std, w, label="Preserve Score σ", color="#1976D2", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Standard Deviation")
    ax.set_title("Score Variability in Fixed Policy Pipeline", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "vf_fixed_variance.png"), dpi=150); plt.close(fig)
    print("  ✓ vf_fixed_variance.png")


def vf_runtime_comparison(data, out):
    """Bar: Average runtime per version (Adaptive)."""
    labels, runtimes = [], []
    for ver, info in data.items():
        labels.append(ver)
        times = [r.get("total_runtime_seconds", 0) for r in info["adaptive"]
                 if r.get("total_runtime_seconds")]
        runtimes.append(np.mean(times) if times else 0)

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(labels, runtimes, color=C_ADAPTIVE[:len(labels)], alpha=0.85)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f"{bar.get_height():.1f}s", ha="center", fontweight="bold")
    ax.set_ylabel("Average Runtime (seconds)")
    ax.set_title("Runtime Comparison (Adaptive Policy)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "vf_runtime_comparison.png"), dpi=150); plt.close(fig)
    print("  ✓ vf_runtime_comparison.png")


def vf_memory_ablation(data, out):
    """Bar: V3 (with memory) vs V5 (no memory) — privacy & preserve comparison."""
    v3_key = "LLM-H4.5-Mem\n(SS-enabled)"
    v5_key = "LLM-H4.5-NoMem\n(SS-enabled)"
    if v3_key not in data or v5_key not in data:
        print("  ⚠ Skipping vf_memory_ablation (missing data)")
        return

    v3_a = sc(data[v3_key]["adaptive"])
    v5_a = sc(data[v5_key]["adaptive"])

    metrics = ["Privacy Score", "Preserve Score", "WER", "CER", "Speaker Privacy"]
    def vals(chunks):
        return [
            np.mean([c["metrics"]["privacy"]["privacy_score"] for c in chunks]) if chunks else 0,
            np.mean([c["metrics"]["utility"]["preserve_score"] for c in chunks]) if chunks else 0,
            np.mean([c["metrics"]["privacy"]["wer"] for c in chunks]) if chunks else 0,
            np.mean([c["metrics"]["privacy"]["cer"] for c in chunks]) if chunks else 0,
            np.mean([c["metrics"]["privacy"]["speaker_privacy"] for c in chunks]) if chunks else 0,
        ]

    v3_v = vals(v3_a); v5_v = vals(v5_a)
    x = np.arange(len(metrics)); w = 0.35
    fig, ax = plt.subplots(figsize=(13, 6))
    b1 = ax.bar(x - w/2, v3_v, w, label="LLM With Memory",    color="#0D47A1", alpha=0.85)
    b2 = ax.bar(x + w/2, v5_v, w, label="LLM Without Memory", color="#FF5722", alpha=0.85)
    for bars in [b1, b2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.15)
    ax.set_title("Memory Ablation: With vs Without Experience Memory", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "vf_memory_ablation.png"), dpi=150); plt.close(fig)
    print("  ✓ vf_memory_ablation.png")


def vf_rule_vs_llm(data, out):
    """Bar: V4 (Rule-Based) vs V3 (LLM) vs V5 (LLM No Memory) comparison."""
    keys = ["LLM-H4.5-Mem\n(SS-enabled)", "Rule-NoMem\n(SS-enabled)", "LLM-H4.5-NoMem\n(SS-enabled)"]
    present = [k for k in keys if k in data]
    if len(present) < 2:
        print("  ⚠ Skipping vf_rule_vs_llm (need at least 2 variants)")
        return

    metrics = ["Privacy Score", "Preserve Score", "WER", "CER"]
    def vals(chunks):
        return [
            np.mean([c["metrics"]["privacy"]["privacy_score"] for c in chunks]) if chunks else 0,
            np.mean([c["metrics"]["utility"]["preserve_score"] for c in chunks]) if chunks else 0,
            np.mean([c["metrics"]["privacy"]["wer"] for c in chunks]) if chunks else 0,
            np.mean([c["metrics"]["privacy"]["cer"] for c in chunks]) if chunks else 0,
        ]

    colors_map = {
        "LLM-H4.5-Mem\n(SS-enabled)": "#0D47A1",
        "Rule-NoMem\n(SS-enabled)": "#4CAF50",
        "LLM-H4.5-NoMem\n(SS-enabled)": "#FF5722",
    }
    label_map = {
        "LLM-H4.5-Mem\n(SS-enabled)": "LLM + Memory",
        "Rule-NoMem\n(SS-enabled)": "Rule-Based",
        "LLM-H4.5-NoMem\n(SS-enabled)": "LLM No Memory",
    }

    x = np.arange(len(metrics)); w = 0.8 / len(present)
    fig, ax = plt.subplots(figsize=(13, 6))
    for i, k in enumerate(present):
        v = vals(sc(data[k]["adaptive"]))
        ax.bar(x + i*w - (len(present)-1)*w/2, v, w,
               label=label_map.get(k, k), color=colors_map.get(k, "#999"), alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.15)
    ax.set_title("Rule-Based vs LLM Agent Comparison", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "vf_rule_vs_llm.png"), dpi=150); plt.close(fig)
    print("  ✓ vf_rule_vs_llm.png")



# ═══════════════════════════════════════════════════════════════════════
# V.F2 — Trial / Retry Analysis
# ═══════════════════════════════════════════════════════════════════════

def vf2_retry_stats_table(data, out):
    """PNG table: % chunks with retry, avg trials, max trials per version."""
    rows = []
    for ver, info in data.items():
        s = sc(info["adaptive"])
        if not s:
            continue
        trial_counts = [c.get("trials", 0) for c in s]
        with_retry = sum(1 for t in trial_counts if t > 1)
        rows.append([
            ver,
            str(len(s)),
            f"{100 * with_retry / len(s):.1f}%",
            f"{np.mean(trial_counts):.2f}",
            f"{np.median(trial_counts):.0f}",
            str(max(trial_counts)),
        ])

    if not rows:
        return
    cols = ["Version", "Speech Chunks", "% With Retry", "Avg Trials", "Median Trials", "Max Trials"]
    fig, ax = plt.subplots(figsize=(16, 1.5 + 0.45 * len(rows)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=cols, loc="center",
                     cellLoc="center", colColours=["#E3F2FD"] * len(cols))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    ax.set_title("Retry Statistics (Adaptive Policy — Speech Chunks Only)", fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "vf2_retry_stats_table.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ vf2_retry_stats_table.png")


def vf2_trials_vs_privacy(data, out):
    """Scatter: trial count vs final privacy score — each point = 1 speech chunk."""
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, (ver, info) in enumerate(data.items()):
        s = sc(info["adaptive"])
        if not s:
            continue
        trials  = [c.get("trials", 0) for c in s]
        privacy = [c["metrics"]["privacy"]["privacy_score"] for c in s]
        # Jitter x slightly so points don't overlap
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(trials))
        ax.scatter(np.array(trials) + jitter, privacy,
                   alpha=0.45, s=25, label=ver, color=C_ADAPTIVE[i])
    ax.axhline(0.65, color="orange", ls="--", lw=1, alpha=0.7, label="High (0.65)")
    ax.axhline(0.80, color="red",    ls="--", lw=1, alpha=0.7, label="Very-high (0.80)")
    ax.set_xlabel("Number of Trials"); ax.set_ylabel("Final Privacy Score")
    ax.set_title("Trial Count vs Final Privacy Score", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "vf2_trials_vs_privacy.png"), dpi=150)
    plt.close(fig)
    print("  ✓ vf2_trials_vs_privacy.png")


def vf2_improvement_curve(data, out):
    """Line chart: avg privacy & preserve score grouped by trial count.

    For each trial count (1, 2, 3, 4), compute the average final privacy
    and preserve scores of chunks that used exactly that many trials.
    This shows whether more trials lead to better outcomes.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for i, (ver, info) in enumerate(data.items()):
        s = sc(info["adaptive"])
        if not s:
            continue
        # Group by trial count
        by_trial: dict[int, list] = {}
        for c in s:
            t = c.get("trials", 0)
            if t < 1:
                continue
            by_trial.setdefault(t, []).append(c)

        trial_nums = sorted(by_trial.keys())
        if not trial_nums:
            continue
        avg_priv = [np.mean([c["metrics"]["privacy"]["privacy_score"] for c in by_trial[t]]) for t in trial_nums]
        avg_pres = [np.mean([c["metrics"]["utility"]["preserve_score"] for c in by_trial[t]]) for t in trial_nums]
        counts   = [len(by_trial[t]) for t in trial_nums]

        ax1.plot(trial_nums, avg_priv, "o-", label=ver, color=C_ADAPTIVE[i], linewidth=2, markersize=6)
        ax2.plot(trial_nums, avg_pres, "s-", label=ver, color=C_ADAPTIVE[i], linewidth=2, markersize=6)

        # Annotate counts
        for t, p, n in zip(trial_nums, avg_priv, counts):
            ax1.annotate(f"n={n}", (t, p), textcoords="offset points",
                         xytext=(5, 5), color=C_ADAPTIVE[i])

    ax1.axhline(0.65, color="orange", ls="--", lw=1, alpha=0.7)
    ax1.axhline(0.80, color="red",    ls="--", lw=1, alpha=0.7)
    ax1.set_xlabel("Number of Trials"); ax1.set_ylabel("Avg Privacy Score")
    ax1.set_title("Privacy Score by Trial Count", fontweight="bold")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.axhline(0.80, color="red", ls="--", lw=1, alpha=0.7)
    ax2.set_xlabel("Number of Trials"); ax2.set_ylabel("Avg Preserve Score")
    ax2.set_title("Preserve Score by Trial Count", fontweight="bold")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle("Score Improvement Curve by Trial Count", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "vf2_improvement_curve.png"), dpi=150)
    plt.close(fig)
    print("  ✓ vf2_improvement_curve.png")


def vf2_tradeoff_by_trials(data, out):
    """Scatter: Preserve (x) vs Privacy (y), colored by trial count.

    Each point = one speech chunk. Color intensity = number of trials used.
    Shows whether chunks needing more trials cluster in specific regions.
    """
    fig, axes = plt.subplots(1, min(len(data), 5), figsize=(5*min(len(data), 5), 5),
                              sharey=True, squeeze=False)
    axes = axes[0]

    for i, (ver, info) in enumerate(data.items()):
        if i >= 5:
            break
        ax = axes[i]
        s = sc(info["adaptive"])
        if not s:
            ax.set_title(ver); continue

        preserve = [c["metrics"]["utility"]["preserve_score"] for c in s]
        privacy  = [c["metrics"]["privacy"]["privacy_score"]  for c in s]
        trials   = [c.get("trials", 1) for c in s]

        scatter = ax.scatter(preserve, privacy, c=trials, cmap="YlOrRd",
                             s=30, alpha=0.7, vmin=1, vmax=4, edgecolors="grey", linewidths=0.3)
        ax.axhline(0.65, color="orange", ls="--", lw=0.8, alpha=0.5)
        ax.axvline(0.80, color="red",    ls="--", lw=0.8, alpha=0.5)
        ax.set_xlabel("Preserve Score"); ax.set_title(ver)
        if i == 0:
            ax.set_ylabel("Privacy Score")

    # Colorbar
    cbar = fig.colorbar(scatter, ax=axes[-1], shrink=0.8)
    cbar.set_label("Trials")

    fig.suptitle("Privacy–Utility Trade-off Colored by Trial Count", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "vf2_tradeoff_by_trials.png"), dpi=150)
    plt.close(fig)
    print("  ✓ vf2_tradeoff_by_trials.png")


# ═══════════════════════════════════════════════════════════════════════
# V.G — Summary Table
# ═══════════════════════════════════════════════════════════════════════

def vg_summary_table(data, out):
    """Generate summary table as text + PNG."""
    rows = []
    for ver, info in data.items():
        sa = sc(info["adaptive"])
        sf = sc(info["fixed"])
        all_a = all_chunks(info["adaptive"])
        recipes = {}
        for c in sa:
            r = c.get("recipe_applied", {})
            if r:
                name = r.get("recipe_name", "?")
                recipes[name] = recipes.get(name, 0) + 1

        avg_trials = np.mean([c.get("trials", 0) for c in sa if c.get("trials", 0) > 0]) if sa else 0
        avg_runtime = np.mean([r.get("total_runtime_seconds", 0) for r in info["adaptive"]
                               if r.get("total_runtime_seconds")]) or 0

        rows.append({
            "Version": ver,
            "Files": len(info["adaptive"]),
            "Total Chunks": len(all_a),
            "Speech Chunks": len(sa),
            "Avg Privacy (A)": f"{np.mean([c['metrics']['privacy']['privacy_score'] for c in sa]):.4f}" if sa else "N/A",
            "Avg Preserve (A)": f"{np.mean([c['metrics']['utility']['preserve_score'] for c in sa]):.4f}" if sa else "N/A",
            "Avg Privacy (F)": f"{np.mean([c['metrics']['privacy']['privacy_score'] for c in sf]):.4f}" if sf else "N/A",
            "Avg Preserve (F)": f"{np.mean([c['metrics']['utility']['preserve_score'] for c in sf]):.4f}" if sf else "N/A",
            "Avg Trials": f"{avg_trials:.2f}",
            "Avg Runtime (s)": f"{avg_runtime:.1f}",
            "Recipes": ", ".join(f"{k.replace('RECIPE_','')}:{v}" for k, v in sorted(recipes.items())),
        })

    # Save as text
    with open(os.path.join(out, "summary.txt"), "w") as f:
        for row in rows:
            f.write(f"\n{'='*70}\n")
            for k, v in row.items():
                f.write(f"  {k}: {v}\n")
    print("  ✓ summary.txt")

    # Save as PNG table
    if not rows:
        return
    col_keys = list(rows[0].keys())
    cell_text = [[row[k] for k in col_keys] for row in rows]

    fig, ax = plt.subplots(figsize=(20, 2 + 0.5 * len(rows)))
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=col_keys, loc="center",
                     cellLoc="center", colColours=["#E3F2FD"] * len(col_keys))
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.5)
    ax.set_title("Summary: All Pipeline Versions", fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "vg_summary_table.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ vg_summary_table.png")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Plot charts for paper RQ1-RQ3 across all pipeline versions")
    parser.add_argument("--out-dir", default="plots/paper_rq",
                        help="Output directory (default: plots/paper_rq)")
    args = parser.parse_args()

    out = args.out_dir
    os.makedirs(out, exist_ok=True)

    print("Loading data...")
    data = load_all()
    if not data:
        print("ERROR: No data loaded"); sys.exit(1)

    print(f"\n{'='*60}")
    print("V.A — Privacy Performance Comparison (RQ1)")
    print(f"{'='*60}")
    va_avg_privacy_score(data, out)
    va_privacy_boxplot(data, out)
    va_privacy_sub_metrics(data, out)
    va_pass_rate(data, out)

    print(f"\n{'='*60}")
    print("V.B — Effect of Speech Characteristics (RQ2)")
    print(f"{'='*60}")
    vb_speech_ratio_vs_privacy(data, out)
    vb_vad_confidence_vs_privacy(data, out)
    vb_speech_ratio_histogram(data, out)

    print(f"\n{'='*60}")
    print("V.C — Transformation Selection Behavior (RQ2)")
    print(f"{'='*60}")
    vc_recipe_usage(data, out)
    vc_trials_distribution(data, out)
    vc_trials_histogram(data, out)

    print(f"\n{'='*60}")
    print("V.D — Impact of Source Separation (RQ2+RQ3)")
    print(f"{'='*60}")
    vd_source_sep_comparison(data, out)

    print(f"\n{'='*60}")
    print("V.E — Utility Preservation (RQ3)")
    print(f"{'='*60}")
    ve_avg_preserve_score(data, out)
    ve_preserve_boxplot(data, out)
    ve_preserve_sub_scores(data, out)
    ve_classification_metrics(data, out)
    ve_privacy_preserve_tradeoff(data, out)

    print(f"\n{'='*60}")
    print("V.F — Metric Variability & Ablation")
    print(f"{'='*60}")
    vf_fixed_variance(data, out)
    vf_runtime_comparison(data, out)
    vf_memory_ablation(data, out)
    vf_rule_vs_llm(data, out)

    print(f"\n{'='*60}")
    print("V.F2 — Trial / Retry Analysis")
    print(f"{'='*60}")
    vf2_retry_stats_table(data, out)
    vf2_trials_vs_privacy(data, out)
    vf2_improvement_curve(data, out)
    vf2_tradeoff_by_trials(data, out)

    print(f"\n{'='*60}")
    print("V.G — Summary")
    print(f"{'='*60}")
    vg_summary_table(data, out)

    total = sum(1 for f in os.listdir(out) if f.endswith(".png"))
    print(f"\nDone — {total} charts + summary saved to {out}/")


if __name__ == "__main__":
    main()
