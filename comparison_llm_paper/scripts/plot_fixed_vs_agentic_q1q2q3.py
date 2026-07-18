#!/usr/bin/env python3
"""Q1 + Q2 + Q3: Fixed Baseline vs Agentic AI Pipeline comparison.

Generates charts comparing ONLY Fixed Baseline and Agentic AI pipelines
(no separate LLM decision log needed).

Usage:
    python scripts/plot_fixed_vs_agentic_q1q2q3.py <run_log.json> [--out-dir DIR]

Q1 Charts (Privacy Effectiveness):
  1. Avg privacy_score per pipeline
  2. Privacy sub-metrics breakdown (content_privacy, speaker_privacy, WER, CER)
  3. Per-chunk privacy_score (Fixed vs Agentic, per target)
  4. Privacy score box plot
  5. Privacy radar (privacy_score, speaker, content, WER, CER)
  6. Privacy heatmap (chunk × pipeline)
  7. Privacy delta bar (Agentic − Fixed)

Q2 Charts (Speech Confidence → Transformation):
  8. Scatter: speech_ratio vs recipe
  9. Scatter: speech_ratio vs privacy_score
  10. Scatter: speech_ratio vs preserve_score
  11. Grouped bar: avg speech_ratio per recipe
  12. Bubble: speech_ratio vs privacy_score (size=trials)
  13. Heatmap: speech_ratio bins vs recipe (Agentic only)

Q3 Charts (Environmental Preservation):
  14. Classification accuracy (mAP, F1, accuracy)
  15. Preserve sub-scores radar
  16. Preserve stacked bar
  17. Privacy–Preserve trade-off scatter
  18. Preserve delta bar (Agentic − Fixed)
  19. Preserve box plot
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PIPE_SHORT = {
    "Fixed Baseline — privacy_target='high'": "Fixed high",
    "Fixed Baseline — privacy_target='very_high'": "Fixed v.high",
    "Agentic Pipeline — privacy_target='high'": "Agentic high",
    "Agentic Pipeline — privacy_target='very_high'": "Agentic v.high",
}
PIPE_COLORS = {
    "Fixed high": "#2196F3",
    "Fixed v.high": "#1565C0",
    "Agentic high": "#FF9800",
    "Agentic v.high": "#E91E63",
}
RECIPE_COLORS = {
    "RECIPE_MID_BAND_ATTEN": "#2196F3",
    "RECIPE_LOWPASS_HIGHPASS_MIX": "#E91E63",
    None: "#9E9E9E",
}
RECIPE_SHORT = {
    "RECIPE_MID_BAND_ATTEN": "MidBand",
    "RECIPE_LOWPASS_HIGHPASS_MIX": "StrongBlur",
    None: "No transform",
}


def load(path):
    with open(path) as f:
        return json.load(f)


def short(label):
    return PIPE_SHORT.get(label, label[:20])


def speech_chunks(pipe):
    return [c for c in pipe["chunks"] if c.get("had_speech") and "privacy" in c]


# ═══════════════════════════════════════════════════════════════════════
# Q1: Privacy Effectiveness
# ═══════════════════════════════════════════════════════════════════════

def q1_avg_privacy_score(data, out_dir):
    """Chart 1: Avg privacy_score per pipeline."""
    entries = []
    for pipe in data["pipelines"]:
        label = short(pipe["label"])
        sc = speech_chunks(pipe)
        if not sc:
            continue
        avg = np.mean([c["privacy"]["privacy_score"] for c in sc])
        entries.append((label, avg, PIPE_COLORS.get(label, "#999")))

    labels = [e[0] for e in entries]
    vals = [e[1] for e in entries]
    colors = [e[2] for e in entries]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, 0.6, color=colors, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(0.65, color="orange", ls="--", lw=1.2, alpha=0.7, label="high threshold (0.65)")
    ax.axhline(0.80, color="red", ls="--", lw=1.2, alpha=0.7, label="very_high threshold (0.80)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Avg Privacy Score", fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_title("Q1: Average Privacy Score — Fixed vs Agentic", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "q1_avg_privacy_score.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")


def q1_privacy_sub_metrics(data, out_dir):
    """Chart 2: Privacy sub-metrics breakdown."""
    entries = []
    for pipe in data["pipelines"]:
        label = short(pipe["label"])
        sc = speech_chunks(pipe)
        if not sc:
            continue
        entries.append({
            "label": label,
            "content_privacy": np.mean([c["privacy"]["content_privacy"] for c in sc]),
            "speaker_privacy": np.mean([c["privacy"]["speaker_privacy"] for c in sc]),
            "wer": np.mean([c["privacy"]["wer"] for c in sc]),
            "cer": np.mean([c["privacy"]["cer"] for c in sc]),
        })

    labels = [e["label"] for e in entries]
    x = np.arange(len(labels))
    width = 0.18
    metrics = [
        ("content_privacy", "Content Privacy", "#7E57C2"),
        ("speaker_privacy", "Speaker Privacy", "#EF5350"),
        ("wer", "WER", "#42A5F5"),
        ("cer", "CER", "#66BB6A"),
    ]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, (key, mlabel, color) in enumerate(metrics):
        vals = [e[key] for e in entries]
        bars = ax.bar(x + i * width, vals, width, label=mlabel, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title("Q1: Privacy Sub-Metrics — Fixed vs Agentic", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "q1_privacy_sub_metrics.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")


def q1_per_chunk_privacy(data, out_dir):
    """Chart 3: Per-chunk privacy_score Fixed vs Agentic."""
    pipe_map = {short(p["label"]): p for p in data["pipelines"]}
    for target, fk, ak in [("high", "Fixed high", "Agentic high"),
                            ("very_high", "Fixed v.high", "Agentic v.high")]:
        if fk not in pipe_map or ak not in pipe_map:
            continue
        fc = pipe_map[fk]["chunks"]
        ac = pipe_map[ak]["chunks"]
        n = min(len(fc), len(ac))
        idx = [i for i in range(n)
               if fc[i].get("had_speech") and "privacy" in fc[i]
               and ac[i].get("had_speech") and "privacy" in ac[i]]
        if not idx:
            continue

        fv = [fc[i]["privacy"]["privacy_score"] for i in idx]
        av = [ac[i]["privacy"]["privacy_score"] for i in idx]
        x = np.arange(len(idx))
        w = 0.35

        fig, ax = plt.subplots(figsize=(max(8, len(idx) * 1.5), 5))
        b1 = ax.bar(x - w / 2, fv, w, label=fk, color=PIPE_COLORS[fk], alpha=0.85)
        b2 = ax.bar(x + w / 2, av, w, label=ak, color=PIPE_COLORS[ak], alpha=0.85)
        for bars, vals in [(b1, fv), (b2, av)]:
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)

        thresh = 0.65 if target == "high" else 0.80
        ax.axhline(thresh, color="red", ls="--", lw=1.2, alpha=0.6, label=f"threshold ({thresh})")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Chunk {i}" for i in idx], fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Privacy Score", fontsize=11)
        ax.set_title(f"Q1: Per-Chunk Privacy — Fixed vs Agentic ({target})",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, f"q1_per_chunk_privacy_{target}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  ✓ {path}")


def q1_privacy_boxplot(data, out_dir):
    """Chart 4: Privacy score box plot."""
    labels, box_data, colors = [], [], []
    for pipe in data["pipelines"]:
        label = short(pipe["label"])
        sc = speech_chunks(pipe)
        if not sc:
            continue
        labels.append(label)
        box_data.append([c["privacy"]["privacy_score"] for c in sc])
        colors.append(PIPE_COLORS.get(label, "#999"))

    if not box_data:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(box_data, patch_artist=True, tick_labels=labels)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.axhline(0.65, color="orange", ls="--", lw=1, alpha=0.6, label="high threshold")
    ax.axhline(0.80, color="red", ls="--", lw=1, alpha=0.6, label="very_high threshold")
    ax.set_ylabel("Privacy Score", fontsize=11)
    ax.set_title("Q1: Privacy Score Distribution — Fixed vs Agentic", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    path = os.path.join(out_dir, "q1_privacy_boxplot.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")


def q1_privacy_radar(data, out_dir):
    """Chart 5: Privacy radar."""
    cats = ["privacy_score", "speaker_privacy", "content_privacy", "wer", "cer"]
    cat_labels = ["Privacy\nScore", "Speaker\nPrivacy", "Content\nPrivacy", "WER", "CER"]
    N = len(cats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for pipe in data["pipelines"]:
        label = short(pipe["label"])
        color = PIPE_COLORS.get(label, "#999")
        sc = speech_chunks(pipe)
        if not sc:
            continue
        vals = [np.mean([c["privacy"][cat] for c in sc]) for cat in cats]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", color=color, lw=2, label=label)
        ax.fill(angles, vals, color=color, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cat_labels, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_title("Q1: Privacy Radar — Fixed vs Agentic", fontsize=13, fontweight="bold", pad=20)
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.35, 1.1))
    fig.tight_layout()
    path = os.path.join(out_dir, "q1_privacy_radar.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")


def q1_privacy_heatmap(data, out_dir):
    """Chart 6: Privacy heatmap."""
    pipelines = data["pipelines"]
    labels = [short(p["label"]) for p in pipelines]
    n_chunks = len(pipelines[0]["chunks"])
    matrix = []
    for pipe in pipelines:
        row = []
        for c in pipe["chunks"]:
            if c.get("had_speech") and "privacy" in c:
                row.append(c["privacy"]["privacy_score"])
            else:
                row.append(0.0)
        matrix.append(row)
    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(max(8, n_chunks * 1.2), max(4, len(labels) * 0.8)))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(n_chunks))
    ax.set_xticklabels([f"Chunk {i}" for i in range(n_chunks)], fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(n_chunks):
            v = matrix[i, j]
            color = "white" if v < 0.5 or v > 0.85 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=color)
    plt.colorbar(im, ax=ax, label="Privacy Score", shrink=0.8)
    ax.set_title("Q1: Privacy Score Heatmap — Fixed vs Agentic", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "q1_privacy_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")


def q1_privacy_delta(data, out_dir):
    """Chart 7: Privacy delta (Agentic − Fixed)."""
    pipe_map = {short(p["label"]): p for p in data["pipelines"]}
    for target, fk, ak in [("high", "Fixed high", "Agentic high"),
                            ("very_high", "Fixed v.high", "Agentic v.high")]:
        if fk not in pipe_map or ak not in pipe_map:
            continue
        fc = pipe_map[fk]["chunks"]
        ac = pipe_map[ak]["chunks"]
        n = min(len(fc), len(ac))
        idx = [i for i in range(n)
               if fc[i].get("had_speech") and "privacy" in fc[i]
               and ac[i].get("had_speech") and "privacy" in ac[i]]
        if not idx:
            continue

        deltas = [ac[i]["privacy"]["privacy_score"] - fc[i]["privacy"]["privacy_score"] for i in idx]
        x = np.arange(len(idx))
        colors = ["#4CAF50" if d >= 0 else "#EF5350" for d in deltas]

        fig, ax = plt.subplots(figsize=(max(8, len(idx) * 1.2), 5))
        bars = ax.bar(x, deltas, 0.6, color=colors, alpha=0.85)
        for bar, v in zip(bars, deltas):
            sign = "+" if v >= 0 else ""
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.005 if v >= 0 else -0.02),
                    f"{sign}{v:.3f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=8, fontweight="bold")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Chunk {i}" for i in idx], fontsize=8)
        ax.set_ylabel("Δ Privacy Score (Agentic − Fixed)", fontsize=10)
        ax.set_title(f"Q1: Privacy Improvement — Agentic vs Fixed ({target})",
                     fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, f"q1_privacy_delta_{target}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  ✓ {path}")


# ═══════════════════════════════════════════════════════════════════════
# Q2: Speech Confidence → Transformation
# ═══════════════════════════════════════════════════════════════════════

def _extract_rows(data):
    rows = []
    for pipe in data["pipelines"]:
        label = short(pipe["label"])
        for c in pipe["chunks"]:
            if not c.get("had_speech"):
                continue
            rows.append({
                "pipeline": label,
                "speech_ratio": c.get("speech_ratio", 0),
                "vad_confidence": c.get("vad_confidence", 0),
                "recipe": c.get("recipe"),
                "trials": c.get("trials", 1),
                "privacy_score": c.get("privacy", {}).get("privacy_score", 0),
                "preserve_score": c.get("preserve", {}).get("preserve_score", 0),
            })
    return rows


def q2_speech_ratio_vs_recipe(rows, out_dir):
    """Chart 8: Scatter speech_ratio vs recipe."""
    fig, ax = plt.subplots(figsize=(10, 5))
    recipe_names = sorted({r["recipe"] for r in rows if r["recipe"]})
    recipe_idx = {r: i for i, r in enumerate(recipe_names)}

    for pipe_label, color in PIPE_COLORS.items():
        subset = [r for r in rows if r["pipeline"] == pipe_label and r["recipe"]]
        if not subset:
            continue
        x = [s["speech_ratio"] for s in subset]
        y = [recipe_idx[s["recipe"]] + np.random.uniform(-0.15, 0.15) for s in subset]
        ax.scatter(x, y, c=color, alpha=0.7, s=50, label=pipe_label, edgecolors="white", lw=0.5)

    ax.set_yticks(range(len(recipe_names)))
    ax.set_yticklabels([RECIPE_SHORT.get(r, r) for r in recipe_names], fontsize=10)
    ax.set_xlabel("Speech Ratio", fontsize=11)
    ax.set_title("Q2: Speech Ratio → Recipe Selection — Fixed vs Agentic",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    fig.tight_layout()
    path = os.path.join(out_dir, "q2_speech_ratio_vs_recipe.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")


def q2_speech_ratio_vs_scores(rows, out_dir):
    """Charts 9 & 10: speech_ratio vs privacy/preserve scores."""
    for metric, ylabel, title, thresh in [
        ("privacy_score", "Privacy Score", "Q2: Speech Ratio vs Privacy Score", None),
        ("preserve_score", "Preserve Score", "Q2: Speech Ratio vs Preserve Score", 0.80),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for recipe, color in RECIPE_COLORS.items():
            subset = [r for r in rows if r["recipe"] == recipe]
            if not subset:
                continue
            x = [s["speech_ratio"] for s in subset]
            y = [s[metric] for s in subset]
            ax.scatter(x, y, c=color, alpha=0.7, s=50,
                       label=RECIPE_SHORT.get(recipe, str(recipe)), edgecolors="white", lw=0.5)
        if thresh:
            ax.axhline(thresh, color="red", ls="--", lw=1, alpha=0.6, label=f"threshold ({thresh})")
        ax.set_xlabel("Speech Ratio", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{title} — Fixed vs Agentic", fontsize=13, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0, 1.1)
        fig.tight_layout()
        safe = metric.replace("_", "")
        path = os.path.join(out_dir, f"q2_speech_ratio_vs_{safe}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  ✓ {path}")


def q2_avg_speech_ratio_by_recipe(rows, out_dir):
    """Chart 11: Avg speech_ratio per recipe per pipeline."""
    pipelines = sorted({r["pipeline"] for r in rows})
    recipes = sorted({r["recipe"] for r in rows if r["recipe"]})
    if not recipes:
        return
    x = np.arange(len(recipes))
    width = 0.8 / max(len(pipelines), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, pipe in enumerate(pipelines):
        avgs = []
        for recipe in recipes:
            subset = [r for r in rows if r["pipeline"] == pipe and r["recipe"] == recipe]
            avgs.append(np.mean([s["speech_ratio"] for s in subset]) if subset else 0)
        bars = ax.bar(x + i * width, avgs, width, label=pipe,
                      color=PIPE_COLORS.get(pipe, "#999"), alpha=0.85)
        for bar, v in zip(bars, avgs):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x + width * (len(pipelines) - 1) / 2)
    ax.set_xticklabels([RECIPE_SHORT.get(r, r) for r in recipes], fontsize=10)
    ax.set_ylabel("Avg Speech Ratio", fontsize=11)
    ax.set_title("Q2: Avg Speech Ratio by Recipe — Fixed vs Agentic",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    path = os.path.join(out_dir, "q2_avg_speech_ratio_by_recipe.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")


def q2_bubble_speech_privacy(rows, out_dir):
    """Chart 12: Bubble — speech_ratio vs privacy_score, size=trials."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for pipe_label, color in PIPE_COLORS.items():
        subset = [r for r in rows if r["pipeline"] == pipe_label]
        if not subset:
            continue
        x = [s["speech_ratio"] for s in subset]
        y = [s["privacy_score"] for s in subset]
        sizes = [s["trials"] * 60 for s in subset]
        ax.scatter(x, y, s=sizes, c=color, alpha=0.55, label=pipe_label,
                   edgecolors="white", lw=0.5)
    ax.set_xlabel("Speech Ratio", fontsize=11)
    ax.set_ylabel("Privacy Score", fontsize=11)
    ax.set_title("Q2: Speech Ratio vs Privacy Score (size = trials)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    path = os.path.join(out_dir, "q2_bubble_speech_privacy.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")


def q2_heatmap_speech_recipe(rows, out_dir):
    """Chart 13: Heatmap — speech_ratio bins vs recipe (Agentic only)."""
    agentic = [r for r in rows if "Agentic" in r["pipeline"] and r["recipe"]]
    if not agentic:
        return
    bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    bin_labels = ["0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]
    recipes = sorted({r["recipe"] for r in agentic})
    matrix = np.zeros((len(recipes), len(bins)))
    for r in agentic:
        for j, (lo, hi) in enumerate(bins):
            if lo <= r["speech_ratio"] < hi:
                ri = recipes.index(r["recipe"])
                matrix[ri, j] += 1
                break

    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(bin_labels, fontsize=9)
    ax.set_yticks(range(len(recipes)))
    ax.set_yticklabels([RECIPE_SHORT.get(r, r) for r in recipes], fontsize=10)
    ax.set_xlabel("Speech Ratio Bin", fontsize=11)
    for i in range(len(recipes)):
        for j in range(len(bins)):
            v = int(matrix[i, j])
            ax.text(j, i, str(v), ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if v > matrix.max() * 0.6 else "black")
    plt.colorbar(im, ax=ax, label="Count", shrink=0.8)
    ax.set_title("Q2: Speech Ratio × Recipe — Agentic Only",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "q2_heatmap_speech_recipe.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")


# ═══════════════════════════════════════════════════════════════════════
# Q3: Environmental Preservation
# ═══════════════════════════════════════════════════════════════════════

def q3_classification_accuracy(data, out_dir):
    """Chart 14: Classification accuracy (mAP, F1, accuracy)."""
    metrics = ["mAP", "f1", "accuracy"]
    entries = []
    for pipe in data["pipelines"]:
        label = short(pipe["label"])
        sc = speech_chunks(pipe)
        if not sc:
            continue
        vals = {m: np.mean([c["utility"][m] for c in sc if "utility" in c]) for m in metrics}
        entries.append({"label": label, **vals})

    labels = [e["label"] for e in entries]
    x = np.arange(len(labels))
    width = 0.22
    colors_m = ["#42A5F5", "#66BB6A", "#FFA726"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (m, col) in enumerate(zip(metrics, colors_m)):
        vals = [e[m] for e in entries]
        bars = ax.bar(x + i * width, vals, width, label=m.upper(), color=col, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title("Q3: Classification Accuracy — Fixed vs Agentic",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "q3_classification_accuracy.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")


def q3_preserve_radar(data, out_dir):
    """Chart 15: Preserve sub-scores radar."""
    cats = ["s_loud", "s_hf", "s_sc", "s_con", "s_psy"]
    cat_labels = ["Loudness", "High-Freq", "Spectral\nCentroid", "mAP", "Psycho-\nacoustic"]
    N = len(cats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for pipe in data["pipelines"]:
        label = short(pipe["label"])
        color = PIPE_COLORS.get(label, "#999")
        sc = speech_chunks(pipe)
        if not sc:
            continue
        vals = [np.mean([c["preserve"][cat] for c in sc if "preserve" in c]) for cat in cats]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", color=color, lw=2, label=label)
        ax.fill(angles, vals, color=color, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cat_labels, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_title("Q3: Preserve Sub-Scores Radar — Fixed vs Agentic",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.35, 1.1))
    fig.tight_layout()
    path = os.path.join(out_dir, "q3_preserve_radar.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")


def q3_preserve_stacked_bar(data, out_dir):
    """Chart 16: Preserve stacked bar."""
    cats = ["s_loud", "s_hf", "s_sc", "s_con", "s_psy"]
    cat_labels = ["Loudness", "High-Freq", "Spectral Centroid", "mAP", "Psychoacoustic"]
    cat_colors = ["#42A5F5", "#AB47BC", "#26A69A", "#FFA726", "#EF5350"]

    entries = []
    for pipe in data["pipelines"]:
        label = short(pipe["label"])
        sc = speech_chunks(pipe)
        if not sc:
            continue
        vals = {cat: np.mean([c["preserve"][cat] for c in sc if "preserve" in c]) for cat in cats}
        entries.append({"label": label, **vals})

    labels = [e["label"] for e in entries]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bottom = np.zeros(len(labels))
    for cat, clabel, color in zip(cats, cat_labels, cat_colors):
        vals = np.array([e[cat] for e in entries])
        ax.bar(x, vals, 0.6, bottom=bottom, label=clabel, color=color, alpha=0.85)
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v > 0.05:
                ax.text(i, b + v / 2, f"{v:.2f}", ha="center", va="center", fontsize=7)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Cumulative Sub-Score", fontsize=11)
    ax.set_title("Q3: Preserve Sub-Scores Stacked — Fixed vs Agentic",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "q3_preserve_stacked_bar.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")


def q3_privacy_preserve_tradeoff(data, out_dir):
    """Chart 17: Privacy–Preserve trade-off scatter."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for pipe in data["pipelines"]:
        label = short(pipe["label"])
        color = PIPE_COLORS.get(label, "#999")
        sc = speech_chunks(pipe)
        if not sc:
            continue
        x = [c["privacy"]["privacy_score"] for c in sc]
        y = [c["preserve"]["preserve_score"] for c in sc]
        ax.scatter(x, y, c=color, alpha=0.7, s=55, label=label, edgecolors="white", lw=0.5)

    ax.axhline(0.80, color="green", ls="--", lw=1, alpha=0.5, label="preserve ≥ 0.80")
    ax.axvline(0.65, color="orange", ls="--", lw=1, alpha=0.5, label="privacy ≥ 0.65")
    ax.set_xlabel("Privacy Score", fontsize=11)
    ax.set_ylabel("Preserve Score", fontsize=11)
    ax.set_title("Q3: Privacy–Preserve Trade-off — Fixed vs Agentic",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    path = os.path.join(out_dir, "q3_privacy_preserve_tradeoff.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")


def q3_preserve_delta(data, out_dir):
    """Chart 18: Preserve delta (Agentic − Fixed)."""
    pipe_map = {short(p["label"]): p for p in data["pipelines"]}
    for target, fk, ak in [("high", "Fixed high", "Agentic high"),
                            ("very_high", "Fixed v.high", "Agentic v.high")]:
        if fk not in pipe_map or ak not in pipe_map:
            continue
        fc = pipe_map[fk]["chunks"]
        ac = pipe_map[ak]["chunks"]
        n = min(len(fc), len(ac))
        idx = [i for i in range(n)
               if fc[i].get("had_speech") and "preserve" in fc[i]
               and ac[i].get("had_speech") and "preserve" in ac[i]]
        if not idx:
            continue

        deltas = [ac[i]["preserve"]["preserve_score"] - fc[i]["preserve"]["preserve_score"]
                  for i in idx]
        x = np.arange(len(idx))
        colors = ["#4CAF50" if d >= 0 else "#EF5350" for d in deltas]

        fig, ax = plt.subplots(figsize=(max(8, len(idx) * 1.2), 5))
        bars = ax.bar(x, deltas, 0.6, color=colors, alpha=0.85)
        for bar, v in zip(bars, deltas):
            sign = "+" if v >= 0 else ""
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.005 if v >= 0 else -0.02),
                    f"{sign}{v:.3f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=8, fontweight="bold")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Chunk {i}" for i in idx], fontsize=8)
        ax.set_ylabel("Δ Preserve Score (Agentic − Fixed)", fontsize=10)
        ax.set_title(f"Q3: Preserve Improvement — Agentic vs Fixed ({target})",
                     fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, f"q3_preserve_delta_{target}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  ✓ {path}")


def q3_preserve_boxplot(data, out_dir):
    """Chart 19: Preserve score box plot."""
    labels, box_data, colors = [], [], []
    for pipe in data["pipelines"]:
        label = short(pipe["label"])
        sc = speech_chunks(pipe)
        if not sc:
            continue
        labels.append(label)
        box_data.append([c["preserve"]["preserve_score"] for c in sc if "preserve" in c])
        colors.append(PIPE_COLORS.get(label, "#999"))

    if not box_data:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(box_data, patch_artist=True, tick_labels=labels)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.axhline(0.80, color="red", ls="--", lw=1, alpha=0.6, label="preserve threshold (0.80)")
    ax.set_ylabel("Preserve Score", fontsize=11)
    ax.set_title("Q3: Preserve Score Distribution — Fixed vs Agentic",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    path = os.path.join(out_dir, "q3_preserve_boxplot.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Q1+Q2+Q3: Fixed vs Agentic comparison charts")
    parser.add_argument("run_log", help="Path to run log JSON")
    parser.add_argument("--out-dir", default="plots/fixed_vs_agentic_q1q2q3",
                        help="Output directory (default: plots/fixed_vs_agentic_q1q2q3)")
    args = parser.parse_args()

    data = load(args.run_log)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("Q1: Privacy Effectiveness")
    print("=" * 60)
    q1_avg_privacy_score(data, out_dir)
    q1_privacy_sub_metrics(data, out_dir)
    q1_per_chunk_privacy(data, out_dir)
    q1_privacy_boxplot(data, out_dir)
    q1_privacy_radar(data, out_dir)
    q1_privacy_heatmap(data, out_dir)
    q1_privacy_delta(data, out_dir)

    print()
    print("=" * 60)
    print("Q2: Speech Confidence → Transformation")
    print("=" * 60)
    rows = _extract_rows(data)
    q2_speech_ratio_vs_recipe(rows, out_dir)
    q2_speech_ratio_vs_scores(rows, out_dir)
    q2_avg_speech_ratio_by_recipe(rows, out_dir)
    q2_bubble_speech_privacy(rows, out_dir)
    q2_heatmap_speech_recipe(rows, out_dir)

    print()
    print("=" * 60)
    print("Q3: Environmental Preservation")
    print("=" * 60)
    q3_classification_accuracy(data, out_dir)
    q3_preserve_radar(data, out_dir)
    q3_preserve_stacked_bar(data, out_dir)
    q3_privacy_preserve_tradeoff(data, out_dir)
    q3_preserve_delta(data, out_dir)
    q3_preserve_boxplot(data, out_dir)

    # Count output
    total = sum(1 for f in os.listdir(out_dir) if f.endswith(".png"))
    print(f"\nDone — {total} charts saved to {out_dir}/")


if __name__ == "__main__":
    main()
