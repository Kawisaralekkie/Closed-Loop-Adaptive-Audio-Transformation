#!/usr/bin/env python3
"""Plot v1 vs v2 parameter comparison charts.

Usage:
    python scripts/plot_v1_vs_v2.py <run_v1v2_log.json> [--out-dir DIR]
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "v1 MidBand (high)": "#2196F3",
    "v2 MidBand (high)": "#FF9800",
    "v1 StrongBlur (very_high)": "#1565C0",
    "v2 StrongBlur (very_high)": "#E91E63",
}


def load(path):
    with open(path) as f:
        return json.load(f)


def speech_chunks(pipe):
    return [c for c in pipe["chunks"] if c.get("had_speech") and "privacy" in c]


# ── Chart 1: Avg Privacy & Preserve Score ─────────────────────────────

def plot_avg_scores(data, out_dir):
    labels, priv, pres, colors = [], [], [], []
    for pipe in data["pipelines"]:
        label = pipe["label"]
        sc = speech_chunks(pipe)
        if not sc:
            continue
        labels.append(label)
        priv.append(np.mean([c["privacy"]["privacy_score"] for c in sc]))
        pres.append(np.mean([c["preserve"]["preserve_score"] for c in sc]))
        colors.append(COLORS.get(label, "#999"))

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x - w/2, priv, w, label="Privacy Score", color=colors, alpha=0.85)
    b2 = ax.bar(x + w/2, pres, w, label="Preserve Score",
                color=[c + "80" for c in colors], alpha=0.6, edgecolor=colors, linewidth=2)
    for bars, vals in [(b1, priv), (b2, pres)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(0.65, color="orange", ls="--", lw=1, alpha=0.6, label="privacy high (0.65)")
    ax.axhline(0.80, color="red", ls="--", lw=1, alpha=0.6, label="privacy very_high / preserve (0.80)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title("v1 vs v2: Average Privacy & Preserve Score", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "v1v2_avg_scores.png"), dpi=150)
    plt.close(fig)
    print(f"  ✓ v1v2_avg_scores.png")


# ── Chart 2: Per-chunk Privacy Score ──────────────────────────────────

def plot_per_chunk_privacy(data, out_dir):
    for tool in ["MidBand", "StrongBlur"]:
        v1 = [p for p in data["pipelines"] if p["label"].startswith(f"v1 {tool}")][0]
        v2 = [p for p in data["pipelines"] if p["label"].startswith(f"v2 {tool}")][0]
        sc1, sc2 = speech_chunks(v1), speech_chunks(v2)
        n = min(len(sc1), len(sc2))
        if n == 0:
            continue
        vals1 = [sc1[i]["privacy"]["privacy_score"] for i in range(n)]
        vals2 = [sc2[i]["privacy"]["privacy_score"] for i in range(n)]
        x = np.arange(n)
        w = 0.35
        fig, ax = plt.subplots(figsize=(max(8, n * 2), 5))
        b1 = ax.bar(x - w/2, vals1, w, label=f"v1 {tool}", color=COLORS[v1["label"]], alpha=0.85)
        b2 = ax.bar(x + w/2, vals2, w, label=f"v2 {tool}", color=COLORS[v2["label"]], alpha=0.85)
        for bars, vals in [(b1, vals1), (b2, vals2)]:
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        thresh = 0.65 if "MidBand" in tool else 0.80
        ax.axhline(thresh, color="red", ls="--", lw=1.2, alpha=0.6, label=f"threshold ({thresh})")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Speech {i+1}" for i in range(n)], fontsize=10)
        ax.set_ylabel("Privacy Score", fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.set_title(f"v1 vs v2: Per-Chunk Privacy — {tool}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"v1v2_per_chunk_privacy_{tool.lower()}.png"), dpi=150)
        plt.close(fig)
        print(f"  ✓ v1v2_per_chunk_privacy_{tool.lower()}.png")


# ── Chart 3: Privacy Sub-metrics ──────────────────────────────────────

def plot_privacy_sub_metrics(data, out_dir):
    entries = []
    for pipe in data["pipelines"]:
        sc = speech_chunks(pipe)
        if not sc:
            continue
        entries.append({
            "label": pipe["label"],
            "WER": np.mean([c["privacy"]["wer"] for c in sc]),
            "CER": np.mean([c["privacy"]["cer"] for c in sc]),
            "Speaker Privacy": np.mean([c["privacy"]["speaker_privacy"] for c in sc]),
            "Content Privacy": np.mean([c["privacy"]["content_privacy"] for c in sc]),
        })
    labels = [e["label"] for e in entries]
    metrics = [("WER", "#42A5F5"), ("CER", "#66BB6A"), ("Speaker Privacy", "#EF5350"), ("Content Privacy", "#7E57C2")]
    x = np.arange(len(labels))
    w = 0.18
    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (m, color) in enumerate(metrics):
        vals = [e[m] for e in entries]
        bars = ax.bar(x + i * w, vals, w, label=m, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title("v1 vs v2: Privacy Sub-Metrics", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "v1v2_privacy_sub_metrics.png"), dpi=150)
    plt.close(fig)
    print(f"  ✓ v1v2_privacy_sub_metrics.png")


# ── Chart 4: Preserve Sub-scores Radar ────────────────────────────────

def plot_preserve_radar(data, out_dir):
    cats = ["s_loud", "s_hf", "s_sc", "s_con", "s_psy"]
    cat_labels = ["Loudness", "High-Freq", "Spectral\nCentroid", "mAP", "Psycho-\nacoustic"]
    N = len(cats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for pipe in data["pipelines"]:
        label = pipe["label"]
        color = COLORS.get(label, "#999")
        sc = speech_chunks(pipe)
        if not sc:
            continue
        vals = [np.mean([c["preserve"][cat] for c in sc]) for cat in cats]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", color=color, lw=2, label=label)
        ax.fill(angles, vals, color=color, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cat_labels, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title("v1 vs v2: Preserve Sub-Scores Radar", fontsize=13, fontweight="bold", pad=20)
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.4, 1.1))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "v1v2_preserve_radar.png"), dpi=150)
    plt.close(fig)
    print(f"  ✓ v1v2_preserve_radar.png")


# ── Chart 5: Privacy–Preserve Trade-off Scatter ──────────────────────

def plot_tradeoff(data, out_dir):
    fig, ax = plt.subplots(figsize=(10, 7))
    for pipe in data["pipelines"]:
        label = pipe["label"]
        color = COLORS.get(label, "#999")
        sc = speech_chunks(pipe)
        if not sc:
            continue
        x = [c["privacy"]["privacy_score"] for c in sc]
        y = [c["preserve"]["preserve_score"] for c in sc]
        ax.scatter(x, y, c=color, s=80, alpha=0.8, label=label, edgecolors="white", lw=0.8)
    ax.axhline(0.80, color="green", ls="--", lw=1, alpha=0.5, label="preserve ≥ 0.80")
    ax.axvline(0.65, color="orange", ls="--", lw=1, alpha=0.5, label="privacy ≥ 0.65 (high)")
    ax.axvline(0.80, color="red", ls="--", lw=1, alpha=0.5, label="privacy ≥ 0.80 (very_high)")
    ax.set_xlabel("Privacy Score", fontsize=12)
    ax.set_ylabel("Preserve Score", fontsize=12)
    ax.set_title("v1 vs v2: Privacy–Preserve Trade-off", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(0.7, 1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "v1v2_tradeoff.png"), dpi=150)
    plt.close(fig)
    print(f"  ✓ v1v2_tradeoff.png")


# ── Chart 6: Parameter Comparison Table ───────────────────────────────

def plot_param_table(out_dir):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")
    table_data = [
        ["Parameter", "v1 MidBand", "v2 MidBand", "v1 StrongBlur", "v2 StrongBlur"],
        ["band_hz / cutoff", "(500, 3000)", "(500, 3000)", "800", "1000"],
        ["atten_db / lowpass_mix", "25.0 dB", "20.0 dB", "0.70", "0.55"],
        ["highband_start", "—", "—", "2000", "2500"],
        ["highband_mix", "—", "—", "0.30", "0.15"],
        ["midband_gain_db", "—", "—", "3.0", "2.0"],
        ["noise_snr_db", "—", "—", "10.0", "18.0"],
        ["pitch_shift", "No", "Available", "No", "Available"],
    ]
    colors_row = [["#E3F2FD"] * 5] + [["white"] * 5] * (len(table_data) - 1)
    table = ax.table(cellText=table_data, cellColours=colors_row, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1565C0")
            cell.set_text_props(color="white", fontweight="bold")
    ax.set_title("v1 vs v2: Parameter Changes", fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "v1v2_param_table.png"), dpi=150)
    plt.close(fig)
    print(f"  ✓ v1v2_param_table.png")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="v1 vs v2 parameter comparison charts")
    parser.add_argument("run_log", help="Path to run_v1v2 JSON log")
    parser.add_argument("--out-dir", default="plots/v1_vs_v2", help="Output directory")
    args = parser.parse_args()

    data = load(args.run_log)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("v1 vs v2 Parameter Comparison Charts")
    print("=" * 60)
    plot_avg_scores(data, out_dir)
    plot_per_chunk_privacy(data, out_dir)
    plot_privacy_sub_metrics(data, out_dir)
    plot_preserve_radar(data, out_dir)
    plot_tradeoff(data, out_dir)
    plot_param_table(out_dir)

    total = sum(1 for f in os.listdir(out_dir) if f.endswith(".png"))
    print(f"\nDone — {total} charts saved to {out_dir}/")


if __name__ == "__main__":
    main()
