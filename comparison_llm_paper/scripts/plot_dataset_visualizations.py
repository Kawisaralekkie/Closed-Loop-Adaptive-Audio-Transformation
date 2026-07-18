#!/usr/bin/env python3
"""Dataset-characterisation figures for the CitySpeechMix ORIGINAL audio.

PART 1 — Representative visualisations (one figure per category):
    * env    : environment-only chunk   (had_speech=False, speech_ratio=0)
    * speech : speech-only chunk         (highest speech_ratio)
    * mix    : mixed chunk               (speech_ratio ~ 0.5)
  Each figure has two stacked panels:
    top    = Waveform      (amp_original, 64000 samples over time)
    bottom = Spectrum      (spectrum_original magnitude over frequency)

PART 2 — Distribution figures across all 2226 chunks:
    * amplitude histogram : Original RMS per chunk (from original_chunk_summary.csv)
    * spectrum band bars  : energy_original per band (from band_reduction_by_mode.csv,
                            subset=overall, mode=ALL_MODES)

All figures are written as PNG under --out-dir (default: plots/dataset_viz).

Representative chunks are chosen from run_metrics_per_chunk.csv (mode=fixed);
their npz are read from the fixed npz tree (original audio is mode-independent).

Usage:
    python3 scripts/plot_dataset_visualizations.py \
        --run-dir logs/s3/20260701_171530 \
        --npz-mode-root logs/amplitude_npz/fixed \
        --out-dir plots/dataset_viz
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BANDS = [(0, 500), (500, 3000), (3000, 8000)]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _load_fixed_rows(run_dir: str) -> list[dict]:
    path = os.path.join(run_dir, "run_metrics_per_chunk.csv")
    return [r for r in csv.DictReader(open(path, newline="")) if r.get("mode") == "fixed"]


def _pick_representatives(rows: list[dict]) -> dict:
    env = [r for r in rows if str(r.get("had_speech", "")).lower() == "false"]
    sp = [r for r in rows if str(r.get("had_speech", "")).lower() == "true"]
    env_sorted = sorted(env, key=lambda r: _f(r.get("amp_orig_rms")))
    return {
        "env": env_sorted[len(env_sorted) // 2] if env else None,
        "speech": max(sp, key=lambda r: _f(r.get("speech_ratio"))) if sp else None,
        "mix": min(sp, key=lambda r: abs(_f(r.get("speech_ratio")) - 0.5)) if sp else None,
    }


def _find_npz(npz_mode_root: str, source_file: str, run_id: str, chunk_index: str) -> str | None:
    cand = os.path.join(npz_mode_root, "**", source_file,
                        f"{run_id}_{chunk_index}_amplitude.npz")
    hits = glob.glob(cand, recursive=True)
    if hits:
        return hits[0]
    # fallback: any npz for that source/chunk
    cand2 = os.path.join(npz_mode_root, "**", source_file, f"*_{chunk_index}_amplitude.npz")
    hits2 = glob.glob(cand2, recursive=True)
    return hits2[0] if hits2 else None


def _plot_representative(label: str, meta: dict, npz_path: str, out_dir: str) -> str:
    d = np.load(npz_path)
    sr = int(d["sample_rate"])
    amp = np.asarray(d["amp_original"], dtype=np.float64)
    freqs = np.asarray(d["freqs_hz"], dtype=np.float64)
    spec = np.asarray(d["spectrum_original"], dtype=np.float64)
    t = np.arange(len(amp)) / sr

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7))
    title = (f"{label.upper()} — {meta['source_file']} chunk {meta['chunk_index']}  "
             f"(speech_ratio={meta.get('speech_ratio')}, rms={_f(meta.get('amp_orig_rms')):.4f})")
    fig.suptitle(title, fontweight="bold", fontsize=11)

    # Waveform
    ax1.plot(t, amp, color="#1976D2", linewidth=0.4)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"Waveform  ({len(amp)} samples @ {sr} Hz)", fontsize=10)
    ax1.set_ylim(-1.05, 1.05)
    ax1.grid(alpha=0.3)

    # Spectrum (magnitude vs frequency)
    ax2.plot(freqs, spec, color="#D32F2F", linewidth=0.4)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude")
    ax2.set_title(f"Spectrum  ({len(spec)} bins, 0–{int(freqs.max())} Hz)", fontsize=10)
    ax2.set_xlim(0, freqs.max())
    ax2.grid(alpha=0.3)
    # shade the core speech band
    ax2.axvspan(500, 3000, color="orange", alpha=0.12, label="500–3000 Hz (speech band)")
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(out_dir, f"representative_{label}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_rms_histogram(run_dir: str, out_dir: str) -> str:
    path = os.path.join(run_dir, "original_chunk_summary.csv")
    rms = [_f(r.get("amp_rms")) for r in csv.DictReader(open(path, newline=""))]
    rms = [v for v in rms if v > 0]
    mean_rms = float(np.mean(rms)) if rms else 0.0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(rms, bins=50, color="#42A5F5", edgecolor="white", alpha=0.85)
    ax.axvline(mean_rms, color="red", ls="--", lw=1.5,
               label=f"mean RMS = {mean_rms:.4f}")
    ax.axvline(1.0, color="black", ls=":", lw=1.2, label="clip level (1.0)")
    ax.set_xlabel("Original amplitude RMS (per chunk)")
    ax.set_ylabel("Number of chunks")
    ax.set_title(f"Distribution of Original RMS across {len(rms)} chunks\n"
                 f"(loudness spread — not silent, not clipping)", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "hist_original_rms.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_band_energy(run_dir: str, out_dir: str) -> str:
    path = os.path.join(run_dir, "band_reduction_by_mode.csv")
    labels = [f"{lo}-{hi}" for lo, hi in BANDS]
    energy = {lab: 0.0 for lab in labels}
    for r in csv.DictReader(open(path, newline="")):
        if r.get("subset") == "overall" and r.get("mode") == "ALL_MODES":
            if r.get("band_hz") in energy:
                energy[r["band_hz"]] = _f(r.get("energy_original"))

    vals = [energy[lab] for lab in labels]
    total = sum(vals) or 1.0
    pct = [100 * v / total for v in vals]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#90A4AE", "#FB8C00", "#66BB6A"]
    bars = ax.bar(labels, vals, color=colors, edgecolor="black")
    for b, v, p in zip(bars, vals, pct):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{p:.1f}%",
                ha="center", va="bottom", fontweight="bold")
    ax.set_xlabel("Frequency band (Hz)")
    ax.set_ylabel("Total original spectral energy")
    ax.set_title("Where the energy lives — CitySpeechMix original audio\n"
                 "(summed magnitude per band, all 11,130 chunks)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "band_energy_original.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Dataset visualisation figures")
    ap.add_argument("--run-dir", default="logs/s3/20260701_171530")
    ap.add_argument("--npz-mode-root", default="logs/amplitude_npz/fixed")
    ap.add_argument("--out-dir", default="plots/dataset_viz")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    written = []

    # PART 1 — representative chunks
    rows = _load_fixed_rows(args.run_dir)
    reps = _pick_representatives(rows)
    for label, meta in reps.items():
        if not meta:
            print(f"  (no chunk for {label})")
            continue
        npz_path = _find_npz(args.npz_mode_root, meta["source_file"],
                             meta["run_id"], meta["chunk_index"])
        if not npz_path:
            print(f"  (npz not found for {label}: {meta['source_file']} chunk {meta['chunk_index']})")
            continue
        p = _plot_representative(label, meta, npz_path, args.out_dir)
        written.append(p)
        print(f"  \u2713 {p}   <- {meta['source_file']} chunk {meta['chunk_index']}")

    # PART 2 — distributions
    written.append(_plot_rms_histogram(args.run_dir, args.out_dir))
    print(f"  \u2713 {written[-1]}")
    written.append(_plot_band_energy(args.run_dir, args.out_dir))
    print(f"  \u2713 {written[-1]}")

    print(f"\nDone. {len(written)} figures in {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
