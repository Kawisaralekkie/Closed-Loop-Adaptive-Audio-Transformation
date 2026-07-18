#!/usr/bin/env python3
"""Aggregate representative figures over ALL 2226 chunks per category.

Instead of picking a single example chunk, each figure summarises EVERY chunk
in its category by averaging across chunks (sample-aligned):

  category (by speech_ratio of the fixed-mode original chunk):
      env     : speech_ratio == 0            (environment only)
      mix     : 0 < speech_ratio < 0.8       (speech + environment)
      speech  : speech_ratio >= 0.8          (speech dominant)

  each figure has two panels, both aggregated over the category:
      top    = mean amplitude ENVELOPE  = mean_over_chunks |amp_original[i]|
               (per sample index i; a real waveform can't be averaged directly
                because phases differ, so we average the rectified envelope)
               with a light band showing +/- 1 std.
      bottom = mean magnitude SPECTRUM  = mean_over_chunks spectrum_original[k]
               (per frequency bin k) with +/- 1 std shading.

Original audio is mode-independent, so we read the FIXED npz tree and
de-duplicate against the report run_id (=> exactly 2226 unique chunks).

Also regenerates band_energy_original.png using the 2226-chunk (single mode)
energy so it matches the true dataset size (not the 5x mode-summed total).

Outputs (default --out-dir plots/dataset_viz):
    representative_agg_env.png
    representative_agg_speech.png
    representative_agg_mix.png
    band_energy_original_2226.png

Usage:
    python3 scripts/plot_representative_aggregate.py \
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


def _category(sr: float, had_speech: bool) -> str:
    if not had_speech or sr == 0:
        return "env"
    if sr >= 0.8:
        return "speech"
    return "mix"


def _load_report(run_dir: str):
    path = os.path.join(run_dir, "run_metrics_per_chunk.csv")
    rows = [r for r in csv.DictReader(open(path, newline="")) if r.get("mode") == "fixed"]
    lut = {}
    for r in rows:
        key = (r.get("source_file", ""), str(r.get("chunk_index", "")))
        lut[key] = {
            "run_id": (r.get("run_id", "") or "").strip(),
            "speech_ratio": _f(r.get("speech_ratio")),
            "had_speech": str(r.get("had_speech", "")).lower() == "true",
        }
    return lut


class _Agg:
    """Streaming mean/std accumulator for fixed-length vectors."""
    def __init__(self):
        self.n = 0
        self.sum = None
        self.sumsq = None

    def add(self, v: np.ndarray):
        if self.sum is None:
            self.sum = np.zeros_like(v, dtype=np.float64)
            self.sumsq = np.zeros_like(v, dtype=np.float64)
        self.sum += v
        self.sumsq += v * v
        self.n += 1

    def mean_std(self):
        mean = self.sum / self.n
        var = np.maximum(self.sumsq / self.n - mean * mean, 0.0)
        return mean, np.sqrt(var)


def _plot_category(label, env_mean, env_std, spec_mean, spec_std,
                   freqs, sr, n_chunks, out_dir) -> str:
    t = np.arange(len(env_mean)) / sr
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7))
    fig.suptitle(f"{label.upper()} — aggregate of {n_chunks} chunks "
                 f"(CitySpeechMix original)", fontweight="bold", fontsize=12)

    # Envelope panel
    ax1.plot(t, env_mean, color="#1976D2", lw=0.6, label="mean |amplitude|")
    ax1.fill_between(t, np.maximum(env_mean - env_std, 0), env_mean + env_std,
                     color="#1976D2", alpha=0.15, label="±1 std")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("|Amplitude| (mean over chunks)")
    ax1.set_title(f"Amplitude envelope  ({len(env_mean)} samples @ {sr} Hz, averaged)",
                  fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8)

    # Spectrum panel
    ax2.plot(freqs, spec_mean, color="#D32F2F", lw=0.6, label="mean magnitude")
    ax2.fill_between(freqs, np.maximum(spec_mean - spec_std, 0), spec_mean + spec_std,
                     color="#D32F2F", alpha=0.15, label="±1 std")
    ax2.axvspan(500, 3000, color="orange", alpha=0.12, label="500–3000 Hz (speech band)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude (mean over chunks)")
    ax2.set_title(f"Mean spectrum  ({len(spec_mean)} bins, 0–{int(freqs.max())} Hz, averaged)",
                  fontsize=10)
    ax2.set_xlim(0, freqs.max())
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(out_dir, f"representative_agg_{label}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_band_energy_2226(run_dir: str, out_dir: str) -> str:
    """Band energy using the single-mode (2226-chunk) original energy."""
    path = os.path.join(run_dir, "band_reduction_by_mode.csv")
    labels = [f"{lo}-{hi}" for lo, hi in BANDS]
    energy = {lab: 0.0 for lab in labels}
    for r in csv.DictReader(open(path, newline="")):
        # fixed mode, overall subset = 2226 unique original chunks
        if r.get("subset") == "overall" and r.get("mode") == "fixed":
            if r.get("band_hz") in energy:
                energy[r["band_hz"]] = _f(r.get("energy_original"))
    vals = [energy[l] for l in labels]
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
                 "(summed magnitude per band, 2226 unique chunks)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "band_energy_original_2226.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate representative figures (all 2226 chunks)")
    ap.add_argument("--run-dir", default="logs/s3/20260701_171530")
    ap.add_argument("--npz-mode-root", default="logs/amplitude_npz/fixed")
    ap.add_argument("--out-dir", default="plots/dataset_viz")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    report = _load_report(args.run_dir)

    npz_files = sorted(glob.glob(os.path.join(args.npz_mode_root, "**", "*.npz"), recursive=True))
    if not npz_files:
        raise SystemExit(f"No npz under {args.npz_mode_root}")

    env_amp, env_spec = _Agg(), _Agg()
    mix_amp, mix_spec = _Agg(), _Agg()
    sp_amp, sp_spec = _Agg(), _Agg()
    aggs = {"env": (env_amp, env_spec), "mix": (mix_amp, mix_spec), "speech": (sp_amp, sp_spec)}
    counts = {"env": 0, "mix": 0, "speech": 0}
    seen = set()
    freqs = None
    sr = 16000
    kept = dropped = unmatched = 0

    for i, npz_path in enumerate(npz_files, 1):
        base = os.path.basename(npz_path)
        parts = npz_path.replace("\\", "/").split("/")
        source_file = next((p for p in parts if p.endswith(".wav")), "")
        run_id = base.split("_", 1)[0]
        stem = base.replace("_amplitude.npz", "")
        chunk_index = stem.rsplit("_", 1)[-1] if "_" in stem else ""
        info = report.get((source_file, chunk_index))
        if info is None:
            unmatched += 1; continue
        if info["run_id"] and run_id != info["run_id"]:
            dropped += 1; continue
        if (source_file, chunk_index) in seen:
            continue
        seen.add((source_file, chunk_index))
        try:
            d = np.load(npz_path)
            amp = np.abs(np.asarray(d["amp_original"], dtype=np.float64))
            spec = np.asarray(d["spectrum_original"], dtype=np.float64)
            if freqs is None:
                freqs = np.asarray(d["freqs_hz"], dtype=np.float64)
                sr = int(d["sample_rate"])
        except Exception as exc:
            print(f"  SKIP {npz_path}: {exc}"); continue
        cat = _category(info["speech_ratio"], info["had_speech"])
        a_amp, a_spec = aggs[cat]
        if len(amp) == 64000:
            a_amp.add(amp)
        if freqs is not None and len(spec) == len(freqs):
            a_spec.add(spec)
        counts[cat] += 1
        kept += 1
        if i % 1000 == 0:
            print(f"  [{i}/{len(npz_files)}] kept={kept}")

    written = []
    for label in ("env", "speech", "mix"):
        a_amp, a_spec = aggs[label]
        if a_amp.n == 0 or a_spec.n == 0:
            print(f"  (no data for {label})"); continue
        env_mean, env_std = a_amp.mean_std()
        spec_mean, spec_std = a_spec.mean_std()
        p = _plot_category(label, env_mean, env_std, spec_mean, spec_std,
                           freqs, sr, counts[label], args.out_dir)
        written.append(p)
        print(f"  \u2713 {p}   (n={counts[label]})")

    p = _plot_band_energy_2226(args.run_dir, args.out_dir)
    written.append(p)
    print(f"  \u2713 {p}")

    print(f"\nkept={kept} dropped_stale={dropped} unmatched={unmatched}")
    print(f"counts: env={counts['env']} mix={counts['mix']} speech={counts['speech']}")
    print(f"Done. {len(written)} figures in {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
