#!/usr/bin/env python3
"""Export amplitude arrays (time-domain) and spectra of audio before/after blur.

Loads an original WAV and its processed (blurred) WAV, then exports the raw
amplitude samples and the magnitude spectrum of each, so they can be analysed
externally (e.g. in pandas / numpy / Excel).

Outputs per pair:
  <out_dir>/<stem>_amplitude.csv   — columns: sample_index, time_s, amp_original, amp_processed
  <out_dir>/<stem>_spectrum.csv    — columns: freq_hz, mag_original, mag_processed
  <out_dir>/<stem>_arrays.npz      — raw numpy arrays (orig, proc, sr, freqs, spec_orig, spec_proc)

Usage:
    # single pair
    python3 scripts/export_amplitude_arrays.py \
        --original path/to/original.wav \
        --processed path/to/blurred.wav \
        --out-dir plots/amplitude_analysis

    # batch: match original vs processed by filename in two folders
    python3 scripts/export_amplitude_arrays.py \
        --original-dir logs/s3/.../originals \
        --processed-dir logs/s3/.../rule_based_ss \
        --out-dir plots/amplitude_analysis
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np
import soundfile as sf


def _load_mono(path: str) -> tuple[np.ndarray, int]:
    """Load a WAV file as mono float64 amplitude array."""
    data, sr = sf.read(path, dtype="float64")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


def _magnitude_spectrum(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (freqs_hz, magnitude) from a real FFT of the whole signal."""
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)
    return freqs, spec


def export_pair(original_path: str, processed_path: str, out_dir: str) -> None:
    """Export amplitude + spectrum CSV/NPZ for one original/processed pair."""
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(processed_path))[0]

    orig, sr_o = _load_mono(original_path)
    proc, sr_p = _load_mono(processed_path)

    if sr_o != sr_p:
        print(f"  WARNING: sample rate mismatch ({sr_o} vs {sr_p}) for {stem}")
    sr = sr_o

    # Align lengths (pad the shorter with zeros so indices line up 1:1)
    n = max(len(orig), len(proc))
    if len(orig) < n:
        orig = np.pad(orig, (0, n - len(orig)))
    if len(proc) < n:
        proc = np.pad(proc, (0, n - len(proc)))

    # ── 1. Amplitude CSV (time-domain) ──
    amp_path = os.path.join(out_dir, f"{stem}_amplitude.csv")
    with open(amp_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_index", "time_s", "amp_original", "amp_processed"])
        for i in range(n):
            w.writerow([i, i / sr, f"{orig[i]:.8f}", f"{proc[i]:.8f}"])

    # ── 2. Spectrum CSV (frequency-domain magnitude) ──
    freqs_o, spec_o = _magnitude_spectrum(orig, sr)
    freqs_p, spec_p = _magnitude_spectrum(proc, sr)
    spec_path = os.path.join(out_dir, f"{stem}_spectrum.csv")
    with open(spec_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["freq_hz", "mag_original", "mag_processed"])
        for i in range(len(freqs_o)):
            w.writerow([f"{freqs_o[i]:.4f}", f"{spec_o[i]:.8f}", f"{spec_p[i]:.8f}"])

    # ── 3. Raw numpy arrays (compact, for programmatic analysis) ──
    npz_path = os.path.join(out_dir, f"{stem}_arrays.npz")
    np.savez_compressed(
        npz_path,
        sample_rate=sr,
        amp_original=orig,
        amp_processed=proc,
        freqs_hz=freqs_o,
        spectrum_original=spec_o,
        spectrum_processed=spec_p,
    )

    # ── 4. Console summary ──
    print(f"  ✓ {stem}")
    print(f"      samples={n}  duration={n/sr:.2f}s  sr={sr}Hz")
    print(f"      amp_original : min={orig.min():.4f} max={orig.max():.4f} rms={np.sqrt(np.mean(orig**2)):.4f}")
    print(f"      amp_processed: min={proc.min():.4f} max={proc.max():.4f} rms={np.sqrt(np.mean(proc**2)):.4f}")
    print(f"      → {amp_path}")
    print(f"      → {spec_path}")
    print(f"      → {npz_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export amplitude arrays before/after blur")
    ap.add_argument("--original", help="Path to a single original WAV")
    ap.add_argument("--processed", help="Path to a single processed/blurred WAV")
    ap.add_argument("--original-dir", help="Folder of original WAVs (batch mode)")
    ap.add_argument("--processed-dir", help="Folder of processed WAVs (batch mode)")
    ap.add_argument("--out-dir", default="plots/amplitude_analysis", help="Output directory")
    args = ap.parse_args()

    # Single-pair mode
    if args.original and args.processed:
        print(f"Exporting amplitude arrays to {args.out_dir}/\n")
        export_pair(args.original, args.processed, args.out_dir)
        return

    # Batch mode: match by filename
    if args.original_dir and args.processed_dir:
        proc_files = sorted(glob.glob(os.path.join(args.processed_dir, "*.wav")))
        if not proc_files:
            print(f"ERROR: no WAV files in {args.processed_dir}")
            sys.exit(1)
        print(f"Batch exporting {len(proc_files)} pairs to {args.out_dir}/\n")
        for proc in proc_files:
            name = os.path.basename(proc)
            orig = os.path.join(args.original_dir, name)
            if not os.path.exists(orig):
                print(f"  SKIP {name} — no matching original in {args.original_dir}")
                continue
            export_pair(orig, proc, args.out_dir)
        return

    ap.error("Provide either (--original AND --processed) or "
             "(--original-dir AND --processed-dir)")


if __name__ == "__main__":
    main()
