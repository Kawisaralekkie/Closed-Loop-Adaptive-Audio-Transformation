#!/usr/bin/env python3
"""Convert amplitude .npz artifacts into CSV files.

Each amplitude .npz (produced by the pipeline) contains:
    sample_rate         () int
    amp_original        (N,)  float32   time-domain amplitude, original
    amp_processed       (N,)  float32   time-domain amplitude, after blur
    freqs_hz            (M,)  float32   frequency axis for the spectra
    spectrum_original   (M,)  float32   magnitude spectrum, original
    spectrum_processed  (M,)  float32   magnitude spectrum, after blur

For every .npz this writes (next to it, or under --out-dir):
    <stem>_amplitude.csv   columns: sample_index, time_s, amp_original, amp_processed
    <stem>_spectrum.csv    columns: freq_hz, mag_original, mag_processed

Modes:
    --mode both        write both amplitude + spectrum CSVs (default)
    --mode amplitude   only the time-domain amplitude CSV
    --mode spectrum    only the frequency-domain spectrum CSV

Usage:
    # single file
    python3 scripts/npz_to_csv.py path/to/xxx_amplitude.npz

    # a whole folder (all *.npz directly inside it)
    python3 scripts/npz_to_csv.py logs/amplitude_npz/fixed/.../00_007687.wav/

    # recurse through a tree, mirror structure under an output dir
    python3 scripts/npz_to_csv.py logs/amplitude_npz/fixed --recursive \
        --out-dir plots/npz_csv/fixed

    # only spectra, and don't overwrite existing CSVs
    python3 scripts/npz_to_csv.py some_dir --recursive --mode spectrum --skip-existing
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np


def _find_npz(path: str, recursive: bool) -> list[str]:
    """Return the list of .npz files implied by *path*."""
    if os.path.isfile(path):
        return [path] if path.endswith(".npz") else []
    if os.path.isdir(path):
        pattern = os.path.join(path, "**", "*.npz") if recursive \
            else os.path.join(path, "*.npz")
        return sorted(glob.glob(pattern, recursive=recursive))
    return []


def _out_path(npz_path: str, suffix: str, in_root: str | None,
              out_dir: str | None) -> str:
    """Compute the CSV output path for one npz + suffix."""
    stem = os.path.splitext(os.path.basename(npz_path))[0]
    fname = f"{stem}_{suffix}.csv"
    if out_dir is None:
        return os.path.join(os.path.dirname(npz_path), fname)
    # Mirror the sub-tree under out_dir when converting a directory.
    if in_root and os.path.isdir(in_root):
        rel = os.path.relpath(os.path.dirname(npz_path), in_root)
        target_dir = os.path.join(out_dir, rel)
    else:
        target_dir = out_dir
    os.makedirs(target_dir, exist_ok=True)
    return os.path.join(target_dir, fname)


def _write_amplitude(d, out_path: str) -> None:
    sr = int(d["sample_rate"]) if "sample_rate" in d.files else 16000
    orig = d["amp_original"]
    proc = d["amp_processed"] if "amp_processed" in d.files else np.zeros_like(orig)
    n = max(len(orig), len(proc))
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_index", "time_s", "amp_original", "amp_processed"])
        for i in range(n):
            o = orig[i] if i < len(orig) else ""
            p = proc[i] if i < len(proc) else ""
            w.writerow([i, f"{i / sr:.8f}",
                        f"{o:.8f}" if o != "" else "",
                        f"{p:.8f}" if p != "" else ""])


def _write_spectrum(d, out_path: str) -> None:
    freqs = d["freqs_hz"]
    so = np.asarray(d["spectrum_original"], dtype=np.float64)
    sp = np.asarray(d["spectrum_processed"], dtype=np.float64) \
        if "spectrum_processed" in d.files else np.zeros_like(so)

    # dB = 20*log10(mag). Guard against log(0) with a tiny epsilon floor.
    eps = 1e-12
    so_db = 20.0 * np.log10(np.maximum(so, eps))
    sp_db = 20.0 * np.log10(np.maximum(sp, eps))
    # reduction (positive = processed is quieter than original at that freq)
    reduction_db = so_db - sp_db
    reduction_pct = np.where(so > eps, (1.0 - sp / np.maximum(so, eps)) * 100.0, 0.0)

    n = len(freqs)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["freq_hz", "mag_original", "mag_processed",
                    "mag_original_dB", "mag_processed_dB",
                    "reduction_dB", "reduction_pct"])
        for i in range(n):
            w.writerow([
                f"{freqs[i]:.4f}",
                f"{so[i]:.8f}", f"{sp[i]:.8f}",
                f"{so_db[i]:.4f}", f"{sp_db[i]:.4f}",
                f"{reduction_db[i]:.4f}", f"{reduction_pct[i]:.4f}",
            ])


def convert_one(npz_path: str, mode: str, in_root: str | None,
                out_dir: str | None, skip_existing: bool) -> list[str]:
    written: list[str] = []
    try:
        d = np.load(npz_path)
    except Exception as exc:
        print(f"  SKIP {npz_path}: {exc}")
        return written

    if mode in ("amplitude", "both"):
        out = _out_path(npz_path, "amplitude", in_root, out_dir)
        if skip_existing and os.path.exists(out):
            print(f"  = exists {out}")
        else:
            _write_amplitude(d, out)
            written.append(out)
            print(f"  ✓ {out}")

    if mode in ("spectrum", "both"):
        out = _out_path(npz_path, "spectrum", in_root, out_dir)
        if skip_existing and os.path.exists(out):
            print(f"  = exists {out}")
        else:
            _write_spectrum(d, out)
            written.append(out)
            print(f"  ✓ {out}")

    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert amplitude .npz files to CSV")
    ap.add_argument("path", help="A .npz file OR a directory containing .npz files")
    ap.add_argument("--recursive", action="store_true",
                    help="Recurse into sub-directories when PATH is a folder")
    ap.add_argument("--mode", choices=["both", "amplitude", "spectrum"],
                    default="both", help="Which CSV(s) to write (default: both)")
    ap.add_argument("--out-dir", default=None,
                    help="Write CSVs here (mirrors sub-tree). Default: beside each npz")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Do not overwrite a CSV that already exists")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only convert the first N npz files (useful for a quick test)")
    args = ap.parse_args()

    npz_files = _find_npz(args.path, args.recursive)
    if not npz_files:
        print(f"No .npz files found for: {args.path}"
              + ("" if args.recursive else "  (try --recursive if it's a tree)"))
        sys.exit(1)

    if args.limit:
        npz_files = npz_files[:args.limit]

    in_root = args.path if os.path.isdir(args.path) else None
    print(f"Converting {len(npz_files)} npz file(s)  mode={args.mode}\n")

    total_csv = 0
    for npz_path in npz_files:
        total_csv += len(convert_one(npz_path, args.mode, in_root,
                                     args.out_dir, args.skip_existing))

    print(f"\nDone. Wrote {total_csv} CSV file(s) from {len(npz_files)} npz file(s).")


if __name__ == "__main__":
    main()
