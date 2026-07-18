#!/usr/bin/env python3
"""Convert selected PNG figures to TIFF (300 dpi).

By default converts ONLY the 3 paper figures that were edited:
    • privacy_utility_tradeoff.png   (Privacy vs Utility Trade-off)
    • speech_ratio_vs_privacy.png    (Speech Ratio vs Privacy Score)
    • avg_utility_score.png          (Semantic Preservation Metrics)

Usage:
    python3 scripts/convert_png_to_tiff.py
    # override the file list:
    python3 scripts/convert_png_to_tiff.py --files privacy_utility_tradeoff.png speech_ratio_vs_privacy.png
    # convert every PNG in the folder instead:
    python3 scripts/convert_png_to_tiff.py --all
"""

import argparse
import os
import glob
from PIL import Image

# The 3 edited figures to convert by default.
DEFAULT_FILES = [
    "privacy_utility_density.png",
    "speech_ratio_facets.png",
    "semantic_preservation.png",
]


def main():
    parser = argparse.ArgumentParser(description="Convert PNG to TIFF at 300 dpi")
    parser.add_argument("--input-dir", default="plots/comparison_llm_paper", help="Directory with PNG files")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: same as input)")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for TIFF output (default: 300)")
    parser.add_argument("--files", nargs="+", default=None,
                        help="Specific PNG filenames to convert (default: the 3 edited figures)")
    parser.add_argument("--all", action="store_true",
                        help="Convert every PNG in the input directory")
    args = parser.parse_args()

    out_dir = args.output_dir or args.input_dir
    os.makedirs(out_dir, exist_ok=True)

    if args.all:
        png_files = sorted(glob.glob(os.path.join(args.input_dir, "*.png")))
    else:
        names = args.files or DEFAULT_FILES
        png_files = [os.path.join(args.input_dir, n) for n in names]

    print(f"Converting {len(png_files)} PNG file(s) from {args.input_dir}")

    converted = 0
    for png_path in png_files:
        basename = os.path.splitext(os.path.basename(png_path))[0]
        if not os.path.exists(png_path):
            print(f"  ⚠ skip (not found): {os.path.basename(png_path)}")
            continue
        tiff_path = os.path.join(out_dir, f"{basename}.tiff")
        img = Image.open(png_path)
        img.save(tiff_path, format="TIFF", dpi=(args.dpi, args.dpi), compression="tiff_lzw")
        size_mb = os.path.getsize(tiff_path) / 1024 / 1024
        print(f"  ✓ {basename}.tiff ({size_mb:.1f} MB)")
        converted += 1

    print(f"\nDone — {converted} TIFF file(s) saved to {out_dir}/")


if __name__ == "__main__":
    main()
