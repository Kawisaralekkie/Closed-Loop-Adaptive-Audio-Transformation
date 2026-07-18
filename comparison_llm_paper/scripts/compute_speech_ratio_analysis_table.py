"""
compute_speech_ratio_analysis.py
================================

Compute Section V-D quantitative analysis numbers per advisor [PR6]:
1. PrivacyScore range (min, max) by speech-ratio band for each config
2. Recipe usage counts by speech-ratio band (LLM configs only)

Usage:
    python compute_speech_ratio_analysis.py /path/to/per_chunk_metrics_paper.csv

Expected CSV columns:
    config_name (str): e.g., "rule_nomem_ss", "llm_h45_mem_ss"
    chunk_id (str)
    speech_ratio (float): 0.0 to 1.0
    privacy_score (float): 0.0 to 1.0
    recipe (str): e.g., "midband_attenuation", "strong_blurring"
"""

import sys
import pandas as pd
import numpy as np


def speech_ratio_band(sr: float) -> str:
    """Classify speech ratio into band."""
    if sr < 0.3:
        return "low"
    elif sr < 0.5:
        return "medium"
    else:
        return "high"


def main(csv_path: str):
    df = pd.read_csv(csv_path)
    df["sr_band"] = df["speech_ratio"].apply(speech_ratio_band)

    print("=" * 70)
    print("PRIVACY SCORE RANGES BY SPEECH-RATIO BAND (Section V-D)")
    print("=" * 70)

    for config in df["configuration"].unique():
        sub = df[df["configuration"] == config]
        print(f"\n[{config}]  (N={len(sub)} chunks)")
        for band in ["low", "medium", "high"]:
            band_sub = sub[sub["sr_band"] == band]
            if len(band_sub) == 0:
                print(f"  {band}: no chunks")
                continue
            ps = band_sub["privacy_score"]
            print(f"  {band:6s} (n={len(band_sub):4d}): "
                  f"PrivacyScore range = [{ps.min():.3f}, {ps.max():.3f}], "
                  f"mean = {ps.mean():.3f}, std = {ps.std():.3f}")

    print()
    print("=" * 70)
    print("RECIPE USAGE BY SPEECH-RATIO BAND (LLM configs only)")
    print("=" * 70)

    if "recipe" not in df.columns:
        print("\nNOTE: 'recipe' column not in CSV.")
        print("Need to extract recipe choice from per-chunk LLM logs.")
        print("Skipping recipe analysis.")
        return

    llm_configs = [c for c in df["configuration"].unique() if "llm" in c.lower()]
    for config in llm_configs:
        sub = df[df["configuration"] == config]
        print(f"\n[{config}]")
        for band in ["low", "medium", "high"]:
            band_sub = sub[sub["sr_band"] == band]
            if len(band_sub) == 0:
                continue
            recipe_counts = band_sub["recipe"].value_counts()
            total = recipe_counts.sum()
            print(f"  {band:6s} (n={total:4d}):")
            for recipe, count in recipe_counts.items():
                pct = 100.0 * count / total
                print(f"    {recipe:30s}: {count:4d} ({pct:5.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compute_speech_ratio_analysis.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
