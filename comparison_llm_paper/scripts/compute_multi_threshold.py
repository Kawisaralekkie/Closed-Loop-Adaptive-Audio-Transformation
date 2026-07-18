"""
Multi-Threshold Acceptance Rate Calculator
==========================================

Computes acceptance rates at multiple threshold combinations from per-chunk JSON reports.
Designed for ISCIT 2026 paper revision.

Usage:
    python compute_multi_threshold.py --input_dir /path/to/configs/

Expected directory structure:
    input_dir/
        LLM-H4.5-Mem/
            *.json
        LLM-H4.5-NoMem-SS/
            *.json
        LLM-H4.5-Mem-SS/
            *.json
        Rule-NoMem-SS/
            *.json

Outputs:
    - multi_threshold_acceptance.csv  (acceptance rates at each (T_p, T_u) combination)
    - per_chunk_metrics.csv  (TC@3, TA@1, PreserveScore, PrivacyScore per chunk)
    - summary_table.csv  (mean +/- std per config)
    - threshold_heatmap.png (optional visualization)
"""

import argparse
import json
import os
import csv
from pathlib import Path
from collections import defaultdict


# ---------------------------------------------------------------------------
# 1. Per-chunk metric computation
# ---------------------------------------------------------------------------

def compute_tc3(top3_orig, top3_proc):
    """Top-3 Consistency: |A intersect B| / 3"""
    if not top3_orig or not top3_proc:
        return None
    set_a = {item["label"] for item in top3_orig}
    set_b = {item["label"] for item in top3_proc}
    return len(set_a & set_b) / 3.0


def compute_ta1(top3_orig, top3_proc):
    """Top-1 Agreement: 1 if top-1 matches, else 0"""
    if not top3_orig or not top3_proc:
        return None
    return 1.0 if top3_orig[0]["label"] == top3_proc[0]["label"] else 0.0


def compute_preserve_score(top3_orig, top3_proc):
    """PreserveScore = (TC@3 + TA@1) / 2"""
    tc3 = compute_tc3(top3_orig, top3_proc)
    ta1 = compute_ta1(top3_orig, top3_proc)
    if tc3 is None or ta1 is None:
        return None, None, None
    return tc3, ta1, (tc3 + ta1) / 2.0


def extract_chunk_metrics(chunk):
    """Return dict of metrics for a single chunk, or None if no speech / data missing."""
    # Only process chunks with speech
    if not chunk.get("had_speech", False):
        return None

    metrics = chunk.get("metrics", {})
    privacy = metrics.get("privacy", {})

    top3_orig = chunk.get("classification_top3_original", [])
    top3_proc = chunk.get("classification_top3", [])

    tc3, ta1, preserve_new = compute_preserve_score(top3_orig, top3_proc)
    if tc3 is None:
        return None

    return {
        "chunk_id": chunk.get("chunk_id"),
        "speech_ratio": chunk.get("speech_ratio"),
        "vad_confidence": chunk.get("vad_confidence"),
        "wer": privacy.get("wer"),
        "cer": privacy.get("cer"),
        "content_privacy": privacy.get("content_privacy"),
        "speaker_privacy": privacy.get("speaker_privacy"),
        "privacy_score": privacy.get("privacy_score"),
        "tc3": tc3,
        "ta1": ta1,
        "preserve_new": preserve_new,
        "trials_used": chunk.get("trials"),
        "used_ss": chunk.get("used_source_separation", False),
        "recipe": (chunk.get("recipe_applied") or {}).get("recipe_name"),
    }


def load_config_chunks(config_dir):
    """Load all chunks from JSON files in a config directory."""
    chunks_data = []
    json_files = list(Path(config_dir).glob("*.json"))
    print(f"  Found {len(json_files)} JSON files in {config_dir.name}")

    for jf in json_files:
        try:
            with open(jf, "r") as f:
                report = json.load(f)
        except Exception as e:
            print(f"  ! Failed to load {jf.name}: {e}")
            continue

        for chunk in report.get("chunks", []):
            m = extract_chunk_metrics(chunk)
            if m is not None:
                m["source_id"] = report.get("source_id", jf.stem)
                chunks_data.append(m)

    print(f"  -> {len(chunks_data)} speech chunks extracted")
    return chunks_data


# ---------------------------------------------------------------------------
# 2. Multi-threshold acceptance rate computation
# ---------------------------------------------------------------------------

def compute_acceptance_rate(chunks, t_p, t_u, t_s):
    """
    Acceptance rate under (T_p, T_u, T_s).
    A chunk is accepted when ALL three conditions hold simultaneously.
    """
    if not chunks:
        return 0.0
    n_accept = sum(
        1 for c in chunks
        if (c["privacy_score"] is not None and c["privacy_score"] >= t_p)
        and (c["preserve_new"] is not None and c["preserve_new"] >= t_u)
        and (c["speaker_privacy"] is not None and c["speaker_privacy"] >= t_s)
    )
    return n_accept / len(chunks)


def compute_marginal_pass_rates(chunks, t_p, t_u, t_s):
    """Pass rates for each criterion individually."""
    n = len(chunks) if chunks else 0
    if n == 0:
        return 0.0, 0.0, 0.0
    return (
        sum(1 for c in chunks if c["privacy_score"] is not None and c["privacy_score"] >= t_p) / n,
        sum(1 for c in chunks if c["preserve_new"] is not None and c["preserve_new"] >= t_u) / n,
        sum(1 for c in chunks if c["speaker_privacy"] is not None and c["speaker_privacy"] >= t_s) / n,
    )


# ---------------------------------------------------------------------------
# 3. Statistics utilities (no numpy, vanilla python)
# ---------------------------------------------------------------------------

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


# ---------------------------------------------------------------------------
# 4. Main report generation
# ---------------------------------------------------------------------------

def generate_reports(all_chunks, output_dir, t_p_grid, t_u_grid, t_s_default=0.40):
    """
    all_chunks: dict[config_name] -> list[chunk_dict]
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---------- (a) Summary table per config ----------
    summary_path = Path(output_dir) / "summary_table.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Configuration", "n_chunks",
            "PrivacyScore_mean", "PrivacyScore_std",
            "PreserveScore_mean", "PreserveScore_std",
            "TC@3_mean", "TC@3_std",
            "TA@1_rate",
            "SpeakerPrivacy_mean", "SpeakerPrivacy_std",
            "WER_mean", "CER_mean",
            "AvgTrials",
        ])
        for cfg, chunks in all_chunks.items():
            w.writerow([
                cfg, len(chunks),
                f"{mean([c['privacy_score'] for c in chunks]):.4f}",
                f"{std([c['privacy_score'] for c in chunks]):.4f}",
                f"{mean([c['preserve_new'] for c in chunks]):.4f}",
                f"{std([c['preserve_new'] for c in chunks]):.4f}",
                f"{mean([c['tc3'] for c in chunks]):.4f}",
                f"{std([c['tc3'] for c in chunks]):.4f}",
                f"{mean([c['ta1'] for c in chunks]):.4f}",
                f"{mean([c['speaker_privacy'] for c in chunks]):.4f}",
                f"{std([c['speaker_privacy'] for c in chunks]):.4f}",
                f"{mean([c['wer'] for c in chunks]):.4f}",
                f"{mean([c['cer'] for c in chunks]):.4f}",
                f"{mean([c['trials_used'] for c in chunks]):.2f}",
            ])
    print(f"\n[OK] Summary table written: {summary_path}")

    # ---------- (b) Multi-threshold acceptance grid ----------
    grid_path = Path(output_dir) / "multi_threshold_acceptance.csv"
    with open(grid_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["Configuration", "T_p", "T_u", "T_s",
                  "AcceptanceRate", "PrivacyPass", "UtilityPass", "SpeakerPass", "n_chunks"]
        w.writerow(header)
        for cfg, chunks in all_chunks.items():
            for t_p in t_p_grid:
                for t_u in t_u_grid:
                    accept = compute_acceptance_rate(chunks, t_p, t_u, t_s_default)
                    p_pass, u_pass, s_pass = compute_marginal_pass_rates(chunks, t_p, t_u, t_s_default)
                    w.writerow([cfg, f"{t_p:.2f}", f"{t_u:.2f}", f"{t_s_default:.2f}",
                                f"{accept:.4f}", f"{p_pass:.4f}", f"{u_pass:.4f}", f"{s_pass:.4f}",
                                len(chunks)])
    print(f"[OK] Multi-threshold grid written: {grid_path}")

    # ---------- (c) Per-chunk metrics ----------
    per_chunk_path = Path(output_dir) / "per_chunk_metrics.csv"
    with open(per_chunk_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Configuration", "source_id", "chunk_id", "speech_ratio", "vad_confidence",
            "wer", "cer", "content_privacy", "speaker_privacy", "privacy_score",
            "tc3", "ta1", "preserve_new", "trials_used", "used_ss", "recipe",
        ])
        for cfg, chunks in all_chunks.items():
            for c in chunks:
                w.writerow([cfg, c.get("source_id"), c["chunk_id"], c["speech_ratio"], c["vad_confidence"],
                            c["wer"], c["cer"], c["content_privacy"], c["speaker_privacy"], c["privacy_score"],
                            c["tc3"], c["ta1"], c["preserve_new"], c["trials_used"], c["used_ss"], c["recipe"]])
    print(f"[OK] Per-chunk metrics written: {per_chunk_path}")

    # ---------- (d) Print compact summary to stdout ----------
    print("\n" + "=" * 80)
    print("COMPACT SUMMARY (for paper)")
    print("=" * 80)
    print(f"{'Configuration':<22} {'Privacy':>10} {'Preserve':>10} {'TC@3':>8} {'TA@1':>8} {'Trials':>8}")
    print("-" * 80)
    for cfg, chunks in all_chunks.items():
        print(f"{cfg:<22} "
              f"{mean([c['privacy_score'] for c in chunks]):>10.4f} "
              f"{mean([c['preserve_new'] for c in chunks]):>10.4f} "
              f"{mean([c['tc3'] for c in chunks]):>8.4f} "
              f"{mean([c['ta1'] for c in chunks]):>8.2%} "
              f"{mean([c['trials_used'] for c in chunks]):>8.2f}")

    print("\n" + "=" * 80)
    print("ACCEPTANCE RATE GRID (T_s = 0.40)")
    print("=" * 80)
    # focus on key cells
    key_cells = [(0.50, 0.60), (0.60, 0.55), (0.65, 0.55), (0.65, 0.70), (0.65, 0.80)]
    print(f"{'Configuration':<22}", end="")
    for tp, tu in key_cells:
        print(f" T_p={tp:.2f},T_u={tu:.2f}", end="")
    print()
    print("-" * (22 + len(key_cells) * 18))
    for cfg, chunks in all_chunks.items():
        print(f"{cfg:<22}", end="")
        for tp, tu in key_cells:
            r = compute_acceptance_rate(chunks, tp, tu, 0.40)
            print(f"     {r:>10.2%} ", end="")
        print()


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------

CONFIG_NAMES = [
    "LLM-H4.5-Mem",
    "LLM-H4.5-NoMem-SS",
    "LLM-H4.5-Mem-SS",
    "Rule-NoMem-SS",
]

# Map config names to actual log paths (same as plot_comparison_llm.py)
CONFIG_PATHS = {
    "LLM-H4.5-Mem": "logs/s3/20260425_094650/llm_with_memory_no_ss",
    "LLM-H4.5-NoMem-SS": "logs/s3/20260426_102345/llm_no_memory",
    "LLM-H4.5-Mem-SS": "logs/s3/20260425_095019/llm_with_memory",
    "Rule-NoMem-SS": "logs/s3/20260426_102213/rule_based_ss",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default=None,
                   help="Directory containing one subfolder per config. "
                        "If not provided, uses hardcoded paths from CONFIG_PATHS.")
    p.add_argument("--output_dir", default="./multi_threshold_output")
    p.add_argument("--t_p_grid", default="0.50,0.55,0.60,0.65,0.70")
    p.add_argument("--t_u_grid", default="0.50,0.55,0.60,0.65,0.70,0.75,0.80")
    p.add_argument("--t_s", type=float, default=0.40)
    args = p.parse_args()

    t_p_grid = [float(x) for x in args.t_p_grid.split(",")]
    t_u_grid = [float(x) for x in args.t_u_grid.split(",")]

    print("Loading per-config JSON reports...")
    all_chunks = {}

    if args.input_dir:
        # Use subfolder-per-config structure
        for cfg in CONFIG_NAMES:
            cfg_dir = Path(args.input_dir) / cfg
            if not cfg_dir.exists():
                print(f"  ! Config dir not found: {cfg_dir}, skipping")
                continue
            print(f"\n[{cfg}]")
            all_chunks[cfg] = load_config_chunks(cfg_dir)
    else:
        # Use hardcoded paths (aligned with plot_comparison_llm.py)
        for cfg, path in CONFIG_PATHS.items():
            cfg_dir = Path(path)
            if not cfg_dir.exists():
                print(f"  ! Path not found: {path}, skipping")
                continue
            print(f"\n[{cfg}] -> {path}")
            all_chunks[cfg] = load_config_chunks(cfg_dir)

    if not all_chunks:
        raise SystemExit("No configs loaded. Check --input_dir.")

    generate_reports(all_chunks, args.output_dir, t_p_grid, t_u_grid, args.t_s)
    print(f"\nDone. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
