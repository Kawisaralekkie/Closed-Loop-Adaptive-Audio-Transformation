#!/usr/bin/env python3
"""Generate paper TABLE II (summary) + TABLE III (acceptance rates) from reports.

Reads the SS-enabled run reports, computes per-configuration statistics over
SPEECH chunks only (N=1121), and writes two CSVs + prints markdown tables.

Utility  = (TC@3 + TA@1) / 2   (semantic preservation)
TA@1     = % chunks whose processed top-1 label == original top-1 label
Accept   = privacy_score >= T_p  AND  utility >= T_u  AND  speaker_privacy >= T_s

Usage:
    python3 scripts/make_summary_tables.py [RUN_DIR]
      RUN_DIR default = logs/s3/20260715_final
"""
import csv
import glob
import json
import os
import sys

import numpy as np

RUN = sys.argv[1] if len(sys.argv) > 1 else "logs/s3/20260715_final"
OUT = "plots/comparison_llm_paper"
T_S = 0.50  # speaker-privacy threshold (Ts in caption)

# (folder, display label) in the paper's row order
CONFIGS = [
    ("llm_with_memory_no_ss", "LLM-H4.5-Mem"),
    ("llm_with_memory",       "LLM-H4.5-Mem-SS"),
    ("llm_no_memory",         "LLM-H4.5-NoMem-SS"),
    ("rule_based_ss",         "Rule-NoMem-SS"),
]
# Operating points (T_privacy, T_utility)
OPS = [(0.50, 0.55), (0.60, 0.55), (0.60, 0.70), (0.65, 0.70), (0.65, 0.80)]


def _top3_sets(c):
    orig = c.get("classification_top3_original") or []
    proc = c.get("classification_top3") or []
    return orig, proc


def _semantic(c):
    """(TC@3, TA@1) for a chunk, or None if classification missing."""
    orig, proc = _top3_sets(c)
    if not orig or not proc:
        return None
    a = set(p["label"] for p in orig[:3])
    b = set(p["label"] for p in proc[:3])
    tc3 = len(a & b) / 3.0
    ta1 = 1.0 if orig[0]["label"] == proc[0]["label"] else 0.0
    return tc3, ta1


def collect(folder):
    """Return list of per-speech-chunk dicts with the metrics we need."""
    rows = []
    files = glob.glob(os.path.join(RUN, folder, "*_report.json"))
    for fp in files:
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        for c in d.get("chunks", []):
            if not c.get("had_speech"):
                continue
            sem = _semantic(c)
            if sem is None:
                continue
            pm = c.get("metrics", {}).get("privacy", {}) or {}
            priv = pm.get("privacy_score")
            spk = pm.get("speaker_privacy")
            wer = pm.get("wer")
            if priv is None:
                continue
            tc3, ta1 = sem
            rows.append({
                "privacy": float(priv),
                "utility": (tc3 + ta1) / 2.0,
                "tc3": tc3, "ta1": ta1,
                "wer": float(wer) if wer is not None else np.nan,
                "spk": float(spk) if spk is not None else np.nan,
            })
    return rows


def ms(vals):
    v = np.asarray([x for x in vals if x == x], dtype=float)  # drop nan
    return (float(v.mean()), float(v.std())) if v.size else (float("nan"), float("nan"))


def main():
    os.makedirs(OUT, exist_ok=True)
    data = {}
    for folder, label in CONFIGS:
        rows = collect(folder)
        data[label] = rows
        print(f"  {label:20} speech chunks = {len(rows)}")

    # ── TABLE II — summary ──
    t2 = os.path.join(OUT, "table2_summary.csv")
    with open(t2, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Configuration", "Privacy", "Utility", "WER", "TA@1", "SpkPriv"])
        print("\n### TABLE II — Summary (mean ± std, N per config shown above)\n")
        print(f"| {'Configuration':20} | Privacy | Utility | WER | TA@1 | SpkPriv |")
        print("|" + "-"*22 + "|" + "|".join(["-"*9]*5) + "|")
        for _, label in CONFIGS:
            r = data[label]
            pm, ps = ms([x["privacy"] for x in r])
            um, us = ms([x["utility"] for x in r])
            wm, _ = ms([x["wer"] for x in r])
            ta1 = 100.0 * np.mean([x["ta1"] for x in r]) if r else float("nan")
            km, ks = ms([x["spk"] for x in r])
            w.writerow([label, f"{pm:.3f}±{ps:.3f}", f"{um:.3f}±{us:.3f}",
                        f"{wm:.3f}", f"{ta1:.1f}%", f"{km:.3f}±{ks:.3f}"])
            print(f"| {label:20} | {pm:.3f}±{ps:.3f} | {um:.3f}±{us:.3f} | {wm:.3f} | {ta1:.1f}% | {km:.3f}±{ks:.3f} |")
    print(f"\n  ✓ {t2}")

    # ── TABLE III — acceptance rates (%) ──
    t3 = os.path.join(OUT, "table3_acceptance.csv")
    with open(t3, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Configuration"] + [f"({tp:.2f},{tu:.2f})" for tp, tu in OPS])
        print(f"\n### TABLE III — Acceptance rates (%) at 5 operating points (Ts={T_S}; N per config)\n")
        hdr = " | ".join([f"({tp:.2f},{tu:.2f})" for tp, tu in OPS])
        print(f"| {'Configuration':20} | {hdr} |")
        print("|" + "-"*22 + "|" + "|".join(["-"*13]*len(OPS)) + "|")
        for _, label in CONFIGS:
            r = data[label]
            n = len(r)
            cells = []
            for tp, tu in OPS:
                acc = sum(1 for x in r
                          if x["privacy"] >= tp and x["utility"] >= tu
                          and (x["spk"] != x["spk"] or x["spk"] >= T_S)) / n * 100 if n else 0.0
                cells.append(f"{acc:.1f}")
            w.writerow([label] + cells)
            print(f"| {label:20} | " + " | ".join(f"{c:>11}" for c in cells) + " |")
    print(f"\n  ✓ {t3}")


if __name__ == "__main__":
    main()
