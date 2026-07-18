#!/usr/bin/env python3
"""Sample chunks for MANUAL utility evaluation against CitySpeechMix ground truth.

Why: YAMNet is only a PROXY classifier. To validate that the proxy utility
metrics (TC@3 / TA@1 / mAP) reflect reality, we hand-check a random sample of
chunks: for each, we lay the YAMNet top-3 predictions next to the dataset's
ground-truth AudioSet labels and let a human mark whether YAMNet was correct.

The script emits a review sheet (CSV) with blank human-input columns. Once the
human fills it in, run with --score <filled.csv> to compute agreement stats
(YAMNet top-1 accuracy vs human judgement, top-3 recall, Cohen's kappa proxy).

Usage (generate sheet):
    python3 scripts/sample_manual_utility_eval.py \
        --reports-dir logs/s3/20260429_093321/llm_with_memory \
        --metadata s3://<BUCKET>/cityspeechmix/metadata/metadata.csv \
        --n 60 --out manual_eval_sheet.csv

Usage (score a filled sheet):
    python3 scripts/sample_manual_utility_eval.py --score manual_eval_sheet_filled.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import os
import random


# ---------------------------------------------------------------------------
def _read_metadata(path: str) -> dict:
    """Return {fname: (label1_audioset, label2_audioset)} from metadata.csv."""
    if path.startswith("s3://"):
        import boto3
        rest = path[len("s3://"):]
        b, _, k = rest.partition("/")
        body = boto3.client("s3").get_object(Bucket=b, Key=k)["Body"].read().decode()
        reader = csv.DictReader(io.StringIO(body))
    else:
        reader = csv.DictReader(open(path))
    gt = {}
    for row in reader:
        fname = row.get("fname") or row.get("filename") or ""
        if fname:
            gt[os.path.basename(fname)] = (
                (row.get("label1_audioset") or "").strip(),
                (row.get("label2_audioset") or "").strip(),
            )
    return gt


def _load_chunks(reports_dir: str) -> list[dict]:
    """Flatten all chunks (with classification) from *_report.json files."""
    chunks = []
    for f in sorted(glob.glob(os.path.join(reports_dir, "*_report.json"))):
        d = json.load(open(f))
        src = d.get("source_id", "")
        for c in d.get("chunks", []):
            if c.get("classification_top3"):
                c["_source_id"] = src
                chunks.append(c)
    return chunks


def _fname_from_source(source_id: str) -> str:
    """Best-effort map report source_id → dataset fname (e.g. '00_007687.wav')."""
    base = os.path.basename(str(source_id))
    if not base.endswith(".wav"):
        base += ".wav"
    return base


# ---------------------------------------------------------------------------
def generate(args) -> None:
    random.seed(args.seed)
    gt = _read_metadata(args.metadata)
    print(f"Loaded ground truth for {len(gt)} files")
    chunks = _load_chunks(args.reports_dir)
    print(f"Loaded {len(chunks)} classified chunks from {args.reports_dir}")

    speech = [c for c in chunks if c.get("had_speech")]
    env = [c for c in chunks if not c.get("had_speech")]
    # Stratified: keep the natural speech/env ratio but cap at --n total.
    n_speech = min(len(speech), round(args.n * len(speech) / max(len(chunks), 1)))
    n_env = min(len(env), args.n - n_speech)
    sample = random.sample(speech, n_speech) + random.sample(env, n_env)
    random.shuffle(sample)
    print(f"Sampled {len(sample)} chunks (speech={n_speech}, env={n_env})")

    out = args.out
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "sample_id", "source_file", "chunk_id", "had_speech",
            "gt_label1_env", "gt_label2_speech",
            "yamnet_top1", "yamnet_top1_conf", "yamnet_top2", "yamnet_top3",
            # ── human-input columns (leave blank; fill during review) ──
            "human_top1_correct(Y/N)", "human_gt_in_top3(Y/N)",
            "human_true_label", "notes",
        ])
        for i, c in enumerate(sample, 1):
            fname = _fname_from_source(c["_source_id"])
            g1, g2 = gt.get(fname, ("", ""))
            top3 = c.get("classification_top3") or []
            def _lab(k):
                return top3[k]["label"] if k < len(top3) else ""
            def _conf(k):
                return f'{top3[k]["confidence"]:.3f}' if k < len(top3) else ""
            w.writerow([
                i, fname, c.get("chunk_id", ""), c.get("had_speech"),
                g1, g2, _lab(0), _conf(0), _lab(1), _lab(2),
                "", "", "", "",
            ])
    print(f"\n✓ Review sheet written: {out}")
    print("  Fill columns human_top1_correct / human_gt_in_top3 / human_true_label,")
    print(f"  then run:  python3 {os.path.basename(__file__)} --score {out}")


def score(path: str) -> None:
    rows = list(csv.DictReader(open(path)))
    filled = [r for r in rows if (r.get("human_top1_correct(Y/N)") or "").strip()]
    if not filled:
        print("No human-filled rows found — fill the sheet first.")
        return

    def _yes(v):
        return (v or "").strip().upper().startswith("Y")

    n = len(filled)
    top1 = sum(1 for r in filled if _yes(r["human_top1_correct(Y/N)"]))
    top3 = sum(1 for r in filled if _yes(r["human_gt_in_top3(Y/N)"]))
    sp = [r for r in filled if str(r.get("had_speech")).lower() == "true"]
    en = [r for r in filled if str(r.get("had_speech")).lower() != "true"]

    def _acc(rs):
        return (sum(1 for r in rs if _yes(r["human_top1_correct(Y/N)"])) / len(rs)) if rs else float("nan")

    print(f"Scored {n} human-reviewed chunks")
    print(f"  YAMNet Top-1 accuracy (human-judged): {top1/n:.3f}  ({top1}/{n})")
    print(f"  YAMNet Top-3 hit rate  (human-judged): {top3/n:.3f}  ({top3}/{n})")
    print(f"  Top-1 accuracy — speech chunks: {_acc(sp):.3f}  (n={len(sp)})")
    print(f"  Top-1 accuracy — env chunks:    {_acc(en):.3f}  (n={len(en)})")
    print("\nUse these to state how well the YAMNet PROXY tracks ground truth.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Manual utility-eval sampler (YAMNet proxy vs CitySpeechMix GT)")
    ap.add_argument("--reports-dir", help="Folder with *_report.json (one run/config)")
    ap.add_argument("--metadata", help="metadata.csv path or s3:// uri")
    ap.add_argument("--n", type=int, default=60, help="Sample size (default 60)")
    ap.add_argument("--out", default="manual_eval_sheet.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--score", help="Score a human-filled sheet instead of generating")
    args = ap.parse_args()

    if args.score:
        score(args.score)
        return
    if not args.reports_dir or not args.metadata:
        ap.error("--reports-dir and --metadata are required to generate a sheet")
    generate(args)


if __name__ == "__main__":
    main()
