#!/usr/bin/env python3
"""Validate Silero VAD speech detection against the dataset's ground-truth labels.

Ground truth comes from ``label2_sonyc`` in the CitySpeechMixed metadata:
  • non-NaN  → the clip CONTAINS speech (positive)
  • NaN      → the clip is voice-free  (negative)

For each clip we run Silero VAD (the same SpeechScanTool used in the pipeline)
and compare its "speech detected" decision to the ground truth, then report
accuracy / precision / recall / F1 and a confusion matrix.

Audio is read from S3 (or local). Works for both train and test split CSVs,
or directly from metadata.csv.

Usage:
    # from a split CSV (has s3_key column)
    python3 scripts/validate_vad_against_labels.py \
        --csv data/cityspeechmixed_meta/test_split.csv \
        --audio-bucket <RAW_AUDIO_BUCKET> \
        --out-csv plots/vad_validation_test.csv

    # directly from metadata.csv (builds s3_key from fname + --audio-prefix)
    python3 scripts/validate_vad_against_labels.py \
        --csv metadata.csv \
        --audio-bucket <RAW_AUDIO_BUCKET> \
        --audio-prefix cityspeechmix/cityspeechmixed/ \
        --out-csv plots/vad_validation_all.csv
"""

from __future__ import annotations

import argparse
import os
import tempfile

import pandas as pd


def _read_csv_any(path: str) -> pd.DataFrame:
    if path.startswith("s3://"):
        import boto3
        rest = path[len("s3://"):]
        bucket, _, key = rest.partition("/")
        d = tempfile.mkdtemp()
        local = os.path.join(d, os.path.basename(key))
        boto3.client("s3").download_file(bucket, key, local)
        return pd.read_csv(local)
    return pd.read_csv(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate Silero VAD vs label2_sonyc ground truth")
    ap.add_argument("--csv", required=True, help="Split CSV or metadata.csv")
    ap.add_argument("--audio-bucket", required=True, help="S3 bucket containing the audio WAVs")
    ap.add_argument("--audio-prefix", default="cityspeechmix/cityspeechmixed/",
                    help="S3 prefix (used to build s3_key from fname when CSV has no s3_key)")
    ap.add_argument("--vad-threshold", type=float, default=0.5, help="Silero VAD threshold")
    ap.add_argument("--speech-ratio-min", type=float, default=0.0,
                    help="Min speech_ratio to count as 'speech detected' "
                         "(0.0 = any segment counts via has_speech)")
    ap.add_argument("--out-csv", default="plots/vad_validation.csv")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N rows (0 = all)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    import boto3
    import uuid
    from src.contracts.audio_contracts import AudioChunk
    from src.tools.speech_scan_tool import SpeechScanTool

    s3 = boto3.client("s3")
    cache = tempfile.mkdtemp(prefix="vad_val_")
    vad = SpeechScanTool(threshold=args.vad_threshold)

    df = _read_csv_any(args.csv)
    if args.limit:
        df = df.head(args.limit)

    # Build s3_key column if missing
    if "s3_key" not in df.columns:
        df["s3_key"] = args.audio_prefix.rstrip("/") + "/" + df["fname"].astype(str)

    rows = []
    n = len(df)
    print(f"Validating VAD on {n} clips (threshold={args.vad_threshold})...\n")
    for i, r in enumerate(df.itertuples(index=False), 1):
        fname = getattr(r, "fname")
        key = getattr(r, "s3_key")
        # Ground truth: speech present if label2_sonyc is not NaN
        gt_label2 = getattr(r, "label2_sonyc", None)
        gt_has_speech = pd.notna(gt_label2)

        # Download + run VAD
        local = os.path.join(cache, key.replace("/", "_"))
        try:
            if not os.path.exists(local):
                s3.download_file(args.audio_bucket, key, local)
            chunk = AudioChunk(
                chunk_id=fname, run_id=uuid.uuid4(), wav_path=local,
                start_time=0.0, end_time=0.0, duration=0.0, metadata={},
            )
            vad_res = vad.run(chunk)
            pred_has_speech = (
                vad_res.has_speech and vad_res.speech_ratio >= args.speech_ratio_min
            )
        except Exception as exc:
            print(f"  [{i}/{n}] SKIP {fname}: {exc}")
            continue

        rows.append({
            "fname": fname,
            "gt_label2_sonyc": gt_label2 if gt_has_speech else "",
            "gt_has_speech": int(gt_has_speech),
            "vad_has_speech": int(pred_has_speech),
            "vad_speech_ratio": round(vad_res.speech_ratio, 4),
            "vad_n_segments": len(vad_res.segments),
            "correct": int(gt_has_speech == pred_has_speech),
        })
        if i % 50 == 0:
            print(f"  [{i}/{n}] processed")

    if not rows:
        print("ERROR: no clips processed")
        return

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)

    # ── Metrics ──
    tp = int(((out.gt_has_speech == 1) & (out.vad_has_speech == 1)).sum())
    tn = int(((out.gt_has_speech == 0) & (out.vad_has_speech == 0)).sum())
    fp = int(((out.gt_has_speech == 0) & (out.vad_has_speech == 1)).sum())
    fn = int(((out.gt_has_speech == 1) & (out.vad_has_speech == 0)).sum())
    total = len(out)
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    print("\n" + "=" * 60)
    print("VAD vs GROUND-TRUTH (label2_sonyc) VALIDATION")
    print("=" * 60)
    print(f"  Clips evaluated : {total}")
    print(f"  Ground-truth speech / voice-free : {int(out.gt_has_speech.sum())} / {int((out.gt_has_speech==0).sum())}")
    print()
    print("  Confusion matrix:")
    print(f"                      VAD: speech    VAD: no-speech")
    print(f"    GT speech     :   TP={tp:<8}     FN={fn}")
    print(f"    GT voice-free :   FP={fp:<8}     TN={tn}")
    print()
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}  (of VAD-detected speech, how many truly had speech)")
    print(f"  Recall   : {rec:.4f}  (of true-speech clips, how many VAD caught)")
    print(f"  F1       : {f1:.4f}")
    print()
    print(f"  Detailed results → {args.out_csv}")


if __name__ == "__main__":
    main()
