#!/usr/bin/env python3
"""Prepare a stratified 70:30 MULTI-LABEL train/test split for CitySpeechMixed.

Each clip carries up to two AudioSet labels:
  label1_audioset : the urban / environmental sound   (Dog, Engine, Siren, ...)
  label2_audioset : "Speech"  — present ONLY on clips that were mixed with a
                    LibriSpeech utterance (NaN on SONYC-only clips)

Because not every clip contains speech and because the privacy pipeline may
not fully remove human voice, "Speech" must be a first-class label so the
classifier can still detect residual speech after blurring. We therefore build
a MULTI-LABEL target over 8 classes = {7 environmental classes} ∪ {Speech}.

For every clip the target is a multi-hot vector:
    the environmental class (always)  +  Speech (only if label2 == Speech)

Writes:
  train_split.csv   — 70% of clips
  test_split.csv    — 30% of clips
  label_map.json    — {class_name: integer_id} over all 8 classes
  split_summary.csv — per-class counts (multi-label) in train vs test

Split columns:
  fname, label1_audioset, label2_audioset, s3_key,
  label_ids   (semicolon-joined class ids, e.g. "1;7"),
  label_names (semicolon-joined names,     e.g. "Dog;Speech"),
  y_0 .. y_{N-1}  (multi-hot 0/1 columns, one per class id)

Stratification uses label1_audioset (the environmental class) so the env
distribution is preserved; Speech presence rides along per clip.

Usage:
    python3 scripts/prepare_train_test_split.py \
        --metadata metadata.csv \
        --out-dir data/cityspeechmixed_meta \
        --s3-bucket <RAW_AUDIO_BUCKET> \
        --s3-meta-prefix cityspeechmix/cityspeechmixed_meta/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import pandas as pd
from sklearn.model_selection import train_test_split


def _read_csv_any(path: str) -> pd.DataFrame:
    """Read a CSV from a local path OR an ``s3://bucket/key`` URI.

    For S3 the object is downloaded to a temp file first (no s3fs needed;
    only boto3, which is already a project dependency).
    """
    if path.startswith("s3://"):
        import boto3
        rest = path[len("s3://"):]
        bucket, _, key = rest.partition("/")
        local = os.path.join(tempfile.mkdtemp(), os.path.basename(key))
        boto3.client("s3").download_file(bucket, key, local)
        print(f"Downloaded metadata from {path}")
        return pd.read_csv(local)
    return pd.read_csv(path)

DEFAULT_AUDIO_PREFIX = "cityspeechmix/cityspeechmixed/"
ENV_LABEL_COL = "label1_audioset"     # environmental sound (varies)
SPEECH_LABEL_COL = "label2_audioset"  # "Speech" or NaN
FNAME_COL = "fname"


def main() -> None:
    ap = argparse.ArgumentParser(description="Stratified 70:30 MULTI-LABEL split for CitySpeechMixed")
    ap.add_argument("--metadata", required=True,
                    help="Path to metadata.csv — local path OR an s3:// URI")
    ap.add_argument("--out-dir", default="data/cityspeechmixed_meta", help="Output directory")
    ap.add_argument("--test-size", type=float, default=0.30, help="Test fraction (default 0.30)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--audio-prefix", default=DEFAULT_AUDIO_PREFIX,
                    help="S3 key prefix where the audio WAVs live (added as 's3_key')")
    ap.add_argument("--s3-bucket", default=None, help="If set, upload split files to this bucket")
    ap.add_argument("--s3-meta-prefix", default="cityspeechmix/cityspeechmixed_meta/",
                    help="S3 prefix for the metadata/split files")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── 1. Load + clean metadata (local path or s3:// URI) ──
    df = _read_csv_any(args.metadata)
    if ENV_LABEL_COL not in df.columns or FNAME_COL not in df.columns:
        print(f"ERROR: metadata must contain '{FNAME_COL}' and '{ENV_LABEL_COL}' columns")
        sys.exit(1)
    if SPEECH_LABEL_COL not in df.columns:
        print(f"WARNING: '{SPEECH_LABEL_COL}' column missing — no Speech labels will be added")
        df[SPEECH_LABEL_COL] = pd.NA

    n_total = len(df)
    # A clip is usable if it has an fname and an environmental (label1) class.
    df = df.dropna(subset=[FNAME_COL, ENV_LABEL_COL]).drop_duplicates(subset=[FNAME_COL])
    # Normalise the speech column: non-null, non-empty string => has speech.
    df[SPEECH_LABEL_COL] = df[SPEECH_LABEL_COL].where(df[SPEECH_LABEL_COL].notna(), other="")
    df[SPEECH_LABEL_COL] = df[SPEECH_LABEL_COL].astype(str).str.strip()
    n_speech = int((df[SPEECH_LABEL_COL] != "").sum())
    print(f"Loaded {n_total} rows → {len(df)} usable "
          f"({n_speech} with speech, {len(df) - n_speech} environment-only)\n")

    # Build full S3 key for each audio file.
    df["s3_key"] = args.audio_prefix.rstrip("/") + "/" + df[FNAME_COL].astype(str)

    # ── 2. Build the 8-class label map (env classes + Speech) ──
    env_classes = sorted(df[ENV_LABEL_COL].unique())
    speech_classes = sorted({v for v in df[SPEECH_LABEL_COL].unique() if v})  # e.g. {"Speech"}
    classes = env_classes + [c for c in speech_classes if c not in env_classes]
    label_map = {c: i for i, c in enumerate(classes)}
    print(f"Built {len(classes)} classes (multi-label):")
    for c, i in label_map.items():
        if c in env_classes:
            n = int((df[ENV_LABEL_COL] == c).sum())
        else:
            n = int((df[SPEECH_LABEL_COL] == c).sum())
        print(f"  [{i}] {c}: {n} clips")
    print()

    # ── 3. Multi-hot targets per clip ──
    def _labels_for_row(row) -> list[str]:
        labs = [row[ENV_LABEL_COL]]
        sp = row[SPEECH_LABEL_COL]
        if sp:  # non-empty => speech present
            labs.append(sp)
        return labs

    yid_cols = [f"y_{i}" for i in range(len(classes))]
    for i in range(len(classes)):
        df[yid_cols[i]] = 0
    label_ids_list, label_names_list = [], []
    for idx, row in df.iterrows():
        names = _labels_for_row(row)
        ids = sorted(label_map[n] for n in names if n in label_map)
        for j in ids:
            df.at[idx, f"y_{j}"] = 1
        label_ids_list.append(";".join(str(j) for j in ids))
        label_names_list.append(";".join(names))
    df["label_ids"] = label_ids_list
    df["label_names"] = label_names_list

    # ── 4. Stratified split on the environmental class ──
    too_small = [c for c in env_classes if (df[ENV_LABEL_COL] == c).sum() < 2]
    if too_small:
        print(f"WARNING: env classes with <2 samples cannot be stratified: {too_small}")

    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        stratify=df[ENV_LABEL_COL],
        random_state=args.seed,
    )
    train_df = train_df.sort_values(FNAME_COL).reset_index(drop=True)
    test_df = test_df.sort_values(FNAME_COL).reset_index(drop=True)

    # ── 5. Write outputs ──
    cols = ([FNAME_COL, ENV_LABEL_COL, SPEECH_LABEL_COL, "s3_key",
             "label_ids", "label_names"] + yid_cols)
    train_path = os.path.join(args.out_dir, "train_split.csv")
    test_path = os.path.join(args.out_dir, "test_split.csv")
    lmap_path = os.path.join(args.out_dir, "label_map.json")
    summ_path = os.path.join(args.out_dir, "split_summary.csv")

    train_df[cols].to_csv(train_path, index=False)
    test_df[cols].to_csv(test_path, index=False)
    with open(lmap_path, "w") as f:
        json.dump(label_map, f, indent=2)

    # ── 6. Per-class (multi-label) summary ──
    summ_rows = []
    for c, i in label_map.items():
        col = f"y_{i}"
        tr = int(train_df[col].sum())
        te = int(test_df[col].sum())
        tot = tr + te
        summ_rows.append({
            "class_id": i, "class": c, "train": tr, "test": te, "total": tot,
            "test_pct": round(100 * te / tot, 1) if tot else 0.0,
        })
    summ = pd.DataFrame(summ_rows).set_index("class_id")
    summ.to_csv(summ_path)

    print("=" * 64)
    print(f"MULTI-LABEL split complete (test_size={args.test_size}, seed={args.seed})")
    print("=" * 64)
    print(f"  Train clips: {len(train_df)}")
    print(f"  Test  clips: {len(test_df)}")
    print(f"  Ratio: {100*len(train_df)/len(df):.1f} : {100*len(test_df)/len(df):.1f}\n")
    print(summ.to_string(index=False))
    print("\n(Note: counts are per-label; a clip is counted in BOTH its env "
          "class and Speech when applicable, so columns sum to > n_clips.)\n")
    print("Files written:")
    for p in (train_path, test_path, lmap_path, summ_path):
        print(f"  ✓ {p}")

    # ── 7. Optional S3 upload ──
    if args.s3_bucket:
        import boto3
        s3 = boto3.client("s3")
        prefix = args.s3_meta_prefix.rstrip("/") + "/"
        print(f"\nUploading split files to s3://{args.s3_bucket}/{prefix}")
        for p in (train_path, test_path, lmap_path, summ_path):
            key = prefix + os.path.basename(p)
            s3.upload_file(p, args.s3_bucket, key)
            print(f"  ✓ s3://{args.s3_bucket}/{key}")


if __name__ == "__main__":
    main()
