#!/usr/bin/env python3
"""YAMNet MULTI-LABEL transfer-learning trainer for CitySpeechMixed.

Each clip may carry several of 8 classes (7 environmental + Speech); the split
CSVs encode this as multi-hot columns y_0..y_{N-1} (see prepare_train_test_split.py).
Keeping Speech as a class lets the classifier still detect residual human voice
after the privacy pipeline blurs it (and handles the SONYC-only clips that
contain no speech at all).

Pipeline:
  1. Read train/test split CSVs (s3_key + multi-hot y_* columns).
  2. Download each WAV from S3 (cached locally).
  3. Extract YAMNet embeddings (1024-dim / 0.48s frame), backbone FROZEN.
  4. Train a small MULTI-LABEL head (1024 → 512 → 256 → N, sigmoid) with
     binary cross-entropy on frame-level embeddings.
  5. Evaluate on the test split (clip-level: mean-pool frame probabilities,
     threshold at --threshold) with per-class precision/recall/F1.
  6. Save the trained head (.keras) + label_map + metrics, optionally to S3.

Usage:
    python3 scripts/train_yamnet_transfer.py \
        --train-csv s3://BUCKET/cityspeechmix/cityspeechmixed_meta/train_split.csv \
        --test-csv  s3://BUCKET/cityspeechmix/cityspeechmixed_meta/test_split.csv \
        --label-map s3://BUCKET/cityspeechmix/cityspeechmixed_meta/label_map.json \
        --audio-bucket <RAW_AUDIO_BUCKET> \
        --out-dir /opt/ml/model \
        --upload-s3-prefix s3://BUCKET/cityspeechmix/models/yamnet_transfer/
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Threading / OpenMP guards — MUST be set before TensorFlow is imported.
# On macOS (Apple Silicon) TF + tensorflow_hub can deadlock while "Loading
# YAMNet" due to duplicate OpenMP runtimes / oversubscribed thread pools, so
# LOCAL runs are pinned single-threaded. Inside a SageMaker training container
# we WANT all vCPUs, so only apply the aggressive pinning when not on SageMaker.
# ---------------------------------------------------------------------------
_ON_SAGEMAKER = bool(os.environ.get("TRAINING_JOB_NAME") or
                     os.environ.get("SM_TRAINING_ENV") or
                     os.path.isdir("/opt/ml/input"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
if not _ON_SAGEMAKER:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")

import argparse
import json
import tempfile

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------
def _parse_s3(uri: str) -> tuple[str, str]:
    assert uri.startswith("s3://"), f"not an s3 uri: {uri}"
    rest = uri[len("s3://"):]
    bucket, _, key = rest.partition("/")
    return bucket, key


def _read_csv_any(path: str):
    if path.startswith("s3://"):
        import boto3
        b, k = _parse_s3(path)
        d = tempfile.mkdtemp()
        local = os.path.join(d, os.path.basename(k))
        boto3.client("s3").download_file(b, k, local)
        return pd.read_csv(local)
    return pd.read_csv(path)


def _read_json_any(path: str) -> dict:
    if path.startswith("s3://"):
        import boto3
        b, k = _parse_s3(path)
        obj = boto3.client("s3").get_object(Bucket=b, Key=k)
        return json.loads(obj["Body"].read())
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Audio + YAMNet
# ---------------------------------------------------------------------------
_YAMNET = None


def _get_yamnet():
    global _YAMNET
    if _YAMNET is None:
        import tensorflow as tf
        # On local (non-SageMaker) runs, pin TF thread pools to 1 to avoid the
        # macOS OpenMP deadlock during load. SageMaker keeps all vCPUs.
        if not _ON_SAGEMAKER:
            try:
                tf.config.threading.set_inter_op_parallelism_threads(1)
                tf.config.threading.set_intra_op_parallelism_threads(1)
            except RuntimeError:
                pass  # already initialized elsewhere
        import tensorflow_hub as hub
        print("Loading YAMNet from TF-Hub...", flush=True)
        _YAMNET = hub.load("https://tfhub.dev/google/yamnet/1")
        print("YAMNet loaded.", flush=True)
    return _YAMNET


def _load_wav_16k_mono(path: str) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        from math import gcd
        g = gcd(sr, 16000)
        data = resample_poly(data, 16000 // g, sr // g).astype("float32")
    return data


def _embeddings_for_file(local_wav: str) -> np.ndarray:
    import tensorflow as tf
    wav = _load_wav_16k_mono(local_wav)
    scores, embeddings, spectrogram = _get_yamnet()(tf.constant(wav))
    return embeddings.numpy()


def _download(s3_client, bucket: str, key: str, cache_dir: str) -> str:
    local = os.path.join(cache_dir, key.replace("/", "_"))
    if not os.path.exists(local):
        s3_client.download_file(bucket, key, local)
    return local


# ---------------------------------------------------------------------------
# Multi-hot label helpers
# ---------------------------------------------------------------------------
def _multihot_for_row(row, num_classes: int) -> np.ndarray:
    """Build a multi-hot vector from y_0..y_{N-1} columns, or from label_ids."""
    vec = np.zeros(num_classes, dtype=np.float32)
    have_y_cols = all(hasattr(row, f"y_{i}") for i in range(num_classes))
    if have_y_cols:
        for i in range(num_classes):
            vec[i] = float(getattr(row, f"y_{i}") or 0)
        return vec
    # Fallback: parse "label_ids" like "1;7"
    raw = getattr(row, "label_ids", "")
    for tok in str(raw).split(";"):
        tok = tok.strip()
        if tok.isdigit() and int(tok) < num_classes:
            vec[int(tok)] = 1.0
    return vec


# ---------------------------------------------------------------------------
# Feature extraction over a split
# ---------------------------------------------------------------------------
def _extract_split(df: pd.DataFrame, bucket: str, cache_dir: str,
                   num_classes: int, *, frame_level: bool):
    """Extract embeddings + multi-hot labels for every file in df.

    frame_level=True  → (X_frames[n,1024], Y_frames[n,C]) one row per frame.
    frame_level=False → (X_clips[m,1024],  Y_clips[m,C])  one mean-pooled row/clip.
    """
    import boto3
    s3 = boto3.client("s3")
    os.makedirs(cache_dir, exist_ok=True)

    X, Y = [], []
    n = len(df)
    for i, row in enumerate(df.itertuples(index=False), 1):
        key = getattr(row, "s3_key")
        multihot = _multihot_for_row(row, num_classes)
        try:
            local = _download(s3, bucket, key, cache_dir)
            emb = _embeddings_for_file(local)  # (frames, 1024)
        except Exception as exc:
            print(f"  [{i}/{n}] SKIP {key}: {exc}")
            continue
        if emb.shape[0] == 0:
            continue
        if frame_level:
            X.append(emb)
            Y.append(np.tile(multihot, (emb.shape[0], 1)))
        else:
            X.append(emb.mean(axis=0, keepdims=True))
            Y.append(multihot[None, :])
        if i % 50 == 0:
            print(f"  [{i}/{n}] processed")
    if not X:
        raise RuntimeError("No embeddings extracted — check S3 keys/access")
    return np.concatenate(X, axis=0), np.concatenate(Y, axis=0)


# ---------------------------------------------------------------------------
# Model (multi-label: sigmoid + binary cross-entropy)
# ---------------------------------------------------------------------------
def _build_head(num_classes: int):
    import tensorflow as tf
    from tensorflow.keras import layers
    model = tf.keras.Sequential([
        layers.Input(shape=(1024,), name="yamnet_embedding"),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="sigmoid"),  # multi-label
    ], name="yamnet_transfer_head")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc", multi_label=True),
            tf.keras.metrics.BinaryAccuracy(name="bin_acc", threshold=0.5),
        ],
    )
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description="YAMNet MULTI-LABEL transfer learning (CitySpeechMixed)")
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--label-map", required=True)
    ap.add_argument("--audio-bucket", required=True)
    ap.add_argument("--out-dir", default="/opt/ml/model")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Probability threshold for a positive label (default 0.5)")
    ap.add_argument("--upload-s3-prefix", default=None)
    # SageMaker's TensorFlow container auto-injects extra hyperparameters such
    # as --model_dir. Use parse_known_args so those are ignored instead of
    # crashing argparse with "unrecognized arguments".
    args, _unknown = ap.parse_known_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cache_dir = args.cache_dir or tempfile.mkdtemp(prefix="csm_audio_")

    # ── Load splits ──
    label_map = _read_json_any(args.label_map)
    id_to_label = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)
    train_df = _read_csv_any(args.train_csv)
    test_df = _read_csv_any(args.test_csv)
    print(f"Classes={num_classes} (multi-label)  train={len(train_df)}  test={len(test_df)}\n")

    # ── Extract embeddings + multi-hot labels ──
    print("Extracting TRAIN embeddings (frame-level)...")
    X_train, Y_train = _extract_split(train_df, args.audio_bucket, cache_dir,
                                      num_classes, frame_level=True)
    print(f"  train frames: {X_train.shape}  labels: {Y_train.shape}")

    print("Extracting TEST embeddings (clip-level, mean-pooled)...")
    X_test, Y_test = _extract_split(test_df, args.audio_bucket, cache_dir,
                                    num_classes, frame_level=False)
    print(f"  test clips: {X_test.shape}  labels: {Y_test.shape}\n")

    # ── Per-class positive weights (handle imbalance in multi-label BCE) ──
    # weight_pos_c = n_neg_c / n_pos_c ; passed via sample-independent class
    # weighting is not directly supported for multi-output BCE, so we instead
    # report prevalence and rely on AUC/threshold. (Kept simple + robust.)
    pos_rate = Y_train.mean(axis=0)
    print("Train positive rate per class:")
    for i in range(num_classes):
        print(f"  [{i}] {id_to_label.get(i, i)}: {pos_rate[i]:.3f}")
    print()

    # ── Train head ──
    model = _build_head(num_classes)
    model.summary()
    import tensorflow as tf
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ]
    model.fit(
        X_train, Y_train,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    # ── Evaluate (clip-level, multi-label) ──
    from sklearn.metrics import (classification_report, f1_score,
                                 accuracy_score, multilabel_confusion_matrix)
    probs = model.predict(X_test, verbose=0)
    Y_pred = (probs >= args.threshold).astype(int)
    target_names = [id_to_label[i] for i in range(num_classes)]

    subset_acc = accuracy_score(Y_test, Y_pred)              # exact-match ratio
    macro_f1 = f1_score(Y_test, Y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(Y_test, Y_pred, average="micro", zero_division=0)
    report = classification_report(Y_test, Y_pred, target_names=target_names,
                                   zero_division=0)
    mlcm = multilabel_confusion_matrix(Y_test, Y_pred).tolist()

    # per-class F1 for the metrics file
    per_class_f1 = f1_score(Y_test, Y_pred, average=None, zero_division=0)
    per_class = {target_names[i]: round(float(per_class_f1[i]), 4)
                 for i in range(num_classes)}

    print("\n" + "=" * 64)
    print(f"TEST subset-acc(exact match)={subset_acc:.4f}  "
          f"macro-F1={macro_f1:.4f}  micro-F1={micro_f1:.4f}  thr={args.threshold}")
    print("=" * 64)
    print(report)

    # ── Save artifacts ──
    model_path = os.path.join(args.out_dir, "yamnet_transfer_head.keras")
    model.save(model_path)
    with open(os.path.join(args.out_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)
    with open(os.path.join(args.out_dir, "eval_metrics.json"), "w") as f:
        json.dump({
            "task": "multi_label",
            "threshold": args.threshold,
            "subset_accuracy": subset_acc,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "per_class_f1": per_class,
            "multilabel_confusion_matrix": mlcm,
            "labels": target_names,
            "train_pos_rate": {target_names[i]: round(float(pos_rate[i]), 4)
                               for i in range(num_classes)},
            "n_train_frames": int(X_train.shape[0]),
            "n_test_clips": int(X_test.shape[0]),
        }, f, indent=2)
    print(f"\nSaved model → {model_path}")

    # ── Optional S3 upload ──
    if args.upload_s3_prefix:
        import boto3
        s3 = boto3.client("s3")
        b, k = _parse_s3(args.upload_s3_prefix)
        prefix = k.rstrip("/") + "/"
        for fname in ("yamnet_transfer_head.keras", "label_map.json", "eval_metrics.json"):
            s3.upload_file(os.path.join(args.out_dir, fname), b, prefix + fname)
            print(f"  ✓ uploaded s3://{b}/{prefix}{fname}")


if __name__ == "__main__":
    main()
