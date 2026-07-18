"""ClassificationTool — LEGACY single-label (softmax/argmax-style) version.

⚠️  BACKUP COPY — kept so the OLD single-label transfer head can still be run.
    The active tool is ``src/tools/classification_tool.py`` (multi-label).
    This legacy version expects a transfer head trained with a SOFTMAX output
    over mutually-exclusive classes; ``_predict_transfer`` ranks the single
    clip-level distribution. Do NOT use with the new 8-class sigmoid head.

To run the old behaviour, import ``ClassificationTool`` from this module
instead of the package default:

    from src.tools.classification_tool_legacy import ClassificationTool

Requirements: 5.1, 5.2, 5.3
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub

from src.contracts.audio_contracts import AudioChunk
from src.contracts.classification_contracts import (
    ClassPrediction,
    YamnetOutput,
)

_YAMNET_MODEL_HANDLE = "https://tfhub.dev/google/yamnet/1"
_YAMNET_SAMPLE_RATE = 16_000  # YAMNet expects 16 kHz mono


def _maybe_download_s3(path: str) -> str:
    """If *path* is an s3:// uri, download to a temp file and return local path."""
    if not path.startswith("s3://"):
        return path
    import tempfile
    import boto3
    rest = path[len("s3://"):]
    bucket, _, key = rest.partition("/")
    local = os.path.join(tempfile.mkdtemp(), os.path.basename(key))
    boto3.client("s3").download_file(bucket, key, local)
    return local


class ClassificationTool:
    """Classify an AudioChunk using YAMNet (raw or transfer-learned).

    Parameters
    ----------
    transfer_model_path : str | None
        Path (local or ``s3://``) to a trained Keras head that consumes
        YAMNet 1024-dim embeddings and outputs the project taxonomy.
        When None, the raw 521-class AudioSet head is used.
    transfer_label_map : str | dict | None
        ``label_map.json`` path/uri or an already-loaded ``{name: id}`` dict
        matching the transfer head's output order. Required when
        ``transfer_model_path`` is set.

    Notes
    -----
    The model(s) are loaded once on construction and reused across calls.
    """

    def __init__(
        self,
        transfer_model_path: str | None = None,
        transfer_label_map: "str | dict | None" = None,
    ) -> None:
        self._model = hub.load(_YAMNET_MODEL_HANDLE)
        # Load the class map shipped with the TF-Hub model (raw mode).
        class_map_path = self._model.class_map_path().numpy().decode("utf-8")
        with tf.io.gfile.GFile(class_map_path) as f:
            reader = csv.DictReader(f)
            self._class_names: list[str] = [
                row["display_name"] for row in reader
            ]

        # ── Optional transfer-learned head ──
        self._transfer_head = None
        self._transfer_labels: list[str] = []
        if transfer_model_path:
            local_model = _maybe_download_s3(transfer_model_path)
            self._transfer_head = tf.keras.models.load_model(local_model)
            # Resolve label map → ordered list indexed by class id
            if transfer_label_map is None:
                raise ValueError("transfer_label_map is required with transfer_model_path")
            if isinstance(transfer_label_map, str):
                lm = json.load(open(_maybe_download_s3(transfer_label_map)))
            else:
                lm = dict(transfer_label_map)
            # {name: id} → ordered names
            self._transfer_labels = [
                name for name, _ in sorted(lm.items(), key=lambda kv: kv[1])
            ]

    # ------------------------------------------------------------------

    def run(self, chunk: AudioChunk) -> YamnetOutput:
        """Run YAMNet on *chunk* and return a ``YamnetOutput``."""
        data, sr = sf.read(chunk.wav_path, dtype="float32")

        # Ensure mono — average channels if needed.
        if data.ndim > 1:
            data = data.mean(axis=1)

        waveform = tf.cast(data, tf.float32)

        # YAMNet returns (scores, embeddings, spectrogram).
        scores, embeddings, _ = self._model(waveform)

        scores_np: np.ndarray = scores.numpy()          # (N_frames, 521)
        embeddings_np: np.ndarray = embeddings.numpy()  # (N_frames, 1024)

        if self._transfer_head is not None:
            predictions = self._predict_transfer(embeddings_np)
        else:
            predictions = self._predict_raw(scores_np)

        return YamnetOutput(
            chunk_id=chunk.chunk_id,
            embeddings=embeddings_np.tolist(),
            predictions=predictions,
        )

    # ------------------------------------------------------------------
    # Prediction heads
    # ------------------------------------------------------------------

    def _predict_raw(self, scores_np: np.ndarray) -> list[ClassPrediction]:
        """Aggregate raw YAMNet 521-class scores across frames."""
        mean_scores = scores_np.mean(axis=0)
        top_indices = np.argsort(mean_scores)[::-1]

        predictions: list[ClassPrediction] = []
        for idx in top_indices:
            label = self._class_names[idx] if idx < len(self._class_names) else f"class_{idx}"
            confidence = float(mean_scores[idx])
            if confidence <= 0.0:
                break
            predictions.append(ClassPrediction(label=label, confidence=confidence))

        if not predictions:
            best_idx = int(np.argmax(mean_scores))
            predictions.append(
                ClassPrediction(
                    label=self._class_names[best_idx] if best_idx < len(self._class_names) else f"class_{best_idx}",
                    confidence=float(mean_scores[best_idx]),
                )
            )
        return predictions

    def _predict_transfer(self, embeddings_np: np.ndarray) -> list[ClassPrediction]:
        """Apply the transfer head to YAMNet embeddings (clip-level mean-pool).

        Frame-level probabilities are averaged across frames to produce a
        single clip-level distribution over the project taxonomy.
        """
        # (N_frames, n_classes) → mean over frames → (n_classes,)
        frame_probs = self._transfer_head.predict(embeddings_np, verbose=0)
        clip_probs = frame_probs.mean(axis=0)
        order = np.argsort(clip_probs)[::-1]
        predictions = [
            ClassPrediction(
                label=self._transfer_labels[i] if i < len(self._transfer_labels) else f"class_{i}",
                confidence=float(clip_probs[i]),
            )
            for i in order
        ]
        return predictions
