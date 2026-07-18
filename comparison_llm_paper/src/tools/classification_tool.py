"""ClassificationTool — Soundscape classification using YAMNet.

Runs the YAMNet model on an AudioChunk (blurred or unblurred) and returns
a YamnetOutput containing audio embeddings and predicted sound class labels
with confidence scores.

Two modes:
  • Raw YAMNet (default) — 521 AudioSet classes from the pretrained head.
  • Transfer-learned     — when ``transfer_model_path`` / ``transfer_label_map``
    are provided, a custom Keras head maps YAMNet's 1024-dim embeddings to the
    project's urban-sound taxonomy (e.g. the 8 CitySpeechMixed classes).

The transfer head is MULTI-LABEL (sigmoid output, one independent probability
per class) so a single clip can be tagged with several sounds at once — e.g.
``Engine`` + ``Speech`` — and residual human voice can still be detected after
the privacy pipeline blurs it. Each class probability is independent and does
NOT sum to 1 across classes (unlike a softmax single-label head).

The previous single-label (softmax) implementation is preserved in
``src/tools/classification_tool_legacy.py`` for backward compatibility.

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
    transfer_threshold : float
        Probability threshold (default 0.5) above which a class is considered
        PRESENT in the clip for the multi-label head. Predictions are always
        returned fully ranked; this threshold marks the present set (see
        ``present_labels``) and must match the value used at training-time eval.

    Notes
    -----
    The model(s) are loaded once on construction and reused across calls.
    """

    def __init__(
        self,
        transfer_model_path: str | None = None,
        transfer_label_map: "str | dict | None" = None,
        transfer_threshold: float = 0.5,
    ) -> None:
        self._transfer_threshold = float(transfer_threshold)
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
        """Run YAMNet on *chunk* and return a ``YamnetOutput``.

        Parameters
        ----------
        chunk : AudioChunk
            The audio chunk to classify (blurred or unblurred).

        Returns
        -------
        YamnetOutput
            Per-frame embeddings and predicted class labels with confidence
            scores. Predictions come from the transfer head when configured,
            otherwise from raw YAMNet (521 AudioSet classes).
        """
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
        """Apply the MULTI-LABEL transfer head to YAMNet embeddings.

        The head emits one independent sigmoid probability per class for every
        0.48 s YAMNet frame. We mean-pool the per-frame probabilities to obtain
        a single clip-level probability per class (standard multi-label
        aggregation), then return every class as a ``ClassPrediction`` sorted by
        probability, highest first.

        Because the head is multi-label:
          • confidences are INDEPENDENT and do not sum to 1 across classes;
          • more than one class can legitimately have a high confidence (e.g.
            ``Engine`` and ``Speech`` together);
          • a class counts as *present* when its confidence ≥ the configured
            ``transfer_threshold`` (see ``present_labels``).

        The full ranked list is always returned (not truncated at the
        threshold) so downstream consumers can still compute top-k / rank-based
        metrics and pick ``predictions[0]`` as the dominant class.
        """
        # (N_frames, n_classes); guard the single-frame case → keep 2-D.
        frame_probs = np.atleast_2d(self._transfer_head.predict(embeddings_np, verbose=0))
        clip_probs = frame_probs.mean(axis=0)  # (n_classes,) independent sigmoid probs
        order = np.argsort(clip_probs)[::-1]
        return [
            ClassPrediction(
                label=self._transfer_labels[i] if i < len(self._transfer_labels) else f"class_{i}",
                # sigmoid output already lies in [0, 1]; clamp defensively for
                # the ClassPrediction contract against tiny float overshoot.
                confidence=float(np.clip(clip_probs[i], 0.0, 1.0)),
            )
            for i in order
        ]

    def present_labels(self, output: YamnetOutput) -> list[str]:
        """Return the multi-label PRESENT set for a transfer-head result.

        Classes whose independent confidence ≥ ``transfer_threshold`` are
        considered present, in descending-confidence order. Falls back to the
        single top-1 label when nothing clears the threshold, so the result is
        never empty for a non-empty prediction list.
        """
        present = [
            p.label for p in output.predictions
            if p.confidence >= self._transfer_threshold
        ]
        if not present and output.predictions:
            present = [output.predictions[0].label]
        return present
