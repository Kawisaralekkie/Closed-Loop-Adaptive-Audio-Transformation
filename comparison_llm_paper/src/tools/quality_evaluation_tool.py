"""QualityEvaluationTool — Privacy and utility quality evaluation.

Computes privacy metrics (WER, CER, speaker privacy), preservation
sub-scores (s_loud, s_hf, s_sc, s_con, s_psy with psychoacoustic
features), utility metrics (mAP, F1, accuracy), and produces a
QualityDecision against configurable privacy_target thresholds.

When ``use_real_models=True`` (default), privacy metrics are computed
using:
  - **Whisper** (OpenAI) for ASR-based WER/CER
  - **WavLM** (Microsoft) for speaker embedding cosine distance

When ``use_real_models=False``, falls back to lightweight signal-level
proxy estimators (useful for unit tests or environments without GPU).

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 17.1, 17.2, 17.3, 17.4, 17.5
"""

from __future__ import annotations

import logging

import numpy as np
import soundfile as sf

from src.contracts.audio_contracts import AudioChunk
from src.contracts.classification_contracts import YamnetOutput
from src.contracts.metrics_contracts import (
    PRESERVE_SCORE_MIN,
    PRIVACY_SCORE_MIN,
    SPEAKER_PRIVACY_MIN,
    MetricsResult,
    normalize_privacy_target,
    PreserveSubScores,
    PrivacyMetrics,
    PsychoacousticFeatures,
    QualityDecision,
    UtilityMetrics,
)

logger = logging.getLogger(__name__)


class QualityEvaluationTool:
    """Evaluate privacy protection and soundscape preservation quality.

    Computes composite privacy_score and preserve_score for a processed
    audio chunk, then renders a pass/fail QualityDecision.

    Parameters
    ----------
    use_real_models : bool
        When ``True`` (default), use Whisper for WER/CER and WavLM
        for speaker_privacy.  When ``False``, use signal-level proxies.
    content_privacy_mode : str
        How to compute content_privacy from WER/CER:
          • "wer_cer" (default) → wer_weight*WER + cer_weight*CER
          • "wer_only"          → WER alone (CER ignored)
        Override via env var ``CONTENT_PRIVACY_MODE``.
    wer_weight, cer_weight : float
        Weights for the "wer_cer" mode (default 0.6 / 0.4).
        Override via env vars ``CONTENT_WER_WEIGHT`` / ``CONTENT_CER_WEIGHT``.
    content_weight, speaker_weight : float
        Weights combining content_privacy and speaker_privacy into the
        final privacy_score (default 0.7 / 0.3).
        Override via env vars ``PRIVACY_CONTENT_WEIGHT`` / ``PRIVACY_SPEAKER_WEIGHT``.
    """

    def __init__(
        self,
        use_real_models: bool = True,
        content_privacy_mode: str | None = None,
        wer_weight: float | None = None,
        cer_weight: float | None = None,
        content_weight: float | None = None,
        speaker_weight: float | None = None,
        ground_truth_label: str | None = None,
    ) -> None:
        import os
        self._use_real_models = use_real_models
        # File-level ground-truth class used as a fallback when run() is
        # called without an explicit ground_truth_label (transfer-learning
        # eval). None → utility metrics fall back to the confidence proxy.
        self._ground_truth_label_default = ground_truth_label
        # Content-privacy formula configuration (env vars override args)
        self._content_mode = (
            content_privacy_mode
            or os.environ.get("CONTENT_PRIVACY_MODE", "wer_cer")
        ).lower()
        self._wer_weight = (
            wer_weight if wer_weight is not None
            else float(os.environ.get("CONTENT_WER_WEIGHT", "0.6"))
        )
        self._cer_weight = (
            cer_weight if cer_weight is not None
            else float(os.environ.get("CONTENT_CER_WEIGHT", "0.4"))
        )
        self._content_weight = (
            content_weight if content_weight is not None
            else float(os.environ.get("PRIVACY_CONTENT_WEIGHT", "0.7"))
        )
        self._speaker_weight = (
            speaker_weight if speaker_weight is not None
            else float(os.environ.get("PRIVACY_SPEAKER_WEIGHT", "0.3"))
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        original_chunk: AudioChunk,
        processed_chunk: AudioChunk,
        classification_result: YamnetOutput,
        privacy_target: str = "moderate",
        ground_truth_label: str | None = None,
        had_speech: bool = True,
    ) -> MetricsResult:
        """Compute privacy, utility, and preservation metrics.

        Parameters
        ----------
        original_chunk : AudioChunk
            The original (unblurred) audio chunk.
        processed_chunk : AudioChunk
            The blurred/processed audio chunk.
        classification_result : YamnetOutput
            YAMNet classification output for the processed chunk.
        privacy_target : str
            ``"moderate"`` or ``"high"`` — determines the minimum
            acceptable privacy_score threshold.

            NOTE (renamed 2026):
                OLD VOCAB        NEW VOCAB     privacy_score_min
                "high"      →    "moderate"    0.65
                "very_high" →    "high"        0.80
            Legacy values ("high", "very_high") are accepted via
            ``normalize_privacy_target()`` for backward compatibility.
        had_speech : bool
            Whether this chunk contains speech (from the VAD result). When
            ``False`` the chunk is a no-speech / bypass chunk: privacy metrics
            are UNDEFINED (reported as ``None``) and privacy_pass is ``True``
            (nothing to protect). Whisper/WavLM are skipped entirely.

        Returns
        -------
        MetricsResult
            Composite result containing privacy, utility, psychoacoustic
            metrics, and the quality decision.
        """
        original_audio, sr_orig = sf.read(original_chunk.wav_path, dtype="float32")
        processed_audio, sr_proc = sf.read(processed_chunk.wav_path, dtype="float32")

        if original_audio.ndim > 1:
            original_audio = original_audio.mean(axis=1)
        if processed_audio.ndim > 1:
            processed_audio = processed_audio.mean(axis=1)

        # --- Privacy metrics (Req 6.1, 6.2) ---
        # Privacy is ONLY defined for chunks that contain speech. No-speech
        # chunks are bypassed (never blurred); running ASR on them makes
        # Whisper hallucinate words from environmental sound, producing
        # meaningless (sometimes huge) WER/CER. We therefore skip privacy
        # entirely for no-speech chunks and report None (not applicable).
        # This also avoids the Whisper+WavLM compute for every silent chunk.
        if had_speech:
            privacy = self._compute_privacy_metrics(
                original_audio, processed_audio, sr_orig,
                original_path=original_chunk.wav_path,
                processed_path=processed_chunk.wav_path,
            )
        else:
            privacy = PrivacyMetrics()  # all fields None → not applicable

        # --- Psychoacoustic features (Req 17.1–17.4) ---
        psychoacoustic = self._compute_psychoacoustic_features(processed_audio, sr_proc)

        # --- Preservation sub-scores (Req 6.3, 17.5) ---
        sub_scores = self._compute_preserve_sub_scores(
            original_audio, processed_audio, sr_orig, classification_result, psychoacoustic,
        )
        preserve_score = self._aggregate_preserve_score(sub_scores)

        # --- Utility metrics (Req 6.4) ---
        # Real mAP/F1/accuracy when a ground-truth label is available
        # (transfer-learning eval); otherwise a YAMNet confidence proxy.
        # Per-call arg takes precedence; otherwise use the tool-level default.
        gt_label = (
            ground_truth_label if ground_truth_label is not None
            else self._ground_truth_label_default
        )
        utility_raw = self._compute_utility_metrics(
            classification_result, ground_truth_label=gt_label,
        )

        utility = UtilityMetrics(
            mAP=utility_raw["mAP"],
            f1=utility_raw["f1"],
            accuracy=utility_raw["accuracy"],
            preserve_score=preserve_score,
            sub_scores=sub_scores,
            ground_truth_label=utility_raw.get("ground_truth_label"),
            predicted_label=utility_raw.get("predicted_label"),
            top1_correct=utility_raw.get("top1_correct"),
            top3_correct=utility_raw.get("top3_correct"),
            metrics_source=utility_raw.get("metrics_source", "proxy"),
        )

        # --- Quality decision (Req 6.5, 6.6) ---
        # Normalize legacy values: old "high"→"moderate", old "very_high"→"high"
        target_normalized = normalize_privacy_target(privacy_target)
        privacy_score_min = PRIVACY_SCORE_MIN.get(
            target_normalized, PRIVACY_SCORE_MIN["moderate"]
        )
        if had_speech and privacy.privacy_score is not None:
            privacy_pass = (
                privacy.privacy_score >= privacy_score_min
                and (privacy.speaker_privacy or 0.0) >= SPEAKER_PRIVACY_MIN
            )
        else:
            # No speech → no human voice to protect → privacy trivially met.
            privacy_pass = True
        decision = QualityDecision(
            privacy_pass=privacy_pass,
            preserve_pass=preserve_score >= PRESERVE_SCORE_MIN,
            overall_pass=(privacy_pass and preserve_score >= PRESERVE_SCORE_MIN),
            # Record the NORMALIZED target so reports use the new vocabulary
            privacy_target=target_normalized,
            privacy_score_min=privacy_score_min,
            preserve_score_min=PRESERVE_SCORE_MIN,
        )

        return MetricsResult(
            chunk_id=processed_chunk.chunk_id,
            privacy=privacy,
            utility=utility,
            psychoacoustic=psychoacoustic,
            decision=decision,
        )

    # ------------------------------------------------------------------
    # Privacy metrics
    # ------------------------------------------------------------------

    def _compute_privacy_metrics(
        self,
        original: np.ndarray,
        processed: np.ndarray,
        sr: int,
        original_path: str | None = None,
        processed_path: str | None = None,
    ) -> PrivacyMetrics:
        """Compute WER, CER, speaker_privacy, and composite scores.

        When ``use_real_models=True`` (the production setting) real
        models are MANDATORY: Whisper ASR for WER/CER and WavLM for
        speaker_privacy. There is NO silent fallback — if the audio
        paths are missing the call raises, so a run can never quietly
        produce proxy (fake) numbers.

        The signal-level proxy estimators are only used when
        ``use_real_models=False`` is explicitly requested (unit tests /
        GPU-less environments).
        """
        if self._use_real_models:
            # Real models are required — never fall back silently.
            if not (original_path and processed_path):
                raise ValueError(
                    "Real privacy metrics require both original_path and "
                    "processed_path. Refusing to fall back to proxy "
                    "estimators (use_real_models=True)."
                )
            logger.info("Computing real privacy metrics (Whisper + WavLM)...")
            print("    [privacy] Running Whisper ASR + WavLM speaker model...", flush=True)
            wer, cer, speaker_privacy = self._compute_real_privacy(
                original_path, processed_path,
            )
            print(f"    [privacy] Done: WER={wer:.4f} CER={cer:.4f} speaker={speaker_privacy:.4f}", flush=True)
        else:
            # Explicit opt-in proxy path (tests only).
            wer = self._estimate_wer_proxy(original, processed)
            cer = self._estimate_cer_proxy(original, processed)
            speaker_privacy = self._estimate_speaker_privacy_proxy(
                original, processed, sr,
            )

        # content_privacy (Req 6.1) — configurable formula
        if self._content_mode == "wer_only":
            content_privacy = wer
        else:  # "wer_cer"
            content_privacy = self._wer_weight * wer + self._cer_weight * cer
        # privacy_score (Req 6.2) — configurable content/speaker weighting
        privacy_score = (
            self._content_weight * content_privacy
            + self._speaker_weight * speaker_privacy
        )

        return PrivacyMetrics(
            wer=wer,
            cer=cer,
            speaker_privacy=speaker_privacy,
            content_privacy=content_privacy,
            privacy_score=privacy_score,
        )

    # ------------------------------------------------------------------
    # Real model-based privacy metrics (Whisper + ECAPA-TDNN)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_real_privacy(
        original_path: str,
        processed_path: str,
    ) -> tuple[float, float, float]:
        """Compute WER, CER, speaker_privacy using real AI models.

        Uses:
          - Whisper (OpenAI) for ASR transcription → WER/CER
          - ECAPA-TDNN (SpeechBrain) for speaker embedding → cosine distance

        Returns
        -------
        tuple[float, float, float]
            ``(wer, cer, speaker_privacy)`` all in [0, 1].
        """
        from src.tools.asr_evaluator import evaluate_asr_privacy
        from src.tools.speaker_evaluator import evaluate_speaker_privacy

        wer, cer, _, _ = evaluate_asr_privacy(original_path, processed_path)
        speaker_privacy = evaluate_speaker_privacy(original_path, processed_path)

        return wer, cer, speaker_privacy

    # ------------------------------------------------------------------
    # Fallback proxy estimators (no heavy model dependencies)
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_wer_proxy(original: np.ndarray, processed: np.ndarray) -> float:
        """Signal-level WER proxy: normalised energy difference in the
        speech band.  Higher difference → higher WER (more privacy)."""
        min_len = min(len(original), len(processed))
        diff = original[:min_len] - processed[:min_len]
        orig_energy = float(np.sum(original[:min_len] ** 2)) + 1e-12
        return float(np.clip(np.sum(diff ** 2) / orig_energy, 0.0, 1.0))

    @staticmethod
    def _estimate_cer_proxy(original: np.ndarray, processed: np.ndarray) -> float:
        """Signal-level CER proxy: correlation-based distortion measure."""
        min_len = min(len(original), len(processed))
        o, p = original[:min_len], processed[:min_len]
        corr = float(np.abs(np.correlate(o, p, mode="valid")[0]))
        norm = float(np.sqrt(np.sum(o ** 2) * np.sum(p ** 2))) + 1e-12
        return float(np.clip(1.0 - corr / norm, 0.0, 1.0))

    @staticmethod
    def _estimate_speaker_privacy_proxy(
        original: np.ndarray, processed: np.ndarray, sr: int,
    ) -> float:
        """Speaker privacy proxy via spectral envelope divergence.

        NOTE: clip removed — expose true Jensen-Shannon divergence value.
        """
        n_fft = min(1024, len(original), len(processed))
        if n_fft < 4:
            return 1.0
        orig_spec = np.abs(np.fft.rfft(original[:n_fft]))
        proc_spec = np.abs(np.fft.rfft(processed[:n_fft]))
        orig_spec = orig_spec / (orig_spec.sum() + 1e-12)
        proc_spec = proc_spec / (proc_spec.sum() + 1e-12)
        m = 0.5 * (orig_spec + proc_spec) + 1e-12
        div = 0.5 * (
            np.sum(orig_spec * np.log(orig_spec / m + 1e-12))
            + np.sum(proc_spec * np.log(proc_spec / m + 1e-12))
        )
        return float(div)

    # ------------------------------------------------------------------
    # Psychoacoustic features (Req 17.1–17.4)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_psychoacoustic_features(
        audio: np.ndarray, sr: int,
    ) -> PsychoacousticFeatures:
        """Compute psychoacoustic feature proxies for the processed chunk."""
        # Short-term loudness (Req 17.1): RMS energy in dB
        rms = float(np.sqrt(np.mean(audio ** 2)) + 1e-12)
        short_term_loudness = 20.0 * np.log10(rms + 1e-12)

        n_fft = min(2048, len(audio))
        if n_fft < 4:
            return PsychoacousticFeatures(
                short_term_loudness=short_term_loudness,
                sharpness_proxy=0.0,
                roughness_proxy=0.0,
                fluctuation_proxy=0.0,
            )

        spectrum = np.abs(np.fft.rfft(audio[:n_fft]))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

        # Sharpness proxy (Req 17.2): ratio of high-frequency energy
        hf_mask = freqs >= 3000.0
        total_energy = float(np.sum(spectrum ** 2)) + 1e-12
        sharpness_proxy = float(np.sum(spectrum[hf_mask] ** 2) / total_energy)

        # Roughness proxy (Req 17.3): spectral flux / variation
        half = n_fft // 2
        if half > 0 and len(audio) >= n_fft + half:
            spectrum2 = np.abs(np.fft.rfft(audio[half : half + n_fft]))
            roughness_proxy = float(
                np.mean(np.abs(spectrum - spectrum2)) / (np.mean(spectrum) + 1e-12)
            )
        else:
            roughness_proxy = 0.0

        # Fluctuation proxy (Req 17.4): amplitude modulation depth
        frame_size = max(1, sr // 20)  # 50 ms frames
        n_frames = len(audio) // frame_size
        if n_frames > 1:
            frame_energies = np.array([
                np.sqrt(np.mean(audio[i * frame_size : (i + 1) * frame_size] ** 2))
                for i in range(n_frames)
            ])
            fluctuation_proxy = float(
                np.std(frame_energies) / (np.mean(frame_energies) + 1e-12)
            )
        else:
            fluctuation_proxy = 0.0

        return PsychoacousticFeatures(
            short_term_loudness=short_term_loudness,
            sharpness_proxy=sharpness_proxy,
            roughness_proxy=roughness_proxy,
            fluctuation_proxy=fluctuation_proxy,
        )

    # ------------------------------------------------------------------
    # Preservation sub-scores (Req 6.3, 17.5)
    # ------------------------------------------------------------------

    def _compute_preserve_sub_scores(
        self,
        original: np.ndarray,
        processed: np.ndarray,
        sr: int,
        classification_result: YamnetOutput,
        psychoacoustic: PsychoacousticFeatures,
    ) -> PreserveSubScores:
        """Compute the five preservation sub-scores."""
        min_len = min(len(original), len(processed))
        o, p = original[:min_len], processed[:min_len]

        # s_loud: loudness similarity
        # NOTE: clip removed — expose true value range (may go negative if
        # processed loudness differs greatly from original).
        rms_o = float(np.sqrt(np.mean(o ** 2)) + 1e-12)
        rms_p = float(np.sqrt(np.mean(p ** 2)) + 1e-12)
        s_loud = float(1.0 - abs(rms_o - rms_p) / (rms_o + 1e-12))

        # s_hf: high-frequency energy preservation (ratio is naturally in [0,1])
        n_fft = min(2048, min_len)
        if n_fft >= 4:
            spec_o = np.abs(np.fft.rfft(o[:n_fft]))
            spec_p = np.abs(np.fft.rfft(p[:n_fft]))
            freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
            hf = freqs >= 3000.0
            hf_o = float(np.sum(spec_o[hf] ** 2)) + 1e-12
            hf_p = float(np.sum(spec_p[hf] ** 2)) + 1e-12
            s_hf = float(min(hf_o, hf_p) / max(hf_o, hf_p))
        else:
            s_hf = 1.0

        # s_sc: spectral centroid similarity
        # NOTE: clip removed — expose true value range.
        if n_fft >= 4:
            sc_o = float(np.sum(freqs * spec_o) / (np.sum(spec_o) + 1e-12))
            sc_p = float(np.sum(freqs * spec_p) / (np.sum(spec_p) + 1e-12))
            max_sc = max(sc_o, sc_p, 1e-12)
            s_sc = float(1.0 - abs(sc_o - sc_p) / max_sc)
        else:
            s_sc = 1.0

        # s_con: classification confidence preservation
        # NOTE: clip removed — confidence from YAMNet is already in [0,1].
        if classification_result.predictions:
            s_con = float(classification_result.predictions[0].confidence)
        else:
            s_con = 0.0

        # s_psy: psychoacoustic quality (Req 17.5)
        s_psy = self._psychoacoustic_to_score(psychoacoustic)

        return PreserveSubScores(
            s_loud=s_loud,
            s_hf=s_hf,
            s_sc=s_sc,
            s_con=s_con,
            s_psy=s_psy,
        )

    @staticmethod
    def _psychoacoustic_to_score(features: PsychoacousticFeatures) -> float:
        """Map psychoacoustic features to a single sub-score.

        Uses a simple heuristic: penalise extreme loudness, high
        sharpness, high roughness, and high fluctuation.

        NOTE: outer clip removed — expose true value range.
        """
        # Loudness penalty: very quiet or very loud is bad
        loud_norm = 1.0 - abs(features.short_term_loudness + 20.0) / 60.0
        sharp_score = 1.0 - features.sharpness_proxy
        rough_score = 1.0 - features.roughness_proxy
        fluct_score = 1.0 - features.fluctuation_proxy

        return float(
            0.25 * loud_norm + 0.25 * sharp_score + 0.25 * rough_score + 0.25 * fluct_score
        )

    @staticmethod
    def _aggregate_preserve_score(sub: PreserveSubScores) -> float:
        """Weighted average of the five preservation sub-scores.

        NOTE: clip removed — expose true value range of preserve_score.
        """
        return float(
            0.20 * sub.s_loud
            + 0.20 * sub.s_hf
            + 0.20 * sub.s_sc
            + 0.20 * sub.s_con
            + 0.20 * sub.s_psy
        )

    # ------------------------------------------------------------------
    # Utility metrics (Req 6.4)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_utility_metrics(
        classification_result: YamnetOutput,
        ground_truth_label: str | None = None,
    ) -> dict:
        """Derive utility metrics from classification output.

        Two modes:

        • **Ground-truth mode** (``ground_truth_label`` provided — used during
          transfer-learning evaluation): metrics are REAL, computed by
          comparing the predicted labels against the ground-truth class.
            - accuracy : 1.0 if top-1 prediction == GT else 0.0 (TA@1)
            - mAP      : per-clip average precision = 1 / rank of the GT label
                         in the ranked predictions (0.0 if GT not predicted).
            - f1       : per-clip F1 for single-label classification, which
                         equals top-1 accuracy. True dataset macro-F1 must be
                         aggregated post-hoc from ``predicted_label`` + GT.
          Also returns ``predicted_label``, ``top1_correct``, ``top3_correct``,
          and ``metrics_source="ground_truth"``.

        • **Proxy mode** (no GT — raw-YAMNet runs without transfer labels):
          uses the classification confidence distribution as a stand-in.
            - mAP      : mean confidence of top-5 predictions
            - f1       : harmonic-mean proxy from top-1 confidence
            - accuracy : top-1 confidence
          Returns ``metrics_source="proxy"`` and GT fields as ``None``.
        """
        preds = classification_result.predictions or []

        # ── Ground-truth mode: real classification metrics ──
        if ground_truth_label is not None:
            ranked = [p.label for p in preds]
            predicted_label = ranked[0] if ranked else None
            # 1-based rank of the GT label in the ranked predictions
            try:
                rank = ranked.index(ground_truth_label) + 1
            except ValueError:
                rank = 0  # GT not present in predictions
            top1_correct = bool(predicted_label == ground_truth_label)
            top3_correct = bool(ground_truth_label in ranked[:3])
            accuracy = 1.0 if top1_correct else 0.0
            average_precision = (1.0 / rank) if rank > 0 else 0.0
            # Single-label per-clip F1 == top-1 accuracy; macro-F1 is post-hoc.
            f1 = accuracy
            return {
                "mAP": float(average_precision),
                "f1": float(f1),
                "accuracy": float(accuracy),
                "ground_truth_label": ground_truth_label,
                "predicted_label": predicted_label,
                "top1_correct": top1_correct,
                "top3_correct": top3_correct,
                "metrics_source": "ground_truth",
            }

        # ── Proxy mode: confidence-based stand-in (no ground truth) ──
        if not preds:
            return {
                "mAP": 0.0, "f1": 0.0, "accuracy": 0.0,
                "ground_truth_label": None, "predicted_label": None,
                "top1_correct": None, "top3_correct": None,
                "metrics_source": "proxy",
            }

        confidences = [p.confidence for p in preds]
        top_conf = confidences[0]
        top5_conf = confidences[:5]
        mean_top5 = float(np.mean(top5_conf))

        return {
            "mAP": float(np.clip(mean_top5, 0.0, 1.0)),
            "f1": float(np.clip(2.0 * top_conf / (top_conf + 1.0 + 1e-12), 0.0, 1.0)),
            "accuracy": float(np.clip(top_conf, 0.0, 1.0)),
            "ground_truth_label": None,
            "predicted_label": preds[0].label,
            "top1_correct": None,
            "top3_correct": None,
            "metrics_source": "proxy",
        }
