"""SourceSeparationTool — nussl-based speech extraction, blur, and remix.

Separates speech from environmental audio using nussl, applies the specified
blurring method (MidBandAttenuation or StrongBlurring) on the speech track
only, then remixes the blurred speech with the unmodified residual track.
Falls back to StrongBlurringTool when separation fails or quality is too low.

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 8.1, 8.2, 8.3, 8.4, 10.3
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any

import numpy as np
import soundfile as sf

from src.contracts.source_separation_contracts import SourceSeparationResult
from src.contracts.transform_contracts import (
    TransformParams,
    TransformRecipeRef,
    TransformResult,
)
from src.contracts.vad_contracts import SpeechSegment
from src.tools.mid_band_attenuation_tool import MidBandAttenuationTool
from src.tools.strong_blurring_tool import StrongBlurringTool

logger = logging.getLogger(__name__)


class SourceSeparationTool:
    """Separate speech via nussl, blur only the speech track, then remix.

    Parameters
    ----------
    output_dir : str | None
        Directory for writing output WAV files.  Defaults to a temp directory.
    mid_band_tool : MidBandAttenuationTool | None
        Tool for mid-band attenuation blurring on the speech track.
    strong_blur_tool : StrongBlurringTool | None
        Tool for strong blurring on the speech track and for fallback.
    timeout_seconds : float
        Maximum time allowed for nussl separation before fallback.
    min_separation_quality : float
        Minimum separation_quality_score; below this triggers fallback.
    """

    RECIPE_NAME = "RECIPE_SOURCE_SEPARATION"
    RECIPE_VERSION = "1.0"

    def __init__(
        self,
        output_dir: str | None = None,
        mid_band_tool: MidBandAttenuationTool | None = None,
        strong_blur_tool: StrongBlurringTool | None = None,
        timeout_seconds: float = 30.0,
        min_separation_quality: float = 0.3,
    ) -> None:
        self._output_dir = output_dir or tempfile.mkdtemp(prefix="source_sep_")
        self._mid_band_tool = mid_band_tool or MidBandAttenuationTool(output_dir=self._output_dir)
        self._strong_blur_tool = strong_blur_tool or StrongBlurringTool(output_dir=self._output_dir)
        self._timeout_seconds = timeout_seconds
        self._min_separation_quality = min_separation_quality
        self.last_separation_result: SourceSeparationResult | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        wav_path: str,
        segments: list[SpeechSegment],
        blur_method: str = "strong_blur",
        blur_params: dict[str, Any] | None = None,
        chunk_id: str = "",
        trial: int = 0,
    ) -> TransformResult:
        """Separate speech via nussl, blur the speech track, and remix.

        Parameters
        ----------
        wav_path : str
            Path to the input WAV file (16 kHz / 16-bit / mono).
        segments : list[SpeechSegment]
            Speech segments from VAD.
        blur_method : str
            ``"midband_attenuation"`` or ``"strong_blur"`` — applied to the
            speech track only.
        blur_params : dict | None
            Extra keyword arguments forwarded to the chosen blur tool.
        chunk_id : str
            Identifier for the chunk being processed.
        trial : int
            Trial number for retry-ladder tracking.

        Returns
        -------
        TransformResult
            Contains the path to the remixed (or fallback-blurred) WAV.
        """
        blur_params = blur_params or {}

        try:
            speech_wav, residual_wav, quality_score, elapsed_ms = self._separate(
                wav_path, chunk_id,
            )
        except Exception as exc:
            logger.warning(
                "Source separation failed for chunk_id=%s error_message=%s "
                "fallback_strategy=StrongBlurringTool",
                chunk_id,
                str(exc),
            )
            self.last_separation_result = SourceSeparationResult(
                chunk_id=chunk_id,
                speech_wav_path="",
                residual_wav_path="",
                remix_wav_path="",
                separation_quality_score=0.0,
                processing_time_ms=0.0,
                nussl_model_name="TimbreClustering",
                fallback_used=True,
            )
            return self._fallback_blur(wav_path, segments, chunk_id, trial)

        if quality_score < self._min_separation_quality:
            logger.warning(
                "Separation quality too low for chunk_id=%s "
                "separation_quality_score=%.3f fallback_strategy=StrongBlurringTool",
                chunk_id,
                quality_score,
            )
            self.last_separation_result = SourceSeparationResult(
                chunk_id=chunk_id,
                speech_wav_path=speech_wav,
                residual_wav_path=residual_wav,
                remix_wav_path="",
                separation_quality_score=quality_score,
                processing_time_ms=elapsed_ms,
                nussl_model_name="TimbreClustering",
                fallback_used=True,
            )
            return self._fallback_blur(wav_path, segments, chunk_id, trial)

        # Blur only the speech track using the requested method
        blur_result = self._blur_speech_track(
            speech_wav, segments, blur_method, blur_params, chunk_id, trial,
        )

        # Remix blurred speech with unmodified residual
        remix_path = self._remix(blur_result.blurred_wav_path, residual_wav, chunk_id)

        self.last_separation_result = SourceSeparationResult(
            chunk_id=chunk_id,
            speech_wav_path=speech_wav,
            residual_wav_path=residual_wav,
            remix_wav_path=remix_path,
            separation_quality_score=quality_score,
            processing_time_ms=elapsed_ms,
            nussl_model_name="TimbreClustering",
            fallback_used=False,
        )

        recipe_ref = TransformRecipeRef(
            recipe_name=self.RECIPE_NAME,
            version=self.RECIPE_VERSION,
        )
        params = TransformParams(
            recipe_ref=recipe_ref,
            params={
                "blur_method": blur_method,
                **blur_params,
            },
            trial=trial,
        )

        return TransformResult(
            chunk_id=chunk_id,
            recipe_ref=recipe_ref,
            params=params,
            blurred_wav_path=remix_path,
            trial=trial,
            success=True,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _separate(self, wav_path: str, chunk_id: str = "") -> tuple[str, str, float, float]:
        """Run nussl source separation with a timeout.

        Returns
        -------
        tuple[str, str, float, float]
            (speech_wav_path, residual_wav_path, separation_quality_score,
             processing_time_ms)

        Raises
        ------
        Exception
            Propagated from nussl or on timeout.
        """
        start_ts = time.monotonic()

        def _do_separate() -> tuple[np.ndarray, np.ndarray, int, float]:
            # Patch scipy.signal for nussl compatibility (hamming/hanning
            # were removed in newer scipy versions).
            import scipy.signal as _sig
            import scipy.signal.windows as _win

            for _name in ("hamming", "hanning", "hann", "blackman", "bartlett"):
                if not hasattr(_sig, _name) and hasattr(_win, _name):
                    setattr(_sig, _name, getattr(_win, _name))

            # Mock soxbindings if not available (nussl works without it)
            import sys
            import types as _types

            if "soxbindings" not in sys.modules:
                _mock = _types.ModuleType("soxbindings")
                _mock.SoxError = Exception  # type: ignore[attr-defined]
                sys.modules["soxbindings"] = _mock

            import nussl  # lazy import — heavy dependency

            data, sr = sf.read(wav_path, dtype="float64")
            signal = nussl.AudioSignal(audio_data_array=data, sample_rate=sr)
            separator = nussl.separation.primitive.TimbreClustering(signal)
            estimates = separator()

            speech = estimates[0].audio_data.squeeze()
            residual = estimates[1].audio_data.squeeze() if len(estimates) > 1 else data - speech

            # Quality heuristic: energy ratio of speech vs original
            orig_energy = np.mean(data ** 2) + 1e-12
            speech_energy = np.mean(speech ** 2) + 1e-12
            quality = float(np.clip(speech_energy / orig_energy, 0.0, 1.0))

            return speech, residual, sr, quality

        # Execute with timeout via ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_do_separate)
            try:
                speech, residual, sr, quality = future.result(timeout=self._timeout_seconds)
            except FuturesTimeoutError:
                future.cancel()
                raise TimeoutError(
                    f"nussl separation timed out after {self._timeout_seconds}s "
                    f"for chunk_id={chunk_id}"
                )

        elapsed_ms = (time.monotonic() - start_ts) * 1000.0

        # Write separated tracks to disk
        os.makedirs(self._output_dir, exist_ok=True)
        speech_path = os.path.join(
            self._output_dir,
            f"{chunk_id}_speech.wav" if chunk_id else "speech.wav",
        )
        residual_path = os.path.join(
            self._output_dir,
            f"{chunk_id}_residual.wav" if chunk_id else "residual.wav",
        )
        sf.write(speech_path, speech, sr, subtype="PCM_16")
        sf.write(residual_path, residual, sr, subtype="PCM_16")

        return speech_path, residual_path, quality, elapsed_ms

    def _blur_speech_track(
        self,
        speech_wav: str,
        segments: list[SpeechSegment],
        blur_method: str,
        blur_params: dict[str, Any],
        chunk_id: str,
        trial: int,
    ) -> TransformResult:
        """Apply the chosen blur method on the separated speech track."""
        if blur_method == "midband_attenuation":
            return self._mid_band_tool.run(
                wav_path=speech_wav,
                segments=segments,
                atten_db=blur_params.get("atten_db", 20.0),
                pitch_shift_semitones=blur_params.get("pitch_shift_semitones", 0.0),
                chunk_id=chunk_id,
                trial=trial,
            )
        # Default to strong blur
        return self._strong_blur_tool.run(
            wav_path=speech_wav,
            segments=segments,
            noise_snr_db=blur_params.get("noise_snr_db", 18.0),
            pitch_shift_semitones=blur_params.get("pitch_shift_semitones", 0.0),
            chunk_id=chunk_id,
            trial=trial,
        )

    def _remix(self, blurred_speech_path: str, residual_path: str, chunk_id: str = "") -> str:
        """Sum blurred speech with unmodified residual and write to disk.

        Returns
        -------
        str
            Path to the remixed WAV file.
        """
        blurred_speech, sr = sf.read(blurred_speech_path, dtype="float64")
        residual, _ = sf.read(residual_path, dtype="float64")

        # Ensure same length — pad shorter with zeros
        max_len = max(len(blurred_speech), len(residual))
        if len(blurred_speech) < max_len:
            blurred_speech = np.pad(blurred_speech, (0, max_len - len(blurred_speech)))
        if len(residual) < max_len:
            residual = np.pad(residual, (0, max_len - len(residual)))

        remix = blurred_speech + residual

        os.makedirs(self._output_dir, exist_ok=True)
        remix_filename = f"{chunk_id}_remix.wav" if chunk_id else "remix.wav"
        remix_path = os.path.join(self._output_dir, remix_filename)
        sf.write(remix_path, remix, sr, subtype="PCM_16")

        return remix_path

    def _fallback_blur(
        self,
        wav_path: str,
        segments: list[SpeechSegment],
        chunk_id: str,
        trial: int,
    ) -> TransformResult:
        """Fall back to StrongBlurringTool on the original audio."""
        result = self._strong_blur_tool.run(
            wav_path=wav_path,
            segments=segments,
            chunk_id=chunk_id,
            trial=trial,
        )
        # Override recipe ref so the caller knows source separation was attempted
        recipe_ref = TransformRecipeRef(
            recipe_name=self.RECIPE_NAME,
            version=self.RECIPE_VERSION,
        )
        return TransformResult(
            chunk_id=result.chunk_id,
            recipe_ref=recipe_ref,
            params=TransformParams(
                recipe_ref=recipe_ref,
                params={"fallback_used": True},
                trial=trial,
            ),
            blurred_wav_path=result.blurred_wav_path,
            trial=trial,
            success=True,
        )
