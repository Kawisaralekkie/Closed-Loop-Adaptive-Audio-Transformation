"""MidBandAttenuationTool — Fixed Baseline voice blurring via mid-band attenuation.

Attenuates the core speech frequency band ([500, 3000] Hz) by a configurable
decibel amount on speech segments only, preserving non-speech segments
bit-identical to the original.  Optionally applies pitch shifting to
degrade speaker identity.

Requirements: 3.1, 3.2, 3.3, 3.4
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, resample_poly
from math import gcd

from src.contracts.transform_contracts import (
    TransformParams,
    TransformRecipeRef,
    TransformResult,
)
from src.contracts.vad_contracts import SpeechSegment


class MidBandAttenuationTool:
    """Attenuate the mid-band speech frequencies on speech segments only.

    Parameters
    ----------
    output_dir : str | None
        Directory for writing blurred WAV files. Defaults to a temp directory.
    """

    RECIPE_NAME = "RECIPE_MID_BAND_ATTEN"
    RECIPE_VERSION = "1.0"

    def __init__(self, output_dir: str | None = None) -> None:
        self._output_dir = output_dir or tempfile.mkdtemp(prefix="mid_band_atten_")

    def run(
        self,
        wav_path: str,
        segments: list[SpeechSegment],
        band_hz: tuple[int, int] = (500, 3000),
        atten_db: float = 20.0,
        lowpass_cutoff: int = 0,
        pitch_shift_semitones: float = 0.0,
        scope: str = "speech_only",
        chunk_id: str = "",
        trial: int = 0,
    ) -> TransformResult:
        """Attenuate *band_hz* by *atten_db* on speech segments.

        Non-speech regions are copied bit-identical from the original.
        Optionally applies a lowpass filter at *lowpass_cutoff* Hz
        (0 = disabled).
        """
        data, sr = sf.read(wav_path, dtype="float64")

        # Build the attenuated version of the full signal
        attenuated = self._attenuate_band(data, sr, band_hz, atten_db)

        # Compose output: speech segments get attenuated audio,
        # non-speech segments stay bit-identical to original.
        output = data.copy()
        for seg in segments:
            start_sample = int(seg.start * sr)
            end_sample = int(seg.end * sr)
            start_sample = max(0, start_sample)
            end_sample = min(len(data), end_sample)
            seg_audio = attenuated[start_sample:end_sample]

            # Apply lowpass filter if requested
            if lowpass_cutoff > 0:
                seg_audio = self._lowpass(seg_audio, sr, lowpass_cutoff)

            # Apply pitch shifting if requested
            if pitch_shift_semitones != 0.0:
                seg_audio = self._pitch_shift(seg_audio, sr, pitch_shift_semitones)

            output[start_sample:end_sample] = seg_audio

        # Write output WAV
        os.makedirs(self._output_dir, exist_ok=True)
        out_filename = f"{chunk_id}_mid_band_atten.wav" if chunk_id else "mid_band_atten.wav"
        blurred_wav_path = os.path.join(self._output_dir, out_filename)
        sf.write(blurred_wav_path, output, sr, subtype="PCM_16")

        recipe_ref = TransformRecipeRef(
            recipe_name=self.RECIPE_NAME,
            version=self.RECIPE_VERSION,
        )
        params = TransformParams(
            recipe_ref=recipe_ref,
            params={
                "band_hz": list(band_hz),
                "atten_db": atten_db,
                "lowpass_cutoff": lowpass_cutoff,
                "pitch_shift_semitones": pitch_shift_semitones,
                "scope": scope,
            },
            trial=trial,
        )

        return TransformResult(
            chunk_id=chunk_id,
            recipe_ref=recipe_ref,
            params=params,
            blurred_wav_path=blurred_wav_path,
            trial=trial,
            success=True,
        )

    @staticmethod
    def _lowpass(data: np.ndarray, sr: int, cutoff: int) -> np.ndarray:
        """Apply a zero-phase lowpass Butterworth filter at *cutoff* Hz."""
        nyquist = sr / 2.0
        sos = butter(N=5, Wn=cutoff / nyquist, btype="low", output="sos")
        return sosfiltfilt(sos, data)

    @staticmethod
    def _attenuate_band(
        data: np.ndarray,
        sr: int,
        band_hz: tuple[int, int],
        atten_db: float,
    ) -> np.ndarray:
        """Return a copy of *data* with *band_hz* attenuated by *atten_db*.

        Uses a bandpass filter to isolate the target band, scales it down,
        and subtracts the difference from the original signal.
        """
        low, high = band_hz
        nyquist = sr / 2.0

        # Design a bandpass filter to extract the target band
        sos_bp = butter(
            N=5,
            Wn=[low / nyquist, high / nyquist],
            btype="bandpass",
            output="sos",
        )
        band_signal = sosfiltfilt(sos_bp, data)

        # Compute the linear gain reduction
        gain = 10.0 ** (-atten_db / 20.0)

        # attenuated = original - band + band * gain
        #            = original - band * (1 - gain)
        attenuated = data - band_signal * (1.0 - gain)

        return attenuated

    @staticmethod
    def _pitch_shift(
        data: np.ndarray,
        sr: int,
        semitones: float,
    ) -> np.ndarray:
        """Shift pitch by resampling (no external dependencies).

        Resamples the signal to change its pitch, then truncates or
        zero-pads to match the original length.
        """
        ratio = 2.0 ** (semitones / 12.0)
        # Use rational approximation for resample_poly
        # Shift up → resample to shorter → stretch back
        # We resample by factor (1/ratio) then take original length
        up = 100
        down = max(1, int(round(up * ratio)))
        d = gcd(up, down)
        up, down = up // d, down // d

        resampled = resample_poly(data, up, down)

        # Match original length
        orig_len = len(data)
        if len(resampled) >= orig_len:
            return resampled[:orig_len]
        else:
            out = np.zeros(orig_len, dtype=data.dtype)
            out[: len(resampled)] = resampled
            return out
