"""StrongBlurringTool — Fixed Baseline voice blurring via composite multi-technique processing.

Applies lowpass filtering, high-band mixing, mid-band preservation with gain,
noise injection, and optional pitch shifting on speech segments only.
Non-speech segments are preserved bit-identical to the original.

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
"""

from __future__ import annotations

import os
import tempfile
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, resample_poly

from src.contracts.transform_contracts import (
    TransformParams,
    TransformRecipeRef,
    TransformResult,
)
from src.contracts.vad_contracts import SpeechSegment


class StrongBlurringTool:
    """Apply composite multi-technique blurring on speech segments only.

    Parameters
    ----------
    output_dir : str | None
        Directory for writing blurred WAV files.  Defaults to a temp directory.
    """

    RECIPE_NAME = "RECIPE_LOWPASS_HIGHPASS_MIX"
    RECIPE_VERSION = "1.0"

    def __init__(self, output_dir: str | None = None) -> None:
        self._output_dir = output_dir or tempfile.mkdtemp(prefix="strong_blur_")

    def run(
        self,
        wav_path: str,
        segments: list[SpeechSegment],
        lowpass_cutoff: int = 1000,
        lowpass_mix: float = 0.55,
        highband_start: int = 2500,
        highband_mix: float = 0.15,
        midband_range: tuple[int, int] = (1200, 2200),
        midband_gain_db: float = 2.0,
        noise_band: tuple[int, int] = (700, 2800),
        noise_snr_db: float = 18.0,
        band_hz: tuple[int, int] | None = None,
        atten_db: float = 0.0,
        pitch_shift_semitones: float = 0.0,
        chunk_id: str = "",
        trial: int = 0,
    ) -> TransformResult:
        """Apply composite blurring on speech segments.

        The techniques applied to each speech segment are:

        1. **Lowpass filtering** — mix the lowpass-filtered signal at *lowpass_mix*
           with the original at *(1 - lowpass_mix)*.
        2. **High-band mixing** — add high-frequency content (above
           *highband_start* Hz) scaled by *highband_mix*.
        3. **Mid-band preservation** — boost the *midband_range* by
           *midband_gain_db* to retain environmental tonal cues.
        4. **Noise injection** — add band-limited noise in *noise_band* at the
           specified *noise_snr_db* relative to the speech signal energy.
        5. **Pitch shifting** (optional) — shift pitch by *pitch_shift_semitones*
           to degrade speaker identity.

        Non-speech regions are copied bit-identical from the original.

        Returns
        -------
        TransformResult
            Contains the path to the blurred WAV and metadata.
        """
        data, sr = sf.read(wav_path, dtype="float64")
        nyquist = sr / 2.0

        # Pre-compute filtered versions of the full signal
        lowpassed = self._lowpass(data, sr, lowpass_cutoff)
        highpassed = self._highpass(data, sr, highband_start)
        midband_boosted = self._boost_band(data, sr, midband_range, midband_gain_db)

        # Pre-compute mid-band attenuated version if band_hz is specified
        if band_hz is not None and atten_db > 0:
            attenuated = self._attenuate_band(data, sr, band_hz, atten_db)
        else:
            attenuated = None

        # Compose output — start with an exact copy (non-speech stays intact)
        output = data.copy()

        for seg in segments:
            s = max(0, int(seg.start * sr))
            e = min(len(data), int(seg.end * sr))
            if s >= e:
                continue

            seg_orig = data[s:e]

            # 0. Mid-band attenuation (if enabled)
            if attenuated is not None:
                seg_orig = attenuated[s:e]

            # 1. Lowpass mix
            blurred = lowpass_mix * lowpassed[s:e] + (1.0 - lowpass_mix) * seg_orig

            # 2. High-band mixing — add scaled high-frequency content
            blurred = blurred + highband_mix * highpassed[s:e]

            # 3. Mid-band preservation — replace mid-band with boosted version
            blurred = self._apply_midband_preservation(
                blurred, midband_boosted[s:e], sr, midband_range,
            )

            # 4. Noise injection in noise_band at noise_snr_db
            blurred = self._inject_noise(blurred, sr, noise_band, noise_snr_db)

            # 5. Pitch shifting (optional)
            if pitch_shift_semitones != 0.0:
                blurred = self._pitch_shift(blurred, sr, pitch_shift_semitones)

            output[s:e] = blurred

        # Write output WAV
        os.makedirs(self._output_dir, exist_ok=True)
        out_filename = f"{chunk_id}_strong_blur.wav" if chunk_id else "strong_blur.wav"
        blurred_wav_path = os.path.join(self._output_dir, out_filename)
        sf.write(blurred_wav_path, output, sr, subtype="PCM_16")

        recipe_ref = TransformRecipeRef(
            recipe_name=self.RECIPE_NAME,
            version=self.RECIPE_VERSION,
        )
        params = TransformParams(
            recipe_ref=recipe_ref,
            params={
                "lowpass_cutoff": lowpass_cutoff,
                "lowpass_mix": lowpass_mix,
                "highband_start": highband_start,
                "highband_mix": highband_mix,
                "midband_range": list(midband_range),
                "midband_gain_db": midband_gain_db,
                "noise_band": list(noise_band),
                "noise_snr_db": noise_snr_db,
                "band_hz": list(band_hz) if band_hz else None,
                "atten_db": atten_db,
                "pitch_shift_semitones": pitch_shift_semitones,
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

    # ------------------------------------------------------------------
    # DSP helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _lowpass(data: np.ndarray, sr: int, cutoff: int) -> np.ndarray:
        """Apply a zero-phase lowpass filter at *cutoff* Hz."""
        nyquist = sr / 2.0
        sos = butter(N=5, Wn=cutoff / nyquist, btype="low", output="sos")
        return sosfiltfilt(sos, data)

    @staticmethod
    def _attenuate_band(
        data: np.ndarray, sr: int, band_hz: tuple[int, int], atten_db: float,
    ) -> np.ndarray:
        """Attenuate *band_hz* by *atten_db* using a bandpass Butterworth filter."""
        low, high = band_hz
        nyquist = sr / 2.0
        sos_bp = butter(N=5, Wn=[low / nyquist, high / nyquist], btype="bandpass", output="sos")
        band_signal = sosfiltfilt(sos_bp, data)
        gain = 10.0 ** (-atten_db / 20.0)
        return data - band_signal * (1.0 - gain)

    @staticmethod
    def _highpass(data: np.ndarray, sr: int, cutoff: int) -> np.ndarray:
        """Apply a zero-phase highpass filter at *cutoff* Hz."""
        nyquist = sr / 2.0
        sos = butter(N=5, Wn=cutoff / nyquist, btype="high", output="sos")
        return sosfiltfilt(sos, data)

    @staticmethod
    def _boost_band(
        data: np.ndarray,
        sr: int,
        band: tuple[int, int],
        gain_db: float,
    ) -> np.ndarray:
        """Return a copy of *data* with *band* boosted by *gain_db*."""
        nyquist = sr / 2.0
        low, high = band
        sos_bp = butter(
            N=5,
            Wn=[low / nyquist, high / nyquist],
            btype="bandpass",
            output="sos",
        )
        band_signal = sosfiltfilt(sos_bp, data)
        gain = 10.0 ** (gain_db / 20.0)
        return data + band_signal * (gain - 1.0)

    @staticmethod
    def _apply_midband_preservation(
        blurred: np.ndarray,
        boosted: np.ndarray,
        sr: int,
        midband_range: tuple[int, int],
    ) -> np.ndarray:
        """Replace the mid-band content in *blurred* with the boosted version.

        Extracts the mid-band from both signals, removes it from *blurred*,
        and adds the boosted mid-band back.
        """
        nyquist = sr / 2.0
        low, high = midband_range
        sos_bp = butter(
            N=5,
            Wn=[low / nyquist, high / nyquist],
            btype="bandpass",
            output="sos",
        )
        blurred_mid = sosfiltfilt(sos_bp, blurred)
        boosted_mid = sosfiltfilt(sos_bp, boosted)
        return blurred - blurred_mid + boosted_mid

    @staticmethod
    def _inject_noise(
        signal: np.ndarray,
        sr: int,
        noise_band: tuple[int, int],
        snr_db: float,
    ) -> np.ndarray:
        """Add band-limited Gaussian noise at the target SNR."""
        nyquist = sr / 2.0
        low, high = noise_band

        # Generate white noise and bandpass-filter it
        rng = np.random.default_rng()
        white = rng.standard_normal(len(signal))
        sos_bp = butter(
            N=5,
            Wn=[low / nyquist, high / nyquist],
            btype="bandpass",
            output="sos",
        )
        band_noise = sosfiltfilt(sos_bp, white)

        # Scale noise to achieve the desired SNR relative to signal energy
        sig_power = np.mean(signal ** 2) + 1e-12  # avoid division by zero
        noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        current_noise_power = np.mean(band_noise ** 2) + 1e-12
        scale = np.sqrt(noise_power / current_noise_power)

        return signal + scale * band_noise

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
        up = 100
        down = max(1, int(round(up * ratio)))
        d = gcd(up, down)
        up, down = up // d, down // d

        resampled = resample_poly(data, up, down)

        orig_len = len(data)
        if len(resampled) >= orig_len:
            return resampled[:orig_len]
        else:
            out = np.zeros(orig_len, dtype=data.dtype)
            out[: len(resampled)] = resampled
            return out
