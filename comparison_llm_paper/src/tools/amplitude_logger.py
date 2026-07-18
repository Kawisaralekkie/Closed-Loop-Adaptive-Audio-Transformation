"""AmplitudeLogger — compute amplitude-array statistics before/after transform.

Runs inside the cloud pipeline (ECS) and produces a compact summary of the
time-domain amplitude arrays of the original and processed audio, plus an
optional full amplitude dump persisted as an artifact for offline analysis.

The summary (min/max/mean/rms/...) is stored in ``ChunkReport.amplitude_stats``
so it appears directly in the run report JSON. The optional full dump is a
``.npz`` file written next to the blurred WAV and handed to the DataLakeWriter.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def _load_mono(path: str) -> tuple[np.ndarray, int]:
    """Load a WAV file as a mono float64 amplitude array."""
    data, sr = sf.read(path, dtype="float64")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


def _array_stats(x: np.ndarray) -> dict:
    """Summary statistics of a 1-D amplitude array."""
    if x.size == 0:
        return {
            "n_samples": 0, "min": 0.0, "max": 0.0, "mean": 0.0,
            "abs_mean": 0.0, "rms": 0.0, "std": 0.0, "peak": 0.0,
        }
    return {
        "n_samples": int(x.size),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "abs_mean": float(np.mean(np.abs(x))),
        "rms": float(np.sqrt(np.mean(x ** 2))),
        "std": float(np.std(x)),
        "peak": float(np.max(np.abs(x))),
    }


def compute_amplitude_stats(
    original_path: str,
    processed_path: str,
    *,
    dump_full_array: bool = False,
    dump_dir: str | None = None,
    chunk_id: str = "",
) -> tuple[dict, str | None]:
    """Compute amplitude-array stats for original vs processed audio.

    Parameters
    ----------
    original_path : str
        Path to the original (pre-transform) WAV.
    processed_path : str
        Path to the processed (post-transform / blurred) WAV.
    dump_full_array : bool
        When True, persist the full amplitude + spectrum arrays as a
        compressed ``.npz`` artifact for offline analysis.
    dump_dir : str | None
        Directory for the ``.npz`` dump. Defaults to the processed file's
        directory.
    chunk_id : str
        Chunk identifier used in the dump filename.

    Returns
    -------
    tuple[dict, str | None]
        ``(amplitude_stats, npz_path_or_None)``.
        ``amplitude_stats`` is JSON-serialisable and meant for the report.
    """
    try:
        orig, sr_o = _load_mono(original_path)
        proc, sr_p = _load_mono(processed_path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("AMPLITUDE: failed to load wavs for chunk=%s: %s", chunk_id, exc)
        return {}, None

    sr = sr_o
    # Align lengths so the per-sample difference is meaningful
    n = max(len(orig), len(proc))
    if len(orig) < n:
        orig = np.pad(orig, (0, n - len(orig)))
    if len(proc) < n:
        proc = np.pad(proc, (0, n - len(proc)))

    diff = proc - orig
    stats = {
        "sample_rate": int(sr),
        "duration_s": round(n / sr, 4) if sr else 0.0,
        "original": _array_stats(orig),
        "processed": _array_stats(proc),
        # How much the waveform changed overall
        "difference": {
            "abs_mean": float(np.mean(np.abs(diff))),
            "rms": float(np.sqrt(np.mean(diff ** 2))),
            "max_abs": float(np.max(np.abs(diff))) if n else 0.0,
        },
    }

    npz_path: str | None = None
    if dump_full_array:
        out_dir = dump_dir or os.path.dirname(processed_path) or "."
        os.makedirs(out_dir, exist_ok=True)
        stem = chunk_id or os.path.splitext(os.path.basename(processed_path))[0]
        npz_path = os.path.join(out_dir, f"{stem}_amplitude.npz")
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        np.savez_compressed(
            npz_path,
            sample_rate=sr,
            amp_original=orig.astype(np.float32),
            amp_processed=proc.astype(np.float32),
            freqs_hz=freqs.astype(np.float32),
            spectrum_original=np.abs(np.fft.rfft(orig)).astype(np.float32),
            spectrum_processed=np.abs(np.fft.rfft(proc)).astype(np.float32),
        )
        logger.info("AMPLITUDE: dumped full arrays for chunk=%s → %s", chunk_id, npz_path)

    return stats, npz_path
