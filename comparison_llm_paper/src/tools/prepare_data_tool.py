"""PrepareDataTool — audio ingestion, validation, canonicalization, and chunking.

Validates WAV PCM mono input, canonicalizes to 16 kHz / 16-bit / mono,
and chunks into fixed-length overlapping segments.

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
"""

from __future__ import annotations

import math
import os
import struct
import tempfile
import wave
from pathlib import Path
from uuid import UUID

import numpy as np
import soundfile as sf
from pydantic import ValidationError as PydanticValidationError

from src.config import AudioConfig, config
from src.contracts.audio_contracts import (
    AudioChunk,
    AudioIngestRequest,
    AudioIngestResponse,
    CanonicalAudio,
)
from src.contracts.core_ids import CoreIds


class ValidationError(Exception):
    """Raised when input audio fails format validation (Req 1.1, 1.6)."""


class PrepareDataTool:
    """Ingest, validate, canonicalize, and chunk raw audio.

    Parameters
    ----------
    audio_config : AudioConfig | None
        Override default audio settings (sample rate, bit depth, window, overlap).
    output_dir : str | None
        Base directory for writing chunk WAV files.  Defaults to a temp directory.
    """

    def __init__(
        self,
        audio_config: AudioConfig | None = None,
        output_dir: str | None = None,
    ) -> None:
        self._cfg = audio_config or config.audio
        self._output_dir = output_dir or tempfile.mkdtemp(prefix="prepare_data_")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        request: AudioIngestRequest,
        timestamp: str | None = None,
    ) -> AudioIngestResponse:
        """Execute the full ingest pipeline.

        Parameters
        ----------
        request : AudioIngestRequest
            Incoming request with ``source_id`` and ``raw_audio_path``.
        timestamp : str | None
            ISO-8601 timestamp for deterministic ``run_id`` generation.
            Defaults to current UTC time if not provided.

        Returns
        -------
        AudioIngestResponse
            Contains the deterministic ``run_id``, canonical audio metadata,
            and a list of ``AudioChunk`` objects.

        Raises
        ------
        ValidationError
            If the input file is not valid WAV PCM mono.
        """
        if timestamp is None:
            from datetime import datetime, timezone

            timestamp = datetime.now(timezone.utc).isoformat()

        # 1. Validate input (Req 1.1, 1.6)
        self._validate_wav_pcm_mono(request.raw_audio_path)

        # 2. Generate deterministic run_id (Req 13.2)
        run_id: UUID = CoreIds.generate_run_id(request.source_id, timestamp)

        # 3. Canonicalize (Req 1.2)
        canonical = self._canonicalize(request.raw_audio_path, run_id)

        # 4. Chunk (Req 1.3, 1.4, 1.5)
        chunks = self._chunk(canonical, run_id)

        return AudioIngestResponse(
            run_id=run_id,
            source_id=request.source_id,
            canonical_audio=canonical,
            chunks=chunks,
        )

    # ------------------------------------------------------------------
    # Validation (Req 1.1, 1.6)
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_wav_pcm_mono(path: str) -> None:
        """Raise ``ValidationError`` if *path* is not a valid WAV PCM mono file."""
        if not os.path.isfile(path):
            raise ValidationError(f"File not found: {path}")

        try:
            with wave.open(path, "rb") as wf:
                # WAV PCM has compression type 'NONE'
                if wf.getcompname() != "not compressed":
                    raise ValidationError(
                        f"Audio is not PCM (compression: {wf.getcompname()})"
                    )
                if wf.getnchannels() != 1:
                    raise ValidationError(
                        f"Audio is not mono (channels: {wf.getnchannels()})"
                    )
        except wave.Error as exc:
            raise ValidationError(f"Invalid WAV file: {exc}") from exc
        except struct.error as exc:
            raise ValidationError(f"Corrupt WAV header: {exc}") from exc

    # ------------------------------------------------------------------
    # Canonicalization (Req 1.2)
    # ------------------------------------------------------------------

    def _canonicalize(self, raw_path: str, run_id: UUID) -> CanonicalAudio:
        """Read *raw_path* and write a canonical 16 kHz / 16-bit / mono WAV."""
        data, orig_sr = sf.read(raw_path, dtype="float64")

        # Ensure 1-D (mono) — should already be validated, but be safe.
        if data.ndim > 1:
            data = data[:, 0]

        # Resample to target sample rate if needed.
        target_sr = self._cfg.sample_rate
        if orig_sr != target_sr:
            data = self._resample(data, orig_sr, target_sr)

        # Write canonical WAV (16-bit PCM).
        run_dir = os.path.join(self._output_dir, str(run_id))
        os.makedirs(run_dir, exist_ok=True)
        canonical_path = os.path.join(run_dir, "canonical.wav")
        sf.write(canonical_path, data, target_sr, subtype="PCM_16")

        duration = len(data) / target_sr

        return CanonicalAudio(
            wav_path=canonical_path,
            sample_rate=target_sr,
            bit_depth=self._cfg.bit_depth,
            channels=self._cfg.channels,
            duration_seconds=duration,
        )

    @staticmethod
    def _resample(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample *data* from *orig_sr* to *target_sr* using scipy."""
        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(int(orig_sr), int(target_sr))
        up = int(target_sr) // g
        down = int(orig_sr) // g
        return resample_poly(data, up, down).astype(data.dtype)

    # ------------------------------------------------------------------
    # Chunking (Req 1.3, 1.4, 1.5)
    # ------------------------------------------------------------------

    def _chunk(self, canonical: CanonicalAudio, run_id: UUID) -> list[AudioChunk]:
        """Split canonical audio into overlapping chunks.

        Chunk count = ceil((D - O) / (W - O))  where D = duration, W = window, O = overlap.
        Each chunk file is written as ``{run_id}/{chunk_id}.wav``.
        """
        data, sr = sf.read(canonical.wav_path, dtype="float64")
        duration = canonical.duration_seconds
        window = self._cfg.window_size
        overlap = self._cfg.overlap

        # Compute number of chunks (Property 2: Chunking Arithmetic).
        if duration <= window:
            n_chunks = 1
        else:
            n_chunks = math.ceil((duration - overlap) / (window - overlap))

        step_samples = int((window - overlap) * sr)
        window_samples = int(window * sr)
        total_samples = len(data)

        run_dir = os.path.join(self._output_dir, str(run_id))
        os.makedirs(run_dir, exist_ok=True)

        chunks: list[AudioChunk] = []
        for i in range(n_chunks):
            start_sample = i * step_samples
            end_sample = min(start_sample + window_samples, total_samples)
            chunk_data = data[start_sample:end_sample]

            chunk_id = CoreIds.generate_chunk_id(run_id, i)
            chunk_path = os.path.join(run_dir, f"{chunk_id}.wav")
            sf.write(chunk_path, chunk_data, sr, subtype="PCM_16")

            start_time = start_sample / sr
            end_time = end_sample / sr

            chunks.append(
                AudioChunk(
                    chunk_id=chunk_id,
                    run_id=run_id,
                    wav_path=chunk_path,
                    start_time=start_time,
                    end_time=end_time,
                    duration=end_time - start_time,
                )
            )

        return chunks
