"""SpeechScanTool — Voice Activity Detection using Silero VAD.

Runs Silero VAD on an AudioChunk and returns a VADResult with detected
SpeechSegments and an overall speech_ratio.

Requirements: 2.1, 2.2
"""

from __future__ import annotations

import torch
import soundfile as sf

from src.contracts.audio_contracts import AudioChunk
from src.contracts.vad_contracts import SpeechSegment, VADResult


class SpeechScanTool:
    """Detect speech segments in an AudioChunk using Silero VAD.

    Parameters
    ----------
    threshold : float
        Speech probability threshold for Silero VAD (default 0.5).
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self._threshold = threshold
        self._model, self._utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )

    def run(self, chunk: AudioChunk) -> VADResult:
        """Run Silero VAD on *chunk* and return a ``VADResult``.

        Parameters
        ----------
        chunk : AudioChunk
            The audio chunk to analyse.

        Returns
        -------
        VADResult
            Contains detected ``SpeechSegment`` list, ``speech_ratio``,
            and ``has_speech`` flag.
        """
        data, sr = sf.read(chunk.wav_path, dtype="float32")

        # Silero VAD expects 16 kHz mono float32 tensor
        wav_tensor = torch.from_numpy(data)

        get_speech_timestamps = self._utils[0]
        speech_timestamps = get_speech_timestamps(
            wav_tensor,
            self._model,
            sampling_rate=sr,
            threshold=self._threshold,
            return_seconds=False,
        )

        total_samples = len(data)
        duration = total_samples / sr

        # Get per-frame speech probabilities for real confidence values
        # Silero VAD model returns speech probability per frame
        with torch.no_grad():
            self._model.reset_states()
            # Process in chunks of 512 samples (Silero VAD window size for 16kHz)
            window_size = 512
            probs = []
            for i in range(0, len(wav_tensor), window_size):
                chunk_audio = wav_tensor[i:i + window_size]
                if len(chunk_audio) < window_size:
                    chunk_audio = torch.nn.functional.pad(chunk_audio, (0, window_size - len(chunk_audio)))
                prob = self._model(chunk_audio, sr).item()
                probs.append(prob)

        segments: list[SpeechSegment] = []
        speech_samples = 0
        for ts in speech_timestamps:
            start_sec = ts["start"] / sr
            end_sec = ts["end"] / sr

            # Compute average speech probability for this segment
            start_frame = ts["start"] // window_size
            end_frame = min(ts["end"] // window_size + 1, len(probs))
            seg_probs = probs[start_frame:end_frame]
            avg_confidence = sum(seg_probs) / max(len(seg_probs), 1)

            segments.append(
                SpeechSegment(
                    start=start_sec,
                    end=end_sec,
                    confidence=round(avg_confidence, 4),
                )
            )
            speech_samples += ts["end"] - ts["start"]

        speech_ratio = speech_samples / total_samples if total_samples > 0 else 0.0

        return VADResult(
            chunk_id=chunk.chunk_id,
            segments=segments,
            speech_ratio=speech_ratio,
            has_speech=len(segments) > 0,
        )
