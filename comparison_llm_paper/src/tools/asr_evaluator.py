"""ASR-based WER/CER evaluator using OpenAI Whisper.

Transcribes original and processed audio chunks, then computes
Word Error Rate (WER) and Character Error Rate (CER) using
the ``jiwer`` library.

Higher WER/CER on the processed (blurred) audio relative to the
original indicates stronger speech privacy protection.

Requirements: 6.1, 6.2
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

# Lazy-loaded globals — heavy imports deferred until first use.
_whisper_model: Any = None
_WHISPER_MODEL_SIZE = "base"  # good balance of speed vs accuracy on CPU


def _get_whisper_model():
    """Load the Whisper model once and cache it."""
    global _whisper_model
    if _whisper_model is None:
        import whisper

        logger.info("Loading Whisper model '%s' ...", _WHISPER_MODEL_SIZE)
        _whisper_model = whisper.load_model(_WHISPER_MODEL_SIZE)
        logger.info("Whisper model loaded.")
    return _whisper_model


def transcribe(audio_path: str, language: str | None = "en") -> str:
    """Transcribe an audio file using Whisper.

    Parameters
    ----------
    audio_path : str
        Path to a WAV file (16 kHz mono preferred).
    language : str | None
        Language code (e.g. "th" for Thai, "en" for English).
        ``None`` enables auto-detection.

    Returns
    -------
    str
        The transcribed text (lowercased, stripped).
    """
    model = _get_whisper_model()
    result = model.transcribe(audio_path, language=language, fp16=False)
    return result.get("text", "").strip().lower()


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate between reference and hypothesis.

    Returns the raw WER where 0.0 means identical transcriptions and
    higher means more divergence (more privacy). The value is NOT clipped
    to [0, 1]: jiwer can return > 1.0 when the hypothesis is longer than
    the reference (many insertions), and we expose that true range.
    """
    import jiwer

    if not reference and not hypothesis:
        return 0.0
    if not reference:
        # No speech in original — blurred text is spurious; treat as low WER
        return 0.0
    if not hypothesis:
        # Blurring destroyed all speech — maximum privacy
        return 1.0

    # NOTE: clip removed — expose true WER (may exceed 1.0 on heavy insertions).
    return float(jiwer.wer(reference, hypothesis))


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate between reference and hypothesis.

    Returns the raw CER where 0.0 means identical transcriptions and
    higher means more divergence (more privacy). The value is NOT clipped
    to [0, 1] (CER can exceed 1.0 on heavy insertions); we expose the
    true range.
    """
    import jiwer

    if not reference and not hypothesis:
        return 0.0
    if not reference:
        return 0.0
    if not hypothesis:
        return 1.0

    # NOTE: clip removed — expose true CER (may exceed 1.0 on heavy insertions).
    return float(jiwer.cer(reference, hypothesis))


def evaluate_asr_privacy(
    original_path: str,
    processed_path: str,
    language: str | None = "en",
) -> tuple[float, float, str, str]:
    """Transcribe both files and compute WER + CER.

    Parameters
    ----------
    original_path : str
        Path to the original (unblurred) audio.
    processed_path : str
        Path to the processed (blurred) audio.
    language : str | None
        Language code passed to Whisper (e.g. "th", "en").
        ``None`` enables auto-detection.

    Returns
    -------
    tuple[float, float, str, str]
        ``(wer, cer, original_text, processed_text)``
    """
    original_text = transcribe(original_path, language=language)
    processed_text = transcribe(processed_path, language=language)

    wer = compute_wer(original_text, processed_text)
    cer = compute_cer(original_text, processed_text)

    logger.info(
        "ASR privacy: WER=%.4f CER=%.4f | orig=%r | proc=%r",
        wer, cer, original_text[:80], processed_text[:80],
    )

    return wer, cer, original_text, processed_text
