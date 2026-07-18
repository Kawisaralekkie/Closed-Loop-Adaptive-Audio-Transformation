"""Speaker privacy evaluator using speaker embedding models.

Computes speaker embeddings for original and processed audio using
a HuggingFace-hosted speaker verification model, then measures cosine
distance as a speaker privacy score.

Higher cosine distance means the blurred audio sounds less like the
original speaker → better speaker privacy.

Uses ``microsoft/wavlm-base-sv`` (WavLM for speaker verification) via
HuggingFace Transformers — no SpeechBrain dependency required.

Requirements: 6.2
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio

logger = logging.getLogger(__name__)

# Lazy-loaded globals
_feature_extractor: Any = None
_speaker_model: Any = None
_MODEL_NAME = "microsoft/wavlm-base-sv"


def _get_speaker_model():
    """Load the WavLM speaker verification model once."""
    global _feature_extractor, _speaker_model
    if _speaker_model is None:
        from transformers import AutoFeatureExtractor, WavLMForXVector

        logger.info("Loading speaker model '%s' ...", _MODEL_NAME)
        _feature_extractor = AutoFeatureExtractor.from_pretrained(_MODEL_NAME)
        _speaker_model = WavLMForXVector.from_pretrained(_MODEL_NAME)
        _speaker_model.eval()
        logger.info("Speaker model loaded.")
    return _feature_extractor, _speaker_model


def extract_embedding(audio_path: str) -> np.ndarray:
    """Extract a speaker embedding vector from an audio file.

    Parameters
    ----------
    audio_path : str
        Path to a WAV file.

    Returns
    -------
    np.ndarray
        1-D embedding vector (x-vector from WavLM).
    """
    feature_extractor, model = _get_speaker_model()

    # Load audio using soundfile directly (avoids torchcodec dependency)
    audio_np, sr = sf.read(audio_path, dtype='float32')
    
    # Convert to torch tensor with shape (channels, samples)
    if audio_np.ndim == 1:
        signal = torch.from_numpy(audio_np).unsqueeze(0)
    else:
        # Multi-channel: transpose to (channels, samples)
        signal = torch.from_numpy(audio_np.T)
    
    # Resample to 16 kHz if needed
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        signal = resampler(signal)

    # Convert to mono if needed
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)

    # Flatten to 1-D numpy for the feature extractor
    audio_np = signal.squeeze().numpy()

    inputs = feature_extractor(
        audio_np,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.embeddings  # shape: (1, embed_dim)

    return embedding.squeeze().cpu().numpy()


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute speaker distance as ``1 - cosine_similarity``.

    cosine_similarity ∈ [-1, 1] (clamped for numerical safety only), so the
    returned distance is in the range [0, 2]:
        0 = identical speakers,
        1 = orthogonal embeddings,
        2 = perfectly anti-correlated.

    NOTE: The final [0, 1] clip was removed so the true value range is
    observable in reports. Only the cosine_similarity clamp to [-1, 1]
    remains, since values outside that range are numerical artifacts.
    """
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a)) + 1e-12
    norm_b = float(np.linalg.norm(b)) + 1e-12
    cosine_sim = dot / (norm_a * norm_b)
    cosine_sim = float(np.clip(cosine_sim, -1.0, 1.0))  # numerical safety only
    return float(1.0 - cosine_sim)


def evaluate_speaker_privacy(
    original_path: str,
    processed_path: str,
) -> float:
    """Compute speaker privacy score between original and processed audio.

    Parameters
    ----------
    original_path : str
        Path to the original (unblurred) audio.
    processed_path : str
        Path to the processed (blurred) audio.

    Returns
    -------
    float
        Speaker privacy score in [0, 1].
        0 = same speaker identity preserved (no privacy).
        1 = speaker identity fully obscured (maximum privacy).
    """
    emb_orig = extract_embedding(original_path)
    emb_proc = extract_embedding(processed_path)

    score = cosine_distance(emb_orig, emb_proc)

    logger.info("Speaker privacy: cosine_distance=%.4f", score)
    return score
