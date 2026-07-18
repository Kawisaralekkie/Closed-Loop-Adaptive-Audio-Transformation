"""Core identifier generation for the Privacy-Preserving Urban Soundscape system.

Provides deterministic, reproducible identifiers used consistently across all
data contracts and pipeline stages.

- run_id:      UUID v5 (deterministic from source_id + timestamp) — unique per run
- source_id:   str — identifies the IoT/sensor source
- chunk_id:    str — "{run_id}_{chunk_index}" — unique per chunk within a run
- artifact_id: str — "{chunk_id}_{artifact_type}" — unique per artifact

Requirements: 13.2, 16.2
"""

from __future__ import annotations

import uuid

# Consistent namespace UUID for v5 generation across the entire system.
_NAMESPACE = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


class CoreIds:
    """Factory for deterministic pipeline identifiers."""

    @staticmethod
    def generate_run_id(source_id: str, timestamp: str) -> uuid.UUID:
        """Return a deterministic UUID v5 run_id from *source_id* and *timestamp*.

        The same (source_id, timestamp) pair always produces the identical UUID,
        enabling idempotent reruns (Req 13.2).
        """
        name = f"{source_id}:{timestamp}"
        return uuid.uuid5(_NAMESPACE, name)

    @staticmethod
    def generate_chunk_id(run_id: uuid.UUID, index: int) -> str:
        """Return ``"{run_id}_{index}"`` (Req 16.2)."""
        return f"{run_id}_{index}"

    @staticmethod
    def generate_artifact_id(chunk_id: str, artifact_type: str) -> str:
        """Return ``"{chunk_id}_{artifact_type}"`` (Req 16.2)."""
        return f"{chunk_id}_{artifact_type}"
