"""Audit logging for the Privacy-Preserving Urban Soundscape system.

Records processing decisions with metadata and hashes only — no raw audio
data or speech content is ever stored in log entries.

Requirements: 12.1, 12.2
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

logger = logging.getLogger("audit")


# ---------------------------------------------------------------------------
# Audit log entry model
# ---------------------------------------------------------------------------

class AuditLogEntry(BaseModel):
    """Structured audit log entry (Req 12.2).

    Fields: run_id, chunk_id, timestamp, tool_name, decision_parameters,
    outcome.  No raw audio data or speech content.
    """

    run_id: str
    chunk_id: str
    timestamp: str  # ISO 8601
    tool_name: str
    decision_parameters: dict[str, Any] = Field(default_factory=dict)
    outcome: str
    entry_hash: str = ""  # SHA-256 of the entry content (excluding this field)


# ---------------------------------------------------------------------------
# Sensitive-content filter
# ---------------------------------------------------------------------------

_SENSITIVE_KEYS = frozenset({
    "raw_audio",
    "audio_data",
    "speech_content",
    "transcript",
    "wav_bytes",
    "pcm_data",
    "raw_samples",
    "speech_text",
})


def _sanitize(params: dict[str, Any]) -> dict[str, Any]:
    """Strip keys that could contain raw audio or speech content (Req 12.1)."""
    return {
        k: v for k, v in params.items()
        if k.lower() not in _SENSITIVE_KEYS
    }


# ---------------------------------------------------------------------------
# Hash helper
# ---------------------------------------------------------------------------

def _compute_entry_hash(entry: AuditLogEntry) -> str:
    """SHA-256 digest of the entry content (excluding entry_hash itself)."""
    payload = entry.model_dump(exclude={"entry_hash"})
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# AuditLogger
# ---------------------------------------------------------------------------

class AuditLogger:
    """Append-only audit logger that emits structured, hash-verified entries.

    Usage::

        audit = AuditLogger()
        audit.log(
            run_id=run_id,
            chunk_id=chunk_id,
            tool_name="MidBandAttenuationTool",
            decision_parameters={"atten_db": 25},
            outcome="success",
        )
        entries = audit.get_entries()
    """

    def __init__(self) -> None:
        self._entries: list[AuditLogEntry] = []

    # -- public API ---------------------------------------------------------

    def log(
        self,
        *,
        run_id: str | UUID,
        chunk_id: str,
        tool_name: str,
        decision_parameters: dict[str, Any] | None = None,
        outcome: str,
    ) -> AuditLogEntry:
        """Create, store, and return a sanitised audit log entry."""
        safe_params = _sanitize(decision_parameters or {})

        entry = AuditLogEntry(
            run_id=str(run_id),
            chunk_id=chunk_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            tool_name=tool_name,
            decision_parameters=safe_params,
            outcome=outcome,
        )
        entry.entry_hash = _compute_entry_hash(entry)

        self._entries.append(entry)
        logger.info("audit | %s | %s | %s | %s", entry.tool_name, entry.chunk_id, entry.outcome, entry.entry_hash)
        return entry

    def get_entries(self) -> list[AuditLogEntry]:
        """Return a copy of all recorded entries."""
        return list(self._entries)

    def get_entries_for_run(self, run_id: str | UUID) -> list[AuditLogEntry]:
        """Return entries matching *run_id*."""
        rid = str(run_id)
        return [e for e in self._entries if e.run_id == rid]

    def clear(self) -> None:
        """Remove all stored entries (useful in tests)."""
        self._entries.clear()
