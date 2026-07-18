"""Data minimization utilities for the Privacy-Preserving Urban Soundscape system.

Provides raw audio deletion when policy prohibits storage and retention
policy enforcement for expired data.

Requirements: 11.1, 11.4
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from uuid import UUID

from src.security.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


def delete_raw_audio_files(
    raw_audio_path: str,
    canonical_wav_path: str,
    speech_chunk_paths: list[str],
    audit_logger: AuditLogger,
    run_id: str | UUID,
) -> list[str]:
    """Delete raw speech audio files when allow_store_raw_audio is False.

    Deletes:
    - The original raw audio input file
    - The canonical.wav (contains full original audio including speech)
    - Speech chunk WAV files (chunks that contained speech)

    Non-speech chunks are NOT deleted — they don't contain speech.

    Parameters
    ----------
    raw_audio_path : str
        Path to the original raw audio input file.
    canonical_wav_path : str
        Path to the canonical.wav file produced by PrepareDataTool.
    speech_chunk_paths : list[str]
        Paths to chunk WAV files that contained speech.
    audit_logger : AuditLogger
        Audit logger for recording deletion actions.
    run_id : str | UUID
        Run identifier for audit logging.

    Returns
    -------
    list[str]
        Paths that were successfully deleted.
    """
    deleted: list[str] = []

    # Collect all paths to delete
    paths_to_delete = []
    if raw_audio_path:
        paths_to_delete.append(("raw_input", raw_audio_path))
    if canonical_wav_path:
        paths_to_delete.append(("canonical_wav", canonical_wav_path))
    for p in speech_chunk_paths:
        paths_to_delete.append(("speech_chunk", p))

    for file_type, path in paths_to_delete:
        try:
            if os.path.isfile(path):
                os.remove(path)
                deleted.append(path)
                audit_logger.log(
                    run_id=run_id,
                    chunk_id="",
                    tool_name="DataMinimization",
                    decision_parameters={
                        "action": "delete_raw_audio",
                        "file_type": file_type,
                        "path": path,
                    },
                    outcome=f"deleted {file_type}: {path}",
                )
            else:
                logger.debug("File already absent, skipping: %s", path)
        except OSError as exc:
            logger.error("Failed to delete %s (%s): %s", file_type, path, exc)
            audit_logger.log(
                run_id=run_id,
                chunk_id="",
                tool_name="DataMinimization",
                decision_parameters={
                    "action": "delete_raw_audio",
                    "file_type": file_type,
                    "path": path,
                },
                outcome=f"error deleting {file_type}: {exc}",
            )

    audit_logger.log(
        run_id=run_id,
        chunk_id="",
        tool_name="DataMinimization",
        decision_parameters={
            "action": "raw_audio_deletion_summary",
            "total_deleted": len(deleted),
        },
        outcome=f"raw audio deletion complete: {len(deleted)} files removed",
    )

    return deleted


def enforce_retention_policy(
    base_path: str,
    max_retention_days: int,
    audit_logger: AuditLogger,
    run_id: str | UUID,
) -> list[str]:
    """Delete data that exceeds the retention policy.

    Scans *base_path* for files older than *max_retention_days* and
    deletes them.

    Parameters
    ----------
    base_path : str
        Root directory of the data lake to scan.
    max_retention_days : int
        Maximum number of days data may be retained.
    audit_logger : AuditLogger
        Audit logger for recording deletion actions.
    run_id : str | UUID
        Run identifier for audit logging.

    Returns
    -------
    list[str]
        Paths that were deleted due to retention expiry.
    """
    deleted: list[str] = []

    if not os.path.isdir(base_path):
        logger.debug("Retention base path does not exist: %s", base_path)
        return deleted

    cutoff_time = time.time() - (max_retention_days * 86400)

    for root, _dirs, files in os.walk(base_path):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                mtime = os.path.getmtime(fpath)
                if mtime < cutoff_time:
                    os.remove(fpath)
                    deleted.append(fpath)
                    audit_logger.log(
                        run_id=run_id,
                        chunk_id="",
                        tool_name="RetentionPolicy",
                        decision_parameters={
                            "action": "retention_delete",
                            "path": fpath,
                            "max_retention_days": max_retention_days,
                            "file_age_days": (time.time() - mtime) / 86400,
                        },
                        outcome=f"deleted expired file: {fpath}",
                    )
            except OSError as exc:
                logger.error("Retention check failed for %s: %s", fpath, exc)

    audit_logger.log(
        run_id=run_id,
        chunk_id="",
        tool_name="RetentionPolicy",
        decision_parameters={
            "action": "retention_enforcement_summary",
            "base_path": base_path,
            "max_retention_days": max_retention_days,
            "total_deleted": len(deleted),
        },
        outcome=f"retention enforcement complete: {len(deleted)} expired files removed",
    )

    return deleted
