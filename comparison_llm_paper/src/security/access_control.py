"""Least-privilege access control for the Privacy-Preserving Urban Soundscape system.

Each tool and agent is granted only the permissions required for its specific
operation.  The ``AccessControl`` class validates that a component is allowed
to perform a requested action before execution proceeds.

Requirements: 13.1, 13.3
"""

from __future__ import annotations

from enum import Enum
from typing import FrozenSet


# ---------------------------------------------------------------------------
# Permission model
# ---------------------------------------------------------------------------

class Permission(str, Enum):
    """Granular permissions assignable to components."""

    READ_AUDIO = "read_audio"
    WRITE_AUDIO = "write_audio"
    READ_KB = "read_kb"
    RUN_VAD = "run_vad"
    RUN_CLASSIFICATION = "run_classification"
    RUN_BLURRING = "run_blurring"
    COMPUTE_METRICS = "compute_metrics"
    WRITE_DATALAKE = "write_datalake"
    WRITE_REPORT = "write_report"
    AUDIT_LOG = "audit_log"
    MANAGE_ENCRYPTION = "manage_encryption"
    DELETE_AUDIO = "delete_audio"


# ---------------------------------------------------------------------------
# Per-component permission grants (least-privilege, Req 13.1)
# ---------------------------------------------------------------------------

_COMPONENT_PERMISSIONS: dict[str, FrozenSet[Permission]] = {
    "PrepareDataTool": frozenset({
        Permission.READ_AUDIO,
        Permission.WRITE_AUDIO,
    }),
    "SpeechScanTool": frozenset({
        Permission.READ_AUDIO,
        Permission.RUN_VAD,
    }),
    "MidBandAttenuationTool": frozenset({
        Permission.READ_AUDIO,
        Permission.WRITE_AUDIO,
        Permission.RUN_BLURRING,
    }),
    "StrongBlurringTool": frozenset({
        Permission.READ_AUDIO,
        Permission.WRITE_AUDIO,
        Permission.RUN_BLURRING,
    }),
    "ClassificationTool": frozenset({
        Permission.READ_AUDIO,
        Permission.RUN_CLASSIFICATION,
    }),
    "QualityEvaluationTool": frozenset({
        Permission.READ_AUDIO,
        Permission.COMPUTE_METRICS,
    }),
    "DataLakeWriter": frozenset({
        Permission.WRITE_DATALAKE,
        Permission.WRITE_REPORT,
    }),
    "AdaptivePrivacyControlAgent": frozenset({
        Permission.READ_AUDIO,
        Permission.WRITE_AUDIO,
        Permission.READ_KB,
        Permission.RUN_BLURRING,
        Permission.COMPUTE_METRICS,
        Permission.WRITE_REPORT,
    }),
    "KnowledgeBaseLoader": frozenset({
        Permission.READ_KB,
    }),
    "AuditLogger": frozenset({
        Permission.AUDIT_LOG,
    }),
}


# ---------------------------------------------------------------------------
# Access control checker
# ---------------------------------------------------------------------------

class AccessDeniedError(PermissionError):
    """Raised when a component attempts an action it is not permitted."""


class AccessControl:
    """Enforce least-privilege access per tool/agent (Req 13.1, 13.3).

    Usage::

        ac = AccessControl()
        ac.check("SpeechScanTool", Permission.RUN_VAD)       # OK
        ac.check("SpeechScanTool", Permission.WRITE_AUDIO)    # raises AccessDeniedError
    """

    def __init__(
        self,
        overrides: dict[str, FrozenSet[Permission]] | None = None,
    ) -> None:
        self._permissions = dict(_COMPONENT_PERMISSIONS)
        if overrides:
            self._permissions.update(overrides)

    def check(self, component: str, permission: Permission) -> None:
        """Raise ``AccessDeniedError`` if *component* lacks *permission*."""
        granted = self._permissions.get(component, frozenset())
        if permission not in granted:
            raise AccessDeniedError(
                f"{component} does not have '{permission.value}' permission"
            )

    def has_permission(self, component: str, permission: Permission) -> bool:
        """Return ``True`` if *component* holds *permission*."""
        return permission in self._permissions.get(component, frozenset())

    def get_permissions(self, component: str) -> FrozenSet[Permission]:
        """Return the permission set for *component* (empty if unknown)."""
        return self._permissions.get(component, frozenset())
