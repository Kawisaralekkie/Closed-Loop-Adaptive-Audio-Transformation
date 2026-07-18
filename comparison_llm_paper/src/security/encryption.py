"""Encryption helpers for the Privacy-Preserving Urban Soundscape system.

Provides at-rest encryption configuration (S3 server-side encryption) and
in-transit encryption helpers (TLS configuration).

Requirements: 11.2, 11.3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# At-rest encryption (S3 server-side encryption)  — Req 11.2
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class S3EncryptionConfig:
    """Configuration for S3 server-side encryption.

    Generates the ``ExtraArgs`` dict expected by boto3 ``upload_file`` /
    ``put_object`` calls.
    """

    algorithm: str = "aws:kms"  # SSE-KMS by default; also accepts "AES256"
    kms_key_id: str | None = None  # None → default AWS-managed key

    def extra_args(self) -> dict[str, str]:
        """Return boto3-compatible ``ExtraArgs`` for server-side encryption."""
        args: dict[str, str] = {"ServerSideEncryption": self.algorithm}
        if self.kms_key_id and self.algorithm == "aws:kms":
            args["SSEKMSKeyId"] = self.kms_key_id
        return args


def get_s3_encryption_args(
    algorithm: str = "aws:kms",
    kms_key_id: str | None = None,
) -> dict[str, str]:
    """Convenience wrapper returning encryption ``ExtraArgs`` for S3 uploads."""
    return S3EncryptionConfig(algorithm=algorithm, kms_key_id=kms_key_id).extra_args()


# ---------------------------------------------------------------------------
# In-transit encryption (TLS)  — Req 11.3
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TLSConfig:
    """TLS configuration for in-transit encryption.

    Produces a ``botocore.config.Config``-compatible dict that enforces
    HTTPS-only connections.
    """

    verify_ssl: bool = True
    min_tls_version: str = "TLSv1.2"

    def boto_config_kwargs(self) -> dict[str, Any]:
        """Return kwargs suitable for ``botocore.config.Config``."""
        return {
            "verify": self.verify_ssl,
        }

    def s3_endpoint_url(self, region: str) -> str:
        """Return the HTTPS endpoint URL for S3 in *region*."""
        return f"https://s3.{region}.amazonaws.com"


def get_tls_config() -> TLSConfig:
    """Return the default TLS configuration."""
    return TLSConfig()
