"""Knowledge Base models and S3 loader.

Defines Pydantic models for the versioned Knowledge Base (policies, playbooks,
taxonomy) and a loader that reads from S3 with version pinning support.

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 10.1, 10.2, 10.3
"""

from __future__ import annotations

import hashlib
import io
import logging
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Knowledge Base Pydantic models
# ---------------------------------------------------------------------------


class PolicyTransformationRules(BaseModel):
    """Policy rules governing privacy targets and data retention (Req 9.3)."""

    privacy_target: str  # "moderate" or "high" (legacy: "high"/"very_high" — see normalize_privacy_target)
    allow_store_raw_audio: bool
    max_retention_days: int


class RecipeDefinition(BaseModel):
    """A single blurring recipe with parameters and auto-tune rules (Req 10.1)."""

    name: str  # e.g. "RECIPE_MID_BAND_ATTEN"
    params: dict[str, Any]
    use_when: dict[str, Any]  # conditions on analyzer features
    risks: list[str] = Field(default_factory=list)
    mitigations: list[str] = Field(default_factory=list)
    auto_tune_rules: dict[str, Any] = Field(default_factory=dict)


class PrivacyPlaybook(BaseModel):
    """Playbook defining recipes, selection strategy, and evaluation rules (Req 9.4, 10.1-10.3)."""

    analyzer_required_features: list[str]  # Req 10.4
    evaluator_metrics: list[str]
    privacy_score_formula: dict[str, Any]
    preserve_score_formula: dict[str, Any]
    pass_criteria: dict[str, Any]  # keyed by privacy_target
    recipes: list[RecipeDefinition]  # Req 10.1
    selection_strategy: str  # "score_then_try" (Req 10.2)
    max_trials: int  # 2 (Req 10.2)
    fallback_rules: dict[str, Any] = Field(default_factory=dict)
    utility_preserve_target: float = 0.80  # Req 9.4


class SoundLabelTaxonomy(BaseModel):
    """Sound classification taxonomy with class lists and thresholds (Req 9.5)."""

    classes: list[str]
    mappings: dict[str, Any]
    confidence_thresholds: dict[str, float]



class PDFReference(BaseModel):
    """Reference to a PDPA/PII regulatory PDF stored in the KB (Req 9.6)."""

    name: str
    s3_key: str
    description: str = ""


class KnowledgeBase(BaseModel):
    """Top-level Knowledge Base aggregating all KB artifacts (Req 9.2, 9.7)."""

    version: str
    manifest_hash: str
    policies: PolicyTransformationRules
    playbook: PrivacyPlaybook
    taxonomy: SoundLabelTaxonomy
    pdf_references: list[PDFReference] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# S3-backed Knowledge Base loader
# ---------------------------------------------------------------------------

# Expected KB S3 key layout (relative to prefix):
#   kb_manifest.yaml
#   policies/pdpa_policy.yaml
#   policies/retention_policy.yaml
#   playbooks/privacy_playbook.yaml
#   taxonomy/audioset_taxonomy_min.yaml
#   docs/*.pdf

_MANIFEST_KEY = "kb_manifest.yaml"
_POLICY_KEYS = [
    "policies/pdpa_policy.yaml",
    "policies/retention_policy.yaml",
]
_PLAYBOOK_KEY = "playbooks/privacy_playbook.yaml"
_TAXONOMY_KEY = "taxonomy/audioset_taxonomy_min.yaml"


class KnowledgeBaseLoader:
    """Loads the Knowledge Base from an S3 bucket with optional version pinning.

    Parameters
    ----------
    s3_client:
        A boto3 S3 client (or compatible stub for testing).
    """

    def __init__(self, s3_client: Any) -> None:
        self._s3 = s3_client

    # -- public API ---------------------------------------------------------

    def load(
        self,
        s3_bucket: str,
        prefix: str = "kb/",
        version: str | None = None,
    ) -> KnowledgeBase:
        """Load the Knowledge Base from *s3_bucket*.

        If *version* is provided the loader fetches that exact version
        (using S3 object versioning); otherwise the latest version is used.

        Returns a fully validated ``KnowledgeBase`` instance.
        """
        effective_prefix = prefix.rstrip("/") + "/"

        # 1. Load and hash the manifest
        manifest_raw = self._read_yaml(s3_bucket, f"{effective_prefix}{_MANIFEST_KEY}", version)
        manifest_hash = hashlib.sha256(
            yaml.dump(manifest_raw, sort_keys=True).encode()
        ).hexdigest()

        kb_version = manifest_raw.get("version", version or "latest")

        # 2. Load policy files and merge into PolicyTransformationRules
        merged_policies: dict[str, Any] = {}
        for key in _POLICY_KEYS:
            data = self._read_yaml(s3_bucket, f"{effective_prefix}{key}", version)
            merged_policies.update(data)
        policies = PolicyTransformationRules(**merged_policies)

        # 3. Load playbook
        playbook_raw = self._read_yaml(s3_bucket, f"{effective_prefix}{_PLAYBOOK_KEY}", version)
        playbook = PrivacyPlaybook(**playbook_raw)

        # 4. Load taxonomy
        taxonomy_raw = self._read_yaml(s3_bucket, f"{effective_prefix}{_TAXONOMY_KEY}", version)
        taxonomy = SoundLabelTaxonomy(**taxonomy_raw)

        # 5. Discover PDF references (Req 9.6)
        pdf_refs = self._discover_pdfs(s3_bucket, f"{effective_prefix}docs/", version)

        return KnowledgeBase(
            version=str(kb_version),
            manifest_hash=manifest_hash,
            policies=policies,
            playbook=playbook,
            taxonomy=taxonomy,
            pdf_references=pdf_refs,
        )

    # -- internal helpers ---------------------------------------------------

    def _read_yaml(
        self,
        bucket: str,
        key: str,
        version: str | None,
    ) -> dict[str, Any]:
        """Download an S3 object and parse it as YAML."""
        get_kwargs: dict[str, Any] = {"Bucket": bucket, "Key": key}
        if version is not None:
            get_kwargs["VersionId"] = version

        response = self._s3.get_object(**get_kwargs)
        body = response["Body"].read()
        if isinstance(body, str):
            body = body.encode()
        return yaml.safe_load(io.BytesIO(body)) or {}

    def _discover_pdfs(
        self,
        bucket: str,
        docs_prefix: str,
        version: str | None,
    ) -> list[PDFReference]:
        """List PDF files under the docs/ prefix and return references."""
        refs: list[PDFReference] = []
        try:
            response = self._s3.list_objects_v2(Bucket=bucket, Prefix=docs_prefix)
            for obj in response.get("Contents", []):
                key: str = obj["Key"]
                if key.lower().endswith(".pdf"):
                    name = key.rsplit("/", 1)[-1]
                    refs.append(PDFReference(name=name, s3_key=key))
        except Exception:
            logger.warning("Could not list PDF references under %s", docs_prefix)
        return refs
