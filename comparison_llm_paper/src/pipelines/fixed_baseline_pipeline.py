"""Fixed Baseline Pipeline — deterministic voice blurring orchestration.

Orchestrates: PrepareDataTool → SpeechScanTool → routing →
MidBandAttenuationTool (privacy_target="moderate") or StrongBlurringTool
(privacy_target="high") → ClassificationTool →
QualityEvaluationTool → DataLakeWriter.

Each chunk is processed independently; partial failures do not abort the run.

NOTE (renamed 2026): privacy_target vocabulary
    OLD VOCAB        NEW VOCAB     pipeline behaviour
    "high"      →    "moderate"    MidBandAttenuationTool
    "very_high" →    "high"        StrongBlurringTool
Legacy values are accepted via ``normalize_privacy_target()``.

Requirements: 1.1–1.6, 2.1–2.4, 3.1–3.4, 4.1–4.5, 5.1–5.3, 6.1–6.6,
              7.1–7.3, 11.1, 11.4, 12.1, 12.2, 14.2, 14.3
"""

from __future__ import annotations

import logging
import os
import traceback
from datetime import datetime, timezone
from uuid import UUID

from src.config import AppConfig, config
from src.contracts.audio_contracts import AudioChunk, AudioIngestRequest
from src.contracts.metrics_contracts import normalize_privacy_target
from src.contracts.report_contracts import ChunkReport, RunReport
from src.security.audit_logger import AuditLogger
from src.security.data_minimization import delete_raw_audio_files, enforce_retention_policy
from src.tools.classification_tool import ClassificationTool
from src.tools.data_lake_writer import DataLakeWriter
from src.tools.mid_band_attenuation_tool import MidBandAttenuationTool
from src.tools.prepare_data_tool import PrepareDataTool
from src.tools.quality_evaluation_tool import QualityEvaluationTool
from src.tools.speech_scan_tool import SpeechScanTool
from src.tools.strong_blurring_tool import StrongBlurringTool

logger = logging.getLogger(__name__)


class FixedBaselinePipeline:
    """Deterministic fixed-baseline processing pipeline.

    Parameters
    ----------
    privacy_target : str
        ``"moderate"`` selects MidBandAttenuationTool (light blur);
        ``"high"`` selects StrongBlurringTool (heavy blur).

        Legacy values are accepted:
            "high"      → "moderate"   (light blur)
            "very_high" → "high"       (heavy blur)
    prepare_data_tool : PrepareDataTool | None
        Optional pre-constructed tool instance.
    speech_scan_tool : SpeechScanTool | None
        Optional pre-constructed tool instance.
    mid_band_tool : MidBandAttenuationTool | None
        Optional pre-constructed tool instance.
    strong_blur_tool : StrongBlurringTool | None
        Optional pre-constructed tool instance.
    classification_tool : ClassificationTool | None
        Optional pre-constructed tool instance.
    quality_tool : QualityEvaluationTool | None
        Optional pre-constructed tool instance.
    data_lake_writer : DataLakeWriter | None
        Optional pre-constructed tool instance.
    audit_logger : AuditLogger | None
        Optional audit logger instance.
    app_config : AppConfig | None
        Override global config.
    allow_store_raw_audio : bool
        When ``False``, raw speech audio files are deleted after processing
        (Req 11.1).  Defaults to ``True``.
    max_retention_days : int
        Maximum retention period in days.  Expired data in the data lake
        is deleted after each run (Req 11.4).  Defaults to ``90``.
    """

    def __init__(
        self,
        privacy_target: str = "moderate",
        *,
        prepare_data_tool: PrepareDataTool | None = None,
        speech_scan_tool: SpeechScanTool | None = None,
        mid_band_tool: MidBandAttenuationTool | None = None,
        strong_blur_tool: StrongBlurringTool | None = None,
        classification_tool: ClassificationTool | None = None,
        quality_tool: QualityEvaluationTool | None = None,
        data_lake_writer: DataLakeWriter | None = None,
        audit_logger: AuditLogger | None = None,
        app_config: AppConfig | None = None,
        allow_store_raw_audio: bool = True,
        max_retention_days: int = 90,
        ground_truth_label: str | None = None,
    ) -> None:
        # Normalize legacy values: old "high" → new "moderate",
        # old "very_high" → new "high".
        self._privacy_target = normalize_privacy_target(privacy_target)
        self._cfg = app_config or config
        self._prepare = prepare_data_tool
        self._speech_scan = speech_scan_tool
        self._mid_band = mid_band_tool
        self._strong_blur = strong_blur_tool
        self._classification = classification_tool
        self._quality = quality_tool
        self._writer = data_lake_writer
        self._audit = audit_logger or AuditLogger()
        self._allow_store_raw_audio = allow_store_raw_audio
        self._max_retention_days = max_retention_days
        # File-level ground-truth class (label1_audioset) for real
        # mAP/F1/accuracy during transfer-learning evaluation. None → proxy.
        self._ground_truth_label = ground_truth_label
        # Dump full amplitude arrays as .npz artifacts when DUMP_AMPLITUDE_ARRAYS=true
        self._dump_amplitude_arrays = (
            os.environ.get("DUMP_AMPLITUDE_ARRAYS", "false").lower() == "true"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        request: AudioIngestRequest,
        timestamp: str | None = None,
    ) -> RunReport:
        """Execute the full fixed-baseline pipeline.

        Parameters
        ----------
        request : AudioIngestRequest
            Incoming audio ingestion request.
        timestamp : str | None
            ISO-8601 timestamp for deterministic run_id generation.

        Returns
        -------
        RunReport
            Comprehensive report with per-chunk details and
            succeeded/failed counts.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        started_at = datetime.now(timezone.utc)

        # --- Step 1: Ingest, validate, canonicalize, chunk ---
        ingest_response = self._prepare.run(request, timestamp=timestamp)
        run_id = ingest_response.run_id

        self._audit.log(
            run_id=run_id,
            chunk_id="",
            tool_name="PrepareDataTool",
            decision_parameters={"source_id": request.source_id},
            outcome=f"produced {len(ingest_response.chunks)} chunks",
        )

        # --- Step 2–6: Process each chunk independently (Req 14.2, 14.3) ---
        chunk_reports: list[ChunkReport] = []
        artifact_paths: list[str] = []
        speech_chunk_paths: list[str] = []  # Track raw speech chunks for deletion

        for chunk in ingest_response.chunks:
            report, paths = self._process_chunk(chunk, run_id)
            chunk_reports.append(report)
            artifact_paths.extend(paths)
            # Track original chunk WAV paths that contained speech
            if report.had_speech:
                speech_chunk_paths.append(chunk.wav_path)

        succeeded = sum(1 for r in chunk_reports if r.failure is None)
        failed = sum(1 for r in chunk_reports if r.failure is not None)

        finished_at = datetime.now(timezone.utc)
        run_report = RunReport(
            run_id=run_id,
            source_id=request.source_id,
            kb_version="N/A",  # Fixed baseline doesn't use KB
            model_versions={"silero_vad": "5.1", "yamnet": "1"},
            config_params={
                "privacy_target": self._privacy_target,
                "sample_rate": self._cfg.audio.sample_rate,
                "window_size": self._cfg.audio.window_size,
                "overlap": self._cfg.audio.overlap,
            },
            chunks=chunk_reports,
            total_chunks=len(chunk_reports),
            succeeded_chunks=succeeded,
            failed_chunks=failed,
            started_at=started_at.isoformat(),
            created_at=finished_at.isoformat(),
            total_runtime_seconds=round((finished_at - started_at).total_seconds(), 2),
        )

        # --- Step 7: Persist to data lake ---
        if self._writer is not None:
            try:
                persisted = self._writer.run(
                    run_id=str(run_id),
                    artifact_paths=artifact_paths,
                    report=run_report,
                )
                run_report.persisted_paths = persisted
                self._audit.log(
                    run_id=run_id,
                    chunk_id="",
                    tool_name="DataLakeWriter",
                    decision_parameters={"artifact_count": len(artifact_paths)},
                    outcome=f"persisted {len(persisted)} paths",
                )
            except Exception as exc:
                logger.error("DataLakeWriter failed: %s", exc)
                self._audit.log(
                    run_id=run_id,
                    chunk_id="",
                    tool_name="DataLakeWriter",
                    decision_parameters={},
                    outcome=f"error: {exc}",
                )

        # --- Step 8: Data minimization (Req 11.1) ---
        if not self._allow_store_raw_audio:
            delete_raw_audio_files(
                raw_audio_path=request.raw_audio_path,
                canonical_wav_path=ingest_response.canonical_audio.wav_path,
                speech_chunk_paths=speech_chunk_paths,
                audit_logger=self._audit,
                run_id=run_id,
            )

        # --- Step 9: Retention policy enforcement (Req 11.4) ---
        if self._writer is not None:
            enforce_retention_policy(
                base_path=self._writer._base_path,
                max_retention_days=self._max_retention_days,
                audit_logger=self._audit,
                run_id=run_id,
            )

        return run_report

    # ------------------------------------------------------------------
    # Per-chunk processing
    # ------------------------------------------------------------------

    def _process_chunk(
        self,
        chunk: AudioChunk,
        run_id: UUID,
    ) -> tuple[ChunkReport, list[str]]:
        """Process a single chunk through the pipeline.

        Returns a ``(ChunkReport, artifact_paths)`` tuple.  Failures are
        caught and recorded in the report rather than propagated (Req 14.2).
        """
        artifact_paths: list[str] = []

        try:
            # --- Speech detection (Req 2.1, 2.2) ---
            vad_result = self._speech_scan.run(chunk)
            self._audit.log(
                run_id=run_id,
                chunk_id=chunk.chunk_id,
                tool_name="SpeechScanTool",
                decision_parameters={
                    "speech_ratio": vad_result.speech_ratio,
                    "has_speech": vad_result.has_speech,
                },
                outcome="speech_detected" if vad_result.has_speech else "no_speech",
            )

            # --- Routing (Req 2.3, 2.4) ---
            transform_result = None
            if vad_result.has_speech:
                transform_result = self._apply_blurring(chunk, vad_result, run_id)
                artifact_paths.append(transform_result.blurred_wav_path)

                # Build a processed AudioChunk pointing to the blurred WAV
                processed_chunk = AudioChunk(
                    chunk_id=chunk.chunk_id,
                    run_id=run_id,
                    wav_path=transform_result.blurred_wav_path,
                    start_time=chunk.start_time,
                    end_time=chunk.end_time,
                    duration=chunk.duration,
                )
            else:
                processed_chunk = chunk

            # --- Classification (Req 5.1–5.3) ---
            classification_original = self._classification.run(chunk)
            classification_result = self._classification.run(processed_chunk)
            self._audit.log(
                run_id=run_id,
                chunk_id=chunk.chunk_id,
                tool_name="ClassificationTool",
                decision_parameters={
                    "top_label": (
                        classification_result.predictions[0].label
                        if classification_result.predictions
                        else "none"
                    ),
                },
                outcome="classified",
            )

            # --- Quality evaluation (Req 6.1–6.6) ---
            metrics_result = self._quality.run(
                original_chunk=chunk,
                processed_chunk=processed_chunk,
                classification_result=classification_result,
                privacy_target=self._privacy_target,
                ground_truth_label=self._ground_truth_label,
                had_speech=vad_result.has_speech,
            )
            self._audit.log(
                run_id=run_id,
                chunk_id=chunk.chunk_id,
                tool_name="QualityEvaluationTool",
                decision_parameters={
                    "privacy_score": metrics_result.privacy.privacy_score,
                    "preserve_score": metrics_result.utility.preserve_score,
                    "overall_pass": metrics_result.decision.overall_pass,
                },
                outcome="pass" if metrics_result.decision.overall_pass else "fail",
            )

            # Compute VAD confidence mean for logging
            vad_conf = (
                sum(s.confidence for s in vad_result.segments) / len(vad_result.segments)
                if vad_result.segments else 0.0
            )

            # Extract top-3 YAMNet predictions
            top3 = [
                {"label": p.label, "confidence": round(p.confidence, 4)}
                for p in (classification_result.predictions or [])[:3]
            ]
            top3_orig = [
                {"label": p.label, "confidence": round(p.confidence, 4)}
                for p in (classification_original.predictions or [])[:3]
            ]

            # --- Amplitude-array stats (before/after transform) ---
            # Computed for EVERY chunk, including no-speech chunks. When a
            # chunk has no speech, processed_chunk == original (passthrough),
            # so this captures the no-speech environmental baseline — the
            # best-case reference (original == processed, difference ≈ 0).
            amplitude_stats = None
            try:
                from src.tools.amplitude_logger import compute_amplitude_stats
                amplitude_stats, amp_npz = compute_amplitude_stats(
                    original_path=chunk.wav_path,
                    processed_path=processed_chunk.wav_path,
                    dump_full_array=self._dump_amplitude_arrays,
                    chunk_id=chunk.chunk_id,
                )
                if amp_npz:
                    artifact_paths.append(amp_npz)
            except Exception as amp_exc:
                logger.warning("Amplitude stats failed for %s: %s", chunk.chunk_id, amp_exc)

            report = ChunkReport(
                chunk_id=chunk.chunk_id,
                run_id=run_id,
                had_speech=vad_result.has_speech,
                speech_ratio=vad_result.speech_ratio,
                vad_confidence=vad_conf,
                recipe_applied=(
                    transform_result.recipe_ref if transform_result else None
                ),
                params_applied=(
                    transform_result.params if transform_result else None
                ),
                trials=1 if transform_result else 0,
                metrics=metrics_result,
                routing_decision="blurred" if vad_result.has_speech else "bypass",
                classification_top3=top3,
                classification_top3_original=top3_orig,
                amplitude_stats=amplitude_stats,
                ground_truth_label=self._ground_truth_label,
            )

        except Exception as exc:
            logger.error(
                "Chunk %s failed: %s", chunk.chunk_id, exc, exc_info=True,
            )
            self._audit.log(
                run_id=run_id,
                chunk_id=chunk.chunk_id,
                tool_name="FixedBaselinePipeline",
                decision_parameters={},
                outcome=f"error: {exc}",
            )
            report = ChunkReport(
                chunk_id=chunk.chunk_id,
                run_id=run_id,
                had_speech=False,
                trials=0,
                routing_decision="error",
                failure=traceback.format_exc(),
            )

        return report, artifact_paths

    # ------------------------------------------------------------------
    # Blurring dispatch
    # ------------------------------------------------------------------

    def _apply_blurring(
        self,
        chunk: AudioChunk,
        vad_result,
        run_id: UUID,
    ):
        """Apply the appropriate blurring tool based on privacy_target.

        ``privacy_target="moderate"`` → MidBandAttenuationTool (light)
        ``privacy_target="high"``     → StrongBlurringTool (heavy)

        NOTE (renamed 2026): legacy callers passing "high"/"very_high"
        are normalized in __init__ to "moderate"/"high" respectively.
        """
        if self._privacy_target == "high":
            result = self._strong_blur.run(
                wav_path=chunk.wav_path,
                segments=vad_result.segments,
                lowpass_cutoff=500,
                lowpass_mix=0.55,
                highband_start=2500,
                highband_mix=0.15,
                midband_range=(1200, 2200),
                midband_gain_db=2.0,
                noise_band=(700, 2800),
                noise_snr_db=0.0,
                band_hz=(700, 2700),
                atten_db=40.0,
                pitch_shift_semitones=-5.0,
                chunk_id=chunk.chunk_id,
            )
            tool_name = "StrongBlurringTool"
        else:
            result = self._mid_band.run(
                wav_path=chunk.wav_path,
                segments=vad_result.segments,
                # Preset M2 — balanced
                band_hz=(700, 2700),
                atten_db=30.0,
                lowpass_cutoff=950,
                pitch_shift_semitones=-4.0,
                chunk_id=chunk.chunk_id,
            )
            tool_name = "MidBandAttenuationTool"

        self._audit.log(
            run_id=run_id,
            chunk_id=chunk.chunk_id,
            tool_name=tool_name,
            decision_parameters={
                "privacy_target": self._privacy_target,
                "recipe": result.recipe_ref.recipe_name,
            },
            outcome="blurred",
        )
        return result
