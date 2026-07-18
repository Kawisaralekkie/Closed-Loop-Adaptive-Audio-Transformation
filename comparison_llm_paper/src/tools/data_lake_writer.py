"""DataLakeWriter — persist processed artifacts and RunReport to the data lake.

Writes all artifact files and the serialized RunReport to a configurable
storage location, returning the full list of persisted paths.

Requirements: 7.1, 7.2, 7.3
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from src.contracts.report_contracts import RunReport


class DataLakeWriter:
    """Persist artifacts and run reports to the data lake.

    Parameters
    ----------
    base_path : str
        Root directory of the data lake storage.  Artifacts are written
        under ``{base_path}/{run_id}/``.
    """

    def __init__(self, base_path: str) -> None:
        self._base_path = base_path

    def run(
        self,
        run_id: str,
        artifact_paths: list[str],
        report: RunReport,
    ) -> list[str]:
        """Persist all artifacts and the RunReport to the data lake.

        Parameters
        ----------
        run_id : str
            Unique identifier for the processing run.
        artifact_paths : list[str]
            Paths to artifact files (blurred WAVs, embeddings, etc.)
            that should be persisted.
        report : RunReport
            The comprehensive run report to serialize and store.

        Returns
        -------
        list[str]
            Every persisted path, including each input artifact path
            and the RunReport JSON path.
        """
        run_dir = os.path.join(self._base_path, run_id)
        os.makedirs(run_dir, exist_ok=True)

        persisted: list[str] = []

        # Persist each artifact file
        for src_path in artifact_paths:
            dest_path = os.path.join(run_dir, os.path.basename(src_path))
            if os.path.abspath(src_path) != os.path.abspath(dest_path):
                shutil.copy2(src_path, dest_path)
            persisted.append(dest_path)

        # Persist the RunReport as JSON
        report_path = os.path.join(run_dir, "run_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(mode="json"), f, indent=2, default=str)
        persisted.append(report_path)

        return persisted
