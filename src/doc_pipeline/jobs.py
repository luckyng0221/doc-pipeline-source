from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClaimedPDF:
    path: Path
    source_name: str
    run_id: str


def claim_pdf(source: Path, processing: Path, *, move_source: bool) -> ClaimedPDF:
    """Stage one PDF in a uniquely identified job folder with its original name."""
    run_id = f"job_{uuid.uuid4().hex}"
    job_dir = processing / run_id
    job_dir.mkdir(parents=False, exist_ok=False)
    metadata_path = job_dir / "job.json"
    temporary_metadata = job_dir / ".job.json.tmp"
    claimed_path = job_dir / "source.pdf"
    try:
        temporary_metadata.write_text(
            json.dumps({"source_name": source.name}, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary_metadata, metadata_path)
        if move_source:
            os.replace(source, claimed_path)
        else:
            shutil.copy2(source, claimed_path)
    except Exception:
        temporary_metadata.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        claimed_path.unlink(missing_ok=True)
        try:
            job_dir.rmdir()
        except OSError:
            pass
        raise
    return ClaimedPDF(path=claimed_path, source_name=source.name, run_id=run_id)
