from __future__ import annotations

import json
import re
from pathlib import Path

from doc_pipeline.jobs import claim_pdf


def test_claim_stages_a_copy_in_its_own_job_folder(tmp_path: Path) -> None:
    source = tmp_path / "sample invoice.pdf"
    source.write_bytes(b"pdf-data")
    processing = tmp_path / "processing"
    processing.mkdir()

    claimed = claim_pdf(source, processing, move_source=False)

    assert source.exists()
    assert re.fullmatch(r"job_[0-9a-f]{32}", claimed.run_id)
    assert claimed.path == processing / claimed.run_id / "source.pdf"
    assert claimed.path.read_bytes() == b"pdf-data"
    assert claimed.source_name == "sample invoice.pdf"
    metadata = json.loads((claimed.path.parent / "job.json").read_text(encoding="utf-8"))
    assert metadata == {"source_name": "sample invoice.pdf"}


def test_claim_can_move_the_upload(tmp_path: Path) -> None:
    source = tmp_path / "upload.pdf"
    source.write_bytes(b"pdf-data")
    processing = tmp_path / "processing"
    processing.mkdir()

    claimed = claim_pdf(source, processing, move_source=True)

    assert not source.exists()
    assert claimed.path.read_bytes() == b"pdf-data"
