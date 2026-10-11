from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from doc_pipeline.registry import Registry


def test_registry_caches_batches_and_flags_content_duplicates(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.db", dhash_max_distance=6, visual_threshold=0.85)
    registry.save_batch(
        source_sha256="a" * 64,
        config_fingerprint="config-v1",
        source_name="first.pdf",
        batch={"source_name": "first.pdf", "documents": []},
        output_path=tmp_path / "first.xlsx",
        created_at="2026-01-01T00:00:00+00:00",
        run_id="job_first",
        documents=[
            {
                "content_fingerprint": "fingerprint-1",
                "page_hashes": ["0000000000000000"],
                "fields": {"vendor": "Acme"},
            }
        ],
    )

    cached = registry.get_cached_batch("a" * 64, "config-v1")
    assert cached is not None
    assert cached["source_name"] == "first.pdf"
    assert cached["run_id"] == "job_first"
    assert registry.has_exact_hash("a" * 64) is not None
    assert registry.find_possible_duplicate("b" * 64, "fingerprint-1", []) == "a" * 64
    assert registry.find_possible_duplicate("b" * 64, None, ["0000000000000000"]) == "a" * 64


def test_registry_migrates_pre_run_id_input_table(tmp_path: Path) -> None:
    path = tmp_path / "old-registry.db"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE processed_inputs (source_sha256 TEXT NOT NULL, config_fingerprint TEXT NOT NULL, "
        "batch_json TEXT NOT NULL, output_path TEXT NOT NULL, source_name TEXT NOT NULL, created_at TEXT NOT NULL, "
        "PRIMARY KEY (source_sha256, config_fingerprint))"
    )
    connection.execute(
        "CREATE TABLE documents (id INTEGER PRIMARY KEY AUTOINCREMENT, source_sha256 TEXT NOT NULL, "
        "content_fingerprint TEXT, page_hashes_json TEXT NOT NULL, fields_json TEXT NOT NULL, created_at TEXT NOT NULL)"
    )
    connection.execute(
        "INSERT INTO processed_inputs VALUES (?, ?, ?, ?, ?, ?)",
        ("a" * 64, "cfg", json.dumps({"source_name": "a.pdf", "documents": []}), "a.xlsx", "a.pdf", "now"),
    )
    connection.commit()
    connection.close()

    registry = Registry(path, dhash_max_distance=6, visual_threshold=0.85)

    assert registry.get_cached_batch("a" * 64, "cfg")["run_id"] == ""
