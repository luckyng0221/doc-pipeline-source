from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class Registry:
    def __init__(self, path: Path, *, dhash_max_distance: int, visual_threshold: float) -> None:
        self.path = path
        self.dhash_max_distance = dhash_max_distance
        self.visual_threshold = visual_threshold
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS processed_inputs (
                    source_sha256 TEXT NOT NULL,
                    config_fingerprint TEXT NOT NULL,
                    batch_json TEXT NOT NULL,
                    output_path TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    run_id TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (source_sha256, config_fingerprint)
                )
                """
            )
            input_columns = {row["name"] for row in db.execute("PRAGMA table_info(processed_inputs)")}
            if "run_id" not in input_columns:
                db.execute("ALTER TABLE processed_inputs ADD COLUMN run_id TEXT NOT NULL DEFAULT ''")
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_sha256 TEXT NOT NULL,
                    content_fingerprint TEXT,
                    page_hashes_json TEXT NOT NULL,
                    fields_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            db.execute("CREATE INDEX IF NOT EXISTS idx_documents_content ON documents(content_fingerprint)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_sha256)")
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS page_hash_index (
                    document_id INTEGER NOT NULL,
                    band INTEGER NOT NULL,
                    band_value INTEGER NOT NULL,
                    PRIMARY KEY (document_id, band, band_value)
                )
                """
            )
            db.execute("CREATE INDEX IF NOT EXISTS idx_page_hash_band_value ON page_hash_index(band, band_value)")
            unindexed = db.execute(
                "SELECT id, page_hashes_json FROM documents d "
                "WHERE NOT EXISTS (SELECT 1 FROM page_hash_index h WHERE h.document_id=d.id)"
            ).fetchall()
            for row in unindexed:
                try:
                    hashes = json.loads(row["page_hashes_json"])
                except (TypeError, json.JSONDecodeError):
                    continue
                self._insert_page_hashes(db, int(row["id"]), hashes)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def has_exact_hash(self, source_sha256: str) -> dict[str, str] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT output_path, source_name FROM processed_inputs WHERE source_sha256=? "
                "ORDER BY created_at DESC LIMIT 1",
                (source_sha256,),
            ).fetchone()
        return dict(row) if row else None

    def get_cached_batch(self, source_sha256: str, config_fingerprint: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT batch_json, output_path, source_name, run_id FROM processed_inputs "
                "WHERE source_sha256=? AND config_fingerprint=?",
                (source_sha256, config_fingerprint),
            ).fetchone()
        if not row:
            return None
        return {
            "batch": json.loads(row["batch_json"]),
            "output_path": row["output_path"],
            "source_name": row["source_name"],
            "run_id": row["run_id"],
        }

    def find_possible_duplicate(
        self,
        source_sha256: str,
        fingerprint: str | None,
        page_hashes: list[str],
    ) -> str | None:
        with self._connect() as db:
            if fingerprint:
                row = db.execute(
                    "SELECT source_sha256 FROM documents WHERE content_fingerprint=? AND source_sha256<>? "
                    "ORDER BY id DESC LIMIT 1",
                    (fingerprint, source_sha256),
                ).fetchone()
                if row:
                    return str(row["source_sha256"])
            if not page_hashes:
                return None
            rows = self._visual_candidates(db, source_sha256, page_hashes)
        for row in rows:
            try:
                previous_hashes = json.loads(row["page_hashes_json"])
            except (TypeError, json.JSONDecodeError):
                continue
            if self._visual_similarity(page_hashes, previous_hashes) >= self.visual_threshold:
                return str(row["source_sha256"])
        return None

    def _visual_similarity(self, current: list[str], previous: list[str]) -> float:
        if not current or not previous:
            return 0.0
        matches = 0
        for current_hash in current:
            try:
                current_value = int(current_hash, 16)
                distances = [(current_value ^ int(old_hash, 16)).bit_count() for old_hash in previous]
            except (TypeError, ValueError):
                return 0.0
            if distances and min(distances) <= self.dhash_max_distance:
                matches += 1
        return matches / max(len(current), len(previous))

    def _visual_candidates(
        self,
        db: sqlite3.Connection,
        source_sha256: str,
        page_hashes: list[str],
    ) -> list[sqlite3.Row]:
        if self.dhash_max_distance > 7:
            return db.execute(
                "SELECT source_sha256, page_hashes_json FROM documents "
                "WHERE source_sha256<>? ORDER BY id DESC",
                (source_sha256,),
            ).fetchall()
        query_bands = {
            (band, value)
            for page_hash in page_hashes
            for band, value in self._hash_bands(page_hash)
        }
        if not query_bands:
            return []
        db.execute(
            "CREATE TEMP TABLE IF NOT EXISTS current_hash_bands "
            "(band INTEGER NOT NULL, band_value INTEGER NOT NULL, PRIMARY KEY (band, band_value))"
        )
        db.execute("DELETE FROM current_hash_bands")
        db.executemany(
            "INSERT OR IGNORE INTO current_hash_bands (band, band_value) VALUES (?, ?)",
            sorted(query_bands),
        )
        minimum_shared_bands = 8 - self.dhash_max_distance
        return db.execute(
            """
            SELECT d.source_sha256, d.page_hashes_json
            FROM documents d
            JOIN page_hash_index h ON h.document_id=d.id
            JOIN current_hash_bands q ON q.band=h.band AND q.band_value=h.band_value
            WHERE d.source_sha256<>?
            GROUP BY d.id
            HAVING COUNT(DISTINCT h.band)>=?
            ORDER BY d.id DESC
            """,
            (source_sha256, minimum_shared_bands),
        ).fetchall()

    @staticmethod
    def _hash_bands(hash_value: str) -> list[tuple[int, int]]:
        try:
            value = int(hash_value, 16)
        except (TypeError, ValueError):
            return []
        if value < 0 or value >= 1 << 64:
            return []
        return [(band, (value >> (band * 8)) & 0xFF) for band in range(8)]

    @classmethod
    def _insert_page_hashes(cls, db: sqlite3.Connection, document_id: int, hashes: list[str]) -> None:
        bands = {
            (document_id, band, value)
            for hash_value in hashes
            for band, value in cls._hash_bands(hash_value)
        }
        if bands:
            db.executemany(
                "INSERT OR IGNORE INTO page_hash_index (document_id, band, band_value) VALUES (?, ?, ?)",
                sorted(bands),
            )

    def save_batch(
        self,
        *,
        source_sha256: str,
        config_fingerprint: str,
        source_name: str,
        batch: dict[str, Any],
        output_path: Path,
        created_at: str,
        run_id: str,
        documents: list[dict[str, Any]],
    ) -> None:
        serialized = json.dumps(batch, ensure_ascii=False, sort_keys=True, default=str)
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO processed_inputs
                    (source_sha256, config_fingerprint, batch_json, output_path, source_name, run_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_sha256, config_fingerprint) DO UPDATE SET
                    batch_json=excluded.batch_json,
                    output_path=excluded.output_path,
                    source_name=excluded.source_name,
                    run_id=excluded.run_id,
                    created_at=excluded.created_at
                """,
                (source_sha256, config_fingerprint, serialized, str(output_path), source_name, run_id, created_at),
            )
            for item in documents:
                cursor = db.execute(
                    """
                    INSERT INTO documents
                        (source_sha256, content_fingerprint, page_hashes_json, fields_json, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        source_sha256,
                        item.get("content_fingerprint"),
                        json.dumps(item.get("page_hashes", []), ensure_ascii=False),
                        json.dumps(item.get("fields", {}), ensure_ascii=False, sort_keys=True, default=str),
                        created_at,
                    ),
                )
                self._insert_page_hashes(db, int(cursor.lastrowid), item.get("page_hashes", []))
