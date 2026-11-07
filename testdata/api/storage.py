"""SQLite persistence for API assessments (survives server restarts)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "artifacts" / "api_assessments.db"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assessments (
                id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_assessments_patient ON assessments(patient_id)"
        )
        conn.commit()


def save_assessment(data: dict) -> dict:
    aid = data["id"]
    created = data.get("createdAt", "")
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO assessments (id, patient_id, payload, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (aid, data.get("patientId", ""), json.dumps(data), created),
        )
        conn.commit()
    return data


def list_assessments(patient_id: str | None = None) -> list[dict]:
    with _connect() as conn:
        if patient_id:
            rows = conn.execute(
                """
                SELECT payload FROM assessments
                WHERE patient_id = ?
                ORDER BY created_at DESC
                """,
                (patient_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT payload FROM assessments ORDER BY created_at DESC"
            ).fetchall()
    return [json.loads(r["payload"]) for r in rows]


def get_assessment(assessment_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT payload FROM assessments WHERE id = ?",
            (assessment_id,),
        ).fetchone()
    if row is None:
        return None
    return json.loads(row["payload"])
