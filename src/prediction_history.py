import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = Path(
    os.getenv("DATABASE_PATH", str(PROJECT_ROOT / "data" / "predictions.db"))
)


def get_connection() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


@contextmanager
def database_connection():
    connection = get_connection()
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def initialize_database() -> None:
    with database_connection() as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                source_type TEXT NOT NULL,
                file_name TEXT,
                document_hash TEXT,
                predicted_label INTEGER NOT NULL,
                predicted_category TEXT NOT NULL,
                confidence REAL NOT NULL,
                top_3_json TEXT NOT NULL,
                model_trained_at TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_predictions_created_at
            ON predictions(created_at DESC)
            """
        )


def save_prediction(
    prediction: dict,
    *,
    source_type: str,
    file_name: str | None = None,
    document_hash: str | None = None,
    model_trained_at: str | None = None,
) -> dict:
    initialize_database()
    prediction_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    with database_connection() as connection:
        connection.execute(
            """
            INSERT INTO predictions (
                id, created_at, source_type, file_name, document_hash,
                predicted_label, predicted_category, confidence,
                top_3_json, model_trained_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prediction_id,
                created_at,
                source_type,
                file_name,
                document_hash,
                prediction["predicted_label"],
                prediction["predicted_category"],
                prediction["confidence"],
                json.dumps(prediction["top_3"], ensure_ascii=False),
                model_trained_at,
            ),
        )

    return {
        "prediction_id": prediction_id,
        "created_at": created_at,
        "source_type": source_type,
        "file_name": file_name,
        **prediction,
    }


def row_to_prediction(row: sqlite3.Row) -> dict:
    return {
        "prediction_id": row["id"],
        "created_at": row["created_at"],
        "source_type": row["source_type"],
        "file_name": row["file_name"],
        "predicted_label": row["predicted_label"],
        "predicted_category": row["predicted_category"],
        "confidence": row["confidence"],
        "top_3": json.loads(row["top_3_json"]),
        "model_trained_at": row["model_trained_at"],
    }


def list_predictions(limit: int, offset: int) -> dict:
    initialize_database()
    with database_connection() as connection:
        total = connection.execute(
            "SELECT COUNT(*) FROM predictions"
        ).fetchone()[0]
        rows = connection.execute(
            """
            SELECT * FROM predictions
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()

    return {
        "items": [row_to_prediction(row) for row in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


def get_prediction(prediction_id: str) -> dict | None:
    initialize_database()
    with database_connection() as connection:
        row = connection.execute(
            "SELECT * FROM predictions WHERE id = ?",
            (prediction_id,),
        ).fetchone()

    return row_to_prediction(row) if row else None


def delete_prediction(prediction_id: str) -> bool:
    initialize_database()
    with database_connection() as connection:
        cursor = connection.execute(
            "DELETE FROM predictions WHERE id = ?",
            (prediction_id,),
        )

    return cursor.rowcount == 1
