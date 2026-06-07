"""PostgreSQL + pgvector access.

All methods are SYNC (psycopg2) and must be invoked via run_in_threadpool.
Cosine distance uses the native pgvector `<=>` operator; lower == more similar.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from app.config import settings


class Database:
    """Thin psycopg2 wrapper. All methods are SYNC and must be threadpooled."""

    def __init__(self) -> None:
        self._conn = None

    def connect(self) -> None:
        import psycopg2
        from pgvector.psycopg2 import register_vector

        self._conn = psycopg2.connect(
            host=settings.pg_host,
            port=settings.pg_port,
            dbname=settings.pg_db,
            user=settings.pg_user,
            password=settings.pg_password,
        )
        self._conn.autocommit = True
        register_vector(self._conn)

    def ping(self) -> bool:
        with self._conn.cursor() as cur:
            cur.execute("SELECT 1;")
            return cur.fetchone()[0] == 1

    def upsert_face(self, user_id: str, embedding: np.ndarray) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO faces (user_id, embedding)
                VALUES (%s, %s)
                ON CONFLICT (user_id)
                DO UPDATE SET embedding = EXCLUDED.embedding,
                              updated_at = now();
                """,
                (user_id, embedding),
            )

    def nearest(self, user_id: str, embedding: np.ndarray) -> Optional[float]:
        """Return cosine DISTANCE to the stored embedding for user_id, or None."""
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT embedding <=> %s AS distance FROM faces WHERE user_id = %s;",
                (embedding, user_id),
            )
            row = cur.fetchone()
            return float(row[0]) if row else None

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()


db = Database()
