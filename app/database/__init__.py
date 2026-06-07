"""PostgreSQL / pgvector access layer."""

from app.database.postgres import Database, db

__all__ = ["Database", "db"]
