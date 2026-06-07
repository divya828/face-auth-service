"""Env-driven application configuration (see .env.example)."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Model / pipeline
    model_name: str = "ArcFace"
    detector_backend: str = "retinaface"
    embedding_dim: int = 512
    cosine_threshold: float = 0.40  # DISTANCE cutoff: match iff distance <= this

    # PostgreSQL
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_db: str = "faces"
    pg_user: str = "postgres"
    pg_password: str = "postgres"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    rate_limit_max: int = 3
    rate_limit_window_s: int = 60

    # AWS S3 cold storage
    aws_region: str = "us-east-1"
    s3_bucket: str = "payment-fraud-review-snapshots"
    s3_prefix: str = "fraud_reviews/"


settings = Settings()
