-- ---------------------------------------------------------------------------
-- PostgreSQL schema for the face-verification pipeline.
-- Requires the pgvector extension. ArcFace produces 512-d embeddings.
--
-- Distance: cosine (<=>). Lower == more similar. The application enforces a
-- strict cutoff of <= 0.40 for a positive match.
-- ---------------------------------------------------------------------------

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS faces (
    user_id     TEXT PRIMARY KEY,
    embedding   vector(512) NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- IVFFlat index tuned for cosine distance. Run ANALYZE after bulk loads.
-- `lists` should scale roughly with sqrt(row_count); 100 is a sane default
-- for up to ~1M rows. Build the index AFTER loading data for best recall.
CREATE INDEX IF NOT EXISTS faces_embedding_cosine_idx
    ON faces
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
