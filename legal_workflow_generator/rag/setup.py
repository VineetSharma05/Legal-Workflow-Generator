import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

import legal_workflow_generator.config.values as config


def run():
    print("step 1: connecting to postgres...")
    conn = psycopg2.connect(
        dbname="postgres",
        user=config.DB_USER,
        password=os.environ.get("PGPASSWORD"),
        host=config.DB_HOST,
        port=config.DB_PORT,
        connect_timeout=5,
    )
    print("step 2: connected!")
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    print("step 3: checking if db exists...")
    cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{config.DB_NAME}'")
    exists = cur.fetchone()
    print(f"step 4: exists={exists}")

    if not exists:
        cur.execute(f"CREATE DATABASE {config.DB_NAME}")
        print(f"step 5: Created database {config.DB_NAME}")
    else:
        print(f"step 5: Database {config.DB_NAME} already exists")

    cur.close()
    conn.close()

    print("step 6: connecting to new database...")
    conn = psycopg2.connect(
        dbname=config.DB_NAME,
        user=config.DB_USER,
        password=os.environ.get("PGPASSWORD"),
        host=config.DB_HOST,
        port=config.DB_PORT,
        connect_timeout=5,
    )
    print("step 7: connected to new database!")
    cur = conn.cursor()

    print("step 8: enabling pgvector...")
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    print("step 9: pgvector extension is ready")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS laws (
            provision_id TEXT PRIMARY KEY,
            statute_id TEXT NOT NULL,
            provision_type TEXT,
            chapter TEXT,
            chapter_title TEXT,
            number TEXT,
            title TEXT,
            text TEXT,
            sub_structure JSONB,
            plain_english_summary TEXT,
            keywords TEXT[],
            penalty_linked BOOLEAN,
            effective_date DATE,
            embedding vector(384),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    print("step 10: Table 'laws' is ready")

    cur.execute("""
        CREATE INDEX IF NOT EXISTS laws_embedding_idx
        ON laws
        USING hnsw (embedding vector_cosine_ops);
    """)
    print("step 11: Embedding index is ready")

    conn.commit()
    cur.close()
    conn.close()
    print("step 12: all done!")