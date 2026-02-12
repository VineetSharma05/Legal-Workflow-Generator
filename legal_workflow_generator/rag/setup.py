import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

import legal_workflow_generator.config.values as config


def run():
    ## Connect to default postgres database
    conn = psycopg2.connect(
        dbname="postgres",
        user=config.DB_USER,
        host=config.DB_HOST,
        port=config.DB_PORT,
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    ## Check if database exists
    cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{config.DB_NAME}'")
    exists = cur.fetchone()

    ## Create database if it does not exist
    if not exists:
        cur.execute(f"CREATE DATABASE {config.DB_NAME}")
        print(f"Created database {config.DB_NAME}")
    else:
        print(f"Database {config.DB_NAME} already exists")

    cur.close()
    conn.close()

    ## Connect to the new database
    conn = psycopg2.connect(
        dbname=config.DB_NAME,
        user=config.DB_USER,
        host=config.DB_HOST,
        port=config.DB_PORT,
    )
    cur = conn.cursor()

    ## Create laws table
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

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    print("Table 'laws' is ready")

    conn.commit()
    cur.close()
    conn.close()
