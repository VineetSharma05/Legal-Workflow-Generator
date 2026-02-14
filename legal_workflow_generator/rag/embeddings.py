import os
import psycopg2
from psycopg2.extras import execute_batch
from sentence_transformers import SentenceTransformer

import legal_workflow_generator.config.values as config

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 100


def run() -> None:
    print(f"Loading embedding model '{EMBEDDING_MODEL}'...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    conn = psycopg2.connect(
        dbname=config.DB_NAME,
        user=config.DB_USER,
        password=os.environ.get("PGPASSWORD"),
        host=config.DB_HOST,
        port=config.DB_PORT,
    )
    cur = conn.cursor()

    # Fetch all rows that don't have an embedding yet
    cur.execute("""
        SELECT provision_id, text
        FROM laws
        WHERE embedding IS NULL
          AND text IS NOT NULL
    """)
    rows = cur.fetchall()

    if not rows:
        print("No rows to embed")
        cur.close()
        conn.close()
        return

    print(f"Embedding {len(rows)} laws...")

    for batch_start in range(0, len(rows), BATCH_SIZE):
        batch = rows[batch_start : batch_start + BATCH_SIZE]

        provision_ids = [row[0] for row in batch]
        texts = [row[1] for row in batch]

        embeddings = model.encode(texts, show_progress_bar=False)

        execute_batch(
            cur,
            "UPDATE laws SET embedding = %s::vector WHERE provision_id = %s",
            [(embedding.tolist(), provision_id) for embedding, provision_id in zip(embeddings, provision_ids)],
        )

        conn.commit()
        print(f"  Embedded {min(batch_start + BATCH_SIZE, len(rows))}/{len(rows)}")

    cur.close()
    conn.close()
    print("Done embedding all laws")