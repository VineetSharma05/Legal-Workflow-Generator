import os
import psycopg2
from psycopg2.extras import execute_batch
from sentence_transformers import SentenceTransformer

import legal_workflow_generator.config.values as config


BATCH_SIZE = 100

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIM = 768


def build_embedding_text(title, text, summary, keywords):
    parts = []

    if title:
        parts.append(title)

    if text:
        parts.append(text)

    if summary:
        parts.append(f"Summary: {summary}")

    if keywords:
        parts.append("Keywords: " + " ".join(keywords))

    return ". ".join(parts)


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

    cur.execute("""
        SELECT provision_id, title, text, plain_english_summary, keywords
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

        batch = rows[batch_start: batch_start + BATCH_SIZE]

        provision_ids = []
        texts = []

        for row in batch:
            provision_id, title, text, summary, keywords = row

            embedding_text = build_embedding_text(
                title,
                text,
                summary,
                keywords or []
            )

            # Add prefix so document and query embeddings are in the same space
            prefixed_text = f"Indian law provision: {embedding_text}"

            provision_ids.append(provision_id)
            texts.append(prefixed_text)

        embeddings = model.encode(texts, show_progress_bar=False)

        execute_batch(
            cur,
            "UPDATE laws SET embedding = %s::vector WHERE provision_id = %s",
            [
                (embedding.tolist(), provision_id)
                for embedding, provision_id in zip(embeddings, provision_ids)
            ],
        )

        conn.commit()

        print(f"  Embedded {min(batch_start+BATCH_SIZE, len(rows))}/{len(rows)}")

    cur.close()
    conn.close()

    print("Done embedding all laws")