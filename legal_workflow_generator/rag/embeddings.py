import time

import psycopg2
from psycopg2.extras import execute_batch
from sentence_transformers import SentenceTransformer

import legal_workflow_generator.config.values as config
from legal_workflow_generator.rag.ingestion import searchable_text


BATCH_SIZE = 100
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIM = 768
MAX_BATCH_ATTEMPTS = 3


def build_embedding_text(text: str, sub_structure=None) -> str:
    """
    text column already contains combined + preprocessed content from ingestion.py:
    title + plain_english_summary + keywords + raw_text + chapter_title.
    The clauses in sub_structure are appended (same text the BM25 index uses).
    """
    return searchable_text(text, sub_structure)


def _connect():
    return psycopg2.connect(
        dbname=config.DB_NAME,
        user=config.DB_USER,
        password=config.PGPASSWORD,
        host=config.DB_HOST,
        port=config.DB_PORT,
    )


def _write_batch(rows: list) -> None:
    """Store one batch on a fresh connection, retrying if the connection drops.
    One connection held for the whole run was cut mid-way on a ~30 minute CPU run."""
    for attempt in range(1, MAX_BATCH_ATTEMPTS + 1):
        try:
            conn = _connect()
            try:
                with conn, conn.cursor() as cur:
                    execute_batch(cur, "UPDATE laws SET embedding = %s::vector WHERE provision_id = %s", rows)
                return
            finally:
                conn.close()
        except psycopg2.OperationalError:
            if attempt == MAX_BATCH_ATTEMPTS:
                raise
            time.sleep(2 * attempt)


def run(reembed: bool = False) -> None:
    """Embed every row that has no embedding yet.

    reembed=True first clears all embeddings, then fills them again (needed after a change to what gets
    embedded). The run is resumable: if it stops, `python main.py embed` continues with the rows still empty.
    Rows without an embedding are skipped by search until they are filled.
    """
    print(f"Loading embedding model '{EMBEDDING_MODEL}'...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    conn = _connect()
    with conn, conn.cursor() as cur:
        if reembed:
            cur.execute("UPDATE laws SET embedding = NULL")
            print(f"Cleared {cur.rowcount} embeddings")
        cur.execute("""
            SELECT provision_id, text, sub_structure
            FROM laws
            WHERE embedding IS NULL
              AND text IS NOT NULL
            ORDER BY provision_id
        """)
        rows = cur.fetchall()
    conn.close()

    if not rows:
        print("No rows to embed")
        return

    print(f"Embedding {len(rows)} laws...", flush=True)

    for batch_start in range(0, len(rows), BATCH_SIZE):
        batch = rows[batch_start: batch_start + BATCH_SIZE]

        # Prefix aligns document embeddings with query embedding space
        texts = [f"Indian law provision: {build_embedding_text(text, sub)}" for _, text, sub in batch]
        embeddings = model.encode(texts, show_progress_bar=False)

        _write_batch([(emb.tolist(), pid) for emb, (pid, _, _) in zip(embeddings, batch)])
        print(f"  Embedded {min(batch_start + BATCH_SIZE, len(rows))}/{len(rows)}", flush=True)

    print("Done embedding all laws")
