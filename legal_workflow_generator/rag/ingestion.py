import os
from typing import List
import psycopg2
from psycopg2.extras import execute_batch, Json

import legal_workflow_generator.typings.types as T
import legal_workflow_generator.config.values as config


def ingest(laws: List[T.LawSchema]) -> None:
    if not laws:
        print("No laws provided for ingestion")
        return

    conn = psycopg2.connect(
        dbname=config.DB_NAME,
        user=config.DB_USER,
        password=os.environ.get("PGPASSWORD"),
        host=config.DB_HOST,
        port=config.DB_PORT,
    )
    cur = conn.cursor()

    query = """
        INSERT INTO laws (
            provision_id,
            statute_id,
            provision_type,
            chapter,
            chapter_title,
            number,
            title,
            text,
            sub_structure,
            plain_english_summary,
            keywords,
            penalty_linked,
            effective_date
        )
        VALUES (
            %(provision_id)s,
            %(statute_id)s,
            %(provision_type)s,
            %(chapter)s,
            %(chapter_title)s,
            %(number)s,
            %(title)s,
            %(text)s,
            %(sub_structure)s,
            %(plain_english_summary)s,
            %(keywords)s,
            %(penalty_linked)s,
            %(effective_date)s
        )
        ON CONFLICT (provision_id)
        DO UPDATE SET
            statute_id = EXCLUDED.statute_id,
            provision_type = EXCLUDED.provision_type,
            chapter = EXCLUDED.chapter,
            chapter_title = EXCLUDED.chapter_title,
            number = EXCLUDED.number,
            title = EXCLUDED.title,
            text = EXCLUDED.text,
            sub_structure = EXCLUDED.sub_structure,
            plain_english_summary = EXCLUDED.plain_english_summary,
            keywords = EXCLUDED.keywords,
            penalty_linked = EXCLUDED.penalty_linked,
            effective_date = EXCLUDED.effective_date;
    """

    prepared_rows = []

    for law in laws:
        row = law.copy()

        row["sub_structure"] = Json(law.get("sub_structure", {}))
        row["keywords"] = law.get("keywords", [])
        row["plain_english_summary"] = law.get("plain_english_summary", "")
        row["penalty_linked"] = law.get("penalty_linked", False)

        prepared_rows.append(row)

    execute_batch(cur, query, prepared_rows, page_size=100)
    conn.commit()

    cur.close()
    conn.close()

    print(f"Ingested {len(laws)} laws successfully")