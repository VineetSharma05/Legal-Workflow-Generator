"""
Extracts distinctive keywords per legal domain from the ingested law corpus
and persists them to the `domain_keywords` table, where they're loaded by
KeywordDomainClassifier (query/keyword_domain_classifier.py) for deterministic,
zero-LLM-cost domain classification.

Method: each domain's provisions are concatenated into a single pseudo-document
(so there are only as many "documents" as there are domains), then TF-IDF is
fit across those pseudo-documents. This is the standard way to pull terms that
are distinctive to one class out of a small number of classes — a term that's
common everywhere (e.g. "company") scores low everywhere, one that's
concentrated in one domain (e.g. "data fiduciary") scores high only there.
Unigrams and bigrams are both kept since legal terms are often multi-word
("data fiduciary", "internal complaints committee").

Provision text is run through the same stemming + stopword removal used for
the BM25 index (ingestion.normalize_text with for_embedding=False), so the
extracted terms match the tokenization KeywordDomainClassifier applies to
queries at classification time.
"""

import psycopg2
from psycopg2.extras import execute_batch
from sklearn.feature_extraction.text import TfidfVectorizer

import legal_workflow_generator.config.values as config
from legal_workflow_generator.rag.ingestion import normalize_text

DEFAULT_TOP_N = 40


def run(top_n: int = DEFAULT_TOP_N) -> None:
    conn = psycopg2.connect(
        dbname=config.DB_NAME,
        user=config.DB_USER,
        password=config.PGPASSWORD,
        host=config.DB_HOST,
        port=config.DB_PORT,
    )
    cur = conn.cursor()

    cur.execute("""
        SELECT domain, text FROM laws
        WHERE domain IS NOT NULL AND domain != 'unknown' AND text IS NOT NULL
    """)
    rows = cur.fetchall()

    if not rows:
        print("No domain-tagged laws found. Run `python main.py ingest` first.")
        cur.close()
        conn.close()
        return

    domain_texts: dict[str, list[str]] = {}
    for domain, text in rows:
        domain_texts.setdefault(domain, []).append(normalize_text(text, for_embedding=False))

    domains = sorted(domain_texts.keys())
    corpus = [" ".join(domain_texts[d]) for d in domains]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
        lowercase=False,  # text is already lowercased + stemmed by normalize_text
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    terms = vectorizer.get_feature_names_out()

    cur.execute("DELETE FROM domain_keywords")

    insert_rows = []
    for i, domain in enumerate(domains):
        scores = tfidf_matrix[i].toarray().ravel()
        top_indices = scores.argsort()[::-1][:top_n]
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            insert_rows.append((domain, terms[idx], score))

    execute_batch(
        cur,
        "INSERT INTO domain_keywords (domain, term, score) VALUES (%s, %s, %s)",
        insert_rows,
    )
    conn.commit()
    cur.close()
    conn.close()

    print(f"Extracted keywords for {len(domains)} domains ({len(insert_rows)} total terms)")
    for domain in domains:
        n = sum(1 for r in insert_rows if r[0] == domain)
        top5 = [r[1] for r in insert_rows if r[0] == domain][:5]
        print(f"  {domain}: {n} terms — top: {top5}")
