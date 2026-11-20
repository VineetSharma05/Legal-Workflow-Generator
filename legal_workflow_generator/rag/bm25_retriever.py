import psycopg2
from rank_bm25 import BM25Okapi

import legal_workflow_generator.config.values as config
from legal_workflow_generator.rag.ingestion import normalize_text, searchable_text


class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.law_ids = []
        self.law_domains = []
        self.law_texts = []

    def build_index(self):
        """Build BM25 index from database laws"""
        conn = psycopg2.connect(
            dbname=config.DB_NAME,
            user=config.DB_USER,
            password=config.PGPASSWORD,
            host=config.DB_HOST,
            port=config.DB_PORT,
        )
        cur = conn.cursor()

        cur.execute("SELECT provision_id, text, sub_structure, domain FROM laws WHERE text IS NOT NULL")
        rows = cur.fetchall()

        self.law_ids = [row[0] for row in rows]
        self.law_domains = [row[3] for row in rows]
        self.law_texts = [searchable_text(row[1], row[2]) for row in rows]

        # Tokenize with BM25-specific normalization (stopwords + stemming)
        tokenized_corpus = [self._tokenize(text) for text in self.law_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        cur.close()
        conn.close()
        print(f"BM25 index built with {len(self.law_ids)} laws")

    def _tokenize(self, text: str):
        """
        Tokenize using enhanced preprocessing:
        - Section stripping
        - Query expansion
        - Stopword removal
        - Stemming
        """
        # for_embedding=False applies stopword removal + stemming
        normalized = normalize_text(text, for_embedding=False)
        return normalized.split()

    def search(self, query: str, top_k: int = 10, domain: str | None = None, prefixes=None):
        """Search using BM25 and return top_k (provision_id, score) tuples.
        With `domain`, only provisions of that domain are ranked (the scores are unchanged).
        With `prefixes`, only provisions whose id starts with one of them are ranked (e.g. one statute)."""
        if self.bm25 is None:
            raise ValueError("BM25 index not built. Call build_index() first.")

        # Apply same preprocessing to query + expand acronyms
        tokenized_query = self._tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)
        pool = range(len(scores)) if domain is None else [i for i, d in enumerate(self.law_domains) if d == domain]
        if prefixes:
            pre = tuple(prefixes)
            pool = [i for i in pool if self.law_ids[i].startswith(pre)]
        top_indices = sorted(pool, key=lambda i: scores[i], reverse=True)[:top_k]
        results = [(self.law_ids[i], scores[i]) for i in top_indices]
        return results