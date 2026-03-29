import os
import psycopg2
from rank_bm25 import BM25Okapi

import legal_workflow_generator.config.values as config
from legal_workflow_generator.rag.ingestion import normalize_text, expand_query


class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.law_ids = []
        self.law_texts = []

    def build_index(self):
        """Build BM25 index from database laws"""
        conn = psycopg2.connect(
            dbname=config.DB_NAME,
            user=config.DB_USER,
            password=os.environ.get("PGPASSWORD"),
            host=config.DB_HOST,
            port=config.DB_PORT,
        )
        cur = conn.cursor()

        cur.execute("SELECT provision_id, text FROM laws WHERE text IS NOT NULL")
        rows = cur.fetchall()

        self.law_ids = [row[0] for row in rows]
        self.law_texts = [row[1] for row in rows]

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

    def search(self, query: str, top_k: int = 10):
        """Search using BM25 and return top_k (provision_id, score) tuples"""
        if self.bm25 is None:
            raise ValueError("BM25 index not built. Call build_index() first.")

        # Apply same preprocessing to query + expand acronyms
        tokenized_query = self._tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = [(self.law_ids[i], scores[i]) for i in top_indices]
        return results