import os
import psycopg2
from rank_bm25 import BM25Okapi
import re

import legal_workflow_generator.config.values as config


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
        
        # Fetch all laws
        cur.execute("SELECT provision_id, text FROM laws WHERE text IS NOT NULL")
        rows = cur.fetchall()
        
        self.law_ids = [row[0] for row in rows]
        self.law_texts = [row[1] for row in rows]
        
        # Tokenize documents for BM25
        tokenized_corpus = [self._tokenize(text) for text in self.law_texts]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        cur.close()
        conn.close()
        
        print(f"BM25 index built with {len(self.law_ids)} laws")
    
    def _tokenize(self, text):
        """Simple tokenization - lowercase and split"""
        text = text.lower()
        # Remove special characters but keep numbers
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return text.split()
    
    def search(self, query, top_k=10):
        """Search using BM25 and return top_k provision_ids"""
        if self.bm25 is None:
            raise ValueError("BM25 index not built. Call build_index() first.")
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top_k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        # Return provision_ids with scores
        results = [(self.law_ids[i], scores[i]) for i in top_indices]
        return results
