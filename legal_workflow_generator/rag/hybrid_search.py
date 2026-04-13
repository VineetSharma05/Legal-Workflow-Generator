import os
import psycopg2
from sentence_transformers import SentenceTransformer

import legal_workflow_generator.config.values as config
from legal_workflow_generator.rag.bm25_retriever import BM25Retriever


class HybridSearcher:
    def __init__(self, embedding_model="sentence-transformers/all-mpnet-base-v2"):
        self.bm25_retriever = BM25Retriever()
        self.model = SentenceTransformer(embedding_model)
        print(f"Loading embedding model: {embedding_model}")

    def build_index(self):
        """Build BM25 index"""
        self.bm25_retriever.build_index()

    def search(self, query, top_k=3, bm25_candidates=20):
        """
        Hybrid search:
        1. BM25 retrieves top bm25_candidates
        2. Re-rank using semantic embedding similarity
        3. Combined score = 0.3 * BM25 + 0.7 * Semantic
        4. Return top_k results as dicts
        """
        # Step 1: BM25 retrieval
        bm25_results = self.bm25_retriever.search(query, top_k=bm25_candidates)
        candidate_ids = [pid for pid, score in bm25_results]

        if not candidate_ids:
            return []

        # Step 2: Normalize BM25 scores to 0-1
        bm25_scores = {pid: score for pid, score in bm25_results}
        max_bm25 = max(bm25_scores.values()) if bm25_scores else 1
        if max_bm25 == 0:
            max_bm25 = 1
        bm25_scores = {pid: score / max_bm25 for pid, score in bm25_scores.items()}

        # Step 3: Fetch candidates and compute semantic similarity
        conn = psycopg2.connect(
            dbname=config.DB_NAME,
            user=config.DB_USER,
            password=os.environ.get("PGPASSWORD"),
            host=config.DB_HOST,
            port=config.DB_PORT,
        )
        cur = conn.cursor()

        placeholders = ','.join(['%s'] * len(candidate_ids))
        cur.execute(f"""
            SELECT provision_id, title, text, plain_english_summary, statute_id, number, embedding
            FROM laws
            WHERE provision_id IN ({placeholders})
              AND embedding IS NOT NULL
        """, candidate_ids)

        candidates = cur.fetchall()
        query_embedding = self.model.encode(query).tolist()

        # Step 4: Compute combined score
        results = []
        for provision_id, title, text, plain_english_summary, statute_id, number, embedding in candidates:
            cur.execute("""
                SELECT 1 - (embedding <=> %s::vector) AS similarity
                FROM laws
                WHERE provision_id = %s
            """, (query_embedding, provision_id))

            row = cur.fetchone()
            if row is None:
                continue

            semantic_score = row[0]
            bm25_score = bm25_scores.get(provision_id, 0)
            combined_score = 0.3 * bm25_score + 0.7 * semantic_score

            results.append({
                "provision_id": provision_id,
                "title": title,
                "text": text,
                "plain_english_summary": plain_english_summary,
                "statute_id": statute_id,
                "number": number,
                "bm25_score": bm25_score,
                "semantic_score": semantic_score,
                "combined_score": combined_score,
            })

        # Step 5: Sort by combined score descending
        results.sort(key=lambda x: x["combined_score"], reverse=True)

        cur.close()
        conn.close()

        return results[:top_k]