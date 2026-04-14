from typing import Any

from legal_workflow_generator.rag.generator import GeminiAnswerGenerator
from legal_workflow_generator.rag.generator import GroqAnswerGenerator
from legal_workflow_generator.rag.hybrid_search import HybridSearcher


class RagPipeline:
    """
    End-to-end RAG pipeline:
      query -> hybrid retrieval -> Gemini/Llama grounded answer
    """

    def __init__(self, llm_provider="gemini"):
        self.searcher = HybridSearcher()
        # Set up the correct LLM generator based on the flag
        if llm_provider == "groq":
            self.generator = GroqAnswerGenerator()
        else:
            self.generator = GeminiAnswerGenerator()
        self._index_ready = False

    def _ensure_index(self) -> None:
        if not self._index_ready:
            self.searcher.build_index()
            self._index_ready = True

    def run(
        self,
        query: str,
        top_k: int = 3,
        bm25_candidates: int = 20,
    ) -> dict[str, Any]:
        self._ensure_index()

        retrieved = self.searcher.search(
            query,
            top_k=top_k,
            bm25_candidates=bm25_candidates,
        )

        answer = self.generator.generate(query=query, retrieved_chunks=retrieved)

        return {
            "query": query,
            "answer": answer,
            "retrieved": [
                {
                    "provision_id": r.get("provision_id"),
                    "title": r.get("title"),
                    "statute_id": r.get("statute_id"),
                    "number": r.get("number"),
                    "combined_score": r.get("combined_score"),
                }
                for r in retrieved
            ],
        }
