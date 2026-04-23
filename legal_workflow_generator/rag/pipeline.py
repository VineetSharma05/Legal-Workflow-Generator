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

    def _build_retrieval_query(
        self,
        query: str,
        legal_context: dict[str, Any] | None,
    ) -> str:
        """
        Build a retrieval-friendly query using structured context from the query unit.

        This improves retrieval grounding by adding domain/intent/keywords context,
        while keeping backward compatibility when only raw query is available.
        """
        if not legal_context:
            return query

        normalized_query = legal_context.get("normalized_query") or query
        legal_domain = str(legal_context.get("legal_domain") or "").strip().lower()
        keywords = legal_context.get("keywords") or []
        confidence = legal_context.get("confidence")

        intent_raw = legal_context.get("intent")
        intent = getattr(intent_raw, "value", intent_raw)
        intent = str(intent).strip().lower() if intent else ""

        intent_hints = {
            "workflow": "step by step procedure compliance checklist",
            "compliance_check": "compliance requirements obligations penalties verification",
            "qa": "legal explanation interpretation",
        }

        parts = [normalized_query]

        if legal_domain and legal_domain != "unknown":
            parts.append(f"domain {legal_domain.replace('_', ' ')}")

        if keywords:
            clean_keywords = [str(k).strip() for k in keywords if str(k).strip()]
            if clean_keywords:
                parts.append("keywords " + " ".join(clean_keywords[:8]))

        if intent in intent_hints:
            parts.append(intent_hints[intent])

        # For low-confidence intent/domain classification, rely more on normalized text.
        # We still include weak hints because they can improve BM25 recall in many cases.
        if isinstance(confidence, (int, float)) and confidence < 0.5:
            parts.append("broad legal context")

        return " ".join(parts)

    def run(
        self,
        query: str,
        top_k: int = 3,
        bm25_candidates: int = 20,
        legal_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._ensure_index()

        retrieval_query = self._build_retrieval_query(query, legal_context)
        generation_query = (
            legal_context.get("original_query", query) if legal_context else query
        )

        retrieved = self.searcher.search(
            retrieval_query,
            top_k=top_k,
            bm25_candidates=bm25_candidates,
        )

        answer = self.generator.generate(query=generation_query, retrieved_chunks=retrieved)

        return {
            "query": generation_query,
            "retrieval_query": retrieval_query,
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
