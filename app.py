from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── Pipeline imports ─────────────────────────────────────────────────────────
import legal_workflow_generator.config.values  # noqa: F401 — loads .env
from legal_workflow_generator.agent.graph import graph
from legal_workflow_generator.query import process_query
from legal_workflow_generator.rag.pipeline import RagPipeline

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="Legal Workflow Assistant")

STATIC_DIR = (
    Path(__file__).resolve().parent
    / "legal_workflow_generator"
    / "presentation"
    / "static"
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Cache RAG pipeline instances by provider so the embedding model and BM25
# index are loaded only once per provider.
_rag_cache: dict[str, RagPipeline] = {}


def _get_rag(provider: str) -> RagPipeline:
    if provider not in _rag_cache:
        _rag_cache[provider] = RagPipeline(llm_provider=provider)
    return _rag_cache[provider]


# ── Request / response models ───────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str
    provider: str = "gemini"
    top_k: int = 3


class TestRequest(BaseModel):
    query: str
    provider: str = "gemini"
    top_k: int = 3


# ── Helpers ──────────────────────────────────────────────────────────────────
def _serialize_context(ctx: dict) -> dict:
    """Convert a LegalContext dict to a JSON-safe dict."""
    intent = ctx.get("intent")
    return {
        "original_query": ctx.get("original_query"),
        "normalized_query": ctx.get("normalized_query"),
        "intent": intent.value if hasattr(intent, "value") else str(intent),
        "legal_domain": ctx.get("legal_domain"),
        "keywords": ctx.get("keywords", []),
        "confidence": ctx.get("confidence"),
        "keyword_domain": ctx.get("keyword_domain"),
        "domain_agreement": ctx.get("domain_agreement"),
        "domain_confidence": ctx.get("domain_confidence"),
        "domain_source": ctx.get("domain_source"),
    }


def _safe_str(val, max_len: int = 600) -> str:
    """Stringify and optionally truncate a value."""
    s = str(val) if val is not None else ""
    return s[:max_len] if len(s) > max_len else s


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/api/chat")
def chat(req: ChatRequest):
    """
    Full agent pipeline.

    Invokes the compiled LangGraph and returns the final answer together
    with every intermediate pipeline stage for frontend inspection.
    """
    try:
        state = graph.invoke({"query": req.query})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    docs = state.get("retrieved_docs") or []

    return {
        "answer": state.get("answer", ""),
        "abstained": bool(state.get("abstain")),
        "pipeline": {
            "classification": {
                "intent": state.get("intent"),
                "domain": state.get("domain"),
                "all_domains": state.get("all_domains", []),
                "normalized_query": state.get("normalized_query"),
                "keywords": state.get("keywords", []),
                "confidence": state.get("confidence"),
            },
            "retrieval": {
                "docs": [
                    {
                        "provision_id": d.get("provision_id"),
                        "title": d.get("title"),
                        "text": _safe_str(d.get("text")),
                        "combined_score": d.get("combined_score"),
                        "bm25_score": d.get("bm25_score"),
                        "semantic_score": d.get("semantic_score"),
                    }
                    for d in docs
                ],
                "rewritten_query": state.get("rewritten_query"),
                "retry_count": state.get("retry_count_retrieval", 0),
            },
            "context_grading": {
                "grade": state.get("context_grade"),
                "reason": state.get("context_grade_reason"),
            },
            "generation": {
                "citations": state.get("citations", []),
                "retry_count": state.get("retry_count_generation", 0),
            },
            "citation_verification": {
                "verified": state.get("verified_citations", []),
                "failed": state.get("failed_citations", []),
            },
            "groundedness": {
                "grade": state.get("groundedness_grade"),
                "reason": state.get("groundedness_reason"),
            },
            "answerability": {
                "grade": state.get("answerability_grade"),
                "reason": state.get("answerability_reason"),
            },
            "abstain_info": {
                "triggered": bool(state.get("abstain")),
                "reason": state.get("abstain_reason", ""),
            },
        },
        "trace": state.get("trace", []),
    }


@app.post("/api/test/query")
def test_query(req: TestRequest):
    """Query processing unit only (normalize → classify intent → resolve context)."""
    try:
        return _serialize_context(process_query(text=req.query))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/test/rag")
def test_rag(req: TestRequest):
    """RAG pipeline only (hybrid BM25+semantic search → LLM generation)."""
    try:
        pipeline = _get_rag(req.provider)
        result = pipeline.run(query=req.query, top_k=req.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/test/classify")
def test_classify(req: TestRequest):
    """Domain classification only (same as query unit, focused on domain fields)."""
    try:
        return _serialize_context(process_query(text=req.query))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Entrypoint ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)