from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# ── Pipeline imports ─────────────────────────────────────────────────────────
import legal_workflow_generator.config.values  # noqa: F401 — loads .env
from legal_workflow_generator.agent.graph import graph, run_workflow
from legal_workflow_generator.agent.settings import RunSettings, build_config
from legal_workflow_generator.rag.pipeline import RagPipeline
from legal_workflow_generator.agent import nodes as agent_nodes
from legal_workflow_generator.agent.nodes import classify_query as agent_classify_query
from legal_workflow_generator.agent.state import AgentState

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Pre-warm embedding model, BM25 index, domain map, and statute catalog on server startup."""
    import os
    if os.getenv("SKIP_PREWARM", "").strip().lower() not in ("1", "true", "yes"):
        try:
            agent_nodes._get_rag_pipeline()
            agent_nodes._load_domain_map()
            agent_nodes._catalog()
        except Exception as e:
            logger.warning("Startup pre-warm skipped: %s", e)
    yield


# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="Legal Workflow Assistant", lifespan=lifespan)

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
    provider: str = "gemini"             # "gemini" | "groq"  (validated by RunSettings)
    top_k: int = 8                       # provisions per domain, 1..20
    as_of: str | None = None             # ISO date the answer must be valid on; default today
    verify_claims: bool | None = None    # per-step entailment check; default from settings (on)


class TestRequest(BaseModel):
    query: str
    provider: str = "gemini"
    top_k: int = 8
    as_of: str | None = None


# ── Helpers ──────────────────────────────────────────────────────────────────
def _safe_str(val, max_len: int = 600) -> str:
    """Stringify and optionally truncate a value."""
    s = str(val) if val is not None else ""
    return s[:max_len] if len(s) > max_len else s


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


def _build_chat_payload(state: dict, cfg: RunSettings) -> dict:
    """Build the JSON response payload from an AgentState (partial or complete)."""
    docs = state.get("retrieved_docs") or []
    domain_results = state.get("domain_results") or []
    settings = {
        "provider": cfg.provider, "top_k": cfg.top_k, "as_of": cfg.as_of_date.isoformat(),
        "verify_claims": cfg.verify_claims,
    }
    workflow = [
        {
            "domain": result.get("domain"),
            "grade": result.get("grade"),
            "reason": result.get("reason"),
            "retries": result.get("retries", 0),
            "summary": result.get("summary", []),
            "steps": result.get("steps", []),
            "not_covered": result.get("not_covered", ""),
            "cited": result.get("cited", []),
        }
        for result in domain_results
    ]
    return {
        "answer": state.get("stitched_answer") or state.get("answer", ""),
        "abstained": bool(state.get("abstain")),
        "settings": settings,
        "workflow": {
            "domains": workflow,
            "gaps": state.get("domain_gaps", []),
        },
        "pipeline": {
            "settings": settings,
            "classification": {
                "intent": state.get("intent"),
                "domain": state.get("domain"),
                "all_domains": state.get("all_domains", []),
                "normalized_query": state.get("normalized_query"),
                "keywords": state.get("keywords", []),
                "confidence": state.get("confidence"),
            } if state.get("intent") or state.get("domain") else None,
            "retrieval": {
                "docs": [
                    {
                        "provision_id": d.get("provision_id"),
                        "title": d.get("title"),
                        "text": _safe_str(d.get("text")),
                        "effective_date": str(d["effective_date"]) if d.get("effective_date") else None,
                        "combined_score": d.get("combined_score"),
                        "bm25_score": d.get("bm25_score"),
                        "semantic_score": d.get("semantic_score"),
                    }
                    for d in docs
                ],
                "rewritten_query": state.get("rewritten_query"),
                "retry_count": state.get("retry_count_retrieval", 0),
            } if docs or state.get("context_grade") else None,
            "domains": domain_results,
            "domain_gaps": state.get("domain_gaps", []),
            "context_grading": {
                "grade": state.get("context_grade"),
                "reason": state.get("context_grade_reason"),
            },
            "generation": {
                "citations": state.get("citations", []),
                "retry_count": state.get("retry_count_generation", 0),
            } if state.get("citations") is not None else None,
            "citation_verification": {
                "verified": state.get("verified_citations", []),
                "failed": state.get("failed_citations", []),
            } if state.get("verified_citations") is not None else None,
            "claim_verification": _claim_summary(domain_results),
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


@app.post("/api/chat")
def chat(req: ChatRequest):
    """
    Full agent pipeline.

    The request's provider / top_k / as_of / verify_claims are passed INTO the LangGraph run
    (as its config), so every worker and grader uses them. Invalid values are rejected with 400.
    The response carries the final answer, the effective settings and every intermediate stage.
    """
    try:
        cfg = RunSettings.from_config(build_config(provider=req.provider, top_k=req.top_k,
                                                   as_of=req.as_of, verify_claims=req.verify_claims))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        state = run_workflow(req.query, provider=req.provider, top_k=req.top_k,
                             as_of=req.as_of, verify_claims=req.verify_claims)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    return _build_chat_payload(state, cfg)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    """Stream each LangGraph node update as SSE (`stage`), then the complete response (`done`)."""
    raw_cfg = build_config(provider=req.provider, top_k=req.top_k,
                           as_of=req.as_of, verify_claims=req.verify_claims)
    try:
        cfg = RunSettings.from_config(raw_cfg)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    def event_generator():
        acc_state: dict = {"query": req.query, "trace": []}
        try:
            for chunk in graph.stream({"query": req.query}, config=raw_cfg, stream_mode="updates"):
                for node_name, update in chunk.items():
                    if isinstance(update, dict):
                        acc_state.update(update)
                    partial = _build_chat_payload(acc_state, cfg)
                    yield _sse("stage", {
                        "node": node_name,
                        "pipeline": partial["pipeline"],
                        "trace": partial["trace"],
                    })
            yield _sse("done", _build_chat_payload(acc_state, cfg))
        except Exception as e:
            yield _sse("error", {"detail": f"Pipeline error: {e}"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _claim_summary(domain_results: list) -> dict:
    """Counts of claim-verdicts across all domains (entailed / partial / unchecked / dropped)."""
    counts = {"entailed": 0, "partial": 0, "unchecked": 0, "dropped": 0}
    for r in domain_results:
        for st in r.get("steps", []):
            v = st.get("verdict", "unchecked")
            counts[v if v in counts else "unchecked"] += 1
        counts["dropped"] += len(r.get("dropped_steps", []))
    return counts


@app.post("/api/test/query")
def test_query(req: TestRequest):
    """Query processing unit — now uses the agent's classify_query node (multi-domain aware)."""
    try:
        state: AgentState = {"query": req.query, "trace": []}
        cfg = build_config(provider=req.provider, top_k=req.top_k, as_of=req.as_of)
        result_state = agent_classify_query(state, config={"configurable": cfg["configurable"]})
        return {
            "intent": result_state.get("intent"),
            "domain": result_state.get("domain"),
            "all_domains": result_state.get("all_domains", []),
            "normalized_query": result_state.get("normalized_query"),
            "keywords": result_state.get("keywords", []),
            "confidence": result_state.get("confidence"),
            "trace": result_state.get("trace", []),
        }
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


# ── Entrypoint ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)