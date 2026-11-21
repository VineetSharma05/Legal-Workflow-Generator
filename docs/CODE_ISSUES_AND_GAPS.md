# Codebase Issues, Gaps & Architectural Risks

This document highlights structural holes, reliability/concurrency bugs, security concerns, and architectural inconsistencies found while auditing the **Legal Workflow Generator** codebase.

---

## 1. Architectural & Design Inconsistencies

### A. Split LLM Abstraction (README vs. Reality)
- **Where**: [legal_workflow_generator/llm/__init__.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/legal_workflow_generator/llm/__init__.py) vs. [legal_workflow_generator/agent/gemini_client.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/legal_workflow_generator/agent/gemini_client.py) & [legal_workflow_generator/agent/groq_client.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/legal_workflow_generator/agent/groq_client.py)
- **Issue**: `README.md` and `llm/__init__.py` state that *"Every LLM call in the project goes through LangChain chat models built in `legal_workflow_generator/llm/`; no module imports the `google-genai` or `groq` SDKs directly."* In reality:
  - Plain-text calls in `agent/nodes.py` (`_call_gemini`, `_call_groq`) bypass LangChain and call `google.genai.Client` (`RotatingGeminiClient`) and `groq.Groq` (`groq_client.complete`) directly.
  - Conversely, structured-output calls (`structured_llm(...)` in `llm/__init__.py`) use `ChatGoogleGenerativeAI` with only the **first** `GEMINI_API_KEY` — meaning **API key rotation (`RotatingGeminiClient`) and token usage tracking (`USAGE`) do NOT work for structured-output calls**, and Groq's TPM token-budget pacing (`groq_client.py`) is bypassed for structured Groq calls.
- **Impact**: Under heavy load or evaluation runs, structured calls can hit 429s on key #1 without rotating to keys #2..N, and token-cost reporting undercounts all structured calls.

### B. Monolithic `agent/nodes.py` (~1,550 lines)
- **Where**: [legal_workflow_generator/agent/nodes.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/legal_workflow_generator/agent/nodes.py)
- **Issue**: A single file mixes typed data models (`Provision`, `Step`, `DomainResult`), raw `psycopg2` SQL queries, LLM retry/provider routing, temporal/scope heuristics, inline prompt templates, retrieval fusion, grading, claim verification, stitching, and LangGraph node definitions.
- **Impact**: Hard to navigate, test in isolation, or modify prompts without risking regressions in retrieval or DB code.

---

## 2. Reliability & Concurrency Gaps

### A. No Database Connection Pooling
- **Where**: [legal_workflow_generator/agent/nodes.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/legal_workflow_generator/agent/nodes.py#L292) (`_connect()`), `rag/hybrid_search.py`, `rag/embeddings.py`, etc.
- **Issue**: Every DB lookup opens a brand-new `psycopg2.connect(...)` and closes it. Because `domain_pipeline` fans out across up to 5 concurrent worker threads, a single multi-domain query opens and tears down dozens of short-lived Postgres connections.
- **Impact**: High connection overhead and risk of exhausting Postgres `max_connections` under concurrent FastAPI traffic.

### B. Unbounded / Unsynchronized Global Caches & Mutable Process Globals
- **Where**:
  - [app.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/app.py#L29) (`_rag_cache`), [agent/nodes.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/legal_workflow_generator/agent/nodes.py#L66) (`_rag_pipeline`, `_DOMAIN_MAP`, `_HAS_RAW_TEXT`)
  - [agent/nodes.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/legal_workflow_generator/agent/nodes.py#L225) (`LLM_FAILURES`)
- **Issue**:
  - `_rag_pipeline`, `_DOMAIN_MAP`, and `_HAS_RAW_TEXT` are initialized lazily without locks (race condition on first concurrent request).
  - If Postgres is temporarily unreachable on startup, `_load_domain_map()` caches `{}` permanently for the lifetime of the process and never retries.
  - `LLM_FAILURES` is a process-wide counter that is never reset between requests in `app.py`.

### C. Fragile JSON Extraction in `_parse_json`
- **Where**: [legal_workflow_generator/agent/nodes.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/legal_workflow_generator/agent/nodes.py#L272)
- **Issue**: `json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group())` uses a greedy regex. If a plain-text LLM response contains multiple JSON objects or stray `{` / `}` characters in prose before/after the JSON, `re.search` matches from the first `{` to the last `}` and throws `JSONDecodeError`.

### D. Missing API Error & Input Guardrails in `app.py`
- **Where**: [app.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/app.py#L84-L215)
- **Issue**:
  - `/api/chat`, `/api/test/query`, and `/api/test/rag` have **no authentication, no rate limiting, and no max query length** on `ChatRequest.query` / `TestRequest.query`. A single huge or abusive request can trigger many LLM calls and exhaust API quotas.
  - `app.py` catches `Exception as e` and returns `detail=f"Pipeline error: {e}"` directly to the client, leaking internal stack/DB/provider error details.
  - `/api/test/query` and `/api/test/rag` do not validate `provider` or `top_k` with `RunSettings` the way `/api/chat` does.

---

## 3. Data & Retrieval Gaps

### A. Static Anchor & Prefix Fallbacks Coupled to Specific Provision IDs
- **Where**: [legal_workflow_generator/agent/nodes.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/legal_workflow_generator/agent/nodes.py#L281-L341) (`_PREFIX_FALLBACK`, `DOMAIN_ANCHORS`)
- **Issue**: Hardcoded regex triggers (`DOMAIN_ANCHORS`) inject specific `provision_id`s (e.g., `income_tax_act_1961_sec_195`, `copyright_act_1957_sec_17`) to compensate for provisions whose `text` column only contains a header. If the dataset is re-ingested with different IDs or expanded to new statutes, these anchors silently fail or become stale.

### B. `verify_citations` vs. `failed_citations`
- **Where**: [legal_workflow_generator/agent/graph.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/legal_workflow_generator/agent/graph.py#L68) & [agent/nodes.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/legal_workflow_generator/agent/nodes.py#L1385)
- **Issue**: `verify_citations` populates `failed_citations`, and `graph.py` unconditionally transitions `verify_citations -> grade_parallel`. Nothing in `graph.py` routes on `failed_citations` (e.g., to strip or force regeneration if a hallucinated citation slips past structured output). Because `domain_workflow_model` uses a `Literal` enum, this rarely triggers on Gemini, but if a provider ignores the enum constraint, `failed_citations` is only logged and never acted on by the graph router.

### C. Hardcoded `where` SQL Interpolation Pattern
- **Where**: [legal_workflow_generator/agent/nodes.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/legal_workflow_generator/agent/nodes.py#L384) (`_select_laws(where: str, params: tuple)`)
- **Issue**: Currently only called with hardcoded internal SQL strings, so it is not directly exploitable today, but exposing a helper that interpolates raw `where` strings into `f"SELECT {cols} FROM laws WHERE {where}"` is an anti-pattern that invites SQL injection if reused with user-influenced clauses later.

---

## 4. Testing & Eval Gaps

- **Core agent retrieval & graph integration is untested offline**: As noted in `README.md`, whole-package offline coverage is ~53%; `rag/hybrid_search.py`, `rag/bm25_retriever.py`, `rag/setup.py`, and `rag/embeddings.py` have minimal or no unit tests without a live Postgres instance.
- **No CI/CD workflow**: There is no `.github/workflows/` configuration to automatically run `uv run pytest` or linting on pull requests.
