## Installation

- Install uv and other packages
```
pip install uv
uv sync
```

- Setup env file
```
cp .env.example .env
# Edit .env file with correct values
```

- Start database
```
docker compose up
```

- Setup database
```bash
python main.py setup
```
- Ingest the dataset
```bash
python main.py ingest
```

- Generate embeddings
```bash
python main.py embed
```

- Run end-to-end retrieval + Gemini answer generation test
```bash
python tests/test_rag_pipeline.py
```

## Test Query Processing Unit

Run the standalone query processor test:

```bash
python -m tests.test_query
```

## Test RAG Unit

Run the standalone RAG pipeline test:

```bash
python -m tests.test_rag_pipeline --gemini
# or
python -m tests.test_rag_pipeline --llama
```

## Demo: Run Query Unit + RAG Unit Together (Custom Query)

Use this script to run the full flow with your own query:

```bash
python demo_query_rag.py \
	--query "What are the steps to comply with DPDP Act as a SaaS startup?" \
	--provider gemini \
	--top-k 3
```

Useful options:

```bash
# Run only query unit (normalization + intent + legal context)
python demo_query_rag.py --query "Need GST compliance checklist" --query-only

# Use Groq for answer generation
python demo_query_rag.py --query "How to register a private limited company in India?" --provider groq
```



## Phase 3: Agentic RAG (LangGraph Self-Corrective Pipeline)

### Run the Agentic Pipeline

```bash
# Single query
python agent_query.py "What are my DPDP compliance obligations as a SaaS startup?"

# Multi-domain query
python agent_query.py "We have 20 employees including women, export software, and collect user data — what are all our compliance obligations?"
```

### Run Evaluation

```bash
# Phase 3 agentic eval (49 queries)
python eval_agent.py

# Phase 2 baseline eval for comparison (49 queries)
python eval_phase2.py
```

### Architecture

The agentic pipeline consists of 7 LangGraph nodes:

1. **classify_query** — intent + domain detection (multi-domain supported)
2. **retrieve** — hybrid BM25 + semantic search across detected domains
3. **grade_context** — LLM grader: is retrieved context sufficient?
4. **rewrite_query** — query rewriting on failure (max 3 retries)
5. **generate** — structured workflow generation with Act/Section citations
6. **verify_citations** — string-match verification of all cited sections
7. **grade_groundedness** — LLM grader: are claims grounded in retrieved docs?
8. **grade_answerability** — LLM grader: does answer resolve the query?
9. **abstain** — honest refusal when confidence is too low

### Key Results

| Metric | Phase 2 Static | Phase 3 Agentic |
|---|---|---|
| Citation hallucination | 1.12 avg per query | 0 |
| Citation recall@k | 0.61 | 0.77 |
| Correct abstain on impossible queries | 0% | 100% |
| Answer rate | 100% (never abstains) | 93.9% |
| Multi-domain support | No | Yes (up to 5 domains) |

See `PHASE3_EVAL_RESULTS.md` for full evaluation details.