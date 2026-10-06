## Project Layout

```
main.py                     CLI entrypoint for database setup / ingestion / embedding
legal_workflow_generator/   library code (query, rag, agent, llm, workflow units)
scripts/                    runnable helper scripts (agent queries, demos, validation)
evals/                      evaluation harnesses; evals/results/ holds saved runs
tests/                      unit and pipeline tests
datasets/                   source legal corpora
docs/                       architecture diagram and evaluation write-ups
```

All commands below are meant to be run from the repository root.

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

`.env` is loaded from the repository root by `legal_workflow_generator/config/values.py`,
so it applies identically to `main.py`, `scripts/`, `tests/` and `evals/` regardless of
the working directory. `PGPASSWORD` and `GEMINI_API_KEY` are **required and never
defaulted** — if either is missing or empty, the process raises
`MissingEnvironmentVariable` and exits before doing any work. Variables already exported
in your shell take precedence over the `.env` file.

Optional overrides (`PGDATABASE`, `PGUSER`, `PGHOST`, `PGPORT`, `GEMINI_MODEL`,
`GROQ_MODEL`, and `GROQ_API_KEY` for the groq provider) are listed in `.env.example`.
`GEMINI_MODEL` defaults to `models/gemini-3.1-flash-lite` and `GROQ_MODEL` to
`openai/gpt-oss-120b`.

- Start database
```
docker compose up
```

- Test if database is reachable
```bash
uv run pytest -m integration tests/test_conn.py
# quick standalone check without pytest:
python -m tests.test_conn
```

- Setup database
```bash
python main.py setup
```
- Ingest the dataset

To ingest CONDENSED version of the dataset
```bash
python main.py ingest
```
To ingest the COMPLETE version of the dataset
```bash
python main.py ingest complete
```

- Generate embeddings
```bash
python main.py embed
```

## LLM Layer (LangChain)

Every LLM call in the project goes through LangChain chat models built in
`legal_workflow_generator/llm/`; no module imports the `google-genai` or `groq`
SDKs directly (they are only pulled in transitively by `langchain-google-genai`
and `langchain-groq`).

| Helper | Purpose |
|---|---|
| `chat_model(temperature=None)` | shared, cached `ChatGoogleGenerativeAI` (Gemini) |
| `groq_chat_model(temperature=0.0)` | shared, cached `ChatGroq`; only used by the groq RAG provider |
| `structured_llm(schema, temperature=None)` | Gemini model wrapped with `with_structured_output(schema, method="json_schema", include_raw=True)` |
| `invoke_structured(llm, system, user)` | sends a system + user message, returns the parsed Pydantic object, raises `StructuredOutputError` if the reply doesn't validate |

Transient API errors (429/5xx) are retried by the provider client (`MAX_RETRIES = 5`).

### Structured outputs

Agent and query-unit calls never parse free text — each returns a Pydantic
model from `legal_workflow_generator/llm/schemas.py`:

| Schema | Used by | Fields |
|---|---|---|
| `IntentClassification` | `IntentClassifier.classify` | `reason`, `intent` (`qa`/`workflow`/`compliance_check`/`unknown`), `confidence` (0–1) |
| `DomainClassification` | `LegalContextResolver` (incl. self-consistency samples) | `domain` (5 domains or `unknown`), `keywords` (≤5) |
| `DomainList` | `classify_query` → `_llm_domains` | `domains` (all domains the query touches) |
| `SearchQuery` | `_focus_query`, `_rewrite_for_domain` | `query` |
| `RelevanceGrade` | `_grade_domain` | `reason`, `grade` (`sufficient`/`insufficient`) |
| `domain_workflow_model(ids)` | `_generate_domain` | `summary` (≤3), `steps[]` (`action`, `provision_id`, `applies_to`), `not_covered` |
| `GroundednessGrade` | `grade_groundedness` | `reason`, `unsupported_claims`, `grade` (`grounded`/`not_grounded`) |
| `AnswerabilityGrade` | `grade_answerability` | `reason`, `grade` (`answers`/`off_target`) |

Design notes:

- `reason` is declared before `grade` so the model justifies a verdict before committing to it.
- `domain_workflow_model(ids)` is built per call: `provision_id` is an enum of
  the provision ids actually retrieved for that domain, so the model cannot cite
  a provision it wasn't shown.
- Schemas stick to what Gemini's response schema handles reliably — `Literal`
  enums, lists, numeric/length bounds, required fields (no defaults or unions).
- A reply that fails validation raises `StructuredOutputError`. Graders map it to
  their negative grade with the reason `Could not parse grader response`, so a
  malformed reply is distinguishable in traces from a genuine verdict.

The Phase 2 answer generators in `rag/generator.py` (`GeminiAnswerGenerator`,
`GroqAnswerGenerator`, used by `RagPipeline.query` and the demo script) still
return free-form markdown, but also go through the LangChain chat models above.

## Run Chatbot UI
Run the fastapi server and frontend:

```bash
python app.py
```

## View Database Statistics
Run the following command to view database statistics.

```bash
python datasets/counter.py
```


## Running the tests

The suite lives in `tests/` and runs on [pytest](https://docs.pytest.org).
`pytest`, `pytest-cov` and the rest of the dev tooling are declared in the
`dev` dependency group, so `uv sync` installs them:

```bash
uv sync                 # includes the dev group by default
```

### Run the default (fast, offline) suite

These tests fake every network boundary (the LangChain chat models) and never touch Postgres, so
they need no `.env`, no database and no API keys — `tests/conftest.py` injects
dummy values for the required env vars at import time.

```bash
uv run pytest
```

### Run with coverage

```bash
uv run pytest --cov --cov-report=term-missing
# HTML report in htmlcov/
uv run pytest --cov --cov-report=html
```

Coverage is configured in `pyproject.toml` (`[tool.coverage.*]`) and is scoped
to the `legal_workflow_generator` package. The default (offline) suite covers
the query unit thoroughly — `query/normalizer.py`, `query/intent_classifier.py`
and `query/context_resolver.py` all sit at ~90–100% — plus the `llm/` layer,
`rag/generator.py` and the pure text helpers in `rag/ingestion.py`. In `agent/*`
only the structured-output LLM steps are unit-tested; retrieval and the full
graph are only exercised by the integration suite (they need the live database
and Gemini), so whole-package coverage is ~53% without it.

### Integration tests

Tests marked `@pytest.mark.integration` (`test_conn.py`, `test_rag_pipeline.py`,
`test_agent.py`) exercise the real stack and are **deselected by default**
(`addopts = -m 'not integration'`). They need a populated Postgres corpus with
embeddings (`main.py setup && main.py ingest && main.py embed`) and a real
`GEMINI_API_KEY`; each one `skip`s itself cleanly if those aren't available.

```bash
uv run pytest -m integration            # run only the integration tests
uv run pytest -m ''                     # run everything
```

### What's covered

| Test module | What it checks | Needs DB/API |
|---|---|---|
| `test_normalizer.py` | `QueryNormalizer` — cleanup, abbreviation expansion, validation, PDF extraction | no |
| `test_ingestion_text.py` | `rag.ingestion` text helpers (stemming, stopwords, query expansion) | no |
| `test_keyword_domain_classifier.py` | `KeywordDomainClassifier.classify` scoring / bigram matching / thresholds | no |
| `test_intent_classifier.py` | `IntentClassifier` structured-output mapping, low-confidence override, schema-mismatch / API error handling (LLM faked) | no |
| `test_context_resolver.py` | `LegalContextResolver` strategy + keyword/LLM reconciliation, self-consistency voting (both backends faked) | no |
| `test_query.py` | `process_query` end-to-end wiring (LLM faked) | no |
| `test_structured_output.py` | `llm/` schemas (provision-id enum, domain enum), `invoke_structured` error handling, and the agent's structured LLM steps (`_grade_domain`, `_rewrite_for_domain`, `_focus_query`, `_generate_domain`, graders, `_llm_domains`) | no |
| `test_generator.py` | `rag/generator.py` answer generators — message construction, empty-context and empty-response fallbacks (fake LangChain chat model) | no |
| `test_rag_retrieval_query.py` | `RagPipeline._build_retrieval_query` hint building | no |
| `test_conn.py` | Postgres connectivity | yes |
| `test_rag_pipeline.py` | full hybrid retrieval + grounded answer | yes |
| `test_agent.py` | LangGraph agentic pipeline, end to end | yes |

## Demo: Run Query Unit + RAG Unit Together (Custom Query)

Use this script to run the full flow with your own query:

```bash
python scripts/demo_query_rag.py \
	--query "What are the steps to comply with DPDP Act as a SaaS startup?" \
	--provider gemini \
	--top-k 3
```

Useful options:

```bash
# Run only query unit (normalization + intent + legal context)
python scripts/demo_query_rag.py --query "Need GST compliance checklist" --query-only

# Use Groq for answer generation
python scripts/demo_query_rag.py --query "How to register a private limited company in India?" --provider groq
```

You can also drive the query, RAG and demo units through the wrapper script:

```bash
./scripts/run_units.sh query
./scripts/run_units.sh rag --gemini
./scripts/run_units.sh demo --query "What are DPDP compliance steps for a SaaS startup?"
```



## Phase 3: Agentic RAG (LangGraph Self-Corrective Pipeline)

### Run the Agentic Pipeline

```bash
# Single query
python scripts/agent_query.py "What are my DPDP compliance obligations as a SaaS startup?"

# Multi-domain query
python scripts/agent_query.py "We have 20 employees including women, export software, and collect user data — what are all our compliance obligations?"
```

### Run Evaluation

```bash
# Phase 3 agentic eval (49 queries)
python evals/eval_agent.py

# Phase 2 baseline eval for comparison (49 queries)
python evals/eval_phase2.py

# Domain classification eval — isolates LegalContextResolver (51 queries)
python evals/eval_domain_classification.py

# Complete-dataset multi-domain agentic eval (100 queries)
python evals/eval_complete_multidomain.py
```

All four scripts write their timestamped JSON/CSV output to `evals/results/`.

#### Complete-dataset multi-domain eval

`evals/eval_complete_multidomain.py` runs the full LangGraph agent against the
**complete** corpus (`python main.py ingest complete`, ~4.9k provisions across
52 statutes). The older `eval_agent.py` set only covers the condensed corpus,
with one domain per query. Every expected citation in this set is a real
provision id from `datasets/complete/`, and most come from statutes that exist
only in the complete corpus (Income-tax, CGST, SEZ, Stamp, FEMA, LLP, Contract,
Consumer Protection, IBC, Patents, Trademarks, Designs, CERT-In, SPDI, Aadhaar,
the labour codes, and others).

| Category | Queries | What it tests |
|---|---|---|
| `single` | 50 | 10 per domain |
| `multi_2` | 20 | two-domain queries (e.g. data protection + employment) |
| `multi_3` | 12 | three-domain queries |
| `multi_4` | 5 | four-domain queries |
| `multi_5` | 3 | queries touching all five domains |
| `edge` | 10 | non-existent sections, out-of-jurisdiction, speculative and off-topic queries; the agent should abstain |

Metrics, reported overall, per category and per domain:

| Metric | What it tells you |
|---|---|
| `domain_recall` / `domain_precision` / `domain_exact_match` | detected `all_domains` vs. the labeled domain set |
| `primary_domain_accuracy` | the resolver's single domain is one of the labeled domains |
| `citation_recall` | verified citations ∩ expected provisions |
| `citation_domain_coverage` | share of labeled domains with at least one verified citation from that domain's statutes |
| `false_abstain_rate` / `abstain_accuracy` | wrongful abstention on answerable queries / correct abstention on edge cases |

```bash
python evals/eval_complete_multidomain.py --validate          # offline: check expected ids exist in datasets/complete/
python evals/eval_complete_multidomain.py --category multi    # only the 40 multi-domain queries
python evals/eval_complete_multidomain.py --limit 10 --sleep 2
```

#### Domain classification eval

`evals/eval_domain_classification.py` runs `LegalContextResolver` in isolation
(no retrieval, no answer generation) against a hand-labeled set of 51 queries —
8 per domain, 6 boundary queries that plausibly touch two domains, and 5
off-topic queries expected to resolve to `unknown`. It's the ground-truth check
for the two correctness signals the resolver produces on every call:

- **`domain_agreement`** — does the LLM's chosen domain match the free,
  always-computed rule-based (keyword) domain? Disagreement costs nothing to
  detect and is a hint the query may be misclassified.
- **`domain_confidence`** — with self-consistency on, the domain prompt is
  sampled N times at `temperature=0.7` and majority-voted; this is the winning
  vote share (e.g. `0.67` for a 2-of-3 split). Low confidence means the LLM
  itself isn't stable on that query.

Neither signal is useful unless it actually predicts wrongness, so the script
reports, beyond plain accuracy:

| Metric | What it tells you |
|---|---|
| `overall_accuracy` | LLM (or self-consistency majority) domain vs. the labeled domain |
| `rule_based_only_accuracy` | accuracy of the keyword classifier alone, for comparison |
| `agreement_rate` | how often the LLM and rule-based classifier agree |
| `accuracy_when_llm_rule_based_agree` / `..._disagree` | does disagreement actually correlate with being wrong? |
| `accuracy_when_unanimous_vote` / `..._split_vote` | does a split self-consistency vote actually correlate with being wrong? |
| `per_domain_metrics` | precision/recall/F1 per domain |
| `confusion_matrix` | expected domain → predicted domain counts |

Self-consistency is on by default here (3 samples/query) since that's what
produces a non-trivial `domain_confidence` to evaluate — pass
`--no-self-consistency` for a single Gemini call per query (cheaper, but
`domain_confidence` degenerates to 1.0/0.0), or `--samples N` to change the
vote size. This is separate from the `SELF_CONSISTENCY_ENABLED` env var, which
controls the default for the resolver everywhere else (e.g. inside the agent).

### Architecture

The agent is a LangGraph `StateGraph` (`agent/graph.py`, nodes in `agent/nodes.py`):

```
classify_query ──(unknown / confidence < 0.5)──────────────► abstain
     │
     ▼
domain_pipeline ──(no domain produced cited steps)─────────► abstain
     │
     ▼
verify_citations
     │
     ▼
grade_parallel ──(failed grade, retries left)──► regenerate ──► domain_pipeline
     │        └─(failed grade, retries exhausted)─────────► abstain
     ▼
stitch_answer ──► END
```

1. **classify_query** — normalizer + `IntentClassifier` + `LegalContextResolver`,
   then multi-domain detection: the union of retrieval-based detection (which
   statutes appear in the top hits), an LLM call (`DomainList`) and the
   resolver's domain.
2. **domain_pipeline** — one worker per detected domain, run in parallel:
   - *focus* (multi-domain queries only): rewrite the query to cover just this domain (`SearchQuery`)
   - *retrieve*: hybrid BM25 + semantic search, filtered to this domain's provisions (Act sections first, then rules)
   - *grade*: is this context sufficient for this domain's part of the query? (`RelevanceGrade`)
   - *rewrite → retrieve → grade* on failure (`MAX_DOMAIN_RETRIES = 1`, `SearchQuery`)
   - *generate*: summary, compliance steps each tied to one retrieved `provision_id`, and what isn't covered (`domain_workflow_model`)

   The per-domain results are stitched into a single workflow document
   (SUMMARY / ACTION CHECKLIST / SOURCES).
3. **verify_citations** — string-match check that every cited provision was retrieved.
4. **grade_parallel** — runs **grade_groundedness** (`GroundednessGrade`) and
   **grade_answerability** (`AnswerabilityGrade`) concurrently.
5. **regenerate** — re-runs `domain_pipeline` once after a failed grade (`MAX_GENERATION_RETRIES = 1`).
6. **stitch_answer** — flags any detected domain that ended up with no verified citations as a knowledge gap.
7. **abstain** — honest refusal with the reason (low confidence, insufficient context, or a failed grade).

### Key Results

| Metric | Phase 2 Static | Phase 3 Agentic |
|---|---|---|
| Citation hallucination | 1.12 avg per query | 0 |
| Citation recall@k | 0.61 | 0.77 |
| Correct abstain on impossible queries | 0% | 100% |
| Answer rate | 100% (never abstains) | 93.9% |
| Multi-domain support | No | Yes (up to 5 domains) |

See `docs/PHASE3_EVAL_RESULTS.md` for full evaluation details.
