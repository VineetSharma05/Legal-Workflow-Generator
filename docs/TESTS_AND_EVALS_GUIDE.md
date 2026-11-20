# Tests & Evaluations Guide

This document explains every test module in `tests/` and every evaluation harness/runner in `evals/` (plus the root `run_evals_*.py` scripts), what each one does, and how to run them.

---

## 1. Test Suite (`tests/`)

The test suite uses **pytest** (configured in [pyproject.toml](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/pyproject.toml)). By default, `@pytest.mark.integration` tests are **deselected** so `uv run pytest` runs fast and completely offline without Postgres or API keys ([conftest.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/tests/conftest.py) injects dummy env vars at import time).

### Quick Commands
```bash
# Run all fast, offline unit tests (default)
uv run pytest

# Run with coverage report
uv run pytest --cov --cov-report=term-missing

# Run ONLY integration tests (requires running Postgres + embeddings + GEMINI_API_KEY)
uv run pytest -m integration

# Run everything (unit + integration)
uv run pytest -m ""

# Run a single test file
uv run pytest tests/test_workflow_units.py
```

### Offline Unit & Pipeline Tests (No DB / No API Keys)

| File | What It Tests |
|---|---|
| [conftest.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/tests/conftest.py) | Pytest setup; injects dummy `PGPASSWORD` and `GEMINI_API_KEY` so modules can be imported offline. |
| [test_normalizer.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/tests/test_normalizer.py) | `QueryNormalizer`: text cleanup, abbreviation expansion, validation, and PDF text extraction (uses `tests/test_legal.pdf`). |
| [test_intent_classifier.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/tests/test_intent_classifier.py) | `IntentClassifier`: structured-output mapping, low-confidence override, schema-mismatch and API error handling. |
| [test_keyword_domain_classifier.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/tests/test_keyword_domain_classifier.py) | `KeywordDomainClassifier`: TF-IDF scoring, bigram matching, and threshold logic. |
| [test_context_resolver.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/tests/test_context_resolver.py) | `LegalContextResolver`: `llm_fallback` vs. `combine` strategies, keyword/LLM reconciliation, and self-consistency voting. |
| [test_query.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/tests/test_query.py) | `process_query` end-to-end query-unit wiring with faked LLM. |
| [test_structured_output.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/tests/test_structured_output.py) | `llm/` Pydantic schemas (including dynamic `domain_workflow_model`), `invoke_structured` error handling, and agent structured LLM steps (`_grade_domain`, `_rewrite_for_domain`, `_generate_domain`, graders, `_llm_domains`). |
| [test_generator.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/tests/test_generator.py) | `rag/generator.py` (`GeminiAnswerGenerator`, `GroqAnswerGenerator`): message construction, empty-context and empty-response fallbacks. |
| [test_ingestion_text.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/tests/test_ingestion_text.py) | Pure text helpers in `rag/ingestion.py` (stemming, stopword removal, section marker stripping, query expansion). |
| [test_rag_retrieval_query.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/tests/test_rag_retrieval_query.py) | `RagPipeline._build_retrieval_query` hint construction from `LegalContext`. |
| [test_gemini_client.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/tests/test_gemini_client.py) | `RotatingGeminiClient` and `KeyPool`: key rotation, cooldowns, usage tracking, and default temperature injection. |
| [test_workflow_units.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/tests/test_workflow_units.py) | Comprehensive unit tests for `agent/nodes.py` and `agent/settings.py`: `RunSettings`, `Provision`/`Step`/`DomainResult`, temporal filtering, anchors, scope checks, context selection, claim verification, and stitching. |
| [test_workflow_pipeline.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/tests/test_workflow_pipeline.py) | Full LangGraph state transitions (`classify_query` → `domain_pipeline` → `verify_citations` → `grade_parallel` → `stitch_answer` / `abstain`) with faked retrieval and LLM calls. |
| [test_app_routes.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/tests/test_app_routes.py) | FastAPI routes in `app.py` (`/api/chat`, `/api/test/query`, `/api/test/rag`), request validation, and response shapes. |
| [test_complete_dataset.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/tests/test_complete_dataset.py) | Validates structure and integrity of `datasets/complete/combined_complete_dataset.json`. |
| [test_eval_runner.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/tests/test_eval_runner.py) | Tests `evals/runner.py` step orchestration and stop-rule behavior. |
| [test_eval_resume.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/tests/test_eval_resume.py) | Tests resumable JSONL checkpointing in evaluation harnesses. |
| [test_expert_review.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/tests/test_expert_review.py) | Tests blinded review sheet generation and Cohen's linear-weighted kappa scoring in `evals/expert_review/`. |

### Integration & Browser Tests

| File | What It Tests | Requirements |
|---|---|---|
| [test_conn.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/tests/test_conn.py) | Postgres connectivity check (`@pytest.mark.integration`). Can also run standalone: `python -m tests.test_conn`. | Live Postgres |
| [test_rag_pipeline.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/tests/test_rag_pipeline.py) | Full hybrid retrieval + grounded answer generation (`@pytest.mark.integration`). | Live Postgres + embeddings + Gemini API |
| [test_agent.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/tests/test_agent.py) | End-to-end LangGraph agentic pipeline (`@pytest.mark.integration`). | Live Postgres + embeddings + Gemini API |
| [tests/browser/test_workflow_inspector.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/tests/browser/test_workflow_inspector.py) | Browser UI verification for the FastAPI chat & workflow inspector frontend. | Running app / browser tooling |

---

## 2. Evaluation Suite (`evals/` & Root Runners)

Evaluations measure retrieval recall, citation hallucination, abstention accuracy, multi-domain coverage, and step-level entailment. Results are saved to `evals/results/` (see [evals/results/RUNS.md](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/evals/results/RUNS.md) for the index of recorded runs).

### Orchestrated Paper Runs (Recommended)
Both root scripts are **resumable** (completed queries are saved immediately to JSONL and skipped on re-run) and stop cleanly on daily quota exhaustion:
```bash
# Steps 1–4: retrieval baselines, hybrid RAG baseline, full pipeline (base & suggest)
python run_evals_core.py

# Steps 5–6: stability repeat + independent-judge step-level entailment
python run_evals_extra.py
```

### Evaluation Modules

| File | Purpose | Example Usage |
|---|---|---|
| [evals/runner.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/evals/runner.py) | Shared step runner (`Step`, `harness`, `run_all`, `ensure_database`, `Lock`) used by `run_evals_core.py` and `run_evals_extra.py`. | Imported by root runners |
| [evals/run_status.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/evals/run_status.py) | Shared exit codes, hard-stop detection (daily quota/billing), and `FailureTracker`. | Imported by harnesses |
| [evals/eval_complete_multidomain.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/evals/eval_complete_multidomain.py) | **Primary 100-query end-to-end eval** (90 answerable + 10 out-of-scope) against the complete dataset. Supports `--provider`, `--tag`, `--resume`, `--option key=val`. | `python evals/eval_complete_multidomain.py --provider gemini --tag pipeline_v4 --resume` |
| [evals/eval_baselines.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/evals/eval_baselines.py) | Runs retrieval-only metrics (`--mode retrieval`), BM25 RAG, hybrid RAG, and LLM-only baselines on the same 100 queries. | `python evals/eval_baselines.py --mode retrieval` |
| [evals/eval_llm_pure.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/evals/eval_llm_pure.py) | Pure LLM baseline (no retrieval and no corpus provision IDs given to the model). | `python evals/eval_llm_pure.py --resume` |
| [evals/eval_worker_retrieval.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/evals/eval_worker_retrieval.py) | Retrieval-only evaluation of the agent's own per-domain retriever (0 LLM calls). | `python evals/eval_worker_retrieval.py --tag domain_search --option domain_search=domain` |
| [evals/eval_entailment.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/evals/eval_entailment.py) | Step-level entailment evaluation: `gen` (regenerate baseline steps), `export` (export pipeline steps), `judge` (independent LLM judge), `report` (generate markdown/CSV). | `python evals/eval_entailment.py report` |
| [evals/replay_generation.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/evals/replay_generation.py) | Replays **only** generation + claim verification on saved retrieval results to iterate on prompts/models cheaply. | `python evals/replay_generation.py --help` |
| [evals/make_report.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/evals/make_report.py) | Aggregates saved JSONL runs in `evals/results/` into summary tables (`report.md` / `report.json`) with bootstrap confidence intervals (0 API calls). | `python evals/make_report.py` |
| [evals/eval_domain_classification.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/evals/eval_domain_classification.py) | Isolates `LegalContextResolver` on 51 hand-labeled queries; measures accuracy, keyword/LLM agreement, self-consistency calibration, and confusion matrix. | `python evals/eval_domain_classification.py --samples 3` |
| [evals/eval_agent.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/evals/eval_agent.py) | Earlier Phase 3 agentic eval (49 queries). | `python evals/eval_agent.py` |
| [evals/eval_phase2.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/evals/eval_phase2.py) | Phase 2 static RAG baseline eval (49 queries). | `python evals/eval_phase2.py` |
| [evals/eval_pipeline.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/evals/eval_pipeline.py) | 50-query pipeline eval harness. | `python evals/eval_pipeline.py` |
| [evals/eval_pipeline_10query.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/evals/eval_pipeline_10query.py) | Quick 10-query smoke eval script. | `python evals/eval_pipeline_10query.py` |
| [evals/expert_review/build_review_sheet.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/evals/expert_review/build_review_sheet.py) | Builds blinded CSV review sheets from evaluation runs for human legal experts (see [RUBRIC.md](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/evals/expert_review/RUBRIC.md)). | `python evals/expert_review/build_review_sheet.py --help` |
| [evals/expert_review/score_reviews.py](file:///Users/yathartha/Desktop/Legal-Workflow-Generator/evals/expert_review/score_reviews.py) | Aggregates filled expert-review CSVs and computes inter-rater agreement (linear-weighted kappa). | `python evals/expert_review/score_reviews.py --help` |
