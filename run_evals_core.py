#!/usr/bin/env python3
"""
Core evaluation for the paper (Gemini, gemini-3.1-flash-lite, temperature 0, query date 2026-10-15).

  1. Pipeline, current code, LLM-suggested provisions OFF   -> eval_pipeline_base.jsonl
  2. Pipeline, LLM-suggested provisions ON                   -> eval_pipeline_suggest.jsonl
  3. Single-pass hybrid RAG without abstention               -> baseline_rag_hybrid.jsonl
  4. Retrieval only, no LLM (baseline retrievers and the pipeline's own per-domain retriever)
  The archived baselines (LLM only x2, BM25 RAG, hybrid RAG with abstention) are reused as they are.

Results go to evals/results/gemini_v2/ (report.md is rebuilt at the end of every session).

Safe to stop and re-run: every finished query is saved at once, and a re-run skips saved queries. When all
Gemini keys fail (daily quota, billing) the run stops cleanly; run this script again the next day.

    python run_evals_core.py
"""
import sys

from evals.runner import Step, copy_archived_baselines, harness, run_all

PIPE = "evals/eval_complete_multidomain.py"
BASE = "evals/eval_baselines.py"

STEPS = [
    # no LLM calls
    Step("retrieval_baselines", harness(BASE, "--mode", "retrieval"), llm=False,
         output="baseline_retrieval_metrics.json"),
    Step("retrieval_worker_global", harness("evals/eval_worker_retrieval.py", "--tag", "global"), llm=False,
         output="worker_retrieval_global.json"),
    Step("retrieval_worker_domain", harness("evals/eval_worker_retrieval.py", "--tag", "domain_search",
                                            "--option", "domain_search=domain"), llm=False,
         output="worker_retrieval_domain_search.json"),
    # ~100 calls
    Step("rag_hybrid", harness(BASE, "--mode", "rag_hybrid", "--provider", "gemini", "--resume", "--sleep", "1")),
    # ~1,000 calls each
    Step("pipeline_base", harness(PIPE, "--provider", "gemini", "--tag", "pipeline_base", "--resume",
                                  "--sleep", "0", "--option", "suggest_provisions=false")),
    Step("pipeline_suggest", harness(PIPE, "--provider", "gemini", "--tag", "pipeline_suggest", "--resume",
                                     "--sleep", "0", "--option", "suggest_provisions=true")),
]

if __name__ == "__main__":
    copy_archived_baselines()
    sys.exit(run_all("CORE evaluation (steps 1-4)", STEPS))
