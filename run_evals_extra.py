#!/usr/bin/env python3
"""
Follow-up evaluation for the paper (run after run_evals_core.py has finished).

  5. Repeat of the pipeline with LLM-suggested provisions ON (stability) -> eval_pipeline_suggest_r2.jsonl
  6. Step-level entailment for every system, judged by an independent model (JUDGE_MODEL, default
     models/gemini-3.5-flash; the generator and the pipeline's own checker use gemini-3.1-flash-lite):
       - regenerate LLM-only and hybrid RAG answers WITH their steps saved (the archived runs kept only ids)
       - export the pipeline's final steps and its pre-verification steps from eval_pipeline_suggest.jsonl
       - judge every (step, cited provision) pair; write entailment_report.md and audit_sample.csv

Results go to evals/results/gemini_v2/. Safe to stop and re-run; see run_evals_core.py.

    python run_evals_extra.py
"""
import sys

from evals.runner import RESULTS, Step, harness, run_all

PIPE = "evals/eval_complete_multidomain.py"
ENT = "evals/eval_entailment.py"


def pipeline_complete(tag: str) -> bool:
    import json
    p = RESULTS / f"eval_{tag}_summary.json"
    return p.exists() and json.loads(p.read_text()).get("summary", {}).get("complete", False)


STEPS = [
    Step("pipeline_suggest_r2", harness(PIPE, "--provider", "gemini", "--tag", "pipeline_suggest_r2", "--resume",
                                        "--sleep", "0", "--option", "suggest_provisions=true")),
    Step("steps_llm_only", harness(ENT, "gen", "--system", "llm_only", "--resume", "--sleep", "1")),
    Step("steps_rag_hybrid", harness(ENT, "gen", "--system", "rag_hybrid", "--resume", "--sleep", "1")),
    Step("steps_rag_hybrid_abstain", harness(ENT, "gen", "--system", "rag_hybrid_abstain", "--resume", "--sleep", "1")),
    Step("steps_pipeline", harness(ENT, "export", "--tag", "pipeline_suggest"), llm=False),
    Step("judge_llm_only", harness(ENT, "judge", "--system", "llm_only", "--resume", "--sleep", "1")),
    Step("judge_rag_hybrid", harness(ENT, "judge", "--system", "rag_hybrid", "--resume", "--sleep", "1")),
    Step("judge_rag_hybrid_abstain", harness(ENT, "judge", "--system", "rag_hybrid_abstain", "--resume", "--sleep", "1")),
    Step("judge_full_final", harness(ENT, "judge", "--system", "full_final", "--resume", "--sleep", "1"),
         needs="steps_full_final.jsonl"),
    Step("judge_full_pre", harness(ENT, "judge", "--system", "full_pre", "--resume", "--sleep", "1"),
         needs="steps_full_pre.jsonl"),
    Step("entailment_report", harness(ENT, "report"), llm=False),
]

if __name__ == "__main__":
    if not pipeline_complete("pipeline_suggest"):
        sys.exit("run_evals_core.py has not finished the pipeline_suggest run yet "
                 "(evals/results/gemini_v2/eval_pipeline_suggest_summary.json is missing or incomplete). "
                 "Finish the core run first.")
    sys.exit(run_all("EXTRA evaluation (steps 5-6)", STEPS))
