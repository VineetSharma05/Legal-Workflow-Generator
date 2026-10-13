# Evaluation status (current)

This page is the source of truth for **what has been measured on which corpus**. Older write-ups
(`PHASE3_EVAL_RESULTS.md`, `DOMAIN_CLASSIFICATION_RELIABILITY.md`) describe a 196-provision prototype
and are historical.

## Corpus

| Corpus | File | Provisions |
|---|---|---|
| Complete (current benchmark corpus) | `datasets/complete/combined_complete_dataset.json` | **4,888** |
| Condensed | `datasets/condensed/combined_condensed_dataset.json` | 206 |
| Prototype (historical) | — | 196 |

`eval_pipeline.py` refuses to start unless the complete corpus file has 4,888 provisions
(`--expect-provisions` overrides this). Its manifest records the file's SHA-256 and the row count of the
live `laws` table, so results can be traced to the exact data.

About 45% of complete-corpus provisions (2,197 of 4,888) have no `effective_date`. Those are treated as
in force on every date. Temporal rules are only defined for the Income-tax Act, 1961 (see `TEMPORAL_RULES`
in `agent/nodes.py`).

## Benchmark (50-query gold set, `eval_pipeline.py`)

- Existing `eval_results.jsonl`: **48 valid rows of 50**. One row is a connection error, and one case was never run.
  These rows **predate** the current code and carry no provenance, so they must not be quoted as current results.
- **Re-run pending.** Requires Postgres with the complete corpus ingested and embedded, plus a Gemini key.
  Expect well over an hour at the default 12 requests per minute.

```bash
python main.py setup && python main.py ingest complete && python main.py embed
python eval_pipeline.py --fresh --provider gemini --top-k 8 --out evals/results/eval_4888_<date>.jsonl
python eval_pipeline.py --summary --out evals/results/eval_4888_<date>.jsonl
```

Each run writes `<out>.manifest.json` (commit, settings, corpus hash and counts). Incomplete cases, where an
LLM call was unavailable, are written as errors and retried on the next run. They are never scored as answers.

## Expert review (workflow correctness and usefulness)

- Rubric: `evals/expert_review/RUBRIC.md`.
- Build blinded review sheets: `python evals/expert_review/build_review_sheet.py --results <jsonl> --out review/`
- Score and aggregate: `python evals/expert_review/score_reviews.py review/reviewer_*.csv --out review/summary.md`
- **No expert scores exist yet.** The harness is tested, but the results are pending a qualified reviewer.

## Browser tests

`pytest -m browser tests/browser` drives the chat UI and workflow inspector in Chromium. `/api/chat` is
mocked, so no database or key is needed. It needs Playwright and a Chromium build.
