# Evaluation runs

All runs: 100 test queries (90 answerable, 10 that should be refused), generator `models/gemini-3.1-flash-lite`,
as-of date 2026-10-15. Recall = share of the expected provisions cited, averaged over the 90 answerable queries.
Faithfulness judge: `models/gemini-3.5-flash`.

Only runs at **temperature 0** are comparable with each other and with the baselines.

| Run (file stem) | Folder | Temperature | Pipeline | Recall | Out-of-scope refused | Answerable refused |
|---|---|---|---|---|---|---|
| `eval_pipeline_v4_t0` | `gemini_v4/` | 0 | v4: v3 + context selector | **0.776** | 10/10 | 2 |
| `eval_pipeline_v4` | `gemini_v2/` | API default | v4 (89 answerable queries ran before the foreign-law rule) | 0.809 | 10/10 | 0 |
| `eval_pipeline_v3` | `gemini_v2/` | API default | phase 3: prompt rules, statute search, original wording, scope check | 0.651 | 10/10 | 1 |
| `eval_pipeline_suggest` / `_r2` | `gemini_v2/` | 0 | planner suggestions on | 0.647 | 8/10 | 2 |
| `eval_pipeline_base` | `gemini_v2/` | 0 | starting pipeline | 0.603 | 9/10 | 3 |

Other folders and files:

- `dev/`: three runs on a 64-query development set (54 queries v3 scored below 1.0 plus 10 random controls),
  default temperature: `dev_base` (v3 rerun, 0.599), `dev_select` (selector v1, 0.611), `dev_select2`
  (selector v2, 0.720).
- `devjudge_dev_base/`, `devjudge_dev_select2/`: faithfulness of the two dev runs judged by flash-lite. Internal
  comparison only (the judge is the generator's own model).
- `gemini_v2/v3_default_temp/`, `gemini_v4/v4_default_temp/`: partial gemini-3.5-flash judging of v3 (54 queries)
  and v4 (53 queries) at the default temperature. Superseded; to be redone on `eval_pipeline_v4_t0`.
- `gemini_v4/judged_*` and `steps_*` for `llm_only`, `rag_hybrid`, `rag_hybrid_abstain`: copies of the baseline
  judgements from `gemini_v2/`, so the entailment report can compare every system from one folder.
- `gemini_v2/judged_rag_hybrid_abstain.jsonl`: rows 70-100 were judged at the default temperature; the first 69
  at temperature 0.
- `gemini_v2/eval_pipeline_v4.before_edgefix.jsonl`: v4 before the 10 out-of-scope queries were re-run with the
  foreign-law rule.
