# Evaluation runs

All runs: 100 test queries (90 answerable, 10 that should be refused), generator `models/gemini-3.1-flash-lite`,
as-of date 2026-10-15. Recall = share of the expected provisions cited, averaged over the 90 answerable queries.
Faithfulness judge: `models/gemini-3.5-flash`.

Only runs at **temperature 0** are comparable with each other and with the baselines.

| Run (file stem) | Folder | Temperature | Pipeline | Recall | Out-of-scope refused | Answerable refused |
|---|---|---|---|---|---|---|
| `eval_pipeline_v5_t0` | `gemini_v5/` | 0 | v5: v4 + overlap snippets, verify_chars=4000, domain guide, none_abstain, strict_grade | **0.816** | 10/10 | 2 |
| `eval_fresh_v5_t0` | `gemini_v5/` | 0 | v5 on the 50 held-out queries. **Not held-out for v5**: v5 was designed after studying v4's failures on them | 0.858 | 9/10 | 0 |
| `eval_dev_v5` / `eval_dev_v4_rerun` | `gemini_v5/` | 0 | v5 and a same-day v4 rerun on the 64 dev queries (`evals/dev_queries.txt`) | 0.754 / 0.703 | - | 2 / 1 |
| `eval_pipeline_v4_t0` | `gemini_v4/` | 0 | v4: v3 + context selector | 0.776 | 10/10 | 2 |
| `eval_fresh_v4_t0` | `gemini_v4/` | 0 | v4 frozen, **50 held-out queries** (`evals/fresh_queries.py`, `--test-set fresh`; 40 answerable) | 0.793 | 5/10 | 1 |
| `eval_pipeline_v4_t02` | `gemini_v4/` | 0.2 | v4, same code as `_t0` (stability check) | 0.787 | 10/10 | 0 |
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
- `gemini_v5/eval_pipeline_v5_t0.before_intentfix.jsonl`: v5 before the guided-domain override was limited to
  questions the intent classifier accepted as legal; two out-of-scope queries ("recommend a cloud hosting
  provider", "write me a poem about corporate law") were answered. Only those two rows were re-run.
- The `citation_recall` in each run's `_summary.json` is a different average from the Recall column here
  (v4_t0 0.748, v5_t0 0.785); the Recall column is the per-query mean that `make_heldout_table.py` reports.
- `gemini_v2/eval_pipeline_v4.before_edgefix.jsonl`: v4 before the 10 out-of-scope queries were re-run with the
  foreign-law rule.
