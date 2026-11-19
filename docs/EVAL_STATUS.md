# Evaluation status (current)

This page records what has been measured, on which corpus, and what is still open. Older write-ups
(`PHASE3_EVAL_RESULTS.md`, `DOMAIN_CLASSIFICATION_RELIABILITY.md`) describe a 196-provision prototype
and are historical.

## Corpus

| Corpus | File | Provisions |
|---|---|---|
| Complete (benchmark corpus) | `datasets/complete/combined_complete_dataset.json` | **4,888** |
| Condensed | `datasets/condensed/combined_condensed_dataset.json` | 206 |
| Prototype (historical) | not in repo | 196 |

The benchmark run log reports `BM25 index built with 4888 laws`. The database row count has not yet been
checked with SQL (see open items).

About 45% of complete-corpus provisions (2,197 of 4,888) have no `effective_date`. Those are treated as
in force on every date. Temporal rules are defined only for the Income-tax Act, 1961, and they add a note
to the answer. They never exclude a provision.

## Benchmark (50-query gold set)

The 50 queries were run in two sessions because the Gemini free-tier daily quota (500 requests per day)
ran out partway through the first session.

- **Session 1:** S01–S10, T01–T08, H01–H05, F01–F03, V03 and R01 completed. The remaining cases hit quota
  failures. Those rows were discarded.
- **Session 2:** the remaining R, A and D cases were run after the fix described below.

Results were combined from the per-case lines. R01 appears in both sessions and is counted once.

### Combined results (50 queries)

| Result | Count | Share |
|---|---|---|
| Fully correct (answered, all expected provisions cited, domain recall 1.00) | 26 of 40 answerable | 65% |
| Partly correct (answered, some expected provisions missing) | 10 of 40 | 25% |
| Wrong abstain (refused an answerable query) | 4 of 40 | 10% |
| Correct abstain (refused an out-of-scope query) | 10 of 10 | 100% |
| Overall correct decisions | 36 of 50 | 72% |
| Expected provisions cited | 65 of 82 | 79% |

Domain recall was 1.00 on every valid answerable case. Domain precision was not combined across sessions.

### Failures

- Partly correct: T01, T03, T06, T07, H02, H03, H05, F01, V01, V02.
- Wrong abstain: S05, R02, R03, R06. R02, R03 and R06 cite nothing, and the summary's `false_abstain` count matches.
- Expected provisions not cited: `llp_act_2008` (S05), `income_tax_act_` (T01, T06, H05), `cgst_act_2017_sec_` (T03),
  `copyright_act_1957_sec_` (T07), `esia_1948_sec_` and `ca_2013_sec_` (H02), `ca_2013_sec_` (H03, F01, R03), `epf_1952_sec_`
  (V01, V02, R02), `copyright_act_1957_sec_17` (V02), `it_intermediary_guidelines_2021` (R06).

## Fix applied after the first session

Provider failures were logged and swallowed, so a quota error looked like an abstention. The runner now
records every final provider failure, treats the case as incomplete, and stops the run on a per-day quota.
Covered by `tests/test_eval_runner.py`.

## Not yet measured

- **Claim-verification accuracy.** The verifier's verdicts are recorded, but they have not been compared
  with human judgements. This needs a labelled set of step–provision pairs, which will come from expert review.
- **Temporal correctness.** No query in the gold set tests an effective-date or supersession case.
- **Practical usefulness.** Requires expert review (see below).
- **Stability.** Only one combined run exists. Repeat runs are needed before quoting the numbers as stable.

## Open items

1. **Statute check.** Run `SELECT DISTINCT statute_id FROM laws ORDER BY 1;`. Missing `llp_act_2008`, `ca_2013`,
   `epf_1952`, `esia_1948`, `it_intermediary_guidelines_2021` or `income_tax_act` would make those misses
   gold-label or corpus gaps, not retrieval failures.
2. **Income-tax misses.** Check whether the gold set expects 1961 Act IDs or 2025 Act IDs, and whether either is loaded.
3. **Repeat run** for stability.
4. **Expert review.** Harness is in `evals/expert_review/`. No scores exist yet.
5. **CI.** `.github/workflows/tests.yml` needs committing to run the offline and browser suites on push.

## Expert review

- Rubric: `evals/expert_review/RUBRIC.md`
- Build blinded sheets: `python evals/expert_review/build_review_sheet.py --results <jsonl> --out review/`
- Score and aggregate: `python evals/expert_review/score_reviews.py review/reviewer_*.csv --out review/summary.md`

## Browser tests

`pytest -m browser tests/browser` drives the chat UI and workflow inspector in Chromium. `/api/chat` is
mocked, so no database or key is needed. Requires Playwright and a Chromium build.