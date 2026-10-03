# Domain Classification Reliability — What Was Done and Why

This documents a work session focused on one question: `LegalContextResolver`
picks the legal domain (`data_protection`, `corporate_governance`, `ip_licensing`,
`taxation`, `employment`, or `unknown`) for every query. Originally that came
from a single Gemini call, with a hand-typed keyword list as a fallback only
when the call errored out — no way to tell whether the LLM's answer was
actually correct, and no way to measure classification quality at all.

The session moved through two approaches. The first added correctness signals
on top of the LLM call without changing who made the decision. The second
replaced the LLM as the *primary* decision-maker with a deterministic
classifier derived from the corpus itself, demoting the LLM to a fallback or
cross-check. Both are documented here because the first phase's findings are
part of why the second happened.

## The constraint that shaped everything

Building a trained classifier was ruled out early — a real classifier needs a
labeled training set, and this project doesn't have one at any usable scale.
But that constraint applies to *training*, not to *evaluation*: measuring how
well an existing classifier performs needs orders of magnitude less data than
training one from scratch. A hand-labeled set of 50-200 examples is enough to
get a real accuracy number, a confusion matrix, and per-domain precision/recall —
which is exactly what `evals/eval_agent.py` and `evals/eval_phase2.py` already
do (49 labeled queries, ~90-96% domain accuracy reported). What was missing was
the same treatment applied specifically to the classifier in isolation, plus a
way to catch bad classifications *without* labels, for production traffic that
was never hand-labeled. `evals/eval_domain_classification.py` was built for
that — it isolates `LegalContextResolver` (no retrieval, no generation) and
runs it against 51 hand-written queries: 8 clear examples per domain, 6
boundary queries deliberately written to plausibly span two domains, and 5
off-topic queries that should resolve to `unknown`.

## Phase 1: correctness signals on top of the LLM classifier

Two signals were added to `LegalContextResolver`, chosen for very different
cost profiles:

**`domain_agreement`** — a rule-based keyword classifier was already in the
code as an error-path fallback, only invoked when the Gemini call itself
failed. It was changed to always run and be compared against the LLM's
answer, whether or not the LLM succeeded — no extra API call, and
disagreement between the two is a zero-cost hint that a query might be
misclassified.

**`domain_confidence`** — a self-consistency check. The domain-classification
prompt is sampled N times (default 3) at `temperature=0.7`, and the resolver
majority-votes across the samples; the winning share (e.g. `0.67` for a 2-of-3
split) is returned as `domain_confidence`. This is the standard self-consistency
technique from Wang et al. 2022, *"Self-Consistency Improves Chain of Thought
Reasoning in Language Models"* (arXiv:2203.11171) — originally proposed to
improve accuracy on reasoning tasks by voting across multiple reasoning paths,
and now also widely used as an uncertainty estimator: low agreement across
samples correlates with the model being wrong. The framing of using resampling
specifically to *catch hallucination* in a black-box, no-training-data setting
is closer to Manakul, Liusie & Gales 2023, *"SelfCheckGPT: Zero-Resource
Black-Box Hallucination Detection for Generative Large Language Models"*
(arXiv:2303.08896), and to Kuhn, Gal & Farquhar 2023, *"Semantic Uncertainty"*
(ICLR 2023, arXiv:2302.09664), both of which detect unreliable LLM output by
measuring how much the output changes under repeated sampling rather than by
checking it against ground truth.

Self-consistency directly multiplies API cost — N samples means N Gemini calls
for one classification step, on top of an already multi-call agent pipeline
that was already hitting rate limits in prior eval runs. Because of that, it
was built to be trivially toggled: off by default via `SELF_CONSISTENCY_ENABLED`
in `.env`, or per-instance via `LegalContextResolver(self_consistency=True)`.

### Phase 1 results (51 queries, self-consistency on, 3 samples)

| Metric | Value |
|---|---|
| Overall accuracy | 90% (46/51) |
| Rule-based-only accuracy | 86% |
| LLM/rule-based agreement rate | 92% (47/51) |
| Accuracy when they agree | 91% (n=47) |
| Accuracy when they disagree | 75% (n=4) |
| Avg self-consistency confidence | 1.00 — every query got a unanimous 3/3 vote |

`domain_agreement` produced a real, if small-sample, signal — disagreement
cases scored meaningfully lower (75%) than agreement cases (91%), and caught
the clearest possible case: *"how to raise a seed round from VCs"* (expected
`unknown`) was the one query where the LLM and rule-based classifier disagreed
**and** the LLM was wrong. Self-consistency, as configured, produced no signal
at all — across all 51 queries and 153 samples, there was not one split vote,
including on the 5 queries the classifier got wrong. The errors were
*confidently* wrong, not *unstable*, which is a distinct failure mode from
what self-consistency is built to catch. Paying 3x the API cost for a signal
that never fired was a bad trade as configured — plausibly because a
single-word answer out of 6 domains sits on a much sharper probability peak
than the open-ended reasoning chains self-consistency was designed for, so
`temperature=0.7` may simply not have been perturbing the output. Raising
`SELF_CONSISTENCY_TEMPERATURE` (a module constant in `context_resolver.py`)
and re-running the eval to check whether variance appears at all remains an
open, unexecuted test.

## Phase 2: replacing the LLM as primary — a deterministic keyword classifier

Phase 1's signals could flag a likely-wrong classification but couldn't fix
it — the domain itself still came from a non-deterministic LLM call, so the
pipeline as a whole was still non-deterministic and its cost still scaled
with LLM usage. The reframing that unblocked this: the "no training data"
constraint applies to labeled *queries* (question → domain), which really are
scarce, but the law corpus is already domain-labeled *documents* — one source
dataset file per domain (`dpdp.json`, `corporategovernance.json`,
`softwarelicensing.json`, `tax.json`, `womanandsex.json`). That's not a
hallucination-detection problem, it's a standard discriminative
feature-extraction problem over already-labeled documents.

**Method.** Each domain's ingested provisions are concatenated into a single
pseudo-document, and TF-IDF (Salton & Buckley 1988, the classic term-weighting
scheme for information retrieval) is fit across those 5 pseudo-documents —
this is the standard way to pull distinctive vocabulary out of a small number
of classes: a term common everywhere scores low everywhere, one concentrated
in one domain scores high only there. Both unigrams and bigrams are kept,
since legal terms are often multi-word ("data fiduciary", "internal complaints
committee"). This runs once at ingestion (`rag/domain_keywords.py`, via
`python main.py extract-keywords`) and persists the top 40 terms per domain to
a new `domain_keywords` table. A new `KeywordDomainClassifier` loads that
table and scores a query by summing the weights of its matched terms — no LLM
call, same query always produces the same answer.

**Schema.** A `statute_id → domain` map was added at ingestion time so every
row in `laws` now carries an explicit `domain` column (previously domain was
only implicit via which dataset file a provision came from); the new
`domain_keywords(domain, term, score)` table holds the extracted vocabulary.

**Combining with the LLM.** `DOMAIN_CLASSIFICATION_STRATEGY` chooses between:
- `llm_fallback` (default) — Gemini is only called when the keyword
  classifier finds no match at all.
- `combine` — Gemini is always called and cross-checked; on disagreement, a
  keyword match strong enough to clear `DOMAIN_KEYWORD_STRONG_MATCH_THRESHOLD`
  is trusted over the LLM, otherwise the LLM's answer wins (a single weak or
  coincidental keyword hit shouldn't outrank it).

### Phase 2 results (same 51-query eval, self-consistency off to isolate this change)

The first run of this eval was contaminated by Gemini free-tier rate limits
(15 requests/minute) — `combine` mode calls Gemini on every query and hit
`429 RESOURCE_EXHAUSTED` on 13 of 38 attempted domain calls, and a re-run after
the API key was refreshed also caught the Postgres container having stopped
between sessions (47/51 queries errored on `Connection refused` before it was
restarted). Numbers below are from the clean re-run, after fixing both:

| | Prior (LLM-only) | `llm_fallback` | `combine` |
|---|---|---|---|
| Overall accuracy | 90% | 88% (45/51) | **98% (50/51)** |
| Queries needing zero LLM calls | 0/51 | **37/51 (73%)** | 4/51 |
| Errors (API/infra) | 0/51 | 0/51 | 0/51 |

**The determinism win is real.** `llm_fallback` resolves 73% of queries with
zero LLM calls — deterministic, reproducible, free — for a ~10pp accuracy cost
against a clean `combine` run. Its 6 errors aren't random: 4 of them (annual
general meetings, board resolutions, NDAs, offer letters) trace to real
keyword-vocabulary coverage gaps — concepts underrepresented in the
196-provision ingested subset — which is a fixable data problem (ingest the
fuller ~1050-provision corpus already sitting in `datasets/`), not a flaw in
the TF-IDF approach itself.

**`combine` mode, run cleanly, is very strong.** 50/51 correct, with 5 of 6
domains at 100% precision and recall (`data_protection`, `corporate_governance`,
`ip_licensing`, `taxation`, `employment`) once API/infra noise was removed.
Its one miss is *"how to raise a seed round from VCs"* (expected `unknown`) —
the same query both strategies and the original all-LLM baseline get wrong,
because the LLM confidently answers `corporate_governance` regardless of which
strategy calls it. That's a genuine LLM-overreach failure mode, not something
either classification approach fixes on its own — `domain_agreement` did flag
it (keyword found nothing, so nothing to agree with), but flagging isn't the
same as correcting, and this is one of the "still open" items below.

**Trade-off, not a clear winner.** `combine` is ~10pp more accurate but calls
Gemini on every single query (up to 2x the calls of the old baseline, once for
intent and once for domain); `llm_fallback` avoids 73% of those calls entirely
at real but modest accuracy cost, and is far more robust to the exact kind of
quota pressure that contaminated the first run of this eval. Which one is
worth it depends on whether the deployment is cost/latency-constrained or
accuracy-constrained — `llm_fallback` remains the configured default
(`DOMAIN_CLASSIFICATION_STRATEGY=llm_fallback`) for now, but `combine`'s clean
numbers make it a reasonable choice too.

## What's still open

- **The one persistent LLM-overreach failure** (*"how to raise a seed round
  from VCs"* → `corporate_governance`) survives every strategy and both API
  keys tried so far — worth a dedicated look, since it's the only error left
  in the `combine` results and it isn't a keyword-coverage problem.
- **The self-consistency temperature test from Phase 1** hasn't been run —
  raise `SELF_CONSISTENCY_TEMPERATURE` and re-check whether split votes start
  appearing, and whether they still predict wrongness at that temperature.
- **Ingesting the fuller corpus** would likely close most of `llm_fallback`'s
  remaining keyword-coverage gaps (AGM, NDA, offer letters) — the 196-provision
  subset is a small fraction of the ~1050 provisions available in `datasets/`.
- **`domain_agreement`, `domain_confidence`, and `domain_source` are still
  observability-only** — they're computed, logged, and surfaced in the agent
  trace, but nothing in `route_after_classify` or elsewhere in the LangGraph
  pipeline acts on them (e.g. triggering `rewrite_query` on disagreement or a
  weak keyword match).
- **The 51-query eval set could grow**, especially around the
  `ip_licensing`/`corporate_governance` and `taxation`/`ip_licensing` boundary
  cases the confusion matrices show are the weakest spots in both approaches.
