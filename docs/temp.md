I read the diagram against the agent implementation in [legal_workflow_generator/agent/](legal_workflow_generator/agent/). Short version: **the diagram is mostly not implemented.** The code is a single-pass self-corrective LangGraph loop that fakes multi-domain support by concatenating all domains into one retrieve+generate call. The diagram's defining ideas — decomposition, per-domain isolated subroutines, and a stitching layer — are absent.

## Element-by-element

| Diagram element | Status | Where / why |
|---|---|---|
| User query | ✅ | `graph.invoke({"query": ...})` |
| Query processing unit (Phase 2) | ✅ | [nodes.py:37](legal_workflow_generator/agent/nodes.py#L37) `classify_query` → normalizer + intent classifier + context resolver |
| **Decomposition layer** (split into sub-tasks, max 3 domains) | ❌ partial | `classify_query` makes an LLM call to detect `all_domains` ([nodes.py:53-71](legal_workflow_generator/agent/nodes.py#L53-L71)), but it does **not** split into independent sub-tasks and does **not** cap at 3 (can return all 5) |
| **Sub-task: domain A / B / C** (parallel branches) | ❌ | No fan-out. One linear graph, one state object. |
| **Per-sub-task subroutine, once per domain** | ❌ | [retrieve](legal_workflow_generator/agent/nodes.py#L81) loops domains *inside a single node* and merges everything into one pool (`sorted(...)[:7]`). Every downstream node works on that merged pool, not per domain. |
| Transform query | ⚠️ | `rewrite_query` node exists, but it's global, not per-domain; `retrieve` also bolts on static hint strings per domain |
| Retrieve | ✅ | `_rag_pipeline.searcher.search` |
| Grade context — cap 3 retries, same domain | ✅ cap / ❌ "same domain" | `route_after_context_grade` abstains at `retry_count_retrieval >= 3` ([graph.py:24](legal_workflow_generator/agent/graph.py#L24)); retry re-runs across all domains, not scoped |
| Generate — answer + inline citations | ✅ / ⚠️ | `generate` node; "citations" are just the retrieved `provision_id`s dumped in a SOURCES block — not per-claim inline citations |
| Verify citations — Section refs exist in chunks | ⚠️ partial | `verify_citations` ([nodes.py:223](legal_workflow_generator/agent/nodes.py#L223)) string-matches citation ids against retrieved doc ids → `verified` / `failed` lists |
| **"fabricated → strip claim, regenerate it, retries exhausted"** | ❌ | `failed_citations` is written and then **nothing reads it**. No claim-stripping, no regenerate trigger, no edge from `verify_citations` back to `generate`. |
| Grade groundedness — cap 2 retries | ✅ / ⚠️ | `route_after_groundedness` regenerates until `retry_count_generation >= 2` — **but** `route_after_verify_citations` ([graph.py:49](legal_workflow_generator/agent/graph.py#L49)) skips groundedness entirely whenever `len(all_domains) > 1`, i.e. exactly the multi-domain case |
| Grade answerable | ✅ | `grade_answerability`; on `off_target` it loops back to `rewrite_query` reusing the retrieval counter |
| Sub-task answered / abstained | ⚠️ | Whole-query `abstain` node, not per-sub-task — one abstain kills the entire answer |
| **Stitching layer** (order steps, merge citations, flag gaps) | ❌ | Does not exist. The single `generate` prompt is just told to "group by domain"; no post-hoc ordering, cross-subtask citation merge, or gap flagging. |
| Final workflow + citations | ✅ | `result["answer"]` + `result["verified_citations"]` |

## What's actually there

The middle "per-sub-task subroutine" box, taken alone, is a fair description of the current graph: retrieve → grade context → (rewrite ↔ retrieve) → generate → verify → grade groundedness → (regenerate ↔ generate) → grade answerable → answered/abstain. That self-correction loop is implemented. It just runs **once, globally**, not N times in isolation per domain.

## Gaps to close if you want the diagram

1. **Decomposition layer** — new node that splits into ≤3 `SubTask` objects (domain + scoped sub-question).
2. **Fan-out** — LangGraph `Send` / subgraph so the subroutine runs per sub-task with its own retry counters and its own doc pool.
3. **Citation-failure path** — wire `failed_citations` to a strip-and-regenerate step; currently dead data.
4. **Stitching layer** — new terminal node to merge per-subtask answers, dedupe/merge citations, order steps, and flag domains that abstained as "gaps."
5. **Per-subtask abstain** — abstain should mark one sub-task, not the whole response.
6. Remove the multi-domain groundedness bypass (Known Limitation #3 in [PHASE3_EVAL_RESULTS.md](docs/PHASE3_EVAL_RESULTS.md)) — the fan-out makes it unnecessary.

Note [docs/architecture.png](docs/architecture.png) in the repo is a different, older block diagram (Query/Retrieval/Workflow/Response units) — it doesn't match the diagram you posted either.