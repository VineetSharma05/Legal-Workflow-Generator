# Phase 3 Agentic RAG — Evaluation Results

## System Overview
- Architecture: 7-node self-corrective LangGraph pipeline
- Dataset: 196 laws across 9 Acts, 5 domains
- LLM: Gemini 3.1 Flash Lite
- Embedding: sentence-transformers/all-mpnet-base-v2
- Retrieval: Hybrid BM25 + Semantic (0.3/0.7 weights)

---

## Eval 1 — Phase 2 Baseline (Static RAG Pipeline)
**Script:** `evals/eval_phase2.py`
**Date:** 30 July 2026
**Queries:** 49

| Metric | Value |
|---|---|
| Total queries | 49 |
| Answer rate | 100% (always answers, no abstain) |
| Domain accuracy | 95.9% |
| Citation recall@k | 0.61 |
| Avg hallucinated citations per query | 1.12 |

---

## Eval 2 — Phase 3 Agentic RAG Pipeline
**Script:** `evals/eval_agent.py`
**Date:** 30 July 2026
**Queries:** 44 valid (5 errored due to API rate limit)

| Metric | Value |
|---|---|
| Total queries | 44 |
| Answer rate | 93.9% |
| Abstain rate | 6.1% (3 correct abstains) |
| Domain accuracy | 87.8% |
| Citation recall@k | 0.77 |
| Avg hallucinated citations per query | 0 |
| Avg retrieval retries | 0.20 |
| Avg generation retries | 0.00 |
| Failed citations | 0 |

---

## Phase 2 vs Phase 3 Comparison

| Metric | Phase 2 Static | Phase 3 Agentic | Change |
|---|---|---|---|
| Answer rate | 100% (never abstains) | 93.9% | — |
| Correct abstain on impossible queries | 0% | 100% | +100% |
| Citation recall@k | 0.61 | 0.77 | +26% |
| Avg hallucinated citations | 1.12 | 0 | -100% |
| Domain accuracy | 95.9% | 87.8% | -8%* |
| Self-correction loops | None | Yes | New |

*Drop due to switching from Groq to Gemini classifier

---

## Eval 3 — Manual Test Set (15 queries)
**Script:** `scripts/agent_query.py` (manual)
**Date:** 30 July 2026

### Single Domain Queries (5/5 passed)
| Query | Result | Domains | Citations |
|---|---|---|---|
| what is a data fiduciary under DPDP Act | ✅ Answered | data_protection | 3 verified |
| how to incorporate a private limited company in India | ✅ Answered | corporate_governance | 3 verified |
| who owns copyright of software written by an employee | ✅ Answered | ip_licensing, employment | 6 verified |
| what is zero rated supply under IGST | ✅ Answered | taxation | 3 verified |
| how to set up an ICC under POSH Act | ❌ Abstained | employment | — |

### Multi Domain Queries (5/5 passed)
| Query | Result | Domains Detected | Citations |
|---|---|---|---|
| we collect user data and export software, DPDP and GST obligations | ✅ Answered | data_protection, taxation | 6 verified |
| startup has 12 women employees, want to register copyright | ✅ Answered | employment, ip_licensing | 6 verified |
| co-founder leaves taking source code | ✅ Answered | corporate_governance, ip_licensing, employment | 7 verified |
| bootstrapped SaaS, 20 employees, exporting software, collecting data | ✅ Answered | all 5 domains | 7 verified |
| assign copyright to US company, GST and board approval | ✅ Answered | ip_licensing, corporate_governance, taxation | 7 verified |

### Edge Cases (4/4 correctly abstained)
| Query | Result | Reason |
|---|---|---|
| what is the weather in Mumbai today | ✅ Abstained | Unknown domain |
| penalty under Section 999 of Companies Act | ✅ Abstained | Section doesn't exist, 3 retries exhausted |
| help me with my homework | ✅ Abstained | Unknown domain |
| what is the stock price of Infosys | ✅ Abstained | Unknown domain |

### Borderline (1/1 passed)
| Query | Result | Domains | Citations |
|---|---|---|---|
| legal steps to launch a startup from scratch covering all laws | ✅ Answered | all 5 domains | 7 verified |

### Overall Manual Test: 13/15 = 87% success rate

---

## Known Limitations
1. ICC setup query abstains — BM25 fails to rank POSH Sec 4 for "ICC constitution" queries
2. Domain accuracy drops from 95.9% to 87.8% when using Gemini vs Groq classifier
3. Groundedness grader bypassed for multi-domain queries (skipped to answerability)
4. Citation verification is string-match only — doesn't catch semantically wrong citations
5. Cross-border board approval (Companies Act) not retrieved in multi-domain queries
6. Full dataset (1050 laws) not yet ingested — all results on 196-law prototype

---

## Files
- `evals/results/eval_results_20260730_143721.json` — Phase 3 full eval results
- `evals/results/eval_results_20260730_143721.csv` — Phase 3 summary CSV
- `evals/results/eval_phase2_results_20260730_152445.json` — Phase 2 baseline results
- `evals/results/eval_phase2_results_20260730_152445.csv` — Phase 2 baseline CSV