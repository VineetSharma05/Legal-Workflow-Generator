# Differentiator and Gaps: Legal Workflow Generator (Indian statutory compliance RAG)

Scope note: about 15 searches and fetches, done 2026-11-21. Most papers were seen through abstracts or snippets. Two were fetched in full: the BITS Pilani Indian hybrid-RAG paper and the Gokhan and Briscoe NLLP 2025 page. Well-known methodology references (Zheng et al. 2023, Dror et al. 2018, Self-RAG, CRAG, Panickssery et al. 2024) are cited by their canonical arXiv or ACL Anthology URLs from prior knowledge. I did not re-fetch them in this session.

## (a) Candidate differentiators ranked by defensibility against existing literature

### Takeaway
The most defensible contribution is the task-plus-resource combination: business-facing, multi-domain Indian statutory *compliance workflows* (not case law, judgment prediction, or bar-exam QA), with gold provision lists, covering the DPDP Act 2023 and Rules 2025. This only becomes publishable if the benchmark is released and validated. Temporal validity and claim-level citation verification are not novel on their own; 2025-2026 papers cover both. They are good components to report, not headline claims.

### Cited Findings
**Indian legal NLP landscape (what is already done):**
- IL-TUR (ACL 2024) is a multi-task Indian legal benchmark (8 tasks, English plus 9 Indian languages) that includes statute identification. It is judgment-centric, not compliance-centric. — [IL-TUR](https://arxiv.org/html/2407.05399v2)
- IL-PCSR (EMNLP 2025 Main) is a joint prior-case and statute retrieval corpus where query documents are judgments with statute citations masked. — [IL-PCSR](https://arxiv.org/html/2511.00268v1)
- ILSIC (arXiv Feb 2026) contains layperson queries mapped to 500+ Indian statutes, for statute identification. It is the closest "lay question to provision" resource, but it is classification/identification, not workflow generation. — [ILSIC](https://arxiv.org/html/2602.00881)
- NyayaRAG does RAG for Indian legal judgment prediction using statutes and precedents. — [NyayaRAG](https://arxiv.org/pdf/2508.00709)
- NyayGraph (NLLP 2025) uses an IPC cross-reference KG for statute identification. — [NLLP 2025.11](https://aclanthology.org/2025.nllp-1.11)
- "Are LLMs Court-Ready?" (NLLP 2025) evaluates frontier models on the All India Bar Exam. — [arXiv 2510.17900](https://arxiv.org/html/2510.17900v1)
- **Closest direct competitor:** "Domain-Partitioned Hybrid RAG for Legal Reasoning: Toward Modular and Explainable Legal AI for India" (BITS Pilani, arXiv preprint Feb 2026). Corpus: Constitution, Central Acts, IPC, about 2,900 SC judgments. Evaluation: a 40-question *synthetic* benchmark, an unnamed LLM judge, no inter-rater agreement, no significance tests. Result: 70% pass rate vs 37.5% for RAG-only. It does NOT address compliance workflows, DPDP, citation-accuracy evaluation, temporal validity (left as future work), or abstention (future work). — [arXiv 2602.23371](https://arxiv.org/html/2602.23371v1)
- LE-BTL (Taxmann x IIT Kharagpur, Feb 2026) is an industry study testing 12 LLMs on 100+ expert-validated questions on income tax, GST, and FEMA. It is not peer reviewed. It shows Indian tax/compliance QA is becoming a contested space. — [Taxmann blog](https://www.taxmann.com/post/blog/taxmann-ai-x-iit-kharagpur-llm-evaluation-le-btl-benchmark-study)
- No dedicated public DPDP Act LLM benchmark was found. Search results were vendor guidance and one weak IJIRT questionnaire paper. — [IJIRT paper](https://ijirt.org/publishedpaper/IJIRT210782_PAPER.pdf); [IBA commentary](https://www.ibanet.org/Building-AI-strategies-according-to-new-data-framework-india)

**Regulatory/compliance RAG elsewhere (framing precedents):**
- Gokhan and Briscoe, "Grounded Answers from Multi-Passage Regulations: Learning-to-Rank for Regulatory RAG" (NLLP 2025). They release ObliQA-MP (a multi-passage extension of ObliQA), add LLM validation that removes about 20% of passages that NLI filtering let through, and use learning-to-rank (lexical+semantic+graph), obligation-centric prompting, and citation-grounded answers scored by RePAS. — [ACL Anthology 2025.nllp-1.10](https://aclanthology.org/2025.nllp-1.10)
- RegNLP 2025 workshop: retrieval + reranking + answer generation pipelines for regulatory compliance. — [2025.regnlp-1.14](https://aclanthology.org/2025.regnlp-1.14); [RegNLP in Action](https://arxiv.org/html/2409.05677v1)
- GraphCompliance builds a policy graph over all GDPR articles and evaluates on 300 GDPR scenarios, with +4.1 to 7.2 micro-F1 over LLM-only and RAG baselines. — [arXiv 2510.26309](https://arxiv.org/html/2510.26309v1)
- PrivaCI-Bench reports that RAG can *reduce* accuracy on GDPR/EU AI Act questions. This is a useful counterpoint to cite. — [arXiv 2502.17041](https://arxiv.org/pdf/2502.17041)
- RegGuard (pharma compliance) uses LLM judges as primary evaluators plus verification by compliance officers. — [arXiv 2601.17826](https://arxiv.org/html/2601.17826v1)
- An EU AI Act QA and article-retrieval benchmark exists. — [arXiv 2603.09435](https://arxiv.org/html/2603.09435v1)

**Temporal validity is already an active subfield (so not a headline novelty):**
- LawInForce-eCFR (AACL 2026): 805 human-validated amendment pairs, 3,220 QA instances, AuthorityFirst-RAG with temporal scoping. — [AACL 2026](https://aclanthology.org/2026.aacl-long.42.pdf)
- "Temporal Misgrounding in Legal RAG" (French tax law, versioned corpus): static RAG never retrieved the date-applicable version, and a date-conditioned retriever reached 98.3% strict accuracy. — [arXiv 2608.09393](https://arxiv.org/pdf/2608.09393)
- SAT-Graph RAG models versioned legal norms for point-in-time retrieval. — [arXiv 2505.00039](https://arxiv.org/html/2505.00039v5)
- Another paper studies application of superseded rules and proposes "as-of date" filtering. — [arXiv 2605.23497](https://arxiv.org/html/2605.23497)

**Citation verification / hallucination (also established):**
- ALCE (EMNLP 2023) defined automatic citation quality evaluation. It notes that its NLI-based scorer cannot detect "partially supports", which lowers citation precision relative to human evaluation. This is directly relevant to the project's 26% "partially supported" figure. — [ALCE](https://aclanthology.org/2023.emnlp-main.398); [arXiv](https://arxiv.org/abs/2305.14627)
- Magesh et al. (Stanford; J. Empirical Legal Studies 2025) define a legal hallucination as incorrect information OR a false assertion that a source supports a proposition. Commercial legal RAG tools (Lexis+ AI, Ask Practical Law AI) hallucinated more than 17% of the time, and Westlaw AI-Assisted Research more than 34%. — [Stanford Law](https://law.stanford.edu/publications/hallucination-free/); [HAI summary](https://hai.stanford.edu/news/ai-trial-legal-models-hallucinate-1-out-6-or-more-benchmarking-queries); [arXiv 2405.20362](https://arxiv.org/abs/2405.20362)
- LegalBench-RAG: 6,858 query-answer pairs, entirely human-annotated by legal experts, with a retrieval-focused benchmark and a mini version. This is the standard a "benchmark" claim gets compared to. — [arXiv 2408.10343](https://arxiv.org/abs/2408.10343)

### Inferences
Ranked from most to least defensible:
1. **Task formulation plus resource (strongest).** No paper found frames Indian law as *business compliance workflow generation with per-step statutory citations across multiple regulatory domains*, and no DPDP Act 2023 / Rules 2025 benchmark was found. The multi-domain split (40 queries needing provisions from 2+ Acts) matches the "multi-passage regulation" gap Gokhan and Briscoe highlight, applied to cross-statute obligations. Claim this as "a task and a benchmark", but only if the dataset (queries, gold provision lists, the 4,888-provision corpus with as-of metadata) is released. Weakness: 100 author-written queries is small next to LegalBench-RAG (6,858, expert-annotated) or LawInForce (3,220). Frame it as a "diagnostic test set / pilot benchmark", not "the benchmark".
2. **Empirical finding on citation fabrication.** LLM-only gave 16.8% unsupported claims and 59 citations to non-existent provisions. Retrieval plus verification brings unsupported claims to roughly 0%. This replicates Magesh et al.'s concern in a new jurisdiction, and the "non-existent provision" count is a crisp, checkable number. It is defensible only once validated by humans (see b).
3. **Precision-under-abstention / scope safety.** Out-of-scope 10/10 refused with 2 false refusals. The deterministic foreign-law scope check is a sensible engineering contribution, but n=10 is far too small to claim anything statistically. Report it as a qualitative safety property or expand it to 30-50 out-of-scope queries.
4. **Cost / small-model angle.** A free-tier flash-lite generator reaching near-zero unsupported claims is attractive to NLLP/regional venues, but it is a practical result rather than a scientific one. Reviewers will ask "compared to what?" A frontier-model baseline is needed for this claim to land.
5. **Claim-level step verification and grader-driven context selection.** These are components with prior art (ALCE, Self-RAG, CRAG). The ablation (0.603 to 0.647 to 0.651 to 0.776 recall) is the evidence that makes them worth reporting. The context selector's +0.125 jump is the most interesting system finding, but it was tuned on data overlapping the test set (see b).
6. **Temporal as-of-date handling (least novel).** Several 2025-2026 papers cover this with dedicated benchmarks. Mention it as a feature with a small targeted test (for example, queries about not-yet-in-force DPDP Rules provisions). Do not claim it as a contribution.

Suggested one-sentence framing: "We introduce [name], a 100-query benchmark and 4,888-provision as-of-dated corpus for multi-statute Indian business compliance (including the DPDP Act 2023 and Rules 2025), and show that step-level citation verification with a small free-tier model eliminates the fabricated-provision citations that LLM-only generation produces, validated by human audit."

### Gaps
- I could not find whether any 2026 DPDP-specific academic RAG paper exists beyond vendor material. Re-check arXiv just before submission.
- The full text of ILSIC and LE-BTL was not read, so I can't confirm whether either covers company law, labour, or GST compliance in workflow form.

## (b) Likely reviewer objections and concrete ~3-day fixes (free-tier feasible)

### Takeaway
The fatal objections are (1) an unvalidated LLM judge, (2) test/dev leakage, and (3) no strong baselines or significance testing. All three can be largely addressed in 3 days: a 150-250 claim human audit with kappa, a clean held-out split reported separately, paired bootstrap CIs, a cross-family judge, and a frontier/standard-RAG baseline on the held-out subset.

### Cited Findings
- LLM-as-judge validity: Zheng et al. (2023) showed strong LLM judges reach above 80% agreement with humans on chat preference, but document position bias, verbosity bias, and self-enhancement bias. — [arXiv 2306.05685](https://arxiv.org/abs/2306.05685)
- Self-preference: LLM evaluators recognise and favour their own generations. — [Panickssery et al. 2024, arXiv 2404.13076](https://arxiv.org/abs/2404.13076)
- Positive self-bias found for GPT models and Claude 3.5 Sonnet, plus higher scores to other models *within the same family*. — [Spiliopoulou et al., arXiv 2508.06709](https://arxiv.org/abs/2508.06709)
- Same-family preference also appears in rubric-based evaluation. — [Pombal et al., arXiv 2604.06996](https://arxiv.org/pdf/2604.06996.pdf); [Who Judges Matters, arXiv 2609.17857](https://arxiv.org/pdf/2609.17857)
- Counterpoint: much measured self-preference may reflect evaluator uncertainty (89.6% on average). — [Roytburg et al., arXiv 2601.22548](https://arxiv.org/pdf/2601.22548)
- Long-CoT before the verdict mitigates harmful self-preference. — [arXiv 2504.03846](https://arxiv.org/html/2504.03846v3)
- Legal judge-human agreement benchmarks:
  - A Colombian law benchmark had one expert blind-score 323 judge decisions: quadratic kappa 0.635, Krippendorff's alpha 0.667. The judge ranked models like humans did but its absolute levels diverged. — [arXiv 2610.03639](https://arxiv.org/pdf/2610.03639)
  - AppellateGen reported kappa 0.452 with 4 experts on a 50-case subset. — [arXiv 2601.01331](https://arxiv.org/pdf/2601.01331)
  - LEXam reported expert-to-expert QWK of about 0.49, which suggests roughly 0.5 is a realistic ceiling. — [arXiv 2505.12864](https://arxiv.org/pdf/2505.12864)
  - Bloomberg Law argues Gwet's AC2 / Krippendorff's alpha can be preferable to Cohen's kappa under skewed label distributions. This applies here because about 99% of claims are "supported", where kappa paradox effects are severe. — [arXiv 2509.12382](https://arxiv.org/html/2509.12382v1)
- Statistical testing standard: Dror et al. (ACL 2018) recommend significance tests matched to the metric, with bootstrap/permutation tests for non-normal metrics, and found many ACL papers failed to report them. — [ACL Anthology P18-1128](https://aclanthology.org/P18-1128/)
- Precedent evaluations in the closest competitor (40 synthetic questions, unnamed judge, no IRR) show what *weak* looks like. Beating that bar visibly is a differentiator in its own right. — [arXiv 2602.23371](https://arxiv.org/html/2602.23371v1)
- Strong baselines reviewers expect: Self-RAG (self-reflective retrieval/critique) — [arXiv 2310.11511](https://arxiv.org/abs/2310.11511); CRAG (corrective retrieval evaluator) — [arXiv 2401.15884](https://arxiv.org/abs/2401.15884); commercial legal tools as in Magesh et al. — [arXiv 2405.20362](https://arxiv.org/abs/2405.20362)

### Inferences (objection, then a ~3-day fix)
1. **"The faithfulness numbers are an LLM grading an LLM."** Fix:
   - Two annotators (both authors, ideally one with law coursework, or a law-student volunteer) independently label a stratified random sample of about 200 claims against the cited provision text: Supported / Partial / Unsupported / Citation-nonexistent.
   - Oversample the LLM-only baseline and "partially supported" claims so rare classes appear. Note that 0% unsupported in the pipeline makes kappa uninformative there.
   - Report Cohen's kappa (annotator-annotator and judge-human), Gwet's AC1, and the confusion matrix.
   - Separately, have a human hand-verify all 59 "non-existent provision" citations, which is deterministic and fast via corpus lookup.
   - Budget is about 4-6 person-hours.
2. **"Same-family judge (Gemini judging Gemini)."** Fix: re-judge a subset (at least 300 claims) with a non-Google model on a free tier, for example an open-weight Llama/Qwen/Mistral via Groq/OpenRouter/HF free endpoints. Report cross-judge agreement and whether system rankings hold. Also ask the judge for reasoning before the verdict (CoT mitigation, arXiv 2504.03846).
3. **"Tuned on the test set."** Fix:
   - Make the headline result the 26 never-tuned queries (0.942 vs 0.872), and call the full-100 numbers "development".
   - Better: write 40-60 fresh queries now, freeze them before running anything, and run the final configuration once. Partners, or a classmate who did not see the system, should ideally write them.
   - State the leakage honestly in a Limitations section.
4. **"No significance / CIs."** Fix:
   - Paired bootstrap (10k resamples over queries) 95% CIs for recall and unsupported-rate for every system and ablation step.
   - Paired permutation test or Wilcoxon signed-rank on per-query recall between adjacent ablation steps.
   - McNemar on per-claim supported/unsupported between LLM-only and pipeline.
   - Report temp-0 vs temp-0.2 variability (79/90 identical) as a stability measure, and ideally run 3 seeds.
   - All of this is CPU-only and takes under a day.
5. **"Weak baselines."** Fix, on the held-out set:
   - (i) a frontier LLM-only baseline with free-tier access if available. The prompt says GPT/Claude commercial access may not be free, so use the strongest available Gemini Pro/free tier, plus one non-Google open model.
   - (ii) "plain RAG" and "RAG+abstain", which already exist.
   - (iii) a lightweight CRAG-style or Self-RAG-style reimplementation (retrieval-evaluator then re-retrieve) as a recognised-method baseline. Even a prompt-level approximation, clearly labelled as such, defuses the objection.
   - Commercial Indian legal tools (for example Manupatra/SCC AI) probably can't be tested in 3 days. Say so explicitly.
6. **"Small, author-written test set."** Fix:
   - Add query-construction provenance: sources such as compliance checklists and MCA/MeitY FAQs, plus the domain distribution.
   - Have a second person validate the gold provision lists on a 30-query subset and report agreement.
   - Frame the result as a pilot benchmark.
7. **"No legal expert evaluation."** Fix: get even one practising lawyer or law faculty member to rate 20-30 full workflows for correctness and completeness on a 1-5 Likert scale. Report it as a small expert study. If that isn't possible, put it in Limitations and avoid claiming "legally correct". Claim "provision-grounded" instead.
8. **"Partially supported = 26%."** Reviewers will read this as "a quarter of steps overclaim". Fix:
   - Break "partial" down via the human audit into acceptable paraphrase/inference vs. real overreach.
   - Cite ALCE's note that NLI-style scorers can't handle partial support.
   - Report a strict metric (supported only) and a lenient one.
9. **"Reproducibility."** Fix:
   - Release code, corpus snapshot with as-of dates, queries, gold lists, all prompts, model IDs/versions, temperatures, and run manifests (these already exist in evals/results), under an open licence on GitHub/Zenodo with a DOI.
   - Note API model drift as a limitation.
10. **"Out-of-scope n=10."** Fix: expand to 30+ out-of-scope queries plus 20+ near-miss in-scope queries (Indian law phrased with foreign terms) to measure false refusals properly.

### Gaps
- I found no study reporting kappa specifically for LLM-judged *claim faithfulness* in legal RAG. That makes the human audit a modest contribution in its own right.
- Free-tier availability of GPT/Claude APIs for baselines was not verified. Treat it as uncertain.

## (c) How comparable published papers framed similar contributions; venue and evaluation norms

### Takeaway
Accepted 2024-2026 legal/regulatory RAG papers frame themselves around (i) a new dataset/benchmark for an under-served jurisdiction or task, (ii) a measured failure mode (multi-passage drop, temporal misgrounding, hallucinated support), and (iii) a targeted method that fixes it, with retrieval metrics plus grounded-generation metrics. Strong papers add human/expert validation. NLLP 2026 (EMNLP) has already closed, so the realistic targets are journals or 2027 workshops.

### Cited Findings
**Framing patterns from titles and claims:**
- **"Problem → benchmark → method":**
  - Gokhan and Briscoe (NLLP 2025) observe retrieval degrading from single- to multi-passage, release ObliQA-MP, then propose learning-to-rank plus obligation-centric prompting. — [2025.nllp-1.10](https://aclanthology.org/2025.nllp-1.10)
  - The same arc appears in Temporal Misgrounding ("static RAG never retrieves the applicable version" → date-conditioned retriever). — [arXiv 2608.09393](https://arxiv.org/pdf/2608.09393)
  - It also appears in LawInForce-eCFR (benchmark + AuthorityFirst-RAG). — [AACL 2026](https://aclanthology.org/2026.aacl-long.42.pdf)
- **"Reliability audit" framing:** "Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools" reached a journal (JELS 2025) and is the anchor citation for legal hallucination definitions. — [Stanford Law](https://law.stanford.edu/publications/hallucination-free/)
- **"Benchmark for jurisdiction X":**
  - IL-TUR (ACL 2024) — [arXiv](https://arxiv.org/html/2407.05399v2)
  - IL-PCSR (EMNLP 2025) — [arXiv](https://arxiv.org/html/2511.00268v1)
  - "Do LLMs Know Colombian Law? A Reliability Benchmark" — [arXiv 2610.03639](https://arxiv.org/pdf/2610.03639)
  - "Are LLM Assistants Regulatory Ready?" (EU AI Act Q&A) — [arXiv 2611.04546](https://arxiv.org/pdf/2611.04546)
- **Evaluation standards observed:**
  - Retrieval: Recall@10/MAP@10/nDCG@10 against dense, hybrid, and fusion baselines (Gokhan and Briscoe).
  - Benchmark construction: expert annotation (LegalBench-RAG) — [arXiv 2408.10343](https://arxiv.org/abs/2408.10343)
  - Judge validation: an expert-validated LLM judge with kappa reported (Colombian, AppellateGen).
  - Primary evaluators: LLM judges plus domain-officer verification (RegGuard) — [arXiv 2601.17826](https://arxiv.org/html/2601.17826v1)
- **Venue facts:** NLLP 2026 (8th edition, EMNLP 2026, Budapest) took long papers of up to 8 pages via OpenReview. Submissions were due 11 Aug 2026, and the workshop was on 28 Oct 2026, so it is closed. — [ACL portal CfP](https://www.aclweb.org/portal/content/nllp-2026); [JURIX list post](https://groups.google.com/g/jurix-foundation/c/hBSN5AEt_so)

### Inferences
- **Recommended title pattern:**
  - "[Name]: A Benchmark and Citation-Verified RAG Pipeline for Multi-Statute Indian Compliance Workflows", or
  - "Grounded or Fabricated? Step-Level Citation Verification for Indian Regulatory Compliance Workflows under the DPDP Act".
- **Lead with the measured failure:** LLM-only fabricates provisions (59 non-existent citations, 16.8% unsupported). Then the benchmark. Then the method plus ablation. Then human-validated faithfulness. This mirrors the accepted framing arc.
- **Minimum evaluation table reviewers expect:**
  - Retrieval recall (and ideally nDCG/MRR) per split with 95% CIs.
  - Faithfulness (strict/lenient) per system.
  - A fabricated-citation count.
  - Abstention precision/recall.
  - An ablation with significance markers.
  - Judge-human agreement.
  - Cost/latency per query. The free-tier angle becomes evidence here.
- **Clearly ahead of the closest Indian competitor (arXiv 2602.23371):**
  - Larger query set (100 vs 40)
  - Gold provision lists
  - Named, cross-family-checked judge
  - Human audit
  - Abstention
  - Temporal handling
  - Compliance focus

  Cite it explicitly as related work and contrast with it.
- **On a journal target:** journals (for example AI & Law, CLSR, or regional Springer journals) put more weight on legal validity and limitations than ACL workshops do. That makes the expert-review item (b.7) and an honest Limitations section more important than extra system tweaks.

### Gaps
- No public OpenReview reviews for NLLP 2025 legal-RAG papers were retrieved. Reviewer-objection patterns above are inferred from methodology literature and the evaluation choices of accepted papers, not from actual review text.
- JURIX 2026/ICAIL 2027 deadlines and AI & Law journal author guidelines were not checked in this session.
