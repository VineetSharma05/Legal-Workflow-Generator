# Related Work Map: Faithful, Citation-Grounded Legal RAG for Indian Regulatory Compliance

Verification legend (applies to every entry):
- **[V]**: confirmed this session from the primary page (arXiv abstract page, ACL Anthology, or publisher/RePEc record).
- **[S]**: confirmed this session only from search-result snippets that pointed at the primary page. Bibliographic details are probably right, but re-check the numbers before citing.
- **[K]**: a well-known paper cited from prior knowledge with its canonical arXiv/ACL ID. It was not re-fetched this session. The IDs are standard, but re-check the numbers against the PDF before the camera-ready.

Project being positioned: LangGraph multi-agent RAG for Indian compliance (DPDP Act 2023 + Rules 2025, IT Act/SPDI, company law, GST/TDS, labour/ESI, IP). It produces step-by-step workflows in which each step cites one statutory provision. Retrieval is hybrid BM25+dense over 4,888 provisions. Other components: per-domain planning, claim-level verification against the cited provision, abstention on out-of-scope or foreign-law questions, and as-of-date handling for provisions not yet in force. Results so far: provision recall 0.776 on 100 queries; LLM-judge unsupported-claim rate 16.8% for LLM-only vs about 0-0.5% for the grounded systems.

---

## Q1. Legal hallucination studies (2024-2026)

### Takeaway
The anchor citations are the Stanford RegLab pair, Dahl et al. 2024 (JLA) and Magesh et al. 2025 (JELS). They show that general LLMs hallucinate on legal queries 58-88% of the time, and that commercial legal RAG tools still hallucinate more than 17% of the time despite "hallucination-free" marketing. Work from 2026 moved to claim-level, citation-support benchmarks (PARCEL at ICAIL 2026, "Who Checks the Citations?"). That is the framing this project's claim-level verifier fits into.

### Cited Findings
- **Magesh, Surani, Dahl, Suzgun, Manning & Ho (2025)**, "Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools," *Journal of Empirical Legal Studies* 22(2):216-242, DOI 10.1111/jels.12413. Accepted 14 March 2025; the preprint is arXiv:2405.20362 (May 2024). In preregistered tests, LexisNexis (Lexis+ AI) and Thomson Reuters (Westlaw AI-Assisted Research, Ask Practical Law AI) tools each hallucinated more than 17% of the time, which is still less than GPT-4. The paper contributes a preregistered dataset and a typology that separates hallucinations from accurate responses. It is the first public evaluation of proprietary legal RAG tools. [V] — [RePEc record](https://econpapers.repec.org/RePEc:wly:empleg:v:22:y:2025:i:2:p:216-242); [Stanford Law page](https://law.stanford.edu/publications/hallucination-free-assessing-the-reliability-of-leading-ai-legal-research-tools); [arXiv](https://arxiv.org/abs/2405.20362v1); [author PDF](https://nlp.stanford.edu/~manning/papers/Magesh-Hallucination%E2%80%90Free-2025.pdf)
  - Per-tool figures from memory: Westlaw AI-AR about 33%, Lexis+ AI about 17%. [K] Verify these in the PDF.
- **Dahl, Magesh, Suzgun & Ho (2024)**, "Large Legal Fictions: Profiling Legal Hallucinations in Large Language Models," *Journal of Legal Analysis* 16(1):64-93, DOI 10.1093/jla/laae003; arXiv:2401.01301. Hallucination rates on verifiable questions about US federal court cases ranged from 58% (GPT-4 / ChatGPT 4) to 88% (Llama 2). The models also showed "contra-factual bias": they accepted false premises in the user's question. [K] — [arXiv](https://arxiv.org/abs/2401.01301); [DOI](https://doi.org/10.1093/jla/laae003)
- **Liu, Stammbach & Henderson (2026)**, "Who Checks the Citations? Benchmarking Legal Hallucination Detection," arXiv:2606.21155 (v1 19 June 2026, v3 4 Oct 2026; preprint). The authors found more than 1,000 court filings with fabricated citations, rising year on year. They release 1,300 brief excerpts with injected citation errors. The best detector, GPT-5 in an agentic setup, reaches 84.4% recall and 55.0% F1 at an average of 15.3 steps per excerpt. All models struggle with subtle error types. [V] — [arXiv](https://arxiv.org/abs/2606.21155)
- **Demir & Canbaz (2026)**, "When Citations Mislead? A Claim-Level Benchmark for Legal Hallucination Detection" (PARCEL), *ICAIL 2026* (Singapore, 8-12 June 2026); arXiv:2610.10971. The benchmark has 3,396 parenthetical claims from New York Court of Appeals decisions, each labelled Supported, Refuted, or Not Found, and is framed as three-way NLI. The best zero-shot accuracy is 0.97, but models still often label unsupported claims as supported. Missing support is harder to detect than contradiction, and fabricated but plausible citations cause the largest drop. [V] — [arXiv](https://arxiv.org/abs/2610.10971)
- **Ovcharov (2026)**, "Citation Grounding: Detecting and Reducing LLM Citation Hallucinations via Legal Citation Graphs," arXiv:2606.00898. Across five systems on 100 Ukrainian legal queries, 13-21% of citations were hallucinated. [S] — [arXiv](https://arxiv.org/abs/2606.00898v1)
- Further 2026 preprints seen only in search results, with findings not verified: "Is this Citation on Point?" (arXiv:2608.12571; reports that agentic checkers are weakest on wrong pincites and content misrepresentation) [S]; "LexAgentHallu: A Hierarchical Benchmark for Profiling Hallucinations in Legal Agents" (arXiv:2609.09754) [S]; "Legal LLM Hallucination Should Be Evaluated as Failure of Legal Warrant" (position paper, arXiv:2609.17546) [S]. — [2608.12571](https://arxiv.org/pdf/2608.12571); [2609.09754](https://arxiv.org/pdf/2609.09754); [2609.17546](https://arxiv.org/pdf/2609.17546)
- **Anaokar et al. (incl. P. Bhattacharyya) (2025)**, "HalluDetect: Detecting, Mitigating, and Benchmarking Hallucinations in Conversational Systems in the Legal Domain," *EMNLP 2025 Industry Track*, pp. 1822-1847. This is Indian legal work, on consumer-grievance chatbots built on LLaMA 3.1 8B. HalluDetect reaches 68.92% F1, 22.47 better than baselines; the abstract does not say whether that is absolute points or relative. The best of five mitigation architectures, AgentBot, has 0.4159 hallucinations per turn and 96.13% token accuracy. [V] — [ACL Anthology](https://aclanthology.org/2025.emnlp-industry.128/)
- **Indian practice context**: an October 2025 Bombay High Court order reportedly quashed a tax assessment that relied on three non-existent, AI-generated precedents. The source is a non-peer-reviewed practitioner blog, so it is usable only as motivation. [S] — [iPleaders](https://blog.ipleaders.in/ai-hallucinated-case-law-fake-citations-india/)

### Inferences
- **How this project differs:** the hallucination literature is almost entirely about US case law and evaluates research or memo answers. This project targets Indian statute and regulation provisions and evaluates procedural compliance workflows, scoring each step against its cited provision. The LLM-only baseline (16.8% unsupported claims plus invalid citations) is of the same order as Magesh et al.'s more-than-17% for commercial RAG tools. However, it is measured by an LLM judge rather than by experts, so the paper should say so explicitly.
- Within the hallucination typology, PARCEL's Supported/Refuted/Not Found scheme is the closest published analogue to this project's claim-level verifier. Adopting compatible labels would make the results directly comparable.

### Gaps
- I found no hallucination study specific to Indian statutes, as opposed to Indian case law (HalluDetect covers consumer chatbots).

---

## Q2. Legal RAG benchmarks and systems (statutory/regulatory focus)

### Takeaway
Statute- and regulation-retrieval benchmarks exist for Belgian law (BSARD), US tax (SARA), the Japanese Civil Code (COLIEE), US housing statutes, Thai law (NitiBench) and ADGM financial regulation (ObliQA/RIRAG). None covers Indian regulatory law, and none evaluates multi-step, multi-domain workflow outputs. The closest task design is RIRAG: regulatory QA with obligation-grounded answers, scored by recall@10 plus a reference-free faithfulness metric, RePASs.

### Cited Findings
**Regulatory compliance RAG**
- **Gokhan, Wang, Gurevych & Briscoe (2024)**, "RIRAG: Regulatory Information Retrieval and Answer Generation," arXiv:2409.05677. Introduces ObliQA: 27,869 questions over ADGM (Abu Dhabi Global Market) financial regulations, split 22,295 train / 2,888 dev / 2,786 test. Retrieval is scored mainly with Recall@10, and answers with RePASs, a reference-free, model-based metric of obligation coverage and entailment. [S] — [arXiv](https://arxiv.org/html/2409.05677v2)
- **Gokhan, Wang, Gurevych & Briscoe (2025)**, "RIRAG-2025 shared task overview," *Proc. 1st Regulatory NLP Workshop (RegNLP 2025)* at COLING 2025, pp. 1-4. 19 teams took part, 16 of which submitted papers, evaluated on 446 hidden questions. Task-specific trained embeddings did marginally better, domain fine-tuning hurt performance, and overall accuracy "remains a challenge." One team (LeSeR) reported Recall@10 0.8201 and MAP@10 0.6655. [S] — [ACL Anthology](https://aclanthology.org/2025.regnlp-1.1); [proceedings](https://aclanthology.org/volumes/2025.regnlp-1/)
- "RAGulating Compliance: A Multi-Agent Knowledge Graph for Regulatory QA," arXiv:2508.09893 (2025). [S] — [arXiv](https://arxiv.org/html/2508.09893)

**Statute retrieval and statutory reasoning**
- **Louis & Spanakis (2022)**, "A Statutory Article Retrieval Dataset in French" (BSARD), *ACL 2022*; arXiv:2108.11792. More than 1,100 French questions from Belgian citizens, labelled against more than 22,600 Belgian statutory articles. BM25 is a strong baseline, and dense retrievers help after fine-tuning. [K] — [arXiv](https://arxiv.org/abs/2108.11792)
- **Holzenberger, Blair-Stanek & Van Durme (2020)**, "A Dataset for Statutory Reasoning in Tax Law Entailment and Question Answering" (SARA), *NLLP Workshop 2020*; arXiv:2005.05257. Built on sections of the US Internal Revenue Code, with hand-written cases; the authors found machine reading models struggle. [K] — [arXiv](https://arxiv.org/abs/2005.05257)
- **COLIEE** (Competition on Legal Information Extraction/Entailment, annual, held with JURISIN/ICAIL). Task 3 is statute-law retrieval over the Japanese Civil Code; Task 4 is legal textual entailment (bar-exam yes/no). [K] — [COLIEE](https://coliee.org/)
- "A Reasoning-Focused Legal Retrieval Benchmark," arXiv:2505.03970 (2025). Includes Bar Exam QA and Housing Statute QA. I believe this is Zheng, Guha et al., CSLAW 2025, but did not verify that. [S] — [arXiv](https://arxiv.org/pdf/2505.03970)
- **NitiBench+** (2026), "A Temporal Thai Legal Question Answering Benchmark Across 17 Domains," arXiv:2611.03576. Notable because it is multi-domain and temporal. [S] — [arXiv](https://arxiv.org/pdf/2611.03576). The original NitiBench (Thai legal QA, arXiv:2502.10868) is [K].

**General legal benchmarks**
- **Guha et al. (2023)**, "LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in LLMs," *NeurIPS 2023 Datasets & Benchmarks*; arXiv:2308.11462. 162 tasks across six reasoning types. [K] — [arXiv](https://arxiv.org/abs/2308.11462)
- **Pipitone & Alami (2024)**, "LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in the Legal Domain," arXiv:2408.10343. 6,858 query-answer pairs over a corpus of about 79M characters, with span-level (snippet) retrieval targets for NDAs, M&A and privacy-policy documents. It is the closest benchmark to precise-snippet retrieval, but it covers contracts rather than statutes. [K] — [arXiv](https://arxiv.org/abs/2408.10343)
- **Chalkidis et al. (2022)**, "LexGLUE: A Benchmark Dataset for Legal Language Understanding in English," *ACL 2022*; arXiv:2110.00976. Seven classification and multiple-choice tasks: ECtHR A/B, SCOTUS, EUR-LEX, LEDGAR, UNFAIR-ToS, CaseHOLD. [K] — [arXiv](https://arxiv.org/abs/2110.00976)
- **Hou et al. (2024/2025)**, "CLERC: A Dataset for Legal Case Retrieval and Retrieval-Augmented Analysis Generation," arXiv:2406.17186 (Findings of NAACL 2025). US case-law retrieval plus generation of analysis with citations; the authors report that LLMs hallucinate citations. [K] — [arXiv](https://arxiv.org/abs/2406.17186)
- **Li, Shao, Wu, Ai, Ma & Liu**, "LeCaRDv2: A Large-Scale Chinese Legal Case Retrieval Dataset," arXiv:2310.17609 (2023; published 2024; SIGIR 2024 venue not confirmed this session). 800 queries and 55,192 candidates drawn from 4.3M criminal cases. Relevance is judged on characterization, penalty and procedure by criminal-law experts. [S] — [arXiv](https://arxiv.org/pdf/2310.17609); [GitHub](https://github.com/THUIR/LeCaRDv2)
- **LRAGE: Legal Retrieval Augmented Generation Evaluation Tool**, arXiv:2504.01840 (2025). [S] — [arXiv](https://arxiv.org/pdf/2504.01840)
- **Evaluating LLM-based Approaches to Legal Citation Prediction: Domain-specific Pre-training, Fine-tuning, or RAG?** (Australian law), arXiv:2412.06272. [S] — [arXiv](https://arxiv.org/pdf/2412.06272)

**Legal LLMs**
- **Cui et al. (2023)**, "ChatLaw: Open-Source Legal LLM with Integrated External Knowledge Bases," arXiv:2306.16092. Chinese; later versions describe a mixture-of-experts, multi-agent design. [K] — [arXiv](https://arxiv.org/abs/2306.16092)
- **Yue et al. (2023)**, "DISC-LawLLM: Fine-tuning LLMs for Intelligent Legal Services," arXiv:2309.11325. Chinese; uses legal-syllogism prompting plus a retrieval module. [K] — [arXiv](https://arxiv.org/abs/2309.11325)
- **Colombo et al. (2024)**, "SaulLM-7B: A pioneering LLM for Law," arXiv:2403.03883; and "SaulLM-54B & SaulLM-141B: Scaling Up Domain Adaptation for the Legal Domain," arXiv:2407.19584 (NeurIPS 2024). English-language legal continued pretraining. [K] — [7B](https://arxiv.org/abs/2403.03883); [54B/141B](https://arxiv.org/abs/2407.19584)

**Privacy and data-protection compliance QA**
- **Ravichander et al. (2019)**, "Question Answering for Privacy Policies" (PrivacyQA), *EMNLP 2019*; arXiv:1911.00841. [K] — [arXiv](https://arxiv.org/abs/1911.00841)
- **Ahmad et al. (2020)**, "PolicyQA: A Reading Comprehension Dataset for Privacy Policies," *Findings of EMNLP 2020*; arXiv:2010.02557. [K] — [arXiv](https://arxiv.org/abs/2010.02557)
- "How Privacy-Savvy Are Large Language Models? A Case Study on Compliance and Privacy Technical Review," arXiv:2409.02375 (GDPR-oriented). [S] — [arXiv](https://arxiv.org/pdf/2409.02375)
- "Enhancing Legal Compliance and Regulation Analysis with Large Language Models," arXiv:2404.17522. [S] — [arXiv](https://arxiv.org/pdf/2404.17522)

### Inferences
- **How this project differs:** (1) The jurisdiction is Indian regulatory law, which no existing statutory-retrieval benchmark covers. (2) The output is an ordered, actionable workflow spanning several legal domains, not a single answer or a ranked list. (3) Grounding is at provision level, with a verifiable citation per step. RIRAG/ObliQA is the nearest task (regulatory obligations, Recall@10 plus a faithfulness metric). The project's 0.776 provision recall can be framed alongside RIRAG-style Recall@k, with a note that the corpus and query distributions differ. BSARD and the RIRAG overview both report BM25 as a strong baseline, which supports the hybrid BM25+dense design.

### Gaps
- I did not fetch the official RIRAG-2025 per-team leaderboard or RePASs scores (see the overview PDF: https://aclanthology.org/2025.regnlp-1.1.pdf).
- LeCaRDv2's venue (SIGIR 2024) is unconfirmed.

---

## Q3. Indian legal NLP (2021-2026)

### Takeaway
Indian legal NLP is dominated by case-law tasks: judgment prediction (ILDC, NyayaAnumana), case and statute retrieval (AILA, IL-TUR), and RAG over Supreme Court judgments. Statute-level QA is only emerging (IndicIPR-QA, 2026, covering IPR statutes only). I found one DPDP-related paper, an agentic data-anonymization framework rather than legal QA. Grounded workflow generation for Indian regulatory compliance appears to be an open niche.

### Cited Findings
- **Joshi, Paul, Sharma, Goyal, Ghosh & Modi (2024)**, "IL-TUR: Benchmark for Indian Legal Text Understanding and Reasoning," *ACL 2024 (long)*; arXiv:2407.05399. A multi-task benchmark whose tasks include statute identification (L-SI), rhetorical roles, judgment prediction, prior-case retrieval, summarization and translation. [K] (the author list was confirmed via [S] reference lists) — [arXiv](https://arxiv.org/abs/2407.05399)
- **Paul, Mandal, Goyal & Ghosh (2023)**, "Pre-trained Language Models for the Legal Domain: A Case Study on Indian Law" (InLegalBERT / InCaseLawBERT), *ICAIL 2023*; arXiv:2209.06049. Further pretraining on about 5.4M Indian legal documents. [K] — [arXiv](https://arxiv.org/abs/2209.06049)
- **Malik et al. (2021)**, "ILDC for CJPE: Indian Legal Documents Corpus for Court Judgment Prediction and Explanation," *ACL 2021*; arXiv:2105.13562. About 35k Supreme Court of India cases. [K] — [arXiv](https://arxiv.org/abs/2105.13562)
- **AILA @ FIRE 2019/2020** (Artificial Intelligence for Legal Assistance; Bhattacharya et al.): precedent retrieval and statute retrieval over a small set of Indian statutes (about 197 sections), plus rhetorical-role labelling in 2020. [K] — [FIRE](http://fire.irsi.res.in/)
- **Nigam et al. (2025)**, "NyayaAnumana & INLegalLlama: The Largest Indian Legal Judgment Prediction Dataset and Specialized Language Model for Enhanced Decision Analysis," *COLING 2025* (main.738); arXiv:2412.08385. 702,945 cases from the Supreme Court, High Courts, tribunals and district courts. INLegalLlama (LLaMA with continued pretraining plus SFT) reports about 90% F1 on prediction, with explanations. [S] — [ACL Anthology](https://aclanthology.org/2025.coling-main.738); [arXiv](https://arxiv.org/abs/2412.08385)
- **NyayaRAG** (2025), "Realistic Legal Judgment Prediction with RAG …," *IJCNLP-AACL 2025 (long)*. RAG over facts, statutes and precedent for Indian judgment prediction. [S] — [ACL Anthology PDF](https://aclanthology.org/2025.ijcnlp-long.92.pdf)
- **Islam & Abulaish (2026)**, "IndicIPR-QA: A Benchmark for Extractive Question Answering on Indian IPR Laws," arXiv:2610.40518. 4,462 QA pairs from 1,116 statutory contexts across seven IP Acts. Pairs were generated by GPT-4o, checked with nine automatic quality checks and a 200-pair human audit, and split by section to prevent leakage. This is the closest Indian statute-QA dataset, relevant to the project's IP domain. [S] — [arXiv](https://arxiv.org/abs/2610.40518)
- **Augmented Question-guided Retrieval (AQgR) of Indian Case Law with LLM, RAG, and Structured Summaries**, arXiv:2508.04710. RAG reduces hallucination in summaries of Indian judgments, and the authors argue ROUGE is inadequate for evaluating them. [S] — [arXiv](https://arxiv.org/html/2508.04710)
- **Rhetorical-Role-Aware RAG for Legal QA over Indian Supreme Court Judgments**, arXiv:2608.06828 (2026). [S] — [arXiv](https://arxiv.org/pdf/2608.06828)
- **Frontiers in Big Data (2026)**, a multi-stage RAG framework for the Indian judiciary with validation and citation generation, reporting 91.2% citation precision. [S] — [Frontiers](https://www.frontiersin.org/journals/big-data/articles/10.3389/fdata.2026.1933830/full)
- "Reducing Hallucinations in Legal AI: A RAG-Based Model for Accurate Legal Guidance" (ResearchGate, 2025). RAG over an Indian statutory corpus with six embedding-LLM pairings; domain pretraining alone did not give satisfactory citation accuracy. The venue is unclear and it is low-tier, so cite with caution. [S] — [ResearchGate](https://www.researchgate.net/publication/402849868_Reducing_Hallucinations_in_Legal_AI_A_Retrieval_Augmented_Generation-Based_Model_for_Accurate_Legal_Guidance)
- **BhashaBench V1** (2025), "A Comprehensive Benchmark for the Quadrant of Indic Domains," arXiv:2510.25409. Includes an Indian legal-domain subset. [S] — [arXiv](https://arxiv.org/pdf/2510.25409)
- **DPDP:** Kulkarni & Ramanathan (2026), "An Agentic Software Framework for Data Governance under DPDP," arXiv:2601.01101 (cs.CY, preprint). It has a KYU agent and a Compliance agent that uses LLaMA with RAG to estimate the sensitivity of data tuples, and applies masking, pseudonymization and generalization across 10 domains. Evaluation uses an "Anonymization Score"; the abstract reports no numbers. [V] — [arXiv](https://arxiv.org/abs/2601.01101)

### Inferences
- **How this project differs:** prior Indian work is about the judiciary and case law (prediction, precedent retrieval, judgment QA) or is single-statute extractive QA (IPR). This project is, as far as I can tell, the first to (a) cover Indian regulatory statutes across several domains, including the DPDP Act 2023 and Rules 2025, (b) generate business-facing compliance procedures, and (c) evaluate citation faithfulness for them. The DPDP paper (2601.01101) runs DPDP compliance logic on data. It does not answer legal questions about the DPDP Act.

### Gaps
- I found no peer-reviewed work on LLM QA over the DPDP Act or Rules 2025, GST/TDS, or Indian labour codes. That absence comes from searching and is not proof; a final ACL Anthology and Semantic Scholar keyword check ("Digital Personal Data Protection", "GST", "Companies Act") before submission is advisable.
- "Nyaya"-branded models beyond NyayaAnumana/NyayaRAG were not catalogued.

---

## Q4. Citation faithfulness, attribution evaluation, corrective RAG, abstention, LLM-judge reliability

### Takeaway
The standard toolkit is: ALCE (citation precision and recall via NLI), AIS/Attributed QA (attribution definitions), FActScore (atomic-claim support), TRUE (NLI-based factual consistency), RAGAS/ARES (automated RAG faithfulness), Self-RAG/CRAG (critique and corrective retrieval), an abstention survey (TACL 2025), and LLM-as-judge validity work. Reviewers will expect LLM-judge faithfulness numbers to be validated against human annotation with agreement statistics (Cohen's or Fleiss' kappa). PARCEL (Q1) shows that LLM judges systematically over-accept unsupported claims.

### Cited Findings
- **Gao, Yen, Yu & Chen (2023)**, "Enabling Large Language Models to Generate Text with Citations" (ALCE), *EMNLP 2023*; arXiv:2305.14627. Automatic citation recall and precision computed with an NLI model (TRUE/T5); even the best models of the time lacked full citation support roughly half the time on ELI5. [K] — [arXiv](https://arxiv.org/abs/2305.14627)
- **Rashkin et al. (2023)**, "Measuring Attribution in Natural Language Generation Models" (AIS), *Computational Linguistics* 49(4); arXiv:2112.12870. [K] — [arXiv](https://arxiv.org/abs/2112.12870)
- **Bohnet et al. (2022)**, "Attributed Question Answering: Evaluation and Modeling for Attributed Large Language Models," arXiv:2212.08037. [K] — [arXiv](https://arxiv.org/abs/2212.08037)
- **Min et al. (2023)**, "FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation," *EMNLP 2023*; arXiv:2305.14251. Decomposes output into atomic facts and checks each against a knowledge source. This is the methodological precedent for claim-level verification. [K] — [arXiv](https://arxiv.org/abs/2305.14251)
- **Honovich et al. (2022)**, "TRUE: Re-evaluating Factual Consistency Evaluation," *NAACL 2022*; arXiv:2204.04991. NLI-based consistency checking. [K] — [arXiv](https://arxiv.org/abs/2204.04991)
- **Es, James, Espinosa-Anke & Schockaert (2024)**, "RAGAS: Automated Evaluation of Retrieval Augmented Generation," *EACL 2024 System Demonstrations*; arXiv:2309.15217. Faithfulness, answer relevance, context relevance. [K] — [arXiv](https://arxiv.org/abs/2309.15217)
- **Saad-Falcon, Khattab, Potts & Zaharia (2024)**, "ARES: An Automated Evaluation Framework for RAG Systems," *NAACL 2024*; arXiv:2311.09476. Uses LLM judges calibrated with a small human-labelled set through prediction-powered inference, a directly relevant template for validating this project's judge. [K] — [arXiv](https://arxiv.org/abs/2311.09476)
- **Asai et al. (2024)**, "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection," *ICLR 2024*; arXiv:2310.11511. [K] — [arXiv](https://arxiv.org/abs/2310.11511)
- **Yan et al. (2024)**, "Corrective Retrieval Augmented Generation" (CRAG), arXiv:2401.15884. A retrieval evaluator triggers corrective actions; this is analogous to the project's grader-driven context selection. [K] — [arXiv](https://arxiv.org/abs/2401.15884)
- **Wen et al. (2025)**, "Know Your Limits: A Survey of Abstention in Large Language Models," *TACL* 2025; arXiv:2407.18418. [K] — [arXiv](https://arxiv.org/abs/2407.18418)
- **Zheng et al. (2023)**, "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena," *NeurIPS 2023 D&B*; arXiv:2306.05685. GPT-4 judges agreed with humans more than 80% of the time, comparable to human-human agreement, but showed position, verbosity and self-enhancement biases. [K] — [arXiv](https://arxiv.org/abs/2306.05685)
- **Thakur et al. (2024)**, "Judging the Judges: Evaluating Alignment and Vulnerabilities in LLMs-as-Judges," arXiv:2406.12624. Argues percent agreement can mislead and recommends Cohen's kappa. [K] — [arXiv](https://arxiv.org/abs/2406.12624)
- "What Do LLM Judges Actually Verify? Matched Controls and Record Gaps for Document Currency," arXiv:2611.01311 (2026). Relevant to judging temporal currency. [S] — [arXiv](https://arxiv.org/pdf/2611.01311)
- **PARCEL** (ICAIL 2026; see Q1): LLMs often label unsupported legal claims as supported even when given the full opinion. This is direct evidence that a legal faithfulness judge needs human calibration. [V] — [arXiv](https://arxiv.org/abs/2610.10971)

### Inferences
- **How this project differs:** the verifier runs at generation time on each workflow step against its single cited provision, combining FActScore-style decomposition with ALCE-style citation precision. Abstention is scope-based (foreign law, out-of-scope), checked deterministically and by the LLM, not only confidence-based. Main weakness: the reported 0-0.5% unsupported rate is a judgment by an LLM, and PARCEL and Thakur et al. imply reviewers will demand a human-annotated subset with kappa. An ARES-style calibration would be one way to answer that.

### Gaps
- I did not search for a legal-specific study of LLM-judge agreement with lawyers beyond PARCEL.

---

## Q5. Temporal validity of law in RAG (versioned statutes, commencement dates)

### Takeaway
This became an active topic in 2025-2026. There is SAT-Graph RAG (point-in-time retrieval over versioned norms), a German temporal statutory QA benchmark (ICAIL 2026, best-paper nominee) showing that version-filtered RAG fixes post-cutoff failures, a Chinese statute RAG with validity windows, and a temporal Thai benchmark. All of these deal with amendments and versions. I found none that handles staggered commencement or notification of provisions not yet in force, which is the DPDP Rules 2025 situation.

### Cited Findings
- **Prior, Schultz & Grabmair (2026)**, "Asking For An Old Friend: Diagnosing and Mitigating Temporal Failure Modes in LLM-based Statutory Question Answering," *ICAIL 2026* (full paper, Best Paper nominee); arXiv:2605.23497. 312 expert-validated, time-sensitive German statutory QA pairs in three categories (post-cutoff amendment, pre-amendment, multi-provision pre-amendment), five LLMs, and four settings (vanilla, web search, two RAG variants with fact-date extraction plus version filtering). Vanilla models degrade badly after the training cutoff. RAG substantially improves every type. Web search is unstable and recency-biased. Conclusion: "temporal validity as a hard constraint." [V] — [arXiv](https://arxiv.org/abs/2605.23497)
- **de Martim (2025)**, "An Ontology-Driven Graph RAG for Legal Norms: A Structural, Temporal, and Deterministic Approach" (SAT-Graph RAG), arXiv:2505.00039 (v5). Treats legal norms as a temporal knowledge graph using LRMoo/FRBRoo work vs. expression versions, and supports deterministic point-in-time retrieval of the version valid on a date. The author name was recalled from memory. [S] — [arXiv](https://arxiv.org/html/2505.00039v5)
- "Deterministic Legal Agents: A Canonical Primitive API for Auditable Reasoning over Temporal Knowledge Graphs," arXiv:2510.06002 (2025). Typed temporal-resolution primitives (valid time vs. transaction time) for legal agents. [S] — [arXiv](https://arxiv.org/html/2510.06002)
- "Can LLMs Time Travel? Enhancing Temporal Consistency in Legal …" arXiv:2605.25920 (2026). Chinese statutes: article chunks carry validity windows and retrieval is period-filtered over all versions of 16 statutes. [S] — [arXiv](https://arxiv.org/pdf/2605.25920)
- "Beyond Probabilistic Similarity: Structural, Temporal, and Causal Limitations of RAG in the Legal Domain," arXiv:2606.09724 (2026). Indexing consolidated snapshots creates near-duplicate chunks that crowd out the top-K, and timestamped triples lose which amending act caused the change. [S] — [arXiv](https://arxiv.org/pdf/2606.09724)
- **NitiBench+** (2026), a temporal Thai legal QA benchmark across 17 domains, arXiv:2611.03576. [S] — [arXiv](https://arxiv.org/pdf/2611.03576)

### Inferences
- **How this project differs:** prior temporal legal RAG handles amendment versioning (which text was valid at time t). This project handles commencement status: whether a provision is in force as of a date. The DPDP Act 2023 and its Rules 2025 have phased commencement, with some obligations notified but not yet operative. That is a distinct, India-relevant temporal failure mode. Prior et al. (ICAIL 2026) is the citation to position against, and their "hard constraint" framing matches the project's as-of-date filter.

### Gaps
- I found no work on commencement-date or not-yet-in-force handling specifically, nor any on Indian statute versioning. Several of the temporal papers are only [S]; fetch them before citing their numbers.

---

## Q6. Multi-agent and LangGraph legal agents; compliance workflow and checklist generation

### Takeaway
Multi-agent legal and compliance LLM systems exist for compliance checking: software requirements vs. regulations (ACM TOSEM), APPI data transfers, financial law with SMT solvers, and contract review. I found no peer-reviewed work on generating step-by-step compliance workflows or checklists with a per-step statutory citation, and none built on LangGraph specifically. The search summary itself noted "the research is about checking compliance, not generating checklists."

### Cited Findings
- "A Multi-Agent RAG Framework for Regulatory Compliance Checking of Software Requirements," *ACM Transactions on Software Engineering and Methodology* (TOSEM), 2025, DOI 10.1145/3785472. Retrieval over regulatory texts, rationale generation, and agent disagreement used to surface ambiguity. [S] — [ACM DL](https://dl.acm.org/doi/10.1145/3785472); [preprint](https://iris.unive.it/bitstream/10278/5115489/1/tosem2025.pdf)
- "Multi-Agent Legal Verifier Systems for Data Transfer Planning," arXiv:2511.10925 (2025). Japan's APPI; specialised agents for statutory interpretation, business context and risk, plus a synthesis step. On 200 cases, accuracy on clear-compliance cases rose from 0.160 (single agent) to 0.900 (multi-agent), per the snippet. [S] — [arXiv](https://arxiv.org/pdf/2511.10925)
- "Neuro-Symbolic Compliance: Integrating LLMs and SMT Solvers for Automated Financial Legal Analysis," arXiv:2601.06181 (2026). [S] — [arXiv](https://arxiv.org/pdf/2601.06181)
- "RAGulating Compliance: A Multi-Agent Knowledge Graph for Regulatory QA," arXiv:2508.09893. [S] — [arXiv](https://arxiv.org/html/2508.09893)
- "Multi-Agent Orchestration of Local LLMs for Contract Review under Resource Constraints," ACM, DOI 10.1145/3800227.3800266. [S] — [ACM DL](https://dl.acm.org/doi/full/10.1145/3800227.3800266)
- "From Regulation to Implementation: A Critical Evaluation of LLM-Assisted Regulatory Compliance in Industry," arXiv:2608.21317 (2026). [S] — [arXiv](https://arxiv.org/pdf/2608.21317)
- HalluDetect's "AgentBot" architecture (EMNLP 2025 Industry; Q1) is an agentic Indian legal chatbot with the fewest hallucinations among five architectures. [V] — [ACL Anthology](https://aclanthology.org/2025.emnlp-industry.128/)
- The DPDP agentic framework (arXiv:2601.01101; Q3) is multi-agent but does data governance, not legal QA. [V]

### Inferences
- **How this project differs:** prior multi-agent compliance systems give a verdict (compliant or not) on a given artefact. This project generates the procedure (ordered steps across domains) from a business question, with per-domain planners and a per-step citation and verification loop. "Compliance workflow generation with provision-level grounding" looks like a defensible novelty claim, provided it is worded as "to our knowledge."

### Gaps
- I found no LangGraph-specific peer-reviewed legal agent paper, and did not search NLLP 2024/2025 or JURIX 2024/2025 proceedings exhaustively for checklist or procedure generation. A manual scan of NLLP 2025 (ACL Anthology) and JURIX 2025 (IOS Press FAIA) tables of contents is recommended before submission.
- The numbers for the [S] multi-agent papers come from search snippets and have not been verified.
