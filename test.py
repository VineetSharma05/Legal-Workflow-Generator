# ── SLIDE 3: MAIN ORCHESTRATOR (RUN_WORKFLOW) ─────────────────────
def RUN_WORKFLOW(query, settings):
    q = normalise(query)
    intent, conf = classify_intent(q)                        # Gemini call
    primary_domain, keywords = resolve_domain(q, intent)
    if conf < 0.5 or primary_domain is UNKNOWN: return ABSTAIN

    # Sum each domain's scores over top 7 hits; keep >= 30% of strongest domain's total
    d_ret   = domains_from_retrieval(q, top_hits=7, min_ratio=0.30)
    d_llm   = domains_from_llm(query)
    domains = union(d_ret, d_llm)
    if domains == EMPTY_SET: domains = {primary_domain}      # Fallback to primary domain

    gen_retries = 0
    while gen_retries <= settings.MAX_GEN_RETRIES:
        results = parallel_map(DOMAIN_WORKER, domains)       # Returns DomainResult per domain
        if not any_domain_sufficient(results): return ABSTAIN

        cited         = VERIFY_CITATIONS(results)            # Cited IDs must be in retrieved set
        answerability = code_grade(cited, domains)           # answers | partial | off_target
        groundedness  = llm_grade(answer, retrieved)         # grounded | not_grounded | unverified

        if groundedness in {grounded, unverified} and answerability in {answers, partial}:
            gaps = {d for d in domains if not has_verified_citation(d, cited)}
            return STITCH(results, gaps, settings.as_of)
        gen_retries += 1
    return ABSTAIN


# ── SLIDE 4: PARALLEL DOMAIN WORKER (DOMAIN_WORKER) ───────────────
def DOMAIN_WORKER(domain, query, settings):
    sub_q = focus(query, domain) if multi_domain else query

    for attempt in range(0, settings.max_domain_retries + 1): # Default <= 1 retry
        # Union of up to 300 BM25 candidates and 100 nearest pgvector neighbours
        bm25_cands = bm25_search(sub_q, limit=300)           # Max-normalised to [0, 1]
        vec_cands  = pgvector_nn(sub_q, limit=100)           # all-mpnet-base-v2 across corpus
        pool       = union(bm25_cands, vec_cands)
        cands      = sort_desc(pool, score = 0.3 * bm25 + 0.7 * semantic)

        cands = keep_domain_no_forms(cands, domain)
        live  = filter(cands, in_force(settings.as_of))      # BEFORE the cut
        docs  = cut(live, act_cap=6, top_k=8) + anchors(domain, sub_q)  # API default top_k=3
        docs  = filter(docs, in_force(settings.as_of))

        if docs and llm_grade_context(sub_q, docs) == sufficient: break
        if attempt == settings.max_domain_retries: break
        sub_q = llm_rewrite(sub_q, reason)

    if grade != sufficient: return DomainResult(grade=insufficient)
    summary, steps = llm_generate(sub_q, docs)               # Steps cite provision_id
    steps = [s for s in steps if s.provision_id in docs]
    for s in steps:
        verdict = llm_entail(s, provision_text(s.provision_id))
        if verdict == unsupported: drop(s)
        if verdict == entailed and numbers_missing(s, text): verdict = partial
    return DomainResult(grade=sufficient, steps=steps, summary=summary, docs=docs)