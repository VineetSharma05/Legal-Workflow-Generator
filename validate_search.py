import os
from legal_workflow_generator.rag.hybrid_search import HybridSearcher

if "PGPASSWORD" not in os.environ:
    print("PGPASSWORD env not set. Please set it and then run the script")
    exit(1)

# ============================================================
# EXPECTED RESULTS — top 3 provision_ids per query
# ============================================================
EXPECTED_RESULTS = {
    # ── Companies Act 2013 ──────────────────────────────────────────────
    "minimum number of founders to start a private company": [
        "ca_2013_sec_3",
        "ca_2013_sec_2",
        "ca_2013_sec_3A",
    ],
    "how to register a startup in India": [
        "ca_2013_sec_7",
        "ca_2013_sec_3",
        "ca_2013_sec_12",
    ],
    "what documents are needed for company incorporation": [
        "ca_2013_sec_7",
        "ca_2013_sec_4",
        "ca_2013_sec_5",
    ],
    "memorandum of association requirements": [
        "ca_2013_sec_4",
        "ca_2013_sec_13",
        "ca_2013_sec_10",
    ],
    "articles of association internal rules": [
        "ca_2013_sec_5",
        "ca_2013_sec_14",
        "ca_2013_sec_10",
    ],
    "what happens if co-founder leaves and company has only one member": [
        "ca_2013_sec_3A",
        "ca_2013_sec_3",
        "ca_2013_sec_2",
    ],
    "how to change company name legally": [
        "ca_2013_sec_13",
        "ca_2013_sec_16",
        "ca_2013_sec_4",
    ],
    "registered office requirements for a new company": [
        "ca_2013_sec_12",
        "ca_2013_sec_10A",
        "ca_2013_sec_7",
    ],
    "can a subsidiary hold shares in its parent company": [
        "ca_2013_sec_19",
        "ca_2013_sec_2",
        "ca_2013_sec_18",
    ],
    "startup must deposit share capital before starting business": [
        "ca_2013_sec_10A",
        "ca_2013_sec_39",
        "ca_2013_sec_7",
    ],
    "penalty for not having a registered office": [
        "ca_2013_sec_12",
        "ca_2013_sec_10A",
        "ca_2013_sec_7",
    ],
    "how to convert private company to public company": [
        "ca_2013_sec_14",
        "ca_2013_sec_13",
        "ca_2013_sec_18",
    ],
    "what is a Section 8 non-profit company": [
        "ca_2013_sec_8",
        "ca_2013_sec_3",
        "ca_2013_sec_2",
    ],
    "separate legal entity after incorporation": [
        "ca_2013_sec_9",
        "ca_2013_sec_10",
        "ca_2013_sec_7",
    ],
    "promoter liability for fraud during registration": [
        "ca_2013_sec_7",
        "ca_2013_sec_34",
        "ca_2013_sec_35",
    ],
    # ── DPDP Act 2023 ───────────────────────────────────────────────────
    "what is a data fiduciary": [
        "dpdp_act_2023_sec_2",
        "dpdp_act_2023_sec_8",
        "dpdp_act_2023_sec_4",
    ],
    "user consent requirements for collecting personal data": [
        "dpdp_act_2023_sec_6",
        "dpdp_act_2023_sec_5",
        "dpdp_act_2023_sec_4",
    ],
    "how to handle personal data breach notification": [
        "dpdp_act_2023_sec_8",
        "dpdp_act_2023_sec_2",
        "dpdp_act_2023_sec_33",
    ],
    "right to erasure of personal data": [
        "dpdp_act_2023_sec_12",
        "dpdp_act_2023_sec_8",
        "dpdp_act_2023_sec_6",
    ],
    "children data protection parental consent": [
        "dpdp_act_2023_sec_9",
        "dpdp_act_2023_sec_6",
        "dpdp_act_2023_sec_8",
    ],
    "cross border data transfer restrictions India": [
        "dpdp_act_2023_sec_16",
        "dpdp_act_2023_sec_38",
        "dpdp_act_2023_sec_17",
    ],
    "significant data fiduciary obligations": [
        "dpdp_act_2023_sec_10",
        "dpdp_act_2023_sec_8",
        "dpdp_act_2023_sec_2",
    ],
    "data protection officer appointment requirements": [
        "dpdp_act_2023_sec_10",
        "dpdp_act_2023_sec_8",
        "dpdp_act_2023_sec_18",
    ],
    "user right to access their own data": [
        "dpdp_act_2023_sec_11",
        "dpdp_act_2023_sec_12",
        "dpdp_act_2023_sec_13",
    ],
    "grievance redressal mechanism for data complaints": [
        "dpdp_act_2023_sec_13",
        "dpdp_act_2023_sec_8",
        "dpdp_act_2023_sec_31",
    ],
    "penalty for violating data protection rules": [
        "dpdp_act_2023_sec_33",
        "dpdp_act_2023_sec_37",
        "dpdp_act_2023_sec_34",
    ],
    "startup exemption from data protection obligations": [
        "dpdp_act_2023_sec_17",
        "dpdp_act_2023_sec_3",
        "dpdp_act_2023_sec_7",
    ],
    "how to withdraw consent from a company": [
        "dpdp_act_2023_sec_6",
        "dpdp_act_2023_sec_8",
        "dpdp_act_2023_sec_12",
    ],
    "what is a consent manager": [
        "dpdp_act_2023_sec_6",
        "dpdp_act_2023_sec_13",
        "dpdp_act_2023_sec_2",
    ],
    "government power to block website for data violations": [
        "dpdp_act_2023_sec_37",
        "dpdp_act_2023_sec_33",
        "dpdp_act_2023_sec_36",
    ],
    # ── Copyright Act 1957 ──────────────────────────────────────────────
    "is software code protected by copyright": [
        "copyright_act_1957_sec_2",
        "copyright_act_1957_sec_13",
        "copyright_act_1957_sec_14",
    ],
    "who owns copyright of code written by an employee": [
        "copyright_act_1957_sec_17",
        "copyright_act_1957_sec_18",
        "copyright_act_1957_sec_16",
    ],
    "how to assign copyright to another person": [
        "copyright_act_1957_sec_18",
        "copyright_act_1957_sec_19",
        "copyright_act_1957_sec_19A",
    ],
    "copyright protection for computer databases": [
        "copyright_act_1957_sec_2",
        "copyright_act_1957_sec_13",
        "copyright_act_1957_sec_14",
    ],
    "what is the duration of copyright assignment if not specified": [
        "copyright_act_1957_sec_19",
        "copyright_act_1957_sec_18",
        "copyright_act_1957_sec_19A",
    ],
    "employer owns copyright of work made during employment": [
        "copyright_act_1957_sec_17",
        "copyright_act_1957_sec_18",
        "copyright_act_1957_sec_16",
    ],
    "how to transfer copyright in writing": [
        "copyright_act_1957_sec_19",
        "copyright_act_1957_sec_18",
        "copyright_act_1957_sec_19A",
    ],
    "copyright infringement of software": [
        "copyright_act_1957_sec_2",
        "copyright_act_1957_sec_14",
        "copyright_act_1957_sec_13",
    ],
    "what happens if assignee does not use the copyright": [
        "copyright_act_1957_sec_19A",
        "copyright_act_1957_sec_18",
        "copyright_act_1957_sec_19",
    ],
    "first publication of software in India": [
        "copyright_act_1957_sec_5",
        "copyright_act_1957_sec_3",
        "copyright_act_1957_sec_13",
    ],
    # ── IT Act 2000 ─────────────────────────────────────────────────────
    "are digital contracts legally valid in India": [
        "it_act_2000_sec_10A",
        "it_act_2000_sec_4",
        "it_act_2000_sec_5",
    ],
    "legal recognition of electronic signatures": [
        "it_act_2000_sec_5",
        "it_act_2000_sec_3",
        "it_act_2000_sec_3A",
    ],
    "what is an intermediary under IT Act": [
        "it_act_2000_sec_2",
        "it_act_2000_sec_11",
        "it_act_2000_sec_6A",
    ],
    "how to store electronic records legally": [
        "it_act_2000_sec_7",
        "it_act_2000_sec_4",
        "it_act_2000_sec_7A",
    ],
    "digital signature authentication requirements": [
        "it_act_2000_sec_3",
        "it_act_2000_sec_15",
        "it_act_2000_sec_16",
    ],
    "extraterritorial application of IT Act for cybercrimes": [
        "it_act_2000_sec_1",
        "it_act_2000_sec_2",
        "it_act_2000_sec_13",
    ],
    "validity of clickwrap terms of service agreement": [
        "it_act_2000_sec_10A",
        "it_act_2000_sec_4",
        "it_act_2000_sec_12",
    ],
    "government filing using electronic records": [
        "it_act_2000_sec_6",
        "it_act_2000_sec_4",
        "it_act_2000_sec_9",
    ],
    "secure electronic record definition": [
        "it_act_2000_sec_14",
        "it_act_2000_sec_15",
        "it_act_2000_sec_16",
    ],
    "who is responsible for automated emails sent by a system": [
        "it_act_2000_sec_11",
        "it_act_2000_sec_12",
        "it_act_2000_sec_13",
    ],
    # ── GST / IGST ──────────────────────────────────────────────────────
    "GST on software exports from India": [
        "igst_act_2017_sec_16",
        "igst_act_2017_sec_13",
        "igst_act_2017_sec_2",
    ],
    "what is zero rated supply for startups": [
        "igst_act_2017_sec_16",
        "igst_act_2017_sec_7",
        "igst_act_2017_sec_6",
    ],
    "IGST on inter-state supply of services": [
        "igst_act_2017_sec_5",
        "igst_act_2017_sec_7",
        "igst_act_2017_sec_12",
    ],
    "input tax credit refund for exporters": [
        "igst_act_2017_sec_16",
        "igst_act_2017_sec_18",
        "igst_act_2017_sec_19",
    ],
    "what is OIDAR online information services GST": [
        "igst_act_2017_sec_2",
        "igst_act_2017_sec_13",
        "igst_act_2017_sec_14",
    ],
    "reverse charge mechanism explained": [
        "igst_act_2017_sec_5",
        "igst_act_2017_sec_20",
        "igst_act_2017_sec_7",
    ],
    "GST liability for e-commerce platforms": [
        "igst_act_2017_sec_5",
        "igst_act_2017_sec_20",
        "igst_act_2017_sec_14",
    ],
    "foreign company selling digital services to Indian users": [
        "igst_act_2017_sec_14",
        "igst_act_2017_sec_13",
        "igst_act_2017_sec_2",
    ],
    "place of supply rules for software services": [
        "igst_act_2017_sec_13",
        "igst_act_2017_sec_12",
        "igst_act_2017_sec_2",
    ],
    "GST compensation cess on luxury goods": [
        "gst_comp_act_2017_sec_8",
        "gst_comp_act_2017_sec_10",
        "gst_comp_act_2017_sec_11",
    ],
    "input tax credit restriction for compensation cess": [
        "gst_comp_act_2017_sec_11",
        "gst_comp_act_2017_sec_8",
        "igst_act_2017_sec_18",
    ],
    "Letter of Undertaking for export without paying GST": [
        "igst_act_2017_sec_16",
        "igst_act_2017_sec_2",
        "igst_act_2017_sec_13",
    ],
    "how are GST proceeds distributed between states": [
        "igst_act_2017_sec_17",
        "igst_act_2017_sec_18",
        "gst_comp_act_2017_sec_10",
    ],
    "online gaming company GST liability": [
        "igst_act_2017_sec_14A",
        "igst_act_2017_sec_5",
        "igst_act_2017_sec_14",
    ],
    "SEZ transactions zero rated supply": [
        "igst_act_2017_sec_16",
        "igst_act_2017_sec_7",
        "igst_act_2017_sec_8",
    ],
    # ── POSH Act 2013 ───────────────────────────────────────────────────
    "how to set up internal complaints committee startup": [
        "posh_act_2013_sec_4",
        "posh_act_2013_sec_19",
        "posh_act_2013_sec_6",
    ],
    "what counts as sexual harassment at workplace": [
        "posh_act_2013_sec_2",
        "posh_act_2013_sec_3",
        "posh_act_2013_sec_19",
    ],
    "complaint filing deadline for harassment case": [
        "posh_act_2013_sec_9",
        "posh_act_2013_sec_11",
        "posh_act_2013_sec_18",
    ],
    "interim relief for harassment victim during inquiry": [
        "posh_act_2013_sec_12",
        "posh_act_2013_sec_11",
        "posh_act_2013_sec_9",
    ],
    "confidentiality rules in harassment proceedings": [
        "posh_act_2013_sec_16",
        "posh_act_2013_sec_17",
        "posh_act_2013_sec_11",
    ],
    "employer duties to prevent sexual harassment": [
        "posh_act_2013_sec_19",
        "posh_act_2013_sec_4",
        "posh_act_2013_sec_3",
    ],
    "annual report requirements under POSH Act": [
        "posh_act_2013_sec_21",
        "posh_act_2013_sec_22",
        "posh_act_2013_sec_23",
    ],
    "penalty for not forming internal complaints committee": [
        "posh_act_2013_sec_26",
        "posh_act_2013_sec_4",
        "posh_act_2013_sec_27",
    ],
    "can a contractor or intern file harassment complaint": [
        "posh_act_2013_sec_2",
        "posh_act_2013_sec_9",
        "posh_act_2013_sec_3",
    ],
    "what if complaint is false or malicious": [
        "posh_act_2013_sec_14",
        "posh_act_2013_sec_11",
        "posh_act_2013_sec_13",
    ],
    "how is compensation determined for harassment victim": [
        "posh_act_2013_sec_15",
        "posh_act_2013_sec_13",
        "posh_act_2013_sec_12",
    ],
    "conciliation process before formal harassment inquiry": [
        "posh_act_2013_sec_10",
        "posh_act_2013_sec_9",
        "posh_act_2013_sec_11",
    ],
    "appeal process after ICC decision": [
        "posh_act_2013_sec_18",
        "posh_act_2013_sec_13",
        "posh_act_2013_sec_27",
    ],
    "harassment inquiry must be completed in how many days": [
        "posh_act_2013_sec_11",
        "posh_act_2013_sec_13",
        "posh_act_2013_sec_9",
    ],
    "government inspection of workplace harassment records": [
        "posh_act_2013_sec_25",
        "posh_act_2013_sec_23",
        "posh_act_2013_sec_22",
    ],
    # ── Equal Remuneration Act 1976 ─────────────────────────────────────
    "equal pay for men and women doing same work": [
        "era_1976_sec_4",
        "era_1976_sec_2",
        "era_1976_sec_3",
    ],
    "can employer reduce salary to fix gender pay gap": [
        "era_1976_sec_4",
        "era_1976_sec_10",
        "era_1976_sec_3",
    ],
    "discrimination in hiring women prohibited": [
        "era_1976_sec_5",
        "era_1976_sec_4",
        "era_1976_sec_10",
    ],
    "penalty for paying unequal wages": [
        "era_1976_sec_10",
        "era_1976_sec_4",
        "era_1976_sec_7",
    ],
    "employer register maintenance for equal pay compliance": [
        "era_1976_sec_8",
        "era_1976_sec_9",
        "era_1976_sec_10",
    ],
    "labour inspector rights to audit workplace pay records": [
        "era_1976_sec_9",
        "era_1976_sec_8",
        "era_1976_sec_10",
    ],
    "what is remuneration under equal remuneration act": [
        "era_1976_sec_2",
        "era_1976_sec_4",
        "era_1976_sec_3",
    ],
    "women discrimination in promotion or training": [
        "era_1976_sec_5",
        "era_1976_sec_4",
        "era_1976_sec_10",
    ],
    "how to file complaint for unequal pay": [
        "era_1976_sec_7",
        "era_1976_sec_9",
        "era_1976_sec_10",
    ],
    "advisory committee for women employment opportunities": [
        "era_1976_sec_6",
        "era_1976_sec_5",
        "era_1976_sec_2",
    ],
    # ── Cross-statute / Scenario-based ─────────────────────────────────
    "founder personal liability for company debts": [
        "ca_2013_sec_3A",
        "ca_2013_sec_35",
        "ca_2013_sec_9",
    ],
    "startup data privacy and user consent obligations": [
        "dpdp_act_2023_sec_6",
        "dpdp_act_2023_sec_8",
        "dpdp_act_2023_sec_5",
    ],
    "employee IP ownership when working at a startup": [
        "copyright_act_1957_sec_17",
        "copyright_act_1957_sec_18",
        "copyright_act_1957_sec_19",
    ],
    "can a startup raise money from the public": [
        "ca_2013_sec_23",
        "ca_2013_sec_2",
        "ca_2013_sec_26",
    ],
    "what laws apply when a startup collects user data": [
        "dpdp_act_2023_sec_4",
        "dpdp_act_2023_sec_3",
        "it_act_2000_sec_2",
    ],
    "SaaS company exporting services GST treatment": [
        "igst_act_2017_sec_16",
        "igst_act_2017_sec_2",
        "igst_act_2017_sec_13",
    ],
    "startup hiring discrimination and equal pay": [
        "era_1976_sec_5",
        "era_1976_sec_4",
        "posh_act_2013_sec_2",
    ],
    "workplace safety legal obligations for tech companies": [
        "posh_act_2013_sec_19",
        "posh_act_2013_sec_4",
        "era_1976_sec_5",
    ],
    "what happens if startup submits fake documents to government": [
        "ca_2013_sec_7",
        "ca_2013_sec_34",
        "ca_2013_sec_35",
    ],
    "legal steps to launch a startup in India from scratch": [
        "ca_2013_sec_3",
        "ca_2013_sec_7",
        "ca_2013_sec_10A",
    ],
}

# ============================================================
# VALIDATION
# ============================================================


def validate_all(searcher, expected, top_k=3):
    total_queries = len(expected)
    exact_matches = 0  # all 3 correct
    partial_matches = 0  # at least 1 correct
    top1_matches = 0  # top result correct
    total_hits = 0
    total_possible = total_queries * top_k

    failed_queries = []
    passed_queries = []

    print("=" * 80)
    print(f"VALIDATION REPORT — Hybrid Search (BM25 0.3 + Semantic 0.7)")
    print(f"Total queries: {total_queries}  |  top_k={top_k}")
    print("=" * 80)

    for query, exp_ids in expected.items():
        results = searcher.search(query, top_k=top_k)
        got_ids = [r["provision_id"] for r in results]
        hits = [e for e in exp_ids if e in got_ids]
        hit_count = len(hits)
        total_hits += hit_count

        top1_ok = len(got_ids) > 0 and got_ids[0] == exp_ids[0]
        all_ok = set(got_ids) == set(exp_ids[:top_k])

        if top1_ok:
            top1_matches += 1
        if hit_count == top_k:
            exact_matches += 1
        if hit_count >= 1:
            partial_matches += 1
        else:
            failed_queries.append(query)

        if all_ok:
            passed_queries.append(query)

        # Per-query result line
        status = (
            " FULL"
            if all_ok
            else (f"  {hit_count}/{top_k}" if hit_count > 0 else " MISS")
        )
        print(f"\n[{status}] {query}")
        for i, r in enumerate(results, 1):
            marker = "correct" if r["provision_id"] in exp_ids else "wrong"
            print(
                f"  {marker} {i}. {r['provision_id']:<40} | BM25:{r['bm25_score']:.3f} Sem:{r['semantic_score']:.3f} Comb:{r['combined_score']:.3f}"
            )
        missing = [e for e in exp_ids if e not in got_ids]
        if missing:
            print(f"     Expected but missing: {missing}")

    # ── Summary ────────────────────────────────────────────────────────
    recall = total_hits / total_possible * 100
    top1_acc = top1_matches / total_queries * 100
    exact_pct = exact_matches / total_queries * 100
    partial_pct = partial_matches / total_queries * 100

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Total queries          : {total_queries}")
    print(f"  Total hits             : {total_hits} / {total_possible}")
    print(f"  Recall@{top_k}              : {recall:.1f}%")
    print(
        f"  Top-1 Accuracy         : {top1_acc:.1f}%  ({top1_matches}/{total_queries})"
    )
    print(
        f"  Full match (all {top_k}/3)  : {exact_pct:.1f}%  ({exact_matches}/{total_queries})"
    )
    print(
        f"  Partial match (>=1/3)   : {partial_pct:.1f}%  ({partial_matches}/{total_queries})"
    )
    print(f"  Complete misses        : {total_queries - partial_matches}")

    if failed_queries:
        print(f"\n   FAILED QUERIES ({len(failed_queries)}):")
        for q in failed_queries:
            print(f"     - {q}")

    print("=" * 80)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    searcher = HybridSearcher()
    searcher.build_index()
    validate_all(searcher, EXPECTED_RESULTS, top_k=3)
