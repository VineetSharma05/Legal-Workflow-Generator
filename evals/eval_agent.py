import os
import json
import time
import csv
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("PGPASSWORD", "sanj2005")

RESULTS_DIR = Path(__file__).resolve().parent / "results"

from legal_workflow_generator.agent.graph import graph

# ── 50 test queries with ground truth ────────────────────────────────────────
TEST_QUERIES = [
    # DPDP / Data Protection
    {"query": "what is a data fiduciary", "domain": "data_protection", "expected_sections": ["dpdp_act_2023_sec_2", "dpdp_act_2023_sec_4"]},
    {"query": "user consent requirements for collecting personal data", "domain": "data_protection", "expected_sections": ["dpdp_act_2023_sec_6", "dpdp_act_2023_sec_5"]},
    {"query": "how to handle personal data breach notification", "domain": "data_protection", "expected_sections": ["dpdp_act_2023_sec_8", "dpdp_act_2023_sec_33"]},
    {"query": "right to erasure of personal data", "domain": "data_protection", "expected_sections": ["dpdp_act_2023_sec_12", "dpdp_act_2023_sec_6"]},
    {"query": "children data protection parental consent", "domain": "data_protection", "expected_sections": ["dpdp_act_2023_sec_9", "dpdp_act_2023_sec_6"]},
    {"query": "cross border data transfer restrictions India", "domain": "data_protection", "expected_sections": ["dpdp_act_2023_sec_16", "dpdp_act_2023_sec_17"]},
    {"query": "significant data fiduciary obligations", "domain": "data_protection", "expected_sections": ["dpdp_act_2023_sec_10", "dpdp_act_2023_sec_8"]},
    {"query": "data protection officer appointment requirements", "domain": "data_protection", "expected_sections": ["dpdp_act_2023_sec_10", "dpdp_act_2023_sec_8"]},
    {"query": "penalty for violating data protection rules", "domain": "data_protection", "expected_sections": ["dpdp_act_2023_sec_33", "dpdp_act_2023_sec_34"]},
    {"query": "what is a consent manager", "domain": "data_protection", "expected_sections": ["dpdp_act_2023_sec_6", "dpdp_act_2023_sec_2"]},

    # Corporate Governance
    {"query": "minimum number of founders to start a private company", "domain": "corporate_governance", "expected_sections": ["ca_2013_sec_3", "ca_2013_sec_2"]},
    {"query": "what documents are needed for company incorporation", "domain": "corporate_governance", "expected_sections": ["ca_2013_sec_7", "ca_2013_sec_4"]},
    {"query": "memorandum of association requirements", "domain": "corporate_governance", "expected_sections": ["ca_2013_sec_4", "ca_2013_sec_13"]},
    {"query": "how to change company name legally", "domain": "corporate_governance", "expected_sections": ["ca_2013_sec_16", "ca_2013_sec_13"]},
    {"query": "registered office requirements for a new company", "domain": "corporate_governance", "expected_sections": ["ca_2013_sec_12", "ca_2013_sec_7"]},
    {"query": "can a subsidiary hold shares in its parent company", "domain": "corporate_governance", "expected_sections": ["ca_2013_sec_19", "ca_2013_sec_18"]},
    {"query": "how to convert private company to public company", "domain": "corporate_governance", "expected_sections": ["ca_2013_sec_14", "ca_2013_sec_18"]},
    {"query": "what is a Section 8 non-profit company", "domain": "corporate_governance", "expected_sections": ["ca_2013_sec_8", "ca_2013_sec_3"]},
    {"query": "promoter liability for fraud during registration", "domain": "corporate_governance", "expected_sections": ["ca_2013_sec_7", "ca_2013_sec_34"]},
    {"query": "founder personal liability for company debts", "domain": "corporate_governance", "expected_sections": ["ca_2013_sec_3A", "ca_2013_sec_9"]},

    # IP / Copyright
    {"query": "is software code protected by copyright", "domain": "ip_licensing", "expected_sections": ["copyright_act_1957_sec_13", "copyright_act_1957_sec_2"]},
    {"query": "who owns copyright of code written by an employee", "domain": "ip_licensing", "expected_sections": ["copyright_act_1957_sec_17", "copyright_act_1957_sec_16"]},
    {"query": "how to assign copyright to another person", "domain": "ip_licensing", "expected_sections": ["copyright_act_1957_sec_18", "copyright_act_1957_sec_19"]},
    {"query": "copyright protection for computer databases", "domain": "ip_licensing", "expected_sections": ["copyright_act_1957_sec_2", "copyright_act_1957_sec_13"]},
    {"query": "what happens if assignee does not use the copyright", "domain": "ip_licensing", "expected_sections": ["copyright_act_1957_sec_19A", "copyright_act_1957_sec_19"]},
    {"query": "copyright infringement of software", "domain": "ip_licensing", "expected_sections": ["copyright_act_1957_sec_13", "copyright_act_1957_sec_14"]},
    {"query": "employer owns copyright of work made during employment", "domain": "ip_licensing", "expected_sections": ["copyright_act_1957_sec_17", "copyright_act_1957_sec_16"]},
    {"query": "employee IP ownership when working at a startup", "domain": "ip_licensing", "expected_sections": ["copyright_act_1957_sec_17", "copyright_act_1957_sec_18"]},

    # GST / Tax
    {"query": "GST on software exports from India", "domain": "taxation", "expected_sections": ["igst_act_2017_sec_16", "igst_act_2017_sec_13"]},
    {"query": "what is zero rated supply for startups", "domain": "taxation", "expected_sections": ["igst_act_2017_sec_16", "igst_act_2017_sec_6"]},
    {"query": "input tax credit refund for exporters", "domain": "taxation", "expected_sections": ["igst_act_2017_sec_18", "igst_act_2017_sec_19"]},
    {"query": "what is OIDAR online information services GST", "domain": "taxation", "expected_sections": ["igst_act_2017_sec_14", "igst_act_2017_sec_2"]},
    {"query": "GST liability for e-commerce platforms", "domain": "taxation", "expected_sections": ["igst_act_2017_sec_5", "igst_act_2017_sec_20"]},
    {"query": "foreign company selling digital services to Indian users", "domain": "taxation", "expected_sections": ["igst_act_2017_sec_14", "igst_act_2017_sec_13"]},
    {"query": "place of supply rules for software services", "domain": "taxation", "expected_sections": ["igst_act_2017_sec_12", "igst_act_2017_sec_13"]},
    {"query": "SaaS company exporting services GST treatment", "domain": "taxation", "expected_sections": ["igst_act_2017_sec_16", "igst_act_2017_sec_13"]},
    {"query": "online gaming company GST liability", "domain": "taxation", "expected_sections": ["igst_act_2017_sec_14A", "igst_act_2017_sec_5"]},
    {"query": "Letter of Undertaking for export without paying GST", "domain": "taxation", "expected_sections": ["igst_act_2017_sec_16", "igst_act_2017_sec_2"]},

    # Employment / POSH / ERA
    {"query": "how to set up internal complaints committee startup", "domain": "employment", "expected_sections": ["posh_act_2013_sec_4", "posh_act_2013_sec_6"]},
    {"query": "what counts as sexual harassment at workplace", "domain": "employment", "expected_sections": ["posh_act_2013_sec_3", "posh_act_2013_sec_2"]},
    {"query": "complaint filing deadline for harassment case", "domain": "employment", "expected_sections": ["posh_act_2013_sec_9", "posh_act_2013_sec_18"]},
    {"query": "confidentiality rules in harassment proceedings", "domain": "employment", "expected_sections": ["posh_act_2013_sec_16", "posh_act_2013_sec_17"]},
    {"query": "employer duties to prevent sexual harassment", "domain": "employment", "expected_sections": ["posh_act_2013_sec_19", "posh_act_2013_sec_4"]},
    {"query": "equal pay for men and women doing same work", "domain": "employment", "expected_sections": ["era_1976_sec_4", "era_1976_sec_2"]},
    {"query": "discrimination in hiring women prohibited", "domain": "employment", "expected_sections": ["era_1976_sec_5", "era_1976_sec_4"]},
    {"query": "startup hiring discrimination and equal pay", "domain": "employment", "expected_sections": ["era_1976_sec_5", "era_1976_sec_4"]},

    # Edge cases — should abstain
    {"query": "what is the penalty under section 999 of DPDP Act", "domain": "data_protection", "expected_sections": []},
    {"query": "legal steps to launch a startup in India from scratch", "domain": "corporate_governance", "expected_sections": ["ca_2013_sec_3", "ca_2013_sec_7"]},
    {"query": "how are GST proceeds distributed between states", "domain": "taxation", "expected_sections": ["igst_act_2017_sec_17", "igst_act_2017_sec_18"]},
]

def evaluate():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []
    
    total = len(TEST_QUERIES)
    abstained = 0
    total_retries_retrieval = 0
    total_retries_generation = 0
    citation_hits = 0
    citation_total = 0
    domain_correct = 0

    print(f"\n{'='*60}")
    print(f"AGENTIC RAG EVALUATION — {total} queries")
    print(f"{'='*60}\n")

    for i, test in enumerate(TEST_QUERIES, 1):
        query = test["query"]
        expected = set(test["expected_sections"])
        print(f"[{i:02d}/{total}] {query[:60]}...")

        try:
            start = time.time()
            result = graph.invoke({"query": query})
            elapsed = round(time.time() - start, 2)

            verified = set(result.get("verified_citations", []))
            hits = len(verified & expected) if expected else 0
            total_exp = len(expected) if expected else 1
            recall = round(hits / total_exp, 2) if expected else None

            retries_r = result.get("retry_count_retrieval", 0)
            retries_g = result.get("retry_count_generation", 0)
            did_abstain = result.get("abstain", False)

            total_retries_retrieval  += retries_r
            total_retries_generation += retries_g
            if did_abstain:
                abstained += 1
            if recall is not None:
                citation_hits  += hits
                citation_total += total_exp

            # domain check
            classified_domain = str(result.get("domain", "")).lower()
            expected_domain   = test["domain"].lower()
            domain_match = expected_domain in classified_domain
            if domain_match:
                domain_correct += 1

            status = "ABSTAIN" if did_abstain else "✓"
            print(f"       {status} | retries_r={retries_r} retries_g={retries_g} | recall={recall} | {elapsed}s")

            results.append({
                "query": query,
                "expected_domain": test["domain"],
                "classified_domain": classified_domain,
                "domain_correct": domain_match,
                "expected_sections": list(expected),
                "verified_citations": list(verified),
                "citation_hits": hits,
                "citation_recall": recall,
                "retry_count_retrieval": retries_r,
                "retry_count_generation": retries_g,
                "abstained": did_abstain,
                "abstain_reason": result.get("abstain_reason", ""),
                "elapsed_seconds": elapsed,
                "trace": result.get("trace", []),
            })

        except Exception as e:
            print(f"       ERROR: {e}")
            results.append({"query": query, "error": str(e)})

        time.sleep(5)  # avoid rate limiting

    # ── summary ───────────────────────────────────────────────────────────────
    answered = total - abstained
    avg_retries_r = round(total_retries_retrieval / total, 2)
    avg_retries_g = round(total_retries_generation / total, 2)
    citation_recall = round(citation_hits / citation_total, 2) if citation_total else 0
    domain_acc = round(domain_correct / total * 100, 1)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total queries        : {total}")
    print(f"Answered             : {answered} ({round(answered/total*100,1)}%)")
    print(f"Abstained            : {abstained} ({round(abstained/total*100,1)}%)")
    print(f"Domain accuracy      : {domain_acc}%")
    print(f"Citation recall@k    : {citation_recall}")
    print(f"Avg retrieval retries: {avg_retries_r}")
    print(f"Avg generation retries: {avg_retries_g}")
    print(f"{'='*60}\n")

    # ── save results ──────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"eval_results_{timestamp}.json"
    csv_path  = RESULTS_DIR / f"eval_results_{timestamp}.csv"

    with open(json_path, "w") as f:
        json.dump({"summary": {
            "total": total, "answered": answered, "abstained": abstained,
            "domain_accuracy_pct": domain_acc,
            "citation_recall": citation_recall,
            "avg_retrieval_retries": avg_retries_r,
            "avg_generation_retries": avg_retries_g,
        }, "results": results}, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "query","expected_domain","classified_domain","domain_correct",
            "citation_recall","retry_count_retrieval","retry_count_generation",
            "abstained","elapsed_seconds"
        ])
        writer.writeheader()
        for r in results:
            if "error" not in r:
                writer.writerow({k: r.get(k,"") for k in writer.fieldnames})

    print(f"Results saved to:")
    print(f"  {json_path}")
    print(f"  {csv_path}")

if __name__ == "__main__":
    evaluate()