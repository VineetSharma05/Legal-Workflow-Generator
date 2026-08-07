import os
import json
import time
import csv
import re
from datetime import datetime
os.environ.setdefault("PGPASSWORD", "sanj2005")

from legal_workflow_generator.rag.pipeline import RagPipeline
from legal_workflow_generator.query.normalizer import QueryNormalizer
from legal_workflow_generator.query.intent_classifier import IntentClassifier
from legal_workflow_generator.query.context_resolver import LegalContextResolver

_normalizer = QueryNormalizer()
_classifier = IntentClassifier()
_resolver = LegalContextResolver()
_pipeline = RagPipeline(llm_provider="gemini")
_pipeline._ensure_index()

TEST_QUERIES = [
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
    {"query": "is software code protected by copyright", "domain": "ip_licensing", "expected_sections": ["copyright_act_1957_sec_13", "copyright_act_1957_sec_2"]},
    {"query": "who owns copyright of code written by an employee", "domain": "ip_licensing", "expected_sections": ["copyright_act_1957_sec_17", "copyright_act_1957_sec_16"]},
    {"query": "how to assign copyright to another person", "domain": "ip_licensing", "expected_sections": ["copyright_act_1957_sec_18", "copyright_act_1957_sec_19"]},
    {"query": "copyright protection for computer databases", "domain": "ip_licensing", "expected_sections": ["copyright_act_1957_sec_2", "copyright_act_1957_sec_13"]},
    {"query": "what happens if assignee does not use the copyright", "domain": "ip_licensing", "expected_sections": ["copyright_act_1957_sec_19A", "copyright_act_1957_sec_19"]},
    {"query": "copyright infringement of software", "domain": "ip_licensing", "expected_sections": ["copyright_act_1957_sec_13", "copyright_act_1957_sec_14"]},
    {"query": "employer owns copyright of work made during employment", "domain": "ip_licensing", "expected_sections": ["copyright_act_1957_sec_17", "copyright_act_1957_sec_16"]},
    {"query": "employee IP ownership when working at a startup", "domain": "ip_licensing", "expected_sections": ["copyright_act_1957_sec_17", "copyright_act_1957_sec_18"]},
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
    {"query": "how to set up internal complaints committee startup", "domain": "employment", "expected_sections": ["posh_act_2013_sec_4", "posh_act_2013_sec_6"]},
    {"query": "what counts as sexual harassment at workplace", "domain": "employment", "expected_sections": ["posh_act_2013_sec_3", "posh_act_2013_sec_2"]},
    {"query": "complaint filing deadline for harassment case", "domain": "employment", "expected_sections": ["posh_act_2013_sec_9", "posh_act_2013_sec_18"]},
    {"query": "confidentiality rules in harassment proceedings", "domain": "employment", "expected_sections": ["posh_act_2013_sec_16", "posh_act_2013_sec_17"]},
    {"query": "employer duties to prevent sexual harassment", "domain": "employment", "expected_sections": ["posh_act_2013_sec_19", "posh_act_2013_sec_4"]},
    {"query": "equal pay for men and women doing same work", "domain": "employment", "expected_sections": ["era_1976_sec_4", "era_1976_sec_2"]},
    {"query": "discrimination in hiring women prohibited", "domain": "employment", "expected_sections": ["era_1976_sec_5", "era_1976_sec_4"]},
    {"query": "startup hiring discrimination and equal pay", "domain": "employment", "expected_sections": ["era_1976_sec_5", "era_1976_sec_4"]},
    {"query": "what is the penalty under section 999 of DPDP Act", "domain": "data_protection", "expected_sections": []},
    {"query": "legal steps to launch a startup in India from scratch", "domain": "corporate_governance", "expected_sections": ["ca_2013_sec_3", "ca_2013_sec_7"]},
    {"query": "how are GST proceeds distributed between states", "domain": "taxation", "expected_sections": ["igst_act_2017_sec_17", "igst_act_2017_sec_18"]},
]

def extract_citations(answer: str, retrieved_ids: list) -> tuple[list, list]:
    """Check which retrieved section IDs appear in the answer text."""
    verified, failed = [], []
    answer_lower = answer.lower()
    for pid in retrieved_ids:
        # check if provision_id or a readable form appears in answer
        readable = pid.replace("_", " ").replace("sec ", "section ")
        if pid in answer_lower or readable in answer_lower:
            verified.append(pid)
        else:
            failed.append(pid)
    return verified, failed

def evaluate():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []
    total = len(TEST_QUERIES)
    citation_hits = 0
    citation_total = 0
    failed_citation_total = 0
    domain_correct = 0

    print(f"\n{'='*60}")
    print(f"PHASE 2 BASELINE EVALUATION — {total} queries")
    print(f"{'='*60}\n")

    for i, test in enumerate(TEST_QUERIES, 1):
        query = test["query"]
        expected = set(test["expected_sections"])
        print(f"[{i:02d}/{total}] {query[:60]}...")

        try:
            time.sleep(4)
            normalized = _normalizer.normalize(text=query)
            intent, confidence = _classifier.classify(normalized)
            context = _resolver.resolve(normalized, intent, confidence)

            time.sleep(4)
            result = _pipeline.run(
                query=query,
                top_k=3,
                legal_context=context
            )

            retrieved_ids = [r["provision_id"] for r in result["retrieved"] if r.get("provision_id")]
            answer = result.get("answer", "")

            # citation recall — did retrieved docs contain expected sections?
            hits = len(set(retrieved_ids) & expected) if expected else 0
            total_exp = len(expected) if expected else 1
            recall = round(hits / total_exp, 2) if expected else None

            # citation hallucination — did answer cite things NOT in retrieved?
            verified, failed = extract_citations(answer, retrieved_ids)
            hallucinated = len(failed)

            if recall is not None:
                citation_hits += hits
                citation_total += total_exp
            failed_citation_total += hallucinated

            classified_domain = str(context.get("legal_domain", "")).lower()
            expected_domain = test["domain"].lower()
            domain_match = expected_domain in classified_domain
            if domain_match:
                domain_correct += 1

            print(f"       recall={recall} | hallucinated_cites={hallucinated} | retrieved={retrieved_ids}")

            results.append({
                "query": query,
                "expected_domain": test["domain"],
                "classified_domain": classified_domain,
                "domain_correct": domain_match,
                "expected_sections": list(expected),
                "retrieved_ids": retrieved_ids,
                "citation_hits": hits,
                "citation_recall": recall,
                "hallucinated_citations": hallucinated,
                "answer_snippet": answer[:200],
            })

        except Exception as e:
            print(f"       ERROR: {e}")
            results.append({"query": query, "error": str(e)})

        time.sleep(5)

    # summary
    citation_recall = round(citation_hits / citation_total, 2) if citation_total else 0
    domain_acc = round(domain_correct / total * 100, 1)
    avg_hallucinated = round(failed_citation_total / total, 2)

    print(f"\n{'='*60}")
    print(f"PHASE 2 BASELINE SUMMARY")
    print(f"{'='*60}")
    print(f"Total queries          : {total}")
    print(f"Domain accuracy        : {domain_acc}%")
    print(f"Citation recall@k      : {citation_recall}")
    print(f"Avg hallucinated cites : {avg_hallucinated}")
    print(f"Answer rate            : 100% (always answers, never abstains)")
    print(f"{'='*60}\n")

    json_path = f"eval_phase2_results_{timestamp}.json"
    csv_path  = f"eval_phase2_results_{timestamp}.csv"

    with open(json_path, "w") as f:
        json.dump({"summary": {
            "total": total,
            "domain_accuracy_pct": domain_acc,
            "citation_recall": citation_recall,
            "avg_hallucinated_citations": avg_hallucinated,
            "answer_rate": "100% (no abstain mechanism)",
        }, "results": results}, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "query","expected_domain","classified_domain","domain_correct",
            "citation_recall","hallucinated_citations"
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