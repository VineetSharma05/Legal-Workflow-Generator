"""
Evaluates LegalContextResolver's domain classification in isolation
(no retrieval, no generation) against a hand-labeled test set.

What this measures — and why it exists
---------------------------------------
The resolver's domain field comes from an LLM call, with a keyword-based
rule-based classifier used only as a fallback when that call errors out. Two
correctness signals were added on top of that (see context_resolver.py):

  - domain_agreement  : does the LLM's domain match the free, always-computed
                        rule-based domain? Disagreement is a zero-cost hint
                        that a query might be misclassified.
  - domain_confidence : when self-consistency is enabled, the LLM is sampled
                        N times at temperature>0 and majority-voted; the
                        winning vote share (e.g. 2/3) is returned here. A
                        split vote means the LLM itself isn't stable on the
                        query.

Neither signal is worth anything unless it actually correlates with being
wrong. This script is what answers that: it runs the resolver against
queries with a known correct domain and reports, on top of plain accuracy,
whether disagreement/low-confidence queries are in fact the ones the
resolver gets wrong.
"""

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Loads .env and fails fast if a required variable is missing.
import legal_workflow_generator.config.values  # noqa: F401

RESULTS_DIR = Path(__file__).resolve().parent / "results"

from legal_workflow_generator.query.normalizer import QueryNormalizer
from legal_workflow_generator.query.intent_classifier import IntentClassifier
from legal_workflow_generator.query.context_resolver import LegalContextResolver

ALL_DOMAINS = [
    "data_protection", "corporate_governance", "ip_licensing",
    "taxation", "employment", "unknown",
]

# ── Hand-written test set ────────────────────────────────────────────────────
# 8 clear-cut queries per domain, 6 boundary queries that plausibly touch two
# domains (labeled with the dominant one), and 5 off-topic/non-legal queries
# that should resolve to "unknown". Boundary and unknown cases are the ones
# most likely to produce LLM/rule-based disagreement or a split self-consistency
# vote — they're what this eval is really stress-testing.
TEST_QUERIES = [
    # data_protection
    {"query": "what is a data fiduciary under DPDP Act", "domain": "data_protection"},
    {"query": "user consent requirements for collecting personal data", "domain": "data_protection"},
    {"query": "how to handle a personal data breach notification", "domain": "data_protection"},
    {"query": "do we need a data protection officer for our startup", "domain": "data_protection"},
    {"query": "cross border transfer of user data outside India", "domain": "data_protection"},
    {"query": "children's data privacy and parental consent rules", "domain": "data_protection"},
    {"query": "penalty for violating DPDP Act provisions", "domain": "data_protection"},
    {"query": "what rights does a user have to delete their personal data", "domain": "data_protection"},

    # corporate_governance
    {"query": "how to incorporate a private limited company in India", "domain": "corporate_governance"},
    {"query": "minimum number of directors required for a private company", "domain": "corporate_governance"},
    {"query": "what is the difference between MOA and AOA", "domain": "corporate_governance"},
    {"query": "annual general meeting requirements for a startup", "domain": "corporate_governance"},
    {"query": "board resolution process for approving a new investment", "domain": "corporate_governance"},
    {"query": "how do I change my company's registered office address", "domain": "corporate_governance"},
    {"query": "shareholder rights when a new investor joins", "domain": "corporate_governance"},
    {"query": "can a foreign national be a director of an Indian company", "domain": "corporate_governance"},

    # ip_licensing
    {"query": "who owns the copyright of software written by an employee", "domain": "ip_licensing"},
    {"query": "how to license our SaaS product to another company", "domain": "ip_licensing"},
    {"query": "is using open source code in our product a legal risk", "domain": "ip_licensing"},
    {"query": "how to register a trademark for our startup name", "domain": "ip_licensing"},
    {"query": "what happens if a competitor infringes our patent", "domain": "ip_licensing"},
    {"query": "do we need an NDA before pitching to investors", "domain": "ip_licensing"},
    {"query": "assigning IP ownership from a freelancer to the company", "domain": "ip_licensing"},
    {"query": "trade secret protection for our proprietary algorithm", "domain": "ip_licensing"},

    # taxation
    {"query": "GST registration requirements for a new startup", "domain": "taxation"},
    {"query": "is GST applicable on software exported outside India", "domain": "taxation"},
    {"query": "TDS deduction rules on employee salaries", "domain": "taxation"},
    {"query": "what is angel tax and does it apply to us", "domain": "taxation"},
    {"query": "input tax credit eligibility for SaaS subscriptions", "domain": "taxation"},
    {"query": "income tax filing deadline for a private limited company", "domain": "taxation"},
    {"query": "GST implications of selling to customers in the US", "domain": "taxation"},
    {"query": "advance tax payment schedule for startups", "domain": "taxation"},

    # employment
    {"query": "how to set up an internal complaints committee under POSH Act", "domain": "employment"},
    {"query": "provident fund contribution requirements for employees", "domain": "employment"},
    {"query": "what should an offer letter legally include", "domain": "employment"},
    {"query": "equal pay obligations between male and female employees", "domain": "employment"},
    {"query": "notice period and termination rules for employees", "domain": "employment"},
    {"query": "ESOP vesting schedule legal requirements", "domain": "employment"},
    {"query": "employee state insurance eligibility criteria", "domain": "employment"},
    {"query": "maternity leave entitlement under Indian labour law", "domain": "employment"},

    # boundary — plausibly spans two domains, labeled with the dominant one
    {"query": "who owns code written by an employee that we later want to license out", "domain": "ip_licensing"},
    {"query": "GST treatment of royalty income from licensing our patent", "domain": "taxation"},
    {"query": "board approval needed before assigning copyright to a US company", "domain": "ip_licensing"},
    {"query": "an employee is leaving and taking source code with them, what are our options", "domain": "ip_licensing"},
    {"query": "do we need consent before sharing employee health data with insurers", "domain": "data_protection"},
    {"query": "tax implications of ESOP allotted to employees", "domain": "taxation"},

    # off-topic — should resolve to unknown
    {"query": "what is the best pizza place in Bangalore", "domain": "unknown"},
    {"query": "how do I improve my startup's social media engagement", "domain": "unknown"},
    {"query": "what's the weather forecast for Mumbai this week", "domain": "unknown"},
    {"query": "recommend a good CRM tool for our sales team", "domain": "unknown"},
    {"query": "how to raise a seed round from VCs", "domain": "unknown"},
]


def per_domain_metrics(results: list[dict]) -> dict:
    metrics = {}
    for d in ALL_DOMAINS:
        tp = sum(1 for r in results if r["expected_domain"] == d and r["classified_domain"] == d)
        fp = sum(1 for r in results if r["expected_domain"] != d and r["classified_domain"] == d)
        fn = sum(1 for r in results if r["expected_domain"] == d and r["classified_domain"] != d)
        support = tp + fn
        if support == 0 and tp + fp == 0:
            continue
        precision = round(tp / (tp + fp), 2) if (tp + fp) else 0.0
        recall = round(tp / (tp + fn), 2) if (tp + fn) else 0.0
        f1 = round(2 * precision * recall / (precision + recall), 2) if (precision + recall) else 0.0
        metrics[d] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return metrics


def confusion_matrix(results: list[dict]) -> dict:
    matrix = defaultdict(Counter)
    for r in results:
        matrix[r["expected_domain"]][r["classified_domain"]] += 1
    return {expected: dict(predicted) for expected, predicted in matrix.items()}


def bucket_accuracy(results: list[dict], key: str, predicate) -> tuple[int, float]:
    subset = [r for r in results if predicate(r[key])]
    if not subset:
        return 0, 0.0
    correct = sum(1 for r in subset if r["correct"])
    return len(subset), round(correct / len(subset), 2)


def evaluate(self_consistency: bool, samples: int):
    resolver = LegalContextResolver(
        self_consistency=self_consistency,
        self_consistency_samples=samples,
    )
    normalizer = QueryNormalizer()
    classifier = IntentClassifier()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total = len(TEST_QUERIES)
    results = []

    print(f"\n{'='*60}")
    print(f"DOMAIN CLASSIFICATION EVALUATION — {total} queries")
    print(f"self_consistency={self_consistency} samples={samples if self_consistency else '-'}")
    print(f"{'='*60}\n")

    for i, test in enumerate(TEST_QUERIES, 1):
        query = test["query"]
        expected_domain = test["domain"]
        print(f"[{i:02d}/{total}] {query[:60]}...")

        try:
            normalized = normalizer.normalize(text=query)
            intent, intent_confidence = classifier.classify(normalized)
            context = resolver.resolve(normalized, intent, intent_confidence)

            classified_domain = context["legal_domain"]
            rule_based_domain = context["rule_based_domain"]
            domain_agreement = context["domain_agreement"]
            domain_confidence = context["domain_confidence"]
            correct = classified_domain == expected_domain
            rule_based_correct = rule_based_domain == expected_domain

            print(
                f"       expected={expected_domain} classified={classified_domain} "
                f"({'OK' if correct else 'WRONG'}) | agreement={domain_agreement} "
                f"confidence={domain_confidence:.2f}"
            )

            results.append({
                "query": query,
                "expected_domain": expected_domain,
                "classified_domain": classified_domain,
                "rule_based_domain": rule_based_domain,
                "correct": correct,
                "rule_based_correct": rule_based_correct,
                "domain_agreement": domain_agreement,
                "domain_confidence": domain_confidence,
            })

        except Exception as e:
            print(f"       ERROR: {e}")
            results.append({"query": query, "expected_domain": expected_domain, "error": str(e)})

        time.sleep(3)  # avoid rate limiting

    scored = [r for r in results if "error" not in r]
    errored = total - len(scored)

    overall_accuracy = round(sum(r["correct"] for r in scored) / len(scored), 2) if scored else 0.0
    rule_based_accuracy = round(sum(r["rule_based_correct"] for r in scored) / len(scored), 2) if scored else 0.0
    agreement_rate = round(sum(r["domain_agreement"] for r in scored) / len(scored), 2) if scored else 0.0
    avg_confidence = round(sum(r["domain_confidence"] for r in scored) / len(scored), 2) if scored else 0.0

    n_agree, acc_when_agree = bucket_accuracy(scored, "domain_agreement", lambda v: v is True)
    n_disagree, acc_when_disagree = bucket_accuracy(scored, "domain_agreement", lambda v: v is False)
    n_unanimous, acc_when_unanimous = bucket_accuracy(scored, "domain_confidence", lambda v: v >= 0.999)
    n_split, acc_when_split = bucket_accuracy(scored, "domain_confidence", lambda v: v < 0.999)

    summary = {
        "total": total,
        "errored": errored,
        "self_consistency_enabled": self_consistency,
        "self_consistency_samples": samples if self_consistency else None,
        "overall_accuracy": overall_accuracy,
        "rule_based_only_accuracy": rule_based_accuracy,
        "agreement_rate": agreement_rate,
        "accuracy_when_llm_rule_based_agree": {"n": n_agree, "accuracy": acc_when_agree},
        "accuracy_when_llm_rule_based_disagree": {"n": n_disagree, "accuracy": acc_when_disagree},
        "avg_domain_confidence": avg_confidence,
        "accuracy_when_unanimous_vote": {"n": n_unanimous, "accuracy": acc_when_unanimous},
        "accuracy_when_split_vote": {"n": n_split, "accuracy": acc_when_split},
        "per_domain_metrics": per_domain_metrics(scored),
        "confusion_matrix": confusion_matrix(scored),
    }

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total queries                        : {total} ({errored} errored)")
    print(f"Overall accuracy (LLM/majority)       : {overall_accuracy}")
    print(f"Rule-based-only accuracy               : {rule_based_accuracy}")
    print(f"LLM/rule-based agreement rate          : {agreement_rate}")
    print(f"  accuracy when they agree             : {acc_when_agree} (n={n_agree})")
    print(f"  accuracy when they disagree          : {acc_when_disagree} (n={n_disagree})")
    if self_consistency:
        print(f"Avg self-consistency confidence        : {avg_confidence}")
        print(f"  accuracy on unanimous votes          : {acc_when_unanimous} (n={n_unanimous})")
        print(f"  accuracy on split votes              : {acc_when_split} (n={n_split})")
    print(f"{'='*60}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"eval_domain_classification_{timestamp}.json"
    csv_path = RESULTS_DIR / f"eval_domain_classification_{timestamp}.csv"

    with open(json_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "query", "expected_domain", "classified_domain", "rule_based_domain",
            "correct", "rule_based_correct", "domain_agreement", "domain_confidence",
        ])
        writer.writeheader()
        for r in results:
            if "error" not in r:
                writer.writerow({k: r.get(k, "") for k in writer.fieldnames})

    print("Results saved to:")
    print(f"  {json_path}")
    print(f"  {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LegalContextResolver domain classification, "
                    "domain_agreement, and domain_confidence against a labeled test set."
    )
    parser.add_argument(
        "--no-self-consistency",
        action="store_true",
        help="Disable self-consistency sampling (single Gemini call per query, "
             "domain_confidence will be trivially 1.0 or 0.0).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Self-consistency samples per query (default: 3). Ignored with --no-self-consistency.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(self_consistency=not args.no_self_consistency, samples=args.samples)
