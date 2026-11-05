#!/usr/bin/env python3
"""
Retrieval-only evaluation of the pipeline's OWN per-domain retriever (no LLM calls).

For every answerable query of the 100-query set and every gold domain of that query, this runs the
domain worker's retrieval step exactly as the pipeline does (search -> domain filter -> in-force
handling -> act/rule cut to top_k -> anchors) with the raw query as the search text, and checks which
gold provisions of that domain end up in the worker's context. It answers "does the gold provision
reach the generator?" separately from "does the generator cite it?".

The LLM-written focus query and aspect queries are NOT used (they need LLM calls), so the numbers are
a lower bound for multi-domain queries, where the pipeline searches with a focus rewrite.

Usage (from the repo root; needs Postgres, no API key):
  python evals/eval_worker_retrieval.py                       # default settings
  python evals/eval_worker_retrieval.py --option domain_search=domain --option top_k=10
  python evals/eval_worker_retrieval.py --tag mytag           # also save per-query rows
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evals"))

from eval_complete_multidomain import RESULTS_DIR, TEST_QUERIES, category, domain_of, ratio  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--option", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--as-of", default="2026-10-15")
    ap.add_argument("--no-anchors", action="store_true", help="Leave out the hand-written anchor provisions.")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    import legal_workflow_generator.agent.nodes as N
    from legal_workflow_generator.agent.settings import build_config, use_settings

    opts = dict(o.split("=", 1) for o in args.option)
    s = use_settings(build_config(as_of=args.as_of, **opts))
    rag = N._get_rag_pipeline()
    rag._ensure_index()
    N._load_domain_map()

    buckets = defaultdict(lambda: {"hits": 0, "gold": 0, "docs": 0, "workers": 0})
    rows = []
    for t in TEST_QUERIES:
        if t["should_abstain"]:
            continue
        cat = category(t)
        group = "single" if cat == "single" else "multi"
        found = []
        for dom in t["domains"]:
            gold = [g for g in t["expected_sections"] if domain_of(g) == dom]
            cands = N._hydrate(N._fused_candidates(rag, [t["query"]], dom))
            if s.future_provisions == "exclude":
                cands = [d for d in cands if N._in_force(d, s.as_of_date)]
            docs = N._cut(cands, s)
            if not args.no_anchors:
                docs = N._with_anchors(dom, t["query"], docs)
            ids = {d.get("provision_id") for d in docs}
            hit = [g for g in gold if g in ids]
            found += hit
            for b in {group, "all", cat}:
                buckets[b]["hits"] += len(hit)
                buckets[b]["gold"] += len(gold)
                buckets[b]["docs"] += len(docs)
                buckets[b]["workers"] += 1
        rows.append({"query": t["query"], "category": cat, "gold": t["expected_sections"], "in_context": found})

    print(f"settings: {opts or 'defaults'} as_of={args.as_of} anchors={'off' if args.no_anchors else 'on'}")
    order = ["all", "single", "multi", "multi_2", "multi_3", "multi_4", "multi_5"]
    for b in order:
        v = buckets[b]
        print(f"  {b:8s} gold in worker context: {v['hits']:3d}/{v['gold']:3d} = {ratio(v['hits'], v['gold'])}"
              f"   (mean docs per worker {v['docs'] / max(v['workers'], 1):.1f})")
    if args.tag:
        out = RESULTS_DIR / f"worker_retrieval_{args.tag}.json"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"settings": opts, "as_of": args.as_of, "anchors": not args.no_anchors,
                                   "summary": {b: {**buckets[b], "recall": ratio(buckets[b]["hits"], buckets[b]["gold"])}
                                               for b in order},
                                   "rows": rows}, indent=2))
        print(f"saved {out}")


if __name__ == "__main__":
    main()
