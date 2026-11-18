#!/usr/bin/env python3
"""
Replay ONLY the generation (+ claim verification) step of saved pipeline runs.

For each sampled domain worker of a saved run (default: eval_pipeline_suggest.jsonl), rebuild the exact context
the worker had (its saved provision ids, hydrated from Postgres) and the sub-query it used (the focus query for
multi-domain questions, read from the trace), then run the CURRENT _generate_domain and _verify_claims on it.
Comparing with the saved run isolates the effect of a generator/verifier change from retrieval noise, at
~2 LLM calls per worker instead of ~11 per query.

Sample = every worker where a gold provision was in context but not cited ("skip"), plus the same number of
workers where every in-context gold provision was cited ("control", to catch regressions).

    python evals/replay_generation.py                 # default run, writes replay_<tag>.jsonl
    python evals/replay_generation.py --limit 10      # smoke test
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GEMINI_TEMPERATURE", "0")

RESULTS = Path(os.getenv("EVAL_RESULTS_DIR", ROOT / "evals" / "results" / "gemini_v2"))
_FOCUS = re.compile(r"\[(\w+)\] focus → '(.*)'$")


def workers(run_file: Path):
    rows = [json.loads(line) for line in run_file.open(encoding="utf-8")]
    skip, control = [], []
    for qi, r in enumerate(rows):
        if r["should_abstain"] or r["abstained"]:
            continue
        focus = {}
        for line in r.get("trace") or []:
            m = _FOCUS.search(line)
            if m:
                focus[m.group(1)] = m.group(2)
        cited = set(r["verified_citations"])
        for dr in r["domain_results"]:
            if dr["grade"] != "sufficient" or not dr["steps"]:
                continue
            ids = [d["provision_id"] for d in dr["docs"]]
            gold_ctx = [g for g in r["expected_sections"] if g in ids]
            if not gold_ctx:
                continue
            w = {"qi": qi, "query": r["query"], "domain": dr["domain"], "sub_q": focus.get(dr["domain"], r["query"]),
                 "ids": ids, "gold_in_context": gold_ctx, "old_cited": dr["cited"]}
            (skip if any(g not in cited for g in gold_ctx) else control).append(w)
    return skip, control[:len(skip)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run", default="eval_pipeline_suggest.jsonl")
    ap.add_argument("--tag", default="gen_v2")
    ap.add_argument("--as-of", default="2026-10-15")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import legal_workflow_generator.agent.nodes as N
    from legal_workflow_generator.agent.settings import build_config, use_settings

    use_settings(build_config(provider="gemini", as_of=args.as_of))
    N._load_domain_map()
    skip, control = workers(RESULTS / args.run)
    sample = [("skip", w) for w in skip] + [("control", w) for w in control]
    if args.limit:
        sample = sample[:args.limit]

    out = RESULTS / f"replay_{args.tag}.jsonl"
    done = set()
    if out.exists():
        done = {(j["qi"], j["domain"]) for j in map(json.loads, out.open(encoding="utf-8"))}
    print(f"{len(skip)} skip + {len(control)} control workers; {len(done)} already saved -> {out}")

    with out.open("a", encoding="utf-8") as f:
        for kind, w in sample:
            if (w["qi"], w["domain"]) in done:
                continue
            docs = N._hydrate([{"provision_id": i} for i in w["ids"]])
            by_id = {d["provision_id"]: d for d in docs}
            docs = [N.Provision.from_dict(by_id[i]) for i in w["ids"] if i in by_id]
            t0 = time.perf_counter()
            try:
                _, steps, _ = N._generate_domain(w["domain"], w["sub_q"], docs)
                kept, dropped, msg = N._verify_claims(w["domain"], steps, docs) if steps else ([], [], "no steps")
            except N.LLMUnavailable as e:
                print(f"LLM unavailable, stopping: {e}")
                return 75
            new_cited = list(dict.fromkeys(s.provision_id for s in kept))
            rec = {**w, "kind": kind, "new_cited": new_cited, "verify": msg,
                   "dropped": [s.provision_id for s in dropped], "seconds": round(time.perf_counter() - t0, 1)}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            hit_old = sum(g in w["old_cited"] for g in w["gold_in_context"])
            hit_new = sum(g in new_cited for g in w["gold_in_context"])
            print(f"[{kind:7s}] {w['domain']:20s} gold {hit_old}->{hit_new}/{len(w['gold_in_context'])} "
                  f"cites {len(w['old_cited'])}->{len(new_cited)}  {w['query'][:60]}", flush=True)
    summarize(out)
    return 0


def summarize(out: Path):
    rows = [json.loads(line) for line in out.open(encoding="utf-8")]
    for kind in ("skip", "control", None):
        sel = [r for r in rows if kind is None or r["kind"] == kind]
        if not sel:
            continue
        gold = sum(len(r["gold_in_context"]) for r in sel)
        old = sum(g in r["old_cited"] for r in sel for g in r["gold_in_context"])
        new = sum(g in r["new_cited"] for r in sel for g in r["gold_in_context"])
        oc = sum(len(r["old_cited"]) for r in sel)
        nc = sum(len(r["new_cited"]) for r in sel)
        print(f"{kind or 'all':8s} workers={len(sel):3d} gold cited {old}/{gold} -> {new}/{gold}   "
              f"citations {oc} -> {nc}   gold share of citations {old / max(oc, 1):.2f} -> {new / max(nc, 1):.2f}")


if __name__ == "__main__":
    sys.exit(main())
