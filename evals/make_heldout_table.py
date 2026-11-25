"""
Fill the held-out results table (paper Table IX) from one eval_complete_multidomain.py run.

    python evals/make_heldout_table.py --tag fresh_v4_t0 --results-dir evals/results/gemini_v4 \
        [--judged evals/results/gemini_v4/judged_fresh_v4_t0.jsonl]

Definitions (answerable = should_abstain is False; columns Single / Multi / Edge split by query type):
  Answered                 queries that were not declined
  Domain recall/precision  mean over answerable queries of the per-query domain recall / precision
  Gold-provision recall    mean over answerable queries of (gold provisions cited / gold provisions); the 95% CI is a
                           percentile bootstrap over queries (10,000 resamples, seed 0)
  Recall on in-force gold  the same, counting only gold provisions whose effective_date is on or before the run's
                           as-of date (unknown dates count as in force); queries with no in-force gold are skipped
  Statute-level recall     mean over answerable queries of (gold statutes with at least one cited provision / gold
                           statutes); a statute is the provision id up to "_sec_", "_rule_", "_reg_", ...
  Abstain accuracy         share of out-of-scope (edge) queries that were declined
  Wrongly declined         answerable queries that were declined (count and share)
  Citation precision       pooled over answerable queries: cited provisions that are gold / all cited provisions
  Entailed share           share of the pipeline's step-citation pairs labelled "entailed" by the judge file, if given
  LLM calls / tokens       per query, from the run's usage counters; latency = mean wall-clock seconds per query
"""
import argparse
import glob
import json
import random
import re
import statistics
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIT = re.compile(r"_(sec|rule|reg|dir|schedule|para|form|clause|art|regulation|order)_.*$")


def statute(pid: str) -> str:
    return UNIT.sub("", pid)


def effective_dates() -> dict:
    out = {}
    for path in glob.glob(str(ROOT / "datasets" / "complete" / "complete_*.json")):
        for o in json.load(open(path, encoding="utf-8")):
            out[o["provision_id"]] = o.get("effective_date") or None
    return out


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def boot_ci(xs, n=10000, seed=0):
    if not xs:
        return None, None
    rnd = random.Random(seed)
    ms = sorted(sum(rnd.choices(xs, k=len(xs))) / len(xs) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n) - 1]


def per_query(r, eff, as_of):
    gold, cited = r["expected_sections"], set(r["verified_citations"])
    inforce = [g for g in gold if not eff.get(g) or date.fromisoformat(str(eff[g])[:10]) <= as_of]
    gs = {statute(g) for g in gold}
    return {
        "recall": r["citation_recall"],
        "recall_inforce": (sum(g in cited for g in inforce) / len(inforce)) if inforce else None,
        "statute_recall": (sum(any(statute(c) == s for c in cited) for s in gs) / len(gs)) if gs else None,
    }


def fmt(x, pct=False, nd=3):
    if x is None:
        return "-"
    return f"{x * 100:.1f}%" if pct else f"{x:.{nd}f}"


def column(rows, eff, judged):
    ans = [r for r in rows if not r["should_abstain"]]
    edge = [r for r in rows if r["should_abstain"]]
    pq = [per_query(r, eff, date.fromisoformat(r.get("as_of") or "2026-10-15")) for r in ans]
    rec = [p["recall"] for p in pq]
    lo, hi = boot_ci(rec)
    cited = sum(len(r["verified_citations"]) for r in ans)
    hits = sum(len(set(r["verified_citations"]) & set(r["expected_sections"])) for r in ans)
    tok = [r.get("tokens", {}) for r in rows]
    c = {
        "Queries (n)": str(len(rows)),
        "Answered": f"{sum(not r['abstained'] for r in rows)}",
        "Domain recall": fmt(mean([r["domain_recall"] for r in ans])) if ans else "-",
        "Domain precision": fmt(mean([r["domain_precision"] for r in ans])) if ans else "-",
        "Gold-provision recall [95% CI]": (f"{fmt(mean(rec))} [{fmt(lo)}, {fmt(hi)}]" if ans else "-"),
        "Recall on in-force gold": fmt(mean([p["recall_inforce"] for p in pq])) if ans else "-",
        "Statute-level recall": fmt(mean([p["statute_recall"] for p in pq])) if ans else "-",
        "Abstain accuracy": (f"{sum(r['abstained'] for r in edge)}/{len(edge)}" if edge else "-"),
        "Wrongly declined": (f"{sum(r['abstained'] for r in ans)}/{len(ans)}" if ans else "-"),
        "Citation precision": fmt(hits / cited) if cited else "-",
        "Entailed share (independent judge)": "-",
        "LLM calls per query": fmt(mean([t.get("calls") for t in tok]), nd=1),
        "Tokens per query": f"{mean([t.get('total_tokens') for t in tok]):,.0f}" if tok else "-",
        "Mean latency (s)": fmt(mean([r.get("elapsed_seconds") for r in rows]), nd=1),
    }
    if judged is not None:
        qs = {r["query"] for r in rows}
        labels = [j for j in judged if j.get("query") in qs]
        if labels:
            c["Entailed share (independent judge)"] = fmt(
                sum(str(j.get("label", "")).lower() == "entailed" for j in labels) / len(labels), pct=True)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--results-dir", default="evals/results/gemini_v4")
    ap.add_argument("--judged", default=None, help="optional judge jsonl with one row per step-citation pair")
    ap.add_argument("--out", default=None, help="markdown output path (default <results-dir>/table_<tag>.md)")
    a = ap.parse_args()
    rdir = Path(a.results_dir) if Path(a.results_dir).is_absolute() else ROOT / a.results_dir
    rows = [json.loads(l) for l in open(rdir / f"eval_{a.tag}.jsonl", encoding="utf-8")]
    judged = [json.loads(l) for l in open(a.judged, encoding="utf-8")] if a.judged else None
    eff = effective_dates()
    groups = {
        "Overall": rows,
        "Single": [r for r in rows if r["category"] == "single"],
        "Multi": [r for r in rows if not r["should_abstain"] and r["category"] != "single"],
        "Edge": [r for r in rows if r["should_abstain"]],
    }
    cols = {g: column(rs, eff, judged) for g, rs in groups.items()}
    overall_only = {"Citation precision", "Entailed share (independent judge)", "LLM calls per query",
                    "Tokens per query"}
    lines = ["| Metric | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
    for m in cols["Overall"]:
        cells = []
        for g in groups:
            v = cols[g][m]
            if m in overall_only and g != "Overall":
                v = "-"
            if g == "Edge" and m not in ("Queries (n)", "Answered", "Abstain accuracy", "Mean latency (s)"):
                v = "-"
            if g != "Edge" and g != "Overall" and m == "Abstain accuracy":
                v = "-"
            cells.append(v)
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    text = "\n".join(lines)
    print(text)
    out = Path(a.out) if a.out else rdir / f"table_{a.tag}.md"
    out.write_text(text + "\n", encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
