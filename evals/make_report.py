#!/usr/bin/env python3
"""
Combine every saved run in the results folder into the paper's tables (no API calls).

Reads  eval_<tag>.jsonl        pipeline runs (evals/eval_complete_multidomain.py)
       baseline_<run>.jsonl    baselines     (evals/eval_baselines.py, evals/eval_llm_pure.py)
Writes report.md and report.json in the same folder.

All systems are scored with the same rules on the same 100 queries:
  gold-provision recall   gold ids cited / gold ids, pooled over the 90 answerable queries
  statute-level recall    a gold id counts as hit when any cited provision is from the same statute
  citation precision      gold hits / valid cited ids on answerable queries
  an abstention earns no credit (nothing is shown to the user)
Confidence intervals are 95% percentile bootstrap intervals over queries (2,000 resamples, seed 0);
paired differences resample the same queries for both systems. A run that does not yet have all 100
queries is reported with its count and marked INCOMPLETE; its numbers are not final.

Usage:  EVAL_RESULTS_DIR=evals/results/gemini_v2 python evals/make_report.py
"""
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_complete_multidomain import RESULTS_DIR, TEST_QUERIES, category, summarize, per_domain  # noqa: E402

PIPELINE_RUNS = [("pipeline_base", "Pipeline (suggestions off)"),
                 ("pipeline_suggest", "Pipeline (suggestions on)"),
                 ("pipeline_suggest_r2", "Pipeline (suggestions on, repeat)")]
BASELINE_RUNS = [("llm_only", "LLM only (ids given)"), ("llm_pure", "LLM only (names only)"),
                 ("rag_bm25", "RAG, BM25"), ("rag_dense", "RAG, dense"),
                 ("rag_hybrid", "RAG, hybrid (no abstain)"), ("rag_hybrid_abstain", "RAG, hybrid (abstain)")]
N_BOOT, SEED = 2000, 0
QUERIES = [t["query"] for t in TEST_QUERIES]
TESTS = {t["query"]: t for t in TEST_QUERIES}


def load_jsonl(p: Path) -> list:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def statute_map() -> dict:
    import eval_baselines as eb
    return {pid: o.get("statute_id") or pid for pid, o in eb.load_corpus().items()}


def load_system(kind: str, name: str) -> dict:
    """{query: {"cited": set, "abstained": bool, "latency": float}} with the latest row per query."""
    rows = load_jsonl(RESULTS_DIR / (f"eval_{name}.jsonl" if kind == "pipeline" else f"baseline_{name}.jsonl"))
    out = {}
    for r in rows:
        if r["query"] not in TESTS:
            continue
        if kind == "pipeline":
            cited = set() if r["abstained"] else set(r.get("verified_citations") or [])
        else:
            cited = set() if r["abstained"] else set(r.get("valid_cited") or [])
        out[r["query"]] = {"cited": cited, "abstained": bool(r["abstained"]),
                           "latency": float(r.get("elapsed_seconds") or 0.0), "row": r}
    return out


def counts(sys_rows: dict, queries: list, statutes: dict) -> tuple:
    """(gold hits, gold total, statute-level hits, valid cited) over the answerable queries in `queries`."""
    hits = gold = shits = cited = 0
    for q in queries:
        t = TESTS[q]
        if t["should_abstain"] or q not in sys_rows:
            continue
        c = sys_rows[q]["cited"]
        exp = t["expected_sections"]
        cited_st = {statutes.get(x, x) for x in c}
        hits += sum(g in c for g in exp)
        shits += sum(statutes.get(g, g) in cited_st for g in exp)
        gold += len(exp)
        cited += len(c)
    return hits, gold, shits, cited


def recall(sys_rows, queries, statutes, statute_level=False):
    h, g, sh, _ = counts(sys_rows, queries, statutes)
    return (sh if statute_level else h) / g if g else None


def boot_ci(fn, queries, n=N_BOOT, seed=SEED):
    rng = random.Random(seed)
    vals = sorted(v for v in (fn([rng.choice(queries) for _ in queries]) for _ in range(n)) if v is not None)
    if not vals:
        return None
    return vals[int(0.025 * len(vals))], vals[min(len(vals) - 1, int(0.975 * len(vals)))]


def fmt(v, d=3):
    return "--" if v is None else f"{v:.{d}f}"


def fmt_ci(ci):
    return "" if ci is None else f" [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci[0] < 0 else f" [{ci[0]:.3f}, {ci[1]:.3f}]"


def system_metrics(rows: dict, statutes: dict) -> dict:
    ans = [q for q in QUERIES if not TESTS[q]["should_abstain"]]
    single = [q for q in ans if category(TESTS[q]) == "single"]
    multi = [q for q in ans if category(TESTS[q]).startswith("multi_")]
    edge = [q for q in QUERIES if TESTS[q]["should_abstain"]]
    h, g, sh, cited = counts(rows, ans, statutes)
    lat = [rows[q]["latency"] for q in QUERIES if q in rows]
    by_dom = {}
    for n in range(1, 6):
        qs = [q for q in ans if len(TESTS[q]["domains"]) == n]
        by_dom[n] = {"n": len(qs), "recall": recall(rows, qs, statutes)}
    present = [q for q in ans if q in rows]
    return {
        "n": len(rows), "complete": len(rows) == len(QUERIES),
        "recall": recall(rows, ans, statutes),
        "recall_ci": boot_ci(lambda qs: recall(rows, qs, statutes), present) if present else None,
        "recall_single": recall(rows, single, statutes), "recall_multi": recall(rows, multi, statutes),
        "statute_recall": recall(rows, ans, statutes, True),
        "statute_recall_single": recall(rows, single, statutes, True),
        "statute_recall_multi": recall(rows, multi, statutes, True),
        "precision": h / cited if cited else None,
        "edge_refused": sum(rows[q]["abstained"] for q in edge if q in rows), "edge_n": sum(q in rows for q in edge),
        "wrongly_declined": sum(rows[q]["abstained"] for q in ans if q in rows), "ans_n": len(present),
        "wrongly_declined_single": sum(rows[q]["abstained"] for q in single if q in rows),
        "wrongly_declined_multi": sum(rows[q]["abstained"] for q in multi if q in rows),
        "latency_mean": statistics.mean(lat) if lat else None,
        "latency_median": statistics.median(lat) if lat else None,
        "by_domains": by_dom,
    }


def paired(a: dict, b: dict, statutes: dict, subset: str) -> dict | None:
    qs = [q for q in QUERIES if not TESTS[q]["should_abstain"] and q in a and q in b]
    if subset == "multi":
        qs = [q for q in qs if category(TESTS[q]).startswith("multi_")]
    elif subset == "single":
        qs = [q for q in qs if category(TESTS[q]) == "single"]
    if not qs:
        return None

    def diff(sample):
        ra, rb = recall(a, sample, statutes), recall(b, sample, statutes)
        return None if ra is None or rb is None else ra - rb
    return {"diff": diff(qs), "ci": boot_ci(diff, qs), "n": len(qs)}


def main():
    statutes = statute_map()
    systems = []
    for name, label in PIPELINE_RUNS:
        rows = load_system("pipeline", name)
        if rows:
            systems.append(("pipeline", name, label, rows))
    for name, label in BASELINE_RUNS:
        rows = load_system("baseline", name)
        if rows:
            systems.append(("baseline", name, label, rows))
    if not systems:
        sys.exit(f"No results in {RESULTS_DIR}")

    metrics = {name: system_metrics(rows, statutes) for _, name, _, rows in systems}
    L = [f"# Evaluation report ({RESULTS_DIR})", "",
         "Gold-provision recall is pooled over the 90 answerable queries; [95% bootstrap CI over queries].", ""]

    L += ["## Systems (Table VII)", "",
          "| System | n | Recall (all) [95% CI] | Single | Multi | Stat.-level (all) | Precision | Edge refused "
          "| Wrongly declined | Latency mean / median (s) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for kind, name, label, _ in systems:
        m = metrics[name]
        flag = "" if m["complete"] else " **INCOMPLETE**"
        L.append(f"| {label}{flag} | {m['n']} | {fmt(m['recall'])}{fmt_ci(m['recall_ci'])} | {fmt(m['recall_single'])} | "
                 f"{fmt(m['recall_multi'])} | {fmt(m['statute_recall'])} | {fmt(m['precision'])} | "
                 f"{m['edge_refused']}/{m['edge_n']} | {m['wrongly_declined']}/{m['ans_n']} "
                 f"({m['wrongly_declined_single']} single, {m['wrongly_declined_multi']} multi) | "
                 f"{fmt(m['latency_mean'], 1)} / {fmt(m['latency_median'], 1)} |")

    L += ["", "## Recall by number of gold domains (Table VIII)", "",
          "| System | " + " | ".join(f"{n} dom. (n={metrics[systems[0][1]]['by_domains'][n]['n']})" for n in range(1, 6)) + " |",
          "|---|" + "---|" * 5]
    for _, name, label, _ in systems:
        L.append(f"| {label} | " + " | ".join(fmt(metrics[name]["by_domains"][n]["recall"]) for n in range(1, 6)) + " |")

    L += ["", "## Statute-level recall", "", "| System | All | Single | Multi |", "|---|---|---|---|"]
    for _, name, label, _ in systems:
        m = metrics[name]
        L.append(f"| {label} | {fmt(m['statute_recall'])} | {fmt(m['statute_recall_single'])} | "
                 f"{fmt(m['statute_recall_multi'])} |")

    pipes = [(n, l, r) for k, n, l, r in systems if k == "pipeline"]
    others = [(n, l, r) for _, n, l, r in systems]
    pair_out = {}
    if pipes:
        L += ["", "## Paired differences in gold-provision recall (pipeline minus other system)", "",
              "| Pipeline run | Against | All [95% CI] | Single [95% CI] | Multi [95% CI] |", "|---|---|---|---|---|"]
        for pn, pl, pr in pipes:
            for on, ol, orows in others:
                if on == pn:
                    continue
                cells = []
                for sub in ("all", "single", "multi"):
                    d = paired(pr, orows, statutes, sub)
                    pair_out[f"{pn}__vs__{on}__{sub}"] = d
                    cells.append("--" if not d or d["diff"] is None else f"{d['diff']:+.3f}{fmt_ci(d['ci'])}")
                L.append(f"| {pl} | {ol} | " + " | ".join(cells) + " |")

    # pipeline-only routing metrics (Table IV / V / VI), from the harness's own summary functions
    for pn, pl, pr in pipes:
        rows = [v["row"] for v in pr.values()]
        by_cat = defaultdict(list)
        for r in rows:
            by_cat["single" if r["category"] == "single" else "edge" if r["category"] == "edge" else "multi"].append(r)
            if r["category"].startswith("multi_"):
                by_cat[r["category"]].append(r)
        L += ["", f"## {pl}: routing and coverage (Tables IV-VI)", "",
              "| Metric | Overall | Single | Multi | Edge |", "|---|---|---|---|---|"]
        s_all = summarize(rows)
        s = {k: summarize(v) for k, v in by_cat.items()}
        for key, label in (("answered_pct", "Answered %"), ("domain_recall", "Domain recall"),
                           ("domain_precision", "Domain precision"), ("domain_exact_match", "Domain exact match"),
                           ("primary_domain_accuracy", "Primary domain accuracy"),
                           ("citation_domain_coverage", "Citations cover every asked domain"),
                           ("avg_elapsed_seconds", "Mean latency (s)")):
            L.append(f"| {label} | {s_all.get(key)} | " + " | ".join(
                str(s.get(c, {}).get(key, "--")) for c in ("single", "multi", "edge")) + " |")
        L += ["", "| Domains | n | Domain recall | Gold-prov. recall | Domain cover | Latency (s) |",
              "|---|---|---|---|---|---|"]
        for n in range(2, 6):
            c = s.get(f"multi_{n}")
            if c:
                L.append(f"| {n} | {c['n']} | {c['domain_recall']} | {c['citation_recall']} | "
                         f"{c['citation_domain_coverage']} | {c['avg_elapsed_seconds']} |")
        L += ["", "| Domain | n | Detect. recall | Cited recall | FP |", "|---|---|---|---|---|"]
        for d, v in per_domain(rows).items():
            L.append(f"| {d} | {v['n']} | {v['detection_recall']} | {v['cited_recall']} | {v['false_positive_detections']} |")
        tok = sum((r.get("tokens") or {}).get("total_tokens", 0) for r in rows)
        calls = sum((r.get("tokens") or {}).get("calls", 0) for r in rows)
        if calls:
            L += ["", f"LLM calls: {calls} ({calls / len(rows):.1f} per query); tokens: {tok} ({tok / len(rows):.0f} per query)."]

    # run-to-run stability when both pipeline runs with suggestions are complete
    a, b = metrics.get("pipeline_suggest"), metrics.get("pipeline_suggest_r2")
    if a and b:
        L += ["", "## Stability (two runs, same settings)", "", "| Metric | Run 1 | Run 2 | Mean | |diff| |",
              "|---|---|---|---|---|"]
        for key in ("recall", "recall_single", "recall_multi", "statute_recall", "precision"):
            if a[key] is not None and b[key] is not None:
                L.append(f"| {key} | {fmt(a[key])} | {fmt(b[key])} | {fmt((a[key] + b[key]) / 2)} | {fmt(abs(a[key] - b[key]))} |")
        for key in ("edge_refused", "wrongly_declined"):
            L.append(f"| {key} | {a[key]} | {b[key]} | {(a[key] + b[key]) / 2} | {abs(a[key] - b[key])} |")

    text = "\n".join(L) + "\n"
    (RESULTS_DIR / "report.md").write_text(text, encoding="utf-8")
    (RESULTS_DIR / "report.json").write_text(json.dumps({"systems": metrics, "paired": pair_out}, indent=2, default=list),
                                            encoding="utf-8")
    print(text)
    print(f"Saved {RESULTS_DIR / 'report.md'} and report.json")


if __name__ == "__main__":
    main()
