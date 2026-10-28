#!/usr/bin/env python3
"""
Step-level entailment evaluation for the baselines (and the full pipeline).

Put next to eval_baselines.py in evals/ and run from the repo root.

  # 1. regenerate baselines, this time SAVING the steps (resumable, ~100 calls each)
  python evals/eval_entailment.py gen --system llm_only
  python evals/eval_entailment.py gen --system rag_hybrid
  python evals/eval_entailment.py gen --system rag_hybrid_abstain
  (add --resume after a quota stop)

  # 2. judge every (step, cited provision) pair with an INDEPENDENT model
  python evals/eval_entailment.py judge --system llm_only --judge-model <other model>
  ... same for the other systems (and full_pre / full_final, see below)

  # 3. table + bootstrap CIs + a stratified sample for human labelling
  python evals/eval_entailment.py report

Files written to RESULTS_DIR:  steps_<system>.jsonl, judged_<system>.jsonl,
entailment_report.md, audit_sample.csv

Full pipeline: write steps_full_pre.jsonl (steps generated BEFORE the identifier /
entailment / numeric filters) and steps_full_final.jsonl (steps kept) in the same schema:
  {"query":..., "category":..., "should_abstain":bool, "abstained":bool,
   "steps":[{"action":"...", "raw_ids":["dpdp_act_2023_sec_8"]}]}
then run `judge --system full_pre` and `judge --system full_final`.
"""
import argparse
import csv
import json
import random
import re
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_complete_multidomain import RESULTS_DIR, TEST_QUERIES, category  # noqa: E402
import eval_baselines as eb  # noqa: E402

GEN_SYSTEMS = {
    "llm_only": ("llm_only", False),
    "rag_hybrid": ("rag_hybrid", False),
    "rag_hybrid_abstain": ("rag_hybrid", True),
}
ALL_SYSTEMS = ["llm_only", "rag_hybrid", "rag_hybrid_abstain", "full_pre", "full_final"]
NAMES = {
    "llm_only": "LLM only", "rag_hybrid": "Single-pass RAG",
    "rag_hybrid_abstain": "Single-pass RAG + abstain",
    "full_pre": "Full pipeline (generated, pre-verification)",
    "full_final": "Full pipeline (final steps)",
}
VERDICTS = ["entailed", "partial", "unsupported", "invalid_id", "judge_error"]


def path(kind, system):
    return RESULTS_DIR / f"{kind}_{system}.jsonl"


def read_jsonl(p):
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def clean_id(s):
    return str(s).strip().strip("[]()\"'`. ")


# ── 1. generation with steps saved ───────────────────────────────────────────
def cmd_gen(args):
    mode, allow_abstain = GEN_SYSTEMS[args.system]
    model = args.model
    if model is None:
        import legal_workflow_generator.config.values as cfg
        model = cfg.GEMINI_MODEL
    print(f"Generator model: {model}")
    provs = eb.load_corpus()
    as_of = date.fromisoformat(args.as_of)
    ret = eb.Retriever(provs, as_of, dense=True) if mode == "rag_hybrid" else None
    client = eb.make_client()
    out = path("steps", args.system)
    done = {r["query"] for r in read_jsonl(out)} if args.resume else set()
    if not args.resume and out.exists():
        out.unlink()
    for i, t in enumerate(TEST_QUERIES, 1):
        if t["query"] in done:
            continue
        retrieved = ret.top(t["query"], "hybrid", args.k) if ret else []
        prompt = eb.build_prompt(mode, t["query"], retrieved, provs, allow_abstain)
        start = time.time()
        try:
            raw = eb.call_llm(client, model, prompt)
        except eb.DailyQuota as e:
            print(f"Daily quota hit: {e}\nRerun with --resume later.")
            break
        parsed = eb.parse_json(raw)
        steps = []
        for s in parsed.get("steps") or []:
            ids = s.get("provision_id") or s.get("provision_ids") or []
            ids = [ids] if isinstance(ids, str) else list(ids)
            steps.append({"action": str(s.get("action", "")), "raw_ids": [clean_id(x) for x in ids]})
        abstained = bool(parsed.get("abstain")) or (allow_abstain and not steps)
        row = {"query": t["query"], "category": category(t), "should_abstain": t["should_abstain"],
               "abstained": abstained, "steps": [] if abstained else steps,
               "retrieved": retrieved, "raw": raw, "unparsed": not parsed,
               "elapsed_seconds": round(time.time() - start, 2), "model": model}
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{i:03d}/100] {category(t):8s} steps={len(row['steps'])} abstained={abstained}")
        time.sleep(args.sleep)


# ── 2. judging ───────────────────────────────────────────────────────────────
JUDGE_PROMPT = (
    "You are auditing legal compliance claims against statute text.\n"
    "For each ITEM decide whether the PROVISION TEXT supports the CLAIM. Judge ONLY from the "
    "provision text shown, not from your own legal knowledge.\n"
    "- entailed: the provision states or directly implies everything the claim asserts.\n"
    "- partial: the provision supports part of the claim, or the claim adds specifics "
    "(numbers, deadlines, penalties, entities, conditions) that the provision does not contain.\n"
    "- unsupported: the provision is about a different matter, does not say this, or contradicts it.\n"
    'Return JSON only: {"verdicts":[{"item":0,"verdict":"entailed|partial|unsupported","reason":"<=20 words"}]}\n\n'
)


def numbers_missing(claim, text):
    norm = lambda s: s.replace(",", "")
    nums = re.findall(r"\d+(?:\.\d+)?", norm(claim))
    body = norm(text)
    return any(not re.search(rf"(?<![\d.]){re.escape(n)}(?![\d])", body) for n in nums)


def judge_items(client, model, items):
    blocks = "\n\n".join(
        f"### ITEM {k}\nCLAIM: {it['claim']}\nPROVISION [{it['pid']}]:\n{it['text'][:12000]}"
        for k, it in enumerate(items))
    parsed = eb.parse_json(eb.call_llm(client, model, JUDGE_PROMPT + blocks))
    res = {}
    for v in parsed.get("verdicts") or []:
        try:
            res[int(v["item"])] = (str(v.get("verdict", "")).lower().strip(), str(v.get("reason", "")))
        except (KeyError, ValueError, TypeError):
            continue
    return res


def cmd_judge(args):
    if args.judge_model is None:
        sys.exit("Pass --judge-model. Use a different model (ideally a different family) from the "
                 "generator and the pipeline's own entailment checker, otherwise the judge is not independent.")
    provs = eb.load_corpus()
    lower = {p.lower(): p for p in provs}
    client = eb.make_client()
    src = read_jsonl(path("steps", args.system))
    if not src:
        sys.exit(f"No {path('steps', args.system)}")
    out = path("judged", args.system)
    done = {r["query"] for r in read_jsonl(out)} if args.resume else set()
    if not args.resume and out.exists():
        out.unlink()
    for i, r in enumerate(src, 1):
        if r["query"] in done:
            continue
        pairs, items = [], []
        for s in r["steps"]:
            for rid in s["raw_ids"] or [""]:
                pid = lower.get(rid.lower())
                p = {"claim": s["action"], "raw_id": rid, "pid": pid, "verdict": None,
                     "reason": "", "numeric_flag": False}
                if pid is None:
                    p["verdict"] = "invalid_id"
                else:
                    items.append({"claim": s["action"], "pid": pid, "text": eb.doc_text(provs[pid])})
                pairs.append(p)
        if items:
            try:
                res = judge_items(client, args.judge_model, items)
            except eb.DailyQuota as e:
                print(f"Daily quota hit: {e}\nRerun with --resume.")
                return
            k = 0
            for p in pairs:
                if p["verdict"] is not None:
                    continue
                v, why = res.get(k, ("judge_error", "missing"))
                if v not in ("entailed", "partial", "unsupported"):
                    v = "judge_error"
                p["verdict"], p["reason"] = v, why
                if v == "entailed" and numbers_missing(p["claim"], items[k]["text"]):
                    p["verdict"], p["numeric_flag"] = "partial", True
                k += 1
            time.sleep(args.sleep)
        row = {k: r[k] for k in ("query", "category", "should_abstain", "abstained")}
        row["pairs"] = pairs
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{i:03d}/{len(src)}] {len(pairs)} pairs")


# ── 3. report ────────────────────────────────────────────────────────────────
def ratio(a, b):
    return a / b if b else None


def stats(rows):
    pairs = [p for r in rows for p in r["pairs"]]
    c = Counter(p["verdict"] for p in pairs)
    valid = c["entailed"] + c["partial"] + c["unsupported"]
    answered = [r for r in rows if not r["abstained"] and r["pairs"]]
    bad = sum(any(p["verdict"] in ("unsupported", "invalid_id") for p in r["pairs"]) for r in answered)
    return {
        "pairs": len(pairs), **{v: c[v] for v in VERDICTS},
        "invalid_id_rate": ratio(c["invalid_id"], len(pairs)),
        "supported_among_valid": ratio(c["entailed"], valid),
        "unsupported_among_valid": ratio(c["unsupported"], valid),
        "answers": len(answered), "answers_with_bad_step": ratio(bad, len(answered)),
    }


def boot(rows, key, n=1000, seed=0):
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        s = stats([rng.choice(rows) for _ in rows])[key]
        if s is not None:
            vals.append(s)
    if not vals:
        return None
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def fmt(v, ci=None):
    if v is None:
        return "--"
    s = f"{v:.3f}"
    return s + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")


def cmd_report(args):
    lines = []
    all_rows = {s: read_jsonl(path("judged", s)) for s in ALL_SYSTEMS}
    all_rows = {s: r for s, r in all_rows.items() if r}
    for group, title in (("answerable", "Answerable queries (single + multi)"),
                         ("single", "Single-domain"), ("multi", "Multi-domain")):
        lines += [f"\n### {title}",
                  "| System | pairs | entailed | partial | unsupported | invalid ID | "
                  "unsupported among valid [95% CI] | answers with >=1 bad step [95% CI] |",
                  "|---|---|---|---|---|---|---|---|"]
        for s, rows in all_rows.items():
            sel = [r for r in rows if not r["should_abstain"]]
            if group == "single":
                sel = [r for r in sel if r["category"] == "single"]
            if group == "multi":
                sel = [r for r in sel if r["category"].startswith("multi_")]
            if not sel:
                continue
            st = stats(sel)
            lines.append(
                f"| {NAMES[s]} | {st['pairs']} | {st['entailed']} | {st['partial']} | {st['unsupported']} | "
                f"{st['invalid_id']} | {fmt(st['unsupported_among_valid'], boot(sel, 'unsupported_among_valid'))} | "
                f"{fmt(st['answers_with_bad_step'], boot(sel, 'answers_with_bad_step'))} |")
    lines += ["\n### Edge cases that should have been refused but were answered",
              "| System | answered of 10 | steps | invalid ID | judged unsupported |", "|---|---|---|---|---|"]
    for s, rows in all_rows.items():
        edge = [r for r in rows if r["should_abstain"]]
        if not edge:
            continue
        st = stats(edge)
        ans = sum(not r["abstained"] for r in edge)
        lines.append(f"| {NAMES[s]} | {ans} | {st['pairs']} | {st['invalid_id']} | {st['unsupported']} |")
    judge_errs = sum(stats(r)["judge_error"] for r in all_rows.values())
    lines.append(f"\nJudge errors (excluded from rates): {judge_errs}")
    text = "\n".join(lines)
    print(text)
    (RESULTS_DIR / "entailment_report.md").write_text(text, encoding="utf-8")

    # stratified audit sample for human labelling (judge validation)
    rng = random.Random(1)
    pool = {}
    for s, rows in all_rows.items():
        for r in rows:
            for p in r["pairs"]:
                if p["verdict"] in ("entailed", "partial", "unsupported"):
                    pool.setdefault((s, p["verdict"]), []).append((s, r["query"], p))
    sample = []
    for key, items in pool.items():
        rng.shuffle(items)
        sample += items[: args.audit_per_cell]
    rng.shuffle(sample)
    with open(RESULTS_DIR / "audit_sample.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "query", "claim", "provision_id", "judge_verdict", "judge_reason",
                    "human_verdict (entailed/partial/unsupported)"])
        for s, q, p in sample:
            w.writerow([s, q, p["claim"], p["pid"], p["verdict"], p["reason"], ""])
    print(f"\nWrote {len(sample)} rows to audit_sample.csv (label the last column by hand, "
          "hide the judge columns while labelling, then compute Cohen's kappa).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--system", required=True, choices=list(GEN_SYSTEMS))
    g.add_argument("--model", default=None)
    g.add_argument("--k", type=int, default=8)
    g.add_argument("--as-of", default="2026-10-15")
    g.add_argument("--sleep", type=float, default=4.0)
    g.add_argument("--resume", action="store_true")
    j = sub.add_parser("judge")
    j.add_argument("--system", required=True, choices=ALL_SYSTEMS)
    j.add_argument("--judge-model", default=None)
    j.add_argument("--sleep", type=float, default=2.0)
    j.add_argument("--resume", action="store_true")
    r = sub.add_parser("report")
    r.add_argument("--audit-per-cell", type=int, default=12)
    args = ap.parse_args()
    {"gen": cmd_gen, "judge": cmd_judge, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()