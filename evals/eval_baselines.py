#!/usr/bin/env python3
"""
Baselines + retrieval metrics for the Legal Workflow Generator, in one file.

Place next to eval_complete_multidomain.py (evals/) and run from the repo root.
It reuses that file's 100 queries, gold labels and helpers, so every baseline is
scored on exactly the same set. No GPU and no Postgres needed: BM25 and the
mpnet embeddings are built in memory from datasets/complete/*.json (embeddings
are cached to disk after the first run).

Modes
-----
  retrieval    BM25 / dense / hybrid retrieval only: Recall@k and MRR of the gold
               provisions. ZERO API calls. Tests "is retrieval the bottleneck?".
  llm_only     (a) Gemini, no retrieval. 1 call per query.
  rag_hybrid   (b) single-pass RAG: hybrid retriever + one generation call, no
               grading / verification / abstention. 1 call per query.
  rag_bm25     (c) same, BM25 only.
  rag_dense    (d) same, dense only.
  full         the real pipeline (graph.invoke), scored with the same metrics.
  report       combine saved results into one comparison table (markdown + LaTeX).

Examples
--------
  pip install rank_bm25 sentence-transformers numpy
  python evals/eval_baselines.py --mode retrieval
  python evals/eval_baselines.py --mode rag_hybrid --sleep 4
  python evals/eval_baselines.py --mode rag_hybrid --resume      # after a quota stop
  python evals/eval_baselines.py --mode report

Runs are saved incrementally to evals/results/baseline_<mode>.jsonl and are
resumable: if the daily Gemini quota is hit the script stops cleanly; rerun with
--resume the next day.
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_complete_multidomain import (  # noqa: E402
    COMPLETE_DIR, FEMA, RESULTS_DIR, STATUTE_PREFIX_DOMAIN, TEST_QUERIES,
    category, domain_of, mean, ratio,
)

CACHE_DIR = RESULTS_DIR / "cache"
KS = (3, 8, 20)
RAG_KIND = {"rag_hybrid": "hybrid", "rag_bm25": "bm25", "rag_dense": "dense"}
ALL_MODES = ["llm_only", "rag_hybrid", "rag_bm25", "rag_dense", "full"]


# ── corpus and retrieval ─────────────────────────────────────────────────────
def load_corpus() -> dict:
    provs = {}
    for path in sorted(COMPLETE_DIR.glob("complete_*.json")):
        for o in json.loads(path.read_text(encoding="utf-8")):
            provs.setdefault(o["provision_id"], o)
    return provs


def doc_text(o: dict) -> str:
    sub = o.get("sub_structure") or {}
    parts = [o.get("title", ""), o.get("text", "")] + [str(v) for v in sub.values()]
    return " ".join(p for p in parts if p)


def in_force(o: dict, as_of: date) -> bool:
    eff = (o.get("effective_date") or "").strip()
    if not eff:
        return True  # unknown date -> treated as in force, same as the pipeline
    try:
        return date.fromisoformat(eff[:10]) <= as_of
    except ValueError:
        return True


def tok(text: str) -> list:
    return re.findall(r"[a-z0-9]+", text.lower())


class Retriever:
    """In-memory BM25 + mpnet over the whole corpus (no domain routing)."""

    def __init__(self, provs: dict, as_of: date, dense: bool = True):
        from rank_bm25 import BM25Okapi

        self.provs = provs
        self.ids = list(provs)
        self.docs = [doc_text(provs[i]) for i in self.ids]
        self.mask = np.array([in_force(provs[i], as_of) for i in self.ids])
        self.bm25 = BM25Okapi([tok(d) for d in self.docs])
        self.model, self.emb = None, None
        if dense:
            self._load_dense()

    def _load_dense(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        import hashlib  # stable digest (python's built-in str hash is salted per process)
        sig = hashlib.md5(("|".join(self.ids) + str(sum(len(d) for d in self.docs))).encode()).hexdigest()[:12]
        path = CACHE_DIR / f"mpnet_{sig}.npy"
        if path.exists():
            self.emb = np.load(path)
            return
        print(f"Embedding {len(self.docs)} provisions on CPU (one-off, cached to {path.name}) ...")
        self.emb = self.model.encode(self.docs, batch_size=32, normalize_embeddings=True,
                                     show_progress_bar=True)
        np.save(path, self.emb)

    def scores(self, query: str, kind: str) -> np.ndarray:
        b = np.asarray(self.bm25.get_scores(tok(query)), dtype=float)
        if b.max() > 0:
            b = b / b.max()
        if kind == "bm25":
            s = b
        else:
            d = self.emb @ self.model.encode([query], normalize_embeddings=True)[0]
            s = d if kind == "dense" else 0.3 * b + 0.7 * d
        return np.where(self.mask, s, -np.inf)

    def ranking(self, query: str, kind: str) -> list:
        s = self.scores(query, kind)
        return [self.ids[i] for i in np.argsort(-s) if s[i] > -np.inf]

    def top(self, query: str, kind: str, k: int) -> list:
        return self.ranking(query, kind)[:k]


# ── retrieval-only metrics (no API) ──────────────────────────────────────────
def retrieval_eval(ret: Retriever, queries: list) -> dict:
    out = {}
    for kind in ("bm25", "dense", "hybrid"):
        buckets = defaultdict(list)
        for t in queries:
            if t["should_abstain"]:
                continue
            pos = {pid: r for r, pid in enumerate(ret.ranking(t["query"], kind))}
            gold = t["expected_sections"]
            ranks = [pos.get(g, 10**9) for g in gold]
            row = {f"hits@{k}": sum(r < k for r in ranks) for k in KS}
            row.update(gold=len(gold), hit8=float(any(r < 8 for r in ranks)),
                       mrr=(1 / (min(ranks) + 1)) if min(ranks) < 10**9 else 0.0)
            cat = category(t)
            buckets["all"].append(row)
            buckets["multi" if cat.startswith("multi_") else cat].append(row)
        out[kind] = {
            c: {
                "n": len(rs),
                **{f"recall@{k}": ratio(sum(r[f"hits@{k}"] for r in rs), sum(r["gold"] for r in rs)) for k in KS},
                "any_gold_in_top8": mean([r["hit8"] for r in rs]),
                "mrr": mean([r["mrr"] for r in rs]),
            }
            for c, rs in buckets.items()
        }
    return out


# ── LLM plumbing ─────────────────────────────────────────────────────────────
class DailyQuota(Exception):
    pass


class ModelUnavailable(Exception):
    pass


def make_client():
    try:  # loads .env and uses the key pool from the project
        import legal_workflow_generator.config.values  # noqa: F401
        from legal_workflow_generator.agent.gemini_client import RotatingGeminiClient
        return RotatingGeminiClient()
    except Exception as e:  # fall back to a single plain key
        print(f"(RotatingGeminiClient unavailable: {e}; using GEMINI_API_KEY directly)")
        from google import genai
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        key = os.environ.get("GEMINI_API_KEY", "").split(",")[0].strip()
        return genai.Client(api_key=key)


def call_llm(client, model: str, prompt: str, retries: int = 5) -> str:
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=model, contents=prompt,
                config={"temperature": 0, "response_mime_type": "application/json"})
            return getattr(resp, "text", None) or str(resp)
        except Exception as e:
            msg = str(e)
            if "NOT_FOUND" in msg or "no longer available" in msg:
                raise ModelUnavailable(msg)
            if "PerDay" in msg or "per day" in msg.lower():
                raise DailyQuota(msg)
            if attempt == retries - 1:
                raise
            m = re.search(r"retry in ([\d.]+)s", msg)
            time.sleep(min(float(m.group(1)) + 1 if m else 3 * 2 ** attempt, 90))


def parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}


HINTS = {
    "ca_2013_": "Companies Act 2013", "css_2020_": "Code on Social Security 2020",
    "oshwc_2020_": "OSH and Working Conditions Code 2020", "esia_1948_": "ESI Act 1948",
    "fa_1948_": "Factories Act 1948", "ida_1947_": "Industrial Disputes Act 1947",
    "epf_1952_": "EPF and MP Act 1952", "poga_1972_": "Payment of Gratuity Act 1972",
    "poba_1965_": "Payment of Bonus Act 1965", "mba_1961_": "Maternity Benefit Act 1961",
    "posh_2013_": "POSH Act 2013", "rpwd_2016_": "Rights of Persons with Disabilities Act 2016",
    "cla_1970_": "Contract Labour Act 1970", "app_1961_": "Apprentices Act 1961",
    "pow_1936_": "Payment of Wages Act 1936", "bocw_1996_": "BOCW Act 1996",
    "eca_1923_": "Employees Compensation Act 1923", "cicra_2005_": "Credit Information Companies Act 2005",
    "rti_act_2005_": "RTI Act 2005", "it_act_2000_": "IT Act 2000 (IP-domain copy)",
}
ID_FORMAT = (
    "Identifier format: <prefix> + 'sec_<number>' for sections (e.g. ca_2013_sec_135), "
    "'rule_<n>' for rules, 'reg_<n>' for regulations, 'dir_<n>' for directions, 'sch_<n>' for schedules."
)


def prefix_list() -> str:
    prefixes = sorted(set(STATUTE_PREFIX_DOMAIN) | {FEMA})
    return "\n".join(f"- {p}" + (f"  ({HINTS[p]})" if p in HINTS else "") for p in prefixes)


def build_prompt(mode: str, query: str, ctx: list, provs: dict, allow_abstain: bool) -> str:
    abstain = ('\nIf the provisions do not answer the question, return {"abstain": true, "reason": "..."} '
               'instead of steps.' if allow_abstain else "")
    if mode == "llm_only":
        ab = ('\nIf you cannot answer from Indian statutes, return {"abstain": true, "reason": "..."}.'
              if allow_abstain else "")
        return (
            "You are an assistant for Indian startup compliance questions. Answer from your own "
            "knowledge of Indian law as an ordered list of compliance steps. For each step cite the "
            "supporting provision using the identifier scheme below.\n"
            f"{ID_FORMAT}\nKnown statute prefixes:\n{prefix_list()}\n{ab}\n"
            'Return JSON only: {"steps":[{"action":"...","provision_id":"..."}]}\n'
            f"Question: {query}"
        )
    blocks = "\n\n".join(
        f"[{pid}] {provs[pid].get('title', '')}\n{doc_text(provs[pid])[:1500]}" for pid in ctx)
    return (
        "You are an assistant for Indian startup compliance questions. Using ONLY the provisions "
        "below, answer as an ordered list of compliance steps. Each step must cite the provision_id "
        f"exactly as shown in square brackets.{abstain}\n"
        'Return JSON only: {"steps":[{"action":"...","provision_id":"..."}]}\n\n'
        f"PROVISIONS:\n{blocks}\n\nQuestion: {query}"
    )


def extract_cited(parsed: dict, raw: str, provs: dict, lower: dict) -> list:
    cited = []
    for s in parsed.get("steps") or []:
        ids = s.get("provision_id") or s.get("provision_ids") or []
        cited += [ids] if isinstance(ids, str) else list(ids)
    if not cited:  # fall back to any corpus id mentioned in the raw text
        cited = [t for t in re.findall(r"[A-Za-z0-9_]+", raw) if t.lower() in lower]
    out = []
    for c in cited:
        # models often echo ids as "[ca_2013_sec_7]", with quotes, or several ids in one string
        for piece in re.split(r"[,;\s]+", str(c).strip()):
            piece = piece.strip("[]()\"'`.")
            if piece:
                out.append(lower.get(piece.lower(), piece))
    return sorted(set(out))


# ── running one query ────────────────────────────────────────────────────────
def make_row(t: dict, cited: list, provs: dict, abstained: bool, reason: str,
             elapsed: float, retrieved: list) -> dict:
    valid = [c for c in cited if c in provs]
    expected = set(t["expected_sections"])
    return {
        "query": t["query"], "category": category(t), "should_abstain": t["should_abstain"],
        "expected_domains": sorted(set(t["domains"]) - {"unknown"}),
        "expected_sections": sorted(expected), "cited": cited, "valid_cited": valid,
        "hits": len(set(valid) & expected),
        "cited_domains": sorted({domain_of(c) for c in valid} - {None}),
        "abstained": abstained, "abstain_reason": reason,
        "ungrounded": [c for c in valid if retrieved and c not in retrieved],
        "retrieved": retrieved, "elapsed_seconds": round(elapsed, 2),
    }


def run_query(t, mode, ctx):
    start = time.time()
    if mode == "full":
        r = ctx["graph"].invoke({"query": t["query"]})
        return make_row(t, sorted(set(r.get("verified_citations", []))), ctx["provs"],
                        bool(r.get("abstain", False)), r.get("abstain_reason", ""),
                        time.time() - start, [])
    retrieved = []
    if mode in RAG_KIND:
        retrieved = ctx["ret"].top(t["query"], RAG_KIND[mode], ctx["k"])
    prompt = build_prompt(mode, t["query"], retrieved, ctx["provs"], ctx["allow_abstain"])
    raw = call_llm(ctx["client"], ctx["model"], prompt)
    parsed = parse_json(raw)
    abstained = bool(parsed.get("abstain")) or (ctx["allow_abstain"] and not (parsed.get("steps")))
    cited = [] if abstained else extract_cited(parsed, raw, ctx["provs"], ctx["lower"])
    row = make_row(t, cited, ctx["provs"], abstained, str(parsed.get("reason", "")),
                   time.time() - start, retrieved)
    row["model"] = ctx["model"]
    row["unparsed"] = not parsed  # model output was not usable JSON
    return row


# ── metrics ──────────────────────────────────────────────────────────────────
def metrics(rows: list) -> dict:
    ans = [r for r in rows if not r["should_abstain"]]
    edge = [r for r in rows if r["should_abstain"]]
    cited = sum(len(r["cited"]) for r in rows)
    valid = sum(len(r["valid_cited"]) for r in rows)
    valid_ans = sum(len(r["valid_cited"]) for r in ans)
    cov = [ratio(len(set(r["cited_domains"]) & set(r["expected_domains"])), len(r["expected_domains"]))
           for r in ans]
    ung = sum(len(r["ungrounded"]) for r in rows)
    return {
        "n": len(rows),
        "gold_provision_recall": ratio(sum(r["hits"] for r in ans), sum(len(r["expected_sections"]) for r in ans)),
        "citation_precision": ratio(sum(r["hits"] for r in ans), valid_ans),
        "citation_validity": ratio(valid, cited),
        "ungrounded_citation_rate": ratio(ung, valid) if any(r["retrieved"] for r in rows) else None,
        "domain_coverage": mean(cov),
        "answered_pct": ratio(sum(not r["abstained"] for r in rows) * 100, len(rows)),
        "wrongly_declined": ratio(sum(r["abstained"] for r in ans), len(ans)),
        "abstain_accuracy": ratio(sum(r["abstained"] for r in edge), len(edge)),
        "avg_latency_s": mean([r["elapsed_seconds"] for r in rows]),
        "unparsed_outputs": sum(bool(r.get("unparsed")) for r in rows),
    }


def summarize(rows: list) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups["overall"].append(r)
        groups["multi" if r["category"].startswith("multi_") else r["category"]].append(r)
    return {g: metrics(rs) for g, rs in groups.items()}


# ── report ───────────────────────────────────────────────────────────────────
def load_rows(mode: str) -> list:
    path = RESULTS_DIR / f"baseline_{mode}.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def report_old():
    names = {"llm_only": "LLM, no retrieval", "rag_hybrid": "Single-pass RAG (hybrid)",
             "rag_bm25": "BM25 only", "rag_dense": "Dense only", "full": "Full pipeline"}
    cols = ["gold_provision_recall", "citation_precision", "citation_validity",
            "abstain_accuracy", "wrongly_declined", "avg_latency_s"]
    print("| System | n | Gold-prov. recall | Cit. precision | Cit. validity | Abstain acc. | Wrongly declined | Latency (s) |")
    print("|---|---|---|---|---|---|---|---|")
    tex = []
    for mode in ALL_MODES:
        rows = load_rows(mode)
        if not rows:
            continue
        m = summarize(rows)["overall"]
        vals = [m[c] if m[c] is not None else "--" for c in cols]
        print(f"| {names[mode]} | {m['n']} | " + " | ".join(str(v) for v in vals) + " |")
        tex.append(f"{names[mode]} & " + " & ".join(str(v) for v in vals) + r" \\")
    print("\nLaTeX rows:\n" + "\n".join(tex))


# ── main ─────────────────────────────────────────────────────────────────────
def report():
    """One combined view: every system side by side, per category, plus retrieval metrics."""
    names = {"llm_only": "LLM, no retrieval", "rag_hybrid": "Single-pass RAG (hybrid)",
             "rag_bm25": "BM25 only", "rag_dense": "Dense only", "full": "Full pipeline"}
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    def f(v):
        return "--" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))

    data = {m: summarize(load_rows(m)) for m in ALL_MODES if load_rows(m)}
    if not data:
        print("No saved results found in", RESULTS_DIR)
    for group, title in (("overall", "Overall"), ("single", "Single-domain"),
                         ("multi", "Multi-domain"), ("edge", "Edge cases (should abstain)")):
        out(f"\n### {title}")
        out("| System | n | Gold-prov. recall | Cit. precision | Cit. validity | Domain cover "
            "| Abstain acc. | Wrongly declined | Latency (s) |")
        out("|---|---|---|---|---|---|---|---|---|")
        for mode, summ in data.items():
            m = summ.get(group)
            if m:
                out(f"| {names[mode]} | {m['n']} | {f(m['gold_provision_recall'])} | "
                    f"{f(m['citation_precision'])} | {f(m['citation_validity'])} | "
                    f"{f(m['domain_coverage'])} | {f(m['abstain_accuracy'])} | "
                    f"{f(m['wrongly_declined'])} | {f(m['avg_latency_s'])} |")

    rpath = RESULTS_DIR / "baseline_retrieval_metrics.json"
    if rpath.exists():
        ret = json.loads(rpath.read_text())
        out("\n### Retrieval only (no LLM): gold provisions in the ranked list")
        out("| Retriever | Queries | Subset | Recall@3 | Recall@8 | Recall@20 | Any gold in top 8 | MRR |")
        out("|---|---|---|---|---|---|---|---|")
        for kind, groups in ret.items():
            for g in ("all", "single", "multi"):
                m = groups.get(g)
                if m:
                    out(f"| {kind} | {m['n']} | {g} | {f(m['recall@3'])} | {f(m['recall@8'])} | "
                        f"{f(m['recall@20'])} | {f(m['any_gold_in_top8'])} | {f(m['mrr'])} |")

    out("\n### LaTeX rows (overall)")
    for mode, summ in data.items():
        m = summ["overall"]
        out(f"{names[mode]} & {f(m['gold_provision_recall'])} & {f(m['citation_validity'])} & "
            f"{f(m['abstain_accuracy'])} & {f(m['wrongly_declined'])} \\\\")

    path = RESULTS_DIR / "baseline_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved {path}")


def select(args) -> list:
    sel = [t for t in TEST_QUERIES
           if args.category == "all"
           or (args.category == "multi" and category(t).startswith("multi_"))
           or category(t) == args.category]
    return sel[: args.limit] if args.limit else sel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["retrieval", "report"] + ALL_MODES)
    ap.add_argument("--model", default=None,
                    help="Model name. Defaults to the pipeline's own GEMINI_MODEL setting, so the "
                         "baselines and the full pipeline use the same model.")
    ap.add_argument("--k", type=int, default=8, help="Provisions given to the generator (pipeline uses 8).")
    ap.add_argument("--as-of", default="2026-10-15", help="Query date for the in-force filter.")
    ap.add_argument("--category", choices=["all", "single", "multi", "edge"], default="all")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=4.0, help="Seconds between LLM calls.")
    ap.add_argument("--allow-abstain", action="store_true",
                    help="Tell the baseline it may abstain. Default off (the paper defines the "
                         "baselines as having no abstention).")
    ap.add_argument("--resume", action="store_true", help="Skip queries already saved for this mode.")
    args = ap.parse_args()

    if args.mode == "report":
        return report()

    if args.mode != "full":
        # The baselines read the JSON corpus, not Postgres. The project config still insists
        # these exist, so fill in placeholders (a real value in the environment wins).
        for var, val in (("PGPASSWORD", "unused"), ("PGHOST", "localhost"), ("PGUSER", "unused"),
                         ("PGDATABASE", "unused"), ("PGPORT", "5432")):
            os.environ.setdefault(var, val)

    if args.model is None and args.mode not in ("retrieval", "full"):
        try:
            import legal_workflow_generator.config.values as cfg
            args.model = cfg.GEMINI_MODEL
        except Exception:
            sys.exit("Could not read GEMINI_MODEL from the project config. "
                     "Pass it explicitly, e.g. --model <name used by the pipeline>.")
        print(f"Using the pipeline's model: {args.model}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    queries = select(args)
    provs = load_corpus()
    as_of = date.fromisoformat(args.as_of)
    print(f"{len(provs)} provisions loaded; {len(queries)} queries; as_of={as_of}")

    if args.mode == "retrieval":
        ret = Retriever(provs, as_of, dense=True)
        res = retrieval_eval(ret, queries)
        path = RESULTS_DIR / "baseline_retrieval_metrics.json"
        path.write_text(json.dumps(res, indent=2))
        for kind, groups in res.items():
            print(f"\n{kind}")
            for g, m in groups.items():
                print(f"  {g:8s} n={m['n']:3d}  R@3={m['recall@3']} R@8={m['recall@8']} "
                      f"R@20={m['recall@20']}  any@8={m['any_gold_in_top8']}  MRR={m['mrr']}")
        print(f"\nSaved {path}")
        return

    ctx = {"provs": provs, "lower": {p.lower(): p for p in provs}, "model": args.model,
           "k": args.k, "allow_abstain": args.allow_abstain}
    if args.mode in RAG_KIND:
        ctx["ret"] = Retriever(provs, as_of, dense=RAG_KIND[args.mode] != "bm25")
    if args.mode == "full":
        import legal_workflow_generator.config.values  # noqa: F401
        from legal_workflow_generator.agent.graph import graph
        ctx["graph"] = graph
    else:
        ctx["client"] = make_client()

    out_path = RESULTS_DIR / f"baseline_{args.mode}.jsonl"
    done = {r["query"] for r in load_rows(args.mode)} if args.resume else set()
    if not args.resume and out_path.exists():
        out_path.unlink()

    errors = 0
    try:
        for i, t in enumerate(queries, 1):
            if t["query"] in done:
                continue
            print(f"[{i:03d}/{len(queries)}] ({category(t)}) {t['query'][:70]}...")
            try:
                row = run_query(t, args.mode, ctx)
            except DailyQuota as e:
                print(f"\nDaily quota exhausted: {e}\nProgress saved. Rerun with --resume later.")
                break
            except ModelUnavailable as e:
                print(f"\nModel '{args.model}' is not available to your keys: {str(e)[:200]}\n"
                      "Set GEMINI_MODEL (or pass --model) to a model your keys can use, then rerun. "
                      "Nothing was saved for this query.")
                break
            except Exception as e:
                errors += 1
                print(f"        ERROR: {e}")
                continue
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(f"        cited={len(row['cited'])} valid={len(row['valid_cited'])} "
                  f"hits={row['hits']}/{len(row['expected_sections'])} "
                  f"abstained={row['abstained']} {row['elapsed_seconds']}s")
            time.sleep(args.sleep)
    except KeyboardInterrupt:
        print("\nInterrupted. Progress saved; rerun with --resume.")

    rows = load_rows(args.mode)
    if rows:
        summ = summarize(rows)
        (RESULTS_DIR / f"baseline_{args.mode}_summary.json").write_text(json.dumps(summ, indent=2))
        print(f"\n{'=' * 60}\n{args.mode}: {len(rows)}/{len(queries)} done, {errors} errors this session")
        for g, m in summ.items():
            print(f"  {g:8s} n={m['n']:3d} recall={m['gold_provision_recall']} prec={m['citation_precision']} "
                  f"valid={m['citation_validity']} abstain_acc={m['abstain_accuracy']} "
                  f"wrong_decl={m['wrongly_declined']} lat={m['avg_latency_s']}s")


if __name__ == "__main__":
    main()