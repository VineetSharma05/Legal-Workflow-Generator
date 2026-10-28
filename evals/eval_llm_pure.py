#!/usr/bin/env python3
"""
Pure LLM baseline: no retrieval AND no identifiers given to the model.

The model is only asked for the NAME of the Act/Rules (with year) and the section
number for each step, in plain words. This code then maps those citations to corpus
provision ids. This is a fairer "Gemini alone" baseline than eval_baselines.py's
llm_only mode, which hands the model the id scheme and statute prefixes.

Put next to eval_baselines.py and eval_complete_multidomain.py (evals/).

Modes
-----
  selftest   ZERO API calls. Tests the citation resolver against your real corpus
             (every statute's own name must resolve back to it) and the parsing.
  run        Calls Gemini once per query. Saves each model citation (act + section)
             raw, so the mapping can be improved later without new API calls.
  rescore    ZERO API calls. Re-maps the saved raw citations with the current resolver
             and recomputes the metrics. Use after adding aliases.

Citation outcomes (reported separately, so mapping gaps never look like hallucinations)
  resolved      act and section found in the corpus -> counts as a valid citation
  nosection     act recognised but that section is not in the corpus -> counted INVALID
                (likely a hallucinated or non-existent section)
  unmatched     the act name could not be mapped to any statute -> EXCLUDED from validity,
                reported as a mapping gap; add an alias and run `rescore`

Usage
-----
  python eval_llm_pure.py --mode selftest
  python eval_llm_pure.py --mode run
  python eval_llm_pure.py --mode run --resume        # after a quota stop
  python eval_llm_pure.py --mode rescore
"""

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_complete_multidomain import RESULTS_DIR, TEST_QUERIES, category  # noqa: E402
import eval_baselines as eb  # noqa: E402

MODE_NAME = "llm_pure"  # results/baseline_llm_pure.jsonl

STOP = {"the", "of", "and", "for", "in", "on", "to", "a", "an", "india", "indian", "no", "by", "with"}

# abbreviation used in provision ids  ->  words used in full Act names
ABBR = {
    "dpdp": "digital personal data protection", "cgst": "central goods services tax",
    "igst": "integrated goods services tax", "utgst": "union territory goods services tax",
    "gst": "goods services tax", "ibc": "insolvency bankruptcy", "llp": "limited liability partnership",
    "css": "code social security", "oshwc": "occupational safety health working conditions",
    "posh": "prevention sexual harassment", "poga": "payment gratuity", "poba": "payment bonus",
    "mba": "maternity benefit", "epf": "employees provident funds miscellaneous provisions",
    "esi": "employees state insurance", "esia": "employees state insurance",
    "ida": "industrial disputes", "fa": "factories", "rpwd": "rights persons disabilities",
    "cla": "contract labour regulation abolition", "app": "apprentices", "pow": "payment wages",
    "bocw": "building other construction workers", "eca": "employees compensation",
    "spdi": "sensitive personal data information", "cert": "computer emergency response team",
    "cicra": "credit information companies regulation", "rti": "right information",
    "ca": "companies", "it": "information technology", "fema": "foreign exchange management",
    "sez": "special economic zones", "ieso": "industrial employment standing orders",
    "tua": "trade unions", "wj": "working journalists", "cal": "child adolescent labour",
    "era": "equal remuneration", "ismw": "inter state migrant workmen",
}
PHRASE_FIXES = (("trade marks", "trademarks"), ("trade mark", "trademark"), ("income-tax", "income tax"),
                ("co-operative", "cooperative"))

ID_RE = re.compile(r"^(?P<prefix>.+?)_(?:sec|section|rule|reg|dir|sch|schedule|annex|art)_(?P<num>.+)$", re.I)
SHORT_RE = re.compile(r"(?:may be called|shall be called|cited as)\s+(?:the\s+)?(.+?)(?:\.|,\s+and\b|$)", re.I)


def words(text: str) -> list:
    text = text.lower()
    for a, b in PHRASE_FIXES:
        text = text.replace(a, b)
    return re.findall(r"[a-z0-9]+", text)


def section_numbers(section) -> list:
    """'135' -> ['135']; 'Section 43A' -> ['43a']; '10(1)(a)' -> ['10']; '5, 6' -> ['5','6']."""
    out = []
    for part in re.split(r"[,;/&]|\band\b", str(section or "")):
        m = re.search(r"\d+[a-z]{0,3}", part.lower())
        if m:
            out.append(m.group(0))
    return out


class Resolver:
    def __init__(self, provs: dict):
        self.sections = defaultdict(dict)   # prefix -> {section number (lower): provision_id}
        srcs = defaultdict(set)
        for pid, o in provs.items():
            m = ID_RE.match(pid)
            if not m:
                continue
            prefix, num = m.group("prefix"), m.group("num").lower()
            self.sections[prefix].setdefault(num, pid)
            srcs[prefix].add(prefix.replace("_", " "))
            if o.get("statute_id"):
                srcs[prefix].add(str(o["statute_id"]).replace("_", " "))
            if num == "1":
                mt = SHORT_RE.search(o.get("text") or "")
                if mt:
                    srcs[prefix].add(mt.group(1))
        self.bags = {p: self._tokens(" ".join(s)) for p, s in srcs.items()}
        df = Counter(t for bag in self.bags.values() for t in bag)
        n = max(len(self.bags), 1)
        self.w = {t: max(0.15, math.log((n + 1) / (c + 0.5))) for t, c in df.items()}
        self.years = {p: set(re.findall(r"(?:18|19|20)\d\d", p + " " + " ".join(s))) for p, s in srcs.items()}

    @staticmethod
    def _tokens(text: str) -> set:
        toks = {w for w in words(text) if w not in STOP}
        extra = set()
        for w in toks:                      # id abbreviation -> its full words
            if w in ABBR:
                extra |= {x for x in ABBR[w].split() if x not in STOP}
        return toks | extra

    def _query_tokens(self, act: str) -> set:
        toks = self._tokens(act)
        for ab, full in ABBR.items():       # full words -> id abbreviation
            fw = {x for x in full.split() if x not in STOP}
            if fw and fw <= toks:
                toks.add(ab)
        return toks

    def _score(self, q: set, prefix: str) -> float:
        bag = self.bags[prefix]
        tot = sum(self.w.get(t, 0.15) for t in bag)
        return sum(self.w.get(t, 0.15) for t in bag & q) / tot if tot else 0.0

    def resolve(self, act: str, num: str):
        """-> ('ok', provision_id) | ('nosection', None) | ('unmatched', None)"""
        q = self._query_tokens(act or "")
        if not q or not self.bags:
            return "unmatched", None
        years = set(re.findall(r"(?:18|19|20)\d\d", act or ""))
        cands = [p for p in self.bags if not years or years & self.years.get(p, set())] or list(self.bags)
        scored = sorted(((self._score(q, p), p) for p in cands), reverse=True)
        best = scored[0][0]
        if best < 0.6:
            return "unmatched", None
        for s, p in scored:
            if s < best - 0.05:
                break
            if num in self.sections[p]:
                return "ok", self.sections[p][num]
        return "nosection", None


# ── prompt, parsing, rows ────────────────────────────────────────────────────
def build_prompt(query: str, allow_abstain: bool) -> str:
    ab = ('\nIf you cannot answer from Indian statutes, return {"abstain": true, "reason": "..."}.'
          if allow_abstain else "")
    return (
        "You are an assistant for Indian startup compliance questions. Answer from your own knowledge "
        "of Indian law as an ordered list of compliance steps. For each step, cite the legal "
        "provision(s) that support it by the full name of the Act or Rules (with year) and the "
        f"section or rule number.{ab}\n"
        'Return JSON only: {"steps":[{"action":"...","citations":[{"act":"Companies Act, 2013","section":"135"}]}]}\n'
        f"Question: {query}"
    )


def raw_citations(parsed: dict) -> list:
    out = []
    for s in parsed.get("steps") or []:
        if not isinstance(s, dict):
            continue
        cs = s.get("citations")
        for c in (cs if isinstance(cs, list) else [s]):
            if isinstance(c, dict):
                act = c.get("act") or c.get("statute") or c.get("law") or ""
                sec = c.get("section") or c.get("provision") or c.get("rule") or ""
                if act or sec:
                    out.append([str(act), str(sec)])
    return out


def build_row(t, raw_cits, resolver, provs, abstained, reason, elapsed):
    cited, unmatched = [], []
    for act, sec in raw_cits:
        nums = section_numbers(sec)
        if not nums:
            unmatched.append(f"{act} | {sec}")
            continue
        for n in nums:
            status, pid = resolver.resolve(act, n)
            if status == "ok":
                cited.append(pid)
            elif status == "nosection":
                cited.append(f"NOSECTION: {act} s.{n}")   # in provs? no -> counted invalid
            else:
                unmatched.append(f"{act} | {sec}")
    row = eb.make_row(t, sorted(set(cited)), provs, abstained, reason, elapsed, [])
    row["unmatched"] = unmatched
    row["raw_citations"] = raw_cits
    return row


def run_one(t, ctx):
    start = time.time()
    raw = eb.call_llm(ctx["client"], ctx["model"], build_prompt(t["query"], ctx["allow_abstain"]))
    parsed = eb.parse_json(raw)
    abstained = bool(parsed.get("abstain")) or (ctx["allow_abstain"] and not parsed.get("steps"))
    cits = [] if abstained else raw_citations(parsed)
    row = build_row(t, cits, ctx["resolver"], ctx["provs"], abstained, str(parsed.get("reason", "")),
                    time.time() - start)
    row.update(model=ctx["model"], unparsed=not parsed, raw_output=(raw or "")[:2000])
    return row


def print_summary(rows: list):
    summ = eb.summarize(rows)
    raw_total = sum(len(r.get("raw_citations", [])) for r in rows)
    unm = sum(len(r.get("unmatched", [])) for r in rows)
    print(f"\n{'=' * 60}\npure LLM (no ids given): {len(rows)} queries")
    for g, m in summ.items():
        print(f"  {g:8s} n={m['n']:3d} recall={m['gold_provision_recall']} prec={m['citation_precision']} "
              f"valid={m['citation_validity']} abstain_acc={m['abstain_accuracy']} "
              f"wrong_decl={m['wrongly_declined']} unparsed={m['unparsed_outputs']} lat={m['avg_latency_s']}s")
    print(f"  mapping gaps (act name not recognised): {unm} of {raw_total} citations "
          f"({round(100 * unm / raw_total, 1) if raw_total else 0}%) - excluded from validity.")
    if raw_total and unm / raw_total > 0.1:
        print("  Over 10% unmatched: recall is understated. Run `--mode rescore` after adding aliases to ABBR.")
        shown = Counter(u.split(" | ")[0] for r in rows for u in r.get("unmatched", []))
        print("  most common unmatched act names:", shown.most_common(8))
    (RESULTS_DIR / f"baseline_{MODE_NAME}_summary.json").write_text(json.dumps(summ, indent=2))
    m = summ["overall"]
    print(f"\nLaTeX row: LLM, no retrieval, no ids & {m['gold_provision_recall']} & "
          f"{m['citation_validity']} & {m['abstain_accuracy']} & {m['wrongly_declined']} \\\\")


# ── offline self-test ────────────────────────────────────────────────────────
class _Resp:
    def __init__(self, text):
        self.text = text


class _FakeModels:
    def generate_content(self, model, contents, config=None):
        steps = [{"action": "a", "citations": [{"act": "Companies Act, 2013", "section": "Section 135"}]},
                 {"action": "b", "citations": [{"act": "The Totally Made Up Act, 1999", "section": "7"}]},
                 {"action": "c", "citations": [{"act": "Companies Act, 2013", "section": "99999"}]}]
        return _Resp("```json\n" + json.dumps({"steps": steps}) + "\n```")


class _FakeClient:
    models = _FakeModels()


def selftest(provs: dict) -> int:
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  PASS  " if cond else "  FAIL  ") + msg)
        ok = ok and bool(cond)

    print("Offline self-test (no API calls)")
    res = Resolver(provs)
    print(f"  {len(res.bags)} statutes indexed from the corpus")

    # 1. every statute's own name must resolve back to itself (or a duplicate with the same section)
    fails, total = [], 0
    for prefix, secs in res.sections.items():
        name = prefix.replace("_", " ")
        for num in list(secs)[:2]:
            total += 1
            status, pid = res.resolve(name, num)
            if status != "ok" or ID_RE.match(pid).group("num").lower() != num:
                fails.append((name, num, status))
    rate = 1 - len(fails) / total if total else 0
    check(rate >= 0.95, f"statute-name round trip: {total - len(fails)}/{total} resolve ({rate:.0%})")
    for f in fails[:15]:
        print("        unresolved:", f)

    # 2. natural-language names (only checked when the target exists in your corpus)
    natural = [("Companies Act, 2013", "135", "ca_2013_sec_135"),
               ("Digital Personal Data Protection Act, 2023", "8", "dpdp_act_2023_sec_8"),
               ("Central Goods and Services Tax Act, 2017", "16", "cgst_act_2017_sec_16"),
               ("Indian Contract Act, 1872", "27", "contract_act_1872_sec_27"),
               ("Payment of Gratuity Act, 1972", "4", "poga_1972_sec_4"),
               ("Industrial Disputes Act, 1947", "25f", "ida_1947_sec_25F"),
               ("Code on Social Security, 2020", "114", "css_2020_sec_114"),
               ("Patents Act, 1970", "3", "patents_act_1970_sec_3"),
               ("Trade Marks Act, 1999", "29", "trademarks_act_1999_sec_29"),
               ("Limited Liability Partnership Act, 2008", "7", "llp_act_2008_sec_7"),
               ("Insolvency and Bankruptcy Code, 2016", "53", "ibc_2016_sec_53"),
               ("Income-tax Act, 1961", "192", "income_tax_act_1961_sec_192")]
    present = [(a, n, e) for a, n, e in natural if e in provs]
    for a, n, e in present:
        status, pid = res.resolve(a, n)
        check(pid == e, f"'{a}' s.{n} -> {pid}")
    print(f"  ({len(natural) - len(present)} natural-language checks skipped: id not in this corpus)")

    # 3. parsing and the full path with a fake model
    check(section_numbers("Section 43A") == ["43a"] and section_numbers("10(1)(a)") == ["10"]
          and section_numbers("5, 6") == ["5", "6"], "section number parsing")
    t = next(x for x in TEST_QUERIES if not x["should_abstain"])
    ctx = {"client": _FakeClient(), "model": "fake", "allow_abstain": False, "resolver": res, "provs": provs}
    row = run_one(t, ctx)
    check(len(row["raw_citations"]) == 3, "fake model: 3 raw citations parsed from fenced JSON")
    check(any(u.startswith("The Totally Made Up Act") for u in row["unmatched"]),
          "a made-up Act is reported as a mapping gap, not as a valid citation")
    check(not any("Made Up" in c for c in row["valid_cited"]), "made-up citation never counted valid")
    if "ca_2013_sec_135" in provs:
        check("ca_2013_sec_135" in row["valid_cited"], "'Companies Act, 2013 / Section 135' resolved")
        check(any(c.startswith("NOSECTION") for c in row["cited"]), "non-existent section counted invalid")
    print("\nSELFTEST PASSED" if ok else "\nSELFTEST FAILED - see FAIL lines above")
    return 0 if ok else 1


# ── main ─────────────────────────────────────────────────────────────────────
def early_abort(session: list):
    if len(session) != 3:
        return None
    if sum(bool(r.get("unparsed")) for r in session) >= 2:
        return "2 of the first 3 model outputs were not valid JSON"
    if sum(len(r["raw_citations"]) for r in session) == 0:
        return "the first 3 answers contained no citations at all"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["selftest", "run", "rescore"])
    ap.add_argument("--model", default=None, help="Defaults to the pipeline's GEMINI_MODEL.")
    ap.add_argument("--category", choices=["all", "single", "multi", "edge"], default="all")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=4.0)
    ap.add_argument("--allow-abstain", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    for var, val in (("PGPASSWORD", "unused"), ("PGHOST", "localhost"), ("PGUSER", "unused"),
                     ("PGDATABASE", "unused"), ("PGPORT", "5432")):
        os.environ.setdefault(var, val)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    provs = eb.load_corpus()
    resolver = Resolver(provs)

    if args.mode == "selftest":
        sys.exit(selftest(provs))

    by_query = {t["query"]: t for t in TEST_QUERIES}
    out_path = RESULTS_DIR / f"baseline_{MODE_NAME}.jsonl"

    if args.mode == "rescore":
        old = eb.load_rows(MODE_NAME)
        if not old:
            sys.exit(f"No saved results at {out_path}. Run --mode run first.")
        bak = out_path.with_suffix(".bak")
        if not bak.exists():
            bak.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")
        new = []
        for r in old:
            row = build_row(by_query[r["query"]], r["raw_citations"], resolver, provs, r["abstained"],
                            r.get("abstain_reason", ""), r["elapsed_seconds"])
            for k in ("model", "unparsed", "raw_output"):
                row[k] = r.get(k)
            new.append(row)
        out_path.write_text("".join(json.dumps(r) + "\n" for r in new), encoding="utf-8")
        print(f"Re-mapped {len(new)} saved answers with the current resolver (backup: {bak.name}).")
        return print_summary(new)

    if args.model is None:
        try:
            import legal_workflow_generator.config.values as cfg
            args.model = cfg.GEMINI_MODEL
        except Exception:
            sys.exit("Could not read GEMINI_MODEL from the project config. Pass --model <name>.")
        print(f"Using the pipeline's model: {args.model}")

    queries = [t for t in TEST_QUERIES
               if args.category == "all" or (args.category == "multi" and category(t).startswith("multi_"))
               or category(t) == args.category]
    queries = queries[: args.limit] if args.limit else queries
    print(f"{len(provs)} provisions loaded; {len(queries)} queries")

    ctx = {"client": eb.make_client(), "model": args.model, "allow_abstain": args.allow_abstain,
           "resolver": resolver, "provs": provs}
    done = {r["query"] for r in eb.load_rows(MODE_NAME)} if args.resume else set()
    if not args.resume and out_path.exists():
        out_path.unlink()

    session, errors = [], 0
    try:
        for i, t in enumerate(queries, 1):
            if t["query"] in done:
                continue
            print(f"[{i:03d}/{len(queries)}] ({category(t)}) {t['query'][:70]}...")
            try:
                row = run_one(t, ctx)
            except eb.DailyQuota as e:
                print(f"\nDaily quota exhausted: {str(e)[:200]}\nProgress saved. Rerun with --resume later.")
                break
            except eb.ModelUnavailable as e:
                print(f"\nModel '{args.model}' is not available to your keys: {str(e)[:200]}\nNothing saved.")
                break
            except Exception as e:
                errors += 1
                print(f"        ERROR: {e}")
                continue
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            session.append(row)
            print(f"        cited={len(row['cited'])} valid={len(row['valid_cited'])} "
                  f"unmatched={len(row['unmatched'])} hits={row['hits']}/{len(row['expected_sections'])} "
                  f"abstained={row['abstained']} {row['elapsed_seconds']}s")
            reason = early_abort(session)
            if reason:
                bad = {r["query"] for r in session}
                keep = [r for r in eb.load_rows(MODE_NAME) if r["query"] not in bad]
                out_path.write_text("".join(json.dumps(r) + "\n" for r in keep), encoding="utf-8")
                print(f"\nABORTED EARLY to save your quota: {reason}.\nFirst raw output:\n"
                      f"{session[0].get('raw_output', '')[:800]}\nThe 3 rows were discarded.")
                return
            time.sleep(args.sleep)
    except KeyboardInterrupt:
        print("\nInterrupted. Progress saved; rerun with --resume.")

    rows = eb.load_rows(MODE_NAME)
    if rows:
        print_summary(rows)
        print(f"({errors} errors this session)")


if __name__ == "__main__":
    main()