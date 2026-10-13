"""Run: python eval_pipeline.py
Scores the pipeline against a hand-written gold set.
Gold sets are judgements, edit them freely. Each 'groups' entry is a list of
provision-id prefixes; the group counts as HIT if any cited id starts with any prefix."""
import statistics
import time

from legal_workflow_generator.agent.graph import graph

CASES = [
    {"q": "hiring a freelance developer abroad — who owns the code, withholding tax on payments to a non-resident, and is he a workman",
     "domains": {"taxation", "ip_licensing", "employment"},
     "groups": [["copyright_act_1957_sec_17"], ["copyright_act_1957_sec_19"],
                ["income_tax_act_1961_sec_195", "income_tax_act_2025_sec_393"], ["ida_1947_sec_2"]],
     "forbidden": ["eca_1923_sec_15b"]},
    {"q": "We are a startup collecting users' personal data and hiring 60 employees. What DPDP obligations and PF/ESI registration do we need?",
     "domains": {"data_protection", "employment"},
     "groups": [["dpdp_act_2023_sec_"], ["epf_1952_sec_"], ["esia_1948_sec_"]], "forbidden": []},
    {"q": "We are incorporating a private limited company and issuing shares to a foreign investor. What company law, FEMA and GST registration steps apply?",
     "domains": {"corporate_governance", "taxation"},
     "groups": [["ca_2013_sec_3", "ca_2013_sec_7"], ["cgst_act_2017_sec_22", "cgst_act_2017_sec_25", "cgst_rules_2017_rule_8"]],
     "forbidden": ["income_tax_act_1961_sec_195"]},
    {"q": "We built a mobile app. How do we protect our logo and source code, and what user consent do we need for collecting data?",
     "domains": {"data_protection", "ip_licensing"},
     "groups": [["dpdp_act_2023_sec_6"], ["trademarks_act_1999_sec_18"], ["copyright_act_1957_sec_"]],
     "forbidden": ["designs_act_2000_sec_23"]},
    {"q": "An employee resigned and took our source code. Who owns the copyright and what are the rules on termination and retrenchment?",
     "domains": {"ip_licensing", "employment"},
     "groups": [["copyright_act_1957_sec_17"], ["ida_1947_sec_25f"], ["ida_1947_sec_2"]], "forbidden": []},
    {"q": "We are acquiring a startup. What are directors' duties, the stamp duty on the share transfer, and what happens to employees under retrenchment rules?",
     "domains": {"corporate_governance", "taxation", "employment"},
     "groups": [["ca_2013_sec_166"], ["indian_stamp_act_1899_sec_9b"], ["ida_1947_sec_25ff"], ["ida_1947_sec_25f"]],
     "forbidden": ["ida_1947_sec_25fff"]},
    {"q": "We hired a contractor in Germany to design our logo. Who owns the copyright and do we deduct tax on the payment?",
     "domains": {"ip_licensing", "taxation"},
     "groups": [["copyright_act_1957_sec_17"], ["income_tax_act_1961_sec_195", "income_tax_act_2025_sec_393"]],
     "forbidden": ["income_tax_act_1961_sec_91"]},
    {"q": "What is the penalty for a personal data breach under DPDP?",
     "domains": {"data_protection"}, "groups": [["dpdp_act_2023_sch_1"]], "forbidden": []},
    {"q": "asdf blorp 123", "abstain": True},
    {"q": "What is the punishment for murder?", "abstain": True},
]


MIN_SPACING = 65      # seconds between query STARTS (free tier = 15 requests/min, ~12 calls per query)
RETRY_WAIT = 70       # wait before retrying a query that failed (e.g. quota)


def run(q):
    t = time.perf_counter()
    out = graph.invoke({"query": q})
    return out, time.perf_counter() - t


def run_safe(q):
    for attempt in (1, 2):
        try:
            return run(q)
        except Exception as e:
            print(f"    ! error ({type(e).__name__}): {str(e)[:120]}")
            if attempt == 1:
                print(f"    waiting {RETRY_WAIT}s and retrying once...")
                time.sleep(RETRY_WAIT)
    return None, 0.0


dom_p, dom_r, lat = [], [], []
grp_hit = grp_tot = forb = forb_tot = abst_ok = abst_tot = errors = 0

for i, c in enumerate(CASES, 1):
    started = time.perf_counter()
    out, dt = run_safe(c["q"])
    if out is None:
        errors += 1
        print(f"[{i}] ERROR - excluded from metrics")
    else:
        lat.append(dt)
        abstained = bool(out.get("abstain"))
        if c.get("abstain"):
            abst_tot += 1
            abst_ok += int(abstained)
            print(f"[{i}] abstain expected -> {'OK' if abstained else 'FAIL'}  ({dt:.1f}s)")
        else:
            got = set(out.get("all_domains") or [])
            exp = c["domains"]
            p = len(got & exp) / len(got) if got else 0.0
            r = len(got & exp) / len(exp)
            dom_p.append(p); dom_r.append(r)
            cited = [x.lower() for x in out.get("verified_citations", [])]
            hits = [any(x.startswith(pref) for x in cited for pref in g) for g in c["groups"]]
            grp_hit += sum(hits); grp_tot += len(hits)
            bad = [f for f in c["forbidden"] if f in cited]
            forb += len(bad); forb_tot += len(c["forbidden"])
            print(f"[{i}] domains P={p:.2f} R={r:.2f} | expected cited {sum(hits)}/{len(hits)} "
                  f"| noise {bad or '-'} | abstained={abstained} | {dt:.1f}s")
            missing = [g[0] for g, h in zip(c["groups"], hits) if not h]
            if missing:
                print(f"      missing: {missing}")

    if i < len(CASES):                       # pace to stay under the per-minute quota
        wait = MIN_SPACING - (time.perf_counter() - started)
        if wait > 0:
            print(f"    (pausing {wait:.0f}s for rate limit)")
            time.sleep(wait)

mean = lambda xs: statistics.mean(xs) if xs else float("nan")
print("\n=== SUMMARY ===")
print(f"queries errored       : {errors}")
print(f"domain precision      : {mean(dom_p):.2f}")
print(f"domain recall         : {mean(dom_r):.2f}")
print(f"expected-citation hit : {grp_hit}/{grp_tot} = {grp_hit / max(grp_tot, 1):.0%}")
print(f"known-noise citations : {forb}/{forb_tot}")
print(f"abstention accuracy   : {abst_ok}/{abst_tot}")
if lat:
    print(f"latency median / max  : {statistics.median(lat):.1f}s / {max(lat):.1f}s")