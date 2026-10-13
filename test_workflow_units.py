"""
Unit / integration tests for the LangGraph workflow. No network, no database, no API quota:
the LLM, the searcher and the DB are replaced by fakes. Run from the project root:

    pytest tests/test_workflow_units.py -q
"""
import datetime
import random
import threading
import time

import pytest

from legal_workflow_generator.agent import nodes as N
from legal_workflow_generator.agent import settings as S
from legal_workflow_generator.agent.graph import graph, run_workflow  # noqa: F401


# ─────────────────────────────── fakes ───────────────────────────────
ALL_DOCS = [
    {"provision_id": "copyright_act_1957_sec_17", "title": "First owner of copyright", "combined_score": 0.6,
     "text": "the author of a work shall be the first owner of the copyright; employer owns under a contract of service"},
    {"provision_id": "copyright_act_1957_sec_19", "title": "Mode of assignment", "combined_score": 0.5,
     "text": "assignment of copyright is valid only if in writing signed by the assignor"},
    {"provision_id": "income_tax_act_1961_sec_195", "title": "Other sums", "combined_score": 0.6,
     "text": "any person responsible for paying to a non-resident any sum chargeable shall deduct income-tax"},
    {"provision_id": "income_tax_act_1961_sec_9", "title": "Income deemed to accrue", "combined_score": 0.5,
     "text": "the following incomes shall be deemed to accrue or arise in India"},
    {"provision_id": "ida_1947_sec_2", "title": "Definitions", "combined_score": 0.6,
     "text": "workman means any person employed to do manual, skilled, technical work"},
    {"provision_id": "dpdp_act_2023_sec_6", "title": "Consent", "combined_score": 0.4,
     "text": "consent must be free, specific, informed"},
]


class FakeSearcher:
    def search(self, query, top_k=3, bm25_candidates=20, vector_candidates=0):
        return [dict(d) for d in ALL_DOCS]


class FakeRag:
    searcher = FakeSearcher()

    def _ensure_index(self):
        pass


class FakeNormalizer:
    def normalize(self, text):
        return text.lower()


class FakeIntent:
    def classify(self, text):
        return "qa", 0.95


class FakeResolver:
    def resolve(self, normalized, intent, confidence):
        return {"intent": intent, "legal_domain": "ip_licensing", "normalized_query": normalized,
                "keywords": [], "confidence": confidence, "original_query": normalized}


def _prompt_kind(system, user):
    s = (system + " " + user)
    for key, kind in (("domain classifier", "domains"), ("search query writer", "focus"),
                      ("relevance grader", "grade"), ("search query optimizer", "rewrite"),
                      ("fact-checker", "verify"), ("grounding verifier", "grounded"),
                      ("workflow assistant", "generate")):
        if key in s:
            return kind
    return "other"


class ScriptedLLM:
    """Deterministic stand-in for the LLM. `verdicts` controls what the claim verifier answers."""

    def __init__(self, verdicts=("partial", "unsupported"), jitter=False, fail=None):
        self.verdicts, self.jitter, self.fail = list(verdicts), jitter, fail
        self.calls = []
        self._lock = threading.Lock()

    def __call__(self, system, user, *a, **k):
        if self.jitter:
            time.sleep(random.random() * 0.02)
        if self.fail:
            raise self.fail
        kind = _prompt_kind(system, user)
        with self._lock:
            self.calls.append(kind)
        import json
        import re
        if kind == "domains":
            return "ip_licensing, taxation, employment"
        if kind == "focus":
            m = re.search(r"covering ONLY the (.+?) aspects", user)
            return f"focus {m.group(1) if m else ''}"
        if kind == "grade":
            return '{"grade": "sufficient", "reason": "ok"}'
        if kind == "rewrite":
            return "rewritten query"
        if kind == "grounded":
            return '{"grade": "grounded", "reason": "ok"}'
        if kind == "generate":
            ids = list(dict.fromkeys(re.findall(r"^\[([a-z0-9_]+)\]", user.split("Provisions:")[1], re.M)))
            steps = [{"action": f"do thing {i}", "provision_id": pid, "applies_to": "private"}
                     for i, pid in enumerate(ids[:3], 1)]
            steps.append({"action": "invented", "provision_id": "made_up_sec_1", "applies_to": "both"})
            return json.dumps({"summary": ["a summary"], "steps": steps, "not_covered": ""})
        if kind == "verify":
            n = len(re.findall(r"^\d+\. CLAIM:", user, re.M))
            return json.dumps({"verdicts": [{"n": i + 1, "verdict": self.verdicts[i % len(self.verdicts)],
                                             "reason": "r"} for i in range(n)]})
        return "{}"


@pytest.fixture(autouse=True)
def fast_limits():
    old = {p: l.rpm for p, l in N.LIMITERS.items()}
    for p in N.LIMITERS:
        N.set_rate_limit(p, 100000)
    yield
    for p, r in old.items():
        N.set_rate_limit(p, r)


@pytest.fixture
def wired(monkeypatch):
    llm = ScriptedLLM()
    monkeypatch.setattr(N, "_get_rag_pipeline", lambda: FakeRag())
    monkeypatch.setattr(N, "_detect_domains_from_retrieval", lambda q: [])
    monkeypatch.setattr(N, "_normalizer", FakeNormalizer())
    monkeypatch.setattr(N, "_intent_classifier", FakeIntent())
    monkeypatch.setattr(N, "_context_resolver", FakeResolver())
    monkeypatch.setattr(N, "_hydrate", lambda docs: docs)
    monkeypatch.setattr(N, "_fetch_by_ids", lambda ids: [])
    monkeypatch.setattr(N, "_load_domain_map", lambda: {})
    monkeypatch.setattr(N, "_llm", llm)
    return llm


QUERY = "hiring a freelance developer abroad: who owns the code, withholding tax on payment, and is he a workman"


# ─────────────────────────────── settings (item 1) ───────────────────────────────
def test_settings_defaults_and_validation():
    s = S.RunSettings()
    assert s.provider in S.ALLOWED_PROVIDERS and s.top_k == 8
    for bad in ({"provider": "openai"}, {"top_k": 0}, {"top_k": 99}, {"as_of": "13/10/2026"}, {"max_domain_retries": 9}):
        with pytest.raises(ValueError):
            S.RunSettings(**bad)


def test_from_config_coerces_and_ignores_unknown_keys():
    cfg = {"configurable": {"provider": " GROQ ", "top_k": "5", "verify_claims": "false", "thread_id": "x",
                            "__pregel_task_id": "y", "as_of": "2026-04-01"}}
    s = S.RunSettings.from_config(cfg)
    assert (s.provider, s.top_k, s.verify_claims, s.as_of_date) == ("groq", 5, False, datetime.date(2026, 4, 1))
    assert S.build_config(provider="gemini", top_k=None) == {"configurable": {"provider": "gemini"}}


_ORIGINAL_LLM = N._llm          # the real N._llm (provider routing, rate limiter, retries)


def test_provider_and_top_k_reach_every_worker(wired, monkeypatch):
    scripted = ScriptedLLM()
    seen = {"groq": 0}

    def fake_groq(system, user):
        seen["groq"] += 1
        return scripted(system, user)

    def no_gemini(system, user):
        raise AssertionError("gemini must not be used when provider=groq")

    monkeypatch.setattr(N, "_llm", _ORIGINAL_LLM)
    monkeypatch.setitem(N._PROVIDERS, "groq", fake_groq)
    monkeypatch.setitem(N._PROVIDERS, "gemini", no_gemini)
    out = run_workflow(QUERY, provider="groq", top_k=2, trace_timing=False)
    assert seen["groq"] > 5                                        # classify, 3 workers, grader all used groq
    assert any("provider=groq top_k=2" in t for t in out["trace"])
    for d in ("ip_licensing", "taxation", "employment"):
        n = [x for x in out["retrieved_docs"] if N._domain_of(x["provision_id"]) == d]
        assert len(n) <= 2                                         # top_k limits provisions per domain


def test_invalid_provider_is_rejected_by_the_workflow(wired):
    with pytest.raises(ValueError):
        run_workflow(QUERY, provider="openai")
    with pytest.raises(ValueError):
        run_workflow(QUERY, top_k=500)


# ─────────────────────────────── typed models (item 2) ───────────────────────────────
def test_step_from_llm_validates_ids_and_applies_to():
    valid = {"ida_1947_sec_2": "ida_1947_sec_2"}
    ok = N.Step.from_llm({"action": " a ", "provision_id": "IDA_1947_SEC_2", "applies_to": "weird"}, valid)
    assert ok and ok.provision_id == "ida_1947_sec_2" and ok.applies_to == "unspecified" and ok.action == "a"
    assert N.Step.from_llm({"action": "a", "provision_id": "nope"}, valid) is None
    assert N.Step.from_llm({"action": "", "provision_id": "ida_1947_sec_2"}, valid) is None
    assert N.Step.from_llm("not a dict", valid) is None


def test_domain_results_are_typed(wired):
    out = run_workflow(QUERY, trace_timing=False)
    assert out["abstain"] is False
    r = N.DomainResult(domain="taxation", steps=[N.Step("x", "income_tax_act_1961_sec_195")])
    d = r.to_dict()
    assert d["steps"][0]["provision_id"] == "income_tax_act_1961_sec_195" and d["grade"] == "insufficient"


# ─────────────────────────────── claim verification (item 3) ───────────────────────────────
def test_unsupported_steps_are_dropped_partial_kept(wired):
    out = run_workflow(QUERY, trace_timing=False)
    text = out["stitched_answer"]
    assert "invented" not in text                                   # unknown ids never reach the checklist
    assert any("verify_claims →" in t and "dropped" in t for t in out["trace"])
    assert "[NOTE: only partly supported by the cited text]" in text   # claim 1 of each domain is 'partial'


def test_verification_can_be_switched_off(wired):
    out = run_workflow(QUERY, verify_claims=False, trace_timing=False)
    assert not any("verify_claims →" in t for t in out["trace"])
    assert "verify" not in wired.calls


def test_verifier_failures_never_drop_steps(monkeypatch):
    steps = [N.Step("a", "p1"), N.Step("b", "p2")]
    docs = [{"provision_id": "p1", "title": "t", "text": "x"}, {"provision_id": "p2", "title": "t", "text": "y"}]
    monkeypatch.setattr(N, "_llm", lambda *a, **k: (_ for _ in ()).throw(N.LLMUnavailable("quota")))
    kept, dropped, msg = N._verify_claims("taxation", steps, docs)
    assert len(kept) == 2 and not dropped and "skipped" in msg
    monkeypatch.setattr(N, "_llm", lambda *a, **k: "not json")
    kept, dropped, msg = N._verify_claims("taxation", steps, docs)
    assert len(kept) == 2 and "unparseable" in msg


# ─────────────────────────────── rate limiting (item 4) ───────────────────────────────
def test_rate_limiter_blocks_when_window_is_full():
    lim = N.RateLimiter(rpm=3, window=0.4)
    t = time.monotonic()
    for _ in range(4):
        lim.acquire()
    assert time.monotonic() - t >= 0.3 and lim.calls == 4


def test_parallel_call_cap():
    lim = N.RateLimiter(rpm=1000, max_parallel=2)
    live, peak, lock = 0, 0, threading.Lock()

    def work():
        nonlocal live, peak
        with lim.slot():
            with lock:
                live += 1
                peak = max(peak, live)
            time.sleep(0.05)
            with lock:
                live -= 1

    ts = [threading.Thread(target=work) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert peak <= 2


def test_quota_errors_are_reported_as_unavailable_not_as_no_provisions(wired, monkeypatch):
    monkeypatch.setattr(N.time, "sleep", lambda s: None)
    monkeypatch.setattr(N, "_llm", _ORIGINAL_LLM)
    monkeypatch.setitem(N._PROVIDERS, "gemini", lambda s, u: (_ for _ in ()).throw(
        RuntimeError("429 RESOURCE_EXHAUSTED. Please retry in 0.01s")))
    out = run_workflow(QUERY, trace_timing=False)
    assert out["abstain"] is True and "LLM unavailable" in out["abstain_reason"]
    assert "No matching provisions" not in out["answer"]
    assert any("ERROR → LLM unavailable" in t for t in out["trace"])


def test_daily_quota_is_not_retried(monkeypatch):
    calls = []
    monkeypatch.setattr(N.time, "sleep", lambda s: None)
    monkeypatch.setitem(N._PROVIDERS, "gemini", lambda s, u: calls.append(1) or (_ for _ in ()).throw(
        RuntimeError("429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel PerDay")))
    with pytest.raises(N.LLMUnavailable):
        _ORIGINAL_LLM("s", "u")
    assert len(calls) == 1


# ─────────────────────────────── deterministic trace (item 5) ───────────────────────────────
def test_trace_is_identical_across_runs_even_when_threads_finish_in_random_order(monkeypatch, wired):
    traces = []
    for seed in range(6):
        random.seed(seed)
        monkeypatch.setattr(N, "_llm", ScriptedLLM(jitter=True))
        traces.append(run_workflow(QUERY, trace_timing=False)["trace"])
    assert all(t == traces[0] for t in traces[1:])
    order = [t.split("]")[0] for t in traces[0] if t.startswith("[")]
    firsts = list(dict.fromkeys(order))
    assert firsts == [f"[{d}" for d in N._DOMAIN_ORDER if f"[{d}" in firsts]   # fixed domain order


# ─────────────────────────────── temporal (item 7) ───────────────────────────────
def test_temporal_note_and_in_force():
    d = datetime.date
    assert "Income-tax Act, 2025" in N._temporal_note("income_tax_act_1961_sec_195", d(2026, 10, 13))
    assert N._temporal_note("income_tax_act_1961_sec_195", d(2026, 3, 31)) == ""
    assert N._temporal_note("copyright_act_1957_sec_17", d(2026, 10, 13)) == ""
    assert N._in_force({"effective_date": d(2027, 5, 1)}, d(2026, 10, 13)) is False
    assert N._in_force({"effective_date": d(1958, 1, 21)}, d(2026, 10, 13)) is True
    assert N._in_force({"effective_date": None}, d(2026, 10, 13)) is True


def test_superseded_statute_is_flagged_in_the_answer_and_date_is_respected(wired):
    now = run_workflow(QUERY, as_of="2026-10-13", trace_timing=False)["stitched_answer"]
    before = run_workflow(QUERY, as_of="2026-01-01", trace_timing=False)["stitched_answer"]
    assert "superseded by the Income-tax Act, 2025" in now
    assert "superseded by the Income-tax Act, 2025" not in before


def test_provisions_not_yet_in_force_are_excluded(wired, monkeypatch):
    future = datetime.date(2030, 1, 1)
    monkeypatch.setattr(N, "_hydrate", lambda docs: [
        {**d, "effective_date": future if d["provision_id"] == "income_tax_act_1961_sec_9" else None} for d in docs])
    out = run_workflow(QUERY, as_of="2026-10-13", trace_timing=False)
    assert "income_tax_act_1961_sec_9" not in [d["provision_id"] for d in out["retrieved_docs"]]
    assert any("excluded 1 provision(s) not in force" in t for t in out["trace"])


# ─────────────────────────────── helpers ───────────────────────────────
def test_snippet_centres_on_rare_keyword_and_full_text_includes_clauses():
    text = "intro " * 400 + "the workman means any person" + " tail" * 400
    assert "workman" in N._snippet(text, "is he a workman", 300)
    ft = N._full_text("Definitions", "header", {"s": '"workman" means ...', "a": "appropriate government"})
    assert "(s)" in ft and "workman" in ft and ft.startswith("Definitions")


@pytest.mark.parametrize("question,fires", [
    ("We hired a contractor in Germany. Do we deduct tax on the payment?", True),
    ("paying a foreign agency for the design; do we deduct tax", True),
    ("We hired a developer in the US to build our app and need to deduct tax on the payment", True),
    ("issuing shares to a foreign investor; what GST registration steps apply", False),
    ("a pending funding round with foreign customers", False),
])
def test_tax_anchor_triggers(question, fires):
    a = [x for x in N.DOMAIN_ANCHORS if x["domain"] == "taxation"][0]
    q = question.lower()
    import re
    assert bool(re.search(a["trigger"], q) and re.search(a["also"], q)) is fires


def test_stitch_shows_failed_error_and_unverifiable_domains():
    ok = N.DomainResult("ip_licensing", grade="sufficient", summary=["s"],
                        steps=[N.Step("owns it", "copyright_act_1957_sec_17", verdict="partial")],
                        cited=["copyright_act_1957_sec_17"])
    gap = N.DomainResult("taxation", grade="insufficient", reason="nothing matched")
    err = N.DomainResult("employment", grade="error", reason="LLM unavailable: quota")
    bad = N.DomainResult("data_protection", grade="sufficient", dropped_steps=[N.Step("x", "p")],
                         reason="Every generated step failed verification against the provision it cited.")
    text = N._stitch_sections("q", [ok, gap, err, bad])
    assert "No matching provisions found" in text and "Please retry" in text and "failed verification" in text
    assert "only partly supported" in text