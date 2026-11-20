"""
Unit / integration tests for the LangGraph workflow. No network, no database, no API quota:
the LLM, the searcher and the DB are replaced by fakes. Run from the project root:

    pytest tests/test_workflow_units.py -q
"""
import datetime
import json
import random
import re
import threading
import time
import typing

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
    def classify(self, text, provider="gemini", raise_errors=False):
        return "qa", 0.95


class FakeResolver:
    def resolve(self, normalized, intent, confidence, provider="gemini", raise_errors=False):
        return {"intent": intent, "legal_domain": "ip_licensing", "normalized_query": normalized,
                "keywords": [], "confidence": confidence, "original_query": normalized}


def _prompt_kind(system, user):
    """Kind of a plain-text call (_llm(system, user)): query planning and the claim verifier."""
    s = (system + " " + user)
    for key, kind in (("search query writer", "focus"), ("fact-checker", "verify")):
        if key in s:
            return kind
    return "other"


# Kind of a structured call (_llm(Schema, system, user)), by the schema the caller asks for.
_SCHEMA_KIND = {"DomainList": "domains", "RelevanceGrade": "grade", "SearchQuery": "rewrite",
                "GroundednessGrade": "grounded", "AnswerabilityGrade": "answerable", "DomainWorkflow": "generate"}


def _workflow_ids(schema):
    """The provision ids a domain_workflow_model(ids) schema allows (its provision_id enum)."""
    step = typing.get_args(schema.model_fields["steps"].annotation)[0]
    return step, list(typing.get_args(step.model_fields["provision_id"].annotation))


class ScriptedLLM:
    """Deterministic stand-in for N._llm, at its real signature:
    structured _llm(Schema, system, user) -> Schema instance; plain-text _llm(system, user) -> str.
    `verdicts` controls what the claim verifier answers."""

    def __init__(self, verdicts=("partial", "unsupported"), jitter=False, fail=None):
        self.verdicts, self.jitter, self.fail = list(verdicts), jitter, fail
        self.calls = []
        self._lock = threading.Lock()

    def __call__(self, schema, system, user=None, *a, **k):
        if self.jitter:
            time.sleep(random.random() * 0.02)
        if self.fail:
            raise self.fail
        if isinstance(schema, str):                       # plain-text call: (system, user)
            system, user = schema, system
            kind = _prompt_kind(system, user)
        else:
            kind = _SCHEMA_KIND.get(schema.__name__, "other")
        with self._lock:
            self.calls.append(kind)
        if kind == "domains":
            return schema(domains=["ip_licensing", "taxation", "employment"])
        if kind == "focus":
            m = re.search(r"covering ONLY the (.+?) aspects", user)
            return f"focus {m.group(1) if m else ''}"
        if kind == "grade":
            return schema(reason="ok", grade="sufficient")
        if kind == "rewrite":
            return schema(query="rewritten query")
        if kind == "grounded":
            return schema(reason="ok", unsupported_claims=[], grade="grounded")
        if kind == "answerable":
            return schema(reason="ok", grade="answers")
        if kind == "generate":
            step, ids = _workflow_ids(schema)
            out = schema(summary=["a summary"], not_covered="",
                         steps=[{"action": f"do thing {i}", "provision_id": pid, "applies_to": "private"}
                                for i, pid in enumerate(ids[:3], 1)])
            # an invented id, bypassing schema validation (a provider that ignores the enum): must never surface
            out.steps.append(step.model_construct(action="invented", provision_id="made_up_sec_1", applies_to="both"))
            return out
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

    def fake_groq(system, user, schema=None):          # provider callable: (system, user, schema=None)
        seen["groq"] += 1
        return scripted(schema, system, user) if schema else scripted(system, user)

    def no_gemini(system, user, schema=None):
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
    monkeypatch.setitem(N._PROVIDERS, "gemini", lambda s, u, schema=None: (_ for _ in ()).throw(
        RuntimeError("429 RESOURCE_EXHAUSTED. Please retry in 0.01s")))
    out = run_workflow(QUERY, trace_timing=False)
    assert out["abstain"] is True and "LLM unavailable" in out["abstain_reason"]
    assert "No matching provisions" not in out["answer"]
    assert any("ERROR → LLM unavailable" in t for t in out["trace"])


def test_daily_quota_is_not_retried(monkeypatch):
    calls = []
    monkeypatch.setattr(N.time, "sleep", lambda s: None)
    monkeypatch.setitem(N._PROVIDERS, "gemini", lambda s, u, schema=None: calls.append(1) or (_ for _ in ()).throw(
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
    out = run_workflow(QUERY, as_of="2026-10-13", future_provisions="exclude", trace_timing=False)
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

# ─────────────────────────────── follow-up checks (review of phase3-agentic) ───────────────────────────────
def test_domain_results_and_stitched_answer_reach_the_graph_output(wired):
    out = run_workflow(QUERY, trace_timing=False)
    assert isinstance(out["domain_results"], list) and out["domain_results"]
    assert all(isinstance(r, dict) and "steps" in r and "verification" in r for r in out["domain_results"])
    assert out["answer"] == out["stitched_answer"]                 # the API returns the stitched text
    assert out["as_of"] == datetime.date.today().isoformat()
    assert f"Valid as of: {out['as_of']}" in out["answer"]


def test_expired_provisions_do_not_shrink_top_k(wired, monkeypatch):
    """A provision not in force on as_of must not push a valid one out of the top-k window."""
    future = datetime.date(2030, 1, 1)
    # rank the future-dated section FIRST, so it would take a slot if the cut came before the date filter
    ranked = [ALL_DOCS[3]] + [d for d in ALL_DOCS if d is not ALL_DOCS[3]]
    monkeypatch.setattr(FakeSearcher, "search", lambda self, q, **k: [dict(d) for d in ranked])
    monkeypatch.setattr(N, "_hydrate", lambda docs: [
        {**d, "effective_date": future if d["provision_id"] == "income_tax_act_1961_sec_9" else None}
        for d in docs])
    out = run_workflow(QUERY, as_of="2026-10-13", top_k=1, future_provisions="exclude", trace_timing=False)
    tax = [d["provision_id"] for d in out["retrieved_docs"] if N._domain_of(d["provision_id"]) == "taxation"]
    assert tax == ["income_tax_act_1961_sec_195"]                  # the valid provision took the slot


def test_unchecked_steps_are_flagged_when_the_checker_was_unavailable(monkeypatch):
    steps = [N.Step("a", "copyright_act_1957_sec_17")]
    res = N.DomainResult("ip_licensing", grade="sufficient", steps=steps, verification="skipped")
    text = N._stitch_sections("q", [res])
    assert "checker was unavailable" in text


def test_query_side_gemini_calls_use_the_shared_limiter(monkeypatch):
    from legal_workflow_generator.agent.llm_limits import LIMITERS
    from legal_workflow_generator.query.intent_classifier import IntentClassifier

    entered = []

    class CountingSlot:
        def __enter__(self):
            entered.append(1)

        def __exit__(self, *a):
            return False

    class FakeLimiter:
        def slot(self):
            return CountingSlot()

    class FakeModels:
        def generate_content(self, **kw):
            class R:
                text = "INTENT: qa\nCONFIDENCE: 0.9\nREASON: r"
            return R()

    clf = IntentClassifier.__new__(IntentClassifier)
    clf.client = type("C", (), {"models": FakeModels()})()
    monkeypatch.setitem(LIMITERS, "gemini", FakeLimiter())
    clf.classify({"original": "q", "normalized": "q", "source": "t"})
    assert entered == [1]


def test_classification_uses_the_selected_provider(wired, monkeypatch):
    seen = []

    class RecordingIntent(FakeIntent):
        def classify(self, text, provider="gemini", raise_errors=False):
            seen.append(("intent", provider))
            return super().classify(text)

    class RecordingResolver(FakeResolver):
        def resolve(self, normalized, intent, confidence, provider="gemini", raise_errors=False):
            seen.append(("resolver", provider))
            return super().resolve(normalized, intent, confidence)

    monkeypatch.setattr(N, "_intent_classifier", RecordingIntent())
    monkeypatch.setattr(N, "_context_resolver", RecordingResolver())
    run_workflow(QUERY, provider="groq", trace_timing=False)
    assert seen == [("intent", "groq"), ("resolver", "groq")]


def test_llm_outage_during_classification_abstains_with_a_service_reason(wired, monkeypatch):
    class DownIntent(FakeIntent):
        def classify(self, text, provider="gemini", raise_errors=False):
            raise RuntimeError("404 model_not_found")

    monkeypatch.setattr(N, "_intent_classifier", DownIntent())
    out = run_workflow(QUERY, provider="groq", trace_timing=False)
    assert out["abstain"] is True
    assert "could not be reached" in out["abstain_reason"] and "model_not_found" in out["abstain_reason"]
    assert "rephrase" not in out["abstain_reason"]


def test_intent_classifier_calls_groq_and_reads_markdown_labels(monkeypatch):
    from legal_workflow_generator.agent import groq_client
    from legal_workflow_generator.query.intent_classifier import IntentClassifier
    from legal_workflow_generator.typings.types import QueryIntent

    class NoGemini:
        def generate_content(self, **kw):
            raise AssertionError("gemini must not be used when provider=groq")

    monkeypatch.setattr(groq_client, "complete", lambda system, user, temperature=None:
                        "**INTENT:** workflow\n**CONFIDENCE:** 0.9\n**REASON:** r")
    clf = IntentClassifier.__new__(IntentClassifier)
    clf.client = type("C", (), {"models": NoGemini()})()
    nq = {"original": "q", "normalized": "q", "source": "t"}
    assert clf.classify(nq, provider="groq") == (QueryIntent.WORKFLOW, 0.9)


def test_intent_classifier_raises_api_errors_only_when_asked(monkeypatch):
    from legal_workflow_generator.agent import groq_client
    from legal_workflow_generator.query.intent_classifier import IntentClassifier
    from legal_workflow_generator.typings.types import QueryIntent

    def down(system, user, temperature=None):
        raise RuntimeError("503 UNAVAILABLE")

    monkeypatch.setattr(groq_client, "complete", down)
    clf = IntentClassifier.__new__(IntentClassifier)
    nq = {"original": "q", "normalized": "q", "source": "t"}
    assert clf.classify(nq, provider="groq") == (QueryIntent.UNKNOWN, 0.0)
    with pytest.raises(RuntimeError):
        clf.classify(nq, provider="groq", raise_errors=True)


def test_groundedness_grader_does_not_see_system_annotations(monkeypatch):
    seen = {}

    def grader(schema, system, user, grader=False):      # structured boundary: _llm(Schema, system, user)
        seen["user"] = user
        return schema(reason="r", unsupported_claims=[], grade="grounded")

    monkeypatch.setattr(N, "_llm", grader)
    answer = ("Summary\n- Data Protection: some of these obligations are not yet in force on 2026-11-04; "
              "each step below shows the date it applies from.\n"
              "1. Obtain consent (dpdp_act_2023_sec_6) [NOTE: Not yet in force on 2026-11-04; applies from 2027-05-13.]")
    state = {"answer": answer, "retrieved_docs": [], "domain": "data_protection", "all_domains": ["data_protection"]}
    N._groundedness(state)
    assert "Obtain consent" in seen["user"]
    assert "2027-05-13" not in seen["user"] and "not yet in force" not in seen["user"]


def test_groundedness_grader_sees_the_full_text_of_cited_provisions_only(monkeypatch):
    seen = {}

    def grader(schema, system, user, grader=False):      # structured boundary: _llm(Schema, system, user)
        seen["user"] = user
        return schema(reason="r", unsupported_claims=[], grade="grounded")

    monkeypatch.setattr(N, "_llm", grader)
    long_text = "x " * 600 + "grievance redressal mechanism"          # clause past the first 800 chars
    docs = [{"provision_id": "a", "full_text": long_text}, {"provision_id": "b", "full_text": "uncited text"}]
    state = {"answer": "set up grievance redressal (a)", "retrieved_docs": docs, "verified_citations": ["a"],
             "domain": "data_protection", "all_domains": ["data_protection"]}
    N._groundedness(state)
    assert "grievance redressal mechanism" in seen["user"] and "uncited text" not in seen["user"]


def test_numbers_in_a_step_must_appear_in_the_cited_text():
    docs = [{"provision_id": "p1", "title": "t", "text": "file within 48 hours; fee Rs 1,00,000"}]
    steps = [N.Step("notify within 72 hours", "p1"), N.Step("notify within 48 hours", "p1")]
    verdict = {"verdicts": [{"n": 1, "verdict": "entailed", "reason": "r"},
                            {"n": 2, "verdict": "entailed", "reason": "r"}]}
    import json
    N_llm = lambda *a, **k: json.dumps(verdict)           # noqa: E731
    old = N._llm
    N._llm = N_llm
    try:
        kept, dropped, _ = N._verify_claims("data_protection", steps, docs)
    finally:
        N._llm = old
    assert kept[0].verdict == "partial" and "72" in kept[0].verdict_reason       # 72 is not in the text
    assert kept[1].verdict == "entailed"                                          # 48 is
    assert N._numbers_missing("pay Rs 1,00,000", "fee Rs 1,00,000") == []       # separators are ignored
    assert N._numbers_missing("pay 5 lakh", "fee Rs 1,00,000") == ["5"]


# ─────────────────────────────── typed Provision model ───────────────────────────────
def test_provision_is_typed_and_dict_compatible():
    raw = {"provision_id": "ida_1947_sec_2", "title": "Definitions", "text": None, "combined_score": 0.7,
           "bm25_score": 0.2, "searcher_extra": "dropped", "effective_date": datetime.date(1947, 4, 1)}
    p = N.Provision.from_dict(raw)
    assert isinstance(p, N.Provision)
    assert p.provision_id == "ida_1947_sec_2" and p.text == "" and p.full_text == ""   # None -> ''
    assert p.get("title") == "Definitions" and p["combined_score"] == 0.7 and "searcher_extra" not in p
    assert p.get("missing", "dflt") == "dflt"
    with pytest.raises(KeyError):
        p["searcher_extra"]
    assert p.to_dict()["effective_date"] == datetime.date(1947, 4, 1)


def test_domain_results_hold_typed_provisions_but_graph_state_stays_plain(wired):
    out = run_workflow(QUERY, trace_timing=False)
    assert out["retrieved_docs"] and all(type(d) is dict for d in out["retrieved_docs"])
    r = N.DomainResult(domain="taxation", docs=[N.Provision.from_dict(ALL_DOCS[2])])
    assert isinstance(r.docs[0], N.Provision)
    assert r.to_dict()["docs"] == [{"provision_id": "income_tax_act_1961_sec_195", "title": "Other sums",
                                    "combined_score": 0.6}]


# ─────────────────────────────── broad-question retrieval + unconfirmed domains ───────────────────────────────
def test_plan_queries_parses_json_and_falls_back_on_plain_text(monkeypatch):
    monkeypatch.setattr(N, "_llm", lambda *a, **k:
                        '{"focus": "dpdp compliance", "aspects": ["data fiduciary duties", "consent", "consent", ""]}')
    assert N._plan_queries("data_protection", "q", "fb") == ("dpdp compliance", ["data fiduciary duties", "consent"], [])
    monkeypatch.setattr(N, "_llm", lambda *a, **k: "just a query")
    assert N._plan_queries("data_protection", "q", "fb") == ("just a query", [], [])


FAKE_CATALOG = {d: {"lines": [], "ids": {}, "prefixes": {}} for d in N.VALID_DOMAINS}
FAKE_CATALOG["taxation"]["ids"] = {"cgst_act_2017_sec_9": "cgst_act_2017_sec_9",
                                   "income_tax_act_1961_sec_195": "income_tax_act_1961_sec_195"}
FAKE_CATALOG["ip_licensing"]["ids"] = {"it_act_2000_sec_4": "it_act_2000_sec_4"}


def test_suggestions_keep_only_existing_ids_of_the_domain(monkeypatch):
    monkeypatch.setattr(N, "_catalog", lambda: FAKE_CATALOG)
    raw = ["[CGST_ACT_2017_SEC_9]", "cgst_act_2017_sec_9(1)", "cgst_act_2017_sec_999",
           "ca_2013_sec_7", "income_tax_act_1961_sec_195", ""]
    assert N._resolve_suggestions("taxation", raw) == ["cgst_act_2017_sec_9", "income_tax_act_1961_sec_195"]
    # the IT Act is split over two prefixes: a suggestion under the wrong one still resolves
    assert N._resolve_suggestions("ip_licensing", ["information_technology_act_2000_sec_4"]) == ["it_act_2000_sec_4"]


def test_plan_queries_returns_suggestions_only_when_enabled(monkeypatch):
    from legal_workflow_generator.agent.settings import build_config, use_settings
    monkeypatch.setattr(N, "_catalog", lambda: FAKE_CATALOG)
    monkeypatch.setattr(N, "_llm", lambda *a, **k:
                        '{"focus": "gst", "aspects": [], "provisions": ["cgst_act_2017_sec_9", "made_up_sec_1"]}')
    use_settings(build_config(suggest_provisions=True, suggest_cap=4))
    assert N._plan_queries("taxation", "q", "fb") == ("gst", [], ["cgst_act_2017_sec_9"])
    use_settings(build_config(suggest_provisions=False))
    assert N._plan_queries("taxation", "q", "fb") == ("gst", [], [])
    use_settings(None)


def test_fusion_lifts_a_provision_that_aspect_queries_rank_high(monkeypatch):
    ranked = {"broad": ["p_defs", "p_rules_power"] + [f"p_noise{i}" for i in range(10)] + ["p_duties"],
              "duties": ["p_duties", "p_defs"], "consent": ["p_consent", "p_duties"]}
    monkeypatch.setattr(N, "_domain_candidates",
                        lambda rag, q, dom: [{"provision_id": p} for p in ranked[q]])
    out = [d["provision_id"] for d in N._fused_candidates(None, ["broad", "duties", "consent"], "x")]
    assert set(out[:2]) == {"p_duties", "p_defs"} and out.index("p_consent") < out.index("p_rules_power")
    assert len(out) == len(set(out)) == 14                        # nothing lost, nothing duplicated


def test_unconfirmed_domain_with_irrelevant_provisions_is_dropped(wired, monkeypatch):
    real_grade = N._grade_domain
    monkeypatch.setattr(N, "_grade_domain", lambda dom, q, docs: ("insufficient", "unrelated")
                        if dom == "employment" else real_grade(dom, q, docs))
    out = run_workflow(QUERY, trace_timing=False)                  # LLM suggests employment; retrieval does not
    assert "employment" not in out["all_domains"] and "Employment" not in out["answer"]
    assert any("dropped unconfirmed domain(s) ['employment']" in t for t in out["trace"])


def test_future_provisions_are_kept_and_flagged_by_default(wired, monkeypatch):
    """Phased statutes (DPDP Act duties apply from 13 May 2027) must still be answered, with the date shown."""
    future = datetime.date(2027, 5, 13)
    monkeypatch.setattr(N, "_hydrate", lambda docs: [
        {**d, "effective_date": future if d["provision_id"] == "copyright_act_1957_sec_17" else None} for d in docs])
    out = run_workflow(QUERY, as_of="2026-11-03", trace_timing=False)
    assert any(d["provision_id"] == "copyright_act_1957_sec_17" for d in out["retrieved_docs"])
    assert "[NOTE: Not yet in force on 2026-11-03; applies from 2027-05-13.]" in out["answer"]
    assert "some of these obligations are not yet in force on 2026-11-03" in out["answer"]
    assert any("not yet in force on 2026-11-03 (kept" in t for t in out["trace"])
    with pytest.raises(ValueError):
        S.RunSettings(future_provisions="ignore")


def test_machinery_provisions_go_behind_substantive_ones_unless_asked_about():
    docs = [{"provision_id": "a", "title": "Power to make rules"}, {"provision_id": "b", "title": "Consent"},
            {"provision_id": "c", "title": "Short title and commencement"},
            {"provision_id": "d", "title": "General obligations of Data Fiduciary"}]
    assert [d["provision_id"] for d in N._demote_machinery(docs, "how do I comply")] == ["b", "d", "a", "c"]
    assert [d["provision_id"] for d in N._demote_machinery(docs, "what rules can the government make")] == ["a", "b", "c", "d"]


# ─────────────────────────────── named-statute search ───────────────────────────────
STATUTE_CATALOG = {d: {"lines": [], "ids": {}, "prefixes": {}} for d in N.VALID_DOMAINS}
STATUTE_CATALOG["data_protection"]["prefixes"] = {"spdi_rules_2011_sec_": "", "dpdp_act_2023_sec_": "",
                                                  "dpdp_rules_2025_sec_": ""}
STATUTE_CATALOG["taxation"]["prefixes"] = {"esi_act_1948_sec_": "", "sez_act_2005_sec_": ""}
STATUTE_CATALOG["employment"]["prefixes"] = {"esia_1948_sec_": ""}


@pytest.mark.parametrize("question, domain, expected", [
    ("what counts as sensitive personal data under the SPDI rules", "data_protection", ["spdi_rules_2011_"]),
    ("DPDP consent requirements", "data_protection", ["dpdp_act_2023_", "dpdp_rules_2025_"]),
    ("ESI contributions for our staff", "employment", ["esia_1948_"]),   # only this domain's prefix
    ("ESI contributions for our staff", "taxation", ["esi_act_1948_"]),
    ("tax exemptions for a unit in a special economic zone", "data_protection", []),   # other domain
    ("how should we handle customer data securely", "data_protection", []),           # names no statute
])
def test_named_statutes_match_names_of_this_domain_only(monkeypatch, question, domain, expected):
    monkeypatch.setattr(N, "_catalog", lambda: STATUTE_CATALOG)
    assert N._named_statutes(question, domain) == expected


def test_question_naming_no_statute_never_loads_the_catalog(monkeypatch):
    def boom():
        raise AssertionError("catalog must not be loaded")
    monkeypatch.setattr(N, "_catalog", boom)
    assert N._named_statutes("who owns code written by a freelancer", "ip_licensing") == []


def test_named_statute_search_joins_the_fusion_and_can_be_switched_off(monkeypatch):
    from legal_workflow_generator.agent.settings import build_config, use_settings
    monkeypatch.setattr(N, "_catalog", lambda: STATUTE_CATALOG)
    seen = []

    def cands(rag, q, dom, prefixes=None):
        seen.append(prefixes)
        return [{"provision_id": "spdi_rules_2011_sec_3"}] if prefixes else [{"provision_id": "it_act_sec_43a"}]
    monkeypatch.setattr(N, "_domain_candidates", cands)
    q = "what is sensitive personal data under the SPDI rules"
    use_settings(build_config(statute_search=True))
    out = [d["provision_id"] for d in N._fused_candidates(None, [q], "data_protection", named_from=q)]
    assert seen == [None, ["spdi_rules_2011_"]] and "spdi_rules_2011_sec_3" in out
    seen.clear()
    use_settings(build_config(statute_search=False))
    out = [d["provision_id"] for d in N._fused_candidates(None, [q], "data_protection", named_from=q)]
    assert seen == [None] and out == ["it_act_sec_43a"]
    use_settings(None)


def test_bm25_prefix_filter_ranks_only_that_statute():
    from legal_workflow_generator.rag.bm25_retriever import BM25Retriever
    from rank_bm25 import BM25Okapi
    r = BM25Retriever()
    r.law_ids = ["income_tax_act_1961_sec_1", "spdi_rules_2011_sec_3", "income_tax_act_1961_sec_2"]
    r.law_domains = ["taxation", "data_protection", "taxation"]
    docs = ["sensitive personal data password", "sensitive personal data information", "sensitive data tax"]
    r.bm25 = BM25Okapi([r._tokenize(d) for d in docs])
    assert [p for p, _ in r.search("sensitive personal data", top_k=5, prefixes=["spdi_rules_2011_"])] \
        == ["spdi_rules_2011_sec_3"]
    assert len(r.search("sensitive personal data", top_k=5)) == 3


# ─────────────────────────────── statute text shown to the LLM ───────────────────────────────
def test_full_text_puts_the_original_opening_clause_before_the_other_clauses():
    sub = {"2": "petroleum is levied later", "3": "reverse charge"}
    raw = "(1) there shall be levied a tax on the value determined under section 15"
    ft = N._full_text("Levy and collection", "levy and collection. there shall be levied ... section and", sub, raw)
    assert ft.index("(1) there shall be levied") < ft.index("(2) petroleum") < ft.index("(3) reverse")
    assert "section 15" in ft and "section and" not in ft      # the normalised search blob is not shown
    assert N._numbers_missing("tax on the value under section 15", ft) == []


def test_full_text_without_raw_text_keeps_the_old_layout():
    ft = N._full_text("T", "blob", {"2": "two"})
    assert ft == "T\n(2) two\nblob"
    assert N._full_text("T", "blob", {"2": "two"}, raw="   ") == ft


# ─────────────────────────────── deterministic scope check ───────────────────────────────
SCOPE_CATALOG = {d: {"lines": [], "prefixes": {}, "ids": {}} for d in N.VALID_DOMAINS}
SCOPE_CATALOG["data_protection"]["ids"] = {"dpdp_act_2023_sec_33": "dpdp_act_2023_sec_33",
                                           "dpdp_act_2023_sec_8": "dpdp_act_2023_sec_8"}
SCOPE_CATALOG["employment"]["ids"] = {"poga_1972_sec_4": "poga_1972_sec_4"}
AS_OF = datetime.date(2026, 10, 15)


@pytest.mark.parametrize("query, refused", [
    ("what is the penalty under section 999 of the DPDP Act", True),
    ("explain section 500 of the Payment of Gratuity Act", True),
    ("what does section 8 of the DPDP Act require", False),
    ("penalty under section 33 of the DPDP Act", False),
    ("what does section 12 say", False),                          # names no statute: not checked
    ("what will the GST rate on SaaS subscriptions be in 2035", True),
    ("DPDP obligations that start in 2027", False),               # as_of + 1 stays in scope
    ("advance tax instalments under the Income Tax Act 2025", False),
    ("who owns code written by a freelancer", False),
])
def test_scope_check(monkeypatch, query, refused):
    monkeypatch.setattr(N, "_catalog", lambda: SCOPE_CATALOG)
    assert bool(N._scope_check(query, AS_OF)) is refused


def test_out_of_scope_query_abstains_without_any_llm_call(wired, monkeypatch):
    monkeypatch.setattr(N, "_catalog", lambda: SCOPE_CATALOG)
    out = run_workflow("what is the penalty under section 999 of the DPDP Act", as_of="2026-10-15")
    assert out["abstain"] and "section 999" in out["abstain_reason"] and wired.calls == []


# ── select_context: grader picks the generator's context from a wider pool ─────────
def _pool(*ids, anchor=()):
    return [{"provision_id": i, "title": i, "text": "t", "anchor": i in anchor} for i in ids]


def test_grade_select_keeps_only_pool_ids_in_order(monkeypatch):
    pool = _pool("a_sec_1", "a_sec_2", "a_sec_3")
    monkeypatch.setattr(N, "_llm", lambda *a, **k:
                        '{"grade": "sufficient", "reason": "r", "relevant": ["a_sec_3", "made_up_sec_9", "A_SEC_1", "a_sec_3"]}')
    grade, reason, picked = N._grade_select("employment", "q", pool)
    assert grade == "sufficient" and picked == ["a_sec_3", "a_sec_1"]


def test_grade_select_unparseable_reply_is_insufficient(monkeypatch):
    monkeypatch.setattr(N, "_llm", lambda *a, **k: "no json here")
    assert N._grade_select("employment", "q", _pool("a_sec_1"))[0] == "insufficient"


def test_keep_selected_adds_picks_to_the_ordinary_window():
    pool = _pool("anc_1", *[f"r{i}" for i in range(20)], "sug_1", anchor=("anc_1",))
    kept = [d["provision_id"] for d in N._keep_selected(pool, ["r15", "r2"], ["sug_1"], ["r0", "r1", "r2"])]
    assert kept == ["r15", "r2", "r0", "r1", "anc_1", "sug_1"]     # picks first, window, then anchors + suggestions


def test_keep_selected_pads_small_context_and_defaults_window():
    pool = _pool(*[f"r{i}" for i in range(15)])
    assert [d["provision_id"] for d in N._keep_selected(pool, [], [], ["r3"])] == ["r3", "r0", "r1", "r2", "r4"]
    kept = N._keep_selected(pool, [], [])
    assert kept == N._cut(pool, S.current_settings())             # default window: the ordinary top_k cut


def test_generator_prompt_lists_key_provisions(monkeypatch):
    seen = {}

    def fake_llm(*args, **kwargs):      # works for _llm(system, user) and the structured _llm(Schema, system, user)
        seen["u"] = next(a for a in args if isinstance(a, str) and "Legal area:" in a)
        return '{"steps": []}'
    monkeypatch.setattr(N, "_llm", fake_llm)
    N._generate_domain("employment", "q", _pool("a_sec_1"), ["a_sec_1"])
    assert "marked these provisions as needed for a complete answer: a_sec_1" in seen["u"]
    seen.clear()
    N._generate_domain("employment", "q", _pool("a_sec_1"))
    assert "marked these provisions" not in seen["u"]


def test_select_settings_validated():
    with pytest.raises(ValueError):
        S.RunSettings(select_pool=0)
    assert S.RunSettings.from_config({"configurable": {"select_context": "true", "select_pool": "16"}}).select_pool == 16
