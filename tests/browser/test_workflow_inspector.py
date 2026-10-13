"""Browser-level tests for the chat UI and the workflow inspector.

The real FastAPI app is served on localhost, but `/api/chat` is intercepted by Playwright and answered
with a fixture, so these tests need no database, no Gemini/Groq key and no network.

Run:   pytest -m browser tests/browser
Needs: `pip install playwright` and a Chromium build. Set PLAYWRIGHT_CHROMIUM_EXECUTABLE to use a
       specific binary; otherwise Playwright's bundled browser (`python -m playwright install chromium`).
Tests skip themselves (with the reason) when the browser cannot be started.
"""
import json
import os
import socket
import threading
import time

import pytest

pytestmark = pytest.mark.browser

playwright_sync = pytest.importorskip("playwright.sync_api", reason="playwright is not installed")


# ── fixtures ─────────────────────────────────────────────────────────────────
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def base_url():
    import uvicorn
    import app as app_module   # imports the FastAPI app (no DB or LLM call at import time)

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app_module.app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def browser():
    try:
        with playwright_sync.sync_playwright() as p:
            launch_args = ["--no-sandbox", "--disable-gpu", "--headless=new"]
            exe = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE") or None
            try:
                b = p.chromium.launch(executable_path=exe, args=launch_args)
            except Exception as e:   # noqa: BLE001
                pytest.skip(f"could not start Chromium: {e.__class__.__name__}: {str(e)[:200]}")
            yield b
            b.close()
    except Exception as e:  # noqa: BLE001
        if isinstance(e, pytest.skip.Exception):
            raise
        pytest.skip(f"Playwright unavailable: {e}")


@pytest.fixture
def page(browser, base_url):
    pg = browser.new_page(viewport={"width": 1280, "height": 900})
    pg.goto(base_url + "/")
    yield pg
    pg.close()


def chat_fixture(**over):
    """A response shaped exactly like the /api/chat contract."""
    body = {
        "answer": "COMPLIANCE WORKFLOW\nSTITCHED ANSWER\n⚠ KNOWLEDGE GAPS IDENTIFIED",
        "abstained": False,
        "settings": {"provider": "groq", "top_k": 4, "as_of": "2026-10-13", "verify_claims": True},
        "pipeline": {
            "settings": {"provider": "groq", "top_k": 4, "as_of": "2026-10-13", "verify_claims": True},
            "classification": {"intent": "workflow", "domain": "ip_licensing", "all_domains": ["ip_licensing"],
                               "normalized_query": "who owns code", "keywords": ["copyright"], "confidence": 0.9},
            "retrieval": {"docs": [{"provision_id": "copyright_act_1957_sec_17", "title": "First owner",
                                    "text": "...", "effective_date": None, "combined_score": 0.8,
                                    "bm25_score": 0.5, "semantic_score": 0.9}],
                          "rewritten_query": None, "retry_count": 0},
            "domains": [{
                "domain": "ip_licensing", "grade": "sufficient", "reason": "ok", "retries": 0,
                "steps": [
                    {"action": "Contractor owns code unless assigned", "provision_id": "copyright_act_1957_sec_17",
                     "applies_to": "private", "verdict": "entailed", "verdict_reason": "", "note": ""},
                    {"action": "Assignment needs 30 days notice", "provision_id": "copyright_act_1957_sec_17",
                     "applies_to": "private", "verdict": "partial", "verdict_reason": "figure not in text",
                     "note": ""},
                ],
                "dropped_steps": [{"action": "invented rule", "provision_id": "copyright_act_1957_sec_17",
                                   "applies_to": "both", "verdict": "unsupported", "verdict_reason": "",
                                   "note": ""}],
                "cited": ["copyright_act_1957_sec_17"], "verification": "ok", "error": "",
            }],
            "domain_gaps": ["- Taxation & GST: insufficient information found in knowledge base"],
            "context_grading": {"grade": "sufficient", "reason": "ip_licensing: sufficient"},
            "generation": {"citations": ["copyright_act_1957_sec_17"], "retry_count": 0},
            "citation_verification": {"verified": ["copyright_act_1957_sec_17"], "failed": []},
            "claim_verification": {"entailed": 1, "partial": 1, "unchecked": 0, "dropped": 1},
            "groundedness": {"grade": "grounded", "reason": "ok"},
            "answerability": {"grade": "partial", "reason": "Covered: ip_licensing."},
            "abstain_info": {"triggered": False, "reason": ""},
        },
        "trace": ["classify → intent=workflow", "[ip_licensing] result → 2 steps"],
    }
    body.update(over)
    return body


def send_query(page, text="Who owns code written by a contractor?"):
    page.fill("#chat-input", text)
    page.click("#send-btn")


# ── tests ────────────────────────────────────────────────────────────────────
def test_provider_and_top_k_selected_in_the_ui_are_sent_to_the_api(page):
    captured = {}

    def handle(route):
        captured["body"] = json.loads(route.request.post_data)
        route.fulfill(status=200, content_type="application/json", body=json.dumps(chat_fixture()))

    page.route("**/api/chat", handle)
    page.select_option("#provider-select", "groq")
    page.eval_on_selector("#topk-slider", "el => { el.value = 4; el.dispatchEvent(new Event('input')); }")
    send_query(page)
    page.wait_for_selector("text=STITCHED ANSWER")

    assert captured["body"]["provider"] == "groq"
    assert captured["body"]["top_k"] == 4
    assert captured["body"]["query"].startswith("Who owns code")


def test_answer_shows_the_knowledge_gap_section_returned_by_the_api(page):
    page.route("**/api/chat", lambda r: r.fulfill(status=200, content_type="application/json",
                                                  body=json.dumps(chat_fixture())))
    send_query(page)
    page.wait_for_selector("text=KNOWLEDGE GAPS IDENTIFIED")
    assert page.locator("text=STITCHED ANSWER").count() == 1


def test_inspector_shows_settings_and_per_domain_claim_verdicts(page):
    page.route("**/api/chat", lambda r: r.fulfill(status=200, content_type="application/json",
                                                  body=json.dumps(chat_fixture())))
    send_query(page)
    page.wait_for_selector(".pipeline-toggle")
    page.click(".pipeline-toggle")

    stage = page.locator(".stage", has_text="Domains & Claim Checks")
    assert stage.count() == 1
    assert "entailed 1" in stage.locator(".stage-summary").inner_text()
    assert "dropped 1" in stage.locator(".stage-summary").inner_text()
    stage.locator(".stage-header").click()
    body = stage.locator(".stage-body").inner_text()
    assert "[entailed] Contractor owns code unless assigned" in body
    assert "[partial] Assignment needs 30 days notice" in body
    assert "DROPPED: invented rule" in body
    assert "Taxation & GST" in body                       # knowledge gap listed

    settings = page.locator(".stage", has_text="Run Settings")
    assert "groq" in settings.locator(".stage-summary").inner_text()
    assert "top_k=4" in settings.locator(".stage-summary").inner_text()


def test_api_error_is_shown_to_the_user(page):
    page.route("**/api/chat", lambda r: r.fulfill(
        status=400, content_type="application/json",
        body=json.dumps({"detail": "provider must be one of ('gemini', 'groq'), got 'openai'"})))
    send_query(page, "anything")
    page.wait_for_selector("text=provider must be one of")
