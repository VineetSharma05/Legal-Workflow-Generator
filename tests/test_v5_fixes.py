"""v5 changes (all behind settings, off by default so v4 runs are reproducible)."""
import pytest

import legal_workflow_generator.agent.nodes as nodes
from legal_workflow_generator.agent.settings import RunSettings, _CURRENT


@pytest.fixture
def settings():
    def use(**kw):
        _CURRENT.set(RunSettings(**kw))
    yield use
    _CURRENT.set(None)


LONG = ("Any person claiming refund of any tax may make an application before the expiry of two years. "
        + "Filler text about unrelated procedure and appeals. " * 60
        + "A registered person may claim refund of unutilised input tax credit on zero rated supplies made "
          "without payment of tax. "
        + "More filler text about withholding of refunds from defaulters. " * 40)


def test_defaults_keep_v4_behaviour():
    s = RunSettings()
    assert (s.snippet_mode, s.verify_chars, s.domain_guide, s.none_abstain, s.strict_grade) == \
        ("rarest", 1500, False, False, False)


def test_bad_settings_are_rejected():
    with pytest.raises(ValueError):
        RunSettings(snippet_mode="middle")
    with pytest.raises(ValueError):
        RunSettings(verify_chars=50)


def test_from_config_parses_v5_options():
    s = RunSettings.from_config({"configurable": {"snippet_mode": "OVERLAP", "verify_chars": "4000",
                                                  "domain_guide": "true", "strict_grade": "1"}})
    assert (s.snippet_mode, s.verify_chars, s.domain_guide, s.strict_grade) == ("overlap", 4000, True, True)


def test_overlap_snippet_finds_the_claimed_sentence(settings):
    claim = "claim refund of unutilised input tax credit for zero rated supplies"
    settings(snippet_mode="overlap")
    out = nodes._snippet(LONG, claim, 1500)
    assert len(out) <= 1510
    assert "zero rated supplies" in out
    assert out.startswith("Any person claiming refund")        # the opening is always kept


def test_short_text_is_returned_whole(settings):
    settings(snippet_mode="overlap")
    assert nodes._snippet("short provision", "anything at all", 1500) == "short provision"


def test_strict_rules_are_empty_when_off(settings):
    settings()
    assert nodes._strict_rule("taxation") == "" and nodes._strict_pick_rule() == ""
    settings(strict_grade=True)
    assert "Income-tax Act" in nodes._strict_rule("taxation")
    assert '"relevant" must be []' in nodes._strict_pick_rule()


@pytest.mark.parametrize("reply,expected", [
    ("NONE", None),
    ("none of these", None),
    ("taxation, employment", ["taxation", "employment"]),
    ("corporate_governance\ndata_protection", ["corporate_governance", "data_protection"]),
    ("garbage", []),
])
def test_guided_domain_list_parsing(settings, monkeypatch, reply, expected):
    settings(domain_guide=True)
    seen = {}
    monkeypatch.setattr(nodes, "_llm", lambda system, user, **kw: seen.setdefault("p", user) and reply)
    assert nodes._llm_domains("q") == expected
    assert "Consumer Protection Act" in seen["p"] and "NONE" in seen["p"]
