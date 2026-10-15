"""RotatingGeminiClient: falls back to the next API key when a call fails."""

import pytest

from legal_workflow_generator.agent import gemini_client as gc


class _Models:
    def __init__(self, key, log, failing):
        self.key, self.log, self.failing = key, log, failing

    def generate_content(self, **kwargs):
        self.log.append(self.key)
        if self.key in self.failing:
            raise RuntimeError(self.failing[self.key])
        return f"ok:{self.key}"


@pytest.fixture
def fake_keys(monkeypatch):
    log, failing = [], {}

    class _Client:
        def __init__(self, api_key):
            self.models = _Models(api_key, log, failing)

    monkeypatch.setattr(gc.genai, "Client", _Client)
    return log, failing


def test_config_splits_comma_separated_keys(monkeypatch):
    import importlib
    import legal_workflow_generator.config.values as values

    monkeypatch.setenv("GEMINI_API_KEY", " k1, k2 ,,k3 ")
    try:
        importlib.reload(values)
        assert values.GEMINI_API_KEYS == ["k1", "k2", "k3"]
        assert values.GEMINI_API_KEY == "k1"
    finally:
        monkeypatch.undo()
        importlib.reload(values)


def test_uses_first_key_when_it_works(fake_keys):
    log, _ = fake_keys
    client = gc.RotatingGeminiClient(keys=["a", "b"])
    assert client.models.generate_content(model="m", contents="x") == "ok:a"
    assert log == ["a"]


def test_rate_limited_key_falls_back_and_sticks(fake_keys):
    log, failing = fake_keys
    failing["a"] = "429 RESOURCE_EXHAUSTED. Please retry in 30s."
    client = gc.RotatingGeminiClient(keys=["a", "b", "c"])

    assert client.models.generate_content(model="m", contents="x") == "ok:b"
    # next call starts on the working key; the failed one is cooling down
    assert client.models.generate_content(model="m", contents="x") == "ok:b"
    assert log == ["a", "b", "b"]


def test_any_failure_rotates(fake_keys):
    log, failing = fake_keys
    failing["a"] = "400 API_KEY_INVALID"
    failing["b"] = "503 UNAVAILABLE"
    client = gc.RotatingGeminiClient(keys=["a", "b", "c"])
    assert client.models.generate_content(model="m", contents="x") == "ok:c"
    assert log == ["a", "b", "c"]


def test_all_keys_failing_raises_last_error(fake_keys):
    log, failing = fake_keys
    failing.update({"a": "429 quota a", "b": "429 GenerateRequestsPerDay quota b"})
    client = gc.RotatingGeminiClient(keys=["a", "b"])
    with pytest.raises(RuntimeError, match="PerDay"):
        client.models.generate_content(model="m", contents="x")
    assert log == ["a", "b"]


def test_cooling_keys_are_retried_when_nothing_else_is_ready(fake_keys):
    log, failing = fake_keys
    failing["a"] = "429 retry in 10s"
    client = gc.RotatingGeminiClient(keys=["a"])
    with pytest.raises(RuntimeError):
        client.models.generate_content(model="m", contents="x")
    del failing["a"]
    assert client.models.generate_content(model="m", contents="x") == "ok:a"


def test_clients_share_one_pool(fake_keys, monkeypatch):
    log, failing = fake_keys
    monkeypatch.setattr(gc, "_shared_pool", gc.KeyPool(["a", "b"]))
    failing["a"] = "429"
    gc.RotatingGeminiClient().models.generate_content(model="m", contents="x")
    log.clear()
    gc.RotatingGeminiClient().models.generate_content(model="m", contents="x")
    assert log == ["b"]


def test_bad_key_is_skipped_for_long_cooldown():
    assert gc._cooldown_for("403 PERMISSION_DENIED. project denied") == gc._DAILY_COOLDOWN
    assert gc._cooldown_for("429 RESOURCE_EXHAUSTED retry in 11.5s") == pytest.approx(12.5)
