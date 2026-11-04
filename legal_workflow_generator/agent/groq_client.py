"""Groq chat completion, used by every LLM call when the run's provider is "groq"."""
import threading

import legal_workflow_generator.config.values as app_config

_client = None
_lock = threading.Lock()


def _get_client():
    global _client
    with _lock:
        if _client is None:
            if not app_config.GROQ_API_KEY:
                raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file to use the groq provider.")
            from groq import Groq
            _client = Groq(api_key=app_config.GROQ_API_KEY)
    return _client


def complete(system: str, user: str, temperature: float | None = None) -> str:
    messages: list = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    r = _get_client().chat.completions.create(model=app_config.GROQ_MODEL, messages=messages, temperature=temperature)
    return (r.choices[0].message.content or "").strip()
