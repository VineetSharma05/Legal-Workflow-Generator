"""
Shared LangChain chat models + structured-output helper.

Every LLM call in the project goes through a model built here; no module talks
to the google-genai or groq SDKs directly. Agent and query-unit calls return a
Pydantic object via `with_structured_output`; nothing parses free text.
Transient API errors (429/5xx) are retried by the provider client (`max_retries`).
"""
from functools import lru_cache
from typing import TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from pydantic import BaseModel

from legal_workflow_generator.config.values import GEMINI_API_KEY, GEMINI_MODEL, GROQ_API_KEY, GROQ_MODEL

T = TypeVar("T", bound=BaseModel)

MAX_RETRIES = 5


class StructuredOutputError(Exception):
    """The model replied, but the reply did not validate against the schema."""


@lru_cache(maxsize=None)
def chat_model(temperature: float | None = None) -> ChatGoogleGenerativeAI:
    kwargs = {} if temperature is None else {"temperature": temperature}
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
        max_retries=MAX_RETRIES,
        **kwargs,
    )


@lru_cache(maxsize=None)
def groq_chat_model(temperature: float = 0.0) -> ChatGroq:
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. Add it to your .env file to use the groq provider."
        )
    return ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=temperature,
        max_retries=MAX_RETRIES,
    )


def structured_llm(schema: type[T], temperature: float | None = None) -> Runnable:
    """Runnable that returns {"raw", "parsed", "parsing_error"} for `schema`."""
    return chat_model(temperature).with_structured_output(schema, method="json_schema", include_raw=True)


def invoke_structured(llm: Runnable, system: str, user: str) -> BaseModel:
    """
    Call a `structured_llm` runnable and return the parsed object.
    Raises StructuredOutputError on a schema mismatch, so callers can tell a
    malformed reply apart from a genuine negative verdict.
    """
    out = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    if out.get("parsing_error") is not None or out.get("parsed") is None:
        raise StructuredOutputError(str(out.get("parsing_error") or "empty structured response"))
    return out["parsed"]
