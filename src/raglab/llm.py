"""Talking to a chat LLM through any OpenAI-compatible endpoint.

Configure with environment variables (the lab container reads ``.env``):

* ``LLM_BASE_URL`` - e.g. ``http://host.docker.internal:8317/v1`` (cliproxyapi)
* ``LLM_API_KEY``  - the key for that endpoint
* ``LLM_MODEL``    - a model name the endpoint offers (see ``list_models()``)
* ``LLM_REASONING_EFFORT`` - optional: low / medium / high for reasoning models
"""

from __future__ import annotations

from functools import lru_cache
import os


def _client():
    from openai import OpenAI

    return OpenAI(base_url=os.getenv("LLM_BASE_URL", "http://host.docker.internal:8317/v1"),
                  api_key=os.getenv("LLM_API_KEY", "missing"), timeout=60, max_retries=1)


def list_models() -> list[str]:
    return sorted(m.id for m in _client().models.list().data)


@lru_cache(maxsize=1)
def llm_available() -> bool:
    """True when the endpoint answers and the configured model exists."""
    if not os.getenv("LLM_API_KEY") or not os.getenv("LLM_MODEL"):
        return False
    try:
        return os.environ["LLM_MODEL"] in list_models()
    except Exception:
        return False


def explain_setup() -> str:
    return ("No LLM configured. Put LLM_API_KEY and LLM_MODEL in the project's .env file, "
            "then run `scripts/lab up` again. Retrieval lessons work without it.")


def chat(prompt: str | list[dict], system: str | None = None, temperature: float = 0.0,
         max_tokens: int = 2000) -> str:
    """Send a prompt (or a full message list) and return the reply text.

    ``max_tokens`` is generous because reasoning models count their hidden
    "thinking" tokens against it.
    """
    if not llm_available():
        raise RuntimeError(explain_setup())
    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    if system:
        messages = [{"role": "system", "content": system}] + messages
    extra_body = {}
    if os.getenv("LLM_REASONING_EFFORT"):
        extra_body["reasoning_effort"] = os.environ["LLM_REASONING_EFFORT"]
    reply = _client().chat.completions.create(
        model=os.environ["LLM_MODEL"], messages=messages,
        temperature=temperature, max_tokens=max_tokens, extra_body=extra_body or None)
    return (reply.choices[0].message.content or "").strip()
