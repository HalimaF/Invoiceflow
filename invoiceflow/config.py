"""Configuration from environment variables."""

from __future__ import annotations

import os

from .llm import GroqClient, HeuristicLLM, LLMClient

# Default model is configurable because Groq's model catalogue changes;
# check https://console.groq.com/docs/models for current IDs.
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


def db_path() -> str:
    return os.getenv("INVOICEFLOW_DB", "invoiceflow.db")


def build_llm() -> LLMClient:
    """LLM_PROVIDER=groq|mock. Defaults to groq when GROQ_API_KEY is set, else mock."""
    provider = os.getenv("LLM_PROVIDER") or ("groq" if os.getenv("GROQ_API_KEY") else "mock")
    if provider == "mock":
        return HeuristicLLM()
    if provider == "groq":
        return GroqClient(
            api_key=os.getenv("GROQ_API_KEY", ""),
            model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
            base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        )
    raise ValueError(f"Unknown LLM_PROVIDER {provider!r} (expected 'groq' or 'mock')")
