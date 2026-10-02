"""LLM clients.

GroqClient talks to Groq's OpenAI-compatible chat completions API.
HeuristicLLM is an offline stand-in so the whole pipeline runs (and is tested)
without an API key; it only understands the simple format used in samples/.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

import httpx


class LLMError(Exception):
    """The model answered, but not with something we can use."""


class LLMClient(Protocol):
    def complete_json(self, task: str, system: str, user: str) -> dict:
        """Return the model's answer parsed as a JSON object."""
        ...


class GroqClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.groq.com/openai/v1",
        timeout: float = 60.0,
        http_client: httpx.Client | None = None,
    ):
        if not api_key:
            raise ValueError("GROQ_API_KEY is required for GroqClient")
        self.model = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._http = http_client or httpx.Client(timeout=timeout)

    def complete_json(self, task: str, system: str, user: str) -> dict:
        body = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        # Transport and HTTP errors (bad key, rate limit) are not the model's
        # fault and retrying the prompt won't fix them, so they propagate as-is.
        response = self._http.post(self._url, json=body, headers=self._headers)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMError(f"model returned invalid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise LLMError("model returned JSON that is not an object")
        return parsed


_FIELD_PATTERNS = {
    "vendor_name": r"^vendor:\s*(.+)$",
    "invoice_number": r"^invoice\s*(?:no|number|#)\.?:\s*(.+)$",
    "invoice_date": r"^(?:invoice\s+)?date:\s*(.+)$",
    "due_date": r"^due(?:\s+date)?:\s*(.+)$",
    "po_number": r"^po(?:\s*number)?:\s*(.+)$",
    "currency": r"^currency:\s*(.+)$",
    "bank_account": r"^bank(?:\s+account)?:\s*(.+)$",
    "subtotal": r"^subtotal:\s*([\d,.]+)",
    "tax": r"^tax:\s*([\d,.]+)",
    "total": r"^total:\s*([\d,.]+)",
}
_NUMERIC = {"subtotal", "tax", "total"}


def _num(value: str) -> float:
    return float(value.replace(",", ""))


class HeuristicLLM:
    """Deterministic, offline replacement for an LLM (demo mode and tests)."""

    def complete_json(self, task: str, system: str, user: str) -> dict:
        if task == "extract_invoice":
            return self._extract(user)
        if task == "draft_vendor_query":
            return self._draft_query(user)
        raise LLMError(f"HeuristicLLM does not support task {task!r}")

    def _extract(self, text: str) -> dict:
        result: dict = {"line_items": []}
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line.count("|") == 3:
                desc, qty, price, amount = (p.strip() for p in line.split("|"))
                try:
                    result["line_items"].append(
                        {"description": desc, "quantity": _num(qty),
                         "unit_price": _num(price), "amount": _num(amount)}
                    )
                except ValueError:
                    pass  # header row such as "Description | Qty | ..."
                continue
            for field, pattern in _FIELD_PATTERNS.items():
                if field in result:
                    continue
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    result[field] = _num(value) if field in _NUMERIC else value
                    break
        return result

    def _draft_query(self, user: str) -> dict:
        info = json.loads(user)
        issues = "\n".join(f"- {f['message']}" for f in info["findings"])
        return {
            "subject": f"Query on invoice {info['invoice_number']}",
            "body": (
                f"Dear {info['vendor_name']},\n\n"
                f"We could not process invoice {info['invoice_number']} because:\n"
                f"{issues}\n\n"
                "Please send a corrected invoice or clarify these points.\n\n"
                "Kind regards,\nAccounts Payable"
            ),
        }
