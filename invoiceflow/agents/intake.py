"""Intake agent: raw invoice text -> validated ExtractedInvoice.

The agentic part is the self-correction loop: when the model's answer fails
schema validation, the exact validation errors are sent back so it can fix them.
"""

from __future__ import annotations

from pydantic import ValidationError

from ..llm import LLMClient, LLMError
from ..models import AgentResult, ExtractedInvoice, Finding, Severity

SYSTEM_PROMPT = """You are an accounts-payable data extraction agent.
Extract the invoice into ONE JSON object with exactly these keys:
vendor_name (string), invoice_number (string), invoice_date (YYYY-MM-DD),
due_date (YYYY-MM-DD or null), po_number (string or null), currency (ISO 4217 code),
line_items (array of {description, quantity, unit_price, amount}),
subtotal (number), tax (number), total (number), bank_account (string or null).
Numbers must be plain numbers without currency symbols or thousands separators.
Copy values exactly as printed. Use null for anything that is not on the invoice; never guess."""


def _format_errors(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors()[:10]:
        loc = ".".join(str(p) for p in err["loc"]) or "(root)"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)


class IntakeAgent:
    name = "intake"

    def __init__(self, llm: LLMClient, max_attempts: int = 3):
        self.llm = llm
        self.max_attempts = max_attempts

    def run(self, text: str) -> tuple[ExtractedInvoice | None, AgentResult]:
        if not text.strip():
            return None, AgentResult(
                agent=self.name, confidence=0.0, reasoning="Document contains no text.",
                findings=[Finding(code="EMPTY_DOCUMENT", severity=Severity.ERROR,
                                  message="No text could be read from the document")],
            )

        base_prompt = f"Invoice text:\n---\n{text}\n---"
        feedback: str | None = None
        history: list[str] = []
        for attempt in range(1, self.max_attempts + 1):
            prompt = base_prompt
            if feedback:
                prompt += (f"\n\nYour previous answer was rejected: {feedback}\n"
                           "Return the complete corrected JSON object.")
            try:
                raw = self.llm.complete_json("extract_invoice", SYSTEM_PROMPT, prompt)
                invoice = ExtractedInvoice.model_validate(raw)
            except LLMError as exc:
                feedback = str(exc)
            except ValidationError as exc:
                feedback = _format_errors(exc)
            else:
                missing = [f for f in ("due_date", "po_number") if getattr(invoice, f) is None]
                confidence = max(0.0, 1.0 - 0.15 * (attempt - 1) - 0.05 * len(missing))
                reasoning = f"Extracted on attempt {attempt}."
                if history:
                    reasoning += " Corrected after: " + " | ".join(history)
                findings = [Finding(code="MISSING_FIELD", severity=Severity.INFO,
                                    message=f"{f} not present on invoice") for f in missing]
                return invoice, AgentResult(
                    agent=self.name, confidence=round(confidence, 2), reasoning=reasoning,
                    findings=findings, data={"attempts": attempt},
                )
            history.append(f"attempt {attempt}: {feedback}")

        return None, AgentResult(
            agent=self.name, confidence=0.0,
            reasoning=f"Could not extract a valid invoice in {self.max_attempts} attempts.",
            findings=[Finding(code="EXTRACTION_FAILED", severity=Severity.ERROR,
                              message=" | ".join(history))],
            data={"attempts": self.max_attempts},
        )
