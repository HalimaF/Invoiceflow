"""Vendor-query agent: drafts an email asking the vendor to fix their invoice."""

from __future__ import annotations

import json

from ..llm import LLMClient, LLMError
from ..models import AgentResult, ExtractedInvoice, Finding

# Issues the vendor can fix by sending a corrected invoice. Fraud signals such as
# BANK_ACCOUNT_CHANGED are deliberately excluded: those must be verified through
# known contact details, never by replying to the invoice itself.
VENDOR_FIXABLE = {
    "LINE_MATH", "SUBTOTAL_MISMATCH", "TOTAL_MISMATCH", "DUE_BEFORE_INVOICE",
    "NO_PO", "PO_NOT_FOUND", "LINE_NOT_ON_PO", "QTY_EXCEEDS_PO", "PRICE_VARIANCE",
    "CURRENCY_MISMATCH",
}

SYSTEM_PROMPT = """You are an accounts-payable assistant writing to a vendor.
Write a short, polite, professional email explaining exactly what is wrong with
their invoice and what they need to send back. Use only the facts provided.
Return JSON: {"subject": string, "body": string}."""


class VendorQueryAgent:
    name = "vendor_query"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    @staticmethod
    def fixable(findings: list[Finding]) -> list[Finding]:
        return [f for f in findings if f.code in VENDOR_FIXABLE]

    def run(self, inv: ExtractedInvoice, findings: list[Finding]) -> AgentResult:
        relevant = self.fixable(findings)
        user = json.dumps({
            "vendor_name": inv.vendor_name,
            "invoice_number": inv.invoice_number,
            "po_number": inv.po_number,
            "findings": [f.model_dump(mode="json") for f in relevant],
        })
        try:
            draft = self.llm.complete_json("draft_vendor_query", SYSTEM_PROMPT, user)
        except LLMError as exc:
            return AgentResult(agent=self.name, confidence=0.0,
                               reasoning=f"Could not draft email: {exc}")
        if not isinstance(draft.get("subject"), str) or not isinstance(draft.get("body"), str):
            return AgentResult(agent=self.name, confidence=0.0,
                               reasoning="Model reply was missing subject/body.")
        return AgentResult(
            agent=self.name, confidence=0.8,
            reasoning=f"Drafted vendor email covering {len(relevant)} issue(s); awaiting human send.",
            data={"subject": draft["subject"], "body": draft["body"]},
        )
