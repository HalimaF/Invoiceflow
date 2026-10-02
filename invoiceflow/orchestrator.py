"""Orchestrator: runs the agents, decides the next step, records every decision.

Design: the invoice lifecycle is a deterministic state machine (predictable and
auditable); LLM agents are used only inside the steps that need judgement
(extraction, vendor communication). Anything an agent is unsure about goes to a
human review queue instead of being guessed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from .agents import (ApprovalAgent, ApprovalPolicy, IntakeAgent, MatchingAgent,
                     ValidationAgent, VendorQueryAgent)
from .llm import LLMClient
from .models import AgentResult, InvoiceStatus
from .store import Store


@dataclass
class ProcessingOutcome:
    invoice_id: int
    status: InvoiceStatus
    reason: str
    assigned_to: str | None = None
    results: list[AgentResult] = field(default_factory=list)


class Orchestrator:
    def __init__(self, store: Store, llm: LLMClient, review_threshold: float = 0.75,
                 policy: ApprovalPolicy | None = None):
        self.store = store
        self.review_threshold = review_threshold
        self.intake = IntakeAgent(llm)
        self.validation = ValidationAgent(store)
        self.matching = MatchingAgent(store)
        self.approval = ApprovalAgent(policy)
        self.vendor_query = VendorQueryAgent(llm)

    def process(self, text: str, source_name: str, today: date | None = None) -> ProcessingOutcome:
        invoice_id = self.store.create_invoice(source_name, text)
        self.store.add_event(invoice_id, "system", "received", {"source": source_name})
        try:
            return self._run(invoice_id, text, today)
        except Exception as exc:
            # Record the failure on the invoice so it is visible in the audit
            # trail, then re-raise so the caller still sees the real error.
            self.store.set_status(invoice_id, InvoiceStatus.FAILED)
            self.store.add_event(invoice_id, "system", "failed",
                                 {"error": f"{type(exc).__name__}: {exc}"})
            raise

    def _finish(self, invoice_id: int, status: InvoiceStatus, reason: str,
                results: list[AgentResult], assigned_to: str | None = None) -> ProcessingOutcome:
        self.store.set_status(invoice_id, status, assigned_to)
        self.store.add_event(invoice_id, "orchestrator", f"status:{status.value}",
                             {"reason": reason, "assigned_to": assigned_to})
        return ProcessingOutcome(invoice_id, status, reason, assigned_to, results)

    def _run(self, invoice_id: int, text: str, today: date | None) -> ProcessingOutcome:
        invoice, intake_result = self.intake.run(text)
        self.store.add_agent_result(invoice_id, intake_result)
        results = [intake_result]
        if invoice is None:
            return self._finish(invoice_id, InvoiceStatus.NEEDS_REVIEW,
                                "Extraction failed; needs manual data entry.", results)
        self.store.save_extracted(invoice_id, invoice)

        validation_result = self.validation.run(invoice, invoice_id, today)
        self.store.add_agent_result(invoice_id, validation_result)
        matching_result = self.matching.run(invoice)
        self.store.add_agent_result(invoice_id, matching_result)
        results += [validation_result, matching_result]

        findings = validation_result.findings + matching_result.findings
        errors = validation_result.errors + matching_result.errors
        lowest = min(r.confidence for r in results)

        if errors or lowest < self.review_threshold:
            reasons = [e.code for e in errors] or [f"low confidence ({lowest:.2f})"]
            if self.vendor_query.fixable(findings):
                query_result = self.vendor_query.run(invoice, findings)
                self.store.add_agent_result(invoice_id, query_result)
                results.append(query_result)
            return self._finish(invoice_id, InvoiceStatus.NEEDS_REVIEW,
                                "Needs review: " + ", ".join(reasons), results)

        approval_result = self.approval.run(invoice, matching_result.data.get("matched", False))
        self.store.add_agent_result(invoice_id, approval_result)
        results.append(approval_result)
        if approval_result.data["route"] == "auto":
            return self._finish(invoice_id, InvoiceStatus.APPROVED, approval_result.reasoning, results)
        return self._finish(invoice_id, InvoiceStatus.PENDING_APPROVAL, approval_result.reasoning,
                            results, assigned_to=approval_result.data["approver"])

    def decide(self, invoice_id: int, approve: bool, actor: str, note: str = "") -> InvoiceStatus:
        """A human approves or rejects an invoice that is waiting on them."""
        record = self.store.get_invoice(invoice_id)
        if record is None:
            raise KeyError(f"invoice {invoice_id} not found")
        waiting = {InvoiceStatus.NEEDS_REVIEW.value, InvoiceStatus.PENDING_APPROVAL.value}
        if record["status"] not in waiting:
            raise ValueError(f"invoice {invoice_id} is {record['status']}, not awaiting a decision")
        status = InvoiceStatus.APPROVED if approve else InvoiceStatus.REJECTED
        self.store.set_status(invoice_id, status)
        self.store.add_event(invoice_id, actor, "approved" if approve else "rejected", {"note": note})
        return status
