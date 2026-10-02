"""Approval agent: decides who (if anyone) must approve a clean invoice."""

from __future__ import annotations

from dataclasses import dataclass

from ..models import AgentResult, ExtractedInvoice


@dataclass(frozen=True)
class ApprovalPolicy:
    auto_approve_limit: float = 1_000.0
    manager_limit: float = 10_000.0
    manager: str = "manager@company.example"
    director: str = "finance.director@company.example"


class ApprovalAgent:
    name = "approval"

    def __init__(self, policy: ApprovalPolicy | None = None):
        self.policy = policy or ApprovalPolicy()

    def run(self, inv: ExtractedInvoice, po_matched: bool) -> AgentResult:
        p = self.policy
        if po_matched and inv.total <= p.auto_approve_limit:
            route, approver = "auto", None
            reasoning = f"PO-matched and total {inv.total} <= auto-approve limit {p.auto_approve_limit}."
        elif inv.total <= p.manager_limit:
            route, approver = "manager", p.manager
            reasoning = (f"Total {inv.total} <= {p.manager_limit}" +
                         ("" if po_matched else " but no PO match") + "; manager approval required.")
        else:
            route, approver = "director", p.director
            reasoning = f"Total {inv.total} exceeds {p.manager_limit}; finance director approval required."
        return AgentResult(agent=self.name, confidence=1.0, reasoning=reasoning,
                           data={"route": route, "approver": approver})
