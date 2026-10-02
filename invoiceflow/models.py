"""Data models shared by all agents."""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class LineItem(BaseModel):
    description: str = Field(min_length=1)
    quantity: float = Field(gt=0)
    unit_price: float = Field(ge=0)
    amount: float = Field(ge=0)


class ExtractedInvoice(BaseModel):
    """Structured invoice produced by the intake agent."""

    vendor_name: str = Field(min_length=1)
    invoice_number: str = Field(min_length=1)
    invoice_date: date
    due_date: date | None = None
    po_number: str | None = None
    currency: str = Field(default="USD", min_length=3, max_length=3)
    line_items: list[LineItem] = Field(min_length=1)
    subtotal: float = Field(ge=0)
    tax: float = Field(default=0.0, ge=0)
    total: float = Field(ge=0)
    bank_account: str | None = None


class POLine(BaseModel):
    description: str
    quantity: float = Field(gt=0)
    unit_price: float = Field(ge=0)


class PurchaseOrder(BaseModel):
    po_number: str
    vendor_name: str
    currency: str = "USD"
    lines: list[POLine]

    @property
    def total(self) -> float:
        return round(sum(line.quantity * line.unit_price for line in self.lines), 2)


class Vendor(BaseModel):
    name: str
    bank_account: str | None = None
    email: str | None = None


class InvoiceStatus(str, Enum):
    RECEIVED = "received"
    NEEDS_REVIEW = "needs_review"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    FAILED = "failed"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class Finding(BaseModel):
    code: str
    severity: Severity
    message: str


class AgentResult(BaseModel):
    """What every agent returns: a decision the orchestrator and a human can audit."""

    agent: str
    confidence: float = Field(ge=0, le=1)
    reasoning: str
    findings: list[Finding] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.WARNING]
