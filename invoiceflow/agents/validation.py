"""Validation agent: arithmetic, dates, duplicates and vendor/bank checks."""

from __future__ import annotations

from datetime import date

from ..models import AgentResult, ExtractedInvoice, Finding, Severity
from ..store import Store


class ValidationAgent:
    name = "validation"

    def __init__(self, store: Store, tolerance: float = 0.01):
        self.store = store
        self.tol = tolerance

    def run(self, inv: ExtractedInvoice, invoice_id: int, today: date | None = None) -> AgentResult:
        today = today or date.today()
        findings: list[Finding] = []

        def add(code: str, severity: Severity, message: str) -> None:
            findings.append(Finding(code=code, severity=severity, message=message))

        for i, item in enumerate(inv.line_items, start=1):
            expected = round(item.quantity * item.unit_price, 2)
            if abs(expected - item.amount) > self.tol:
                add("LINE_MATH", Severity.ERROR,
                    f"Line {i} '{item.description}': {item.quantity} x {item.unit_price} = {expected}, invoice says {item.amount}")

        line_sum = round(sum(item.amount for item in inv.line_items), 2)
        if abs(line_sum - inv.subtotal) > self.tol:
            add("SUBTOTAL_MISMATCH", Severity.ERROR,
                f"Line items sum to {line_sum}, subtotal says {inv.subtotal}")
        if abs(inv.subtotal + inv.tax - inv.total) > self.tol:
            add("TOTAL_MISMATCH", Severity.ERROR,
                f"Subtotal {inv.subtotal} + tax {inv.tax} = {round(inv.subtotal + inv.tax, 2)}, total says {inv.total}")

        if inv.due_date and inv.due_date < inv.invoice_date:
            add("DUE_BEFORE_INVOICE", Severity.ERROR, "Due date is before invoice date")
        if inv.invoice_date > today:
            add("FUTURE_DATE", Severity.WARNING, f"Invoice is dated in the future ({inv.invoice_date})")

        dup = self.store.find_duplicate(inv, exclude_id=invoice_id)
        if dup is not None:
            add("DUPLICATE_INVOICE", Severity.ERROR,
                f"Invoice {inv.invoice_number} from this vendor was already received (invoice #{dup})")
        else:
            similar = self.store.find_similar(inv, exclude_id=invoice_id)
            if similar is not None:
                add("POSSIBLE_DUPLICATE", Severity.WARNING,
                    f"Same vendor and amount as invoice #{similar} within 7 days")

        vendor = self.store.get_vendor(inv.vendor_name)
        if vendor is None:
            add("UNKNOWN_VENDOR", Severity.WARNING, f"'{inv.vendor_name}' is not in the vendor master")
        elif inv.bank_account and vendor.bank_account and \
                inv.bank_account.replace(" ", "").upper() != vendor.bank_account.replace(" ", "").upper():
            add("BANK_ACCOUNT_CHANGED", Severity.ERROR,
                "Bank account on invoice differs from the vendor master - possible payment fraud; "
                "verify with the vendor using known contact details")

        errors = sum(f.severity == Severity.ERROR for f in findings)
        warnings = sum(f.severity == Severity.WARNING for f in findings)
        confidence = max(0.0, 1.0 - 0.3 * errors - 0.1 * warnings)
        reasoning = "All checks passed." if not findings else \
            f"{errors} error(s), {warnings} warning(s) found."
        return AgentResult(agent=self.name, confidence=round(confidence, 2),
                           reasoning=reasoning, findings=findings)
