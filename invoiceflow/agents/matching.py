"""Matching agent: two-way match of invoice lines against the purchase order."""

from __future__ import annotations

from difflib import SequenceMatcher

from ..models import AgentResult, ExtractedInvoice, Finding, POLine, Severity
from ..store import Store, normalize_name


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.casefold(), b.casefold()).ratio()


class MatchingAgent:
    name = "matching"

    def __init__(self, store: Store, price_tolerance: float = 0.02, min_similarity: float = 0.6):
        self.store = store
        self.price_tolerance = price_tolerance
        self.min_similarity = min_similarity

    def run(self, inv: ExtractedInvoice) -> AgentResult:
        if not inv.po_number:
            return AgentResult(
                agent=self.name, confidence=0.5, reasoning="Invoice has no PO number; cannot match.",
                findings=[Finding(code="NO_PO", severity=Severity.WARNING,
                                  message="No purchase order number on the invoice")],
                data={"matched": False},
            )

        po = self.store.get_po(inv.po_number)
        if po is None:
            return AgentResult(
                agent=self.name, confidence=0.2, reasoning=f"PO {inv.po_number} does not exist.",
                findings=[Finding(code="PO_NOT_FOUND", severity=Severity.ERROR,
                                  message=f"Purchase order {inv.po_number} not found")],
                data={"matched": False},
            )

        findings: list[Finding] = []

        def add(code: str, severity: Severity, message: str) -> None:
            findings.append(Finding(code=code, severity=severity, message=message))

        if normalize_name(po.vendor_name) != normalize_name(inv.vendor_name):
            add("PO_VENDOR_MISMATCH", Severity.ERROR,
                f"PO {po.po_number} was issued to {po.vendor_name}, not {inv.vendor_name}")
        if po.currency.upper() != inv.currency.upper():
            add("CURRENCY_MISMATCH", Severity.ERROR, f"PO is in {po.currency}, invoice in {inv.currency}")

        matches = []
        unused: list[POLine] = list(po.lines)
        for item in inv.line_items:
            best = max(unused, key=lambda pl: _similarity(pl.description, item.description), default=None)
            score = _similarity(best.description, item.description) if best else 0.0
            if best is None or score < self.min_similarity:
                add("LINE_NOT_ON_PO", Severity.ERROR, f"'{item.description}' is not on PO {po.po_number}")
                continue
            unused.remove(best)
            matches.append({"invoice_line": item.description, "po_line": best.description,
                            "similarity": round(score, 2)})
            if item.quantity > best.quantity:
                add("QTY_EXCEEDS_PO", Severity.ERROR,
                    f"'{item.description}': invoiced {item.quantity}, ordered {best.quantity}")
            if best.unit_price > 0:
                variance = (item.unit_price - best.unit_price) / best.unit_price
                if abs(variance) > self.price_tolerance:
                    add("PRICE_VARIANCE", Severity.ERROR,
                        f"'{item.description}': unit price {item.unit_price} vs PO {best.unit_price} "
                        f"({variance:+.1%}, tolerance {self.price_tolerance:.0%})")

        errors = sum(f.severity == Severity.ERROR for f in findings)
        confidence = max(0.0, 1.0 - 0.3 * errors)
        reasoning = (f"All {len(matches)} line(s) match PO {po.po_number}." if not findings
                     else f"{errors} mismatch(es) against PO {po.po_number}.")
        return AgentResult(agent=self.name, confidence=round(confidence, 2), reasoning=reasoning,
                           findings=findings,
                           data={"matched": not findings, "po_number": po.po_number,
                                 "po_total": po.total, "line_matches": matches})
