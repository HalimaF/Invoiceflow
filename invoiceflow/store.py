"""SQLite persistence: vendors, purchase orders, invoices and the audit trail."""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any

from .models import AgentResult, ExtractedInvoice, InvoiceStatus, PurchaseOrder, Vendor

_SUFFIXES = {"inc", "llc", "ltd", "limited", "corp", "corporation", "co", "pvt", "plc", "gmbh"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS vendors (
    name_key TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    bank_account TEXT,
    email TEXT
);
CREATE TABLE IF NOT EXISTS purchase_orders (
    po_number TEXT PRIMARY KEY,
    data_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS invoices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    status TEXT NOT NULL,
    vendor_key TEXT,
    invoice_number TEXT,
    invoice_date TEXT,
    total REAL,
    data_json TEXT,
    assigned_to TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_id INTEGER NOT NULL REFERENCES invoices(id),
    ts TEXT NOT NULL,
    actor TEXT NOT NULL,
    event TEXT NOT NULL,
    confidence REAL,
    payload_json TEXT NOT NULL
);
"""


def normalize_name(name: str) -> str:
    """'Acme Office Supplies, Inc.' -> 'acme office supplies'"""
    words = re.sub(r"[^\w\s]", " ", name.casefold()).split()
    return " ".join(w for w in words if w not in _SUFFIXES)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str = ":memory:"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    # ---- reference data -------------------------------------------------
    def upsert_vendor(self, vendor: Vendor) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO vendors VALUES (?, ?, ?, ?)",
                (normalize_name(vendor.name), vendor.name, vendor.bank_account, vendor.email),
            )

    def get_vendor(self, name: str) -> Vendor | None:
        row = self.conn.execute(
            "SELECT * FROM vendors WHERE name_key = ?", (normalize_name(name),)
        ).fetchone()
        if row is None:
            return None
        return Vendor(name=row["name"], bank_account=row["bank_account"], email=row["email"])

    def upsert_po(self, po: PurchaseOrder) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO purchase_orders VALUES (?, ?)",
                (po.po_number.upper(), po.model_dump_json()),
            )

    def get_po(self, po_number: str) -> PurchaseOrder | None:
        row = self.conn.execute(
            "SELECT data_json FROM purchase_orders WHERE po_number = ?",
            (po_number.strip().upper(),),
        ).fetchone()
        return PurchaseOrder.model_validate_json(row["data_json"]) if row else None

    def seed(self, data: dict) -> None:
        for v in data.get("vendors", []):
            self.upsert_vendor(Vendor.model_validate(v))
        for p in data.get("purchase_orders", []):
            self.upsert_po(PurchaseOrder.model_validate(p))

    # ---- invoices -------------------------------------------------------
    def create_invoice(self, source_name: str, raw_text: str) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO invoices (source_name, raw_text, status, created_at) VALUES (?, ?, ?, ?)",
                (source_name, raw_text, InvoiceStatus.RECEIVED.value, _now()),
            )
        return int(cur.lastrowid)

    def save_extracted(self, invoice_id: int, inv: ExtractedInvoice) -> None:
        with self.conn:
            self.conn.execute(
                """UPDATE invoices SET vendor_key = ?, invoice_number = ?, invoice_date = ?,
                   total = ?, data_json = ? WHERE id = ?""",
                (normalize_name(inv.vendor_name), inv.invoice_number.strip().upper(),
                 inv.invoice_date.isoformat(), inv.total, inv.model_dump_json(), invoice_id),
            )

    def set_status(self, invoice_id: int, status: InvoiceStatus, assigned_to: str | None = None) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE invoices SET status = ?, assigned_to = ? WHERE id = ?",
                (status.value, assigned_to, invoice_id),
            )

    def get_invoice(self, invoice_id: int) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM invoices WHERE id = ?", (invoice_id,)).fetchone()
        if row is None:
            return None
        record = dict(row)
        record["data"] = json.loads(record.pop("data_json")) if record["data_json"] else None
        return record

    def list_invoices(self, status: InvoiceStatus | None = None) -> list[dict[str, Any]]:
        sql = "SELECT id, source_name, status, vendor_key, invoice_number, total, assigned_to, created_at FROM invoices"
        params: tuple = ()
        if status:
            sql += " WHERE status = ?"
            params = (status.value,)
        return [dict(r) for r in self.conn.execute(sql + " ORDER BY id", params)]

    def find_duplicate(self, inv: ExtractedInvoice, exclude_id: int) -> int | None:
        """An earlier invoice with the same vendor and invoice number."""
        row = self.conn.execute(
            "SELECT id FROM invoices WHERE vendor_key = ? AND invoice_number = ? AND id != ? ORDER BY id LIMIT 1",
            (normalize_name(inv.vendor_name), inv.invoice_number.strip().upper(), exclude_id),
        ).fetchone()
        return int(row["id"]) if row else None

    def find_similar(self, inv: ExtractedInvoice, exclude_id: int, days: int = 7) -> int | None:
        """Same vendor and total within `days`, but a different invoice number."""
        row = self.conn.execute(
            """SELECT id FROM invoices WHERE vendor_key = ? AND ABS(total - ?) < 0.01
               AND invoice_number != ? AND id != ?
               AND ABS(julianday(invoice_date) - julianday(?)) <= ? ORDER BY id LIMIT 1""",
            (normalize_name(inv.vendor_name), inv.total, inv.invoice_number.strip().upper(),
             exclude_id, inv.invoice_date.isoformat(), days),
        ).fetchone()
        return int(row["id"]) if row else None

    # ---- audit trail ----------------------------------------------------
    def add_event(self, invoice_id: int, actor: str, event: str,
                  payload: dict[str, Any] | None = None, confidence: float | None = None) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO audit_events (invoice_id, ts, actor, event, confidence, payload_json) VALUES (?, ?, ?, ?, ?, ?)",
                (invoice_id, _now(), actor, event, confidence, json.dumps(payload or {}, default=str)),
            )

    def add_agent_result(self, invoice_id: int, result: AgentResult) -> None:
        self.add_event(invoice_id, result.agent, "agent_result",
                       result.model_dump(mode="json"), result.confidence)

    def get_events(self, invoice_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM audit_events WHERE invoice_id = ? ORDER BY id", (invoice_id,)
        ).fetchall()
        events = []
        for r in rows:
            e = dict(r)
            e["payload"] = json.loads(e.pop("payload_json"))
            events.append(e)
        return events
