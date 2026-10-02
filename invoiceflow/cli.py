"""Command-line interface.

    python -m invoiceflow seed samples/seed.json
    python -m invoiceflow process samples/invoices/*.txt
    python -m invoiceflow list
    python -m invoiceflow show 3
    python -m invoiceflow decide 2 --approve --by alice@company.example
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import build_llm, db_path
from .documents import read_document
from .orchestrator import Orchestrator
from .store import Store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="invoiceflow", description="Agentic invoice processing")
    parser.add_argument("--db", default=None, help="SQLite path (default: $INVOICEFLOW_DB or invoiceflow.db)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("seed", help="load vendors and purchase orders from JSON")
    p.add_argument("file")
    p = sub.add_parser("process", help="run invoices through the agents")
    p.add_argument("files", nargs="+")
    sub.add_parser("list", help="list invoices")
    p = sub.add_parser("show", help="show an invoice and its audit trail")
    p.add_argument("invoice_id", type=int)
    p = sub.add_parser("decide", help="approve or reject an invoice awaiting a human")
    p.add_argument("invoice_id", type=int)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--approve", action="store_true")
    g.add_argument("--reject", action="store_true")
    p.add_argument("--by", required=True, help="who is deciding")
    p.add_argument("--note", default="")

    args = parser.parse_args(argv)
    store = Store(args.db or db_path())

    if args.command == "seed":
        with open(args.file, encoding="utf-8") as fh:
            store.seed(json.load(fh))
        print(f"Seeded from {args.file}")
    elif args.command == "process":
        orch = Orchestrator(store, build_llm())
        for path in args.files:
            outcome = orch.process(read_document(path), source_name=path)
            who = f" -> {outcome.assigned_to}" if outcome.assigned_to else ""
            print(f"#{outcome.invoice_id} {path}: {outcome.status.value}{who}\n    {outcome.reason}")
    elif args.command == "list":
        for row in store.list_invoices():
            print(f"#{row['id']:<4} {row['status']:<17} {row['invoice_number'] or '-':<12} "
                  f"{row['total'] if row['total'] is not None else '-':>10}  {row['source_name']}")
    elif args.command == "show":
        record = store.get_invoice(args.invoice_id)
        if record is None:
            print(f"Invoice {args.invoice_id} not found", file=sys.stderr)
            return 1
        record.pop("raw_text")
        print(json.dumps(record, indent=2, default=str))
        print("\nAudit trail:")
        for e in store.get_events(args.invoice_id):
            conf = f" (confidence {e['confidence']:.2f})" if e["confidence"] is not None else ""
            print(f"  {e['ts']}  {e['actor']:<13} {e['event']}{conf}")
            payload = e["payload"]
            if payload.get("reasoning"):
                print(f"      {payload['reasoning']}")
            for f in payload.get("findings", []):
                print(f"      [{f['severity']}] {f['code']}: {f['message']}")
            if payload.get("data", {}).get("body"):
                print("      Draft email:\n        " + payload["data"]["body"].replace("\n", "\n        "))
    elif args.command == "decide":
        status = Orchestrator(store, build_llm()).decide(
            args.invoice_id, approve=args.approve, actor=args.by, note=args.note)
        print(f"#{args.invoice_id} -> {status.value}")
    return 0
