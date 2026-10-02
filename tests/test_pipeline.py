import json
import unittest
from datetime import date
from pathlib import Path

import httpx

from invoiceflow.agents import IntakeAgent, MatchingAgent, ValidationAgent
from invoiceflow.llm import GroqClient, HeuristicLLM, LLMError
from invoiceflow.models import ExtractedInvoice, InvoiceStatus
from invoiceflow.orchestrator import Orchestrator
from invoiceflow.store import Store, normalize_name

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples" / "invoices"
TODAY = date(2026, 10, 1)


def seeded_store() -> Store:
    store = Store(":memory:")
    store.seed(json.loads((ROOT / "samples" / "seed.json").read_text()))
    return store


def invoice(**overrides) -> ExtractedInvoice:
    data = {
        "vendor_name": "Acme Office Supplies Inc.", "invoice_number": "T-1",
        "invoice_date": "2026-09-15", "due_date": "2026-10-15", "po_number": "PO-1001",
        "currency": "USD",
        "line_items": [
            {"description": "Printer paper A4 (box)", "quantity": 10, "unit_price": 25, "amount": 250},
            {"description": "Toner cartridge", "quantity": 2, "unit_price": 80, "amount": 160},
        ],
        "subtotal": 410, "tax": 61.5, "total": 471.5,
        "bank_account": "PK36SCBL0000001123456702",
    }
    data.update(overrides)
    return ExtractedInvoice.model_validate(data)


class ScriptedLLM:
    """Returns pre-set answers in order and records the prompts it received."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def complete_json(self, task, system, user):
        self.prompts.append(user)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class IntakeAgentTests(unittest.TestCase):
    def test_self_corrects_after_invalid_answer(self):
        good = invoice().model_dump(mode="json")
        bad = dict(good, invoice_date="not a date")
        llm = ScriptedLLM([bad, good])
        result_invoice, result = IntakeAgent(llm).run("some invoice text")
        self.assertIsNotNone(result_invoice)
        self.assertEqual(result.data["attempts"], 2)
        self.assertLess(result.confidence, 1.0)
        self.assertIn("invoice_date", llm.prompts[1])  # error was fed back to the model

    def test_recovers_from_unparseable_output(self):
        llm = ScriptedLLM([LLMError("bad json"), invoice().model_dump(mode="json")])
        result_invoice, _ = IntakeAgent(llm).run("text")
        self.assertIsNotNone(result_invoice)

    def test_gives_up_after_max_attempts(self):
        llm = ScriptedLLM([{}, {}, {}])
        result_invoice, result = IntakeAgent(llm, max_attempts=3).run("text")
        self.assertIsNone(result_invoice)
        self.assertEqual(result.errors[0].code, "EXTRACTION_FAILED")

    def test_empty_document(self):
        result_invoice, result = IntakeAgent(ScriptedLLM([])).run("   ")
        self.assertIsNone(result_invoice)
        self.assertEqual(result.errors[0].code, "EMPTY_DOCUMENT")


class ValidationAgentTests(unittest.TestCase):
    def setUp(self):
        self.store = seeded_store()
        self.agent = ValidationAgent(self.store)

    def codes(self, inv, invoice_id=999):
        return {f.code for f in self.agent.run(inv, invoice_id, today=TODAY).findings}

    def test_clean_invoice(self):
        self.assertEqual(self.codes(invoice()), set())

    def test_arithmetic_errors(self):
        inv = invoice(subtotal=400, total=500)
        self.assertEqual(self.codes(inv), {"SUBTOTAL_MISMATCH", "TOTAL_MISMATCH"})

    def test_line_math(self):
        items = invoice().model_dump()["line_items"]
        items[0]["amount"] = 260
        self.assertIn("LINE_MATH", self.codes(invoice(line_items=items, subtotal=420, total=481.5)))

    def test_bank_account_change_flagged(self):
        self.assertIn("BANK_ACCOUNT_CHANGED", self.codes(invoice(bank_account="PK00OTHER")))

    def test_unknown_vendor_and_future_date(self):
        codes = self.codes(invoice(vendor_name="Nobody Ltd", invoice_date="2027-01-01", due_date=None))
        self.assertEqual(codes, {"UNKNOWN_VENDOR", "FUTURE_DATE"})

    def test_duplicate_detection(self):
        first = self.store.create_invoice("a", "x")
        self.store.save_extracted(first, invoice())
        second = self.store.create_invoice("b", "x")
        self.assertIn("DUPLICATE_INVOICE", self.codes(invoice(vendor_name="ACME Office Supplies, LLC"), second))
        self.assertIn("POSSIBLE_DUPLICATE", self.codes(invoice(invoice_number="T-2"), second))


class MatchingAgentTests(unittest.TestCase):
    def setUp(self):
        self.agent = MatchingAgent(seeded_store())

    def test_full_match(self):
        result = self.agent.run(invoice())
        self.assertTrue(result.data["matched"])
        self.assertEqual(result.findings, [])

    def test_price_variance_and_quantity(self):
        items = invoice().model_dump()["line_items"]
        items[0].update(quantity=12, unit_price=27, amount=324)
        codes = {f.code for f in self.agent.run(invoice(line_items=items)).findings}
        self.assertEqual(codes, {"QTY_EXCEEDS_PO", "PRICE_VARIANCE"})

    def test_missing_and_unknown_po(self):
        self.assertEqual(self.agent.run(invoice(po_number=None)).findings[0].code, "NO_PO")
        self.assertEqual(self.agent.run(invoice(po_number="PO-9999")).findings[0].code, "PO_NOT_FOUND")

    def test_wrong_vendor_and_unlisted_line(self):
        items = invoice().model_dump()["line_items"] + [
            {"description": "Espresso machine", "quantity": 1, "unit_price": 10, "amount": 10}]
        codes = {f.code for f in self.agent.run(
            invoice(vendor_name="Bright Print Co.", line_items=items)).findings}
        self.assertEqual(codes, {"PO_VENDOR_MISMATCH", "LINE_NOT_ON_PO"})


class OrchestratorTests(unittest.TestCase):
    def test_sample_invoices_end_to_end(self):
        store = seeded_store()
        orch = Orchestrator(store, HeuristicLLM())
        outcomes = {p.name: orch.process(p.read_text(), p.name, today=TODAY)
                    for p in sorted(SAMPLES.glob("*.txt"))}

        self.assertEqual(outcomes["01_clean_small.txt"].status, InvoiceStatus.APPROVED)
        o2 = outcomes["02_needs_manager.txt"]
        self.assertEqual(o2.status, InvoiceStatus.PENDING_APPROVAL)
        self.assertEqual(o2.assigned_to, "manager@company.example")

        o3 = outcomes["03_price_variance.txt"]
        self.assertEqual(o3.status, InvoiceStatus.NEEDS_REVIEW)
        self.assertIn("vendor_query", [r.agent for r in o3.results])

        o4 = outcomes["04_duplicate_new_bank.txt"]
        self.assertEqual(o4.status, InvoiceStatus.NEEDS_REVIEW)
        self.assertIn("DUPLICATE_INVOICE", o4.reason)
        self.assertIn("BANK_ACCOUNT_CHANGED", o4.reason)
        # fraud signal must not trigger an email to the (possibly spoofed) sender
        self.assertNotIn("vendor_query", [r.agent for r in o4.results])

        events = [e["event"] for e in store.get_events(o3.invoice_id)]
        self.assertEqual(events[0], "received")
        self.assertEqual(events[-1], "status:needs_review")

    def test_human_decision(self):
        store = seeded_store()
        orch = Orchestrator(store, HeuristicLLM())
        outcome = orch.process((SAMPLES / "02_needs_manager.txt").read_text(), "inv", today=TODAY)
        self.assertEqual(orch.decide(outcome.invoice_id, True, "alice"), InvoiceStatus.APPROVED)
        with self.assertRaises(ValueError):
            orch.decide(outcome.invoice_id, False, "bob")

    def test_failure_is_recorded_and_reraised(self):
        store = seeded_store()
        orch = Orchestrator(store, ScriptedLLM([RuntimeError("network down")]))
        with self.assertRaises(RuntimeError):
            orch.process("text", "inv")
        record = store.list_invoices()[0]
        self.assertEqual(record["status"], "failed")
        self.assertIn("network down", store.get_events(record["id"])[-1]["payload"]["error"])


class GroqClientTests(unittest.TestCase):
    def client(self, handler):
        return GroqClient("key-123", "test-model", http_client=httpx.Client(transport=httpx.MockTransport(handler)))

    def test_request_shape_and_parsing(self):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            seen["auth"] = request.headers["authorization"]
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"a": 1}'}}]})

        self.assertEqual(self.client(handler).complete_json("t", "sys", "usr"), {"a": 1})
        self.assertEqual(seen["url"], "https://api.groq.com/openai/v1/chat/completions")
        self.assertEqual(seen["auth"], "Bearer key-123")
        self.assertEqual(seen["body"]["response_format"], {"type": "json_object"})
        self.assertEqual(seen["body"]["messages"][0], {"role": "system", "content": "sys"})

    def test_invalid_json_raises_llm_error(self):
        handler = lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "nope"}}]})
        with self.assertRaises(LLMError):
            self.client(handler).complete_json("t", "s", "u")

    def test_http_error_propagates(self):
        handler = lambda r: httpx.Response(401, json={"error": "bad key"})
        with self.assertRaises(httpx.HTTPStatusError):
            self.client(handler).complete_json("t", "s", "u")


class HelperTests(unittest.TestCase):
    def test_normalize_name(self):
        self.assertEqual(normalize_name("Acme Office Supplies, Inc."), "acme office supplies")


if __name__ == "__main__":
    unittest.main()
