# InvoiceFlow

**An agentic AI system that automates accounts-payable invoice processing.**

An invoice arrives → AI agents extract it, validate it, match it to the purchase
order, catch duplicates and payment-fraud signals, and either approve it, route it
to the right approver, or send it to a human with a drafted email to the vendor.
Every decision is recorded with the agent's confidence and reasoning.

## How it works

```
             ┌──────────────────────────── Orchestrator (state machine) ───────────────────────────┐
 invoice ──► │ Intake agent ─► Validation agent ─► Matching agent ─► ┬─► Approval agent ─► approved │
 (text/PDF)  │  (LLM, self-     (math, dates,       (2-way PO match,   │                  └► pending   │
             │   correcting)     duplicates, bank)   price/qty)        └─► Vendor-query agent         │
             │                                                              (LLM) ─► needs_review     │
             └────────────────────── every step written to the audit trail ─────────────────────────┘
```

| Agent | What it does |
|---|---|
| **Intake** | Uses an LLM to turn invoice text into a strict schema. If the answer fails validation, the exact errors are fed back and the model corrects itself (up to 3 attempts). |
| **Validation** | Line and total arithmetic, due-date sanity, duplicate and near-duplicate invoices, unknown vendors, and **bank-account changes** (a common invoice-fraud pattern). |
| **Matching** | Matches invoice lines to PO lines (fuzzy description match), flags price variance beyond tolerance, over-billed quantities, wrong vendor or currency. |
| **Approval** | Policy routing: auto-approve small PO-matched invoices, otherwise manager or finance director. |
| **Vendor query** | Drafts an email to the vendor explaining what to fix. Never used for fraud signals; those must be verified through known contacts. |

**Design choice:** the lifecycle is a deterministic state machine, so behaviour is
predictable and auditable. LLMs are used only where judgment is needed
(extraction, writing to vendors). Low-confidence or failed checks go to a human
instead of being guessed.

## Quick start

Requires Python 3.10+.

```bash
pip install -r requirements.txt

python -m invoiceflow seed samples/seed.json          # vendors + purchase orders
python -m invoiceflow process samples/invoices/*.txt
python -m invoiceflow list
python -m invoiceflow show 3                          # full audit trail + drafted email
python -m invoiceflow decide 2 --approve --by alice@company.example
```

Example output:

```
#1 samples/invoices/01_clean_small.txt: approved
    PO-matched and total 471.5 <= auto-approve limit 1000.0.
#2 samples/invoices/02_needs_manager.txt: pending_approval -> manager@company.example
    Total 6555.0 <= 10000.0; manager approval required.
#3 samples/invoices/03_price_variance.txt: needs_review
    Needs review: PRICE_VARIANCE
#4 samples/invoices/04_duplicate_new_bank.txt: needs_review
    Needs review: DUPLICATE_INVOICE, BANK_ACCOUNT_CHANGED
```

### Using a real LLM (Groq, free tier)

Without an API key the project runs with an offline heuristic extractor that
understands the sample format, so the demo and tests need no network. For real
invoices, use [Groq](https://console.groq.com):

```bash
# variables are read from the environment (see .env.example for the full list)
export GROQ_API_KEY=gsk_...
export GROQ_MODEL=llama-3.3-70b-versatile   # any current Groq chat model that supports JSON mode
```

`GroqClient` calls Groq's OpenAI-compatible `/chat/completions` endpoint with JSON mode.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover each agent, the intake self-correction loop, the end-to-end pipeline
on the sample invoices, human decisions, failure recording, and the Groq request
format (via a mocked HTTP transport).

## Project layout

```
invoiceflow/
  agents/        intake, validation, matching, approval, vendor_query
  orchestrator.py  lifecycle state machine + human decisions
  llm.py         GroqClient and offline HeuristicLLM
  store.py       SQLite: vendors, purchase orders, invoices, audit events
  cli.py         command-line interface
samples/         seed data and example invoices
tests/
```

## Roadmap

- [ ] REST API (FastAPI) and a web dashboard with an approval inbox
- [ ] OCR for scanned invoices and images (Tesseract or a Hugging Face document model)
- [ ] Three-way matching with goods receipts and partial deliveries
- [ ] Email ingestion and sending the drafted vendor queries
- [ ] Payment scheduling to capture early-payment discounts
