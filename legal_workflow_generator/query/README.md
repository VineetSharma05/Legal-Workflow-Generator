# Query Processing Unit

> Part of the **Legal Workflow Generator** — an AI-powered compliance system for Indian tech startups.

This module is the first stage of the pipeline. It takes raw user input (text or PDF) and converts it into a structured `LegalContext` object that the Retrieval Unit uses to search the legal database.

---

## What It Does

```
User Input (text / PDF / both)
        ↓
   QueryNormalizer        → cleans and standardizes input
        ↓
   IntentClassifier       → classifies what the user wants
        ↓
   LegalContextResolver   → detects legal domain + keywords
        ↓
   LegalContext object    → passed to Retrieval Unit
```

---

## Folder Structure

```
legal_workflow_generator/
├── query/
│   ├── __init__.py           # Single entry point — process_query()
│   ├── normalizer.py         # QueryNormalizer class
│   ├── intent_classifier.py  # IntentClassifier class
│   └── context_resolver.py   # LegalContextResolver class
└── typings/
    └── types.py              # NormalizedQuery, QueryIntent, LegalContext
```

---

## Setup

### 1. Clone the repo and switch to this branch

```bash
git clone https://github.com/VineetSharma05/Legal-Workflow-Generator.git
cd Legal-Workflow-Generator
git checkout query-processing-unit
```

### 2. Create and activate virtual environment

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows Command Prompt
.\.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install uv
uv sync
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

> Get a free Groq API key at [console.groq.com](https://console.groq.com)

> ⚠️ Never commit your `.env` file — it is already in `.gitignore`

---

## Usage

### Simple — use the single entry point

```python
from legal_workflow_generator.query import process_query

# Text query
context = process_query(text="What are the steps to comply with DPDP Act?")

# PDF only
context = process_query(pdf_path="company_details.pdf")

# PDF + text together
context = process_query(
    text="Am I compliant with GST?",
    pdf_path="company_details.pdf"
)

print(context)
```

### Output — LegalContext object

```json
{
  "original_query": "What are the steps to comply with DPDP Act?",
  "normalized_query": "what are the steps to comply with digital personal data protection act",
  "intent": "workflow",
  "legal_domain": "data_protection",
  "keywords": ["compliance", "digital", "personal", "data", "startup"],
  "confidence": 0.9
}
```

---

## Intents

The classifier outputs one of 4 intents:

| Intent | Meaning | Example |
|---|---|---|
| `QA` | User wants a legal question answered | "What is the DPDP Act?" |
| `WORKFLOW` | User wants step-by-step process | "How do I register my startup?" |
| `COMPLIANCE_CHECK` | User wants to check if they are compliant | "Am I compliant with GST?" |
| `UNKNOWN` | Query is unclear or not legal | "What is the best pizza in Bangalore?" |

---

## Legal Domains

The context resolver maps queries to one of 5 domains:

| Domain | Covers |
|---|---|
| `data_protection` | DPDP Act, privacy, data breach, consent |
| `corporate_governance` | Company registration, MCA, directors, shares |
| `ip_licensing` | Patents, trademarks, copyright, open source |
| `taxation` | GST, TDS, income tax, ITR filing |
| `employment` | Hiring, PF, ESI, ESOP, labour law |

---

## Supported Abbreviations

The normalizer automatically expands these:

| Abbreviation | Expands To |
|---|---|
| `DPDP` | Digital Personal Data Protection |
| `GST` | Goods and Services Tax |
| `TDS` | Tax Deducted at Source |
| `FEMA` | Foreign Exchange Management Act |
| `SEBI` | Securities and Exchange Board of India |
| `ESOP` | Employee Stock Option Plan |
| `NDA` | Non Disclosure Agreement |
| `MCA` | Ministry of Corporate Affairs |
| `PF` | Provident Fund |
| `ESI` | Employee State Insurance |
| `LLP` | Limited Liability Partnership |
| `ROC` | Registrar of Companies |
| `MSME` | Micro Small and Medium Enterprises |

---

## Error Handling

The normalizer raises clear errors for bad input:

```python
# Empty input
process_query()
# → ValueError: At least one of text or pdf_path must be provided

# Only spaces or special characters
process_query(text="   !@#$%  ")
# → ValueError: Query is empty after normalization

# Too short
process_query(text="GST")
# → ValueError: Query is too short to be meaningful

# PDF not found
process_query(pdf_path="missing.pdf")
# → FileNotFoundError: PDF not found: missing.pdf

# Scanned image PDF with no extractable text
process_query(pdf_path="scanned.pdf")
# → ValueError: Could not extract any text from the PDF
```

Queries longer than 500 words are automatically truncated with a warning logged.

---

## Design Decisions

### Why Groq (llama-3.3-70b) for classification?
- Free tier is generous enough for a capstone project
- Constrained to fixed output format — hallucination risk is near zero
- Understands legal context far better than keyword matching
- `temperature=0.1` keeps responses deterministic

### Why rule-based fallback in context resolver?
- If Groq API is unavailable, the system still works
- Domain detection falls back to keyword counting
- Never crashes the pipeline

### Why pdfplumber over pypdf?
- Better at handling structured/table-heavy legal documents
- More accurate text extraction from formatted PDFs

### Why remove spell correction?
- Generic spell checkers mangle legal terms (`"esops"` → `"sops"`)
- Groq handles minor typos intelligently anyway
- Simpler = more reliable

---

## Running Tests

```bash
# Basic tests
python test.py

# Intent classification tests
python test_intent.py

# Full pipeline tests
python test_query.py
```

---

## Module Details

### `QueryNormalizer`

```python
normalizer = QueryNormalizer()
result = normalizer.normalize(text="...", pdf_path="...")
# Returns NormalizedQuery: { original, normalized, source }
```

### `IntentClassifier`

```python
classifier = IntentClassifier()
intent, confidence = classifier.classify(normalized_query)
# Returns tuple: (QueryIntent, float)
```

### `LegalContextResolver`

```python
resolver = LegalContextResolver()
context = resolver.resolve(normalized_query, intent, confidence)
# Returns LegalContext object
```

---

## Known Limitations

- Single domain detection only — multi-domain queries (e.g. "ESOP tax implications for employees") are mapped to the most relevant domain
- Hindi-only queries pass through normalization unchanged and are typically classified as `UNKNOWN` or mapped to the closest intent by Groq
- Very long PDFs (500+ words after normalization) are truncated

---

## Dependencies

| Package | Purpose |
|---|---|
| `groq` | LLM API for intent classification and context resolution |
| `pdfplumber` | PDF text extraction |
| `python-dotenv` | Loading `.env` file |

---