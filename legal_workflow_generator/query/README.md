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
│   ├── context_resolver.py   # LegalContextResolver class
│   └── keyword_domain_classifier.py  # rule-based (TF-IDF keyword) domain classifier
├── llm/
│   ├── __init__.py           # shared LangChain chat models + invoke_structured()
│   └── schemas.py            # IntentClassification, DomainClassification, ...
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

Create a `.env` file in the project root (or copy `.env.example`):

```
PGPASSWORD=your_postgres_password
GEMINI_API_KEY=your_gemini_api_key_here
```

Both are required — the config module raises `MissingEnvironmentVariable` at
import time if either is missing. The query unit's LLM calls use Gemini only;
`GROQ_API_KEY` is optional and only needed for the `--provider groq` RAG answer
generator.

> Get a Gemini API key at [aistudio.google.com](https://aistudio.google.com/apikey)

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

### Why Gemini with LangChain structured output for classification?
- Both LLM calls go through `ChatGoogleGenerativeAI` with `with_structured_output`,
  so the reply is validated against a Pydantic schema instead of being parsed
  from text:
  - `IntentClassifier` → `IntentClassification` (`reason`, `intent`, `confidence` in 0–1)
  - `LegalContextResolver` → `DomainClassification` (`domain`, up to 5 `keywords`)
- `intent` and `domain` are `Literal` enums, so the model cannot return a label
  outside the fixed set
- Understands legal context far better than keyword matching
- The single-shot call uses the model's default temperature; self-consistency
  votes sample at `temperature=0.7`
- A reply that fails validation, or an API error, degrades safely: the intent
  becomes `UNKNOWN` with confidence `0.0`, and the LLM domain is treated as
  missing

### Why a rule-based keyword classifier in the context resolver?
- `KeywordDomainClassifier` (TF-IDF terms extracted from the corpus) runs first
  on every query — deterministic and free
- With the default `llm_fallback` strategy, Gemini is only called when keywords
  find no match; with `combine`, both run and are cross-checked
  (`domain_agreement`)
- If the Gemini call fails, a keyword match still resolves the domain, so the
  pipeline doesn't crash

### Why pdfplumber over pypdf?
- Better at handling structured/table-heavy legal documents
- More accurate text extraction from formatted PDFs

### Why remove spell correction?
- Generic spell checkers mangle legal terms (`"esops"` → `"sops"`)
- The LLM classifiers handle minor typos intelligently anyway
- Simpler = more reliable

---

## Running Tests

The query-unit tests are part of the pytest suite at the repo root and run
fully offline — the structured LLM runnables are replaced by a fake that returns
parsed schema objects (`FakeStructuredLLM` in `tests/conftest.py`):

```bash
# Normalizer, intent classifier, context resolver, and process_query wiring
uv run pytest tests/test_normalizer.py tests/test_intent_classifier.py \
              tests/test_context_resolver.py tests/test_query.py
```

See the "Running the tests" section of the top-level `README.md` for coverage
and the integration suite.

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
# confidence < 0.5 is overridden to QueryIntent.UNKNOWN
# classifier.llm is the structured_llm(IntentClassification) runnable
```

### `LegalContextResolver`

```python
resolver = LegalContextResolver()
context = resolver.resolve(normalized_query, intent, confidence)
# Returns LegalContext object
# resolver.llm / resolver.sampling_llm are structured_llm(DomainClassification)
# runnables (default temperature / 0.7 for self-consistency votes)
```

---

## Known Limitations

- Single domain detection only — multi-domain queries (e.g. "ESOP tax implications for employees") are mapped to the most relevant domain
- Hindi-only queries pass through normalization unchanged and are typically classified as `UNKNOWN` or mapped to the closest intent by the LLM
- Very long PDFs (500+ words after normalization) are truncated

---

## Dependencies

| Package | Purpose |
|---|---|
| `langchain-google-genai` | Gemini chat model (structured output) for intent classification and context resolution |
| `pydantic` | Output schemas for the structured LLM calls (`llm/schemas.py`) |
| `pdfplumber` | PDF text extraction |
| `python-dotenv` | Loading `.env` file |

---