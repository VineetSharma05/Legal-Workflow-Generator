## Installation

- Install uv and other packages
```
pip install uv
uv sync
```

- Setup env file
```
cp .env.example .env
# Edit .env file with correct values
```

- Start database
```
docker compose up
```

- Setup database
```bash
python main.py setup
```
- Ingest the dataset
```bash
python main.py ingest
```

- Generate embeddings
```bash
python main.py embed
```

- Run end-to-end retrieval + Gemini answer generation test
```bash
python tests/test_rag_pipeline.py
```

## Test Query Processing Unit

Run the standalone query processor test:

```bash
python -m tests.test_query
```

## Test RAG Unit

Run the standalone RAG pipeline test:

```bash
python -m tests.test_rag_pipeline --gemini
# or
python -m tests.test_rag_pipeline --llama
```

## Demo: Run Query Unit + RAG Unit Together (Custom Query)

Use this script to run the full flow with your own query:

```bash
python demo_query_rag.py \
	--query "What are the steps to comply with DPDP Act as a SaaS startup?" \
	--provider gemini \
	--top-k 3
```

Useful options:

```bash
# Run only query unit (normalization + intent + legal context)
python demo_query_rag.py --query "Need GST compliance checklist" --query-only

# Use Groq for answer generation
python demo_query_rag.py --query "How to register a private limited company in India?" --provider groq
```
