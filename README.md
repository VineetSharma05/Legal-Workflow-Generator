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
