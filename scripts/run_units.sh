#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_units.sh query
  ./scripts/run_units.sh rag
  ./scripts/run_units.sh demo --query "<your query>" [--provider gemini|groq] [--top-k N] [--query-only] [--use-original-query]

Modes:
  query   Run the query-unit pytest module (offline, mocked).
  rag     Run the RAG-pipeline integration pytest module (needs DB + GEMINI_API_KEY).
  demo    Run combined query + RAG demo script with a custom query.

Examples:
  ./scripts/run_units.sh query
  ./scripts/run_units.sh rag
  ./scripts/run_units.sh demo --query "What are DPDP compliance steps for a SaaS startup?" --provider gemini --top-k 3
EOF
}

if command -v uv >/dev/null 2>&1; then
  PYTHON_CMD=(uv run python)
  PYTEST_CMD=(uv run pytest)
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD=(python)
  PYTEST_CMD=(python -m pytest)
else
  echo "Error: neither 'uv' nor 'python' was found in PATH."
  exit 1
fi

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

mode="$1"
shift

case "$mode" in
  query)
  "${PYTEST_CMD[@]}" tests/test_query.py "$@"
    ;;

  rag)
  "${PYTEST_CMD[@]}" -m integration tests/test_rag_pipeline.py "$@"
    ;;

  demo)
  "${PYTHON_CMD[@]}" scripts/demo_query_rag.py "$@"
    ;;

  -h|--help|help)
    usage
    ;;

  *)
    echo "Error: unknown mode '$mode'"
    echo
    usage
    exit 1
    ;;
esac
