#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage:
  ./run_units.sh query
  ./run_units.sh rag [--gemini|--llama]
  ./run_units.sh demo --query "<your query>" [--provider gemini|groq] [--top-k N] [--query-only] [--use-original-query]

Modes:
  query   Run only the query unit test script.
  rag     Run only the RAG unit test script.
  demo    Run combined query + RAG demo script with a custom query.

Examples:
  ./run_units.sh query
  ./run_units.sh rag --gemini
  ./run_units.sh rag --llama
  ./run_units.sh demo --query "What are DPDP compliance steps for a SaaS startup?" --provider gemini --top-k 3
EOF
}

if command -v uv >/dev/null 2>&1; then
  PYTHON_CMD=(uv run python)
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD=(python)
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
    if [[ $# -ne 0 ]]; then
      echo "Error: query mode does not take extra arguments."
      echo
      usage
      exit 1
    fi
  "${PYTHON_CMD[@]}" -m tests.test_query
    ;;

  rag)
    if [[ $# -eq 0 ]]; then
      set -- --gemini
    fi
  "${PYTHON_CMD[@]}" -m tests.test_rag_pipeline "$@"
    ;;

  demo)
  "${PYTHON_CMD[@]}" demo_query_rag.py "$@"
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
