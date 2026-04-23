import argparse
import json
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run query processing and RAG together with a custom query."
    )
    parser.add_argument(
        "-q",
        "--query",
        required=True,
        help="Custom legal query to run through the pipeline.",
    )
    parser.add_argument(
        "-p",
        "--provider",
        choices=["gemini", "groq"],
        default="gemini",
        help="LLM provider for final grounded answer.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of provisions to retrieve.",
    )
    parser.add_argument(
        "--query-only",
        action="store_true",
        help="Run only query processing unit and skip RAG.",
    )
    parser.add_argument(
        "--use-original-query",
        action="store_true",
        help="Use original query for retrieval instead of normalized query.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from legal_workflow_generator.query import process_query
    except Exception as e:
        print(f"Failed to initialize query unit: {e}")
        print("Make sure .env is configured (at least GROQ_API_KEY).")
        return 1

    try:
        context = process_query(text=args.query)
    except Exception as e:
        print(f"Query processing failed: {e}")
        return 1

    print("\n=== QUERY UNIT OUTPUT ===")
    print(
        json.dumps(
            {
                "original_query": context["original_query"],
                "normalized_query": context["normalized_query"],
                "intent": context["intent"].value,
                "legal_domain": context["legal_domain"],
                "keywords": context["keywords"],
                "confidence": context["confidence"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    if args.query_only:
        return 0

    rag_query = (
        context["original_query"] if args.use_original_query else context["normalized_query"]
    )

    try:
        from legal_workflow_generator.rag.pipeline import RagPipeline
        pipeline = RagPipeline(llm_provider=args.provider)
        result = pipeline.run(
            query=rag_query,
            top_k=args.top_k,
            legal_context=context,
        )
    except Exception as e:
        print(f"RAG pipeline failed: {e}")
        print(
            "Check setup: docker db running, PGPASSWORD set, and dataset ingested+embedded. "
            "For Gemini provider, GEMINI_API_KEY is required."
        )
        return 1

    print("\n=== RAG UNIT OUTPUT ===")
    print("User query used for answer:", result["query"])
    print("Query used for retrieval:", result.get("retrieval_query", result["query"]))

    print("\nRetrieved provisions:")
    for i, item in enumerate(result["retrieved"], start=1):
        score = item.get("combined_score")
        score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
        print(
            f"{i}. {item.get('provision_id')} | {item.get('title')} "
            f"(score={score_text})"
        )

    print("\nGenerated answer:\n")
    print(result["answer"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
