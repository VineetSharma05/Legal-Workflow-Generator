import json

from legal_workflow_generator.rag.pipeline import RagPipeline


if __name__ == "__main__":
    # Query grounded in combined_dataset.json (Companies Act, formation requirements)
    query = (
        "What are the minimum founder requirements to start a private company in India?"
    )

    pipeline = RagPipeline()
    result = pipeline.run(query=query, top_k=3)

    print("\n=== QUERY ===")
    print(result["query"])

    print("\n=== RETRIEVED PROVISIONS ===")
    for i, item in enumerate(result["retrieved"], start=1):
        print(
            f"{i}. {item['provision_id']} | {item['title']} "
            f"(score={item['combined_score']:.3f})"
        )

    print("\n=== GEMINI ANSWER ===")
    print(result["answer"])

    # Optional JSON dump for debugging / API integration
    print("\n=== RAW RESULT JSON ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))
