import json
import argparse

from legal_workflow_generator.rag.pipeline import RagPipeline


if __name__ == "__main__":
    # 1. Set up the argument parser
    parser = argparse.ArgumentParser(description="Run the Legal RAG Pipeline.")
    
    # 2. Create a mutually exclusive group (forces the user to pick one)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-g", "--gemini", action="store_true", help="Use Google Gemini")
    group.add_argument("-l", "--llama", action="store_true", help="Use Groq Llama 3")
    
    # 3. Parse the arguments
    args = parser.parse_args()

    # 4. Determine the provider based on the flag
    provider = "groq" if args.llama else "gemini"
    
    print(f"Initializing pipeline with {provider.upper()}...")

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

    print("\n=== " + provider.upper() + " ANSWER ===")
    print(result["answer"])

    # Optional JSON dump for debugging / API integration
    print("\n=== RAW RESULT JSON ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))
