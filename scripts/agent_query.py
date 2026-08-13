import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("PGPASSWORD", "sanj2005")

from legal_workflow_generator.agent.graph import graph

def run(query: str):
    print(f"\n🔍 Query: {query}\n")
    result = graph.invoke({"query": query})

    print(result["answer"])

    print("\n📋 FULL TRACE:")
    for step in result.get("trace", []):
        print(f"   {step}")

    print("\nGRADES:")
    print(f"  context_grade: {result.get('context_grade')}")
    print(f"  groundedness_grade: {result.get('groundedness_grade')}")
    print(f"  answerability_grade: {result.get('answerability_grade')}")
    print(f"  retry_r: {result.get('retry_count_retrieval')}")
    print(f"  retry_g: {result.get('retry_count_generation')}")

    if result["abstain"]:
        print(f"\n⛔ ABSTAINED: {result['abstain_reason']}")
        return

    print(f"\n✅ Citations verified: {result['verified_citations']}")
    if result["failed_citations"]:
        print(f"❌ Citations failed: {result['failed_citations']}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/agent_query.py \"your legal question here\"")
        sys.exit(1)
    run(" ".join(sys.argv[1:]))