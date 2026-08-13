from legal_workflow_generator.agent.graph import graph

result = graph.invoke(
    {
        "query": "What are the steps to comply with DPDP Act as a SaaS startup?",
    }
)

print("\n=== FINAL ANSWER ===")
print(result["answer"])
print("\n=== CITATIONS ===")
print(result["verified_citations"])
print("\n=== ABSTAINED ===")
print(result["abstain"])
print("\n=== TRACE ===")
for step in result["trace"]:
    print(" →", step)

