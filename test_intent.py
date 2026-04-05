from legal_workflow_generator.query.normalizer import QueryNormalizer
from legal_workflow_generator.query.intent_classifier import IntentClassifier

normalizer = QueryNormalizer()
classifier = IntentClassifier()

queries = [
    # Edge case 1 — very vague query
    "tell me about taxes",
    
    # Edge case 2 — mixed intent (both QA and workflow)
    "What is FEMA and how do I comply with it?",
    
    # Edge case 3 — very long query
    "I am a startup founder in India and I recently started a SaaS company and I want to know all the steps I need to follow to make sure I am fully compliant with all Indian laws including data protection employment and taxation",
    
    # Edge case 4 — Hindi/informal language
    "mujhe startup ke liye kya karna hoga",
    
    # Edge case 5 — just abbreviations
    "DPDP GST SEBI",
    
    # Edge case 6 — negative/already done
    "I have already registered my company is there anything else I need to do for compliance?",
    
    # Edge case 7 — penalty related
    "what happens if I dont comply with DPDP Act",
    
    # Edge case 8 — completely random
    "what is the best pizza place in bangalore",
]

for q in queries:
    normalized = normalizer.normalize(text=q)
    intent, confidence = classifier.classify(normalized)
    print(f"Query    : {q}")
    print(f"Normalized: {normalized['normalized']}")
    print(f"Intent   : {intent} | Confidence: {confidence}")
    print()