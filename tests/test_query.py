from legal_workflow_generator.query import process_query
import json

q = "What are the steps to comply with DPDP Act as a SaaS startup?"
context = process_query(text=q)

# Print full context object nicely
print(json.dumps({
    "original_query": context['original_query'],
    "normalized_query": context['normalized_query'],
    "intent": context['intent'].value,
    "legal_domain": context['legal_domain'],
    "keywords": context['keywords'],
    "confidence": context['confidence'],
}, indent=2))

"""queries = [
    # Extreme 1 — Completely empty string
    "",
    
    # Extreme 2 — Only spaces
    "     ",
    
    # Extreme 3 — Only special characters
    "!@#$%^&*()",
    
    # Extreme 4 — SQL injection attempt
    "SELECT * FROM laws WHERE 1=1; DROP TABLE laws;",
    
    # Extreme 5 — Very long query (500+ words)
    "I am a founder of a tech startup in India " * 50,
    
    # Extreme 6 — All caps
    "WHAT IS THE DPDP ACT AND HOW DO I COMPLY WITH IT",
    
    # Extreme 7 — Mixed languages
    "What is DPDP Act? मुझे इसके बारे में जानना है",
    
    # Extreme 8 — Numbers only
    "123456789",
    
    # Extreme 9 — Repeated words
    "compliance compliance compliance compliance compliance",
    
    # Extreme 10 — Code injection attempt
    "<script>alert('hack')</script> what is GST",
    
    # Extreme 11 — Contradictory query
    "I want to avoid all taxes legally and illegally",
    
    # Extreme 12 — Very specific with fake law
    "What is section 999 of the fake XYZ Act 2099?",
]

for q in queries:
    try:
        context = process_query(text=q)
        print(f"Query     : {q[:60]}...")
        print(f"Intent    : {context['intent']}")
        print(f"Domain    : {context['legal_domain']}")
        print(f"Keywords  : {context['keywords']}")
        print(f"Confidence: {context['confidence']}")
    except Exception as e:
        print(f"Query     : {q[:60]}...")
        print(f"ERROR     : {type(e).__name__}: {e}")
    print()"""