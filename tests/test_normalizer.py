from legal_workflow_generator.query.normalizer import QueryNormalizer

normalizer = QueryNormalizer()

# Test 1 — plain text
result = normalizer.normalize(text="How do I comply with DPDP Act??")
print(result)

# Test 2 — text with abbreviations
result = normalizer.normalize(text="What are GST rules for my startup?")
print(result)

# Test 3 — typo
result = normalizer.normalize(text="What is the complience process for SEBI?")
print(result)

# Test 4 — PDF only
result = normalizer.normalize(pdf_path="test_legal.pdf")
print(result)

# Test 5 — PDF + text together
result = normalizer.normalize(
    text="What are my compliance obligations?",
    pdf_path="test_legal.pdf"
)
print(result)
"""

from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="This company needs to comply with DPDP Act and GST regulations.", ln=True)
pdf.output("test_legal.pdf")"""