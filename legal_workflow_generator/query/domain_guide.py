"""
What each of the five domains holds, for the routing prompts (settings.domain_guide, v5).

Written from the corpus itself (the statutes the `laws` table files under each domain), not from any test query.
The domain names alone mislead routing: the Contract Act, Consumer Protection Act and IBC sit under
corporate_governance, and FEMA and the Stamp Act under taxation. Keep this in step with the corpus.
"""

DOMAIN_GUIDE = {
    "corporate_governance": (
        "company and LLP law, contracts and consumer protection: Companies Act 2013, LLP Act 2008, "
        "Indian Contract Act 1872 (formation, validity, capacity, breach, damages, indemnity, guarantee, agency), "
        "Consumer Protection Act 2019 (consumer rights, unfair trade practices, product liability, consumer "
        "commissions, the Central Consumer Protection Authority), Insolvency and Bankruptcy Code 2016"),
    "data_protection": (
        "personal data, cyber security and information law: DPDP Act 2023 and DPDP Rules 2025, SPDI Rules 2011, "
        "IT Act 2000 (data and cyber provisions), IT Intermediary Guidelines 2021, CERT-In Directions 2022, "
        "Aadhaar Act 2016, Credit Information Companies (Regulation) Act 2005 (credit information and credit "
        "reports), RTI Act 2005, Telecommunications Act 2023 and Telecom Cyber Security Rules 2024, "
        "RBI directive on storage of payment system data"),
    "ip_licensing": (
        "intellectual property: Copyright Act 1957, Patents Act 1970, Trade Marks Act 1999, Designs Act 2000 "
        "(registration, cancellation, infringement), IT Act 2000 provisions on software and electronic works"),
    "taxation": (
        "tax, duties and foreign investment: CGST Act and Rules 2017, IGST Act 2017, UTGST Act 2017, "
        "Income-tax Act 1961 and Income-tax Act 2025, Indian Stamp Act 1899, SEZ Act 2005, FEMA regulations on "
        "transfer or issue of securities to persons resident outside India (foreign investment, RBI reporting), "
        "ESI Act 1948 contributions"),
    "employment": (
        "employment and workplace law: wages, bonus, gratuity, provident fund, ESI and social security codes, "
        "maternity benefit, POSH Act 2013, industrial disputes, standing orders, trade unions, contract and "
        "migrant labour, apprentices, factories and occupational safety code, employees' compensation, equal "
        "remuneration, child labour, persons with disabilities"),
}

# Laws the corpus does not hold. Listed so the router can say "none" instead of forcing a nearest domain.
NOT_COVERED = ("criminal law, civil and criminal procedure, arbitration, customs duty, environmental law, "
               "banking and securities regulation, competition law, property and land law, family law, and the "
               "law of any other country")


def guide_text() -> str:
    lines = [f"- {d}: {txt}" for d, txt in DOMAIN_GUIDE.items()]
    return "\n".join(lines) + f"\nThe knowledge base does NOT cover: {NOT_COVERED}."
