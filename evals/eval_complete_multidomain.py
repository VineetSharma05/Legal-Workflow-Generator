"""
End-to-end agentic eval against the COMPLETE dataset (datasets/complete/,
~4.9k provisions across 52 statutes), with a heavy multi-domain component.

Why a new eval
--------------
eval_agent.py was written for the condensed corpus: 49 queries, one domain
each, and expected citations drawn from only ~7 statutes. The complete corpus
adds Income-tax, CGST, SEZ, Stamp, FEMA, LLP, Contract, Consumer Protection,
IBC, Patents, Trademarks, Designs, CERT-In, SPDI, Aadhaar, the labour codes and
~20 more labour statutes — and real startup questions routinely span several of
them at once. This eval stresses exactly that.

Test set (100 queries)
----------------------
  - 50 single-domain   : 10 per domain, targeting statutes that only exist in
                         the complete corpus.
  - 40 multi-domain    : 20 two-domain, 12 three-domain, 5 four-domain and
                         3 five-domain queries, each labeled with every domain
                         it touches and expected citations from those domains.
  - 10 edge cases      : non-existent sections, out-of-jurisdiction, speculative
                         and off-topic queries — the agent should abstain.

Metrics
-------
  domain_recall / domain_precision / domain_exact_match
      all_domains detected by classify_query vs. the labeled domain set.
  primary_domain_accuracy
      the resolver's single domain is one of the labeled domains.
  citation_recall@k
      verified citations ∩ expected provisions / expected provisions.
  citation_domain_coverage
      share of labeled domains for which at least one verified citation came
      from a statute in that domain (did the stitched answer cover every
      domain the user asked about?).
  abstain_accuracy / false_abstain_rate
      correct abstention on edge cases / wrongful abstention on answerable ones.
Every metric is also broken down per category (single, multi_2 … multi_5,
edge) and per domain.

Usage
-----
  python evals/eval_complete_multidomain.py                 # full run (100 queries)
  python evals/eval_complete_multidomain.py --category multi
  python evals/eval_complete_multidomain.py --limit 10 --sleep 2
  python evals/eval_complete_multidomain.py --provider groq
  python evals/eval_complete_multidomain.py --validate      # offline: check every
                                                            # expected id exists in
                                                            # datasets/complete/
"""

import argparse
import csv
import os
import glob
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# EVAL_RESULTS_DIR lets a run (e.g. the Groq comparison) keep its files apart from earlier runs.
RESULTS_DIR = Path(os.getenv("EVAL_RESULTS_DIR") or Path(__file__).resolve().parent / "results")
if not RESULTS_DIR.is_absolute():
    RESULTS_DIR = ROOT / RESULTS_DIR
COMPLETE_DIR = ROOT / "datasets" / "complete"

DP, CG, IP, TAX, EMP = (
    "data_protection", "corporate_governance", "ip_licensing", "taxation", "employment",
)
ALL_DOMAINS = [DP, CG, IP, TAX, EMP]

# provision_id prefix → domain, mirroring which datasets/complete/complete_*.json
# file each statute lives in. Used to attribute verified citations to a domain.
STATUTE_PREFIX_DOMAIN = {
    # complete_dpdp.json
    "dpdp_act_2023_": DP, "dpdp_rules_2025_": DP, "information_technology_act_2000_": DP,
    "aadhaar_act_2016_": DP, "cert_in_directions_2022_": DP, "spdi_rules_2011_": DP,
    "it_intermediary_guidelines_2021_": DP, "cicra_2005_": DP,
    "telecom_cyber_security_rules_2024_": DP, "telecommunications_act_2023_": DP,
    "rti_act_2005_": DP, "rbi_payment_data_storage_2018_": DP,
    # complete_corporategovernance.json
    "ca_2013_": CG, "contract_act_1872_": CG, "consumer_protection_act_2019_": CG,
    "llp_act_2008_": CG, "ibc_2016_": CG,
    # complete_softwarelicensing.json
    "patents_act_1970_": IP, "trademarks_act_1999_": IP, "designs_act_2000_": IP,
    "copyright_act_1957_": IP, "it_act_2000_": IP,
    # complete_tax.json
    "income_tax_act_1961_": TAX, "income_tax_act_2025_": TAX, "cgst_rules_2017_": TAX,
    "cgst_act_2017_": TAX, "cgst_forms_2017_": TAX, "esi_act_1948_": TAX,
    "indian_stamp_act_1899_": TAX, "sez_act_2005_": TAX, "fema_": TAX,
    "utgst_act_2017_": TAX, "igst_act_2017_": TAX,
    # complete_employee_laws.json
    "css_2020_": EMP, "oshwc_2020_": EMP, "esia_1948_": EMP, "fa_1948_": EMP,
    "rpwd_2016_": EMP, "ida_1947_": EMP, "epf_1952_": EMP, "bocw_1996_": EMP,
    "ieso_rules_1946_": EMP, "ieso_1946_": EMP, "eca_1923_": EMP, "poba_1965_": EMP,
    "bcw_1966_": EMP, "app_1961_": EMP, "ismw_1979_": EMP, "cla_1970_": EMP,
    "mba_1961_": EMP, "tua_1926_": EMP, "wj_1955_": EMP, "pow_1936_": EMP,
    "posh_2013_": EMP, "cal_1986_": EMP, "poga_1972_": EMP, "era_1976_": EMP,
}

FEMA = "fema_transfer_issue_foreign_security_regulations_2004_"


def q(query, domains, sections, should_abstain=False):
    return {"query": query, "domains": domains, "expected_sections": sections,
            "should_abstain": should_abstain}


# ── 100 test queries ─────────────────────────────────────────────────────────
TEST_QUERIES = [
    # ── single-domain: data_protection ──────────────────────────────────────
    q("what notice must a data fiduciary give before collecting personal data",
      [DP], ["dpdp_act_2023_sec_5", "dpdp_rules_2025_sec_3"]),
    q("how quickly must we report a personal data breach to the Data Protection Board and affected users",
      [DP], ["dpdp_act_2023_sec_8", "dpdp_rules_2025_sec_7"]),
    q("verifiable parental consent requirements for processing children's personal data",
      [DP], ["dpdp_act_2023_sec_9", "dpdp_rules_2025_sec_10"]),
    q("registration conditions and obligations of a consent manager",
      [DP], ["dpdp_rules_2025_sec_4", "dpdp_rules_2025_sch_1"]),
    q("what reasonable security safeguards are mandatory under the DPDP rules",
      [DP], ["dpdp_rules_2025_sec_6", "dpdp_act_2023_sec_8"]),
    q("can we transfer personal data of Indian users to servers outside India",
      [DP], ["dpdp_rules_2025_sec_15", "spdi_rules_2011_sec_7"]),
    q("mandatory reporting of cyber security incidents to CERT-In within 6 hours",
      [DP], ["cert_in_directions_2022_dir_2", "cert_in_directions_2022_annex_1"]),
    q("how long must we retain ICT system logs under the CERT-In directions",
      [DP], ["cert_in_directions_2022_dir_4", "cert_in_directions_2022_preamble"]),
    q("what counts as sensitive personal data or information under the SPDI rules",
      [DP], ["spdi_rules_2011_sec_3", "spdi_rules_2011_sec_2"]),
    q("compensation liability of a body corporate that fails to protect sensitive personal data",
      [DP], ["information_technology_act_2000_sec_43A", "spdi_rules_2011_sec_8"]),

    # ── single-domain: corporate_governance ─────────────────────────────────
    q("minimum number of partners and designated partners required to form an LLP",
      [CG], ["llp_act_2008_sec_6", "llp_act_2008_sec_7"]),
    q("how to convert a private limited company into an LLP",
      [CG], ["llp_act_2008_sec_56", "llp_act_2008_schedule_3"]),
    q("is a non-compete clause in a commercial contract enforceable in India",
      [CG], ["contract_act_1872_sec_27", "contract_act_1872_sec_23"]),
    q("damages recoverable for breach of contract when a liquidated damages clause exists",
      [CG], ["contract_act_1872_sec_74", "contract_act_1872_sec_73"]),
    q("what is a contract of indemnity and what are the rights of the indemnity holder",
      [CG], ["contract_act_1872_sec_124", "contract_act_1872_sec_125"]),
    q("when must a company spend on corporate social responsibility",
      [CG], ["ca_2013_sec_135"]),
    q("process for appointing the auditor of a company at the annual general meeting",
      [CG], ["ca_2013_sec_139", "ca_2013_sec_96"]),
    q("issuing new shares to existing shareholders through a rights issue",
      [CG], ["ca_2013_sec_62", "ca_2013_sec_39"]),
    q("who is not eligible to be a resolution applicant under the insolvency code",
      [CG], ["ibc_2016_sec_29A", "ibc_2016_sec_30"]),
    q("unfair trade practices by an e-commerce marketplace towards consumers",
      [CG], ["consumer_protection_act_2019_sec_93", "consumer_protection_act_2019_sec_2"]),

    # ── single-domain: ip_licensing ─────────────────────────────────────────
    q("what is not an invention under Indian patent law, are software algorithms excluded",
      [IP], ["patents_act_1970_sec_3"]),
    q("how long does a patent last in India",
      [IP], ["patents_act_1970_sec_53"]),
    q("who is entitled to apply for a patent, the inventor or the employer",
      [IP], ["patents_act_1970_sec_6"]),
    q("must a patent assignment be in writing to be valid",
      [IP], ["patents_act_1970_sec_68"]),
    q("grounds on which a trademark application can be refused",
      [IP], ["trademarks_act_1999_sec_9", "trademarks_act_1999_sec_11"]),
    q("how long is a trademark registration valid and how do we renew it",
      [IP], ["trademarks_act_1999_sec_25"]),
    q("what amounts to infringement of a registered trademark and what relief is available",
      [IP], ["trademarks_act_1999_sec_29", "trademarks_act_1999_sec_135"]),
    q("licensing our trademark to a franchisee as a registered user",
      [IP], ["trademarks_act_1999_sec_48", "trademarks_act_1999_sec_49"]),
    q("what is piracy of a registered design",
      [IP], ["designs_act_2000_sec_22", "designs_act_2000_sec_2"]),
    q("legal validity of contracts formed electronically through click-wrap acceptance",
      [IP], ["it_act_2000_sec_10A", "it_act_2000_sec_4"]),

    # ── single-domain: taxation ─────────────────────────────────────────────
    q("turnover threshold for compulsory GST registration",
      [TAX], ["cgst_act_2017_sec_22", "cgst_act_2017_sec_24"]),
    q("conditions for claiming input tax credit under the CGST Act",
      [TAX], ["cgst_act_2017_sec_16"]),
    q("can a small startup opt for the GST composition scheme",
      [TAX], ["cgst_act_2017_sec_10"]),
    q("what details must a GST tax invoice contain",
      [TAX], ["cgst_act_2017_sec_31"]),
    q("obligation of e-commerce operators to collect tax at source under GST",
      [TAX], ["cgst_act_2017_sec_52"]),
    q("advance tax instalments and due dates under the Income Tax Act 2025",
      [TAX], ["income_tax_act_2025_sec_408", "income_tax_act_2025_sec_404"]),
    q("TDS obligations on salary paid to employees",
      [TAX], ["income_tax_act_1961_sec_192", "income_tax_act_2025_sec_392"]),
    q("tax exemptions available to a unit set up in a Special Economic Zone",
      [TAX], ["sez_act_2005_sec_7", "sez_act_2005_sec_15"]),
    q("consequences of executing a share purchase agreement that is not duly stamped",
      [TAX], ["indian_stamp_act_1899_sec_35", "indian_stamp_act_1899_sec_62"]),
    q("can an Indian startup set up a wholly owned subsidiary abroad under FEMA",
      [TAX], [FEMA + "reg_6", FEMA + "reg_15"]),

    # ── single-domain: employment ───────────────────────────────────────────
    q("maternity benefit eligibility and payment under the Code on Social Security",
      [EMP], ["css_2020_sec_60", "mba_1961_sec_5"]),
    q("is a creche facility mandatory for our office",
      [EMP], ["mba_1961_sec_11A"]),
    q("gratuity eligibility after five years of continuous service",
      [EMP], ["poga_1972_sec_4", "css_2020_sec_54"]),
    q("social security schemes for gig and platform workers and the aggregator contribution",
      [EMP], ["css_2020_sec_114", "css_2020_sec_113"]),
    q("conditions precedent for lawful retrenchment of workmen",
      [EMP], ["ida_1947_sec_25F", "ida_1947_sec_25N"]),
    q("minimum and maximum statutory bonus payable to employees",
      [EMP], ["poba_1965_sec_10", "poba_1965_sec_11"]),
    q("employer contributions to the Employees Provident Fund",
      [EMP], ["epf_1952_sec_6", "epf_1952_sec_5"]),
    q("obligations of employers who engage apprentices",
      [EMP], ["app_1961_sec_11", "app_1961_sec_4"]),
    q("non-discrimination against persons with disabilities in employment and equal opportunity policy",
      [EMP], ["rpwd_2016_sec_20", "rpwd_2016_sec_21"]),
    q("licence requirements for engaging contract labour through a contractor",
      [EMP], ["cla_1970_sec_12", "cla_1970_sec_7"]),

    # ── multi-domain: 2 domains ─────────────────────────────────────────────
    q("we want to collect biometric attendance data from employees — what consent and employer duties apply",
      [DP, EMP], ["dpdp_act_2023_sec_6", "spdi_rules_2011_sec_3", "oshwc_2020_sec_6"]),
    q("our payments app stores card transaction data and charges a fee — data localisation and GST obligations",
      [DP, TAX], ["rbi_payment_data_storage_2018_dir_1", "cgst_act_2017_sec_9"]),
    q("are company directors and officers personally liable when the company suffers a data breach",
      [DP, CG], ["information_technology_act_2000_sec_85", "dpdp_act_2023_sec_33", "ca_2013_sec_2"]),
    q("we license our proprietary database to clients — copyright in the database and duties for the personal data inside it",
      [DP, IP], ["copyright_act_1957_sec_13", "copyright_act_1957_sec_2", "dpdp_act_2023_sec_8"]),
    q("an employee who invented a patentable process is being retrenched — who can apply for the patent and what retrenchment conditions apply",
      [IP, EMP], ["patents_act_1970_sec_6", "patents_act_1970_sec_68", "ida_1947_sec_25F"]),
    q("gratuity payable to a resigning employee and TDS on the final salary settlement",
      [EMP, TAX], ["poga_1972_sec_4", "income_tax_act_1961_sec_192"]),
    q("Companies Act allotment rules and stamp duty when issuing shares to investors",
      [CG, TAX], ["ca_2013_sec_39", "ca_2013_sec_62", "indian_stamp_act_1899_sec_35"]),
    q("converting our partnership firm into an LLP and transferring the firm's registered trademark to it",
      [CG, IP], ["llp_act_2008_sec_55", "trademarks_act_1999_sec_45"]),
    q("can we enforce a non-compete against a terminated employee and what retrenchment compensation is due",
      [CG, EMP], ["contract_act_1872_sec_27", "ida_1947_sec_25F"]),
    q("withholding tax on royalty paid to a foreign company for a software patent licence",
      [IP, TAX], ["income_tax_act_1961_sec_195", "income_tax_act_1961_sec_9", "patents_act_1970_sec_48"]),
    q("our e-commerce marketplace shares customer data with sellers — consumer protection and data protection obligations",
      [DP, CG], ["consumer_protection_act_2019_sec_93", "dpdp_act_2023_sec_8"]),
    q("handling a sexual harassment complaint while keeping the complainant's personal data confidential",
      [EMP, DP], ["posh_2013_sec_9", "posh_2013_sec_16", "dpdp_act_2023_sec_8"]),
    q("ESI — which employees must be insured and how the principal employer pays contributions",
      [EMP, TAX], ["esia_1948_sec_38", "esia_1948_sec_39", "esi_act_1948_sec_40"]),
    q("foreign SaaS company serving Indian users — GST on online information services and transferring user data out of India",
      [TAX, DP], ["igst_act_2017_sec_14", "dpdp_rules_2025_sec_15"]),
    q("our app uses Aadhaar authentication — restrictions on sharing Aadhaar data and copyright protection for the app code",
      [DP, IP], ["aadhaar_act_2016_sec_8", "aadhaar_act_2016_sec_29", "copyright_act_1957_sec_13"]),
    q("our company is in liquidation — what happens to unpaid employee wages and provident fund dues",
      [CG, EMP], ["ibc_2016_sec_53", "ibc_2016_sec_36", "pow_1936_sec_3"]),
    q("exporting software from an SEZ unit — zero-rated GST supply and copyright protection for the software",
      [TAX, IP], ["igst_act_2017_sec_16", "sez_act_2005_sec_7", "copyright_act_1957_sec_13"]),
    q("what must the annual report disclose about POSH compliance and CSR spending",
      [EMP, CG], ["posh_2013_sec_22", "ca_2013_sec_135"]),
    q("how long can we retain customer personal data after the service ends and what tax invoices must we issue",
      [DP, TAX], ["dpdp_rules_2025_sec_8", "cgst_act_2017_sec_31"]),
    q("vendor contract for outsourced processing of customer data — indemnity clause and data fiduciary obligations",
      [CG, DP], ["contract_act_1872_sec_124", "dpdp_act_2023_sec_8"]),

    # ── multi-domain: 3 domains ─────────────────────────────────────────────
    q("setting up a private limited company with 15 employees — incorporation, GST registration and provident fund obligations",
      [CG, TAX, EMP], ["ca_2013_sec_7", "cgst_act_2017_sec_22", "epf_1952_sec_6"]),
    q("launching a consumer mobile app — registering the brand name, getting user consent and incorporating the company",
      [IP, DP, CG], ["trademarks_act_1999_sec_18", "dpdp_act_2023_sec_6", "ca_2013_sec_7"]),
    q("hiring a freelance developer abroad — who owns the code, withholding tax on payments to a non-resident, and is he a workman",
      [IP, TAX, EMP], ["copyright_act_1957_sec_17", "copyright_act_1957_sec_18", "income_tax_act_1961_sec_195", "ida_1947_sec_2"]),
    q("an employee leaked customer data — breach reporting, penalty for disclosure, and liability of the company's officers",
      [DP, EMP, CG], ["dpdp_rules_2025_sec_7", "information_technology_act_2000_sec_72A", "information_technology_act_2000_sec_85"]),
    q("selling our startup's patents and trademarks to an acquirer — assignment formalities and stamp duty",
      [IP, TAX, CG], ["patents_act_1970_sec_68", "trademarks_act_1999_sec_45", "indian_stamp_act_1899_sec_35"]),
    q("fintech lending app — payment data localisation, GST on processing fees and liability for misleading advertisements",
      [DP, TAX, CG], ["rbi_payment_data_storage_2018_dir_1", "cgst_act_2017_sec_9", "consumer_protection_act_2019_sec_89"]),
    q("our factory — worker safety duties of the employer, CCTV monitoring of workers and protecting our product design",
      [EMP, DP, IP], ["oshwc_2020_sec_6", "dpdp_act_2023_sec_6", "designs_act_2000_sec_22"]),
    q("winding up an LLP — settling employee gratuity and bonus dues, and stamp duty on the deed transferring its assets",
      [CG, EMP, TAX], ["llp_act_2008_sec_64", "poga_1972_sec_4", "poba_1965_sec_8", "indian_stamp_act_1899_sec_35"]),
    q("hiring persons with disabilities — non-discrimination duties, consent for processing their data, and counting it towards CSR",
      [EMP, DP, CG], ["rpwd_2016_sec_20", "dpdp_rules_2025_sec_11", "ca_2013_sec_135"]),
    q("online gaming platform — GST on online services, intermediary safe harbour and copyright in game content",
      [TAX, DP, IP], ["igst_act_2017_sec_14", "information_technology_act_2000_sec_79", "copyright_act_1957_sec_14"]),
    q("a co-founder who is also an employee is leaving — assignment of the code he wrote and enforceability of his non-compete",
      [CG, IP, EMP], ["copyright_act_1957_sec_17", "copyright_act_1957_sec_19", "contract_act_1872_sec_27"]),
    q("outsourced payroll — TDS on salaries, ESI contributions and protecting employees' financial data",
      [TAX, EMP, DP], ["income_tax_act_1961_sec_192", "esia_1948_sec_39", "spdi_rules_2011_sec_3"]),

    # ── multi-domain: 4 domains ─────────────────────────────────────────────
    q("we are a 50-person SaaS company with many women employees, we collect user data and bill overseas clients — what are all our compliance obligations",
      [EMP, DP, TAX, CG], ["posh_2013_sec_4", "dpdp_act_2023_sec_8", "igst_act_2017_sec_16", "ca_2013_sec_92"]),
    q("edtech startup — incorporating the company, trademarking the brand, GST on online courses and consent for children's data",
      [CG, IP, TAX, DP], ["ca_2013_sec_7", "trademarks_act_1999_sec_18", "cgst_act_2017_sec_9", "dpdp_act_2023_sec_9"]),
    q("delivery startup using gig workers — platform worker social security, location data consent, GST TCS on marketplace sales and our app trademark",
      [EMP, DP, TAX, IP], ["css_2020_sec_114", "dpdp_act_2023_sec_6", "cgst_act_2017_sec_52", "trademarks_act_1999_sec_18"]),
    q("acquiring a smaller startup — transferring employees' gratuity, stamp duty on the share transfer, assigning their patents and issuing our shares as consideration",
      [EMP, TAX, IP, CG], ["poga_1972_sec_4", "indian_stamp_act_1899_sec_35", "patents_act_1970_sec_68", "ca_2013_sec_62"]),
    q("healthtech clinic chain run as an LLP — security of patient health data, nurses' maternity benefits and GST on services",
      [DP, EMP, CG, TAX], ["dpdp_rules_2025_sec_6", "mba_1961_sec_5", "llp_act_2008_sec_7", "cgst_act_2017_sec_9"]),

    # ── multi-domain: 5 domains ─────────────────────────────────────────────
    q("complete legal compliance checklist for an Indian AI startup with employees, user data, proprietary model code, GST billing and outside investors",
      [CG, EMP, DP, IP, TAX], ["ca_2013_sec_7", "ca_2013_sec_62", "posh_2013_sec_4", "dpdp_act_2023_sec_8",
                               "copyright_act_1957_sec_13", "cgst_act_2017_sec_22"]),
    q("we manufacture IoT devices in an SEZ, employ 200 factory workers, collect device telemetry, hold design registrations and are issuing new shares — what laws apply",
      [TAX, EMP, DP, IP, CG], ["sez_act_2005_sec_15", "oshwc_2020_sec_6", "dpdp_act_2023_sec_8",
                               "designs_act_2000_sec_22", "ca_2013_sec_62"]),
    q("our company is going insolvent — dues to employees, handling customer data during liquidation, selling our patents, pending tax dues and creditor priority",
      [CG, EMP, DP, IP, TAX], ["ibc_2016_sec_53", "ibc_2016_sec_36", "esia_1948_sec_94",
                               "dpdp_act_2023_sec_8", "patents_act_1970_sec_68"]),

    # ── edge cases: should abstain ──────────────────────────────────────────
    q("what is the penalty under section 999 of the DPDP Act", [DP], [], should_abstain=True),
    q("explain section 500 of the Payment of Gratuity Act", [EMP], [], should_abstain=True),
    q("Delaware franchise tax filing deadline for a US C-corporation", ["unknown"], [], should_abstain=True),
    q("GDPR Article 17 obligations for a company established only in the EU", ["unknown"], [], should_abstain=True),
    q("what will the GST rate on SaaS subscriptions be in 2035", [TAX], [], should_abstain=True),
    q("what is the best pizza place in Bangalore", ["unknown"], [], should_abstain=True),
    q("how do I improve my startup's Instagram engagement", ["unknown"], [], should_abstain=True),
    q("recommend a cloud hosting provider for our backend", ["unknown"], [], should_abstain=True),
    q("write me a poem about corporate law", ["unknown"], [], should_abstain=True),
    q("what is the capital of Australia", ["unknown"], [], should_abstain=True),
]


# ── helpers ──────────────────────────────────────────────────────────────────
def category(test: dict) -> str:
    if test["should_abstain"]:
        return "edge"
    n = len(test["domains"])
    return "single" if n == 1 else f"multi_{n}"


def domain_of(pid: str) -> str | None:
    # longest prefix wins (e.g. "ieso_rules_1946_" before "ieso_1946_" is irrelevant
    # here, but keeps "information_technology_act_2000_" distinct from "it_act_2000_")
    for prefix in sorted(STATUTE_PREFIX_DOMAIN, key=len, reverse=True):
        if pid.startswith(prefix):
            return STATUTE_PREFIX_DOMAIN[prefix]
    return None


def ratio(num: float, den: float) -> float | None:
    return round(num / den, 3) if den else None


def mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def validate() -> int:
    """Offline check: every expected provision id exists in datasets/complete/."""
    known = set()
    for path in glob.glob(str(COMPLETE_DIR / "complete_*.json")):
        with open(path) as f:
            known.update(o["provision_id"] for o in json.load(f))

    problems = []
    for i, t in enumerate(TEST_QUERIES, 1):
        for pid in t["expected_sections"]:
            if pid not in known:
                problems.append(f"[{i:03d}] unknown provision id: {pid}")
            elif domain_of(pid) not in t["domains"]:
                problems.append(f"[{i:03d}] {pid} ({domain_of(pid)}) not in labeled domains {t['domains']}")
        if not t["should_abstain"] and not t["expected_sections"]:
            problems.append(f"[{i:03d}] answerable query has no expected sections")

    counts = Counter(category(t) for t in TEST_QUERIES)
    print(f"{len(TEST_QUERIES)} queries | {dict(sorted(counts.items()))}")
    print(f"{len(known)} provision ids in {COMPLETE_DIR}")
    for p in problems:
        print("  " + p)
    print("OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


def summarize(rows: list[dict]) -> dict:
    answerable = [r for r in rows if not r["should_abstain"]]
    edge = [r for r in rows if r["should_abstain"]]
    cit_hits = sum(r["citation_hits"] for r in answerable)
    cit_total = sum(len(r["expected_sections"]) for r in answerable)
    return {
        "n": len(rows),
        "domain_recall": mean([r["domain_recall"] for r in answerable]),
        "domain_precision": mean([r["domain_precision"] for r in answerable]),
        "domain_exact_match": mean([float(r["domain_exact_match"]) for r in answerable]),
        "primary_domain_accuracy": mean([float(r["primary_domain_correct"]) for r in answerable]),
        "citation_recall": ratio(cit_hits, cit_total),
        "citation_domain_coverage": mean([r["citation_domain_coverage"] for r in answerable]),
        "answered_pct": ratio(sum(not r["abstained"] for r in rows) * 100, len(rows)),
        "false_abstain_rate": ratio(sum(r["abstained"] for r in answerable), len(answerable)),
        "abstain_accuracy": ratio(sum(r["abstained"] for r in edge), len(edge)),
        "avg_elapsed_seconds": mean([r["elapsed_seconds"] for r in rows]),
    }


def per_domain(rows: list[dict]) -> dict:
    out = {}
    for d in ALL_DOMAINS:
        subset = [r for r in rows if not r["should_abstain"] and d in r["expected_domains"]]
        if not subset:
            continue
        out[d] = {
            "n": len(subset),
            "detection_recall": mean([float(d in r["predicted_domains"]) for r in subset]),
            "cited_recall": mean([float(d in r["cited_domains"]) for r in subset]),
            "false_positive_detections": sum(
                1 for r in rows
                if not r["should_abstain"] and d in r["predicted_domains"] and d not in r["expected_domains"]
            ),
        }
    return out


# ── main loop ────────────────────────────────────────────────────────────────
def score_result(test: dict, result: dict, elapsed: float) -> dict:
    expected_domains = set(test["domains"]) - {"unknown"}
    expected = set(test["expected_sections"])
    predicted = set(result.get("all_domains") or [result.get("domain")]) - {None, "unknown"}
    did_abstain = bool(result.get("abstain", False))
    raw_verified = set(result.get("verified_citations", []))
    # an abstention shows the user no citations, so it earns no credit (same rule as the baselines)
    verified = set() if did_abstain else raw_verified
    cited_domains = {domain_of(p) for p in verified} - {None}
    hits = len(verified & expected)
    overlap = predicted & expected_domains
    return {
        "query": test["query"],
        "category": category(test),
        "should_abstain": test["should_abstain"],
        "expected_domains": sorted(expected_domains),
        "predicted_domains": sorted(predicted),
        "primary_domain": result.get("domain"),
        "domain_recall": ratio(len(overlap), len(expected_domains)),
        "domain_precision": ratio(len(overlap), len(predicted)),
        "domain_exact_match": predicted == expected_domains,
        "primary_domain_correct": result.get("domain") in expected_domains,
        "expected_sections": sorted(expected),
        "verified_citations": sorted(verified),
        "verified_citations_before_abstain": sorted(raw_verified),
        "citation_hits": hits,
        "citation_recall": ratio(hits, len(expected)),
        "cited_domains": sorted(cited_domains),
        "citation_domain_coverage": ratio(len(cited_domains & expected_domains), len(expected_domains)),
        "abstained": did_abstain,
        "abstain_correct": did_abstain == test["should_abstain"],
        "abstain_reason": result.get("abstain_reason", ""),
        "llm_error": result.get("llm_error", ""),
        "domain_errors": [d.get("domain") for d in result.get("domain_results") or [] if d.get("grade") == "error"],
        "groundedness_grade": result.get("groundedness_grade"),
        "answerability_grade": result.get("answerability_grade"),
        "retry_count_retrieval": result.get("retry_count_retrieval", 0),
        "retry_count_generation": result.get("retry_count_generation", 0),
        "domain_detection_method": result.get("domain_detection_method", ""),
        "domain_results": result.get("domain_results", []),
        "answer": result.get("answer", ""),
        "elapsed_seconds": round(elapsed, 2),
        "trace": result.get("trace", []),
    }


def is_incomplete(row: dict) -> bool:
    """An LLM failure (quota, outage) is not a result: such rows are re-run instead of being scored.
    Besides hard errors this catches the stages that carry on after a failed call (claim check "skipped",
    groundedness "unverified", LLM domain detection missing): `llm_failures` counts every call that finally
    failed while the query ran."""
    return (bool(row.get("llm_error")) or bool(row.get("domain_errors"))
            or int(row.get("llm_failures") or 0) > 0)


def incomplete_reason(row: dict) -> str:
    if row.get("llm_error"):
        return row["llm_error"]
    if row.get("domain_errors"):
        return f"LLM error in domain(s) {row['domain_errors']}"
    return f"{row.get('llm_failures')} LLM call(s) failed: {row.get('llm_failure_last', '')}"


def check_manifest(path: Path, manifest: dict, resume: bool) -> str:
    """A resumed run must use the same settings as the rows already saved; returns an error message or ''."""
    keys = ("provider", "model", "as_of", "settings", "temperature")
    if resume and path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        diff = [k for k in keys if old.get(k) != manifest.get(k)]
        if diff:
            return (f"saved rows in this tag were produced with different settings ({', '.join(diff)}):\n"
                    f"  saved: { {k: old.get(k) for k in diff} }\n  now:   { {k: manifest.get(k) for k in diff} }\n"
                    f"Use another --tag, or drop --resume to start this tag again.")
        manifest["sessions"] = old.get("sessions", [])
    manifest.setdefault("sessions", [])
    manifest["sessions"].append({"started": datetime.now().isoformat(timespec="seconds"), "commit": git_commit()})
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return ""


def git_commit() -> str:
    import subprocess
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10,
                             cwd=ROOT)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                               text=True, timeout=10, cwd=ROOT).stdout.strip()
        return (out.stdout.strip() or "unknown") + ("+dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def build_summary(rows: list, provider: str, total: int, errors: list, extra: dict | None = None) -> dict:
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
        if r["category"].startswith("multi_"):
            by_cat["multi_all"].append(r)
    return {
        "provider": provider, **(extra or {}),
        "total": total, "scored": len(rows), "errored": len(errors),
        "overall": summarize(rows) if rows else {},
        "by_category": {c: summarize(rs) for c, rs in sorted(by_cat.items())},
        "per_domain": per_domain(rows),
    }


def print_summary(summary: dict) -> None:
    o = summary["overall"]
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"Scored queries            : {summary['scored']} of {summary['total']} ({summary['errored']} errored)")
    for key in ("answered_pct", "domain_recall", "domain_precision", "domain_exact_match",
                "primary_domain_accuracy", "citation_recall", "citation_domain_coverage",
                "false_abstain_rate", "abstain_accuracy", "avg_elapsed_seconds"):
        print(f"{key:26s}: {o.get(key)}")
    print("\nBy category (domain_recall / exact / citation_recall / coverage / abstain_acc):")
    for c, s in summary["by_category"].items():
        print(f"  {c:10s} n={s['n']:3d}  {s['domain_recall']} / {s['domain_exact_match']} / "
              f"{s['citation_recall']} / {s['citation_domain_coverage']} / {s['abstain_accuracy']}")
    print("\nPer domain (detection_recall / cited_recall / false_positives):")
    for d, s in summary["per_domain"].items():
        print(f"  {d:22s} n={s['n']:3d}  {s['detection_recall']} / {s['cited_recall']} / "
              f"{s['false_positive_detections']}")
    print(f"{'='*70}\n")


def provider_name(provider: str | None) -> str:
    return provider or os.getenv("LLM_PROVIDER", "gemini")


def evaluate(selected: list[dict], sleep_s: float, provider: str | None = None, as_of: str | None = None,
             tag: str = "pipeline", resume: bool = False, options: dict | None = None):
    # Imported lazily so --validate works without a database or API key.
    import legal_workflow_generator.config.values as app_config  # noqa: F401  (loads .env, fails fast)
    from legal_workflow_generator.agent.graph import graph
    from legal_workflow_generator.agent.settings import RunSettings, build_config
    from legal_workflow_generator.agent import groq_client, gemini_client
    usage_src = groq_client.USAGE if provider_name(provider) == "groq" else gemini_client.USAGE

    cfg = build_config(provider=provider, as_of=as_of, **(options or {}))
    settings = RunSettings.from_config(cfg)
    provider = settings.provider
    model = app_config.GROQ_MODEL if provider == "groq" else app_config.GEMINI_MODEL

    import legal_workflow_generator.agent.nodes as nodes
    from run_status import EXIT_DONE, EXIT_PARTIAL, EXIT_STOPPED, FailureTracker

    temperature = os.getenv("GEMINI_TEMPERATURE" if provider == "gemini" else "GROQ_TEMPERATURE", "")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows_path = RESULTS_DIR / f"eval_{tag}.jsonl"
    manifest = {"tag": tag, "provider": provider, "model": model, "as_of": settings.as_of_date.isoformat(),
                "temperature": temperature,
                "settings": {k: v for k, v in vars(settings).items() if k not in ("as_of", "provider", "trace_timing")}}
    problem = check_manifest(RESULTS_DIR / f"eval_{tag}.manifest.json", manifest, resume)
    if problem:
        print("ERROR: " + problem)
        return 1

    wanted = {t["query"] for t in selected}
    old = load_jsonl(rows_path) if resume else []
    done = {r["query"]: r for r in old if not is_incomplete(r) and r["query"] in wanted}
    if not resume and rows_path.exists():
        rows_path.unlink()
    elif resume:   # keep only valid rows; the rest are re-run below
        rows_path.write_text("".join(json.dumps(r, default=str) + "\n" for r in done.values()), encoding="utf-8")

    total = len(selected)
    errors = []
    tracker = FailureTracker()
    print(f"\n{'='*70}")
    print(f"COMPLETE-DATASET MULTI-DOMAIN AGENT EVALUATION — {total} queries "
          f"(provider={provider}, model={model}, as_of={settings.as_of_date}, tag={tag}, "
          f"{len(done)} already done)")
    print(f"{'='*70}\n")

    try:
        for i, test in enumerate(selected, 1):
            query, cat = test["query"], category(test)
            if query in done:
                continue
            print(f"[{i:03d}/{total}] ({cat}) {query[:70]}...", flush=True)
            before = dict(usage_src)
            fails_before = nodes.LLM_FAILURES["count"]
            try:
                start = time.time()
                result = graph.invoke({"query": query}, config=cfg)
                row = score_result(test, result, time.time() - start)
            except Exception as e:  # noqa: BLE001
                msg = f"{type(e).__name__}: {e}"
                print(f"        ERROR (not saved, retried on the next run): {msg[:200]}")
                errors.append({"query": query, "category": cat, "error": msg})
                tracker.failed(msg)
                if tracker.should_stop:
                    break
                continue
            row["llm_failures"] = nodes.LLM_FAILURES["count"] - fails_before
            if row["llm_failures"]:
                row["llm_failure_last"] = nodes.LLM_FAILURES["last"]
            row["tokens"] = {k: usage_src[k] - before[k] for k in usage_src}
            row.update(provider=provider, model=model, as_of=settings.as_of_date.isoformat(),
                       temperature=temperature, run_date=datetime.now().isoformat(timespec="seconds"))
            if is_incomplete(row):
                msg = incomplete_reason(row)
                print(f"        INCOMPLETE (not saved, retried on the next run): {msg[:200]}")
                errors.append({"query": query, "category": cat, "error": msg})
                tracker.failed(msg)
                if tracker.should_stop:
                    break
                continue
            tracker.ok()
            with open(rows_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
            done[query] = row
            status = "ABSTAIN" if row["abstained"] else "ANSWER"
            ok = "OK" if row["abstain_correct"] else "WRONG"
            print(f"        {status} ({ok}) | domains exp={row['expected_domains']} got={row['predicted_domains']} "
                  f"| cit_recall={row['citation_recall']} cov={row['citation_domain_coverage']} "
                  f"| {row['elapsed_seconds']}s | {row['tokens']['total_tokens']} tok", flush=True)
            if row["abstained"]:
                print(f"        reason: {row['abstain_reason'][:200]}")
            time.sleep(sleep_s)
    except KeyboardInterrupt:
        tracker.stop_reason = "interrupted (Ctrl+C)"

    order = {t["query"]: n for n, t in enumerate(TEST_QUERIES)}
    rows = sorted(done.values(), key=lambda r: order.get(r["query"], 10**6))
    summary = build_summary(rows, provider, total, errors,
                            {"model": model, "as_of": settings.as_of_date.isoformat(), "tag": tag,
                             "options": options or {},
                             "tokens": sum(r.get("tokens", {}).get("total_tokens", 0) for r in rows)})
    print_summary(summary)
    sum_path = RESULTS_DIR / f"eval_{tag}_summary.json"
    summary["complete"] = len(rows) == total
    sum_path.write_text(json.dumps({"summary": summary, "errors": errors}, indent=2), encoding="utf-8")

    fields = ["query", "category", "expected_domains", "predicted_domains", "primary_domain",
              "domain_recall", "domain_precision", "domain_exact_match", "citation_recall",
              "citation_domain_coverage", "abstained", "abstain_correct",
              "retry_count_retrieval", "retry_count_generation", "elapsed_seconds"]
    csv_path = RESULTS_DIR / f"eval_{tag}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("|".join(r[k]) if isinstance(r[k], list) else r[k]) for k in fields})
    print(f"Results saved to:\n  {rows_path}\n  {sum_path}\n  {csv_path}")
    print(f"\n{len(rows)}/{total} queries have a saved result.")
    if tracker.should_stop:
        print(f"STOPPED: {tracker.stop_reason}\nProgress is saved; run the same command again later to continue.")
        return EXIT_STOPPED
    if len(rows) < total:
        print("Some queries failed this pass; run the same command again to retry them.")
        return EXIT_PARTIAL
    return EXIT_DONE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic RAG eval on the complete dataset with multi-domain queries (100 queries)."
    )
    parser.add_argument("--category", choices=["all", "single", "multi", "edge"], default="all",
                        help="Run only a subset of the test set (default: all).")
    parser.add_argument("--provider", choices=["gemini", "groq"], default=None,
                        help="LLM provider for every agent call (default: LLM_PROVIDER env var, else gemini).")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N selected queries.")
    parser.add_argument("--sleep", type=float, default=5.0,
                        help="Seconds to sleep between queries to avoid rate limiting (default: 5).")
    parser.add_argument("--as-of", default="2026-10-15",
                        help="Query date for the in-force filter (default 2026-10-15, the paper's date).")
    parser.add_argument("--tag", default="pipeline",
                        help="Name of the run; results go to <results dir>/eval_<tag>.jsonl.")
    parser.add_argument("--resume", action="store_true", help="Keep saved rows of this tag and run the rest.")
    parser.add_argument("--option", action="append", default=[], metavar="KEY=VALUE",
                        help="Extra RunSettings option, e.g. --option top_k=8 (repeatable).")
    parser.add_argument("--summary", action="store_true", help="Only summarise the saved rows of --tag.")
    parser.add_argument("--validate", action="store_true",
                        help="Offline: verify every expected provision id exists in datasets/complete/ "
                             "and belongs to a labeled domain, then exit. No DB or API calls.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.validate:
        sys.exit(validate())

    selected = [
        t for t in TEST_QUERIES
        if args.category == "all"
        or (args.category == "multi" and category(t).startswith("multi_"))
        or category(t) == args.category
    ]
    if args.limit:
        selected = selected[: args.limit]
    options = dict(o.split("=", 1) for o in args.option)
    if args.summary:
        rows = load_jsonl(RESULTS_DIR / f"eval_{args.tag}.jsonl")
        print_summary(build_summary(rows, rows[0].get("provider", "?") if rows else "?", len(selected), []))
        sys.exit(0)
    sys.exit(evaluate(selected, sleep_s=args.sleep, provider=args.provider, as_of=args.as_of, tag=args.tag,
                      resume=args.resume, options=options))
