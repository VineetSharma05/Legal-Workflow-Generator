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
  python evals/eval_complete_multidomain.py --validate      # offline: check every
                                                            # expected id exists in
                                                            # datasets/complete/
"""

import argparse
import csv
import glob
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS_DIR = Path(__file__).resolve().parent / "results"
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
def evaluate(selected: list[dict], sleep_s: float):
    # Imported lazily so --validate works without a database or API key.
    import legal_workflow_generator.config.values  # noqa: F401  (loads .env, fails fast)
    from legal_workflow_generator.agent.graph import graph

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total = len(selected)
    rows, errors = [], []

    print(f"\n{'='*70}")
    print(f"COMPLETE-DATASET MULTI-DOMAIN AGENT EVALUATION — {total} queries")
    print(f"{'='*70}\n")

    for i, test in enumerate(selected, 1):
        query, cat = test["query"], category(test)
        expected_domains = set(test["domains"]) - {"unknown"}
        expected = set(test["expected_sections"])
        print(f"[{i:03d}/{total}] ({cat}) {query[:70]}...")

        try:
            start = time.time()
            result = graph.invoke({"query": query})
            elapsed = round(time.time() - start, 2)

            predicted = set(result.get("all_domains") or [result.get("domain")]) - {None, "unknown"}
            verified = set(result.get("verified_citations", []))
            cited_domains = {domain_of(p) for p in verified} - {None}
            did_abstain = bool(result.get("abstain", False))

            hits = len(verified & expected)
            overlap = predicted & expected_domains
            row = {
                "query": query,
                "category": cat,
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
                "citation_hits": hits,
                "citation_recall": ratio(hits, len(expected)),
                "cited_domains": sorted(cited_domains),
                "citation_domain_coverage": ratio(len(cited_domains & expected_domains), len(expected_domains)),
                "abstained": did_abstain,
                "abstain_correct": did_abstain == test["should_abstain"],
                "abstain_reason": result.get("abstain_reason", ""),
                "retry_count_retrieval": result.get("retry_count_retrieval", 0),
                "retry_count_generation": result.get("retry_count_generation", 0),
                "domain_detection_method": result.get("domain_detection_method", ""),
                "elapsed_seconds": elapsed,
                "trace": result.get("trace", []),
            }
            rows.append(row)

            status = "ABSTAIN" if did_abstain else "ANSWER"
            ok = "OK" if row["abstain_correct"] else "WRONG"
            print(
                f"        {status} ({ok}) | domains exp={sorted(expected_domains)} got={sorted(predicted)} "
                f"| cit_recall={row['citation_recall']} cov={row['citation_domain_coverage']} | {elapsed}s"
            )
            if did_abstain:
              print(f"        reason: {row['abstain_reason']}")
              print(f"        retries: retrieval={row['retry_count_retrieval']} generation={row['retry_count_generation']}")
              print(f"        trace: {' | '.join(row['trace'])}")
        except Exception as e:
            print(f"        ERROR: {e}")
            errors.append({"query": query, "category": cat, "error": str(e)})

        time.sleep(sleep_s)  # avoid rate limiting

    # ── summary ──────────────────────────────────────────────────────────────
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
        if r["category"].startswith("multi_"):
            by_cat["multi_all"].append(r)

    summary = {
        "total": total,
        "errored": len(errors),
        "overall": summarize(rows) if rows else {},
        "by_category": {c: summarize(rs) for c, rs in sorted(by_cat.items())},
        "per_domain": per_domain(rows),
    }

    o = summary["overall"]
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"Total queries             : {total} ({len(errors)} errored)")
    for key in ("answered_pct", "domain_recall", "domain_precision", "domain_exact_match",
                "primary_domain_accuracy", "citation_recall", "citation_domain_coverage",
                "false_abstain_rate", "abstain_accuracy"):
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

    # ── save results ─────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"eval_complete_multidomain_{timestamp}.json"
    csv_path = RESULTS_DIR / f"eval_complete_multidomain_{timestamp}.csv"

    with open(json_path, "w") as f:
        json.dump({"summary": summary, "results": rows, "errors": errors}, f, indent=2)

    fields = ["query", "category", "expected_domains", "predicted_domains", "primary_domain",
              "domain_recall", "domain_precision", "domain_exact_match", "citation_recall",
              "citation_domain_coverage", "abstained", "abstain_correct",
              "retry_count_retrieval", "retry_count_generation", "elapsed_seconds"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("|".join(r[k]) if isinstance(r[k], list) else r[k]) for k in fields})

    print("Results saved to:")
    print(f"  {json_path}")
    print(f"  {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic RAG eval on the complete dataset with multi-domain queries (100 queries)."
    )
    parser.add_argument("--category", choices=["all", "single", "multi", "edge"], default="all",
                        help="Run only a subset of the test set (default: all).")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N selected queries.")
    parser.add_argument("--sleep", type=float, default=5.0,
                        help="Seconds to sleep between queries to avoid rate limiting (default: 5).")
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
    evaluate(selected, sleep_s=args.sleep)
