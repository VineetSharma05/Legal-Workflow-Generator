"""
Fresh held-out test set (50 queries) for eval_complete_multidomain.py.

Design rules
------------
  * Written AFTER v4 was frozen, never run against any system, no query text shared with the 100-query set.
  * Every in-scope query targets provisions from statutes that exist only in the complete corpus.
  * No scope-check rule or prompt rule was written with these in view.
  * 10 edge queries are all outside the corpus (criminal, family, securities, arbitration, real estate,
    motor vehicles, customs, environment, foreign privacy law, pure off-topic).

  Mix: 10 single | 14 multi_2 | 9 multi_3 | 4 multi_4 | 3 multi_5 | 10 edge
"""

DP, CG, IP, TAX, EMP = ("data_protection", "corporate_governance", "ip_licensing", "taxation", "employment")


def q(query, domains, sections, should_abstain=False):
    return {"query": query, "domains": domains, "expected_sections": sections,
            "should_abstain": should_abstain}


TEST_QUERIES = [
    # ── single-domain (10) ───────────────────────────────────────────────────
    q("what is the punishment for a business that misuses a customer's Aadhaar identity information after authentication",
      [DP], ["aadhaar_act_2016_sec_40", "aadhaar_act_2016_sec_43"]),
    q("how fast must a telecom operator report a security incident to the government and must it appoint a security officer",
      [DP], ["telecom_cyber_security_rules_2024_sec_7", "telecom_cyber_security_rules_2024_sec_6"]),
    q("can we cancel a contract because the other side pressured us from a position of dominance or misled us before signing",
      [CG], ["contract_act_1872_sec_19", "contract_act_1872_sec_16", "contract_act_1872_sec_18"]),
    q("when is a manufacturer liable if a defective product injures a buyer, and when does it escape liability",
      [CG], ["consumer_protection_act_2019_sec_84", "consumer_protection_act_2019_sec_87"]),
    q("can someone force us to license our patent if we are not working it in India",
      [IP], ["patents_act_1970_sec_84", "patents_act_1970_sec_83"]),
    q("can a registered trademark be taken off the register if the owner has never used it",
      [IP], ["trademarks_act_1999_sec_47", "trademarks_act_1999_sec_57"]),
    q("how do we get a refund of unutilised GST input credit on exported services and what interest do we get if it is delayed",
      [TAX], ["cgst_act_2017_sec_54", "cgst_act_2017_sec_56"]),
    q("we found an agreement that was never properly stamped, how do we get the Collector to fix the duty so it can be used in court",
      [TAX], ["indian_stamp_act_1899_sec_31", "indian_stamp_act_1899_sec_40"]),
    q("is a strike illegal while conciliation proceedings are pending in a company",
      [EMP], ["ida_1947_sec_24", "ida_1947_sec_23"]),
    q("what can an employer lawfully deduct from wages for damaged equipment or fines",
      [EMP], ["pow_1936_sec_10", "pow_1936_sec_8", "pow_1936_sec_7"]),

    # ── multi-domain: 2 domains (14) ─────────────────────────────────────────
    q("a contractor brings inter-State migrant workmen to our site: can we process their personal data for registration without separate consent, what are the contractor's duties and what goes on the government portal",
      [DP, EMP], ["dpdp_act_2023_sec_7", "ismw_1979_sec_12", "oshwc_2020_sec_21"]),
    q("we run a crypto exchange: how long must we keep KYC and transaction records for CERT-In and how long must GST books of account be kept",
      [DP, TAX], ["cert_in_directions_2022_dir_6", "cgst_act_2017_sec_36"]),
    q("an unlisted public company wants to convert into an LLP: what are the conditions, how is the conversion registered and is the transfer of its assets to the LLP taxed as capital gains",
      [CG, TAX], ["llp_act_2008_sec_57", "llp_act_2008_sec_58", "llp_act_2008_schedule_4", "income_tax_act_1961_sec_47"]),
    q("a newspaper startup employs journalists: who owns copyright in staff articles and what gratuity are journalists entitled to",
      [IP, EMP], ["copyright_act_1957_sec_17", "wj_1955_sec_5"]),
    q("choosing a name for our new LLP when a similar trademark is already registered, can the name be reserved and what rights does the mark owner have",
      [CG, IP], ["llp_act_2008_sec_15", "llp_act_2008_sec_16", "trademarks_act_1999_sec_28"]),
    q("a worker died in an accident at our factory: what compensation is the employer liable for and is it deductible as a business expense",
      [TAX, EMP], ["eca_1923_sec_3", "eca_1923_sec_4", "income_tax_act_1961_sec_37"]),
    q("a borrower disputes an inaccurate credit report from a credit bureau: what correction rights exist and can he complain as a consumer",
      [DP, CG], ["cicra_2005_sec_21", "cicra_2005_sec_20", "consumer_protection_act_2019_sec_30"]),
    q("our social platform hosts user videos and a rights holder demands removal: grievance officer duties and what exclusive rights the copyright owner has over the video",
      [DP, IP], ["it_intermediary_guidelines_2021_sec_3", "copyright_act_1957_sec_14"]),
    q("claiming depreciation on a patent we acquired and registering the patent assignment in our name",
      [TAX, IP], ["income_tax_act_2025_sec_33", "patents_act_1970_sec_69"]),
    q("can a 17 year old be engaged as an apprentice and who must sign the apprenticeship contract",
      [CG, EMP], ["app_1961_sec_4", "app_1961_sec_3", "contract_act_1872_sec_11"]),
    q("an insolvent company sold its building to a related party far below market value and we also need to know what stamp duty applies to the transfer deed",
      [CG, TAX], ["ibc_2016_sec_45", "ibc_2016_sec_46", "indian_stamp_act_1899_sec_3"]),
    q("who is personally liable in an LLP when provident fund contributions are not paid",
      [CG, EMP], ["llp_act_2008_sec_8", "epf_1952_sec_14"]),
    q("an author assigned book rights to our publishing house and now disputes the assignment because we are not exploiting it, and she asks about tax relief on her royalty income",
      [TAX, IP], ["income_tax_act_2025_sec_151", "copyright_act_1957_sec_19A"]),
    q("an apprentice helped create our new product design: is an apprentice a worker under labour law and what rights does design registration give us",
      [EMP, IP], ["app_1961_sec_18", "designs_act_2000_sec_11"]),

    # ── multi-domain: 3 domains (9) ──────────────────────────────────────────
    q("we want to pay part of the annual bonus to employees in shares: private placement conditions, tax on the share benefit and how bonus already paid is adjusted",
      [CG, TAX, EMP], ["ca_2013_sec_42", "income_tax_act_2025_sec_17", "poba_1965_sec_17", "poba_1965_sec_19"]),
    q("credit scoring app: registering our brand as a trademark, privacy principles for credit information and complaints about unfair practices to the consumer authority",
      [IP, DP, CG], ["trademarks_act_1999_sec_23", "cicra_2005_sec_20", "consumer_protection_act_2019_sec_17"]),
    q("in-house R&D lab: tax deduction for scientific research spending, employer's duty to keep the lab safe and the rights a patent gives us",
      [TAX, EMP, IP], ["income_tax_act_1961_sec_35", "oshwc_2020_sec_23", "patents_act_1970_sec_48"]),
    q("data centre business: keeping subscriber records for five years, paying GST on time and a clause limiting our customers' right to sue",
      [DP, TAX, CG], ["cert_in_directions_2022_dir_5", "cgst_act_2017_sec_49", "contract_act_1872_sec_28"]),
    q("news startup: working hours and leave for journalists, privacy policy for reader data and who owns what the journalists write",
      [EMP, DP, IP], ["wj_1955_sec_6", "spdi_rules_2011_sec_4", "copyright_act_1957_sec_17"]),
    q("closing down an undertaking with 200 workers: voluntary liquidation, permission to close and cancelling the GST registration",
      [CG, EMP, TAX], ["ibc_2016_sec_59", "ida_1947_sec_25-O", "cgst_act_2017_sec_29"]),
    q("franchise network: licensing our mark to franchisees as registered users, an agreement too vague to enforce and the tax invoice franchisees must get",
      [IP, CG, TAX], ["trademarks_act_1999_sec_48", "trademarks_act_1999_sec_49", "contract_act_1872_sec_29", "cgst_act_2017_sec_31"]),
    q("payroll vendor leaked employee bank details: compensation under the IT Act, can we cancel the vendor contract for fraud and are salaries still due on time",
      [DP, CG, EMP], ["information_technology_act_2000_sec_43A", "contract_act_1872_sec_17", "contract_act_1872_sec_19", "pow_1936_sec_5"]),
    q("two founders co-own a patent and run an LLP together: can one license it alone, how do partners relate to each other and can the LLP's losses be carried forward",
      [IP, CG, TAX], ["patents_act_1970_sec_50", "llp_act_2008_sec_23", "income_tax_act_1961_sec_72"]),

    # ── multi-domain: 4 domains (4) ──────────────────────────────────────────
    q("restaurant chain run as an LLP: creche for women staff, how we collect customer information on CCTV and apps, GST registration limits and annual accounts",
      [EMP, DP, TAX, CG], ["css_2020_sec_67", "spdi_rules_2011_sec_5", "cgst_act_2017_sec_22", "llp_act_2008_sec_34"]),
    q("wearable health device maker: a rival opposes our trademark application, the device collects users' health readings, factory staff daily hours and IGST on imported components",
      [IP, DP, EMP, TAX], ["trademarks_act_1999_sec_21", "dpdp_act_2023_sec_4", "fa_1948_sec_54", "igst_act_2017_sec_7", "igst_act_2017_sec_11"]),
    q("buying a distressed company through an insolvency resolution plan: plan approval, workers' provident fund priority, its credit data and unpaid GST recovery",
      [CG, EMP, DP, TAX], ["ibc_2016_sec_31", "epf_1952_sec_11", "cicra_2005_sec_19", "cgst_act_2017_sec_79"]),
    q("advertising agency LLP: assignment of copyright from freelance writers, penalties for misleading ads, GST credit notes to clients and staff bonus timing",
      [IP, CG, TAX, EMP], ["copyright_act_1957_sec_19", "consumer_protection_act_2019_sec_20", "cgst_act_2017_sec_34", "poba_1965_sec_19"]),

    # ── multi-domain: 5 domains (3) ──────────────────────────────────────────
    q("converting our partnership into an LLP for an ed-tech business: gratuity for existing staff, disclosure of student data, assigning our brand to the LLP and GST returns",
      [CG, EMP, DP, IP, TAX], ["llp_act_2008_sec_55", "llp_act_2008_sec_58", "css_2020_sec_53", "spdi_rules_2011_sec_6", "trademarks_act_1999_sec_38", "cgst_act_2017_sec_39"]),
    q("medical device maker: liability for a defective device, factory occupier duties, security of patient Aadhaar data, design registration cancellation risk and GST outward supply details",
      [CG, EMP, DP, IP, TAX], ["consumer_protection_act_2019_sec_84", "fa_1948_sec_7", "aadhaar_act_2016_sec_30", "designs_act_2000_sec_19", "cgst_act_2017_sec_37"]),
    q("consumer app company expanding abroad: RBI approval for overseas investment when eligibility norms are not met, board composition of the Indian parent, equal recruitment of men and women, duties of a significant social media intermediary and whether using our mark on exports counts as use",
      [TAX, CG, EMP, DP, IP], ["fema_transfer_issue_foreign_security_regulations_2004_reg_9", "ca_2013_sec_149", "era_1976_sec_5", "it_intermediary_guidelines_2021_sec_4", "trademarks_act_1999_sec_56"]),

    # ── edge: outside the corpus, should abstain (10) ────────────────────────
    q("what is the punishment for cheating and criminal breach of trust by a company director under the Bharatiya Nyaya Sanhita",
      ["unknown"], [], should_abstain=True),
    q("what are the grounds for divorce under the Hindu Marriage Act", ["unknown"], [], should_abstain=True),
    q("what are the open offer obligations when an acquirer crosses 25 percent under the SEBI takeover regulations",
      ["unknown"], [], should_abstain=True),
    q("on what grounds can an arbitral award be set aside by a court in India", ["unknown"], [], should_abstain=True),
    q("a builder delayed possession of my flat for three years, what relief does RERA give", ["unknown"], [], should_abstain=True),
    q("what is the penalty for driving under the influence of alcohol under the Motor Vehicles Act", ["unknown"], [], should_abstain=True),
    q("does a California resident have the right to delete data from our Indian startup under the CCPA", ["unknown"], [], should_abstain=True),
    q("what customs duty applies when we import laptops for our employees", ["unknown"], [], should_abstain=True),
    q("do we need environmental clearance and a pollution control board consent before starting a chemical plant",
      ["unknown"], [], should_abstain=True),
    q("what is a good recipe for soft idli batter", ["unknown"], [], should_abstain=True),
]
