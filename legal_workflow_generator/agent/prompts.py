"""Prompt builders for the LangGraph agent pipeline."""


def grade_domain_prompt(domain_label: str, sub_q: str, chunks: str) -> tuple[str, str]:
    system = "You are a legal relevance grader."
    user = f"""Query ({domain_label} only): {sub_q}

Retrieved provisions:
{chunks}

Grade ONLY whether these provisions answer the {domain_label} part of the query.
A multi-part question may also ask about other legal areas (e.g. employment, tax, IP, corporate, data protection) — NEVER mark "insufficient" because other parts of the question belong to a different domain, and do not mention other legal areas in your reason.
All provisions are Indian statutes and the user is an Indian business. A person, contractor or work being
located abroad does NOT make Indian provisions irrelevant; judge whether the provision answers the question
under Indian law.
"sufficient": at least one provision directly addresses the {domain_label} part of the query.
"insufficient": none of the provisions address the {domain_label} part of the query."""
    return system, user


def generate_domain_prompt(domain_label: str, sub_q: str, chunks: str, key_ids: list | None = None) -> tuple[str, str]:
    key_rule = ""
    if key_ids:
        key_rule = (f"\n- A reviewer marked these provisions as needed for a complete answer: {', '.join(key_ids)}. "
                    "Write at least one step for each of them, stating what that provision itself requires, allows "
                    "or provides for this query. Leave one out only if its text, as given below, does not apply "
                    "to the query.")
    system = "You are a legal workflow assistant for Indian tech startups."
    user = f"""Query: {sub_q}
Legal area: {domain_label}

Using ONLY the provisions below, list the compliance actions relevant to this legal area for the query.
Do not invent sections or obligations. Address specifics in the query (company size, employee type, etc.).
For each step, say who the provision applies to ("private", "government", "both" or "unspecified"), based only on the text.

Rules:
- Discuss ONLY {domain_label}. Do not mention other legal areas (tax, data protection, etc.) in the summary.
- Include a step only if the provision directly responds to the query. Skip provisions that concern other kinds of
  workplaces (factories, mines, plantations, schools) or are unrelated to the specific question.
- If a provision states a default rule that answers the question (for example who is the first owner of a work,
  or what the default is when a contract is silent), INCLUDE it as a step and state both the default and its exception.
- A provision must answer the actual question asked. Do NOT include one merely because it shares a word with the
  query (for example "abroad" or "contractor") if it regulates a different subject (such as accident compensation
  or migrant-worker conditions). Leave it out.
- A provision that only gives the government or a regulator a power (to make rules, appoint or pay members,
  run its own proceedings) is not a duty of the user. Do not turn it into a step.
- Definitions may be used to say who a duty applies to, but prefer the provisions that state the duties themselves.
- Cite the provision that itself creates the duty, right, condition or consequence asked about. When rules,
  regulations or directions given below spell out how to carry out a duty created by an Act section that is also
  given below, write one step citing the Act section (the duty) and another citing the rule (the procedure,
  timeline or form). Do not cite only the rule.
- When one provision states a general rule and another covers the specific situation in the query, cite the
  specific one, and also the general one if the answer depends on it.
- If several provisions each answer a different part of the query, write a step for each of them.{key_rule}
- If the query asks what a law is, base the summary on what the provisions say about its scope and purpose.
- Provisions marked NOT YET IN FORCE still count: use them when they answer the query, because the user has to
  prepare for them. Do not write commencement dates into the action text; the system adds them to each step.
- If nothing in the provisions answers part of the query, say so in "not_covered" instead of padding the steps.

Provisions:
{chunks}

"""
    return system, user
