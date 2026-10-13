# Expert review rubric: workflow correctness and practical usefulness

Automated scores (domain recall, citation hit rate) check that the pipeline found the right
statutes. They do not check whether a lawyer would accept the advice. This rubric is for that.

Reviewers are qualified Indian lawyers (or supervised paralegals who report to one). They see
each workflow **blind**: no pipeline grades, no settings, no retrieval scores. They score the
text, and they can open the cited provision text in the sheet to check it.

## Correctness (score 1 to 5 each)

| Code | Criterion | 1 | 3 | 5 |
|---|---|---|---|---|
| C1 | **Citation faithfulness**: does each step say what its cited provision says? | Most steps misstate the provision | Some steps overreach or add detail the text lacks | Every step is faithful to its cited provision |
| C2 | **Applicability**: right law, right party (private company, government, both), right trigger | Wrong law or wrong party for most steps | One or two steps apply to the wrong party or situation | Every step applies to the situation as described |
| C3 | **Completeness**: are the mandatory obligations for the question covered? | A core obligation or penalty is missing | Minor obligations missing, or a required step is only partly covered | Nothing material is missing for the question asked |
| C4 | **No invented obligations**: does the workflow avoid steps with no basis in Indian law? | Several invented or incorrect obligations | One questionable step | No invented obligations |
| C5 | **Temporal validity**: is the advice valid on the stated date? Are repealed or future provisions flagged? | Relies on a repealed or not-yet-in-force provision without a note | Flags are present but incomplete | Dates and transitions are handled correctly |

## Usefulness (score 1 to 5 each)

| Code | Criterion | 1 | 3 | 5 |
|---|---|---|---|---|
| U1 | **Actionability**: could a startup founder act on each step without further research? | Vague; no concrete action | Some concrete actions, some vague | Each step is a concrete, owner-able action |
| U2 | **Clarity and ordering**: is the checklist readable and prioritised? | Hard to follow | Readable but unordered | Clear, sensible ordering by urgency or dependency |
| U3 | **Honest gaps**: when the knowledge base does not cover part of the question, is that said clearly? | Gaps hidden or the answer pretends to be complete | Gaps mentioned but unclear | Gaps stated precisely with a pointer to a professional |

## Overall

| Code | Question | Values |
|---|---|---|
| O1 | Would you send this to a client with light edits? | `yes` / `with_edits` / `no` |
| O2 | Does it contain a **critical error** (a wrong obligation, deadline, penalty or rate that could cause the client to break the law)? | `Y` / `N` |
| comment | Anything the scores do not capture (name the step id if relevant) | free text |

A `critical error` (O2 = Y) is reported separately and cannot be offset by high scores elsewhere.

## Procedure

1. Build the sheet: `python evals/expert_review/build_review_sheet.py --results eval_results.jsonl --out review/`.
2. Each reviewer fills only their own CSV. Cases marked `double=yes` are scored by two reviewers, which measures agreement.
3. Score each row independently and do not discuss rows before submitting.
4. Aggregate: `python evals/expert_review/score_reviews.py review/reviewer_*.csv --out review/summary.md`.
5. Report the mean for each criterion, the critical-error rate with its count, and the agreement on double-scored rows.
   Treat agreement below about 0.6 (linear weighted kappa) as a sign that the rubric needs clarifying before the scores are trusted.
