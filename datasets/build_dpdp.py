"""
Builds datasets/complete/complete_dpdp.json by concatenating the reviewed
per-statute fragments in datasets/dpdp_parts/ in filename order.

Fragments are kept separate so each new source can be reviewed on its own
(and diffed on its own) before it lands in the combined file. The build is a
full rewrite every time, so it stays deterministic and re-runnable - never
append to complete_dpdp.json by hand.

    python datasets/build_dpdp.py           # build + validate
    python datasets/build_dpdp.py --check   # validate only, write nothing

Part 00 is the DPDP Act itself, seeded from condensed_dpdp.json with its ten
duplicate section 11-20 entries dropped.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REQUIRED_KEYS = [
    "provision_id", "statute_id", "provision_type", "chapter", "chapter_title",
    "number", "title", "text", "sub_structure", "plain_english_summary",
    "keywords", "penalty_linked", "effective_date",
]

# Keys a provision may carry beyond REQUIRED_KEYS; `part` is used by the
# employee-laws dataset for Acts divided into Parts rather than Chapters.
OPTIONAL_KEYS = ["part"]


def validate(provisions: list[dict]) -> list[str]:
    errors = []

    for i, p in enumerate(provisions):
        where = f"[{i}] {p.get('provision_id', '<no provision_id>')}"

        missing = [k for k in REQUIRED_KEYS if k not in p]
        if missing:
            errors.append(f"{where}: missing key(s) {missing}")

        extra = [k for k in p if k not in REQUIRED_KEYS and k not in OPTIONAL_KEYS]
        if extra:
            errors.append(f"{where}: unexpected key(s) {extra}")

        for k in ("provision_id", "statute_id", "provision_type", "number", "title", "text"):
            if not p.get(k):
                errors.append(f"{where}: '{k}' must be a non-empty string")

        if not isinstance(p.get("sub_structure"), dict):
            errors.append(f"{where}: 'sub_structure' must be an object")
        if not isinstance(p.get("keywords"), list):
            errors.append(f"{where}: 'keywords' must be a list")
        if not isinstance(p.get("penalty_linked"), bool):
            errors.append(f"{where}: 'penalty_linked' must be a boolean")

    dupes = {k: v for k, v in Counter(p.get("provision_id") for p in provisions).items() if v > 1}
    if dupes:
        errors.append(f"duplicate provision_id(s): {dupes}")

    return errors


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    parts_dir = repo_root / "datasets" / "dpdp_parts"
    out_path = repo_root / "datasets" / "complete" / "complete_dpdp.json"

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="validate only, do not write the output file")
    args = ap.parse_args()

    parts = sorted(parts_dir.glob("*.json"))
    if not parts:
        print(f"error: no fragments in {parts_dir}", file=sys.stderr)
        return 1

    combined: list[dict] = []
    for part in parts:
        provisions = json.load(part.open())
        combined.extend(provisions)
        by_statute = Counter(p.get("statute_id") for p in provisions)
        detail = ", ".join(f"{k} ({v})" for k, v in by_statute.items())
        print(f"  {part.name:<34} {len(provisions):>4}   {detail}")

    errors = validate(combined)
    if errors:
        print("\nvalidation FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print(f"\n  {'TOTAL':<34} {len(combined):>4}")
    for statute, n in Counter(p["statute_id"] for p in combined).most_common():
        types = Counter(p["provision_type"] for p in combined if p["statute_id"] == statute)
        print(f"    {statute:<28} {n:>4}   {dict(types)}")

    if args.check:
        print("\nvalidation passed (--check: nothing written)")
        return 0

    with out_path.open("w") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"\nwrote {out_path.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
