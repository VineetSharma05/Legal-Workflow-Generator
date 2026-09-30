"""
Combines every per-domain dataset in datasets/complete/ into a single
datasets/complete/combined_complete_dataset.json, mirroring what
combined_condensed_dataset.json is for the condensed corpus.

The merge is a plain concatenation in filename order and it never edits the
per-domain files - those stay the source of truth. The output file itself is
skipped on re-runs, so the script is idempotent.

Duplicate provision_ids are reported but NOT silently dropped, because
`laws.provision_id` is the PRIMARY KEY in rag/setup.py and ingestion upserts
with ON CONFLICT (provision_id) DO UPDATE. Any duplicate therefore collapses
to one row at ingestion, last one winning, so the count here will exceed the
row count in Postgres. Use --dedup to resolve that in the file instead, or
--strict to fail the build on collisions.

    python datasets/combine_complete.py            # combine + report
    python datasets/combine_complete.py --dedup    # keep first of each id
    python datasets/combine_complete.py --strict   # exit 1 if ids collide
    python datasets/combine_complete.py --check    # report only, write nothing
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

OUTPUT_NAME = "combined_complete_dataset.json"

REQUIRED_KEYS = [
    "provision_id", "statute_id", "provision_type", "chapter", "chapter_title",
    "number", "title", "text", "sub_structure", "plain_english_summary",
    "keywords", "penalty_linked", "effective_date",
]


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    complete_dir = repo_root / "datasets" / "complete"
    out_path = complete_dir / OUTPUT_NAME

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dedup", action="store_true", help="keep only the first entry for each provision_id")
    ap.add_argument("--strict", action="store_true", help="exit non-zero if any provision_id collides")
    ap.add_argument("--check", action="store_true", help="report only, do not write the output file")
    args = ap.parse_args()

    sources = sorted(p for p in complete_dir.glob("*.json") if p.name != OUTPUT_NAME)
    if not sources:
        print(f"error: no per-domain datasets in {complete_dir}", file=sys.stderr)
        return 1

    combined: list[dict] = []
    # provision_id -> [source filenames], to attribute collisions to their files
    seen_in: dict[str, list[str]] = defaultdict(list)
    schema_problems: list[str] = []

    print(f"{len(sources)} source file(s) in {complete_dir.relative_to(repo_root)}/\n")

    for src in sources:
        provisions = json.load(src.open())

        for p in provisions:
            missing = [k for k in REQUIRED_KEYS if k not in p]
            if missing:
                schema_problems.append(
                    f"{src.name}: {p.get('provision_id', '<no provision_id>')} missing {missing}"
                )
            seen_in[p.get("provision_id")].append(src.name)

        combined.extend(provisions)

        statutes = Counter(p.get("statute_id") for p in provisions)
        print(f"  {src.name:<38} {len(provisions):>5} provisions, {len(statutes):>2} statute(s)")

    collisions = {pid: files for pid, files in seen_in.items() if len(files) > 1}

    # A collision is worse when it spans two domain files: the two entries are
    # then different provisions of different statutes fighting over one row.
    cross_file = {pid: files for pid, files in collisions.items() if len(set(files)) > 1}

    print(f"\n  {'TOTAL':<38} {len(combined):>5} provisions, {len(set(p['statute_id'] for p in combined)):>2} statutes")
    print(f"  {'unique provision_ids':<38} {len(seen_in):>5}")

    if schema_problems:
        print(f"\nschema problems ({len(schema_problems)}):")
        for s in schema_problems[:20]:
            print(f"  - {s}")
        if len(schema_problems) > 20:
            print(f"  ... and {len(schema_problems) - 20} more")

    if collisions:
        print(f"\nduplicate provision_id(s): {len(collisions)} id(s), "
              f"{len(combined) - len(seen_in)} entries in excess")
        by_file = Counter(f for files in collisions.values() for f in files)
        for fname, n in by_file.most_common():
            print(f"  {fname:<38} {n:>5} entries with a colliding id")
        if cross_file:
            print(f"  NOTE: {len(cross_file)} id(s) collide ACROSS files: "
                  f"{list(cross_file)[:5]}{' ...' if len(cross_file) > 5 else ''}")
        worst = sorted(collisions.items(), key=lambda kv: -len(kv[1]))[:5]
        print("  worst offenders: " + ", ".join(f"{pid} (x{len(f)})" for pid, f in worst))
        print("  laws.provision_id is a PRIMARY KEY, so ingestion keeps only the last of each.")

    if args.dedup:
        kept, first_seen = [], set()
        for p in combined:
            pid = p.get("provision_id")
            if pid in first_seen:
                continue
            first_seen.add(pid)
            kept.append(p)
        print(f"\n--dedup: {len(combined)} -> {len(kept)} provisions "
              f"({len(combined) - len(kept)} dropped)")
        combined = kept

    if args.check:
        print("\n--check: nothing written")
    else:
        with out_path.open("w") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"\nwrote {out_path.relative_to(repo_root)}  ({len(combined)} provisions)")

    if args.strict and collisions:
        print("\n--strict: failing because provision_ids collide", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
