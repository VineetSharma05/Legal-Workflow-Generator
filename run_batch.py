"""
Run a batch of queries from a text file through the agent graph.

Usage:
    python run_batch.py queries.txt

Each non-empty line in the file is treated as one query.
Full output (answer + trace) for every query is written to batch_results.txt.
A short summary table is printed to the terminal at the end.
"""

import sys
import time
from legal_workflow_generator.agent.graph import graph


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_batch.py queries.txt")
        sys.exit(1)

    path = sys.argv[1]
    with open(path, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(queries)} queries from {path}\n")

    summary_rows = []
    out_path = "batch_results.txt"

    with open(out_path, "w", encoding="utf-8") as out:
        for i, query in enumerate(queries, start=1):
            print(f"[{i}/{len(queries)}] Running: {query[:70]}...")
            start = time.time()
            try:
                result = graph.invoke({"query": query})
                elapsed = time.time() - start
                error = None
            except Exception as e:
                elapsed = time.time() - start
                result = {}
                error = str(e)

            out.write("=" * 100 + "\n")
            out.write(f"QUERY {i}: {query}\n")
            out.write("=" * 100 + "\n")

            if error:
                out.write(f"ERROR: {error}\n\n")
                summary_rows.append((i, query, "ERROR", "-", elapsed))
            else:
                out.write("\n--- ANSWER ---\n")
                out.write(str(result.get("answer", "<none>")) + "\n")

                out.write("\n--- ABSTAINED ---\n")
                out.write(str(result.get("abstain")) + "\n")

                out.write("\n--- VERIFIED CITATIONS ---\n")
                out.write(str(result.get("verified_citations", [])) + "\n")

                out.write("\n--- FAILED CITATIONS ---\n")
                out.write(str(result.get("failed_citations", [])) + "\n")

                out.write("\n--- TRACE ---\n")
                for step in result.get("trace", []):
                    out.write(f" → {step}\n")

                out.write("\n\n")

                abstained = result.get("abstain", False)
                n_verified = len(result.get("verified_citations", []))
                n_failed = len(result.get("failed_citations", []))
                status = "ABSTAIN" if abstained else "ANSWERED"
                summary_rows.append(
                    (i, query, status, f"{n_verified}v/{n_failed}f", round(elapsed, 1))
                )

            print(f"    done in {elapsed:.1f}s")

    print(f"\nFull results written to {out_path}\n")

    print("=" * 100)
    print(f"{'#':<3} {'Status':<10} {'Citations':<12} {'Time':<6} Query")
    print("=" * 100)
    for i, query, status, cit, elapsed in summary_rows:
        print(f"{i:<3} {status:<10} {cit:<12} {elapsed:<6} {query[:70]}")


if __name__ == "__main__":
    main()
