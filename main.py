import json
import sys
from pathlib import Path

import legal_workflow_generator.rag as rag


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} [setup|ingest [condensed|complete]|embed|extract-keywords]")
        exit(1)

    if sys.argv[1] == "setup":
        rag.setup.run()

    elif sys.argv[1] == "ingest":
        dataset_type = sys.argv[2] if len(sys.argv) > 2 else "condensed"

        if dataset_type == "complete":
            DATASET_FILE = Path("./datasets/complete/combined_complete_dataset.json")
        elif dataset_type == "condensed":
            DATASET_FILE = Path("./datasets/condensed/combined_condensed_dataset.json")
        else:
            print(f"Unknown dataset type: {dataset_type}")
            print(f"Usage: {sys.argv[0]} ingest [condensed|complete]")
            exit(1)

        if not DATASET_FILE.is_file():
            print(
                f"Could not find {DATASET_FILE}. Make sure you have generated it first."
            )
            exit(1)

        with open(DATASET_FILE, "r") as f:
            dataset = json.load(f)

        rag.ingestion.ingest(dataset)

    elif sys.argv[1] == "embed":
        rag.embeddings.run()

    elif sys.argv[1] == "extract-keywords":
        rag.domain_keywords.run()

    else:
        print(f"Unknown command: {sys.argv[1]}")
        print(f"Usage: {sys.argv[0]} [setup|ingest|embed|extract-keywords]")
        exit(1)


if __name__ == "__main__":
    main()

