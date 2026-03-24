import json
import sys
from pathlib import Path

import legal_workflow_generator.rag as rag


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} [setup|ingest|embed]")
        exit(1)

    if sys.argv[1] == "setup":
        rag.setup.run()

    elif sys.argv[1] == "ingest":
        DATASET_FILE = Path("combined_dataset.json")

        if not DATASET_FILE.is_file():
            print(
                f"Could not find {DATASET_FILE} in current folder. Make sure you have the {DATASET_FILE} in the current folder"
            )
            exit(1)

        with open(DATASET_FILE, "r") as f:
            dataset = json.load(f)

        rag.ingestion.ingest(dataset)

    elif sys.argv[1] == "embed":
        rag.embeddings.run()

    else:
        print(f"Unknown command: {sys.argv[1]}")
        print(f"Usage: {sys.argv[0]} [setup|ingest|embed]")
        exit(1)


if __name__ == "__main__":
    main()