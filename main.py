import json
import sys
from pathlib import Path

import legal_workflow_generator.rag as rag


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} [setup|ingest]")
        exit(1)

    if sys.argv[1] == "setup":
        rag.setup.run()

    elif sys.argv[1] == "ingest":
        DATASET_FILE = Path("dataset.json")

        if not DATASET_FILE.is_file():
            print(
                f"Could not find {DATASET_FILE} in current folder. Make sure you have the {DATASET_FILE} in the current folder"
            )
            exit(1)

        with open(DATASET_FILE, "r") as f:
            dataset = json.load(f)

        rag.ingestion.ingest(dataset)


if __name__ == "__main__":
    main()
