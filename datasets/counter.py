import json
from pathlib import Path

root = Path("datasets/")

groups = {}
for folder in sorted(p for p in root.iterdir() if p.is_dir()):
    rows = []
    for f in sorted(folder.glob("*.json")):
        try:
            n = len(json.load(f.open()))
        except Exception as e:
            n = f"error: {type(e).__name__}"
        rows.append((f.name, n))
    groups[folder] = rows

for folder, rows in groups.items():
    print(f"{folder.name}/")

    individuals = [(name, n) for name, n in rows if not name.startswith("combined_")]
    combined = [(name, n) for name, n in rows if name.startswith("combined_")]

    if not combined:
        combined = [(f"combined_{folder.name}_dataset.json", "to_be_compiled")]

    width = max(len(name) for name, _ in individuals + combined)

    for name, n in individuals:
        print(f"  {name:<{width}}  {n:>6}")

    print(f"  {'':<{width}}  {'─' * 6}")

    for name, n in combined:
        print(f"  {name:<{width}}  {n:>6}")

    print()