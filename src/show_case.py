"""Interactive helper to display the raw JSON of a ToolSword case."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    base_dir = Path("toolsword_cases")
    file_name = input(f"Enter JSON filename (relative to {base_dir}): ").strip()
    file_path = base_dir / file_name if file_name else base_dir / "data_confict.json"

    try:
        index = int(input("Enter case index: ").strip())
    except ValueError:
        raise SystemExit("Index must be an integer.")

    if not file_path.exists():
        raise SystemExit(f"File not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        entry = data[index]
    except (IndexError, TypeError):
        raise SystemExit(f"No case at index {index} in {file_path}")

    print(json.dumps(entry, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
