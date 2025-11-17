# src/parser.py
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

@dataclass
class ParsedTest:
    id: str
    stage: str
    scenario: str
    query: str
    # Mapping name -> tool dict (convenient for lookups)
    tools: Dict[str, Any]
    # Richest representation: original list of tool dicts, in order
    tools_list: List[Dict[str, Any]]
    dialog: List[Dict[str, Any]]
    raw: Dict[str, Any]


class ToolSwordParser:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def iter_file_cases(self, file_path: str | Path):
        """Yield all cases from a single JSON file."""
        file_path = Path(file_path)
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for i, entry in enumerate(data):
            yield self._parse_entry(entry, file_path.stem, i)

    def iter_all_cases(self, pattern: str = "data_*.json", shuffle: bool = False):
        """Yield all cases from all files in the directory, optionally shuffled."""
        files = sorted(self.directory.glob(pattern))
        # NOTE: shuffle param is unused for now; you can add random.shuffle(files) if needed
        for file in files:
            for case in self.iter_file_cases(file):
                yield case

    def get_case(self, file_path: str | Path, index: int) -> ParsedTest:
        """Return a single case by file + index without scanning all cases."""
        file_path = Path(file_path)
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        try:
            entry = data[index]
        except IndexError:
            raise IndexError(f"File {file_path} has no case at index {index}")

        return self._parse_entry(entry, file_path.stem, index)

    def _parse_entry(self, entry: dict, file_stem: str, index: int) -> ParsedTest:
        """Convert a raw JSON entry into ParsedTest."""
        raw_tools = entry.get("tools", []) or []

        # Richest representation: keep the list as-is (filtered to dicts)
        tools_list: List[Dict[str, Any]] = [
            t for t in raw_tools if isinstance(t, dict)
        ]

        # Convenience mapping: name -> tool dict
        tools_by_name: Dict[str, Any] = {}
        for t in tools_list:
            name = t.get("name")
            if isinstance(name, str):
                # If there are duplicates, the last wins – but tools_list keeps them all
                tools_by_name[name] = t

        return ParsedTest(
            id=f"{file_stem}_{index}",
            stage=entry.get("stage", "").lower(),
            scenario=entry.get("scenario", ""),
            query=entry.get("query", ""),
            tools=tools_by_name,
            tools_list=tools_list,
            dialog=entry.get("dialog", []) or [],
            raw=entry,
        )


if __name__ == "__main__":
    import random
    from pprint import pprint

    parser = ToolSwordParser("toolsword_cases")

    mode = input(
        "Pick mode: (1) random from all cases, (2) random from random file, "
        "(3) select file + index: "
    ).strip()

    if mode == "1":
        # Mode 1: random case from all cases
        all_cases = list(parser.iter_all_cases())
        if not all_cases:
            print("No cases found!")
            exit(1)
        case = random.choice(all_cases)

    elif mode == "2":
        # Mode 2: pick a random file and a random index
        files = list(Path("toolsword_cases").glob("data_*.json"))
        if not files:
            print("No JSON files found!")
            exit(1)
        file_choice = random.choice(files)
        with file_choice.open("r", encoding="utf-8") as f:
            data = json.load(f)
        index_choice = random.randint(0, len(data) - 1)
        case = parser.get_case(file_choice, index_choice)

    elif mode == "3":
        file_name = input("Enter JSON filename (relative to toolsword_cases): ").strip()
        file_path = Path("toolsword_cases") / file_name
        try:
            index_choice = int(input("Enter case index: ").strip())
        except ValueError:
            print("Index must be an integer.")
            exit(1)
        try:
            case = parser.get_case(file_path, index_choice)
        except Exception as exc:  # pragma: no cover - interactive helper
            print(f"Failed to fetch case: {exc}")
            exit(1)

    else:
        print("Invalid mode!")
        exit(1)

    # Compact summary
    print(f"\nCase ID: {case.id}")
    print(f"Stage: {case.stage}, Scenario: {case.scenario}")
    print(f"Query: {case.query}")
    print(f"Dialog turns: {len(case.dialog)}")
    print("Tool names:", list(case.tools.keys()))

    expand = input("\nShow full tools details (rich list)? [y/N]: ").lower()
    if expand == "y":
        print("\nFull tools_list (as in JSON):")
        print(json.dumps(case.tools_list, indent=2, ensure_ascii=False))

    show_raw = input("\nShow raw JSON entry? [y/N]: ").lower()
    if show_raw == "y":
        print(json.dumps(case.raw, indent=2, ensure_ascii=False))
