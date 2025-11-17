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
    tools: Dict[str, Any]
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

    def iter_all_cases(self, pattern="data_*.json", shuffle=False):
        """Yield all cases from all files in the directory, optionally shuffled."""
        files = sorted(self.directory.glob(pattern))
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
        tools = {}
        for t in raw_tools:
            if isinstance(t, dict) and "name" in t:
                tools[t["name"]] = t

        return ParsedTest(
            id=f"{file_stem}_{index}",
            stage=entry.get("stage", "").lower(),
            scenario=entry.get("scenario", ""),
            query=entry.get("query", ""),
            tools=tools,
            dialog=entry.get("dialog", []) or [],
            raw=entry,
        )



if __name__ == "__main__":
    import random, json
    from pathlib import Path
    from pprint import pprint

    parser = ToolSwordParser("toolsword_cases")

    mode = input("Pick mode: (1) random from all cases, (2) random from random file: ")

    if mode.strip() == "1":
        # Mode 1: random case from all cases
        all_cases = list(parser.iter_all_cases())
        if not all_cases:
            print("No cases found!")
            exit(1)
        case = random.choice(all_cases)

    elif mode.strip() == "2":
        # Mode 2: pick a random file and a random index
        files = list(Path("toolsword_cases").glob("data_*.json"))
        if not files:
            print("No JSON files found!")
            exit(1)
        file_choice = random.choice(files)
        with file_choice.open("r", encoding="utf-8") as f:
            data = json.load(f)
        index_choice = random.randint(0, len(data)-1)
        case = parser.get_case(file_choice, index_choice)

    else:
        print("Invalid mode!")
        exit(1)

    # Compact print
    print(f"\nCase ID: {case.id}")
    print(f"Stage: {case.stage}, Scenario: {case.scenario}")
    print(f"Query: {case.query}")
    print(f"Dialog turns: {len(case.dialog)}")
    print("Tools:", list(case.tools.keys()))

    expand = input("\nShow full tools details? [y/N]: ").lower()
    if expand == "y":
        pprint(case.tools)

    show_raw = input("\nShow raw JSON? [y/N]: ").lower()
    if show_raw == "y":
        print(json.dumps(case.raw, indent=2, ensure_ascii=False))
        