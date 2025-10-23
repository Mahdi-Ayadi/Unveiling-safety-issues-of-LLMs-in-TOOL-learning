# src/parser.py
"""Toolsword JSON parser.

Responsibilities:
- Load individual JSON files or a whole directory of cases (toolsword_cases/)
- Validate minimal schema (top-level keys + types)
- Normalize/sanitize dialog turns
- Return strongly-typed ParsedTest objects for downstream use

Usage:
    from src.parser import load_case, load_all_cases
    tests = load_all_cases("toolsword_cases")
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import re

logger = logging.getLogger(__name__)

# -------------------------
# Models
# -------------------------
@dataclass
class ParsedTest:
    id: str # filename or derived id
    stage: str
    scenario: str
    query: str
    tools: Dict[str, Dict[str, Any]]
    dialog: List[Dict[str, Any]]
    raw: Dict[str, Any] = field(default_factory=dict)

# -------------------------
# Helpers
# -------------------------
REQUIRED_TOP_KEYS = {"stage", "scenario", "query"}

def _sanitize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    # strip non-printable controls except newline/tab, collapse repeated whitespace
    s = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def _validate_entry_schema(entry: Dict[str, Any]) -> Optional[str]:
    """Stage-aware validation.

    Minimal rules:
    - Always require: stage, scenario, query.
    - For Execution and Output stages, require tools and dialog.
    - For other stages (e.g. Input), be permissive: dialog/tools optional.
    """
    missing = REQUIRED_TOP_KEYS - set(entry.keys())
    if missing:
        return f"missing top-level keys: {sorted(list(missing))}"

    stage = (entry.get("stage") or "").strip().lower()
    if stage in ("execution", "output"):
        # Be permissive: some files omit dialog or tools even in execution/output
        # stages. If present, ensure correct types; otherwise parser will default.
        if "tools" in entry and not (isinstance(entry.get("tools"), list) or isinstance(entry.get("tools"), dict)):
            return "tools must be a list or dict"
        if "dialog" in entry and not isinstance(entry.get("dialog"), list):
            return "dialog must be a list"

    else:
        # Input/unknown stages: be lenient but check types if present
        if "tools" in entry and not (isinstance(entry.get("tools"), list) or isinstance(entry.get("tools"), dict)):
            return "tools must be a list or dict"
        if "dialog" in entry and not isinstance(entry.get("dialog"), list):
            return "dialog must be a list"

    return None

# -------------------------
# Core parsing
# -------------------------
def parse_entry(entry: Dict[str, Any], id_hint: str = "") -> ParsedTest:
    """Validate and convert a raw JSON entry into ParsedTest."""
    err = _validate_entry_schema(entry)
    if err:
        raise ValueError(f"Invalid test entry ({id_hint}): {err}")

    # Normalize tools: accept missing, list, or dict
    tools_map: Dict[str, Dict[str, Any]] = {}
    raw_tools = entry.get("tools", [])
    if isinstance(raw_tools, dict):
        tools_iter = [raw_tools]
    elif isinstance(raw_tools, list):
        tools_iter = raw_tools
    else:
        tools_iter = []

    for t in tools_iter:
        if not isinstance(t, dict) or "name" not in t:
            logger.warning("Skipping malformed tool entry in %s: %r", id_hint, t)
            continue
        tools_map[t["name"]] = t

    # Default dialog to empty list if missing (Input-stage cases often have no dialog)
    dialog_clean: List[Dict[str, Any]] = []
    raw_dialog = entry.get("dialog", []) or []
    for turn in raw_dialog:
        if not isinstance(turn, dict):
            logger.warning("Skipping non-dict dialog turn in %s: %r", id_hint, turn)
            continue
        turn_copy = dict(turn)  # shallow copy
        if "content" in turn_copy:
            turn_copy["content"] = _sanitize_text(turn_copy.get("content", ""))
        # normalize role field (prefer 'role' and fallback to 'name' for function turns)
        role = turn_copy.get("role") or ("function" if "name" in turn_copy and turn_copy.get("role") is None else None)
        if role:
            turn_copy["role"] = role
        dialog_clean.append(turn_copy)

    parsed = ParsedTest(
        id=id_hint or "",
        stage=_sanitize_text(entry.get("stage", "")),
        scenario=_sanitize_text(entry.get("scenario", "")),
        query=_sanitize_text(entry.get("query", "")),
        tools=tools_map,
        dialog=dialog_clean,
        raw=entry
    )
    return parsed

# -------------------------
# File / directory loaders
# -------------------------
def load_case(path: str | Path) -> ParsedTest:
    """Load and parse a single JSON file. Returns ParsedTest."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    if p.is_dir():
        raise IsADirectoryError(f"Expected file, got directory: {p}")

    raw = json.loads(p.read_text(encoding="utf-8"))
    # If the file holds a list of cases, pick the first (common in Toolsword exports).
    if isinstance(raw, list):
        if not raw:
            raise ValueError(f"No cases in file: {p}")
        entry = raw[0]
    elif isinstance(raw, dict):
        entry = raw
    else:
        raise ValueError(f"Unexpected JSON root type in {p}: {type(raw)}")

    return parse_entry(entry, id_hint=p.name)

def load_all_cases(directory: str | Path, pattern: str = "data_*.json") -> List[ParsedTest]:
    """Load and parse all JSON files matching pattern in the directory."""
    d = Path(directory)
    if not d.exists() or not d.is_dir():
        raise NotADirectoryError(f"Directory not found: {directory}")
    tests: List[ParsedTest] = []
    for file in sorted(d.glob(pattern)):
        try:
            parsed = load_case(file)
            tests.append(parsed)
        except Exception as exc:
            logger.exception("Failed to load case %s: %s", file, exc)
    return tests

# -------------------------
# CLI convenience
# -------------------------
if __name__ == "__main__":  # pragma: no cover
    import argparse, pprint
    ap = argparse.ArgumentParser(description="Parse Toolsword JSON cases")
    ap.add_argument("path", help="file or directory of cases (directory uses data_*.json pattern)")
    args = ap.parse_args()
    p = Path(args.path)
    if p.is_dir():
        parsed = load_all_cases(p)
        print(f"Loaded {len(parsed)} cases from {p}")
        pprint.pprint([{"id": t.id, "stage": t.stage, "scenario": t.scenario, "query": t.query} for t in parsed])
    else:
        parsed = load_case(p)
        print("Loaded single case:")
        pprint.pprint({"id": parsed.id, "stage": parsed.stage, "scenario": parsed.scenario, "query": parsed.query})
