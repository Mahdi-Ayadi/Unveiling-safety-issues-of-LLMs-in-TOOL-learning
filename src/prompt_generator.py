"""Minimal LangChain runner for ToolSword attack scenarios."""

from __future__ import annotations

import json
from typing import Any, Iterable, List, Mapping

from .parser import ParsedTest, ToolSwordParser
from textwrap import indent

# from langchain_core.language_models import BaseLanguageModel

def _format_tools(tools: Any) -> str:
    """
    Render tools in a natural, deployment-like way.

    `tools` can be:
    - a dict {name: meta_dict}
    - a list of dicts [{"name": ..., "description": ..., "parameters": ...}, ...]
    """
    if not tools:
        return ""

    # Normalize to list of dicts
    if isinstance(tools, Mapping):
        tool_list = []
        for name, meta in tools.items():
            if not isinstance(meta, dict):
                meta = {}
            tool_entry = {"name": name}
            tool_entry.update(meta)
            tool_list.append(tool_entry)
    else:
        tool_list = [t for t in tools if isinstance(t, dict)]

    lines: List[str] = []
    lines.append("You can use the following tools to help the user:")

    for tool in tool_list:
        name = tool.get("name", "")
        desc = tool.get("description") or tool.get("desc") or ""
        params = tool.get("parameters")

        # One-line summary
        if desc:
            lines.append(f"- {name}: {desc}")
        else:
            lines.append(f"- {name}")

        # Show JSON schema indented underneath
        if params:
            params_json = json.dumps(params, indent=2, ensure_ascii=False)
            lines.append("  Arguments (JSON schema):")
            lines.append(indent(params_json, "    "))

    return "\n".join(lines).strip()


def _format_dialog(dialog: Iterable[dict]) -> str:
    if not dialog:
        return ""
    turns = []
    for turn in dialog:
        role = (turn.get("role") or "user").capitalize()
        content = turn.get("content") or ""
        turns.append(f"{role}: {content}".strip())
    return "Conversation so far:\n" + "\n".join(turns)


def build_attack_prompt(case: ParsedTest) -> str:
    """
    Build a realistic prompt for a tool-using assistant,
    without mentioning attacks/evaluation/stage/scenario.
    """
    sections: List[str] = []

    # 1) Tools first: like a system-level description of available tools
    tools_block = _format_tools(case.tools_list or case.tools)
    if tools_block:
        sections.append(tools_block)
        sections.append(
            "\nWhen it helps, call one of these tools with appropriate arguments. "
            "Otherwise, answer directly."
        )

    # 2) Previous conversation, if any
    dialog_block = _format_dialog(case.dialog)
    if dialog_block:
        sections.append("")  # blank line
        sections.append(dialog_block)

    # 3) Current user query
    sections.append("")  # blank line
    sections.append("User:")
    sections.append(case.query)

    return "\n\n".join(sections).strip()


if __name__ == "__main__":
    import random
    from pathlib import Path

    base_dir = Path("toolsword_cases")
    parser = ToolSwordParser(base_dir)

    mode = input("Pick mode: (1) random case from specific file, (2) random case from all files: ").strip()

    if mode == "1":
        file_input = input(f"Enter JSON filename (relative to {base_dir}): ").strip()
        file_path = (base_dir / file_input) if file_input else base_dir / "data_confict.json"
        cases = list(parser.iter_file_cases(file_path))
        if not cases:
            raise SystemExit(f"No cases found in {file_path}")
        case = random.choice(cases)
    else:
        cases = list(parser.iter_all_cases())
        if not cases:
            raise SystemExit("No cases found in directory.")
        case = random.choice(cases)

    print(f"\nCase picked: {case.id}")
    print("-" * 40)
    print(build_attack_prompt(case))
