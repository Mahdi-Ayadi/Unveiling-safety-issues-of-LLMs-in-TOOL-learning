import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from parser import load_all_cases, load_case, ParsedTest

TOOLS_DIR = Path(__file__).resolve().parents[1] / 'toolsword_cases'


def test_load_all_cases_type_and_count():
    cases = load_all_cases(TOOLS_DIR)
    assert isinstance(cases, list)
    assert all(isinstance(c, ParsedTest) for c in cases)
    # at least one case should be present
    assert len(cases) > 0


def test_input_stage_defaults_dialog_empty():
    # Find an Input-stage case and verify dialog defaulted to []
    cases = load_all_cases(TOOLS_DIR)
    found = False
    for c in cases:
        if c.stage.lower() == 'input':
            found = True
            assert isinstance(c.dialog, list)
            assert len(c.dialog) == 0
    assert found, "No Input-stage case found in toolsword_cases for this test"


def test_load_single_file_returns_parsedtest():
    # pick one file
    file = TOOLS_DIR / 'data_RC.json'
    parsed = load_case(file)
    assert isinstance(parsed, ParsedTest)
    assert parsed.id == 'data_RC.json'
    assert hasattr(parsed, 'stage')
    assert hasattr(parsed, 'tools')
    assert isinstance(parsed.tools, dict)
