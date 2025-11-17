from llm_runner import validation

def test_no_exec_tag():
    input_data = {
        "safety_tags": ["no-exec"],
        "allowed_backends": [],
        "tool_list": []
    }
    is_safety_tags,_ = validation(input_data)
    assert is_safety_tags is False

def test_safe_tag():
    input_data = {
        "safety_tags": ["safe"],
        "allowed_backends": [],
        "tool_list": []
    }
    is_safety_tags,_ = validation(input_data)
    assert is_safety_tags is True

def test_internal_only_with_external_backend():
    input_data = {
        "safety_tags": ["internal-only"],
        "allowed_backends": ["gemma3:7b"],  # backend externe
        "tool_list": []
    }
    is_safety_tags,_ = validation(input_data)
    assert is_safety_tags is False
