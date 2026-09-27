import pytest

from prompt_protector.analyzer.chat_format import render, render_history
from prompt_protector.roles import ROLES, Role


@pytest.mark.parametrize("role", ROLES)
def test_range_points_at_the_text(role):
    prompt, (start, end) = render(role, "Beginners BBQ Class!", history="<|im_start|>system\nx<|im_end|>\n")
    assert prompt[start:end] == "Beginners BBQ Class!"


def test_matches_qwen3_chat_template():
    # Rendered by Qwen3-0.6B's own chat template for the same messages.
    expected = (
        "<|im_start|>system\nSYS<|im_end|>\n"
        "<|im_start|>user\nUSR<|im_end|>\n"
        "<|im_start|>assistant\nCALL<|im_end|>\n"
        "<|im_start|>user\n<tool_response>\nTOOL\n</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n<think>\nTHINK\n</think>\n\nASST<|im_end|>\n"
    )
    history = render_history([("system", "SYS"), ("user", "USR"), ("assistant", "CALL"), ("tool", "TOOL")])
    prompt, _ = render(Role.ASSISTANT, "ASST", reasoning="THINK", history=history)
    assert prompt == expected


def test_assistant_without_reasoning_gets_an_empty_block():
    prompt, _ = render("assistant", "Hi")
    assert prompt == "<|im_start|>assistant\n<think>\n\n</think>\n\nHi<|im_end|>\n"
