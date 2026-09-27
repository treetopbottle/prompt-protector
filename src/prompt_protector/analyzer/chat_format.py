"""Qwen3's chat format, one message at a time, as the role probe sees it in training and scoring."""

from ..roles import Role

# Opening and closing text around a message's content.
OPEN = {
    Role.SYSTEM: "<|im_start|>system\n",
    Role.USER: "<|im_start|>user\n",
    Role.TOOL: "<|im_start|>user\n<tool_response>\n",
    Role.REASONING: "<|im_start|>assistant\n<think>\n",
    Role.ASSISTANT: "<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n",
}
CLOSE = {
    Role.SYSTEM: "<|im_end|>\n",
    Role.USER: "<|im_end|>\n",
    Role.TOOL: "\n</tool_response><|im_end|>\n",
    Role.REASONING: "\n</think>\n\n",
    Role.ASSISTANT: "<|im_end|>\n",
}


def render(role: Role | str, text: str, reasoning: str = "", history: str = "") -> tuple[str, tuple[int, int]]:
    """Render `text` as one message, after `history`; return the prompt and the character range of `text`.

    An assistant message follows a reasoning block, empty unless `reasoning` is given.
    """
    role = Role(role)
    opening = history + OPEN[role].format(reasoning=reasoning)
    return opening + text + CLOSE[role], (len(opening), len(opening) + len(text))


def render_history(messages: list[tuple[Role | str, str]]) -> str:
    """Earlier turns as Qwen3 renders them; assistant turns in history have no reasoning block."""
    out = ""
    for role, text in messages:
        role = Role(role)
        if role == Role.ASSISTANT:
            out += f"<|im_start|>assistant\n{text}<|im_end|>\n"
        else:
            out += OPEN[role] + text + CLOSE[role]
    return out
