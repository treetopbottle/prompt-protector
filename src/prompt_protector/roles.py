"""The roles a segment of a prompt can have."""

from enum import StrEnum


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    TOOL = "tool"
    ASSISTANT = "assistant"
    REASONING = "reasoning"


ROLES = tuple(Role)
