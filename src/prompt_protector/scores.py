"""Role Score: the probability per role that a model perceives a text as that role."""

import json
from dataclasses import dataclass, fields

from .roles import ROLES, Role

# RoleAnalyzer.04: the probabilities sum to 1 (± 0.01).
SUM_TOLERANCE = 0.01


@dataclass(frozen=True)
class RoleScore:
    system: float
    user: float
    tool: float
    assistant: float
    reasoning: float

    def __post_init__(self):
        values = [getattr(self, f.name) for f in fields(self)]
        if any(not 0 <= v <= 1 for v in values):
            raise ValueError(f"probabilities must be between 0 and 1: {values}")
        total = sum(values)
        if abs(total - 1) > SUM_TOLERANCE + 1e-9:
            raise ValueError(f"probabilities sum to {total:.4f}, not 1 (± {SUM_TOLERANCE})")

    @classmethod
    def from_dict(cls, probabilities: dict[str, float]) -> "RoleScore":
        return cls(**{role.value: float(probabilities[role]) for role in ROLES})

    def __getitem__(self, role: Role | str) -> float:
        return getattr(self, Role(role).value)

    def to_dict(self) -> dict[str, float]:
        return {role.value: self[role] for role in ROLES}

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def confusion(self, declared: Role | str) -> float:
        """How little the text sounds like the role it is labeled as: 1 − P(declared)."""
        return 1 - self[declared]

    def top(self, exclude: Role | str | None = None) -> Role:
        """The highest scoring role, optionally other than `exclude` (the Report's `perceived`)."""
        candidates = [role for role in ROLES if role != exclude]
        return max(candidates, key=self.__getitem__)
