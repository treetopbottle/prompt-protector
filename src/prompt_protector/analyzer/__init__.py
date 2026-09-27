"""Role Analyzer: scores a text with a local model and returns a Role Score."""

from collections.abc import Mapping
from typing import Any, Protocol

from ..scores import RoleScore


class RoleAnalyzer(Protocol):
    def score(self, text: str) -> RoleScore: ...


def load_analyzer(config: Mapping[str, Any] | None = None) -> RoleAnalyzer:
    """Create the analyzer named by `config["backend"]`; the other keys go to the backend."""
    options = dict(config or {})
    backend = options.pop("backend", "zero_shot")
    if backend == "zero_shot":
        from .zero_shot import ZeroShotAnalyzer

        return ZeroShotAnalyzer(**options)
    raise ValueError(f"unknown analyzer backend '{backend}'")
