"""Role Analyzer: scores a text with a local model and returns a Role Score."""

from collections.abc import Mapping
from typing import Any, Protocol

from ..roles import Role
from ..scores import RoleScore


class RoleAnalyzer(Protocol):
    def score(self, text: str, role: Role | str) -> RoleScore:
        """How the model perceives `text` when it appears under its declared `role`."""
        ...


def load_analyzer(config: Mapping[str, Any] | None = None) -> RoleAnalyzer:
    """Create the analyzer named by `config["backend"]`; the other keys go to the backend."""
    options = dict(config or {})
    backend = options.pop("backend", "probe")
    if backend == "probe":
        from .probe import ProbeAnalyzer

        return ProbeAnalyzer(**options)
    raise ValueError(f"unknown analyzer backend '{backend}'")
