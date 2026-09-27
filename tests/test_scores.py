import json

import pytest

from prompt_protector.analyzer import load_analyzer
from prompt_protector.roles import ROLES, Role
from prompt_protector.scores import RoleScore

# The Role Score example from SPECIFICATION.md.
SPEC_EXAMPLE = {"system": 0.05, "user": 0.10, "tool": 0.80, "assistant": 0.02, "reasoning": 0.03}


def score(**overrides):
    return RoleScore.from_dict(SPEC_EXAMPLE | overrides)


def test_roles_match_the_spec():
    assert [r.value for r in ROLES] == ["system", "user", "tool", "assistant", "reasoning"]


def test_accepts_sum_within_tolerance():
    assert score(tool=0.805).tool == 0.805  # sums to 1.005


@pytest.mark.parametrize("tool", [0.70, 0.90])  # sums to 0.9 and 1.1
def test_rejects_sum_outside_tolerance(tool):
    with pytest.raises(ValueError, match="sum"):
        score(tool=tool)


def test_rejects_negative_probability():
    with pytest.raises(ValueError, match="between 0 and 1"):
        score(user=-0.05, tool=0.95)


def test_json_matches_the_spec_schema():
    data = json.loads(score().to_json())
    assert data == SPEC_EXAMPLE
    assert list(data) == [r.value for r in ROLES]


def test_lookup_by_role_or_name():
    s = score()
    assert s[Role.TOOL] == s["tool"] == 0.80


def test_confusion_is_one_minus_declared():
    assert score().confusion("tool") == pytest.approx(0.20)
    assert score().confusion(Role.SYSTEM) == pytest.approx(0.95)


def test_top_and_top_other_than_declared():
    s = score()
    assert s.top() == Role.TOOL
    assert s.top(exclude="tool") == Role.USER


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="unknown analyzer backend"):
        load_analyzer({"backend": "crystal_ball"})
