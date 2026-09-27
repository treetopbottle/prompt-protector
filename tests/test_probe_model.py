"""Role Analyzer rules RoleAnalyzer.01–04 with the real model. Run with `uv run pytest -m model`."""

import json
import socket
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from prompt_protector.analyzer.probe import ProbeAnalyzer
from prompt_protector.roles import ROLES

pytestmark = pytest.mark.model

CASES_DIR = Path(__file__).parent.parent / "eval" / "cases"
THRESHOLD = 0.5  # no policy defaults yet


def case(case_id):
    for path in CASES_DIR.glob("*.json"):
        for c in json.loads(path.read_text(encoding="utf-8")):
            if c["id"] == case_id:
                return c
    raise KeyError(case_id)


def segment(case_id, index):
    seg = case(case_id)["segments"][index]
    return seg["text"], seg["role"]


def injected_span(case_id):
    c = case(case_id)
    finding = c["expected"]["findings"][0]
    return finding["span"], c["segments"][finding["segment"]]["role"]


def no_network(*args, **kwargs):
    raise AssertionError("the Role Analyzer tried to use the network")


@pytest.fixture(scope="module")
def analyzer():
    # Loading happens with the network blocked too.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(socket.socket, "connect", no_network)
        yield ProbeAnalyzer()


@pytest.mark.parametrize(
    "case_id, index",
    [("ben-002", 3), ("ben-020", 2), ("hn-014", 2)],
    ids=["search results", "calendar API", "git log with 'ignore previous'"],
)
def test_01_plain_tool_output_scores_highest_as_tool(analyzer, case_id, index):
    text, role = segment(case_id, index)
    assert analyzer.score(text, role).top() == "tool"


def test_02_instructions_in_tool_output_score_as_system_or_user(analyzer):
    score = analyzer.score(*injected_span("ind-001"))
    assert max(score.system, score.user) > THRESHOLD


@pytest.mark.parametrize("case_id", ["dir-003", "dir-010"], ids=["CoT forgery", "fake <think> block"])
def test_03_chain_of_thought_in_user_segment_scores_as_reasoning(analyzer, case_id):
    text, role = segment(case_id, 1)
    assert analyzer.score(text, role).reasoning > THRESHOLD


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(text=st.one_of(st.just(""), st.just(" \n\t"), st.text(max_size=300)), role=st.sampled_from(ROLES))
def test_04_probabilities_sum_to_one(analyzer, text, role):
    score = analyzer.score(text, role)
    assert abs(sum(score.to_dict().values()) - 1) <= 0.01


def test_04_scoring_uses_no_network(analyzer, monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", no_network)
    analyzer.score("Rotterdam has about 670,000 inhabitants.", "tool")
