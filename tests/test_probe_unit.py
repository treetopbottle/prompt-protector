"""Probe backend tests that need no model."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from prompt_protector.analyzer import probe
from prompt_protector.analyzer.probe import ProbeAnalyzer, content_mask, role_probabilities
from prompt_protector.roles import ROLES


class CharTokenizer:
    """One token per character."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def bare_analyzer(max_tokens=2048):
    analyzer = object.__new__(ProbeAnalyzer)
    analyzer.tok, analyzer.decoder, analyzer.max_tokens = CharTokenizer(), None, max_tokens
    analyzer.w, analyzer.b = np.zeros((5, 4), np.float32), np.zeros(5, np.float32)
    return analyzer


def test_content_mask_keeps_tokens_inside_the_range():
    offsets = [(0, 3), (3, 5), (5, 9), (9, 10), (10, 12), (12, 12)]
    assert content_mask(offsets, (3, 10)).tolist() == [False, True, True, True, False, False]


def test_content_mask_drops_a_token_that_crosses_the_end():
    # e.g. "." merged with the closing "\n" into one token
    assert content_mask([(0, 4), (4, 6)], (0, 5)).tolist() == [True, False]


@given(arrays(np.float32, (3, 4), elements=st.floats(-1e3, 1e3, width=32)))
def test_probabilities_sum_to_one(states):
    rng = np.random.default_rng(0)
    w, b = rng.normal(size=(5, 4)).astype(np.float32), rng.normal(size=5).astype(np.float32)
    p = role_probabilities(w, b, states)
    assert p.shape == (3, 5)
    assert np.allclose(p.sum(axis=1), 1, atol=1e-5)
    assert (p >= 0).all()


def test_probabilities_follow_the_logits():
    w = np.eye(5, 4, dtype=np.float32)
    p = role_probabilities(w, np.zeros(5, np.float32), np.array([[0, 0, 5, 0]], np.float32))
    assert int(p.argmax()) == 2


def test_long_text_keeps_head_and_tail():
    assert bare_analyzer(max_tokens=4).truncate("abcdefghij") == "ab […] ij"
    assert bare_analyzer(max_tokens=4).truncate("abcd") == "abcd"


@pytest.mark.parametrize("text", ["", "   "])
def test_text_without_tokens_scores_as_declared(monkeypatch, text):
    monkeypatch.setattr(probe, "content_states", lambda *args: np.zeros((0, 4), np.float32))
    score = bare_analyzer().score(text, "tool")
    assert score.to_dict() == {r.value: float(r == "tool") for r in ROLES}


def test_missing_model_raises_model_not_found():
    # An unknown revision is never in the cache, and nothing is downloaded.
    with pytest.raises(probe.ModelNotFound, match="download-model"):
        probe.load_decoder("Qwen/Qwen3-0.6B", "0" * 40, layer=1)


def test_rejects_a_probe_with_other_roles(tmp_path):
    path = tmp_path / "probe.npz"
    np.savez(path, w=np.zeros((2, 4)), b=np.zeros(2), layer=1, model="m", revision="r", roles=["user", "tool"])
    with pytest.raises(ValueError, match="expected"):
        ProbeAnalyzer(path)
