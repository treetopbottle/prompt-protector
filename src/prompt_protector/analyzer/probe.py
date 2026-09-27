"""Role probe backend: a linear probe on a middle layer of a small local model.

The probe was trained on neutral text rendered in every role (see scripts/train_probe.py), so it reads how
the model internally perceives a token's role. A text's Role Score is the mean over its tokens.
"""

from pathlib import Path

import numpy as np

from ..roles import ROLES, Role
from ..scores import RoleScore
from .chat_format import render

DEFAULT_PROBE = Path(__file__).parent / "probes" / "qwen3-0.6b.npz"


class ModelNotFound(RuntimeError):
    pass


def load_decoder(model_id: str, revision: str, layer: int):
    """Load the tokenizer and the model's first `layer` layers, from the local cache only."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True)
        causal = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, local_files_only=True, dtype=torch.bfloat16
        )
    except OSError as e:
        raise ModelNotFound(f'{model_id} is not downloaded; run "prompt-protector download-model" first') from e
    decoder = causal.model
    decoder.layers = decoder.layers[:layer]
    decoder.norm = torch.nn.Identity()  # the probe reads the layer's raw output
    return tok, decoder.eval()


def content_mask(offsets: list[tuple[int, int]], char_range: tuple[int, int]) -> np.ndarray:
    """Tokens that lie entirely inside `char_range`."""
    start, end = char_range
    return np.array([start <= s < e <= end for s, e in offsets], dtype=bool)


def content_states(tok, decoder, prompt: str, char_range: tuple[int, int]) -> np.ndarray:
    """Hidden states (tokens x dim) of the tokens inside `char_range`."""
    import torch

    enc = tok(prompt, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    with torch.inference_mode():
        hs = decoder(input_ids=enc["input_ids"], use_cache=False).last_hidden_state[0]
    mask = content_mask(enc["offset_mapping"][0].tolist(), char_range)
    return hs[torch.from_numpy(mask)].float().numpy()


def role_probabilities(w: np.ndarray, b: np.ndarray, states: np.ndarray) -> np.ndarray:
    """Per-token role probabilities (tokens x roles) from the probe's affine map and a softmax."""
    logits = states @ w.T + b
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


class ProbeAnalyzer:
    def __init__(self, probe: str | Path = DEFAULT_PROBE, max_tokens: int = 2048):
        data = np.load(probe)
        if [str(r) for r in data["roles"]] != [r.value for r in ROLES]:
            raise ValueError(f"probe {probe} has roles {list(data['roles'])}, expected {[r.value for r in ROLES]}")
        self.w, self.b = data["w"], data["b"]
        self.layer = int(data["layer"])
        self.model_id, self.revision = str(data["model"]), str(data["revision"])
        self.max_tokens = max_tokens
        self.tok, self.decoder = load_decoder(self.model_id, self.revision, self.layer)

    def truncate(self, text: str) -> str:
        """Keep the first and last `max_tokens / 2` tokens of a long text, with "[…]" between them."""
        ids = self.tok.encode(text, add_special_tokens=False)
        if len(ids) <= self.max_tokens:
            return text
        half = self.max_tokens // 2
        return self.tok.decode(ids[:half]) + " […] " + self.tok.decode(ids[-half:])

    def score(self, text: str, role: Role | str) -> RoleScore:
        """Score `text` rendered alone under its declared `role`."""
        role = Role(role)
        prompt, char_range = render(role, self.truncate(text))
        states = content_states(self.tok, self.decoder, prompt, char_range)
        if len(states) == 0:
            # No tokens to judge (empty text): nothing contradicts the declared role.
            return RoleScore.from_dict({r: float(r == role) for r in ROLES})
        mean = role_probabilities(self.w, self.b, states).mean(axis=0)
        return RoleScore.from_dict(dict(zip(ROLES, mean.tolist(), strict=True)))
