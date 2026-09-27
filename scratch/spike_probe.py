"""Step 1 spike of docs/plans/role-analyzer-probe.md: do role probes work on a small Qwen3 model?

Follows the paper's recipe: neutral C4 text wrapped in every role, a logistic regression per layer
on content-token hidden states. Then scores the same 22 texts as the zero-shot spike.

Usage:
  uv run --extra train python scratch/spike_probe.py extract [--model M] [--docs 250] [--limit N]
  uv run --extra train python scratch/spike_probe.py train [--model M]
  uv run --extra train python scratch/spike_probe.py score [--model M] [--layer L]

Hidden states and probes are cached in scratch/cache/<model>/ (git-ignored).
"""

import argparse
import gzip
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROLES = ["system", "user", "tool", "assistant", "reasoning"]
REPO = Path(__file__).parent.parent
CASES_DIR = REPO / "eval" / "cases"
C4_GLOB = "datasets--allenai--c4/snapshots/*/en/c4-validation.00000-of-00008.json.gz"
SEQ_LEN = 512
SUBSAMPLE = 4  # keep every 4th content token for training
SEED = 123
CASE_IDS = ["ben-002", "ben-016", "ind-001", "dir-003", "sty-001a", "sty-001e"]

# Qwen3 chat template, one message at a time: (opening, closing) around the content.
OPEN = {
    "system": "<|im_start|>system\n",
    "user": "<|im_start|>user\n",
    "tool": "<|im_start|>user\n<tool_response>\n",
    "reasoning": "<|im_start|>assistant\n<think>\n",
    "assistant": "<|im_start|>assistant\n<think>\n{partner}\n</think>\n\n",
}
CLOSE = {
    "system": "<|im_end|>\n",
    "user": "<|im_end|>\n",
    "tool": "\n</tool_response><|im_end|>\n",
    "reasoning": "\n</think>\n\n",
    "assistant": "<|im_end|>\n",
}
TEST_PREFIX = "<|im_start|>system\n<|im_end|>\n"  # the paper's test prefix for Qwen3

# --variant history: earlier turns put before every training sequence, so that "text comes before
# this" is not a cue for `assistant`. The same history is used for all 5 roles of a document.
HISTORIES = [[], ["system"], ["system", "user"], ["user"], ["system", "user", "assistant"]]


def render(role, text, partner=""):
    """Return the rendered message and the character range of `text` in it."""
    opening = OPEN[role].format(partner=partner)
    return opening + text + CLOSE[role], (len(opening), len(opening) + len(text))


def render_history(roles, texts):
    """Earlier turns as Qwen3 renders them; assistant turns in history have no reasoning block."""
    out = ""
    for role, text in zip(roles, texts, strict=False):
        if role == "assistant":
            out += f"<|im_start|>assistant\n{text}<|im_end|>\n"
        else:
            out += OPEN[role] + text + CLOSE[role]
    return out


def cache_dir(model_id, variant=""):
    d = REPO / "scratch" / "cache" / (model_id.replace("/", "--") + (f"-{variant}" if variant else ""))
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_model(model_id):
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True, dtype=torch.bfloat16).eval()
    return tok, model.model  # the decoder without the output layer


def probe_layers(model):
    n = len(model.layers)
    return list(range(2, n + 1, 2))


@torch.inference_mode()
def hidden_states(tok, model, prompt, layers):
    enc = tok(prompt, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    out = model(input_ids=enc["input_ids"], output_hidden_states=True)
    hs = torch.stack([out.hidden_states[layer][0] for layer in layers])  # layers x tokens x dim
    return hs.float().numpy(), enc["offset_mapping"][0].tolist()


def token_mask(offsets, char_range):
    start, end = char_range
    return np.array([s >= start and e <= end and e > s for s, e in offsets])


# --- extract ---------------------------------------------------------------------------------------


def load_docs(tok, n_docs):
    path = next(Path.home().glob(f".cache/huggingface/hub/{C4_GLOB}"))
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        texts = [json.loads(line)["text"] for line in fh]
    rng = random.Random(SEED)
    picked = rng.sample(texts, n_docs)
    return [tok.decode(tok.encode(t, add_special_tokens=False)[:SEQ_LEN]) for t in picked]


def extract(args):
    tok, model = load_model(args.model)
    layers = probe_layers(model)
    docs = load_docs(tok, args.docs)[: args.limit or None]
    rng = np.random.default_rng(SEED)
    partner_lens = (rng.beta(0.5, 4.0, size=len(docs)) * (SEQ_LEN / 2 + 1)).astype(int)
    partners = [
        tok.decode(tok.encode(docs[(i + 1) % len(docs)], add_special_tokens=False)[: partner_lens[i]]).strip()
        for i in range(len(docs))
    ]

    def snippet(i):
        n = max(8, int(rng.beta(0.5, 4.0) * (SEQ_LEN / 2 + 1)))
        return tok.decode(tok.encode(docs[i % len(docs)], add_special_tokens=False)[:n]).strip()

    histories = []
    for d in range(len(docs)):
        roles = HISTORIES[rng.integers(len(HISTORIES))] if args.variant == "history" else []
        histories.append(render_history(roles, [snippet(d + 2 + k) for k in range(len(roles))]))

    feats, labels, doc_ids, positions = [[] for _ in layers], [], [], []
    t0, total = time.perf_counter(), len(docs) * len(ROLES)
    for d, doc in enumerate(docs):
        for r, role in enumerate(ROLES):
            prompt, char_range = render(role, doc, partners[d])
            h = histories[d]
            prompt, char_range = h + prompt, (char_range[0] + len(h), char_range[1] + len(h))
            hs, offsets = hidden_states(tok, model, prompt, layers)
            idx = np.flatnonzero(token_mask(offsets, char_range))
            keep = idx[rng.integers(SUBSAMPLE) :: SUBSAMPLE]
            for li in range(len(layers)):
                feats[li].append(hs[li, keep].astype(np.float16))
            labels += [r] * len(keep)
            doc_ids += [d] * len(keep)
            positions += (keep - idx[0]).tolist()
        done = (d + 1) * len(ROLES)
        if (d + 1) % 10 == 0 or d + 1 == len(docs):
            eta = (time.perf_counter() - t0) / done * (total - done)
            print(f"{done}/{total} sequences, {len(labels)} tokens, eta {eta / 60:.1f} min", flush=True)

    out = cache_dir(args.model, args.variant)
    for li, layer in enumerate(layers):
        np.save(out / f"hs_layer{layer:02d}.npy", np.concatenate(feats[li]))
    np.savez(out / "labels.npz", labels=labels, doc_ids=doc_ids, positions=positions, layers=layers)
    print(f"saved {len(labels)} tokens x {len(layers)} layers to {out}")


# --- train -----------------------------------------------------------------------------------------


def fit(x_train, y_train, c):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x_train)
    clf = LogisticRegression(C=c, max_iter=2000).fit(scaler.transform(x_train), y_train)
    # Fold the scaler into the weights, so scoring is a single affine map.
    w = clf.coef_ / scaler.scale_
    b = clf.intercept_ - (w * scaler.mean_).sum(axis=1)
    return w.astype(np.float32), b.astype(np.float32)


def predict(w, b, x):
    logits = x @ w.T + b
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def train(args):
    d = cache_dir(args.model, args.variant)
    meta = np.load(d / "labels.npz")
    y, docs, layers = meta["labels"], meta["doc_ids"], meta["layers"].tolist()
    rng = np.random.default_rng(SEED)
    test_docs = rng.choice(np.unique(docs), size=max(1, len(np.unique(docs)) // 10), replace=False)
    test = np.isin(docs, test_docs)
    print(f"{(~test).sum()} train tokens, {test.sum()} test tokens from {len(test_docs)} held-out docs\n")

    def evaluate(layer, c):
        x = np.load(d / f"hs_layer{layer:02d}.npy").astype(np.float32)
        t0 = time.perf_counter()
        w, b = fit(x[~test], y[~test], c)
        pred = predict(w, b, x[test]).argmax(axis=1)
        acc = (pred == y[test]).mean()
        per_role = [(pred[y[test] == r] == r).mean() for r in range(len(ROLES))]
        return w, b, acc, per_role, time.perf_counter() - t0

    print(f"{'layer':>5} {'C':>6}  {'acc':>5}  " + " ".join(f"{r[:5]:>5}" for r in ROLES) + "  fit s")
    results = {}
    for layer in layers:
        w, b, acc, per_role, secs = evaluate(layer, args.c)
        results[layer] = (acc, args.c)
        np.savez(d / f"probe_layer{layer:02d}.npz", w=w, b=b, layer=layer, c=args.c, acc=acc)
        print(f"{layer:5d} {args.c:6g}  {acc:5.3f}  " + " ".join(f"{a:5.3f}" for a in per_role) + f"  {secs:5.0f}")

    best = sorted(results, key=lambda la: -results[la][0])[:3]
    print(f"\nC grid on the best layers {best}:")
    for layer in best:
        for c in (1e-3, 1e-2, 1.0):
            w, b, acc, per_role, secs = evaluate(layer, c)
            line = f"{layer:5d} {c:6g}  {acc:5.3f}  " + " ".join(f"{a:5.3f}" for a in per_role) + f"  {secs:5.0f}"
            if acc > results[layer][0]:
                results[layer] = (acc, c)
                np.savez(d / f"probe_layer{layer:02d}.npz", w=w, b=b, layer=layer, c=c, acc=acc)
                line += "  (kept)"
            print(line)


# --- score -----------------------------------------------------------------------------------------


def load_cases():
    cases = {c["id"]: c for p in CASES_DIR.glob("*.json") for c in json.loads(p.read_text()) if "segments" in c}
    return [cases[cid] for cid in CASE_IDS]


def render_alone(role, text):
    """One message on its own, as at training time; assistant gets an empty reasoning block."""
    return render(role, text, partner="")


def render_untagged(text):
    return TEST_PREFIX + "\n" + text, (len(TEST_PREFIX) + 1, len(TEST_PREFIX) + 1 + len(text))


def render_context(segments):
    """The whole prompt; assistant turns without a reasoning block, as Qwen3 renders history."""
    prompt, ranges = "", []
    for seg in segments:
        if seg["role"] == "assistant":
            opening, closing = "<|im_start|>assistant\n", "<|im_end|>\n"
        else:
            opening, closing = OPEN[seg["role"]], CLOSE[seg["role"]]
        start = len(prompt) + len(opening)
        ranges.append((start, start + len(seg["text"])))
        prompt += opening + seg["text"] + closing
    return prompt, ranges


def score(args):
    d = cache_dir(args.model, args.variant)
    probes = {int(p.stem[-2:]): np.load(p) for p in sorted(d.glob("probe_layer*.npz"))}
    tok, model = load_model(args.model)
    layers = sorted(probes)

    # Texts: every segment plus every expected span; expected roles as in the zero-shot spike.
    items = []  # (name, case, segment index, char range in the segment text or None, declared, expected, injected)
    for case in load_cases():
        spans = {f["segment"]: f for f in case["expected"]["findings"]}
        for i, seg in enumerate(case["segments"]):
            injected = i in spans
            expected = spans[i]["perceived"] if injected else [seg["role"]]
            items.append((f"{case['id']} seg{i}", case, i, None, seg["role"], expected, injected))
            if injected:
                s = seg["text"].index(spans[i]["span"])
                rng = (s, s + len(spans[i]["span"]))
                items.append((f"{case['id']} span", case, i, rng, seg["role"], expected, True))

    def mean_probs(prompt, char_range):
        hs, offsets = hidden_states(tok, model, prompt, layers)
        mask = token_mask(offsets, char_range)
        return {
            layer: predict(probes[layer]["w"], probes[layer]["b"], hs[li, mask]).mean(axis=0)
            for li, layer in enumerate(layers)
        }

    results = {}  # placement -> list of {layer: probs}
    t0 = time.perf_counter()
    for placement in ("alone", "untagged", "context"):
        rows = []
        for _, case, i, span, declared, _, _ in items:
            seg = case["segments"][i]
            text = seg["text"] if span is None else seg["text"][span[0] : span[1]]
            if placement == "alone":
                rows.append(mean_probs(*render_alone(declared, text)))
            elif placement == "untagged":
                rows.append(mean_probs(*render_untagged(text)))
            else:
                prompt, ranges = render_context(case["segments"])
                start = ranges[i][0]
                rng = ranges[i] if span is None else (start + span[0], start + span[1])
                rows.append(mean_probs(prompt, rng))
        results[placement] = rows
    print(f"scored {len(items)} texts x 3 placements in {time.perf_counter() - t0:.0f}s\n")

    def correct(probs, expected):
        return ROLES[int(np.argmax(probs))] in expected

    print("Correct top role: benign (of 14) / injected in perceived (of 8) / injected top != declared (of 8)")
    print(f"{'layer':>5}  " + "  ".join(f"{p:>16}" for p in results))
    for layer in layers:
        cells = []
        for rows in results.values():
            b = sum(correct(r[layer], it[5]) for r, it in zip(rows, items, strict=True) if not it[6])
            inj = [(r[layer], it) for r, it in zip(rows, items, strict=True) if it[6]]
            ip = sum(correct(p, it[5]) for p, it in inj)
            flagged = sum(ROLES[int(np.argmax(p))] != it[4] for p, it in inj)
            cells.append(f"{b:2d} / {ip} / {flagged}")
        print(f"{layer:5d}  " + "  ".join(f"{c:>16}" for c in cells))

    layer = args.layer or max(layers, key=lambda la: probes[la]["acc"])
    for placement, rows in results.items():
        print(f"\n{placement}, layer {layer}:")
        print(f"{'text':14} {'decl':9} {'top':9} " + " ".join(f"{r[:5]:>5}" for r in ROLES) + "  confus")
        for r, (name, _, _, _, declared, expected, _) in zip(rows, items, strict=True):
            p = r[layer]
            top = ROLES[int(np.argmax(p))]
            mark = " ok" if top in expected else ""
            conf = 1 - p[ROLES.index(declared)]
            print(f"{name:14} {declared:9} {top:9} " + " ".join(f"{v:5.2f}" for v in p) + f"  {conf:6.2f}{mark}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["extract", "train", "score"])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--variant", default="", help="'history': random earlier turns before training text")
    parser.add_argument("--docs", type=int, default=250)
    parser.add_argument("--limit", type=int, default=0, help="extract: only the first N docs (smoke test)")
    parser.add_argument("--c", type=float, default=0.1, help="train: regularization for the layer sweep")
    parser.add_argument("--layer", type=int, default=0, help="score: layer for the detail tables")
    args = parser.parse_args()
    {"extract": extract, "train": train, "score": score}[args.command](args)


if __name__ == "__main__":
    main()
