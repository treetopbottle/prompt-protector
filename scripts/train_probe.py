"""Train a role probe and write it as a probe file for the Role Analyzer.

Follows "Prompt Injection as Role Confusion" (arXiv:2603.12277): neutral C4 text is rendered once in each
role, with the same random earlier turns before all 5 versions of a document, so that only the role
differs. A logistic regression learns the role from one layer's hidden state of each content token.
See docs/reports/2026-09-27-probe-spike.md for how the settings were chosen.

Usage:
  uv run --extra model --extra train python scripts/train_probe.py \\
      --model Qwen/Qwen3-0.6B --layer 16 --out src/prompt_protector/analyzer/probes/qwen3-0.6b.npz

Needs the model and the C4 validation shard in the Hugging Face cache:
  uv run hf download allenai/c4 en/c4-validation.00000-of-00008.json.gz --repo-type dataset
"""

import argparse
import gzip
import json
import random
import time
from pathlib import Path

import numpy as np

from prompt_protector.analyzer.chat_format import render, render_history
from prompt_protector.analyzer.probe import content_states, load_decoder, role_probabilities
from prompt_protector.roles import ROLES, Role

REVISIONS = {
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
}
C4_SHARD = "datasets--allenai--c4/snapshots/*/en/c4-validation.00000-of-00008.json.gz"
SEED = 123
SUBSAMPLE = 4  # keep every 4th content token
# Earlier turns put before a document; the same for all 5 of its roles.
HISTORIES = [[], [Role.SYSTEM], [Role.SYSTEM, Role.USER], [Role.USER], [Role.SYSTEM, Role.USER, Role.ASSISTANT]]


def load_docs(tok, n_docs, seq_len):
    path = next(Path.home().glob(f".cache/huggingface/hub/{C4_SHARD}"))
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        texts = [json.loads(line)["text"] for line in fh]
    picked = random.Random(SEED).sample(texts, n_docs)
    return [tok.decode(tok.encode(t, add_special_tokens=False)[:seq_len]) for t in picked]


def extract(tok, decoder, docs, seq_len):
    """Hidden states of every 4th content token, labeled with their role and document."""
    rng = np.random.default_rng(SEED)

    def snippet(i):
        n = max(8, int(rng.beta(0.5, 4.0) * (seq_len / 2 + 1)))
        return tok.decode(tok.encode(docs[i % len(docs)], add_special_tokens=False)[:n]).strip()

    states, labels, doc_ids = [], [], []
    t0 = time.perf_counter()
    for d, doc in enumerate(docs):
        roles = HISTORIES[rng.integers(len(HISTORIES))]
        history = render_history([(r, snippet(d + 2 + k)) for k, r in enumerate(roles)])
        reasoning = snippet(d + 1)
        for label, role in enumerate(ROLES):
            prompt, char_range = render(role, doc, reasoning=reasoning, history=history)
            hs = content_states(tok, decoder, prompt, char_range)
            keep = hs[rng.integers(SUBSAMPLE) :: SUBSAMPLE]
            states.append(keep.astype(np.float16))
            labels += [label] * len(keep)
            doc_ids += [d] * len(keep)
        if (d + 1) % 25 == 0 or d + 1 == len(docs):
            eta = (time.perf_counter() - t0) / (d + 1) * (len(docs) - d - 1)
            print(f"{d + 1}/{len(docs)} documents, {len(labels)} tokens, eta {eta / 60:.1f} min", flush=True)
    return np.concatenate(states).astype(np.float32), np.array(labels), np.array(doc_ids)


def fit(x, y, c):
    """Logistic regression on standardized states, folded back into one affine map on raw states."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x)
    clf = LogisticRegression(C=c, max_iter=2000).fit(scaler.transform(x), y)
    w = clf.coef_ / scaler.scale_
    b = clf.intercept_ - (w * scaler.mean_).sum(axis=1)
    return w.astype(np.float32), b.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", choices=sorted(REVISIONS))
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--c", type=float, default=0.1, help="inverse regularization strength")
    parser.add_argument("--docs", type=int, default=250)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    revision = REVISIONS[args.model]
    tok, decoder = load_decoder(args.model, revision, args.layer)
    docs = load_docs(tok, args.docs, args.seq_len)
    x, y, doc_ids = extract(tok, decoder, docs, args.seq_len)

    # Hold out 10% of the documents; the probe is trained on the rest.
    held_out_docs = np.random.default_rng(SEED).choice(len(docs), size=max(1, len(docs) // 10), replace=False)
    test = np.isin(doc_ids, held_out_docs)
    w, b = fit(x[~test], y[~test], args.c)
    pred = role_probabilities(w, b, x[test]).argmax(axis=1)
    accuracy = float((pred == y[test]).mean())
    per_role = {r.value: float((pred[y[test] == i] == i).mean()) for i, r in enumerate(ROLES)}
    print(f"held-out token accuracy {accuracy:.3f}: " + ", ".join(f"{r} {a:.3f}" for r, a in per_role.items()))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        w=w,
        b=b,
        layer=args.layer,
        model=args.model,
        revision=revision,
        roles=[r.value for r in ROLES],
        c=args.c,
        docs=args.docs,
        seq_len=args.seq_len,
        seed=SEED,
        held_out_accuracy=accuracy,
        held_out_accuracy_per_role=json.dumps(per_role),
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
