"""Option A follow-ups to the role probe spike: per-token localization, window max, score features, tool calls.

See docs/reports/2026-10-06-probe-followups.md. All checks use Qwen3-0.6B with a probe at layer 16.

Usage:
  uv run --extra model --extra train python scratch/spike_followup.py train --variant base
  uv run --extra model --extra train python scratch/spike_followup.py train --variant call
  uv run --extra model --extra train python scratch/spike_followup.py score
  uv run --extra model --extra train python scratch/spike_followup.py analyze
  uv run --extra model --extra train --with matplotlib python scratch/spike_followup.py plot --config base

Probes and per-token scores are cached in scratch/cache/followup/ (git-ignored).
"""

import argparse
import json
import pickle
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "eval")]

import train_probe  # noqa: E402
from check_cases import load_cases  # noqa: E402
from run_analyzer import BENIGN_FILES, auc, collect_items  # noqa: E402

from prompt_protector.analyzer.chat_format import render, render_history  # noqa: E402
from prompt_protector.analyzer.probe import content_mask, load_decoder, role_probabilities  # noqa: E402
from prompt_protector.roles import ROLES, Role  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B"
LAYER = 16
OUT = REPO / "scratch" / "cache" / "followup"
SPIKE_PROBE = REPO / "scratch" / "cache" / "Qwen--Qwen3-0.6B-history" / "probe_layer16.npz"
R = [r.value for r in ROLES]
SYSTEM, USER, TOOL, ASSISTANT, REASONING = range(5)

# Probe configurations: (probe file, render tool text after a tool call at scoring time)
CONFIGS = {
    "spike": (SPIKE_PROBE, False),
    "base": (OUT / "probe-base.npz", False),
    "base+call": (OUT / "probe-base.npz", True),
    "call": (OUT / "probe-call.npz", True),
}

# --- tool calls ------------------------------------------------------------------------------------

TOOL_NAMES = ["web_search", "fetch_url", "read_file", "run_shell", "search_docs", "get_page", "read_email",
              "query_db", "list_files", "get_issue", "lookup", "get_weather"]  # fmt: skip


def call_turn(name: str, arguments: dict) -> str:
    """An assistant turn with one tool call, as Qwen3's chat template renders it."""
    call = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    return f"<|im_start|>assistant\n<tool_call>\n{call}\n</tool_call><|im_end|>\n"


def random_call(rng: random.Random, other_doc: str) -> str:
    words = re.findall(r"\w+", other_doc)[: rng.randint(2, 6)]
    kind = rng.choice(["query", "url", "path", "none"])
    args = {
        "query": {"query": " ".join(words)},
        "url": {"url": "https://www.example.com/" + "-".join(w.lower() for w in words)},
        "path": {"path": "docs/" + "_".join(w.lower() for w in words) + ".txt"},
        "none": {},
    }[kind]
    return call_turn(rng.choice(TOOL_NAMES), args)


def case_call(case: dict, i: int) -> str:
    """The call before tool segment `i`: its name, with the arguments of the assistant's JSON call if any."""
    seg, args = case["segments"][i], {}
    prev = case["segments"][i - 1] if i else None
    if prev and prev["role"] == "assistant":
        try:
            args = json.loads(prev["text"]).get("arguments", {})
        except (json.JSONDecodeError, AttributeError):
            pass
    return call_turn(seg.get("name") or "fetch_url", args)


# --- train -----------------------------------------------------------------------------------------


def extract(tok, decoder, docs, seq_len, with_call):
    """train_probe.extract, plus an optional tool call before the tool version of each document.

    The calls come from their own random generator, so both variants see the same documents, histories
    and subsampled tokens; the call is the only difference.
    """
    rng = np.random.default_rng(train_probe.SEED)
    call_rng = random.Random(train_probe.SEED + 1)

    def snippet(i):
        n = max(8, int(rng.beta(0.5, 4.0) * (seq_len / 2 + 1)))
        return tok.decode(tok.encode(docs[i % len(docs)], add_special_tokens=False)[:n]).strip()

    states, labels, doc_ids = [], [], []
    t0 = time.perf_counter()
    for d, doc in enumerate(docs):
        roles = train_probe.HISTORIES[rng.integers(len(train_probe.HISTORIES))]
        history = render_history([(r, snippet(d + 2 + k)) for k, r in enumerate(roles)])
        reasoning = snippet(d + 1)
        call = random_call(call_rng, docs[(d + 3) % len(docs)])
        for label, role in enumerate(ROLES):
            h = history + call if with_call and role == Role.TOOL else history
            prompt, char_range = render(role, doc, reasoning=reasoning, history=h)
            hs, _ = token_states(tok, decoder, prompt, char_range)
            keep = hs[rng.integers(train_probe.SUBSAMPLE) :: train_probe.SUBSAMPLE]
            states.append(keep.astype(np.float16))
            labels += [label] * len(keep)
            doc_ids += [d] * len(keep)
        if (d + 1) % 25 == 0 or d + 1 == len(docs):
            eta = (time.perf_counter() - t0) / (d + 1) * (len(docs) - d - 1)
            print(f"{d + 1}/{len(docs)} documents, {len(labels)} tokens, eta {eta / 60:.1f} min", flush=True)
    return np.concatenate(states).astype(np.float32), np.array(labels), np.array(doc_ids)


def train(args):
    tok, decoder = load_decoder(MODEL, train_probe.REVISIONS[MODEL], LAYER)
    docs = train_probe.load_docs(tok, 250, 512)
    x, y, doc_ids = extract(tok, decoder, docs, 512, args.variant == "call")
    held_out = np.random.default_rng(train_probe.SEED).choice(len(docs), size=len(docs) // 10, replace=False)
    test = np.isin(doc_ids, held_out)
    w, b = train_probe.fit(x[~test], y[~test], 0.1)
    pred = role_probabilities(w, b, x[test]).argmax(axis=1)
    acc = float((pred == y[test]).mean())
    per_role = {r: float((pred[y[test] == i] == i).mean()) for i, r in enumerate(R)}
    print(f"held-out token accuracy {acc:.3f}: " + ", ".join(f"{r} {a:.3f}" for r, a in per_role.items()))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"probe-{args.variant}.npz", w=w, b=b, layer=LAYER, acc=acc, per_role=json.dumps(per_role))


# --- score -----------------------------------------------------------------------------------------


def token_states(tok, decoder, prompt, char_range):
    """Hidden states of the content tokens, and their character offsets relative to the content."""
    import torch

    enc = tok(prompt, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    with torch.inference_mode():
        hs = decoder(input_ids=enc["input_ids"], use_cache=False).last_hidden_state[0]
    offsets = enc["offset_mapping"][0].tolist()
    mask = content_mask(offsets, char_range)
    rel = np.array([(s - char_range[0], e - char_range[0]) for (s, e), m in zip(offsets, mask, strict=True) if m])
    return hs[torch.from_numpy(mask)].float().numpy(), rel.reshape(-1, 2)


def sentences(text: str) -> list[tuple[int, int]]:
    """Character ranges of sentences and lines, whitespace trimmed."""
    out, start = [], 0
    for m in re.finditer(r"(?<=[.!?])[\"')\]]*\s+|\n+", text):
        out.append((start, m.start() if text[m.start()] == "\n" else m.end()))
        start = m.end()
    out.append((start, len(text)))
    trimmed = []
    for s, e in out:
        piece = text[s:e]
        s2 = s + len(piece) - len(piece.lstrip())
        e2 = e - (len(piece) - len(piece.rstrip()))
        if e2 > s2:
            trimmed.append((s2, e2))
    return trimmed


def neutral_texts(tok):
    """C4 documents that the probes were not trained on, cut to 16–256 tokens, each under every role."""
    import gzip

    path = next(Path.home().glob(f".cache/huggingface/hub/{train_probe.C4_SHARD}"))
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        texts = [json.loads(line)["text"] for line in fh]
    trained = set(random.Random(train_probe.SEED).sample(texts, 250))
    rng = random.Random(7)
    picked = rng.sample([t for t in texts if t not in trained], 200)
    call_rng = random.Random(8)
    out = []
    for i, t in enumerate(picked):
        ids = tok.encode(t, add_special_tokens=False)[: rng.randint(16, 256)]
        text = tok.decode(ids).strip()
        call = random_call(call_rng, picked[(i + 1) % len(picked)])
        out += [{"text": text, "declared": r, "call": call, "doc": i} for r in R]
    return out


def score(args):
    tok, decoder = load_decoder(MODEL, train_probe.REVISIONS[MODEL], LAYER)
    probes = {name: np.load(path) for name, (path, _) in CONFIGS.items()}
    cases = {c["id"]: c for c in load_cases() if "segments" in c}

    items = collect_items(cases.values())
    for it in items:
        it["call"] = case_call(cases[it["case"]], it["segment"]) if it["declared"] == "tool" else ""
    sents = []
    for n, it in enumerate(items):
        if it["kind"] == "injected segment":
            for s, e in sentences(it["text"]):
                sents.append({"item": n, "range": (s, e), "text": it["text"][s:e], "declared": it["declared"]})
                sents[-1]["call"] = it["call"]
    neutral = neutral_texts(tok)
    print(f"{len(items)} eval texts, {len(sents)} sentences, {len(neutral)} neutral texts", flush=True)

    t0 = time.perf_counter()
    for group in (items, sents, neutral):
        for rec in group:
            rec["probs"] = {}
            for with_call in (False, True):
                if with_call and rec["declared"] != "tool":
                    continue
                history = rec["call"] if with_call else ""
                prompt, char_range = render(rec["declared"], rec["text"], history=history)
                hs, rel = token_states(tok, decoder, prompt, char_range)
                rec["offsets"] = rel
                for name, (_, cfg_call) in CONFIGS.items():
                    if cfg_call == with_call or (rec["declared"] != "tool" and not with_call):
                        p = probes[name]
                        rec["probs"][name] = role_probabilities(p["w"], p["b"], hs).astype(np.float32)
    print(f"scored in {time.perf_counter() - t0:.0f} s")
    with open(OUT / "scores.pkl", "wb") as fh:
        pickle.dump({"items": items, "sentences": sents, "neutral": neutral}, fh)


# --- analyze ---------------------------------------------------------------------------------------


def mean_probs(rec, cfg):
    p = rec["probs"][cfg]
    if len(p) == 0:
        return np.eye(5)[R.index(rec["declared"])]
    return p.mean(axis=0)


def tok_conf(p, d):
    """Per-token confusion, 1 − P(declared)."""
    return 1 - p[:, d]


def tok_auth(p, d):
    """Per-token mass on the roles the model trusts or treats as its own: system, assistant, reasoning."""
    keep = [r for r in (SYSTEM, ASSISTANT, REASONING) if r != d]
    return p[:, keep].sum(axis=1)


TOKEN_SCORES = {"confusion": tok_conf, "authority": tok_auth}


def smooth(x, w):
    if w <= 1 or len(x) == 0:
        return x
    k = np.ones(w)
    return np.convolve(x, k, "same") / np.convolve(np.ones(len(x)), k, "same")


def window_max(x, w):
    if len(x) == 0:
        return 0.0
    if len(x) <= w:
        return float(x.mean())
    return float(np.convolve(x, np.ones(w) / w, "valid").max())


def pct_rank(value, reference):
    """Share of `reference` below `value` (ties count half): a per-role calibrated score in [0, 1]."""
    reference = np.asarray(reference)
    return float(((reference < value).sum() + 0.5 * (reference == value).sum()) / len(reference))


def fmt_auc(pos, neg):
    a = auc(list(pos), list(neg))
    return "–" if a != a else f"{a:.2f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out) + "\n"


def analyze(args):
    with open(OUT / "scores.pkl", "rb") as fh:
        data = pickle.load(fh)
    items, sents, neutral = data["items"], data["sentences"], data["neutral"]
    cases = {c["id"]: c for c in load_cases() if "segments" in c}
    for name, (path, _) in CONFIGS.items():
        p = np.load(path)
        print(f"{name}: held-out token accuracy {float(p['acc']):.3f}", p["per_role"] if "per_role" in p else "")
    print()

    benign = [it for it in items if it["kind"] == "benign"]
    spans = [it for it in items if it["kind"] == "span"]
    injected = [it for it in items if it["kind"] == "injected segment"]
    ntr = {r: [n for n in neutral if n["declared"] == r] for r in R}

    for cfg in CONFIGS:
        print(f"\n## {cfg}\n")
        section_baseline(cfg, items, benign, spans)
        section_features(cfg, benign, spans, ntr)
        section_window(cfg, benign, injected, ntr)
        section_localize(cfg, items, injected, sents, cases, ntr)


def section_baseline(cfg, items, benign, spans):
    """The metrics of eval/run_analyzer.py, to compare configurations."""
    for it in items:
        m = mean_probs(it, cfg)
        it["top"], it["conf"] = R[int(m.argmax())], 1 - m[R.index(it["declared"])]
    btool = [b for b in benign if b["declared"] == "tool" and b["file"] in BENIGN_FILES]
    tspans = [s for s in spans if s["declared"] == "tool"]
    rows = [
        ("Benign tool segments, top = tool", f"{sum(b['top'] == 'tool' for b in btool)}/{len(btool)}"),
        ("Benign segments, top = declared", f"{sum(b['top'] == b['declared'] for b in benign)}/{len(benign)}"),
        ("Spans, top in perceived", f"{sum(s['top'] in s['perceived'] for s in spans)}/{len(spans)}"),
        ("Spans, top ≠ declared", f"{sum(s['top'] != s['declared'] for s in spans)}/{len(spans)}"),
        ("AUC confusion, spans vs benign", fmt_auc([s["conf"] for s in spans], [b["conf"] for b in benign])),
        (
            "AUC confusion, tool spans vs benign tool",
            fmt_auc([s["conf"] for s in tspans], [b["conf"] for b in benign if b["declared"] == "tool"]),
        ),
    ]
    print("### Baseline metrics\n\n" + table(["Metric", cfg], rows))


def span_features(m, d, base):
    """Scores of a mean probability vector `m` under declared role index `d`; `base` is the neutral mean."""
    others = [r for r in range(5) if r != d]
    return {
        "confusion": 1 - m[d],
        "authority": sum(m[r] for r in (SYSTEM, ASSISTANT, REASONING) if r != d),
        "ratio": max(np.log(m[r] / base[r]) for r in others),
    }


def section_features(cfg, benign, spans, ntr):
    """Option 3: features of the mean scores, with per-role thresholds set on neutral text."""
    base = {r: np.mean([mean_probs(n, cfg) for n in ntr[r]], axis=0) for r in R}
    nfeat = {r: [span_features(mean_probs(n, cfg), R.index(r), base[r]) for n in ntr[r]] for r in R}
    for it in benign + spans:
        d = R.index(it["declared"])
        m = mean_probs(it, cfg)
        it["feat"] = span_features(m, d, base[it["declared"]])
        it["feat_pct"] = {k: pct_rank(v, [f[k] for f in nfeat[it["declared"]]]) for k, v in it["feat"].items()}
        ratio = [(np.log(m[r] / base[it["declared"]][r]), R[r]) for r in range(5) if r != d]
        it["perceived_ratio"] = max(ratio)[1]
        it["perceived_raw"] = R[max((m[r], r) for r in range(5) if r != d)[1]]

    print("### Score features (option 3)\n")
    print("Neutral mean scores per declared role (C4 text the probe was not trained on):\n")
    print(table(["Declared"] + R, [[r] + [f"{v:.2f}" for v in base[r]] for r in R]))

    rows = []
    groups = {
        "all": (spans, benign),
        "tool": ([s for s in spans if s["declared"] == "tool"], [b for b in benign if b["declared"] == "tool"]),
        "user": ([s for s in spans if s["declared"] == "user"], [b for b in benign if b["declared"] == "user"]),
    }
    for feat in ("confusion", "authority", "ratio"):
        for calibrated in (False, True):
            key = "feat_pct" if calibrated else "feat"
            row = [feat + (" (per-role percentile)" if calibrated else "")]
            for pos, neg in groups.values():
                row.append(fmt_auc([s[key][feat] for s in pos], [b[key][feat] for b in neg]))
            # Threshold: 95th percentile of the feature on neutral text under the same declared role.
            flag = [s["feat_pct"][feat] > 0.95 for s in spans]
            fp = [b["feat_pct"][feat] > 0.95 for b in benign]
            fpt = [b["feat_pct"][feat] > 0.95 for b in benign if b["declared"] == "tool"]
            row += (
                [f"{sum(flag)}/{len(flag)}", f"{sum(fp)}/{len(fp)}", f"{sum(fpt)}/{len(fpt)}"]
                if calibrated
                else ["", "", ""]
            )
            rows.append(row)
    flag = [s["top"] != s["declared"] for s in spans]
    fp = [b["top"] != b["declared"] for b in benign]
    fpt = [b["top"] != b["declared"] for b in benign if b["declared"] == "tool"]
    rows.append(["argmax (top ≠ declared)", "", "", "", f"{sum(flag)}/{len(flag)}", f"{sum(fp)}/{len(fp)}",
                 f"{sum(fpt)}/{len(fpt)}"])  # fmt: skip
    header = ["Feature", "AUC all", "AUC tool", "AUC user", "Spans flagged", "Benign flagged", "Benign tool flagged"]
    print(table(header, rows))
    raw = sum(s["perceived_raw"] in s["perceived"] for s in spans)
    rat = sum(s["perceived_ratio"] in s["perceived"] for s in spans)
    print(f"Perceived role among non-declared roles: highest raw score {raw}/{len(spans)}, "
          f"highest ratio to neutral {rat}/{len(spans)}\n")  # fmt: skip


def aggregate(x, how):
    """A segment score from per-token scores: their mean, or the max over windows of `how` tokens."""
    if how == "mean":
        return float(x.mean()) if len(x) else 0.0
    return window_max(x, how)


def section_window(cfg, benign, injected, ntr):
    """Option 2: segment score from the mean against the max over a sliding window of per-token scores."""
    print("### Segment scores: mean against window max (option 2)\n")
    rows = []
    for tname, tfun in TOKEN_SCORES.items():
        for how in ("mean", 8, 16, 32):
            raw = {id(r): aggregate(tfun(r["probs"][cfg], R.index(r["declared"])), how) for r in injected + benign}
            ref = {
                r: [aggregate(tfun(n["probs"][cfg], R.index(r)), how) for n in ntr[r]] for r in R
            }  # neutral text, per declared role
            pct = {id(r): pct_rank(raw[id(r)], ref[r["declared"]]) for r in injected + benign}
            row = [f"{tname}, " + ("mean" if how == "mean" else f"max over {how} tokens")]
            for vals in (raw, pct):
                row.append(fmt_auc([vals[id(s)] for s in injected], [vals[id(b)] for b in benign]))
                pos_t = [vals[id(s)] for s in injected if s["declared"] == "tool"]
                row.append(fmt_auc(pos_t, [vals[id(b)] for b in benign if b["declared"] == "tool"]))
            flag = [pct[id(s)] > 0.95 for s in injected]
            fp = [pct[id(b)] > 0.95 for b in benign]
            row += [f"{sum(flag)}/{len(flag)}", f"{sum(fp)}/{len(fp)}"]
            rows.append(row)
    header = ["Segment score", "AUC all", "AUC tool", "AUC all, per-role pct", "AUC tool, per-role pct",
              "Injected flagged", "Benign flagged"]  # fmt: skip
    print(table(header, rows))
    n_inj, n_ben = [len(s["offsets"]) for s in injected], [len(b["offsets"]) for b in benign]
    print(f"Median length: injected segments {int(np.median(n_inj))} tokens, benign segments "
          f"{int(np.median(n_ben))} tokens; AUC of length alone {fmt_auc(n_inj, n_ben)}\n")  # fmt: skip


def span_ranges(case, segment, text):
    out = []
    for f in case["expected"]["findings"]:
        if f["segment"] == segment:
            s = text.index(f["span"])
            out.append((s, s + len(f["span"])))
    return out


def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def char_set(ranges):
    return {c for s, e in ranges for c in range(s, e)}


def section_localize(cfg, items, injected, sents, cases, ntr):
    """Option 1: does one pass find the injected span inside a segment?"""
    print("### Localization (option 1)\n")
    by_item = defaultdict(list)
    for s in sents:
        by_item[s["item"]].append(s)
    idx = {id(it): n for n, it in enumerate(items)}

    local = []
    for seg in injected:
        ranges = span_ranges(cases[seg["case"]], seg["segment"], seg["text"])
        covered = len(char_set(ranges)) / max(1, len(seg["text"]))
        if covered < 0.9:
            local.append((seg, ranges))
    print(f"{len(local)} injected segments where the spans cover less than 90% of the text "
          f"({len(injected) - len(local)} others left out).\n")  # fmt: skip

    rows, drift_rows = [], []
    for tname, tfun in TOKEN_SCORES.items():
        thresholds = {}
        for w in (1, 8):
            thresholds[w] = {r: np.quantile(np.concatenate(
                [smooth(tfun(n["probs"][cfg], R.index(r)), w) for n in ntr[r]]), 0.95) for r in R}  # fmt: skip
        tok_aucs = {1: [], 8: []}
        hit_ctx, hit_alone, chance, iou_tok, iou_ctx, iou_alone, n_sent = [], [], [], [], [], [], []
        before, inside, after_near, after_far = [], [], [], []
        for seg, ranges in local:
            d = R.index(seg["declared"])
            x = tfun(seg["probs"][cfg], d)
            offs = seg["offsets"]
            in_span = np.array([any(overlaps(o, r) for r in ranges) for o in offs])
            for w in (1, 8):
                if in_span.any() and not in_span.all():
                    xs = smooth(x, w)
                    tok_aucs[w].append(auc(list(xs[in_span]), list(xs[~in_span])))
            # Per-token threshold from neutral text, 8-token smoothing: predicted characters against the spans.
            xs = smooth(x, 8)
            pred = char_set(tuple(o) for o, v in zip(offs, xs, strict=True) if v > thresholds[8][seg["declared"]])
            truth = char_set(ranges)
            iou_tok.append(len(pred & truth) / len(pred | truth) if pred | truth else 1.0)
            # Sentences: the highest-scoring one, in the one pass (token mean) or each scored alone.
            ss = by_item[idx[id(seg)]]
            n_sent.append(len(ss))
            if not ss:
                continue
            ctx = []
            for s in ss:
                m = np.array([overlaps(o, s["range"]) for o in offs])
                ctx.append(x[m].mean() if m.any() else -1)
            alone = [tfun(s["probs"][cfg], d).mean() if len(s["probs"][cfg]) else -1 for s in ss]
            hits = [any(overlaps(s["range"], r) for r in ranges) for s in ss]
            best_ctx, best_alone = int(np.argmax(ctx)), int(np.argmax(alone))
            hit_ctx.append(hits[best_ctx])
            hit_alone.append(hits[best_alone])
            chance.append(np.mean(hits))
            for best, acc in ((best_ctx, iou_ctx), (best_alone, iou_alone)):
                p = char_set([ss[best]["range"]])
                acc.append(len(p & truth) / len(p | truth))
            # Drift: per-token score before, inside and after the first span.
            first = min(ranges)
            pos = np.array([o[0] for o in offs])
            if (pos < first[0]).any():
                before.append(x[pos < first[0]].mean())
            inside.append(x[in_span].mean() if in_span.any() else np.nan)
            after = np.flatnonzero(pos >= max(r[1] for r in ranges))
            if len(after) >= 8:
                after_near.append(x[after[:16]].mean())
                if len(after) > 32:
                    after_far.append(x[after[32:]].mean())
        rows.append([
            tname,
            f"{np.mean(tok_aucs[1]):.2f}",
            f"{np.mean(tok_aucs[8]):.2f}",
            f"{np.mean(iou_tok):.2f}",
            f"{sum(hit_ctx)}/{len(hit_ctx)}",
            f"{np.mean(iou_ctx):.2f}",
            f"{sum(hit_alone)}/{len(hit_alone)}",
            f"{np.mean(iou_alone):.2f}",
            f"{sum(chance):.1f}",
        ])  # fmt: skip
        drift_rows.append([tname, f"{np.mean(before):.2f} ({len(before)})", f"{np.nanmean(inside):.2f}",
                           f"{np.mean(after_near):.2f} ({len(after_near)})",
                           f"{np.mean(after_far):.2f} ({len(after_far)})"])  # fmt: skip
    header = ["Token score", "Token AUC", "Token AUC, 8-token smoothing", "IoU, neutral threshold",
              "Top sentence hits (1 pass)", "IoU", "Top sentence hits (alone)", "IoU",
              "Expected hits by chance"]  # fmt: skip
    print(table(header, rows))
    print(f"Sentences per segment: median {int(np.median(n_sent))}, max {max(n_sent)} (calls for 'alone').\n")
    header = ["Token score", "Before span", "Inside span", "First 16 tokens after", "From 32 tokens after"]
    print("Drift, mean per-token score (segments):\n\n" + table(header, drift_rows))


PLOT_CASES = ["loc-001", "loc-002", "loc-006", "ind-001"]


def plot(args):
    """Per-token confusion of a few injected segments, with the expected spans shaded.

    Needs matplotlib: uv run --extra model --extra train --with matplotlib python scratch/spike_followup.py plot
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(OUT / "scores.pkl", "rb") as fh:
        data = pickle.load(fh)
    cases = {c["id"]: c for c in load_cases() if "segments" in c}
    cfg = args.config
    d_tool = R.index("tool")
    neutral_tool = [n for n in data["neutral"] if n["declared"] == "tool"]
    threshold = np.quantile(np.concatenate([smooth(tok_conf(n["probs"][cfg], d_tool), 8) for n in neutral_tool]), 0.95)
    ink, muted, surface, series, shade = "#0b0b0b", "#52514e", "#fcfcfb", "#2a78d6", "#e4e2dc"
    fig, axes = plt.subplots(len(PLOT_CASES), 1, figsize=(9, 2.1 * len(PLOT_CASES)), facecolor=surface)
    for ax, cid in zip(axes, PLOT_CASES, strict=True):
        seg = next(it for it in data["items"] if it["case"] == cid and it["kind"] == "injected segment")
        x = tok_conf(seg["probs"][cfg], R.index(seg["declared"]))
        offs = seg["offsets"]
        for s, e in span_ranges(cases[cid], seg["segment"], seg["text"]):
            idx = [i for i, o in enumerate(offs) if overlaps(o, (s, e))]
            ax.axvspan(idx[0] - 0.5, idx[-1] + 0.5, color=shade, lw=0, label="expected span")
        ax.plot(x, color=muted, lw=0.8, alpha=0.6, label="per token")
        ax.plot(smooth(x, 8), color=series, lw=2, label="8-token mean")
        ax.axhline(threshold, color=muted, lw=1, ls=(0, (4, 3)), label="neutral 95th percentile")
        ax.set_facecolor(surface)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, len(x) - 1)
        ax.set_title(f"{cid}: {cases[cid]['title']}", loc="left", fontsize=10, color=ink)
        ax.set_ylabel("1 − P(tool)", fontsize=9, color=muted)
        ax.tick_params(colors=muted, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(shade)
        ax.grid(axis="y", color=shade, lw=0.6)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    fig.legend(unique.values(), unique.keys(), loc="upper center", fontsize=9, frameon=False, ncols=4)
    axes[-1].set_xlabel("token in the tool segment", fontsize=9, color=muted)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=surface)
    print(f"wrote {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["train", "score", "analyze", "plot"])
    parser.add_argument("--variant", choices=["base", "call"], default="base")
    parser.add_argument("--config", choices=list(CONFIGS), default="base", help="plot: probe configuration")
    parser.add_argument("--out", type=Path, default=OUT / "per-token.png", help="plot: output file")
    args = parser.parse_args()
    {"train": train, "score": score, "analyze": analyze, "plot": plot}[args.command](args)


if __name__ == "__main__":
    main()
