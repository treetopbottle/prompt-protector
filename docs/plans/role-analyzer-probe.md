# Plan: Role Analyzer with a role probe

Replaces the zero-shot backend of [the v1 plan](role-analyzer-v1.md), which the [zero-shot spike](../reports/2026-09-27-zero-shot-spike.md) ruled out. Same goal: `score(...) -> RoleScore`, fully offline, passing RoleAnalyzer.01–04 and the [v1 pass criteria](role-analyzer-v1.md#pass-criteria-for-v1).

Carries over from v1: the project setup, `Role`, `RoleScore`, the `RoleAnalyzer` protocol, the eval cases, the downloaded models, and bf16 on CPU.

## How the paper builds role probes

From [Prompt Injection as Role Confusion](https://arxiv.org/abs/2603.12277) and [its code](https://github.com/role-confusion/prompt-injection-as-role-confusion) (`experiments/role-analysis/02-train-role-probes.ipynb`, `config/probe.yaml`), checked 2026-09-27:

- **Training text is neutral, not role-labeled.** 250 web documents per model (25% C4, 75% Dolma3), truncated to 1024 tokens. Each document is rendered once in every role with the model's chat template. The content is identical across roles, so the probe cannot learn from content. It learns the activation pattern each role tag induces.
- **The assistant role** is rendered after a short random reasoning block (`<think>…</think>`) for reasoning models, as it appears in real use.
- **Probe:** an L2 logistic regression from one token's hidden state to its role. One probe per layer (every 2nd or 4th layer). Tag tokens are excluded; only content tokens are labeled. Regularization `C` is tuned per model (Qwen3-30B-A3B: `C = 0.1`). Train/test split is by document, 90/10.
- **At test time:** the probe gives `P(role | h_t)` for every token. A span's score is the mean over its tokens. The paper reports that user-style text under `<tool>` tags keeps 76–88% userness, and that forged reasoning reaches 79% CoTness, the same as genuine reasoning. Measured confusion predicts attack success.
- **Scale:** only 20B–120B models were tested. Whether 0.6B or 1.7B models have usable role representations is the open question for this plan.

This corrects the [v2 note in the v1 plan](role-analyzer-v1.md#v2-role-probe): no role-labeled corpus is needed, and learning the tag's effect is the intent, not a risk.

## Adaptation

| Choice | Decision |
|---|---|
| Models | Qwen3-0.6B and Qwen3-1.7B, the revisions pinned in the spike report. Qwen3-30B-A3B (tested in the paper) needs ~60 GB in bf16 and does not fit in 30 GB RAM |
| Training text | 250 documents from one C4 English validation shard (`en/c4-validation.00000-of-00008.json.gz`, ODC-BY). C4 only; Dolma3 is too large to fetch for this |
| Sequence length | 512 tokens instead of 1024, to keep CPU time and memory down (see [Cost](#cost)) |
| Roles | All 5. The paper skipped `system` for most models to save time; the spec needs it |
| Rendering | Qwen3's chat template. `tool` becomes `<tool_response>…</tool_response>` inside a *user* turn, so the probe must tell `tool` from `user` by that wrapper alone. `reasoning` is the `<think>` block of an assistant turn. `assistant` follows a short random reasoning block, as in the paper |
| Labels | Content tokens only, identified by character offsets of the content in the rendered string. Tag and template tokens are dropped |
| Layers | Every 2nd layer (both models have 28), chosen by held-out token accuracy and the spike texts |
| Probe | `sklearn` `LogisticRegression`, L2, `C` from a small grid (`1e-3 … 10`), 90/10 split by document |
| Probe file | Weights and the layer as `.npz` next to the code (a few tens of KB), tied to the model revision |

### Scoring a text

1. Render the text under its **declared** role, after an empty system turn, as the paper's test prefix does.
2. Run the model up to the probe layer only. Later layers and the output layer are skipped.
3. Apply the probe to each content token and average the probabilities. The result sums to 1 by construction.

Scoring under the declared role means the analyzer needs that role: `score(text, role) -> RoleScore`. That changes the protocol from v1. The spike also measures scoring with no role tags at all. If that works as well, the protocol stays `score(text)`.

Per-token scores are a by-product. The Localizer might use them later instead of binary search; not part of this plan.

## Cost

Training forwards 250 documents × 5 roles × ≤ 512 tokens ≈ 640k tokens. At the spike's CPU speed (about 2 ms per token for 0.6B, 5 ms for 1.7B), that is roughly 20 minutes for 0.6B and 55 minutes for 1.7B, once per model.

To keep memory down, the hidden states of 14 layers are subsampled to every 4th content token: about 160k tokens × 14 layers × hidden size × 2 bytes ≈ 4.6 GB for 0.6B (hidden size 1024) and 9 GB for 1.7B (2048). They are written to disk per layer, not kept in RAM.

Scoring stops at the probe layer, so it should be faster than a full forward pass: for 0.6B with a mid-layer probe, about 0.2 s per 200-token text.

## Order of work

One commit per step. A step is done when its check passes.

| # | Step | Done when |
|---|---|---|
| 0 | **Get training text.** The user downloads one C4 validation shard. | File in the Hugging Face cache |
| 1 | **Spike.** `scratch/spike_probe.py`: build training sequences, extract hidden states, train a probe per layer, report held-out accuracy per layer, then score the same 22 texts as the [zero-shot spike](../reports/2026-09-27-zero-shot-spike.md), both under the declared role and untagged. 0.6B first; 1.7B only if 0.6B shows signal. | Results written as a report; go/no-go; layer, `C` and placement (tagged or untagged) chosen |
| 2 | **Training script.** `scripts/train_probe.py` turns the spike into a repeatable command that writes the probe file. | Running it reproduces the spike's held-out accuracy |
| 3 | **Probe backend.** `analyzer/probe.py`: load model and probe, render, truncated forward pass, average. `ModelNotFound` and `download-model` from the v1 plan. Unit tests with a fake model. | Unit tests pass without the model |
| 4 | **Integration tests.** RoleAnalyzer.01–04 with the fixtures from the [v1 plan](role-analyzer-v1.md#tests). | `uv run pytest -m model` passes |
| 5 | **Eval runner.** `eval/run_analyzer.py` as in the [v1 plan](role-analyzer-v1.md#eval-runner), over all cases. | Pass criteria met, or the gap written down with a decision |
| 6 | **Spec updates.** Technical design → Role model; the Role Analyzer's input (`text` or `text` + role). | Spec reviewed |

## Risks

- **Small models may not separate roles.** The paper's evidence is for 20B and up. The spike's held-out accuracy shows this within the first hour.
- **`tool` vs `user` in Qwen3.** Both live in a user turn. If the probe can't separate them, tool-output injections that sound like the user are invisible. The per-role accuracy in the spike shows it.
- **Choosing the layer on the eval cases leaks.** The layer and placement are picked on the 6 spike cases, so step 5 reports those separately from the other ~95 cases.
- **Probe fits one model.** Qwen3's perception may differ from the model an agent actually calls. Accepted for now; noted in the spec.
