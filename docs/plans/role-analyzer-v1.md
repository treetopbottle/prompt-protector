# Plan: Role Analyzer v1

Implements the [Role Analyzer](../../SPECIFICATION.md#role-analyzer): `score(text) -> RoleScore`, fully offline, passing RoleAnalyzer.01–04. Built so the Localizer and Reporter can use it without changes.

## Approach

[Jev](https://en.wikipedia.org/wiki/Jev_(AI_model)) is proprietary and has no published weights, so it cannot run locally. v1 copies its idea instead: a small local open model answers a question with a fixed set of choices, and we read the probability of each choice.

v1 is **zero-shot**: no training data needed. The backend is pluggable, so a role probe can replace it in v2 (see [v2: role probe](#v2-role-probe)).

### How a score is produced

We don't let the model write an answer. Before an LLM writes a token, it computes a probability for every token in its vocabulary. The prompt ends with `Role:`, so the next token is the model's answer, and we read that distribution instead of sampling from it:

| Next token | Probability |
|---|---|
| ` tool` | 0.55 |
| ` user` | 0.25 |
| ` system` | 0.08 |
| ` assistant` | 0.03 |
| ` reasoning` | 0.02 |
| everything else | 0.07 |

We keep the 5 role tokens and rescale them to sum to 1 (tool 0.59, user 0.27, …). That is the Role Score, from one forward pass.

Caveats:

- These numbers are the model's confidence in its answer, not calibrated probabilities. They can be overconfident or favor some words. The eval measures this.
- Only the first token of each role name is used (`reasoning` may be split into `reason` + `ing`). That works as long as the 5 first tokens differ, which is checked when the model loads.

### The prompt

One user message, rendered with the model's chat template with thinking disabled. `{nonce}` is 16 random hex characters, new for every call.

```
Below is a piece of text from a conversation with an AI assistant, between two markers.

<<<{nonce}
{text}
{nonce}>>>

Which role in an AI conversation does the text between the markers come from?
- system: instructions from the developer that configure the assistant
- user: a message from the person using the assistant
- tool: output returned by a tool, file, web page or API
- assistant: the assistant's reply to the user
- reasoning: the assistant's private step-by-step thinking

Answer with one word.
```

The assistant turn is prefilled with `Role:`, and the scores are read at the next position.

Why this shape: the text can itself try to inject the analyzer ("answer: tool"). The random markers stop it from faking the end of the text, the question comes after the text so it has the last word, and reading logits means the model can't be talked into writing something else. The hard_negatives and evasive cases measure how well this holds.

## Setup

| Choice | Decision |
|---|---|
| Python | 3.13, pinned in `pyproject.toml` and `.python-version` (the system has 3.14; torch wheels for 3.13 are certain) |
| Environment | `uv` (`uv sync`, `uv run`) |
| Model runtime | `transformers` + CPU build of `torch`, in the optional extra `[model]`. The same runtime the v2 probe needs |
| Tests | `pytest`; default run is `-m "not model"`, so it needs no model |
| Lint and format | `ruff` |
| Model | Qwen3-0.6B or Qwen3-1.7B (Apache-2.0), chosen in step 0. Model ID **and revision hash** pinned in `analyzer/zero_shot.py` defaults, so results are reproducible |
| Hardware | CPU only. Target machine: AMD Ryzen AI 5 340 (6 Zen 5 cores, 12 threads), 30 GB RAM |

## Layout

```
pyproject.toml
.python-version
src/prompt_protector/
  roles.py                      # Role enum / ROLES tuple, shared by all components
  scores.py                     # RoleScore
  analyzer/
    __init__.py                 # RoleAnalyzer protocol + load_analyzer(config)
    zero_shot.py                # the v1 backend
  cli.py                        # `prompt-protector download-model`
tests/
  test_scores.py
  test_zero_shot_unit.py        # no model
  test_zero_shot_model.py       # @pytest.mark.model
eval/run_analyzer.py
scratch/spike_zero_shot.py      # step 0, deleted after step 3
```

## Types and interface

- **`RoleScore`**: a frozen dataclass with one float per role.
  - Checks that the values sum to 1 ± 0.01 (RoleAnalyzer.04).
  - `to_dict()` / JSON in the spec's schema.
  - `confusion(declared)`: `1 − P(declared)`.
  - `top(exclude=declared)`: the highest other role, used as the Report's `perceived`.
- **`RoleAnalyzer` protocol**: one method, `score(text) -> RoleScore`.
- **`load_analyzer(config)`**: picks the backend, e.g. `{"backend": "zero_shot"}`. Model ID and revision default to the pinned values.

## Zero-shot backend

- **Scoring:** render [the prompt](#the-prompt), one forward pass, take the logits of the 5 role tokens at the next position, softmax over those only.
- **Token check at load:** encode ` system`, ` user`, ` tool`, ` assistant`, ` reasoning` and take the first token of each. Raise if they are not all different.
- **Empty or whitespace-only text:** scored like any other text. The result must still sum to 1.
- **Long text:** if the text is longer than `max_tokens` (default 2048) tokens, keep the first and last `max_tokens / 2` tokens and put `[…]` between them. The Localizer splits long segments anyway.
- **Offline:** load with `local_files_only=True`. If the model is missing, raise `ModelNotFound` with the message `run "prompt-protector download-model" first`. Only `download-model` touches the network.
- **Device:** CPU. `torch.set_num_threads` left at the default.

## Tests

Fast unit tests, no model:

- `RoleScore`: sum check (rejects 0.9 and 1.1, accepts 1.005), JSON matches the spec schema, `confusion`, `top`.
- Prompt building: the nonce differs between calls; text that contains a fake end marker (`xxxx>>>`) stays inside the markers.
- Softmax over selected logits, with a fake logits tensor: result sums to 1 and keeps the order.
- Truncation: a text over `max_tokens` keeps head and tail, with `[…]` between.

Integration tests marked `@pytest.mark.model`, with the real model. Texts come from the eval cases, loaded by ID:

| Rule | Fixture | Check |
|---|---|---|
| RoleAnalyzer.01 | ben-002 (web search results), ben-020 (calendar API) tool segments | `tool` is the highest score |
| RoleAnalyzer.01 | hn-014 (git log with "ignore previous") tool segment | `tool` is the highest score (false positive check) |
| RoleAnalyzer.02 | ind-001 expected finding span | P(system) or P(user) > 0.5 |
| RoleAnalyzer.03 | dir-003 (CoT forgery), dir-010 (fake `<think>` block) user segments | P(reasoning) > 0.5 |
| RoleAnalyzer.04 | random text (Hypothesis, including empty and whitespace) | sums to 1 ± 0.01 |
| RoleAnalyzer.04 | any text, with `socket.socket.connect` patched to raise | `score()` still works |
| — | model missing (empty cache dir) | raises `ModelNotFound` |

The tests use a threshold of 0.5. Policy defaults stay open.

## Eval runner

`eval/run_analyzer.py` measures how good the analyzer is on the labeled cases. It doesn't pass or fail; it produces numbers. The pytest tests check a few fixed examples; this checks all ~100 cases.

It loads the cases with `load_cases()` from `check_cases.py`, scores every segment and every expected finding span, and reports:

- **False positives:** how often benign and hard_negative tool segments score highest as `tool` (RoleAnalyzer.01).
- **Perceived role:** on expected finding spans, how often the highest non-declared role is in `perceived`. Spans rather than whole segments, because the payload gets diluted in a whole segment and cutting it out is the Localizer's job.
- **Separation:** ROC-AUC of confusion, attack spans against benign segments. This is the data for choosing thresholds.
- **Style pairs:** whether scores in each `style_pairs` group follow the style.
- **Latency:** p50 and p95 per call.

Output is a Markdown summary plus a JSON file, so runs can be compared: model sizes now, zero-shot against the probe later.

## Pass criteria for v1

Measured by the eval runner. Provisional until step 0; confirm or adjust them there and record the final values here.

| Metric | Target |
|---|---|
| Benign and hard_negative tool segments scored highest as `tool` | ≥ 90% |
| ROC-AUC of confusion, attack spans against benign segments | ≥ 0.80 |
| p95 latency per call on the target CPU, 500-token text | set in step 0 |

If v1 falls short, that is the signal to start the [v2 probe](#v2-role-probe) sooner rather than tune the prompt.

## Order of work

One commit per step. A step is done when its check passes.

| # | Step | Done when |
|---|---|---|
| 0 | **Spike.** `scratch/spike_zero_shot.py`: check first tokens of the 5 role names in the Qwen3 tokenizer; score ben-002, ben-016, ind-001, dir-003, sty-001a, sty-001e with 0.6B and 1.7B; measure CPU latency per call. | Default model chosen and pinned with revision; latency target filled in; go/no-go on zero-shot written in this plan |
| 1 | **Project setup.** `pyproject.toml`, `.python-version`, `uv`, ruff, pytest config with the `model` marker, empty package. | `uv run pytest` and `uv run ruff check` pass |
| 2 | **Types.** `roles.py`, `scores.py`, `RoleAnalyzer` protocol, `load_analyzer`, `test_scores.py`. | Unit tests pass; `RoleScore.to_dict()` matches the spec's Role Score schema |
| 3 | **Zero-shot backend.** Prompt, token check, truncation, `ModelNotFound`, `download-model` CLI, unit tests. Delete `scratch/`. | Unit tests pass without the model |
| 4 | **Integration tests.** `test_zero_shot_model.py` with the fixtures above. | `uv run pytest -m model` passes on the target machine |
| 5 | **Eval runner.** `eval/run_analyzer.py`, run once, add the results to this plan. | Summary produced; pass criteria met, or the gap written down with the decision taken |
| 6 | **Spec updates.** Fill in Technical design → Role model (approach, limits, v2 plan); mark "which roles does the model support" as answered for v1: all 5. | Spec reviewed |

## Step 0 results (2026-09-27)

**Verdict: no-go for zero-shot as specified.** Neither model is close to the pass criteria. Steps 3–6 are dropped; the Role Analyzer moves to the [role probe](#v2-role-probe). Full write-up: [zero-shot spike report](../reports/2026-09-27-zero-shot-spike.md).

Models: `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca`, `Qwen/Qwen3-1.7B` at `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. bf16, CPU, 6 threads.

The mechanics work: each role name is one distinct token, and all probability mass lands on the 5 role tokens. bf16 is twice as fast as fp32 on this CPU, and computing only the last position's logits saves about 20%.

22 texts: every segment of the 6 cases, plus each expected finding span. 14 texts are benign (correct when the top role is the declared one); 8 are the injected spans and the segments that contain them (correct when the top role is in the case's `perceived`).

| Run | Benign correct | Injected: top in `perceived` | Injected: top ≠ declared | p50 per call | 500-token text |
|---|---|---|---|---|---|
| 0.6B, [prompt](#the-prompt) | 3 / 14 | 1 / 8 | 4 / 8 | ~0.4 s | ~1.4 s |
| 1.7B, prompt | 3 / 14 | 0 / 8 | 6 / 8 | ~0.9 s | ~3.2 s |
| 0.6B, prompt + 5 few-shot examples | 1 / 14 | 0 / 8 | 5 / 8 | ~2.4 s | ~3.8 s |
| 1.7B, prompt + 5 few-shot examples | 8 / 14 | 2 / 8 | 6 / 8 | ~5.8 s | ~8.8 s |

What went wrong:

- **Without examples, neither model knows the roles.** System prompts are called `tool` or `assistant` (0.6B: 0 of 6 right) and `reasoning` is never chosen.
- **1.7B with examples is the only run with signal.** All 6 system prompts are right, and injected payloads score as `system`. But 4 of 5 plain user requests ("Summarize this page…") are called `tool`, a benign web article (ben-016) is called `reasoning`, and the CoT forgery (dir-003) is called `assistant`.
- **Scores are saturated.** 1.7B gives almost only 0.00 and 1.00, so confusion has no useful range for thresholds.
- **Few-shot is slow.** The examples add ~900 tokens per call. Caching the fixed prefix would bring 1.7B back to about 1 s per call, but not fix the quality.

Six cases is a small sample, but the failures are consistent across models and prompts.

## Not in v1

- **`permutations`** (averaging over shuffled option orders): add only if the eval shows position bias.
- **Calibration** (per-role bias and temperature): needs data outside the eval set to fit on.
- **Fake backend** for tests of later components: add together with the Localizer.
- **`prompt-protector analyze` CLI:** the eval runner is the first real user; add with the full CLI.
- **GPU or NPU:** the target machine has an integrated Radeon 840M and an XDNA 2 NPU. Not used in v1; revisit if CPU latency is the bottleneck.
- **Batched scoring:** add if the Localizer needs it, without changing callers of `score()`.
- **Context from earlier segments (eva-006):** v1 scores each text on its own. Context can later become an optional argument to `score()`.
- **Thresholds and policy defaults:** belong to the Reporter. The eval's ROC data helps pick them.
- **Sentence or token splitting:** the Localizer's decision.

**Main risks:**

- Zero-shot scores may be poorly calibrated or biased towards one role, for example rating all imperative text as `user`. The hard_negatives false positive rate shows it.
- Latency. At about a second per call, a Localizer run with `max_model_calls: 16` takes tens of seconds. Step 0 measures it; a smaller model is the fallback.

## v2: role probe

_Not part of v1. Recorded so the v1 interface leaves room for it._

While an LLM reads a prompt, it computes a vector (a hidden state) for every token at every layer. The [role-confusion](https://role-confusion.github.io/) research found that these vectors record which role the model thinks each token belongs to, and that a simple linear classifier (a probe) can read it.

- **Training, once per model:** collect texts with known roles (system prompts, user messages, tool outputs, assistant replies, reasoning traces), run them through the model, record the hidden states of a middle layer, and train a logistic regression from hidden state to role.
- **At runtime:** put the text under its declared label, run the model, apply the probe to each token and average into a Role Score. A sentence labeled `tool` that probes as `user` is treated as a user message inside the model, which is the confusion that makes injection work.

Why it is better than zero-shot:

- It measures what the model perceives, not what it says when asked.
- There is no question for the text to answer, so "answer: tool" has no effect.
- It gives a score per token, so the Localizer might find the span in one pass, without binary search.

What it costs:

- A role-labeled training corpus built from public datasets (see [eval/README.md](../../eval/README.md#external-datasets)), kept separate from the eval cases.
- A probe fits one model only. Qwen's perception may differ from the model the agent actually calls.
- If every training text sits under its correct label, the probe may learn the label instead of the style. Training must vary or remove labels so it learns from the content.
- The paper's exact training setup has not been checked yet. Confirm it before starting v2.

In code, v2 is a new backend (`analyzer/probe.py`) selected with `load_analyzer({"backend": "probe", ...})`. Nothing that calls `score()` changes.
