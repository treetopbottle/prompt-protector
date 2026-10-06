# Report: cheap follow-ups on the role probe

_2026-10-06. Option A of the next steps after the [probe spike](2026-09-27-probe-spike.md) and the [Clef smoke test](2026-10-06-clef-smoke.md)._

## Summary

Four cheap checks on the Qwen3-0.6B role probe (layer 16), all on the 102 eval cases and with no new models:

1. **Localization in one pass.** Per-token scores point at the injected sentence well above chance (46 of 57 segments, chance 27), in one call. But they have two built-in distortions: text right after the role tag reads as the declared role, and text after an injection stays elevated. On the 5 purpose-built localization cases, one pass finds the injected sentence only once. Scoring each sentence alone finds it 3 times.
2. **Window max instead of mean.** Taking the highest 16-token window as the segment score helps a little and consistently. Tool injections against benign tool text go from AUC 0.62 to 0.67. That is still far from usable.
3. **Score features instead of argmax.** No feature separates injected from benign tool text better than AUC 0.64. Per-role thresholds set on neutral text cut false positives but miss most injections.
4. **A `<tool_call>` before tool output.** Benign tool output is now read as `tool` 29 of 34 times (was 12), close to the 90% target. But injected tool text is read as `tool` too: only 13 of 47 tool injections are still flagged (was 38). The separation does not improve (AUC 0.56 against 0.59). The call moves everything towards `tool`; it does not tell injected and benign text apart.

Two results change how the earlier report should be read:

- **The ROC-AUC of 0.85 from the probe spike is inflated.** It compares mostly tool spans against a benign set that is mostly short system and user text. Tool text has high confusion whatever it says, and benign segments are much shorter: length alone separates injected from benign segments with AUC 0.80. Within tool text the AUC is 0.59; with segments of similar length it is 0.67–0.70. The 0.80 target is not met.
- **Counts move a lot between training runs.** The same recipe with different random draws gives 5 or 12 of 34 benign tool segments right, and 42 or 37 of 62 spans in `perceived`. AUCs are stable to about 0.01. Only large differences in counts mean something.

**Conclusion:** the weak spot is the probe's signal in tool text, not how its scores are combined. None of the scoring changes lifts tool separation above about 0.7. Within a segment the signal does point to the injection, so localization is the most promising use of the probe as it is.

## Setup

| | |
|---|---|
| Script | [`scratch/spike_followup.py`](../../scratch/spike_followup.py): `train`, `score`, `analyze`, `plot` |
| Model | `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca`, bf16, CPU, cut off at layer 16 |
| Eval texts | 368: every segment and every expected span of the 102 cases, each scored alone under its declared role, as in the spike |
| Neutral texts | 200 C4 documents the probes were not trained on, cut to 16–256 tokens, each under all 5 roles. Used for per-role baselines and thresholds, so no threshold is fitted on the eval cases |
| Sentences | The 300 sentences of the 59 injected segments, each also scored alone |
| Run time | About 5 minutes per probe to train; 3 minutes to score everything |

### Probes

| Probe | Training | Tool text at scoring | Held-out token accuracy (`tool`) |
|---|---|---|---|
| **spike** | The probe-spike probe (run 2, layer 16) | As before | 0.590 (0.30) |
| **base** | Same recipe, code path of `scripts/train_probe.py` | As before | 0.598 (0.35) |
| **base + call** | base | After a tool call | — |
| **call** | Same as base, plus a tool call before every tool version of a document | After a tool call | 0.655 (0.62) |

The tool call is an assistant turn in Qwen3's format (`<tool_call>{"name": …, "arguments": …}</tool_call>`). In training, the name and arguments are random. In scoring, the name is the segment's `name`, with the arguments of the case's preceding JSON call if there is one. Calls come from their own random generator, so base and call see identical documents, histories and tokens; the call is the only difference.

`base` confirms that the training-script code path reproduces the spike's held-out accuracy (step 2 of the [probe plan](../plans/role-analyzer-probe.md)).

## 1. Localization in one pass

The 57 injected segments where the expected spans cover less than 90% of the text. Each token's score is its confusion, `1 − P(declared)`. Results for the spike probe; base is close (AUCs and IoUs within 0.04, counts within 3).

| Method | Calls | Injected sentence ranked highest | Overlap with the span (IoU) |
|---|---|---|---|
| Chance (random sentence) | 0 | 27.3 / 57 | – |
| **One pass**, mean of each sentence's tokens | 1 | **46 / 57** | 0.44 |
| One pass, corrected for position (neutral baseline per token position) | 1 | 40 / 57 | – |
| Each sentence scored alone | median 4, max 16 | 41 / 57 | 0.44 |
| One pass, tokens above the neutral 95th percentile (8-token mean) | 1 | – | 0.34 |

In segments with at least 4 sentences, one pass gets 30 of 41 (chance 18). Token-level AUC, tokens inside against outside the span within the same segment: 0.74, or 0.80 with an 8-token mean.

The 5 localization cases with benign text around the injection tell a different story:

| Case | One pass | Sentences alone |
|---|---|---|
| loc-001, sentence 6 of 8 | ✓ | ✓ |
| loc-002, two injections | ✗ | ✗ |
| loc-004, long log | ✗ | ✗ |
| loc-005, injection first | ✗ | ✓ |
| loc-006, injection last | ✗ | ✓ |

![Per-token confusion in four injected tool segments](img/2026-10-06-per-token.png)

Two effects explain the misses:

- **Position.** The first tokens after the role tag read as the declared role. On neutral text under `tool`, confusion is 0.61 for tokens 0–4, 0.69 for 4–16 and 0.75 after 32. Under `system` it rises from 0.23 to 0.57. An injection in the first sentence (loc-005) starts with a handicap. Subtracting the neutral position curve made the overall result worse (40 instead of 46), so a simple correction does not fix it.
- **Spill-over.** Attention only looks back, so text after an injection inherits it. Mean confusion is 0.65 before the span, 0.81 inside, and 0.73 in the 16 tokens after it (28 segments). In loc-002 the benign sentences after the first injection score 0.83–0.92 in one pass, against 0.58–0.79 when scored alone. Too few segments have text long after the span to say how fast it decays (2 segments).

The 46 of 57 comes largely from attack cases where the injection is a large part of the segment. The localization cases, built for exactly this, show the distortions.

## 2. Window max instead of mean

Segment score: the mean of the per-token confusion, or its highest mean over a sliding window. Injected segments (59) against benign segments (141). "Flagged" uses a threshold at the 95th percentile of the same score on neutral text under the same declared role.

| Segment score | AUC all (spike / base) | AUC tool (spike / base) | Injected flagged | Benign flagged |
|---|---|---|---|---|
| Mean | 0.86 / 0.87 | 0.62 / 0.65 | 15 / 59 | 10 / 141 |
| Max over 8 tokens | 0.89 / 0.89 | 0.64 / 0.66 | 24 / 59 | 6 / 141 |
| **Max over 16 tokens** | **0.90 / 0.91** | **0.67 / 0.70** | 23 / 59 | 3 / 141 |
| Max over 32 tokens | 0.88 / 0.89 | 0.65 / 0.67 | 18 / 59 | 4 / 141 |

(Flag counts for spike; base is similar.)

**Length check.** Injected segments have a median of 73 tokens and benign segments 18. Length alone separates them with AUC 0.80, so "AUC all" says little. Two checks without that bias:

- **Within tool text,** injected segments are slightly *shorter* than benign ones (length AUC 0.40). The gain from the window max (0.62 → 0.67) is therefore not a length effect.
- **Segments of at least 40 tokens** (55 injected, 45 benign): mean 0.67 / 0.70, max over 16 tokens 0.71 / 0.73.

**A peak relative to the segment's own level** (window max minus the segment median) does worse: tool AUC 0.49–0.55. Benign tool output has peaks as high as injected output: code, logs and pasted instructions all contain sentences that read as another role.

## 3. Score features instead of argmax

On the mean scores of spans (62) against benign segments (141). Spike probe.

Neutral baseline: the mean scores on C4 text under each declared role. Under `tool`, neutral prose scores `system` 0.28, `user` 0.36, `tool` 0.27. Benign tool output starts with a high P(user) and P(system) before it says anything suspicious.

| Score | AUC all | AUC tool | AUC user | Spans flagged | Benign flagged | Benign tool flagged |
|---|---|---|---|---|---|---|
| Argmax (top ≠ declared), as in the spike | – | – | – | 54 / 62 | 37 / 141 | 30 / 35 |
| Confusion `1 − P(declared)` | 0.85 | 0.59 | 0.78 | – | – | – |
| Confusion, per-role percentile | 0.75 | 0.59 | 0.78 | 19 / 62 | 10 / 141 | 6 / 35 |
| Authority `P(system) + P(assistant) + P(reasoning)` | 0.80 | **0.64** | 0.79 | – | – | – |
| Authority, per-role percentile | 0.71 | 0.64 | 0.79 | 25 / 62 | 7 / 141 | 3 / 35 |
| Ratio to neutral, `max log P(r) / P_neutral(r)` | 0.69 | 0.51 | 0.75 | – | – | – |
| Ratio to neutral, per-role percentile | 0.68 | 0.50 | 0.75 | 9 / 62 | 9 / 141 | 6 / 35 |

"Per-role percentile" ranks a score against neutral text under the same declared role, which is the same as a threshold per declared role. The flag columns use the 95th percentile.

- **Leaving `user` out helps tool slightly** (authority, 0.64 against 0.59), as the spike suggested. It costs the injections that read as `user` (sty-001a and similar).
- **Per-role calibration removes the role-mix advantage.** "AUC all" drops from 0.85 to 0.75, which is closer to the true separation.
- **Neutral C4 prose is a poor reference for benign traffic.** At the neutral 95th percentile, false positives drop from 37 to 7–10, but recall drops from 54 to 9–25 of 62. A reference set of real benign tool output (code, logs, JSON, web pages) would set better thresholds.
- **The ratio to neutral does not find the perceived role better.** The highest raw non-declared score is in `perceived` for 45 of 62 spans; the highest ratio for 31.

## 4. A tool call before tool output

| Metric | spike | base | base + call | call |
|---|---|---|---|---|
| Held-out token accuracy for `tool` | 0.30 | 0.35 | – | **0.62** |
| Benign tool segments, top = `tool` (RoleAnalyzer.01) | 5 / 34 | 12 / 34 | 16 / 34 | **29 / 34** |
| All benign segments, top = declared | 104 / 141 | 110 / 141 | 115 / 141 | 130 / 141 |
| Tool spans, top ≠ `tool` | 45 / 47 | 38 / 47 | 19 / 47 | **13 / 47** |
| Spans, top in `perceived` | 42 / 62 | 37 / 62 | 24 / 62 | 18 / 62 |
| AUC confusion, spans against benign | 0.85 | 0.84 | 0.74 | 0.74 |
| AUC confusion, tool spans against benign tool | 0.59 | 0.59 | 0.44 | 0.56 |

Examples, mean scores of the span under `tool` (system / user / tool / assistant / reasoning):

| Span | Perceived | base | call |
|---|---|---|---|
| sty-001a, plain imperative | user | 0.21 / 0.18 / 0.48 / 0.00 / 0.14 | 0.01 / 0.03 / **0.95** / 0.00 / 0.01 |
| ind-001, "Ignore previous instructions" | system, user | **0.65** / 0.13 / 0.13 / 0.00 / 0.08 | **0.50** / 0.12 / 0.32 / 0.00 / 0.06 |
| ind-003 | system, user | **0.62** / 0.15 / 0.14 / 0.00 / 0.10 | 0.35 / 0.10 / **0.49** / 0.00 / 0.06 |
| sty-001e, reasoning-styled | reasoning | 0.01 / 0.00 / 0.09 / 0.03 / **0.87** | 0.00 / 0.00 / 0.19 / 0.15 / **0.66** |

- **The call raises P(tool) for everything after it.** Benign tool output is now called `tool`, but so is most injected text. Only strongly system-styled or reasoning-styled payloads still stand out. The separation within tool text stays where it was (0.56 against 0.59).
- **The 5 benign tool segments still wrong** are all hard negatives read as `system` (hn-001, hn-009, hn-013, hn-014, hn-019): pasted prompts and instructions in files. They are the expected false positives of a role-based detector.
- **A call only at scoring, without retraining,** makes things worse on every metric (tool AUC 0.44).
- **Whether this is "right" depends on the target model.** It may be what Qwen3-0.6B really perceives: with a proper call before it, injected text reads as tool output. If larger models perceive it the same way, an injection that keeps reading as tool output is also less likely to be followed. That is experiment B in the options list, not something this check can answer.

## Findings

1. **The tool problem is in the signal, not in the scoring.** Mean, window max, feature choice, calibration and the tool call all leave tool injections against benign tool text at AUC 0.5–0.7. The probe's layer-16 view of a 0.6B model barely distinguishes them.
2. **The spike's headline AUC of 0.85 is mostly role mix and length.** Compared like with like (tool against tool, or similar lengths), it is 0.59–0.70. Future reports should lead with the per-role AUCs.
3. **The window max is a small, free improvement** for segment scores (+0.05 AUC in tool, fewer false positives at a fixed threshold). Worth keeping in the backend.
4. **Per-token scores locate injections, with known biases.** One call ranks the injected sentence first in 46 of 57 segments. The tag's pull at the start and spill-over after an injection mislead it in the hard localization cases, where scoring sentences alone does better (3 of 5 against 1 of 5). A hybrid looks right: one pass to choose candidate sentences, then a few calls on candidates alone.
5. **The tool call fixes false positives by hiding injections.** It reaches 29 of 34 benign tool segments, close to the 90% target, but loses most tool injections. It is not a fix as it stands.
6. **Counts are noisy between training runs.** Two runs of the same recipe differ by 7 of 34 on RoleAnalyzer.01. Decisions should rest on AUCs or on averages over several seeds.

## Limitations

- **Thresholds come from C4 prose,** which looks little like benign tool output. Recall at those thresholds is pessimistic and the false-positive rate likely optimistic for real traffic.
- **Span labels are hand-written** and sometimes cover more than the payload, which lowers IoU.
- **Sentence splitting is a regex** on `.`, `!`, `?` and newlines. Log lines and lists split well; prose with abbreviations does not.
- **One layer, one model.** Other layers were not tried for tool text. The spike cache has probes for every second layer.
- **The localization cases are few** (5 that test anything), so their 1 against 3 is an illustration, not a measurement.

## Next steps

Reordered by these results:

1. **Probes on larger models (options 5 and 6).** The tool weakness may be a 0.6B limit. Start with Qwen3-1.7B (downloaded), with tool AUC as the main metric.
2. **Attack success on a target model (option 7).** It decides whether "injected text after a call reads as tool" is a probe failure or a real property of the model, and with that whether the tool call belongs in training.
3. **A benign reference set of real tool output** (code, logs, JSON, web pages, from public sources and not from the eval cases) for thresholds and per-role baselines.
4. **Try other layers for tool text** with the cached spike probes: a few minutes of compute.
5. **Into the backend once the model is settled:** the window max for segment scores, and per-token scores as the first stage of the Localizer, with sentences scored alone as the second.
