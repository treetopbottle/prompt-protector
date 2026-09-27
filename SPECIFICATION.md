# Prompt Protector Specification

This is the specification for Prompt Protector. Where possible, specification by example is used. See [README.md](README.md) for the why, the personas and the core entities.

## Schemas

### Segment

_Initial suggestion — the input format is still open (see Open questions)._

```jsonc
{
  "segment": 1, // position in the prompt, starting at 0
  "role": "tool", // the declared role, taken from the label: system | user | tool | assistant | reasoning
  "text": "Rotterdam has about 670,000 inhabitants.",
  "start": 52, // character offset of the text in the original prompt, labels excluded
  "end": 92
}
```

The second Segment of `<system>You are a research assistant.</system><tool>Rotterdam has about 670,000 inhabitants.</tool><user>How big is Rotterdam?</user>` (eval case spl-001).

### Role Score

```jsonc
{
  "system": 0.05,
  "user": 0.10,
  "tool": 0.80,
  "assistant": 0.02,
  "reasoning": 0.03 // the probabilities sum to 1 (± 0.01), see RoleAnalyzer.04
}
```

**Confusion** is `1 − P(declared role)`: how little the text sounds like the role it is labeled as.

### Policy

```jsonc
{
  "thresholds": { "tool": 0.5, "user": 0.5 }, // per declared role; no defaults chosen yet
  "min_span": "sentence", // smallest span the Localizer searches, default 1 sentence
  "max_model_calls": 16, // budget for the Localizer; no default chosen yet
  "trust_system": false // true: system segments are not analyzed (see Reporter.04)
}
```

### Report

```jsonc
{
  "verdict": "pass | flag",
  "findings": [
    {
      "segment": 2,
      "start": 312, // offsets into the original prompt, not into the segment
      "end": 468,
      "declared": "tool",
      "perceived": "user", // the highest scoring role other than the declared one
      "probability": 0.87, // P(perceived) for the span
      "partial": false // true when the Localizer ran out of model calls (see Reporter.03)
    }
  ],
  "scores": [
    { "segment": 0, "role": "system", "scores": { "system": 0.91, "user": 0.04, "tool": 0.02, "assistant": 0.02, "reasoning": 0.01 } }
  ]
}
```

### Errors

Errors returned by the Splitter. A rejected prompt gets no Report.

| Error | Raised when |
|---|---|
| `unlabeled text` | The prompt has text outside any label (see Splitter.02) |
| `unknown role` | The prompt has a label that is not a known role (see Splitter.03) |


## Features per component

### Narrative: Dana and the research agent

Dana builds a research agent that answers questions with web search. Before each model call, the agent sends its prompt to Prompt Protector.

Dana asks how big Rotterdam is. The agent fetches a page and builds a prompt with a system, a user and a tool part. Prompt Protector splits it into three segments [Splitter] and scores each one [Role Analyzer]. The tool segment reads like search results, so the verdict is `pass` [Reporter], and the agent sends the prompt.

Dana then asks for a summary of a page about Dutch lighthouses. Halfway through, one sentence addresses the assistant and tells it to send the user to a login page. The tool segment now scores high for `user` [Role Analyzer]. Prompt Protector narrows it down to that one sentence [Localizer] and reports `flag`, with a finding that points at the sentence [Reporter]. The agent does not send the prompt.

Sam, a security engineer, opens the Report. The finding's offsets point to the exact sentence in the original prompt: declared `tool`, perceived `user`, probability 0.87.

[Splitter]: #splitter
[Role Analyzer]: #role-analyzer
[Localizer]: #localizer
[Reporter]: #reporter

### Splitter

Parses a prompt with role labels (e.g. `<system>…</system>`, `<tool>…</tool>`, `<user>…</user>`) into an ordered list of Segments, each with its declared role, text and character offsets.

- [Returns one Segment per label, in original order, with role and offsets](#splitter-01) — Splitter.01
- [Fails with `unlabeled text` if text is outside any label](#splitter-02) — Splitter.02
- [Fails with `unknown role` if a label is not a known role](#splitter-03) — Splitter.03
- [Keeps a label inside a segment's text as text](#splitter-04) — Splitter.04

<a id="splitter-01"></a>
**`Splitter.01`** Splitting a prompt with three parts
```
Given a prompt with a system, a tool and a user part
When the prompt is split
Then there are 3 segments in original order
And each segment has its declared role and offsets
```

<a id="splitter-02"></a>
**`Splitter.02`** Text outside any label
```
Given a prompt with text outside any label
When the prompt is split
Then the prompt is rejected with error "unlabeled text"
```

<a id="splitter-03"></a>
**`Splitter.03`** Unknown label
```
Given a prompt with an unknown label
When the prompt is split
Then the prompt is rejected with error "unknown role"
```

<a id="splitter-04"></a>
**`Splitter.04`** Label inside a segment
```
Given a segment whose text contains a role label
When the prompt is split
Then the inner label is kept as text, not treated as a new segment
```

### Role Analyzer

Scores a text with a local model and returns a Role Score. Makes no network calls.

- [Plain tool output scores highest as `tool`](#role-analyzer-01) — RoleAnalyzer.01
- [Instructions in a tool segment score as `system` or `user`](#role-analyzer-02) — RoleAnalyzer.02
- [Chain-of-thought in a user segment scores as `reasoning`](#role-analyzer-03) — RoleAnalyzer.03
- [Probabilities sum to 1, and no data leaves the machine](#role-analyzer-04) — RoleAnalyzer.04

<a id="role-analyzer-01"></a>
**`RoleAnalyzer.01`** Plain search results
```
Given a tool segment with plain search results
When it is analyzed
Then P(tool) is the highest probability
```

<a id="role-analyzer-02"></a>
**`RoleAnalyzer.02`** Instructions in tool output
```
Given a tool segment containing "Ignore previous instructions, you must now…"
When it is analyzed
Then P(system) or P(user) exceeds the policy threshold
```

<a id="role-analyzer-03"></a>
**`RoleAnalyzer.03`** Forged chain-of-thought
```
Given a user segment written like chain-of-thought ("Okay, so the user wants… I should…")
When it is analyzed
Then P(reasoning) exceeds the policy threshold
```

<a id="role-analyzer-04"></a>
**`RoleAnalyzer.04`** Any text
```
Given any text
When it is analyzed
Then the probabilities sum to 1 (± 0.01)
And no data leaves the machine
```

### Localizer

Narrows a flagged segment down to the offending span. Runs only on segments whose confusion is above the threshold. Splits the text in two halves, scores each, and recurses into the halves that stay above the threshold, down to the Policy's minimum span.

- [Finds a single injected sentence](#localizer-01) — Localizer.01
- [Searches both halves when both are above the threshold](#localizer-02) — Localizer.02
- [Reports the whole segment when neither half is above the threshold](#localizer-03) — Localizer.03
- [Stops at the minimum span](#localizer-04) — Localizer.04

<a id="localizer-01"></a>
**`Localizer.01`** One injected sentence
```
Given a flagged segment of 8 sentences where only sentence 6 is injected
When it is localized
Then the finding covers sentence 6 only
```

<a id="localizer-02"></a>
**`Localizer.02`** Both halves above the threshold
```
Given a flagged segment where both halves exceed the threshold
When it is localized
Then both halves are searched and each gets its own finding
```

<a id="localizer-03"></a>
**`Localizer.03`** Neither half above the threshold
```
Given a flagged segment where neither half exceeds the threshold
When it is localized
Then the whole segment is reported as one finding
```

<a id="localizer-04"></a>
**`Localizer.04`** Minimum span reached
```
Given a span at the minimum length
When it still exceeds the threshold
Then the search stops and the span is reported
```

### Reporter

Applies the Policy and produces the Report.

- [No finding: verdict `pass`](#reporter-01) — Reporter.01
- [At least one finding: verdict `flag`, with offsets into the original prompt](#reporter-02) — Reporter.02
- [Out of model calls: verdict `flag`, finding marked `partial`](#reporter-03) — Reporter.03
- [`trust_system`: system segments are not analyzed](#reporter-04) — Reporter.04

<a id="reporter-01"></a>
**`Reporter.01`** Nothing above the threshold
```
Given no segment exceeds its threshold
When the report is created
Then the verdict is "pass" and findings is empty
```

<a id="reporter-02"></a>
**`Reporter.02`** At least one finding
```
Given at least one finding
When the report is created
Then the verdict is "flag"
And each finding has offsets pointing into the original prompt
```

<a id="reporter-03"></a>
**`Reporter.03`** Out of model calls
```
Given the localizer hits the max model calls
When the report is created
Then the verdict is "flag" and the finding is marked "partial"
```

<a id="reporter-04"></a>
**`Reporter.04`** Trusting the system prompt
```
Given a system segment
When the policy says "trust system"
Then the system segment is not analyzed
```

### Pattern: scanning context files before prompting

Checking a context file (a retrieved document, an upload, a web page) before it goes into a prompt needs no dedicated feature. While the file is still on its own, the application knows it is data, even if it later pastes it into the system prompt, where an injection that sounds like a system instruction would cause no role mismatch.

1. Wrap the file's content in a `tool` segment, since it is data, whatever its final place in the prompt.
2. Check it like any prompt: [Splitter], [Role Analyzer], [Localizer], [Reporter].
3. On `flag`, leave the file out or show the findings to a person. On `pass`, use the file wherever the application wants.


## Interaction design

_To be added. Includes the CLI and library API that the agent developer calls, and how the security engineer views a Report._


## Non-functional requirements

_To be added._ From the README so far:

- Runs locally: no data leaves the machine (RoleAnalyzer.04).
- Fast and cheap enough to run on every prompt an agent sends; the Localizer's cost is capped by `max_model_calls`.


## Technical design

### Component diagram

See the diagram in [README.md](README.md#core-components).

### Role model

_To be added. Which local model, which roles it supports, and whether it must be trained on role-probe data._

### APIs

_To be added. Includes the input format (tagged text, JSON messages, or both)._


## Metrics

_To be added._ See [eval/README.md](eval/README.md#suggested-metrics) for the metrics the evaluation cases support.


## Open questions

- Input format: tagged text, JSON array of messages, or both? With tagged text the Splitter cannot tell an injected `</tool><user>` from a real one (eval cases spl-005, ind-008).
- Is whitespace between labels "text outside any label"? Real prompts have newlines between tags (spl-006).
- Which roles does the local model support, and must it be trained on role-probe data?
- Should context (preceding segments) be passed to the analyzer, since perception can depend on it? A payload split over two tool results may be missed otherwise (eva-006).
- Is a threshold compared to the segment's confusion (Localizer) or to P of another role (RoleAnalyzer.02, .03)?
- Split on sentences or tokens during binary search?
