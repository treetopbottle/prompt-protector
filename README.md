# Prompt Protector

Detects prompt injection by finding text that pretends to be a role it is not.

## Why?

**Mission:** Stop prompt injection before a prompt reaches the LLM.

**Vision:** Every prompt an agent sends is checked for [role confusion](https://role-confusion.github.io/), locally, fast and cheaply.

LLMs don't trust role tags alone. They also judge a role by its writing style. Tool output that *sounds* like a system prompt or like the model's own reasoning gets treated as one. Prompt Protector measures how each part of a prompt will be perceived and flags the parts that don't match their label.

## Who?

| Persona | Needs |
|---|---|
| **Agent developer** | Checks prompts before sending them. Wants a simple API and a clear verdict. |
| **Security engineer** | Investigates flagged prompts. Wants to see exactly which text is suspicious and why. |
| **Platform operator** | Runs the tool on their own hardware. Wants low latency, no data leaving the machine. |

## What?

| Entity | Description |
|---|---|
| **Prompt** | The full input, with labeled parts. |
| **Role** | A label: `system`, `user`, `tool`, `assistant`, `reasoning`. |
| **Segment** | One labeled part of a prompt: a declared role plus its text. |
| **Role Score** | Probability per role that the model perceives a text as that role. |
| **Finding** | A span of text whose perceived role differs from its declared role. |
| **Report** | The verdict for a prompt: `pass` or `flag`, plus all findings. |
| **Policy** | Thresholds and settings that decide when a score becomes a finding. |

## Core components

1. **Splitter** – Parses a prompt into segments.
2. **Role Analyzer** – Scores a text with a local typed model (e.g. a [Jev](https://en.wikipedia.org/wiki/Jev_(AI_model))-like model) and returns role probabilities as JSON.
3. **Localizer** – Uses binary search to narrow a suspicious segment down to the offending span.
4. **Reporter** – Applies the policy and produces the report.

```
Prompt ─▶ Splitter ─▶ Segments ─▶ Role Analyzer ─▶ Role Scores
                                        ▲                │ mismatch?
                                        └── Localizer ◀──┘
                                                 │
                                                 ▼
                                             Reporter ─▶ Report (JSON)
```

See [SPECIFICATION.md](SPECIFICATION.md) for features and rules.
