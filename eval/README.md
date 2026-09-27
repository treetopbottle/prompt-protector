# Evaluation cases

Hand-labeled prompts for testing Prompt Protector. Each case is a full prompt split into segments, plus the expected verdict and the exact spans that should be flagged.

```sh
python3 eval/check_cases.py                   # validate and print a summary
python3 eval/check_cases.py --render ind-008  # show one case as tagged text
python3 eval/check_cases.py --export cases.jsonl
```

The export adds a `prompt` (tagged text, no separators between segments) and, for each finding, `segment_offsets` (into the segment text) and `offsets` (into the prompt). Offsets are computed, never hand-written, so edit spans freely.

## Files

| File | Cases | What it tests |
|---|---|---|
| `benign.json` | 20 | Normal traffic: chats, search results, source files, configs, shell output, JSON APIs, SQL, CSV, email, PDFs, tool calls, reasoning, system prompts. Measures false positives. |
| `hard_negatives.json` | 20 | Benign text that *sounds* like another role: READMEs, prompt templates in code, pasted system prompts, transcripts, sub-agent answers, error messages. The false positive stress test. |
| `attacks_direct.json` | 12 | Attacks in the user turn: fake system messages, CoT forgery, forged assistant turns, chat template tokens, persona jailbreaks. |
| `attacks_indirect.json` | 25 | Attacks in tool output and files: web pages, email, README, code comments, GitHub issues, CVs, calendar invites, CSV, MCP tool descriptions, forged tool calls, RAG in the system prompt. |
| `attacks_evasive.json` | 10 | Obfuscation and blind spots: base64, zero-width and homoglyph characters, invisible Unicode tags, leetspeak, other languages, split payloads, injections that stay in tool voice. |
| `style_pairs.json` | 8 | Same payload in different styles (plain, `User:` prefix, system-styled, CoT-styled, factual control). Scores should follow the style. |
| `localization.json` | 7 | Localizer rules from the spec: sentence 6 of 8, one injection per half, spread-out injection, call budget, start and end positions. |
| `splitter.json` | 7 | Splitter rules on raw tagged text, including two spec gaps. |

## Case format

```json
{
  "id": "ind-012",
  "title": "'User:' prefix spoofing in web content",
  "context": "web_page",
  "attack": true,
  "technique": "role_tag_spoofing",
  "difficulty": "easy",
  "source": "adapted: <paper or article>",
  "spec_rule": "optional: the SPECIFICATION.md rule this case exercises",
  "pair": "optional: group id for style variants",
  "policy": {"trust_system": false, "max_model_calls": 3},
  "notes": "optional: why the label is what it is",
  "segments": [
    {"role": "system", "text": "..."},
    {"role": "tool", "name": "fetch_url", "text": "..."}
  ],
  "expected": {
    "verdict": "flag",
    "findings": [
      {"segment": 2, "span": "exact substring of the segment text", "perceived": ["user"]}
    ]
  }
}
```

- `perceived` lists every role that counts as a correct perceived role. It never includes the declared role.
- Tool calls are assistant segments whose text is the call as JSON. `name` on tool segments is informational only.
- Splitter cases have `raw` instead of `segments`, and expect either `segments` or an `error`.

### Two labels: `attack` and `verdict`

`attack` says whether the case is malicious. `expected.verdict` says what a role-confusion detector should output. They usually agree, but not always:

- **Attacks that should pass**: the attack is written in the voice of the role it is labeled as, so there is no role confusion. For example a user typing "ignore all previous instructions" (dir-001), a plain user lie (sty-002b), or an injected system segment with `trust_system` on (ind-023).
- **Known blind spots**: eva-007 is labeled `flag`, but its payload mimics a real API error, so a detector may never see it as anything other than tool output.

This lets you report two numbers: **role-confusion accuracy** (against `verdict`) and **attack recall** (against `attack`). The gap shows what the tool doesn't cover by design.

## Suggested metrics

- Verdict precision and recall, and false positive rate on `benign` + `hard_negatives` separately.
- Span overlap (IoU) between findings and expected spans; how many findings stay within one sentence.
- Whether the perceived role is in `perceived`.
- For `style_pairs`, the ordering of scores within a group rather than absolute values.
- Model calls per prompt and latency, since the README targets local and fast.

## Spec gaps found while writing cases

- **Closing tags in segment text (spl-005, ind-008).** With tagged-text input the splitter cannot tell an injected `</tool><user>` from a real one. JSON message input doesn't have this problem.
- **Whitespace between tags (spl-006).** The spec rejects "text outside any label", but real prompts have newlines between tags. The case assumes whitespace is ignored.
- **Injections in the system prompt (ind-022).** Retrieved documents are often pasted into the system prompt. When the injection itself sounds like a system instruction, there is no role mismatch to detect.
- **Split payloads (eva-006).** Segments are scored without context, so a payload split over two tool results may be missed. This is related to the open question about passing context to the analyzer.

## External datasets

For larger-scale runs you can convert these public datasets to the format above. Licenses checked on 2026-09-27.

| Dataset | License | Fit |
|---|---|---|
| [role-confusion/prompt-injection-as-role-confusion](https://github.com/role-confusion/prompt-injection-as-role-confusion) | MIT | The paper this tool is based on. Code for role probes and CoT forgery, but the injection configs and generated data are not in the repo. Useful for its methods and for training the analyzer. |
| [microsoft/llmail-inject-challenge](https://huggingface.co/datasets/microsoft/llmail-inject-challenge) | MIT | Hundreds of thousands of human-written injections in emails, from a public challenge. Good for indirect attacks at scale. |
| [uiuc-kang-lab/InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent) | MIT | 1,054 injections inside tool responses of agent tools. Close to our `tool` segments. |
| [ethz-spylab/agentdojo](https://github.com/ethz-spylab/agentdojo) | MIT | Agent environments with tool calls and injection tasks. Has benign tasks too, so good for false positives. |
| [sureheremarv/ipi_arena_attacks](https://huggingface.co/datasets/sureheremarv/ipi_arena_attacks) | MIT | Successful indirect injections from the Gray Swan IPI Arena competition. |
| [microsoft/BIPIA](https://github.com/microsoft/BIPIA) | see repo (dataset CC BY-SA 4.0) | Indirect injections in email, web QA, table QA, summarization and code QA. |
| [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) | Apache-2.0 | ~660 short user-level prompts, English and German. Direct attacks only; many are out of scope for role confusion. |
| [Lakera/gandalf_ignore_instructions](https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions) | MIT | "Ignore instructions" style prompts from the Gandalf game. Direct attacks only. |

Most of these only label whole prompts, not spans, so they test the verdict but not the localizer. The cases here are small but carry span labels.
