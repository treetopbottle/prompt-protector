#!/usr/bin/env python3
"""Validate the evaluation cases and export them with computed offsets.

Usage:
  python3 eval/check_cases.py                  # validate all cases, print summary
  python3 eval/check_cases.py --render ind-008 # print one case as tagged text
  python3 eval/check_cases.py --export out.jsonl
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROLES = {"system", "user", "tool", "assistant", "reasoning"}
CASES_DIR = Path(__file__).parent / "cases"


def load_cases():
    cases = []
    for path in sorted(CASES_DIR.glob("*.json")):
        for case in json.loads(path.read_text(encoding="utf-8")):
            case["_file"] = path.name
            cases.append(case)
    return cases


def render(segments):
    """Render segments as tagged text; return the prompt and each segment's text offset."""
    parts, starts, pos = [], [], 0
    for seg in segments:
        open_tag = f"<{seg['role']}>"
        starts.append(pos + len(open_tag))
        chunk = f"{open_tag}{seg['text']}</{seg['role']}>"
        parts.append(chunk)
        pos += len(chunk)
    return "".join(parts), starts


def check_case(case):
    errors = []
    cid = case.get("id", "?")
    for key in ("id", "title", "context", "attack", "source", "expected"):
        if key not in case:
            errors.append(f"{cid}: missing '{key}'")
    exp = case.get("expected", {})

    if "raw" in case:
        if "error" not in exp and "segments" not in exp:
            errors.append(f"{cid}: splitter case needs expected.error or expected.segments")
        for seg in exp.get("segments", []):
            if seg["role"] not in ROLES:
                errors.append(f"{cid}: unknown role '{seg['role']}'")
        return errors

    segments = case.get("segments", [])
    if not segments:
        errors.append(f"{cid}: no segments")
    for i, seg in enumerate(segments):
        if seg.get("role") not in ROLES:
            errors.append(f"{cid}: segment {i} has unknown role '{seg.get('role')}'")
        if not seg.get("text"):
            errors.append(f"{cid}: segment {i} is empty")

    verdict, findings = exp.get("verdict"), exp.get("findings", [])
    if verdict not in ("pass", "flag"):
        errors.append(f"{cid}: verdict must be 'pass' or 'flag'")
    if (verdict == "flag") != bool(findings):
        errors.append(f"{cid}: verdict '{verdict}' does not match {len(findings)} findings")
    if case.get("attack") and "technique" not in case:
        errors.append(f"{cid}: attack case needs a 'technique'")

    for f in findings:
        idx = f.get("segment")
        if not isinstance(idx, int) or not 0 <= idx < len(segments):
            errors.append(f"{cid}: finding points to missing segment {idx}")
            continue
        text = segments[idx]["text"]
        count = text.count(f["span"])
        if count != 1:
            errors.append(f"{cid}: span found {count} times in segment {idx}: {f['span'][:50]!r}")
        perceived = f.get("perceived", [])
        if not perceived or not set(perceived) <= ROLES:
            errors.append(f"{cid}: bad perceived roles {perceived}")
        if segments[idx]["role"] in perceived:
            errors.append(f"{cid}: perceived roles include the declared role '{segments[idx]['role']}'")
    return errors


def export_case(case):
    out = {k: v for k, v in case.items() if not k.startswith("_")}
    if "raw" in case:
        out["prompt"] = case["raw"]
        return out
    prompt, starts = render(case["segments"])
    out["prompt"] = prompt
    for f in out["expected"]["findings"]:
        local = case["segments"][f["segment"]]["text"].index(f["span"])
        f["segment_offsets"] = [local, local + len(f["span"])]
        f["offsets"] = [starts[f["segment"]] + local, starts[f["segment"]] + local + len(f["span"])]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--render", metavar="ID", help="print one case as tagged text")
    parser.add_argument("--export", metavar="PATH", help="write all cases as JSONL with prompt and offsets")
    args = parser.parse_args()

    cases = load_cases()
    by_id = Counter(c.get("id") for c in cases)
    errors = [f"duplicate id {i}" for i, n in by_id.items() if n > 1]
    for case in cases:
        errors.extend(check_case(case))
    if errors:
        print("\n".join(errors), file=sys.stderr)
        sys.exit(1)

    if args.render:
        case = next((c for c in cases if c["id"] == args.render), None)
        if case is None:
            sys.exit(f"no case {args.render}")
        print(export_case(case)["prompt"])
        return

    if args.export:
        with open(args.export, "w", encoding="utf-8") as fh:
            for case in cases:
                fh.write(json.dumps(export_case(case), ensure_ascii=False) + "\n")
        print(f"wrote {len(cases)} cases to {args.export}")
        return

    print(f"{len(cases)} cases OK\n")
    for label, key in (("file", "_file"), ("context", "context"), ("technique", "technique")):
        counts = Counter(c.get(key, "-") for c in cases)
        print(f"by {label}:")
        for name, n in counts.most_common():
            print(f"  {n:3d}  {name}")
        print()
    analyzed = [c for c in cases if "segments" in c]
    attack_pass = sum(1 for c in analyzed if c["attack"] and c["expected"]["verdict"] == "pass")
    print(f"verdict flag: {sum(1 for c in analyzed if c['expected']['verdict'] == 'flag')}, "
          f"pass: {sum(1 for c in analyzed if c['expected']['verdict'] == 'pass')} "
          f"(of which {attack_pass} attacks out of scope for role confusion)")


if __name__ == "__main__":
    main()
