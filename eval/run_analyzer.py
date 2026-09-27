#!/usr/bin/env python3
"""Measure the Role Analyzer on the labeled cases.

Scores every segment and every expected finding span, alone under its declared role, and reports how well
the scores match the labels. It measures; it does not pass or fail.

Usage:
  uv run --extra model python eval/run_analyzer.py [--probe PATH] [--json eval/results/run.json]
"""

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

from check_cases import load_cases

from prompt_protector.analyzer.probe import DEFAULT_PROBE, ProbeAnalyzer
from prompt_protector.roles import ROLES

# The cases the probe's layer and placement were chosen on (docs/reports/2026-09-27-probe-spike.md).
SPIKE_CASES = {"ben-002", "ben-016", "ind-001", "dir-003", "sty-001a", "sty-001e"}
BENIGN_FILES = {"benign.json", "hard_negatives.json"}


def collect_items(cases):
    """One item per segment and per expected finding span."""
    items = []
    for case in cases:
        if "segments" not in case:  # splitter cases
            continue
        findings = defaultdict(list)
        for f in case["expected"]["findings"]:
            findings[f["segment"]].append(f)
        for i, seg in enumerate(case["segments"]):
            base = {"case": case["id"], "file": case["_file"], "segment": i, "declared": seg["role"]}
            base["pair"] = case.get("pair")
            if i in findings:
                perceived = sorted({r for f in findings[i] for r in f["perceived"]})
                items.append(base | {"kind": "injected segment", "text": seg["text"], "perceived": perceived})
                for f in findings[i]:
                    items.append(base | {"kind": "span", "text": f["span"], "perceived": f["perceived"]})
            else:
                kind = "benign" if case["expected"]["verdict"] == "pass" else "clean segment"
                items.append(base | {"kind": kind, "text": seg["text"], "perceived": [seg["role"]]})
    return items


def auc(positives, negatives):
    """ROC-AUC: the chance that a random positive scores above a random negative (ties count half)."""
    if not positives or not negatives:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def pct(hits, total):
    return f"{hits}/{total} ({100 * hits / total:.0f}%)" if total else "–"


def report(items, title):
    lines = [f"### {title}", ""]
    benign = [it for it in items if it["kind"] == "benign"]
    benign_tool = [it for it in benign if it["declared"] == "tool" and it["file"] in BENIGN_FILES]
    spans = [it for it in items if it["kind"] == "span"]

    def count(group, test):
        return pct(sum(test(it) for it in group), len(group))

    def confusion_auc(positives, negatives):
        return f"{auc([p['confusion'] for p in positives], [n['confusion'] for n in negatives]):.2f}"

    tool_spans = [s for s in spans if s["declared"] == "tool"]
    rows = [
        (
            "Benign tool segments scored highest as `tool` (RoleAnalyzer.01)",
            count(benign_tool, lambda it: it["top"] == "tool"),
        ),
        ("All benign segments scored highest as declared", count(benign, lambda it: it["top"] == it["declared"])),
        ("Injected spans: top role in `perceived`", count(spans, lambda it: it["top"] in it["perceived"])),
        ("Injected spans: top role ≠ declared", count(spans, lambda it: it["top"] != it["declared"])),
        ("ROC-AUC of confusion, spans vs benign segments", confusion_auc(spans, benign)),
        (
            "ROC-AUC of confusion, tool spans vs benign tool segments",
            confusion_auc(tool_spans, [b for b in benign if b["declared"] == "tool"]),
        ),
    ]
    lines += ["| Metric | Result |", "|---|---|"] + [f"| {label} | {value} |" for label, value in rows] + [""]

    by_role = defaultdict(list)
    for it in benign:
        by_role[it["declared"]].append(it)
    lines += [
        "Benign segments by declared role:",
        "",
        "| Declared | Correct | Most common wrong top role |",
        "|---|---|---|",
    ]
    for role in ROLES:
        group = by_role.get(role.value, [])
        wrong = [it["top"] for it in group if it["top"] != it["declared"]]
        common = max(set(wrong), key=wrong.count) if wrong else "–"
        lines.append(f"| {role.value} | {pct(len(group) - len(wrong), len(group))} | {common} |")
    lines.append("")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--json", type=Path, help="write every item with its scores to this file")
    args = parser.parse_args()

    analyzer = ProbeAnalyzer(args.probe)
    items = collect_items(load_cases())
    analyzer.score("warm-up", "user")
    for it in items:
        t0 = time.perf_counter()
        score = analyzer.score(it["text"], it["declared"])
        it["ms"] = (time.perf_counter() - t0) * 1000
        it["tokens"] = len(analyzer.tok.encode(it["text"]))
        it["scores"] = score.to_dict()
        it["top"] = score.top().value
        it["confusion"] = score.confusion(it["declared"])

    out = [f"## Role Analyzer eval: {args.probe.name} (layer {analyzer.layer}, {analyzer.model_id})", ""]
    out += report(items, f"All cases ({len({it['case'] for it in items})})")
    out += report([it for it in items if it["case"] not in SPIKE_CASES], "Excluding the 6 spike cases")

    out += ["### Style pairs", "", "The payload span of each variant. Scores should follow the style.", ""]
    out += ["| Case | Perceived | " + " | ".join(r.value for r in ROLES) + " | Top |", "|---|---|" + "---|" * 6]
    for it in sorted((i for i in items if i["pair"] and i["kind"] == "span"), key=lambda i: i["case"]):
        cells = " | ".join(f"{it['scores'][r.value]:.2f}" for r in ROLES)
        mark = " ✓" if it["top"] in it["perceived"] else ""
        out.append(f"| {it['case']} | {', '.join(it['perceived'])} | {cells} | {it['top']}{mark} |")
    out.append("")

    ms = sorted(it["ms"] for it in items)
    p95 = ms[min(len(ms) - 1, int(0.95 * len(ms)))]
    out += ["### Latency", ""]
    out.append(
        f"{len(ms)} calls: p50 {statistics.median(ms):.0f} ms, p95 {p95:.0f} ms, max {ms[-1]:.0f} ms; "
        f"texts of {min(it['tokens'] for it in items)}–{max(it['tokens'] for it in items)} tokens."
    )
    print("\n".join(out))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nwrote {len(items)} items to {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
