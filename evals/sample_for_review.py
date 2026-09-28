"""Renders `docs/eval_label_sample.md`: a fixed-seed random sample of 30
items from each eval set, for the maintainer to check before either set
is used for anything (Section 15.4's evals are ground truth like the R
fixtures -- reviewed once, not just generated and trusted).

    python evals/sample_for_review.py

No live API call anywhere in this script.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from build_eval_sets import SEED

ROOT = Path(__file__).parent
OUT = ROOT.parent / "docs" / "eval_label_sample.md"
SAMPLE_SIZE = 30


def _load(name: str) -> list[dict[str, Any]]:
    path = ROOT / name
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _render_intent(item: dict[str, Any]) -> str:
    lines = [f"### #{item['id']}", "", f"**Input:** `{item['text']}`"]
    if item["context"]:
        lines.append(f"**Context:** `{item['context']}`")
    lines.append(f"**Label:** `{item['label']}`")
    return "\n".join(lines)


def _render_spec(item: dict[str, Any]) -> str:
    return "\n".join(
        [f"### #{item['id']}", "", f"**Input:** `{item['text']}`", f"**Label:** `{item['label']}`"]
    )


def main() -> None:
    intents = _load("intent_utterances.jsonl")
    specs = _load("question_specs.jsonl")

    intent_sample = random.Random(SEED).sample(intents, SAMPLE_SIZE)
    spec_sample = random.Random(SEED).sample(specs, SAMPLE_SIZE)

    sections = [
        "# Eval label sample (for review)",
        "",
        f"Random sample of {SAMPLE_SIZE} items from each eval set (Section 15.4), "
        f"drawn with `random.Random({SEED}).sample(...)` -- seed `{SEED}` "
        "(today's date), stated here so the sample is reproducible.",
        "",
        "These are ground truth, like the R fixtures: every label below was "
        "derived from the same template slots used to build its input text "
        "(`evals/build_eval_sets.py`), not asserted separately by hand. "
        "Full sets: `evals/intent_utterances.jsonl` "
        f"({len(intents)} items), `evals/question_specs.jsonl` ({len(specs)} items).",
        "",
        "**No live API call has been made anywhere in building or sampling "
        "these.** This document is for review before any eval run.",
        "",
        "---",
        "",
        f"## Intent parsing sample ({SAMPLE_SIZE} of {len(intents)})",
        "",
    ]
    sections += [_render_intent(item) + "\n" for item in intent_sample]
    sections += [
        "---",
        "",
        f"## QuestionSpec building sample ({SAMPLE_SIZE} of {len(specs)})",
        "",
    ]
    sections += [_render_spec(item) + "\n" for item in spec_sample]

    OUT.write_text("\n".join(sections), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
