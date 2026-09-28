"""Renders the two maintainer-review documents for Section 15.4's eval
sets:

- `docs/eval_template_rules.md`: one row per template in
  `build_eval_sets.py` -- the pattern, one example it produces, and the
  rule that assigns its label. Review this to check the *generating
  logic*, not individual sentences (they're all correct by construction
  from the same slots).
- `docs/eval_realistic_items.md`: every hand-written realistic item
  (`realistic_items.py`), in full, not a sample -- each of those labels
  is a judgment call and needs its own look.

    python evals/render_docs.py

No live API call anywhere in this script.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import build_eval_sets as bes

ROOT = Path(__file__).parent
DOCS = ROOT.parent / "docs"


def _load(name: str) -> list[dict[str, Any]]:
    with (ROOT / name).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def render_template_rules() -> None:
    # Populating TEMPLATE_BLOCKS is a side effect of calling the builders.
    bes.TEMPLATE_BLOCKS.clear()
    bes.build_intent_templates()
    bes.build_question_spec_templates()

    lines = [
        "# Eval template rules (for review)",
        "",
        "One row per template in `evals/build_eval_sets.py`: the pattern it "
        "fills in, one example sentence it produces, and the rule that "
        "assigns that example's label. Every item a template produces is "
        "correct by construction -- the label is derived from the same "
        "slots used to build the sentence -- so reviewing the rule here "
        "covers every sentence the template can produce, not just the one "
        "shown.",
        "",
        "Replaces the earlier `docs/eval_label_sample.md` (a random sample "
        "of template output), which reviewed sentences one at a time "
        "rather than the logic that generates and labels all of them.",
        "",
        "**No live API call has been made anywhere in building or rendering these.**",
        "",
    ]

    for eval_set, title in [
        ("intent", "## Intent parsing templates (`parse_intent`)"),
        ("question_spec", "## QuestionSpec building templates (`build_question_spec`)"),
    ]:
        lines += [
            title,
            "",
            "| Template | Pattern | Example | Label | Rule |",
            "|---|---|---|---|---|",
        ]
        for block in bes.TEMPLATE_BLOCKS:
            if block.eval_set != eval_set:
                continue
            pattern = block.pattern.replace("|", "\\|")
            example_text = block.example_text.replace("|", "\\|")
            label = str(block.example_label).replace("|", "\\|")
            rule = block.rule.replace("|", "\\|")
            lines.append(f"| {block.name} | `{pattern}` | `{example_text}` | `{label}` | {rule} |")
        lines.append("")

    (DOCS / "eval_template_rules.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {DOCS / 'eval_template_rules.md'}")


def _render_realistic_item(item: dict[str, Any]) -> str:
    lines = [f"### #{item['id']}", "", f"**Input:** `{item['text']}`"]
    if item["context"]:
        lines.append(f"**Context:** `{item['context']}`")
    label = dict(item["label"])
    note = label.pop("note", None)
    expected_correction = label.pop("expected_correction", None)
    needs_clarification = label.pop("needs_clarification", False)
    lines.append(f"**Label:** `{label}`")
    lines.append(f"**Needs clarification:** {'yes' if needs_clarification else 'no'}")
    if expected_correction:
        lines.append(f"**Expected correction note:** {expected_correction}")
    if note:
        lines.append(f"**Note:** {note}")
    return "\n".join(lines)


def render_realistic_items() -> None:
    intents = [item for item in _load("intent_utterances.jsonl") if item["source"] == "realistic"]
    specs = [item for item in _load("question_specs.jsonl") if item["source"] == "realistic"]

    lines = [
        "# Eval realistic items (for review)",
        "",
        f"Every hand-written realistic item in both eval sets -- {len(intents)} intent, "
        f"{len(specs)} QuestionSpec, not a sample. Each label here is a judgment call "
        "(typos, vague references, compound requests, requests that push against the "
        "product's own rules, and naturally-phrased ambiguous designs), so each carries "
        "a note explaining the call and a needs-clarification flag marking whether the "
        "correct system behaviour is to ask rather than proceed.",
        "",
        "**No live API call has been made anywhere in building or rendering these.**",
        "",
        "---",
        "",
        f"## Intent parsing: realistic items ({len(intents)})",
        "",
    ]
    lines += [_render_realistic_item(item) + "\n" for item in intents]
    lines += ["---", "", f"## QuestionSpec building: realistic items ({len(specs)})", ""]
    lines += [_render_realistic_item(item) + "\n" for item in specs]

    (DOCS / "eval_realistic_items.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {DOCS / 'eval_realistic_items.md'}")


def main() -> None:
    render_template_rules()
    render_realistic_items()


if __name__ == "__main__":
    main()
