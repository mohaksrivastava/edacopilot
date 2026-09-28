"""Builds the two Section 15.4 eval sets from templates (deterministic,
seeded), rather than hand-labelling free-form sentences one at a time --
a label is correct by construction here, because it is derived from the
same slots used to build the sentence, not asserted separately and
independently of it.

    python evals/build_eval_sets.py

writes `evals/intent_utterances.jsonl` (~200 items, `parse_intent`) and
`evals/question_specs.jsonl` (~150 items, `build_question_spec`), each
line one JSON object: `{"id": ..., "text": ..., "context": {...},
"label": {...}}`. `context` carries anything other than the raw text the
call needs (e.g. `awaiting_answer` for `ANSWER_CLARIFICATION`).

These are ground truth, like the R fixtures (Section 15.1's own framing
for reference values) -- not run against anything live until reviewed.
"""

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path
from typing import Any

SEED = 20260928  # today's date; stated again in docs/eval_label_sample.md

ROOT = Path(__file__).parent
NUMERIC = ["income", "age", "wait_minutes", "satisfaction", "test_score", "weight", "reaction_time"]
GROUPS = ["gender", "region", "clinic", "treatment_group", "department", "cohort"]
TIME_COLS = ["month", "quarter", "week"]
PERSONAS = ["professor", "consultant", "maverick"]
STAGES = ["profile", "missingness", "outliers", "explore", "hypothesis", "export"]
FUNCTIONS = ["welch_t", "mann_whitney", "student_t", "kruskal_wallis", "spearman", "kendall_tau"]
TERMS = [
    "a p-value",
    "Cohen's d",
    "a confidence interval",
    "Levene's test",
    "the Shapiro-Wilk test",
    "effect size",
    "a type I error",
]
BRANCH_NAMES = ["alt-cleaning", "kept-outliers", "sensitivity-check", "reviewer-request"]


def _item(
    idx: int, text: str, label: dict[str, Any], context: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {"id": idx, "text": text, "context": context or {}, "label": label}


# --------------------------------------------------------------------------
# intent_utterances.jsonl
# --------------------------------------------------------------------------


def build_intent_utterances() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    n = 0

    def add(text: str, label: dict[str, Any], context: dict[str, Any] | None = None) -> None:
        nonlocal n
        n += 1
        items.append(_item(n, text, label, context))

    # ASK_ANALYSIS (~50): comparison / association / trend phrasings.
    compare_templates = [
        "is {a} different between {b} groups?",
        "does {a} differ by {b}?",
        "compare {a} across {b}",
        "how does {a} vary with {b}?",
        "check whether {a} depends on {b}",
    ]
    for i, b in enumerate(GROUPS):
        a = NUMERIC[i % len(NUMERIC)]
        template = compare_templates[i % len(compare_templates)]
        add(template.format(a=a, b=b), {"type": "ask_analysis"})
    assoc_templates = [
        "is there a relationship between {a} and {b}?",
        "does {a} correlate with {b}?",
        "how are {a} and {b} related?",
    ]
    for i in range(15):
        a, b = NUMERIC[i % len(NUMERIC)], NUMERIC[(i + 1) % len(NUMERIC)]
        template = assoc_templates[i % len(assoc_templates)]
        add(template.format(a=a, b=b), {"type": "ask_analysis"})
    trend_templates = [
        "has {a} changed over {t}?",
        "is there a trend in {a} across {t}?",
    ]
    for i in range(10):
        a, t = NUMERIC[i % len(NUMERIC)], TIME_COLS[i % len(TIME_COLS)]
        template = trend_templates[i % len(trend_templates)]
        add(template.format(a=a, t=t), {"type": "ask_analysis"})
    for i in range(20):
        a, b = NUMERIC[i % len(NUMERIC)], GROUPS[(i + 2) % len(GROUPS)]
        add(f"is {a} higher in one {b} than another?", {"type": "ask_analysis"})

    # ACCEPT (~20)
    accept_plain = ["yes", "sounds good", "run it", "accept", "go with that", "that works", "do it"]
    for text in accept_plain:
        add(text, {"type": "accept", "persona": None})
    for persona in PERSONAS:
        add(f"go with {persona}", {"type": "accept", "persona": persona})
        add(f"use {persona}'s proposal", {"type": "accept", "persona": persona})
        add(f"I'll take the {persona} option", {"type": "accept", "persona": persona})
        add(f"{persona} sounds right", {"type": "accept", "persona": persona})

    # MODIFY (~10): alpha changes.
    for alpha in ["0.01", "0.1", "0.001", "0.005", "0.02"]:
        add(f"set alpha to {alpha}", {"type": "modify", "alpha": float(alpha)})
        add(f"use alpha = {alpha}", {"type": "modify", "alpha": float(alpha)})

    # OVERRIDE (~15)
    for fn in FUNCTIONS:
        add(f"override to {fn}", {"type": "override", "function": fn})
        add(f"use {fn} anyway", {"type": "override", "function": fn})
    add("I still want to use welch_t", {"type": "override", "function": "welch_t"})

    # EXPLAIN (~20)
    for term in TERMS:
        add(f"what is {term}?", {"type": "explain", "topic": term})
        add(f"explain {term}", {"type": "explain", "topic": term})
    for fn in FUNCTIONS[:6]:
        add(f"why not {fn}?", {"type": "explain", "topic": fn})

    # GOTO_STAGE (~10)
    for stage in STAGES:
        add(f"go to the {stage} stage", {"type": "goto_stage", "target_stage": stage})
    for stage in STAGES[:4]:
        add(f"jump to {stage}", {"type": "goto_stage", "target_stage": stage})

    # UNDO (~8)
    for text in [
        "undo",
        "go back",
        "revert that",
        "undo the last step",
        "take that back",
        "go back one step",
    ]:
        add(text, {"type": "undo"})
    add("undo please", {"type": "undo"})
    add("can you revert", {"type": "undo"})

    # BRANCH (~8)
    for name in BRANCH_NAMES:
        add(f"create a branch called {name}", {"type": "branch", "name": name})
        add(f"branch {name}", {"type": "branch", "name": name})

    # SWITCH_BRANCH (~8)
    for name in BRANCH_NAMES:
        add(f"switch to branch {name}", {"type": "switch_branch", "name": name})
        add(f'switch to the "{name}" branch', {"type": "switch_branch", "name": name})

    # SKIP (~8)
    for text in [
        "skip this",
        "not now",
        "move on",
        "skip",
        "let's move on",
        "not right now",
        "pass on this",
        "skip ahead",
    ]:
        add(text, {"type": "skip"})

    # EXPORT (~8)
    for text in [
        "export this to a notebook",
        "write it up",
        "export the notebook",
        "can you export this",
        "write this up as code",
        "generate the notebook",
        "export",
        "write up the analysis",
    ]:
        add(text, {"type": "export"})

    # SETTINGS (~8)
    for text in [
        "show me the ledger",
        "what have I run so far",
        "show history",
        "what have I done",
        "show me what tests ran",
        "ledger",
        "what's my test history",
        "show the ledger",
    ]:
        add(text, {"type": "settings"})

    # SHOW (~15)
    for i in range(len(NUMERIC)):
        a, b = NUMERIC[i % len(NUMERIC)], NUMERIC[(i + 3) % len(NUMERIC)]
        add(f"plot {a} against {b}", {"type": "show"})
    for i in range(7):
        a = NUMERIC[i % len(NUMERIC)]
        add(f"show me the distribution of {a}", {"type": "show"})

    # ANSWER_CLARIFICATION (~15), awaiting_answer=True context.
    for text, confirmed in [
        ("yes", True),
        ("yeah", True),
        ("correct", True),
        ("that's right", True),
    ]:
        add(
            text,
            {"type": "answer_clarification", "confirmed": confirmed},
            context={"awaiting_answer": True},
        )
    for text, confirmed in [
        ("no", False),
        ("nope", False),
        ("that's wrong", False),
        ("not quite", False),
    ]:
        add(
            text,
            {"type": "answer_clarification", "confirmed": confirmed},
            context={"awaiting_answer": True},
        )
    for word, design in [
        ("paired", "paired"),
        ("independent", "independent"),
        ("repeated", "repeated"),
        ("same people", "paired"),
        ("matched", "paired"),
        ("different people", "independent"),
        ("same subjects", "paired"),
    ]:
        add(
            word,
            {"type": "answer_clarification", "design": design},
            context={"awaiting_answer": True},
        )

    # OTHER (~15): genuinely unclear, below the confidence floor.
    for text in [
        "hmm",
        "I don't know",
        "what about it",
        "ok",
        "?",
        "tell me more",
        "interesting",
        "hm not sure",
        "maybe",
        "I guess",
        "sure whatever",
        "huh",
        "well",
        "right",
        "I see",
    ]:
        add(text, {"type": "other"})

    return items


# --------------------------------------------------------------------------
# question_specs.jsonl
# --------------------------------------------------------------------------


def build_question_specs() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    n = 0

    def add(text: str, label: dict[str, Any]) -> None:
        nonlocal n
        n += 1
        items.append(_item(n, text, label))

    # COMPARE_GROUPS -- independent design, stated explicitly (~15).
    for i in range(15):
        a, b = NUMERIC[i % len(NUMERIC)], GROUPS[i % len(GROUPS)]
        add(
            f"is {a} different between {b} groups? these are unrelated, independent samples",
            {
                "goal": "compare_groups",
                "variables": {"outcome": a, "group": b},
                "design": "independent",
            },
        )

    # COMPARE_GROUPS -- paired design, stated explicitly (~7: one per
    # NUMERIC column, since the template has only that one free slot).
    for a in NUMERIC:
        add(
            f"did {a} change after the intervention, same subjects measured before and after",
            {
                "goal": "compare_groups",
                "variables": {"outcome": a, "group": "period"},
                "design": "paired",
            },
        )

    # COMPARE_GROUPS -- repeated design, stated explicitly (~9).
    for i in range(9):
        a, b = NUMERIC[i % len(NUMERIC)], GROUPS[(i + 1) % len(GROUPS)]
        add(
            f"{a} was measured three times on the same subjects across {b}",
            {
                "goal": "compare_groups",
                "variables": {"outcome": a, "group": b},
                "design": "repeated",
            },
        )

    # COMPARE_GROUPS -- ambiguous design: the correct label is an
    # ambiguity, not a guess (Section 7.1's own rule: design is never
    # inferred from wording alone when the wording does not settle it).
    for i in range(18):
        a, b = NUMERIC[i % len(NUMERIC)], GROUPS[i % len(GROUPS)]
        add(
            f"is {a} different by {b}?",
            {
                "goal": "compare_groups",
                "variables": {"outcome": a, "group": b},
                "design": "ambiguous",
                "note": "wording does not say independent vs. paired/repeated",
            },
        )

    # ASSOCIATION (~15): genuine unique pairs, not a fixed-offset cycle
    # over the same modulus (which repeats every len(NUMERIC) steps).
    for a, b in itertools.islice(itertools.combinations(NUMERIC, 2), 15):
        add(
            f"is there a relationship between {a} and {b}?",
            {"goal": "association", "variables": {"x": a, "y": b}, "design": "unknown"},
        )

    # DISTRIBUTION_FIT (~12)
    for i in range(12):
        a = NUMERIC[i % len(NUMERIC)]
        ref = [50, 100, 0, 3.5, 20][i % 5]
        add(
            f"is the average {a} equal to {ref}?",
            {
                "goal": "distribution_fit",
                "variables": {"outcome": a},
                "design": "unknown",
                "reference_value": ref,
            },
        )

    # EQUIVALENCE (~12) -- shaped like compare_groups plus bounds.
    for i in range(12):
        a, b = NUMERIC[i % len(NUMERIC)], GROUPS[i % len(GROUPS)]
        add(
            f"is {a} practically the same between {b} groups, within a small margin?",
            {
                "goal": "equivalence",
                "variables": {"outcome": a, "group": b},
                "design": "independent",
            },
        )

    # TREND (~12)
    for i in range(12):
        a, t = NUMERIC[i % len(NUMERIC)], TIME_COLS[i % len(TIME_COLS)]
        add(
            f"has {a} changed over {t}?",
            {"goal": "trend", "variables": {"outcome": a, "time": t}, "design": "unknown"},
        )

    # DESCRIBE (~14: two distinct phrasings per NUMERIC column)
    for a in NUMERIC:
        add(
            f"summarize {a}", {"goal": "describe", "variables": {"outcome": a}, "design": "unknown"}
        )
        add(
            f"give me an overview of {a}",
            {"goal": "describe", "variables": {"outcome": a}, "design": "unknown"},
        )

    # MISSINGNESS (~14)
    for a in NUMERIC:
        add(
            f"why is {a} missing so often?",
            {"goal": "missingness", "variables": {"outcome": a}, "design": "unknown"},
        )
        add(
            f"what's going on with the missing values in {a}?",
            {"goal": "missingness", "variables": {"outcome": a}, "design": "unknown"},
        )

    # OUTLIERS (~14)
    for a in NUMERIC:
        add(
            f"are there outliers in {a}?",
            {"goal": "outliers", "variables": {"outcome": a}, "design": "unknown"},
        )
        add(
            f"does {a} have any extreme values?",
            {"goal": "outliers", "variables": {"outcome": a}, "design": "unknown"},
        )

    # TRANSFORM (~14)
    for a in NUMERIC:
        add(
            f"should {a} be log-transformed?",
            {"goal": "transform", "variables": {"outcome": a}, "design": "unknown"},
        )
        add(
            f"does {a} need a transformation before modelling?",
            {"goal": "transform", "variables": {"outcome": a}, "design": "unknown"},
        )

    return items


def main() -> None:
    intents = build_intent_utterances()
    specs = build_question_specs()

    with (ROOT / "intent_utterances.jsonl").open("w", encoding="utf-8") as fh:
        for item in intents:
            fh.write(json.dumps(item) + "\n")

    with (ROOT / "question_specs.jsonl").open("w", encoding="utf-8") as fh:
        for item in specs:
            fh.write(json.dumps(item) + "\n")

    print(f"wrote {len(intents)} intent utterances, {len(specs)} question specs")
    # A read-only sanity check that the fixed seed is actually usable for
    # sampling later, not a claim about this script's own randomness (it
    # has none -- every item here is template-derived, not sampled).
    random.Random(SEED)


if __name__ == "__main__":
    main()
