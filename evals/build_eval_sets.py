"""Builds the two Section 15.4 eval sets from two sources per set:

- **template** items (this file): generated from slot-filled patterns, so
  a label is correct by construction -- derived from the same slots used
  to build the sentence, not asserted separately from it. Good for
  coverage of every `IntentType`/`Goal`; nothing like what a person
  actually types.
- **realistic** items (`realistic_items.py`): hand-written, covering
  typos, vague references, compound requests, requests that push against
  the product's own rules, and naturally-phrased ambiguous designs. Every
  label there is a judgment call, carrying its own `note`.
- **gold** items (not yet built): reserved for a future round of
  maintainer-labelled items drawn from real usage.

    python evals/build_eval_sets.py

writes `evals/intent_utterances.jsonl` and `evals/question_specs.jsonl`,
each line one JSON object: `{"id", "text", "context", "label", "source"}`.
`context` carries anything other than the raw text the call needs (e.g.
`awaiting_answer` for `ANSWER_CLARIFICATION`). Every `label` carries
`needs_clarification` -- whether the correct system behaviour is to ask
rather than proceed -- so a future eval report can score that dimension,
and every item's top-level `source` is `"template"`, `"realistic"`, or
(reserved) `"gold"`, so a report can give metrics per source as well as
overall.

These are ground truth, like the R fixtures (Section 15.1's own framing
for reference values) -- not run against anything live until reviewed.
`evals/render_docs.py` renders the maintainer-facing review documents:
`docs/eval_template_rules.md` (one row per template, its rule, and an
example) and `docs/eval_realistic_items.md` (every realistic item, since
each of those labels is a judgment call, not a sample of them).
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from realistic_items import REALISTIC_INTENT_ITEMS, REALISTIC_QUESTION_SPEC_ITEMS

SEED = 20260928  # today's date; stated again in the rendered docs

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


@dataclass
class TemplateBlock:
    """One documented template: what it generates and why the label
    follows (`docs/eval_template_rules.md`'s rows)."""

    eval_set: str  # "intent" or "question_spec"
    name: str
    pattern: str
    rule: str
    example_text: str = ""
    example_label: dict[str, Any] = field(default_factory=dict)


TEMPLATE_BLOCKS: list[TemplateBlock] = []


def _register(
    eval_set: str,
    name: str,
    pattern: str,
    rule: str,
    example_text: str,
    example_label: dict[str, Any],
) -> None:
    TEMPLATE_BLOCKS.append(
        TemplateBlock(eval_set, name, pattern, rule, example_text, example_label)
    )


def _item(
    idx: int,
    text: str,
    label: dict[str, Any],
    context: dict[str, Any] | None,
    source: str,
) -> dict[str, Any]:
    return {"id": idx, "text": text, "context": context or {}, "label": label, "source": source}


# --------------------------------------------------------------------------
# intent_utterances.jsonl -- template items
# --------------------------------------------------------------------------


def build_intent_templates() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    def add(
        text: str, label: dict[str, Any], context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        label = {"needs_clarification": False, **label}
        item = _item(0, text, label, context, "template")
        items.append(item)
        return item

    # ASK_ANALYSIS: comparison phrasing.
    compare_templates = [
        "is {a} different between {b} groups?",
        "does {a} differ by {b}?",
        "compare {a} across {b}",
        "how does {a} vary with {b}?",
        "check whether {a} depends on {b}",
    ]
    first = None
    for i, b in enumerate(GROUPS):
        a = NUMERIC[i % len(NUMERIC)]
        template = compare_templates[i % len(compare_templates)]
        result = add(template.format(a=a, b=b), {"type": "ask_analysis"})
        first = first or result
    _register(
        "intent",
        "ASK_ANALYSIS: compare",
        " / ".join(compare_templates),
        "Any phrasing asking whether an outcome differs by/varies with/depends on a group column maps "
        "to ask_analysis; the specific comparison verb doesn't change the intent type.",
        first["text"],
        first["label"],
    )

    # ASK_ANALYSIS: association phrasing.
    assoc_templates = [
        "is there a relationship between {a} and {b}?",
        "does {a} correlate with {b}?",
        "how are {a} and {b} related?",
    ]
    first = None
    for i in range(15):
        a, b = NUMERIC[i % len(NUMERIC)], NUMERIC[(i + 1) % len(NUMERIC)]
        template = assoc_templates[i % len(assoc_templates)]
        result = add(template.format(a=a, b=b), {"type": "ask_analysis"})
        first = first or result
    _register(
        "intent",
        "ASK_ANALYSIS: association",
        " / ".join(assoc_templates),
        "Phrasing asking about a relationship/correlation between two numeric columns maps to ask_analysis.",
        first["text"],
        first["label"],
    )

    # ASK_ANALYSIS: trend phrasing.
    trend_templates = ["has {a} changed over {t}?", "is there a trend in {a} across {t}?"]
    first = None
    for i in range(10):
        a, t = NUMERIC[i % len(NUMERIC)], TIME_COLS[i % len(TIME_COLS)]
        template = trend_templates[i % len(trend_templates)]
        result = add(template.format(a=a, t=t), {"type": "ask_analysis"})
        first = first or result
    _register(
        "intent",
        "ASK_ANALYSIS: trend",
        " / ".join(trend_templates),
        "Phrasing asking about change over a time column maps to ask_analysis.",
        first["text"],
        first["label"],
    )

    # ASK_ANALYSIS: "higher in one X than another".
    first = None
    for i in range(len(NUMERIC)):
        a, b = NUMERIC[i % len(NUMERIC)], GROUPS[(i + 2) % len(GROUPS)]
        result = add(f"is {a} higher in one {b} than another?", {"type": "ask_analysis"})
        first = first or result
    _register(
        "intent",
        "ASK_ANALYSIS: higher-in-one",
        "is {a} higher in one {b} than another?",
        "A magnitude-comparison phrasing over a group column still maps to ask_analysis.",
        first["text"],
        first["label"],
    )

    # ACCEPT: no persona named.
    accept_plain = ["yes", "sounds good", "run it", "accept", "go with that", "that works", "do it"]
    first = None
    for text in accept_plain:
        result = add(text, {"type": "accept", "persona": None})
        first = first or result
    _register(
        "intent",
        "ACCEPT: plain",
        " / ".join(accept_plain),
        "A bare agreement word/phrase with no persona named maps to accept with persona=None (accepts "
        "whichever proposal is on screen).",
        first["text"],
        first["label"],
    )

    # ACCEPT: persona named.
    first = None
    for persona in PERSONAS:
        add(f"go with {persona}", {"type": "accept", "persona": persona})
        add(f"use {persona}'s proposal", {"type": "accept", "persona": persona})
        add(f"I'll take the {persona} option", {"type": "accept", "persona": persona})
        result = add(f"{persona} sounds right", {"type": "accept", "persona": persona})
        first = first or result
    _register(
        "intent",
        "ACCEPT: persona named",
        "go with {persona} / use {persona}'s proposal / ...",
        "Naming a persona (professor/consultant/maverick) sets label.persona to that name.",
        first["text"],
        first["label"],
    )

    # MODIFY: alpha.
    first = None
    for alpha in ["0.01", "0.1", "0.001", "0.005", "0.02"]:
        add(f"set alpha to {alpha}", {"type": "modify", "alpha": float(alpha)})
        result = add(f"use alpha = {alpha}", {"type": "modify", "alpha": float(alpha)})
        first = first or result
    _register(
        "intent",
        "MODIFY: alpha",
        "set alpha to {x} / use alpha = {x}",
        "A stated significance level maps to modify with label.alpha set to that float.",
        first["text"],
        first["label"],
    )

    # OVERRIDE: function named.
    first = None
    for fn in FUNCTIONS:
        result = add(f"override to {fn}", {"type": "override", "function": fn})
        add(f"use {fn} anyway", {"type": "override", "function": fn})
        first = first or result
    add("I still want to use welch_t", {"type": "override", "function": "welch_t"})
    _register(
        "intent",
        "OVERRIDE: function named",
        "override to {fn} / use {fn} anyway",
        "A registered function name present in the text sets label.function to it (matched against the "
        "registry, longest name first -- see `_named_function` in llm/fallback.py).",
        first["text"],
        first["label"],
    )

    # EXPLAIN: term.
    first = None
    for term in TERMS:
        result = add(f"what is {term}?", {"type": "explain", "topic": term})
        add(f"explain {term}", {"type": "explain", "topic": term})
        first = first or result
    _register(
        "intent",
        "EXPLAIN: term",
        "what is {term}? / explain {term}",
        "A glossary-style term after 'what is'/'explain' sets label.topic to that term verbatim.",
        first["text"],
        first["label"],
    )

    # EXPLAIN: why not a function.
    first = None
    for fn in FUNCTIONS[:6]:
        result = add(f"why not {fn}?", {"type": "explain", "topic": fn})
        first = first or result
    _register(
        "intent",
        "EXPLAIN: why not",
        "why not {fn}?",
        "'why not X' sets label.topic to the named function, same as a term.",
        first["text"],
        first["label"],
    )

    # GOTO_STAGE.
    first = None
    for stage in STAGES:
        result = add(f"go to the {stage} stage", {"type": "goto_stage", "target_stage": stage})
        first = first or result
    for stage in STAGES[:4]:
        add(f"jump to {stage}", {"type": "goto_stage", "target_stage": stage})
    _register(
        "intent",
        "GOTO_STAGE",
        "go to the {stage} stage / jump to {stage}",
        "A named stage sets label.target_stage to it.",
        first["text"],
        first["label"],
    )

    # UNDO.
    undo_texts = [
        "undo",
        "go back",
        "revert that",
        "undo the last step",
        "take that back",
        "go back one step",
    ]
    first = None
    for text in undo_texts:
        result = add(text, {"type": "undo"})
        first = first or result
    add("undo please", {"type": "undo"})
    add("can you revert", {"type": "undo"})
    _register(
        "intent",
        "UNDO",
        " / ".join(undo_texts),
        "A fixed set of undo/revert phrasings, no slots.",
        first["text"],
        first["label"],
    )

    # BRANCH.
    first = None
    for name in BRANCH_NAMES:
        result = add(f"create a branch called {name}", {"type": "branch", "name": name})
        add(f"branch {name}", {"type": "branch", "name": name})
        first = first or result
    _register(
        "intent",
        "BRANCH",
        "create a branch called {name} / branch {name}",
        "A named branch sets label.name to it.",
        first["text"],
        first["label"],
    )

    # SWITCH_BRANCH.
    first = None
    for name in BRANCH_NAMES:
        result = add(f"switch to branch {name}", {"type": "switch_branch", "name": name})
        add(f'switch to the "{name}" branch', {"type": "switch_branch", "name": name})
        first = first or result
    _register(
        "intent",
        "SWITCH_BRANCH",
        'switch to branch {name} / switch to the "{name}" branch',
        "A named branch sets label.name to it.",
        first["text"],
        first["label"],
    )

    # SKIP.
    skip_texts = [
        "skip this",
        "not now",
        "move on",
        "skip",
        "let's move on",
        "not right now",
        "pass on this",
        "skip ahead",
    ]
    first = None
    for text in skip_texts:
        result = add(text, {"type": "skip"})
        first = first or result
    _register(
        "intent",
        "SKIP",
        " / ".join(skip_texts),
        "A fixed set of skip/decline phrasings, no slots.",
        first["text"],
        first["label"],
    )

    # EXPORT.
    export_texts = [
        "export this to a notebook",
        "write it up",
        "export the notebook",
        "can you export this",
        "write this up as code",
        "generate the notebook",
        "export",
        "write up the analysis",
    ]
    first = None
    for text in export_texts:
        result = add(text, {"type": "export"})
        first = first or result
    _register(
        "intent",
        "EXPORT",
        " / ".join(export_texts),
        "A fixed set of export/write-up phrasings, no slots.",
        first["text"],
        first["label"],
    )

    # SETTINGS.
    settings_texts = [
        "show me the ledger",
        "what have I run so far",
        "show history",
        "what have I done",
        "show me what tests ran",
        "ledger",
        "what's my test history",
        "show the ledger",
    ]
    first = None
    for text in settings_texts:
        result = add(text, {"type": "settings"})
        first = first or result
    _register(
        "intent",
        "SETTINGS",
        " / ".join(settings_texts),
        "A fixed set of history/ledger phrasings, no slots.",
        first["text"],
        first["label"],
    )

    # SHOW: plot.
    first = None
    for i in range(len(NUMERIC)):
        a, b = NUMERIC[i % len(NUMERIC)], NUMERIC[(i + 3) % len(NUMERIC)]
        result = add(f"plot {a} against {b}", {"type": "show"})
        first = first or result
    _register(
        "intent",
        "SHOW: plot",
        "plot {a} against {b}",
        "A plot/chart request over two columns maps to show.",
        first["text"],
        first["label"],
    )

    # SHOW: distribution.
    first = None
    for a in NUMERIC:
        result = add(f"show me the distribution of {a}", {"type": "show"})
        first = first or result
    _register(
        "intent",
        "SHOW: distribution",
        "show me the distribution of {a}",
        "A distribution request over one column maps to show.",
        first["text"],
        first["label"],
    )

    # ANSWER_CLARIFICATION: confirmation words (only valid with awaiting_answer=True).
    first = None
    for text, confirmed in [
        ("yes", True),
        ("yeah", True),
        ("correct", True),
        ("that's right", True),
    ]:
        result = add(
            text,
            {"type": "answer_clarification", "confirmed": confirmed},
            context={"awaiting_answer": True},
        )
        first = first or result
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
    _register(
        "intent",
        "ANSWER_CLARIFICATION: confirmation",
        "yes/yeah/correct/... , no/nope/...",
        "A bare confirmation word is only read as answer_clarification when context.awaiting_answer is "
        "True (a question is actually pending); otherwise the same word is OTHER (see the realistic set's "
        "category F for the false-positive case).",
        first["text"],
        first["label"],
    )

    # ANSWER_CLARIFICATION: design word.
    first = None
    for word, design in [
        ("paired", "paired"),
        ("independent", "independent"),
        ("repeated", "repeated"),
        ("same people", "paired"),
        ("matched", "paired"),
        ("different people", "independent"),
        ("same subjects", "paired"),
    ]:
        result = add(
            word,
            {"type": "answer_clarification", "design": design},
            context={"awaiting_answer": True},
        )
        first = first or result
    _register(
        "intent",
        "ANSWER_CLARIFICATION: design word",
        "paired / independent / repeated / same people / ...",
        "A bare design-indicating word, only when context.awaiting_answer is True, sets label.design to "
        "the Design enum value it maps to (paired/independent/repeated).",
        first["text"],
        first["label"],
    )

    # OTHER: genuinely unclear, below the confidence floor.
    other_texts = [
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
    ]
    first = None
    for text in other_texts:
        result = add(text, {"type": "other", "needs_clarification": True})
        first = first or result
    _register(
        "intent",
        "OTHER",
        " / ".join(other_texts),
        "Text matching no other rule falls below Section 9.1's confidence floor and maps to other, with "
        "needs_clarification=True -- the orchestrator asks rather than guessing.",
        first["text"],
        first["label"],
    )

    return items


# --------------------------------------------------------------------------
# question_specs.jsonl -- template items
# --------------------------------------------------------------------------


def build_question_spec_templates() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    def add(text: str, label: dict[str, Any]) -> dict[str, Any]:
        label = {"needs_clarification": False, **label}
        item = _item(0, text, label, None, "template")
        items.append(item)
        return item

    # COMPARE_GROUPS -- independent design, stated explicitly.
    first = None
    for i in range(15):
        a, b = NUMERIC[i % len(NUMERIC)], GROUPS[i % len(GROUPS)]
        result = add(
            f"is {a} different between {b} groups? these are unrelated, independent samples",
            {
                "goal": "compare_groups",
                "variables": {"outcome": a, "group": b},
                "design": "independent",
            },
        )
        first = first or result
    _register(
        "question_spec",
        "COMPARE_GROUPS: independent (stated)",
        "is {a} different between {b} groups? these are unrelated, independent samples",
        "The text names the outcome and group columns and explicitly says the samples are independent, "
        "so design=independent -- read off the wording, never inferred from the data.",
        first["text"],
        first["label"],
    )

    # COMPARE_GROUPS -- paired design, stated explicitly.
    first = None
    for a in NUMERIC:
        result = add(
            f"did {a} change after the intervention, same subjects measured before and after",
            {
                "goal": "compare_groups",
                "variables": {"outcome": a, "group": "period"},
                "design": "paired",
            },
        )
        first = first or result
    _register(
        "question_spec",
        "COMPARE_GROUPS: paired (stated)",
        "did {a} change after the intervention, same subjects measured before and after",
        "'same subjects ... before and after' explicitly states paired design.",
        first["text"],
        first["label"],
    )

    # COMPARE_GROUPS -- repeated design, stated explicitly.
    first = None
    for i in range(9):
        a, b = NUMERIC[i % len(NUMERIC)], GROUPS[(i + 1) % len(GROUPS)]
        result = add(
            f"{a} was measured three times on the same subjects across {b}",
            {
                "goal": "compare_groups",
                "variables": {"outcome": a, "group": b},
                "design": "repeated",
            },
        )
        first = first or result
    _register(
        "question_spec",
        "COMPARE_GROUPS: repeated (stated)",
        "{a} was measured three times on the same subjects across {b}",
        "'measured three times on the same subjects' explicitly states repeated design.",
        first["text"],
        first["label"],
    )

    # COMPARE_GROUPS -- design genuinely unstated: design MUST stay
    # "unknown" (the real Design enum has no "ambiguous" value), and
    # needs_clarification=True is what actually marks the case as one a
    # downstream ambiguity check must catch (Section 7.1: design is
    # never inferred from wording alone when the wording does not settle
    # it -- an earlier draft of this file mislabelled this case with a
    # literal design="ambiguous" string, which doesn't correspond to
    # anything the real QuestionSpec/Design enum can hold).
    first = None
    for i in range(18):
        a, b = NUMERIC[i % len(NUMERIC)], GROUPS[i % len(GROUPS)]
        result = add(
            f"is {a} different by {b}?",
            {
                "goal": "compare_groups",
                "variables": {"outcome": a, "group": b},
                "design": "unknown",
                "needs_clarification": True,
                "note": "wording does not say independent vs. paired/repeated",
            },
        )
        first = first or result
    _register(
        "question_spec",
        "COMPARE_GROUPS: design unstated (needs clarification)",
        "is {a} different by {b}?",
        "No cue distinguishes independent samples from paired/repeated measurement, so design stays "
        "'unknown' (not a guess) and needs_clarification=True: the downstream validate_spec/ambiguity "
        "check must ask, per Section 7.1's rule that design is never inferred from wording alone.",
        first["text"],
        first["label"],
    )

    # ASSOCIATION.
    first = None
    for a, b in itertools.islice(itertools.combinations(NUMERIC, 2), 15):
        result = add(
            f"is there a relationship between {a} and {b}?",
            {"goal": "association", "variables": {"x": a, "y": b}, "design": "unknown"},
        )
        first = first or result
    _register(
        "question_spec",
        "ASSOCIATION",
        "is there a relationship between {a} and {b}?",
        "design='unknown' here because the goal has no design concept at all, not because anything is "
        "ambiguous -- needs_clarification stays False.",
        first["text"],
        first["label"],
    )

    # DISTRIBUTION_FIT.
    first = None
    for i in range(12):
        a = NUMERIC[i % len(NUMERIC)]
        ref = [50, 100, 0, 3.5, 20][i % 5]
        result = add(
            f"is the average {a} equal to {ref}?",
            {
                "goal": "distribution_fit",
                "variables": {"outcome": a},
                "design": "unknown",
                "reference_value": ref,
            },
        )
        first = first or result
    _register(
        "question_spec",
        "DISTRIBUTION_FIT",
        "is the average {a} equal to {ref}?",
        "The stated number sets reference_value; design is not applicable to this goal.",
        first["text"],
        first["label"],
    )

    # EQUIVALENCE -- design must be stated in the text to be labelled,
    # same rule as COMPARE_GROUPS (an earlier draft asserted
    # design="independent" here without the text actually saying so,
    # which is the same class of bug as the "ambiguous" one above: the
    # pattern now states it explicitly, matching what it claims).
    first = None
    for i in range(12):
        a, b = NUMERIC[i % len(NUMERIC)], GROUPS[i % len(GROUPS)]
        result = add(
            f"is {a} practically the same between {b} groups (independent, unrelated samples), "
            "within a small margin?",
            {
                "goal": "equivalence",
                "variables": {"outcome": a, "group": b},
                "design": "independent",
            },
        )
        first = first or result
    _register(
        "question_spec",
        "EQUIVALENCE",
        "is {a} practically the same between {b} groups (independent, unrelated samples), within a small margin?",
        "Equivalence testing between groups carries the same design ambiguity as compare_groups; the "
        "pattern states 'independent, unrelated samples' explicitly so design=independent is actually "
        "read off the text, not assumed the way an earlier draft of this generator did.",
        first["text"],
        first["label"],
    )

    # TREND.
    first = None
    for i in range(12):
        a, t = NUMERIC[i % len(NUMERIC)], TIME_COLS[i % len(TIME_COLS)]
        result = add(
            f"has {a} changed over {t}?",
            {"goal": "trend", "variables": {"outcome": a, "time": t}, "design": "unknown"},
        )
        first = first or result
    _register(
        "question_spec",
        "TREND",
        "has {a} changed over {t}?",
        "design is not applicable to this goal.",
        first["text"],
        first["label"],
    )

    # DESCRIBE.
    first = None
    for a in NUMERIC:
        result = add(
            f"summarize {a}", {"goal": "describe", "variables": {"outcome": a}, "design": "unknown"}
        )
        first = first or result
        add(
            f"give me an overview of {a}",
            {"goal": "describe", "variables": {"outcome": a}, "design": "unknown"},
        )
    _register(
        "question_spec",
        "DESCRIBE",
        "summarize {a} / give me an overview of {a}",
        "design is not applicable to this goal.",
        first["text"],
        first["label"],
    )

    # MISSINGNESS.
    first = None
    for a in NUMERIC:
        result = add(
            f"why is {a} missing so often?",
            {"goal": "missingness", "variables": {"outcome": a}, "design": "unknown"},
        )
        first = first or result
        add(
            f"what's going on with the missing values in {a}?",
            {"goal": "missingness", "variables": {"outcome": a}, "design": "unknown"},
        )
    _register(
        "question_spec",
        "MISSINGNESS",
        "why is {a} missing so often? / what's going on with ... {a}?",
        "design is not applicable to this goal.",
        first["text"],
        first["label"],
    )

    # OUTLIERS.
    first = None
    for a in NUMERIC:
        result = add(
            f"are there outliers in {a}?",
            {"goal": "outliers", "variables": {"outcome": a}, "design": "unknown"},
        )
        first = first or result
        add(
            f"does {a} have any extreme values?",
            {"goal": "outliers", "variables": {"outcome": a}, "design": "unknown"},
        )
    _register(
        "question_spec",
        "OUTLIERS",
        "are there outliers in {a}? / does {a} have any extreme values?",
        "design is not applicable to this goal.",
        first["text"],
        first["label"],
    )

    # TRANSFORM.
    first = None
    for a in NUMERIC:
        result = add(
            f"should {a} be log-transformed?",
            {"goal": "transform", "variables": {"outcome": a}, "design": "unknown"},
        )
        first = first or result
        add(
            f"does {a} need a transformation before modelling?",
            {"goal": "transform", "variables": {"outcome": a}, "design": "unknown"},
        )
    _register(
        "question_spec",
        "TRANSFORM",
        "should {a} be log-transformed? / does {a} need a transformation ...?",
        "design is not applicable to this goal.",
        first["text"],
        first["label"],
    )

    return items


# --------------------------------------------------------------------------
# combine template + realistic, renumber, write
# --------------------------------------------------------------------------


def _finalize(template_items: list[dict[str, Any]], realistic: list[tuple]) -> list[dict[str, Any]]:
    combined = list(template_items)
    for text, label, context in realistic:
        label = {"needs_clarification": False, **label}
        combined.append(_item(0, text, label, context, "realistic"))
    for idx, item in enumerate(combined, start=1):
        item["id"] = idx
    return combined


def build_intent_utterances() -> list[dict[str, Any]]:
    return _finalize(build_intent_templates(), REALISTIC_INTENT_ITEMS)


def build_question_specs() -> list[dict[str, Any]]:
    return _finalize(build_question_spec_templates(), REALISTIC_QUESTION_SPEC_ITEMS)


def main() -> None:
    intents = build_intent_utterances()
    specs = build_question_specs()

    with (ROOT / "intent_utterances.jsonl").open("w", encoding="utf-8") as fh:
        for item in intents:
            fh.write(json.dumps(item) + "\n")

    with (ROOT / "question_specs.jsonl").open("w", encoding="utf-8") as fh:
        for item in specs:
            fh.write(json.dumps(item) + "\n")

    by_source: dict[str, int] = {}
    for item in intents + specs:
        by_source[item["source"]] = by_source.get(item["source"], 0) + 1
    print(
        f"wrote {len(intents)} intent utterances, {len(specs)} question specs -- by source: {by_source}"
    )


if __name__ == "__main__":
    main()
