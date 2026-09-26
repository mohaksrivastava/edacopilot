"""Regenerate docs/persona_picks.md: every trap scenario x persona pick.

Like `generate_eligibility_table.py`, this is a review artefact rendered
from the live code rather than written by hand, and
`test_persona_picks_doc.py` fails if the committed file and a fresh render
disagree. A stale table of "what each persona would do" is worse than none:
it would be read as a statement about current behaviour.

    python scripts/generate_persona_picks.py          # rewrite the file
    python scripts/generate_persona_picks.py --check  # exit 1 if stale
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from edacopilot.eligibility import (
    Design,
    Goal,
    QuestionSpec,
    select_candidates,
    validate_spec,
)
from edacopilot.personas import PERSONA_ORDER, load_personas, propose_with_rationales, render_card
from tests.scenarios.generators import (
    heavy_tails_small_n,
    heteroscedastic_groups,
    ordinal_as_numeric,
    paired_as_independent,
    worked_example_7_4,
)

DOC_PATH = Path(__file__).resolve().parents[1] / "docs" / "persona_picks.md"


@dataclass(frozen=True)
class Scenario:
    name: str
    note: str
    df: pd.DataFrame
    spec: QuestionSpec
    # The spec as it arrives *before* the user has confirmed anything. When
    # set, its ambiguities are rendered above the picks, because the picks
    # below them are the state after those questions were answered -- and a
    # table that showed only the answer would read as though the system had
    # decided the design itself (Section 7.1: "Design is always confirmed by
    # the user").
    asked_first: QuestionSpec | None = None


def _compare(
    df: pd.DataFrame, outcome: str, group: str, design: Design, **extra: Any
) -> QuestionSpec:
    variables: dict[str, str] = {"outcome": outcome, "group": group}
    variables.update({k: v for k, v in extra.items() if isinstance(v, str)})
    return QuestionSpec(
        goal=Goal.COMPARE_GROUPS,
        variables=variables,
        design=design,
        confirmed_by_user={"design"},
    )


def scenarios() -> list[Scenario]:
    rng = np.random.default_rng(1)
    clean = pd.DataFrame(
        {
            "value": np.concatenate([rng.normal(10, 2, 40), rng.normal(12, 2, 40)]),
            "group": ["A"] * 40 + ["B"] * 40,
        }
    )
    paired = paired_as_independent()
    ordinal = ordinal_as_numeric()

    return [
        Scenario(
            "worked example 7.4",
            "income by gender, n = 38/41, strong right skew (ARCHITECTURE.md Section 7.4)",
            worked_example_7_4(),
            _compare(worked_example_7_4(), "income", "gender", Design.INDEPENDENT),
        ),
        Scenario(
            "clean two groups",
            "normal, equal variance, equal n -- the case where nothing is compromised",
            clean,
            _compare(clean, "value", "group", Design.INDEPENDENT),
        ),
        Scenario(
            "heteroscedastic_groups",
            "unequal variances and unequal n (Section 15.3)",
            heteroscedastic_groups(),
            _compare(heteroscedastic_groups(), "value", "group", Design.INDEPENDENT),
        ),
        Scenario(
            "heavy_tails_small_n",
            "t(2) tails at n = 12 per group (Section 15.3)",
            heavy_tails_small_n(),
            _compare(heavy_tails_small_n(), "value", "group", Design.INDEPENDENT),
        ),
        Scenario(
            "paired_as_independent (after the design question is answered)",
            "the trap resolved: the user confirmed the paired design (Section 15.3)",
            paired,
            _compare(paired, "score", "condition", Design.PAIRED, subject="subject_id"),
            asked_first=QuestionSpec(
                goal=Goal.COMPARE_GROUPS,
                variables={"outcome": "score", "group": "condition"},
                design=Design.INDEPENDENT,
                user_text="Is the score different between the two conditions?",
            ),
        ),
        Scenario(
            "ordinal_as_numeric (confirmed ordinal)",
            "a 1-5 Likert scale, confirmed as ordinal rather than measured (Section 15.3)",
            ordinal,
            QuestionSpec(
                goal=Goal.ASSOCIATION,
                variables={"x": "satisfaction", "y": "tenure_months"},
                design=Design.INDEPENDENT,
                confirmed_by_user={"design", "x", "y"},
            ),
        ),
    ]


def render() -> str:
    lines: list[str] = [
        "# Persona picks by scenario",
        "",
        "**Generated** by `scripts/generate_persona_picks.py` from the live persona",
        "policies, the eligibility engine and the template rationales. Do not edit by",
        "hand: `tests/unit/edacopilot/test_persona_picks_doc.py` fails if this file and a",
        "fresh render disagree.",
        "",
        "Each row is one persona's proposal for one scenario, with the one-line rationale",
        "it would show. The rationales are Section 10.5's deterministic templates -- what",
        "the system says with the LLM switched off, and what it falls back to when an",
        "LLM-written rationale fails its fact-check (Section 8.4).",
        "",
        "Rule 2 holds throughout: a persona chooses among what the eligibility engine",
        "(Section 7) already ruled valid, and can never propose an INELIGIBLE method.",
        "",
        "## The personas (Section 8.1)",
        "",
        "| Persona | Method pool | Normality | CLT shortcut | Preference order |",
        "|---|---|---|---|---|",
    ]
    for policy in load_personas():
        pool = ", ".join(f"`{tag}`" for tag in sorted(policy.method_pool_tags))
        normality = f"borderline is **{policy.normality.borderline_is}**"
        shortcut = (
            "Cochran's rule (n > 25*skew^2)"
            if policy.normality.clt_shortcut == "cochran"
            else "none"
        )
        prefer = " > ".join(f"`{key}`" for key in policy.prefer)
        lines.append(
            f"| **{policy.display_name}** | {pool} | {normality} | {shortcut} | {prefer} |"
        )
    lines += ["", "## Picks", ""]

    for scenario in scenarios():
        validated = validate_spec(scenario.spec, scenario.df)
        lines += [f"### {scenario.name}", "", f"> {scenario.note}", ""]

        if scenario.asked_first is not None:
            asked = validate_spec(scenario.asked_first, scenario.df)
            lines += [
                "**This table is the state *after* the user answered.** The question "
                "below is raised first, and no persona is consulted until it is: "
                "`select_candidates` refuses a spec with open ambiguities "
                "(`AmbiguousSpecError`), and a persona can only ever choose from a "
                "`CandidateSet`, so there is no path to the card underneath without "
                "passing through the question.",
                "",
                *(f"> {question}" for question in asked.ambiguities),
                "",
            ]

        if validated.ambiguities:
            lines += [
                "The engine stops here and asks before any persona is consulted:",
                "",
                *(f"- {question}" for question in validated.ambiguities),
                "",
            ]
            continue

        candidates = select_candidates(validated, scenario.df)
        verdict = propose_with_rationales(candidates)
        statuses = candidates.statuses()

        lines += [
            f"Family: `{candidates.family}`. Card: **{verdict.card}**.",
            "",
            f"> {render_card(verdict)}",
            "",
            "| Persona | Pick | Status | Rationale |",
            "|---|---|---|---|",
        ]
        for persona_id in PERSONA_ORDER:
            persona_pick = verdict.pick_for(persona_id)
            function = persona_pick.function
            if function is None:
                lines.append(
                    f"| **{persona_pick.display_name}** | — | — | {persona_pick.rationale} |"
                )
                continue
            engine_status = statuses[function].value
            seen = persona_pick.eligibility.value if persona_pick.eligibility else engine_status
            status = (
                f"`{engine_status}`"
                if seen == engine_status
                else f"`{engine_status}` → `{seen}` (relaxed)"
            )
            lines.append(
                f"| **{persona_pick.display_name}** | `{function}` | {status} "
                f"| {persona_pick.rationale} |"
            )
        lines.append("")

        relaxed = [p for p in verdict.picks if p.relaxations]
        if relaxed:
            lines += ["Relaxations applied:", ""]
            lines += [f"- **{p.display_name}**: {p.relaxations[0]}" for p in relaxed]
            lines.append("")

    lines += [
        "## How to read a relaxed status",
        "",
        "A persona may upgrade a CAVEAT to ELIGIBLE only through a rule its own YAML",
        "declares, and only for soft assumptions:",
        "",
        "- `clt_shortcut: cochran` clears a **normality** caveat when Cochran's rule",
        "  (n > 25*skew^2 per group) is met. Note that `normality_or_large_n` is already",
        "  decided by Cochran's rule inside the eligibility engine, because Section 6.7",
        "  grants those methods the escape by name -- so the persona shortcut only bites",
        "  on methods declaring plain `normality`, such as one-way ANOVA.",
        "- `borderline_is: pass` clears a caveat **all** of whose causes are BORDERLINE.",
        "  One FAIL anywhere blocks it, so a persona cannot reach eligibility by",
        "  declining to look at a check.",
        "- `variance.strategy: always_robust` is a *tightening*: the persona declines to",
        "  propose a method that assumes equal variance when that assumption is failing,",
        "  rather than waiving the caveat on it.",
        "",
        "Hard assumptions are never touched by any of these.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 if the file is stale")
    args = parser.parse_args()

    rendered = render()
    if args.check:
        current = DOC_PATH.read_text(encoding="utf-8") if DOC_PATH.exists() else ""
        if current != rendered:
            print(f"{DOC_PATH} is stale; run: python {Path(__file__).name}", file=sys.stderr)
            return 1
        return 0

    DOC_PATH.write_text(rendered, encoding="utf-8")
    print(f"wrote {DOC_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
