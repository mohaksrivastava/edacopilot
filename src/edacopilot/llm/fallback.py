"""Every LLM call, implemented without a model (ARCHITECTURE.md, Section 10.5).

Rule 9: the system must run with the LLM switched off. This module is how.
It is not a degraded mode kept alive out of duty — it is how the entire test
suite runs, and it is what the system falls back to when a provider is
unreachable or an LLM's answer fails its fact-check (Section 8.4).

Section 10.5 lists five fallbacks. Three are here:

- `parse_intent`: keyword and regex rules over the user's text.
- `build_question_spec`: refuses to guess, and returns the *fields* a
  form-style card should ask for instead (the card itself is the
  orchestrator's to build — this module has no opinion about UI).
- `answer_free_question`: a lookup in `glossary.yaml`, or a plain
  "not available offline".

The other two live where their data does: `write_rationale` is
`personas/rationale.py`, and `elicit_mnar`'s question bank arrives with the
missingness stage in M10.

**The parser guesses narrowly and says when it is guessing.** A keyword
ruleset that stretched to cover everything would misroute confidently,
which is worse than not routing at all: Section 9.1 has the orchestrator
ask a clarifying question below `CONFIDENCE_FLOOR`, and that path only
helps if the confidence is honest. So a rule fires on an unambiguous marker
or not at all.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

import pandas as pd
import yaml

from edacopilot.eligibility.spec import Design, Goal
from edacopilot.orchestrator.intents import Intent, IntentType
from edacore.contracts import SemanticType
from edacore.profiling import infer_semantic_types

GLOSSARY_PATH: Final = Path(__file__).parent / "glossary.yaml"

NOT_IN_GLOSSARY: Final = (
    "That term is not in the offline glossary. With an LLM configured "
    "(Section 10), the system can answer it against this session's facts."
)

PERSONA_WORDS: Final[dict[str, str]] = {
    "professor": "professor",
    "consultant": "consultant",
    "maverick": "maverick",
}

# Confidence for a rule that fired on an unambiguous marker ("undo",
# "explain"), versus one that fired on a weaker signal. The second sits
# below Section 9.1's floor on purpose, so the orchestrator asks.
CERTAIN: Final = 0.95
LIKELY: Final = 0.8
UNSURE: Final = 0.4


# --------------------------------------------------------------------------
# glossary (Section 10.5's `answer_free_question`)
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_glossary() -> tuple[dict[str, str], dict[str, str]]:
    """The glossary's terms and its alias table, loaded once."""
    data = yaml.safe_load(GLOSSARY_PATH.read_text(encoding="utf-8"))
    terms = {key: " ".join(value.split()) for key, value in data["terms"].items()}
    aliases = dict(data.get("aliases") or {})
    unknown = sorted(target for target in aliases.values() if target not in terms)
    if unknown:
        raise ValueError(f"glossary aliases point at undefined terms: {unknown}")
    return terms, aliases


def normalise_topic(topic: str) -> str:
    """Reduce a phrase to a glossary key candidate.

    Strips the question wrapper ("what is a ...?"), punctuation and
    pluralisation, so `explain("What is sphericity?")` and
    `explain("sphericity")` are the same lookup.
    """
    text = topic.strip().lower()
    text = re.sub(r"^(what(?:'s| is| are| does)?|why|how|explain|define|tell me about)\b", "", text)
    # Drop the apostrophe rather than the letter around it, so "Cohen's d"
    # becomes "cohens d" and reaches its alias instead of "cohen s d".
    text = text.replace("\u2019", "'").replace("'", "")
    text = re.sub(r"\b(a|an|the|mean|means|meaning|do|does|did)\b", " ", text)
    text = re.sub(r"[^a-z0-9_ ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def answer_free_question(topic: str) -> str:
    """Section 10.5: a glossary lookup, or "not available offline"."""
    terms, aliases = load_glossary()
    candidate = normalise_topic(topic)
    for key in _lookup_keys(candidate):
        if key in terms:
            return terms[key]
        if key in aliases:
            return terms[aliases[key]]
    return NOT_IN_GLOSSARY


def _lookup_keys(candidate: str) -> list[str]:
    """The forms of a phrase worth trying, most specific first."""
    underscored = candidate.replace(" ", "_")
    keys = [candidate, underscored]
    if candidate.endswith("s"):
        keys += [candidate[:-1], underscored[:-1]]
    return keys


def glossary_terms() -> list[str]:
    terms, _ = load_glossary()
    return sorted(terms)


# --------------------------------------------------------------------------
# intent parsing (Section 10.5's `parse_intent`)
# --------------------------------------------------------------------------

_ANALYSIS_WORDS = re.compile(
    r"\b(differ|difference|different|compare|comparison|higher|lower|greater|"
    r"relat(?:ed|ionship)|associat|correlat|predict|vary|affect|impact|"
    r"between|across|versus|vs)\b"
)
_STAGE_WORDS = re.compile(r"\b(?:go to|jump to|move to|switch to)\s+(?:the\s+)?([a-z_]+)")
_POSTHOC_WORDS = re.compile(r"\b(post[- ]?hoc|pairwise|which (?:groups?|pairs?)|follow[- ]?up)\b")
_ALPHA = re.compile(r"\balpha\s*(?:=|of|to)?\s*(0?\.\d+)\b")
_YES = re.compile(r"^(yes|yep|yeah|correct|confirmed?|that'?s right)\b")
_NO = re.compile(r"^(no|nope|not quite|that'?s wrong)\b")
_DESIGN_ANSWER = re.compile(r"\b(paired|repeated|independent|matched|same (?:people|subjects))\b")


def parse_intent(text: str, *, awaiting_answer: bool = False) -> Intent:
    """Map free text to an `Intent` with keyword rules (Section 10.5).

    `awaiting_answer` says the last card was a question. It changes how a
    bare "paired" or "yes" is read — as an answer rather than as a new
    request — which is the one piece of conversational state a keyword
    parser genuinely needs.
    """
    raw = text.strip()
    lowered = raw.lower()
    if not lowered:
        return Intent(type=IntentType.OTHER, confidence=0.0, payload={"text": raw})

    if awaiting_answer and (
        _YES.match(lowered) or _NO.match(lowered) or _DESIGN_ANSWER.search(lowered)
    ):
        return Intent(
            type=IntentType.ANSWER_CLARIFICATION,
            payload={"text": raw, **_answer_payload(lowered)},
            confidence=CERTAIN,
        )

    if re.search(r"\bundo\b|\bgo back\b|\brevert\b", lowered):
        return Intent(type=IntentType.UNDO, payload={"text": raw}, confidence=CERTAIN)

    if match := re.search(r"\bswitch (?:to )?(?:the )?branch\s+['\"]?([\w -]+)", lowered):
        return Intent(
            type=IntentType.SWITCH_BRANCH,
            payload={"text": raw, "name": match.group(1).strip()},
            confidence=CERTAIN,
        )
    if match := re.search(
        r"\b(?:new |create a |make a |start a )?branch\s+['\"]?([\w -]+)", lowered
    ):
        return Intent(
            type=IntentType.BRANCH,
            payload={"text": raw, "name": match.group(1).strip()},
            confidence=CERTAIN,
        )

    if re.search(r"\bskip\b|\bnot now\b|\bmove on\b", lowered):
        return Intent(type=IntentType.SKIP, payload={"text": raw}, confidence=CERTAIN)

    if re.search(r"\bexport\b|\bnotebook\b|\bwrite (?:it )?up\b", lowered):
        return Intent(type=IntentType.EXPORT, payload={"text": raw}, confidence=CERTAIN)

    if match := _STAGE_WORDS.search(lowered):
        return Intent(
            type=IntentType.GOTO_STAGE,
            target_stage=match.group(1),
            payload={"text": raw},
            confidence=CERTAIN,
        )

    if match := _ALPHA.search(lowered):
        return Intent(
            type=IntentType.MODIFY,
            payload={"text": raw, "alpha": float(match.group(1))},
            confidence=CERTAIN,
        )

    if re.search(r"\boverride\b|\banyway\b|\bregardless\b|\bi still want\b", lowered):
        return Intent(
            type=IntentType.OVERRIDE,
            payload={"text": raw, "function": _named_function(lowered)},
            confidence=LIKELY,
        )

    if re.search(
        r"\baccept\b|\bgo with\b|\buse\b.*\bproposal\b|\bsounds good\b|\brun it\b", lowered
    ):
        return Intent(
            type=IntentType.ACCEPT,
            persona=_named_persona(lowered),
            payload={"text": raw},
            confidence=CERTAIN,
        )

    if _POSTHOC_WORDS.search(lowered):
        return Intent(
            type=IntentType.ASK_ANALYSIS,
            payload={"text": raw, "post_hoc": True},
            confidence=CERTAIN,
        )

    if re.search(
        r"\bexplain\b|\bwhat (?:is|are|does)\b|\bwhy (?:not|is|did)\b|\bwhat'?s\b", lowered
    ):
        return Intent(
            type=IntentType.EXPLAIN,
            payload={"text": raw, "topic": raw},
            confidence=CERTAIN,
        )

    if re.search(r"\bledger\b|\bwhat have i (?:run|done)\b|\bhistory\b", lowered):
        return Intent(
            type=IntentType.SETTINGS, payload={"text": raw, "show": "ledger"}, confidence=LIKELY
        )

    if re.search(r"\b(?:plot|chart|graph|show me|visuali[sz]e|histogram|scatter)\b", lowered):
        return Intent(type=IntentType.SHOW, payload={"text": raw}, confidence=LIKELY)

    if _ANALYSIS_WORDS.search(lowered):
        return Intent(type=IntentType.ASK_ANALYSIS, payload={"text": raw}, confidence=LIKELY)

    # Nothing matched. Say so honestly: below Section 9.1's floor the
    # orchestrator asks rather than acting on a guess.
    return Intent(type=IntentType.OTHER, payload={"text": raw}, confidence=UNSURE)


def _answer_payload(lowered: str) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if _YES.match(lowered):
        payload["confirmed"] = True
    if _NO.match(lowered):
        payload["confirmed"] = False
    if match := _DESIGN_ANSWER.search(lowered):
        word = match.group(1)
        if word in ("paired", "matched") or word.startswith("same "):
            payload["design"] = Design.PAIRED.value
        elif word == "repeated":
            payload["design"] = Design.REPEATED.value
        else:
            payload["design"] = Design.INDEPENDENT.value
    return payload


def _named_persona(lowered: str) -> str | None:
    for word, persona in PERSONA_WORDS.items():
        if word in lowered:
            return persona
    return None


def _named_function(lowered: str) -> str | None:
    """A registered function named in the text, if any.

    Matched against the registry rather than a hand-written list, so an
    override can name any method the catalogue actually has.
    """
    from edacore.registry import registry

    candidates = [
        spec.name
        for spec in registry.list()
        if spec.kind in ("test", "posthoc") and spec.name in lowered.replace(" ", "_")
    ]
    # Longest first: "welch_t" must not win over "welch_anova".
    return max(candidates, key=len) if candidates else None


# --------------------------------------------------------------------------
# question-spec form (Section 10.5's `build_question_spec`)
# --------------------------------------------------------------------------


class SpecFormField:
    """One field of the form-style card the fallback offers instead of a spec."""

    def __init__(self, name: str, prompt: str, options: list[str], required: bool = True) -> None:
        self.name = name
        self.prompt = prompt
        self.options = options
        self.required = required

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"SpecFormField({self.name!r}, options={self.options!r})"


def build_question_spec_form(df: pd.DataFrame) -> list[SpecFormField]:
    """The fields a form-style card should ask for (Section 10.5).

    Without a model there is no honest way to turn "is income different by
    region?" into a `QuestionSpec`, so the fallback does not try: it shows
    the user the fields and lets them fill them in. The options come from
    the actual frame, so the form cannot offer a column that is not there.
    """
    types = infer_semantic_types(df)
    numeric = sorted(
        name
        for name, (semantic_type, _) in types.items()
        if semantic_type in (SemanticType.CONTINUOUS, SemanticType.DISCRETE)
    )
    grouping = sorted(
        name
        for name, (semantic_type, _) in types.items()
        if semantic_type in (SemanticType.NOMINAL, SemanticType.ORDINAL, SemanticType.BINARY)
    )
    every = sorted(df.columns)

    return [
        SpecFormField("goal", "What are you asking?", [goal.value for goal in Goal]),
        SpecFormField("outcome", "Which column is the outcome?", numeric or every),
        SpecFormField("group", "Which column defines the groups?", grouping or every),
        SpecFormField(
            "design",
            "Are the groups independent, or the same subjects measured more than once?",
            [design.value for design in Design],
        ),
        SpecFormField(
            "alpha", "Significance level (blank uses the persona default)", [], required=False
        ),
    ]
