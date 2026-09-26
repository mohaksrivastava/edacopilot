"""The offline glossary (ARCHITECTURE.md, Section 10.5).

Section 10.5 asks for "one line per term, ~150 terms covering every
assumption, test, effect size and missingness concept in the catalogue".
The coverage half of that is derived from the live registry rather than
listed here, so registering a new test or declaring a new assumption
without defining it fails CI — which is the only way a glossary stays
current.

The style half is checked too, loosely: one sentence, no undefined symbols,
and no definition that merely restates the name. A glossary whose entry for
`sphericity` reads "the sphericity assumption" has the same coverage and
none of the value.
"""

from __future__ import annotations

import pytest

from edacopilot.llm import NOT_IN_GLOSSARY, answer_free_question, load_glossary
from edacore.registry import registry

TERMS, ALIASES = load_glossary()


def required_terms() -> set[str]:
    """Every catalogue term Section 10.5 names, from the live registry."""
    required = {
        spec.name for spec in registry.list() if spec.kind in ("test", "posthoc", "effect", "check")
    }
    for spec in registry.list():
        required |= set(spec.assumptions.hard) | set(spec.assumptions.soft)
    return required


def test_every_catalogue_term_is_defined() -> None:
    missing = sorted(required_terms() - set(TERMS))
    assert not missing, (
        f"{len(missing)} catalogue term(s) have no glossary entry: {missing}. "
        f"`session.explain(...)` would answer 'not available offline' for each."
    )


def test_the_glossary_is_about_the_size_section_10_5_calls_for() -> None:
    """~150 terms. A floor, not an exact count: the catalogue grows."""
    assert len(TERMS) >= 150


def test_missingness_concepts_are_covered() -> None:
    """Section 10.5 names them explicitly, and none is in the registry yet
    (Section 6.3 lands in M10), so nothing else would catch their absence."""
    assert {
        "mcar",
        "mar",
        "mnar",
        "littles_test",
        "listwise_deletion",
        "single_imputation",
        "multiple_imputation",
        "mean_imputation",
        "missingness_predictors",
    } <= set(TERMS)


@pytest.mark.parametrize("term", sorted(TERMS))
def test_each_definition_is_one_plain_sentence(term: str) -> None:
    definition = TERMS[term]
    assert definition.strip(), f"'{term}' has an empty definition"
    assert definition[0].isupper() or definition[0] in "`'", f"'{term}' does not start a sentence"
    assert definition.rstrip().endswith("."), f"'{term}' does not end in a full stop"
    assert "\n" not in definition, f"'{term}' spans more than one line"
    # Two sentences are allowed (what it is, then when it matters); three
    # is a paragraph, and Section 10.5 asked for a line.
    assert definition.count(". ") <= 2, f"'{term}' is longer than two sentences"


@pytest.mark.parametrize("term", sorted(TERMS))
def test_no_definition_merely_restates_its_own_name(term: str) -> None:
    """The failure mode a coverage test alone cannot catch."""
    words = [word for word in term.split("_") if len(word) > 3]
    definition = TERMS[term].lower()
    stripped = definition
    for word in words:
        stripped = stripped.replace(word, "")
    assert len(stripped) > 0.5 * len(definition), (
        f"'{term}' is defined mostly in terms of its own name: {TERMS[term]!r}"
    )


def test_every_alias_points_at_a_defined_term() -> None:
    """`load_glossary` raises on a dangling alias; this says so out loud."""
    dangling = sorted(target for target in ALIASES.values() if target not in TERMS)
    assert not dangling


@pytest.mark.parametrize(
    ("asked", "expected"),
    [
        ("sphericity", "sphericity"),
        ("What is sphericity?", "sphericity"),
        ("what does sphericity mean", "sphericity"),
        ("mann-whitney", "mann_whitney"),
        ("mann_whitney", "mann_whitney"),
        ("what is a p-value?", "p_value"),
        ("p value", "p_value"),
        ("explain Cohen's d", "cohens_d"),
        ("effect sizes", "effect_size"),
        ("outliers", "outlier"),
        ("what is MNAR", "mnar"),
        ("Levene's test", "check_equal_variance_levene"),
    ],
)
def test_a_question_reaches_the_right_entry(asked: str, expected: str) -> None:
    assert answer_free_question(asked) == TERMS[expected]


def test_an_unknown_term_says_so_rather_than_guessing() -> None:
    """Section 10.5's other half: "or 'not available offline'".

    Guessing would be worse than silence here -- a confidently wrong
    definition of a statistical term is exactly the failure this product
    exists to prevent.
    """
    assert answer_free_question("frobnicator") == NOT_IN_GLOSSARY
    assert answer_free_question("") == NOT_IN_GLOSSARY


def test_definitions_name_the_consequence_where_there_is_one() -> None:
    """Section 1.3's "teach by consequence", spot-checked on the
    assumptions where ignoring the violation does specific damage."""
    consequences = {
        "sphericity": "false-positive",
        "equal_variance": "wrong",
        "independent": "cannot check",
        "linearity": "curved",
        "mnar": "untestable",
        "single_imputation": "understates",
        "mean_imputation": "shrinks",
        "p_value": "not that the effect is large",
    }
    for term, expected in consequences.items():
        assert expected in TERMS[term].lower(), f"'{term}' does not say what goes wrong"
