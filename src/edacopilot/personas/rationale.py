"""Template rationales: the deterministic fallback (Section 10.5).

Section 10.5 requires the system to run with the LLM switched off, and
Section 15's test suite runs that way always. These templates are what a
persona says then -- and what it falls back to when the LLM's output fails
the fact-check in Section 8.4.

Two constraints shape every sentence here:

- **Rule 1: the LLM never computes a statistic, and neither does this.**
  Every number in a rationale is interpolated from an `AssumptionCheck`
  that `edacore` already produced. There is no arithmetic in this module.
- **A rationale must say what goes wrong, not just what was chosen.**
  `AssumptionCheck.consequence` is written for a junior analyst for exactly
  this purpose (Section 6.6), so the templates quote it rather than
  paraphrasing it into something vaguer.

`explanation_style` selects the voice, and only the voice: the same facts
are cited whichever persona is speaking. A persona that reported different
facts depending on its style would be presenting a preference as evidence.
"""

from __future__ import annotations

from edacore.contracts import AssumptionCheck, CheckStatus, Eligibility

from .engine import PersonaPick, PersonaVerdict, Proposal
from .policy import PersonaPolicy, get_persona

# How each explanation_style opens a proposal. The trailing space is part
# of the template.
_OPENERS = {
    "precise": "{method} is the method whose assumptions this data actually meets",
    "brief": "Use {method}",
    "curious": "{method} is worth a look here",
}

_CAVEAT_OPENERS = {
    "precise": "{method} is the least compromised option, but not a clean one",
    "brief": "{method}, with a caveat worth knowing",
    "curious": "{method}, though it is not assumption-free either",
}


def _method_name(function: str) -> str:
    """A function name as prose: `mann_whitney` -> "Mann-Whitney"."""
    special = {
        "mann_whitney": "Mann-Whitney",
        "welch_t": "Welch's t-test",
        "student_t": "Student's t-test",
        "yuen_trimmed_t": "Yuen's trimmed-mean t-test",
        "brunner_munzel": "the Brunner-Munzel test",
        "ks_two_sample": "the Kolmogorov-Smirnov test",
        "kruskal_wallis": "Kruskal-Wallis",
        "one_way_anova": "one-way ANOVA",
        "welch_anova": "Welch's ANOVA",
        "alexander_govern": "the Alexander-Govern test",
        "permutation_test_2s": "a permutation test",
        "permutation_test_paired": "a paired permutation test",
        "permutation_anova": "a permutation ANOVA",
        "bootstrap_diff": "a bootstrap of the difference",
        "bootstrap_one_sample": "a bootstrap interval",
        "paired_t": "a paired t-test",
        "wilcoxon_signed_rank": "the Wilcoxon signed-rank test",
        "sign_test_paired": "the sign test",
        "sign_test": "the sign test",
        "one_sample_t": "a one-sample t-test",
        "wilcoxon_one_sample": "the Wilcoxon signed-rank test",
        "chi2_independence": "a chi-square test of independence",
        "chi2_goodness_of_fit": "a chi-square goodness-of-fit test",
        "fisher_exact": "Fisher's exact test",
        "g_test": "the G-test",
        "two_proportion_z": "a two-proportion test",
        "mcnemar": "McNemar's test",
        "binomial_test": "an exact binomial test",
        "repeated_measures_anova": "a repeated-measures ANOVA",
        "friedman": "the Friedman test",
        "cochran_q": "Cochran's Q",
        "two_way_anova": "a two-way ANOVA",
        "aligned_rank_transform_anova": "an aligned-rank-transform ANOVA",
        "pearson": "Pearson's correlation",
        "spearman": "Spearman's correlation",
        "kendall_tau": "Kendall's tau",
        "point_biserial": "a point-biserial correlation",
        "partial_correlation": "a partial correlation",
        "distance_correlation": "distance correlation",
        "mutual_information": "mutual information",
        "tost_equivalence": "a TOST equivalence test",
        "cochran_armitage_trend": "the Cochran-Armitage trend test",
    }
    return special.get(function, function.replace("_", " "))


def _blocking(pick: PersonaPick, checks: dict[str, AssumptionCheck]) -> list[AssumptionCheck]:
    if pick.candidate is None:
        return []
    return [
        checks[fact_id]
        for fact_id in pick.candidate.reasons
        if fact_id in checks
        and checks[fact_id].status in (CheckStatus.FAIL, CheckStatus.BORDERLINE)
    ]


def _untestable(pick: PersonaPick, checks: dict[str, AssumptionCheck]) -> list[AssumptionCheck]:
    if pick.candidate is None:
        return []
    return [
        checks[fact_id]
        for fact_id in pick.candidate.reasons
        if fact_id in checks and checks[fact_id].status is CheckStatus.UNTESTABLE
    ]


def _first_sentence(text: str) -> str:
    stripped = text.strip()
    for marker in ("; ", " -- "):
        if marker in stripped:
            stripped = stripped.split(marker)[0]
    return stripped.rstrip(".")


def render_rationale(
    pick: PersonaPick, checks: dict[str, AssumptionCheck], policy: PersonaPolicy | None = None
) -> str:
    """One line for one persona's pick, in that persona's voice.

    Every clause after the opener is built from an `AssumptionCheck` the
    engine produced -- the consequence sentence for a caveat, the rule text
    for a relaxation, the consequence for anything untestable.
    """
    policy = policy or get_persona(pick.persona)
    style = policy.explanation_style

    if pick.candidate is None:
        return (
            f"{policy.display_name} has no method to offer here: nothing in this family "
            f"is both eligible and inside this persona's method pool."
        )

    if pick.concurs_with:
        # A persona that defers still says why, in its own voice. Silence
        # would read as absence; the user should know the alternative was
        # considered and found unnecessary.
        with_whom = get_persona(pick.concurs_with).display_name
        return (
            f"{policy.display_name} concurs with {with_whom} on "
            f"{_method_name(pick.candidate.function)}: {pick.concur_note}."
        )

    method = _method_name(pick.candidate.function)
    caveated = pick.eligibility is Eligibility.CAVEAT
    opener = (_CAVEAT_OPENERS if caveated else _OPENERS)[style].format(method=method)

    parts: list[str] = [opener]

    blocking = _blocking(pick, checks)
    if caveated and blocking:
        worst = blocking[0]
        parts.append(_first_sentence(worst.consequence).lower())
    elif pick.relaxations:
        parts.append(_first_sentence(pick.relaxations[0]))
    elif style == "precise":
        parts.append("every assumption it makes is checked and holds here")
    elif style == "curious":
        parts.append("it assumes less than the standard choice")

    untestable = _untestable(pick, checks)
    if untestable and style != "brief":
        parts.append(f"though {_first_sentence(untestable[0].consequence).lower()}")

    sentence = "; ".join(parts) + "."
    return sentence[0].upper() + sentence[1:]


def render_verdict(verdict: PersonaVerdict, checks: list[AssumptionCheck]) -> PersonaVerdict:
    """Attach a rendered rationale to every pick and proposal."""
    by_id = {check.fact_id: check for check in checks}
    picks = [
        pick.model_copy(update={"rationale": render_rationale(pick, by_id)})
        for pick in verdict.picks
    ]
    by_persona = {pick.persona: pick for pick in picks}
    proposals = [
        proposal.model_copy(update={"rationale": by_persona[proposal.personas[0]].rationale})
        for proposal in verdict.proposals
    ]
    return verdict.model_copy(update={"picks": picks, "proposals": proposals})


def render_card(verdict: PersonaVerdict) -> str:
    """The card headline Section 8.3 describes, as plain text."""
    if verdict.is_consensus:
        only = verdict.proposals[0]
        if not only.function:
            return "No persona has a method to offer for this question."
        if only.is_equivalence:
            return (
                f"All three personas reach the same conclusion: "
                f"{_equivalence_line(only)}. {only.equivalence_note[0].upper()}"
                f"{only.equivalence_note[1:]}."
            )
        return f"All three personas agree: {_method_name(only.function)}."
    return " ".join(_proposal_line(p) for p in verdict.proposals)


def _equivalence_line(proposal: Proposal) -> str:
    """Name each persona's method, since they differ even though the answer
    does not -- the user should see what was actually proposed."""
    by_method: dict[str, list[str]] = {}
    for persona_id, function in proposal.equivalent_methods.items():
        by_method.setdefault(function, []).append(get_persona(persona_id).display_name)
    return ", ".join(
        f"{' + '.join(names)} → {_method_name(function)}" for function, names in by_method.items()
    )


def _proposal_line(proposal: Proposal) -> str:
    if not proposal.function:
        return f"{' + '.join(proposal.display_names)}: no method to offer."
    if proposal.is_equivalence:
        return f"{_equivalence_line(proposal)} ({proposal.equivalence_note})."
    return f"{' + '.join(proposal.display_names)} → {_method_name(proposal.function)}."
