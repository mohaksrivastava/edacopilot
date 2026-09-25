# Persona picks by scenario

**Generated** by `scripts/generate_persona_picks.py` from the live persona
policies, the eligibility engine and the template rationales. Do not edit by
hand: `tests/unit/edacopilot/test_persona_picks_doc.py` fails if this file and a
fresh render disagree.

Each row is one persona's proposal for one scenario, with the one-line rationale
it would show. The rationales are Section 10.5's deterministic templates -- what
the system says with the LLM switched off, and what it falls back to when an
LLM-written rationale fails its fact-check (Section 8.4).

Rule 2 holds throughout: a persona chooses among what the eligibility engine
(Section 7) already ruled valid, and can never propose an INELIGIBLE method.

## The personas (Section 8.1)

| Persona | Method pool | Normality | CLT shortcut | Preference order |
|---|---|---|---|---|
| **Professor** | `exact`, `nonparametric`, `parametric` | borderline is **fail** | none | `assumptions_clearly_met` > `exact_over_asymptotic` |
| **Consultant** | `nonparametric`, `parametric` | borderline is **pass** | Cochran's rule (n > 25*skew^2) | `fewest_steps` > `interpretability` > `robust_default` |
| **Maverick** | `maverick`, `nonparametric_advanced`, `resampling`, `robust` | borderline is **fail** | none | `robust` > `resampling` > `informative_effect_sizes` |

## Picks

### worked example 7.4

> income by gender, n = 38/41, strong right skew (ARCHITECTURE.md Section 7.4)

Family: `two_independent_numeric`. Card: **divergence**.

> Professor + Consultant → Mann-Whitney. Maverick → Yuen's trimmed-mean t-test.

| Persona | Pick | Status | Rationale |
|---|---|---|---|
| **Professor** | `mann_whitney` | `eligible` | Mann-Whitney is the method whose assumptions this data actually meets; every assumption it makes is checked and holds here; though independence depends on how the data was collected, not just its values. |
| **Consultant** | `mann_whitney` | `eligible` | Use Mann-Whitney. |
| **Maverick** | `yuen_trimmed_t` | `eligible` | Yuen's trimmed-mean t-test is worth a look here; it assumes less than the standard choice; though independence depends on how the data was collected, not just its values. |

### clean two groups

> normal, equal variance, equal n -- the case where nothing is compromised

Family: `two_independent_numeric`. Card: **consensus**.

> All three personas reach the same conclusion: Professor + Maverick → Student's t-test, Consultant → Welch's t-test. With equal variances confirmed, Welch's correction is negligible and the two tests give the same answer.

| Persona | Pick | Status | Rationale |
|---|---|---|---|
| **Professor** | `student_t` | `eligible` | Student's t-test is the method whose assumptions this data actually meets; every assumption it makes is checked and holds here; though independence depends on how the data was collected, not just its values. |
| **Consultant** | `welch_t` | `eligible` | Use Welch's t-test. |
| **Maverick** | `student_t` | `eligible` | Maverick concurs with Professor on Student's t-test: every assumption these methods make is met, so an alternative would add unfamiliarity without adding information. |

### heteroscedastic_groups

> unequal variances and unequal n (Section 15.3)

Family: `two_independent_numeric`. Card: **divergence**.

> Professor + Consultant → Welch's t-test. Maverick → Yuen's trimmed-mean t-test.

| Persona | Pick | Status | Rationale |
|---|---|---|---|
| **Professor** | `welch_t` | `eligible` | Welch's t-test is the method whose assumptions this data actually meets; every assumption it makes is checked and holds here; though independence depends on how the data was collected, not just its values. |
| **Consultant** | `welch_t` | `eligible` | Use Welch's t-test. |
| **Maverick** | `yuen_trimmed_t` | `eligible` | Yuen's trimmed-mean t-test is worth a look here; it assumes less than the standard choice; though independence depends on how the data was collected, not just its values. |

### heavy_tails_small_n

> t(2) tails at n = 12 per group (Section 15.3)

Family: `two_independent_numeric`. Card: **divergence**.

> Professor + Consultant → Mann-Whitney. Maverick → Yuen's trimmed-mean t-test.

| Persona | Pick | Status | Rationale |
|---|---|---|---|
| **Professor** | `mann_whitney` | `eligible` | Mann-Whitney is the method whose assumptions this data actually meets; every assumption it makes is checked and holds here; though independence depends on how the data was collected, not just its values. |
| **Consultant** | `mann_whitney` | `eligible` | Use Mann-Whitney. |
| **Maverick** | `yuen_trimmed_t` | `eligible` | Yuen's trimmed-mean t-test is worth a look here; it assumes less than the standard choice; though independence depends on how the data was collected, not just its values. |

### paired_as_independent (design confirmed paired)

> the trap resolved: the user confirmed the repeated-measures design (Section 15.3)

Family: `two_paired_numeric`. Card: **consensus**.

> All three personas agree: a paired t-test.

| Persona | Pick | Status | Rationale |
|---|---|---|---|
| **Professor** | `paired_t` | `eligible` | A paired t-test is the method whose assumptions this data actually meets; every assumption it makes is checked and holds here. |
| **Consultant** | `paired_t` | `eligible` | Use a paired t-test. |
| **Maverick** | `paired_t` | `eligible` | Maverick concurs with Professor on a paired t-test: every assumption these methods make is met, so an alternative would add unfamiliarity without adding information. |

### ordinal_as_numeric (confirmed ordinal)

> a 1-5 Likert scale, confirmed as ordinal rather than measured (Section 15.3)

Family: `ordinal_any`. Card: **consensus**.

> All three personas agree: Spearman's correlation.

| Persona | Pick | Status | Rationale |
|---|---|---|---|
| **Professor** | `spearman` | `eligible` | Spearman's correlation is the method whose assumptions this data actually meets; every assumption it makes is checked and holds here. |
| **Consultant** | `spearman` | `eligible` | Use Spearman's correlation. |
| **Maverick** | `spearman` | `eligible` | Maverick concurs with Professor on Spearman's correlation: nothing in this family is inside this persona's method pool. |

## How to read a relaxed status

A persona may upgrade a CAVEAT to ELIGIBLE only through a rule its own YAML
declares, and only for soft assumptions:

- `clt_shortcut: cochran` clears a **normality** caveat when Cochran's rule
  (n > 25*skew^2 per group) is met. Note that `normality_or_large_n` is already
  decided by Cochran's rule inside the eligibility engine, because Section 6.7
  grants those methods the escape by name -- so the persona shortcut only bites
  on methods declaring plain `normality`, such as one-way ANOVA.
- `borderline_is: pass` clears a caveat **all** of whose causes are BORDERLINE.
  One FAIL anywhere blocks it, so a persona cannot reach eligibility by
  declining to look at a check.
- `variance.strategy: always_robust` is a *tightening*: the persona declines to
  propose a method that assumes equal variance when that assumption is failing,
  rather than waiving the caveat on it.

Hard assumptions are never touched by any of these.
