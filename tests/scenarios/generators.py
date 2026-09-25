"""Synthetic "trap" datasets (ARCHITECTURE.md, Section 15.3).

Each generator plants one problem a junior analyst commonly misses. Scenarios
are added alongside the milestone that exercises them, not before: M1 added
ordinal_as_numeric, sentinel_missing, target_leakage, duplicate_rows and
pii_present; M4 adds the three the eligibility engine is judged on
(paired_as_independent, heteroscedastic_groups, heavy_tails_small_n) and
reuses ordinal_as_numeric. Still pending: mar_by_group -> M10,
non_stationary_ts -> M12, simpsons_paradox, huge_n_tiny_effect,
many_comparisons.

Every generator is seeded for reproducibility — there is no run-to-run
variance to chase down if a scenario test fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ordinal_as_numeric(n: int = 200) -> pd.DataFrame:
    """A 1-5 Likert scale stored as plain int. Must be flagged as an ordinal
    candidate, not treated as a generic discrete/continuous numeric column."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "customer_id": range(1, n + 1),
            "satisfaction": rng.integers(1, 6, size=n),
            "tenure_months": rng.integers(1, 120, size=n),
        }
    )


def sentinel_missing(n: int = 200) -> pd.DataFrame:
    """-999 used in place of NaN for missing income. Must be detected as a
    sentinel before any numeric summary is computed from raw values."""
    rng = np.random.default_rng(1)
    income = rng.normal(60000, 15000, size=n).round(2)
    income[rng.random(n) < 0.08] = -999
    return pd.DataFrame({"customer_id": range(1, n + 1), "income": income})


def target_leakage(n: int = 200) -> tuple[pd.DataFrame, str]:
    """A column that is a direct copy of the target. Must be flagged in
    PROFILE. Returns (df, target_column_name)."""
    rng = np.random.default_rng(2)
    outcome = rng.integers(0, 2, size=n)
    df = pd.DataFrame(
        {
            "customer_id": range(1, n + 1),
            "outcome": outcome,
            "outcome_encoded": outcome,
            "tenure_months": rng.integers(1, 120, size=n),
        }
    )
    return df, "outcome"


def duplicate_rows(n: int = 200, duplicate_frac: float = 0.05) -> pd.DataFrame:
    """~5% exact duplicate rows appended to the dataset. Must be detected in
    PROFILE."""
    rng = np.random.default_rng(3)
    base = pd.DataFrame(
        {
            "customer_id": range(1, n + 1),
            "age": rng.integers(18, 80, size=n),
            "region": rng.choice(["A", "B", "C"], size=n),
        }
    )
    n_dupes = max(1, round(n * duplicate_frac))
    dupes = base.sample(n=n_dupes, random_state=3)
    return pd.concat([base, dupes], ignore_index=True)


def pii_present(n: int = 200) -> pd.DataFrame:
    """Email and phone columns. Strict mode should be suggested and nothing
    from these columns should ever reach the LLM."""
    rng = np.random.default_rng(4)
    return pd.DataFrame(
        {
            "customer_id": range(1, n + 1),
            "email": [f"user{i}@example.com" for i in range(n)],
            "phone": [f"555-{100 + (i % 800):03d}-{1000 + i:04d}" for i in range(n)],
            "age": rng.integers(18, 80, size=n),
        }
    )


def paired_as_independent(n_subjects: int = 30) -> pd.DataFrame:
    """The same subjects measured before and after, stored long, so the
    table looks exactly like two independent groups.

    Section 15.3: the system must raise the design ambiguity. Nothing about
    the VALUES gives this away -- only `subject_id` appearing under both
    levels of `condition` does, which is why Section 7.1 cross-checks ids
    rather than trusting the shape of the data.

    The effect is real but modest, and the within-subject correlation is
    strong: treating these as independent has ample power to be wrong in an
    interesting way rather than failing to reject either way.
    """
    rng = np.random.default_rng(11)
    subject_level = rng.normal(100, 15, n_subjects)
    before = subject_level + rng.normal(0, 4, n_subjects)
    after = subject_level + 6 + rng.normal(0, 4, n_subjects)
    return pd.DataFrame(
        {
            "subject_id": np.tile(np.arange(1, n_subjects + 1), 2),
            "condition": ["before"] * n_subjects + ["after"] * n_subjects,
            "score": np.concatenate([before, after]),
        }
    )


def heteroscedastic_groups(n_a: int = 60, n_b: int = 25) -> pd.DataFrame:
    """Unequal variances AND unequal group sizes -- the combination that
    makes Student's t actively misleading rather than merely inefficient.

    Section 15.3: Student t -> CAVEAT, Welch preferred. The larger group
    has the smaller variance, which is the direction that makes the pooled
    test anti-conservative.
    """
    rng = np.random.default_rng(12)
    return pd.DataFrame(
        {
            "value": np.concatenate([rng.normal(50, 4, n_a), rng.normal(53, 16, n_b)]),
            "group": ["A"] * n_a + ["B"] * n_b,
        }
    )


def heavy_tails_small_n(n_per_group: int = 12) -> pd.DataFrame:
    """t-distributed with 2 degrees of freedom at n=12 per group.

    Section 15.3: parametric methods -> CAVEAT. With df=2 the variance is
    undefined in the population, so the sample SD a t-test depends on is
    not estimating anything stable -- and at n=12 there is no CLT rescue.
    """
    rng = np.random.default_rng(13)
    return pd.DataFrame(
        {
            "value": np.concatenate(
                [
                    rng.standard_t(2, n_per_group) * 5 + 20,
                    rng.standard_t(2, n_per_group) * 5 + 24,
                ]
            ),
            "group": ["A"] * n_per_group + ["B"] * n_per_group,
        }
    )


def worked_example_7_4() -> pd.DataFrame:
    """ARCHITECTURE.md Section 7.4's worked example: income by gender,
    n = 38 and 41, strong right skew.

    The seed is chosen so every check in Section 7.4's table lands on the
    status that table gives (Shapiro FAIL in both groups at p < 0.001,
    descriptive skew BORDERLINE in both at 1.89 / 1.70 against the spec's
    1.9 / 2.2, Levene FAIL, sample size PASS, same-shape PASS). The exact
    statistics cannot match -- the spec quotes numbers from a dataset it
    does not ship -- so the reproduction is of the statuses and the
    candidate table they produce.
    """
    rng = np.random.default_rng(121)
    return pd.DataFrame(
        {
            "income": np.concatenate(
                [rng.lognormal(10.5, 0.45, 38), rng.lognormal(10.6, 0.62, 41)]
            ),
            "gender": ["F"] * 38 + ["M"] * 41,
        }
    )
