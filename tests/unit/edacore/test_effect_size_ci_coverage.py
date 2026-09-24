"""Simulated coverage for the two effect sizes whose CI is a genuine
bootstrap with no R fixture to match bit-for-bit: kendalls_w and
epsilon_squared (see effect_sizes.py's module docstring for why — R
itself bootstraps these too, just with plain percentile rather than BCa).

Generates many independent datasets from a known population, computes each
one's 95% CI, and checks the fraction of intervals containing the true
population value falls in [0.93, 0.97] — a tolerance band around the
nominal 95%, wide enough to absorb Monte Carlo noise at 500 trials
(binomial SE at p=0.95, n=500 is ~1%, so +/-2pp is a ~2-sigma band) without
being so loose it would pass a badly miscalibrated CI.

Runs in a few seconds per test now that the BCa bootstrap is pure numpy
(see effect_sizes.py's `_bca_matrix_row_bootstrap_ci` /
`_bca_grouped_bootstrap_ci`); still marked slow since 500 trials is a lot
of test-suite time for two tests. `pytest -m "not slow"` skips these.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import edacore

pytestmark = pytest.mark.slow


def _true_epsilon_squared(means: list[float], sd: float, n_per_group: int) -> float:
    """Population rank-epsilon-squared for k normal groups, approximated by
    a large one-off Monte Carlo sample (not part of the per-trial loop)."""
    rng = np.random.default_rng(999)
    groups = [rng.normal(m, sd, n_per_group * 50) for m in means]
    h_stat, _ = stats.kruskal(*groups)
    n_total = sum(len(g) for g in groups)
    return float(h_stat / (n_total - 1))


def _true_kendalls_w(
    n_subj: int, condition_effects: np.ndarray, sd_subject: float, sd_noise: float
) -> float:
    """Population Kendall's W, approximated the same way."""
    rng = np.random.default_rng(999)
    n_big = n_subj * 200
    subject_fx = rng.normal(0, sd_subject, (n_big, 1))
    data = subject_fx + condition_effects + rng.normal(0, sd_noise, (n_big, len(condition_effects)))
    fr_stat, _ = stats.friedmanchisquare(*[data[:, j] for j in range(len(condition_effects))])
    return float(fr_stat / (n_big * (len(condition_effects) - 1)))


def test_epsilon_squared_ci_coverage() -> None:
    means = [50.0, 55.0, 62.0]
    sd = 10.0
    n_per_group = 30
    true_value = _true_epsilon_squared(means, sd, n_per_group)

    n_trials = 500
    covered = 0
    for trial in range(n_trials):
        rng = np.random.default_rng(trial)
        df = pd.DataFrame(
            {
                "value": np.concatenate([rng.normal(m, sd, n_per_group) for m in means]),
                "group": sum(([f"g{i}"] * n_per_group for i in range(len(means))), []),
            }
        )
        result = edacore.effect_sizes.epsilon_squared(df, "value", "group")
        if result["ci_low"] <= true_value <= result["ci_high"]:
            covered += 1

    coverage = covered / n_trials
    msg = f"epsilon_squared 95% CI covered {coverage:.1%} of {n_trials} trials"
    assert 0.93 <= coverage <= 0.97, msg


def test_kendalls_w_ci_coverage() -> None:
    n_subj, k = 25, 4
    condition_effects = np.array([0.0, 2.0, 5.0, 4.0])
    sd_subject, sd_noise = 8.0, 3.0
    true_value = _true_kendalls_w(n_subj, condition_effects, sd_subject, sd_noise)

    n_trials = 500
    covered = 0
    for trial in range(n_trials):
        rng = np.random.default_rng(trial)
        subject_fx = rng.normal(0, sd_subject, (n_subj, 1))
        values = subject_fx + condition_effects + rng.normal(0, sd_noise, (n_subj, k))
        df = pd.DataFrame(
            {
                "subject": np.repeat(np.arange(n_subj), k),
                "condition": np.tile(np.arange(k), n_subj),
                "value": values.ravel(),
            }
        )
        result = edacore.effect_sizes.kendalls_w(df, "value", "condition", "subject")
        if result["ci_low"] <= true_value <= result["ci_high"]:
            covered += 1

    coverage = covered / n_trials
    msg = f"kendalls_w 95% CI covered {coverage:.1%} of {n_trials} trials"
    assert 0.93 <= coverage <= 0.97, msg
