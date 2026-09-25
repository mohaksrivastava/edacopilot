# M3 acceptance audit: effect size + CI (Section 6.7)

M3's acceptance criterion (Section 16) is "every test returns effect size +
CI", and Section 6.7 says "All return `TestResult` including effect size with
CI (bootstrap if no analytic CI)".

**Status after m3.4 (2026-09-25): met.** 36 of 45 test functions return an
effect size with a confidence interval; the remaining 9 are exempt, each for a
reason recorded below. The m3.3 audit (2026-09-24) found only 9 of 45
compliant; this document records what each of the other 27 now returns and
against what R reference it is checked.

The criterion is no longer audited by hand. `test_m3_effect_size_criterion.py`
calls every registered `kind="test"` function in stage `hypothesis` and fails
CI if a non-exempt one returns no effect size, no CI, or a CI that does not
contain its own point estimate — and equally if an *exempt* one starts
returning an effect size, which would make its exemption stale. That test's
`_EXEMPT` dict is the machine-readable twin of the exemption table below, and a
further test asserts the two do not drift apart.

The per-number checks against R live in each module's own test file, not here.

## Conventions that apply throughout

- **One-sided intervals.** Proportion-of-variance measures (η², partial η²,
  ω², Cramér's V, Cohen's w, φ) are nonnegative by construction, so their
  intervals follow `effectsize`'s `alternative="greater"` default: a
  noncentral-distribution lower bound with the upper bound fixed at the
  measure's maximum (1, or `sqrt(1/min(p) - 1)` for a goodness-of-fit w).
- **The estimate is sometimes the effect size.** For a correlation, a
  proportion, a relative effect, a trimmed-mean difference or a 2×2 odds
  ratio, `effect_size` mirrors `estimate` and `effect_size_ci` mirrors `ci`
  rather than inventing a second interval.
- **Unstandardized measures carry no magnitude label.** `effect_magnitude`
  stays `None` for the measures listed in `effect_sizes.UNLABELLED_MEASURES`
  (proportions, raw and trimmed mean differences, odds ratios, the
  Brunner–Munzel relative effect, mutual information, dCor). Calling a mean
  difference "medium" without units would be meaningless.
- **TOST is reported at 1 − 2α**, not 1 − α, for both its intervals — the
  level at which "the interval lies inside the bounds" and "both one-sided
  tests reject" are the same statement, and what TOSTER itself reports.

## One sample

| Function | Effect size | CI method | R reference |
|---|---|---|---|
| `one_sample_t` | Cohen's d | noncentral t (df = n−1) | `effectsize::cohens_d(mu=)` |
| `wilcoxon_one_sample` | rank-biserial | Fisher z, signed-rank SE | `effectsize::rank_biserial(mu=)` |
| `sign_test` | proportion | Clopper–Pearson (exact) | `stats::binom.test` |
| `binomial_test` | Cohen's h | exact: h(p) is monotone, so the Clopper–Pearson bounds for p map onto h | `stats::binom.test` + the arcsine transform |
| `chi2_goodness_of_fit` | Cohen's w | noncentral χ², one-sided | `effectsize::cohens_w(p=)` |
| `bootstrap_one_sample` | — (exempt) | BCa bootstrap on `ci` | Section 6.7 specifies "CI only" |

## Two independent groups

| Function | Effect size | CI method | R reference |
|---|---|---|---|
| `student_t` | Hedges' g | noncentral t | `effectsize::hedges_g` |
| `welch_t` | Hedges' g(av) | noncentral t on the **Welch** df | `effectsize::hedges_g(pooled_sd=FALSE)` |
| `yuen_trimmed_t` | trimmed-mean difference | Yuen's own t interval (shared with `ci`) | `WRS2::yuen` |
| `mann_whitney` | rank-biserial | Fisher z, Mann–Whitney SE | `effectsize::rank_biserial` |
| `brunner_munzel` | relative effect | p̂ ± t(df)·SE, the test's own df | `brunnermunzel::brunnermunzel.test`'s `conf.int` |
| `permutation_test_2s` | mean difference | BCa bootstrap (seeded) | none — resampling |
| `bootstrap_diff` | — (exempt) | BCa bootstrap on `ci` | Section 6.7 specifies "CI of diff" |
| `ks_two_sample` | — (exempt) | — | D is the effect size; no standard CI |

`welch_t` was the one outright bug the audit found: up to m3.3 it reported
Cohen's d(av) under the name `hedges_g` — right denominator, missing the J
small-sample correction, and no interval.

## Two paired groups

| Function | Effect size | CI method | R reference |
|---|---|---|---|
| `paired_t` | d_z | noncentral t | `effectsize::cohens_d(paired=TRUE)` |
| `wilcoxon_signed_rank` | rank-biserial (paired) | Fisher z, **signed-rank** SE — not Mann–Whitney's | `effectsize::rank_biserial(paired=TRUE)` |
| `sign_test_paired` | proportion | Clopper–Pearson (exact) | `stats::binom.test` |
| `permutation_test_paired` | mean difference | BCa bootstrap over pairs (seeded) | none — resampling |

## k independent / k related groups

| Function | Effect size | CI method | R reference |
|---|---|---|---|
| `one_way_anova` | η² (ω² in `validity_notes`) | noncentral F, one-sided | `effectsize::eta_squared` |
| `welch_anova` | ω² | noncentral F on the Welch df | `effectsize::omega_squared(oneway.test(...))` |
| `alexander_govern` | — (exempt) | — | Section 6.7 lists "—" |
| `kruskal_wallis` | ε² | BCa bootstrap (stratified) | `effectsize::rank_epsilon_squared` |
| `permutation_anova` | η² | noncentral F, one-sided | `effectsize::eta_squared` |
| `repeated_measures_anova` | partial η² | noncentral F on the **uncorrected** df | `effectsize::eta_squared(afex::aov_ez(...))` |
| `friedman` | Kendall's W | BCa bootstrap over subjects | `effectsize::kendalls_w` |
| `cochran_q` | — (exempt) | — | Section 6.7 lists "—" |

The sphericity correction changes a repeated-measures test's df, not its
effect size: `effectsize::eta_squared` on the afex fit returns exactly
`F_to_eta2` at the uncorrected df, which the fixture records so the Python
test can assert the identity rather than assume it.

## Factorial

| Function | Effect size | CI method | R reference |
|---|---|---|---|
| `two_way_anova` (per term) | partial η² | noncentral F, one-sided | `effectsize::F_to_eta2` on `car::Anova(type=3)` |
| `aligned_rank_transform_anova` (per term) | partial η² | noncentral F, one-sided | `effectsize::F_to_eta2` on `ARTool::art` + `anova` |

Section 6.7 is silent on a factorial effect size. Partial η² per term is what
`effectsize::eta_squared` reports for these objects and the only choice that is
comparable across terms of one model. For the ART version, read it as an effect
size for the rank-transformed response, not for the raw outcome.

## Categorical association

| Function | Effect size | CI method | R reference |
|---|---|---|---|
| `chi2_independence` | Cramér's V | noncentral χ², one-sided | `effectsize::cramers_v` |
| `fisher_exact` 2×2 | conditional-MLE odds ratio | exact conditional (shared with `ci`) | `stats::fisher.test` |
| `fisher_exact` r×c | — (exempt) | — | no odds ratio beyond 2×2 |
| `g_test` | Cramér's V | noncentral χ², one-sided | `effectsize::cramers_v` |
| `mcnemar` | odds ratio b01/b10 | exact conditional: p/(1−p) at `binom.test`'s Clopper–Pearson bounds | constructed (base R reports no effect size) |
| `two_proportion_z` | Cohen's h (risk difference in `estimate` + `ci`) | arcsine | `effectsize::cohens_h` |
| `cochran_armitage_trend` | — (exempt) | — | Section 6.7 lists "—" |

`mcnemar` returns no interval when there are no discordant pairs, with a
warning: the odds ratio is undefined there, not merely imprecise.

## Correlation

| Function | Effect size | CI method | R reference |
|---|---|---|---|
| `pearson` | r | Fisher z (shared with `ci`) | `stats::cor.test` |
| `point_biserial` | r | Fisher z (shared with `ci`) | `stats::cor.test` |
| `spearman` | ρ | Fisher z, SE = 1/√(n−3) | `DescTools::SpearmanRho` |
| `kendall_tau` | τ-b | delta-method ASE from the joint table | `DescTools::KendallTauB` |
| `partial_correlation` | partial r | Fisher z, SE = 1/√(n−k−3) | **none** — see below |
| `correlation_matrix` | per pair, as its method | as its method | as its method |
| `distance_correlation` | — (exempt) | — | no analytic CI; a bootstrap of dCor is biased near 0 |
| `mutual_information` | — (exempt) | — | no standard CI for the KSG estimator |

`cor.test` reports no interval for ρ or τ, which is why DescTools is the
reference for both. τ-b's interval is deliberately *not* a Fisher-z one: the
delta-method variance accounts for ties, and on the tied fixture the two
disagree materially (a test asserts that, so the shortcut cannot creep back
in). That table is (distinct x) × (distinct y), i.e. n × n for tie-free
continuous data, so above `MAX_KENDALL_TABLE_CELLS` (4M cells, about n = 2000
tie-free) the CI degrades to Fisher z with a warning saying so; the estimate
and p-value are unaffected.

**`partial_correlation` is the one interval in edacore with no independent
reference.** No installed R package reports a CI for a partial correlation, so
the fixture evaluates the same published Fisher-z formula in R that the Python
side implements. That is a check on the arithmetic, not on the method. What the
Python test does establish independently is that the df is n − k − 3 and
responds to the number of control variables.

## Distribution / equivalence / other

| Function | Effect size | CI method | R reference |
|---|---|---|---|
| `anderson_ksamp` | — (exempt) | — | distribution-equality test; no standard effect size |
| `tost_equivalence` | Hedges' g (pooled or av, following `var_equal`) | noncentral t at 1 − 2α | `TOSTER::t_TOST`'s `effsize`, and `effectsize::hedges_g(ci=1-2*alpha)` |
| `runs_test` | — (exempt) | — | randomness test; no standard effect size |

## Exemptions (9)

| Function | Reason |
|---|---|
| `ks_two_sample` | D is the effect size; no standard CI exists for it |
| `alexander_govern` | Section 6.7 lists "—" |
| `cochran_q` | Section 6.7 lists "—" |
| `cochran_armitage_trend` | Section 6.7 lists "—" |
| `anderson_ksamp` | distribution-equality test; no standard effect size |
| `runs_test` | randomness test; no standard effect size |
| `distance_correlation` | no analytic CI; a bootstrap of dCor is biased near 0 |
| `mutual_information` | no standard CI for the KSG estimator |
| `bootstrap_one_sample` / `bootstrap_diff` | Section 6.7 specifies "CI only" / "CI of diff" — the estimate *is* the result, carried in `ci` |
| `fisher_exact` r×c | no odds ratio beyond a 2×2 table (the 2×2 path is not exempt) |

## Not covered by the criterion: post-hoc procedures

`PairwiseComparison` has `estimate` and `ci` but no effect-size fields. Only
`tukey_hsd` returns a per-pair CI; `games_howell`, `permutation_posthoc` and
`paired_posthoc` return an estimate without one, and `dunn_test`,
`nemenyi_friedman`, `conover_friedman` and `mcnemar_posthoc` return neither.
Section 16's criterion says "every **test**", and a `PostHocResult` is not a
`TestResult`, so these are out of scope for M3 rather than exempt from it.

Every pairwise `estimate` is `mean(group_a) - mean(group_b)` (checked), but
`tukey_hsd` orders pairs (B, A) while the others use (A, B): cosmetic only.

## Counts

| | m3.3 audit | after m3.4 |
|---|---|---|
| Effect size **with** CI | 9 | 36 |
| Effect size, no CI | 20 | 0 |
| CI computed elsewhere but not attached | 7 | 0 |
| Exempt | 9 | 9 |

45 rows: `fisher_exact` counts twice (2×2, r×c) and `correlation_matrix` not at
all (it inherits its method's status).
