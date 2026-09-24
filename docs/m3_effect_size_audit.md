# M3 acceptance audit: effect size + CI (Section 6.7)

Audit for m3.3 (2026-09-24). M3's acceptance criterion (Section 16) is "every
test returns effect size + CI", and Section 6.7 says "All return `TestResult`
including effect size with CI (bootstrap if no analytic CI)".

**Result: the criterion is not met.** 9 of 45 test functions fully satisfy it.
Method: every registered `kind="test"` function in stage `hypothesis` was
called on synthetic data suited to its design, recording what it actually
returns (not what its docstring says): `estimate`/`ci` and
`effect_size`/`effect_size_ci`.

Legend:
- **OK**: effect size and its CI are returned.
- **WIRE**: the CI already exists elsewhere in edacore (an R-verified
  `effect_sizes` function, or this function's own `ci`) but isn't put in
  `effect_size_ci`. Mechanical fix, no new statistics.
- **ADD**: an effect size is returned without a CI; an R reference for the
  CI exists (named), so it can be added and verified the usual way.
- **EXEMPT (proposed)**: no effect size or no standard CI, with the reason.

## One sample

| Function | Effect size returned | CI | Status | Note / R reference for the fix |
|---|---|---|---|---|
| `one_sample_t` | Cohen's d | no | ADD | `effectsize::cohens_d(x, mu=)`: noncentral-t, `_ncp_ci` already in edacore |
| `wilcoxon_one_sample` | rank-biserial | no | ADD | `effectsize::rank_biserial(x, mu=)` |
| `sign_test` | proportion | no | ADD | Clopper-Pearson, `binom.test` |
| `binomial_test` | Cohen's h | no (`ci` holds the CI for p) | ADD | h is monotone in p: transform `binom.test`'s CI |
| `chi2_goodness_of_fit` | Cohen's w | no | ADD | `effectsize::cohens_w(x, p=)`: noncentral chi², `_ncx2_ci_low` exists |
| `bootstrap_one_sample` | (estimate itself) | yes (`ci`) | OK | spec: "CI only" |

## Two independent groups

| Function | Effect size returned | CI | Status | Note / R reference for the fix |
|---|---|---|---|---|
| `student_t` | Cohen's d | no | WIRE | spec says Hedges' g; `effect_sizes.hedges_g` has an R-verified CI |
| `welch_t` | **labelled `hedges_g`, actually Cohen's d_av without the J correction** | no | ADD + **bug** | decision needed (see below); `effectsize::hedges_g(pooled_sd=FALSE)` |
| `yuen_trimmed_t` | none (trimmed-mean difference is in `estimate`, with `ci`) | in `ci` | WIRE | spec: trimmed-mean difference (ξ optional) |
| `mann_whitney` | rank-biserial | no | WIRE | `effect_sizes.rank_biserial` has an R-verified CI |
| `brunner_munzel` | name `relative_effect`, value only in `estimate` | no | ADD | `brunnermunzel::brunnermunzel.test` reports the CI of p-hat |
| `permutation_test_2s` | none (mean diff in `estimate`) | no | ADD | spec: bootstrap if no analytic CI (`bootstrap_effect_ci`) |
| `bootstrap_diff` | (estimate itself) | yes (`ci`) | OK | spec: "CI of diff" |
| `ks_two_sample` | none (D is `statistic`) | no | EXEMPT (proposed) | D is the effect size; no standard CI for it |

## Two paired groups

| Function | Effect size returned | CI | Status | Note / R reference for the fix |
|---|---|---|---|---|
| `paired_t` | d_z | no | WIRE | `effect_sizes.d_z` has an R-verified CI |
| `wilcoxon_signed_rank` | rank-biserial | no | ADD | `effectsize::rank_biserial(paired=TRUE)` |
| `sign_test_paired` | proportion | no | ADD | Clopper-Pearson, `binom.test` |
| `permutation_test_paired` | none (mean diff in `estimate`) | no | ADD | bootstrap (spec rule) |

## k independent / k related groups

| Function | Effect size returned | CI | Status | Note / R reference for the fix |
|---|---|---|---|---|
| `one_way_anova` | η² (ω² in `validity_notes`) | yes | OK | |
| `welch_anova` | ω² (F-based) | no | ADD | `effectsize::omega_squared(oneway.test(...))` |
| `alexander_govern` | none | no | EXEMPT (proposed) | spec lists "—" |
| `kruskal_wallis` | ε² | yes | OK | |
| `permutation_anova` | η² | yes | OK | |
| `repeated_measures_anova` | partial η² | no | ADD | `effectsize::eta_squared(afex fit)` |
| `friedman` | Kendall's W | yes | OK | |
| `cochran_q` | none | no | EXEMPT (proposed) | spec lists "—" |

## Factorial

| Function | Effect size returned | CI | Status | Note / R reference for the fix |
|---|---|---|---|---|
| `two_way_anova` (per term) | none | no | ADD | spec is silent; partial η² per term, `effectsize::eta_squared(car::Anova(...))` |
| `aligned_rank_transform_anova` (per term) | none | no | ADD | partial η² on the aligned-rank F, `effectsize::eta_squared(ARTool anova)` |

## Categorical association

| Function | Effect size returned | CI | Status | Note / R reference for the fix |
|---|---|---|---|---|
| `chi2_independence` | Cramér's V | yes | OK | |
| `fisher_exact` 2×2 | conditional-MLE odds ratio | in `ci` only | WIRE | copy `ci` into `effect_size_ci` |
| `fisher_exact` r×c | none | no | EXEMPT (proposed) | no odds ratio beyond 2×2; could add Cramér's V (as chi2) |
| `g_test` | Cramér's V | yes | OK | |
| `mcnemar` | odds ratio b/c | no | ADD | exact conditional CI: `binom.test(b, b+c)` transformed to p/(1-p) |
| `two_proportion_z` | Cohen's h (risk difference in `estimate` + `ci`) | yes | OK | |
| `cochran_armitage_trend` | none | no | EXEMPT (proposed) | spec lists "—" |

## Correlation

| Function | Effect size returned | CI | Status | Note / R reference for the fix |
|---|---|---|---|---|
| `pearson` | r in `estimate` | in `ci` | WIRE | r *is* the effect size; set the `effect_size*` fields |
| `point_biserial` | r in `estimate` | in `ci` | WIRE | as pearson |
| `correlation_matrix` | per pair, as its method | as its method | follows the method | |
| `spearman` | rho in `estimate` | no | ADD | decision needed (see below); `DescTools::SpearmanRho(conf.level=)` |
| `kendall_tau` | tau-b in `estimate` | no | ADD | decision needed (see below); `DescTools::KendallTauB(conf.level=)` |
| `partial_correlation` | partial r in `estimate` | no | ADD | Fisher z with n - 3 - k; no R package installed that reports it |
| `distance_correlation` | dCor in `estimate` | no | EXEMPT (proposed) | no analytic CI; a bootstrap is possible but biased near 0 |
| `mutual_information` | MI in `estimate` | no | EXEMPT (proposed) | no standard CI for the KSG estimator |

## Distribution / equivalence / other

| Function | Effect size returned | CI | Status | Note / R reference for the fix |
|---|---|---|---|---|
| `anderson_ksamp` | none | no | EXEMPT (proposed) | a distribution-equality test; no standard effect size |
| `tost_equivalence` | none (mean diff in `estimate`) | no | ADD | `TOSTER::t_TOST` reports the 1 - 2α CI of the difference and Hedges' g + CI |
| `runs_test` | none | no | EXEMPT (proposed) | randomness test; no standard effect size |

## Post-hoc procedures (`PostHocResult`)

`PairwiseComparison` has `estimate` and `ci` but no effect-size fields. Only
`tukey_hsd` returns a per-pair CI. `games_howell`, `permutation_posthoc` and
`paired_posthoc` return an estimate without a CI. `dunn_test`,
`nemenyi_friedman`, `conover_friedman` and `mcnemar_posthoc` return neither.
Section 16's criterion says "every test", so post-hocs are arguably out of
scope. Listed here for completeness.

Every pairwise `estimate` is `mean(group_a) - mean(group_b)` (checked), but
`tukey_hsd` orders pairs (B, A) while the others use (A, B): cosmetic only.

## Counts

| Status | Rows |
|---|---|
| OK | 9 |
| WIRE | 7 |
| ADD | 20 |
| EXEMPT (proposed) | 9 |

45 rows: `fisher_exact` counts twice (2×2, r×c) and `correlation_matrix`
not at all (it inherits its method's status).
