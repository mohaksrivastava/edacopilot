# Eligibility table

**Generated** by `scripts/generate_eligibility_table.py` from the live function
registry and the Section 7.3 rules. Do not edit by hand:
`tests/unit/edacopilot/test_eligibility_table_doc.py` fails if this file and a
fresh render disagree, so an assumption cannot change without this table changing
with it.

This is what the eligibility engine actually enforces, per method family. Read it
against ARCHITECTURE.md Section 6.7's own tables: where they differ, one of the two
is wrong.

## How a status is decided (Section 7.2)

| Outcome | Rule |
|---|---|
| `INELIGIBLE` | any **hard** assumption's deciding check returns FAIL |
| `CAVEAT` | no hard failure, but a **soft** assumption returns FAIL or BORDERLINE |
| `ELIGIBLE` | neither |

UNTESTABLE never changes a status. Independence of sampling and exchangeability are
facts about how the data was collected, not about its values (Section 5.2), so they
are surfaced as reasons on every candidate that rests on them and left for the user
to vouch for. A method whose *question parameters* are missing (equivalence bounds, a
reference value, a covariate) is reported under `unavailable` rather than as
INELIGIBLE -- that is a question to ask, not a verdict about the data.

## Which check decides each assumption

| Assumption | Decided by | Rule |
|---|---|---|
| `balanced` | `check_balanced_design` | every subject appears exactly once at every level of the within factor |
| `binary` | `check_measurement_level` | binary |
| `binary_outcome` | `check_measurement_level` | binary |
| `bivariate_normality` | `check_normality_shapiro` | on each margin. Marginal normality does not imply joint normality; it is the standard practical proxy, and only a FAIL is conclusive |
| `categorical` | `check_measurement_level` | nominal, ordinal or binary |
| `categorical_outcome` | `check_measurement_level` | nominal, ordinal or binary |
| `continuous` | `check_measurement_level` | semantic type must be continuous |
| `equal_variance` | `check_equal_variance_levene` | median-centred (Brown-Forsythe) |
| `exchangeability` | -- (UNTESTABLE) | a property of how the data was collected |
| `expected_counts` | `check_expected_counts` + `check_expected_counts_gof` | all expected >= 1 and <= 20% of cells < 5; from the table's margins for a two-way question, from the reference proportions for a goodness-of-fit one |
| `independent` | `confirmed design` + `check_design_crossing` | the user's confirmed design must be `independent`, and no subject id may appear under more than one group. With no id column, `check_independence_design` is carried as UNTESTABLE so the candidate shows what it is assuming |
| `linearity` | `check_linearity` | RESET test |
| `min_n_per_group>=2` | `check_sample_size` | every group n >= 2 |
| `monotonicity` | `check_monotonicity` | fraction of the lowess smooth's movement that reverses direction |
| `no_influential_outliers` | `check_influential_outliers` | max Cook's D > 1 fail, > 0.5 borderline |
| `normality` | `check_normality_shapiro` + `check_normality_descriptive` | per group |
| `normality_of_differences` | `check_normality_shapiro` | on the per-subject differences, pivoted from long format when needed |
| `normality_or_large_n` | `check_normality_shapiro` + `check_normality_descriptive` | as `normality`, except that a failure is waived when every group has n >= 100 and \|skew\| < 2.0 (the CLT escape, a CONVENTION not a result); the Shapiro results are still reported as evidence |
| `normality_within_groups` | `check_normality_shapiro` | split by the binary column |
| `np_at_least_10` | `check_proportion_counts` | >= 10 successes and >= 10 failures per group |
| `numeric` | `check_measurement_level` | continuous or discrete |
| `numeric_or_ordinal` | `check_measurement_level` | continuous, discrete or ordinal |
| `numeric_outcome` | `check_measurement_level` | continuous or discrete |
| `ordered_sequence` | -- (UNTESTABLE) | whether the stored row order is the meaningful one |
| `ordinal_exposure` | `check_measurement_level` | ordinal |
| `ordinal_or_higher` | `check_measurement_level` | continuous, discrete, ordinal or binary |
| `paired` | `confirmed design` | the user's confirmed design must be `paired` |
| `paired_binary` | `check_measurement_level` | every paired column is binary |
| `repeated` | `confirmed design` | the user's confirmed design must be `repeated` |
| `same_shape` | `check_same_shape` | KS on centred/scaled groups; with k > 2 every pair, Holm-adjusted, worst pair decides |
| `sphericity` | `check_sphericity_mauchly` | Mauchly's W |
| `symmetry` | `check_symmetry` | \|skew\| < 0.5 pass, < 1 borderline |
| `symmetry_of_differences` | `check_symmetry` | on the per-subject differences |

## Method families (Section 7.3)

### `two_independent_numeric`

| Method | Hard assumptions | Soft assumptions |
|---|---|---|
| `student_t` | `numeric_outcome`, `independent`, `min_n_per_group>=2` | `normality_or_large_n`, `equal_variance` |
| `welch_t` | `numeric_outcome`, `independent` | `normality_or_large_n` |
| `yuen_trimmed_t` | `numeric_outcome`, `independent` | -- |
| `mann_whitney` | `ordinal_or_higher`, `independent` | `same_shape` |
| `brunner_munzel` | `ordinal_or_higher`, `independent` | -- |
| `permutation_test_2s` | `numeric_outcome`, `independent`, `exchangeability` | -- |
| `bootstrap_diff` | `numeric_outcome`, `independent` | -- |
| `ks_two_sample` | `continuous`, `independent` | -- |

### `two_paired_numeric`

| Method | Hard assumptions | Soft assumptions |
|---|---|---|
| `paired_t` | `numeric_outcome`, `paired` | `normality_of_differences` |
| `wilcoxon_signed_rank` | `numeric_or_ordinal`, `paired` | `symmetry_of_differences` |
| `sign_test_paired` | `ordinal_or_higher`, `paired` | -- |
| `permutation_test_paired` | `numeric_outcome`, `paired` | -- |

### `k_independent_numeric`

| Method | Hard assumptions | Soft assumptions |
|---|---|---|
| `one_way_anova` | `numeric_outcome`, `independent` | `normality`, `equal_variance` |
| `welch_anova` | `numeric_outcome`, `independent` | `normality_or_large_n` |
| `alexander_govern` | `numeric_outcome`, `independent` | `normality` |
| `kruskal_wallis` | `ordinal_or_higher`, `independent` | `same_shape` |
| `permutation_anova` | `numeric_outcome`, `independent` | -- |

### `k_repeated_numeric`

| Method | Hard assumptions | Soft assumptions |
|---|---|---|
| `repeated_measures_anova` | `balanced`, `numeric_outcome`, `repeated` | `normality`, `sphericity` |
| `friedman` | `ordinal_or_higher`, `repeated` | -- |

### `k_repeated_binary`

| Method | Hard assumptions | Soft assumptions |
|---|---|---|
| `cochran_q` | `binary_outcome`, `repeated` | -- |

### `factorial_numeric`

| Method | Hard assumptions | Soft assumptions |
|---|---|---|
| `two_way_anova` | `numeric_outcome`, `independent` | `normality`, `equal_variance` |
| `aligned_rank_transform_anova` | `ordinal_or_higher`, `independent` | -- |

### `binary_outcome_groups`

| Method | Hard assumptions | Soft assumptions |
|---|---|---|
| `two_proportion_z` | `binary_outcome`, `independent`, `np_at_least_10` | -- |
| `chi2_independence` | `categorical`, `expected_counts`, `independent` | -- |
| `fisher_exact` | `categorical`, `independent` | -- |

### `equivalence_two_groups`

| Method | Hard assumptions | Soft assumptions |
|---|---|---|
| `tost_equivalence` | `numeric_outcome`, `independent` | `normality_or_large_n` |

### `numeric_numeric`

| Method | Hard assumptions | Soft assumptions |
|---|---|---|
| `pearson` | `numeric` | `linearity`, `bivariate_normality`, `no_influential_outliers` |
| `spearman` | `ordinal_or_higher` | `monotonicity` |
| `kendall_tau` | `ordinal_or_higher` | -- |
| `distance_correlation` | `numeric` | -- |
| `mutual_information` | `numeric` | -- |

### `ordinal_any`

> cochran_armitage_trend only applies to a binary outcome against an ordinal exposure

| Method | Hard assumptions | Soft assumptions |
|---|---|---|
| `spearman` | `ordinal_or_higher` | `monotonicity` |
| `kendall_tau` | `ordinal_or_higher` | -- |
| `cochran_armitage_trend` | `binary_outcome`, `ordinal_exposure` | -- |

### `binary_numeric`

| Method | Hard assumptions | Soft assumptions |
|---|---|---|
| `point_biserial` | `binary`, `numeric` | `normality_within_groups` |
| `mann_whitney` | `ordinal_or_higher`, `independent` | `same_shape` |

### `two_categorical_independent`

| Method | Hard assumptions | Soft assumptions |
|---|---|---|
| `chi2_independence` | `categorical`, `expected_counts`, `independent` | -- |
| `fisher_exact` | `categorical`, `independent` | -- |
| `g_test` | `categorical`, `expected_counts`, `independent` | -- |

### `two_binary_paired`

| Method | Hard assumptions | Soft assumptions |
|---|---|---|
| `mcnemar` | `paired_binary`, `paired` | -- |

### `numeric_numeric_controlled`

> Section 7.3 lists partial_correlation under the correlation methods without giving it a family of its own; it gets one here because it is the only association method that needs a third variable, and offering it without one would always fail

| Method | Hard assumptions | Soft assumptions |
|---|---|---|
| `partial_correlation` | `numeric` | `linearity` |

### `one_sample_numeric`

| Method | Hard assumptions | Soft assumptions |
|---|---|---|
| `one_sample_t` | `numeric_outcome` | `normality_or_large_n` |
| `wilcoxon_one_sample` | `numeric_or_ordinal` | `symmetry` |
| `sign_test` | `ordinal_or_higher` | -- |
| `bootstrap_one_sample` | `numeric_outcome` | -- |

### `one_sample_categorical`

| Method | Hard assumptions | Soft assumptions |
|---|---|---|
| `binomial_test` | `binary_outcome` | -- |
| `chi2_goodness_of_fit` | `categorical_outcome`, `expected_counts` | -- |

## Goals with no methods yet

| Goal | Served by |
|---|---|
| `describe` | profiling (Section 6.1), which describes rather than tests |
| `missingness` | M10 (Section 6.3) |
| `outliers` | M11 (Section 6.4) |
| `transform` | M11 (Section 6.5) |
| `trend` | M12 (Section 6.9): the `ts_stationarity` family will offer `stationarity_verdict`, `stl_decompose`, `detect_change_points` |

Asking for one of these raises `UnsupportedQuestionError` naming the milestone,
rather than reporting perfectly good methods as INELIGIBLE: they are not ineligible,
they do not exist yet.

## Coverage

- Goals with rules: `association`, `compare_groups`, `distribution_fit`, `equivalence`, `trend`
- Families: 16
- Distinct assumption names in use: 33
