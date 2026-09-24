#!/usr/bin/env Rscript
# Generates R reference values for edacore's tests (ARCHITECTURE.md Section
# 15.1). Every number edacore's Python tests compare against must come from
# running this script — never typed by hand. Re-run and commit the outputs
# whenever a new check/effect size is added or an input dataset changes.
#
# Two kinds of output land in tests/fixtures/r_reference/:
#   data/<name>.csv   the exact input data (so Python loads *this*, not a
#                      Python-side RNG draw that merely shares a seed —
#                      R and Python RNGs are not bit-compatible)
#   <name>.json        the R function's output, at full double precision
#
# Scope: ARCHITECTURE.md Section 6.6 (assumption checks) and Section 6.8
# (effect sizes, multiplicity, power) — the functions milestone M2 covers.
# Section 6.7 (hypothesis tests) is M3's; add its fixtures alongside that
# work, not here.

required_packages <- c(
  "jsonlite",   # fixture output
  "nortest",    # Anderson-Darling, Lilliefors
  "moments",    # skewness, kurtosis, D'Agostino test
  "car",        # Levene's test, VIF, Durbin-Watson
  "lmtest",     # Breusch-Pagan, Harvey-Collier, RESET
  "effectsize", # cohens_d, hedges_g, glass_delta, eta/omega/epsilon^2, ...
  "pwr"         # power analysis
)
missing_packages <- required_packages[!sapply(required_packages, requireNamespace, quietly = TRUE)]
if (length(missing_packages) > 0) {
  stop(
    "Missing required R packages: ", paste(missing_packages, collapse = ", "), "\n",
    "Install with: install.packages(c(", paste(sprintf('"%s"', missing_packages), collapse = ", "), "))"
  )
}

# Run this from the repo root (e.g. `Rscript scripts/generate_r_fixtures.R`).
# Deliberately not auto-detecting the script's own location: that requires
# parsing commandArgs() for "--file=", which is absent (and silently wrong)
# under `Rscript -e 'source(...)'` or when sourced from an R session.
repo_root <- getwd()
if (!dir.exists(file.path(repo_root, "tests")) || !file.exists(file.path(repo_root, "pyproject.toml"))) {
  stop(
    "Run this script from the edacopilot repo root (expected ./tests and ",
    "./pyproject.toml under the current directory: ", repo_root, ")"
  )
}
out_dir <- file.path(repo_root, "tests", "fixtures", "r_reference")
data_dir <- file.path(out_dir, "data")
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

write_fixture <- function(name, obj) {
  jsonlite::write_json(obj, file.path(out_dir, paste0(name, ".json")), auto_unbox = TRUE, digits = 15, na = "null")
  invisible(NULL)
}

write_data <- function(name, df) {
  write.csv(df, file.path(data_dir, paste0(name, ".csv")), row.names = FALSE)
  invisible(NULL)
}

set.seed(20260101)

# ---------------------------------------------------------------------------
# Datasets shared across several checks/effect sizes
# ---------------------------------------------------------------------------

n_normal <- 60
normal_sample <- rnorm(n_normal, mean = 50, sd = 10)
write_data("normal_sample", data.frame(x = normal_sample))

n_skewed <- 60
skewed_sample <- rexp(n_skewed, rate = 1 / 10)
write_data("skewed_sample", data.frame(x = skewed_sample))

n_group <- 40
group_equal_var <- data.frame(
  value = c(rnorm(n_group, mean = 100, sd = 15), rnorm(n_group, mean = 108, sd = 15)),
  group = rep(c("A", "B"), each = n_group)
)
write_data("two_groups_equal_var", group_equal_var)

group_unequal_var <- data.frame(
  value = c(rnorm(n_group, mean = 100, sd = 8), rnorm(n_group, mean = 100, sd = 28)),
  group = rep(c("A", "B"), each = n_group)
)
write_data("two_groups_unequal_var", group_unequal_var)

n_pair <- 35
paired_before <- rnorm(n_pair, mean = 70, sd = 12)
paired_after <- paired_before + rnorm(n_pair, mean = 4, sd = 6)
write_data("paired_before_after", data.frame(before = paired_before, after = paired_after))

n_k <- 30
k_group_df <- data.frame(
  value = c(rnorm(n_k, 50, 10), rnorm(n_k, 55, 10), rnorm(n_k, 62, 10)),
  group = rep(c("A", "B", "C"), each = n_k)
)
write_data("three_groups", k_group_df)

# 2x2 contingency table: treatment vs outcome
contingency_2x2 <- matrix(c(30, 20, 15, 35), nrow = 2, byrow = TRUE,
                           dimnames = list(exposure = c("exposed", "unexposed"), outcome = c("event", "no_event")))
write_data("contingency_2x2", as.data.frame(as.table(contingency_2x2)))

# 3x3 contingency table (general r x c, for bias-corrected Cramer's V)
contingency_3x3 <- matrix(c(20, 15, 5, 10, 25, 15, 5, 10, 20), nrow = 3, byrow = TRUE,
                           dimnames = list(row = c("r1", "r2", "r3"), col = c("c1", "c2", "c3")))
write_data("contingency_3x3", as.data.frame(as.table(contingency_3x3)))

# Regression-style x/y data with mild heteroscedasticity and a touch of
# autocorrelation in residuals, for linearity / homoscedasticity / DW checks
n_reg <- 80
x_reg <- seq(1, 40, length.out = n_reg)
ar_noise <- as.numeric(arima.sim(model = list(ar = 0.5), n = n_reg, sd = 3))
y_reg <- 5 + 1.8 * x_reg + ar_noise + rnorm(n_reg, sd = 0.3 * x_reg)
write_data("regression_xy", data.frame(x = x_reg, y = y_reg))

# Monotonic-but-nonlinear pair (for check_monotonicity)
x_mono <- seq(1, 50, length.out = 50)
y_mono <- log(x_mono) + rnorm(50, sd = 0.15)
write_data("monotonic_xy", data.frame(x = x_mono, y = y_mono))

# Multicollinear predictors (for VIF)
n_vif <- 100
x1_vif <- rnorm(n_vif)
x2_vif <- rnorm(n_vif)
x3_vif <- 0.9 * x1_vif + 0.1 * rnorm(n_vif) # strongly collinear with x1
y_vif <- 1 + 0.5 * x1_vif + 0.3 * x2_vif + rnorm(n_vif, sd = 0.5)
write_data("vif_predictors", data.frame(y = y_vif, x1 = x1_vif, x2 = x2_vif, x3 = x3_vif))

# Time-ordered series for Ljung-Box (autocorrelated) and its residuals
n_ts <- 100
ts_series <- as.numeric(arima.sim(model = list(ar = 0.6), n = n_ts))
write_data("autocorrelated_series", data.frame(t = seq_len(n_ts), x = ts_series))

# Repeated-measures wide data (subject x 4 conditions) for Mauchly's test
n_subj <- 25
rm_base <- rnorm(n_subj, 50, 8)
rm_wide <- data.frame(
  subject = seq_len(n_subj),
  t1 = rm_base + rnorm(n_subj, 0, 3),
  t2 = rm_base + rnorm(n_subj, 2, 4),
  t3 = rm_base + rnorm(n_subj, 5, 6),
  t4 = rm_base + rnorm(n_subj, 4, 9)
)
write_data("repeated_measures_wide", rm_wide)

# p-values for multiplicity adjustment
raw_pvalues <- c(0.001, 0.008, 0.02, 0.03, 0.04, 0.06, 0.09, 0.15, 0.4, 0.8)
write_data("raw_pvalues", data.frame(p = raw_pvalues))

# ---------------------------------------------------------------------------
# 6.6 Assumption checks
# ---------------------------------------------------------------------------

# check_normality_shapiro
sh_normal <- shapiro.test(normal_sample)
write_fixture("check_normality_shapiro__normal_sample", list(
  r_function = "stats::shapiro.test", data = "normal_sample.csv",
  statistic = unname(sh_normal$statistic), p_value = sh_normal$p.value
))
sh_skewed <- shapiro.test(skewed_sample)
write_fixture("check_normality_shapiro__skewed_sample", list(
  r_function = "stats::shapiro.test", data = "skewed_sample.csv",
  statistic = unname(sh_skewed$statistic), p_value = sh_skewed$p.value
))

# check_normality_dagostino
dag_normal <- moments::agostino.test(normal_sample)
write_fixture("check_normality_dagostino__normal_sample", list(
  r_function = "moments::agostino.test", data = "normal_sample.csv",
  statistic = unname(dag_normal$statistic["skew"]), p_value = dag_normal$p.value
))
dag_skewed <- moments::agostino.test(skewed_sample)
write_fixture("check_normality_dagostino__skewed_sample", list(
  r_function = "moments::agostino.test", data = "skewed_sample.csv",
  statistic = unname(dag_skewed$statistic["skew"]), p_value = dag_skewed$p.value
))

# check_normality_anderson
ad_normal <- nortest::ad.test(normal_sample)
write_fixture("check_normality_anderson__normal_sample", list(
  r_function = "nortest::ad.test", data = "normal_sample.csv",
  statistic = unname(ad_normal$statistic), p_value = ad_normal$p.value
))
ad_skewed <- nortest::ad.test(skewed_sample)
write_fixture("check_normality_anderson__skewed_sample", list(
  r_function = "nortest::ad.test", data = "skewed_sample.csv",
  statistic = unname(ad_skewed$statistic), p_value = ad_skewed$p.value
))

# check_normality_lilliefors
lf_normal <- nortest::lillie.test(normal_sample)
write_fixture("check_normality_lilliefors__normal_sample", list(
  r_function = "nortest::lillie.test", data = "normal_sample.csv",
  statistic = unname(lf_normal$statistic), p_value = lf_normal$p.value
))
lf_skewed <- nortest::lillie.test(skewed_sample)
write_fixture("check_normality_lilliefors__skewed_sample", list(
  r_function = "nortest::lillie.test", data = "skewed_sample.csv",
  statistic = unname(lf_skewed$statistic), p_value = lf_skewed$p.value
))

# check_normality_descriptive (skewness / excess kurtosis)
write_fixture("check_normality_descriptive__normal_sample", list(
  r_function = "moments::skewness, moments::kurtosis", data = "normal_sample.csv",
  skewness = moments::skewness(normal_sample),
  excess_kurtosis = moments::kurtosis(normal_sample) - 3
))
write_fixture("check_normality_descriptive__skewed_sample", list(
  r_function = "moments::skewness, moments::kurtosis", data = "skewed_sample.csv",
  skewness = moments::skewness(skewed_sample),
  excess_kurtosis = moments::kurtosis(skewed_sample) - 3
))

# qq_correlation (probability-plot correlation coefficient; base R, no package)
qq_corr <- function(x) {
  x <- sort(x)
  cor(x, qnorm(ppoints(length(x))))
}
write_fixture("qq_correlation__normal_sample", list(
  r_function = "cor(sort(x), qnorm(ppoints(length(x))))", data = "normal_sample.csv",
  r = qq_corr(normal_sample)
))
write_fixture("qq_correlation__skewed_sample", list(
  r_function = "cor(sort(x), qnorm(ppoints(length(x))))", data = "skewed_sample.csv",
  r = qq_corr(skewed_sample)
))

# check_equal_variance_levene (car::leveneTest, center="median" = Brown-Forsythe)
lev_equal <- car::leveneTest(value ~ group, data = group_equal_var, center = "median")
write_fixture("check_equal_variance_levene__equal_var", list(
  r_function = 'car::leveneTest(center="median")', data = "two_groups_equal_var.csv",
  statistic = lev_equal[1, "F value"], df1 = lev_equal[1, "Df"], df2 = lev_equal[2, "Df"],
  p_value = lev_equal[1, "Pr(>F)"]
))
lev_unequal <- car::leveneTest(value ~ group, data = group_unequal_var, center = "median")
write_fixture("check_equal_variance_levene__unequal_var", list(
  r_function = 'car::leveneTest(center="median")', data = "two_groups_unequal_var.csv",
  statistic = lev_unequal[1, "F value"], df1 = lev_unequal[1, "Df"], df2 = lev_unequal[2, "Df"],
  p_value = lev_unequal[1, "Pr(>F)"]
))

# check_equal_variance_bartlett
bart_equal <- bartlett.test(value ~ group, data = group_equal_var)
write_fixture("check_equal_variance_bartlett__equal_var", list(
  r_function = "stats::bartlett.test", data = "two_groups_equal_var.csv",
  statistic = unname(bart_equal$statistic), df = unname(bart_equal$parameter), p_value = bart_equal$p.value
))
bart_unequal <- bartlett.test(value ~ group, data = group_unequal_var)
write_fixture("check_equal_variance_bartlett__unequal_var", list(
  r_function = "stats::bartlett.test", data = "two_groups_unequal_var.csv",
  statistic = unname(bart_unequal$statistic), df = unname(bart_unequal$parameter), p_value = bart_unequal$p.value
))

# check_equal_variance_fligner
fk_equal <- fligner.test(value ~ group, data = group_equal_var)
write_fixture("check_equal_variance_fligner__equal_var", list(
  r_function = "stats::fligner.test", data = "two_groups_equal_var.csv",
  statistic = unname(fk_equal$statistic), df = unname(fk_equal$parameter), p_value = fk_equal$p.value
))
fk_unequal <- fligner.test(value ~ group, data = group_unequal_var)
write_fixture("check_equal_variance_fligner__unequal_var", list(
  r_function = "stats::fligner.test", data = "two_groups_unequal_var.csv",
  statistic = unname(fk_unequal$statistic), df = unname(fk_unequal$parameter), p_value = fk_unequal$p.value
))

# check_variance_ratio (descriptive; base R, no package)
variance_ratio <- function(df) {
  sds <- tapply(df$value, df$group, sd)
  max(sds) / min(sds)
}
write_fixture("check_variance_ratio__equal_var", list(
  r_function = "max(sd)/min(sd) by group", data = "two_groups_equal_var.csv",
  ratio = variance_ratio(group_equal_var)
))
write_fixture("check_variance_ratio__unequal_var", list(
  r_function = "max(sd)/min(sd) by group", data = "two_groups_unequal_var.csv",
  ratio = variance_ratio(group_unequal_var)
))

# check_expected_counts (stats::chisq.test$expected)
chi_expected <- suppressWarnings(chisq.test(contingency_2x2))$expected
write_fixture("check_expected_counts__contingency_2x2", list(
  r_function = "chisq.test(x)$expected", data = "contingency_2x2.csv",
  expected = as.vector(t(chi_expected)),
  min_expected = min(chi_expected),
  pct_cells_below_5 = mean(chi_expected < 5) * 100
))

# check_sphericity_mauchly
rm_matrix <- as.matrix(rm_wide[, c("t1", "t2", "t3", "t4")])
rm_model <- lm(rm_matrix ~ 1)
mauchly_result <- mauchly.test(rm_model, X = ~1)
write_fixture("check_sphericity_mauchly__repeated_measures", list(
  r_function = "stats::mauchly.test", data = "repeated_measures_wide.csv",
  statistic = unname(mauchly_result$statistic), p_value = unname(mauchly_result$p.value)
))

# check_linearity: Harvey-Collier + RESET (lmtest), against the regression fit
reg_fit <- lm(y ~ x, data = data.frame(x = x_reg, y = y_reg))
hc_test <- lmtest::harvtest(reg_fit)
reset_test <- lmtest::resettest(reg_fit, power = 2:3, type = "fitted")
write_fixture("check_linearity__regression_xy", list(
  r_function = "lmtest::harvtest, lmtest::resettest", data = "regression_xy.csv",
  harvey_collier_statistic = unname(hc_test$statistic), harvey_collier_p_value = hc_test$p.value,
  reset_statistic = unname(reset_test$statistic), reset_p_value = reset_test$p.value
))

# check_monotonicity: Spearman correlation (base R)
spear <- cor.test(x_mono, y_mono, method = "spearman")
write_fixture("check_monotonicity__monotonic_xy", list(
  r_function = 'cor.test(method="spearman")', data = "monotonic_xy.csv",
  rho = unname(spear$estimate), p_value = spear$p.value
))

# check_homoscedasticity_bp (Breusch-Pagan)
bp_test <- lmtest::bptest(reg_fit)
write_fixture("check_homoscedasticity_bp__regression_xy", list(
  r_function = "lmtest::bptest", data = "regression_xy.csv",
  statistic = unname(bp_test$statistic), df = unname(bp_test$parameter), p_value = bp_test$p.value
))

# check_autocorrelation_dw (Durbin-Watson)
dw_test <- car::durbinWatsonTest(reg_fit)
write_fixture("check_autocorrelation_dw__regression_xy", list(
  r_function = "car::durbinWatsonTest", data = "regression_xy.csv",
  statistic = dw_test$dw, p_value = dw_test$p
))

# check_ljung_box
lb_test <- Box.test(ts_series, lag = 10, type = "Ljung-Box")
write_fixture("check_ljung_box__autocorrelated_series", list(
  r_function = 'stats::Box.test(type="Ljung-Box")', data = "autocorrelated_series.csv",
  lag = 10, statistic = unname(lb_test$statistic), df = unname(lb_test$parameter), p_value = lb_test$p.value
))

# check_multicollinearity_vif
vif_fit <- lm(y ~ x1 + x2 + x3, data = data.frame(y = y_vif, x1 = x1_vif, x2 = x2_vif, x3 = x3_vif))
vif_values <- car::vif(vif_fit)
write_fixture("check_multicollinearity_vif__vif_predictors", list(
  r_function = "car::vif", data = "vif_predictors.csv",
  vif = as.list(vif_values)
))

# check_same_shape: two-sample KS on centred/scaled groups (base R)
center_scale <- function(x) (x - mean(x)) / sd(x)
grp_a <- group_equal_var$value[group_equal_var$group == "A"]
grp_b <- group_equal_var$value[group_equal_var$group == "B"]
ks_shape <- suppressWarnings(ks.test(center_scale(grp_a), center_scale(grp_b)))
write_fixture("check_same_shape__two_groups_equal_var", list(
  r_function = "stats::ks.test on centred/scaled groups", data = "two_groups_equal_var.csv",
  statistic = unname(ks_shape$statistic), p_value = ks_shape$p.value
))

# ---------------------------------------------------------------------------
# 6.8 Effect sizes
# ---------------------------------------------------------------------------

es_two_groups <- effectsize::cohens_d(value ~ group, data = group_equal_var)
write_fixture("cohens_d__two_groups_equal_var", list(
  r_function = "effectsize::cohens_d", data = "two_groups_equal_var.csv",
  estimate = es_two_groups$Cohens_d, ci_low = es_two_groups$CI_low, ci_high = es_two_groups$CI_high
))

es_hedges <- effectsize::hedges_g(value ~ group, data = group_equal_var)
write_fixture("hedges_g__two_groups_equal_var", list(
  r_function = "effectsize::hedges_g", data = "two_groups_equal_var.csv",
  estimate = es_hedges$Hedges_g, ci_low = es_hedges$CI_low, ci_high = es_hedges$CI_high
))

es_glass <- effectsize::glass_delta(value ~ group, data = group_unequal_var)
write_fixture("glass_delta__two_groups_unequal_var", list(
  r_function = "effectsize::glass_delta", data = "two_groups_unequal_var.csv",
  estimate = es_glass$Glass_delta, ci_low = es_glass$CI_low, ci_high = es_glass$CI_high
))

es_dz <- effectsize::cohens_d(paired_after, paired_before, paired = TRUE)
write_fixture("d_z__paired_before_after", list(
  r_function = "effectsize::cohens_d(paired=TRUE)", data = "paired_before_after.csv",
  estimate = es_dz$Cohens_d, ci_low = es_dz$CI_low, ci_high = es_dz$CI_high
))

es_rb <- effectsize::rank_biserial(value ~ group, data = group_equal_var)
write_fixture("rank_biserial__two_groups_equal_var", list(
  r_function = "effectsize::rank_biserial", data = "two_groups_equal_var.csv",
  estimate = es_rb$r_rank_biserial, ci_low = es_rb$CI_low, ci_high = es_rb$CI_high
))

# For two independent samples, Cliff's delta and the rank-biserial
# correlation are the same quantity; effectsize labels the column
# "r_rank_biserial" here too rather than "Cliffs_delta".
es_cliffs <- effectsize::cliffs_delta(value ~ group, data = group_equal_var)
write_fixture("cliffs_delta__two_groups_equal_var", list(
  r_function = "effectsize::cliffs_delta", data = "two_groups_equal_var.csv",
  estimate = es_cliffs$r_rank_biserial, ci_low = es_cliffs$CI_low, ci_high = es_cliffs$CI_high
))

aov_fit <- aov(value ~ group, data = k_group_df)
es_eta2 <- effectsize::eta_squared(aov_fit, partial = FALSE)
write_fixture("eta_squared__three_groups", list(
  r_function = "effectsize::eta_squared(partial=FALSE)", data = "three_groups.csv",
  estimate = es_eta2$Eta2, ci_low = es_eta2$CI_low, ci_high = es_eta2$CI_high
))

# For one-way designs, partial eta^2 == eta^2 (effectsize warns and returns
# eta^2 under the same "Eta2" column, not a separate "Eta2_partial").
es_peta2 <- effectsize::eta_squared(aov_fit, partial = TRUE)
write_fixture("partial_eta_squared__three_groups", list(
  r_function = "effectsize::eta_squared(partial=TRUE)", data = "three_groups.csv",
  estimate = es_peta2$Eta2, ci_low = es_peta2$CI_low, ci_high = es_peta2$CI_high
))

es_omega2 <- effectsize::omega_squared(aov_fit)
write_fixture("omega_squared__three_groups", list(
  r_function = "effectsize::omega_squared", data = "three_groups.csv",
  estimate = es_omega2$Omega2, ci_low = es_omega2$CI_low, ci_high = es_omega2$CI_high
))

kw_fit <- kruskal.test(value ~ group, data = k_group_df)
es_epsilon2 <- effectsize::rank_epsilon_squared(value ~ group, data = k_group_df)
write_fixture("epsilon_squared__three_groups", list(
  r_function = "effectsize::rank_epsilon_squared", data = "three_groups.csv",
  kruskal_statistic = unname(kw_fit$statistic), kruskal_p_value = kw_fit$p.value,
  estimate = es_epsilon2$rank_epsilon_squared, ci_low = es_epsilon2$CI_low, ci_high = es_epsilon2$CI_high
))

rm_long <- data.frame(
  subject = rep(rm_wide$subject, times = 4),
  condition = factor(rep(c("t1", "t2", "t3", "t4"), each = n_subj)),
  value = c(rm_wide$t1, rm_wide$t2, rm_wide$t3, rm_wide$t4)
)
write_data("repeated_measures_long", rm_long)
es_kendall_w <- effectsize::kendalls_w(value ~ condition | subject, data = rm_long)
write_fixture("kendalls_w__repeated_measures", list(
  r_function = "effectsize::kendalls_w", data = "repeated_measures_long.csv",
  estimate = es_kendall_w$Kendalls_W, ci_low = es_kendall_w$CI_low, ci_high = es_kendall_w$CI_high
))

es_cramers_v <- effectsize::cramers_v(contingency_3x3, adjust = TRUE)
write_fixture("cramers_v__contingency_3x3", list(
  r_function = "effectsize::cramers_v(adjust=TRUE)", data = "contingency_3x3.csv",
  estimate = es_cramers_v$Cramers_v_adjusted, ci_low = es_cramers_v$CI_low, ci_high = es_cramers_v$CI_high
))

es_phi <- effectsize::phi(contingency_2x2, adjust = FALSE)
write_fixture("phi__contingency_2x2", list(
  r_function = "effectsize::phi", data = "contingency_2x2.csv",
  estimate = es_phi$phi, ci_low = es_phi$CI_low, ci_high = es_phi$CI_high
))

# oddsratio()/riskratio() (like cohens_h above) read the table as
# columns = compared groups, row 1 = "success"; our matrix has groups as
# rows, so transpose. (Odds ratio happens to be transpose-invariant --
# ad/bc is the same either way -- but risk ratio is NOT: leaving it
# untransposed silently computed P(exposed|event)/P(exposed|no_event)
# instead of the intended P(event|exposed)/P(event|unexposed).)
es_or <- effectsize::oddsratio(t(contingency_2x2))
write_fixture("odds_ratio__contingency_2x2", list(
  r_function = "effectsize::oddsratio(t(x))", data = "contingency_2x2.csv",
  estimate = es_or$Odds_ratio, ci_low = es_or$CI_low, ci_high = es_or$CI_high
))

es_rr <- effectsize::riskratio(t(contingency_2x2))
write_fixture("risk_ratio__contingency_2x2", list(
  r_function = "effectsize::riskratio(t(x))", data = "contingency_2x2.csv",
  estimate = es_rr$Risk_ratio, ci_low = es_rr$CI_low, ci_high = es_rr$CI_high
))

# effectsize has no risk-difference function (only riskratio/oddsratio); a
# difference of proportions with its Wald CI is exactly what
# stats::prop.test computes for a 2-sample table, so use that instead.
rd_x <- contingency_2x2[, "event"]
rd_n <- rowSums(contingency_2x2)
rd_test <- prop.test(rd_x, rd_n, correct = FALSE)
write_fixture("risk_difference__contingency_2x2", list(
  r_function = "stats::prop.test(correct=FALSE)", data = "contingency_2x2.csv",
  estimate = unname(rd_test$estimate[1] - rd_test$estimate[2]),
  ci_low = rd_test$conf.int[1], ci_high = rd_test$conf.int[2]
))

p1 <- contingency_2x2["exposed", "event"] / sum(contingency_2x2["exposed", ])
p2 <- contingency_2x2["unexposed", "event"] / sum(contingency_2x2["unexposed", ])
# cohens_h(x) takes a 2x2 table, not two scalar proportions: it reads
# p1/p2 as Obs[1,1]/colSum1 and Obs[1,2]/colSum2, so columns must be the
# two compared groups. Our matrix has groups as rows, so transpose it.
es_h <- effectsize::cohens_h(t(contingency_2x2))
write_fixture("cohens_h__contingency_2x2", list(
  r_function = "effectsize::cohens_h(t(x))", data = "contingency_2x2.csv",
  p1 = unname(p1), p2 = unname(p2), estimate = es_h$Cohens_h,
  ci_low = es_h$CI_low, ci_high = es_h$CI_high
))

chi_3x3 <- chisq.test(contingency_3x3)
es_w <- effectsize::cohens_w(contingency_3x3)
write_fixture("cohens_w__contingency_3x3", list(
  r_function = "effectsize::cohens_w", data = "contingency_3x3.csv",
  chisq_statistic = unname(chi_3x3$statistic), estimate = es_w$Cohens_w
))

# ---------------------------------------------------------------------------
# 6.8 Multiplicity
# ---------------------------------------------------------------------------

write_fixture("adjust_pvalues__raw_pvalues", list(
  r_function = "stats::p.adjust", data = "raw_pvalues.csv",
  holm = p.adjust(raw_pvalues, method = "holm"),
  bonferroni = p.adjust(raw_pvalues, method = "bonferroni"),
  fdr_bh = p.adjust(raw_pvalues, method = "BH"),
  fdr_by = p.adjust(raw_pvalues, method = "BY")
))

# ---------------------------------------------------------------------------
# 6.8 Power
# ---------------------------------------------------------------------------

pwr_t <- pwr::pwr.t.test(d = 0.5, sig.level = 0.05, power = 0.8, type = "two.sample")
write_fixture("required_sample_size__two_sample_t", list(
  r_function = "pwr::pwr.t.test", d = 0.5, sig.level = 0.05, power = 0.8,
  n_per_group = pwr_t$n
))

pwr_anova <- pwr::pwr.anova.test(k = 3, f = 0.25, sig.level = 0.05, power = 0.8)
write_fixture("required_sample_size__anova", list(
  r_function = "pwr::pwr.anova.test", k = 3, f = 0.25, sig.level = 0.05, power = 0.8,
  n_per_group = pwr_anova$n
))

pwr_r <- pwr::pwr.r.test(r = 0.3, sig.level = 0.05, power = 0.8)
write_fixture("required_sample_size__correlation", list(
  r_function = "pwr::pwr.r.test", r = 0.3, sig.level = 0.05, power = 0.8,
  n = pwr_r$n
))

pwr_chisq <- pwr::pwr.chisq.test(w = 0.3, df = 4, sig.level = 0.05, power = 0.8)
write_fixture("required_sample_size__chisq", list(
  r_function = "pwr::pwr.chisq.test", w = 0.3, df = 4, sig.level = 0.05, power = 0.8,
  n = pwr_chisq$N
))

pwr_mde_t <- pwr::pwr.t.test(n = 40, sig.level = 0.05, power = 0.8, type = "two.sample")
write_fixture("minimum_detectable_effect__two_sample_t", list(
  r_function = "pwr::pwr.t.test", n = 40, sig.level = 0.05, power = 0.8,
  d = pwr_mde_t$d
))

cat("Wrote fixtures to", out_dir, "\n")
