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
# (effect sizes, multiplicity, power) from M2/M2.1; Section 6.7's
# one-sample, two-independent-group, and two-paired-group hypothesis tests
# from M3 part 1; Section 6.7's k-independent-group, k-related-group, and
# post-hoc tests from M3 part 2a. Factorial, categorical association, and
# correlation (the rest of Section 6.7) are later M3 work; add fixtures
# alongside that, not here.

required_packages <- c(
  "jsonlite",       # fixture output
  "nortest",        # Anderson-Darling, Lilliefors
  "moments",        # skewness, kurtosis, D'Agostino test
  "car",            # Levene's test, VIF, Durbin-Watson
  "lmtest",         # Breusch-Pagan, Harvey-Collier, RESET
  "effectsize",     # cohens_d, hedges_g, glass_delta, eta/omega/epsilon^2, ...
  "pwr",            # power analysis
  "WRS2",           # yuen_trimmed_t (M3)
  "brunnermunzel",  # brunner_munzel (M3)
  "PMCMRplus",      # Games-Howell, Friedman post-hoc Nemenyi/Conover (M3 part 2a)
  "FSA",            # Dunn's test, matched at its default (two-sided, Holm) (M3 part 2a)
  "afex",           # repeated-measures ANOVA with GG/HF correction (M3 part 2a)
  "DescTools",      # Cochran's Q (M3 part 2a)
  "onewaytests"     # Alexander-Govern test (M3 part 2a)
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

# Recursively checks a fixture for NULL/NA before it's written. A NULL
# almost always means a column name that doesn't exist on the R object
# (silently returns NULL via `$`) -- exactly the bug that produced
# cliffs_delta's and partial_eta_squared's first (empty) fixtures in M2.
# NA usually means an upstream computation failed quietly. Either way it's
# a bug to fix in this script, never something to work around downstream.
check_no_null_na <- function(obj, path) {
  if (is.null(obj)) {
    stop("Fixture value is NULL at '", path, "' -- likely a wrong column name ",
         "(R's $ returns NULL silently); fix the source computation, don't ",
         "patch around it downstream.")
  }
  if (is.list(obj)) {
    nms <- names(obj)
    for (i in seq_along(obj)) {
      key <- if (!is.null(nms) && nzchar(nms[i])) nms[i] else as.character(i)
      check_no_null_na(obj[[i]], paste0(path, ".", key))
    }
  } else if ((is.numeric(obj) || is.logical(obj)) && any(is.na(obj))) {
    stop("Fixture value is NA at '", path, "' -- an upstream computation ",
         "failed quietly; fix the source computation, don't patch around ",
         "it downstream.")
  }
  invisible(NULL)
}

write_fixture <- function(name, obj) {
  check_no_null_na(obj, name)
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
  chisq_statistic = unname(chi_3x3$statistic), estimate = es_w$Cohens_w,
  ci_low = es_w$CI_low, ci_high = es_w$CI_high
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

# ---------------------------------------------------------------------------
# 6.7 Hypothesis tests: one sample, two independent groups, two paired
# groups (M3 part 1). Reuses normal_sample, two_groups_equal_var/
# unequal_var, and paired_before/paired_after from the M2 section above.
# ---------------------------------------------------------------------------

## One sample --------------------------------------------------------------

mu0 <- 48

t1 <- t.test(normal_sample, mu = mu0)
write_fixture("one_sample_t__normal_sample", list(
  r_function = "stats::t.test", data = "normal_sample.csv", mu0 = mu0,
  statistic = unname(t1$statistic), df = unname(t1$parameter), p_value = t1$p.value,
  estimate = unname(t1$estimate), ci_low = t1$conf.int[1], ci_high = t1$conf.int[2]
))

w1 <- wilcox.test(normal_sample, mu = mu0, conf.int = TRUE)
write_fixture("wilcoxon_one_sample__normal_sample", list(
  r_function = "stats::wilcox.test", data = "normal_sample.csv", mu0 = mu0,
  statistic = unname(w1$statistic), p_value = w1$p.value
))

# Sign test = binomial test on the sign of (x - mu0), ties excluded.
signs <- normal_sample - mu0
n_pos <- sum(signs > 0)
n_nonzero <- sum(signs != 0)
sign_result <- binom.test(n_pos, n_nonzero, p = 0.5)
write_fixture("sign_test__normal_sample", list(
  r_function = "stats::binom.test on sign(x - mu0)", data = "normal_sample.csv", mu0 = mu0,
  n_positive = n_pos, n_nonzero = n_nonzero,
  statistic = unname(sign_result$statistic), p_value = sign_result$p.value,
  estimate = unname(sign_result$estimate)
))

n_binom <- 60
binary_sample <- rbinom(n_binom, 1, 0.35)
write_data("binary_sample", data.frame(x = binary_sample))

p0 <- 0.5
bt <- binom.test(sum(binary_sample), n_binom, p = p0)
write_fixture("binomial_test__binary_sample", list(
  r_function = "stats::binom.test", data = "binary_sample.csv", p0 = p0,
  successes = sum(binary_sample), n = n_binom,
  statistic = unname(bt$statistic), p_value = bt$p.value, estimate = unname(bt$estimate),
  ci_low = bt$conf.int[1], ci_high = bt$conf.int[2]
))

category_counts <- c(A = 18, B = 25, C = 30, D = 12)
write_data(
  "category_counts",
  data.frame(category = names(category_counts), count = as.integer(category_counts))
)
expected_probs <- c(A = 0.2, B = 0.3, C = 0.3, D = 0.2)
cgof <- chisq.test(category_counts, p = expected_probs)
write_fixture("chi2_goodness_of_fit__category_counts", list(
  r_function = "stats::chisq.test", data = "category_counts.csv",
  expected_probs = unname(expected_probs),
  statistic = unname(cgof$statistic), df = unname(cgof$parameter), p_value = cgof$p.value
))

## Two independent groups ---------------------------------------------------

st_eq <- t.test(value ~ group, data = group_equal_var, var.equal = TRUE)
write_fixture("student_t__two_groups_equal_var", list(
  r_function = "stats::t.test(var.equal=TRUE)", data = "two_groups_equal_var.csv",
  statistic = unname(st_eq$statistic), df = unname(st_eq$parameter), p_value = st_eq$p.value,
  estimate_diff = unname(st_eq$estimate[1] - st_eq$estimate[2]),
  ci_low = st_eq$conf.int[1], ci_high = st_eq$conf.int[2]
))

wt_uneq <- t.test(value ~ group, data = group_unequal_var, var.equal = FALSE)
write_fixture("welch_t__two_groups_unequal_var", list(
  r_function = "stats::t.test(var.equal=FALSE)", data = "two_groups_unequal_var.csv",
  statistic = unname(wt_uneq$statistic), df = unname(wt_uneq$parameter), p_value = wt_uneq$p.value,
  estimate_diff = unname(wt_uneq$estimate[1] - wt_uneq$estimate[2]),
  ci_low = wt_uneq$conf.int[1], ci_high = wt_uneq$conf.int[2]
))

yuen_result <- WRS2::yuen(value ~ group, data = group_unequal_var, tr = 0.2)
write_fixture("yuen_trimmed_t__two_groups_unequal_var", list(
  r_function = "WRS2::yuen(tr=0.2)", data = "two_groups_unequal_var.csv",
  statistic = unname(yuen_result$test), df = unname(yuen_result$df),
  p_value = yuen_result$p.value, estimate_diff = unname(yuen_result$diff),
  ci_low = yuen_result$conf.int[1], ci_high = yuen_result$conf.int[2]
))

mw <- wilcox.test(value ~ group, data = group_equal_var, conf.int = TRUE)
write_fixture("mann_whitney__two_groups_equal_var", list(
  r_function = "stats::wilcox.test", data = "two_groups_equal_var.csv",
  statistic = unname(mw$statistic), p_value = mw$p.value
))

bm <- brunnermunzel::brunnermunzel.test(value ~ group, data = group_equal_var)
write_fixture("brunner_munzel__two_groups_equal_var", list(
  r_function = "brunnermunzel::brunnermunzel.test", data = "two_groups_equal_var.csv",
  statistic = unname(bm$statistic), df = unname(bm$parameter), p_value = bm$p.value,
  estimate = unname(bm$estimate)
))

ks2 <- suppressWarnings(ks.test(
  group_equal_var$value[group_equal_var$group == "A"],
  group_equal_var$value[group_equal_var$group == "B"]
))
write_fixture("ks_two_sample__two_groups_equal_var", list(
  r_function = "stats::ks.test", data = "two_groups_equal_var.csv",
  statistic = unname(ks2$statistic), p_value = ks2$p.value
))

# Permutation test: exact (full enumeration), not Monte Carlo. A
# permutation p-value is exact by construction once every distinct
# assignment is enumerated -- no package needed, no R/Python RNG mismatch
# to work around, and no "loose tolerance" comparison required. Full
# enumeration is only tractable for small n, hence a dedicated small
# dataset here rather than reusing two_groups_equal_var (n=40 per group
# would mean C(80,40) ~ 10^23 assignments).
n_perm_small <- 6
perm_small_a <- round(rnorm(n_perm_small, mean = 20, sd = 5), 2)
perm_small_b <- round(rnorm(n_perm_small, mean = 25, sd = 5), 2)
write_data(
  "permutation_two_sample_small",
  data.frame(value = c(perm_small_a, perm_small_b), group = rep(c("A", "B"), each = n_perm_small))
)

combined_small <- c(perm_small_a, perm_small_b)
n_total_small <- length(combined_small)
observed_diff <- mean(perm_small_a) - mean(perm_small_b)

# All C(12, 6) = 924 ways to choose which 6 of the 12 values form group A;
# the complement is group B. Each combination is one point in the exact
# permutation distribution of the mean-difference statistic.
all_combos <- combn(n_total_small, n_perm_small)
perm_diffs <- apply(all_combos, 2, function(idx) {
  mean(combined_small[idx]) - mean(combined_small[-idx])
})
perm_p <- mean(abs(perm_diffs) >= abs(observed_diff) - 1e-10)
write_fixture("permutation_test_2s__small_exact", list(
  r_function = "manual exact enumeration via combn (no package: C(12,6)=924 assignments)",
  data = "permutation_two_sample_small.csv",
  observed_diff = observed_diff, p_value = perm_p, n_permutations = ncol(all_combos)
))

## Two paired groups ---------------------------------------------------------

pt <- t.test(paired_after, paired_before, paired = TRUE)
write_fixture("paired_t__paired_before_after", list(
  r_function = "stats::t.test(paired=TRUE)", data = "paired_before_after.csv",
  statistic = unname(pt$statistic), df = unname(pt$parameter), p_value = pt$p.value,
  estimate = unname(pt$estimate), ci_low = pt$conf.int[1], ci_high = pt$conf.int[2]
))

wsr <- wilcox.test(paired_after, paired_before, paired = TRUE, conf.int = TRUE)
write_fixture("wilcoxon_signed_rank__paired_before_after", list(
  r_function = "stats::wilcox.test(paired=TRUE)", data = "paired_before_after.csv",
  statistic = unname(wsr$statistic), p_value = wsr$p.value
))

diffs <- paired_after - paired_before
n_pos_p <- sum(diffs > 0)
n_nonzero_p <- sum(diffs != 0)
sign_paired <- binom.test(n_pos_p, n_nonzero_p, p = 0.5)
write_fixture("sign_test_paired__paired_before_after", list(
  r_function = "stats::binom.test on sign(after - before)", data = "paired_before_after.csv",
  n_positive = n_pos_p, n_nonzero = n_nonzero_p,
  statistic = unname(sign_paired$statistic), p_value = sign_paired$p.value,
  estimate = unname(sign_paired$estimate)
))

# Paired permutation test: exact (full enumeration) sign-flips, not Monte
# Carlo. 2^6 = 64 sign patterns is small enough to enumerate completely;
# again a small dedicated dataset (2^35 patterns for the full 35-pair
# paired_before_after would not be tractable to enumerate).
n_pair_small <- 6
perm_small_before <- round(rnorm(n_pair_small, mean = 70, sd = 10), 2)
perm_small_after <- perm_small_before + round(rnorm(n_pair_small, mean = 4, sd = 5), 2)
write_data(
  "permutation_paired_small",
  data.frame(before = perm_small_before, after = perm_small_after)
)

diffs_small <- perm_small_after - perm_small_before
observed_diff_paired <- mean(diffs_small)

# All 2^6 = 64 ways to flip the sign of each paired difference.
sign_patterns <- as.matrix(expand.grid(rep(list(c(-1, 1)), n_pair_small)))
perm_diffs_paired <- apply(sign_patterns, 1, function(signs) mean(diffs_small * signs))
perm_p_paired <- mean(abs(perm_diffs_paired) >= abs(observed_diff_paired) - 1e-10)
write_fixture("permutation_test_paired__small_exact", list(
  r_function = "manual exact enumeration via expand.grid sign flips (no package: 2^6=64 patterns)",
  data = "permutation_paired_small.csv",
  observed_diff = observed_diff_paired, p_value = perm_p_paired,
  n_permutations = nrow(sign_patterns)
))

# ---------------------------------------------------------------------------
# 6.7 Hypothesis tests: k independent groups, k related groups, and their
# post-hoc tests (M3 part 2a). Reuses three_groups (k independent) and
# repeated_measures_long/wide (k related) from the M2 section above.
# ---------------------------------------------------------------------------

## k independent groups ------------------------------------------------------

fit_3g <- aov(value ~ group, data = k_group_df)
anova_3g <- summary(fit_3g)[[1]]
write_fixture("one_way_anova__three_groups", list(
  r_function = "stats::aov", data = "three_groups.csv",
  statistic = unname(anova_3g["group", "F value"]),
  df1 = unname(anova_3g["group", "Df"]), df2 = unname(anova_3g["Residuals", "Df"]),
  p_value = unname(anova_3g["group", "Pr(>F)"])
))

tukey_3g <- TukeyHSD(fit_3g)$group
write_fixture("tukey_hsd__three_groups", list(
  r_function = "stats::TukeyHSD", data = "three_groups.csv",
  comparisons = rownames(tukey_3g),
  diff = unname(tukey_3g[, "diff"]), lwr = unname(tukey_3g[, "lwr"]),
  upr = unname(tukey_3g[, "upr"]), p_adj = unname(tukey_3g[, "p adj"])
))

kw_3g <- kruskal.test(value ~ group, data = k_group_df)
write_fixture("kruskal_wallis__three_groups", list(
  r_function = "stats::kruskal.test", data = "three_groups.csv",
  statistic = unname(kw_3g$statistic), df = unname(kw_3g$parameter), p_value = kw_3g$p.value
))

dunn_3g <- FSA::dunnTest(value ~ group, data = k_group_df, method = "holm")$res
write_fixture("dunn_test__three_groups", list(
  r_function = "FSA::dunnTest(method='holm')", data = "three_groups.csv",
  comparisons = dunn_3g$Comparison,
  statistic = dunn_3g$Z, p_unadj = dunn_3g$P.unadj, p_adj = dunn_3g$P.adj
))

# Small dataset (3 groups x 3 = 9 obs) for exact-enumeration permutation
# ANOVA: C(9,3)*C(6,3) = 1680 distinct group assignments, small enough to
# enumerate completely rather than rely on Monte Carlo (same reasoning as
# the M3 part 1 two-sample/paired permutation fixtures).
n_perm_g <- 3
perm_anova_small <- data.frame(
  value = round(c(rnorm(n_perm_g, 10, 2), rnorm(n_perm_g, 14, 2), rnorm(n_perm_g, 18, 2)), 2),
  group = rep(c("A", "B", "C"), each = n_perm_g)
)
write_data("permutation_anova_small", perm_anova_small)

f_stat_from_assignment <- function(values, groups) {
  fit <- aov(values ~ groups)
  summary(fit)[[1]][["F value"]][1]
}
observed_f <- f_stat_from_assignment(perm_anova_small$value, perm_anova_small$group)
idx_all <- 1:9
n_extreme <- 0
n_total <- 0
for (grpA in combn(idx_all, n_perm_g, simplify = FALSE)) {
  remaining <- setdiff(idx_all, grpA)
  for (grpB in combn(remaining, n_perm_g, simplify = FALSE)) {
    grpC <- setdiff(remaining, grpB)
    assignment <- character(9)
    assignment[grpA] <- "A"
    assignment[grpB] <- "B"
    assignment[grpC] <- "C"
    f_perm <- f_stat_from_assignment(perm_anova_small$value, assignment)
    n_total <- n_total + 1
    if (f_perm >= observed_f - 1e-10) n_extreme <- n_extreme + 1
  }
}
write_fixture("permutation_anova__small_exact", list(
  r_function = "manual exact enumeration via combn (no package: C(9,3)*C(6,3)=1680 assignments)",
  data = "permutation_anova_small.csv",
  observed_f = observed_f, p_value = n_extreme / n_total, n_permutations = n_total
))

# Heteroscedastic dataset for welch_anova / alexander_govern / games_howell.
n_uneq_g <- 15
k_group_unequal_var <- data.frame(
  value = c(rnorm(n_uneq_g, 10, 3), rnorm(n_uneq_g, 12, 8), rnorm(n_uneq_g, 15, 15)),
  group = factor(rep(c("A", "B", "C"), each = n_uneq_g))
)
write_data("k_groups_unequal_var", k_group_unequal_var)

wa <- oneway.test(value ~ group, data = k_group_unequal_var, var.equal = FALSE)
wa_omega2 <- suppressWarnings(effectsize::omega_squared(wa))
write_fixture("welch_anova__k_groups_unequal_var", list(
  r_function = "stats::oneway.test(var.equal=FALSE)", data = "k_groups_unequal_var.csv",
  statistic = unname(wa$statistic), df1 = unname(wa$parameter[1]), df2 = unname(wa$parameter[2]),
  p_value = wa$p.value, omega_squared = wa_omega2[[1]][1]
))

ag <- onewaytests::ag.test(value ~ group, data = k_group_unequal_var, verbose = FALSE)
write_fixture("alexander_govern__k_groups_unequal_var", list(
  r_function = "onewaytests::ag.test", data = "k_groups_unequal_var.csv",
  statistic = unname(ag$statistic), df = unname(ag$parameter), p_value = ag$p.value
))

gh <- PMCMRplus::gamesHowellTest(value ~ group, data = k_group_unequal_var)
gh_pairs <- expand.grid(row = rownames(gh$p.value), col = colnames(gh$p.value))
gh_pairs$statistic <- as.vector(gh$statistic)
gh_pairs$p_value <- as.vector(gh$p.value)
gh_pairs <- gh_pairs[!is.na(gh_pairs$p_value), ]
write_fixture("games_howell__k_groups_unequal_var", list(
  r_function = "PMCMRplus::gamesHowellTest", data = "k_groups_unequal_var.csv",
  comparisons = paste(gh_pairs$row, "-", gh_pairs$col),
  statistic = gh_pairs$statistic, p_value = gh_pairs$p_value
))

## k related groups -----------------------------------------------------------

rm_k_mat <- as.matrix(rm_wide[, c("t1", "t2", "t3", "t4")])
rownames(rm_k_mat) <- as.character(rm_wide$subject)

rm_fit <- afex::aov_ez(
  id = "subject", dv = "value", data = rm_long, within = "condition",
  anova_table = list(es = "pes", correction = "GG")
)
rm_gg <- rm_fit$anova_table
rm_uncorrected <- afex::aov_ez(
  id = "subject", dv = "value", data = rm_long, within = "condition",
  anova_table = list(es = "pes", correction = "none")
)$anova_table
rm_hf <- suppressWarnings(afex::aov_ez(
  id = "subject", dv = "value", data = rm_long, within = "condition",
  anova_table = list(es = "pes", correction = "HF")
)$anova_table)
write_fixture("repeated_measures_anova__repeated_measures", list(
  r_function = "afex::aov_ez", data = "repeated_measures_long.csv",
  statistic = unname(rm_gg[1, "F"]), pes = unname(rm_gg[1, "pes"]),
  df1_uncorrected = unname(rm_uncorrected[1, "num Df"]), df2_uncorrected = unname(rm_uncorrected[1, "den Df"]),
  p_value_uncorrected = unname(rm_uncorrected[1, "Pr(>F)"]),
  df1_gg = unname(rm_gg[1, "num Df"]), df2_gg = unname(rm_gg[1, "den Df"]),
  p_value_gg = unname(rm_gg[1, "Pr(>F)"]),
  df1_hf = unname(rm_hf[1, "num Df"]), df2_hf = unname(rm_hf[1, "den Df"]),
  p_value_hf = unname(rm_hf[1, "Pr(>F)"])
))

pw_pt <- pairwise.t.test(
  rm_long$value, rm_long$condition,
  paired = TRUE, p.adjust.method = "holm"
)$p.value
pw_pairs <- expand.grid(row = rownames(pw_pt), col = colnames(pw_pt))
pw_pairs$p_adj <- as.vector(pw_pt)
pw_pairs <- pw_pairs[!is.na(pw_pairs$p_adj), ]
write_fixture("repeated_measures_posthoc__repeated_measures", list(
  r_function = "stats::pairwise.t.test(paired=TRUE, p.adjust.method='holm')",
  data = "repeated_measures_long.csv",
  comparisons = paste(pw_pairs$row, "-", pw_pairs$col), p_adj = pw_pairs$p_adj
))

fr <- friedman.test(rm_k_mat)
write_fixture("friedman__repeated_measures", list(
  r_function = "stats::friedman.test", data = "repeated_measures_wide.csv",
  statistic = unname(fr$statistic), df = unname(fr$parameter), p_value = fr$p.value
))

nem <- PMCMRplus::frdAllPairsNemenyiTest(rm_k_mat)
nem_pairs <- expand.grid(row = rownames(nem$p.value), col = colnames(nem$p.value))
nem_pairs$statistic <- as.vector(nem$statistic)
nem_pairs$p_value <- as.vector(nem$p.value)
nem_pairs <- nem_pairs[!is.na(nem_pairs$p_value), ]
write_fixture("nemenyi_friedman__repeated_measures", list(
  r_function = "PMCMRplus::frdAllPairsNemenyiTest", data = "repeated_measures_wide.csv",
  comparisons = paste(nem_pairs$row, "-", nem_pairs$col),
  statistic = nem_pairs$statistic, p_value = nem_pairs$p_value
))

con <- PMCMRplus::frdAllPairsConoverTest(rm_k_mat)
con_pairs <- expand.grid(row = rownames(con$p.value), col = colnames(con$p.value))
con_pairs$statistic <- as.vector(con$statistic)
con_pairs$p_value <- as.vector(con$p.value)
con_pairs <- con_pairs[!is.na(con_pairs$p_value), ]
write_fixture("conover_friedman__repeated_measures", list(
  r_function = "PMCMRplus::frdAllPairsConoverTest", data = "repeated_measures_wide.csv",
  comparisons = paste(con_pairs$row, "-", con_pairs$col),
  statistic = con_pairs$statistic, p_value = con_pairs$p_value
))

# Binary repeated-measures dataset for cochran_q + pairwise McNemar post-hoc.
n_cq <- 25
set.seed(20260101 + 1)
cochran_binary <- data.frame(
  t1 = rbinom(n_cq, 1, 0.5),
  t2 = rbinom(n_cq, 1, 0.65),
  t3 = rbinom(n_cq, 1, 0.35)
)
write_data("cochran_q_binary", cochran_binary)

cq <- DescTools::CochranQTest(as.matrix(cochran_binary))
write_fixture("cochran_q__cochran_q_binary", list(
  r_function = "DescTools::CochranQTest", data = "cochran_q_binary.csv",
  statistic = unname(cq$statistic), df = unname(cq$parameter), p_value = cq$p.value
))

cq_pairs <- combn(colnames(cochran_binary), 2, simplify = FALSE)
mcn_stat <- sapply(cq_pairs, function(p) {
  mcnemar.test(table(cochran_binary[[p[1]]], cochran_binary[[p[2]]]), correct = TRUE)$statistic
})
mcn_p <- sapply(cq_pairs, function(p) {
  mcnemar.test(table(cochran_binary[[p[1]]], cochran_binary[[p[2]]]), correct = TRUE)$p.value
})
mcn_p_adj <- p.adjust(mcn_p, method = "holm")
write_fixture("cochran_q_posthoc__cochran_q_binary", list(
  r_function = "stats::mcnemar.test(correct=TRUE) pairwise + p.adjust(method='holm')",
  data = "cochran_q_binary.csv",
  comparisons = sapply(cq_pairs, paste, collapse = " - "),
  statistic = unname(mcn_stat), p_unadj = unname(mcn_p), p_adj = unname(mcn_p_adj)
))

cat("Wrote fixtures to", out_dir, "\n")
