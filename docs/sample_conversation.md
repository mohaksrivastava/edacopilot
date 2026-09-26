# A session, as the user sees it

**Generated** by `scripts/generate_transcripts.py` from the live orchestrator.
Do not edit by hand: `tests/conversations/test_conversations.py` fails if this
file and a fresh render disagree.

This is ARCHITECTURE.md Section 7.4's worked example driven end to end through
the Python API, with every card rendered exactly as the system produces it.
Rendering is plain text because M7 has no UI: M8 draws these same `Card`
objects as ipywidgets, and nothing about their content changes.

Everything here runs with **no LLM**. Section 10.5's deterministic fallback
supplies the intent parsing and the rationales, which is also how the whole
test suite runs (rule 9).

Three things to look for, because they are what the design is for:

1. **Nothing ran until it was accepted** (rule 3). The proposal card lists
   eight methods and runs none of them.
2. **The personas disagree, and the card says so** rather than picking for the
   user. Section 8.3's divergence detection only fires when the disagreement
   is real; on clean data the same question produces one consensus card.
3. **The result carries its validity notes** (Section 6.12). A p-value of 0.085
   is not "no difference", and the card says which interval the data is
   actually compatible with.

---

```text
# The Section 7.4 worked example, end to end

`income` by `gender`, n = 38 and 41, strong right skew. Section 7.4 predicts a two-proposal divergence card: Professor and Consultant on Mann-Whitney, the Maverick on Yuen's trimmed-mean t-test, which is what the card below shows.

--------------------------------------------------------------------------

YOU:
    session.goto_stage("profile")

EDACOPILOT:
    [info] Dataset profile
    stage: profile

    79 rows x 2 columns, read as **cross sectional**.

    | Column | Type | Confidence | Missing | Unique |
    |---|---|---|---|---|
    | `income` | continuous | 90% | — | 79 |
    | `gender` | binary | 100% | — | 2 |

    Next:
      [Look at the variables]   (session.goto_stage("explore"))
      [Ask a question of the data]   (session.ask(goal="compare_groups", outcome=..., group=...))

--------------------------------------------------------------------------

YOU:
    session.ask(goal="compare_groups", outcome="income", group="gender",
                design="independent", confirmed_by_user=["design"])

    (The design is stated and confirmed up front here. The next transcript shows what happens when it is not.)

EDACOPILOT:
    [divergence] The personas disagree (2 proposals)
    stage: hypothesis

    Professor + Consultant → Mann-Whitney. Maverick → Yuen's trimmed-mean t-test.

    Family: `two_independent_numeric`. 8 method(s) considered, 6 eligible.

    | Method | Status | Estimates |
    |---|---|---|
    | `yuen_trimmed_t` | eligible | difference in trimmed means (robust to outliers) |
    | `bootstrap_diff` | eligible | difference in a summary statistic, with a bootstrap CI (no p-value) |
    | `permutation_test_2s` | eligible | difference in means (independent groups, permutation-based p-value) |
    | `mann_whitney` | eligible | stochastic dominance between two independent groups |
    | `brunner_munzel` | eligible | relative effect P(X < Y) + 0.5*P(X == Y) between two independent groups |
    | `ks_two_sample` | eligible | whether two independent samples come from the same distribution |
    | `welch_t` | caveat | difference in means (independent groups, unequal variance) |
    | `student_t` | caveat | difference in means (independent groups, equal variance) |

    Nothing has run. Accept a proposal, or override with a method of your own.

    Diagnostics (these decided which methods are valid):
      UNTESTABLE  exchangeability.untestable
                  -> a permutation test assumes the group labels are exchangeable under the null; that follows from how the data was collected, not from its values
      UNTESTABLE  independence.design
                  -> Independence depends on how the data was collected, not just its values; confirm the design with whoever collected it.
      PASS        independent.design=independent.required=independent
                  -> An independent-samples test treats every row as a separate subject; if the same subject appears more than once, it counts one person's repeated readings as independent evidence and reports more certainty than the data supports.
      PASS        measurement_level.income.continuous statistic=0.9
                  -> 'income' is continuous, which a method built for continuous data can use.
      FAIL        normality_or_large_n.group=F.variable=income statistic=38
                  -> This sample is skewed enough, for its size, that the mean's sampling distribution is not yet symmetric -- so a test built on it can be off-centre, not merely imprecise.
      PASS        same_shape.income.gender statistic=0.1142 p=0.9213
                  -> Different distribution shapes mean a Mann-Whitney result reflects general stochastic dominance, not specifically a difference in medians.
      PASS        sample_size.income.gender statistic=38
                  -> Too few observations in a group makes the test's asymptotic assumptions unreliable and its result unstable.
      FAIL        variance.levene.income.gender statistic=7.878 p=0.006337
                  -> Unequal variances bias standard-error estimates in tests that assume equal spread; prefer a Welch-type correction.

    Also computed (evidence, not a verdict on any method):
      FAIL        measurement_level.income.binary statistic=0.9
                  -> 'income' looks like continuous, not binary; a method built for binary data may not mean what it claims here.
      PASS        measurement_level.income.discrete statistic=0.9
                  -> 'income' is continuous, which a method built for discrete data can use.
      FAIL        measurement_level.income.ordinal statistic=0.9
                  -> 'income' looks like continuous, not ordinal; a method built for ordinal data may not mean what it claims here.
      BORDERLINE  normality.descriptive.income.gender=F statistic=1.89
                  -> Skew or heavy tails distort the mean and standard-deviation-based methods; check alongside a formal normality test.
      BORDERLINE  normality.descriptive.income.gender=M statistic=1.704
                  -> Skew or heavy tails distort the mean and standard-deviation-based methods; check alongside a formal normality test.
      FAIL        normality.shapiro.income.gender=F statistic=0.8431 p=8.972e-05
                  -> If normality fails, parametric tests that assume it may mislead; prefer a robust or nonparametric alternative.
      FAIL        normality.shapiro.income.gender=M statistic=0.8234 p=1.769e-05
                  -> If normality fails, parametric tests that assume it may mislead; prefer a robust or nonparametric alternative.

    Proposals:
      Professor -> mann_whitney [eligible]
          Mann-Whitney is the method whose assumptions this data actually meets; every assumption it makes is checked and holds here; though independence depends on how the data was collected, not just its values.
      Consultant -> mann_whitney [eligible]
          Use Mann-Whitney.
      Maverick -> yuen_trimmed_t [eligible]
          Yuen's trimmed-mean t-test is worth a look here; it assumes less than the standard choice; though independence depends on how the data was collected, not just its values.

    Plots:
      - p0
      - p1

    Next:
      [Professor + Consultant: mann_whitney]   (session.accept("professor"))
      [Maverick: yuen_trimmed_t]   (session.accept("maverick"))
      [Use something else (e.g. bootstrap_diff)]   (session.override("bootstrap_diff"))
      [Why this method?]   (session.explain("mann_whitney"))

--------------------------------------------------------------------------

YOU:
    session.explain("student_t")

EDACOPILOT:
    [info] Why student_t is caveat
    stage: hypothesis

    `student_t` is **caveat** for this question.

    It estimates: difference in means (independent groups, equal variance).

    Hard assumptions: `numeric_outcome`, `independent`, `min_n_per_group>=2`.
    Soft assumptions: `normality_or_large_n`, `equal_variance`.

    Here, these are what decided it:

    - **normality_or_large_n.group=F.variable=income** — FAIL (cochran_rule, threshold: Cochran's rule (n > 25*skew^2) needs group 'F' to have n > 89.3 for its skew of 1.89; it has 38)
      - This sample is skewed enough, for its size, that the mean's sampling distribution is not yet symmetric -- so a test built on it can be off-centre, not merely imprecise.
    - **variance.levene.income.gender** — FAIL (levene_median, threshold: p < 0.05 -> fail; 0.05 <= p < 0.10 -> borderline)
      - Unequal variances bias standard-error estimates in tests that assume equal spread; prefer a Welch-type correction.
    - **independence.design** — UNTESTABLE (repeated_id_check, threshold: no id_col given, or no repeats found)
      - Independence depends on how the data was collected, not just its values; confirm the design with whoever collected it.

    Diagnostics (these decided which methods are valid):
      FAIL        normality_or_large_n.group=F.variable=income statistic=38
                  -> This sample is skewed enough, for its size, that the mean's sampling distribution is not yet symmetric -- so a test built on it can be off-centre, not merely imprecise.
      FAIL        variance.levene.income.gender statistic=7.878 p=0.006337
                  -> Unequal variances bias standard-error estimates in tests that assume equal spread; prefer a Welch-type correction.
      UNTESTABLE  independence.design
                  -> Independence depends on how the data was collected, not just its values; confirm the design with whoever collected it.

--------------------------------------------------------------------------

YOU:
    session.accept("professor")

EDACOPILOT:
    [result] mann_whitney: stochastic dominance between two independent groups
    stage: hypothesis

    Test #1 in this session, recorded as step `s0` on branch `main`.

    Code:

    ```python
    edacore.stattests.two_sample.mann_whitney(df, group_col='gender', value_col='income', ci=0.95, nan_policy='omit')
    ```

    Result:
      test        mann_whitney (stochastic dominance of 'income' between 'F' and 'M')
      U           603
      p (raw)     0.08509
      p (adj)     0.08509
      effect      rank_biserial = -0.2259  95% CI [-0.4513, 0.02651] (small)
      n used      F=38, M=41

    Validity notes:
      - Not significant does not mean no effect. The CI for the difference ranges from -0.451 to 0.0265; consider an equivalence test if 'no difference' is the claim.

    Next:
      [Test for equivalence instead]   (session.ask(goal="equivalence", ..., equivalence_bounds=(lo, hi)))
      [Explain normality_or_large_n]   (session.explain("normality_or_large_n"))
      [Ask another question]   (session.ask(goal=..., outcome=..., group=...))

--------------------------------------------------------------------------

YOU:
    session.ledger()

EDACOPILOT:
    [info] Test ledger
    stage: hypothesis

    1 test(s) counted toward the session adjustment, using **holm**.

    | # | test | family | p (raw) | p (adj, session) | p (adj, family) | counts |
    |---|---|---|---|---|---|---|
    | t0 | `mann_whitney` | income comparisons | 0.08509 | 0.08509 | 0.08509 | yes |

    A post-hoc procedure is recorded but does not count: its comparisons are already adjusted within their own family, and pooling them here would correct them twice (Section 12.3).

--------------------------------------------------------------------------

SESSION AFTERWARDS

  branch: main   version: v0
  branches: main
  versions: v0
  tests counted toward the session adjustment: 1

  provenance:
    s0  hypothesis  mann_whitney             via professor

  code (rule 7 — every accepted step, runnable):
    edacore.stattests.two_sample.mann_whitney(df, group_col='gender', value_col='income', ci=0.95, nan_policy='omit')
```
