# edacopilot — Architecture & Build Specification

> **Status:** v1 specification. Project name: `edacopilot`.
> **Audience:** the maintainer and AI coding agents (e.g. Claude Sonnet / Haiku) building the system from scratch.
> **How to use this document:** read Sections 1–3 fully before writing any code. Build in the milestone order in Section 16. Each milestone has acceptance criteria. Do not skip ahead.

---

## 0. Rules for coding agents (read first)

These rules are non-negotiable. If a task seems to require breaking one, stop and ask the maintainer.

1. **The LLM never computes a statistic.** Every number shown to the user (p-values, effect sizes, means, counts, CIs, diagnostics) comes from a deterministic function in `edacore`. The LLM only interprets intent, fills structured specs, and writes explanations that reference already-computed facts.
2. **No persona can propose an ineligible method.** Eligibility is decided by deterministic code (Section 7). Personas choose among eligible candidates only.
3. **One step at a time.** The system proposes, the user decides. Nothing executes without explicit user acceptance, and the system never chains steps automatically.
4. **No user tracking.** Do not store, score, or report anything about the user's skill, habits, or behaviour. Only the *analysis* is recorded (provenance).
5. **Raw rows never go to the LLM** (by default). Only schema, summary statistics, and diagnostics. See Section 11.
6. **Every function is pure and registered.** No function mutates its input DataFrame. Every public function is registered in the function registry with metadata and a code template (Section 5).
7. **Every accepted step must be reproducible** as plain Python code in the exported notebook.
8. **No GPL dependencies.** The project is Apache-2.0. Do not add `pingouin` (GPL-3) or other copyleft libraries. Implement what is needed on top of `scipy` / `statsmodels`.
9. **The system must run with the LLM switched off** ("deterministic mode"), using template text. This is how the test suite runs.
10. **Every statistical function needs a test** against a reference value (R output or published example). See Section 15.

---

## 1. Product definition

### 1.1 What it is
A Jupyter-based EDA co-pilot for **junior data scientists** who tend to skip assumption and validity checks. It walks them through EDA one step at a time. For each decision it shows proposals from three personas, compares their trade-offs, and waits for the user to choose.

### 1.2 What it is not
- Not an autonomous analyst. It never runs a full analysis on its own.
- Not a modelling / AutoML tool. It stops at EDA and inferential checks.
- Not a performance tracker or tutor that grades the user.
- Not a replacement for domain knowledge. Where statistics can't answer a question (e.g. MNAR), it asks the user.

### 1.3 Design principles
| Principle | Implication |
|---|---|
| Validity first | Assumption diagnostics always run before any test result is shown. |
| Transparent | Every recommendation shows the diagnostics behind it. |
| Teach by consequence | Each persona comment names *what goes wrong* if an assumption is ignored, in one sentence. |
| Show disagreement, hide noise | If all personas agree, show one consensus card. Show three cards only when they differ. |
| Friction on overrides | Choosing a method with a violated assumption requires a typed reason, which is written into the exported code as a comment. |
| Reproducible | Accepted path → runnable notebook. |
| Provider-agnostic | Works with Claude, OpenAI, and local models via Ollama. |
| Small-model friendly | Most decisions are deterministic, so Haiku-class or 7–8B local models are sufficient. |

### 1.4 v1 scope
- **Data types:** tabular cross-sectional, time series (single and panel), and text columns within tabular data.
- **Data size:** designed for up to ~5M rows × 500 columns in memory with pandas. Larger data: sample for diagnostics (explicitly disclosed to user). *(Assumption — see Section 17.)*
- **Interface:** Jupyter (Lab, Notebook 7, VS Code notebooks) via an ipywidgets panel plus IPython magics.

---

## 2. The three personas

Personas are **policy configurations** over the same function library, not independent agents. They differ in how they treat borderline evidence and trade-offs, never in whether validity matters.

| Persona | Stance | Typical behaviour |
|---|---|---|
| **Professor** (statistician) | Strictest valid path | Checks every assumption with a formal test *and* a visual/descriptive check. Treats borderline results as violations. Applies multiple-testing correction by default. Prefers methods whose assumptions are clearly met. Default anchor. |
| **Consultant** | Simplest valid path, sufficient for the decision | Uses robust-by-default choices (e.g. Welch over Student). Relies on large-sample reasoning (CLT) where legitimate. Minimises data loss and steps. Still never picks an ineligible method. |
| **Maverick** (scientist) | Valid alternative that is more robust or informative | Prefers resampling (permutation, bootstrap), robust estimators (trimmed means, MAD), model-based imputation, distance correlation, etc. Restricted to library methods tagged `maverick`. Never proposes unimplemented methods. |

Full policy spec in Section 8.

---

## 3. System architecture

### 3.1 Layers

```
┌──────────────────────────────────────────────────────────────┐
│  UI LAYER (edacopilot.ui)                                    │
│  ipywidgets chat panel · proposal cards · %eda magics        │
└───────────────▲──────────────────────────────┬───────────────┘
                │ render(Card)                 │ UserAction
┌───────────────┴──────────────────────────────▼───────────────┐
│  ORCHESTRATOR (edacopilot.orchestrator)                      │
│  turn loop · stage state machine · intent routing            │
├──────────────────────────────────────────────────────────────┤
│  STAGE MODULES ("sub-agents", edacopilot.stages)             │
│  profile · quality · missingness · outliers · transform ·    │
│  explore · hypothesis · timeseries · text · export           │
├───────────────────────┬──────────────────────────────────────┤
│  ELIGIBILITY ENGINE   │  PERSONA ENGINE                      │
│  (deterministic)      │  (deterministic pick + LLM rationale)│
├───────────────────────┴──────────────────────────────────────┤
│  LLM LAYER (edacopilot.llm)                                  │
│  LiteLLM adapter · prompt templates · structured output ·    │
│  ContextBuilder (privacy filter) · deterministic fallback    │
├──────────────────────────────────────────────────────────────┤
│  SESSION (edacopilot.session)                                │
│  DatasetStore (version DAG) · ProvenanceLog · TestLedger     │
├──────────────────────────────────────────────────────────────┤
│  EDACORE (separate package, NO LLM dependency)               │
│  function registry · profiling · assumptions · tests ·       │
│  effect sizes · missingness · outliers · transforms ·        │
│  timeseries · text · viz · export templates                  │
└──────────────────────────────────────────────────────────────┘
```

`edacore` must be usable on its own as a plain statistics library (`import edacore`). This keeps the deterministic core testable and gives the project value even without the agent layer.

### 3.2 The turn loop (core interaction)

```
User message / button
        │
        ▼
[1] Intent parser (LLM, structured) ──► Intent
        │
        ▼
[2] Route to stage module
        │
        ▼
[3] Build QuestionSpec (LLM fills from user text + schema; validated by pydantic
    and against the actual dataset; ambiguities → ask user)
        │
        ▼
[4] Run diagnostics (edacore, deterministic)
        │
        ▼
[5] Eligibility engine → ranked Candidates (ELIGIBLE / CAVEAT / INELIGIBLE)
        │
        ▼
[6] Persona engine → each persona's pick (deterministic, from policy)
        │
        ├── all picks identical? → Consensus card
        └── picks differ?        → Divergence card (2–3 proposals + comparison)
        │
        ▼
[7] LLM writes rationale / pros / cons grounded in fact IDs → fact-check → render
        │
        ▼
      WAIT for user
        │
        ▼
[8] On accept: execute function → Result / new DatasetVersion → ProvenanceLog
    (+ override reason if needed) → TestLedger (if test) → render result +
    validity notes → suggest (not run) next step → WAIT
```

### 3.3 Why stage modules instead of autonomous sub-agents
EDA stages are sequential and share state: cleaning changes what tests see. Independent agents passing messages lose context and multiply cost. Each stage module is therefore a **scoped skill**: its own prompt templates, its own allowed function subset, and its own eligibility rules, all operating on one shared `Session`. The orchestrator is the only component that talks to the user.

### 3.4 Where the LLM is used (complete list)
| Call | Purpose | Output | Model tier |
|---|---|---|---|
| `parse_intent` | Classify user message | `Intent` | small / local |
| `build_question_spec` | Map user question to a structured spec | `QuestionSpec` | small–medium |
| `write_rationale` | Persona explanations, pros/cons, comparison | `RationaleBundle` | medium (local OK) |
| `elicit_mnar` | Domain questions about why data may be missing | `list[ElicitationQuestion]` | medium |
| `suggest_next_step` | Suggest what to look at next (never executes) | `NextStepSuggestion` | small |
| `answer_free_question` | Explain a concept or result, grounded in session facts | `GroundedAnswer` | medium |
| `select_adhoc_function` | Ad-hoc read-only requests ("plot X by Y") | `FunctionCall` from registry subset | small–medium |

Everything else is deterministic. Every call has a deterministic fallback (Section 10.5).

---

## 4. Tech stack and repository layout

### 4.1 Dependencies
| Purpose | Library | Notes |
|---|---|---|
| Language | Python ≥ 3.11 | |
| DataFrames | `pandas` ≥ 2.2 (pyarrow backend) | Polars support deferred to v2 |
| Stats | `scipy` ≥ 1.13, `statsmodels` ≥ 0.14 | Core of all tests |
| ML utilities | `scikit-learn` | Imputers, isolation forest, LOF, MCD, mutual info |
| Post-hoc tests | `scikit-posthocs` (MIT) | Dunn, Nemenyi |
| Change points | `ruptures` (BSD) | Optional extra `[timeseries]` |
| Language detection | `langdetect` (Apache-2.0) | Optional extra `[text]` |
| Contracts | `pydantic` ≥ 2 | All data contracts and LLM outputs |
| LLM | `litellm` | Provider-agnostic: Anthropic, OpenAI, Ollama, etc. |
| Plots | `matplotlib` (default), `plotly` (optional) | Static by default for reproducible export |
| UI | `ipywidgets` ≥ 8, `IPython` | Works in JupyterLab, Notebook 7, VS Code |
| Export | `nbformat` | Notebook generation |
| Storage | `pyarrow` (parquet) | Dataset versions on disk |
| Dev | `pytest`, `hypothesis`, `ruff`, `mypy`, `pre-commit` | |

### 4.2 Repository layout
```
edacopilot/
├── README.md
├── ARCHITECTURE.md              # this document
├── LICENSE                      # Apache-2.0
├── pyproject.toml               # extras: [timeseries], [text], [plotly], [all], [dev]
├── .github/workflows/ci.yml
├── src/
│   ├── edacore/                 # deterministic library, NO LLM imports
│   │   ├── __init__.py
│   │   ├── registry.py          # @register decorator, FunctionSpec, lookup
│   │   ├── contracts.py         # shared pydantic models (Section 5)
│   │   ├── profiling.py
│   │   ├── quality.py
│   │   ├── missingness/
│   │   │   ├── diagnose.py      # Little's test, pattern analysis
│   │   │   └── impute.py
│   │   ├── outliers.py
│   │   ├── transforms/
│   │   │   ├── numeric.py
│   │   │   ├── categorical.py
│   │   │   ├── datetime.py
│   │   │   └── text.py
│   │   ├── assumptions.py       # all assumption checks
│   │   ├── stattests/           # statistical tests (named to avoid clashing with pytest)
│   │   │   ├── one_sample.py
│   │   │   ├── two_sample.py
│   │   │   ├── k_sample.py
│   │   │   ├── categorical.py
│   │   │   ├── correlation.py
│   │   │   ├── equivalence.py
│   │   │   └── posthoc.py
│   │   ├── effect_sizes.py
│   │   ├── multiplicity.py
│   │   ├── power.py
│   │   ├── timeseries.py
│   │   ├── text.py
│   │   ├── viz.py
│   │   ├── validity_notes.py    # deterministic interpretation guards
│   │   └── codegen.py           # code templates → python source
│   └── edacopilot/              # agent layer
│       ├── __init__.py          # start(), load_ipython_extension()
│       ├── config.py
│       ├── session/
│       │   ├── session.py
│       │   ├── dataset_store.py
│       │   ├── provenance.py
│       │   └── ledger.py
│       ├── eligibility/
│       │   ├── engine.py
│       │   └── rules/           # one module per QuestionSpec goal
│       ├── personas/
│       │   ├── engine.py
│       │   ├── professor.yaml
│       │   ├── consultant.yaml
│       │   └── maverick.yaml
│       ├── stages/              # one module per stage (Section 9.3)
│       ├── orchestrator/
│       │   ├── loop.py
│       │   ├── state_machine.py
│       │   └── intents.py
│       ├── llm/
│       │   ├── client.py        # LiteLLM wrapper, retries, structured output
│       │   ├── context.py       # ContextBuilder / privacy filter
│       │   ├── factcheck.py
│       │   ├── fallback.py      # deterministic templates
│       │   └── prompts/         # one .md template per LLM call
│       ├── ui/
│       │   ├── panel.py
│       │   ├── cards.py
│       │   └── magics.py
│       └── export/
│           ├── notebook.py
│           └── report.py
├── tests/
│   ├── unit/edacore/            # reference-value tests
│   ├── unit/edacopilot/
│   ├── fixtures/r_reference/    # JSON outputs generated from R
│   ├── scenarios/               # synthetic trap datasets
│   └── conversations/           # golden conversation tests (mocked LLM)
├── evals/                       # LLM-layer evals (Section 15.4)
├── scripts/
│   └── generate_r_fixtures.R
└── examples/
    └── quickstart.ipynb
```

---

## 5. Core contracts and function registry

All contracts live in `edacore/contracts.py` (statistical) and `edacopilot/session/*.py` (session). Use pydantic v2 `BaseModel`, frozen where possible.

### 5.1 Column and dataset profile
```python
class SemanticType(str, Enum):
    CONTINUOUS = "continuous"; DISCRETE = "discrete"; BINARY = "binary"
    NOMINAL = "nominal"; ORDINAL = "ordinal"; DATETIME = "datetime"
    TEXT = "text"; IDENTIFIER = "identifier"; CONSTANT = "constant"; MIXED = "mixed"

class ColumnProfile(BaseModel):
    name: str
    dtype: str                      # pandas dtype
    semantic_type: SemanticType
    semantic_confidence: float      # 0–1; < 0.8 → confirm with user
    n: int; n_missing: int; n_unique: int
    sentinel_candidates: list[Any]  # e.g. -999, "NA", ""
    summary: dict[str, float | int | str]   # type-specific stats
    flags: list[str]                # "high_cardinality", "numeric_coded_categorical", ...

class DatasetProfile(BaseModel):
    n_rows: int; n_cols: int
    columns: list[ColumnProfile]
    structure: Literal["cross_sectional", "time_series", "panel", "repeated_measures", "unknown"]
    time_index: str | None
    entity_id: str | None           # panel / repeated measures key
    target: str | None
    duplicate_rows: int
    leakage_candidates: list[str]
    pii_columns: list[str]
```

### 5.2 Diagnostics and assumption checks
```python
class CheckStatus(str, Enum):
    PASS = "pass"; BORDERLINE = "borderline"; FAIL = "fail"
    NOT_APPLICABLE = "n/a"; UNTESTABLE = "untestable"   # e.g. independence → ask user

class AssumptionCheck(BaseModel):
    fact_id: str                    # stable ID, e.g. "normality.shapiro.group=A"
    assumption: str                 # "normality", "equal_variance", "independence", ...
    method: str                     # "shapiro_wilk"
    scope: dict[str, str]           # {"variable": "income", "group": "A"}
    statistic: float | None
    p_value: float | None
    threshold: str                  # human-readable rule applied
    status: CheckStatus
    consequence: str                # one sentence: what goes wrong if ignored
    plot_ref: str | None
```

### 5.3 Candidates and results
```python
class Eligibility(str, Enum):
    ELIGIBLE = "eligible"; CAVEAT = "caveat"; INELIGIBLE = "ineligible"

class Candidate(BaseModel):
    function: str                   # registry name, e.g. "welch_t_test"
    params: dict[str, Any]
    eligibility: Eligibility
    reasons: list[str]              # fact_ids supporting the status
    tags: set[str]                  # "parametric", "nonparametric", "robust", "resampling", ...
    estimand: str                   # what it actually tests, e.g. "difference in means"

class TestResult(BaseModel):
    fact_id: str
    function: str
    estimand: str
    statistic: float; statistic_name: str
    df: float | tuple[float, float] | None
    p_value: float | None
    p_adjusted: float | None        # filled by TestLedger
    estimate: float | None          # e.g. mean difference
    ci: tuple[float, float] | None; ci_level: float
    effect_size: float | None; effect_size_name: str | None
    effect_size_ci: tuple[float, float] | None
    effect_magnitude: Literal["negligible", "small", "medium", "large"] | None
    n: dict[str, int]
    warnings: list[str]
    validity_notes: list[str]       # from validity_notes.py

class TransformRecord(BaseModel):
    function: str; params: dict[str, Any]
    columns_affected: list[str]
    rows_before: int; rows_after: int
    summary_before: dict; summary_after: dict
    warnings: list[str]
```

### 5.4 Function registry
Every public `edacore` function is registered:
```python
@register(
    name="welch_t_test",
    kind="test",                    # profile|check|test|effect|transform|impute|detect|viz|export
    stage="hypothesis",
    tags={"parametric", "two_sample", "independent", "robust_to_unequal_var"},
    assumptions={
        "hard": ["independent_groups", "numeric_outcome", "min_n_per_group>=2"],
        "soft": ["normality_or_large_n"],
    },
    estimand="difference in means",
    code_template="scipy.stats.ttest_ind({a}, {b}, equal_var=False)",
    read_only=True,                 # False for transforms
)
def welch_t_test(df: pd.DataFrame, outcome: str, group: str,
                 groups: tuple[str, str], alternative: str = "two-sided",
                 ci_level: float = 0.95) -> TestResult: ...
```

Registry requirements:
- `registry.get(name)`, `registry.list(stage=..., kind=..., tags=...)`.
- `registry.json_schema(name)` → JSON schema of params (for ad-hoc LLM function selection).
- `registry.to_code(name, params, df_var="df")` → runnable Python string for export. Every function must render to code that uses only public libraries or `edacore` itself.
- **Hard vs soft assumptions:** a failed *hard* assumption → `INELIGIBLE`. A failed or borderline *soft* assumption → `CAVEAT`. Personas decide what to do with caveats.

### 5.5 Universal function conventions
1. First argument is `df: pd.DataFrame`; column names are strings.
2. Never mutate inputs. Transforms return `(new_df, TransformRecord)`.
3. Handle missing values explicitly: every test takes `nan_policy: Literal["omit","raise"]="omit"` and reports dropped n in `warnings`. Untreated missingness is handled **pairwise (per test)**, never listwise across the session; the result card always shows the n actually used.
4. Random functions take `random_state: int` (default from session config) — required for reproducible permutation/bootstrap results.
5. Large-n guard: if a function is O(n²) or slow (distance correlation, LOF, Shapiro at n>5000), it samples with a disclosed `sample_n` and adds a warning.
6. Return rich objects, never bare tuples.

---

## 6. Function catalogue (edacore)

Legend for **Persona** column: **P** Professor, **C** Consultant, **M** Maverick — which personas' method pools include the function. "All" = shared utility. Signatures omit `df`, `nan_policy`, `random_state` where obvious.

### 6.1 Profiling (`profiling.py`) — stage `profile`
| Function | Purpose | Returns |
|---|---|---|
| `profile_dataset(target=None)` | Runs everything below, builds `DatasetProfile` | `DatasetProfile` |
| `infer_semantic_types()` | Semantic type + confidence per column | `dict[str, (SemanticType, float)]` |
| `detect_identifier_columns()` | Unique-per-row, monotonic, or ID-like names | `list[str]` |
| `detect_numeric_coded_categoricals(max_unique=15)` | Integers that are really categories (e.g. 1–5 codes, region codes) | `list[str]` |
| `detect_ordinal_candidates()` | Likert-like scales, ordered labels ("low/med/high") | `list[str]` |
| `detect_mixed_types()` | Columns mixing numbers and strings | `dict[str, sample_values]` |
| `detect_sentinel_values(candidates=(-999,-99,9999,"NA","N/A","null","",".","?"))` | Disguised missing values | `dict[str, list]` |
| `detect_constant_columns(near_zero_var_ratio=0.95)` | Constant / near-zero-variance | `list[str]` |
| `detect_duplicates(subset=None)` | Exact and key-based duplicate rows | `DuplicateReport` |
| `detect_structure()` | Cross-sectional / time series / panel / repeated measures; finds time index and entity key | `StructureReport` |
| `detect_leakage_candidates(target)` | Near-perfect association with target, post-outcome timestamps, target-derived names | `list[LeakageFlag]` |
| `detect_weight_columns()` | Columns that look like survey/sampling weights; v1 only warns that weights are not applied | `list[str]` |
| `detect_pii(columns=None)` | Emails, phone numbers, national-ID-like patterns, names | `dict[str, list[str]]` |
| `summarize_numeric(col)` | n, missing, mean, sd, median, IQR, MAD, min/max, skew, kurtosis, zeros, negatives | `dict` |
| `summarize_categorical(col)` | Levels, frequencies, rare levels, entropy | `dict` |
| `summarize_datetime(col)` | Range, frequency guess, gaps, future dates | `dict` |
| `summarize_text(col)` | Length distribution, empty/whitespace share, token stats | `dict` |

### 6.2 Data quality and cleaning (`quality.py`) — stage `quality`
| Function | Purpose | Persona |
|---|---|---|
| `standardize_column_names(style="snake")` | Clean names | All |
| `coerce_types(mapping)` | Cast with a report of values that failed to parse | All |
| `replace_sentinels(mapping)` | Convert sentinels to NaN | All |
| `normalize_strings(cols, strip=True, case="lower", unicode="NFKC")` | Whitespace, case, unicode | All |
| `suggest_label_merges(col, threshold=0.85)` | Fuzzy-matched near-duplicate category labels ("Dehli"/"Delhi") — suggestion only | All |
| `apply_label_mapping(col, mapping)` | Apply user-confirmed merges | All |
| `drop_duplicates(subset=None, keep="first")` | Remove duplicates | All |
| `validate_ranges(rules)` | Rule-based checks, e.g. `age in [0,120]` | All |
| `validate_cross_field(rules)` | e.g. `end_date >= start_date` | All |
| `filter_rows(query)` | Filter with pandas query string; records rows removed | All |
| `drop_columns(cols, reason)` | Drop with recorded reason | All |

### 6.3 Missingness (`missingness/`) — stage `missingness`
**Diagnosis**
| Function | Purpose | Notes |
|---|---|---|
| `missing_summary()` | Per-column and per-row missing counts/shares | |
| `missing_patterns(top_k=20)` | Frequency of each missingness pattern | |
| `nullity_correlation()` | Correlation of missing indicators | Co-missing columns hint at a shared cause |
| `littles_mcar_test(cols)` | Little's (1988) MCAR test | See spec below. Rejecting → evidence against MCAR. Not rejecting ≠ proof of MCAR. |
| `missingness_predictors(col, predictors=None)` | Logistic regression of `is_missing(col)` on observed variables; AUC + significant predictors | Evidence *for MAR vs MCAR* only |
| `compare_observed_vs_missing(col)` | For each other variable, compare distributions between rows where `col` is missing vs present (SMD, KS) | |
| `mnar_elicitation_prompts(col)` | Deterministic question templates for the user (the LLM rephrases them) | MNAR **cannot** be tested from observed data |
| `delta_sensitivity(col, deltas, analysis_fn)` | Pattern-mixture δ-adjustment: shift imputed values by δ and rerun an analysis to see if conclusions change | M, P |

**Little's MCAR test spec.** Estimate μ and Σ with EM on all rows. For each missingness pattern *j* with *m_j* rows and observed variable set *o_j*: `d² = Σ_j m_j (ȳ_j − μ_{o_j})ᵀ Σ_{o_j}⁻¹ (ȳ_j − μ_{o_j})`, `df = Σ_j |o_j| − p`. p-value from χ²(df). Numeric columns only; warn if categorical columns are excluded. Validate against R `naniar::mcar_test`.

**Treatment**
| Function | Purpose | Persona | Valid under |
|---|---|---|---|
| `drop_rows_listwise(cols)` | Complete-case analysis | C | MCAR (unbiased); disclose n lost |
| `drop_columns_by_missing(threshold)` | Drop high-missing columns | C, P | Any, with reason |
| `impute_simple(cols, strategy)` | mean / median / mode / constant | C | MCAR; shrinks variance (warn) |
| `impute_by_group(cols, group, strategy)` | Group-wise median/mode | C, P | MAR on group |
| `impute_knn(cols, k=5)` | KNN imputation | M, P | MAR |
| `impute_iterative(cols, estimator="bayesian_ridge", m=1)` | MICE. v1 produces **one** imputed dataset for downstream analysis; `m>1` is allowed only for `compare_imputations` diagnostics. Rubin's-rules pooling is v2. | P, M | MAR |
| `impute_missforest(cols)` | IterativeImputer with RandomForest | M | MAR, nonlinear |
| `impute_timeseries(col, method)` | ffill / bfill / linear / time / spline / seasonal (STL) | All | Depends on gap pattern |
| `add_missing_indicator(cols)` | Keep an `is_missing_*` flag | P, M | Any; recommended with imputation |
| `compare_imputations(original, imputed_versions)` | Distribution shift (KS), variance ratio, correlation change | P, M | Diagnostic |

### 6.4 Outliers (`outliers.py`) — stage `outliers`
**Detection** (return `OutlierReport`: indices, scores, method, threshold)
| Function | Method | Assumes | Persona |
|---|---|---|---|
| `detect_iqr(col, k=1.5)` | Tukey fences | Roughly symmetric | C |
| `detect_zscore(col, z=3)` | Standard score | Normality | P (only if normality passes) |
| `detect_modified_zscore(col, z=3.5)` | MAD-based (Iglewicz–Hoaglin) | None | P, M |
| `detect_adjusted_boxplot(col)` | Medcouple-adjusted fences | Skewed data | P, M |
| `detect_grubbs(col, alpha)` | Grubbs test (single outlier) | Normality | P |
| `detect_mahalanobis_robust(cols)` | MCD-based robust distance | Elliptical | P, M |
| `detect_isolation_forest(cols, contamination="auto")` | Isolation forest | None | M |
| `detect_lof(cols, n_neighbors=20)` | Local outlier factor | None | M |
| `detect_ts_outliers(col, method="stl_resid"|"rolling_mad")` | Time-series aware | Needs time index | All |
| `outlier_impact(col, outliers, stat="mean")` | Statistic with vs without outliers | — | All |

**Treatment**
| Function | Purpose | Persona |
|---|---|---|
| `flag_outliers(col, report)` | Add indicator column, change nothing else | P |
| `remove_outliers(report, reason)` | Remove rows; reason required | C |
| `winsorize(col, limits=(0.01,0.01))` | Cap at percentiles | C, M |
| `cap_at_fences(col, report)` | Cap at detection thresholds | C |
| `replace_with_nan(col, report)` | Treat as missing → missingness stage | P |

Rule: outliers the user confirms are *data errors* are removed/corrected; genuine extreme values are flagged or handled with robust methods. The system asks which it is.

### 6.5 Transformations (`transforms/`) — stage `transform`
**Numeric** (each returns `TransformRecord` with skewness/kurtosis before/after)
| Function | Notes | Persona |
|---|---|---|
| `log_transform(col, base="e", offset="auto")` | Requires positives; offset disclosed | C |
| `log1p_transform(col)` | For counts with zeros | C |
| `sqrt_transform(col)` | Counts | C |
| `reciprocal_transform(col)` | Rates | M |
| `box_cox(col)` | Positive only; λ via MLE | P |
| `yeo_johnson(col)` | Any sign | P, M |
| `quantile_normal(col)` | Rank → normal scores | M |
| `rank_transform(col)` | Ranks | M |
| `standardize(col)` / `minmax_scale(col)` / `robust_scale(col)` | Scaling | All |
| `bin_numeric(col, method="quantile"|"equal_width"|"custom", bins)` | Discretisation; warns about information loss | C |
| `assess_transformation(col, candidates)` | Compares candidate transforms by skewness, normality statistic, interpretability | All |

**Categorical**
| Function | Notes |
|---|---|
| `group_rare_levels(col, min_freq=0.01)` | Into "Other" |
| `one_hot_encode(col, drop_first=False)` | |
| `ordinal_encode(col, order)` | Order must be confirmed by user |
| `target_encode(col, target, cv=5)` | Out-of-fold only; leakage warning always shown |
| `frequency_encode(col)` | |

**Bivariate / multivariate**
| Function | Notes |
|---|---|
| `create_interaction(a, b)` | Product or categorical cross |
| `create_ratio(a, b)` | Guards against zero denominators |
| `polynomial_features(col, degree)` | |
| `pca_summary(cols, n_components=None)` | Explained variance, loadings (exploratory only) |

**Datetime**
| Function | Notes |
|---|---|
| `extract_datetime_parts(col, parts)` | year, month, dow, hour, is_weekend, quarter |
| `time_since(col, reference)` | Durations |

**Time series** (see also 6.9)
| Function | Notes |
|---|---|
| `difference(col, lag=1)` / `seasonal_difference(col, period)` | For stationarity |
| `detrend(col, method="linear"|"stl")` | |
| `resample(col, freq, agg)` | Regularise frequency |
| `lag_features(col, lags)` / `rolling_features(col, windows, stats)` | Entity-aware for panels |

**Text** (see also 6.10)
| Function | Notes |
|---|---|
| `text_length_features(col)` | chars, words, sentences |
| `normalize_text(col, lower, strip_punct, remove_urls)` | |
| `tfidf_top_terms(col, top_k=30, by=None)` | Deterministic; optionally per group |

### 6.6 Assumption checks (`assumptions.py`) — shared by all stages
Every check returns an `AssumptionCheck` with `status` and a one-sentence `consequence`.

| Function | Assumption | Default rule | Note |
|---|---|---|---|
| `check_normality_shapiro(col, by=None)` | Normality | p < α → FAIL; α ≤ p < 0.10 → BORDERLINE | n ≤ 5000; beyond that sample + warn |
| `check_normality_dagostino(col, by=None)` | Normality | as above | Needs n ≥ 20 |
| `check_normality_anderson(col, by=None)` | Normality | critical value at 5% | |
| `check_normality_lilliefors(col, by=None)` | Normality | as above | statsmodels |
| `check_normality_descriptive(col, by=None)` | Normality | \|skew\| < 1 and \|excess kurtosis\| < 2 → PASS; < 2 / < 7 → BORDERLINE | Always run alongside a formal test |
| `qq_correlation(col, by=None)` | Normality | Filliben-type r; data for QQ plot | Feeds `viz.qq_plot` |
| `check_equal_variance_levene(col, group)` | Homoscedasticity | median-centred (Brown–Forsythe) | Default variance check |
| `check_equal_variance_bartlett(col, group)` | Homoscedasticity | | Only valid if normality passes |
| `check_equal_variance_fligner(col, group)` | Homoscedasticity | | Nonparametric |
| `check_variance_ratio(col, group)` | Homoscedasticity | max/min SD ratio < 2 → PASS | Descriptive complement |
| `check_sample_size(col, group, min_n)` | Adequate n | per-group n vs method minimum | Hard assumption for many tests |
| `check_expected_counts(table)` | χ² validity | all expected ≥ 1 and ≤ 20% cells < 5 | Hard for χ² |
| `check_independence_design()` | Independence | UNTESTABLE → ask user about design | Checks for repeated IDs first |
| `check_paired_structure(a, b, id=None)` | Pairing | detects shared IDs / equal-length matched rows | Wrong design is the most common junior error |
| `check_sphericity_mauchly(df_long, subject, within, dv)` | Sphericity | p < α → FAIL (apply GG/HF correction) | Repeated-measures ANOVA |
| `check_linearity(x, y)` | Linearity | lowess deviation + Harvey–Collier / RESET | For Pearson / regression |
| `check_monotonicity(x, y)` | Monotonicity | Spearman vs lowess shape | For Spearman |
| `check_homoscedasticity_bp(x, y)` | Residual variance | Breusch–Pagan | Regression-based EDA |
| `check_autocorrelation_dw(resid)` / `check_ljung_box(col, lags)` | Independence over time | | Time-ordered data |
| `check_multicollinearity_vif(cols)` | Multicollinearity | VIF > 5 BORDERLINE, > 10 FAIL | |
| `check_measurement_level(col, required)` | Scale | semantic type matches method | Hard (e.g. ordinal → no Pearson) |
| `check_same_shape(col, group)` | Same distribution shape | KS on centred/scaled groups | Needed to read Mann–Whitney as a median test |

### 6.7 Hypothesis tests (`stattests/`) — stage `hypothesis`
All return `TestResult` including effect size with CI (bootstrap if no analytic CI).

**One sample**
| Function | Hard assumptions | Soft | Persona | Effect size |
|---|---|---|---|---|
| `one_sample_t(col, mu0)` | numeric | normality or large n | P, C | Cohen's d |
| `wilcoxon_one_sample(col, mu0)` | numeric/ordinal | symmetry | P, C | rank-biserial |
| `sign_test(col, mu0)` | ordinal+ | — | P, M | proportion |
| `binomial_test(col, p0)` | binary | — | All | Cohen's h |
| `chi2_goodness_of_fit(col, expected)` | categorical, expected counts | — | All | Cohen's w |
| `bootstrap_one_sample(col, stat="mean")` | numeric | — | M | CI only |

**Two independent groups**
| Function | Hard | Soft | Persona | Effect size |
|---|---|---|---|---|
| `student_t(outcome, group)` | numeric, independent, n ≥ 2 | normality, equal variance | P (only if both pass) | Hedges' g |
| `welch_t(outcome, group)` | numeric, independent | normality or large n | P, C | Hedges' g |
| `yuen_trimmed_t(outcome, group, trim=0.2)` | numeric, independent | — | M, P | trimmed mean difference; Wilcox–Tian ξ |
| `mann_whitney(outcome, group)` | ordinal+, independent | same shape (for median interpretation) | P, C | rank-biserial, Cliff's δ |
| `brunner_munzel(outcome, group)` | ordinal+, independent | — | M, P | stochastic superiority |
| `permutation_test_2s(outcome, group, stat="mean_diff", n_perm=10000)` | independent, exchangeability | — | M | observed diff |
| `bootstrap_diff(outcome, group, stat="mean"|"median")` | independent | — | M | CI of diff |
| `ks_two_sample(outcome, group)` | continuous | — | M | D statistic |

**Two paired groups**
| Function | Hard | Soft | Persona | Effect size |
|---|---|---|---|---|
| `paired_t(a, b)` | paired numeric | normality of differences or large n | P, C | d_z |
| `wilcoxon_signed_rank(a, b)` | paired ordinal+ | symmetry of differences | P, C | rank-biserial |
| `sign_test_paired(a, b)` | paired ordinal | — | P | proportion |
| `permutation_test_paired(a, b)` | paired | — | M | mean diff |

**k independent groups** (+ post-hoc)
| Function | Hard | Soft | Persona | Effect size | Post-hoc |
|---|---|---|---|---|---|
| `one_way_anova(outcome, group)` | numeric, independent | normality, equal variance | P | η², ω² | `tukey_hsd` |
| `welch_anova(outcome, group)` | numeric, independent | normality or large n | P, C | ω² | `games_howell` |
| `alexander_govern(outcome, group)` | numeric, independent | normality | M | — | `games_howell` |
| `kruskal_wallis(outcome, group)` | ordinal+, independent | same shape | P, C | ε² | `dunn_test` (with p-adjust) |
| `permutation_anova(outcome, group)` | independent | — | M | η² | pairwise permutation |

**k related groups**
| Function | Hard | Soft | Persona | Effect size | Post-hoc |
|---|---|---|---|---|---|
| `repeated_measures_anova(dv, subject, within)` | balanced, numeric | normality, sphericity (GG/HF if FAIL) | P | partial η² | pairwise paired t + Holm |
| `friedman(dv, subject, within)` | ordinal+ | — | P, C | Kendall's W | `nemenyi` / Conover |
| `cochran_q(dv, subject, within)` | binary | — | All | — | pairwise McNemar |

**Factorial**
| Function | Notes | Persona |
|---|---|---|
| `two_way_anova(outcome, factors, typ=2)` | Interaction test; typ=3 if unbalanced | P |
| `aligned_rank_transform_anova(outcome, factors)` | Nonparametric factorial (ART) | M |

**Categorical association**
| Function | Hard | Persona | Effect size |
|---|---|---|---|
| `chi2_independence(a, b)` | expected counts OK | P, C | Cramér's V (bias-corrected) |
| `fisher_exact(a, b)` | 2×2 (or r×c via simulation) | P, C | odds ratio + CI |
| `g_test(a, b)` | expected counts OK | M | Cramér's V |
| `mcnemar(a, b)` | paired binary | All | odds ratio |
| `two_proportion_z(outcome, group)` | binary, n·p ≥ 10 | C | risk difference, Cohen's h |
| `cochran_armitage_trend(binary, ordinal)` | ordinal exposure | P, M | — |

**Correlation**
| Function | Hard | Soft | Persona |
|---|---|---|---|
| `pearson(x, y)` | both continuous | linearity, bivariate normality (for CI), no influential outliers | P, C |
| `spearman(x, y)` | ordinal+ | monotonicity | P, C |
| `kendall_tau(x, y)` | ordinal+ | many ties → tau-b | P |
| `point_biserial(binary, x)` | binary + continuous | normality within groups | P |
| `partial_correlation(x, y, covars, method)` | | | P, M |
| `distance_correlation(x, y)` | numeric | — (permutation p) | M |
| `mutual_information(x, y)` | any | — | M |
| `correlation_matrix(cols, method, adjust="holm")` | | | All |

**Distribution / equivalence / other**
| Function | Purpose | Persona |
|---|---|---|
| `anderson_ksamp(outcome, group)` | k-sample distribution comparison | M |
| `tost_equivalence(outcome, group, low, high)` | Equivalence ("no meaningful difference") | P, M |
| `runs_test(col)` | Randomness of sequence | P |

### 6.8 Effect sizes, multiplicity, power
| Module | Functions |
|---|---|
| `effect_sizes.py` | `cohens_d`, `hedges_g`, `glass_delta`, `d_z`, `rank_biserial`, `cliffs_delta`, `eta_squared`, `partial_eta_squared`, `omega_squared`, `epsilon_squared`, `kendalls_w`, `cramers_v(bias_correct=True)`, `phi`, `odds_ratio`, `risk_ratio`, `risk_difference`, `cohens_h`, `cohens_w`, `magnitude_label(value, measure)` (documented conventional thresholds, with a note that thresholds are field-dependent), `bootstrap_effect_ci(fn, ...)` |
| `multiplicity.py` | `adjust_pvalues(pvals, method="holm"|"bonferroni"|"fdr_bh"|"fdr_by")` |
| `power.py` | `required_sample_size(test, effect, alpha, power)`, `minimum_detectable_effect(test, n, alpha, power)`. **No post-hoc "observed power"** — it is a function of the p-value and misleading. |

### 6.9 Time series (`timeseries.py`) — stage `timeseries`
| Function | Purpose |
|---|---|
| `infer_frequency(time_col)` | Regular frequency guess and confidence |
| `check_regular_index(time_col, entity=None)` | Gaps, duplicates, irregular spacing |
| `adf_test(col, regression="c"|"ct")` | Unit-root test (H0: non-stationary) |
| `kpss_test(col, regression="c"|"ct")` | Stationarity test (H0: stationary) |
| `stationarity_verdict(col)` | Combines ADF + KPSS using the table below |
| `stl_decompose(col, period)` | Trend / seasonal / residual |
| `seasonality_strength(col, period)` | Hyndman's F_S and F_T |
| `acf_pacf(col, nlags)` | Values + confidence bands |
| `ljung_box(col, lags)` | Residual autocorrelation |
| `detect_change_points(col, method="pelt", penalty)` | `ruptures` (optional extra) |
| `cross_correlation(x, y, max_lag)` | Lead/lag relationships (warns: needs prewhitening) |
| `granger_causality(x, y, max_lag)` | M only; requires stationarity; always shown with "predictive, not causal" note |
| `rolling_summary(col, window)` | Rolling mean / SD for visual stability check |
| `panel_summary(entity, time, col)` | Per-entity coverage, balance |

**ADF + KPSS interpretation (deterministic):**
| ADF rejects | KPSS rejects | Verdict |
|---|---|---|
| Yes | No | Stationary |
| No | Yes | Non-stationary (difference) |
| Yes | Yes | Difference-stationary vs trend: try `regression="ct"`, then differencing |
| No | No | Inconclusive (often too little data) |

Time-series awareness elsewhere: when `structure` is `time_series` or `panel`, the eligibility engine marks independence as FAIL for cross-sectional tests unless the user confirms independence, and outlier/imputation stages offer time-aware methods first.

### 6.10 Text (`text.py`) — stage `text`
Raw text is **never** sent to the LLM. All analysis is deterministic.
| Function | Purpose |
|---|---|
| `text_profile(col)` | Length distribution, empty share, unique share, avg tokens |
| `detect_encoding_issues(col)` | Mojibake, control characters, mixed scripts |
| `detect_language(col, sample_n=1000)` | Language mix (optional extra) |
| `near_duplicate_texts(col, threshold=0.9)` | Normalised exact + shingle Jaccard |
| `top_ngrams(col, n=(1,2), top_k=30, by=None)` | Frequent terms, optionally by group |
| `distinctive_terms(col, group)` | Log-odds with informative prior between groups |
| `text_length_by_group(col, group)` | Feeds normal hypothesis tests |
| `pii_scan(col)` | Reuses `detect_pii` |

### 6.11 Visualisation (`viz.py`)
All return a `matplotlib.figure.Figure` and a `plot_ref` ID stored in the session. Code template must reproduce the plot.

`histogram`, `kde`, `ecdf`, `boxplot`, `violin`, `strip_by_group`, `qq_plot`, `scatter_lowess`, `pair_plot(cols, max_cols=6)`, `correlation_heatmap`, `bar_counts`, `mosaic`, `missing_matrix`, `missing_heatmap`, `missing_by_group`, `outlier_plot`, `before_after(col, record)`, `ts_line(col, time, entity=None)`, `stl_plot`, `acf_plot`, `pacf_plot`, `change_point_plot`, `text_length_hist`, `top_terms_bar`, `effect_size_forest(results)`.

### 6.12 Validity notes (`validity_notes.py`)
Deterministic interpretation guards attached to every `TestResult`. Each is a function `(result, context) -> str | None`:
| Guard | Fires when | Note (template) |
|---|---|---|
| `non_significance` | p ≥ α | "Not significant does not mean no effect. The CI for the difference ranges from {lo} to {hi}; consider an equivalence test if 'no difference' is the claim." |
| `tiny_effect_large_n` | p < α and magnitude = negligible | "Statistically significant but negligible in size; with n={n}, very small differences become significant." |
| `large_effect_small_n` | magnitude ≥ medium and n small | "Large estimated effect with a wide CI; treat as preliminary." |
| `correlation_not_causation` | any correlation/association | "Association only; it does not show that one variable causes the other." |
| `multiple_testing` | ledger count > 1 | "This is test #{k} in this session; the adjusted p-value ({method}) is {p_adj}." |
| `mann_whitney_shape` | MW run and same-shape FAIL | "Groups differ in shape, so this tests whether one group tends to have larger values, not a difference in medians." |
| `post_selection` | the test was chosen after looking at the same data (e.g. the pair of groups with the biggest visual gap) | "This comparison was chosen after exploring the data, so its p-value is optimistic." |
| `imputed_data` | outcome contains imputed values | "{k}% of values are imputed; single imputation understates uncertainty." |
| `outliers_removed` | rows removed in outlier stage affect this variable | "{k} outliers were removed earlier; results may differ with them included." |

---

## 7. Eligibility engine (deterministic)

### 7.1 QuestionSpec
The LLM turns a user question into a `QuestionSpec`. The engine then validates it against the data.
```python
class Goal(str, Enum):
    DESCRIBE = "describe"                     # univariate description
    COMPARE_GROUPS = "compare_groups"         # outcome across groups
    ASSOCIATION = "association"               # two variables
    DISTRIBUTION_FIT = "distribution_fit"     # vs a reference / one-sample
    EQUIVALENCE = "equivalence"
    TREND = "trend"                           # time series
    MISSINGNESS = "missingness"
    OUTLIERS = "outliers"
    TRANSFORM = "transform"

class Design(str, Enum):
    INDEPENDENT = "independent"; PAIRED = "paired"; REPEATED = "repeated"; UNKNOWN = "unknown"

class QuestionSpec(BaseModel):
    goal: Goal
    variables: dict[str, str]                 # {"outcome": "income", "group": "region"}
    design: Design
    alternative: Literal["two-sided", "less", "greater"] = "two-sided"
    alpha: float | None = None                # None → persona/session default
    equivalence_bounds: tuple[float, float] | None = None
    user_text: str                            # original wording, for provenance
    confirmed_by_user: set[str] = set()       # fields the user explicitly confirmed
    ambiguities: list[str] = []               # questions to ask before proceeding
```

**Validation (deterministic, before any test):**
1. All named columns exist; types match the goal (e.g. `COMPARE_GROUPS` needs a grouping column with 2+ levels).
2. `design` is cross-checked with `check_paired_structure` and `detect_structure`. If the LLM said INDEPENDENT but IDs repeat across groups → add ambiguity: "Each customer appears in both groups — are these paired measurements?" **Design is always confirmed by the user** before a test runs; it is the single most consequential choice.
3. If `ambiguities` is non-empty, the orchestrator asks the user and does not proceed.

### 7.2 Engine algorithm
```
def select_candidates(spec, session) -> CandidateSet:
    rules = RULES[spec.goal]                    # eligibility/rules/<goal>.py
    method_family = rules.family(spec, profile)  # e.g. "two_independent_numeric"
    checks = run_required_checks(method_family)  # AssumptionCheck list, cached per data version
    candidates = []
    for fn in rules.methods(method_family):
        hard = evaluate(fn.assumptions.hard, checks)
        soft = evaluate(fn.assumptions.soft, checks)
        status = INELIGIBLE if any(h.status == FAIL for h in hard) else \
                 CAVEAT if any(s.status in (FAIL, BORDERLINE) for s in soft) else ELIGIBLE
        candidates.append(Candidate(fn, status, reasons=[fact_ids]))
    return CandidateSet(spec, checks, rank(candidates))
```
Ranking within a status: fewer caveats first, then broader validity (tag `robust`), then interpretability.

### 7.3 Method families (v1)
| Family | Trigger | Methods considered |
|---|---|---|
| `one_sample_numeric` | DISTRIBUTION_FIT, numeric | one_sample_t, wilcoxon_one_sample, sign_test, bootstrap_one_sample |
| `one_sample_categorical` | DISTRIBUTION_FIT, categorical | binomial_test, chi2_goodness_of_fit |
| `two_independent_numeric` | COMPARE_GROUPS, 2 levels, INDEPENDENT | student_t, welch_t, yuen_trimmed_t, mann_whitney, brunner_munzel, permutation_test_2s, bootstrap_diff |
| `two_paired_numeric` | COMPARE_GROUPS, 2 levels, PAIRED | paired_t, wilcoxon_signed_rank, sign_test_paired, permutation_test_paired |
| `k_independent_numeric` | COMPARE_GROUPS, ≥3 levels, INDEPENDENT | one_way_anova, welch_anova, alexander_govern, kruskal_wallis, permutation_anova |
| `k_repeated_numeric` | COMPARE_GROUPS, ≥3 levels, REPEATED | repeated_measures_anova, friedman |
| `k_repeated_binary` | binary outcome, REPEATED | cochran_q |
| `factorial_numeric` | 2+ grouping factors | two_way_anova, aligned_rank_transform_anova |
| `two_categorical_independent` | ASSOCIATION, both categorical | chi2_independence, fisher_exact, g_test |
| `two_binary_paired` | ASSOCIATION, paired binary | mcnemar |
| `binary_outcome_groups` | COMPARE_GROUPS, binary outcome | two_proportion_z, chi2_independence, fisher_exact |
| `numeric_numeric` | ASSOCIATION, both numeric | pearson, spearman, kendall_tau, distance_correlation, mutual_information |
| `ordinal_any` | ASSOCIATION, ordinal involved | spearman, kendall_tau, cochran_armitage_trend (binary vs ordinal) |
| `binary_numeric` | ASSOCIATION | point_biserial (≡ t-test), mann_whitney |
| `equivalence_two_groups` | EQUIVALENCE | tost_equivalence |
| `ts_stationarity` | TREND | stationarity_verdict, stl_decompose, change points |

### 7.4 Worked example (two independent groups)
Data: `income` by `gender`, n = 38 and 41, strong right skew.

| Check | Result | Status |
|---|---|---|
| normality.shapiro.group=F | W=0.86, p<0.001 | FAIL |
| normality.shapiro.group=M | W=0.88, p<0.001 | FAIL |
| normality.descriptive | skew 1.9 / 2.2 | BORDERLINE |
| variance.levene | p=0.03 | FAIL |
| sample_size | 38 / 41 | PASS |
| same_shape | KS p=0.41 | PASS |

| Candidate | Status | Why |
|---|---|---|
| student_t | CAVEAT | normality FAIL, equal variance FAIL |
| welch_t | CAVEAT | normality FAIL (n moderate) |
| yuen_trimmed_t | ELIGIBLE | robust to skew and unequal variance |
| mann_whitney | ELIGIBLE | same shape PASS → readable as a median shift |
| brunner_munzel | ELIGIBLE | |
| permutation_test_2s | ELIGIBLE | |

Persona picks (Section 8): Professor → `mann_whitney` (clean assumptions, strict reading of normality FAIL). Consultant → `mann_whitney` too (its CLT rule needs n ≥ 30 **and** |skew| < 2 — skew fails). Maverick → `yuen_trimmed_t` (robust, keeps a mean-based interpretation). Result: divergence card with two distinct proposals (Professor + Consultant merged, Maverick separate).

### 7.5 User overrides
- Choosing a `CAVEAT` candidate that the chosen persona didn't pick, or any `INELIGIBLE` candidate, requires a typed reason.
- `INELIGIBLE` overrides additionally show the failed hard assumption and its consequence, and require a second confirmation.
- The reason is stored in `ProvenanceLog` and emitted as a code comment on export.

---

## 8. Persona engine

### 8.1 Policy schema (`personas/*.yaml`)
```yaml
# professor.yaml
id: professor
display_name: "Professor"
alpha: 0.05
normality:
  method: [shapiro, descriptive, qq]     # all must PASS for "normal"
  borderline_is: fail
  clt_shortcut: false
variance:
  method: levene_median
  borderline_is: fail
method_pool_tags: [parametric, nonparametric, exact]
prefer: [assumptions_clearly_met, exact_over_asymptotic]
multiple_testing: { default: holm, apply: always }
missing:
  prefer: [impute_iterative, add_missing_indicator]
  max_listwise_loss_pct: 5
  require_mechanism_discussion: true
outliers:
  prefer: [flag_outliers, replace_with_nan]
  detection: [detect_modified_zscore, detect_adjusted_boxplot]
transforms:
  prefer: [box_cox, yeo_johnson]
explanation_style: "precise, cites each assumption, names consequences"
```
```yaml
# consultant.yaml
id: consultant
display_name: "Consultant"
alpha: 0.05
normality:
  method: [descriptive]
  borderline_is: pass
  clt_shortcut: { min_n_per_group: 30, max_abs_skew: 2 }   # legitimate large-sample reasoning
variance:
  strategy: always_robust        # always use Welch-type methods; no variance test needed
method_pool_tags: [parametric, nonparametric]
prefer: [fewest_steps, interpretability, robust_default]
multiple_testing: { default: holm, apply: when_more_than: 3 }
missing:
  prefer: [drop_rows_listwise, impute_simple]
  max_listwise_loss_pct: 10      # above this, falls back to impute_by_group
outliers:
  prefer: [winsorize, cap_at_fences]
  detection: [detect_iqr]
transforms:
  prefer: [log_transform, log1p_transform, sqrt_transform]
explanation_style: "brief, decision-focused, plain language"
```
```yaml
# maverick.yaml
id: maverick
display_name: "Maverick"
alpha: 0.05
normality:
  method: [shapiro, descriptive]
  borderline_is: fail
method_pool_tags: [resampling, robust, nonparametric_advanced, maverick]
prefer: [robust, resampling, informative_effect_sizes]
multiple_testing: { default: fdr_bh, apply: always }
missing:
  prefer: [impute_missforest, impute_knn, delta_sensitivity]
outliers:
  prefer: [flag_outliers]
  detection: [detect_isolation_forest, detect_lof, detect_mahalanobis_robust]
transforms:
  prefer: [yeo_johnson, quantile_normal, rank_transform]
constraints:
  only_registered_functions: true
  must_state_tradeoff: true      # e.g. "less familiar to reviewers"
explanation_style: "curious, explains why the alternative adds information"
```

### 8.2 Pick algorithm (deterministic)
```
def pick(persona, candidate_set) -> Candidate | None:
    pool = [c for c in candidate_set if c.eligibility != INELIGIBLE
            and c.tags & persona.method_pool_tags]
    reclassify each c using persona.normality / persona.variance rules
        (e.g. consultant's CLT shortcut can turn a normality CAVEAT into ELIGIBLE;
         professor's borderline_is=fail can turn ELIGIBLE into CAVEAT)
    keep only ELIGIBLE after reclassification; if empty, keep least-caveated
    sort by persona.prefer
    return pool[0]
```
Persona reclassification may only *relax soft* assumptions via rules listed in its YAML, and only in documented, defensible ways (CLT shortcut with thresholds). It can never touch hard assumptions.

### 8.3 Divergence detection
Two proposals are "the same" if they share `function` and materially identical `params`. Group personas by proposal:
- 1 group → **Consensus card**: the method, one-line reason, the diagnostics table (collapsed), and "All three personas agree."
- 2–3 groups → **Divergence card**: one column per distinct proposal, labelled with the persona(s), plus a comparison section.

### 8.4 Rationale generation (LLM)
Input: `CandidateSet`, persona picks, persona `explanation_style`, relevant `AssumptionCheck`s with fact IDs.
Output schema:
```python
class PersonaRationale(BaseModel):
    persona: str
    summary: str                     # ≤ 2 sentences
    pros: list[str]                  # ≤ 3
    cons: list[str]                  # ≤ 3
    consequence_if_ignored: str      # 1 sentence
    cited_facts: list[str]           # fact_ids used

class RationaleBundle(BaseModel):
    rationales: list[PersonaRationale]
    comparison: str                  # ≤ 4 sentences: when you'd pick each
```
**Fact-check (deterministic, `llm/factcheck.py`):** every number in the text must match a value in a cited fact (tolerance for rounding); every `cited_facts` ID must exist; method names must match the picks. On failure: one retry with the error message, then fall back to templates.

---

## 9. Orchestrator and stage modules

### 9.1 Intents
```python
class IntentType(str, Enum):
    ASK_ANALYSIS = "ask_analysis"       # "is income different by region?"
    ACCEPT = "accept"                   # accept persona X's proposal
    MODIFY = "modify"                   # "use alpha 0.01", "use median instead"
    OVERRIDE = "override"               # choose a non-recommended candidate
    EXPLAIN = "explain"                 # "why not a t-test?", "what is sphericity?"
    GOTO_STAGE = "goto_stage"
    UNDO = "undo"; BRANCH = "branch"; SWITCH_BRANCH = "switch_branch"
    SHOW = "show"                       # ad-hoc read-only: "plot age vs income"
    SKIP = "skip"; EXPORT = "export"; SETTINGS = "settings"
    ANSWER_CLARIFICATION = "answer_clarification"
    OTHER = "other"

class Intent(BaseModel):
    type: IntentType
    persona: Literal["professor", "consultant", "maverick"] | None = None
    target_stage: str | None = None
    payload: dict[str, Any] = {}
    confidence: float
```
Button clicks produce `Intent`s directly (no LLM). Free text goes through `parse_intent`. If `confidence < 0.6`, the orchestrator asks a clarifying question with 2–3 suggested interpretations as buttons.

### 9.2 Stage state machine
```
INGEST → PROFILE → QUALITY → MISSINGNESS → OUTLIERS → TRANSFORM
       → EXPLORE (univariate & bivariate) → HYPOTHESIS → EXPORT
Conditional: TIMESERIES (if structure ∈ {time_series, panel}),
             TEXT (if any TEXT column)
```
- The machine defines the **suggested order**, not a forced one. The user can jump to any stage at any time (`GOTO_STAGE`). If they jump ahead past an unvisited earlier stage, the orchestrator shows a one-line warning ("Missing values haven't been reviewed; 12% of `income` is missing and tests will drop those rows") and proceeds.
- Each stage has an `entry_summary()` (what it will look at, from the profile) and a `next_suggestions()` list.
- Stage completion is recorded but never enforced.

### 9.3 Stage module interface
```python
class StageModule(Protocol):
    name: str
    allowed_functions: list[str]         # registry names this stage may call
    def entry_summary(self, session) -> Card: ...
    def build_spec(self, intent, session) -> QuestionSpec: ...            # uses LLM
    def diagnose(self, spec, session) -> list[AssumptionCheck]: ...       # deterministic
    def candidates(self, spec, checks, session) -> CandidateSet: ...      # eligibility
    def execute(self, candidate, session) -> StepOutcome: ...             # deterministic
    def next_suggestions(self, session) -> list[Suggestion]: ...
```
Per-stage specifics:
| Stage | Key questions it poses to the user | Notable behaviour |
|---|---|---|
| PROFILE | Confirm low-confidence semantic types; confirm target; confirm ID/time/entity columns | Blocks tests on columns with unconfirmed type < 0.8 confidence |
| QUALITY | Confirm sentinel replacement; confirm label merges; confirm range rules | Never auto-merges labels |
| MISSINGNESS | Why might values be missing? (MNAR elicitation) | Presents MCAR evidence → MAR evidence → user's domain answers, then treatment proposals. States clearly that MNAR is untestable. |
| OUTLIERS | Error or genuine extreme value? | Errors → remove/correct; genuine → flag/robust |
| TRANSFORM | What must stay interpretable? | Shows before/after plots and skewness |
| EXPLORE | Which variables matter? | Univariate summaries and bivariate plots; read-only |
| HYPOTHESIS | Confirm design (independent/paired/repeated) | Full eligibility + persona flow; TestLedger updated |
| TIMESERIES | Confirm frequency, period | Stationarity verdict, decomposition |
| TEXT | Which text columns to profile? | No raw text to LLM |
| EXPORT | Which branch to export? | Notebook + markdown report |

### 9.4 Cards (UI payloads)
```python
class Card(BaseModel):
    kind: Literal["info", "question", "consensus", "divergence", "result",
                  "warning", "override_confirm", "export"]
    title: str
    body_md: str
    diagnostics: list[AssumptionCheck] = []
    proposals: list[ProposalView] = []      # for consensus/divergence
    result: TestResult | TransformRecord | None = None
    plots: list[str] = []                   # plot_refs
    actions: list[ActionButton] = []        # each maps to an Intent
```

### 9.5 Next-step suggestions
After each accepted step, the orchestrator shows 2–4 suggested next actions as buttons (deterministic rules first, optionally reworded by the LLM). Examples: after a significant k-group test → "Run post-hoc comparisons"; after imputation → "Compare distributions before/after imputation". It never executes them.

---

## 10. LLM layer

### 10.1 Client
- `litellm.completion` behind `LLMClient` with: model name from config, timeout, retries (2, exponential backoff), temperature 0 for structured calls, 0.3 for rationale.
- **Structured output strategy:** request JSON matching a pydantic schema (JSON mode / `response_format` where the provider supports it; otherwise instruct + parse). Parse with `Model.model_validate_json`. On failure, retry once with the validation error appended. Then fall back to the deterministic path.
- Do not rely on native tool-calling for the core flow; local models handle it unreliably. `select_adhoc_function` also uses structured JSON: `{"function": name, "params": {...}}`, validated against `registry.json_schema`, restricted to `read_only=True` functions.

### 10.2 Config (`edacopilot.toml` or `start(...)` kwargs)
```toml
[llm]
default_model = "anthropic/claude-haiku-4-5"
rationale_model = "anthropic/claude-sonnet-5"   # optional per-call override
# local example:
# default_model = "ollama/qwen2.5:7b-instruct"
# api_base = "http://localhost:11434"
deterministic_mode = false

[privacy]
level = "standard"          # "strict" | "standard"
alias_column_names = false  # true in strict mode

[analysis]
alpha = 0.05
random_state = 42
max_rows_for_exact = 5000
```
Model names are examples; the maintainer sets current ones.

### 10.3 Prompt templates (`llm/prompts/*.md`)
One file per call. Each template has: role statement, hard rules, the JSON schema, 2–3 few-shot examples, and a slot for the context block. Shared hard rules included in every template:
```
- You do not calculate statistics. Use only numbers present in CONTEXT.
- Refer to facts by their fact_id in cited_facts.
- Never recommend a method not present in CANDIDATES.
- Never mention methods marked INELIGIBLE except to explain why they are excluded.
- Keep each field within its length limit. Plain language, no jargon without a gloss.
- Output JSON only, matching SCHEMA.
```

### 10.4 Context budget
Context is built per call, not as the full session history:
- Dataset profile summary (compact: name, type, n, missing %, 3–4 key stats per column; truncate to the most relevant 40 columns for the current spec).
- Current stage, current spec, candidates, checks relevant to this step.
- Last 6 conversational turns (text only).
- Session summary: accepted steps as one line each.
Target ≤ 6k tokens per call so 8k-context local models work.

### 10.5 Deterministic fallback
`llm/fallback.py` implements every LLM call without a model:
- `parse_intent`: keyword/regex rules + buttons.
- `build_question_spec`: a form-style card (dropdowns for goal, variables, design).
- `write_rationale`: templates filled from `AssumptionCheck.consequence` and persona YAML.
- `elicit_mnar`: static question bank per semantic type.
- `answer_free_question`: glossary lookup (`llm/glossary.yaml`, **one line per term**, ~150 terms covering every assumption, test, effect size and missingness concept in the catalogue) or "not available offline".

This is used when `deterministic_mode = true`, when the LLM is unreachable, and in all unit/conversation tests.

---

## 11. Privacy and the ContextBuilder

`llm/context.py` is the **only** path from session data to the LLM. Nothing else may put data in a prompt.

| Data | `standard` | `strict` |
|---|---|---|
| Column names | sent | replaced by aliases (`col_01`), mapped back locally |
| Semantic types, n, missing % | sent | sent |
| Summary statistics | sent (rounded to 3 s.f.) | sent (rounded) |
| Category labels | top 10 levels sent | not sent (level counts only) |
| Test results / diagnostics | sent | sent |
| Raw rows | **never** | **never** |
| Raw text values | **never** | **never** |
| Columns flagged as PII | name + type only | alias + type only |

Additional rules:
- On `start()`, if PII is detected, show a warning card and default to `strict` until the user confirms.
- Keep a local copy of every outgoing prompt in the session so the user can audit exactly what was sent: `session.llm_audit()`. This log never leaves the machine.
- When using a local model (Ollama), show a banner that data stays on the machine; privacy rules still apply to keep behaviour identical.

---

## 12. Session state

### 12.1 DatasetStore (version DAG)
```python
class DatasetVersion(BaseModel):
    version_id: str                  # "v0", "v1", ...
    parent: str | None
    branch: str                      # "main", or user-named
    created_by_step: str | None
    path: Path                       # parquet under .edacopilot/<session_id>/
    profile: DatasetProfile          # recomputed lazily (cheap diff where possible)
```
- `v0` is the original data, read-only.
- Each accepted transform/cleaning/imputation creates a child version.
- `UNDO` moves the branch head to the parent. `BRANCH` creates a new branch from any version (e.g. compare consultant-cleaned vs professor-cleaned data). `SWITCH_BRANCH` changes the active head.
- Diagnostics are cached per `(version_id, check, scope)`.
- Memory: keep only the active head and `v0` in memory; others load from parquet on demand.

### 12.2 ProvenanceLog
```python
class Step(BaseModel):
    step_id: str
    stage: str
    branch: str
    question: QuestionSpec | None
    checks: list[str]                # fact_ids
    proposals: list[ProposalView]    # what each persona proposed
    chosen: Candidate
    chosen_via: Literal["professor", "consultant", "maverick", "consensus", "override"]
    override_reason: str | None
    input_version: str; output_version: str | None
    result_fact_id: str | None
    code: str                        # from registry.to_code
    timestamp: datetime
```
This records the analysis only. It stores no evaluation of the user.

### 12.3 TestLedger
- Every executed hypothesis test is appended with its family label (e.g. "income comparisons").
- Adjusted p-values are recomputed for the whole session (and per family) after each new test, using the active persona method or the user's session setting.
- The result card always shows raw p, adjusted p, and the test count.
- `session.ledger()` shows all tests with raw/adjusted p-values.

### 12.4 Persistence
- `.edacopilot/<session_id>/session.json` (steps, ledger, config, branch heads) + parquet versions.
- `edacopilot.resume(session_id)` restores after a kernel restart.
- Add `.edacopilot/` to the user's `.gitignore` guidance in the README (it may contain data).

---

## 13. Jupyter UI

### 13.1 Entry points
```python
import edacopilot as eda
session = eda.start(df, target="default_flag", name="loan_eda")   # renders the panel
# or
%load_ext edacopilot
%eda start df --target default_flag
%eda ask "Does loan amount differ between men and women?"
%eda undo | %eda branch <name> | %eda ledger | %eda export
```
A pure-Python API mirrors every action (`session.ask(...)`, `session.accept("professor")`, `session.override(candidate, reason=...)`, `session.export(...)`) so the tool is scriptable and testable without widgets.

### 13.2 Panel layout (ipywidgets)
```
┌ edacopilot · loan_eda · branch: main · v3 · Stage: HYPOTHESIS ─────────┐
│ [Profile][Quality][Missing][Outliers][Transform][Explore][Tests][TS][Text][Export] │
├──────────────────────────────────────────────────────────────────────────┤
│ Chat history (cards)                                                     │
│ ┌ Divergence card ─────────────────────────────────────────────────────┐ │
│ │ Diagnostics ▸ (collapsible table, status colour-coded)               │ │
│ │ ┌ Professor + Consultant ─┐ ┌ Maverick ───────────────┐              │ │
│ │ │ Mann–Whitney U          │ │ Yuen trimmed-mean t     │              │ │
│ │ │ pros / cons / if ignored│ │ pros / cons / if ignored│              │ │
│ │ │ [Accept]                │ │ [Accept]                │              │ │
│ │ └─────────────────────────┘ └─────────────────────────┘              │ │
│ │ Comparison: ...                                                      │ │
│ │ [Other methods ▾] [Explain] [Modify] [Skip]                          │ │
│ └──────────────────────────────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────────────────────────┤
│ [ type a question...                                          ] [Send]  │
└──────────────────────────────────────────────────────────────────────────┘
```
- Stage tabs show visited/unvisited state; clicking one = `GOTO_STAGE`.
- "Other methods" lists all candidates with status badges; choosing one triggers the override flow.
- Plots render inline in cards (matplotlib → PNG in an `Output` widget).
- Colour-coding: PASS green, BORDERLINE amber, FAIL red, UNTESTABLE grey. Also use icons/text so it's not colour-only.
- Everything must work in VS Code notebooks (test there; avoid JupyterLab-only APIs).

### 13.3 Future (v2)
A JupyterLab sidebar extension (TypeScript) that talks to the same Python session, and "insert code cell" into the current notebook. Out of scope for v1 because it needs a frontend build.

---

## 14. Export

- `session.export(path="eda_<name>.ipynb", branch="main")`: builds a notebook with nbformat:
  1. Header: dataset, date, edacopilot version, random_state, config.
  2. One markdown + code cell pair per accepted step: markdown states the question, diagnostics summary, chosen method and why; code is from `registry.to_code`. Override reasons appear as a comment: `# OVERRIDE: normality failed (Shapiro p<0.001); reason: "n=180 per group, CLT applies"`.
  3. Final cell: test ledger with adjusted p-values.
- The exported notebook must run top-to-bottom on the original data and reproduce every number (acceptance test in CI).
- `session.export_report(path="eda_<name>.md")`: human-readable summary with tables and embedded plots.
- `session.export_script(path=...)`: `.py` equivalent.

---

## 15. Testing and evaluation

### 15.1 Reference-value tests (edacore)
- `scripts/generate_r_fixtures.R` runs the R equivalent of every test/check/effect size on fixed datasets (built-in R datasets + seeded synthetic data) and writes JSON to `tests/fixtures/r_reference/`.
  - Examples: `t.test` (Welch/Student), `wilcox.test`, `kruskal.test`, `aov` + `TukeyHSD`, `oneway.test`, `friedman.test`, `chisq.test`, `fisher.test`, `mcnemar.test`, `cor.test` (all methods), `shapiro.test`, `car::leveneTest`, `naniar::mcar_test`, `WRS2::yuen`, `brunnermunzel::brunnermunzel.test`, `effectsize::*`, `tseries::adf.test`, `tseries::kpss.test`, `p.adjust`.
- Python tests assert agreement to 1e-6 (or documented tolerance for resampling methods with fixed seeds).
- Property-based tests (`hypothesis`): no input mutation; invariance to row order; correct handling of NaN; sensible errors on degenerate input (constant column, one group empty).

### 15.2 Eligibility tests
A table-driven test file per method family: synthetic data with known properties → expected status for every candidate. E.g. heavy-tailed n=15 per group → `student_t` CAVEAT, `mann_whitney` ELIGIBLE; paired IDs → all independent tests INELIGIBLE once the design is confirmed as paired.

### 15.3 Scenario ("trap") datasets
`tests/scenarios/` contains generators with planted problems junior analysts commonly miss. Each scenario has an expected detection:
| Scenario | Planted problem | System must |
|---|---|---|
| `paired_as_independent` | Same subjects in both groups | Raise the design ambiguity |
| `ordinal_as_numeric` | Likert 1–5 stored as int | Flag as ordinal candidate; block Pearson until confirmed |
| `sentinel_missing` | -999 for missing income | Detect sentinel before summaries |
| `simpsons_paradox` | Association reverses within strata | Suggest stratified look in EXPLORE when a confounder-like categorical exists |
| `target_leakage` | Column derived from target | Flag in PROFILE |
| `mar_by_group` | Missingness depends on region | `missingness_predictors` finds region |
| `heteroscedastic_groups` | Unequal variances, unequal n | Student t → CAVEAT; Welch preferred |
| `heavy_tails_small_n` | t-distributed, n=12 | Parametric → CAVEAT |
| `huge_n_tiny_effect` | n=1e6, d=0.01 | `tiny_effect_large_n` note fires |
| `non_stationary_ts` | Random walk | Stationarity verdict = non-stationary; independence FAIL for cross-sectional tests |
| `many_comparisons` | 20 null comparisons | Ledger adjusts; ~1 raw "significant", ~0 adjusted |
| `duplicate_rows` | 5% exact duplicates | Detected in PROFILE |
| `pii_present` | Email + phone columns | Strict mode suggested; nothing sent to LLM |

### 15.4 LLM-layer evals (`evals/`)
Run manually or nightly, per configured model (including a local Ollama model):
| Eval | Metric | Target |
|---|---|---|
| Intent parsing | accuracy on ~200 labelled utterances | ≥ 95% (Haiku-class), ≥ 90% (7–8B local) |
| QuestionSpec building | exact match on goal/variables/design for ~150 questions | ≥ 90%; design errors must surface as ambiguities, not silent mistakes |
| Rationale factuality | % passing fact-check on first try | ≥ 95% |
| Rationale grounding | no ineligible methods recommended | 100% |
| Privacy | no raw rows / text in any prompt (automated scan of prompt logs) | 100% |

### 15.5 Conversation tests
Golden transcripts in `tests/conversations/` replay a sequence of user actions against a mocked LLM (recorded structured outputs) and assert the cards, steps, versions and exported notebook. Run in CI.

### 15.6 CI (`.github/workflows/ci.yml`)
- Matrix: Python 3.11, 3.12; Ubuntu + Windows.
- Steps: `ruff check`, `ruff format --check`, `mypy src/`, `pytest -m "not llm"`, export round-trip test (exported notebook executes and reproduces numbers via `nbclient`).
- LLM evals are **not** in CI (cost, non-determinism).

---

## 16. Build plan (milestones)

Build strictly in this order. Each milestone ends with its acceptance criteria passing in CI.

| # | Milestone | Deliverables | Acceptance criteria |
|---|---|---|---|
| M0 | Scaffolding | Repo layout, pyproject with extras, CI, pre-commit, `contracts.py`, `registry.py`, `codegen.py` | CI green; registry registers a dummy function and renders its code |
| M1 | Profiling | 6.1 complete; `DatasetProfile` | Trap scenarios `ordinal_as_numeric`, `sentinel_missing`, `target_leakage`, `duplicate_rows`, `pii_present` detected |
| M2 | Assumptions + effect sizes + multiplicity | 6.6, 6.8 | All match R fixtures |
| M3 | Hypothesis tests | 6.7 all families | All match R fixtures; every test returns effect size + CI |
| M4 | Eligibility engine | Section 7, all families in 7.3 | Table-driven eligibility tests pass; `paired_as_independent` raises ambiguity |
| M5 | Persona engine (deterministic) | Section 8 minus LLM rationale; template rationales | Worked example 7.4 reproduces exactly; divergence detection correct |
| M6 | Session | Section 12 | Undo/branch/switch round-trip; ledger adjusts correctly; resume after restart |
| M7 | Orchestrator in deterministic mode | Section 9 with fallback intents; Python API (`session.ask/accept/override`) | Conversation tests pass with no LLM |
| M8 | Jupyter UI | Section 13 | Manual checklist in JupyterLab, Notebook 7, VS Code; all actions reachable by buttons |
| M9 | LLM layer | Section 10–11; LiteLLM; prompts; fact-check; ContextBuilder | Evals in 15.4 meet targets on one API model and one Ollama model; privacy scan 100% |
| M10 | Missingness | 6.3 incl. Little's test, elicitation flow | Matches `naniar::mcar_test`; `mar_by_group` detected |
| M11 | Outliers + transforms + quality | 6.2, 6.4, 6.5 | Before/after records correct; no input mutation |
| M12 | Time series | 6.9 | `non_stationary_ts` scenario passes; ADF/KPSS match R |
| M13 | Text | 6.10 | Profiling works; prompt logs contain no raw text |
| M14 | Export | Section 14 | Exported notebooks for all conversation tests execute and reproduce all numbers |
| M15 | Docs + example | README, quickstart notebook, CONTRIBUTING | New user can run quickstart end-to-end in < 10 minutes |

**Suggested agent split:** use a stronger model (Sonnet-class) for M3, M4, M5, M9 and M10, where statistical judgment and design matter. A smaller model (Haiku-class) is fine for M0, M1, M6, M8, M11, M13–M15 given this spec, with review.

---

## 17. Assumptions and open questions

### 17.1 Assumptions made in this spec (change if wrong)
1. Python + pandas; Polars deferred.
2. License Apache-2.0; no GPL dependencies.
3. Data fits in memory (≤ ~5M rows); larger data is sampled for diagnostics with disclosure.
4. ipywidgets UI for v1 (not a TypeScript JupyterLab extension).
5. English-only UI and explanations in v1.
6. Default α = 0.05; conventional effect-size thresholds with a caveat that they are field-dependent.
7. Single user, single machine; no server component.
8. Regression modelling is out of scope, except diagnostic checks used inside EDA (linearity, VIF, Breusch–Pagan).

### 17.2 Resolved decisions
| # | Question | Decision |
|---|---|---|
| 1 | Project name | `edacopilot` |
| 2 | Bayesian alternatives for the maverick | Deferred to v2 (avoids a PyMC dependency in v1) |
| 3 | Survey weights and design effects | Out of scope for v1. All functions assume simple random samples; the PROFILE stage shows a note if a column looks like a weight (e.g. named `weight`, `wt`, `pw`) saying weights are not applied |
| 4 | Multiple imputation pooling | v1 runs downstream tests on a single imputed dataset, with the `imputed_data` validity note on every affected result |
| 5 | Missing-data default for tests | Pairwise (per-test `omit`) with the n used disclosed on every result |
| 6 | Language of explanations | English only |
| 7 | Glossary depth | One line per term |

### 17.3 v2 roadmap (not to be built in v1)
- Bayesian estimation for the maverick (BEST, Bayes factors) as an optional `[bayes]` extra.
- Survey weights and complex designs (weighted summaries, design-based tests).
- Multiple imputation with Rubin's-rules pooling across downstream tests.
- JupyterLab sidebar extension and "insert code cell into the current notebook".
- Polars backend for larger data.
- Regression modelling beyond EDA diagnostics.
