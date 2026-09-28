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
| Stats | `scipy` ≥ 1.15, `statsmodels` ≥ 0.14 | Core of all tests |
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
├── .github/workflows/nightly.yml
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
│       │   ├── intents.py
│       │   └── cards.py        # Card/ActionButton as plain data (Section 9.4);
│       │                       # ui/cards.py renders them
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

Section 6.1 names three more return types (`DuplicateReport`, `StructureReport`,
`list[LeakageFlag]`) that this section didn't originally spell out. Added during M1:
```python
class DuplicateReport(BaseModel):
    n_exact: int
    exact_duplicate_indices: list[int]
    key_columns: list[str] | None    # None if detect_duplicates was called with subset=None
    n_key_duplicates: int; key_duplicate_indices: list[int]

class StructureReport(BaseModel):
    structure: Literal["cross_sectional", "time_series", "panel", "repeated_measures", "unknown"]
    time_index: str | None; entity_id: str | None
    confidence: float
    reasons: list[str]              # human-readable evidence for the verdict

class LeakageFlag(BaseModel):
    column: str
    reason: str                     # e.g. "near-perfect correlation with target (r=0.997)"
    method: str                     # "name_match" | "pearson_correlation" | "perfect_predictability"
    score: float | None             # e.g. the correlation, when the method produces one
```

Section 6.7's k-group post-hoc column (`tukey_hsd`, `games_howell`, `dunn_test`, ...)
names procedures that each produce *multiple* pairwise comparisons, which
`TestResult` (one result per call) doesn't fit. Added during M3 part 2a:
```python
class PairwiseComparison(BaseModel):
    group_a: str; group_b: str
    statistic: float | None; p_value: float; p_adjusted: float | None
    estimate: float | None; ci: tuple[float, float] | None

class PostHocResult(BaseModel):
    fact_id: str; function: str; method: str
    p_adjust_method: str | None
    comparisons: list[PairwiseComparison]
    n: dict[str, int]; warnings: list[str]
```

### 5.4 Function registry
Every public `edacore` function is registered:
```python
@register(
    name="welch_t_test",
    kind="test",                    # profile|check|test|effect|posthoc|transform|impute|detect|viz|export
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
| `check_same_shape(col, group)` | Same distribution shape | KS on centred/scaled groups; k>2 → all pairs, Holm-adjusted | Needed to read Mann–Whitney (or Kruskal–Wallis) as a median test |

Added in M4, because Section 7's eligibility engine has to resolve *every*
hard/soft assumption name the Section 6.7 registry uses, and these five were
named without a check behind them:

| Function | Assumption | Default rule | Note |
|---|---|---|---|
| `check_design_crossing(group, id_col)` | Independence / pairing | any id under >1 group level → FAIL (related design); ids repeating inside one level → FAIL (clustered) | The long-format counterpart of `check_paired_structure`. Distinguishing the two failures matters: one means "use a paired method", the other means "these rows are clustered" |
| `check_symmetry(col, by=None)` | Symmetry | \|skew\| < 0.5 → PASS; < 1 → BORDERLINE | What the Wilcoxon tests actually assume; normality is sufficient but far stronger |
| `check_influential_outliers(x, y)` | No influential outliers | max Cook's D > 1 → FAIL; > 0.5 → BORDERLINE | Influence, not outlyingness. General outlier detection is 6.4 (M11) |
| `check_balanced_design(subject, within)` | Balance | every subject × condition cell holds exactly 1 row | Hard for `repeated_measures_anova` |
| `check_proportion_counts(outcome, group, event)` | Normal approximation for proportions | ≥ 10 successes **and** ≥ 10 failures per group | Hard for `two_proportion_z` (Section 6.7 lists n·p ≥ 10 under Hard) |
| `check_expected_counts_gof(col, expected)` | χ² validity | as `check_expected_counts`, from reference proportions | Hard for `chi2_goodness_of_fit`, which has no two-way table |

### 6.7 Hypothesis tests (`stattests/`) — stage `hypothesis`
All return `TestResult` including effect size with CI (bootstrap if no analytic CI).
Nine tests are exempt, each for a recorded reason (no standard effect size, or
no standard CI for the one it has): see `docs/m3_effect_size_audit.md`, whose
exemption list `tests/unit/edacore/test_m3_effect_size_criterion.py` enforces in
CI — that test fails both if a non-exempt test loses its effect size or CI and
if an exempt one gains one.

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
| `welch_t(outcome, group)` | numeric, independent | normality or large n | P, C | Hedges' g(av) (average-variance denominator, Welch df) |
| `yuen_trimmed_t(outcome, group, trim=0.2)` | numeric, independent | — | M, P | trimmed mean difference; Wilcox–Tian ξ |
| `mann_whitney(outcome, group)` | ordinal+, independent | same shape (for median interpretation) | P, C | rank-biserial, Cliff's δ |
| `brunner_munzel(outcome, group)` | ordinal+, independent | — | M, P | stochastic superiority (relative effect + its CI) |
| `permutation_test_2s(outcome, group, stat="mean_diff", n_perm=10000)` | independent, exchangeability | — | M | observed diff (+ seeded BCa bootstrap CI) |
| `bootstrap_diff(outcome, group, stat="mean"|"median")` | independent | — | M | CI of diff |
| `ks_two_sample(outcome, group)` | continuous | — | M | D statistic |

**Two paired groups**
| Function | Hard | Soft | Persona | Effect size |
|---|---|---|---|---|
| `paired_t(a, b)` | paired numeric | normality of differences or large n | P, C | d_z |
| `wilcoxon_signed_rank(a, b)` | paired ordinal+ | symmetry of differences | P, C | rank-biserial |
| `sign_test_paired(a, b)` | paired ordinal | — | P | proportion |
| `permutation_test_paired(a, b)` | paired | — | M | mean diff (+ seeded BCa bootstrap CI over pairs) |

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
| Function | Notes | Persona | Effect size |
|---|---|---|---|
| `two_way_anova(outcome, factors, typ=2)` | Interaction test; typ=3 if unbalanced | P | partial η² per term |
| `aligned_rank_transform_anova(outcome, factors)` | Nonparametric factorial (ART) | M | partial η² per term, on the aligned-rank F |

**Categorical association**
| Function | Hard | Persona | Effect size |
|---|---|---|---|
| `chi2_independence(a, b)` | expected counts OK | P, C | Cramér's V (bias-corrected) |
| `fisher_exact(a, b)` | 2×2, or r×c (exact enumeration; seeded Monte Carlo above 2M tables) | P, C | odds ratio + CI (2×2 only) |
| `g_test(a, b)` | expected counts OK | M | Cramér's V |
| `mcnemar(a, b)` | paired binary | All | odds ratio b01/b10 + exact conditional CI |
| `two_proportion_z(outcome, group)` | binary, n·p ≥ 10 | C | risk difference, Cohen's h |
| `cochran_armitage_trend(binary, ordinal)` | ordinal exposure | P, M | — |

**Correlation**

The coefficient *is* the effect size for every function here, so `effect_size`
mirrors `estimate` and shares its interval. `cor.test` reports no CI for
Spearman's rho or Kendall's tau; those follow `DescTools::SpearmanRho`
(Fisher z, SE = 1/√(n−3)) and `DescTools::KendallTauB` (delta-method ASE from
the joint table, which unlike Fisher z accounts for ties).

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
| `tost_equivalence(outcome, group, low, high, alpha=0.05)` | Equivalence ("no meaningful difference"). Both intervals are at 1 − 2α (TOSTER's convention), with Hedges' g as the effect size | P, M |
| `runs_test(col)` | Randomness of sequence | P |

### 6.8 Effect sizes, multiplicity, power
| Module | Functions |
|---|---|
| `effect_sizes.py` | `cohens_d`, `hedges_g`, `glass_delta`, `d_z`, `rank_biserial`, `cliffs_delta`, `eta_squared`, `partial_eta_squared`, `omega_squared`, `epsilon_squared`, `kendalls_w`, `cramers_v(bias_correct=True)`, `phi`, `odds_ratio`, `risk_ratio`, `risk_difference`, `cohens_h`, `cohens_w`, `magnitude_label(value, measure)` (documented conventional thresholds, with a note that thresholds are field-dependent), `bootstrap_effect_ci(fn, ...)`. Added in M3.4 for the Section 6.7 tests that needed a differently parameterized version: `cohens_d_one_sample`, `hedges_g_av` (average-variance denominator, Welch df), `rank_biserial_one_sample`, `rank_biserial_paired` (signed-rank SE, not Mann-Whitney's), `cohens_h_one_sample` (exact, via the monotone arcsine transform of a Clopper-Pearson interval), `cohens_w_gof`; plus two unregistered conversions from already-computed test output, `partial_eta_squared_from_f` and `omega_squared_from_f`, and `UNLABELLED_MEASURES`, the set of effect sizes that deliberately carry no magnitude label |
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

`histogram`, `kde`, `ecdf`, `boxplot`, `violin`, `strip_by_group`, `qq_plot`, `scatter_lowess`, `pair_plot(cols, max_cols=6)`, `correlation_heatmap`, `bar_counts`, `mosaic`, `missing_matrix`, `missing_heatmap`, `missing_by_group`, `outlier_plot(df, col, flagged=None, whis=1.5)`†, `before_after(before, after, col, record=None)`†, `ts_line(col, time, entity=None)`, `stl_plot`, `acf_plot`, `pacf_plot`, `change_point_plot(df, col, change_points, time=None)`†, `text_length_hist`, `top_terms_bar`, `effect_size_forest(results)`.

† Signature departs from a plain name because of the "Three signatures take
what a later milestone will detect" point below — kept here rather than
just in the prose so the catalogue line matches `viz.py` exactly.

All 25 implemented in M8. Four points settled there:

- **They return a `Figure`; the session mints the `plot_ref`.** The two
  halves of the sentence above sit on opposite sides of the `edacore`
  boundary and have to: `edacore` draws, because that is a pure function
  of the data, and `session.plot(...)` writes the PNG under
  `.edacopilot/<session_id>/plots/` and returns the id, because only the
  session knows where that is. A user calling `edacore.viz` directly gets
  a figure and does what they like with it.
- **No pyplot.** Each builds a bare `matplotlib.figure.Figure`. `pyplot`
  keeps a global registry of every figure it creates and releases none, so
  a library going through it would leak one per call into the host kernel
  and start warning at twenty. A test asserts the registry stays empty
  across all 25.
- **A `plot_ref` is keyed on function, params *and dataset version*.**
  Re-rendering the same diagnostic reuses the PNG; the same plot of a
  transformed frame is a different plot, because it is a claim about a
  different state of the data.
- **Three signatures take what a later milestone will detect**, so the
  plot exists now and the detector plugs in without a signature change:
  `outlier_plot(df, col, flagged=None)` (M11's detectors pass indices;
  defaults to the Tukey fence), `change_point_plot(df, col, change_points,
  time=None)` (M12 passes the points), and `before_after(before, after,
  col, record=None)` — 6.11 writes it as `(col, record)`, but a
  `TransformRecord` carries summaries rather than data, so both frames
  have to be passed.

Rule 7 is enforced for plots the same way it is for statistics, by
`tests/helpers/plot_check.py`: the rendered code is executed and the
*plotted data* compared — the artists' own vertices, bar rectangles,
scatter offsets and image arrays — rather than pixels. An image comparison
would fail on a matplotlib point release or a font substitution without
anything being wrong, and nobody can diagnose a pixel diff.

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

Implemented in `edacore/validity_notes.py` (M6.1). Four guards are decided
from the `TestResult` alone; the other five are about the *session* — how
many tests have run, what the same-shape check said, whether the outcome
was imputed, whether rows were dropped, whether this comparison was chosen
after looking. `edacore` has no session and must not grow one (Section 3.1:
it is usable on its own as a plain statistics library), so the caller
assembles a `ValidityContext` and passes it in; the HYPOTHESIS stage builds
one, and a bare `attach(result)` still yields the four result-only guards.

Notes **append**. Several `edacore` tests already use
`TestResult.validity_notes` for facts with nowhere else to go (Mauchly's
test and the Huynh-Feldt correction on a repeated-measures ANOVA, the run
count on a runs test); `attach` adds after those and skips duplicates, so
it is idempotent.

Two conventions Section 6.12 left open, decided with the maintainer:

- **"n small"** for `large_effect_small_n` means the *smallest group* has
  fewer than 30 observations, not the total. The note's claim is that the
  CI is wide, and CI width is driven by the smallest cell: a 200-vs-12
  comparison is preliminary whatever its total says.
- **`mann_whitney_shape` also covers `kruskal_wallis`**, which declares the
  same `same_distribution_shape` assumption. The misreading it prevents —
  reporting a rank test as a difference in medians — is identical there,
  and `check_same_shape` already handles k > 2 groups with Holm adjustment
  across the pairs.

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
    # Added in M4, alongside equivalence_bounds, which plays the same role
    # for EQUIVALENCE: a DISTRIBUTION_FIT question is asked *against* a
    # reference, and every method in that goal's families needs it
    # (one_sample_t's mu0, binomial_test's p0, chi2_goodness_of_fit's
    # expected proportions). Without them those candidates can never be
    # built, so the spec has nowhere to put the question's own parameters.
    reference_value: float | None = None
    reference_proportions: dict[str, float] | None = None
```

Implemented in `edacopilot/eligibility/spec.py`. `validate_spec(spec, df)`
returns a copy carrying `ambiguities`; `select_candidates` raises
`AmbiguousSpecError` if any remain, so step 3 below is enforced rather than
merely expected of the orchestrator. Problems the user cannot resolve by
answering a question — a column that does not exist, a missing required
role, a grouping column with one level — raise `InvalidSpecError` instead:
those are spec-building bugs, not questions.

**Validation (deterministic, before any test):**
1. All named columns exist; types match the goal (e.g. `COMPARE_GROUPS` needs a grouping column with 2+ levels).
2. `design` is cross-checked with `check_design_crossing` (M4's long-format counterpart of `check_paired_structure`, which answers the same question for wide data) and `detect_structure`. If the LLM said INDEPENDENT but IDs repeat across groups → add ambiguity: "Each customer appears in both groups — are these paired measurements?" **Design is always confirmed by the user** before a test runs; it is the single most consequential choice. The engine never edits `spec.design` itself, in either direction: the mirror-image error (design says paired, but no id appears in both groups) is also a question, and so is the third case M4 added, ids repeating *within* a single group, which means the rows are clustered rather than paired.
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
Ranking within a status: fewer caveats first, then broader validity (tag `robust`), then interpretability. The function name is the final tie-break, so the order is total and stable across runs (rule 7).

Implemented in `edacopilot/eligibility/engine.py`, with the assumption-name →
check translation in `eligibility/checks.py`. Four points that the pseudocode
leaves implicit and M4 had to settle:

- **UNTESTABLE never blocks.** Independence of sampling, exchangeability and
  the meaningfulness of row order are facts about data collection, not
  values (Section 5.2). They are attached as `reasons` to every candidate
  that rests on them — including ELIGIBLE ones — and left for the user.
- **Evidence is not the verdict.** A resolver returns *graded* checks (which
  decide the assumption) separately from *evidence* (shown, never graded).
  `check_design_crossing` FAILing means "these rows are not independent",
  which is fatal to an independent-samples test and is exactly what a paired
  test requires; grading it directly made every paired test ineligible on
  paired data.
- **Only a hard FAIL makes a method INELIGIBLE**, as written — a hard
  BORDERLINE would be silently ignored. Hard assumptions are type, design and
  count checks with no borderline band, and a test asserts that stays true.
- **Missing question parameters are not ineligibility.** A method whose
  params cannot be built (no equivalence bounds, no reference value, no
  covariate) is reported under `CandidateSet.unavailable` with the reason.
  That is a question to ask, not a verdict about the data.

A family may also declare a `prepare` hook that derives a column its
assumptions are about: a factorial design's homoscedasticity assumption is
about its *cells* (factor A × factor B), not about either factor's margin.

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

`docs/eligibility_table.md` renders this table as implemented — every method
with its hard and soft assumptions and the check that decides each one. It is
generated from the live registry by `scripts/generate_eligibility_table.py`,
and a test fails if the committed file and a fresh render disagree, so an
assumption cannot change without the table changing with it.

`ts_stationarity`'s methods are Section 6.9, which lands in M12. Asking a
TREND question raises `UnsupportedQuestionError` naming the milestone rather
than reporting those methods as INELIGIBLE: they are not ineligible, they do
not exist yet. The same applies to DESCRIBE, MISSINGNESS, OUTLIERS and
TRANSFORM.

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

Implemented in `orchestrator/loop.py` (M7). One clause is easy to read past
and decides the design: the friction applies to `CAVEAT` and `INELIGIBLE`,
**and only those**. Choosing an `ELIGIBLE` method no persona happened to
name needs nothing — the engine already ruled it valid, so it is a choice
among valid methods rather than an override of a judgement. Demanding a
reason there would train the user to type anything to get past the box,
which would devalue the reason on the overrides that matter.
`Session.record_step` enforces the same rule, so the guard cannot be
bypassed by calling the session directly.

The second confirmation is a *separate* argument (`confirm=True`), not a
second chance to supply the reason, and it does not carry over to a later
override: two decisions, two acts.

---

## 8. Persona engine

### 8.1 Policy schema (`personas/*.yaml`)
A persona's `normality.method` / `variance.method` list names the checks it
**cites** when it explains itself; it never permits ignoring a FAIL from a
check it does not cite — relaxation happens only through the explicitly
named rules below (`clt_shortcut`, `borderline_is`), and a persona that
could reach eligibility by declining to look at a check would be skipping an
assumption silently, which is what rule 2 exists to prevent.

```yaml
# professor.yaml
id: professor
display_name: "Professor"
alpha: 0.05
normality:
  method: [shapiro, descriptive, qq]     # all must PASS for "normal"
  borderline_is: fail
  clt_shortcut: none                     # no large-sample escape at all
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
  clt_shortcut: cochran      # n > 25*skew^2 per group (Cochran 1977)
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
proposal_policy:                       # speak up only when it adds something
  propose_when: [soft_assumption_not_met, outliers_flagged]
  otherwise_concur_with: professor
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
Persona reclassification may only *relax soft* assumptions via rules listed in its YAML, and only in documented, defensible ways (the CLT shortcut). It can never touch hard assumptions.

Implemented in `edacopilot/personas/` (M5). The policies are loaded and
schema-validated from the YAML files with `extra="forbid"`, so a misspelled
key is a load-time error rather than a persona that silently stops applying
one of its own rules. There are exactly three relaxation/tightening rules,
and nothing else in the module can change a status:

| Rule | Direction | Effect |
|---|---|---|
| `normality.clt_shortcut: cochran` | relax | clears a **normality** caveat when Cochran's rule is met |
| `normality.borderline_is: pass` | relax | clears a caveat *all* of whose causes are BORDERLINE; one FAIL anywhere blocks it |
| `variance.strategy: always_robust` | tighten | declines to propose any method that assumes equal variance |

`borderline_is: pass` being blocked by a single FAIL is the load-bearing
detail: `normality.method` lists the checks a persona *cites*, not ones it
may ignore. A persona that could discard a Shapiro FAIL by leaving
`shapiro` off its list would be skipping an assumption silently, which is
what rule 2 exists to prevent.

Note that `normality_or_large_n` is already decided by Cochran's rule
inside the eligibility engine, because Section 6.7 grants those methods
the escape by name. The persona shortcut therefore only bites on methods
declaring plain `normality` — one-way ANOVA, Alexander–Govern, two-way
ANOVA, repeated-measures ANOVA. The engine records Cochran's verdict on
any family containing a normality-family assumption, so a persona never
depends on whether some *other* candidate happened to surface it.

Ranking: the persona's `prefer` keys are applied in order as sort
criteria, with the eligibility engine's own ranking as the final tie-break
so the order is total and stable (rule 7). An unrecognised `prefer` key is
a load-time error — silently skipping it would leave the persona ranking by
something other than what its file says.

**`proposal_policy` (M5.1).** A persona may decline to propose its own
method and concur with another instead. The Maverick does: it offers an
alternative only when at least one soft assumption is BORDERLINE/FAIL, or
outliers are flagged for the variables involved; otherwise (and whenever
its own pool is empty) it concurs with the Professor. The reasoning is its
own `must_state_tradeoff` constraint — an unfamiliar method is a real cost
("less familiar to reviewers"), and on data that meets every assumption
there is no matching benefit, only a divergence card where the personas do
not actually disagree. Concurring is not silence: the persona still appears
on the card with a one-line note saying it looked and had nothing to add.
The `outliers_flagged` trigger currently reads
`check_influential_outliers`, the only outlier diagnostic that exists;
Section 6.4's general detection (M11) will feed the same trigger.

**The CLT shortcut is Cochran's rule** (`clt_shortcut: cochran`): the
large-sample condition is met, per group, when `n > 25 * skew^2`. It
replaced a pair of flat thresholds (`min_n_per_group: 30`,
`max_abs_skew: 2`) in M4.1, because those answered the wrong question —
whether n is "big" in the abstract rather than big *relative to how skewed
this sample is*. Cochran's rule scales the requirement with the problem: a
near-symmetric sample needs almost no n, a badly skewed one needs a lot.
The same rule decides `normality_or_large_n` in the eligibility engine
(`eligibility/checks.py`), so a persona's shortcut is either on or off, not
a second, different threshold.

It bounds SKEWNESS only, and says nothing about heavy tails: a symmetric
heavy-tailed sample has skew near zero and satisfies it at any n. The
formal normality checks still run and are still reported, so the failure
stays visible; it simply no longer produces a caveat by itself. The
professor keeps `clt_shortcut: none` precisely because of cases like that.

### 8.3 Divergence detection
Two proposals are "the same" if they share `function` and materially identical `params`, **or** if a practical-equivalence rule says they answer the same question under a condition this data meets. Group personas by proposal:
- 1 group → **Consensus card**: the method, one-line reason, the diagnostics table (collapsed), and "All three personas agree."
- 2–3 groups → **Divergence card**: one column per distinct proposal, labelled with the persona(s), plus a comparison section.

**Practical equivalence (M5.1).** Rules live in
`personas/equivalences.yaml`, as config rather than code, for the same
reason the personas do — a rule can be reviewed and added without touching
the pick algorithm. The first is `student_t ≈ welch_t when equal_variance
PASSes`. Two personas naming different methods that agree under a condition
the data meets are **one** proposal: presenting that as a choice asks the
user to decide between two answers that are not different, which teaches
them to read the card as noise. The consensus card then names each
persona's method and why the conclusion is the same.

Two safeguards make this collapse-only. A rule's `when_passes` condition is
mandatory and an *absent* check does not satisfy it — if the condition was
never evaluated there is no evidence the methods agree. And a rule can only
ever turn a divergence into a consensus, never the reverse.

`docs/persona_picks.md` renders every trap scenario against all three
personas, with the template rationale each would show. Like the eligibility
table it is generated (`scripts/generate_persona_picks.py`) with a test
that fails if the committed file goes stale.

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

Implemented in `orchestrator/cards.py` (M7) as plain data — no ipywidgets,
no matplotlib, no HTML. `Card.to_text()` renders the whole card as plain
text, which is what makes Section 15.5's golden transcripts real
transcripts rather than summaries of them; M8's `ui/cards.py` draws the
same objects.

Three additions to the field list above:

- `result` also accepts a `PostHocResult`. Section 9.4 predates that model,
  and a post-hoc's comparison table is exactly a result card's content.
- `stage` records which stage produced the card, since Section 9.2's
  warnings and Section 9.5's suggestions are both stage-scoped.
- `graded` names which of the `diagnostics` actually decided an assumption.
  Section 7.2's engine records *evidence* alongside verdicts
  (`Resolution.evidence`), and some evidence reads FAIL while deciding
  nothing — a `measurement_level` attempt against a scale no method
  required. Both belong on the card (Section 7.4's own table lists
  Shapiro's result, which is evidence under `normality_or_large_n`), but
  they must not look alike: a reader who notices the diagnostics are padded
  with irrelevant failures stops reading them, which is the one thing this
  project cannot afford. The renderer prints verdicts first, then
  "Also computed (evidence, not a verdict on any method)".

Next-step suggestions (Section 9.5) are `actions`, not a separate field:
they are buttons, and the rule that they are never executed automatically
is a property of the loop, not of the card.

### 9.5 Next-step suggestions
After each accepted step, the orchestrator shows 2–4 suggested next actions as buttons (deterministic rules first, optionally reworded by the LLM). Examples: after a significant k-group test → "Run post-hoc comparisons"; after imputation → "Compare distributions before/after imputation". It never executes them.

M7's rules, in the order they are offered: the post-hoc a **significant**
omnibus licenses; an equivalence test after a non-significant result; an
explanation of a graded check that FAILed; the ledger once more than one
test has run; and "ask another question". The 2–4 range is enforced rather
than hoped for — a clean significant result with no failed assumption and
no post-hoc would otherwise offer one button, and a lone button reads as a
dead end at the moment the user most needs somewhere to go, so the effect
size's explanation fills the gap.

A post-hoc is offered only after a *significant* omnibus: running pairwise
comparisons after a non-significant one is the inflation the omnibus exists
to prevent.

---

## 10. LLM layer

### 10.1 Client
- `litellm.completion` behind `LLMClient` with: model name from config, timeout, retries (2, exponential backoff), temperature 0 for structured calls, 0.3 for rationale.
- **Structured output strategy:** request JSON matching a pydantic schema (JSON mode / `response_format` where the provider supports it; otherwise instruct + parse). Parse with `Model.model_validate_json`. On failure, retry once with the validation error appended. Then fall back to the deterministic path.
- Do not rely on native tool-calling for the core flow; local models handle it unreliably. `select_adhoc_function` also uses structured JSON: `{"function": name, "params": {...}}`, validated against `registry.json_schema`, restricted to `read_only=True` functions.

### 10.2 Config (`edacopilot.toml` or `start(...)` kwargs)
```toml
[llm]
default_model = "gemini/gemini-3.8-flash"
rationale_model = "gemini/gemini-3.8-flash"   # optional per-call override
deterministic_mode = false

# A named alternative provider profile. Switching to it -- for local use,
# or back to a different hosted provider -- is `active_profile = "ollama"`
# below, never a code change (Section 1.3's "provider-agnostic" principle).
[llm.profiles.ollama]
default_model = "ollama/qwen2.5:7b-instruct"
api_base = "http://localhost:11434"
# active_profile = "ollama"   # uncomment to switch, config-only

[privacy]
level = "standard"          # "strict" | "standard"
alias_column_names = false  # true in strict mode

[analysis]
alpha = 0.05
random_state = 42
max_rows_for_exact = 5000
```
Model names are examples; the maintainer sets current ones. `gemini/` and
`ollama/` are LiteLLM provider prefixes read straight from `GEMINI_API_KEY`
and a local `api_base` respectively -- see Section 18's M9 entry for why
Gemini, not Anthropic, is the shipped default.

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

Implemented in `llm/fallback.py` and `llm/glossary.yaml` (M7);
`write_rationale` lives with its data in `personas/rationale.py` (M5), and
`elicit_mnar`'s question bank arrives with the missingness stage in M10.

- **`parse_intent` guesses narrowly and says when it is guessing.** A rule
  fires on an unambiguous marker or not at all, and an unmatched message
  returns `OTHER` below Section 9.1's floor so the orchestrator asks. A
  ruleset stretched to cover everything would misroute *confidently*, which
  is worse than not routing: the clarifying-question path only helps if the
  confidence is honest.
- **`build_question_spec` refuses to guess.** Without a model there is no
  honest way to turn "is income different by region?" into a spec, so the
  fallback returns the form's fields — with options taken from the actual
  frame, so it cannot offer a column that is not there.
- **The glossary has 207 terms**, above Section 10.5's ~150, and its
  coverage is derived from the live registry by
  `tests/unit/edacopilot/test_glossary.py`: registering a test or declaring
  an assumption without defining it fails CI. The style is checked too —
  one or two sentences, ending in a full stop, and not defined mostly in
  terms of its own name — because a glossary whose entry for `sphericity`
  reads "the sphericity assumption" has the same coverage and none of the
  value. An alias table maps the phrasings a user would actually type
  ("what is a p-value?", "Cohen's d", "Levene's test") onto the entries.

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

Implemented in `edacopilot/session/dataset_store.py` (M6). Three notes:

- **Parquet is written with zstd, not pandas' default snappy.** Measured on
  this project's WSL setup, a 1M-row / 5-column frame (41 MB in memory)
  writes to `/mnt/c` in 2.12 s with snappy and 1.06 s with zstd, for 19.3 MB
  against 12.7 MB; the same write to WSL's native filesystem takes 0.14 s.
  The Windows drive is reached over a 9p filesystem roughly an order of
  magnitude slower, so on a Windows-drive checkout the bottleneck is bytes
  crossing that boundary — which makes the smaller encoding faster *and*
  smaller. On native storage the two are within noise, so nothing is lost by
  preferring zstd everywhere.
- **An undo does not delete the version it moved off.** Only the branch head
  moves, so an undo the user regrets is recoverable by switching back or
  branching from it.
- `v0`'s parquet cannot be overwritten (`ReadOnlyVersionError`). It is the
  one state a session can always fall back to; a bug that rewrote it would
  be unrecoverable rather than merely wrong.

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

Implemented in `edacopilot/session/provenance.py` (M6). Rule 4 is enforced
rather than trusted: `assert_no_user_evaluation()` fails if `Step` ever
gains a field like `warnings_ignored`, `time_taken` or `skill`, and a test
proves the guard itself catches one. The log is append-only — an undo moves
the dataset head, it does not erase the fact that the step happened.

`Step.code` is rendered from `registry.to_code` at accept time, with the
function's own defaults filled in from its signature. A `Candidate` carries
only the params that identify the analysis (the columns, the question's
values), and a code template also references the ones with defaults (`ci`,
`nan_policy`, `random_state`). Rendering them explicitly is deliberate: the
exported notebook should state what it ran, including the seed, so the
number can be reproduced without knowing this version's defaults. A test
evaluates the recorded line and checks it returns the same p-value (rule 7).

M7 added two things `Step.code` has to carry for that promise to hold in
every case:

- **An override's reason, as a comment above the call** (Section 7.5's last
  clause). Someone reading the exported notebook meets the justification at
  the line it applies to, not in a log they would have to go and find.
- **A reshape, where the method needs one.** `edacore`'s paired tests take
  two *columns* while a paired question arrives in long form (outcome,
  group, subject). M4 recorded the two level names as the candidate's `a`
  and `b` params and left the pivot to the orchestrator deliberately — the
  eligibility engine reasons about the user's own columns, and pivoting
  inside it would make every check report against a frame the user never
  saw. So the step records the pivot as source too, because a notebook
  whose paired t-test ran against a frame the notebook never built would
  reproduce nothing.

### 12.3 TestLedger
- Every executed hypothesis test is appended with its family label (e.g. "income comparisons").
- Adjusted p-values are recomputed for the whole session (and per family) after each new test, using the active persona method or the user's session setting.
- The result card always shows raw p, adjusted p, and the test count.
- `session.ledger()` shows all tests with raw/adjusted p-values.
- **Omnibus tests vs. post-hoc results**: an omnibus test (`one_way_anova`,
  `welch_anova`, `kruskal_wallis`, `friedman`, `repeated_measures_anova`,
  `cochran_q`, `alexander_govern`, `permutation_anova`, and their
  Section 6.7 factorial/categorical/correlation counterparts) enters the
  ledger as exactly **one** entry, using its own `TestResult.p_value` —
  never one entry per implied comparison. A post-hoc procedure
  (`tukey_hsd`, `games_howell`, `dunn_test`, `nemenyi_friedman`,
  `conover_friedman`, `paired_posthoc`, `mcnemar_posthoc`,
  `permutation_posthoc`) returns a `PostHocResult` whose
  `PairwiseComparison.p_adjusted` values are adjusted **within that
  procedure's own family only** (`PostHocResult.p_adjust_method` and
  `len(comparisons)` name the method and family size) — they are not
  pooled into the session-wide adjustment `session.ledger()` computes
  across `TestResult` entries, and they do not add to the session test
  count. Rationale: a post-hoc family's multiplicity is already handled by
  its own procedure (Tukey's studentized range, Dunn's Holm step-down,
  etc.); re-adjusting those already-adjusted p-values against unrelated
  session tests would double-correct and make the family's own adjusted
  p-values uninterpretable. The omnibus test that triggered the post-hoc
  *does* count normally, since running the post-hoc doesn't change how
  many session-level conclusions that omnibus result itself supports.

### 12.4 Persistence
- `.edacopilot/<session_id>/session.json` (steps, ledger, config, branch heads) + parquet versions.
- `edacopilot.resume(session_id)` restores after a kernel restart.
- Add `.edacopilot/` to the user's `.gitignore` guidance in the README (it may contain data).

Implemented in `edacopilot/session/session.py` (M6). `session.json` is
written to a temporary file and renamed, so a kernel that dies mid-write
costs the last step rather than the whole history. Resuming validates that
every version's parquet is still present and says so plainly if the
directory was moved, instead of failing later inside pandas on a path the
user never typed.

The README carries the data warning, and `.edacopilot/` is in this repo's
own `.gitignore` too — a test run that forgets `tmp_path` should not be able
to commit a dataset.

**The eligibility hook.** `Session.candidates(spec)` passes the
DatasetStore's own version id as Section 7.2's `dataset_version` and holds
one `CheckCache` per version. A transform therefore produces a new version
with a new cache, and cannot read an assumption check computed about its
parent — which is the correctness property that cache exists for. A test
asserts the same `fact_id` returns a different statistic across versions,
rather than merely that the ids differ.

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

Implemented in `edacopilot/ui/` (M8). Four things worth stating:

- **The panel owns no logic.** Every button hands the `Intent` its card
  already carried to `Orchestrator.handle` — the same method the Python
  API calls. There is no path in the UI that reads eligibility, picks a
  persona or decides anything, which is what makes Section 13.1's claim
  true rather than aspirational. It is tested as an equality: two
  sessions, one driven through the API and one by invoking the buttons'
  own click handlers, must end byte-identical.
- **Colour is never the only signal.** `STATUS_MARKS` pairs every
  `CheckStatus` with an icon and a word as well as a colour, in one place
  so no renderer can drift from it. This is not a box-tick: these cards
  exist to tell someone whether an assumption failed, and a red/green
  distinction is invisible to roughly one man in twelve.
- **Portability is a constraint on what is used, not a thing to test for
  afterwards.** Core ipywidgets only; plots go in as PNG bytes in an
  `Image` rather than through a matplotlib display hook, because that is
  the one approach that behaves identically in Lab, Notebook 7 and VS
  Code. A test parses the UI modules' imports and fails on `ipylab`,
  `jupyterlab*`, `jupyter_server` or `notebook`; another asserts the panel
  emits `application/vnd.jupyter.widget-view+json`, in-process and
  (nightly) through a real `ipykernel` subprocess.
- **The override flow is the one place the UI adds a step.** The API takes
  the reason as an argument; a panel has to collect it. So an
  `override_confirm` card grows a text box with Accept disabled until it
  is non-empty, and an INELIGIBLE override grows a second confirmation
  checkbox that is also required. Both gates stay enforced in the
  orchestrator — the widget only makes them reachable with a mouse, and
  disabling the button is so the user is not invited to press something
  that will be refused.

Glossary terms in card text are wrapped in `<abbr title=...>` carrying
their one-line definition, so an explanation arrives where the consequence
does. `<abbr>` rather than a JavaScript popover because it is the one
mechanism that behaves the same in all three hosts and stays reachable by
keyboard.

`docs/ui_manual_checklist.md` is M8's acceptance criterion: the things
only a human in a real front end can judge, with the much longer list of
what is already automated.

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
- **Process rule**: after any edit to `generate_r_fixtures.R`, run it fully
  and confirm via `git diff` (or an equivalent before/after byte
  comparison) that no *existing* fixture file changed, before writing any
  Python that reads the new fixtures. A wrong R-object reference (the
  actual bug behind M3 part 1's and part 2a's own CI-fix rounds) either
  crashes the script outright or, worse, silently computes against the
  wrong data — the only way to catch the second case is diffing every
  fixture, not just checking the script exits 0.
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

M7 adds six, and there is no mock in them because in M7 there is nothing to
mock: the system runs in deterministic mode all the way down (rule 9). When
M9 adds the LLM, the recorded outputs slot in at `parse_intent` and
`build_question_spec` and these same transcripts keep their meaning — which
is why they drive the Python API rather than a UI.

A transcript is compared as a whole: every card, plus the provenance,
versions, ledger count and recorded code afterwards. That fails on a
wording change as well as a behaviour change, deliberately — the wording of
a card is most of what this product is, and a card nobody reads is a card
that did not work, so it should not be able to change without someone
looking at the diff. Regenerate with `python scripts/generate_transcripts.py`.
`docs/sample_conversation.md` is rendered from the same transcript by the
same renderer, so the document cannot drift from what the tests assert.

### 15.6 CI (`.github/workflows/ci.yml`, `.github/workflows/nightly.yml`)
- `test` job matrix: Python 3.11, 3.12; Ubuntu + Windows. Steps: `ruff check`,
  `ruff format --check`, `pytest -m "not llm and not slow"`, export
  round-trip test (exported notebook executes and reproduces numbers via
  `nbclient`).
- `lint` job: `mypy src/`, run once (Ubuntu, Python 3.12, numpy pinned) rather
  than per matrix leg — mypy's inferred types are sensitive to which numpy
  version pip resolves, so running it once with a pinned version keeps its
  pass/fail deterministic instead of drifting per-leg (Section 18, M3 part 1).
- `test-min-versions` job: Section 4.1's dependency floors pinned exactly.
- `@pytest.mark.slow` tests (coverage simulations) do **not** run on
  push/PR CI at all — they run on a nightly schedule (`nightly.yml`,
  `workflow_dispatch` also available for on-demand runs).
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
| M7 | Orchestrator in deterministic mode | Section 9 with fallback intents; 6.12; Python API (`session.ask/accept/override`) | Conversation tests pass with no LLM |
| M8 | Jupyter UI | Section 13; 6.11 (`viz.py`) | Manual checklist in JupyterLab, Notebook 7, VS Code; all actions reachable by buttons; every `Card.plots` ref renders |
| M9 | LLM layer | Section 10–11; LiteLLM; prompts; fact-check; ContextBuilder | Evals in 15.4 meet targets on one API model and one Ollama model; privacy scan 100% |
| M10 | Missingness | 6.3 incl. Little's test, elicitation flow | Matches `naniar::mcar_test`; `mar_by_group` detected |
| M11 | Outliers + transforms + quality | 6.2, 6.4, 6.5 | Before/after records correct; no input mutation |
| M12 | Time series | 6.9 | `non_stationary_ts` scenario passes; ADF/KPSS match R |
| M13 | Text | 6.10 | Profiling works; prompt logs contain no raw text |
| M14 | Export | Section 14 | Exported notebooks for all conversation tests execute and reproduce all numbers |
| M15 | Docs + example | README, quickstart notebook, CONTRIBUTING | New user can run quickstart end-to-end in < 10 minutes |

**Suggested agent split:** use a stronger model (Sonnet-class) for M3, M4, M5, M9 and M10, where statistical judgment and design matter. A smaller model (Haiku-class) is fine for M0, M1, M6, M8, M11, M13–M15 given this spec, with review.

**Section 6 coverage.** Every subsection of the function catalogue is named
by exactly one milestone above. The mapping, cross-checked in m7.1:

| Catalogue | Milestone | |
|---|---|---|
| 6.1 Profiling | M1 | done |
| 6.2 Data quality | M11 | |
| 6.3 Missingness | M10 | |
| 6.4 Outliers | M11 | |
| 6.5 Transformations | M11 | |
| 6.6 Assumption checks | M2 | done |
| 6.7 Hypothesis tests | M3 | done |
| 6.8 Effect sizes, multiplicity, power | M2 | done |
| 6.9 Time series | M12 | |
| 6.10 Text | M13 | |
| 6.11 Visualisation | M8 | added in m7.1 |
| 6.12 Validity notes | M7 | added in m7.1; delivered early, in m6.1 |

Two were missing when the check was run, and both had already cost
something. **6.12** was in Section 4.2's layout and in no milestone, so all
nine interpretation guards went unwritten until m6.1 caught it — M7's
result cards would have shipped without them, which is the one part of a
result card this product exists to provide. **6.11** was likewise
unassigned, which is why M7's EXPLORE stage is numeric-only: Section 9.3
promises it "bivariate plots" and there were no plotting functions to call.
It is M8's now, alongside the UI that has to draw them, and `Card.plots`
already carries the refs so nothing in the orchestrator changes when they
arrive.

A catalogue subsection with no milestone does not announce itself — it
simply never gets built, and the gap surfaces as a milestone that cannot
meet its own acceptance criteria. Worth re-running this cross-check
whenever Section 6 grows.

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

---

## 18. Changelog

Newest first. One entry per milestone (or per round of fixes against an
already-"complete" milestone); each links back to its git tag.

### 2026-09-28 — M9 (tag `m9`)
The LLM layer (Sections 3.4, 10, 11): `LLMClient` over LiteLLM, structured
output for all seven calls, the fact-check, and the `ContextBuilder` --
the first code in this repository that can call a model at all.

- **Provider switched from Anthropic to Google Gemini**, a maintainer
  decision made mid-milestone, not a spec default: `GEMINI_API_KEY`
  replaces `ANTHROPIC_API_KEY`, and `gemini/gemini-3.8-flash` (LiteLLM's
  `gemini/` prefix, which reads the key directly) replaces
  `anthropic/claude-haiku-4-5` as `default_model`. `gemini-3.8-flash` was
  chosen because it is the model Google's own model-listing page banners
  as newly released and recommends for new production use at the time of
  this change -- confirmed by fetching that page's raw HTML rather than
  trusting a summarized digest of it, since a summarizer with no real
  knowledge of a future date is exactly where a plausible-sounding but
  invented model name would slip through unnoticed. Ollama
  (`ollama/qwen2.5:7b-instruct`) remains a config profile, wired and unit
  tested; it was not exercised live this milestone because Ollama was
  unreachable from the development machine at the time (WSL2 -> Windows
  host networking). Switching provider is `active_profile` or
  `default_model` in `edacopilot.toml` -- Section 10.2's table now shows
  Gemini as the default and Ollama as the profile example.
- **`tenacity` is now a direct dependency**, not treated as transitive.
  LiteLLM's own `num_retries` kwarg (Section 10.1's "retries (2,
  exponential backoff)") wraps the call in a `tenacity.Retrying` at call
  time and raises `ImportError` if `tenacity` is not importable -- found
  by an unmocked smoke call during development that failed for exactly
  this reason. That call could not be confirmed not to have reached the
  network before failing, which is the reason the next point exists.
- **A suite-wide safety net: `litellm.completion` is patched to raise in
  every test by default** (`tests/conftest.py`, autouse). M7 and M8's
  entire test suite predates the LLM layer and calls
  `session.ask("free text")` throughout on the assumption that nothing
  reaches a network; M9 makes that call path capable of reaching one for
  the first time. Patching it to raise means every such call falls
  through to `LLMFallbackRequired` and Section 10.5's deterministic
  path -- the same behaviour those tests already asserted -- rather than
  either attempting a real call or requiring hundreds of pre-existing
  tests to be individually rewritten. A test exercising the LLM path
  re-patches `litellm.completion` itself for its own scope.
- **That net does not reach a subprocess.** `examples/m8_demo.ipynb`
  (m8.1) executes in its own Jupyter kernel via `nbclient`, and
  `scripts/generate_transcripts.py` runs `tests/conversations/`'s
  transcripts as a standalone script -- both outside pytest, so a
  monkeypatch in the test process has no effect on either. Both call
  `session.ask("free text")` somewhere (a post-hoc proposal, in both
  cases), which is exactly the path M9 wired to try a live model first.
  Both now set `config={"llm": {"deterministic_mode": True}}` explicitly
  at session creation (the notebook's own cells; `Transcript.start` for
  every golden transcript), rather than relying on the pytest-only net.
  Caught by an unmocked run of the notebook test before it was fixed --
  the fix and this note both belong to closing M9's own gap, not to
  m8.1's original tag.
- **That net does not reach a subprocess.** `examples/m8_demo.ipynb`
  (m8.1) executes in its own Jupyter kernel via `nbclient`, and
  `scripts/generate_transcripts.py` runs `tests/conversations/`'s
  transcripts as a standalone script -- both outside pytest, so a
  monkeypatch in the test process has no effect on either. Both call
  `session.ask("free text")` somewhere (a post-hoc proposal, in both
  cases), which is exactly the path M9 wired to try a live model first.
  Both now set `config={"llm": {"deterministic_mode": True}}` explicitly
  at session creation (the notebook's own cells;
  `Transcript.start` for every golden transcript), rather than relying on
  the pytest-only net. Caught by an unmocked run of the notebook test
  before it was fixed -- the fix and this note both belong to closing
  M9's own gap, not to m8.1's original tag.
- **`LLMClient.complete_structured`** (`llm/client.py`): one retry with
  the validation error appended on a schema failure (Section 10.1); any
  other failure -- timeout, malformed JSON twice, deterministic_mode on,
  the provider unreachable -- raises `LLMFallbackRequired` rather than
  propagating, which is the single seam every one of the seven calls
  relies on to keep rule 9 true without duplicating a try/except at each
  call site.
- **`write_rationale`'s second failure mode is Section 8.4's fact-check**
  (`llm/factcheck.py`), not a schema violation: valid JSON, valid shape,
  but a number, a `fact_id`, or a method name that does not trace back to
  what the model was actually given. One retry with the fact-check's own
  failure message appended, then `LLMFallbackRequired` -- so a rationale
  that passes the schema but fails to ground itself falls back to
  `personas/rationale.py`'s templates exactly like a malformed one does.
- **`ContextBuilder`** (`llm/context.py`) is the only path from session
  data to a prompt. Building it surfaced one privacy bug worth recording:
  a text-like column with (almost) all-unique values, if classified
  NOMINAL/ORDINAL rather than TEXT by the profiler, would have had its
  "top levels" computed and sent -- which for a column where every value
  is essentially unique is almost every raw value, exactly what Section
  11 says never leaves the session. Fixed by also checking the
  profiler's own `high_cardinality` flag, not semantic type alone, before
  computing top levels. A canary test (plant a unique string in a
  free-text column, assert it never appears in a rendered prompt, both
  privacy levels) is what caught it.
- **Two calls' "deterministic fallback" already existed before this
  milestone, just not in `llm/fallback.py`.** Section 10.5 lists five
  fallbacks; `suggest_next_step` and `select_adhoc_function` were never
  on that list because their deterministic behaviour predates the LLM
  layer entirely -- `stage.next_suggestions`/`result_suggestions` and the
  orchestrator's own keyword column-matcher (`_show`) respectively. Both
  calls are implemented and unit tested in `llm/calls.py`; only
  `parse_intent` and `build_question_spec` are wired live into the
  orchestrator this milestone (Section 15.5 names these two specifically
  as where a golden transcript's mocked LLM slots in), guarded by the
  same fallback seam. `write_rationale`, `elicit_mnar`,
  `answer_free_question`, `suggest_next_step` and `select_adhoc_function`
  are complete and tested standalone, ready to wire into their stage call
  sites as those stages' own milestones (M10-M13) touch them.
- **`elicit_mnar` has no deterministic fallback at all yet** (Section
  10.5 already says so: "arrives with the missingness stage in M10").
  Without a working LLM, a caller has nothing to fall back to until then.
- **Evals (Section 15.4) and `docs/sample_conversation_llm.md` are not
  part of this tag.** Both require live, approved calls against a real
  model and are run as a separate, explicitly approved step after `m9`
  is tagged, per the standing rule that any paid-API call outside the
  mocked test suite needs a go-ahead first.

### 2026-09-28 — M8.1 (tag `m8.1`)
Four loose ends in the M8 deliverable, closed before M9 starts.

- **`examples/m8_demo.ipynb`**, new (there was no `examples/` directory
  before this). Walks Section 7.4's worked example end to end through the
  Python API: the seeded income/gender dataset, the divergence card, an
  override with a typed reason on a `CAVEAT` candidate, an undo, and —
  since Section 7.4's dataset has only two groups — a second three-group
  dataset to show a significant omnibus offering, and waiting for
  acceptance of, its paired post-hoc. `tests/test_examples_notebook.py`
  executes it top to bottom via `nbclient` on every run; it needed no new
  CI step, because it is an ordinary test under `tests/` and the existing
  `pytest -m "not llm and not slow"` step already collects it.
- **The empty-reason-disables-Accept test already existed.**
  `tests/unit/edacopilot/test_ui.py::test_accept_is_disabled_until_a_reason_is_typed`
  (plus a whitespace-only variant and the `INELIGIBLE` double-confirmation
  case) was written with M8. Checked, not re-added.
- **A "Show plot" button for a `PASS` check that has one to draw.**
  Section 1.3's "hide noise" still keeps a passing check's plot out of the
  card by default, but `plot_for_check` was already a pure function of the
  check and the question (`orchestrator/diagnostic_plots.py`) — it simply
  was never called for anything but `FAIL`/`BORDERLINE`. The card now adds
  a button per plottable `PASS` check (`IntentType.SHOW_DIAGNOSTIC_PLOT`,
  carrying the check's `fact_id`); the click reaches
  `Orchestrator.show_diagnostic_plot`, which draws on demand through the
  same `plot_ref` cache the eager path uses, so repeats reuse the PNG. The
  button is withheld from `UNTESTABLE`/`NOT_APPLICABLE` checks even when an
  assumption name matches a plot builder — there is no computed check
  behind those, so a plot would show data without the evidence it claims
  to illustrate. `Session.show_diagnostic_plot(fact_id)` is the matching
  API method, following the one-`Session`-method-per-action convention.
  This changed five golden transcripts' button lists (new buttons only,
  same underlying behaviour) — regenerated with
  `scripts/generate_transcripts.py`.
- **Section 6.11's three signature departures are now in the catalogue
  line itself**, not only in the prose bullet below it: `outlier_plot(df,
  col, flagged=None, whis=1.5)`, `change_point_plot(df, col,
  change_points, time=None)`, and `before_after(before, after, col,
  record=None)`, each marked `†` back to the existing explanation. No
  implementation changed — `viz.py` already matched this; only the
  catalogue line was stale against it.

### 2026-09-26 — M8 (tag `m8`)
The Jupyter UI (Section 13) and the visualisation catalogue (Section 6.11),
which m7.1's audit had just moved into this milestone. Still no LLM call
anywhere.

Deliverable for review: `docs/ui_manual_checklist.md` — the things only a
human in a real front end can judge, and the longer list of what no longer
needs checking by hand.

**Part A — `edacore/viz.py`.** All 25 functions Section 6.11 lists.

- **They return a `Figure`; the session mints the `plot_ref`.** Section
  6.11's sentence spans the `edacore` boundary and had to be split:
  `edacore` draws, `session.plot(...)` writes the PNG and returns the id.
  Giving `edacore` a session would have broken Section 3.1's promise that
  it stands alone as a statistics library.
- **No pyplot.** Each builds a bare `Figure`, because `pyplot` never
  releases a figure and a library going through it leaks one per call into
  the host kernel. A test asserts the global registry stays empty across
  all 25 — the kind of thing that is invisible until a long notebook
  starts warning.
- **Rule 7 for plots**, by executing the rendered code and comparing the
  artists' own arrays rather than pixels. A pixel comparison would fail on
  a matplotlib point release or a font substitution, and a pixel diff is
  not something anyone can act on.
- **A `plot_ref` is keyed on the dataset version as well as the function
  and its parameters.** The same diagnostic rendered twice reuses one PNG;
  the same plot of a transformed frame is a different plot, because it is
  a claim about a different state of the data.
- **Plots are wired into cards.** A FAILing or BORDERLINE check gets the
  picture that shows what failed — a Q-Q plot for normality, boxes for
  variance, a violin for shape — and the five normality checks on one
  question collapse onto one Q-Q plot rather than five. Passing checks get
  none: Section 1.3's "hide noise" applies to evidence as much as to
  personas, and a card the user learns to scroll past is worse than a
  shorter one. EXPLORE gained the univariate and bivariate plots Section
  9.3 had been promising since M7, where it shipped numeric-only precisely
  because 6.11 did not exist.

**Part B — `edacopilot/ui/`.**

- **The panel owns no logic**, and that is tested as an equality rather
  than asserted: two sessions, one driven through the Python API and one
  by invoking the buttons' own click handlers, end byte-identical —
  including a whole transcript and the full override flow. Because of
  that, M7's conversation tests cover the UI too, and the only bugs M8 can
  introduce are rendering bugs.
- **Every card kind renders, twice over**: once from a synthetic card per
  kind, and once from every card the six golden transcripts actually
  produce. The second matters because a hand-made example fills fields the
  real thing sometimes leaves empty — a result with no CI, a divergence
  whose proposal has no function — and those are what break a renderer.
- **Status is never colour alone.** `STATUS_MARKS` pairs each
  `CheckStatus` with an icon and a word, in one place, with tests that the
  icons and words are distinct as well as present.
- **Portability was designed in rather than checked afterwards**: core
  ipywidgets only, PNG bytes in an `Image` instead of a matplotlib display
  hook. One test parses the UI modules' imports and fails on `ipylab`,
  `jupyterlab*`, `jupyter_server` or `notebook`; another asserts the panel
  emits `application/vnd.jupyter.widget-view+json`, which is what all
  three hosts render — in-process, and nightly through a real `ipykernel`
  subprocess, which is the only test here that exercises the kernel path
  at all.
- **Glossary terms in card text carry their definition** in an
  `<abbr title=...>`, so the explanation arrives where the consequence
  does. Terms inside code spans are left alone, and words shorter than
  four characters are skipped — underlining every "the" would make a card
  unreadable and teach nothing.
- **The override flow is the one place the UI adds a step**, because the
  API takes the reason as an argument and a panel has to collect it.
  Accept is disabled until the reason is non-empty, and for an INELIGIBLE
  method until the second confirmation is ticked too. The orchestrator
  still enforces both; disabling the button is so the user is not invited
  to press something that will be refused.
- **`%eda` magics and `eda.start(df, ...)`** per Section 13.1, each a
  single `session.*` call. Argument parsing is `shlex` plus a small table
  rather than `argparse`, which exits the process on a bad argument — in a
  notebook that means killing the kernel.

Three smaller things:

- **`ipykernel` is now a `[dev]` dependency**, not a hard one. VS Code can
  only select a virtualenv as a kernel if it is installed there, but the
  library never imports it and a user's own Jupyter install provides one.
- **A root `conftest.py` sets the Agg backend** before anything imports
  pyplot, and closes figures after every test. Backend selection would
  otherwise depend on what is installed on the machine — an interactive
  backend locally and Agg in CI, which is the shape of difference that
  makes a suite pass in one place and fail in the other.
- **`Text.on_submit` and `Layout(overflow_y=...)` are both gone in
  ipywidgets 8.** Running the panel under `-W error::DeprecationWarning`
  found them; a test now pins that no deprecated widget API is used.

### 2026-09-26 — M7.1 (tag `m7.1`)
Three checks against M7, two of which found something, plus a round of
reviewed glossary wording.

- **Fourteen glossary entries replaced with the maintainer's own wording**,
  after review of the terms flagged as uncertain in M7's report. Most were
  clarity fixes; two were substantive. `partial_eta_squared` now says what
  it is (variance explained by one effect after removing the others, equal
  to eta-squared in a one-way design) rather than asserting a reporting
  convention. `post_selection_inference` is renamed
  `data_driven_comparison` — the old name described the field, not the
  thing the user did — with the old phrasing kept as an alias, since it is
  what a reader would type. `provenance` and `data_version` are now
  labelled as edacopilot vocabulary rather than statistics, and the three
  `*_posthoc` wrappers say what they run and how they adjust instead of
  gesturing at "corrected within the family". The wording is pinned by
  substance in `test_glossary.py`, so a copy-edit is allowed and a change
  of meaning is not.

- **`epsilon_squared` was one name doing one job under a name that means
  another.** The question was whether it served as both the ANOVA measure
  and the Kruskal-Wallis rank measure; it does not — the registered
  function is rank epsilon-squared, H / (N - 1), it is reported only by
  `kruskal_wallis`, and it is checked against
  `effectsize::rank_epsilon_squared`. The ambiguity was in the *name*:
  "epsilon-squared" unqualified usually means Kelley's bias-corrected
  ANOVA measure, a different quantity from a different test. A result card
  reading `epsilon_squared = 0.1246` would be read by anyone who knows the
  ANOVA measure as the ANOVA measure, which is the class of misreading
  this project exists to prevent, so the function, the reported
  `effect_size_name` and the magnitude-threshold key are all now
  `rank_epsilon_squared`. The glossary carries **both** names: the bare one
  explains that it is ambiguous and says which measure this project
  computes, and `rank_epsilon_squared` defines the one it actually reports.
  No split of the *implementation* was needed because there is only one
  measure — but the name needed splitting, which is the same problem one
  step earlier.
- **The post-hoc pairings were already the specified ones**, and now each
  has a test. The pairing is statistical, not stylistic: Tukey's
  studentized range assumes the equal variance that Welch's ANOVA and
  Alexander-Govern were chosen to avoid assuming, so offering Tukey after
  either would hand back the assumption the user had just been steered away
  from — with the correction silently wrong rather than visibly absent.
  `test_post_hoc_pairings.py` asserts the table against the specification,
  drives all eight pairings end to end from a significant omnibus through
  the suggested button to the executed procedure, and checks both
  directions of the case that matters: Tukey never after Welch or
  Alexander-Govern, Tukey always after one-way ANOVA.
- **Two Section 6 subsections belonged to no milestone.** Both were in
  Section 4.2's repository layout, which is why they read as planned.
  **6.12** (validity notes) went unwritten until M6.1 caught it; M7's
  result cards would have shipped without a single interpretation guard.
  **6.11** (visualisation) is why M7's EXPLORE stage is numeric-only —
  Section 9.3 promises it "bivariate plots" and there were no plotting
  functions to call. 6.11 is now M8's, alongside the UI that has to draw
  them, and 6.12 is recorded against M7. Section 16 carries the full
  mapping, and `test_spec_coverage.py` parses the specification and fails
  if a subsection is ever added without one: the failure mode is not a
  visible gap but a milestone that quietly cannot meet its own acceptance
  criteria, years of context later.

### 2026-09-26 — M7 (tag `m7`)
The orchestrator in deterministic mode (Section 9, with 7.5 and 10.5):
the turn loop, the stage state machine, the cards, and the Python API that
M8's panel will drive. No LLM call anywhere in this milestone.

Deliverable for review: `docs/sample_conversation.md` — Section 7.4's
worked example as plain-text cards, exactly as a user sees them.

- **The conversational API is on `Session`** and every method returns a
  `Card`: `ask`, `answer`, `accept`, `override`, `modify`, `explain`,
  `goto_stage`, `skip`, `undo`, `branch`, `switch_branch`. `ledger()` keeps
  returning rows, because it is data rather than a card and a notebook user
  wants the table; `orchestrator.ledger_card()` renders it when a card is
  what is wanted.
  - `ask()` takes a `QuestionSpec`, keyword arguments (`goal=`, `outcome=`,
    `group=`, `design=`), or free text. Free text goes through Section
    10.5's keyword parser; anything it cannot resolve to columns returns
    the form-style card rather than a guess.
  - **Three renames were needed** to put the conversational verbs on the
    names Section 9 uses. M6's `Session.accept(stage=..., chosen=...)` is
    now `record_step`, which is what it does; the `branch` *property* is
    now `active_branch`, freeing `branch(name)` for the action; and the
    store-level `switch_branch` is now `switch_to_branch`. The old names
    were the right ones for the new meanings, and leaving them on the
    plumbing would have made the API read as an afterthought.
- **Rule 3 is a property, not a promise.** Only `accept` and `override`
  reach `record_step`. A hypothesis property test drives random sequences
  of the seventeen non-accepting actions and asserts no step, no ledger
  entry and no data version appears; a second test checks the call graph
  itself, so a conversational method added later that forgot the rule fails
  immediately rather than when someone remembers to add it to the list.
- **The design question reaches the user before any persona card.** Not by
  convention but by construction: `select_candidates` raises
  `AmbiguousSpecError` on a spec with open ambiguities, and a persona can
  only choose from a `CandidateSet`, so there is no path to a proposal that
  does not pass through the question. Tested for all three design claims on
  `paired_as_independent`, including that the orchestrator never edits
  `spec.design` itself.
- **Section 7.5's friction applies to CAVEAT and INELIGIBLE only** — see
  Section 7.5 for why an ELIGIBLE method needs none. An INELIGIBLE override
  shows the failed hard assumption *and* its consequence, and needs the
  reason and a separate `confirm=True`; five parametrised combinations of
  the two gates assert none of them runs anything, and the internal path
  raises `OverrideRequired` rather than returning a card, so a bug there
  fails loudly instead of running the method.
- **Stage modules.** PROFILE, EXPLORE (read-only) and HYPOTHESIS are
  implemented; QUALITY, MISSINGNESS, OUTLIERS, TRANSFORM, TIMESERIES, TEXT
  and EXPORT return an info card naming the milestone that brings them, and
  **refuse** to build a spec or execute. A stub that returned something
  empty would let a user believe the stage had run, which is the failure
  this project exists to prevent. `allowed_functions` is enforced, not
  documented: a stage that calls outside its scope raises.
- **Section 9.2's jump warning names a number from this dataset.**
  "Missing values haven't been reviewed; 12% of `income` is missing and
  tests will drop those rows" teaches something; "you skipped missingness"
  does not. One line, with the other skipped stages in a parenthetical —
  a wall of warnings at every jump is a wall the user learns to scroll
  past. When several stages were skipped, missingness speaks first: missing
  rows are dropped without appearing anywhere the user looks, while a
  duplicate at least leaves a visible count behind. EXPLORE never warns
  (read-only), and the two conditional stages sit outside the main line so
  a frame with no text columns is never warned about TEXT.
- **Which stage you are in, and which you have visited, are persisted**
  (`state_version` 2). Section 9.2's warning has to survive a kernel
  restart or it becomes a lie. The transient half — a proposal on screen, a
  question awaiting an answer — deliberately is not: a resumed session
  re-asks rather than accepting into a context the user no longer has in
  front of them.
- **Post-hoc candidates inherit the omnibus's eligibility.** A post-hoc is
  in no Section 7.3 family: it is not something the user asks for, it is
  something a significant omnibus licenses. Rather than invent a fresh
  ELIGIBLE verdict for it — which would claim a check that never ran — the
  candidate carries the status and reasons of the omnibus the user
  accepted, and the card says so. Its p-values stay out of the session
  adjustment exactly as Section 12.3 requires, asserted in the transcript's
  ledger.
- **Six golden transcripts** (Section 15.5), covering the 7.4 worked
  example end to end, `paired_as_independent` from design question to
  result, an INELIGIBLE override with both gates, undo followed by a
  branch, a significant Kruskal-Wallis followed by an accepted post-hoc,
  and a jump to HYPOTHESIS with untreated missing data.

Three fixes to earlier milestones that only became visible once cards
existed to show their output:

- **`check_measurement_level`'s consequence read "'income' looks like
  continuous, not continuous" on every PASS.** The wording was
  unconditional. It is phrased for the status it actually has now — that
  single line, repeated down a diagnostics table, is how a reader learns to
  stop reading diagnostics.
- **Evidence was indistinguishable from verdicts.** `CandidateSet` now
  records which fact_ids graded an assumption (`graded`, and
  `graded_checks()`), so a card can print verdicts and evidence separately
  instead of showing a `measurement_level` FAIL against a scale no method
  required as though it had decided something. Section 9.4 has the full
  argument.
- **A numpy scalar in `Candidate.params` made a session unwritable.**
  `cochran_armitage_trend`'s `event` param is taken from the column's own
  values, so it arrives as `numpy.int64`, and pydantic cannot serialise
  one. The failure surfaced far from the cause — not when the value was
  stored but later, when `session.json` was written, on the step the user
  had just accepted. `edacore.contracts.to_jsonable` converts at the
  boundary, on every `Any`-typed field that can hold data-derived values.

### 2026-09-26 — M6.1 (tag `m6.1`)
Three fixes ahead of M7, each closing a gap M6 left that M7 would have
built on.

- **Serialisation audit (Section 12.4).** Five fields across both packages
  are unordered collections; M6 fixed the two it happened to trip over
  (`Candidate.tags`, `QuestionSpec.confirmed_by_user`). The other three —
  `FunctionSpec.tags`, `EquivalenceRule.functions`,
  `PersonaPolicy.method_pool_tags` — now sort on dump too. Fixing the
  instances is not the same as fixing the class of bug, so
  `test_canonical_serialisation.py` walks every pydantic model in both
  packages and fails on any set-typed field without a sorting serializer:
  a new model cannot slip through and resurface months later as an
  intermittent resume failure. Only sets needed this. `dict` preserves
  insertion order, and every dict on these models is built by iterating a
  list (a family's methods, `PERSONA_ORDER`, the dataset's columns, the
  store's versions in creation order), so its order is already a function
  of the value that built it.
- **`paired_as_independent` ordering confirmed, and shown.** The design
  ambiguity does reach the user before any persona card, by construction
  rather than by convention (see the M7 entry). `docs/persona_picks.md`
  already showed the state *after* confirmation; it now renders the
  question that comes first, from the live engine, above the table — a
  table that showed only the answer would read as though the system had
  decided the design itself, which is the one thing Section 7.1 forbids.
- **Validity notes (Section 6.12) did not exist.** `validity_notes.py` was
  in Section 4.2's layout and in no milestone's deliverables, so all nine
  guards were missing and M7's result cards would have shipped without
  them. Implemented with a test per guard for the case it fires on *and*
  the case it must stay silent on — a guard that fires on everything
  teaches nothing, and one that never fires is decoration. The two
  conventions Section 6.12 left open are recorded there.

### 2026-09-25 — M6 (tag `m6`)
The session layer (Section 12), in `edacopilot/session/`: the dataset version
DAG, the provenance log, the test ledger, and persistence that survives a
kernel restart.

- **DatasetStore** (12.1). Every accepted transform creates a child version;
  nothing is edited in place, which is rule 6 at the storage level. `v0`'s
  parquet cannot be overwritten — it is the one state a session can always
  fall back to, so a bug that rewrote it would be unrecoverable rather than
  merely wrong. Only `v0` and the active head stay in memory, asserted via
  `store.in_memory()` rather than assumed. An undo moves the branch head and
  does **not** delete the version, so an undo the user regrets is
  recoverable.
- **Parquet uses zstd, not pandas' default snappy**, decided from
  measurement rather than preference. 1M rows x 5 columns (41 MB in memory)
  to `/mnt/c`: snappy 2.12 s / 19.3 MB, zstd 1.06 s / 12.7 MB. The same
  write to WSL's native filesystem takes 0.14 s, so `/mnt/c` is roughly an
  order of magnitude slower — it is reached over 9p — and the bottleneck is
  bytes crossing that boundary, which is why the smaller encoding is also
  the faster one. On native storage the two are within noise (0.10 s vs
  0.08 s), so nothing is lost by using zstd everywhere.
  `test_parquet_performance.py` (slow, nightly) asserts the *decision* —
  materially smaller, not materially slower — because a wall-clock
  assertion would fail on a loaded runner for reasons unrelated to the code.
  It runs on `tmp_path`, which is native, so it reproduces the native rows
  and not the `/mnt/c` ones; the size win is the half that generalises.
- **ProvenanceLog** (12.2), append-only. Rule 4 is enforced, not trusted:
  `assert_no_user_evaluation()` fails if `Step` ever gains a field like
  `warnings_ignored` or `time_taken`, and a test proves the guard catches
  one. An undo does not erase a step — "what did I do and then undo?" is a
  fair question.
- **`Step.code` is complete and runnable.** A `Candidate` carries only the
  params that identify the analysis, so the first version rendered
  `# could not render code for welch_t: code_template references '{ci}'` —
  which would have made rule 7's promise hollow. The defaults are now filled
  from the function's own signature, and a test evaluates the recorded line
  and asserts it returns the same p-value. Rendering defaults explicitly is
  deliberate: an exported notebook should say what it ran, including the
  seed.
- **TestLedger** (12.3). Adjusted p-values are *recomputed* after each test,
  not accumulated, because Holm is a step-down method — adding a test
  changes every earlier adjusted p-value, and a ledger that computed each
  entry once would show stale numbers for all but the newest. Adjustment
  delegates to `edacore.multiplicity.adjust_pvalues`, and the acceptance
  test compares against that function on the same inputs for all four
  methods. Section 12.3's rule holds exactly: an omnibus test is one entry,
  a post-hoc is recorded but excluded from both the session adjustment and
  the session count, and the sharpest form of that — adding a post-hoc
  leaves every real test's adjusted p-value untouched — is asserted
  directly.
- **The eligibility hook is wired.** `Session.candidates()` passes the
  DatasetStore version id as Section 7.2's `dataset_version` and keeps one
  `CheckCache` per version, so a transform cannot read a check computed
  about its parent. The test asserts the same `fact_id` returns a *different
  statistic* across versions, rather than merely that the version ids
  differ — the latter would pass even if the cache were shared.
- **Persistence** (12.4). `session.json` is written to a temp file and
  renamed, so a kernel dying mid-write costs the last step rather than the
  history. `resume()` validates that every version's parquet is present and
  says the directory was moved, instead of failing later inside pandas.
  M6's acceptance criterion is a filesystem round-trip: a busy session
  (tests, a transform, an override, a branch) resumed from disk has
  `to_state()` equal to the original, identical frames for every version,
  and can be continued.
- **README and `.gitignore`**: `.edacopilot/` holds the user's data as
  parquet, so the README warns about it prominently and this repo ignores it
  too — a test run that forgets `tmp_path` should not be able to commit a
  dataset.
- **Serialising a `set` was not canonical**, found by the acceptance test
  failing once in about ten full runs and passing on retry — the shape of
  bug that gets written off as flaky. CPython iterates a set in hash-table
  order, and two sets with the same members can iterate differently when
  their insertion histories collided differently, which depends on the
  per-process string hash seed. So `Candidate.tags` and
  `QuestionSpec.confirmed_by_user` dumped in an order that was not a
  function of their value: a session written to JSON and read back produced
  an *equal* model whose dump differed. Both now have a `field_serializer`
  that sorts. This matters beyond the test — non-canonical state makes
  `session.json` diffs noisy and any state comparison unreliable — so the
  property is pinned by its own test (built from two different insertion
  orders), since the round-trip test cannot catch it reliably. Verified
  across several `PYTHONHASHSEED` values.

### 2026-09-25 — M5.1 (tag `m5.1`)
Divergence cards that only appear when the personas actually disagree. A
card the user learns to dismiss is worse than no card, because the one time
it matters it gets dismissed too.

- **Practical equivalence (Section 8.3)**, as config in
  `personas/equivalences.yaml`. The first rule: `student_t` and `welch_t`
  are one proposal when the `equal_variance` check PASSes. Two safeguards
  keep it collapse-only — a rule's `when_passes` condition is mandatory and
  an *absent* check does not satisfy it (not evaluated is not the same as
  passed), and a rule can turn a divergence into a consensus but never the
  reverse. The consensus card then names each persona's method and why the
  conclusion is the same.
- **The Maverick proposes only when it adds something** (`proposal_policy`
  in `maverick.yaml`): at least one soft assumption BORDERLINE/FAIL, or
  outliers flagged. Otherwise, and whenever its own pool is empty, it
  concurs with the Professor. Its own `must_state_tradeoff` constraint is
  the argument — an unfamiliar method costs ("less familiar to reviewers")
  and on clean data buys nothing. Concurring is not silence: the persona
  still appears with a one-line note saying it looked and had nothing to
  add, which also replaces the old "no method to offer" on ordinal
  association questions. `outliers_flagged` currently reads
  `check_influential_outliers`, the only outlier diagnostic that exists;
  Section 6.4's detection (M11) feeds the same trigger.
- **Section 8.1 now says plainly** that a persona's `method` list names the
  checks it *cites* and never permits ignoring a FAIL from a check it does
  not cite. That was M5's reading, decided on the strength of 7.4; it is
  now written down rather than inferred.
- **Scenario cards after the change**: 3 consensus, 3 divergence (was 1
  consensus, 5 divergence). `clean two groups` became a consensus by
  equivalence (Professor → Student's t, Consultant → Welch, Maverick
  concurring); `paired_as_independent` and `ordinal_as_numeric` became
  consensus because the Maverick concurs instead of dissenting or going
  silent. **Section 7.4 is unchanged**: a two-proposal divergence card,
  with the Maverick proposing on its own because normality and variance
  both fail there.

### 2026-09-25 — M5 (tag `m5`)
The persona engine (Section 8), deterministic parts, in
`edacopilot/personas/`. Section 8.4's LLM-written rationales stay for M9;
what is here is Section 10.5's fallback, which is also how the whole test
suite runs. Deliverable for review: `docs/persona_picks.md`.

- **Section 7.4 reproduces exactly**: Professor + Consultant →
  `mann_whitney`, Maverick → `yuen_trimmed_t`, as a divergence card with
  two proposals (the first merged, the second separate). The spec's own
  stated reason for the consultant's pick also holds: its CLT shortcut does
  not fire, because Cochran's rule needs n > 89 for a skew of 1.89 and
  there are 38.
- **Rule 2 is a property test, not a spot check.** `pick` can reach a
  candidate by four routes — the eligible pool, the least-caveated
  fallback, and two relaxation rules — and a targeted test only proves the
  route it walks. Hypothesis drives random data (normal, lognormal,
  heavy-tailed, uniform, ordinal; 2–4 groups; n from 3 to 60; equal and
  unequal spread) through the engine and asserts no persona ever proposes
  an INELIGIBLE candidate, never picks outside its own method pool, and
  never relaxes anything that was not a CAVEAT to begin with.
- **Personas are data, and the schema is what keeps them honest.**
  `extra="forbid"` throughout: a misspelled `method_pool_tag` would
  otherwise leave the persona with an empty pool and no indication why it
  had stopped proposing anything. An unrecognised `prefer` key is likewise
  a load error, since silently skipping it would leave the persona ranking
  by something other than its file. A fourth persona file is rejected —
  Section 8 defines three, and a fourth changes what a divergence card
  means.
- **`borderline_is: pass` is blocked by a single FAIL.** This is the one
  design decision in M5 worth flagging. Section 8.1's consultant lists
  `normality.method: [descriptive]`, which could be read as licence to
  ignore a Shapiro FAIL it never consults. It is not read that way here:
  `method` lists the checks a persona *cites*, and relaxation happens only
  through the two named rules. A persona that could reach eligibility by
  declining to look at a check would be skipping an assumption silently,
  which is exactly what rule 2 exists to prevent — and it would also break
  7.4, where the consultant is supposed to land on the same rank-based
  method as the professor.
- **`always_robust` means always.** The consultant now declines any method
  declaring `equal_variance`, not merely when the variance check fails.
  That is what its YAML comment says ("always use Welch-type methods; no
  variance test needed") and it is the defensible reading: Welch costs
  almost nothing when variances are equal, so a persona optimising for
  fewest defensible steps has no reason to test first and then decide. The
  visible effect is on *clean* data, where the three personas now diverge
  three ways (Professor → Student's t, Consultant → Welch, Maverick →
  Yuen) instead of two. 7.4 is unaffected.
- **The engine now records Cochran's rule on any family with a
  normality-family assumption**, whether or not a method declares
  `normality_or_large_n`. Without that, a persona's CLT shortcut would work
  in the ANOVA family (where `welch_anova` surfaces the fact) and silently
  not work in the factorial family (where nothing does.)
- **Template rationales** are built from `AssumptionCheck.consequence`, not
  paraphrased: a rationale has to say what goes wrong, and that sentence is
  already written for a junior analyst (Section 6.6). A test asserts every
  number appearing in a rationale also appears in a check's own text, which
  is rule 1 applied to the fallback layer — the template engine computes
  nothing.
- **One finding for your decision**: the Maverick has *no method to offer*
  for an ordinal association question. The `ordinal_any` family offers
  `spearman`, `kendall_tau` and `cochran_armitage_trend`, and none carries
  a tag in the maverick's pool (`resampling`, `robust`,
  `nonparametric_advanced`, `maverick`). That is correct given Section
  8.1's tags, and the card says so plainly rather than inventing a pick —
  but if the Maverick should have something to say there, the fix is to tag
  `kendall_tau` (or `distance_correlation`, already tagged `resampling`,
  by adding it to that family) rather than to change the pick algorithm.
- **New declared dependency**: `pyyaml>=6`, plus `types-PyYAML` for mypy.
  It was already present transitively via litellm; Section 8.1's policies
  are YAML files, so the dependency is now real rather than a happy
  accident, and the min-versions CI leg pins it.

### 2026-09-25 — M4.1 (tag `m4.1`)
Pre-M5 hardening. No new milestone scope; three gaps closed and one
threshold replaced.

- **Status tests for every assumption check**
  (`tests/unit/edacore/test_check_statuses.py`). The R-fixture tests prove
  each check computes the right *number*; nothing proved it reached the
  right *verdict* from that number. `check_monotonicity` matched R's
  Spearman rho to 1e-9 for two milestones while reporting FAIL on perfectly
  monotone data. All 29 registered checks now have a clean dataset they
  must not FAIL on and a violating dataset they must FAIL on.

  Asserted across ten seeds, not one, and this is the point of the file:
  these are 5%-size tests, so they reject clean data about once in twenty
  runs by construction. A single hand-picked seed would pass for a correct
  check and for one that is wrong 60% of the time alike. The requirement is
  "never FAIL on clean data, and PASS in at least 8 of 10" — which an
  inverted check fails instantly (the pre-M4 `check_monotonicity` scored
  0/10) — plus "FAIL on all 10 violating seeds", since a violation built to
  be unambiguous should leave no room for leniency.

  **No further wrong statuses were found.** The two that initially looked
  wrong were not: `check_linearity` rejecting linear data was a chance
  rejection at the nominal 5% rate (it passes 10/10 with the seed band),
  and `check_normality_descriptive` returning BORDERLINE on exponential
  data is its documented rule (skew 2, excess kurtosis 6 sit inside the
  BORDERLINE band) — the violating dataset was strengthened to lognormal
  rather than the check changed.
- **Assumption-vocabulary invariant**
  (`tests/unit/edacopilot/test_assumption_registry_invariant.py`). Section
  5.4 lets a function declare assumptions as free strings, so a typo or a
  copied method can name one the engine has never heard of. Every hard and
  soft assumption on every registered function — not just hypothesis tests
  — must now resolve to a registered check or to an explicit ask-user
  handler returning UNTESTABLE, and each resolution must produce real
  evidence with a fact_id, a threshold and a consequence. Both directions
  are asserted: an undeclared resolver is dead code that reads as live
  behaviour, so that fails too. Verified to fail as intended by registering
  a function with a bogus assumption.
- **`clt_shortcut` is now Cochran's rule**: the large-sample condition is
  met, per group, when `n > 25 * skew^2`. It replaces the flat pair
  (`LARGE_N_FOR_CLT = 100`, `MAX_SKEW_FOR_CLT = 2`), which answered the
  wrong question — whether n is "big" in the abstract rather than big
  *relative to how skewed this sample is*. Cochran's rule scales the
  requirement with the problem, and the verdict now names the binding group
  and its arithmetic ("group 'F' has skew 1.89, so it needs n > 89.3 and
  has 38"). Section 8.1 updated: the consultant's `clt_shortcut: cochran`,
  the professor's `clt_shortcut: none`.

  **Section 7.4 is unchanged**: at n = 38/41 with skew 1.89/1.70 the rule
  requires n > 89.3 and n > 72.6, so the escape cannot fire and `welch_t`
  stays CAVEAT. One scenario did change, for the better: on
  `heteroscedastic_groups` (n = 60/25, near-symmetric) Cochran's rule is
  met, so `welch_t` becomes ELIGIBLE and only `student_t` is caveated —
  which is exactly what Section 15.3's "Student t → CAVEAT; Welch
  preferred" describes.

  **What it does not cover**: Cochran's rule bounds SKEWNESS. A symmetric
  heavy-tailed sample has skew near zero and satisfies it at any n, even
  though the sample variance a t-test leans on is badly behaved. The
  normality checks still run and are still reported as evidence, so the
  failure stays visible — it simply no longer produces a caveat by itself.
  `heavy_tails_small_n` still caveats the parametric methods because its
  sample skew is large (2.73 at n=12), not because the rule catches heavy
  tails.

### 2026-09-25 — M4 (tag `m4`)
The deterministic eligibility engine (Section 7), in
`edacopilot/eligibility/`. First real code in the agent layer; still no LLM
anywhere in it, which is what makes rule 2 ("no persona can propose an
ineligible method") enforceable at all. Deliverable for review:
`docs/eligibility_table.md`.

- **The design cross-check is the whole point, so it never decides.**
  `validate_spec` compares the design the question was read as against the
  subject ids, and every branch it cannot settle becomes a question in the
  user's own words — including the mirror-image error (design says paired,
  but no id appears in both groups) and the case where the id column was
  never named and `detect_structure` found it (the question says so).
  `select_candidates` raises `AmbiguousSpecError` on a spec that still has
  ambiguities, so Section 7.1 step 3 is enforced, not merely expected of
  the orchestrator. `spec.design` comes back exactly as it went in.
- **Ids repeating *within* one group is clustering, not pairing**, and the
  new `check_design_crossing` reports the two differently. Conflating them
  would push the user toward a paired test that has no pairs to work with.
- **The registry's assumption metadata did not match Section 6.7's tables**,
  and since the engine is driven by that metadata, every gap was an engine
  bug waiting to happen. Reconciled: `independent` added to the 8 two-sample
  and 9 k-sample/categorical methods that need it, `paired`/`repeated` to
  the 7 related-samples ones, `min_n_per_group>=2` to `student_t`,
  `same_shape` to `mann_whitney`, `exchangeability` to `permutation_test_2s`.
  Two changes that alter behaviour and are worth your eye: `ks_two_sample`'s
  outcome assumption is now `continuous` (Section 6.7's Hard column) rather
  than `numeric_or_ordinal`, and `two_proportion_z`'s n·p ≥ 10 moved from
  soft to hard, also per that column — so both are now INELIGIBLE rather
  than caveated on data they do not fit.
- **Five Section 6.6 checks were named by Section 6.7 but did not exist**:
  `check_design_crossing`, `check_symmetry`, `check_influential_outliers`,
  `check_balanced_design`, `check_proportion_counts`, plus
  `check_expected_counts_gof` (the hard assumption `chi2_goodness_of_fit`
  declares had no one-sample check behind it, so it could never block).
  Each is verified against R where R has an equivalent.
- **`check_monotonicity` was backwards** — a pre-existing M2 bug the engine
  exposed. It took Spearman's own p-value as its verdict, so a *small* p (a
  strong monotone association) was reported as FAILING monotonicity: the
  most perfectly monotone data possible got the caveat. The M2 test compared
  only rho and p against R and never asserted a status, which is how it
  survived. It now measures what Section 6.6 always said ("Spearman vs
  lowess shape"): the fraction of the lowess smooth's movement that reverses
  direction. A U shape — strong, perfectly ordered, Spearman rho ≈ 0 — now
  FAILs, which is the case the check exists for.
- **`check_influential_outliers` uses 1.0 / 0.5, not 4/n.** The 4/n rule
  screens which individual points to look at; at any real n some point
  exceeds it by chance, so as a verdict on the dataset it made nearly every
  correlation a caveat — which trains the user to ignore the warning.
- **`check_same_shape` now handles k > 2 groups** (Kruskal-Wallis declares
  it too, and the old code raised). Every pair, Holm-adjusted, worst pair
  decides. Holm rather than the raw minimum: over C(k,2) pairs an unadjusted
  minimum rejects on chance alone. At k=2 Holm is the identity, so two-group
  behaviour and its fixture are unchanged.
- **Evidence vs verdict.** A resolver returns graded checks separately from
  evidence. `check_design_crossing` FAILing is fatal to an
  independent-samples test and is precisely what a paired test needs; the
  first version graded it directly and made every paired test INELIGIBLE on
  paired data. The same split lets the CLT escape in `normality_or_large_n`
  actually replace the Shapiro failure it exists to excuse, rather than
  sitting uselessly beside it.
- **Untestable assumptions are surfaced, never silently passed.**
  Independence with no id column resolves through
  `check_independence_design` to UNTESTABLE and appears on every candidate
  that assumes it, so an ELIGIBLE t-test still shows that independence rests
  on the user's word.
- **Section 7.4's worked example reproduces**: every check lands on the
  status that table gives (Shapiro FAIL/FAIL, descriptive skew BORDERLINE at
  1.89/1.70 against the spec's 1.9/2.2, Levene FAIL, sample size PASS,
  same-shape PASS) and all six candidate statuses match. The statistics
  cannot match — the spec quotes numbers from a dataset it does not ship.
  One documented difference: Section 7.3's `two_independent_numeric` family
  has eight methods, Section 7.4's table shows six; the two it omits
  (`bootstrap_diff`, `ks_two_sample`) are eligible here, asserted separately
  rather than folded in quietly.
- **Spec extensions**, each mirroring something already in Section 7.1:
  `QuestionSpec.reference_value` and `reference_proportions` (a
  DISTRIBUTION_FIT question is asked against a reference, and
  `equivalence_bounds` already did this for EQUIVALENCE); a family-level
  `prepare` hook (a factorial design's homoscedasticity assumption is about
  its cells, not either factor's margin); `check_measurement_level`'s
  `fact_id` now includes the required scale, since the same column is
  legitimately checked against different scales by different candidates and
  those are different facts.
- **`docs/eligibility_table.md` is generated**, not written, by
  `scripts/generate_eligibility_table.py`, and a test fails if the committed
  file and a fresh render disagree. A hand-maintained table would drift the
  first time an assumption changed, and a stale one is worse than none: it
  would be read as a statement about the current code.
- **Scenario generators**: `paired_as_independent`, `heteroscedastic_groups`
  and `heavy_tails_small_n` added to `tests/scenarios/generators.py`
  (Section 15.3), each asserted against the behaviour that table names.
- **One convention worth your decision**: `LARGE_N_FOR_CLT = 100` with
  `MAX_SKEW_FOR_CLT = 2.0` is the threshold above which a normality failure
  is waived for a mean-based test. Section 6.7 says "normality or large n"
  without fixing "large". It is set high deliberately, because Section 7.4
  expects a CAVEAT at n = 38/41; persona policies (Section 8.1) layer their
  own stricter rules on top.

### 2026-09-25 — M3.4 (tag `m3.4`)
Closes M3's acceptance criterion (Section 16: "every test returns effect
size + CI"), which the m3.3 audit found was met by only 9 of 45 test
functions. All 27 gaps are fixed; the 9 exemptions from that audit stand.
No new dependency, no version-floor change.

- **The criterion is now enforced, not audited.**
  `test_m3_effect_size_criterion.py` calls every registered `kind="test"`
  function in stage `hypothesis` and asserts it returns an effect size,
  a named measure, and a CI containing its own point estimate — or is on
  an explicit exemption list with a reason. It fails both ways: a new
  test function with no row in its call table fails, and an *exempt*
  function that starts returning an effect size fails too (its exemption
  would be stale). A companion test asserts the list matches
  `docs/m3_effect_size_audit.md`, which was rewritten to record what each
  test now returns and against which R function it is checked.
- **welch_t reported the wrong number.** Its `effect_size_name` said
  `hedges_g` but the value was Cohen's d(av): the right
  (average-variance) denominator with no J small-sample correction. Now
  `hedges_g_av`, matching `effectsize::hedges_g(pooled_sd=FALSE)` —
  including the detail that both J and the noncentral-t interval use the
  **Welch** df, not n1+n2-2.
- **student_t returned Cohen's d** where Section 6.7 specifies Hedges' g.
  Now g, with the CI `effect_sizes.hedges_g` already computed and had
  verified against R since M2.1 but which nothing was wired to.
- **Six new registered effect sizes** (Section 6.8), each a
  parameterization Section 6.7's tests needed and each verified against a
  new R fixture: `cohens_d_one_sample`, `hedges_g_av`,
  `rank_biserial_one_sample`, `rank_biserial_paired`,
  `cohens_h_one_sample`, `cohens_w_gof`. Two facts worth recording,
  because guessing either would have produced a plausible-looking wrong
  interval: the paired/one-sample rank-biserial SE is
  `sqrt((2n^3+3n^2+n)/6)/(n(n+1)/2)`, *not* the Mann-Whitney SE already in
  the module (a test asserts the two differ on the same data); and a
  goodness-of-fit Cohen's w has upper bound `sqrt(1/min(p) - 1)`, not 1
  and not `cohens_w`'s `sqrt(min(nrow, ncol) - 1)`.
- **Exact intervals where an exact one exists.** `sign_test`,
  `sign_test_paired` and `binomial_test` now use Clopper-Pearson rather
  than a normal approximation; Cohen's h and the McNemar odds ratio are
  strictly monotone transforms of a binomial proportion, so their
  intervals are the transformed Clopper-Pearson bounds — exact, not
  delta-method. base R reports no effect size for `mcnemar.test` at all,
  so that construction (not an R function's output) is what the fixture
  records, and `mcnemar` now returns no interval, with a warning, when
  there are no discordant pairs.
- **Kendall's tau-b CI is deliberately not a Fisher-z interval.**
  `cor.test` reports no CI for tau or rho, so per your decision these
  follow DescTools: `SpearmanRho` (Fisher z, SE = 1/sqrt(n-3)) and
  `KendallTauB`, whose delta-method ASE is computed from the joint
  contingency table and therefore accounts for ties. On the tied fixture
  the two disagree materially; a test asserts that, so the shortcut
  cannot creep back in. DescTools' `ConDisPairs` is O((r*c)^2), so it was
  ported via one 2-D cumulative sum instead. That table is still
  (distinct x) x (distinct y), i.e. n x n for tie-free continuous data, so
  above `MAX_KENDALL_TABLE_CELLS` (4M cells, about n=2000 tie-free) the CI
  degrades to a Fisher-z interval with a warning saying so — a table that
  large has hardly any ties for the delta method to account for, which is
  exactly when the two agree most closely. The estimate and p-value are
  unaffected.
- **Partial eta-squared per term** for `two_way_anova`,
  `aligned_rank_transform_anova` and `repeated_measures_anova`, via
  `effectsize::F_to_eta2` — which is what `effectsize::eta_squared`
  itself computes for `car::Anova`, ARTool and afex objects (checked
  directly, both routes agree). For repeated measures this settles a real
  ambiguity: the sphericity correction changes the **test's** df, not the
  effect size's, so the interval uses the uncorrected F and df while
  `TestResult.df` stays GG-corrected. The fixture records both so the
  Python test asserts the identity rather than assuming it.
- **Two fixtures were added purely because the existing ones were
  vacuous.** `k_groups_unequal_var` has Welch F < 1 and
  `repeated_measures_wide` has RM F < 1, so omega^2 and partial eta^2 are
  both 0 with interval [0, 1] — true, and passable by almost any
  implementation. `k_groups_unequal_var_effect` and
  `repeated_measures_effect_long` have a real effect, so the lower bound
  is strictly positive and the noncentral-F inversion is actually tested.
  Both the null and the effect case are now asserted.
- **tost_equivalence reports at 1 - 2*alpha**, with a new `alpha`
  parameter. That is TOSTER's own convention and the level at which "the
  interval lies inside the bounds" and "both one-sided tests reject" are
  the same statement; reporting a 95% interval next to a 5% TOST decision
  would invite exactly the misreading this tool exists to prevent. Its
  SMD is Hedges' g — pooled or average-variance, following `var_equal`,
  so a Welch-based TOST does not smuggle the equal-variance assumption
  back in through its effect size. TOSTER's own SMD row and
  `effectsize::hedges_g(ci = 1 - 2*alpha)` agree exactly, which is why the
  existing machinery is reused.
- **Permutation tests get a seeded BCa bootstrap CI**, per Section 6.7's
  "bootstrap if no analytic CI". Not R-matchable (resampling draws), so
  what the tests assert is: the interval exists, brackets the observed
  difference, is disclosed in `warnings`, and is reproduced exactly by the
  same seed (rule 7). The paired version resamples **pairs**, never the
  two columns independently.
- **`_ncp_ci` no longer raises on a degenerate sample.** A zero
  within-group SD makes the standardized effect size infinite and its
  interval undefined; it now returns NaN bounds. Before M3.4 nothing
  called it on such data, but wiring `d_z` into `paired_t` did, turning
  "this sample has no variance" into a crash partway through an otherwise
  valid result.
- **Magnitude labels are now explicit both ways.** `UNLABELLED_MEASURES`
  names the effect sizes that deliberately get no label — proportions,
  raw and trimmed mean differences, odds ratios, the Brunner-Munzel
  relative effect, mutual information, dCor — because calling a mean
  difference "medium" without units is meaningless. A test asserts every
  reported measure is either in the threshold table (and carries a label)
  or in that set (and does not). Correlation thresholds use Cohen's r
  conventions; `kendall_tau_b` shares them with a documented caveat that
  tau is systematically smaller than r for the same association, so the
  label understates it.
- **One interval has no independent reference**, and is flagged as such
  in the audit and in the code: no installed R package reports a CI for a
  partial correlation, so the fixture evaluates the same published
  Fisher-z formula in R that Python implements. That checks the
  arithmetic, not the method. The Python test does independently
  establish that the df is n - k - 3 and responds to the number of
  controlled variables.
- **Signature changes** (all additive, all with defaults, so existing
  call sites are unaffected): `ci` added to `wilcoxon_one_sample`,
  `sign_test`, `chi2_goodness_of_fit`, `mann_whitney`, `brunner_munzel`,
  `wilcoxon_signed_rank`, `sign_test_paired`, `welch_anova`,
  `repeated_measures_anova`, `two_way_anova`,
  `aligned_rank_transform_anova`, `spearman`, `kendall_tau`,
  `partial_correlation`, `mcnemar`; `ci_level` to `fisher_exact` (`ci` is
  already its returned interval); `ci` + `n_boot` to both permutation
  tests; `alpha` to `tost_equivalence`.
- **Tests**: 435 passing, up from 298, on Python 3.11. 23 new R fixtures;
  `git diff tests/fixtures` confirmed no existing fixture changed (the
  new R section is appended last and either reuses objects already built
  or sets its own seed).

### 2026-09-24 — M3.3 (tag `m3.3`)
Pre-M4 fixes. Development moved to WSL2 (Python 3.11 venv `.venv-wsl`,
Windows R called from WSL).

- **Line endings**: added `.gitattributes` (`* text=auto eol=lf`). Windows
  git (autocrlf) and the Windows R install write CRLF, which WSL git
  reported as 142 modified files with no content change.
- **Spearman**: implemented R's `C_pRho` (AS 89, Best & Roberts 1975) from
  the published algorithm. That means full enumeration of the null
  distribution of S for n<=9, the Edgeworth series for 9<n<=1290, and R's
  `round(q) + 2*lower_tail` convention. Previously an n!-bounded
  permutation enumeration (n<=8) plus scipy's t approximation. Now matches
  `cor.test` to <=4e-15 relative at n=8, 9, 20, 49, 100 (new fixtures
  `spearman__rank_corr_n{9,20,49,100}`); the old code was off by up to
  1.1% (n=9). `statistic` is now R's S (was rho, duplicating `estimate`),
  per the M3 part 2b statistic-field convention. R's `prho.c` is GPL-2+:
  the implementation follows the published algorithm, not R's C code.
- **Kendall**: exact null distribution of T via
  `scipy.stats.kendalltau(method="exact")` for n<50 without ties,
  reporting `statistic` = T as R does; otherwise z with R's tie-corrected
  variance, and the p-value computed from that z. Matches `cor.test` to
  <=4e-15 at n=9, 20, 49 (exact) and 100 (normal); the old code was off
  by up to 1.7% (n=49).
- **scipy floor 1.13 -> 1.15** (1.15.0: January 2025, >12 months old).
  The r×c Fisher skip and the "needs scipy>=1.15" error path are removed.
  The full fast suite passes against the exact floor pins (numpy 1.26.0,
  pandas 2.2.0, scipy 1.15.0, statsmodels 0.14.0; 297 tests, 0 skipped).
  The `anderson(method=...)` fallback stays (scipy 1.17 is <12 months).
- **Fisher r×c: real bug, and M3 part 2b's diagnosis was wrong.** scipy's
  r×c `fisher_exact` is *not* exact: it defaults to an unseeded
  `MonteCarloMethod` (9999 draws). Repeated calls on the 3×3 fixture gave
  0.0001-0.0005, so the p-value was non-reproducible (rule 7). The old
  "must differ from R" test passed by luck: at p=0.0001 the relative gap
  to R was 1.4e-3, just outside its 1e-3 margin. The "different exact
  conventions" explanation in the m3.2b entry below, and your decision
  based on it, rested on that misreading. Your tolerance hypothesis: R's
  r×c path (FEXACT) uses `tol = 3.45254e-7` on the log path length, i.e.
  P(table) <= P(obs)·exp(3.45e-7). The 1+1e-7 `relErr` is R's 2×2
  branch. Tolerance was not the cause of the 0.0002 vs 0.0001001 gap.
  Fix: `_fisher_rxc_exact_p` enumerates every table with the observed
  margins and applies R's FEXACT tolerance. It matches R to ~1e-12,
  agrees with scipy's own exact `PermutationMethod` on a small table, and
  takes 0.05 s for the fixture (356,481 tables). Above
  `MAX_FISHER_TABLES` (2M) it falls back to a seeded Monte Carlo p-value
  (new `random_state` parameter), disclosed in `warnings`.
- **anderson_ksamp**: scipy 1.15 accepts `method=PermutationMethod`. When
  the asymptotic p-value hits scipy's cap (<=0.001 or >=0.25), it is
  replaced by a seeded 9999-resample permutation p-value (new
  `random_state` parameter), disclosed. Fixture: 1e-4 (the resolution
  floor) instead of 0.001; R's asymptotic value is 1e-13, so it still
  can't match R there. At the upper cap the gain is large: null data
  gave scipy's 0.25 vs a permutation p of 0.92.
- **mutual_information: two real bugs**, found by the new sklearn
  comparison (sklearn now imports locally):
  1. The KSG max-norm neighbourhoods are not scale-invariant. Scaling x
     by 1000 at rho=0.7 took the estimate from 0.35 to 0.05 nats (true:
     0.34). Now each variable is scaled to unit variance, as sklearn does.
  2. Tied values (e.g. Likert data) made a radius <= 0 and returned
     MI = **inf**. Now seeded 1e-10 jitter (new `random_state`), as
     Kraskov et al. and sklearn do, with a warning that KSG assumes
     continuous data; a constant column raises.

  It now matches `sklearn.feature_selection.mutual_info_regression` to
  <=1e-15 (16 CI cases: rho 0-0.95, x scale 1 and 1000, k 3 and 5).
- `chi2_goodness_of_fit`: `expected` proportions not summing to 1 now
  raise a clear error (was an obscure scipy error, found by the audit).
- **M3 acceptance audit** (`docs/m3_effect_size_audit.md`): Section 16's
  "every test returns effect size + CI" is **not met**. 9 of 45 do. Of
  the rest, 7 only need an existing, R-verified CI wired into
  `effect_size_ci`, 20 need a new CI (each with an R reference named), and
  9 have no standard effect size or CI (proposed exempt). Also found:
  `welch_t` labels its effect size `hedges_g` but returns Cohen's d_av
  without the J correction. Remediation is pending your decisions and
  not done in m3.3.

### 2026-09-25 — M3 part 2b (tag `m3.2b`)
Section 6.7's factorial, categorical association, correlation, and
distribution/equivalence/other tests: `two_way_anova`,
`aligned_rank_transform_anova`, `chi2_independence`, `fisher_exact`,
`g_test`, `mcnemar`, `two_proportion_z`, `cochran_armitage_trend`,
`pearson`, `spearman`, `kendall_tau`, `point_biserial`,
`partial_correlation`, `distance_correlation`, `mutual_information`,
`correlation_matrix`, `anderson_ksamp`, `tost_equivalence`, `runs_test`
(19 functions; `two_way_anova`, `aligned_rank_transform_anova` and
`correlation_matrix` return `list[TestResult]`, one per term/pair — each is
its own omnibus test for Section 12.3's ledger rule).

- **New R packages** (6): `ARTool`, `ppcor`, `energy`, `kSamples`,
  `TOSTER`, `randtests` (`car`, `DescTools` already present). Per the
  process rule, the R script was run fully and every pre-existing fixture
  diffed against a pre-edit backup before any Python was written: none
  changed; 18 fixtures + 9 datasets added.
- **Your continuity-correction decision** (match R everywhere):
  `chi2_independence` needed nothing (scipy's `chi2_contingency` already
  defaults to Yates at df=1, same as `chisq.test`); `two_proportion_z`:
  R's `prop.test` for two samples is exactly a Yates-corrected 2x2
  chi-square (read from its source), so statistic/p reuse
  `chi2_contingency(correction=True)` and only the CI (R's
  continuity-corrected Wald interval) is ported; statsmodels has no
  Yates-style option (`test_proportions_2indep`'s `correction` is a
  Miettinen-Nurminen df correction, unrelated). `mcnemar`: corrected by
  default.
- **Real bug found in shipped M3 part 2a code**: `mcnemar_posthoc` (and
  the new `mcnemar`) applied the continuity correction unconditionally.
  R's `mcnemar.test` only corrects when the off-diagonals differ
  (`any(x - t(x) != 0)`); for b01 == b10 the statistic must be exactly 0
  but came out (|0|-1)^2/n != 0. The 2a fixture never had a symmetric
  pair, so it slipped through; caught here by the new fixture. Fixed in
  both places, with a regression test.
- **Type III SS trap**: confirmed on the unbalanced fixture that default
  treatment contrasts give a qualitatively wrong answer (factor_b p=0.034
  vs. 0.0006 with sum-to-zero contrasts). `typ=3` uses `C(x, Sum)`;
  matches `car::Anova(type=3)` with `contr.sum` to ~1e-12. `typ=2` (the
  spec's default) is contrast-invariant and needs no fix, and has no R
  fixture of its own (only Type III was requested as the trap).
- **ART ANOVA**: ported the alignment procedure (align per effect,
  rank, keep only that effect's Type III F row); matches `ARTool::art +
  anova` on all three terms to ~1e-12. Two-factor designs only.
- **Odds ratio**: three different quantities exist (sample ad/bc = 3.5;
  scipy `fisher_exact`'s unconditional MLE; R's conditional MLE 3.4536).
  `fisher_exact` reports the conditional MLE via
  `contingency.odds_ratio(kind="conditional")`, labelled
  `odds_ratio_conditional_mle`. It agrees with R to ~2.5e-5 relative, not
  1e-6: same estimand, two root-finders (scipy's `brentq(xtol=1e-13)` on a
  steep noncentral-hypergeometric mean vs R's). Tested at 1e-4, with the
  reason in the test — a numerical-solver tolerance like the resampling
  carve-out in Section 15.1, not a methodology difference.
- **Fisher r×c — your decision**: scipy's exact p-value (0.0002 here) and
  R's `fisher.test` (0.0001001) are both exact but use different
  "as extreme" conventions; kept scipy's, documented and warned in the
  result. **Floor finding**: r×c needs scipy>=1.15; at the declared floor
  (1.13) `fisher_exact` is 2×2-only (found by running the suite in a
  venv pinned to the exact floor versions before pushing). Rather than
  raise the floor silently, r×c raises a clear "needs scipy>=1.15" error
  there and its test skips. Raising the floor to 1.15 is a one-line
  change if you prefer it.
- **G-test**: `DescTools::GTest` defaults to `correct="none"` (not
  Williams); ported its exact formula, Williams available as an option.
- **Cochran-Armitage**: ported `DescTools::CochranArmitageTest`'s formula
  (integer scores 1..k). The sign depends on which outcome level is the
  "event" (R uses the table's first column), so `event` is explicit.
- **Spearman/Kendall exact p-values**: real gap. scipy's `spearmanr` has no
  exact mode (asymptotic p=0.0039 vs R's exact 0.00724 on the n=8
  fixture, ~2x). Implemented exact mode as full permutation enumeration
  bounded by `MAX_EXACT_PERMUTATIONS` (bit-identical to R at n=8).
  **Limitation, disclosed in results**: that only covers n<=8 (n! grows
  fast), whereas R stays exact to n<1290 (Spearman) / n<50 (Kendall) via
  specialised algorithms — so for n≈9-49 without ties this module falls
  back to the asymptotic p-value and may differ from R; a warning says so.
  Kendall with ties already matched R via scipy's `method="auto"`.
- **Statistic-field convention fix**: `pearson`/`point_biserial` initially
  put r in `statistic` (scipy's `.statistic`) while R's, and this
  codebase's, `statistic` is the test statistic that drives the p-value;
  now `t`, with r in `estimate`. `kendall_tau`'s `statistic` is R's
  tie-corrected z (ported from `cor.test`'s source), tau in `estimate`.
- **Partial correlation** ported from the precision-matrix formulation
  (general in the number of covariates), ~1e-9 vs `ppcor::pcor.test`.
  **Distance correlation** statistic matches `energy::dcor` to ~1e-9; its
  permutation p-value cannot match R (RNG) and is tested by simulation
  (tiny under y=x², rarely significant under independence).
- **Mutual information**: no R reference. Implemented a KSG k-NN
  estimator on `scipy.spatial.cKDTree` rather than
  `sklearn.feature_selection.mutual_info_regression`, which fails to
  import in this environment (Application Control blocks
  `_expected_mutual_info_fast` — the same local-only class of problem as
  mypy's DLL earlier), so it could not be verified. Tested against
  -0.5*ln(1-rho^2) at rho 0.3/0.7, n=1500, tolerance 0.05 nats (measured
  |bias| <= ~0.015 at n=3000; the tolerance is far smaller than the
  effect being distinguished).
- **anderson_ksamp**: scipy's statistic equals R's standardized T.AD
  (version 2); R's fixture is stored rounded to 3 decimals so it's
  compared at 1e-3. scipy's p-value is floored at 0.001 (R: 1e-13), and
  capped at 0.25 above; both are disclosed in `warnings`.
- **TOST**: bounds are RAW (outcome units), matching TOSTER's default
  `eqbound_type="raw"`; stated in the result and docstring. p-value is
  max of the two one-sided p-values. **Runs test**: median dichotomization
  (values equal to the median dropped) and normal approximation, matching
  `randtests::runs.test` defaults. Its fixture happens to land at exactly
  z=0, so directionality is additionally tested on clustered/alternating
  sequences.

### 2026-09-25 — M3.2a.1 (tag `m3.2a.1`)
Pre-work for M3 part 2b, no new stattests functions:

- **TestLedger rule** (Section 12.3): an omnibus test enters the session
  ledger as exactly one entry, using its own `p_value`; a post-hoc
  procedure's `PostHocResult` is adjusted within its own family only
  (labelled with `p_adjust_method` and `len(comparisons)`), never pooled
  into the session-wide adjustment, and never adds to the session test
  count. `TestLedger` itself belongs to the not-yet-built `edacopilot`
  session layer, so "implement and test" scoped to what `edacore` is
  responsible for right now: `tests/unit/edacore/test_ledger_rules.py`
  locks in that every omnibus `TestResult.p_adjusted` stays unset (never
  self-computed) and every registered post-hoc function's
  `PostHocResult` is correctly labelled and within-family-adjusted, for
  all omnibus/post-hoc functions built so far.
- `PostHocResult`/`PairwiseComparison` in Section 5: already added during
  M3 part 2a: no additional change needed, confirmed still present.
- **Process rule** (Section 15.1 and `generate_r_fixtures.R`'s own header
  comment): after any edit to the R fixture script, run it fully and diff
  every existing fixture against its pre-edit state before writing any
  Python — codifying the discipline that caught the wrong-R-object-name
  bug twice (M3 part 1, M3 part 2a).

### 2026-09-25 — M3 part 2a (tag `m3.2a`)
Section 6.7's k-independent-group, k-related-group, and post-hoc tests:
`one_way_anova`, `welch_anova`, `alexander_govern`, `kruskal_wallis`,
`permutation_anova`, `repeated_measures_anova`, `friedman`, `cochran_q`,
and post-hoc procedures `tukey_hsd`, `games_howell`, `dunn_test`,
`permutation_posthoc`, `paired_posthoc`, `nemenyi_friedman`,
`conover_friedman`, `mcnemar_posthoc` (16 functions).

- **New contract type**: `PostHocResult`/`PairwiseComparison` (Section 5)
  — a post-hoc procedure produces multiple pairwise comparisons per call,
  which doesn't fit `TestResult`'s one-result shape. Added the same way
  `DuplicateReport`/`StructureReport`/`LeakageFlag` were for M1. New
  `FunctionKind` literal `"posthoc"` alongside `test`/`effect`/etc.
- **R packages** (5 new, none previously installed): `PMCMRplus`
  (Games-Howell + Friedman post-hoc, one dependency instead of two since
  you named it for Friedman with no alternative), `FSA` (Dunn's test —
  installed `dunn.test` too, purely to compare defaults empirically before
  choosing; `dunn.test` isn't a project dependency), `afex` (repeated-
  measures ANOVA with GG/HF correction), `DescTools` (Cochran's Q),
  `onewaytests` (Alexander-Govern). All installed and used at their own
  package defaults unless noted below.
- **`generate_r_fixtures.R` had the same class of bug as M3 part 1** twice
  over: referenced `two_groups_equal_var`/`two_groups_unequal_var` (CSV
  filenames) instead of the actual R variables `group_equal_var`/
  `group_unequal_var`; fixed all six call sites. Also hit two real
  R-package requirements the docs don't make obvious: `PMCMRplus::
  gamesHowellTest` requires its group column to be an actual factor (a
  character column fails with "all group levels must be finite", not a
  type error); `PMCMRplus::frdAllPairsNemenyiTest`/`frdAllPairsConoverTest`
  require explicit block (subject) rownames on the input matrix or they
  fail with "number of levels differs". Reran end-to-end; diffed every
  pre-existing fixture byte-for-byte against a backup taken before the
  rerun — nothing from M2/M2.1/M2.2/M3-part-1 changed, only the 14 new
  fixtures were added.
- **Convention choices made and disclosed** (per your "say which"/"state
  which" instructions, not asked as questions since each was a named,
  delegated choice):
  - Dunn's test: **FSA::dunnTest**, matched at its own default (two-sided
    p-values, Holm-adjusted) — `dunn.test`'s default is one-sided,
    unadjusted, confirmed by running both on the same data (FSA's
    unadjusted p is exactly double dunn.test's reported p). FSA's design
    (a `method` argument for the adjustment) also matches Section 6.7's
    own "`dunn_test` (with p-adjust)" framing directly.
  - `repeated_measures_anova` matches **afex::aov_ez's default** output
    exactly: primary statistic/df/p_value are Greenhouse-Geisser-corrected
    (afex's own default), with the uncorrected values, Huynh-Feldt
    correction (capped at epsilon=1 when >1, matching R's own behavior),
    and Mauchly's sphericity test (reusing `check_sphericity_mauchly` from
    M2.1) carried in `validity_notes`. GG/HF epsilon are computed by
    reading `car:::summary.Anova.mlm`'s source directly (same one that
    backs afex) and porting its exact formula — eigenvalues of the
    covariance matrix of Helmert-contrast-transformed scores, reusing
    `assumptions._helmert_contrasts`, the same orthonormal basis already
    verified for Mauchly's W.
  - `one_way_anova`'s two listed effect sizes (η², ω²) don't fit
    `TestResult`'s one `effect_size` slot the way mann_whitney's
    rank-biserial/Cliff's-δ pair did in M3 part 1 (those are the *same*
    number under two names; η² and ω² are genuinely different formulas).
    Kept η² as the primary `effect_size` (first-listed, most commonly
    requested by name) and put ω²'s value in `validity_notes`.
  - `welch_anova`'s ω² is **not** the classical SS-based formula
    `edacore.effect_sizes.omega_squared` uses for `one_way_anova` —
    confirmed by reading `effectsize::omega_squared`'s source: for a
    Welch `htest` object it dispatches to the F-based approximation
    `max(0, ((F-1)*df1)/(F*df1+df2+1))` using the Welch F/df directly, a
    genuinely different number from the classical formula on the same
    data (own module docstring has the reasoning).
  - `yuen_trimmed_t`'s precedent (M3 part 1: keep a signed statistic,
    document R's unsigned convention) repeated for `alexander_govern` and
    is **not** needed for `tukey_hsd`'s statistic, since R's own
    `TukeyHSD` already reports a signed mean difference.
- **`conover_friedman` real discrepancy, found and resolved**:
  scikit-posthocs' `posthoc_conover_friedman` does *not* match
  `PMCMRplus::frdAllPairsConoverTest`'s default at its own default
  (`p_adjust=None`) — confirmed the gap isn't explainable by any
  monotonic p-value adjustment (scikit-posthocs' raw p for one pair was
  0.742, R's default was 0.988; no adjustment method only ever increases
  p-values enough to bridge that from the raw value cleanly the way it
  did for other pairs). Root cause: scikit-posthocs' own `p_adjust`
  parameter accepts a `'single-step'` option (studentized-range-based,
  distinct from a post-hoc p.adjust() layer) that exactly reproduces
  PMCMRplus's default once selected — not a formula difference, a missed
  parameter. Documented in `conover_friedman`'s docstring; not asked about
  since it was a diagnosable bug (a parameter to find), not a genuine
  convention choice.
- **Real bug found and fixed while testing** (not an R discrepancy — a
  self-inconsistency within the Python code): `tukey_hsd`'s
  `group_a`/`group_b` labels didn't match what `estimate` (the mean
  difference) represented — statsmodels' `meandiffs` is
  `mean(group_t) - mean(group_c)`, but the comparison was labeled
  `group_a=group_c, group_b=group_t`, making `estimate` silently equal
  `mean(group_b) - mean(group_a)` instead of the reverse. A sanity test
  (`estimate == means[group_a] - means[group_b]`) caught it; fixed by
  swapping which side maps to `group_a`. Also gave `tukey_hsd` a
  `PairwiseComparison.ci` (lower/upper) it hadn't had at all, and added
  `brunner_munzel`-style missing-field coverage: `permutation_posthoc`
  reports which pairs fell back to Monte Carlo, mirroring M3 part 1's
  `permutation_test_2s`/`permutation_test_paired`.
- `cochran_q` needed no separate formula: `DescTools::CochranQTest`
  delegates directly to `stats::friedman.test` (confirmed by reading its
  source) — Cochran's Q is exactly Friedman's test applied to binary
  data, so `cochran_q` calls the same `scipy.stats.friedmanchisquare`
  path as `friedman`.
- **`test-min-versions` CI failure, root-caused and fixed**: `tukey_hsd`
  used `TukeyHSDResults.group_t`/`.group_c`, which don't exist at all on
  statsmodels 0.14.0 (this project's declared floor) — confirmed by
  installing exactly that pinned version in a throwaway venv and
  reproducing the `AttributeError` directly, then rerunning the full fast
  suite against it after the fix. Replaced with
  `itertools.combinations(result.groupsunique, 2)`, which `meandiffs`/
  `pvalues`/`confint` are documented to follow (it's the same order the
  printed summary table uses) and is stable across the version range.

### 2026-09-24 — M3.1.1 (tag `m3.1.1`)
CI restructuring, ahead of M3 part 2a:

- `@pytest.mark.slow` tests (coverage simulations) moved out of push/PR CI
  entirely, into a new `.github/workflows/nightly.yml` (`cron` daily +
  `workflow_dispatch` for on-demand runs). Push/PR CI now runs fast tests
  only on every leg.
- `mypy src/` moved out of the `test` matrix (5 legs) into its own `lint`
  job: Ubuntu, Python 3.12, numpy pinned to 2.5.3. This directly addresses
  the failure mode from M3 part 1's CI-fix round — mypy's inferred types
  disagreed between numpy 2.4.6 (resolved for the 3.11 legs) and 2.5.x
  (resolved for 3.12), producing false-positive `no-any-return` errors
  that had nothing to do with the code changing. Running mypy once, with a
  pinned numpy, makes its pass/fail deterministic regardless of which
  numpy version each matrix leg's own dependency resolution happens to
  pick.

### 2026-09-24 — M3 part 1 (tag `m3.1`)
Section 6.7's one-sample, two-independent-group, and two-paired-group
hypothesis tests: `one_sample_t`, `wilcoxon_one_sample`, `sign_test`,
`binomial_test`, `chi2_goodness_of_fit`, `bootstrap_one_sample`,
`student_t`, `welch_t`, `yuen_trimmed_t`, `mann_whitney`, `brunner_munzel`,
`permutation_test_2s`, `bootstrap_diff`, `ks_two_sample`, `paired_t`,
`wilcoxon_signed_rank`, `sign_test_paired`, `permutation_test_paired`.

- `generate_r_fixtures.R` had a real bug blocking every two-sample/paired
  fixture: the M3 section referenced R objects `two_groups_equal_var` /
  `two_groups_unequal_var` (the *CSV filenames*), but those data frames
  were actually assigned to `group_equal_var` / `group_unequal_var`
  earlier in the script. Fixed all six call sites; reran end-to-end;
  diffed every pre-existing fixture byte-for-byte against a backup taken
  before the rerun — nothing from M2/M2.1/M2.2 changed, only the 16 new
  M3 fixtures and the two new small-n permutation datasets were added.
- Permutation tests (`permutation_test_2s`, `permutation_test_paired`) use
  exact full enumeration (`scipy.stats.permutation_test(...,
  n_resamples=np.inf)`) when the number of distinct arrangements is
  ≤100,000, Monte Carlo above that; the mode used is reported in
  `TestResult.warnings`. Exact mode verified against R's own
  full-enumeration fixtures (`combn`/`expand.grid`, no package) and,
  independently, against a hand-rolled `itertools` enumeration.
- **Four real R/Python default discrepancies found and resolved per the
  maintainer's explicit choice** (not silently patched):
  - *Wilcoxon statistic* (`wilcoxon_one_sample`, `wilcoxon_signed_rank`):
    scipy's `wilcoxon()` always reports `min(W+, W-)`; R's `wilcox.test()`
    always reports `V = W+` (sum of positive-difference ranks). Chose to
    match R's `V` convention.
  - *Wilcoxon/Mann-Whitney p-value method selection*: scipy's `'auto'`
    mode and R's default exact/asymptotic threshold (and R's default
    continuity correction) disagree for some n, giving different
    p-values for identical data. Chose to replicate R's threshold rule
    (exact when n<50 and no ties, else continuity-corrected normal
    approximation) rather than keep scipy's own `'auto'` choice — now
    bit-matches R on every fixture.
  - *`yuen_trimmed_t` statistic sign*: read `WRS2::yuen`'s source
    directly (`print(WRS2::yuen)`) — it hard-codes
    `test <- abs(dif/sqrt(q1+q2))`, an unsigned statistic. Chose to keep
    a signed statistic (consistent with `estimate`'s sign and with how
    `student_t`/`welch_t`/`paired_t` report signed statistics elsewhere
    in this module); magnitude matches R exactly, sign is the documented,
    intentional difference. Also added the CI `yuen_trimmed_t` was
    missing entirely (R's `WRS2::yuen` hard-codes a 95% CI regardless of
    any `alpha`/`ci` argument passed; this implementation takes a `ci`
    parameter properly).
  - `brunner_munzel` was also missing `TestResult.df` (scipy's result
    object doesn't expose it) and had a wrong, invented formula for its
    `estimate`. Fixed by replicating scipy's own internal Satterthwaite-df
    formula (read from `scipy/stats/_stats_py.py`) and correcting the
    relative-effect formula to `(mean rank of y - mean rank of x)/N +
    0.5`; both now match the R fixture.
- One real bug found and fixed while writing tests (not a discrepancy —
  self-inconsistent within Python alone): `one_sample_t`'s CI was on the
  wrong scale. scipy's `ttest_1samp(x, mu0).confidence_interval()` is the
  CI for the raw sample mean (independent of `mu0`), but `estimate` was
  `mean - mu0`. Fixed by shifting the CI bounds by `-mu0` so both are on
  the same scale as `estimate`.
- `kendalls_w` now adds a runtime warning to `TestResult.warnings` (not
  just the docstring) when its CI is computed from fewer than 50 subjects,
  carrying the same coverage figures as the M2.2 docstring note.

### 2026-09-24 — M2.2 (tag `m2.2`)
Resolves the one open item M2.1 left: `kendalls_w`'s BCa coverage came in
at 91.4% against a [93%, 97%] target. Diagnosed rather than assumed either
"bug" or "fine":

- True W used in the coverage simulation is 0.31-0.33 throughout (a
  200x-oversized Monte Carlo sample of the same data-generating process) —
  not near 0 or 1, so the simulation's design wasn't the problem.
- Ran 2000 trials at n=25/50/100 subjects (same DGP, true W stable): 91.1%
  / 93.3% / 94.5%. Coverage climbs monotonically toward nominal as n
  grows and is already inside the target band at n=100 — the signature of
  a finite-sample BCa property, not a bug (a bug would typically produce
  a constant bias or non-monotonic behavior regardless of n, not a clean
  convergence curve).
- Per the maintainer's own decision rule for this outcome: moved the
  committed coverage test to n=100 (still passes: epsilon_squared and
  kendalls_w both green in the same run) and added a small-n validity
  note to `kendalls_w`'s docstring — its CI is narrower than its stated
  confidence level for repeated-measures studies with well under 100
  subjects; the point estimate itself isn't affected.
- CI: coverage-simulation tests are marked `slow` and now run on exactly
  one leg (`ubuntu-latest` / Python 3.12), excluded from every other
  matrix leg and from `test-min-versions` — they're deterministic
  (fixed seeds), so running them on every OS/Python combination bought
  nothing but wall-clock time. New per-leg CI times below M2.1's, once
  this run's numbers are in.
- `generate_r_fixtures.R`: replaced the two permutation-test fixtures'
  Monte Carlo shuffle (10000 random draws, no exact match possible against
  Python's own RNG) with **exact, full-enumeration** permutation tests on
  small dedicated datasets — 6 vs 6 (all C(12,6)=924 group-assignments) for
  the two-sample case, 6 pairs (all 2^6=64 sign-flip patterns) for the
  paired case. Both use only base R (`combn`, `expand.grid`) — no new
  package, and no more RNG-mismatch caveat needed for these two fixtures.

### 2026-09-24 — M2.1 (tag `m2.1`)
Closes out M2's three real discrepancies and one solver-tolerance gap by
reading `effectsize`'s actual R source rather than guessing at its methods
— several M2 assumptions about which functions bootstrap turned out wrong.

- **check_sphericity_mauchly**: ported R's exact two-term (Box-corrected)
  p-value approximation (`stats:::mauchly.test.SSD`) instead of the
  leading-term-only approximation M2 used. Now matches R to ~1e-14, not
  ~1.4% off. `check_normality_anderson` and `check_normality_lilliefors`
  keep their scipy/statsmodels defaults (maintainer's call — both are
  within a reasonable 12-month-freshness bump of a newer scipy floor, see
  below, so not a "fix this properly" gap the way Mauchly was).
- **Noncentral-F CIs** (eta_squared, partial_eta_squared, omega_squared):
  ported effectsize's actual `.get_ncp_F` + eta2 back-conversion — a
  pseudo-F is reconstructed from the point estimate, inverted against
  noncentral F, one-sided (ci_high fixed at 1). Matches R to ~1e-9.
- **Noncentral chi-square CIs** (cramers_v, phi, cohens_w): ported
  `.get_ncp_chi` the same way, including cramers_v's bias-correction and
  k/l adjustment and cohens_w's different ci_high
  (`sqrt(min(nrow,ncol)-1)`, not 1). cramers_v/phi verified to ~1e-8
  against R; cohens_w's formula is implemented but unverified — the M2
  fixture never captured its CI, so there's nothing to check it against
  yet (added to the R script, pending a rerun).
- **rank_biserial, cliffs_delta**: M2 assumed these bootstrap in R. They
  don't — `effectsize::rank_biserial`'s CI is an exact closed form
  (Fisher-z transform, analytic SE), and `cliffs_delta` just calls
  `rank_biserial` internally. Ported the exact formula; matches R to
  ~1e-13. No more bootstrap here at all.
- **kendalls_w, epsilon_squared**: M2 also assumed noncentral-F for
  epsilon_squared specifically (grouping it with eta2/omega2) — wrong too:
  `rank_epsilon_squared`'s formula-interface path is a plain percentile
  bootstrap resampling *within each group* (R's `boot::boot.ci(type=
  "perc")`, R=200), and `kendalls_w`'s is the same but resampling whole
  subjects. Both now use BCa instead of R's plain percentile (per
  instruction) — no fixture to bit-match either way, since a bootstrap's
  specific draws are inherently RNG-dependent. Validated by simulated
  coverage instead (500 trials, `test_effect_size_ci_coverage.py`, marked
  slow): epsilon_squared lands inside the specified [93%, 97%] band;
  **kendalls_w came in at 91.4%**, below it. This is a known finite-sample
  property of BCa at moderate n (25 subjects here), not a caught bug — but
  it wasn't silently loosened to pass; flagged for the maintainer to
  decide (widen the tolerance, grow the test's sample size, or investigate
  further).
- **Performance**: the initial BCa implementation rebuilt a pandas
  DataFrame (concat + relabel) on every one of 2000 bootstrap iterations
  plus jackknife — a single kendalls_w call took 96.7s. Rewrote both
  bootstraps to resample numpy arrays directly (no DataFrame in the hot
  loop): same call now takes 0.39s, ~250x faster, bit-identical results.
  This wasn't cosmetic — at the original speed the full test suite would
  have taken over an hour.
- **scipy floor for `anderson(method=...)`**: traced to scipy 1.17.0
  (released 2026-01-10, ~8.5 months before this entry) via direct
  wheel-by-wheel testing in a real Python 3.11 environment (see below) —
  within the 12-month freshness threshold, so per instruction this was a
  question for the maintainer, not an automatic bump. Decision: keep the
  floor at scipy>=1.13 and the hand-written fallback (verified
  bit-identical to the modern API).
- **Local verification environment**: installed Python 3.11.9 (this
  machine's default is 3.13, for which several declared-minimum package
  versions have no prebuilt wheels — e.g. scipy 1.14.0 tries to compile
  from source and fails with no C compiler available) at
  `.venv311/` (gitignored). Used it to determine the scipy floor above by
  installing actual historical scipy releases, not guessing from
  changelogs.
- `generate_r_fixtures.R`: `write_fixture()` now recursively rejects any
  NULL or NA value before writing, with a message naming the exact path
  (e.g. `cliffs_delta__two_groups_equal_var.estimate`) — this is the
  validation that would have caught M2's `cliffs_delta`/
  `partial_eta_squared` silent-NULL bugs immediately instead of needing
  manual inspection to find them.

### 2026-09-24 — M2 (tag `m2`)
- `edacore.assumptions`: all 23 Section 6.6 checks. 19 have a computable
  statistic and are verified against R fixtures; `check_sample_size`,
  `check_independence_design`, `check_paired_structure`,
  `check_measurement_level` are deterministic (no statistic), so have no
  R fixture and are covered by table-driven tests instead.
- `edacore.effect_sizes`: all 18 Section 6.8 effect-size functions, plus
  `magnitude_label` and `bootstrap_effect_ci` (unregistered helpers).
  `cohens_d`, `hedges_g`, `glass_delta`, `d_z`, `cohens_h`, `odds_ratio`,
  `risk_ratio`, `risk_difference` match R's point estimate *and* CI
  exactly (noncentral-t inversion / log-scale Wald / arcsine, whichever
  `effectsize` uses). The other 10 match R's point estimate but use a
  percentile bootstrap CI instead of `effectsize`'s noncentral-F /
  specialized asymptotic CIs — see "Known discrepancies" below.
- `edacore.multiplicity.adjust_pvalues`: matches R's `p.adjust` to float
  precision for holm/bonferroni/fdr_bh/fdr_by.
- `edacore.power`: `required_sample_size`, `minimum_detectable_effect`,
  ported directly from `pwr` package's R source (not statsmodels' `power`
  classes, which parameterize ANOVA differently and don't cover
  `pwr.r.test`'s bias-corrected correlation formula at all).
- Registry: added `takes_df=False` for functions that don't operate on a
  dataset (`adjust_pvalues`, the two power functions) — Section 6.8 is the
  first place a function's first argument legitimately isn't `df`.
  `registry.register`'s decorator is now typed `F -> F` (was the erased
  `Callable[..., Any]`), so a decorated function forwarding its result to
  another decorated function (e.g. `cliffs_delta` -> `rank_biserial`)
  keeps its real return type under mypy strict.
- `codegen.render_call`'s `"df"` key reservation is now conditional on
  `df_var` actually being injected — a `takes_df=False` function can have
  a genuine parameter named `df` (power's chi-square degrees of freedom)
  without colliding with the dataframe-variable convention.
- R fixtures regenerated after fixing three bugs in
  `generate_r_fixtures.R` itself: `effectsize::riskdifference()` doesn't
  exist (switched to `stats::prop.test`); `cohens_h`/`oddsratio`/
  `riskratio` need the 2x2 table transposed to read "columns = compared
  groups" (odds ratio is transpose-invariant so this was silently
  masked there, but risk ratio was computing the wrong quantity);
  `cliffs_delta`/`partial_eta_squared` were reading nonexistent R column
  names via silent `NULL` (`$` partial-matching happened to save
  `Glass_delta` but not these two).

**Known discrepancies (point estimates match; these don't):**
| Function | What differs | Size |
|---|---|---|
| `check_normality_anderson` | p-value: scipy `method="interpolate"` table vs R's `nortest::ad.test` table | ~6e-4 |
| `check_normality_lilliefors` | p-value: statsmodels' approximation vs R's `nortest::lillie.test` table | ~1.1e-2 |
| `check_sphericity_mauchly` | p-value: this module's chi-square approximation constant vs R's (W itself matches to 1e-9) | ~1.4% relative |
| `rank_biserial`, `cliffs_delta`, `eta_squared`, `partial_eta_squared`, `omega_squared`, `epsilon_squared`, `kendalls_w`, `cramers_v`, `phi`, `cohens_w` | CI: percentile bootstrap (2000 resamples) here vs `effectsize`'s noncentral-F / specialized asymptotic CI | CI width/bounds differ; not systematically checked against R |
| `required_sample_size` (all 4 test families) | n: R's `uniroot` default tolerance (~1.2e-4 relative) is looser than this module's root-finder, so R's own reference n doesn't evaluate to exactly the target power | 1e-6 to 4e-6 |

Open question for the maintainer: for the four p-value/CI discrepancies
(not the `uniroot`-tolerance one, which is just solver precision), should
this module keep its current defaults (standard scipy/statsmodels methods,
clearly documented), or is exact bit-for-bit agreement with R's specific
table/approximation choice required? The latter means porting R's
reference tables/constants for each one individually.

### 2026-09-24 — M1.1 (tag `m1.1`)
- Reconciled `registry.list(stage="profile")` against Section 6.1: all 17
  functions present, nothing missing.
- Added false-positive tests: a clean dataset produces zero flags; a wide
  integer range (age 18-65) isn't flagged ordinal; genuine zeros aren't
  flagged as sentinels; a strong-but-noisy predictor (r≈0.87) isn't flagged
  as leakage.
- Added `detect_structure` scenario tests for all three repeating-entity
  shapes: a single time series, panel data, and repeated IDs across groups
  with no time index (repeated_measures).
- Added a CI leg pinning the declared dependency floors (`pandas==2.2.0`,
  `numpy==1.26.0`, `scipy==1.13.0`, `statsmodels==0.14.0`) rather than only
  ever testing against latest.
- Added `numpy` as an explicit dependency (was only transitive) with the
  1.26.0 floor pandas 2.2.0 itself requires on Python >=3.12.
- Set mypy's `python_version` to 3.12 (the actual minimum that parses
  numpy's PEP 695 stub syntax; was overcautiously set to 3.13 in M1).
- Documented `DuplicateReport`, `StructureReport`, `LeakageFlag` in
  Section 5.3 — Section 6.1 named them as return types but Section 5 never
  defined them.

### 2026-09-24 — M1 (tag `m1`)
- `edacore.profiling`: all 17 Section 6.1 functions implemented and
  registered with code templates (`profile_dataset`, `infer_semantic_types`,
  eight `detect_*` functions, four `summarize_*` functions).
- Trap scenarios `ordinal_as_numeric`, `sentinel_missing`, `target_leakage`,
  `duplicate_rows`, `pii_present` (Section 15.3) detected, each via a
  seeded generator in `tests/scenarios/generators.py`.
- Fixed for pandas 3.0's new default string dtype, which silently broke
  every `dtype == object` check.

### 2026-09-24 — M0 gap-closing (tag `m0`, same commit range)
- Added `tests/helpers/codegen_check.py` (`assert_codegen_matches`): execs
  `registry.to_code(...)` against a fixture DataFrame and checks it equals
  calling the function directly.
- Added `model_dump`/`model_validate` round-trip tests for every Section 5
  model and enum.
- Confirmed `registry.json_schema()` and `registry.list(stage, kind, tags)`
  with combined-filter tests.
- Added a `pip-licenses --fail-on="GPL"` CI step (rule 8: no GPL-family
  dependencies).

### 2026-09-24 — M0 (tag `m0`)
- Repo scaffolding: `src/edacore` + `src/edacopilot` layout, pyproject with
  extras, CI (ruff, mypy, pytest), pre-commit, Apache-2.0 license.
- `edacore.contracts`: the nine Section 5 models/enums, frozen.
- `edacore.registry`: `@register`, `FunctionSpec`, `get`/`list`/
  `json_schema`/`to_code`.
- `edacore.codegen`: `render_call()`.
