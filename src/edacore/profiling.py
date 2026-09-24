"""Dataset profiling (ARCHITECTURE.md, Section 6.1) — stage `profile`.

Every function here is read-only and registered so it can be reproduced as
exported code (rule 7). `profile_dataset` composes the detectors and
summarizers below into a single `DatasetProfile`.

Semantic-type inference and the trap detectors (sentinels, numeric-coded
categoricals, ordinal candidates, structure, leakage, PII) are heuristics —
there is no ground truth to compute them from, unlike the statistical tests
in later sections. Each heuristic is documented at its definition.
"""

from __future__ import annotations

import re
from typing import Any, Literal

import numpy as np
import pandas as pd

from edacore.contracts import (
    ColumnProfile,
    DatasetProfile,
    DuplicateReport,
    LeakageFlag,
    SemanticType,
    StructureReport,
)
from edacore.registry import register

DEFAULT_SENTINEL_CANDIDATES: tuple[Any, ...] = (
    -999,
    -99,
    9999,
    "NA",
    "N/A",
    "null",
    "",
    ".",
    "?",
)

# Common ordered vocabularies used to recognise text columns that are really
# ordinal (Likert-style) rather than nominal.
_ORDINAL_VOCABULARIES: tuple[tuple[str, ...], ...] = (
    ("low", "medium", "high"),
    ("low", "med", "high"),
    ("small", "medium", "large"),
    ("s", "m", "l", "xl"),
    ("s", "m", "l", "xl", "xxl"),
    ("strongly disagree", "disagree", "neutral", "agree", "strongly agree"),
    ("strongly disagree", "disagree", "neither agree nor disagree", "agree", "strongly agree"),
    ("never", "rarely", "sometimes", "often", "always"),
    ("poor", "fair", "good", "very good", "excellent"),
    ("beginner", "intermediate", "advanced", "expert"),
)

_ID_NAME_RE = re.compile(r"(^|_)(id|uuid|guid|key)($|_)", re.IGNORECASE)
_WEIGHT_NAME_RE = re.compile(
    r"^(weight|wt|pw|sample_weight|sampling_weight|design_weight)$", re.IGNORECASE
)
_TIME_NAME_RE = re.compile(r"(date|time|timestamp|datetime)", re.IGNORECASE)
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE_RE = re.compile(r"^\+?[\d][\d\-.\s()]{7,}\d$")
_SSN_RE = re.compile(r"^\d{3}-\d{2}-\d{4}$")

_StructureLiteral = Literal[
    "cross_sectional", "time_series", "panel", "repeated_measures", "unknown"
]


def _is_integer_like(non_null: pd.Series) -> bool:
    if non_null.empty:
        return False
    numeric = pd.to_numeric(non_null, errors="coerce")
    if numeric.isna().any():
        return False
    return bool((numeric == numeric.round()).all())


def _is_stringlike_dtype(series: pd.Series) -> bool:
    """True for legacy object-dtype-of-strings and pandas 3's default `str` dtype.

    A genuinely mixed object column (e.g. `[1, "a", 2.5]`) is object dtype but
    `is_string_dtype` is False for it, so this deliberately excludes it —
    detect_mixed_types below is what's supposed to catch that case.
    """
    return pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)


def _is_likert_like(non_null_numeric: pd.Series) -> bool:
    values = sorted(int(v) for v in non_null_numeric.unique())
    if not (3 <= len(values) <= 9):
        return False
    if values[0] not in (0, 1):
        return False
    return values == list(range(values[0], values[0] + len(values)))


def _looks_like_free_text(sample: pd.Series) -> bool:
    if sample.empty:
        return False
    word_counts = sample.str.split().map(len, na_action="ignore").fillna(0)
    avg_words = float(word_counts.mean())
    uniqueness = sample.nunique() / len(sample)
    return avg_words >= 4 and uniqueness >= 0.5


def _matches_ordinal_vocabulary(sample: pd.Series) -> bool:
    normalized = {str(v).strip().lower() for v in sample.unique()}
    return any(normalized <= set(vocab) for vocab in _ORDINAL_VOCABULARIES)


def _infer_column_semantic_type(df: pd.DataFrame, col: str) -> tuple[SemanticType, float]:
    series = df[col]
    n = len(series)
    non_null = series.dropna()
    n_unique = int(non_null.nunique())

    if n_unique <= 1:
        return SemanticType.CONSTANT, 1.0

    if pd.api.types.is_datetime64_any_dtype(series):
        return SemanticType.DATETIME, 1.0

    is_all_unique = n_unique == n and n > 1
    if is_all_unique and (pd.api.types.is_integer_dtype(series) or _is_stringlike_dtype(series)):
        if _ID_NAME_RE.search(str(col)):
            return SemanticType.IDENTIFIER, 0.9
        if pd.api.types.is_integer_dtype(series):
            # A scattered set of unique integers (e.g. ages in a small sample)
            # isn't an identifier; a dense auto-increment-style run is.
            sorted_vals = sorted(int(v) for v in non_null.unique())
            if sorted_vals == list(range(sorted_vals[0], sorted_vals[0] + len(sorted_vals))):
                return SemanticType.IDENTIFIER, 0.6

    if n_unique == 2:
        return SemanticType.BINARY, 1.0

    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        if not _is_integer_like(non_null):
            return SemanticType.CONTINUOUS, 0.9
        if _is_likert_like(non_null):
            return SemanticType.ORDINAL, 0.75
        if n_unique <= 15:
            return SemanticType.NOMINAL, 0.6
        return SemanticType.DISCRETE, 0.85

    if _is_stringlike_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
        sample = non_null.astype(str)
        if _looks_like_free_text(sample):
            return SemanticType.TEXT, 0.75
        types_seen = {type(v) for v in non_null}
        if len(types_seen) > 1:
            return SemanticType.MIXED, 0.7
        if _matches_ordinal_vocabulary(sample):
            return SemanticType.ORDINAL, 0.85
        return SemanticType.NOMINAL, 0.7

    return SemanticType.MIXED, 0.3


@register(
    name="infer_semantic_types",
    kind="profile",
    stage="profile",
    code_template="edacore.profiling.infer_semantic_types({df})",
)
def infer_semantic_types(df: pd.DataFrame) -> dict[str, tuple[SemanticType, float]]:
    return {col: _infer_column_semantic_type(df, col) for col in df.columns}


@register(
    name="detect_identifier_columns",
    kind="detect",
    stage="profile",
    code_template="edacore.profiling.detect_identifier_columns({df})",
)
def detect_identifier_columns(df: pd.DataFrame) -> list[str]:
    types = infer_semantic_types(df)
    return [
        col for col, (semantic_type, _) in types.items() if semantic_type is SemanticType.IDENTIFIER
    ]


@register(
    name="detect_numeric_coded_categoricals",
    kind="detect",
    stage="profile",
    code_template=(
        "edacore.profiling.detect_numeric_coded_categoricals({df}, max_unique={max_unique})"
    ),
)
def detect_numeric_coded_categoricals(df: pd.DataFrame, max_unique: int = 15) -> list[str]:
    flagged: list[str] = []
    for col in df.columns:
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        non_null = series.dropna()
        if non_null.empty:
            continue
        n_unique = non_null.nunique()
        if n_unique <= 1 or n_unique == len(non_null):
            continue  # constant or identifier, not a "coded categorical"
        if n_unique <= max_unique and _is_integer_like(non_null):
            flagged.append(col)
    return flagged


@register(
    name="detect_ordinal_candidates",
    kind="detect",
    stage="profile",
    code_template="edacore.profiling.detect_ordinal_candidates({df})",
)
def detect_ordinal_candidates(df: pd.DataFrame) -> list[str]:
    flagged: list[str] = []
    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        if non_null.empty:
            continue
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            if _is_integer_like(non_null) and _is_likert_like(non_null):
                flagged.append(col)
        elif _is_stringlike_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
            if _matches_ordinal_vocabulary(non_null.astype(str)):
                flagged.append(col)
    return flagged


@register(
    name="detect_mixed_types",
    kind="detect",
    stage="profile",
    code_template="edacore.profiling.detect_mixed_types({df})",
)
def detect_mixed_types(df: pd.DataFrame) -> dict[str, list[Any]]:
    result: dict[str, list[Any]] = {}
    for col in df.columns:
        series = df[col]
        if not pd.api.types.is_object_dtype(series):
            continue
        non_null = series.dropna()
        types_seen = {type(v) for v in non_null}
        if len(types_seen) <= 1:
            continue
        examples: list[Any] = []
        seen_types: set[type] = set()
        for value in non_null:
            if type(value) not in seen_types:
                examples.append(value)
                seen_types.add(type(value))
            if len(examples) >= 5:
                break
        result[col] = examples
    return result


@register(
    name="detect_sentinel_values",
    kind="detect",
    stage="profile",
    code_template="edacore.profiling.detect_sentinel_values({df}, candidates={candidates!r})",
)
def detect_sentinel_values(
    df: pd.DataFrame, candidates: tuple[Any, ...] = DEFAULT_SENTINEL_CANDIDATES
) -> dict[str, list[Any]]:
    numeric_candidates = [
        c for c in candidates if isinstance(c, int | float) and not isinstance(c, bool)
    ]
    string_candidates = {str(c).strip().lower() for c in candidates if isinstance(c, str)}

    result: dict[str, list[Any]] = {}
    for col in df.columns:
        series = df[col]
        found: list[Any] = []
        if pd.api.types.is_numeric_dtype(series):
            present = set(series.dropna().unique())
            found = [c for c in numeric_candidates if c in present]
        elif _is_stringlike_dtype(series):
            normalized = series.dropna().astype(str).str.strip().str.lower()
            present_norm = set(normalized.unique())
            found = [c for c in string_candidates if c in present_norm]
        if found:
            result[col] = found
    return result


@register(
    name="detect_constant_columns",
    kind="detect",
    stage="profile",
    code_template=(
        "edacore.profiling.detect_constant_columns({df}, near_zero_var_ratio={near_zero_var_ratio})"
    ),
)
def detect_constant_columns(df: pd.DataFrame, near_zero_var_ratio: float = 0.95) -> list[str]:
    flagged: list[str] = []
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            continue
        if series.nunique() <= 1:
            flagged.append(col)
            continue
        top_share = float(series.value_counts(normalize=True).iloc[0])
        if top_share >= near_zero_var_ratio:
            flagged.append(col)
    return flagged


@register(
    name="detect_duplicates",
    kind="detect",
    stage="profile",
    code_template="edacore.profiling.detect_duplicates({df}, subset={subset!r})",
)
def detect_duplicates(df: pd.DataFrame, subset: list[str] | None = None) -> DuplicateReport:
    exact_mask = df.duplicated(keep=False)
    exact_indices = [int(i) for i in df.index[exact_mask]]

    n_key_duplicates = 0
    key_duplicate_indices: list[int] = []
    if subset:
        key_mask = df.duplicated(subset=subset, keep=False)
        key_duplicate_indices = [int(i) for i in df.index[key_mask]]
        n_key_duplicates = int(key_mask.sum())

    return DuplicateReport(
        n_exact=int(exact_mask.sum()),
        exact_duplicate_indices=exact_indices,
        key_columns=subset,
        n_key_duplicates=n_key_duplicates,
        key_duplicate_indices=key_duplicate_indices,
    )


def _find_time_index(df: pd.DataFrame, reasons: list[str]) -> str | None:
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            reasons.append(f"'{col}' is already datetime64")
            return str(col)

    for col in df.columns:
        if not _TIME_NAME_RE.search(str(col)):
            continue
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
        success_rate = float(parsed.notna().mean())
        if success_rate >= 0.95:
            reasons.append(f"'{col}' parses as a datetime ({success_rate:.0%} of values)")
            return str(col)
    return None


def _find_entity_id(df: pd.DataFrame, time_index: str | None, reasons: list[str]) -> str | None:
    candidates = [col for col in df.columns if col != time_index and _ID_NAME_RE.search(str(col))]
    for col in candidates:
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        n = len(non_null)
        n_unique = non_null.nunique()
        if 1 < n_unique < n:
            reasons.append(f"'{col}' repeats ({n_unique} unique of {n} rows)")
            return str(col)
    return None


@register(
    name="detect_structure",
    kind="detect",
    stage="profile",
    code_template="edacore.profiling.detect_structure({df})",
)
def detect_structure(df: pd.DataFrame) -> StructureReport:
    reasons: list[str] = []

    if df.empty:
        return StructureReport(structure="unknown", confidence=0.0, reasons=["dataset is empty"])

    time_index = _find_time_index(df, reasons)
    entity_id = _find_entity_id(df, time_index, reasons)

    structure: _StructureLiteral
    confidence: float
    if time_index is not None and entity_id is not None:
        rows_per_entity = df.groupby(df[entity_id])[time_index].nunique()
        if bool((rows_per_entity > 1).any()):
            structure, confidence = "panel", 0.8
            reasons.append(f"'{entity_id}' repeats across multiple '{time_index}' values -> panel")
        else:
            structure, confidence = "time_series", 0.6
            reasons.append(f"'{time_index}' looks like a regular time index -> time_series")
    elif time_index is not None:
        structure, confidence = "time_series", 0.8
        reasons.append(f"'{time_index}' looks like a regular time index -> time_series")
    elif entity_id is not None:
        structure, confidence = "repeated_measures", 0.6
        reasons.append(f"'{entity_id}' repeats without a time index -> repeated_measures")
    else:
        structure, confidence = "cross_sectional", 0.6
        reasons.append("no time index or repeating entity id found -> cross_sectional")

    return StructureReport(
        structure=structure,
        time_index=time_index,
        entity_id=entity_id,
        confidence=confidence,
        reasons=reasons,
    )


@register(
    name="detect_leakage_candidates",
    kind="detect",
    stage="profile",
    code_template="edacore.profiling.detect_leakage_candidates({df}, target={target!r})",
)
def detect_leakage_candidates(df: pd.DataFrame, target: str) -> list[LeakageFlag]:
    if target not in df.columns:
        raise KeyError(f"target column '{target}' not found")

    flags: list[LeakageFlag] = []
    target_series = df[target]
    target_name_lower = str(target).lower()

    for col in df.columns:
        if col == target:
            continue
        col_series = df[col]

        if target_name_lower in str(col).lower():
            flags.append(
                LeakageFlag(
                    column=str(col),
                    reason=f"column name contains target name '{target}'",
                    method="name_match",
                )
            )
            continue

        if pd.api.types.is_numeric_dtype(col_series) and pd.api.types.is_numeric_dtype(
            target_series
        ):
            paired = pd.concat([col_series, target_series], axis=1).dropna()
            if (
                len(paired) >= 3
                and paired.iloc[:, 0].nunique() > 1
                and paired.iloc[:, 1].nunique() > 1
            ):
                corr = paired.iloc[:, 0].corr(paired.iloc[:, 1])
                if corr is not None and not np.isnan(corr) and abs(corr) >= 0.98:
                    flags.append(
                        LeakageFlag(
                            column=str(col),
                            reason=f"near-perfect correlation with target (r={corr:.3f})",
                            method="pearson_correlation",
                            score=float(corr),
                        )
                    )
                    continue

        paired_cat = pd.concat([col_series, target_series], axis=1).dropna()
        if len(paired_cat) >= 3 and col_series.nunique() < len(paired_cat):
            group_sizes = paired_cat.groupby(paired_cat.columns[0])[paired_cat.columns[1]].nunique()
            if len(group_sizes) > 1 and int(group_sizes.max()) == 1:
                flags.append(
                    LeakageFlag(
                        column=str(col),
                        reason="column value perfectly determines the target",
                        method="perfect_predictability",
                        score=1.0,
                    )
                )

    return flags


@register(
    name="detect_weight_columns",
    kind="detect",
    stage="profile",
    code_template="edacore.profiling.detect_weight_columns({df})",
)
def detect_weight_columns(df: pd.DataFrame) -> list[str]:
    return [str(col) for col in df.columns if _WEIGHT_NAME_RE.match(str(col))]


@register(
    name="detect_pii",
    kind="detect",
    stage="profile",
    code_template="edacore.profiling.detect_pii({df}, columns={columns!r})",
)
def detect_pii(df: pd.DataFrame, columns: list[str] | None = None) -> dict[str, list[str]]:
    target_columns = columns if columns is not None else list(df.columns)
    result: dict[str, list[str]] = {}
    for col in target_columns:
        if col not in df.columns or not _is_stringlike_dtype(df[col]):
            continue
        sample = df[col].dropna().astype(str)
        if sample.empty:
            continue
        kinds: list[str] = []
        if float(sample.str.match(_EMAIL_RE).mean()) >= 0.5:
            kinds.append("email")
        if float(sample.str.match(_PHONE_RE).mean()) >= 0.5:
            kinds.append("phone")
        if float(sample.str.match(_SSN_RE).mean()) >= 0.5:
            kinds.append("national_id")
        if kinds:
            result[col] = kinds
    return result


@register(
    name="summarize_numeric",
    kind="profile",
    stage="profile",
    code_template="edacore.profiling.summarize_numeric({df}, col={col!r})",
)
def summarize_numeric(df: pd.DataFrame, col: str) -> dict[str, Any]:
    series = pd.to_numeric(df[col], errors="coerce")
    non_null = series.dropna()
    n = len(series)
    if non_null.empty:
        return {
            "n": n,
            "n_missing": n,
            "mean": None,
            "sd": None,
            "median": None,
            "iqr": None,
            "mad": None,
            "min": None,
            "max": None,
            "skew": None,
            "kurtosis": None,
            "zeros": 0,
            "negatives": 0,
        }
    q1, q3 = non_null.quantile([0.25, 0.75])
    return {
        "n": n,
        "n_missing": int(series.isna().sum()),
        "mean": float(non_null.mean()),
        "sd": float(non_null.std()),
        "median": float(non_null.median()),
        "iqr": float(q3 - q1),
        "mad": float((non_null - non_null.median()).abs().median()),
        "min": float(non_null.min()),
        "max": float(non_null.max()),
        "skew": float(non_null.skew()),  # type: ignore[arg-type]
        "kurtosis": float(non_null.kurt()),  # type: ignore[arg-type]
        "zeros": int((non_null == 0).sum()),
        "negatives": int((non_null < 0).sum()),
    }


_RARE_LEVEL_THRESHOLD = 0.01


@register(
    name="summarize_categorical",
    kind="profile",
    stage="profile",
    code_template="edacore.profiling.summarize_categorical({df}, col={col!r})",
)
def summarize_categorical(df: pd.DataFrame, col: str) -> dict[str, Any]:
    series = df[col].dropna().astype(str)
    n = len(series)
    if n == 0:
        return {"n": 0, "n_levels": 0, "frequencies": {}, "rare_levels": [], "entropy": 0.0}

    counts = series.value_counts()
    freqs = (counts / n).to_dict()
    rare_levels = [level for level, share in freqs.items() if share < _RARE_LEVEL_THRESHOLD]
    probs = np.array(list(freqs.values()), dtype=float)
    entropy = float(-(probs * np.log2(probs)).sum())
    return {
        "n": n,
        "n_levels": int(counts.shape[0]),
        "frequencies": counts.to_dict(),
        "rare_levels": rare_levels,
        "entropy": entropy,
    }


@register(
    name="summarize_datetime",
    kind="profile",
    stage="profile",
    code_template="edacore.profiling.summarize_datetime({df}, col={col!r})",
)
def summarize_datetime(df: pd.DataFrame, col: str) -> dict[str, Any]:
    series = pd.to_datetime(df[col], errors="coerce", format="mixed")
    non_null = series.dropna().sort_values()
    n = len(series)
    if non_null.empty:
        return {
            "n": n,
            "n_missing": n,
            "min": None,
            "max": None,
            "inferred_freq": None,
            "n_gaps": 0,
            "future_dates": 0,
        }

    diffs = non_null.diff().dropna()
    inferred_freq = pd.infer_freq(non_null) if len(non_null) >= 3 else None
    n_gaps = 0
    if not diffs.empty:
        mode_diff = diffs.mode().iloc[0]
        n_gaps = int((diffs > mode_diff).sum())
    now = pd.Timestamp.now(tz=non_null.dt.tz)
    future_dates = int((non_null > now).sum())

    return {
        "n": n,
        "n_missing": int(series.isna().sum()),
        "min": non_null.min(),
        "max": non_null.max(),
        "inferred_freq": inferred_freq,
        "n_gaps": n_gaps,
        "future_dates": future_dates,
    }


@register(
    name="summarize_text",
    kind="profile",
    stage="profile",
    code_template="edacore.profiling.summarize_text({df}, col={col!r})",
)
def summarize_text(df: pd.DataFrame, col: str) -> dict[str, Any]:
    series = df[col]
    non_null = series.dropna().astype(str)
    n = len(series)
    if non_null.empty:
        return {
            "n": n,
            "n_missing": n,
            "empty_share": 0.0,
            "mean_length": None,
            "mean_tokens": None,
        }

    lengths = non_null.str.len()
    tokens = non_null.str.split().map(len, na_action="ignore")
    empty_share = float((non_null.str.strip() == "").mean())
    return {
        "n": n,
        "n_missing": int(series.isna().sum()),
        "empty_share": empty_share,
        "mean_length": float(lengths.mean()),
        "length_distribution": {
            "min": int(lengths.min()),
            "p25": float(lengths.quantile(0.25)),
            "median": float(lengths.median()),
            "p75": float(lengths.quantile(0.75)),
            "max": int(lengths.max()),
        },
        "mean_tokens": float(tokens.mean()),
    }


def _coerce_summary(summary: dict[str, Any]) -> dict[str, float | int | str]:
    """Narrow a rich summarize_* dict to ColumnProfile.summary's plain type.

    Nested structures (e.g. summarize_text's length_distribution) and None
    values aren't representable in `dict[str, float | int | str]`; the full
    detail stays available by calling the summarize_* function directly.
    """
    coerced: dict[str, float | int | str] = {}
    for key, value in summary.items():
        if isinstance(value, bool):
            coerced[key] = int(value)
        elif isinstance(value, int | float | str):
            coerced[key] = value
        elif isinstance(value, pd.Timestamp):
            coerced[key] = value.isoformat()
    return coerced


_NUMERIC_TYPES = (SemanticType.CONTINUOUS, SemanticType.DISCRETE)
_CATEGORICAL_TYPES = (SemanticType.NOMINAL, SemanticType.ORDINAL, SemanticType.BINARY)


@register(
    name="profile_dataset",
    kind="profile",
    stage="profile",
    code_template="edacore.profiling.profile_dataset({df}, target={target!r})",
)
def profile_dataset(df: pd.DataFrame, target: str | None = None) -> DatasetProfile:
    n_rows, n_cols = df.shape
    semantic_types = infer_semantic_types(df)
    sentinel_hits = detect_sentinel_values(df)
    identifier_cols = set(detect_identifier_columns(df))
    numeric_coded = set(detect_numeric_coded_categoricals(df))
    ordinal_candidates = set(detect_ordinal_candidates(df))
    constant_cols = set(detect_constant_columns(df))

    columns: list[ColumnProfile] = []
    for col in df.columns:
        series = df[col]
        semantic_type, confidence = semantic_types[col]
        n = len(series)
        n_missing = int(series.isna().sum())
        n_unique = int(series.dropna().nunique())

        flags: list[str] = []
        if col in constant_cols:
            flags.append("constant")
        if col in identifier_cols:
            flags.append("identifier")
        if col in numeric_coded:
            flags.append("numeric_coded_categorical")
        if col in ordinal_candidates:
            flags.append("ordinal_candidate")
        if (
            n > 0
            and semantic_type in (SemanticType.NOMINAL, SemanticType.TEXT)
            and n_unique / n > 0.5
        ):
            flags.append("high_cardinality")

        summary: dict[str, Any]
        if semantic_type in _NUMERIC_TYPES:
            summary = summarize_numeric(df, col)
        elif semantic_type == SemanticType.DATETIME:
            summary = summarize_datetime(df, col)
        elif semantic_type == SemanticType.TEXT:
            summary = summarize_text(df, col)
        elif semantic_type in _CATEGORICAL_TYPES:
            summary = summarize_categorical(df, col)
        else:
            summary = {}

        columns.append(
            ColumnProfile(
                name=str(col),
                dtype=str(series.dtype),
                semantic_type=semantic_type,
                semantic_confidence=confidence,
                n=n,
                n_missing=n_missing,
                n_unique=n_unique,
                sentinel_candidates=sentinel_hits.get(col, []),
                summary=_coerce_summary(summary),
                flags=flags,
            )
        )

    structure_report = detect_structure(df)
    duplicate_report = detect_duplicates(df)
    leakage = detect_leakage_candidates(df, target) if target else []
    pii = detect_pii(df)

    return DatasetProfile(
        n_rows=n_rows,
        n_cols=n_cols,
        columns=columns,
        structure=structure_report.structure,
        time_index=structure_report.time_index,
        entity_id=structure_report.entity_id,
        target=target,
        duplicate_rows=duplicate_report.n_exact,
        leakage_candidates=[flag.column for flag in leakage],
        pii_columns=list(pii.keys()),
    )
