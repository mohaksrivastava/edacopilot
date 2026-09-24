"""Shared helpers for stattests submodules: not itself part of the public
edacore API, just cross-cutting logic every test function needs (rule 5.5:
nan_policy handling with dropped-n reporting, reproducible randomness).
"""

from __future__ import annotations

from math import comb
from typing import Literal

import numpy as np
import pandas as pd

NanPolicy = Literal["omit", "raise"]

# Above this many distinct arrangements, permutation tests fall back to
# Monte Carlo sampling instead of full enumeration (exact enumeration's
# cost grows combinatorially: C(12,6)=924 is instant, C(80,40)~10^23 is not).
MAX_EXACT_PERMUTATIONS = 100_000


def n_arrangements_two_sample(n1: int, n2: int) -> int:
    """Number of distinct ways to split n1+n2 units into groups of size n1, n2."""
    return comb(n1 + n2, n1)


def n_arrangements_paired(n: int) -> int:
    """Number of distinct sign-flip patterns for n paired differences."""
    return int(2**n)


def clean_series(
    series: pd.Series, nan_policy: NanPolicy, label: str, warnings: list[str]
) -> np.ndarray:
    """Drop (or raise on) missing values in a single column, per rule 5.5."""
    n_before = len(series)
    clean = series.dropna()
    n_dropped = n_before - len(clean)
    if n_dropped > 0:
        if nan_policy == "raise":
            raise ValueError(f"'{label}' has {n_dropped} missing value(s) and nan_policy='raise'")
        warnings.append(f"{n_dropped} missing value(s) dropped from '{label}' (nan_policy='omit')")
    return clean.to_numpy(dtype=float)


def clean_paired(
    df: pd.DataFrame, a: str, b: str, nan_policy: NanPolicy, warnings: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Drop (or raise on) missing values from a paired pair of columns —
    a pair is dropped together if either side is missing, preserving
    pairing (Section 5.5: 'untreated missingness is handled pairwise per
    test'; for a paired test the pair itself is the unit)."""
    sub = df[[a, b]]
    n_before = len(sub)
    clean = sub.dropna()
    n_dropped = n_before - len(clean)
    if n_dropped > 0:
        if nan_policy == "raise":
            raise ValueError(f"{n_dropped} pair(s) missing '{a}' or '{b}' and nan_policy='raise'")
        warnings.append(
            f"{n_dropped} pair(s) dropped due to missing '{a}' or '{b}' (nan_policy='omit')"
        )
    return clean[a].to_numpy(dtype=float), clean[b].to_numpy(dtype=float)
