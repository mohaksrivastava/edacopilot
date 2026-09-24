"""Shared loaders for R-fixture tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

FIXTURES = Path(__file__).parents[2] / "fixtures" / "r_reference"


def data(name: str) -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "data" / f"{name}.csv")


def expanded(name: str) -> pd.DataFrame:
    """Freq-weighted long table -> one row per observation."""
    df = data(name)
    return df.loc[df.index.repeat(df["Freq"])].reset_index(drop=True)


def ref(name: str) -> dict[str, Any]:
    with (FIXTURES / f"{name}.json").open() as f:
        out: dict[str, Any] = json.load(f)
        return out
