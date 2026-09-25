"""Why the dataset store writes zstd rather than pandas' default snappy.

Marked slow: it writes a 1M-row frame several times, which is too much for
every push. It runs in the nightly workflow, where the point is to catch the
decision going stale rather than to police a wall-clock number.

**The measurement that motivated the choice** (1M rows x 5 columns, 41 MB in
memory, on this project's own WSL setup):

| Location | Compression | Write | Read | Size |
|---|---|---|---|---|
| `/mnt/c` (Windows drive via 9p) | snappy | 2.12 s | 0.58 s | 19.3 MB |
| `/mnt/c` | zstd | 1.06 s | 0.22 s | 12.7 MB |
| WSL native (`/tmp`) | snappy | 0.14 s | 0.03 s | 19.3 MB |
| WSL native | zstd | 0.12 s | 0.02 s | 12.7 MB |

Two things follow. First, `/mnt/c` is roughly an order of magnitude slower
than WSL's own filesystem — 2.1 s against 0.14 s for the same write — so on
a Windows-drive checkout the bottleneck is bytes crossing that boundary, not
compression CPU. Second, and because of that, zstd wins on *both* axes
there: fewer bytes to push through 9p means it is faster **and** smaller.
On native storage the two are near-identical, so nothing is lost by
preferring zstd everywhere.

What this test asserts is the decision, not the numbers: zstd must not be
materially slower than snappy, and must be materially smaller. A wall-clock
assertion would fail on a loaded CI runner for reasons that say nothing
about the code.

**Note on where it runs.** `tmp_path` is on the *native* filesystem, not
`/mnt/c` — so this test reproduces the bottom two rows of the table above,
where the two encodings are within noise on time (measured 0.10 s vs 0.08 s)
and zstd's win is entirely the 34% size reduction. It cannot reproduce the
`/mnt/c` rows, because pytest does not put its temporary directory there.
That is fine for the purpose: the size claim is the one that generalises,
and it is the size claim that makes zstd faster on a slow filesystem.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from edacopilot.session import PARQUET_COMPRESSION

pytestmark = pytest.mark.slow

ROWS = 1_000_000


def _big_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "id": np.arange(ROWS),
            "value": rng.normal(0, 1, ROWS),
            "group": rng.choice(["a", "b", "c"], ROWS),
            "flag": rng.integers(0, 2, ROWS),
            "ts": pd.date_range("2024-01-01", periods=ROWS, freq="s"),
        }
    )


def _time_write(df: pd.DataFrame, path: Path, compression: str) -> tuple[float, float, int]:
    """Median of three, so one slow first write does not decide it."""
    writes, reads = [], []
    for _ in range(3):
        start = time.perf_counter()
        df.to_parquet(path, compression=compression, index=False)
        writes.append(time.perf_counter() - start)
        start = time.perf_counter()
        pd.read_parquet(path)
        reads.append(time.perf_counter() - start)
    return sorted(writes)[1], sorted(reads)[1], path.stat().st_size


def test_zstd_is_not_slower_and_is_smaller_than_snappy(tmp_path: Path) -> None:
    df = _big_frame()
    zstd_write, zstd_read, zstd_size = _time_write(df, tmp_path / "z.parquet", "zstd")
    snappy_write, snappy_read, snappy_size = _time_write(df, tmp_path / "s.parquet", "snappy")

    print(
        f"\n  zstd   write={zstd_write:.2f}s read={zstd_read:.2f}s size={zstd_size / 1e6:.1f} MB"
        f"\n  snappy write={snappy_write:.2f}s read={snappy_read:.2f}s "
        f"size={snappy_size / 1e6:.1f} MB"
    )

    assert PARQUET_COMPRESSION == "zstd"
    # Smaller by a clear margin -- this is the robust half of the claim.
    assert zstd_size < snappy_size * 0.85
    # And not materially slower. The generous factor is deliberate: a loaded
    # runner's timings say nothing about the code, and the size win alone
    # justifies the choice on a slow filesystem.
    assert zstd_write < snappy_write * 2.0


def test_a_million_row_round_trip_is_exact(tmp_path: Path) -> None:
    """The reason to care about parquet at all: a version has to come back
    byte-identical, or undo and branch mean nothing."""
    df = _big_frame()
    path = tmp_path / "v0.parquet"
    df.to_parquet(path, compression=PARQUET_COMPRESSION, index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(path), df)
