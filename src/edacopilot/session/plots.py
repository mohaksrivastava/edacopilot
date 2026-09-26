"""Plot storage (ARCHITECTURE.md, Sections 6.11 and 12.4).

Section 6.11 says each plot function returns "a `matplotlib.figure.Figure`
and a `plot_ref` ID stored in the session". The two halves live on
different sides of the `edacore` boundary, and they have to: `edacore`
draws, because that is a pure function of the data, and the session
stores, because only it knows where `.edacopilot/<session_id>/` is.
`edacore` having a session would break Section 3.1's promise that it is
usable on its own as a plain statistics library.

So a `plot_ref` is minted here. It is a short, stable id (`p0`, `p1`, …)
matching the session's other ids — `v0` for versions, `s0` for steps, `t0`
for ledger entries — and it resolves to a PNG under
`.edacopilot/<session_id>/plots/`.

**Why a file rather than bytes in `session.json`.** A base64 PNG inline
would make the state file tens of megabytes for a session with a few dozen
plots, and every save rewrites it whole. Files are also what the export in
Section 14 needs to reference.

**The same plot is not drawn twice.** A ref is keyed on the function and
its parameters, so re-rendering the same diagnostic — which the
orchestrator does whenever it rebuilds a card — reuses the PNG instead of
filling the directory with identical images.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict

from edacore.contracts import to_jsonable

PLOT_DIR = "plots"
DEFAULT_DPI = 110


class PlotRecord(BaseModel):
    """One rendered plot, as persisted."""

    model_config = ConfigDict(frozen=True)

    plot_ref: str
    function: str
    params: dict[str, Any] = {}
    # The dataset version it was drawn from. A plot is a claim about a
    # particular state of the data, and a transform invalidates it.
    version_id: str = ""
    filename: str = ""

    def key(self) -> str:
        return _key(self.function, self.params, self.version_id)


def _key(function: str, params: dict[str, Any], version_id: str) -> str:
    """A stable identity for "this plot of this data".

    `sort_keys` because a dict built in a different order is the same
    plot; `default=str` because a param can be anything the registry
    accepts, including a list of results.
    """
    return json.dumps(
        {"f": function, "p": to_jsonable(params), "v": version_id},
        sort_keys=True,
        default=str,
    )


@dataclass
class PlotStore:
    """Every plot this session has rendered."""

    directory: Path
    records: dict[str, PlotRecord] = field(default_factory=dict)
    dpi: int = DEFAULT_DPI

    @property
    def plot_directory(self) -> Path:
        return self.directory / PLOT_DIR

    def save(
        self,
        fig: Figure,
        *,
        function: str,
        params: dict[str, Any] | None = None,
        version_id: str = "",
    ) -> str:
        """Write a figure and return its `plot_ref`.

        The figure is closed afterwards. A session that kept every figure
        it ever drew would hold the memory for the life of the kernel, and
        the caller has the PNG.
        """
        params = dict(params or {})
        key = _key(function, params, version_id)
        for record in self.records.values():
            if record.key() == key:
                fig.clear()
                return record.plot_ref

        plot_ref = f"p{len(self.records)}"
        filename = f"{plot_ref}_{function}.png"
        self.plot_directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.plot_directory / filename, dpi=self.dpi, format="png")
        fig.clear()

        self.records[plot_ref] = PlotRecord(
            plot_ref=plot_ref,
            function=function,
            params=params,
            version_id=version_id,
            filename=filename,
        )
        return plot_ref

    def path_for(self, plot_ref: str) -> Path:
        return self.plot_directory / self.record(plot_ref).filename

    def record(self, plot_ref: str) -> PlotRecord:
        try:
            return self.records[plot_ref]
        except KeyError:
            raise KeyError(f"no plot '{plot_ref}' in this session") from None

    def png_bytes(self, plot_ref: str) -> bytes:
        """The image itself, for a UI that renders it inline (Section 13.2)."""
        return self.path_for(plot_ref).read_bytes()

    def refs(self) -> list[str]:
        return list(self.records)

    def to_state(self) -> list[dict[str, Any]]:
        return [record.model_dump(mode="json") for record in self.records.values()]

    @classmethod
    def from_state(cls, directory: Path, state: list[dict[str, Any]]) -> PlotStore:
        records = {item["plot_ref"]: PlotRecord.model_validate(item) for item in state}
        return cls(directory=directory, records=records)

    def __len__(self) -> int:
        return len(self.records)
