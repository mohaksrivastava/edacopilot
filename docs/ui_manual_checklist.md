# M8 manual checklist: JupyterLab, Notebook 7, VS Code

M8's acceptance criterion (ARCHITECTURE.md, Section 16) is a **manual**
checklist in three hosts, plus "all actions reachable by buttons". Most of
what a UI can get wrong is automated in
`tests/unit/edacopilot/test_ui.py` — every card kind renders, every button
carries a valid intent, a click leaves the session in the same state as the
API call, status is never signalled by colour alone. What automation cannot
check is whether the thing is *legible and usable in a real front end*, so
that is what this list is for.

## What is already proven automatically

Do not re-check these by hand; they run in CI.

- Every one of Section 9.4's eight card kinds renders to a widget tree,
  both from synthetic cards and from every card the six golden transcripts
  actually produce.
- Every `ActionButton` carries a valid, confident `Intent`.
- Clicking Accept, a stage tab, or a completed override leaves the session
  byte-identical to the equivalent Python API call.
- Rendering a card executes nothing (rule 3 through the UI).
- Every `CheckStatus` renders with an icon and a word as well as a colour.
- The panel emits `application/vnd.jupyter.widget-view+json`, in-process
  and (nightly) through a real `ipykernel` subprocess.
- No module in `edacopilot.ui` imports `ipylab`, `jupyterlab*`,
  `jupyter_server` or `notebook`.

## Setup

```bash
uv pip install -e ".[dev,all]"          # includes ipykernel
python -m ipykernel install --user --name edacopilot-wsl \
       --display-name "edacopilot (.venv-wsl)"
```

```python
import pandas as pd, numpy as np
import edacopilot as eda

rng = np.random.default_rng(0)
df = pd.DataFrame(
    {
        "income": rng.lognormal(10, 0.6, 79),
        "gender": ["F"] * 38 + ["M"] * 41,
    }
)
session = eda.start(df, name="demo")
session.ask(
    goal="compare_groups",
    outcome="income",
    group="gender",
    design="independent",
    confirmed_by_user=["design"],
)
```

## The checklist

Run in **JupyterLab**, **Notebook 7** and **VS Code notebooks** (in VS
Code, select the `edacopilot (.venv-wsl)` kernel). Each row is a thing that
differs between hosts or that only a human can judge.

| # | Check | Why this host matters |
|---|---|---|
| 1 | The panel appears at all, with header, stage tabs, card area and input box | Widget support differs per host; VS Code loads the widget manager separately |
| 2 | Stage tabs wrap rather than overflow at a narrow window width | Flexbox wrapping is the layout most likely to differ |
| 3 | Clicking a stage tab moves the header's "stage:" and adds a ✓ | Confirms the click round-trip through the kernel, not just the handler |
| 4 | The diagnostics accordion opens and closes | `Accordion` is the most host-sensitive core widget |
| 5 | Q-Q and box plots appear inline, not as broken images or a text placeholder | PNG-in-`Image` is the portable path; confirm it really is |
| 6 | Plots are legible at the host's default width | DPI and default font size differ between hosts |
| 7 | Two persona panels sit side by side, and wrap to stacked when narrow | The divergence card is the densest layout |
| 8 | Hovering a dotted-underlined term shows its definition | `<abbr title>` tooltips are host- and OS-rendered |
| 9 | A term is still reachable by keyboard (Tab to it) | Hover-only would exclude keyboard users |
| 10 | Typing in the input box and pressing **Enter** sends | `continuous_update=False` is the ipywidgets 8 replacement for `on_submit`; verify Enter, not just the Send button |
| 11 | `%load_ext edacopilot` then `%eda ledger` prints or renders | Magics load per host |
| 12 | Override: Accept is greyed out until a reason is typed | The disabled state must be visible, not just present |
| 13 | Override on an ineligible method: Accept stays greyed until the checkbox is also ticked | Section 7.5's second gate |
| 14 | The failed hard assumption and its consequence are readable on that card | This is the friction's whole purpose |
| 15 | With the notebook rendered in greyscale (or with a colour filter), PASS / BORDERLINE / FAIL are still distinguishable | Colour-only signalling would be invisible to ~1 man in 12 |
| 16 | After 20+ cards, scrolling the card area stays responsive | `max_height` + `overflow` behave differently per host |
| 17 | Restarting the kernel and running `eda.resume("<session_id>")` restores stage and visited tabs | Section 12.4 through the UI |
| 18 | Clicking "Show plot" on a passing check appends a small card with that plot, right below the proposal it came from | m8.1: on-demand rendering follows the same click-appends-a-card model as every other action, so nothing about the panel's behaviour is special-cased for it |

## Known limitations in this build

- **No "insert code cell"**. Section 13.3 puts it in v2; it needs a
  frontend extension.
- **The card history is capped at 40** (`ui.MAX_HISTORY`). The provenance
  log is the real history; a Jupyter output cell with hundreds of widget
  trees stops being usable well before it stops being complete.
- **Plots are static PNGs.** Section 4.1 lists plotly as an optional
  extra; interactive plots are not wired up.
- **Tooltips are `<abbr title=...>`**, so they follow the OS's tooltip
  delay and cannot be styled. A JavaScript popover would look better and
  would not work identically in all three hosts.
