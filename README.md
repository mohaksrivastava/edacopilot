# edacopilot

A Jupyter-based EDA co-pilot for junior data scientists. It walks through
exploratory data analysis one step at a time, checking assumptions before
showing results and comparing trade-offs across three personas (Professor,
Consultant, Maverick) before the user decides.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full specification and build
plan. The project is currently at milestone M8 (the Jupyter panel); a
quickstart notebook lands in M15.

Everything runs with **no LLM configured** — that is the supported mode
today, and it is how the test suite runs.

## In a notebook

```python
import pandas as pd
import edacopilot as eda

session = eda.start(pd.read_csv("survey.csv"), target="churn", name="survey")
```

The panel renders inline: stage tabs, cards with their diagnostics and
plots, and a button for each persona's proposal. It works in JupyterLab,
Notebook 7 and VS Code notebooks. There are `%eda` magics too — see
ARCHITECTURE.md Section 13.1.

## Or as plain Python

Every action the panel offers is a method on the session — which is what
makes the tool scriptable, and what the panel itself calls:

```python
session.ask(goal="compare_groups", outcome="income", group="region", design="independent")
# ... read the card, then:
session.accept("professor")
```

**Nothing runs until you accept it.**
[docs/sample_conversation.md](docs/sample_conversation.md) shows a whole
session as the user sees it.

## ⚠️ `.edacopilot/` contains your data — gitignore it

A session writes its state to `.edacopilot/<session_id>/`, and that
directory holds **your dataset**, saved as parquet: the original data plus
one file per transformed version. It sits next to your notebook by default,
so it is easy to commit by accident.

Add this to your project's `.gitignore` before committing anything:

```gitignore
.edacopilot/
```

Committing a session directory publishes the data it was run on, to
everyone with access to the repository, with nothing in the diff to say so.
If you need to share a session, share the `session.json` (history, methods,
p-values, no rows) and leave the parquet files behind.

## Development

```bash
pip install -e ".[dev,all]"
pre-commit install

ruff check .
ruff format --check .
mypy src/
pytest -m "not llm"
```

## License

Apache-2.0. See [LICENSE](LICENSE). No GPL dependencies are accepted
(ARCHITECTURE.md, rule 8).
