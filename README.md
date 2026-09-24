# edacopilot

A Jupyter-based EDA co-pilot for junior data scientists. It walks through
exploratory data analysis one step at a time, checking assumptions before
showing results and comparing trade-offs across three personas (Professor,
Consultant, Maverick) before the user decides.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full specification and build
plan. The project is currently at milestone M0 (scaffolding); a quickstart
notebook and usage docs land in M15.

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
