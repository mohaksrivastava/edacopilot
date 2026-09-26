"""The `%eda` line magics (ARCHITECTURE.md, Section 13.1).

```
%load_ext edacopilot
%eda start df --target default_flag
%eda ask "Does loan amount differ between men and women?"
%eda undo | %eda branch <name> | %eda ledger | %eda export
```

The magics are a thin shell over the Python API and nothing else — every
one resolves to a single `session.*` call. That is not laziness: Section
13.1 says the API mirrors every action so the tool is scriptable and
testable, and the cheapest way to keep that true is for the other
interfaces to have no separate implementation to drift from.

Parsing is `shlex` plus a small argument table rather than `argparse`,
because argparse exits the process on a bad argument, which in a notebook
kernel means killing the kernel.
"""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING, Any

from edacopilot.orchestrator.cards import Card

if TYPE_CHECKING:  # pragma: no cover
    from IPython.core.interactiveshell import InteractiveShell

USAGE = (
    "%eda start <dataframe> [--target COL] [--name NAME]\n"
    '%eda ask "<question>"      %eda explain <term>\n'
    '%eda accept [persona]       %eda override <method> --reason "..." [--confirm]\n'
    "%eda goto <stage>           %eda skip\n"
    "%eda undo                   %eda branch <name>       %eda switch <name>\n"
    "%eda ledger                 %eda steps               %eda code\n"
    "%eda export"
)


class EdaMagicError(RuntimeError):
    """A magic that cannot run, reported rather than raised at the user.

    Raised internally and caught by the magic itself, so a typo prints
    usage instead of a traceback — a stack trace in a notebook for a
    mistyped flag teaches nothing and hides the answer.
    """


def _split(line: str) -> tuple[str, list[str], dict[str, Any]]:
    """`command`, positional arguments and `--flags`."""
    tokens = shlex.split(line.strip())
    if not tokens:
        raise EdaMagicError("no command given")
    command, rest = tokens[0], tokens[1:]

    positional: list[str] = []
    flags: dict[str, Any] = {}
    index = 0
    while index < len(rest):
        token = rest[index]
        if token.startswith("--"):
            name = token[2:].replace("-", "_")
            if index + 1 < len(rest) and not rest[index + 1].startswith("--"):
                flags[name] = rest[index + 1]
                index += 2
            else:
                flags[name] = True
                index += 1
        else:
            positional.append(token)
            index += 1
    return command, positional, flags


def run_magic(line: str, namespace: dict[str, Any]) -> Any:
    """Execute one `%eda` line against a namespace.

    Separated from the IPython plumbing so the whole magic surface is
    testable with a plain dict — no kernel, no shell, no display hook.
    """
    command, positional, flags = _split(line)
    session = namespace.get("_eda_session")

    if command == "start":
        return _start(positional, flags, namespace)

    if session is None:
        raise EdaMagicError("no session yet; run `%eda start <dataframe>` first")

    if command == "ask":
        if not positional:
            raise EdaMagicError('ask needs a question: %eda ask "..."')
        return session.ask(" ".join(positional))
    if command == "explain":
        if not positional:
            raise EdaMagicError("explain needs a term: %eda explain sphericity")
        return session.explain(" ".join(positional))
    if command == "accept":
        return session.accept(positional[0] if positional else None)
    if command == "override":
        if not positional:
            raise EdaMagicError("override needs a method name")
        return session.override(
            positional[0],
            reason=flags.get("reason"),
            confirm=bool(flags.get("confirm", False)),
        )
    if command in ("goto", "stage"):
        if not positional:
            raise EdaMagicError("goto needs a stage name")
        return session.goto_stage(positional[0])
    if command == "skip":
        return session.skip()
    if command == "undo":
        return session.undo()
    if command == "branch":
        if not positional:
            raise EdaMagicError("branch needs a name")
        return session.branch(positional[0])
    if command == "switch":
        if not positional:
            raise EdaMagicError("switch needs a branch name")
        return session.switch_branch(positional[0])
    if command == "ledger":
        return session.orchestrator.ledger_card()
    if command == "steps":
        return session.steps()
    if command == "code":
        return session.code()
    if command == "export":
        return session.goto_stage("export")

    raise EdaMagicError(f"unknown command '{command}'")


def _start(positional: list[str], flags: dict[str, Any], namespace: dict[str, Any]) -> Any:
    from edacopilot import start

    if not positional:
        raise EdaMagicError("start needs the name of a DataFrame in scope")
    name = positional[0]
    if name not in namespace:
        raise EdaMagicError(f"no variable called '{name}' in this notebook")
    session = start(
        namespace[name],
        target=flags.get("target"),
        name=flags.get("name"),
        display=False,
    )
    namespace["_eda_session"] = session
    return session


def load_ipython_extension(ipython: InteractiveShell) -> None:
    """Register `%eda`. Called by `%load_ext edacopilot`."""
    from IPython.core.magic import register_line_magic

    @register_line_magic("eda")  # type: ignore[arg-type]
    def _eda(line: str) -> Any:
        from IPython.display import display

        try:
            result = run_magic(line, ipython.user_ns)
        except EdaMagicError as error:
            print(f"%eda: {error}\n\n{USAGE}")
            return None

        session = ipython.user_ns.get("_eda_session")
        panel = getattr(session, "_panel", None)
        if isinstance(result, Card) and panel is not None:
            panel.show(result)
            return None
        if isinstance(result, Card):
            print(result.to_text())
            return None
        if result is not None and not hasattr(result, "session_id"):
            display(result)  # type: ignore[no-untyped-call]
        return None
