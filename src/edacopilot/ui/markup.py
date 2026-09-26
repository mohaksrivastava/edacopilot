"""Card text to HTML, with glossary terms made clickable (Section 13).

Two jobs, and the second is the interesting one.

**A small Markdown subset.** Cards are written in Markdown (`Card.body_md`)
and the UI has to show it. A full Markdown library would be another
dependency for the handful of constructs the cards actually use — bold,
inline code, fenced code, bullets, tables and headings — so this renders
those and escapes everything else. The subset is deliberately closed: text
arriving here is written by this project, but the *data* interpolated into
it (a column name, a user's override reason) is not, so everything is
escaped first and only known constructs are unescaped back into tags.

**Glossary terms become clickable.** A junior analyst reading "the
sphericity assumption failed" needs to know what sphericity is at that
moment, not after switching context to look it up — and the whole premise
of Section 1.3's "teach by consequence" is that the explanation arrives
where the consequence does. Any term in `llm/glossary.yaml` that appears
in card text is wrapped in a `<abbr>` carrying its one-line definition, so
it shows on hover and on focus, and reads as the definition in a screen
reader. `<abbr title=...>` rather than a JavaScript popover because it is
the one mechanism that behaves identically in JupyterLab, Notebook 7 and
VS Code, none of which will run arbitrary script in an HTML widget the
same way.

Terms are matched only on whole words, longest first, and never inside an
existing tag or a code span — otherwise "the `mann_whitney` result" would
end up with a tooltip nested inside a `<code>` that already says it.
"""

from __future__ import annotations

import re
from collections.abc import Callable

Definer = Callable[[str], str | None]

# Inline constructs, applied after escaping. Order matters: code spans are
# taken first so `**` inside one is left alone.
_CODE_SPAN = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC = re.compile(r"(?<![*\w])\*([^*\n]+)\*(?![*\w])")

# A word that could be a glossary key: letters, digits, underscores, and
# the internal hyphens a term like "mann-whitney" would use.
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?:[-'][A-Za-z0-9_]+)*")

# Regions of already-rendered HTML that term-linking must not enter.
_PROTECTED = re.compile(r"<[^>]+>|<code>.*?</code>", re.DOTALL)


def escape(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def to_html(
    text: str,
    *,
    define: Definer | None = None,
    inline: bool = False,
) -> str:
    """Render card Markdown as HTML.

    `inline=True` skips the block constructs (headings, lists, tables,
    fenced code) and renders a single run of text — what a table cell or a
    list item needs.
    """
    if inline:
        return _link_terms(_inline(escape(text)), define)
    return _link_terms(_blocks(text), define)


# --------------------------------------------------------------------------
# blocks
# --------------------------------------------------------------------------


def _blocks(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if stripped.startswith("```"):
            index, block = _fenced(lines, index)
            out.append(block)
            continue
        if stripped.startswith("|") and _is_table(lines, index):
            index, block = _table(lines, index)
            out.append(block)
            continue
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            content = _inline(escape(stripped[level:].strip()))
            out.append(f"<div style='font-weight:600;margin-top:6px'>{content}</div>")
            index += 1
            continue
        if stripped.startswith(("- ", "* ")):
            index, block = _list(lines, index)
            out.append(block)
            continue
        if not stripped:
            index += 1
            continue

        paragraph: list[str] = []
        while index < len(lines) and lines[index].strip() and not _starts_block(lines[index]):
            paragraph.append(lines[index].strip())
            index += 1
        out.append(f"<p style='margin:4px 0'>{_inline(escape(' '.join(paragraph)))}</p>")
    return "".join(out)


def _starts_block(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith(("```", "|", "#", "- ", "* "))


def _fenced(lines: list[str], index: int) -> tuple[int, str]:
    index += 1
    body: list[str] = []
    while index < len(lines) and not lines[index].strip().startswith("```"):
        body.append(lines[index])
        index += 1
    return index + 1, (
        "<pre style='background:#f6f8fa;padding:8px;border-radius:4px;"
        f"font-size:0.85em;overflow-x:auto'>{escape(chr(10).join(body))}</pre>"
    )


def _is_table(lines: list[str], index: int) -> bool:
    return index + 1 < len(lines) and set(lines[index + 1].strip()) <= set("|-: ")


def _list(lines: list[str], index: int) -> tuple[int, str]:
    items: list[str] = []
    while index < len(lines) and lines[index].strip().startswith(("- ", "* ")):
        items.append(f"<li>{_inline(escape(lines[index].strip()[2:]))}</li>")
        index += 1
    return index, f"<ul style='margin:4px 0 4px 18px;padding:0'>{''.join(items)}</ul>"


def _table(lines: list[str], index: int) -> tuple[int, str]:
    def cells(row: str) -> list[str]:
        return [cell.strip() for cell in row.strip().strip("|").split("|")]

    header = cells(lines[index])
    index += 2  # header and the separator row
    body: list[str] = []
    while index < len(lines) and lines[index].strip().startswith("|"):
        body.append(
            "<tr>"
            + "".join(
                f"<td style='padding:2px 8px;border-top:1px solid #eee'>"
                f"{_inline(escape(cell))}</td>"
                for cell in cells(lines[index])
            )
            + "</tr>"
        )
        index += 1
    head = "".join(
        f"<th style='padding:2px 8px;text-align:left'>{_inline(escape(cell))}</th>"
        for cell in header
    )
    return index, (
        "<table style='border-collapse:collapse;font-size:0.9em;margin:4px 0'>"
        f"<tr>{head}</tr>{''.join(body)}</table>"
    )


def _inline(escaped: str) -> str:
    out = _CODE_SPAN.sub(
        lambda m: (
            f"<code style='background:#f6f8fa;padding:0 3px;border-radius:3px'>{m.group(1)}</code>"
        ),
        escaped,
    )
    out = _BOLD.sub(r"<b>\1</b>", out)
    return _ITALIC.sub(r"<i>\1</i>", out)


# --------------------------------------------------------------------------
# glossary linking
# --------------------------------------------------------------------------


def _link_terms(html: str, define: Definer | None) -> str:
    """Wrap known glossary terms in an `<abbr>` carrying their definition.

    Walks the segments *between* tags, so nothing is inserted inside an
    attribute, and skips the contents of `<code>` — a term already shown
    as code is already labelled.
    """
    if define is None:
        return html

    out: list[str] = []
    position = 0
    in_code = 0
    for match in re.finditer(r"<(/?)(\w+)[^>]*>", html):
        out.append(
            _link_run(html[position : match.start()], define)
            if not in_code
            else html[position : match.start()]
        )
        out.append(match.group(0))
        if match.group(2).lower() == "code":
            in_code += -1 if match.group(1) else 1
            in_code = max(in_code, 0)
        position = match.end()
    out.append(_link_run(html[position:], define) if not in_code else html[position:])
    return "".join(out)


def _link_run(text: str, define: Definer) -> str:
    if not text.strip():
        return text

    def replace(match: re.Match[str]) -> str:
        word = match.group(0)
        definition = define(word)
        if definition is None:
            return word
        return (
            f'<abbr title="{escape(definition)}" '
            f'style="text-decoration:underline dotted;cursor:help">{word}</abbr>'
        )

    return _WORD.sub(replace, text)
