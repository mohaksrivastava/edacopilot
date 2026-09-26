"""Render a transcript as plain text (ARCHITECTURE.md, Section 15.5).

One renderer, two consumers: the golden test compares its output to a
committed file, and `scripts/generate_transcripts.py` writes those files
and `docs/sample_conversation.md`. Sharing the renderer is the point — a
document that could drift from what the tests assert would be worse than
no document.
"""

from __future__ import annotations

from pathlib import Path

from edacopilot.session import Session

from .transcripts import Transcript

GOLDEN_DIR = Path(__file__).parent / "golden"

SEPARATOR = "-" * 74


def run(transcript: Transcript, root: Path) -> tuple[Session, list[str]]:
    """Replay a transcript, returning the session and the rendered turns."""
    session = transcript.start(root)
    blocks: list[str] = []
    for step in transcript.steps:
        card = step.run(session)
        blocks.append(_render_turn(step.say, step.note, card.to_text()))
    return session, blocks


def render(transcript: Transcript, root: Path) -> str:
    session, blocks = run(transcript, root)
    header = [
        f"# {transcript.title}",
        "",
        transcript.note,
        "",
        SEPARATOR,
        "",
    ]
    footer = _footer(session)
    return "\n".join([*header, *blocks, *footer]).rstrip() + "\n"


def _render_turn(say: str, note: str, card_text: str) -> str:
    lines = ["YOU:"]
    lines += [f"    {line}" for line in say.splitlines()]
    lines.append("")
    if note:
        lines += [f"    ({note})", ""]
    lines.append("EDACOPILOT:")
    lines += [f"    {line}".rstrip() for line in card_text.splitlines()]
    lines += ["", SEPARATOR, ""]
    return "\n".join(lines)


def _footer(session: Session) -> list[str]:
    """What the session holds afterwards.

    Section 15.5 asks a transcript to assert "the cards, steps, versions and
    exported notebook", so the last three are part of the golden file rather
    than checked separately: a step that stopped being recorded would
    otherwise pass a card-only comparison.
    """
    lines = ["SESSION AFTERWARDS", ""]
    lines.append(f"  branch: {session.active_branch}   version: {session.version}")
    lines.append(f"  branches: {', '.join(session.branches)}")
    lines.append(f"  versions: {', '.join(session.store.versions)}")
    lines.append(f"  tests counted toward the session adjustment: {session.test_ledger.n_tests}")
    lines.append("")
    lines.append("  provenance:")
    if not session.provenance.steps:
        lines.append("    (no steps — nothing was accepted)")
    for step in session.provenance.steps:
        override = f'  override reason: "{step.override_reason}"' if step.override_reason else ""
        chosen = step.chosen.function if step.chosen else "—"
        lines.append(
            f"    {step.step_id}  {step.stage:<11} {chosen:<24} via {step.chosen_via}{override}"
        )
    lines.append("")
    lines.append("  code (rule 7 — every accepted step, runnable):")
    code = session.code().strip()
    if code:
        lines += [f"    {line}" for line in code.splitlines()]
    else:
        lines.append("    (nothing was accepted)")
    return lines


def golden_path(name: str) -> Path:
    return GOLDEN_DIR / f"{name}.txt"
