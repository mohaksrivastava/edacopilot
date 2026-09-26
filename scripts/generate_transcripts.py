"""Regenerate the golden conversation transcripts and the sample document.

Like `generate_eligibility_table.py` and `generate_persona_picks.py`, these
are artefacts rendered from the live code rather than written by hand, and
`tests/conversations/test_conversations.py` fails if a committed file and a
fresh render disagree.

    python scripts/generate_transcripts.py          # rewrite them
    python scripts/generate_transcripts.py --check  # exit 1 if any is stale
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from tests.conversations.runner import GOLDEN_DIR, golden_path, render
from tests.conversations.transcripts import Transcript, by_name, transcripts

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DOC = ROOT / "docs" / "sample_conversation.md"

# The transcript reproduced in docs/sample_conversation.md, which is M7's
# deliverable for review: Section 7.4's worked example as a user sees it.
SAMPLE = "worked_example_7_4"


def rendered(transcript: Transcript) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        return render(transcript, Path(tmp))


def sample_document(body: str) -> str:
    """Wrap the worked-example transcript as `docs/sample_conversation.md`."""
    return "\n".join(
        [
            "# A session, as the user sees it",
            "",
            "**Generated** by `scripts/generate_transcripts.py` from the live orchestrator.",
            "Do not edit by hand: `tests/conversations/test_conversations.py` fails if this",
            "file and a fresh render disagree.",
            "",
            "This is ARCHITECTURE.md Section 7.4's worked example driven end to end through",
            "the Python API, with every card rendered exactly as the system produces it.",
            "Rendering is plain text because M7 has no UI: M8 draws these same `Card`",
            "objects as ipywidgets, and nothing about their content changes.",
            "",
            "Everything here runs with **no LLM**. Section 10.5's deterministic fallback",
            "supplies the intent parsing and the rationales, which is also how the whole",
            "test suite runs (rule 9).",
            "",
            "Three things to look for, because they are what the design is for:",
            "",
            "1. **Nothing ran until it was accepted** (rule 3). The proposal card lists",
            "   eight methods and runs none of them.",
            "2. **The personas disagree, and the card says so** rather than picking for the",
            "   user. Section 8.3's divergence detection only fires when the disagreement",
            "   is real; on clean data the same question produces one consensus card.",
            "3. **The result carries its validity notes** (Section 6.12). A p-value of 0.085",
            '   is not "no difference", and the card says which interval the data is',
            "   actually compatible with.",
            "",
            "---",
            "",
            "```text",
            body.rstrip(),
            "```",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 if anything is stale")
    args = parser.parse_args()

    stale: list[Path] = []
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    for transcript in transcripts():
        path = golden_path(transcript.name)
        body = rendered(transcript)
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != body:
                stale.append(path)
        else:
            path.write_text(body, encoding="utf-8")

    document = sample_document(rendered(by_name(SAMPLE)))
    if args.check:
        if not SAMPLE_DOC.exists() or SAMPLE_DOC.read_text(encoding="utf-8") != document:
            stale.append(SAMPLE_DOC)
    else:
        SAMPLE_DOC.write_text(document, encoding="utf-8")

    if args.check:
        for path in stale:
            print(f"{path} is stale; run: python {Path(__file__).name}", file=sys.stderr)
        return 1 if stale else 0

    print(f"wrote {len(transcripts())} transcripts and {SAMPLE_DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
