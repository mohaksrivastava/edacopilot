"""Loads a prompt template and assembles the final message (Section 10.3).

One `.md` file per call: role statement, Section 10.3's six hard rules,
2-3 few-shot examples, and a `{{context}}` slot. The JSON schema is *not*
baked into the file -- it is appended here, live, from the pydantic model
each call actually validates against, so the schema in the prompt can
never drift from the schema the response is checked against.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel

    from edacopilot.llm.context import PromptContext

PROMPTS_DIR = Path(__file__).parent

HARD_RULES = """\
- You do not calculate statistics. Use only numbers present in CONTEXT.
- Refer to facts by their fact_id in cited_facts.
- Never recommend a method not present in CANDIDATES.
- Never mention methods marked INELIGIBLE except to explain why they are excluded.
- Keep each field within its length limit. Plain language, no jargon without a gloss.
- Output JSON only, matching SCHEMA."""


def load_template(call: str) -> str:
    return (PROMPTS_DIR / f"{call}.md").read_text(encoding="utf-8")


def render_context(context: PromptContext) -> str:
    """CONTEXT, as JSON -- the model reads it, never edits it."""
    payload = {
        "privacy_level": context.privacy_level,
        "stage": context.stage,
        "dataset": context.dataset_summary,
        "columns": [
            {
                "name": col.name,
                "type": col.semantic_type,
                "n": col.n,
                "n_missing": col.n_missing,
                "summary": col.summary,
                **({"top_levels": col.top_levels} if col.top_levels is not None else {}),
                **({"pii": True} if col.is_pii else {}),
            }
            for col in context.columns
        ],
        "session_summary": context.session_summary,
        "recent_turns": [{"role": role, "text": text} for role, text in context.recent_turns],
    }
    return json.dumps(payload, indent=2, default=str)


def build_messages(
    call: str,
    context: PromptContext,
    schema: type[BaseModel],
    extra: dict[str, object] | None = None,
) -> list[dict[str, str]]:
    """The full message list for one call: template + live schema + context."""
    template = load_template(call)
    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    body = template.replace("{{context}}", render_context(context))
    if extra:
        body += "\n\nADDITIONAL INPUT:\n" + json.dumps(extra, indent=2, default=str)
    body += f"\n\nSCHEMA:\n{schema_json}\n\nHARD RULES:\n{HARD_RULES}"
    return [{"role": "user", "content": body}]
