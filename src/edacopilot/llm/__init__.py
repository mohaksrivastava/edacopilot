"""The LLM layer (ARCHITECTURE.md, Section 10).

Only Section 10.5's deterministic fallback exists in this build. The client,
the prompt templates, the fact-check and the ContextBuilder arrive with M9;
until then `deterministic_mode` is the only mode, which is rule 9 taken at
its word rather than as an aspiration.
"""

from __future__ import annotations

from .fallback import (
    NOT_IN_GLOSSARY,
    SpecFormField,
    answer_free_question,
    build_question_spec_form,
    glossary_terms,
    load_glossary,
    normalise_topic,
    parse_intent,
)

__all__ = [
    "NOT_IN_GLOSSARY",
    "SpecFormField",
    "answer_free_question",
    "build_question_spec_form",
    "glossary_terms",
    "load_glossary",
    "normalise_topic",
    "parse_intent",
]
