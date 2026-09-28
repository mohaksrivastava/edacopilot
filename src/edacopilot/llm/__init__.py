"""The LLM layer (ARCHITECTURE.md, Section 10).

`LLMClient` (`client.py`), the seven calls (`calls.py`), the prompt
templates (`prompts/`), the fact-check (`factcheck.py`) and the
`ContextBuilder` (`context.py`) all arrived with M9. This module's own
`__all__` still only re-exports Section 10.5's deterministic fallback
(`fallback.py`) -- the rest are imported by their submodule path, lazily,
at each call site that needs them, the same way `Orchestrator.module()`
already did for its own import cycle. Eagerly importing them here would
recreate that cycle: `calls.py` reaches `edacopilot.orchestrator.intents`,
and this package is itself first loaded *from inside*
`edacopilot.orchestrator.loop`'s own module body.

`deterministic_mode` is still exactly what rule 9 promises: with it on, or
with no model reachable at all, every call in `calls.py` raises
`LLMFallbackRequired` and its caller runs the same deterministic path this
module provided before M9 existed.
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
