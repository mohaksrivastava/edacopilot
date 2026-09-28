"""The LLM client (ARCHITECTURE.md, Section 10.1).

`litellm.completion` behind a client that: reads the model per call type
from config, retries twice with backoff on a schema-validation failure
(the error text appended to the retry, per Section 10.1), and otherwise
raises `LLMFallbackRequired` rather than propagating a provider error.

That last point is the one seam the whole milestone leans on: "the LLM
never computes a statistic" (rule 1) and "the system must run with the LLM
switched off" (rule 9) both require that a model failure never becomes an
exception the user sees. Every one of Section 3.4's seven calls catches
`LLMFallbackRequired` and runs Section 10.5's deterministic path instead --
one seam, not seven duplicated try/except blocks.
"""

from __future__ import annotations

from typing import Any, TypeVar

import litellm
from pydantic import BaseModel, ValidationError

from .config import LLMConfig

ModelT = TypeVar("ModelT", bound=BaseModel)

# Temperature 0 for every structured call, 0.3 for write_rationale, the one
# call that produces prose rather than a slot-filled schema (Section 10.1).
_RATIONALE_TEMPERATURE = 0.3
_STRUCTURED_TEMPERATURE = 0.0


class LLMFallbackRequired(Exception):
    """Raised when a call could not get a valid structured answer.

    The caller catches this and runs the deterministic fallback
    (`llm/fallback.py` or the orchestrator's own rule-based path) --
    never propagated to the user as an error (Section 10.5).
    """

    def __init__(self, call: str, reason: str) -> None:
        super().__init__(f"{call}: falling back to deterministic mode ({reason})")
        self.call = call
        self.reason = reason


class LLMClient:
    """Thin wrapper over `litellm.completion` (Section 10.1)."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        # Token usage per call type, for Section 10.4's budget report.
        # Overwritten each call rather than accumulated -- this is "how
        # much did the last call to each type cost", not a running total.
        self.last_usage: dict[str, int] = {}

    def complete_structured(
        self,
        call: str,
        messages: list[dict[str, str]],
        schema: type[ModelT],
    ) -> ModelT:
        """One structured call, validated against `schema`.

        On a `ValidationError`, one retry with the error text appended to
        the prompt (Section 10.1). On a second failure, or any error
        reaching the provider at all (timeout, network, auth), raises
        `LLMFallbackRequired` -- deterministic mode's failure path, not an
        exception the caller has to translate itself.
        """
        if self.config.deterministic_mode:
            raise LLMFallbackRequired(call, "deterministic_mode is on")

        model, api_base = self.config.resolve(call)
        temperature = (
            _RATIONALE_TEMPERATURE if call == "write_rationale" else _STRUCTURED_TEMPERATURE
        )
        attempt_messages = list(messages)
        last_error: Exception | None = None

        for _ in range(2):
            try:
                response = litellm.completion(
                    model=model,
                    messages=attempt_messages,
                    api_base=api_base,
                    temperature=temperature,
                    timeout=self.config.timeout,
                    num_retries=self.config.num_retries,
                    response_format=schema,
                )
            except Exception as exc:  # noqa: BLE001 - any provider failure falls back
                last_error = exc
                break

            usage = getattr(response, "usage", None)
            if usage is not None:
                self.last_usage[call] = int(getattr(usage, "total_tokens", 0) or 0)
            content = _content_of(response)
            try:
                return schema.model_validate_json(content)
            except ValidationError as exc:
                last_error = exc
                attempt_messages = [
                    *messages,
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            "That response did not match the required JSON schema. "
                            f"Fix it and resend, matching this validation error exactly:\n{exc}"
                        ),
                    },
                ]
                continue

        raise LLMFallbackRequired(call, str(last_error))

    def token_count(self, call: str, messages: list[dict[str, str]]) -> int:
        """Actual prompt token count for Section 10.4's budget report."""
        model, _ = self.config.resolve(call)
        return int(litellm.token_counter(model=model, messages=messages))


def _content_of(response: Any) -> str:
    """The text of a `litellm.completion` response's first choice."""
    return response.choices[0].message.content or ""
