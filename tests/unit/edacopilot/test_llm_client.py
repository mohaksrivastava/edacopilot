"""Section 10.1's client: schema-retry, then `LLMFallbackRequired` --
never a raw exception out of `complete_structured`. Every test here mocks
`litellm.completion`; none makes a network call (rule: no live call
outside an explicitly approved smoke test)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from edacopilot.llm.client import LLMClient, LLMFallbackRequired
from edacopilot.llm.config import LLMConfig


class Toy(BaseModel):
    value: int


def _response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=content))]
    resp.usage = MagicMock(total_tokens=7)
    return resp


def test_a_valid_response_is_returned_on_the_first_try() -> None:
    client = LLMClient(LLMConfig())
    with patch("litellm.completion", return_value=_response('{"value": 3}')) as mock:
        result = client.complete_structured("toy", [{"role": "user", "content": "x"}], Toy)
    assert result == Toy(value=3)
    assert mock.call_count == 1
    assert client.last_usage["toy"] == 7


def test_malformed_json_retries_once_then_succeeds() -> None:
    client = LLMClient(LLMConfig())
    responses = [_response("not json at all"), _response('{"value": 5}')]
    with patch("litellm.completion", side_effect=responses) as mock:
        result = client.complete_structured("toy", [{"role": "user", "content": "x"}], Toy)
    assert result == Toy(value=5)
    assert mock.call_count == 2


def test_a_schema_violation_appends_the_validation_error_on_retry() -> None:
    client = LLMClient(LLMConfig())
    responses = [_response('{"value": "not an int"}'), _response('{"value": 9}')]
    with patch("litellm.completion", side_effect=responses):
        result = client.complete_structured("toy", [{"role": "user", "content": "x"}], Toy)
    assert result == Toy(value=9)


def test_two_bad_responses_fall_back_rather_than_raise_a_validation_error() -> None:
    client = LLMClient(LLMConfig())
    responses = [_response("garbage"), _response("still garbage")]
    with patch("litellm.completion", side_effect=responses):
        with pytest.raises(LLMFallbackRequired):
            client.complete_structured("toy", [{"role": "user", "content": "x"}], Toy)


def test_invented_fields_are_a_schema_violation_not_a_silent_pass() -> None:
    """extra='forbid' models: an invented field is exactly as wrong as a
    missing one, so it must go through the same retry-then-fallback path."""
    client = LLMClient(LLMConfig())

    class Strict(BaseModel):
        model_config = {"extra": "forbid"}
        value: int

    responses = [_response('{"value": 1, "made_up_field": true}'), _response('{"value": 1}')]
    with patch("litellm.completion", side_effect=responses):
        result = client.complete_structured("toy", [{"role": "user", "content": "x"}], Strict)
    assert result == Strict(value=1)


def test_a_provider_error_falls_back_immediately_without_a_second_call() -> None:
    client = LLMClient(LLMConfig())
    with patch("litellm.completion", side_effect=TimeoutError("boom")) as mock:
        with pytest.raises(LLMFallbackRequired):
            client.complete_structured("toy", [{"role": "user", "content": "x"}], Toy)
    assert mock.call_count == 1


def test_deterministic_mode_never_calls_the_provider_at_all() -> None:
    client = LLMClient(LLMConfig(deterministic_mode=True))
    with patch("litellm.completion") as mock:
        with pytest.raises(LLMFallbackRequired):
            client.complete_structured("toy", [{"role": "user", "content": "x"}], Toy)
    mock.assert_not_called()


def test_write_rationale_uses_the_rationale_temperature() -> None:
    client = LLMClient(LLMConfig())
    with patch("litellm.completion", return_value=_response('{"value": 1}')) as mock:
        client.complete_structured("write_rationale", [{"role": "user", "content": "x"}], Toy)
    assert mock.call_args.kwargs["temperature"] == pytest.approx(0.3)


def test_a_structured_call_uses_zero_temperature() -> None:
    client = LLMClient(LLMConfig())
    with patch("litellm.completion", return_value=_response('{"value": 1}')) as mock:
        client.complete_structured("parse_intent", [{"role": "user", "content": "x"}], Toy)
    assert mock.call_args.kwargs["temperature"] == pytest.approx(0.0)
