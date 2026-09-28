"""LLM and privacy configuration (ARCHITECTURE.md, Section 10.2).

Loaded from `edacopilot.toml` in the current working directory if one
exists, then overridden by `start(..., config={...})` kwargs -- kwargs win,
matching how `start()` already merges a plain `config` dict today. No
`edacopilot.toml` needs to exist: every field has a default, and the
defaults are exactly Section 10.2's table.

Switching provider is a config change, never a code change (Section 1.3's
"provider-agnostic" principle): either point `default_model` at a different
`litellm` model string directly, or set `active_profile` to a name in
`profiles` (the Ollama profile below is one such profile, wired and tested
without being run live).
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_TOML_NAME = "edacopilot.toml"

# gemini/gemini-3.8-flash: chosen 2026-09-28 as the current GA Flash-tier
# Gemini model (Google's own model page banners it as newly released),
# reached through LiteLLM's `gemini/` prefix, which reads GEMINI_API_KEY
# directly. See ARCHITECTURE.md Section 18's M9 entry for why this replaced
# the spec's original Anthropic default.
DEFAULT_MODEL = "gemini/gemini-3.8-flash"


class ProviderProfile(BaseModel):
    """One named, swappable provider configuration (Section 10.2)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    default_model: str
    rationale_model: str | None = None
    api_base: str | None = None


class LLMConfig(BaseModel):
    """Section 10.2's `[llm]` table."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    default_model: str = DEFAULT_MODEL
    rationale_model: str | None = None
    deterministic_mode: bool = False
    timeout: float = 30.0
    num_retries: int = 2
    api_base: str | None = None
    # Name of an entry in `profiles` to use instead of the fields above, or
    # None to use them directly. Switching this one value is the whole of
    # "switch providers via config only".
    active_profile: str | None = None
    profiles: dict[str, ProviderProfile] = Field(default_factory=dict)

    def resolve(self, call: str) -> tuple[str, str | None]:
        """The `(model, api_base)` a given call type should use.

        `write_rationale` is the one call with its own model slot
        (Section 10.2's `rationale_model`), because it is the one call that
        produces prose rather than a slot-filled schema and so is the one
        place a maintainer is likeliest to want a different tier.
        """
        profile = self.profiles.get(self.active_profile) if self.active_profile else None
        if profile is not None:
            model = (
                profile.rationale_model
                if call == "write_rationale" and profile.rationale_model
                else profile.default_model
            )
            return model, profile.api_base
        model = (
            self.rationale_model
            if call == "write_rationale" and self.rationale_model
            else self.default_model
        )
        return model, self.api_base


class PrivacyConfig(BaseModel):
    """Section 11's `[privacy]` table."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    level: Literal["standard", "strict"] = "standard"
    alias_column_names: bool = False


class EdacopilotConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    llm: LLMConfig = Field(default_factory=LLMConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    overrides: dict[str, Any] | None = None, *, toml_path: Path | None = None
) -> EdacopilotConfig:
    """Section 10.2's config: file, then kwargs, then pydantic's own defaults."""
    path = toml_path if toml_path is not None else Path.cwd() / DEFAULT_TOML_NAME
    data: dict[str, Any] = {}
    if path.is_file():
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    merged = _deep_merge(data, overrides or {})
    return EdacopilotConfig.model_validate(merged)
