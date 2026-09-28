"""Section 10.2: switching providers is a config change, never a code
change. Each profile below must resolve to a valid `litellm.completion`
call shape (model + api_base) without making a network call."""

from __future__ import annotations

from edacopilot.llm.config import (
    DEFAULT_MODEL,
    EdacopilotConfig,
    LLMConfig,
    ProviderProfile,
    load_config,
)


def test_the_default_resolves_to_gemini() -> None:
    config = LLMConfig()
    model, api_base = config.resolve("parse_intent")
    assert model == DEFAULT_MODEL == "gemini/gemini-3.8-flash"
    assert api_base is None


def test_an_ollama_profile_resolves_without_touching_the_default() -> None:
    config = LLMConfig(
        active_profile="ollama",
        profiles={
            "ollama": ProviderProfile(
                default_model="ollama/qwen2.5:7b-instruct", api_base="http://localhost:11434"
            )
        },
    )
    model, api_base = config.resolve("parse_intent")
    assert model == "ollama/qwen2.5:7b-instruct"
    assert api_base == "http://localhost:11434"


def test_an_anthropic_override_is_a_config_change_only() -> None:
    config = LLMConfig(default_model="anthropic/claude-haiku-4-5")
    model, _ = config.resolve("write_rationale")
    assert model == "anthropic/claude-haiku-4-5"


def test_rationale_model_overrides_only_the_rationale_call() -> None:
    config = LLMConfig(
        default_model="gemini/gemini-3.8-flash", rationale_model="gemini/gemini-3.8-pro"
    )
    assert config.resolve("write_rationale")[0] == "gemini/gemini-3.8-pro"
    assert config.resolve("parse_intent")[0] == "gemini/gemini-3.8-flash"


def test_a_profiles_rationale_model_overrides_only_the_rationale_call() -> None:
    config = LLMConfig(
        active_profile="local",
        profiles={
            "local": ProviderProfile(
                default_model="ollama/qwen2.5:7b-instruct",
                rationale_model="ollama/qwen2.5:14b-instruct",
            )
        },
    )
    assert config.resolve("write_rationale")[0] == "ollama/qwen2.5:14b-instruct"
    assert config.resolve("parse_intent")[0] == "ollama/qwen2.5:7b-instruct"


def test_toml_and_kwargs_merge_with_kwargs_winning(tmp_path) -> None:  # type: ignore[no-untyped-def]
    toml_path = tmp_path / "edacopilot.toml"
    toml_path.write_text(
        '[llm]\ndefault_model = "anthropic/claude-haiku-4-5"\n[privacy]\nlevel = "strict"\n'
    )
    config = load_config({"llm": {"default_model": "gemini/gemini-3.8-flash"}}, toml_path=toml_path)
    assert config.llm.default_model == "gemini/gemini-3.8-flash"
    assert config.privacy.level == "strict"  # untouched by the override, still read from the file


def test_missing_toml_file_falls_back_to_defaults(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = load_config(toml_path=tmp_path / "does_not_exist.toml")
    assert config == EdacopilotConfig()


def test_an_unknown_field_is_a_load_time_error() -> None:
    """Section 8.1's pattern for persona YAML applies here too: a
    misspelled key should fail loudly, not silently do nothing."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        LLMConfig.model_validate({"defalut_model": "typo"})
