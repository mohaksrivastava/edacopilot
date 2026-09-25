"""Persona policies load, validate, and say what Section 8.1 says.

A persona is data, so the schema is the only thing standing between a typo
in a YAML file and a persona that silently stops applying one of its own
rules. These tests are mostly about that.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from edacopilot.personas import (
    PERSONA_ORDER,
    PREFERENCE_KEYS,
    PersonaPolicy,
    get_persona,
    load_personas,
    load_policy,
)
from edacopilot.personas.policy import PERSONA_DIR


def _raw(persona_id: str) -> dict:
    with (PERSONA_DIR / f"{persona_id}.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_all_three_personas_load_in_a_fixed_order() -> None:
    """Fixed, not alphabetical or filesystem order: a card that reordered
    itself between runs would look like a changed recommendation."""
    policies = load_personas()
    assert [p.id for p in policies] == list(PERSONA_ORDER)


def test_each_persona_matches_section_8_1() -> None:
    professor, consultant, maverick = load_personas()

    assert professor.normality.clt_shortcut == "none"
    assert professor.normality.borderline_is == "fail"
    assert professor.method_pool_tags == {"parametric", "nonparametric", "exact"}

    assert consultant.normality.clt_shortcut == "cochran"
    assert consultant.normality.borderline_is == "pass"
    assert consultant.method_pool_tags == {"parametric", "nonparametric"}
    assert consultant.avoids_equal_variance_methods

    assert maverick.normality.clt_shortcut == "none"
    assert maverick.method_pool_tags == {
        "resampling",
        "robust",
        "nonparametric_advanced",
        "maverick",
    }
    assert maverick.variance is None


def test_only_the_consultant_has_a_clt_shortcut() -> None:
    """M4.1 made the shortcut Cochran's rule and left the professor without
    one, precisely because Cochran's rule bounds skew and says nothing
    about heavy tails."""
    with_shortcut = [p.id for p in load_personas() if p.normality.clt_shortcut != "none"]
    assert with_shortcut == ["consultant"]


def test_every_preference_key_is_implemented() -> None:
    """A preference the pick algorithm does not implement would be silently
    skipped, leaving the persona ranking by something other than its file."""
    for policy in load_personas():
        assert set(policy.prefer) <= set(PREFERENCE_KEYS), policy.id


def test_an_unknown_preference_key_is_rejected_at_load() -> None:
    raw = _raw("professor") | {"prefer": ["assumptions_clearly_met", "vibes"]}
    with pytest.raises(ValidationError, match="unknown preference key"):
        PersonaPolicy.model_validate(raw)


def test_a_misspelled_top_level_key_is_rejected() -> None:
    """`extra="forbid"` earns its keep here: `method_pool_tag` would
    otherwise be ignored, leaving the persona with no method pool and no
    indication why it stopped proposing anything."""
    raw = _raw("professor")
    raw["method_pool_tag"] = raw.pop("method_pool_tags")
    with pytest.raises(ValidationError):
        PersonaPolicy.model_validate(raw)


def test_a_misspelled_nested_key_is_rejected() -> None:
    raw = _raw("consultant")
    raw["normality"]["clt_shortcuts"] = raw["normality"].pop("clt_shortcut")
    with pytest.raises(ValidationError):
        PersonaPolicy.model_validate(raw)


def test_an_unsupported_clt_shortcut_is_rejected() -> None:
    """The shortcut is a named rule, not free text: `cochran` or `none`."""
    raw = _raw("consultant")
    raw["normality"]["clt_shortcut"] = {"min_n_per_group": 30, "max_abs_skew": 2}
    with pytest.raises(ValidationError):
        PersonaPolicy.model_validate(raw)


def test_alpha_must_be_a_probability() -> None:
    with pytest.raises(ValidationError):
        PersonaPolicy.model_validate(_raw("professor") | {"alpha": 1.5})


def test_empty_method_pool_is_rejected() -> None:
    with pytest.raises(ValidationError):
        PersonaPolicy.model_validate(_raw("professor") | {"method_pool_tags": []})


def test_a_file_whose_id_does_not_match_its_name_is_rejected(tmp_path: Path) -> None:
    for persona_id in PERSONA_ORDER:
        raw = _raw(persona_id)
        if persona_id == "maverick":
            raw["id"] = "rebel"
        (tmp_path / f"{persona_id}.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    load_personas.cache_clear()
    with pytest.raises(ValueError, match="declares id 'rebel'"):
        load_personas(tmp_path)
    load_personas.cache_clear()


def test_a_fourth_persona_is_rejected(tmp_path: Path) -> None:
    """Section 8 defines three. A fourth changes what a divergence card
    means, so it needs a spec change, not a dropped file."""
    for persona_id in PERSONA_ORDER:
        (tmp_path / f"{persona_id}.yaml").write_text(
            yaml.safe_dump(_raw(persona_id)), encoding="utf-8"
        )
    (tmp_path / "skeptic.yaml").write_text(
        yaml.safe_dump(_raw("professor") | {"id": "skeptic"}), encoding="utf-8"
    )
    load_personas.cache_clear()
    with pytest.raises(ValueError, match="unexpected persona file"):
        load_personas(tmp_path)
    load_personas.cache_clear()


def test_a_missing_persona_file_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "professor.yaml").write_text(yaml.safe_dump(_raw("professor")), encoding="utf-8")
    load_personas.cache_clear()
    with pytest.raises(FileNotFoundError, match="missing"):
        load_personas(tmp_path)
    load_personas.cache_clear()


def test_get_persona_round_trips_and_rejects_unknown() -> None:
    assert get_persona("maverick").display_name == "Maverick"
    with pytest.raises(KeyError):
        get_persona("nobody")


def test_load_policy_rejects_a_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "professor.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_policy(path)
