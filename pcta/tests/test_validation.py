from __future__ import annotations

import pytest

from pcta.core.schemas import TrialUnitInput, UnitType
from pcta.core.validation import ValidationOptions, validate_units


def _base_unit(**overrides) -> TrialUnitInput:
    data = {
        "trial_id": "T1",
        "unit_type": UnitType.house,
        "unit_id": "H1",
        "treatment": "A",
        "days": 35,
        "birds_placed": 1000,
        "mortality_total": 10,
        "birds_sold": None,
        "feed_delivered_kg": 3000.0,
        "feed_refusals_kg": 50.0,
        "bw_initial_mean_g": 42.0,
        "bw_final_mean_g": 2500.0,
        "bw_final_sd_g": None,
        "final_sample_n": None,
        "diet_cost_per_kg": None,
        "additive_cost_total": 0.0,
        "chick_cost_per_bird": 0.0,
        "other_variable_costs_total": 0.0,
    }
    data.update(overrides)
    return TrialUnitInput(**data)


def test_validation_errors_block() -> None:
    u = _base_unit(birds_placed=0)
    with pytest.raises(ValueError):
        validate_units([u])

    u2 = _base_unit(mortality_total=2000)
    with pytest.raises(ValueError):
        validate_units([u2])

    u3 = _base_unit(feed_refusals_kg=4000.0)
    with pytest.raises(ValueError):
        validate_units([u3])

    u4 = _base_unit(days=0)
    with pytest.raises(ValueError):
        validate_units([u4])


def test_validation_warnings_missing_sd_and_sample_n() -> None:
    u = _base_unit(bw_final_sd_g=None, final_sample_n=None)
    _, warnings = validate_units([u])
    codes = {w.code.value for w in warnings}
    assert "missing_bw_sd_uniformity_unavailable" in codes
    assert "missing_sample_n_ci_unavailable" in codes


def test_validation_replication_warning() -> None:
    u1 = _base_unit(unit_id="H1", treatment="A")
    u2 = _base_unit(unit_id="H2", treatment="B")
    _, warnings = validate_units([u1, u2])
    codes = {w.code.value for w in warnings}
    assert "inferential_disabled_no_replication" in codes


def test_negative_wg_blocks_when_configured() -> None:
    u = _base_unit(bw_initial_mean_g=2500.0, bw_final_mean_g=2400.0)
    with pytest.raises(ValueError):
        validate_units([u], options=ValidationOptions(wg_negative_is_error=True))

    # When not configured, it should not block
    _, warnings = validate_units([u], options=ValidationOptions(wg_negative_is_error=False))
    assert isinstance(warnings, list)
