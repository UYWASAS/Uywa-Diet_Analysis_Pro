from __future__ import annotations

import math

from pcta.core.calculations import compute_unit_kpis
from pcta.core.schemas import TrialUnitInput, UnitType


def _unit(**overrides) -> TrialUnitInput:
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
        "bw_final_sd_g": 250.0,
        "final_sample_n": 100,
        "diet_cost_per_kg": 0.5,
        "additive_cost_total": 100.0,
        "chick_cost_per_bird": 0.4,
        "other_variable_costs_total": 50.0,
    }
    data.update(overrides)
    return TrialUnitInput(**data)


def test_basic_kpis() -> None:
    u = _unit()
    k, warnings = compute_unit_kpis(u)
    assert k.feed_consumed_kg == 2950.0
    assert k.birds_end == 990  # placed - mortality
    assert math.isclose(k.mortality_pct, 1.0, rel_tol=1e-9)

    wg = 2500.0 - 42.0
    assert math.isclose(k.wg_g_per_bird, wg, rel_tol=1e-9)
    assert math.isclose(k.adg_g_per_bird_per_day, wg / 35.0, rel_tol=1e-9)

    # total LWG kg = wg*g * birds / 1e6
    assert math.isclose(k.total_liveweight_gain_kg, wg * 990 / 1_000_000.0, rel_tol=1e-9)

    # CV
    assert math.isclose(k.cv_final_pct or 0.0, 250.0 / 2500.0 * 100.0, rel_tol=1e-9)

    # Economics
    assert math.isclose(k.feed_cost_total or 0.0, 2950.0 * 0.5, rel_tol=1e-9)
    expected_total_cost = (2950.0 * 0.5) + 100.0 + (0.4 * 1000) + 50.0
    assert math.isclose(k.total_cost or 0.0, expected_total_cost, rel_tol=1e-9)


def test_wg_nonpositive_sets_fcr_null() -> None:
    u = _unit(bw_initial_mean_g=2500.0, bw_final_mean_g=2500.0)
    k, warnings = compute_unit_kpis(u)
    assert k.wg_g_per_bird == 0.0
    assert k.fcr is None
    assert any(w.code.value == "wg_nonpositive_fcr_nan" for w in warnings)
