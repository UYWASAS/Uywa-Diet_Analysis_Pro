"""
Core calculation routines for PCTA.

This module computes per-unit commercial poultry metrics and economics.

Commercial calculations (spec):
- feed_consumed_kg = feed_delivered_kg - feed_refusals_kg
- birds_end = birds_sold if exists else birds_placed - mortality_total
- WG_g_per_bird = bw_final_mean_g - bw_initial_mean_g
- ADG = WG / days
- total_liveweight_gain_kg = WG * birds_end / 1e6
- FCR = feed_consumed_kg / total_liveweight_gain_kg
- CV_final = bw_sd / bw_mean * 100 if available

Economics (spec):
- feed_cost_total = feed_consumed_kg * diet_cost_per_kg
- total_cost = feed_cost_total + additive_cost_total + chick_cost_per_bird * birds_placed + other_variable_costs_total
- cost_per_kg_sold = total_cost / (bw_final_mean_g/1000 * birds_end)

Notes:
- Validation (e.g., refusals <= delivered, mortality <= placed, WG >= 0) is
  enforced in validation layer. This module is defensive and emits warnings
  for division-by-zero and missing optional fields.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .schemas import AnalysisWarning, ComputedMetrics, TrialUnitInput, WarningCode


def _safe_div(numer: float, denom: float) -> Optional[float]:
    if denom == 0:
        return None
    return numer / denom


def compute_unit_metrics(unit: TrialUnitInput) -> Tuple[ComputedMetrics, List[AnalysisWarning]]:
    """
    Compute all derived metrics for a single unit.

    Returns:
        (ComputedMetrics, warnings)
    """
    warnings: List[AnalysisWarning] = []

    feed_consumed_kg = unit.feed_delivered_kg - unit.feed_refusals_kg

    if unit.birds_sold is not None:
        birds_end = unit.birds_sold
    else:
        birds_end = unit.birds_placed - unit.mortality_total
        warnings.append(
            AnalysisWarning(
                code=WarningCode.missing_birds_sold_used_estimate,
                message="birds_sold missing; birds_end estimated as birds_placed - mortality_total.",
                context={
                    "unit_id": unit.unit_id,
                    "birds_placed": unit.birds_placed,
                    "mortality_total": unit.mortality_total,
                    "birds_end": birds_end,
                },
            )
        )

    wg_g_per_bird = unit.bw_final_mean_g - unit.bw_initial_mean_g
    adg = wg_g_per_bird / unit.days

    total_liveweight_gain_kg = (wg_g_per_bird * birds_end) / 1_000_000.0

    fcr = _safe_div(feed_consumed_kg, total_liveweight_gain_kg)
    if fcr is None:
        warnings.append(
            AnalysisWarning(
                code=WarningCode.division_by_zero,
                message="FCR could not be computed due to zero total liveweight gain.",
                context={"unit_id": unit.unit_id, "total_liveweight_gain_kg": total_liveweight_gain_kg},
            )
        )

    cv_final_pct: Optional[float] = None
    if unit.bw_final_sd_g is None:
        warnings.append(
            AnalysisWarning(
                code=WarningCode.missing_bw_sd_cv_unavailable,
                message="bw_final_sd_g missing; CV_final_pct not computed.",
                context={"unit_id": unit.unit_id},
            )
        )
    else:
        cv_final_pct = _safe_div(unit.bw_final_sd_g, unit.bw_final_mean_g)
        if cv_final_pct is None:
            warnings.append(
                AnalysisWarning(
                    code=WarningCode.division_by_zero,
                    message="CV_final_pct could not be computed due to zero bw_final_mean_g.",
                    context={"unit_id": unit.unit_id, "bw_final_mean_g": unit.bw_final_mean_g},
                )
            )
        else:
            cv_final_pct *= 100.0

    # Economics
    feed_cost_total: Optional[float] = None
    if unit.diet_cost_per_kg is not None:
        feed_cost_total = feed_consumed_kg * unit.diet_cost_per_kg

    total_cost: Optional[float] = None
    if feed_cost_total is not None and unit.chick_cost_per_bird is not None:
        total_cost = (
            feed_cost_total
            + unit.additive_cost_total
            + unit.chick_cost_per_bird * unit.birds_placed
            + unit.other_variable_costs_total
        )

    cost_per_kg_sold: Optional[float] = None
    sold_kg = (unit.bw_final_mean_g / 1000.0) * birds_end
    if total_cost is not None:
        cost_per_kg_sold = _safe_div(total_cost, sold_kg)
        if cost_per_kg_sold is None:
            warnings.append(
                AnalysisWarning(
                    code=WarningCode.division_by_zero,
                    message="cost_per_kg_sold could not be computed due to zero sold liveweight.",
                    context={"unit_id": unit.unit_id, "sold_kg": sold_kg},
                )
            )

    computed = ComputedMetrics(
        unit_id=unit.unit_id,
        treatment=unit.treatment,
        unit_type=unit.unit_type,
        days=unit.days,
        birds_placed=unit.birds_placed,
        mortality_total=unit.mortality_total,
        birds_end=birds_end,
        feed_delivered_kg=unit.feed_delivered_kg,
        feed_refusals_kg=unit.feed_refusals_kg,
        feed_consumed_kg=feed_consumed_kg,
        bw_initial_mean_g=unit.bw_initial_mean_g,
        bw_final_mean_g=unit.bw_final_mean_g,
        wg_g_per_bird=wg_g_per_bird,
        adg_g_per_bird_per_day=adg,
        total_liveweight_gain_kg=total_liveweight_gain_kg,
        fcr=fcr,
        cv_final_pct=cv_final_pct,
        diet_cost_per_kg=unit.diet_cost_per_kg,
        feed_cost_total=feed_cost_total,
        additive_cost_total=unit.additive_cost_total,
        chick_cost_per_bird=unit.chick_cost_per_bird,
        other_variable_costs_total=unit.other_variable_costs_total,
        total_cost=total_cost,
        cost_per_kg_sold=cost_per_kg_sold,
    )
    return computed, warnings


def compute_all_units(units: List[TrialUnitInput]) -> Tuple[List[ComputedMetrics], List[AnalysisWarning]]:
    """
    Compute metrics for all units.

    Returns:
        (computed_metrics_list, warnings)
    """
    all_metrics: List[ComputedMetrics] = []
    all_warnings: List[AnalysisWarning] = []
    for u in units:
        m, w = compute_unit_metrics(u)
        all_metrics.append(m)
        all_warnings.extend(w)
    return all_metrics, all_warnings
