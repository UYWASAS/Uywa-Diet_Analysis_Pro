"""
KPI calculations for PCTA.

Per-unit calculations (spec):
- feed_consumed_kg = feed_delivered_kg - feed_refusals_kg
- birds_end = birds_sold else birds_placed - mortality_total
- mortality_pct = mortality_total / birds_placed * 100
- WG = bw_final_mean_g - bw_initial_mean_g
- ADG = WG / days
- total_liveweight_gain_kg = WG * birds_end / 1e6
- FCR = feed_consumed_kg / total_liveweight_gain_kg (WG<=0 => NaN + warning)
- CV_final_pct = bw_final_sd_g / bw_final_mean_g * 100 (if sd available)

Economics:
- feed_cost_total = feed_consumed_kg * diet_cost_per_kg (if diet_cost_per_kg provided)
- total_cost = feed_cost_total + additive_cost_total + chick_cost_per_bird*birds_placed + other_variable_costs_total
- kg_sold_est = (bw_final_mean_g/1000) * birds_end
- cost_per_kg_sold = total_cost / kg_sold_est
- cost_per_bird_sold = total_cost / birds_end
- cost_per_kg_gain = total_cost / total_liveweight_gain_kg

This module is defensive (division-by-zero -> None) and emits warnings, but it
assumes blocking validation has already happened in validation.py.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .schemas import AnalysisWarning, ComputedUnitKPIs, TrialUnitInput, WarningCode


def _safe_div(numer: float, denom: float) -> Optional[float]:
    if denom == 0:
        return None
    return numer / denom


def compute_unit_kpis(unit: TrialUnitInput) -> Tuple[ComputedUnitKPIs, List[AnalysisWarning]]:
    warnings: List[AnalysisWarning] = []

    feed_consumed_kg = unit.feed_delivered_kg - unit.feed_refusals_kg

    birds_end = unit.birds_sold if unit.birds_sold is not None else (unit.birds_placed - unit.mortality_total)
    mortality_pct = (unit.mortality_total / unit.birds_placed) * 100.0

    wg = unit.bw_final_mean_g - unit.bw_initial_mean_g
    adg = wg / unit.days

    total_liveweight_gain_kg = (wg * birds_end) / 1_000_000.0

    fcr: Optional[float]
    if wg <= 0 or total_liveweight_gain_kg <= 0:
        fcr = None
        warnings.append(
            AnalysisWarning(
                code=WarningCode.wg_nonpositive_fcr_nan,
                message="WG <= 0; FCR set to null (cannot compute).",
                context={"trial_id": unit.trial_id, "unit_id": unit.unit_id, "wg_g_per_bird": wg},
            )
        )
    else:
        fcr = _safe_div(feed_consumed_kg, total_liveweight_gain_kg)
        if fcr is None:
            warnings.append(
                AnalysisWarning(
                    code=WarningCode.division_by_zero,
                    message="FCR could not be computed due to zero total liveweight gain.",
                    context={"trial_id": unit.trial_id, "unit_id": unit.unit_id, "total_liveweight_gain_kg": total_liveweight_gain_kg},
                )
            )

    cv_final_pct: Optional[float] = None
    if unit.bw_final_sd_g is not None:
        cv = _safe_div(unit.bw_final_sd_g, unit.bw_final_mean_g)
        if cv is None:
            warnings.append(
                AnalysisWarning(
                    code=WarningCode.division_by_zero,
                    message="CV_final_pct could not be computed due to zero final BW mean.",
                    context={"trial_id": unit.trial_id, "unit_id": unit.unit_id, "bw_final_mean_g": unit.bw_final_mean_g},
                )
            )
        else:
            cv_final_pct = cv * 100.0

    # Economics
    feed_cost_total: Optional[float] = None
    if unit.diet_cost_per_kg is not None:
        feed_cost_total = feed_consumed_kg * unit.diet_cost_per_kg

    total_cost: Optional[float] = None
    if feed_cost_total is not None:
        total_cost = (
            feed_cost_total
            + unit.additive_cost_total
            + (unit.chick_cost_per_bird * unit.birds_placed)
            + unit.other_variable_costs_total
        )

    kg_sold_est = (unit.bw_final_mean_g / 1000.0) * birds_end

    cost_per_kg_sold: Optional[float] = None
    cost_per_bird_sold: Optional[float] = None
    cost_per_kg_gain: Optional[float] = None

    if total_cost is not None:
        cost_per_kg_sold = _safe_div(total_cost, kg_sold_est)
        if cost_per_kg_sold is None:
            warnings.append(
                AnalysisWarning(
                    code=WarningCode.division_by_zero,
                    message="cost_per_kg_sold could not be computed due to zero kg_sold_est.",
                    context={"trial_id": unit.trial_id, "unit_id": unit.unit_id, "kg_sold_est": kg_sold_est},
                )
            )

        cost_per_bird_sold = _safe_div(total_cost, float(birds_end))
        if cost_per_bird_sold is None:
            warnings.append(
                AnalysisWarning(
                    code=WarningCode.division_by_zero,
                    message="cost_per_bird_sold could not be computed due to birds_end=0.",
                    context={"trial_id": unit.trial_id, "unit_id": unit.unit_id, "birds_end": birds_end},
                )
            )

        cost_per_kg_gain = _safe_div(total_cost, total_liveweight_gain_kg)
        if cost_per_kg_gain is None:
            warnings.append(
                AnalysisWarning(
                    code=WarningCode.division_by_zero,
                    message="cost_per_kg_gain could not be computed due to zero total liveweight gain.",
                    context={"trial_id": unit.trial_id, "unit_id": unit.unit_id, "total_liveweight_gain_kg": total_liveweight_gain_kg},
                )
            )

    computed = ComputedUnitKPIs(
        trial_id=unit.trial_id,
        unit_type=unit.unit_type,
        unit_id=unit.unit_id,
        treatment=unit.treatment,
        days=unit.days,
        birds_placed=unit.birds_placed,
        mortality_total=unit.mortality_total,
        mortality_pct=mortality_pct,
        birds_end=birds_end,
        feed_delivered_kg=unit.feed_delivered_kg,
        feed_refusals_kg=unit.feed_refusals_kg,
        feed_consumed_kg=feed_consumed_kg,
        bw_initial_mean_g=unit.bw_initial_mean_g,
        bw_final_mean_g=unit.bw_final_mean_g,
        bw_final_sd_g=unit.bw_final_sd_g,
        cv_final_pct=cv_final_pct,
        wg_g_per_bird=wg,
        adg_g_per_bird_per_day=adg,
        total_liveweight_gain_kg=total_liveweight_gain_kg,
        fcr=fcr,
        diet_cost_per_kg=unit.diet_cost_per_kg,
        feed_cost_total=feed_cost_total,
        total_cost=total_cost,
        kg_sold_est=kg_sold_est,
        cost_per_kg_sold=cost_per_kg_sold,
        cost_per_bird_sold=cost_per_bird_sold,
        cost_per_kg_gain=cost_per_kg_gain,
    )
    return computed, warnings


def compute_all_units(units: List[TrialUnitInput]) -> Tuple[List[ComputedUnitKPIs], List[AnalysisWarning]]:
    all_kpis: List[ComputedUnitKPIs] = []
    all_warnings: List[AnalysisWarning] = []
    for u in units:
        k, w = compute_unit_kpis(u)
        all_kpis.append(k)
        all_warnings.extend(w)
    return all_kpis, all_warnings
