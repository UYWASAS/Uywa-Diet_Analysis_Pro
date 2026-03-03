"""
Validation utilities for PCTA.

This module enforces strict, domain-specific validation rules beyond basic
type checking, and emits structured warnings/errors.

Rules (from spec):
- birds_placed > 0
- mortality_total <= birds_placed
- feed_refusals_kg <= feed_delivered_kg
- days > 0
- WG cannot be negative (warn or block) -> implemented as an error by default
  (configurable via ValidationOptions).

Design:
- Separate "validate_*" functions that either:
  - raise ValueError with a clear message (for blocking conditions), or
  - return warnings for non-blocking conditions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .schemas import AnalysisWarning, TrialUnitInput, WarningCode


@dataclass(frozen=True)
class ValidationOptions:
    """
    Options controlling strictness.

    wg_negative_is_error:
        If True, negative weight gain blocks analysis. If False, emits a warning.
        Default True for statistical/data safety.
    """
    wg_negative_is_error: bool = True


def _err(msg: str) -> ValueError:
    return ValueError(msg)


def validate_unit(unit: TrialUnitInput, *, options: ValidationOptions | None = None) -> List[AnalysisWarning]:
    """
    Validate a single unit.

    Returns:
        List of non-blocking warnings.

    Raises:
        ValueError: If any blocking rule is violated.
    """
    opts = options or ValidationOptions()
    warnings: List[AnalysisWarning] = []

    # Basic required constraints (some already enforced by Pydantic, but keep explicit)
    if unit.days <= 0:
        raise _err(f"Unit {unit.unit_id}: days must be > 0.")
    if unit.birds_placed <= 0:
        raise _err(f"Unit {unit.unit_id}: birds_placed must be > 0.")

    # Mortality bounds
    if unit.mortality_total < 0:
        raise _err(f"Unit {unit.unit_id}: mortality_total must be >= 0.")
    if unit.mortality_total > unit.birds_placed:
        raise _err(f"Unit {unit.unit_id}: mortality_total cannot exceed birds_placed.")

    # Birds sold bounds (if provided)
    if unit.birds_sold is not None:
        if unit.birds_sold < 0:
            raise _err(f"Unit {unit.unit_id}: birds_sold must be >= 0.")
        # Allow birds_sold <= birds_placed, but also allow practical cases where birds sold might be less due to culls etc.
        if unit.birds_sold > unit.birds_placed:
            warnings.append(
                AnalysisWarning(
                    code=WarningCode.validation_adjustment,
                    message="birds_sold is greater than birds_placed; please confirm data consistency.",
                    context={"unit_id": unit.unit_id, "birds_sold": unit.birds_sold, "birds_placed": unit.birds_placed},
                )
            )

    # Feed refusals cannot exceed delivered
    if unit.feed_delivered_kg < 0:
        raise _err(f"Unit {unit.unit_id}: feed_delivered_kg must be >= 0.")
    if unit.feed_refusals_kg < 0:
        raise _err(f"Unit {unit.unit_id}: feed_refusals_kg must be >= 0.")
    if unit.feed_refusals_kg > unit.feed_delivered_kg:
        raise _err(f"Unit {unit.unit_id}: feed_refusals_kg cannot exceed feed_delivered_kg.")

    # BW sanity
    if unit.bw_initial_mean_g < 0 or unit.bw_final_mean_g < 0:
        raise _err(f"Unit {unit.unit_id}: body weights must be >= 0.")

    wg = unit.bw_final_mean_g - unit.bw_initial_mean_g
    if wg < 0:
        msg = (
            f"Unit {unit.unit_id}: weight gain (bw_final_mean_g - bw_initial_mean_g) is negative "
            f"({wg:.3f} g)."
        )
        if opts.wg_negative_is_error:
            raise _err(msg)
        warnings.append(
            AnalysisWarning(
                code=WarningCode.wg_negative,
                message=msg,
                context={
                    "unit_id": unit.unit_id,
                    "bw_initial_mean_g": unit.bw_initial_mean_g,
                    "bw_final_mean_g": unit.bw_final_mean_g,
                    "wg_g_per_bird": wg,
                },
            )
        )

    # SD sanity
    if unit.bw_final_sd_g is not None and unit.bw_final_sd_g < 0:
        raise _err(f"Unit {unit.unit_id}: bw_final_sd_g must be >= 0.")

    # Economics sanity (optional fields)
    if unit.diet_cost_per_kg is not None and unit.diet_cost_per_kg < 0:
        raise _err(f"Unit {unit.unit_id}: diet_cost_per_kg must be >= 0.")
    if unit.chick_cost_per_bird is not None and unit.chick_cost_per_bird < 0:
        raise _err(f"Unit {unit.unit_id}: chick_cost_per_bird must be >= 0.")
    if unit.additive_cost_total < 0:
        raise _err(f"Unit {unit.unit_id}: additive_cost_total must be >= 0.")
    if unit.other_variable_costs_total < 0:
        raise _err(f"Unit {unit.unit_id}: other_variable_costs_total must be >= 0.")

    return warnings


def validate_units(
    units: List[TrialUnitInput],
    *,
    options: ValidationOptions | None = None,
) -> Tuple[List[TrialUnitInput], List[AnalysisWarning]]:
    """
    Validate a list of units.

    Returns:
        (units, warnings)

    Raises:
        ValueError: If any unit violates blocking rules.
    """
    all_warnings: List[AnalysisWarning] = []
    for u in units:
        all_warnings.extend(validate_unit(u, options=options))
    return units, all_warnings
