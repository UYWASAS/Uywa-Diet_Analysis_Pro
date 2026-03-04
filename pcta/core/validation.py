"""
Domain validation for PCTA.

Validation philosophy:
- Errors block analysis (raise ValueError)
- Warnings are returned (do not block)

Errors (spec):
- missing ids/treatment
- birds_placed <= 0
- mortality_total < 0 or > birds_placed
- feed_delivered_kg < 0
- feed_refusals_kg < 0 or feed_refusals_kg > feed_delivered_kg
- days <= 0

Warnings (spec):
- missing bw_final_sd_g (uniformity not available)
- missing sample_n (CI not available)
- birds_sold mismatch (inconsistent with birds_placed - mortality_total)
- no replication (min n per treatment < 2) -> warning only; stats module enforces safety

Note: Replication warning is emitted here for early visibility, but inferential
p-values are still strictly controlled by stats.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

from .schemas import AnalysisWarning, TrialUnitInput, WarningCode


@dataclass(frozen=True)
class ValidationOptions:
    """
    Validation configuration.

    wg_negative_is_error:
        If True, negative weight gain blocks analysis.
        If False, negative weight gain yields a warning (calculations will still
        proceed, but FCR may become NaN in calculations.py per safety behavior).
    """
    wg_negative_is_error: bool = False


def _err(msg: str) -> ValueError:
    return ValueError(msg)


def validate_unit(unit: TrialUnitInput, *, options: ValidationOptions | None = None) -> List[AnalysisWarning]:
    """
    Validate a single unit. Returns warnings, raises on errors.
    """
    opts = options or ValidationOptions()
    warnings: List[AnalysisWarning] = []

    # Required identifiers
    if not unit.trial_id.strip():
        raise _err("trial_id is required.")
    if not unit.unit_id.strip():
        raise _err("unit_id is required.")
    if not unit.treatment.strip():
        raise _err("treatment is required.")

    # Days
    if unit.days <= 0:
        raise _err(f"Unit {unit.unit_id}: days must be > 0.")

    # Birds
    if unit.birds_placed <= 0:
        raise _err(f"Unit {unit.unit_id}: birds_placed must be > 0.")
    if unit.mortality_total < 0:
        raise _err(f"Unit {unit.unit_id}: mortality_total must be >= 0.")
    if unit.mortality_total > unit.birds_placed:
        raise _err(f"Unit {unit.unit_id}: mortality_total cannot exceed birds_placed.")

    if unit.birds_sold is not None:
        if unit.birds_sold < 0:
            raise _err(f"Unit {unit.unit_id}: birds_sold must be >= 0.")
        if unit.birds_sold > unit.birds_placed:
            warnings.append(
                AnalysisWarning(
                    code=WarningCode.birds_sold_mismatch,
                    message="birds_sold is greater than birds_placed; please confirm.",
                    context={"unit_id": unit.unit_id, "birds_sold": unit.birds_sold, "birds_placed": unit.birds_placed},
                )
            )
        # Also warn if it doesn't match expected birds_end
        expected_end = unit.birds_placed - unit.mortality_total
        if unit.birds_sold != expected_end:
            warnings.append(
                AnalysisWarning(
                    code=WarningCode.birds_sold_mismatch,
                    message="birds_sold differs from birds_placed - mortality_total; please confirm.",
                    context={
                        "unit_id": unit.unit_id,
                        "birds_sold": unit.birds_sold,
                        "birds_placed": unit.birds_placed,
                        "mortality_total": unit.mortality_total,
                        "birds_placed_minus_mortality": expected_end,
                    },
                )
            )

    # Feed
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
    if wg < 0 and opts.wg_negative_is_error:
        raise _err(
            f"Unit {unit.unit_id}: weight gain (bw_final_mean_g - bw_initial_mean_g) is negative ({wg:.3f} g)."
        )
    if unit.bw_final_sd_g is None:
        warnings.append(
            AnalysisWarning(
                code=WarningCode.missing_bw_sd_uniformity_unavailable,
                message="bw_final_sd_g missing; uniformity (CV) not available.",
                context={"unit_id": unit.unit_id},
            )
        )
    if unit.final_sample_n is None:
        warnings.append(
            AnalysisWarning(
                code=WarningCode.missing_sample_n_ci_unavailable,
                message="final_sample_n missing; confidence intervals not available.",
                context={"unit_id": unit.unit_id},
            )
        )

    return warnings


def validate_units(
    units: List[TrialUnitInput],
    *,
    options: ValidationOptions | None = None,
) -> Tuple[List[TrialUnitInput], List[AnalysisWarning]]:
    """
    Validate all units, raising on the first blocking error.

    Also emits a replication warning if min n per treatment < 2.
    """
    opts = options or ValidationOptions()
    warnings: List[AnalysisWarning] = []

    for u in units:
        warnings.extend(validate_unit(u, options=opts))

    # Replication warning (non-blocking)
    df = pd.DataFrame([u.model_dump() for u in units])
    rep = df.groupby("treatment").size().astype(int).to_dict()
    min_n = int(min(rep.values())) if rep else 0
    if min_n < 2:
        warnings.append(
            AnalysisWarning(
                code=WarningCode.inferential_disabled_no_replication,
                message="Inferential statistics disabled due to lack of replication.",
                context={"replication_by_treatment": rep, "min_n_per_treatment": min_n},
            )
        )

    return units, warnings
