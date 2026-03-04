"""
PCTA schemas (Pydantic).

Defines:
- Standardized input units for commercial poultry trials (house/pen abstraction)
- Parsed input container returned by IO layer
- Warnings model (non-blocking) used across validation/calculation/stats
- Export payload structure for report generation
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class UnitType(str, Enum):
    """Experimental unit type abstraction."""
    house = "house"
    pen = "pen"


class WarningCode(str, Enum):
    """Machine-readable warning codes."""
    inferential_disabled_no_replication = "inferential_disabled_no_replication"
    missing_bw_sd_uniformity_unavailable = "missing_bw_sd_uniformity_unavailable"
    missing_sample_n_ci_unavailable = "missing_sample_n_ci_unavailable"
    birds_sold_mismatch = "birds_sold_mismatch"
    wg_nonpositive_fcr_nan = "wg_nonpositive_fcr_nan"
    missing_days_computed = "missing_days_computed"
    missing_bw_derived_from_weigh_samples = "missing_bw_derived_from_weigh_samples"
    missing_costs_defaulted = "missing_costs_defaulted"
    division_by_zero = "division_by_zero"
    validation_adjustment = "validation_adjustment"


class AnalysisWarning(BaseModel):
    """A non-blocking warning emitted during processing."""
    model_config = ConfigDict(extra="forbid")

    code: WarningCode
    message: str
    context: Dict[str, Any] = Field(default_factory=dict)


class TrialUnitInput(BaseModel):
    """
    Canonical per-unit input record used throughout the app.

    trial_id is required to support multi-trial uploads; downstream UI may filter by trial_id.
    """
    model_config = ConfigDict(extra="forbid")

    trial_id: str = Field(..., min_length=1)
    unit_type: UnitType = Field(UnitType.house)
    unit_id: str = Field(..., min_length=1)
    treatment: str = Field(..., min_length=1)

    # Timing
    days: int = Field(..., gt=0)

    # Birds
    birds_placed: int = Field(..., gt=0)
    mortality_total: int = Field(0, ge=0)
    birds_sold: Optional[int] = Field(default=None, ge=0)

    # Feed
    feed_delivered_kg: float = Field(..., ge=0)
    feed_refusals_kg: float = Field(0.0, ge=0)

    # BW (may be derived by IO from WEIGH_SAMPLES)
    bw_initial_mean_g: float = Field(..., ge=0)
    bw_final_mean_g: float = Field(..., ge=0)
    bw_final_sd_g: Optional[float] = Field(default=None, ge=0)

    # Weigh sample metadata (optional; used for CI availability)
    final_sample_n: Optional[int] = Field(default=None, ge=1)

    # Economics (optional; IO may default missing to 0 for totals)
    diet_cost_per_kg: Optional[float] = Field(default=None, ge=0)
    additive_cost_total: float = Field(0.0, ge=0)
    chick_cost_per_bird: float = Field(0.0, ge=0)
    other_variable_costs_total: float = Field(0.0, ge=0)


class ComputedUnitKPIs(BaseModel):
    """Computed KPIs per unit."""
    model_config = ConfigDict(extra="forbid")

    trial_id: str
    unit_type: UnitType
    unit_id: str
    treatment: str
    days: int

    birds_placed: int
    mortality_total: int
    mortality_pct: float
    birds_end: int

    feed_delivered_kg: float
    feed_refusals_kg: float
    feed_consumed_kg: float

    bw_initial_mean_g: float
    bw_final_mean_g: float
    bw_final_sd_g: Optional[float] = None
    cv_final_pct: Optional[float] = None

    wg_g_per_bird: float
    adg_g_per_bird_per_day: float

    total_liveweight_gain_kg: float
    fcr: Optional[float] = None

    # Economics
    diet_cost_per_kg: Optional[float] = None
    feed_cost_total: Optional[float] = None
    total_cost: Optional[float] = None

    kg_sold_est: float
    cost_per_kg_sold: Optional[float] = None
    cost_per_bird_sold: Optional[float] = None
    cost_per_kg_gain: Optional[float] = None


class ParsedInputMode(str, Enum):
    """Detected input mode."""
    excel_template = "excel_template"
    single_table = "single_table"


class ParsedInput(BaseModel):
    """
    Structured, cleaned representation of uploaded data.

    DataFrames are stored as list-of-records to keep Pydantic-friendly structure
    for caching/session state. Use `to_dataframes()` to reconstruct.
    """
    model_config = ConfigDict(extra="forbid")

    mode: ParsedInputMode
    cleaned_input: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary cleaned input metadata; used for reporting.",
    )

    house_summary_records: List[Dict[str, Any]]
    weigh_samples_records: Optional[List[Dict[str, Any]]] = None
    costs_records: Optional[List[Dict[str, Any]]] = None

    def to_dataframes(self) -> Dict[str, Optional["pd.DataFrame"]]:
        import pandas as pd  # local import

        hs = pd.DataFrame(self.house_summary_records)
        ws = pd.DataFrame(self.weigh_samples_records) if self.weigh_samples_records is not None else None
        cs = pd.DataFrame(self.costs_records) if self.costs_records is not None else None
        return {"house_summary": hs, "weigh_samples": ws, "costs": cs}


class ExportPayload(BaseModel):
    """Payload used by reporting.export_report_xlsx."""
    model_config = ConfigDict(extra="forbid")

    cleaned_input: Dict[str, Any]
    unit_kpis: Any  # pandas.DataFrame at runtime
    treatment_summary: Any  # pandas.DataFrame at runtime
    stats: Any  # pandas.DataFrame at runtime
    warnings: Any  # pandas.DataFrame at runtime
