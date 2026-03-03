"""
Pydantic schemas for PCTA (Poultry Commercial Trial Analyzer).

These models define:
- Input payload structure for trial analysis
- Validation constraints (strict where required)
- Output structures for computed metrics, statistics, and warnings

Statistical safety rule:
- If n per treatment < 2, inferential testing MUST be disabled.
  This rule is enforced in analysis logic (not only schema), but
  schemas include fields to carry replication counts and warnings.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class UnitType(str, Enum):
    """Experimental unit type abstraction."""
    house = "house"
    pen = "pen"


class MetricName(str, Enum):
    """Common metric keys produced by calculations/statistics."""
    feed_consumed_kg = "feed_consumed_kg"
    birds_end = "birds_end"
    wg_g_per_bird = "wg_g_per_bird"
    adg_g_per_bird_per_day = "adg_g_per_bird_per_day"
    total_liveweight_gain_kg = "total_liveweight_gain_kg"
    fcr = "fcr"
    cv_final_pct = "cv_final_pct"

    feed_cost_total = "feed_cost_total"
    total_cost = "total_cost"
    cost_per_kg_sold = "cost_per_kg_sold"


class WarningCode(str, Enum):
    """Machine-readable warning codes."""
    inferential_disabled_no_replication = "inferential_disabled_no_replication"
    wg_negative = "wg_negative"
    missing_birds_sold_used_estimate = "missing_birds_sold_used_estimate"
    missing_bw_sd_cv_unavailable = "missing_bw_sd_cv_unavailable"
    division_by_zero = "division_by_zero"
    validation_adjustment = "validation_adjustment"


class AnalysisWarning(BaseModel):
    """A warning emitted during validation, calculation, or statistics."""
    model_config = ConfigDict(extra="forbid")

    code: WarningCode
    message: str
    context: Dict[str, Any] = Field(default_factory=dict)


class TrialUnitInput(BaseModel):
    """
    Input for a single experimental unit (house or pen).

    Notes:
    - Many fields are optional to allow partial data entry.
    - Core validation rules are enforced in validation layer; schema
      captures types and basic constraints.
    """
    model_config = ConfigDict(extra="forbid")

    unit_id: str = Field(..., min_length=1, description="Unique unit identifier within the dataset.")
    unit_type: UnitType = Field(..., description="Unit type abstraction: house or pen.")
    treatment: str = Field(..., min_length=1, description="Treatment/group identifier.")

    days: int = Field(..., gt=0, description="Trial length in days (must be > 0).")

    birds_placed: int = Field(..., gt=0, description="Number of birds placed at start (must be > 0).")
    mortality_total: int = Field(0, ge=0, description="Total mortality count.")
    birds_sold: Optional[int] = Field(
        default=None,
        ge=0,
        description="Birds sold/marketed. If missing, birds_end is estimated as birds_placed - mortality_total.",
    )

    feed_delivered_kg: float = Field(..., ge=0, description="Total feed delivered (kg).")
    feed_refusals_kg: float = Field(0.0, ge=0, description="Feed refusals/remaining (kg).")

    bw_initial_mean_g: float = Field(..., ge=0, description="Initial mean body weight (g).")
    bw_final_mean_g: float = Field(..., ge=0, description="Final mean body weight (g).")
    bw_final_sd_g: Optional[float] = Field(
        default=None,
        ge=0,
        description="Final body weight SD (g). If provided, CV can be computed.",
    )

    # Economics (optional)
    diet_cost_per_kg: Optional[float] = Field(default=None, ge=0, description="Diet cost per kg.")
    additive_cost_total: float = Field(0.0, ge=0, description="Total additive cost for the unit.")
    chick_cost_per_bird: Optional[float] = Field(default=None, ge=0, description="Chick cost per bird placed.")
    other_variable_costs_total: float = Field(0.0, ge=0, description="Other variable costs total for the unit.")

    # Optional metadata
    site: Optional[str] = None
    flock_id: Optional[str] = None
    notes: Optional[str] = None


class TrialInput(BaseModel):
    """Top-level analysis request body."""
    model_config = ConfigDict(extra="forbid")

    trial_name: str = Field(..., min_length=1)
    units: List[TrialUnitInput] = Field(..., min_length=1)

    # Controls
    alpha: float = Field(0.05, gt=0.0, lt=1.0, description="Significance level for inferential tests.")
    enable_posthoc: bool = Field(True, description="Whether to run posthoc tests when applicable.")


class ComputedMetrics(BaseModel):
    """Computed per-unit metrics."""
    model_config = ConfigDict(extra="forbid")

    unit_id: str
    treatment: str
    unit_type: UnitType
    days: int

    # Bird counts
    birds_placed: int
    mortality_total: int
    birds_end: int

    # Feed + performance
    feed_delivered_kg: float
    feed_refusals_kg: float
    feed_consumed_kg: float

    bw_initial_mean_g: float
    bw_final_mean_g: float
    wg_g_per_bird: float
    adg_g_per_bird_per_day: float
    total_liveweight_gain_kg: float
    fcr: Optional[float] = None
    cv_final_pct: Optional[float] = None

    # Economics
    diet_cost_per_kg: Optional[float] = None
    feed_cost_total: Optional[float] = None
    additive_cost_total: float = 0.0
    chick_cost_per_bird: Optional[float] = None
    other_variable_costs_total: float = 0.0
    total_cost: Optional[float] = None
    cost_per_kg_sold: Optional[float] = None


class GroupSummary(BaseModel):
    """Descriptive summary for a treatment group."""
    model_config = ConfigDict(extra="forbid")

    treatment: str
    n_units: int

    # Summary statistics by metric key (mean, sd, sem, min, max)
    metrics: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Per-metric descriptive summary; keys are metric names.",
    )


class AssumptionChecks(BaseModel):
    """Assumption check results for a given metric."""
    model_config = ConfigDict(extra="forbid")

    metric: str
    shapiro_p: Optional[float] = None
    levene_p: Optional[float] = None
    notes: Optional[str] = None


class InferentialResult(BaseModel):
    """
    Inferential testing result for one metric.

    p_value is optional and MUST NOT be produced when replication is insufficient.
    """
    model_config = ConfigDict(extra="forbid")

    metric: str
    test: Optional[str] = None  # e.g., "anova", "welch_anova", "kruskal"
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    df: Optional[Dict[str, float]] = None
    assumptions: Optional[AssumptionChecks] = None
    posthoc: Optional[Dict[str, Any]] = None  # structure depends on method
    disabled_reason: Optional[str] = None


class AnalysisResult(BaseModel):
    """Full analysis response."""
    model_config = ConfigDict(extra="forbid")

    trial_name: str
    alpha: float

    # Per-unit computed values
    unit_metrics: List[ComputedMetrics]

    # Group-level summaries
    group_summaries: List[GroupSummary]

    # Inferential results per metric
    inferential: List[InferentialResult] = Field(default_factory=list)

    # Replication info (min n across treatments is key for safety rule)
    replication_by_treatment: Dict[str, int] = Field(default_factory=dict)
    min_n_per_treatment: int = 0
    inferential_enabled: bool = False

    warnings: List[AnalysisWarning] = Field(default_factory=list)


# ---------- Upload/template related schemas ----------


class UploadParseMode(str, Enum):
    """How uploaded files should be interpreted."""
    excel = "excel"
    csv = "csv"


class UploadResponse(BaseModel):
    """Response returned after parsing an uploaded dataset into validated units."""
    model_config = ConfigDict(extra="forbid")

    trial_name: Optional[str] = None
    units: List[TrialUnitInput]
    warnings: List[AnalysisWarning] = Field(default_factory=list)


class TemplateResponse(BaseModel):
    """Downloadable template metadata."""
    model_config = ConfigDict(extra="forbid")

    filename: str
    content_type: str = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    description: str
