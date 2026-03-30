"""
Productive KPI calculations (pure logic, no Streamlit).

Computes economic and productive metrics from TrialUnitInput records.
Designed to work with manual price/cost inputs from UI.

KPIs calculated:
- Feed consumed (kg)
- Weight gain (g/bird, total kg)
- FCR (feed conversion ratio)
- Feed cost, bird cost, total cost
- Revenue (from kg sold)
- Margin
- EPEF (European Production Efficiency Factor)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .schemas import TrialUnitInput, AnalysisWarning, WarningCode


# ======================================================================================
# Data classes for KPI inputs and results
# ======================================================================================


@dataclass
class ProductiveKPIInputs:
    """User-provided economic parameters for KPI calculation."""
    price_kg_sold: float  # Price per kg of liveweight ($/kg or €/kg)
    cost_feed_per_kg: float  # Cost of feed per kg consumed ($/kg)
    cost_chick_per_bird: float  # Cost per day-old chick ($/bird)
    other_costs_per_bird: float = 0.0  # Additional costs per bird placed ($/bird)
    mortality_cost_pct: float = 0.0  # % of bird cost recoverable if bird dies (0-100)


@dataclass
class UnitProductiveKPI:
    """Computed KPIs for a single trial unit."""
    trial_id: str
    unit_id: str
    treatment: str
    
    # Input data
    birds_placed: int
    mortality_total: int
    mortality_pct: float
    birds_end: int
    days_cycle: int
    
    # Feed
    feed_consumed_kg: float
    
    # Growth
    bw_initial_g: float
    bw_final_g: float
    wg_total_per_bird_g: float
    adg_per_bird_per_day_g: float
    total_liveweight_gain_kg: float
    
    # Conversion
    fcr: Optional[float]
    
    # Economics
    feed_cost_total: float
    bird_cost_total: float
    other_cost_total: float
    total_cost: float
    
    # Revenue & margin
    kg_sold_estimate: float
    revenue_total: float
    margin_total: float
    margin_pct: Optional[float]
    cost_per_kg_sold: Optional[float]
    
    # Efficiency
    epef: Optional[float]  # European Production Efficiency Factor


# ======================================================================================
# Core calculation functions
# ======================================================================================


def compute_unit_kpis(
    unit: TrialUnitInput,
    inputs: ProductiveKPIInputs,
) -> Tuple[UnitProductiveKPI, List[AnalysisWarning]]:
    """
    Compute all KPIs for a single trial unit.
    
    Args:
        unit: TrialUnitInput record (from house_summary)
        inputs: ProductiveKPIInputs (prices/costs from UI)
    
    Returns:
        (UnitProductiveKPI, list of warnings)
    """
    warnings: List[AnalysisWarning] = []
    
    # --- Basic derived metrics ---
    mortality_pct = (unit.mortality_total / unit.birds_placed * 100.0) if unit.birds_placed > 0 else 0.0
    birds_end = unit.birds_placed - unit.mortality_total
    
    # Feed consumed
    feed_consumed_kg = unit.feed_delivered_kg - unit.feed_refusals_kg
    if feed_consumed_kg < 0:
        feed_consumed_kg = 0.0
        warnings.append(AnalysisWarning(
            code=WarningCode.validation_adjustment,
            message="Feed consumed was negative; set to 0.",
            context={"unit_id": unit.unit_id}
        ))
    
    # --- Weight gain ---
    wg_total_per_bird_g = unit.bw_final_mean_g - unit.bw_initial_mean_g
    adg_per_bird_per_day_g = wg_total_per_bird_g / unit.days if unit.days > 0 else 0.0
    total_liveweight_gain_kg = (wg_total_per_bird_g / 1000.0) * birds_end if birds_end > 0 else 0.0
    
    # --- FCR (Feed Conversion Ratio) ---
    fcr: Optional[float] = None
    if total_liveweight_gain_kg > 0:
        fcr = feed_consumed_kg / total_liveweight_gain_kg
    elif total_liveweight_gain_kg == 0:
        warnings.append(AnalysisWarning(
            code=WarningCode.wg_nonpositive_fcr_nan,
            message="Weight gain is zero or negative; FCR cannot be computed.",
            context={"unit_id": unit.unit_id}
        ))
    
    # --- Economics ---
    # Feed cost
    feed_cost_total = feed_consumed_kg * inputs.cost_feed_per_kg if inputs.cost_feed_per_kg is not None else 0.0
    
    # Bird cost (accounting for mortality)
    bird_cost_total = unit.birds_placed * inputs.cost_chick_per_bird
    if inputs.mortality_cost_pct > 0 and unit.mortality_total > 0:
        # Recover some cost if bird dies (e.g., salvage value)
        recovery = unit.mortality_total * inputs.cost_chick_per_bird * (inputs.mortality_cost_pct / 100.0)
        bird_cost_total -= recovery
    
    # Other costs
    other_cost_total = unit.birds_placed * inputs.other_costs_per_bird
    other_cost_total += unit.additive_cost_total + unit.other_variable_costs_total
    
    # Total cost
    total_cost = feed_cost_total + bird_cost_total + other_cost_total
    
    # ---- Revenue ----
    kg_sold_estimate = total_liveweight_gain_kg  # Assume all gain is sold
    revenue_total = kg_sold_estimate * inputs.price_kg_sold
    
    # ---- Margin ----
    margin_total = revenue_total - total_cost
    margin_pct = (margin_total / revenue_total * 100.0) if revenue_total > 0 else None
    
    # Cost per unit sold
    cost_per_kg_sold = total_cost / kg_sold_estimate if kg_sold_estimate > 0 else None
    
    # ---- EPEF (European Production Efficiency Factor) ----
    epef: Optional[float] = None
    if fcr is not None and fcr > 0 and unit.days > 0:
        viability_pct = 100.0 - mortality_pct
        # EPEF = (Viability% * Weight(kg) * 100) / (Age(days) * FCR)
        # Using final weight per bird in kg
        bw_final_kg = unit.bw_final_mean_g / 1000.0
        if bw_final_kg > 0:
            epef = (viability_pct * bw_final_kg * 100.0) / (unit.days * fcr)
    
    kpi = UnitProductiveKPI(
        trial_id=unit.trial_id,
        unit_id=unit.unit_id,
        treatment=unit.treatment,
        
        birds_placed=unit.birds_placed,
        mortality_total=unit.mortality_total,
        mortality_pct=mortality_pct,
        birds_end=birds_end,
        days_cycle=unit.days,
        
        feed_consumed_kg=feed_consumed_kg,
        
        bw_initial_g=unit.bw_initial_mean_g,
        bw_final_g=unit.bw_final_mean_g,
        wg_total_per_bird_g=wg_total_per_bird_g,
        adg_per_bird_per_day_g=adg_per_bird_per_day_g,
        total_liveweight_gain_kg=total_liveweight_gain_kg,
        
        fcr=fcr,
        
        feed_cost_total=feed_cost_total,
        bird_cost_total=bird_cost_total,
        other_cost_total=other_cost_total,
        total_cost=total_cost,
        
        kg_sold_estimate=kg_sold_estimate,
        revenue_total=revenue_total,
        margin_total=margin_total,
        margin_pct=margin_pct,
        cost_per_kg_sold=cost_per_kg_sold,
        
        epef=epef,
    )
    
    return kpi, warnings


def compute_productive_kpis_batch(
    units: List[TrialUnitInput],
    inputs: ProductiveKPIInputs,
) -> Tuple[List[UnitProductiveKPI], List[AnalysisWarning]]:
    """
    Compute KPIs for multiple units.
    
    Args:
        units: List of TrialUnitInput records
        inputs: Shared ProductiveKPIInputs
    
    Returns:
        (list of UnitProductiveKPI, aggregated warnings)
    """
    kpis: List[UnitProductiveKPI] = []
    all_warnings: List[AnalysisWarning] = []
    
    for unit in units:
        kpi, warnings = compute_unit_kpis(unit, inputs)
        kpis.append(kpi)
        all_warnings.extend(warnings)
    
    return kpis, all_warnings


def kpis_to_dataframe(kpis: List[UnitProductiveKPI]) -> pd.DataFrame:
    """Convert list of UnitProductiveKPI to pandas DataFrame."""
    records = [kpi.__dict__ for kpi in kpis]
    return pd.DataFrame(records)


def compute_summary_by_treatment(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate KPIs by treatment (mean, sum, or relevant metric).
    
    Args:
        df: DataFrame from kpis_to_dataframe
    
    Returns:
        Summary DataFrame grouped by treatment
    """
    if df.empty:
        return pd.DataFrame()
    
    # Metrics to aggregate by treatment
    agg_dict = {
        "birds_placed": "first",
        "mortality_total": "sum",
        "mortality_pct": "mean",
        "birds_end": "sum",
        "days_cycle": "first",
        
        "feed_consumed_kg": "sum",
        "bw_initial_g": "mean",
        "bw_final_g": "mean",
        "wg_total_per_bird_g": "mean",
        "adg_per_bird_per_day_g": "mean",
        "total_liveweight_gain_kg": "sum",
        
        "fcr": "mean",
        
        "feed_cost_total": "sum",
        "bird_cost_total": "sum",
        "other_cost_total": "sum",
        "total_cost": "sum",
        
        "kg_sold_estimate": "sum",
        "revenue_total": "sum",
        "margin_total": "sum",
        "margin_pct": "mean",
        "cost_per_kg_sold": "mean",
        
        "epef": "mean",
    }
    
    summary = df.groupby("treatment", as_index=False).agg(agg_dict)
    summary.insert(0, "num_units", df.groupby("treatment").size().values)
    
    return summary


def compute_total_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute overall totals and KPIs across all units.
    
    Returns:
        Dict with keys like "total_birds_placed", "total_revenue", etc.
    """
    if df.empty:
        return {}
    
    return {
        "num_units": len(df),
        "total_birds_placed": int(df["birds_placed"].sum()),
        "total_mortality": int(df["mortality_total"].sum()),
        "avg_mortality_pct": float(df["mortality_pct"].mean()),
        "total_birds_end": int(df["birds_end"].sum()),
        
        "total_feed_consumed_kg": float(df["feed_consumed_kg"].sum()),
        "avg_feed_consumed_per_bird_kg": float(df["feed_consumed_kg"].sum() / df["birds_end"].sum()) if df["birds_end"].sum() > 0 else 0.0,
        
        "total_liveweight_gain_kg": float(df["total_liveweight_gain_kg"].sum()),
        "avg_wg_per_bird_g": float(df["wg_total_per_bird_g"].mean()),
        "avg_adg_per_bird_per_day_g": float(df["adg_per_bird_per_day_g"].mean()),
        
        "avg_fcr": float(df["fcr"].mean()),
        
        "total_feed_cost": float(df["feed_cost_total"].sum()),
        "total_bird_cost": float(df["bird_cost_total"].sum()),
        "total_other_cost": float(df["other_cost_total"].sum()),
        "total_cost": float(df["total_cost"].sum()),
        
        "total_kg_sold": float(df["kg_sold_estimate"].sum()),
        "total_revenue": float(df["revenue_total"].sum()),
        "total_margin": float(df["margin_total"].sum()),
        "avg_margin_pct": float(df["margin_pct"].mean()) if df["margin_pct"].notna().any() else None,
        "avg_cost_per_kg_sold": float(df["cost_per_kg_sold"].mean()) if df["cost_per_kg_sold"].notna().any() else None,
        
        "avg_epef": float(df["epef"].mean()) if df["epef"].notna().any() else None,
    }
