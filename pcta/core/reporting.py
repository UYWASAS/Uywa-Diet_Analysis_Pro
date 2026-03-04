"""
Reporting helpers for PCTA.

Produces:
- Treatment-level descriptive summaries
- Default metric list selection for stats/charts
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


_ID_COLS = {"trial_id", "unit_type", "unit_id", "treatment"}


def default_metric_list(unit_kpis_df: pd.DataFrame) -> List[str]:
    """
    Choose a stable default list of KPI columns for inferential stats.
    Only includes numeric columns and excludes identifiers.
    """
    numeric_cols = [c for c in unit_kpis_df.columns if c not in _ID_COLS and pd.api.types.is_numeric_dtype(unit_kpis_df[c])]
    # Prefer common metrics first if present
    preferred = [
        "fcr",
        "adg_g_per_bird_per_day",
        "wg_g_per_bird",
        "mortality_pct",
        "feed_consumed_kg",
        "total_liveweight_gain_kg",
        "cv_final_pct",
        "cost_per_kg_sold",
        "cost_per_bird_sold",
        "cost_per_kg_gain",
        "total_cost",
    ]
    out: List[str] = []
    for m in preferred:
        if m in numeric_cols:
            out.append(m)
    for c in numeric_cols:
        if c not in out:
            out.append(c)
    return out


def build_treatment_summary(unit_kpis_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build descriptive treatment summary.

    Outputs columns:
    - treatment
    - n_units
    - for each numeric KPI: mean, sd, sem, min, max
    """
    if unit_kpis_df.empty:
        return pd.DataFrame()

    df = unit_kpis_df.copy()
    numeric_cols = [
        c for c in df.columns
        if c not in _ID_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]

    grouped = df.groupby("treatment", dropna=False)

    rows = []
    for treatment, g in grouped:
        row = {"treatment": treatment, "n_units": int(len(g))}
        for c in numeric_cols:
            x = pd.to_numeric(g[c], errors="coerce").dropna()
            if x.empty:
                row[f"{c}__mean"] = np.nan
                row[f"{c}__sd"] = np.nan
                row[f"{c}__sem"] = np.nan
                row[f"{c}__min"] = np.nan
                row[f"{c}__max"] = np.nan
                continue
            row[f"{c}__mean"] = float(x.mean())
            row[f"{c}__sd"] = float(x.std(ddof=1)) if len(x) >= 2 else np.nan
            row[f"{c}__sem"] = float(x.sem(ddof=1)) if len(x) >= 2 else np.nan
            row[f"{c}__min"] = float(x.min())
            row[f"{c}__max"] = float(x.max())
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("treatment").reset_index(drop=True)
    return out
