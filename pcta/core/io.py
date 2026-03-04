"""
I/O layer for PCTA (Streamlit).

Responsibilities:
- Parse uploaded Excel template or CSV single-table into cleaned DataFrames
- Normalize columns, fill defaults, compute days if needed
- Optionally derive bw_initial/bw_final (and bw_final_sd) from WEIGH_SAMPLES
- Optionally merge COSTS (simple or phase) into unit inputs
- Provide export helper for Excel reporting

Expected Excel sheets:
- HOUSE_SUMMARY (required)
- WEIGH_SAMPLES (optional)
- COSTS (optional)

CSV:
- Treated as HOUSE_SUMMARY equivalent.

The output is a ParsedInput object with record lists to make Streamlit caching easy.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .schemas import (
    AnalysisWarning,
    ExportPayload,
    ParsedInput,
    ParsedInputMode,
    TrialUnitInput,
    UnitType,
    WarningCode,
)
from .utils import normalize_columns, parse_date_like, safe_float, safe_int


REQUIRED_HOUSE_SUMMARY_COLS = {"trial_id", "unit_id", "treatment"}
HOUSE_SUMMARY_DEFAULTS: Dict[str, Any] = {
    "unit_type": "house",
    "feed_refusals_kg": 0.0,
    "mortality_total": 0,
}


@dataclass(frozen=True)
class ParsedInputBundle:
    mode: ParsedInputMode
    house_summary: pd.DataFrame
    weigh_samples: Optional[pd.DataFrame]
    costs: Optional[pd.DataFrame]
    cleaned_input: Dict[str, Any]


def _read_excel_template(file_bytes: bytes) -> ParsedInputBundle:
    xl = pd.ExcelFile(BytesIO(file_bytes))
    sheets = {s.upper(): s for s in xl.sheet_names}

    if "HOUSE_SUMMARY" not in sheets:
        raise ValueError("Excel file missing required sheet: HOUSE_SUMMARY")

    house_summary = xl.parse(sheets["HOUSE_SUMMARY"])
    weigh_samples = xl.parse(sheets["WEIGH_SAMPLES"]) if "WEIGH_SAMPLES" in sheets else None
    costs = xl.parse(sheets["COSTS"]) if "COSTS" in sheets else None

    return ParsedInputBundle(
        mode=ParsedInputMode.excel_template,
        house_summary=house_summary,
        weigh_samples=weigh_samples,
        costs=costs,
        cleaned_input={"input_mode": "excel_template", "sheets": list(xl.sheet_names)},
    )


def _read_single_table(name: str, file_bytes: bytes) -> ParsedInputBundle:
    lower = name.lower()
    if lower.endswith(".csv"):
        df = pd.read_csv(BytesIO(file_bytes))
    elif lower.endswith(".xlsx"):
        xl = pd.ExcelFile(BytesIO(file_bytes))
        df = xl.parse(xl.sheet_names[0])
    else:
        raise ValueError("Unsupported file type. Please upload .xlsx or .csv")

    return ParsedInputBundle(
        mode=ParsedInputMode.single_table,
        house_summary=df,
        weigh_samples=None,
        costs=None,
        cleaned_input={"input_mode": "single_table", "source_name": name},
    )


def parse_uploaded_file(filename: str, file_bytes: bytes) -> ParsedInput:
    """
    Parse uploaded file into ParsedInput.
    This function does NOT validate domain rules (that happens in validation.py),
    but it will normalize column names, apply defaults, and compute derived fields.
    """
    if filename.lower().endswith(".xlsx"):
        try:
            bundle = _read_excel_template(file_bytes)
        except Exception:
            bundle = _read_single_table(filename, file_bytes)
    elif filename.lower().endswith(".csv"):
        bundle = _read_single_table(filename, file_bytes)
    else:
        raise ValueError("Unsupported file type. Please upload .xlsx or .csv")

    hs = normalize_columns(bundle.house_summary)
    ws = normalize_columns(bundle.weigh_samples) if bundle.weigh_samples is not None else None
    cs = normalize_columns(bundle.costs) if bundle.costs is not None else None

    assert hs is not None
    hs = _clean_house_summary(hs)
    if ws is not None:
        ws = _clean_weigh_samples(ws)
    if cs is not None:
        cs = _clean_costs(cs)

    parsed = ParsedInput(
        mode=bundle.mode,
        cleaned_input=bundle.cleaned_input,
        house_summary_records=hs.to_dict(orient="records"),
        weigh_samples_records=ws.to_dict(orient="records") if ws is not None else None,
        costs_records=cs.to_dict(orient="records") if cs is not None else None,
    )
    return parsed


def _clean_house_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col, val in HOUSE_SUMMARY_DEFAULTS.items():
        if col not in df.columns:
            df[col] = val

    missing = [c for c in REQUIRED_HOUSE_SUMMARY_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"HOUSE_SUMMARY missing required columns: {missing}")

    if "days" not in df.columns:
        if "start_date" in df.columns and "end_date" in df.columns:
            start = df["start_date"].apply(parse_date_like)
            end = df["end_date"].apply(parse_date_like)
            df["days"] = (end - start).dt.days
        else:
            raise ValueError("HOUSE_SUMMARY must include 'days' or both 'start_date' and 'end_date'.")

    df["trial_id"] = df["trial_id"].astype(str).str.strip()
    df["unit_id"] = df["unit_id"].astype(str).str.strip()
    df["treatment"] = df["treatment"].astype(str).str.strip()

    df["unit_type"] = df["unit_type"].fillna("house").astype(str).str.strip().str.lower()
    df["days"] = df["days"].apply(safe_int)

    # Parse numeric columns when present
    int_cols = ["birds_placed", "mortality_total", "birds_sold", "final_sample_n"]
    float_cols = [
        "feed_delivered_kg",
        "feed_refusals_kg",
        "bw_initial_mean_g",
        "bw_final_mean_g",
        "bw_final_sd_g",
        "diet_cost_per_kg",
        "additive_cost_total",
        "chick_cost_per_bird",
        "other_variable_costs_total",
    ]

    for c in int_cols:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: (safe_int(x) if pd.notna(x) else None))

    for c in float_cols:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: (safe_float(x) if pd.notna(x) else None))

    if "feed_refusals_kg" in df.columns:
        df["feed_refusals_kg"] = df["feed_refusals_kg"].fillna(0.0)
    if "mortality_total" in df.columns:
        df["mortality_total"] = df["mortality_total"].fillna(0)

    # Costs defaults (if provided)
    for c, default in [
        ("additive_cost_total", 0.0),
        ("chick_cost_per_bird", 0.0),
        ("other_variable_costs_total", 0.0),
    ]:
        if c in df.columns:
            df[c] = df[c].fillna(default)

    return df


def _clean_weigh_samples(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required = {"trial_id", "unit_id", "treatment"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"WEIGH_SAMPLES missing required columns: {missing}")

    if "day" not in df.columns and "date" not in df.columns:
        raise ValueError("WEIGH_SAMPLES must include either 'day' or 'date' column.")

    df["trial_id"] = df["trial_id"].astype(str).str.strip()
    df["unit_id"] = df["unit_id"].astype(str).str.strip()
    df["treatment"] = df["treatment"].astype(str).str.strip()

    if "day" in df.columns:
        df["day"] = df["day"].apply(lambda x: safe_int(x) if pd.notna(x) else None)
    if "date" in df.columns:
        df["date"] = df["date"].apply(lambda x: parse_date_like(x) if pd.notna(x) else None)

    if "sample_n" in df.columns:
        df["sample_n"] = df["sample_n"].apply(lambda x: safe_int(x) if pd.notna(x) else None)
    for c in ["bw_mean_g", "bw_sd_g"]:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: safe_float(x) if pd.notna(x) else None)

    return df


def _clean_costs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required = {"trial_id", "unit_id", "treatment"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"COSTS missing required columns: {missing}")

    df["trial_id"] = df["trial_id"].astype(str).str.strip()
    df["unit_id"] = df["unit_id"].astype(str).str.strip()
    df["treatment"] = df["treatment"].astype(str).str.strip()

    # Phase mode or simple mode
    if "phase" in df.columns:
        if "feed_cost_per_kg" in df.columns:
            df["feed_cost_per_kg"] = df["feed_cost_per_kg"].apply(lambda x: safe_float(x) if pd.notna(x) else None)
        if "additive_cost_total_phase" in df.columns:
            df["additive_cost_total_phase"] = df["additive_cost_total_phase"].apply(
                lambda x: safe_float(x) if pd.notna(x) else 0.0
            )
    else:
        for c in ["diet_cost_per_kg", "additive_cost_total", "chick_cost_per_bird", "other_variable_costs_total"]:
            if c in df.columns:
                df[c] = df[c].apply(lambda x: safe_float(x) if pd.notna(x) else None)

        for c, default in [
            ("additive_cost_total", 0.0),
            ("chick_cost_per_bird", 0.0),
            ("other_variable_costs_total", 0.0),
        ]:
            if c in df.columns:
                df[c] = df[c].fillna(default)

    return df


def _derive_bw_from_weigh_samples(
    hs: pd.DataFrame,
    ws: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[AnalysisWarning]]:
    """
    Fill missing bw_initial_mean_g / bw_final_mean_g / bw_final_sd_g / final_sample_n from WEIGH_SAMPLES
    by using earliest and latest record per (trial_id, unit_id).
    """
    warnings: List[AnalysisWarning] = []
    hs = hs.copy()
    ws = ws.copy()

    # Decide timeline column (prefer day)
    if "day" in ws.columns and ws["day"].notna().any():
        ws["_time"] = ws["day"]
    else:
        ws["_time"] = ws["date"]

    key_cols = ["trial_id", "unit_id"]
    ws = ws.sort_values(key_cols + ["_time"])

    first = ws.groupby(key_cols, as_index=False).first()
    last = ws.groupby(key_cols, as_index=False).last()

    first = first.rename(
        columns={
            "bw_mean_g": "_bw_initial_mean_g",
            "sample_n": "_initial_sample_n",
        }
    )
    last = last.rename(
        columns={
            "bw_mean_g": "_bw_final_mean_g",
            "bw_sd_g": "_bw_final_sd_g",
            "sample_n": "_final_sample_n",
        }
    )

    merged = hs.merge(first[key_cols + ["_bw_initial_mean_g", "_initial_sample_n"]], on=key_cols, how="left")
    merged = merged.merge(last[key_cols + ["_bw_final_mean_g", "_bw_final_sd_g", "_final_sample_n"]], on=key_cols, how="left")

    # Fill missing BW means
    for col_hs, col_ws in [
        ("bw_initial_mean_g", "_bw_initial_mean_g"),
        ("bw_final_mean_g", "_bw_final_mean_g"),
    ]:
        if col_hs in merged.columns:
            missing_mask = merged[col_hs].isna() & merged[col_ws].notna()
            if missing_mask.any():
                merged.loc[missing_mask, col_hs] = merged.loc[missing_mask, col_ws]
                for _, r in merged.loc[missing_mask, ["trial_id", "unit_id"]].drop_duplicates().iterrows():
                    warnings.append(
                        AnalysisWarning(
                            code=WarningCode.missing_bw_derived_from_weigh_samples,
                            message=f"{col_hs} derived from WEIGH_SAMPLES.",
                            context={"trial_id": r["trial_id"], "unit_id": r["unit_id"], "field": col_hs},
                        )
                    )

    # Fill missing SD
    if "bw_final_sd_g" in merged.columns:
        missing_sd = merged["bw_final_sd_g"].isna() & merged["_bw_final_sd_g"].notna()
        if missing_sd.any():
            merged.loc[missing_sd, "bw_final_sd_g"] = merged.loc[missing_sd, "_bw_final_sd_g"]
            for _, r in merged.loc[missing_sd, ["trial_id", "unit_id"]].drop_duplicates().iterrows():
                warnings.append(
                    AnalysisWarning(
                        code=WarningCode.missing_bw_derived_from_weigh_samples,
                        message="bw_final_sd_g derived from WEIGH_SAMPLES.",
                        context={"trial_id": r["trial_id"], "unit_id": r["unit_id"], "field": "bw_final_sd_g"},
                    )
                )

    # Fill missing final_sample_n (DO NOT delete it later)
    if "final_sample_n" not in merged.columns:
        merged["final_sample_n"] = None

    missing_n = merged["final_sample_n"].isna() & merged["_final_sample_n"].notna()
    if missing_n.any():
        merged.loc[missing_n, "final_sample_n"] = merged.loc[missing_n, "_final_sample_n"]

    # Drop helper columns only (keep final_sample_n!)
    merged = merged.drop(columns=[c for c in merged.columns if c.startswith("_bw_") or c in {"_time", "_initial_sample_n", "_final_sample_n"}], errors="ignore")
    return merged, warnings


def _merge_costs_into_house_summary(
    hs: pd.DataFrame,
    costs: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[AnalysisWarning]]:
    """
    Merge COSTS into house summary.

    Supports:
    - Simple mode: diet_cost_per_kg, additive_cost_total, chick_cost_per_bird, other_variable_costs_total
    - Phase mode: approximations (warned)
    """
    warnings: List[AnalysisWarning] = []
    hs = hs.copy()

    key = ["trial_id", "unit_id", "treatment"]

    if "phase" in costs.columns:
        tmp = costs.copy()
        tmp["feed_cost_per_kg"] = tmp.get("feed_cost_per_kg", pd.Series([None] * len(tmp))).apply(
            lambda x: safe_float(x) if pd.notna(x) else None
        )
        tmp["additive_cost_total_phase"] = tmp.get("additive_cost_total_phase", pd.Series([0.0] * len(tmp))).apply(
            lambda x: safe_float(x) if pd.notna(x) else 0.0
        )

        agg = (
            tmp.groupby(key, as_index=False)
            .agg(
                diet_cost_per_kg=("feed_cost_per_kg", "mean"),
                additive_cost_total=("additive_cost_total_phase", "sum"),
            )
        )
        merged = hs.merge(agg, on=key, how="left")

        warnings.append(
            AnalysisWarning(
                code=WarningCode.validation_adjustment,
                message="COSTS phase mode detected. diet_cost_per_kg approximated as mean(feed_cost_per_kg) due to missing phase feed intake.",
                context={},
            )
        )
    else:
        keep_cols = [c for c in ["diet_cost_per_kg", "additive_cost_total", "chick_cost_per_bird", "other_variable_costs_total"] if c in costs.columns]
        merged = hs.merge(costs[key + keep_cols], on=key, how="left")

    # Default missing costs to 0 where appropriate
    for c, default in [
        ("additive_cost_total", 0.0),
        ("chick_cost_per_bird", 0.0),
        ("other_variable_costs_total", 0.0),
    ]:
        if c not in merged.columns:
            merged[c] = default
        else:
            merged[c] = merged[c].fillna(default)

    if "diet_cost_per_kg" not in merged.columns:
        merged["diet_cost_per_kg"] = None

    if merged["diet_cost_per_kg"].isna().all():
        warnings.append(
            AnalysisWarning(
                code=WarningCode.missing_costs_defaulted,
                message="diet_cost_per_kg missing for all units; feed_cost_total and cost KPIs will be unavailable.",
                context={},
            )
        )

    return merged, warnings


def _to_unit_inputs(hs: pd.DataFrame) -> List[TrialUnitInput]:
    units: List[TrialUnitInput] = []
    for rec in hs.to_dict(orient="records"):
        unit_type_raw = str(rec.get("unit_type", "house") or "house").strip().lower()
        unit_type = UnitType.house if unit_type_raw not in ("house", "pen") else UnitType(unit_type_raw)

        units.append(
            TrialUnitInput(
                trial_id=str(rec["trial_id"]).strip(),
                unit_type=unit_type,
                unit_id=str(rec["unit_id"]).strip(),
                treatment=str(rec["treatment"]).strip(),
                days=int(rec["days"]),
                birds_placed=int(rec.get("birds_placed")),
                mortality_total=int(rec.get("mortality_total", 0) or 0),
                birds_sold=(int(rec["birds_sold"]) if rec.get("birds_sold") is not None and pd.notna(rec.get("birds_sold")) else None),
                feed_delivered_kg=float(rec.get("feed_delivered_kg")),
                feed_refusals_kg=float(rec.get("feed_refusals_kg", 0.0) or 0.0),
                bw_initial_mean_g=float(rec.get("bw_initial_mean_g")),
                bw_final_mean_g=float(rec.get("bw_final_mean_g")),
                bw_final_sd_g=(float(rec["bw_final_sd_g"]) if rec.get("bw_final_sd_g") is not None and pd.notna(rec.get("bw_final_sd_g")) else None),
                final_sample_n=(int(rec["final_sample_n"]) if rec.get("final_sample_n") is not None and pd.notna(rec.get("final_sample_n")) else None),
                diet_cost_per_kg=(float(rec["diet_cost_per_kg"]) if rec.get("diet_cost_per_kg") is not None and pd.notna(rec.get("diet_cost_per_kg")) else None),
                additive_cost_total=float(rec.get("additive_cost_total", 0.0) or 0.0),
                chick_cost_per_bird=float(rec.get("chick_cost_per_bird", 0.0) or 0.0),
                other_variable_costs_total=float(rec.get("other_variable_costs_total", 0.0) or 0.0),
            )
        )
    return units


# ---- ParsedInput convenience ----

def _parsed_input_to_units(self: ParsedInput) -> List[TrialUnitInput]:
    dfs = self.to_dataframes()
    hs = dfs["house_summary"]
    ws = dfs["weigh_samples"]
    cs = dfs["costs"]
    assert hs is not None

    # Ensure expected columns exist (so derivation can fill them)
    for col, default in [
        ("bw_initial_mean_g", None),
        ("bw_final_mean_g", None),
        ("bw_final_sd_g", None),
        ("final_sample_n", None),
    ]:
        if col not in hs.columns:
            hs[col] = default

    # Derive BW/sample_n from weigh samples if present
    if ws is not None:
        hs, _io_warnings = _derive_bw_from_weigh_samples(hs, ws)
    else:
        _io_warnings = []

    # Merge costs if present
    if cs is not None:
        hs, _cost_warnings = _merge_costs_into_house_summary(hs, cs)
    else:
        _cost_warnings = []

    # Persist cleaned_input updates (metadata only)
    total_io_warnings = len(_io_warnings) + len(_cost_warnings)
    if total_io_warnings:
        self.cleaned_input = {**self.cleaned_input, "io_warnings_count": total_io_warnings}

    units = _to_unit_inputs(hs)
    return units


ParsedInput.to_units = _parsed_input_to_units  # type: ignore[attr-defined]


# ---- Export ----

def export_report_xlsx(payload: ExportPayload) -> bytes:
    """
    Create an Excel report (bytes) with expected sheets:
    - CLEANED_INPUT
    - UNIT_KPIS
    - TREATMENT_SUMMARY
    - STATS
    - WARNINGS
    """
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        ci = pd.DataFrame([{"key": k, "value": v} for k, v in (payload.cleaned_input or {}).items()])
        ci.to_excel(writer, index=False, sheet_name="CLEANED_INPUT")

        payload.unit_kpis.to_excel(writer, index=False, sheet_name="UNIT_KPIS")
        payload.treatment_summary.to_excel(writer, index=False, sheet_name="TREATMENT_SUMMARY")
        payload.stats.to_excel(writer, index=False, sheet_name="STATS")
        payload.warnings.to_excel(writer, index=False, sheet_name="WARNINGS")

        for sheet in ["CLEANED_INPUT", "UNIT_KPIS", "TREATMENT_SUMMARY", "STATS", "WARNINGS"]:
            ws = writer.sheets[sheet]
            ws.freeze_panes = "A2"

    return out.getvalue()
