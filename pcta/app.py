"""
PCTA - Poultry Commercial Trial Analyzer (Streamlit)

Streamlit UI for:
- Uploading Excel/CSV inputs
- Previewing and validating inputs
- Computing per-unit KPIs
- Treatment summaries and (safe) inferential statistics
- Exporting an Excel report

Safety:
- Inferential statistics are disabled when min replication per treatment < 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from core.calculations import compute_all_units
from core.io import (
    ExportPayload,
    ParsedInput,
    export_report_xlsx,
    parse_uploaded_file,
)
from core.reporting import (
    build_treatment_summary,
    default_metric_list,
)
from core.schemas import AnalysisWarning, TrialUnitInput
from core.stats import StatsOptions, run_inferential_statistics
from core.validation import ValidationOptions, validate_units


@dataclass(frozen=True)
class AppState:
    parsed: Optional[ParsedInput]
    units: Optional[List[TrialUnitInput]]
    warnings: List[AnalysisWarning]
    unit_kpis: Optional[pd.DataFrame]
    treatment_summary: Optional[pd.DataFrame]
    stats_table: Optional[pd.DataFrame]
    report_bytes: Optional[bytes]


def _init_state() -> None:
    if "pcta_state" not in st.session_state:
        st.session_state["pcta_state"] = AppState(
            parsed=None,
            units=None,
            warnings=[],
            unit_kpis=None,
            treatment_summary=None,
            stats_table=None,
            report_bytes=None,
        )


def _set_state(**kwargs) -> None:
    cur: AppState = st.session_state["pcta_state"]
    st.session_state["pcta_state"] = AppState(
        parsed=kwargs.get("parsed", cur.parsed),
        units=kwargs.get("units", cur.units),
        warnings=kwargs.get("warnings", cur.warnings),
        unit_kpis=kwargs.get("unit_kpis", cur.unit_kpis),
        treatment_summary=kwargs.get("treatment_summary", cur.treatment_summary),
        stats_table=kwargs.get("stats_table", cur.stats_table),
        report_bytes=kwargs.get("report_bytes", cur.report_bytes),
    )


def _warnings_to_df(warnings: List[AnalysisWarning]) -> pd.DataFrame:
    if not warnings:
        return pd.DataFrame(columns=["code", "message", "context"])
    return pd.DataFrame(
        [
            {
                "code": w.code.value,
                "message": w.message,
                "context": w.context,
            }
            for w in warnings
        ]
    )


def _format_replication(units: List[TrialUnitInput]) -> Tuple[Dict[str, int], int]:
    df = pd.DataFrame([u.model_dump() for u in units])
    rep = df.groupby("treatment").size().astype(int).to_dict()
    min_n = int(min(rep.values())) if rep else 0
    return rep, min_n


st.set_page_config(page_title="PCTA – Poultry Commercial Trial Analyzer", layout="wide")
_init_state()
state: AppState = st.session_state["pcta_state"]

st.title("PCTA – Poultry Commercial Trial Analyzer")
st.caption(
    "Commercial poultry trial analyzer (houses; compatible with pens). "
    "Statistical safety: inferential testing disabled without replication (n per treatment < 2)."
)

with st.sidebar:
    st.header("Upload")
    uploaded = st.file_uploader(
        "Upload Excel (.xlsx) or CSV (.csv)",
        type=["xlsx", "csv"],
        accept_multiple_files=False,
    )

    st.divider()
    st.header("Analysis options")
    alpha = st.number_input("alpha (significance level)", min_value=0.001, max_value=0.2, value=0.05, step=0.005)
    enable_posthoc = st.checkbox("Enable posthoc (when applicable)", value=True)

    st.divider()
    st.header("Validation strictness")
    wg_negative_is_error = st.checkbox("Block negative weight gain (WG<0)", value=True)

    st.divider()
    run_btn = st.button("Run analysis", type="primary", use_container_width=True)

    st.divider()
    st.header("Export")
    if state.report_bytes is not None:
        st.download_button(
            "Download Excel report",
            data=state.report_bytes,
            file_name="pcta_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    else:
        st.caption("Run analysis to enable export.")

# Upload + parse
if uploaded is not None:
    try:
        parsed = parse_uploaded_file(uploaded.name, uploaded.getvalue())
        _set_state(parsed=parsed, report_bytes=None)
    except Exception as e:
        st.error(f"Failed to parse file: {e}")
        _set_state(parsed=None, units=None, warnings=[], unit_kpis=None, treatment_summary=None, stats_table=None, report_bytes=None)

# Main layout
tabs = st.tabs(["1) Preview", "2) Validation", "3) KPIs", "4) Stats", "5) Export & Notes"])

with tabs[0]:
    st.subheader("Preview")
    if state.parsed is None:
        st.info("Upload a file to preview inputs.")
    else:
        st.write("Detected input mode:", state.parsed.mode)
        cols = st.columns(2)

        with cols[0]:
            st.markdown("**HOUSE_SUMMARY (or single-table)**")
            st.dataframe(state.parsed.house_summary.head(200), use_container_width=True)

        with cols[1]:
            st.markdown("**WEIGH_SAMPLES (optional)**")
            if state.parsed.weigh_samples is None:
                st.caption("Not provided.")
            else:
                st.dataframe(state.parsed.weigh_samples.head(200), use_container_width=True)

        st.markdown("**COSTS (optional)**")
        if state.parsed.costs is None:
            st.caption("Not provided.")
        else:
            st.dataframe(state.parsed.costs.head(200), use_container_width=True)

with tabs[1]:
    st.subheader("Validation")
    if state.units is None:
        st.info("Run analysis to validate units.")
    else:
        rep, min_n = _format_replication(state.units)
        st.write("Replication by treatment:", rep)
        if min_n < 2:
            st.warning("Inferential statistics will be disabled (min n per treatment < 2).")

        if state.warnings:
            st.dataframe(_warnings_to_df(state.warnings), use_container_width=True, hide_index=True)
        else:
            st.success("No warnings.")

with tabs[2]:
    st.subheader("Unit KPIs")
    if state.unit_kpis is None:
        st.info("Run analysis to compute KPIs.")
    else:
        st.dataframe(state.unit_kpis, use_container_width=True, hide_index=True)

        metric_options = [c for c in state.unit_kpis.columns if c not in {"trial_id", "unit_type", "unit_id", "treatment"}]
        metric = st.selectbox("Metric to visualize", options=metric_options, index=metric_options.index("fcr") if "fcr" in metric_options else 0)

        import plotly.express as px  # local import to keep startup fast

        c1, c2 = st.columns(2)

        with c1:
            fig = px.box(
                state.unit_kpis,
                x="treatment",
                y=metric,
                points="all",
                title=f"Distribution by treatment: {metric}",
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            means = (
                state.unit_kpis.groupby("treatment", as_index=False)[metric]
                .mean(numeric_only=True)
                .sort_values("treatment")
            )
            fig2 = px.bar(means, x="treatment", y=metric, title=f"Mean by treatment: {metric}")
            st.plotly_chart(fig2, use_container_width=True)

with tabs[3]:
    st.subheader("Statistics")
    if state.stats_table is None:
        st.info("Run analysis to compute statistics (if enabled by replication).")
    else:
        st.dataframe(state.stats_table, use_container_width=True, hide_index=True)

with tabs[4]:
    st.subheader("Export & Notes")
    st.markdown(
        """
**Disclaimers / safety**
- Inferential statistics are only produced when **each treatment** has at least **2 experimental units** (replication).
- Without replication, the app returns **descriptive summaries only** and a warning.
- Always confirm data consistency (e.g., birds_sold vs birds_placed and mortality).

**Report contents**
- CLEANED_INPUT
- UNIT_KPIS
- TREATMENT_SUMMARY
- STATS (if enabled)
- WARNINGS
"""
    )

    if state.report_bytes is None:
        st.caption("Run analysis to enable export.")


# Run analysis pipeline
if run_btn:
    if state.parsed is None:
        st.error("Upload a file first.")
    else:
        try:
            # Parse -> units (normalized, derived fields)
            units = state.parsed.to_units()

            # Validate
            v_opts = ValidationOptions(wg_negative_is_error=wg_negative_is_error)
            _, v_warnings = validate_units(units, options=v_opts)

            # Compute KPIs per unit
            computed_list, c_warnings = compute_all_units(units)
            unit_kpis_df = pd.DataFrame([m.model_dump() for m in computed_list])

            # Treatment summaries
            treatment_summary_df = build_treatment_summary(unit_kpis_df)

            # Statistics (safe)
            metric_list = default_metric_list(unit_kpis_df)
            s_opts = StatsOptions(alpha=float(alpha), enable_posthoc=bool(enable_posthoc))
            inf_res, rep_by_trt, min_n, enabled, s_warnings = run_inferential_statistics(
                computed_list,
                metrics=metric_list,
                options=s_opts,
            )

            stats_df = pd.DataFrame([r.model_dump() for r in inf_res])

            all_warnings = [*v_warnings, *c_warnings, *s_warnings]

            # Export report bytes
            payload = ExportPayload(
                cleaned_input=state.parsed.cleaned_input,
                unit_kpis=unit_kpis_df,
                treatment_summary=treatment_summary_df,
                stats=stats_df,
                warnings=_warnings_to_df(all_warnings),
            )
            report_bytes = export_report_xlsx(payload)

            _set_state(
                units=units,
                warnings=all_warnings,
                unit_kpis=unit_kpis_df,
                treatment_summary=treatment_summary_df,
                stats_table=stats_df,
                report_bytes=report_bytes,
            )
            st.success("Analysis completed.")
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            _set_state(units=None, warnings=[], unit_kpis=None, treatment_summary=None, stats_table=None, report_bytes=None)

Which file should I generate next?
