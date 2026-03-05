"""
PCTA - Poultry Commercial Trial Analyzer (Streamlit)

UI:
- Login (multi-user, similar to the user's pet app)
- Upload Excel/CSV
- Preview raw/cleaned inputs
- Validate + warnings
- Compute per-unit KPIs
- Descriptive summaries and (safe) inferential stats
- Export an Excel report

CRITICAL SAFETY:
- Inferential statistics are disabled if min replication per treatment < 2 (handled in core.stats).

Branding/theming:
- Sidebar branding block with logo (pcta/assets/logo.png)
- Global CSS to match corporate style (gradient background, dark sidebar, rounded buttons)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from auth import get_current_user, login_ui, logout_button
from core.calculations import compute_all_units
from core.io import export_report_xlsx, parse_uploaded_file
from core.reporting import build_treatment_summary, default_metric_list
from core.schemas import AnalysisWarning, ParsedInput, TrialUnitInput
from core.stats import StatsOptions, run_inferential_statistics
from core.validation import ValidationOptions, validate_units


# ======================== BLOQUE 1: CONFIG + ESTILO CORPORATIVO ========================

st.set_page_config(page_title="PCTA — Poultry Commercial Trial Analyzer", layout="wide")

st.markdown(
    """
    <style>
    /* App background */
    html, body, .stApp, .block-container {
        background: linear-gradient(120deg, #ffffff 0%, #eef4fc 100%) !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #2C3E50 !important;
    }

    /* Sidebar text ONLY (avoid overriding widgets too aggressively) */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] small {
        color: #fff !important;
    }

    /* Sidebar inputs: ensure contrast */
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea,
    section[data-testid="stSidebar"] select {
        background: rgba(255,255,255,0.10) !important;
        color: #fff !important;
        border: 1px solid rgba(255,255,255,0.25) !important;
        border-radius: 8px !important;
    }

    /* Buttons */
    .stButton > button, .stDownloadButton > button {
        background-color: #2176ff !important;
        color: #fff !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.55rem 1.0rem !important;
        font-weight: 700 !important;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background-color: #1254d1 !important;
        color: #fff !important;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.18) !important;
    }

    /* Main container padding */
    .block-container {
        padding: 2rem 3.5rem;
    }

    /* Hide Streamlit footer */
    footer {visibility: hidden !important;}

    /* Cards */
    .pcta-card {
        background: #ffffff;
        border: 1px solid #e6eefb;
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0px 2px 10px rgba(16, 24, 40, 0.06);
    }

    .pcta-card-dark {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 14px;
        padding: 12px 14px;
    }

    .pcta-muted {
        color: #4b5563;
        font-size: 0.95rem;
    }

    .pcta-kpi {
        display:flex;
        gap:18px;
        align-items:stretch;
        flex-wrap:wrap;
        margin-top: 6px;
    }
    .pcta-kpi > div{
        background:#f7faff;
        border:1px solid #dbeafe;
        border-radius:12px;
        padding:12px 14px;
        min-width: 180px;
    }
    .pcta-kpi .label{
        font-size: 0.82rem;
        color:#334155;
        font-weight:700;
        margin-bottom:4px;
    }
    .pcta-kpi .value{
        font-size: 1.25rem;
        color:#0f172a;
        font-weight:900;
        letter-spacing: -0.01em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================== BLOQUE 2: LOGIN GATE ========================

if not st.session_state.get("logged_in", False):
    login_ui()

user = get_current_user()
if not user:
    # Extra safety; login_ui should st.stop() already.
    st.error("El usuario no está autenticado.")
    st.stop()

# ======================== BLOQUE 3: ESTADO DE SESIÓN ========================


@dataclass(frozen=True)
class AppState:
    parsed: Optional[ParsedInput]
    units: Optional[List[TrialUnitInput]]

    unit_kpis_df: Optional[pd.DataFrame]
    treatment_summary_df: Optional[pd.DataFrame]
    stats_df: Optional[pd.DataFrame]

    warnings: List[AnalysisWarning]
    report_bytes: Optional[bytes]


def _init_state() -> None:
    if "pcta_state" not in st.session_state:
        st.session_state["pcta_state"] = AppState(
            parsed=None,
            units=None,
            unit_kpis_df=None,
            treatment_summary_df=None,
            stats_df=None,
            warnings=[],
            report_bytes=None,
        )


def _set_state(**kwargs) -> None:
    cur: AppState = st.session_state["pcta_state"]
    st.session_state["pcta_state"] = AppState(
        parsed=kwargs.get("parsed", cur.parsed),
        units=kwargs.get("units", cur.units),
        unit_kpis_df=kwargs.get("unit_kpis_df", cur.unit_kpis_df),
        treatment_summary_df=kwargs.get("treatment_summary_df", cur.treatment_summary_df),
        stats_df=kwargs.get("stats_df", cur.stats_df),
        warnings=kwargs.get("warnings", cur.warnings),
        report_bytes=kwargs.get("report_bytes", cur.report_bytes),
    )


def _warnings_df(warnings: List[AnalysisWarning]) -> pd.DataFrame:
    if not warnings:
        return pd.DataFrame(columns=["code", "message", "context"])
    return pd.DataFrame([{"code": w.code.value, "message": w.message, "context": w.context} for w in warnings])


def _replication_summary(units: List[TrialUnitInput]) -> Dict[str, int]:
    df = pd.DataFrame([u.model_dump() for u in units])
    return df.groupby("treatment").size().astype(int).to_dict()


_init_state()
state: AppState = st.session_state["pcta_state"]

# ======================== BLOQUE 4: SIDEBAR CORPORATIVO ========================

with st.sidebar:
    # Logo: pcta/assets/logo.png (robust path)
    logo_path = Path(__file__).resolve().parent / "assets" / "logo.png"
    if logo_path.exists():
        st.image(str(logo_path), use_container_width=True)
    else:
        st.markdown(
            """
            <div class="pcta-card-dark">
              <div style="text-align:center; font-weight:900; font-size: 1.05rem; color:#fff;">PCTA</div>
              <div style="text-align:center; font-size: 0.85rem; opacity:0.95; color:#fff;">Poultry Commercial Trial Analyzer</div>
              <div style="text-align:center; font-size: 0.78rem; opacity:0.85; margin-top:6px; color:#fff;">
                Logo missing: expected <code style="color:#fff;">pcta/assets/logo.png</code>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div style="text-align:center; margin-top: 10px; margin-bottom: 10px;">
            <h2 style="font-family:Montserrat,system-ui,sans-serif; margin:0; color:#fff;">
                UYWA Nutrition
            </h2>
            <p style="font-size:13px; margin:0; color:#fff; opacity:0.95;">
                Nutrición de Precisión • Evidencia
            </p>
            <br>
            <hr style="border:1px solid rgba(255,255,255,0.35);">
            <p style="font-size:12px; color:#fff; margin:0;">
                Usuario: <b>{user.get("name","")}</b>
            </p>
            <p style="font-size:11px; color:#fff; margin:0; opacity:0.85;">
                Rol: {user.get("role","")} • {"Premium" if user.get("premium", False) else "Standard"}
            </p>
            <p style="font-size:11px; color:#fff; margin:0; opacity:0.85;">
                © 2026 — Derechos reservados
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    logout_button()

    st.divider()
    st.subheader("Upload")
    uploaded = st.file_uploader(
        "Excel (.xlsx) or CSV (.csv)",
        type=["xlsx", "csv"],
        accept_multiple_files=False,
        label_visibility="visible",
    )

    st.divider()
    st.subheader("Options")
    alpha = st.number_input("alpha", min_value=0.001, max_value=0.2, value=0.05, step=0.005)
    enable_posthoc = st.checkbox("Posthoc (when applicable)", value=True)

    st.divider()
    st.subheader("Validation")
    wg_negative_is_error = st.checkbox("Block negative WG (WG < 0)", value=True)

    st.divider()
    run_btn = st.button("Run analysis", type="primary", use_container_width=True)

    st.divider()
    st.subheader("Export")
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

# ======================== BLOQUE 5: HEADER + TABS ========================

st.title("PCTA — Poultry Commercial Trial Analyzer")
st.markdown(
    "<div class='pcta-muted'>Carga datos de ensayo, valida, calcula KPIs, compara tratamientos y exporta reporte.</div>",
    unsafe_allow_html=True,
)

tabs = st.tabs(["1) Preview", "2) Validation", "3) KPIs", "4) Stats", "5) Export & Notes"])

# ======================== BLOQUE 6: PARSE (ON UPLOAD) ========================

if uploaded is not None:
    try:
        parsed = parse_uploaded_file(uploaded.name, uploaded.getvalue())
        _set_state(
            parsed=parsed,
            units=None,
            unit_kpis_df=None,
            treatment_summary_df=None,
            stats_df=None,
            warnings=[],
            report_bytes=None,
        )
    except Exception as e:
        st.error(f"Failed to parse file: {e}")
        _set_state(
            parsed=None,
            units=None,
            unit_kpis_df=None,
            treatment_summary_df=None,
            stats_df=None,
            warnings=[],
            report_bytes=None,
        )

# ======================== TAB 1: PREVIEW ========================

with tabs[0]:
    st.subheader("Preview")
    if state.parsed is None:
        st.info("Upload a file to preview inputs.")
    else:
        dfs = state.parsed.to_dataframes()
        hs = dfs["house_summary"]
        ws = dfs["weigh_samples"]
        cs = dfs["costs"]

        st.markdown(
            f"""
            <div class="pcta-card">
              <b>Detected input mode:</b> {state.parsed.mode.value}
              <div class="pcta-muted" style="margin-top:6px;">
                HOUSE_SUMMARY is required. WEIGH_SAMPLES and COSTS are optional.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**HOUSE_SUMMARY (or single-table)**")
            st.dataframe(hs.head(200), use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**WEIGH_SAMPLES (optional)**")
            if ws is None:
                st.caption("Not provided.")
            else:
                st.dataframe(ws.head(200), use_container_width=True, hide_index=True)

        st.markdown("**COSTS (optional)**")
        if cs is None:
            st.caption("Not provided.")
        else:
            st.dataframe(cs.head(200), use_container_width=True, hide_index=True)

# ======================== TAB 2: VALIDATION ========================

with tabs[1]:
    st.subheader("Validation")
    if state.units is None:
        st.info("Run analysis to validate.")
    else:
        rep = _replication_summary(state.units)
        min_n = int(min(rep.values())) if rep else 0

        st.markdown(
            f"""
            <div class="pcta-card">
              <b>Replication by treatment:</b> {rep}<br>
              <b>Min n per treatment:</b> {min_n}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if state.warnings:
            st.markdown("#### Warnings")
            st.dataframe(_warnings_df(state.warnings), use_container_width=True, hide_index=True)
        else:
            st.success("No warnings.")

# ======================== TAB 3: KPIs + FIGURES ========================

with tabs[2]:
    st.subheader("Unit KPIs")
    if state.unit_kpis_df is None:
        st.info("Run analysis to compute KPIs.")
    else:
        st.dataframe(state.unit_kpis_df, use_container_width=True, hide_index=True)

        id_cols = {"trial_id", "unit_type", "unit_id", "treatment"}
        metric_options = [
            c
            for c in state.unit_kpis_df.columns
            if c not in id_cols and pd.api.types.is_numeric_dtype(state.unit_kpis_df[c])
        ]
        if not metric_options:
            st.warning("No numeric KPI columns available to chart.")
        else:
            default_metric = "fcr" if "fcr" in metric_options else metric_options[0]
            metric = st.selectbox(
                "Metric to visualize",
                options=metric_options,
                index=metric_options.index(default_metric),
            )

            g = state.unit_kpis_df
            n_units = int(len(g))
            treatments = sorted(set(g["treatment"].astype(str).tolist()))

            st.markdown(
                f"""
                <div class="pcta-kpi">
                  <div><div class="label">Units</div><div class="value">{n_units}</div></div>
                  <div><div class="label">Treatments</div><div class="value">{len(treatments)}</div></div>
                  <div><div class="label">Selected metric</div><div class="value">{metric}</div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            try:
                import plotly.express as px
            except Exception:
                st.error("Plotly is not installed. Add `plotly` to requirements.txt to enable charts.")
                st.stop()

            c1, c2 = st.columns(2)

            with c1:
                fig = px.box(
                    state.unit_kpis_df,
                    x="treatment",
                    y=metric,
                    points="all",
                    title=f"Distribution by treatment: {metric}",
                    template="simple_white",
                )
                fig.update_layout(title_font=dict(size=16), xaxis_title="Treatment", yaxis_title=metric)
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                means = (
                    state.unit_kpis_df.groupby("treatment", as_index=False)[metric]
                    .mean(numeric_only=True)
                    .sort_values("treatment")
                )
                fig2 = px.bar(
                    means,
                    x="treatment",
                    y=metric,
                    title=f"Mean by treatment: {metric}",
                    template="simple_white",
                )
                fig2.update_layout(title_font=dict(size=16), xaxis_title="Treatment", yaxis_title=metric)
                st.plotly_chart(fig2, use_container_width=True)

# ======================== TAB 4: STATS ========================

with tabs[3]:
    st.subheader("Statistics (safe)")
    if state.stats_df is None:
        st.info("Run analysis to compute statistics (requires replication).")
    else:
        st.dataframe(state.stats_df, use_container_width=True, hide_index=True)

# ======================== TAB 5: EXPORT + NOTES ========================

with tabs[4]:
    st.subheader("Export & Notes")
    st.markdown(
        """
        <div class="pcta-card">
          <b>Safety / disclaimers</b>
          <ul>
            <li>Inferential statistics are only produced when <b>each treatment</b> has at least <b>2 experimental units</b> (replication).</li>
            <li>Without replication, PCTA returns descriptive summaries only and a warning.</li>
            <li>Always confirm data consistency (e.g., birds_sold vs birds_placed and mortality).</li>
          </ul>
          <b>Report sheets</b>
          <ul>
            <li>CLEANED_INPUT</li>
            <li>UNIT_KPIS</li>
            <li>TREATMENT_SUMMARY</li>
            <li>STATS</li>
            <li>WARNINGS</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if state.report_bytes is None:
        st.caption("Run analysis to enable export.")

# ======================== BLOQUE 7: RUN PIPELINE ========================

if run_btn:
    if state.parsed is None:
        st.error("Upload a file first.")
    else:
        try:
            units = state.parsed.to_units()

            v_opts = ValidationOptions(wg_negative_is_error=bool(wg_negative_is_error))
            _, v_warnings = validate_units(units, options=v_opts)

            computed, c_warnings = compute_all_units(units)
            unit_kpis_df = pd.DataFrame([m.model_dump() for m in computed])

            treatment_summary_df = build_treatment_summary(unit_kpis_df)

            metrics = default_metric_list(unit_kpis_df)
            s_opts = StatsOptions(alpha=float(alpha), enable_posthoc=bool(enable_posthoc))
            stats_df, rep_by_trt, min_n, enabled, s_warnings = run_inferential_statistics(
                computed,
                metrics=metrics,
                options=s_opts,
            )

            all_warnings = [*v_warnings, *c_warnings, *s_warnings]

            from core.schemas import ExportPayload

            report_bytes = export_report_xlsx(
                ExportPayload(
                    cleaned_input=state.parsed.cleaned_input,
                    unit_kpis=unit_kpis_df,
                    treatment_summary=treatment_summary_df,
                    stats=stats_df,
                    warnings=_warnings_df(all_warnings),
                )
            )

            _set_state(
                units=units,
                unit_kpis_df=unit_kpis_df,
                treatment_summary_df=treatment_summary_df,
                stats_df=stats_df,
                warnings=all_warnings,
                report_bytes=report_bytes,
            )

            st.success("Analysis completed.")
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            _set_state(
                units=None,
                unit_kpis_df=None,
                treatment_summary_df=None,
                stats_df=None,
                warnings=[],
                report_bytes=None,
            )
