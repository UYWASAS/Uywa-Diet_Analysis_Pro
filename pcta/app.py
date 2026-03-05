from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from auth import get_current_user, login_ui, logout_button
from core.calculations import compute_all_units
from core.io import export_report_xlsx, parse_uploaded_file
from core.reporting import build_treatment_summary, default_metric_list
from core.schemas import AnalysisWarning, ParsedInput, TrialUnitInput
from core.stats import StatsOptions, run_inferential_statistics
from core.validation import ValidationOptions, validate_units


# ======================== CONFIG + ESTILO ========================

st.set_page_config(page_title="PCTA — Analizador de Ensayos Comerciales", layout="wide")

st.markdown(
    """
    <style>
    html, body, .stApp, .block-container {
        background: linear-gradient(120deg, #ffffff 0%, #eef4fc 100%) !important;
    }

    section[data-testid="stSidebar"] { background-color: #2C3E50 !important; }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] small {
        color: #fff !important;
    }

    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea,
    section[data-testid="stSidebar"] select {
        background: rgba(255,255,255,0.10) !important;
        color: #fff !important;
        border: 1px solid rgba(255,255,255,0.25) !important;
        border-radius: 8px !important;
    }

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

    .block-container { padding: 2rem 3.5rem; }
    footer {visibility: hidden !important;}

    .pcta-card {
        background: #ffffff;
        border: 1px solid #e6eefb;
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0px 2px 10px rgba(16, 24, 40, 0.06);
    }
    .pcta-muted { color: #4b5563; font-size: 0.95rem; }

    .pcta-kpi {
        display:flex; gap:18px; align-items:stretch; flex-wrap:wrap; margin-top: 6px;
    }
    .pcta-kpi > div{
        background:#f7faff; border:1px solid #dbeafe; border-radius:12px; padding:12px 14px; min-width: 180px;
    }
    .pcta-kpi .label{ font-size: 0.82rem; color:#334155; font-weight:700; margin-bottom:4px; }
    .pcta-kpi .value{ font-size: 1.25rem; color:#0f172a; font-weight:900; letter-spacing: -0.01em; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================== LOGIN ========================

if not st.session_state.get("logged_in", False):
    login_ui()

user = get_current_user()
if not user:
    st.error("El usuario no está autenticado.")
    st.stop()

# ======================== ESTADO ========================


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
        return pd.DataFrame(columns=["codigo", "mensaje", "contexto"])
    return pd.DataFrame(
        [{"codigo": w.code.value, "mensaje": w.message, "contexto": w.context} for w in warnings]
    )


def _replication_summary(units: List[TrialUnitInput]) -> Dict[str, int]:
    df = pd.DataFrame([u.model_dump() for u in units])
    return df.groupby("treatment").size().astype(int).to_dict()


def _numeric_kpi_columns(unit_kpis_df: pd.DataFrame) -> List[str]:
    id_cols = {"trial_id", "unit_type", "unit_id", "treatment"}
    cols = [
        c
        for c in unit_kpis_df.columns
        if c not in id_cols and pd.api.types.is_numeric_dtype(unit_kpis_df[c])
    ]
    return cols


_init_state()
state: AppState = st.session_state["pcta_state"]

# ======================== SIDEBAR (MINIMAL) ========================

with st.sidebar:
    logo_path = Path(__file__).resolve().parent / "assets" / "logo.png"
    if logo_path.exists():
        st.image(str(logo_path), use_container_width=True)

    st.markdown(
        f"""
        <div style="text-align:center; margin-top: 6px; margin-bottom: 6px;">
            <h3 style="margin:0; color:#fff; font-family:Montserrat,system-ui,sans-serif;">UYWA Nutrition</h3>
            <p style="font-size:12px; margin:0; color:#fff; opacity:0.95;">PCTA — Ensayos comerciales</p>
            <hr style="border:1px solid rgba(255,255,255,0.35); margin:10px 0;">
            <p style="font-size:12px; color:#fff; margin:0;">Usuario: <b>{user.get("name","")}</b></p>
            <p style="font-size:11px; color:#fff; margin:0; opacity:0.85;">Rol: {user.get("role","")} • {"Premium" if user.get("premium", False) else "Standard"}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    logout_button()

    st.divider()
    st.subheader("Carga de archivos")

    uploaded_main = st.file_uploader(
        "Ensayo (Excel .xlsx o CSV .csv)",
        type=["xlsx", "csv"],
        accept_multiple_files=False,
        key="uploader_main",
    )

    uploaded_costs_extra = st.file_uploader(
        "Costos (opcional, archivo aparte .xlsx/.csv)",
        type=["xlsx", "csv"],
        accept_multiple_files=False,
        key="uploader_costs_extra",
        help="Si tu archivo principal no trae COSTS, puedes cargar un archivo de costos por separado.",
    )

# ======================== HEADER ========================

st.title("PCTA — Analizador de Ensayos Comerciales Avícolas")
st.markdown(
    "<div class='pcta-muted'>Carga datos, valida, calcula KPIs, compara tratamientos, genera gráficos y exporta reporte.</div>",
    unsafe_allow_html=True,
)

tabs = st.tabs(
    [
        "1) Vista previa",
        "2) Configuración",
        "3) Resultados (KPIs)",
        "4) Estadística",
        "5) Gráficos",
        "6) Exportar",
    ]
)

# ======================== PARSE MAIN FILE ========================

if uploaded_main is not None:
    try:
        parsed = parse_uploaded_file(uploaded_main.name, uploaded_main.getvalue())
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
        st.error(f"No se pudo leer el archivo principal: {e}")
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
    st.subheader("Vista previa del input")
    if state.parsed is None:
        st.info("Carga un archivo principal para ver la vista previa.")
    else:
        dfs = state.parsed.to_dataframes()
        hs = dfs["house_summary"]
        ws = dfs["weigh_samples"]
        cs = dfs["costs"]

        st.markdown(
            f"""
            <div class="pcta-card">
              <b>Modo detectado:</b> {state.parsed.mode.value}
              <div class="pcta-muted" style="margin-top:6px;">
                HOUSE_SUMMARY es obligatorio. WEIGH_SAMPLES y COSTS son opcionales.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**HOUSE_SUMMARY (o tabla única)**")
            st.dataframe(hs.head(200), use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**WEIGH_SAMPLES (opcional)**")
            if ws is None:
                st.caption("No proporcionado.")
            else:
                st.dataframe(ws.head(200), use_container_width=True, hide_index=True)

        st.markdown("**COSTS (opcional)**")
        if cs is None:
            st.caption("No proporcionado dentro del archivo principal.")
        else:
            st.dataframe(cs.head(200), use_container_width=True, hide_index=True)

        if uploaded_costs_extra is not None:
            st.info("También cargaste un archivo de COSTOS aparte. En la siguiente pestaña puedes elegir cómo usarlo.")

# ======================== TAB 2: CONFIGURACIÓN ========================

with tabs[1]:
    st.subheader("Configuración de análisis")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("### Validación")
        wg_negative_is_error = st.checkbox("Bloquear WG negativo (WG < 0)", value=True)
        st.caption("Si está activo, un WG negativo detiene el análisis (error).")

    with colB:
        st.markdown("### Estadística")
        alpha = st.number_input("Nivel de significancia (alpha)", min_value=0.001, max_value=0.2, value=0.05, step=0.005)
        enable_posthoc = st.checkbox("Posthoc (cuando aplique)", value=True)

    st.divider()
    st.markdown("### Datos económicos (no productivos)")
    econ_mode = st.radio(
        "¿Cómo quieres manejar COSTOS?",
        options=[
            "Usar COSTS del archivo principal (si existe)",
            "Usar archivo de costos aparte (si se cargó)",
            "Ingresar costos manuales (simple)",
            "No usar costos (solo productivo)",
        ],
        index=0,
    )

    manual_costs: Dict[str, float] = {}
    if econ_mode == "Ingresar costos manuales (simple)":
        st.caption("Estos valores se aplican a TODOS los tratamientos/unidades (modo simple).")
        c1, c2, c3 = st.columns(3)
        with c1:
            manual_costs["diet_cost_per_kg"] = float(st.number_input("Costo alimento (USD/kg)", min_value=0.0, value=0.0, step=0.01))
        with c2:
            manual_costs["chick_cost_per_bird"] = float(st.number_input("Costo pollito (USD/ave)", min_value=0.0, value=0.0, step=0.01))
        with c3:
            manual_costs["additive_cost_total"] = float(st.number_input("Aditivos total (USD / unidad)", min_value=0.0, value=0.0, step=1.0))
        manual_costs["other_variable_costs_total"] = float(st.number_input("Otros costos variables (USD / unidad)", min_value=0.0, value=0.0, step=1.0))

    st.divider()
    st.markdown("### Variables a analizar (ordenado)")
    st.caption("Primero corre el análisis para ver todos los KPIs disponibles; luego seleccionas cuáles incluir en estadística y gráficos.")
    # La selección real se hace después de computar KPIs; aquí dejamos un placeholder.

    st.divider()
    run_btn = st.button("Correr análisis", type="primary", use_container_width=True)

# ======================== EJECUCIÓN DEL PIPELINE ========================

def _apply_manual_costs_to_units(units: List[TrialUnitInput], costs: Dict[str, float]) -> List[TrialUnitInput]:
    out: List[TrialUnitInput] = []
    for u in units:
        out.append(
            u.model_copy(
                update={
                    "diet_cost_per_kg": costs.get("diet_cost_per_kg", u.diet_cost_per_kg),
                    "additive_cost_total": costs.get("additive_cost_total", u.additive_cost_total),
                    "chick_cost_per_bird": costs.get("chick_cost_per_bird", u.chick_cost_per_bird),
                    "other_variable_costs_total": costs.get("other_variable_costs_total", u.other_variable_costs_total),
                }
            )
        )
    return out


def _parse_extra_costs_file(file) -> Optional[pd.DataFrame]:
    if file is None:
        return None
    p = parse_uploaded_file(file.name, file.getvalue())
    dfs = p.to_dataframes()
    return dfs.get("costs")


if run_btn:
    if state.parsed is None:
        st.error("Primero carga el archivo principal.")
    else:
        try:
            units = state.parsed.to_units()

            # Opcional: costos extra o manuales
            if econ_mode == "Usar archivo de costos aparte (si se cargó)":
                extra_costs_df = _parse_extra_costs_file(uploaded_costs_extra)
                if extra_costs_df is None or extra_costs_df.empty:
                    st.warning("No se detectaron COSTOS en el archivo aparte. Se continúa sin costos.")
                else:
                    # Reusar merge del core: lo más simple es reconstruir ParsedInput con costs_records,
                    # pero para no tocar core ahora, avisamos y dejamos en backlog.
                    st.warning(
                        "Modo COSTOS aparte: pendiente de integración completa al core (merge directo). "
                        "Por ahora usa COSTS en el archivo principal o el modo manual."
                    )

            if econ_mode == "Ingresar costos manuales (simple)":
                units = _apply_manual_costs_to_units(units, manual_costs)

            if econ_mode == "No usar costos (solo productivo)":
                units = _apply_manual_costs_to_units(
                    units,
                    {
                        "diet_cost_per_kg": None,
                        "additive_cost_total": 0.0,
                        "chick_cost_per_bird": 0.0,
                        "other_variable_costs_total": 0.0,
                    },
                )

            # Validación
            v_opts = ValidationOptions(wg_negative_is_error=bool(wg_negative_is_error))
            _, v_warnings = validate_units(units, options=v_opts)

            # KPIs
            computed, c_warnings = compute_all_units(units)
            unit_kpis_df = pd.DataFrame([m.model_dump() for m in computed])

            # Summary
            treatment_summary_df = build_treatment_summary(unit_kpis_df)

            # Métricas por defecto
            metrics_default = default_metric_list(unit_kpis_df)

            # Stats (usa default por ahora; luego permitimos elegir)
            s_opts = StatsOptions(alpha=float(alpha), enable_posthoc=bool(enable_posthoc))
            stats_df, rep_by_trt, min_n, enabled, s_warnings = run_inferential_statistics(
                computed,
                metrics=metrics_default,
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
            st.success("Análisis completado. Ve a pestañas de Resultados / Estadística / Gráficos.")
        except Exception as e:
            st.error(f"Falló el análisis: {e}")
            _set_state(
                units=None,
                unit_kpis_df=None,
                treatment_summary_df=None,
                stats_df=None,
                warnings=[],
                report_bytes=None,
            )

# ======================== TAB 3: RESULTADOS ========================

with tabs[2]:
    st.subheader("Resultados: KPIs por unidad + resumen por tratamiento")

    if state.units is None or state.unit_kpis_df is None:
        st.info("Corre el análisis en la pestaña Configuración.")
    else:
        rep = _replication_summary(state.units)
        min_n = int(min(rep.values())) if rep else 0

        st.markdown(
            f"""
            <div class="pcta-card">
              <b>Replicación por tratamiento:</b> {rep}<br>
              <b>Mínimo n por tratamiento:</b> {min_n}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if state.warnings:
            st.markdown("#### Advertencias")
            st.dataframe(_warnings_df(state.warnings), use_container_width=True, hide_index=True)

        st.markdown("#### KPIs por unidad")
        st.dataframe(state.unit_kpis_df, use_container_width=True, hide_index=True)

        st.markdown("#### Resumen por tratamiento (descriptivo)")
        if state.treatment_summary_df is not None:
            st.dataframe(state.treatment_summary_df, use_container_width=True, hide_index=True)

# ======================== TAB 4: ESTADÍSTICA (SELECCIÓN DE VARIABLES) ========================

with tabs[3]:
    st.subheader("Estadística inferencial (segura)")

    if state.units is None or state.unit_kpis_df is None or state.stats_df is None:
        st.info("Corre el análisis primero.")
    else:
        # Selector ordenado de variables: ahora SÍ puedes elegir
        kpi_cols = _numeric_kpi_columns(state.unit_kpis_df)
        suggested = [m for m in default_metric_list(state.unit_kpis_df) if m in kpi_cols]

        metrics = st.multiselect(
            "Selecciona variables (KPIs) para analizar",
            options=kpi_cols,
            default=suggested[: min(8, len(suggested))],
            help="Selecciona las variables que quieres incluir en la tabla de estadística.",
        )

        if st.button("Recalcular estadística con variables seleccionadas", type="primary"):
            try:
                # Re-calcular stats con métricas elegidas (sin re-calcular todo)
                # Necesitamos los modelos 'computed'. No los guardamos: por simplicidad recalculamos desde units.
                computed, c_warnings = compute_all_units(state.units)
                s_opts = StatsOptions(alpha=float(alpha), enable_posthoc=bool(enable_posthoc))
                stats_df, rep_by_trt, min_n, enabled, s_warnings = run_inferential_statistics(
                    computed,
                    metrics=metrics,
                    options=s_opts,
                )
                # Mantener warnings anteriores + nuevas de stats
                merged_warnings = list(state.warnings) + list(s_warnings)
                _set_state(stats_df=stats_df, warnings=merged_warnings)
                st.success("Estadística actualizada.")
            except Exception as e:
                st.error(f"No se pudo recalcular estadística: {e}")

        st.markdown("#### Tabla de estadística")
        st.dataframe(state.stats_df, use_container_width=True, hide_index=True)

# ======================== TAB 5: GRÁFICOS ========================

with tabs[4]:
    st.subheader("Gráficos")

    if state.unit_kpis_df is None:
        st.info("Corre el análisis primero.")
    else:
        try:
            import plotly.express as px
        except Exception:
            st.error("No está instalado Plotly. Revisa `pcta/requirements.txt`.")
            st.stop()

        kpi_cols = _numeric_kpi_columns(state.unit_kpis_df)
        if not kpi_cols:
            st.warning("No hay KPIs numéricos para graficar.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                metric = st.selectbox("Variable (KPI)", options=kpi_cols, index=0)
            with c2:
                chart_type = st.selectbox("Tipo de gráfico", options=["Boxplot", "Violín", "Barras (media)"], index=0)
            with c3:
                scale = st.selectbox("Escala", options=["Lineal", "Log"], index=0)

            fmt_col1, fmt_col2 = st.columns(2)
            with fmt_col1:
                decimals = st.slider("Decimales", min_value=0, max_value=6, value=2)
            with fmt_col2:
                as_percent = st.checkbox("Mostrar como %", value=False, help="Solo cambia formato visual.")

            df = state.unit_kpis_df.copy()
            y = metric
            if as_percent:
                df["_metric_fmt"] = df[y] * 100.0
                y_plot = "_metric_fmt"
                y_label = f"{metric} (%)"
            else:
                y_plot = y
                y_label = metric

            if chart_type == "Boxplot":
                fig = px.box(df, x="treatment", y=y_plot, points="all", template="simple_white", title=f"{y_label} por tratamiento")
            elif chart_type == "Violín":
                fig = px.violin(df, x="treatment", y=y_plot, box=True, points="all", template="simple_white", title=f"{y_label} por tratamiento")
            else:
                means = df.groupby("treatment", as_index=False)[y_plot].mean(numeric_only=True)
                fig = px.bar(means, x="treatment", y=y_plot, template="simple_white", title=f"Media de {y_label} por tratamiento")

            if scale == "Log":
                fig.update_yaxes(type="log")

            fig.update_traces(hovertemplate="%{x}<br>%{y:." + str(decimals) + "f}<extra></extra>")
            fig.update_layout(yaxis_title=y_label, xaxis_title="Tratamiento", title_font=dict(size=16))
            st.plotly_chart(fig, use_container_width=True)

# ======================== TAB 6: EXPORTAR ========================

with tabs[5]:
    st.subheader("Exportar")

    if state.report_bytes is None:
        st.info("Corre el análisis primero para habilitar el reporte.")
    else:
        st.download_button(
            "Descargar reporte Excel",
            data=state.report_bytes,
            file_name="pcta_reporte.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
