"""
Bloques editables para la app Streamlit (PCTA).

Objetivo:
- Partir el código del app en funciones pequeñas (bloques) para editar sin regenerar todo.
- Mantener un único punto de entrada en pcta/app.py, pero delegando UI y lógica a este módulo.

Este archivo contiene:
- Render de sidebar minimal
- Pestañas: Vista previa, Configuración, Resultados, Estadística, Gráficos, Exportar
- Helpers UI (toggle de tablas, selección de variable, resumen descriptivo)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from pcta.auth import logout_button
from pcta.core.calculations import compute_all_units
from pcta.core.io import export_report_xlsx, parse_uploaded_file
from pcta.core.reporting import build_treatment_summary, default_metric_list
from pcta.core.schemas import AnalysisWarning, ExportPayload, ParsedInput, TrialUnitInput
from pcta.core.stats import StatsOptions, run_inferential_statistics
from pcta.core.validation import ValidationOptions, validate_units


# ------------------------ Estado ------------------------


@dataclass(frozen=True)
class AppState:
    parsed: Optional[ParsedInput]
    units: Optional[List[TrialUnitInput]]
    unit_kpis_df: Optional[pd.DataFrame]
    treatment_summary_df: Optional[pd.DataFrame]
    stats_df: Optional[pd.DataFrame]
    warnings: List[AnalysisWarning]
    report_bytes: Optional[bytes]


def init_state() -> None:
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


def get_state() -> AppState:
    return st.session_state["pcta_state"]


def set_state(**kwargs) -> None:
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


# ------------------------ Helpers ------------------------


def warnings_df(warnings: List[AnalysisWarning]) -> pd.DataFrame:
    if not warnings:
        return pd.DataFrame(columns=["codigo", "mensaje", "contexto"])
    return pd.DataFrame(
        [{"codigo": w.code.value, "mensaje": w.message, "contexto": w.context} for w in warnings]
    )


def replication_summary(units: List[TrialUnitInput]) -> Dict[str, int]:
    df = pd.DataFrame([u.model_dump() for u in units])
    return df.groupby("treatment").size().astype(int).to_dict()


def numeric_kpi_columns(unit_kpis_df: pd.DataFrame) -> List[str]:
    id_cols = {"trial_id", "unit_type", "unit_id", "treatment"}
    cols = [
        c
        for c in unit_kpis_df.columns
        if c not in id_cols and pd.api.types.is_numeric_dtype(unit_kpis_df[c])
    ]
    return cols


def describe_by_treatment(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Estadística descriptiva por tratamiento para una variable.
    Devuelve: n, media, sd, min, p25, mediana, p75, max
    """
    if metric not in df.columns:
        return pd.DataFrame()

    g = (
        df.groupby("treatment")[metric]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .rename(
            columns={
                "count": "n",
                "mean": "media",
                "std": "sd",
                "min": "min",
                "median": "mediana",
                "max": "max",
            }
        )
        .reset_index()
    )

    pcts = (
        df.groupby("treatment")[metric]
        .quantile([0.25, 0.75])
        .unstack(level=1)
        .reset_index()
        .rename(columns={0.25: "p25", 0.75: "p75"})
    )

    out = g.merge(pcts, on="treatment", how="left")
    return out


# ------------------------ Sidebar minimal ------------------------


@dataclass(frozen=True)
class SidebarResult:
    uploaded_main: Optional[object]
    uploaded_costs_extra: Optional[object]


def render_sidebar_minimal(*, user: Dict[str, object]) -> SidebarResult:
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

        # Solo cerrar sesión + carga de archivos
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
        )

    return SidebarResult(uploaded_main=uploaded_main, uploaded_costs_extra=uploaded_costs_extra)


# ------------------------ Parse ------------------------


def maybe_parse_main_upload(uploaded_main: Optional[object]) -> None:
    """
    Si subieron archivo principal, lo parsea y guarda `parsed` en el estado.
    """
    if uploaded_main is None:
        return

    try:
        parsed = parse_uploaded_file(uploaded_main.name, uploaded_main.getvalue())
        set_state(
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
        set_state(
            parsed=None,
            units=None,
            unit_kpis_df=None,
            treatment_summary_df=None,
            stats_df=None,
            warnings=[],
            report_bytes=None,
        )


# ------------------------ Pestaña 1: Vista previa ------------------------


def tab_preview() -> None:
    st.subheader("Vista previa del input")

    state = get_state()
    if state.parsed is None:
        st.info("Carga un archivo principal para ver la vista previa.")
        return

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

    st.markdown("### Mostrar/Ocultar tablas")
    show_hs = st.toggle("Mostrar HOUSE_SUMMARY", value=True)
    show_ws = st.toggle("Mostrar WEIGH_SAMPLES (si existe)", value=False)
    show_cs = st.toggle("Mostrar COSTS (si existe)", value=False)

    if show_hs:
        st.markdown("**HOUSE_SUMMARY (o tabla única)**")
        st.dataframe(hs.head(500), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        if show_ws:
            st.markdown("**WEIGH_SAMPLES (opcional)**")
            if ws is None:
                st.caption("No proporcionado.")
            else:
                st.dataframe(ws.head(500), use_container_width=True, hide_index=True)
    with c2:
        if show_cs:
            st.markdown("**COSTS (opcional)**")
            if cs is None:
                st.caption("No proporcionado dentro del archivo principal.")
            else:
                st.dataframe(cs.head(500), use_container_width=True, hide_index=True)


# ------------------------ Pestaña 2: Configuración + Correr análisis ------------------------


@dataclass(frozen=True)
class RunConfig:
    wg_negative_is_error: bool
    alpha: float
    enable_posthoc: bool
    econ_mode: str
    manual_costs: Dict[str, float]
    metrics_selected: List[str]


def tab_config_and_run(*, uploaded_costs_extra: Optional[object]) -> None:
    """
    Pestaña de configuración.
    - Define validación, alpha/posthoc, costos manuales.
    - Permite seleccionar variables a analizar (si ya hay KPIs).
    - Botón para correr análisis.
    """
    st.subheader("Configuración")

    state = get_state()

    colA, colB = st.columns(2)
    with colA:
        st.markdown("### Validación")
        wg_negative_is_error = st.checkbox("Bloquear WG negativo (WG < 0)", value=True)
    with colB:
        st.markdown("### Estadística")
        alpha = st.number_input(
            "Nivel de significancia (alpha)",
            min_value=0.001,
            max_value=0.2,
            value=0.05,
            step=0.005,
        )
        enable_posthoc = st.checkbox("Posthoc (cuando aplique)", value=True)

    st.divider()
    st.markdown("### Datos económicos (no productivos)")
    econ_mode = st.radio(
        "¿Cómo quieres manejar COSTOS?",
        options=[
            "Usar COSTS del archivo principal (si existe)",
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
            manual_costs["diet_cost_per_kg"] = float(
                st.number_input("Costo alimento (USD/kg)", min_value=0.0, value=0.0, step=0.01)
            )
        with c2:
            manual_costs["chick_cost_per_bird"] = float(
                st.number_input("Costo pollito (USD/ave)", min_value=0.0, value=0.0, step=0.01)
            )
        with c3:
            manual_costs["additive_cost_total"] = float(
                st.number_input("Aditivos total (USD / unidad)", min_value=0.0, value=0.0, step=1.0)
            )
        manual_costs["other_variable_costs_total"] = float(
            st.number_input("Otros costos variables (USD / unidad)", min_value=0.0, value=0.0, step=1.0)
        )

    st.divider()
    st.markdown("### Variables a analizar")
    if state.unit_kpis_df is None:
        st.info("Corre el análisis una vez para ver todos los KPIs disponibles y poder seleccionarlos.")
        metrics_selected: List[str] = []
    else:
        kpi_cols = numeric_kpi_columns(state.unit_kpis_df)
        suggested = [m for m in default_metric_list(state.unit_kpis_df) if m in kpi_cols]
        metrics_selected = st.multiselect(
            "Selecciona variables (KPIs) para estadística y gráficos",
            options=kpi_cols,
            default=suggested[: min(8, len(suggested))],
        )

    st.divider()
    run_btn = st.button(
        "Correr análisis",
        type="primary",
        use_container_width=True,
        disabled=(state.parsed is None),
    )

    if run_btn:
        run_analysis(
            RunConfig(
                wg_negative_is_error=bool(wg_negative_is_error),
                alpha=float(alpha),
                enable_posthoc=bool(enable_posthoc),
                econ_mode=str(econ_mode),
                manual_costs=manual_costs,
                metrics_selected=metrics_selected,
            )
        )


def apply_manual_costs_to_units(units: List[TrialUnitInput], costs: Dict[str, float]) -> List[TrialUnitInput]:
    out: List[TrialUnitInput] = []
    for u in units:
        out.append(
            u.model_copy(
                update={
                    "diet_cost_per_kg": costs.get("diet_cost_per_kg", u.diet_cost_per_kg),
                    "additive_cost_total": costs.get("additive_cost_total", u.additive_cost_total),
                    "chick_cost_per_bird": costs.get("chick_cost_per_bird", u.chick_cost_per_bird),
                    "other_variable_costs_total": costs.get(
                        "other_variable_costs_total", u.other_variable_costs_total
                    ),
                }
            )
        )
    return out


def run_analysis(cfg: RunConfig) -> None:
    state = get_state()
    if state.parsed is None:
        st.error("Primero carga el archivo principal.")
        return

    try:
        units = state.parsed.to_units()

        if cfg.econ_mode == "Ingresar costos manuales (simple)":
            units = apply_manual_costs_to_units(units, cfg.manual_costs)

        if cfg.econ_mode == "No usar costos (solo productivo)":
            units = apply_manual_costs_to_units(
                units,
                {
                    "diet_cost_per_kg": None,
                    "additive_cost_total": 0.0,
                    "chick_cost_per_bird": 0.0,
                    "other_variable_costs_total": 0.0,
                },
            )

        _, v_warnings = validate_units(
            units, options=ValidationOptions(wg_negative_is_error=cfg.wg_negative_is_error)
        )

        computed, c_warnings = compute_all_units(units)
        unit_kpis_df = pd.DataFrame([m.model_dump() for m in computed])

        treatment_summary_df = build_treatment_summary(unit_kpis_df)

        metrics = cfg.metrics_selected or default_metric_list(unit_kpis_df)

        stats_df, rep_by_trt, min_n, enabled, s_warnings = run_inferential_statistics(
            computed,
            metrics=metrics,
            options=StatsOptions(alpha=cfg.alpha, enable_posthoc=cfg.enable_posthoc),
        )

        all_warnings = [*v_warnings, *c_warnings, *s_warnings]

        report_bytes = export_report_xlsx(
            ExportPayload(
                cleaned_input=state.parsed.cleaned_input,
                unit_kpis=unit_kpis_df,
                treatment_summary=treatment_summary_df,
                stats=stats_df,
                warnings=warnings_df(all_warnings),
            )
        )

        set_state(
            units=units,
            unit_kpis_df=unit_kpis_df,
            treatment_summary_df=treatment_summary_df,
            stats_df=stats_df,
            warnings=all_warnings,
            report_bytes=report_bytes,
        )
        st.success("Análisis completado.")
    except Exception as e:
        st.error(f"Falló el análisis: {e}")
        set_state(
            units=None,
            unit_kpis_df=None,
            treatment_summary_df=None,
            stats_df=None,
            warnings=[],
            report_bytes=None,
        )


# ------------------------ Pestaña 3: Resultados ------------------------


def tab_results() -> None:
    st.subheader("Resultados")

    state = get_state()
    if state.units is None or state.unit_kpis_df is None:
        st.info("Corre el análisis en Configuración.")
        return

    rep = replication_summary(state.units)
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
        st.dataframe(warnings_df(state.warnings), use_container_width=True, hide_index=True)

    st.markdown("#### KPIs por unidad")
    st.dataframe(state.unit_kpis_df, use_container_width=True, hide_index=True)

    st.markdown("#### Resumen por tratamiento (descriptivo general)")
    if state.treatment_summary_df is not None:
        st.dataframe(state.treatment_summary_df, use_container_width=True, hide_index=True)


# ------------------------ Pestaña 4: Estadística ------------------------


def tab_stats() -> None:
    st.subheader("Estadística")

    state = get_state()
    if state.unit_kpis_df is None or state.stats_df is None:
        st.info("Corre el análisis primero.")
        return

    kpi_cols = numeric_kpi_columns(state.unit_kpis_df)
    if not kpi_cols:
        st.warning("No hay KPIs numéricos para analizar.")
        return

    st.markdown("### 1) Descriptiva por variable (selecciona una)")
    suggested = [m for m in default_metric_list(state.unit_kpis_df) if m in kpi_cols]
    default_var = suggested[0] if suggested else kpi_cols[0]
    var = st.selectbox("Variable principal", options=kpi_cols, index=kpi_cols.index(default_var))

    desc = describe_by_treatment(state.unit_kpis_df, var)
    if not desc.empty:
        st.dataframe(desc, use_container_width=True, hide_index=True)
    else:
        st.info("No se pudo calcular descriptiva para esa variable.")

    st.divider()
    st.markdown("### 2) Inferencial (segura) — tabla")
    st.dataframe(state.stats_df, use_container_width=True, hide_index=True)


# ------------------------ Pestaña 5: Gráficos ------------------------


def tab_charts() -> None:
    st.subheader("Gráficos")

    state = get_state()
    if state.unit_kpis_df is None:
        st.info("Corre el análisis primero.")
        return

    try:
        import plotly.express as px
    except Exception:
        st.error("No está instalado Plotly. Revisa `pcta/requirements.txt`.")
        return

    kpi_cols = numeric_kpi_columns(state.unit_kpis_df)
    if not kpi_cols:
        st.warning("No hay KPIs numéricos para graficar.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        metric = st.selectbox("Variable (KPI)", options=kpi_cols, index=0, key="chart_metric")
    with c2:
        chart_type = st.selectbox(
            "Tipo de gráfico", options=["Boxplot", "Violín", "Barras (media)"], index=0, key="chart_type"
        )
    with c3:
        scale = st.selectbox("Escala", options=["Lineal", "Log"], index=0, key="chart_scale")

    fmt_col1, fmt_col2 = st.columns(2)
    with fmt_col1:
        decimals = st.slider("Decimales", min_value=0, max_value=6, value=2, key="chart_decimals")
    with fmt_col2:
        as_percent = st.checkbox("Mostrar como %", value=False, key="chart_percent")

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
        fig = px.box(
            df,
            x="treatment",
            y=y_plot,
            points="all",
            template="simple_white",
            title=f"{y_label} por tratamiento",
        )
    elif chart_type == "Violín":
        fig = px.violin(
            df,
            x="treatment",
            y=y_plot,
            box=True,
            points="all",
            template="simple_white",
            title=f"{y_label} por tratamiento",
        )
    else:
        means = df.groupby("treatment", as_index=False)[y_plot].mean(numeric_only=True)
        fig = px.bar(
            means,
            x="treatment",
            y=y_plot,
            template="simple_white",
            title=f"Media de {y_label} por tratamiento",
        )

    if scale == "Log":
        fig.update_yaxes(type="log")

    fig.update_traces(hovertemplate="%{x}<br>%{y:." + str(decimals) + "f}<extra></extra>")
    fig.update_layout(yaxis_title=y_label, xaxis_title="Tratamiento", title_font=dict(size=16))
    st.plotly_chart(fig, use_container_width=True)


# ------------------------ Pestaña 6: Exportar ------------------------


def tab_export() -> None:
    st.subheader("Exportar")
    state = get_state()
    if state.report_bytes is None:
        st.info("Corre el análisis primero para habilitar el reporte.")
        return

    st.download_button(
        "Descargar reporte Excel",
        data=state.report_bytes,
        file_name="pcta_reporte.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
