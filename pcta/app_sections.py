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

    # Variable principal elegida por el usuario
    st.session_state.setdefault("primary_metric", None)
    st.session_state.setdefault("primary_metric_label", None)


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
    # ordenar tratamientos alfabéticamente
    out = out.sort_values("treatment").reset_index(drop=True)
    return out


def fmt_num(x: object, decimals: int = 2) -> str:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return ""
        return f"{float(x):,.{decimals}f}"
    except Exception:
        return str(x)


def styled_descriptive_table(desc: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    """
    Devuelve una tabla ya formateada (strings) para mostrar más agradable.
    """
    if desc.empty:
        return desc
    out = desc.copy()
    for col in ["media", "sd", "min", "p25", "mediana", "p75", "max"]:
        if col in out.columns:
            out[col] = out[col].apply(lambda v: fmt_num(v, decimals=decimals))
    if "n" in out.columns:
        out["n"] = out["n"].astype(int)
    return out


# ------------------------ Sidebar minimal ------------------------


@dataclass(frozen=True)
class SidebarResult:
    uploaded_main: Optional[object]


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

        logout_button()

        st.divider()
        st.subheader("Carga de archivos")

        uploaded_main = st.file_uploader(
            "Ensayo (Excel .xlsx o CSV .csv)",
            type=["xlsx", "csv"],
            accept_multiple_files=False,
            key="uploader_main",
        )

    return SidebarResult(uploaded_main=uploaded_main)


# ------------------------ Parse ------------------------


def maybe_parse_main_upload(uploaded_main: Optional[object]) -> None:
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


# ------------------------ Pipeline (correr) ------------------------


def run_analysis(*, alpha: float, enable_posthoc: bool, wg_negative_is_error: bool) -> None:
    state = get_state()
    if state.parsed is None:
        st.error("Primero carga el archivo principal.")
        return

    try:
        units = state.parsed.to_units()

        _, v_warnings = validate_units(
            units, options=ValidationOptions(wg_negative_is_error=wg_negative_is_error)
        )

        computed, c_warnings = compute_all_units(units)
        unit_kpis_df = pd.DataFrame([m.model_dump() for m in computed])

        treatment_summary_df = build_treatment_summary(unit_kpis_df)

        # Stats por defecto (se recalculará/filtrará en pestaña 3)
        metrics_default = default_metric_list(unit_kpis_df)
        stats_df, rep_by_trt, min_n, enabled, s_warnings = run_inferential_statistics(
            computed,
            metrics=metrics_default,
            options=StatsOptions(alpha=alpha, enable_posthoc=enable_posthoc),
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


# ------------------------ TAB 1: Vista previa + variable principal ------------------------


def tab_preview_and_select_variable() -> None:
    st.subheader("1) Vista previa + selección de variable principal")

    state = get_state()
    if state.parsed is None:
        st.info("Carga un archivo principal para empezar.")
        return

    dfs = state.parsed.to_dataframes()
    hs = dfs["house_summary"]

    st.markdown(
        f"""
        <div class="pcta-card">
          <b>Modo detectado:</b> {state.parsed.mode.value}<br>
          <span class="pcta-muted">Primero corre el análisis (pestaña 2) para generar KPIs; luego eliges tu variable principal.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    show_hs = st.toggle("Mostrar HOUSE_SUMMARY", value=False)
    if show_hs:
        st.dataframe(hs.head(300), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Variable principal")
    if state.unit_kpis_df is None:
        st.warning("Aún no hay KPIs. Ve a la pestaña 2 y corre el análisis.")
        return

    kpi_cols = numeric_kpi_columns(state.unit_kpis_df)
    if not kpi_cols:
        st.warning("No se detectaron KPIs numéricos para seleccionar.")
        return

    # Sugerencias "humanas" (puedes expandir luego)
    labels = {
        "bw_final_mean_g": "Peso vivo final (g)",
        "wg_g_per_bird": "Ganancia de peso (g/ave)",
        "adg_g_per_bird_d": "ADG (g/ave/día)",
        "fcr": "FCR",
        "mortality_pct": "Mortalidad (%)",
    }
    options_with_labels = [f"{labels.get(c, c)}  —  ({c})" for c in kpi_cols]

    # Resolver default
    suggested = [c for c in ["bw_final_mean_g", "wg_g_per_bird", "fcr"] if c in kpi_cols]
    default_metric = st.session_state.get("primary_metric") or (suggested[0] if suggested else kpi_cols[0])
    default_idx = kpi_cols.index(default_metric)

    sel = st.selectbox("Elige la variable principal de trabajo", options=options_with_labels, index=default_idx)
    chosen_metric = kpi_cols[options_with_labels.index(sel)]
    st.session_state["primary_metric"] = chosen_metric
    st.session_state["primary_metric_label"] = labels.get(chosen_metric, chosen_metric)

    st.success(f"Variable principal seleccionada: {st.session_state['primary_metric_label']}")


# ------------------------ TAB 2: Descriptiva + gráficos (variable principal) ------------------------


def tab_descriptive_and_charts() -> None:
    st.subheader("2) Estadística descriptiva + gráficos (variable principal)")

    state = get_state()
    if state.unit_kpis_df is None:
        st.info("Corre el análisis primero (en esta misma pestaña, más abajo).")
    metric = st.session_state.get("primary_metric")

    # Bloque para correr análisis
    with st.expander("Correr / Re-correr análisis", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            wg_negative_is_error = st.checkbox("Bloquear WG negativo (WG < 0)", value=True)
        with c2:
            alpha = st.number_input("Alpha", min_value=0.001, max_value=0.2, value=0.05, step=0.005)
        with c3:
            enable_posthoc = st.checkbox("Posthoc (cuando aplique)", value=True)

        if st.button("Correr análisis ahora", type="primary", use_container_width=True):
            run_analysis(alpha=float(alpha), enable_posthoc=bool(enable_posthoc), wg_negative_is_error=bool(wg_negative_is_error))
            st.rerun()

    state = get_state()
    if state.unit_kpis_df is None:
        return

    # Garantizar variable principal
    kpi_cols = numeric_kpi_columns(state.unit_kpis_df)
    if not metric or metric not in kpi_cols:
        st.warning("Selecciona la variable principal en la pestaña 1.")
        return

    label = st.session_state.get("primary_metric_label") or metric

    # Cards rápidas
    df = state.unit_kpis_df
    n_units = len(df)
    treatments = sorted(df["treatment"].astype(str).unique().tolist())

    st.markdown(
        f"""
        <div class="pcta-kpi">
          <div><div class="label">Variable</div><div class="value">{label}</div></div>
          <div><div class="label">Unidades</div><div class="value">{n_units}</div></div>
          <div><div class="label">Tratamientos</div><div class="value">{len(treatments)}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("### Resumen descriptivo por tratamiento")
    decimals = st.slider("Decimales (descriptiva)", 0, 6, 2)
    desc = describe_by_treatment(df, metric)
    st.dataframe(styled_descriptive_table(desc, decimals=decimals), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Gráficos")

    try:
        import plotly.express as px
    except Exception:
        st.error("No está instalado Plotly. Revisa `pcta/requirements.txt`.")
        return

    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.box(df, x="treatment", y=metric, points="all", template="simple_white", title=f"{label} (distribución)")
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        means = df.groupby("treatment", as_index=False)[metric].mean(numeric_only=True)
        fig2 = px.bar(means, x="treatment", y=metric, template="simple_white", title=f"{label} (media)")
        st.plotly_chart(fig2, use_container_width=True)


# ------------------------ TAB 3: Inferencial (comparación tratamientos) ------------------------


def tab_inferential_compare() -> None:
    st.subheader("3) Comparación de tratamientos (test de medias + gráficos)")

    state = get_state()
    if state.units is None or state.unit_kpis_df is None:
        st.info("Corre el análisis primero.")
        return

    metric = st.session_state.get("primary_metric")
    if not metric or metric not in numeric_kpi_columns(state.unit_kpis_df):
        st.warning("Selecciona variable principal en la pestaña 1.")
        return

    label = st.session_state.get("primary_metric_label") or metric

    treatments_all = sorted(state.unit_kpis_df["treatment"].astype(str).unique().tolist())
    selected_trts = st.multiselect(
        "Selecciona tratamientos a comparar",
        options=treatments_all,
        default=treatments_all,
        help="Puedes comparar un subconjunto de tratamientos.",
    )
    if len(selected_trts) < 2:
        st.warning("Selecciona al menos 2 tratamientos.")
        return

    st.divider()
    st.markdown("### Elegir test de medias")
    # UI listo; por ahora el core decide automáticamente.
    test_mode = st.selectbox(
        "Método (por ahora AUTO)",
        options=[
            "AUTO (recomendado)",
            "ANOVA (pendiente implementación)",
            "Welch ANOVA (pendiente implementación)",
            "Kruskal-Wallis (pendiente implementación)",
        ],
        index=0,
        help="Actualmente el core corre selección automática. Si confirmas, habilito selección real modificando core/stats.py.",
    )
    enable_posthoc = st.checkbox("Hacer posthoc (si aplica)", value=True)
    alpha = st.number_input("Alpha (comparación)", min_value=0.001, max_value=0.2, value=0.05, step=0.005)

    if st.button("Calcular test de medias para la variable principal", type="primary"):
        # Re-calcular stats SOLO para esa métrica y SOLO para tratamientos seleccionados
        try:
            units_filtered = [u for u in state.units if str(u.treatment) in set(selected_trts)]
            computed, _ = compute_all_units(units_filtered)

            stats_df, rep_by_trt, min_n, enabled, warnings = run_inferential_statistics(
                computed,
                metrics=[metric],
                options=StatsOptions(alpha=float(alpha), enable_posthoc=bool(enable_posthoc)),
            )

            st.markdown("### Resultado inferencial")
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

            # Gráfico enfocado a tratamientos seleccionados
            try:
                import plotly.express as px
            except Exception:
                st.error("No está instalado Plotly.")
                return

            df = pd.DataFrame([m.model_dump() for m in computed])
            fig = px.box(
                df[df["treatment"].astype(str).isin(selected_trts)],
                x="treatment",
                y=metric,
                points="all",
                template="simple_white",
                title=f"{label} — tratamientos seleccionados",
            )
            st.plotly_chart(fig, use_container_width=True)

            if warnings:
                st.markdown("### Advertencias")
                st.dataframe(warnings_df(warnings), use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"No se pudo calcular inferencial: {e}")


# ------------------------ Export ------------------------


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
