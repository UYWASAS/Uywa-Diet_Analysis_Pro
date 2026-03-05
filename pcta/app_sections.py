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

    # Variable elegida por el usuario (puede ser raw o KPI)
    st.session_state.setdefault("analysis_variable", None)
    st.session_state.setdefault("analysis_variable_label", None)
    st.session_state.setdefault("analysis_source", None)  # "raw" | "kpi"


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


def numeric_cols(df: pd.DataFrame, *, exclude: Optional[set[str]] = None) -> List[str]:
    exclude = exclude or set()
    out: List[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


def fmt_num(x: object, decimals: int = 2) -> str:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return ""
        return f"{float(x):,.{decimals}f}"
    except Exception:
        return str(x)


def describe_by_group(df: pd.DataFrame, group_col: str, metric: str) -> pd.DataFrame:
    if metric not in df.columns or group_col not in df.columns:
        return pd.DataFrame()

    g = (
        df.groupby(group_col)[metric]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .rename(columns={"count": "n", "mean": "media", "std": "sd", "median": "mediana"})
        .reset_index()
    )
    pcts = (
        df.groupby(group_col)[metric]
        .quantile([0.25, 0.75])
        .unstack(level=1)
        .reset_index()
        .rename(columns={0.25: "p25", 0.75: "p75"})
    )
    out = g.merge(pcts, on=group_col, how="left").sort_values(group_col).reset_index(drop=True)
    return out


def styled_desc(desc: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    if desc.empty:
        return desc
    out = desc.copy()
    for col in ["media", "sd", "min", "p25", "mediana", "p75", "max"]:
        if col in out.columns:
            out[col] = out[col].apply(lambda v: fmt_num(v, decimals=decimals))
    if "n" in out.columns:
        out["n"] = out["n"].fillna(0).astype(int)
    return out


def numeric_kpi_columns(unit_kpis_df: pd.DataFrame) -> List[str]:
    id_cols = {"trial_id", "unit_type", "unit_id", "treatment"}
    return [
        c
        for c in unit_kpis_df.columns
        if c not in id_cols and pd.api.types.is_numeric_dtype(unit_kpis_df[c])
    ]


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


# ------------------------ Pipeline ------------------------


def run_analysis(*, alpha: float, enable_posthoc: bool, wg_negative_is_error: bool) -> None:
    """
    Corre el pipeline completo:
    - to_units
    - validate
    - compute_all_units
    - run_inferential_statistics (default metrics)
    - export_report_xlsx
    """
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

        metrics_default = default_metric_list(unit_kpis_df)
        stats_df, rep_by_trt, min_n, enabled, s_warnings = run_inferential_statistics(
            computed,
            metrics=metrics_default,
            options=StatsOptions(alpha=float(alpha), enable_posthoc=bool(enable_posthoc)),
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


# ------------------------ TAB 1: Elegir variable + correr ------------------------


def tab_1_select_variable_and_run() -> None:
    st.subheader("1) Selección de variable + correr análisis")

    state = get_state()
    if state.parsed is None:
        st.info("Carga un archivo en la barra lateral para comenzar.")
        return

    hs = state.parsed.to_dataframes()["house_summary"].copy()

    st.markdown("### Selecciona variable de trabajo (desde HOUSE_SUMMARY)")
    st.caption("Esta selección funciona ANTES de correr el análisis (input crudo).")

    raw_exclude = {"trial_id", "unit_id", "treatment", "unit_type"}
    raw_numeric = numeric_cols(hs, exclude=raw_exclude)

    if not raw_numeric:
        st.warning("No se detectaron columnas numéricas en HOUSE_SUMMARY para seleccionar.")
        return

    raw_labels = {
        "bw_final_mean_g": "Peso vivo final (g)",
        "bw_initial_mean_g": "Peso vivo inicial (g)",
        "mortality_total": "Mortalidad total (n)",
        "feed_delivered_kg": "Alimento entregado (kg)",
    }

    raw_options = [f"{raw_labels.get(c, c)}  —  ({c})" for c in raw_numeric]
    default_raw = st.session_state.get("analysis_variable") or (
        "bw_final_mean_g" if "bw_final_mean_g" in raw_numeric else raw_numeric[0]
    )
    idx = raw_numeric.index(default_raw) if default_raw in raw_numeric else 0

    sel = st.selectbox("Variable (raw)", options=raw_options, index=idx, key="sel_raw_var")
    chosen = raw_numeric[raw_options.index(sel)]

    st.session_state["analysis_variable"] = chosen
    st.session_state["analysis_variable_label"] = raw_labels.get(chosen, chosen)
    st.session_state["analysis_source"] = "raw"

    st.success(f"Variable seleccionada: {st.session_state['analysis_variable_label']}")

    with st.expander("Vista previa de datos (opcional)", expanded=False):
        show = st.toggle("Mostrar tabla HOUSE_SUMMARY", value=False, key="preview_show_hs")
        if show:
            st.dataframe(hs.head(300), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Correr análisis (para habilitar KPIs y test de medias)")
    c1, c2, c3 = st.columns(3)
    with c1:
        wg_negative_is_error = st.checkbox(
            "Bloquear WG negativo (WG < 0)",
            value=True,
            key="run_wg_negative_is_error",
        )
    with c2:
        alpha = st.number_input(
            "Alpha",
            min_value=0.001,
            max_value=0.2,
            value=0.05,
            step=0.005,
            key="run_alpha",
        )
    with c3:
        enable_posthoc = st.checkbox(
            "Posthoc (cuando aplique)",
            value=True,
            key="run_enable_posthoc",
        )

    if st.button("Correr análisis ahora", type="primary", use_container_width=True, key="run_analysis_btn"):
        run_analysis(
            alpha=float(alpha),
            enable_posthoc=bool(enable_posthoc),
            wg_negative_is_error=bool(wg_negative_is_error),
        )
        st.success("Listo. Ve a la pestaña 2 para ver resultados.")


# ------------------------ TAB 2: Resultados de variable (descriptiva + gráficos) ------------------------


def tab_2_results_for_selected_variable() -> None:
    st.subheader("2) Resultados (descriptiva + gráficos) — variable seleccionada")

    state = get_state()
    if state.parsed is None:
        st.info("Carga un archivo primero.")
        return

    var = st.session_state.get("analysis_variable")
    src = st.session_state.get("analysis_source")
    label = st.session_state.get("analysis_variable_label") or var

    if not var or src != "raw":
        st.warning("Selecciona la variable en la pestaña 1.")
        return

    hs = state.parsed.to_dataframes()["house_summary"].copy()
    if "treatment" not in hs.columns:
        st.error("No existe columna 'treatment' en HOUSE_SUMMARY.")
        return
    if var not in hs.columns:
        st.error(f"La variable seleccionada ({var}) no existe en HOUSE_SUMMARY.")
        return

    n_units = int(len(hs))
    n_trt = int(hs["treatment"].astype(str).nunique())

    st.markdown(
        f"""
        <div class="pcta-kpi">
          <div><div class="label">Variable</div><div class="value">{label}</div></div>
          <div><div class="label">Unidades</div><div class="value">{n_units}</div></div>
          <div><div class="label">Tratamientos</div><div class="value">{n_trt}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("### Resumen descriptivo por tratamiento")
    decimals = st.slider("Decimales", 0, 6, 2, key="desc_decimals")
    desc = describe_by_group(hs, "treatment", var)
    st.dataframe(styled_desc(desc, decimals=decimals), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Gráficos")

    try:
        import plotly.express as px
    except Exception:
        st.error("No está instalado Plotly.")
        return

    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.box(
            hs,
            x="treatment",
            y=var,
            points="all",
            template="simple_white",
            title=f"{label} (distribución)",
        )
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        means = hs.groupby("treatment", as_index=False)[var].mean(numeric_only=True)
        fig2 = px.bar(
            means,
            x="treatment",
            y=var,
            template="simple_white",
            title=f"{label} (media)",
        )
        st.plotly_chart(fig2, use_container_width=True)


# ------------------------ TAB 3: Test de medias sobre la variable seleccionada ------------------------


def tab_3_mean_tests() -> None:
    st.subheader("3) Test de medias (ANOVA / posthoc) + tratamientos a comparar")

    state = get_state()
    if state.units is None or state.unit_kpis_df is None:
        st.info("Primero corre el análisis en la pestaña 1.")
        return

    var = st.session_state.get("analysis_variable")
    label = st.session_state.get("analysis_variable_label") or var
    if not var:
        st.warning("Selecciona variable en la pestaña 1.")
        return

    kpi_cols = numeric_kpi_columns(state.unit_kpis_df)
    if var not in kpi_cols:
        st.warning(
            f"La variable seleccionada ({var}) es de input crudo y no está disponible como KPI calculado. "
            "Para ANOVA/posthoc, selecciona una variable KPI (ej: bw_final_mean_g, fcr, wg_g_per_bird) "
            "o ampliamos el core para inferencia sobre raw."
        )
        return

    treatments_all = sorted(state.unit_kpis_df["treatment"].astype(str).unique().tolist())
    selected_trts = st.multiselect(
        "Tratamientos a comparar",
        options=treatments_all,
        default=treatments_all,
        key="mean_tests_selected_trts",
    )

    if len(selected_trts) < 2:
        st.warning("Selecciona al menos 2 tratamientos.")
        return

    st.divider()
    st.markdown("### Método")
    test_mode = st.selectbox(
        "Test de medias",
        options=[
            "AUTO (según supuestos)",
            "ANOVA (pendiente)",
            "Welch ANOVA (pendiente)",
            "Kruskal-Wallis (pendiente)",
        ],
        index=0,
        key="mean_tests_mode",
    )
    enable_posthoc = st.checkbox("Posthoc (si aplica)", value=True, key="mean_tests_posthoc")
    alpha = st.number_input(
        "Alpha",
        min_value=0.001,
        max_value=0.2,
        value=0.05,
        step=0.005,
        key="mean_tests_alpha",
    )

    if st.button("Calcular test de medias", type="primary", key="mean_tests_run"):
        try:
            units_filtered = [u for u in state.units if str(u.treatment) in set(selected_trts)]
            computed, _ = compute_all_units(units_filtered)

            stats_df, rep_by_trt, min_n, enabled, warnings = run_inferential_statistics(
                computed,
                metrics=[var],
                options=StatsOptions(alpha=float(alpha), enable_posthoc=bool(enable_posthoc)),
            )

            st.markdown(f"### Resultado: {label}")
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

            try:
                import plotly.express as px
            except Exception:
                st.error("No está instalado Plotly.")
                return

            df = pd.DataFrame([m.model_dump() for m in computed])
            fig = px.box(
                df[df["treatment"].astype(str).isin(selected_trts)],
                x="treatment",
                y=var,
                points="all",
                template="simple_white",
                title=f"{label} — tratamientos seleccionados",
            )
            st.plotly_chart(fig, use_container_width=True)

            if warnings:
                st.markdown("### Advertencias")
                st.dataframe(warnings_df(warnings), use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"No se pudo calcular el test: {e}")


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
