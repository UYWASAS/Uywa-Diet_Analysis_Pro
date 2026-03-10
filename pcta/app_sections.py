from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats as sps

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

    # Variable elegida por el usuario (raw)
    st.session_state.setdefault("analysis_variable", None)
    st.session_state.setdefault("analysis_variable_label", None)
    st.session_state.setdefault("analysis_source", None)  # "raw" | "kpi"

    # NUEVO: factor de agrupación (columna categórica)
    st.session_state.setdefault("group_factor", "treatment")


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


def _inject_table_css() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stDataFrame"] {
          border: 1px solid #e6eefb;
          border-radius: 12px;
          overflow: hidden;
        }
        div[data-testid="stDataFrame"] thead tr th {
          background: #f3f7ff !important;
          color: #0f172a !important;
          font-weight: 700 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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


def _categorical_candidates(df: pd.DataFrame) -> List[str]:
    """
    Heurística para columnas que sirven como factor:
    - object/string/bool/category
    (excluye columnas típicamente numéricas aunque tengan pocos niveles)
    """
    out: List[str] = []
    for c in df.columns:
        if c in {"trial_id"}:
            continue
        s = df[c]
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            out.append(c)
        elif pd.api.types.is_bool_dtype(s):
            out.append(c)
        elif pd.api.types.is_categorical_dtype(s):
            out.append(c)
    # prefer treatment si existe
    if "treatment" in df.columns and "treatment" not in out:
        out.insert(0, "treatment")
    return out


def describe_by_group(df: pd.DataFrame, group_col: str, metric: str) -> pd.DataFrame:
    """
    Descriptiva por grupo mejorada:
    - n, media, sd, cv_pct, min, p10, p25, mediana, p75, p90, max, rango
    """
    if metric not in df.columns or group_col not in df.columns:
        return pd.DataFrame()

    sub = df[[group_col, metric]].dropna()
    if sub.empty:
        return pd.DataFrame()

    g = (
        sub.groupby(group_col)[metric]
        .agg(
            n="count",
            media="mean",
            sd=lambda s: float(s.std(ddof=1)),
            min="min",
            p10=lambda s: float(s.quantile(0.10)),
            p25=lambda s: float(s.quantile(0.25)),
            mediana="median",
            p75=lambda s: float(s.quantile(0.75)),
            p90=lambda s: float(s.quantile(0.90)),
            max="max",
        )
        .reset_index()
    )

    g["rango"] = g["max"] - g["min"]
    g["cv_pct"] = g.apply(lambda r: (np.nan if r["media"] == 0 else 100.0 * r["sd"] / abs(r["media"])), axis=1)
    g = g.sort_values(group_col).reset_index(drop=True)
    return g


def _format_desc_table(desc: pd.DataFrame, *, group_col: str, decimals: int) -> pd.DataFrame:
    if desc.empty:
        return desc

    out = desc.copy()
    if "n" in out.columns:
        out["n"] = out["n"].fillna(0).astype(int)

    num_cols = [c for c in out.columns if c not in {group_col, "n"}]
    for c in num_cols:
        if c == "cv_pct":
            out[c] = out[c].apply(lambda v: "" if pd.isna(v) else f"{float(v):.{min(2,decimals)}f}%")
        else:
            out[c] = out[c].apply(lambda v: "" if pd.isna(v) else f"{float(v):,.{decimals}f}")

    rename = {
        group_col: "Grupo",
        "n": "n",
        "media": "Media",
        "sd": "SD",
        "cv_pct": "CV%",
        "min": "Mín",
        "p10": "P10",
        "p25": "P25",
        "mediana": "Mediana",
        "p75": "P75",
        "p90": "P90",
        "max": "Máx",
        "rango": "Rango",
    }
    return out.rename(columns=rename)


def _homogeneity_tests(df: pd.DataFrame, group_col: str, metric: str) -> Dict[str, object]:
    sub = df[[group_col, metric]].dropna()
    if sub.empty:
        return {"levene_p": None, "shapiro_min_p": None, "notes": "Sin datos."}

    groups: List[np.ndarray] = []
    shapiro_pvals: List[float] = []

    for _, g in sub.groupby(group_col):
        arr = g[metric].to_numpy(dtype=float)
        groups.append(arr)
        if arr.size >= 3:
            try:
                shapiro_pvals.append(float(sps.shapiro(arr).pvalue))
            except Exception:
                pass

    levene_p = None
    try:
        if len(groups) >= 2 and all(len(x) >= 2 for x in groups):
            levene_p = float(sps.levene(*groups, center="median").pvalue)
    except Exception:
        levene_p = None

    shapiro_min_p = float(min(shapiro_pvals)) if shapiro_pvals else None

    notes = []
    notes.append("Levene (mediana) evalúa homogeneidad de varianzas (p>0.05 sugiere varianzas similares).")
    notes.append("Shapiro (p>0.05) sugiere normalidad; se reporta el p mínimo entre grupos (si n>=3).")
    return {"levene_p": levene_p, "shapiro_min_p": shapiro_min_p, "notes": " ".join(notes)}


def _correlation_stats(x: pd.Series, y: pd.Series, method: str) -> Dict[str, object]:
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    n = int(len(df))
    if n < 3:
        return {"n": n, "r": None, "r2": None, "p_value": None, "note": "Se requieren al menos 3 pares válidos."}

    if method == "pearson":
        r, p = sps.pearsonr(df["x"].to_numpy(float), df["y"].to_numpy(float))
    else:
        r, p = sps.spearmanr(df["x"].to_numpy(float), df["y"].to_numpy(float))

    r = float(r)
    p = float(p)
    return {"n": n, "r": r, "r2": float(r * r), "p_value": p, "note": ""}


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


# ------------------------ TAB 1: Elegir variable + correr + factor ------------------------


def tab_1_select_variable_and_run() -> None:
    st.subheader("1) Selección de variable + factor + correr análisis")

    state = get_state()
    if state.parsed is None:
        st.info("Carga un archivo en la barra lateral para comenzar.")
        return

    hs = state.parsed.to_dataframes()["house_summary"].copy()

    st.markdown("### Selecciona variable de trabajo (desde HOUSE_SUMMARY)")
    st.caption("Esta selección funciona ANTES de correr el análisis (input crudo).")

    raw_exclude = {"trial_id", "unit_id", "unit_type"}
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

    st.divider()
    st.markdown("### Factor de comparación (agrupación)")

    candidates = _categorical_candidates(hs)
    if not candidates:
        st.warning("No encontré columnas categóricas. Se intentará usar 'treatment' si existe.")
        st.session_state["group_factor"] = "treatment"
    else:
        default_factor = st.session_state.get("group_factor", "treatment")
        if default_factor not in candidates:
            default_factor = candidates[0]

        factor = st.selectbox(
            "Selecciona la columna de agrupación (factor)",
            options=candidates,
            index=candidates.index(default_factor),
            key="group_factor_selectbox",
        )
        st.session_state["group_factor"] = factor
        levels = int(hs[factor].astype(str).nunique(dropna=True))
        st.caption(f"Niveles detectados en {factor}: {levels}")

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


# ------------------------ TAB 2: Resultados (factor flexible + correlación) ------------------------


def tab_2_results_for_selected_variable() -> None:
    st.subheader("2) Resultados (descriptiva + gráficos) — variable seleccionada")
    _inject_table_css()

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

    group_col = st.session_state.get("group_factor", "treatment")
    if group_col not in hs.columns:
        st.error(f"No existe la columna de agrupación '{group_col}' en HOUSE_SUMMARY.")
        return
    if var not in hs.columns:
        st.error(f"La variable seleccionada ({var}) no existe en HOUSE_SUMMARY.")
        return

    n_units = int(len(hs))
    groups = sorted(hs[group_col].astype(str).dropna().unique().tolist())
    n_groups = int(len(groups))
    missing = int(hs[var].isna().sum())
    valid = int(hs[var].notna().sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Variable", str(label))
    c2.metric("Unidades", str(n_units))
    c3.metric("Grupos", str(n_groups))
    c4.metric("Válidos", str(valid))
    st.caption(f"Agrupando por: **{group_col}**")

    st.divider()
    st.markdown("### Datos de la variable seleccionada")

    cols_show = [c for c in ["trial_id", "unit_id", "unit_type", group_col] if c in hs.columns] + [var]
    data_view = hs[cols_show].copy()
    sort_cols = [c for c in [group_col, "unit_id"] if c in data_view.columns]
    if sort_cols:
        data_view = data_view.sort_values(sort_cols, kind="stable")

    with st.expander("Ver tabla de datos", expanded=True):
        st.dataframe(data_view, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Resumen descriptivo por grupo (mejorado)")

    decimals = st.slider("Decimales", 0, 6, 2, key="tab2_desc_decimals")
    desc = describe_by_group(hs, group_col, var)
    desc_fmt = _format_desc_table(desc, group_col=group_col, decimals=decimals)
    st.dataframe(desc_fmt, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Supuestos (informativo)")
    tests = _homogeneity_tests(hs, group_col, var)
    a, b, c = st.columns(3)
    a.metric("Levene p (varianzas)", "—" if tests["levene_p"] is None else f"{tests['levene_p']:.4f}")
    b.metric("Shapiro min p", "—" if tests["shapiro_min_p"] is None else f"{tests['shapiro_min_p']:.4f}")
    c.metric("NA (faltantes)", str(missing))
    st.caption(str(tests["notes"]))

    st.divider()
    st.markdown("### Gráficos")

    try:
        import plotly.express as px
    except Exception:
        st.error("No está instalado Plotly.")
        return

    chart_type = st.radio(
        "Tipo de gráfico",
        options=["Box (con puntos)", "Violín", "Barras (media)", "Dispersión"],
        horizontal=True,
        key="tab2_chart_type",
    )

    col1, col2 = st.columns(2)
    with col1:
        if chart_type == "Box (con puntos)":
            fig1 = px.box(hs, x=group_col, y=var, points="all", template="simple_white", title=f"{label} — distribución")
        elif chart_type == "Violín":
            fig1 = px.violin(hs, x=group_col, y=var, box=True, points="all", template="simple_white", title=f"{label} — distribución")
        elif chart_type == "Barras (media)":
            means = hs.groupby(group_col, as_index=False)[var].mean(numeric_only=True)
            fig1 = px.bar(means, x=group_col, y=var, template="simple_white", title=f"{label} — media")
        else:
            fig1 = px.scatter(hs, x=group_col, y=var, template="simple_white", title=f"{label} — dispersión por grupo")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        agg = hs.groupby(group_col, as_index=False)[var].agg(["mean", "std", "count"]).reset_index()
        agg = agg.rename(columns={"mean": "media", "std": "sd", "count": "n"})
        fig2 = px.bar(agg, x=group_col, y="media", error_y="sd", template="simple_white", title=f"{label} — media ± SD")
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.markdown("### Correlación entre variables (2×2)")

    numeric_candidates = numeric_cols(hs, exclude={"trial_id", "unit_id", "unit_type"})
    if len(numeric_candidates) < 2:
        st.info("No hay suficientes variables numéricas para correlación.")
        return

    scope = st.radio(
        "Alcance de la correlación",
        options=["Toda la población", f"Filtrar por {group_col}"],
        horizontal=True,
        key="corr_scope",
    )

    if scope != "Toda la población":
        selected_levels = st.multiselect(
            f"Niveles incluidos de {group_col}",
            options=groups,
            default=groups,
            key="corr_levels",
        )
        if len(selected_levels) == 0:
            st.warning("Selecciona al menos un nivel.")
            return
        df_corr_base = hs[hs[group_col].astype(str).isin(set(selected_levels))].copy()
    else:
        df_corr_base = hs

    cL, cM, cR = st.columns([1, 1, 1])
    with cL:
        x_var = st.selectbox("Variable X", options=numeric_candidates, index=0, key="corr_x")
    with cM:
        default_y = 1 if len(numeric_candidates) > 1 else 0
        y_var = st.selectbox("Variable Y", options=numeric_candidates, index=default_y, key="corr_y")
    with cR:
        method = st.selectbox("Método", options=["pearson", "spearman"], index=0, key="corr_method")

    if x_var == y_var:
        st.warning("Selecciona dos variables diferentes (X ≠ Y).")
        return

    corr = _correlation_stats(df_corr_base[x_var], df_corr_base[y_var], method=method)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("n (pares)", str(corr["n"]))
    k2.metric("r", "—" if corr["r"] is None else f"{corr['r']:.4f}")
    k3.metric("r²", "—" if corr["r2"] is None else f"{corr['r2']:.4f}")
    k4.metric("p-value", "—" if corr["p_value"] is None else f"{corr['p_value']:.4g}")

    if corr.get("note"):
        st.caption(str(corr["note"]))

    df_plot = df_corr_base[[x_var, y_var, group_col]].dropna()

    if scope == "Toda la población":
        figc = px.scatter(
            df_plot,
            x=x_var,
            y=y_var,
            trendline="ols" if method == "pearson" else None,
            template="simple_white",
            title=f"Correlación ({method}) — población total: {x_var} vs {y_var}",
        )
    else:
        figc = px.scatter(
            df_plot,
            x=x_var,
            y=y_var,
            color=group_col,
            trendline="ols" if method == "pearson" else None,
            template="simple_white",
            title=f"Correlación ({method}) — por {group_col}: {x_var} vs {y_var}",
        )

    st.plotly_chart(figc, use_container_width=True)


# ------------------------ TAB 3: Test de medias (pendiente factor flexible en core) ------------------------


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
            "Para ANOVA/posthoc, selecciona una variable KPI (ej: bw_final_mean_g, fcr, wg_g_per_bird)."
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
    _ = st.selectbox(
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
