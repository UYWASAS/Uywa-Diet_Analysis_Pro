from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats as sps

from pcta.auth import logout_button
from pcta.core.calculations import compute_all_units
from pcta.core.factorial_stats import FactorialOptions, run_factorial_anova_df
from pcta.core.io import export_report_xlsx, parse_uploaded_file
from pcta.core.reporting import build_treatment_summary, default_metric_list
from pcta.core.schemas import AnalysisWarning, ExportPayload, ParsedInput, TrialUnitInput
from pcta.core.stats import StatsOptions, run_inferential_statistics, run_inferential_statistics_df
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

    # NUEVO: modo libre (cualquier Excel/CSV)
    st.session_state.setdefault("raw_df", None)  # pd.DataFrame | None
    st.session_state.setdefault("raw_mode", False)  # bool
    st.session_state.setdefault("raw_sheet_name", None)  # str | None

    # Selección explícita (no asumir nombres)
    st.session_state.setdefault("dv_col", None)  # Y
    st.session_state.setdefault("factor_a", None)  # tratamiento-like
    st.session_state.setdefault("factor_b", None)  # segundo factor
    st.session_state.setdefault("block_col", None)  # bloque (opcional)
    st.session_state.setdefault("filters", {})  # dict[col] -> list[level strings]

    # Correlación
    st.session_state.setdefault("corr_x", None)
    st.session_state.setdefault("corr_y", None)

    # Compatibilidad con UI anterior si existe
    st.session_state.setdefault("analysis_variable", None)
    st.session_state.setdefault("analysis_variable_label", None)
    st.session_state.setdefault("analysis_source", "raw")


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
    return pd.DataFrame([{"codigo": w.code.value, "mensaje": w.message, "contexto": w.context} for w in warnings])


def numeric_cols(df: pd.DataFrame, *, exclude: Optional[set[str]] = None) -> List[str]:
    exclude = exclude or set()
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def categorical_cols(df: pd.DataFrame, *, exclude: Optional[set[str]] = None) -> List[str]:
    exclude = exclude or set()
    out: List[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        s = df[c]
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_bool_dtype(s):
            out.append(c)
        elif pd.api.types.is_categorical_dtype(s):
            out.append(c)
    return out


def apply_filters(df: pd.DataFrame, filters: Dict[str, List[str]]) -> pd.DataFrame:
    out = df
    for col, levels in (filters or {}).items():
        if col not in out.columns:
            continue
        if levels:
            out = out[out[col].astype(str).isin(set(map(str, levels)))]
    return out


def render_design_controls(
    df: pd.DataFrame, *, prefix_key: str
) -> Tuple[str, str, Optional[str], Optional[str], Dict[str, List[str]]]:
    num = numeric_cols(df)
    cat = categorical_cols(df)

    if not num:
        st.error("No hay columnas numéricas disponibles para Y.")
        return "", "", None, None, {}

    if not cat:
        st.error("No hay columnas categóricas disponibles para factores/filtros.")
        return "", "", None, None, {}

    dv_default = st.session_state.get("dv_col") or (st.session_state.get("analysis_variable") if st.session_state.get("analysis_variable") in num else num[0])
    if dv_default not in num:
        dv_default = num[0]

    fa_default = st.session_state.get("factor_a")
    if fa_default not in cat:
        # No asumimos treatment; si existe, lo preferimos, si no el primero
        fa_default = "treatment" if "treatment" in cat else cat[0]

    dv_col = st.selectbox("Variable dependiente (Y)", options=num, index=num.index(dv_default), key=f"{prefix_key}_dv")
    st.session_state["dv_col"] = dv_col
    st.session_state["analysis_variable"] = dv_col
    st.session_state["analysis_variable_label"] = dv_col
    st.session_state["analysis_source"] = "raw"

    factor_a = st.selectbox("Factor A (principal)", options=cat, index=cat.index(fa_default), key=f"{prefix_key}_fa")
    st.session_state["factor_a"] = factor_a

    b_opts = [None] + [c for c in cat if c != factor_a]
    fb_default = st.session_state.get("factor_b")
    if fb_default not in b_opts:
        fb_default = None

    factor_b = st.selectbox(
        "Factor B (opcional)",
        options=b_opts,
        format_func=lambda v: "— Ninguno —" if v is None else str(v),
        index=b_opts.index(fb_default),
        key=f"{prefix_key}_fb",
    )
    st.session_state["factor_b"] = factor_b

    block_opts = [None] + [c for c in cat if c not in {factor_a, factor_b}]
    block_default = st.session_state.get("block_col")
    if block_default not in block_opts:
        block_default = None

    block_col = st.selectbox(
        "Bloque (opcional)",
        options=block_opts,
        format_func=lambda v: "— Ninguno —" if v is None else str(v),
        index=block_opts.index(block_default),
        key=f"{prefix_key}_block",
    )
    st.session_state["block_col"] = block_col

    st.markdown("#### Bloqueos / filtros (pre-análisis)")
    prev_filter_cols = list(st.session_state.get("filters", {}).keys())
    filter_cols = st.multiselect(
        "Columnas a filtrar",
        options=cat,
        default=[c for c in prev_filter_cols if c in cat],
        key=f"{prefix_key}_filter_cols",
    )

    filters: Dict[str, List[str]] = {}
    for col in filter_cols:
        levels = sorted(df[col].dropna().astype(str).unique().tolist())
        default_levels = st.session_state.get("filters", {}).get(col, levels)
        default_levels = [x for x in default_levels if x in levels] or levels
        chosen = st.multiselect(
            f"Niveles permitidos: {col}",
            options=levels,
            default=default_levels,
            key=f"{prefix_key}_filter_levels_{col}",
        )
        filters[col] = chosen

    st.session_state["filters"] = filters
    return dv_col, factor_a, factor_b, block_col, filters


def describe_by_group(df: pd.DataFrame, group_cols: List[str], metric: str) -> pd.DataFrame:
    for gc in group_cols:
        if gc not in df.columns:
            return pd.DataFrame()
    if metric not in df.columns:
        return pd.DataFrame()

    sub = df[group_cols + [metric]].dropna()
    if sub.empty:
        return pd.DataFrame()

    g = (
        sub.groupby(group_cols)[metric]
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
    return g


def format_desc_table(desc: pd.DataFrame, *, group_cols: List[str], decimals: int) -> pd.DataFrame:
    if desc.empty:
        return desc

    out = desc.copy()
    if "n" in out.columns:
        out["n"] = out["n"].fillna(0).astype(int)

    num_cols = [c for c in out.columns if c not in set(group_cols + ["n"])]
    for c in num_cols:
        if c == "cv_pct":
            out[c] = out[c].apply(lambda v: "" if pd.isna(v) else f"{float(v):.{min(2,decimals)}f}%")
        else:
            out[c] = out[c].apply(lambda v: "" if pd.isna(v) else f"{float(v):,.{decimals}f}")

    rename = {gc: f"Factor_{i+1}" for i, gc in enumerate(group_cols)}
    rename.update(
        {
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
    )
    return out.rename(columns=rename)


def homogeneity_tests(df: pd.DataFrame, group_col: str, metric: str) -> Dict[str, object]:
    sub = df[[group_col, metric]].dropna()
    if sub.empty:
        return {"levene_p": None, "shapiro_min_p": None}

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
    return {"levene_p": levene_p, "shapiro_min_p": shapiro_min_p}


def correlation_stats(x: pd.Series, y: pd.Series, method: str) -> Dict[str, object]:
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    n = int(len(df))
    if n < 3:
        return {"n": n, "r": None, "r2": None, "p_value": None}

    if method == "pearson":
        r, p = sps.pearsonr(df["x"].to_numpy(float), df["y"].to_numpy(float))
    else:
        r, p = sps.spearmanr(df["x"].to_numpy(float), df["y"].to_numpy(float))

    r = float(r)
    p = float(p)
    return {"n": n, "r": r, "r2": float(r * r), "p_value": p}


def _read_any_file_to_df(file_name: str, file_bytes: bytes) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Best-effort reader for "free mode".
    - CSV -> a single DF
    - XLSX -> picks first sheet by default (sheet selection handled outside)
    Returns (df, error_message)
    """
    lower = file_name.lower()
    try:
        if lower.endswith(".csv"):
            return pd.read_csv(BytesIO(file_bytes)), None
        if lower.endswith(".xlsx"):
            # caller will typically read specific sheet; default to first sheet
            xls = pd.ExcelFile(BytesIO(file_bytes))
            first = xls.sheet_names[0] if xls.sheet_names else None
            if first is None:
                return pd.DataFrame(), "El archivo Excel no tiene hojas."
            return pd.read_excel(BytesIO(file_bytes), sheet_name=first), None
        return pd.DataFrame(), "Formato no soportado (solo .csv o .xlsx)."
    except Exception as e:
        return pd.DataFrame(), str(e)


def _excel_sheets(file_bytes: bytes) -> List[str]:
    try:
        xls = pd.ExcelFile(BytesIO(file_bytes))
        return list(xls.sheet_names)
    except Exception:
        return []


def _read_excel_sheet(file_bytes: bytes, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name)


def get_active_df() -> Optional[pd.DataFrame]:
    """
    Returns the DataFrame to use for flexible analysis:
    - if raw_mode: st.session_state["raw_df"]
    - else: parsed house_summary
    """
    if st.session_state.get("raw_mode") and isinstance(st.session_state.get("raw_df"), pd.DataFrame):
        return st.session_state["raw_df"]

    state = get_state()
    if state.parsed is None:
        return None

    dfs = state.parsed.to_dataframes()
    if "house_summary" in dfs:
        return dfs["house_summary"].copy()
    return None


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


# ------------------------ Parse / Load ------------------------


def maybe_parse_main_upload(uploaded_main: Optional[object]) -> None:
    """
    Tries to parse using the existing strict PCTA parser.
    If it fails, falls back to "free mode" loading into raw_df.
    """
    if uploaded_main is None:
        return

    file_name = uploaded_main.name
    file_bytes = uploaded_main.getvalue()

    # Try strict parse first (legacy)
    try:
        parsed = parse_uploaded_file(file_name, file_bytes)
        set_state(
            parsed=parsed,
            units=None,
            unit_kpis_df=None,
            treatment_summary_df=None,
            stats_df=None,
            warnings=[],
            report_bytes=None,
        )
        st.session_state["raw_mode"] = False
        st.session_state["raw_df"] = None
        st.session_state["raw_sheet_name"] = None
        return
    except Exception as e:
        # fallback to free mode
        st.warning(f"Modo PCTA estándar no aplica. Activando Modo Libre. Detalle: {e}")

    # Free mode: load
    lower = file_name.lower()
    if lower.endswith(".xlsx"):
        sheets = _excel_sheets(file_bytes)
        if not sheets:
            st.error("No pude leer hojas del Excel.")
            return

        # Keep sheet selection in session
        default_sheet = st.session_state.get("raw_sheet_name") or sheets[0]
        if default_sheet not in sheets:
            default_sheet = sheets[0]

        # Render selector immediately in sidebar area
        with st.sidebar:
            st.subheader("Modo Libre — hoja a usar")
            sheet = st.selectbox("Hoja", options=sheets, index=sheets.index(default_sheet), key="raw_sheet_select")
        st.session_state["raw_sheet_name"] = sheet

        try:
            df = _read_excel_sheet(file_bytes, sheet_name=sheet)
        except Exception as e:
            st.error(f"No pude leer la hoja '{sheet}': {e}")
            return

        st.session_state["raw_mode"] = True
        st.session_state["raw_df"] = df
        set_state(parsed=None, units=None, unit_kpis_df=None, treatment_summary_df=None, stats_df=None, warnings=[], report_bytes=None)
        return

    # CSV
    df, err = _read_any_file_to_df(file_name, file_bytes)
    if err:
        st.error(f"No se pudo cargar el archivo en modo libre: {err}")
        return

    st.session_state["raw_mode"] = True
    st.session_state["raw_df"] = df
    st.session_state["raw_sheet_name"] = None
    set_state(parsed=None, units=None, unit_kpis_df=None, treatment_summary_df=None, stats_df=None, warnings=[], report_bytes=None)


# ------------------------ KPI pipeline (solo modo PCTA estándar) ------------------------


def run_analysis(*, alpha: float, enable_posthoc: bool, wg_negative_is_error: bool) -> None:
    """
    Corre el pipeline KPI legado SOLO si estamos en modo PCTA estándar (parsed != None).
    """
    state = get_state()
    if state.parsed is None:
        st.error("El análisis KPI requiere modo PCTA estándar (archivo con estructura esperada).")
        return

    try:
        units = state.parsed.to_units()
        _, v_warnings = validate_units(units, options=ValidationOptions(wg_negative_is_error=wg_negative_is_error))

        computed, c_warnings = compute_all_units(units)
        unit_kpis_df = pd.DataFrame([m.model_dump() for m in computed])
        treatment_summary_df = build_treatment_summary(unit_kpis_df)

        metrics_default = default_metric_list(unit_kpis_df)
        stats_df, rep_by_trt, min_n, enabled, s_warnings = run_inferential_statistics(
            computed,
            metrics=metrics_default,
            options=StatsOptions(alpha=float(alpha), enable_posthoc=bool(enable_posthoc)),
            group_col="treatment",
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
        set_state(units=None, unit_kpis_df=None, treatment_summary_df=None, stats_df=None, warnings=[], report_bytes=None)


# ------------------------ Tabs ------------------------


def tab_1_select_variable_and_run() -> None:
    st.subheader("1) Definir diseño (Y/A/B/bloque/filtros)")

    df = get_active_df()
    if df is None:
        st.info("Carga un archivo en la barra lateral para comenzar.")
        return

    if st.session_state.get("raw_mode"):
        st.caption("Modo Libre activado: no se requieren nombres estándar de columnas/hojas.")

    st.markdown("### Diseño del análisis")
    render_design_controls(df, prefix_key="tab1_design")

    # Solo si estamos en modo estándar (parsed) mostramos botón KPI/export
    state = get_state()
    if state.parsed is not None:
        st.divider()
        st.markdown("### (Opcional) Correr análisis KPI/Export (modo estándar)")
        c1, c2, c3 = st.columns(3)
        with c1:
            wg_negative_is_error = st.checkbox("Bloquear WG negativo (WG < 0)", value=True, key="run_wg_negative_is_error")
        with c2:
            alpha = st.number_input("Alpha", min_value=0.001, max_value=0.2, value=0.05, step=0.005, key="run_alpha")
        with c3:
            enable_posthoc = st.checkbox("Posthoc (cuando aplique)", value=True, key="run_enable_posthoc")

        if st.button("Correr análisis KPI ahora", type="primary", use_container_width=True, key="run_analysis_btn"):
            run_analysis(alpha=float(alpha), enable_posthoc=bool(enable_posthoc), wg_negative_is_error=bool(wg_negative_is_error))
            st.success("Listo. Ve a Exportar para descargar el reporte KPI.")


def tab_2_results_for_selected_variable() -> None:
    st.subheader("2) Resultados (descriptivo + correlación)")
    _inject_table_css()

    df = get_active_df()
    if df is None:
        st.info("Carga un archivo primero.")
        return

    st.markdown("### Diseño (puedes ajustar aquí)")
    dv_col, factor_a, factor_b, block_col, filters = render_design_controls(df, prefix_key="tab2_design")

    df_f = apply_filters(df, filters)
    if df_f.empty:
        st.error("Con los filtros actuales no quedan filas para analizar.")
        return

    group_cols = [factor_a] + ([factor_b] if factor_b else [])

    st.divider()
    st.markdown("### Datos (post-filtro)")
    cols_show = group_cols + [dv_col]
    with st.expander("Ver tabla", expanded=True):
        st.dataframe(df_f[cols_show].dropna(), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Resumen descriptivo")
    decimals = st.slider("Decimales", 0, 6, 2, key="tab2_decimals")
    desc = describe_by_group(df_f, group_cols, dv_col)
    st.dataframe(format_desc_table(desc, group_cols=group_cols, decimals=decimals), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Supuestos (informativo; sobre Factor A)")
    tests = homogeneity_tests(df_f, factor_a, dv_col)
    c1, c2 = st.columns(2)
    c1.metric("Levene p", "—" if tests["levene_p"] is None else f"{tests['levene_p']:.4f}")
    c2.metric("Shapiro min p", "—" if tests["shapiro_min_p"] is None else f"{tests['shapiro_min_p']:.4f}")

    st.divider()
    st.markdown("### Correlación 2×2 (post-filtro)")
    num = numeric_cols(df_f)
    if len(num) < 2:
        st.info("No hay suficientes variables numéricas para correlación.")
        return

    x_default = st.session_state.get("corr_x") or num[0]
    y_default = st.session_state.get("corr_y") or (num[1] if len(num) > 1 else num[0])
    if x_default not in num:
        x_default = num[0]
    if y_default not in num:
        y_default = num[1] if len(num) > 1 else num[0]

    x_var = st.selectbox("Variable X", options=num, index=num.index(x_default), key="corr_x")
    y_var = st.selectbox("Variable Y", options=num, index=num.index(y_default), key="corr_y")
    method = st.selectbox("Método", options=["pearson", "spearman"], index=0, key="corr_method")

    if x_var == y_var:
        st.warning("Selecciona X ≠ Y.")
        return

    corr = correlation_stats(df_f[x_var], df_f[y_var], method=method)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("n", str(corr["n"]))
    k2.metric("r", "—" if corr["r"] is None else f"{corr['r']:.4f}")
    k3.metric("r²", "—" if corr["r2"] is None else f"{corr['r2']:.4f}")
    k4.metric("p-value", "—" if corr["p_value"] is None else f"{corr['p_value']:.4g}")

    try:
        import plotly.express as px
    except Exception:
        st.error("No está instalado Plotly.")
        return

    df_plot = df_f[[x_var, y_var]].dropna()
    figc = px.scatter(
        df_plot,
        x=x_var,
        y=y_var,
        trendline="ols" if method == "pearson" else None,
        template="simple_white",
        title=f"Correlación ({method}) — {x_var} vs {y_var}",
    )
    st.plotly_chart(figc, use_container_width=True)


def tab_3_mean_tests() -> None:
    st.subheader("3) Inferencial (RAW flexible)")

    df = get_active_df()
    if df is None:
        st.info("Carga un archivo primero.")
        return

    st.markdown("### Diseño (puedes ajustar aquí)")
    dv_col, factor_a, factor_b, block_col, filters = render_design_controls(df, prefix_key="tab3_design")

    df_f = apply_filters(df, filters)
    if df_f.empty:
        st.error("Con los filtros actuales no quedan filas para analizar.")
        return

    st.divider()
    st.markdown("### Opciones inferenciales")
    alpha = st.number_input("Alpha", min_value=0.001, max_value=0.2, value=0.05, step=0.005, key="tab3_alpha")

    # Factorial A×B
    if factor_b:
        include_interaction = st.checkbox("Incluir interacción A:B", value=True, key="tab3_interaction")
        anova_type = st.selectbox("Tipo de ANOVA", options=[2, 3], index=0, key="tab3_anova_type")
        use_block = st.checkbox("Incluir bloque como efecto fijo", value=bool(block_col), key="tab3_use_block")

        aov_df, meta, warnings = run_factorial_anova_df(
            df_f,
            y_col=dv_col,
            factor_a=factor_a,
            factor_b=factor_b,
            block_col=block_col if (use_block and block_col) else None,
            options=FactorialOptions(alpha=float(alpha), include_interaction=bool(include_interaction), anova_type=int(anova_type)),
        )

        st.caption(f"Fórmula: `{meta.get('formula','')}`")
        st.caption(f"min n por celda (A×B): {meta.get('min_n_per_cell')}")
        st.dataframe(aov_df, use_container_width=True, hide_index=True)

        if warnings:
            st.markdown("### Advertencias")
            st.dataframe(warnings_df(warnings), use_container_width=True, hide_index=True)
        return

    # 1-factor
    enable_posthoc = st.checkbox("Posthoc (si aplica)", value=True, key="tab3_enable_posthoc")
    stats_df, rep, min_n, enabled, warnings = run_inferential_statistics_df(
        df_f,
        metric=dv_col,
        group_col=factor_a,
        options=StatsOptions(alpha=float(alpha), enable_posthoc=bool(enable_posthoc)),
    )

    st.markdown("### Replicación por grupo")
    rep_df = pd.DataFrame([{"grupo": k, "n": int(v)} for k, v in sorted(rep.items(), key=lambda kv: str(kv[0]))])
    st.dataframe(rep_df, use_container_width=True, hide_index=True)

    st.markdown("### Resultado inferencial")
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

    if warnings:
        st.markdown("### Advertencias")
        st.dataframe(warnings_df(warnings), use_container_width=True, hide_index=True)

    if not enabled:
        st.warning("Inferencia deshabilitada: se requiere al menos n>=2 por grupo (Factor A) para p-values.")


def tab_export() -> None:
    st.subheader("Exportar (solo modo estándar KPI)")
    state = get_state()
    if state.report_bytes is None:
        st.info("En Modo Libre no se genera el reporte KPI. (Si lo necesitas, lo habilitamos luego).")
        return

    st.download_button(
        "Descargar reporte Excel",
        data=state.report_bytes,
        file_name="pcta_reporte.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
