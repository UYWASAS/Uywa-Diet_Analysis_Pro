"""
PCTA Streamlit sections (modo estándar + modo libre)

Este archivo está organizado con "BLOQUES DE EDICIÓN" para que en iteraciones futuras
podamos modificar solo secciones específicas sin re-generar todo el archivo.

BUSCAR RÁPIDO (Ctrl/Cmd+F):
- BLOQUE DE EDICIÓN: LOAD/INPUT (STRICT -> FREE MODE)
- BLOQUE DE EDICIÓN: UI HELPERS (DESIGN + FILTERS)
- BLOQUE DE EDICIÓN: TAB 1 (SELECCIÓN)
- BLOQUE DE EDICIÓN: TAB 2 (RESULTADOS) — descriptivo + distribuciones + correlación COMPLETA
- BLOQUE DE EDICIÓN: TAB 3 (TEST DE MEDIAS) — omnibus + posthoc bajo demanda
- BLOQUE DE EDICIÓN: TAB 4 (KPIS PRODUCTIVOS) — análisis económico integral
- BLOQUE DE EDICIÓN: EXPORT
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats as sps

from pcta.auth import logout_button
from pcta.core.factorial_stats import FactorialOptions, run_factorial_anova_df
from pcta.core.io import parse_uploaded_file
from pcta.core.schemas import AnalysisWarning, ParsedInput
from pcta.core.stats import StatsOptions, run_inferential_statistics_df

# ===== IMPORTACIONES PARA TAB 4 (necesarias globalmente) =====
try:
    from pcta.core.productive_kpis import (
        ProductiveKPIInputs,
        compute_productive_kpis_batch,
        kpis_to_dataframe,
        compute_summary_by_treatment,
        compute_total_summary,
    )
    from pcta.core.schemas import TrialUnitInput, UnitType
    PRODUCTIVE_KPI_AVAILABLE = True
except ImportError:
    PRODUCTIVE_KPI_AVAILABLE = False
# ===== FIN IMPORTACIONES PARA TAB 4 =====

# ======================================================================================
# State
# ======================================================================================


@dataclass(frozen=True)
class AppState:
    parsed: Optional[ParsedInput]
    warnings: List[AnalysisWarning]


def init_state() -> None:
    if "pcta_state" not in st.session_state:
        st.session_state["pcta_state"] = AppState(parsed=None, warnings=[])

    # Free mode (any file)
    st.session_state.setdefault("raw_mode", False)
    st.session_state.setdefault("raw_df", None)  # pd.DataFrame | None
    st.session_state.setdefault("raw_sheet_name", None)

    # Design selections
    st.session_state.setdefault("dv_col", None)
    st.session_state.setdefault("factor_a", None)
    st.session_state.setdefault("factor_b", None)
    st.session_state.setdefault("block_col", None)
    st.session_state.setdefault("filters", {})

    # Correlation module (independent)
    st.session_state.setdefault("corr_x_var", None)
    st.session_state.setdefault("corr_y_var", None)
    st.session_state.setdefault("corr_method", "pearson")
    st.session_state.setdefault("corr_mode", "global")  # global | by_group | compare_full_post
    st.session_state.setdefault("corr_group_col", None)
    st.session_state.setdefault("corr_scope", "post_filter")  # post_filter | full
    st.session_state.setdefault("corr_global_color_by", None)
    st.session_state.setdefault("corr_show_group_trendlines", True)

    # Distribution plots
    st.session_state.setdefault("show_violin", True)
    st.session_state.setdefault("show_points", True)

    # Tab 3 (mean tests)
    st.session_state.setdefault("tab3_posthoc_policy", "auto_if_significant")  # auto_if_significant | always | never

    # Tab 4 (productive KPIs) - price/cost inputs
    st.session_state.setdefault("tab4_price_kg", 2.50)
    st.session_state.setdefault("tab4_cost_feed_kg", 0.45)
    st.session_state.setdefault("tab4_cost_chick", 0.80)
    st.session_state.setdefault("tab4_other_cost_bird", 0.15)
    st.session_state.setdefault("tab4_mortality_recovery", 20)


def get_state() -> AppState:
    return st.session_state["pcta_state"]


def set_state(*, parsed: Optional[ParsedInput], warnings: List[AnalysisWarning]) -> None:
    st.session_state["pcta_state"] = AppState(parsed=parsed, warnings=warnings)


# ======================================================================================
# Core helpers (data types, formatting, correlation)
# ======================================================================================


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


def numeric_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def categorical_cols(df: pd.DataFrame) -> List[str]:
    out: List[str] = []
    for c in df.columns:
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


def correlation_stats(x: pd.Series, y: pd.Series, method: str) -> Dict[str, object]:
    """
    Compute correlation stats for two numeric series.
    Returns: n, r, r2, p_value
    """
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    n = int(len(d))
    if n < 3:
        return {"n": n, "r": None, "r2": None, "p_value": None}

    if method == "pearson":
        r, p = sps.pearsonr(d["x"].to_numpy(float), d["y"].to_numpy(float))
    else:
        r, p = sps.spearmanr(d["x"].to_numpy(float), d["y"].to_numpy(float))

    r = float(r)
    p = float(p)
    return {"n": n, "r": r, "r2": float(r * r), "p_value": p}


def _fmt_p(p: object) -> str:
    if p is None:
        return "—"
    try:
        pf = float(p)
        if np.isnan(pf):
            return "—"
        return "<0.0001" if pf < 1e-4 else f"{pf:.4g}"
    except Exception:
        return "—"


def _fmt_num(x: object, digits: int = 4) -> str:
    if x is None:
        return "—"
    try:
        xf = float(x)
        if np.isnan(xf):
            return "—"
        return f"{xf:.{digits}f}"
    except Exception:
        return "—"


def _sanitize_corr_xy(cols_num: List[str]) -> Tuple[str, str]:
    """
    Ensure corr_x_var/corr_y_var exist in options and are distinct.
    """
    cur_x = st.session_state.get("corr_x_var")
    cur_y = st.session_state.get("corr_y_var")

    if cur_x not in cols_num:
        st.session_state["corr_x_var"] = cols_num[0]
        cur_x = cols_num[0]

    if cur_y not in cols_num or cur_y == cur_x:
        st.session_state["corr_y_var"] = next((c for c in cols_num if c != cur_x), cur_x)
        cur_y = st.session_state["corr_y_var"]

    return str(cur_x), str(cur_y)


def _posthoc_to_df(posthoc_obj: object) -> pd.DataFrame:
    """
    Expects `posthoc` column from stats module which is typically:
      {"method": "...", "comparisons": [ {...}, {...} ] }
    """
    if not isinstance(posthoc_obj, dict):
        return pd.DataFrame()
    comps = posthoc_obj.get("comparisons")
    if not isinstance(comps, list):
        return pd.DataFrame()
    out = pd.DataFrame(comps)
    for c in ["p_adj", "p_raw", "p_value"]:
        if c in out.columns:
            out[c] = out[c].apply(_fmt_p)
    return out


def describe_by_group(df: pd.DataFrame, group_cols: List[str], metric: str) -> pd.DataFrame:
    sub = df[group_cols + [metric]].dropna()
    if sub.empty:
        return pd.DataFrame()

    g = (
        sub.groupby(group_cols)[metric]
        .agg(
            n="count",
            mean="mean",
            sd=lambda s: float(s.std(ddof=1)),
            min="min",
            p10=lambda s: float(s.quantile(0.10)),
            p25=lambda s: float(s.quantile(0.25)),
            median="median",
            p75=lambda s: float(s.quantile(0.75)),
            p90=lambda s: float(s.quantile(0.90)),
            max="max",
        )
        .reset_index()
    )
    g["range"] = g["max"] - g["min"]
    g["cv_pct"] = g.apply(lambda r: (np.nan if r["mean"] == 0 else 100.0 * r["sd"] / abs(r["mean"])), axis=1)
    return g


# ======================================================================================
# Free mode Excel helpers
# ======================================================================================


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
    Active dataset:
      - Free mode: raw_df
      - Strict mode: parsed.house_summary (or first dataframe if missing)
    """
    if st.session_state.get("raw_mode") and isinstance(st.session_state.get("raw_df"), pd.DataFrame):
        return st.session_state["raw_df"]

    state = get_state()
    if state.parsed is None:
        return None

    dfs = state.parsed.to_dataframes()
    if "house_summary" in dfs:
        return dfs["house_summary"].copy()
    if dfs:
        return next(iter(dfs.values())).copy()
    return None


# ======================================================================================
# BLOQUE DE EDICIÓN: UI HELPERS (DESIGN + FILTERS)
# ======================================================================================


def render_design_controls(
    df: pd.DataFrame, *, prefix_key: str
) -> Tuple[str, str, Optional[str], Optional[str], Dict[str, List[str]]]:
    """
    Selectors for:
      - DV (Y)
      - Factor A / Factor B (optional)
      - Block (optional)
      - Filters (pre-analysis)
    """
    num = numeric_cols(df)
    cat = categorical_cols(df)

    if not num:
        st.error("No hay columnas numéricas disponibles para Y.")
        return "", "", None, None, {}
    if not cat:
        st.error("No hay columnas categóricas disponibles para factores/filtros.")
        return "", "", None, None, {}

    dv_default = st.session_state.get("dv_col")
    if dv_default not in num:
        dv_default = num[0]

    fa_default = st.session_state.get("factor_a")
    if fa_default not in cat:
        fa_default = cat[0]

    dv_col = st.selectbox("Variable dependiente (Y)", options=num, index=num.index(dv_default), key=f"{prefix_key}_dv")
    factor_a = st.selectbox("Factor A (principal)", options=cat, index=cat.index(fa_default), key=f"{prefix_key}_fa")

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

    st.markdown("#### Bloqueos / filtros (pre-análisis)")
    filter_cols = st.multiselect("Columnas a filtrar", options=cat, default=[], key=f"{prefix_key}_filter_cols")
    filters: Dict[str, List[str]] = {}
    for col in filter_cols:
        levels = sorted(df[col].dropna().astype(str).unique().tolist())
        chosen = st.multiselect(
            f"Niveles permitidos: {col}",
            options=levels,
            default=levels,
            key=f"{prefix_key}_filter_levels_{col}",
        )
        filters[col] = chosen

    # persist app-level choices
    st.session_state["dv_col"] = dv_col
    st.session_state["factor_a"] = factor_a
    st.session_state["factor_b"] = factor_b
    st.session_state["block_col"] = block_col
    st.session_state["filters"] = filters

    return dv_col, factor_a, factor_b, block_col, filters


# ======================================================================================
# BLOQUE DE EDICIÓN: LOAD/INPUT (STRICT -> FREE MODE)
# ======================================================================================


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


def maybe_parse_main_upload(uploaded_main: Optional[object]) -> None:
    if uploaded_main is None:
        return

    file_name = uploaded_main.name
    file_bytes = uploaded_main.getvalue()

    # reset
    st.session_state["raw_mode"] = False
    st.session_state["raw_df"] = None
    st.session_state["raw_sheet_name"] = None

    # strict parser
    try:
        parsed = parse_uploaded_file(file_name, file_bytes)
        set_state(parsed=parsed, warnings=[])
        return
    except Exception as e:
        st.warning(f"Modo PCTA estándar no aplica. Activando Modo Libre. Detalle: {e}")
        set_state(parsed=None, warnings=[])

    # free mode
    lower = file_name.lower()
    if lower.endswith(".xlsx"):
        sheets = _excel_sheets(file_bytes)
        if not sheets:
            st.error("No pude leer hojas del Excel en Modo Libre.")
            return
        with st.sidebar:
            st.subheader("Modo Libre — hoja a usar")
            sheet = st.selectbox("Hoja", options=sheets, index=0, key="raw_sheet_select")
        st.session_state["raw_sheet_name"] = sheet
        st.session_state["raw_df"] = _read_excel_sheet(file_bytes, sheet_name=sheet)
        st.session_state["raw_mode"] = True
        return

    if lower.endswith(".csv"):
        st.session_state["raw_df"] = pd.read_csv(BytesIO(file_bytes))
        st.session_state["raw_mode"] = True
        return

    st.error("Formato no soportado. Sube un .xlsx o .csv.")


# ======================================================================================
# BLOQUE DE EDICIÓN: TAB 1 (SELECCIÓN)
# ======================================================================================


def tab_1_select_variable_and_run() -> None:
    st.subheader("1) Definir diseño (Y/A/B/bloque/filtros)")
    df = get_active_df()
    if df is None:
        st.info("Carga un archivo en la barra lateral para comenzar.")
        return
    if st.session_state.get("raw_mode"):
        st.caption("Modo Libre activado: puedes elegir columnas con cualquier nombre.")
    render_design_controls(df, prefix_key="tab1")


# ======================================================================================
# BLOQUE DE EDICIÓN: TAB 2 (RESULTADOS) — DESCRIPTIVO + DISTRIBUCIONES + CORRELACIÓN
# ======================================================================================


def tab_2_results_for_selected_variable() -> None:
    st.subheader("2) Resultados (descriptivo + distribuciones + correlación)")
    _inject_table_css()

    df = get_active_df()
    if df is None:
        st.info("Carga un archivo primero.")
        return

    st.markdown("### A) Diseño (para descriptivo / distribuciones)")
    dv_col, factor_a, factor_b, block_col, filters = render_design_controls(df, prefix_key="tab2_design")
    df_post = apply_filters(df, filters)
    if df_post.empty:
        st.error("Con los filtros actuales no quedan filas.")
        return

    group_cols = [factor_a] + ([factor_b] if factor_b else [])

    st.divider()
    st.markdown("### B) Resumen descriptivo")
    desc = describe_by_group(df_post, group_cols, dv_col)
    st.dataframe(desc, use_container_width=True, hide_index=True)

    try:
        import plotly.express as px
    except Exception:
        st.error("Plotly no está instalado.")
        return

    st.divider()
    st.markdown("### C) Distribuciones (gráficas)")
    bins = st.slider("Bins (histograma)", 5, 100, 30, key="hist_bins")
    st.plotly_chart(px.histogram(df_post, x=dv_col, nbins=bins, template="simple_white"), use_container_width=True)

    show_violin = st.checkbox("Ver violin plot", value=True, key="show_violin")
    show_points = st.checkbox("Mostrar puntos (strip)", value=True, key="show_points")

    if factor_b:
        fig_dist = (
            px.violin(df_post, x=factor_a, y=dv_col, color=factor_b, box=True, points="all" if show_points else False, template="simple_white")
            if show_violin
            else px.box(df_post, x=factor_a, y=dv_col, color=factor_b, points="all" if show_points else False, template="simple_white")
        )
    else:
        fig_dist = (
            px.violin(df_post, x=factor_a, y=dv_col, box=True, points="all" if show_points else False, template="simple_white")
            if show_violin
            else px.box(df_post, x=factor_a, y=dv_col, points="all" if show_points else False, template="simple_white")
        )
    st.plotly_chart(fig_dist, use_container_width=True)

    # ------------------------- Correlación completa -------------------------
    st.divider()
    st.markdown("### D) Correlación (módulo independiente)")

    corr_mode = st.radio(
        "Modo de correlación",
        options=["global", "by_group", "compare_full_post"],
        index=["global", "by_group", "compare_full_post"].index(st.session_state.get("corr_mode", "global")),
        format_func=lambda v: {
            "global": "Global (una población)",
            "by_group": "Por factor (r/p por nivel)",
            "compare_full_post": "Comparar: completo vs post-filtro (un solo plano, color+símbolo)",
        }[v],
        key="corr_mode",
        horizontal=True,
    )
    method = st.selectbox("Método", options=["pearson", "spearman"], index=0, key="corr_method")

    # Compare FULL vs POST
    if corr_mode == "compare_full_post":
        num_full = set(numeric_cols(df))
        num_post = set(numeric_cols(df_post))
        cols_num = sorted(num_full.intersection(num_post))
        if len(cols_num) < 2:
            st.info("Para comparar FULL vs POST, se necesitan >=2 columnas numéricas presentes en ambos datasets.")
            return

        x0, y0 = _sanitize_corr_xy(cols_num)
        c1, c2 = st.columns(2)
        with c1:
            x_var = st.selectbox("Variable X", options=cols_num, index=cols_num.index(x0), key="corr_x_var")
        with c2:
            if st.session_state.get("corr_y_var") == x_var:
                st.session_state["corr_y_var"] = next((c for c in cols_num if c != x_var), x_var)
            y_var = st.selectbox("Variable Y", options=cols_num, index=cols_num.index(st.session_state["corr_y_var"]), key="corr_y_var")

        c_full = correlation_stats(df[x_var], df[y_var], method=method)
        c_post = correlation_stats(df_post[x_var], df_post[y_var], method=method)

        st.markdown("#### Estadísticos (dos poblaciones)")
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("n FULL", str(c_full["n"]))
        a2.metric("r FULL", _fmt_num(c_full["r"]))
        a3.metric("r² FULL", _fmt_num(c_full["r2"]))
        a4.metric("p FULL", _fmt_p(c_full["p_value"]))

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("n POST", str(c_post["n"]))
        b2.metric("r POST", _fmt_num(c_post["r"]))
        b3.metric("r² POST", _fmt_num(c_post["r2"]))
        b4.metric("p POST", _fmt_p(c_post["p_value"]))

        df_full_plot = df[[x_var, y_var]].dropna().copy()
        df_full_plot["_poblacion"] = "FULL"
        df_post_plot = df_post[[x_var, y_var]].dropna().copy()
        df_post_plot["_poblacion"] = "POST-FILTRO"
        overlay = pd.concat([df_full_plot, df_post_plot], ignore_index=True)

        fig = px.scatter(
            overlay,
            x=x_var,
            y=y_var,
            color="_poblacion",
            symbol="_poblacion",
            template="simple_white",
            title=f"FULL vs POST-FILTRO: {x_var} vs {y_var}",
        )
        st.plotly_chart(fig, use_container_width=True)
        return

    # global/by_group scope selector
    corr_scope = st.radio(
        "Dataset base para correlación",
        options=["post_filter", "full"],
        index=0 if st.session_state.get("corr_scope") == "post_filter" else 1,
        format_func=lambda v: "Usar datos post-filtro" if v == "post_filter" else "Usar datos completos (sin filtros)",
        key="corr_scope",
        horizontal=True,
    )
    corr_df = df_post if corr_scope == "post_filter" else df

    cols_num = numeric_cols(corr_df)
    if len(cols_num) < 2:
        st.info("No hay suficientes variables numéricas para correlación.")
        return

    x0, y0 = _sanitize_corr_xy(cols_num)
    c1, c2 = st.columns(2)
    with c1:
        x_var = st.selectbox("Variable X", options=cols_num, index=cols_num.index(x0), key="corr_x_var")
    with c2:
        if st.session_state.get("corr_y_var") == x_var:
            st.session_state["corr_y_var"] = next((c for c in cols_num if c != x_var), x_var)
        y_var = st.selectbox("Variable Y", options=cols_num, index=cols_num.index(st.session_state["corr_y_var"]), key="corr_y_var")

    if corr_mode == "global":
        cat_cols = categorical_cols(corr_df)
        color_opts = [None] + cat_cols
        default_color = st.session_state.get("corr_global_color_by")
        if default_color not in color_opts:
            default_color = None

        color_by = st.selectbox(
            "Color (solo visual en modo Global)",
            options=color_opts,
            index=color_opts.index(default_color),
            format_func=lambda v: "— Sin color —" if v is None else str(v),
            key="corr_global_color_by",
        )

        c = correlation_stats(corr_df[x_var], corr_df[y_var], method=method)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("n", str(c["n"]))
        k2.metric("r", _fmt_num(c["r"]))
        k3.metric("r²", _fmt_num(c["r2"]))
        k4.metric("p-value", _fmt_p(c["p_value"]))

        if color_by is not None:
            st.caption("Nota: el color representa subgrupos, pero los estadísticos mostrados arriba son GLOBAL (una sola población).")
            df_plot = corr_df[[x_var, y_var, color_by]].dropna()
            fig = px.scatter(
                df_plot,
                x=x_var,
                y=y_var,
                color=color_by,
                trendline="ols" if method == "pearson" else None,
                template="simple_white",
                title=f"Scatter (global): {x_var} vs {y_var} — color={color_by}",
            )
        else:
            df_plot = corr_df[[x_var, y_var]].dropna()
            fig = px.scatter(
                df_plot,
                x=x_var,
                y=y_var,
                trendline="ols" if method == "pearson" else None,
                template="simple_white",
                title=f"Scatter (global): {x_var} vs {y_var}",
            )
        st.plotly_chart(fig, use_container_width=True)
        return

    # by_group
    cat_cols = categorical_cols(corr_df)
    if not cat_cols:
        st.warning("No hay columnas categóricas para calcular correlación por factor.")
        return

    default_group = st.session_state.get("corr_group_col")
    if default_group not in cat_cols:
        default_group = factor_a if factor_a in cat_cols else cat_cols[0]

    group_col = st.selectbox(
        "Factor para separar poblaciones (r/p por nivel)",
        options=cat_cols,
        index=cat_cols.index(default_group),
        key="corr_group_col",
    )

    show_trendlines = st.checkbox(
        "Mostrar tendencia por grupo (solo Pearson)",
        value=bool(st.session_state.get("corr_show_group_trendlines", True)),
        key="corr_show_group_trendlines",
    )

    df_xy = corr_df[[group_col, x_var, y_var]].dropna()
    if df_xy.empty:
        st.warning("No hay filas válidas (NA) para X/Y en el dataset actual.")
        return

    rows = []
    for lvl, g in df_xy.groupby(group_col):
        cc = correlation_stats(g[x_var], g[y_var], method=method)
        rows.append({"grupo": str(lvl), "n": cc["n"], "r": cc["r"], "r2": cc["r2"], "p_value": cc["p_value"]})

    stats_by_group = pd.DataFrame(rows).sort_values(["grupo"])
    st.markdown("#### Estadísticos por grupo (poblaciones separadas)")
    st.dataframe(stats_by_group, use_container_width=True, hide_index=True)

    fig = px.scatter(
        df_xy,
        x=x_var,
        y=y_var,
        color=group_col,
        trendline="ols" if (show_trendlines and method == "pearson") else None,
        template="simple_white",
        title=f"Scatter por grupo: {x_var} vs {y_var} — color={group_col}",
    )
    st.plotly_chart(fig, use_container_width=True)


# ======================================================================================
# BLOQUE DE EDICIÓN: TAB 3 (TEST DE MEDIAS) — OMNIBUS + POSTHOC BAJO DEMANDA
# ======================================================================================


def tab_3_mean_tests() -> None:
    st.subheader("3) Test de medias (inferencial) — Omnibus + Posthoc bajo demanda")

    df = get_active_df()
    if df is None:
        st.info("Carga un archivo primero.")
        return

    dv_col, factor_a, factor_b, block_col, filters = render_design_controls(df, prefix_key="tab3")
    df_f = apply_filters(df, filters)

    if df_f.empty:
        st.error("Con los filtros actuales no quedan filas.")
        return

    st.divider()
    alpha = st.number_input("Alpha", min_value=0.001, max_value=0.2, value=0.05, step=0.005, key="tab3_alpha")

    # Factorial A×B
    if factor_b:
        st.markdown("### ANOVA factorial (A×B)")
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
        st.info("Posthoc factorial: siguiente iteración (si lo necesitas).")
        return

    # Omnibus without posthoc
    st.markdown("### Omnibus (selección automática del test)")
    omni_df, rep, min_n, enabled, warnings = run_inferential_statistics_df(
        df_f,
        metric=dv_col,
        group_col=factor_a,
        options=StatsOptions(alpha=float(alpha), enable_posthoc=False),
    )

    if omni_df.empty:
        st.warning("No se pudo calcular el test omnibus.")
        return

    row = omni_df.iloc[0].to_dict()
    p_val = row.get("p_value")
    test_name = str(row.get("test"))

    # Cards summary
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Test", test_name)
    c2.metric("p-value", _fmt_p(p_val))
    c3.metric("Shapiro (min p)", _fmt_p(row.get("assumptions_shapiro_min_p")))
    c4.metric("Levene p", _fmt_p(row.get("assumptions_levene_p")))
    c5.metric("Effect size", _fmt_num(row.get("effect_size"), digits=4))

    st.markdown("#### Detalle omnibus")
    st.dataframe(pd.DataFrame([row]), use_container_width=True, hide_index=True)

    with st.expander("Replicación por grupo", expanded=False):
        rep_df = pd.DataFrame([{"grupo": k, "n": int(v)} for k, v in sorted(rep.items(), key=lambda kv: str(kv[0]))])
        st.dataframe(rep_df, use_container_width=True, hide_index=True)

    if warnings:
        with st.expander("Advertencias", expanded=False):
            st.dataframe(
                pd.DataFrame([{"code": w.code.value, "message": w.message, "context": w.context} for w in warnings]),
                use_container_width=True,
                hide_index=True,
            )

    if not enabled:
        st.warning("Inferencia deshabilitada: se requiere al menos n>=2 por grupo para p-values.")
        return

    # Posthoc policy
    st.divider()
    st.markdown("### Posthoc (test de medias)")

    sig = False
    try:
        if p_val is not None and not (isinstance(p_val, float) and np.isnan(float(p_val))):
            sig = float(p_val) <= float(alpha)
    except Exception:
        sig = False

    policy = st.radio(
        "¿Cuándo correr posthoc?",
        options=["auto_if_significant", "always", "never"],
        index=["auto_if_significant", "always", "never"].index(st.session_state.get("tab3_posthoc_policy", "auto_if_significant")),
        format_func=lambda v: {"auto_if_significant": "Solo si omnibus es significativo", "always": "Siempre", "never": "Nunca"}[v],
        key="tab3_posthoc_policy",
        horizontal=True,
    )

    if policy == "never":
        return

    if policy == "auto_if_significant" and not sig:
        st.info("Omnibus no significativo: posthoc omitido. Cambia a 'Siempre' si deseas explorarlo.")
        return

    # Run with posthoc enabled
    full_df, *_ = run_inferential_statistics_df(
        df_f,
        metric=dv_col,
        group_col=factor_a,
        options=StatsOptions(alpha=float(alpha), enable_posthoc=True),
    )
    if full_df.empty:
        st.warning("No se pudo calcular posthoc.")
        return

    posthoc_obj = full_df.loc[0, "posthoc"] if "posthoc" in full_df.columns else None
    ph_df = _posthoc_to_df(posthoc_obj)

    if ph_df.empty:
        st.warning("No hay comparaciones posthoc disponibles.")
        return

    st.markdown("#### Comparaciones pareadas (posthoc)")
    st.dataframe(ph_df, use_container_width=True, hide_index=True)


# ======================================================================================
# BLOQUE DE EDICIÓN: TAB 4 (KPIS PRODUCTIVOS) — Análisis económico integral
# ======================================================================================

# Importaciones globales para TAB 4 (al inicio del archivo principal, pero las dejamos aquí para claridad)
def tab_4_productive_kpis() -> None:
    """
    KPIs productivos: an��lisis económico integral.
    
    Soporta 2 modos:
    1. MODO PCTA: usa estructura HOUSE_SUMMARY (datos cargados)
    2. MODO MANUAL: el usuario ingresa múltiples tratamientos manualmente y compara
    
    Cálculos:
    - Consumo de alimento, conversión alimenticia (FCR)
    - Ingresos por venta, costos totales, margen
    - EPEF (Índice Europeo de Eficiencia Productiva)
    """
    st.subheader("4) KPIs Productivos — Análisis Económico Integral")

    state = get_state()
    has_pcta_data = state.parsed is not None

    # ---- Elegir modo ----
    st.markdown("### A) Seleccionar modo de entrada de datos")
    
    mode = st.radio(
        "¿Cómo ingresar datos productivos?",
        options=["modo_pcta", "modo_manual"],
        format_func=lambda v: {
            "modo_pcta": "📊 MODO PCTA (cargar desde archivo)",
            "modo_manual": "✏️ MODO MANUAL (múltiples tratamientos con comparación)",
        }[v],
        index=0 if has_pcta_data else 1,
        key="tab4_mode",
        horizontal=False,
    )

    if mode == "modo_pcta":
        _tab4_modo_pcta()
    else:
        _tab4_modo_manual()


def _tab4_modo_pcta() -> None:
    """
    Modo PCTA: usa datos de HOUSE_SUMMARY cargados.
    """
    st.markdown("#### Datos desde archivo PCTA (HOUSE_SUMMARY)")

    df = get_active_df()
    if df is None:
        st.info("Carga un archivo en la barra lateral para comenzar.")
        return

    # Filtración
    st.markdown("**Filtración de datos**")
    dv_col, factor_a, factor_b, block_col, filters = render_design_controls(df, prefix_key="tab4_design")
    df_post = apply_filters(df, filters)
    if df_post.empty:
        st.error("Con los filtros actuales no quedan filas.")
        return

    state = get_state()
    if state.parsed is None:
        st.error("Requiere datos cargados en modo PCTA estándar (con estructura HOUSE_SUMMARY).")
        return

    try:
        units = state.parsed.to_units()
    except Exception as e:
        st.error(f"Error al convertir datos a TrialUnitInput: {e}")
        return

    # Filtrar units
    units_filtered = [u for u in units if all(
        str(getattr(u, k, "")) in map(str, levels)
        for k, levels in filters.items()
        if hasattr(u, k) and levels
    )]

    if not units_filtered:
        st.error("Ninguna unidad coincide con los filtros.")
        return

    # Parámetros económicos
    st.divider()
    _render_economic_inputs(prefix="tab4_")
    
    # Cálculos
    st.divider()
    _render_kpi_results(units_filtered)


def _tab4_modo_manual() -> None:
    """
    Modo Manual: usuario ingresa múltiples tratamientos manualmente.
    Sistema de acordeones para ingresar datos y luego compara.
    """
    if not PRODUCTIVE_KPI_AVAILABLE:
        st.error("❌ Módulo de KPIs productivos no disponible. Verifica las dependencias.")
        return

    st.markdown("#### Ingreso manual de múltiples tratamientos")
    st.info("📋 Ingresa datos de cada tratamiento en los acordeones abajo. Luego analiza y compara.")

    # ---- PANEL INICIAL: Parámetros globales ----
    st.divider()
    st.markdown("### B) Parámetros globales del ensayo")

    col1, col2, col3 = st.columns(3)
    with col1:
        trial_id = st.text_input("ID del ensayo", value="TRIAL_MANUAL", key="tab4_trial_id_global")
    with col2:
        days_cycle = st.number_input("Días de ciclo (igual para todos)", min_value=1, value=42, step=1, key="tab4_days_global")
    with col3:
        num_treatments = st.number_input(
            "¿Cuántos tratamientos deseas evaluar?",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            key="tab4_num_treatments"
        )

    # ---- PANEL: Ingresar datos de cada tratamiento ----
    st.divider()
    st.markdown("### C) Datos de cada tratamiento")

    treatment_units: Dict[str, TrialUnitInput] = {}

    for idx in range(int(num_treatments)):
        with st.expander(
            f"📝 Tratamiento {idx + 1}",
            expanded=(idx == 0)
        ):
            col1, col2, col3 = st.columns(3)

            with col1:
                treatment_name = st.text_input(
                    f"Nombre del tratamiento",
                    value=f"Trat_{chr(65 + idx)}",
                    key=f"tab4_treatment_{idx}"
                )

            with col2:
                unit_id = st.text_input(
                    f"ID unidad (galpón)",
                    value=f"GALPÓN_{idx + 1}",
                    key=f"tab4_unit_id_{idx}"
                )

            with col3:
                st.number_input(
                    f"Número de réplica",
                    min_value=1,
                    value=1,
                    step=1,
                    key=f"tab4_replicate_{idx}"
                )

            # Datos de aves
            st.markdown("##### Datos de aves")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                birds_placed = st.number_input(
                    f"Aves colocadas",
                    min_value=1,
                    value=10000,
                    step=100,
                    key=f"tab4_birds_placed_{idx}"
                )

            with col2:
                mortality_total = st.number_input(
                    f"Aves muertas",
                    min_value=0,
                    value=200,
                    step=10,
                    key=f"tab4_mortality_{idx}"
                )

            with col3:
                bw_initial_g = st.number_input(
                    f"Peso inicial (g)",
                    min_value=1.0,
                    value=42.0,
                    step=1.0,
                    key=f"tab4_bw_initial_{idx}"
                )

            with col4:
                bw_final_g = st.number_input(
                    f"Peso final (g)",
                    min_value=1.0,
                    value=2400.0,
                    step=10.0,
                    key=f"tab4_bw_final_{idx}"
                )

            # Alimento
            st.markdown("##### Consumo de alimento")
            col1, col2 = st.columns(2)

            with col1:
                feed_delivered_kg = st.number_input(
                    f"Alimento suministrado (kg)",
                    min_value=0.0,
                    value=20000.0,
                    step=100.0,
                    key=f"tab4_feed_delivered_{idx}"
                )

            with col2:
                feed_refusals_kg = st.number_input(
                    f"Alimento rechazado (kg)",
                    min_value=0.0,
                    value=500.0,
                    step=50.0,
                    key=f"tab4_feed_refusals_{idx}"
                )

            # Crear TrialUnitInput
            treatment_units[treatment_name] = TrialUnitInput(
                trial_id=trial_id,
                unit_type=UnitType.house,
                unit_id=unit_id,
                treatment=treatment_name,
                days=int(days_cycle),
                birds_placed=int(birds_placed),
                mortality_total=int(mortality_total),
                birds_sold=None,
                feed_delivered_kg=float(feed_delivered_kg),
                feed_refusals_kg=float(feed_refusals_kg),
                bw_initial_mean_g=float(bw_initial_g),
                bw_final_mean_g=float(bw_final_g),
                bw_final_sd_g=None,
                final_sample_n=None,
                diet_cost_per_kg=0.40,
                additive_cost_total=0.0,
                chick_cost_per_bird=0.80,
                other_variable_costs_total=0.0,
            )

    # ---- PANEL: Parámetros económicos ----
    st.divider()
    _render_economic_inputs(prefix="manual_")

    # ---- PANEL: Calcular y comparar ----
    st.divider()
    st.markdown("### D) Calcular y comparar tratamientos")

    if st.button("🧮 Calcular y comparar", key="btn_calc_compare", type="primary"):
        if not treatment_units:
            st.error("No hay tratamientos para analizar.")
            return

        try:
            kpi_inputs = ProductiveKPIInputs(
                price_kg_sold=float(st.session_state.get("manual_tab4_price_kg", 2.50)),
                cost_feed_per_kg=float(st.session_state.get("manual_tab4_cost_feed_kg", 0.45)),
                cost_chick_per_bird=float(st.session_state.get("manual_tab4_cost_chick", 0.80)),
                other_costs_per_bird=float(st.session_state.get("manual_tab4_other_cost_bird", 0.15)),
                mortality_cost_pct=float(st.session_state.get("manual_tab4_mortality_recovery", 20.0)),
            )

            units_list: List[TrialUnitInput] = []
            for treatment_name, unit in treatment_units.items():
                unit.diet_cost_per_kg = kpi_inputs.cost_feed_per_kg
                unit.chick_cost_per_bird = kpi_inputs.cost_chick_per_bird
                unit.other_variable_costs_total = kpi_inputs.other_costs_per_bird * unit.birds_placed
                units_list.append(unit)

            # Ahora compute_productive_kpis_batch ESTÁ DISPONIBLE (importado al nivel del módulo)
            kpis, warnings = compute_productive_kpis_batch(units_list, kpi_inputs)

        except Exception as calc_error:
            st.error(f"❌ Error al calcular KPIs: {calc_error}")
            import traceback
            with st.expander("🔧 Detalles técnicos"):
                st.code(traceback.format_exc())
            return

        if warnings:
            with st.expander("⚠️ Advertencias", expanded=False):
                for w in warnings:
                    st.warning(f"**{w.code.value}**: {w.message}")

        try:
            df_kpis = kpis_to_dataframe(kpis)
        except Exception as df_error:
            st.error(f"❌ Error al convertir a DataFrame: {df_error}")
            return

        # ---- RESULTADOS ----
        st.markdown("### E) Resultados por tratamiento")

        st.markdown("#### Tabla comparativa de KPIs")
        try:
            st.dataframe(
                df_kpis[["treatment", "birds_placed", "mortality_pct", "feed_consumed_kg", 
                          "wg_total_per_bird_g", "fcr", "revenue_total", "total_cost", 
                          "margin_total", "margin_pct", "epef"]].round(2),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "treatment": st.column_config.TextColumn("Tratamiento", width=100),
                    "birds_placed": st.column_config.NumberColumn("Aves", format="%d"),
                    "mortality_pct": st.column_config.NumberColumn("Mort. %", format="%.1f"),
                    "feed_consumed_kg": st.column_config.NumberColumn("Alimento (kg)", format="%.0f"),
                    "wg_total_per_bird_g": st.column_config.NumberColumn("Ganancia/ave (g)", format="%.0f"),
                    "fcr": st.column_config.NumberColumn("FCR", format="%.2f"),
                    "revenue_total": st.column_config.NumberColumn("Ingresos ($)", format="%.2f"),
                    "total_cost": st.column_config.NumberColumn("Costo ($)", format="%.2f"),
                    "margin_total": st.column_config.NumberColumn("Margen ($)", format="%.2f"),
                    "margin_pct": st.column_config.NumberColumn("Margen %", format="%.1f"),
                    "epef": st.column_config.NumberColumn("EPEF", format="%.1f"),
                }
            )
        except KeyError as key_error:
            st.error(f"❌ Error en columnas: {key_error}")
            return

        # ---- Resumen por tratamiento ----
        st.divider()
        st.markdown("### F) Resumen por tratamiento")

        try:
            summary_treatment = compute_summary_by_treatment(df_kpis)
            if not summary_treatment.empty:
                st.dataframe(summary_treatment.round(2), use_container_width=True, hide_index=True)
        except Exception as summary_error:
            st.warning(f"⚠️ Error: {summary_error}")

        # ---- Totales globales ----
        st.divider()
        st.markdown("### G) Totales globales")

        try:
            totals = compute_total_summary(df_kpis)
            if totals:
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Total aves", f"{totals.get('total_birds_placed', 0):,.0f}")
                col2.metric("Alimento total", f"{totals.get('total_feed_consumed_kg', 0):,.0f} kg")
                col3.metric("FCR prom.", f"{totals.get('avg_fcr', 0):.2f}")
                col4.metric("EPEF prom.", f"{totals.get('avg_epef', 0):.1f}" if totals.get('avg_epef') else "—")
                col5.metric("Margen total", f"${totals.get('total_margin', 0):,.2f}")
        except Exception as totals_error:
            st.warning(f"⚠️ Error: {totals_error}")

        # ---- Gráficos ----
        st.divider()
        st.markdown("### H) Gráficos")

        try:
            import plotly.graph_objects as go
            import plotly.express as px

            fig_bars = go.Figure()
            fig_bars.add_trace(go.Bar(name="Ingresos", x=df_kpis["treatment"], y=df_kpis["revenue_total"], marker_color="#4CAF50"))
            fig_bars.add_trace(go.Bar(name="Costo", x=df_kpis["treatment"], y=df_kpis["total_cost"], marker_color="#f44336"))
            fig_bars.add_trace(go.Bar(name="Margen", x=df_kpis["treatment"], y=df_kpis["margin_total"], marker_color="#2176ff"))
            fig_bars.update_layout(title="Ingresos vs Costos", barmode="group", template="plotly_white", height=400)
            st.plotly_chart(fig_bars, use_container_width=True)

        except ImportError:
            st.info("📦 Instala: pip install plotly")
        except Exception as plot_error:
            st.warning(f"Error en gráficos: {plot_error}")

        # ---- Análisis comparativo ----
        st.divider()
        st.markdown("### I) Análisis de diferencias")

        if len(df_kpis) > 1:
            try:
                best_idx = df_kpis["margin_total"].idxmax()
                worst_idx = df_kpis["margin_total"].idxmin()
                
                best_treatment = df_kpis.loc[best_idx, "treatment"]
                best_value = df_kpis.loc[best_idx, "margin_total"]
                worst_treatment = df_kpis.loc[worst_idx, "treatment"]
                worst_value = df_kpis.loc[worst_idx, "margin_total"]

                diff = best_value - worst_value
                pct = (diff / abs(worst_value) * 100) if worst_value != 0 else 0

                st.success(f"🏆 Mejor: {best_treatment} → ${best_value:,.2f}")
                st.error(f"📉 Menor: {worst_treatment} → ${worst_value:,.2f}")
                st.info(f"💰 Ventaja: ${diff:,.2f} ({pct:.1f}%)")

            except Exception as analysis_error:
                st.warning(f"Error: {analysis_error}")


def _render_economic_inputs(prefix: str = "tab4_") -> None:
    """
    Renderiza inputs para parámetros económicos.
    """
    st.markdown("### Parámetros económicos (supuestos)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Precio y costo de alimento")
        st.number_input(
            "Precio de venta ($/kg liveweight)",
            min_value=0.0,
            value=st.session_state.get(f"{prefix}price_kg", 2.50),
            step=0.1,
            key=f"{prefix}price_kg"
        )
        
        st.number_input(
            "Costo de alimento ($/kg consumido)",
            min_value=0.0,
            value=st.session_state.get(f"{prefix}cost_feed_kg", 0.45),
            step=0.01,
            key=f"{prefix}cost_feed_kg"
        )
    
    with col2:
        st.markdown("#### Costos iniciales y adicionales")
        st.number_input(
            "Costo del pollito ($/ave)",
            min_value=0.0,
            value=st.session_state.get(f"{prefix}cost_chick", 0.80),
            step=0.05,
            key=f"{prefix}cost_chick"
        )
        
        st.number_input(
            "Otros costos ($/ave colocada)",
            min_value=0.0,
            value=st.session_state.get(f"{prefix}other_cost_bird", 0.15),
            step=0.05,
            key=f"{prefix}other_cost_bird"
        )

    st.slider(
        "% de valor del pollito recuperado si muere",
        min_value=0,
        max_value=100,
        value=st.session_state.get(f"{prefix}mortality_recovery", 20),
        step=5,
        key=f"{prefix}mortality_recovery"
    )


def _render_kpi_results(units_filtered: List[Any]) -> None:
    """
    Renderiza resultados de KPIs para modo PCTA.
    """
    from pcta.core.productive_kpis import (
        ProductiveKPIInputs,
        compute_productive_kpis_batch,
        kpis_to_dataframe,
        compute_summary_by_treatment,
        compute_total_summary,
    )

    try:
        kpi_inputs = ProductiveKPIInputs(
            price_kg_sold=float(st.session_state.get("tab4_price_kg", 2.50)),
            cost_feed_per_kg=float(st.session_state.get("tab4_cost_feed_kg", 0.45)),
            cost_chick_per_bird=float(st.session_state.get("tab4_cost_chick", 0.80)),
            other_costs_per_bird=float(st.session_state.get("tab4_other_cost_bird", 0.15)),
            mortality_cost_pct=float(st.session_state.get("tab4_mortality_recovery", 20.0)),
        )
    except (ValueError, TypeError) as e:
        st.error(f"❌ Error en parámetros: {e}")
        return

    st.markdown("### E) Resultados por unidad")

    try:
        kpis, warnings = compute_productive_kpis_batch(units_filtered, kpi_inputs)
        df_kpis = kpis_to_dataframe(kpis)

        if warnings:
            with st.expander("⚠️ Advertencias", expanded=False):
                for w in warnings:
                    st.warning(f"**{w.code.value}**: {w.message}")

        st.dataframe(
            df_kpis.round(2),
            use_container_width=True,
            hide_index=True,
            column_config={
                "trial_id": st.column_config.TextColumn("Ensayo"),
                "unit_id": st.column_config.TextColumn("Unidad"),
                "treatment": st.column_config.TextColumn("Tratamiento"),
                "birds_placed": st.column_config.NumberColumn("Aves", format="%d"),
                "mortality_pct": st.column_config.NumberColumn("Mort. %", format="%.1f"),
                "feed_consumed_kg": st.column_config.NumberColumn("Alimento (kg)", format="%.1f"),
                "fcr": st.column_config.NumberColumn("FCR", format="%.2f"),
                "revenue_total": st.column_config.NumberColumn("Ingresos ($)", format="%.2f"),
                "total_cost": st.column_config.NumberColumn("Costo ($)", format="%.2f"),
                "margin_total": st.column_config.NumberColumn("Margen ($)", format="%.2f"),
                "epef": st.column_config.NumberColumn("EPEF", format="%.1f"),
            }
        )

        st.divider()
        st.markdown("### F) Resumen por tratamiento")

        summary_treatment = compute_summary_by_treatment(df_kpis)
        if not summary_treatment.empty:
            st.dataframe(summary_treatment.round(2), use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("### G) Totales globales")

        totals = compute_total_summary(df_kpis)

        if totals:
            col1, col2, col3, col4, col5 = st.columns(5)

            col1.metric("Aves colocadas", f"{totals.get('total_birds_placed', 0):,.0f}")
            col2.metric("Alimento consumido", f"{totals.get('total_feed_consumed_kg', 0):,.0f} kg")
            col3.metric("FCR promedio", f"{totals.get('avg_fcr', 0):.2f}")
            col4.metric("EPEF promedio", f"{totals.get('avg_epef', 0):.1f}" if totals.get('avg_epef') else "—")
            col5.metric("Margen total", f"${totals.get('total_margin', 0):,.2f}")

            financial_summary = pd.DataFrame([
                {"Métrica": "Ingresos", "Valor ($)": totals.get('total_revenue', 0)},
                {"Métrica": "Costo total", "Valor ($)": totals.get('total_cost', 0)},
                {"Métrica": "Margen bruto", "Valor ($)": totals.get('total_margin', 0)},
            ])
            st.dataframe(financial_summary, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"❌ Error: {e}")
        import traceback
        with st.expander("Detalles"):
            st.code(traceback.format_exc())


# ======================================================================================
# BLOQUE DE EDICIÓN: EXPORT
# ======================================================================================


def tab_export() -> None:
    st.subheader("5) Exportar")
    st.info("Export en modo libre: pendiente (lo agregamos si lo necesitas).")
