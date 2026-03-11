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
from pcta.core.factorial_stats import FactorialOptions, run_factorial_anova_df
from pcta.core.io import parse_uploaded_file
from pcta.core.schemas import AnalysisWarning, ParsedInput
from pcta.core.stats import StatsOptions, run_inferential_statistics_df


# ------------------------ Estado ------------------------


@dataclass(frozen=True)
class AppState:
    parsed: Optional[ParsedInput]
    warnings: List[AnalysisWarning]


def init_state() -> None:
    if "pcta_state" not in st.session_state:
        st.session_state["pcta_state"] = AppState(parsed=None, warnings=[])

    # Modo libre
    st.session_state.setdefault("raw_mode", False)
    st.session_state.setdefault("raw_df", None)  # pd.DataFrame | None
    st.session_state.setdefault("raw_sheet_name", None)

    # Selecciones de análisis (diseño)
    st.session_state.setdefault("dv_col", None)
    st.session_state.setdefault("factor_a", None)
    st.session_state.setdefault("factor_b", None)
    st.session_state.setdefault("block_col", None)
    st.session_state.setdefault("filters", {})

    # Correlación (módulo independiente)
    st.session_state.setdefault("corr_x_var", None)
    st.session_state.setdefault("corr_y_var", None)
    st.session_state.setdefault("corr_method", "pearson")
    st.session_state.setdefault("corr_mode", "global")  # global | by_group | compare_full_post
    st.session_state.setdefault("corr_group_col", None)
    st.session_state.setdefault("corr_scope", "post_filter")  # post_filter | full
    st.session_state.setdefault("corr_global_color_by", None)  # None | <cat col>
    st.session_state.setdefault("corr_show_group_trendlines", True)

    # Plots (distribuciones)
    st.session_state.setdefault("show_violin", True)
    st.session_state.setdefault("show_points", True)


def get_state() -> AppState:
    return st.session_state["pcta_state"]


def set_state(*, parsed: Optional[ParsedInput], warnings: List[AnalysisWarning]) -> None:
    st.session_state["pcta_state"] = AppState(parsed=parsed, warnings=warnings)


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


def _excel_sheets(file_bytes: bytes) -> List[str]:
    try:
        xls = pd.ExcelFile(BytesIO(file_bytes))
        return list(xls.sheet_names)
    except Exception:
        return []


def _read_excel_sheet(file_bytes: bytes, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name)


def get_active_df() -> Optional[pd.DataFrame]:
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


def render_design_controls(
    df: pd.DataFrame, *, prefix_key: str
) -> Tuple[str, str, Optional[str], Optional[str], Dict[str, List[str]]]:
    num = numeric_cols(df)
    cat = categorical_cols(df)

    if not num:
        st.error("No hay columnas numéricas para Y.")
        return "", "", None, None, {}
    if not cat:
        st.error("No hay columnas categóricas para factores/filtros.")
        return "", "", None, None, {}

    dv_default = st.session_state.get("dv_col")
    if dv_default not in num:
        dv_default = num[0]

    fa_default = st.session_state.get("factor_a")
    if fa_default not in cat:
        fa_default = cat[0]

    dv_col = st.selectbox("Variable dependiente (Y)", options=num, index=num.index(dv_default), key=f"{prefix_key}_dv")
    st.session_state["dv_col"] = dv_col

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
    st.session_state["filters"] = filters

    return dv_col, factor_a, factor_b, block_col, filters


def describe_by_group(df: pd.DataFrame, group_cols: List[str], metric: str) -> pd.DataFrame:
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

    return {"n": n, "r": float(r), "r2": float(r * r), "p_value": float(p)}


def _sanitize_corr_xy(cols_num: List[str]) -> Tuple[str, str]:
    """
    Ensures corr_x_var/corr_y_var are valid options and distinct.
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


# ------------------------ Sidebar ------------------------


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


# ------------------------ Carga (estricto o libre) ------------------------


def maybe_parse_main_upload(uploaded_main: Optional[object]) -> None:
    if uploaded_main is None:
        return

    file_name = uploaded_main.name
    file_bytes = uploaded_main.getvalue()

    st.session_state["raw_mode"] = False
    st.session_state["raw_df"] = None
    st.session_state["raw_sheet_name"] = None

    try:
        parsed = parse_uploaded_file(file_name, file_bytes)
        set_state(parsed=parsed, warnings=[])
        return
    except Exception as e:
        st.warning(f"Modo PCTA estándar no aplica. Activando Modo Libre. Detalle: {e}")
        set_state(parsed=None, warnings=[])

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


# ------------------------ Tabs ------------------------


def tab_1_select_variable_and_run() -> None:
    st.subheader("1) Definir diseño (Y/A/B/bloque/filtros)")

    df = get_active_df()
    if df is None:
        st.info("Carga un archivo en la barra lateral para comenzar.")
        return

    if st.session_state.get("raw_mode"):
        st.caption("Modo Libre activado: puedes elegir columnas con cualquier nombre.")

    render_design_controls(df, prefix_key="tab1")


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
    decimals = st.slider("Decimales", 0, 6, 2, key="tab2_decimals")
    desc = describe_by_group(df_post, group_cols, dv_col)
    st.dataframe(format_desc_table(desc, group_cols=group_cols, decimals=decimals), use_container_width=True, hide_index=True)

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

    # ---------------- Correlación ----------------
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

    # Compare FULL vs POST in one plot
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
        a2.metric("r FULL", "—" if c_full["r"] is None else f"{c_full['r']:.4f}")
        a3.metric("r² FULL", "—" if c_full["r2"] is None else f"{c_full['r2']:.4f}")
        a4.metric("p FULL", "—" if c_full["p_value"] is None else f"{c_full['p_value']:.4g}")

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("n POST", str(c_post["n"]))
        b2.metric("r POST", "—" if c_post["r"] is None else f"{c_post['r']:.4f}")
        b3.metric("r² POST", "—" if c_post["r2"] is None else f"{c_post['r2']:.4f}")
        b4.metric("p POST", "—" if c_post["p_value"] is None else f"{c_post['p_value']:.4g}")

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

    # global / by_group use a chosen scope
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
        # Option to color by a factor, but clarify that stats are global
        cat_cols = categorical_cols(corr_df)
        color_opts = [None] + cat_cols
        color_default = st.session_state.get("corr_global_color_by")
        if color_default not in color_opts:
            color_default = None

        color_by = st.selectbox(
            "Color (solo visual en modo Global)",
            options=color_opts,
            index=color_opts.index(color_default),
            format_func=lambda v: "— Sin color —" if v is None else str(v),
            key="corr_global_color_by",
        )

        c = correlation_stats(corr_df[x_var], corr_df[y_var], method=method)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("n", str(c["n"]))
        k2.metric("r", "—" if c["r"] is None else f"{c['r']:.4f}")
        k3.metric("r²", "—" if c["r2"] is None else f"{c['r2']:.4f}")
        k4.metric("p-value", "—" if c["p_value"] is None else f"{c['p_value']:.4g}")

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

    # by_group: must show per-group statistics table
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


def tab_3_mean_tests() -> None:
    st.subheader("3) Test de medias (inferencial)")
    df = get_active_df()
    if df is None:
        st.info("Carga un archivo primero.")
        return

    dv_col, factor_a, factor_b, block_col, filters = render_design_controls(df, prefix_key="tab3")
    df_f = apply_filters(df, filters)
    if df_f.empty:
        st.error("Con los filtros actuales no quedan filas.")
        return

    alpha = st.number_input("Alpha", min_value=0.001, max_value=0.2, value=0.05, step=0.005, key="alpha_tab3")

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
        return

    enable_posthoc = st.checkbox("Posthoc (si aplica)", value=True, key="tab3_posthoc")
    stats_df, rep, min_n, enabled, warnings = run_inferential_statistics_df(
        df_f,
        metric=dv_col,
        group_col=factor_a,
        options=StatsOptions(alpha=float(alpha), enable_posthoc=bool(enable_posthoc)),
    )
    st.dataframe(stats_df, use_container_width=True, hide_index=True)


def tab_export() -> None:
    st.subheader("Exportar")
    st.info("Export en modo libre: pendiente (lo agregamos si lo necesitas).")
