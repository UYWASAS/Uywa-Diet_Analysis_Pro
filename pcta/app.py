"""
Entry point (pequeño) para Streamlit.

Este archivo queda minimalista: login, estilo, sidebar, tabs.
El resto vive en pcta/app_sections.py para que puedas editar por bloques.

IMPORTANTE (Streamlit Cloud / ejecución como script):
- Streamlit ejecuta `pcta/app.py` como script y a veces NO agrega la raíz del repo al sys.path.
- Este archivo agrega un "bootstrap" para que `import pcta.*` funcione siempre.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# --- Bootstrap de imports (evita: ModuleNotFoundError: No module named 'pcta') ---
REPO_ROOT = Path(__file__).resolve().parents[1]  # raíz del repo (carpeta padre de pcta/)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pcta.auth import get_current_user, login_ui  # noqa: E402
from pcta.app_sections import (  # noqa: E402
    init_state,
    maybe_parse_main_upload,
    render_sidebar_minimal,
    tab_descriptive_and_charts,
    tab_export,
    tab_inferential_compare,
    tab_preview_and_select_variable,
)


def _apply_global_css() -> None:
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
        section[data-testid="stSidebar"] small { color: #fff !important; }
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
        .pcta-kpi .value{ font-size: 1.1rem; color:#0f172a; font-weight:900; letter-spacing: -0.01em; }
        </style>
        """,
        unsafe_allow_html=True,
    )


_apply_global_css()

# Login
if not st.session_state.get("logged_in", False):
    login_ui()

user = get_current_user()
if not user:
    st.error("El usuario no está autenticado.")
    st.stop()

init_state()

# Sidebar minimal (logout + uploader)
sidebar = render_sidebar_minimal(user=user)

# Parse principal upload (si existe)
maybe_parse_main_upload(sidebar.uploaded_main)

# Header
st.title("PCTA — Analizador de Ensayos Comerciales Avícolas")
st.markdown(
    "<div class='pcta-muted'>Flujo por variable: selecciona variable → descriptiva+gráficos → test de medias.</div>",
    unsafe_allow_html=True,
)

tabs = st.tabs(
    [
        "1) Vista previa",
        "2) Descriptiva + gráficos",
        "3) Test de medias",
        "4) Exportar",
    ]
)

with tabs[0]:
    tab_preview_and_select_variable()

with tabs[1]:
    tab_descriptive_and_charts()

with tabs[2]:
    tab_inferential_compare()

with tabs[3]:
    tab_export()
