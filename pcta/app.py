"""
Entry point (pequeño) para Streamlit.

IMPORTANTE:
- Este archivo DEBE tener `import streamlit as st` antes de usar `st.tabs`.
- Además incluye el bootstrap de sys.path para Streamlit Cloud.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# --- Bootstrap para que `import pcta.*` funcione en Streamlit Cloud ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pcta.auth import get_current_user, login_ui  # noqa: E402
from pcta.app_sections import (  # noqa: E402
    init_state,
    maybe_parse_main_upload,
    render_sidebar_minimal,
    tab_1_select_variable_and_run,
    tab_2_results_for_selected_variable,
    tab_3_mean_tests,
    tab_export,
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
        </style>
        """,
        unsafe_allow_html=True,
    )


_apply_global_css()

# Login gate
if not st.session_state.get("logged_in", False):
    login_ui()

user = get_current_user()
if not user:
    st.error("El usuario no está autenticado.")
    st.stop()

# Init state + sidebar + parse
init_state()
sidebar = render_sidebar_minimal(user=user)
maybe_parse_main_upload(sidebar.uploaded_main)

# Main
st.title("PCTA — Analizador de Ensayos Comerciales Avícolas")
st.markdown(
    "<div class='pcta-muted'>Flujo: Selección de variable → Resultados → Test de medias → Exportar</div>",
    unsafe_allow_html=True,
)

tabs = st.tabs(["1) Selección", "2) Resultados", "3) Test de medias", "4) Exportar"])

with tabs[0]:
    tab_1_select_variable_and_run()
with tabs[1]:
    tab_2_results_for_selected_variable()
with tabs[2]:
    tab_3_mean_tests()
with tabs[3]:
    tab_export()
