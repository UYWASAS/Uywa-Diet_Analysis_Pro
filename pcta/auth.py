"""
Simple multi-user auth (in-memory) for Streamlit.

NOTE:
- This is NOT production-grade authentication.
- Passwords are stored in plain text for simplicity (like the example app).
- For production, use hashing + a real user store + proper session security.

App contract:
- login() sets:
    st.session_state["logged_in"] = True
    st.session_state["user"] = user_dict
"""

from __future__ import annotations

from typing import Dict, Optional

import streamlit as st

# Example users database (edit to match your company users)
# username key MUST be lowercase.
USERS_DB: Dict[str, Dict[str, object]] = {
    "admin": {"username": "admin", "name": "Admin", "password": "admin123", "premium": True, "role": "admin"},
    "analyst": {"username": "analyst", "name": "Analyst", "password": "analyst123", "premium": True, "role": "analyst"},
    "viewer": {"username": "viewer", "name": "Viewer", "password": "viewer123", "premium": False, "role": "viewer"},
}


def get_current_user() -> Optional[Dict[str, object]]:
    if not st.session_state.get("logged_in", False):
        return None
    u = st.session_state.get("user")
    return u if isinstance(u, dict) else None


def logout_button(*, key: str = "logout_btn") -> None:
    if st.button("Cerrar sesión", key=key, use_container_width=True):
        st.session_state["logged_in"] = False
        st.session_state["user"] = None
        st.rerun()


def login_ui() -> None:
    """
    Render login UI and stop execution if not authenticated.
    """
    st.title("Iniciar sesión")
    st.caption("Acceso restringido — ingresa tus credenciales.")

    username = st.text_input("Usuario", key="login_username")
    password = st.text_input("Contraseña", type="password", key="login_password")

    if st.button("Entrar", type="primary"):
        u = USERS_DB.get((username or "").strip().lower())
        if u and u.get("password") == password:
            st.session_state["logged_in"] = True
            st.session_state["user"] = u
            st.success(f"Bienvenido, {u.get('name', u.get('username'))}!")
            st.rerun()
        else:
            st.error("Usuario o contraseña incorrectos.")

    if not st.session_state.get("logged_in", False):
        st.stop()
