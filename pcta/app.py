# ... imports ...
from pcta.app_sections import (
    init_state,
    maybe_parse_main_upload,
    render_sidebar_minimal,
    tab_preview_and_select_variable,
    tab_descriptive_and_charts,
    tab_inferential_compare,
    tab_export,
)

# ... tabs ...
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
