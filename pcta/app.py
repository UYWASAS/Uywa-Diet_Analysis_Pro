from pcta.app_sections import (  # noqa: E402
    init_state,
    maybe_parse_main_upload,
    render_sidebar_minimal,
    tab_1_select_variable_and_run,
    tab_2_results_for_selected_variable,
    tab_3_mean_tests,
    tab_export,
)

# ...

tabs = st.tabs(["1) Selección", "2) Resultados", "3) Test de medias", "4) Exportar"])

with tabs[0]:
    tab_1_select_variable_and_run()
with tabs[1]:
    tab_2_results_for_selected_variable()
with tabs[2]:
    tab_3_mean_tests()
with tabs[3]:
    tab_export()
