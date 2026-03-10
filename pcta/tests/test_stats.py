from __future__ import annotations

import pandas as pd

from pcta.core.calculations import compute_all_units
from pcta.core.schemas import TrialUnitInput, UnitType
from pcta.core.stats import StatsOptions, run_inferential_statistics, run_inferential_statistics_df


def _unit(unit_id: str, treatment: str, bw_final: float) -> TrialUnitInput:
    return TrialUnitInput(
        trial_id="T1",
        unit_type=UnitType.house,
        unit_id=unit_id,
        treatment=treatment,
        days=35,
        birds_placed=1000,
        mortality_total=10,
        birds_sold=None,
        feed_delivered_kg=3000.0,
        feed_refusals_kg=50.0,
        bw_initial_mean_g=42.0,
        bw_final_mean_g=bw_final,
        bw_final_sd_g=None,
        final_sample_n=None,
        diet_cost_per_kg=None,
        additive_cost_total=0.0,
        chick_cost_per_bird=0.0,
        other_variable_costs_total=0.0,
    )


def test_stats_disabled_without_replication() -> None:
    units = [_unit("H1", "A", 2500.0), _unit("H2", "B", 2600.0)]
    kpis, _ = compute_all_units(units)

    df, rep, min_n, enabled, warnings = run_inferential_statistics(
        kpis,
        metrics=["fcr", "wg_g_per_bird"],
        options=StatsOptions(alpha=0.05, enable_posthoc=True),
    )

    assert enabled is False
    assert min_n == 1
    assert "inferential_disabled_no_replication" in {w.code.value for w in warnings}
    assert df["p_value"].isna().all() or (df["p_value"].astype(object).isna().all())


def test_stats_returns_pvalues_with_replication() -> None:
    # Two replicates per treatment
    units = [
        _unit("H1", "A", 2500.0),
        _unit("H2", "A", 2490.0),
        _unit("H3", "B", 2600.0),
        _unit("H4", "B", 2610.0),
    ]
    kpis, _ = compute_all_units(units)

    stats_df, rep, min_n, enabled, warnings = run_inferential_statistics(
        kpis,
        metrics=["wg_g_per_bird"],
        options=StatsOptions(alpha=0.05, enable_posthoc=False),
    )

    assert enabled is True
    assert min_n == 2
    assert stats_df.shape[0] == 1
    assert pd.notna(stats_df.loc[0, "p_value"])


def test_stats_group_col_is_configurable_in_unit_kpis() -> None:
    # Same data but grouping by a different column name should work.
    units = [
        _unit("H1", "A", 2500.0),
        _unit("H2", "A", 2490.0),
        _unit("H3", "B", 2600.0),
        _unit("H4", "B", 2610.0),
    ]
    kpis, _ = compute_all_units(units)

    # rename treatment -> group_x
    df_kpi = pd.DataFrame([m.model_dump() for m in kpis]).rename(columns={"treatment": "group_x"})
    records = df_kpi.to_dict(orient="records")

    stats_df, rep, min_n, enabled, warnings = run_inferential_statistics(
        records,
        metrics=["wg_g_per_bird"],
        options=StatsOptions(alpha=0.05, enable_posthoc=False),
        group_col="group_x",
    )

    assert enabled is True
    assert min_n == 2
    assert set(rep.keys()) == {"A", "B"}
    assert pd.notna(stats_df.loc[0, "p_value"])
    assert stats_df.loc[0, "group_col"] == "group_x"


def test_stats_df_api_disables_when_no_replication() -> None:
    df_raw = pd.DataFrame(
        {
            "tratamien": ["SZ_40", "SZ_80"],  # 1 por grupo
            "peso_final": [2572.0, 2636.0],
        }
    )
    stats_df, rep, min_n, enabled, warnings = run_inferential_statistics_df(
        df_raw,
        metric="peso_final",
        group_col="tratamien",
        options=StatsOptions(alpha=0.05, enable_posthoc=True),
    )

    assert enabled is False
    assert min_n == 1
    assert stats_df.shape[0] == 1
    assert pd.isna(stats_df.loc[0, "p_value"])
    assert stats_df.loc[0, "disabled_reason"]
    assert "inferential_disabled_no_replication" in {w.code.value for w in warnings}


def test_stats_df_api_returns_pvalues_with_replication() -> None:
    df_raw = pd.DataFrame(
        {
            "tratamien": ["SZ_40", "SZ_40", "SZ_80", "SZ_80"],
            "peso_final": [2572.0, 2574.0, 2636.0, 2630.0],
        }
    )
    stats_df, rep, min_n, enabled, warnings = run_inferential_statistics_df(
        df_raw,
        metric="peso_final",
        group_col="tratamien",
        options=StatsOptions(alpha=0.05, enable_posthoc=False),
    )

    assert enabled is True
    assert min_n == 2
    assert stats_df.shape[0] == 1
    assert pd.notna(stats_df.loc[0, "p_value"])
    assert stats_df.loc[0, "group_col"] == "tratamien"
