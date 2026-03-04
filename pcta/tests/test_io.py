from __future__ import annotations

import pandas as pd

from pcta.core.io import _derive_bw_from_weigh_samples


def test_derive_bw_and_final_sample_n_from_weigh_samples() -> None:
    # HOUSE_SUMMARY missing bw fields and final_sample_n (or with NAs)
    hs = pd.DataFrame(
        [
            {
                "trial_id": "T1",
                "unit_id": "H1",
                "treatment": "A",
                "unit_type": "house",
                "days": 35,
                "birds_placed": 1000,
                "mortality_total": 10,
                "feed_delivered_kg": 3000.0,
                "feed_refusals_kg": 0.0,
                "bw_initial_mean_g": None,
                "bw_final_mean_g": None,
                "bw_final_sd_g": None,
                "final_sample_n": None,
            }
        ]
    )

    # WEIGH_SAMPLES has day-based timeline (earliest -> initial, latest -> final)
    ws = pd.DataFrame(
        [
            {
                "trial_id": "T1",
                "unit_id": "H1",
                "treatment": "A",
                "day": 1,
                "sample_n": 50,
                "bw_mean_g": 42.0,
                "bw_sd_g": 2.0,
            },
            {
                "trial_id": "T1",
                "unit_id": "H1",
                "treatment": "A",
                "day": 35,
                "sample_n": 100,
                "bw_mean_g": 2500.0,
                "bw_sd_g": 250.0,
            },
        ]
    )

    merged, warnings = _derive_bw_from_weigh_samples(hs, ws)

    assert float(merged.loc[0, "bw_initial_mean_g"]) == 42.0
    assert float(merged.loc[0, "bw_final_mean_g"]) == 2500.0
    assert float(merged.loc[0, "bw_final_sd_g"]) == 250.0
    assert int(merged.loc[0, "final_sample_n"]) == 100

    # Should emit warnings for derived fields
    assert len(warnings) >= 1
    assert any(w.code.value == "missing_bw_derived_from_weigh_samples" for w in warnings)
