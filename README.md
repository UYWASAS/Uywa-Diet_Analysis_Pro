# PCTA — Poultry Commercial Trial Analyzer

PCTA is a Streamlit app for analyzing commercial poultry trial data (houses with thousands of birds), while remaining compatible with experimental pen trials via a generic **unit abstraction**.

## Quick start

```bash
# from the repo root
pip install -r pcta/requirements.txt
streamlit run pcta/app.py
```

## What it does

Given a trial dataset, PCTA:
- validates inputs (blocking errors + non-blocking warnings)
- computes per-unit KPIs (performance + optional economics)
- summarizes results by treatment (descriptive statistics)
- runs inferential statistics **only when replication exists** (safety rule)
- exports a multi-sheet Excel report

## Input modes

### A) Excel template (recommended)

Excel workbook with sheets:
- **HOUSE_SUMMARY** (required)
- **WEIGH_SAMPLES** (optional)
- **COSTS** (optional; simple or phase)

### B) Single-table file (minimal)

A CSV (or the first sheet of an .xlsx) equivalent to HOUSE_SUMMARY.

## HOUSE_SUMMARY (required) — expected columns

Required identifiers:
- `trial_id`
- `unit_id`
- `treatment`
- `unit_type` (optional; defaults to `house`)

Timing:
- `days` **OR** `start_date` + `end_date` (days will be computed)

Birds:
- `birds_placed`
- `mortality_total`
- `birds_sold` (optional)

Feed:
- `feed_delivered_kg`
- `feed_refusals_kg` (optional; defaults to 0)

Body weight:
- `bw_initial_mean_g` (optional if WEIGH_SAMPLES provided)
- `bw_final_mean_g` (optional if WEIGH_SAMPLES provided)
- `bw_final_sd_g` (optional; can be derived from WEIGH_SAMPLES)

## WEIGH_SAMPLES (optional) — expected columns (long format)

- `trial_id`, `unit_id`, `treatment`
- `day` **or** `date`
- `sample_n`
- `bw_mean_g`
- `bw_sd_g` (optional)

PCTA uses the **earliest** sample as initial BW and the **latest** sample as final BW when those are missing from HOUSE_SUMMARY.

## COSTS (optional)

### Simple COSTS columns

- `trial_id`, `unit_id`, `treatment`
- `diet_cost_per_kg`
- `additive_cost_total` (optional; default 0)
- `chick_cost_per_bird` (optional; default 0)
- `other_variable_costs_total` (optional; default 0)

### Phase COSTS columns (best-effort)

- `trial_id`, `unit_id`, `treatment`, `phase`
- `feed_cost_per_kg`
- `additive_cost_total_phase`

If phase feed intake is not provided, PCTA currently approximates:
- `diet_cost_per_kg` = mean(`feed_cost_per_kg`) across phases
- `additive_cost_total` = sum(`additive_cost_total_phase`) across phases

## Validation rules

### Blocking errors
- missing `trial_id`, `unit_id`, or `treatment`
- `birds_placed <= 0`
- `mortality_total < 0` or `mortality_total > birds_placed`
- `days <= 0`
- `feed_delivered_kg < 0`
- `feed_refusals_kg < 0` or `feed_refusals_kg > feed_delivered_kg`

### Warnings (non-blocking)
- missing `bw_final_sd_g` (uniformity / CV not available)
- missing `final_sample_n` (CI not available)
- `birds_sold` mismatch vs `birds_placed - mortality_total`
- no replication (min n per treatment < 2) → inferential disabled

## KPI calculations (per unit)

- `feed_consumed_kg = feed_delivered_kg - feed_refusals_kg`
- `birds_end = birds_sold` else `birds_placed - mortality_total`
- `mortality_pct = mortality_total / birds_placed * 100`
- `WG = bw_final_mean_g - bw_initial_mean_g`
- `ADG = WG / days`
- `total_liveweight_gain_kg = WG * birds_end / 1e6`
- `FCR = feed_consumed_kg / total_liveweight_gain_kg` (WG<=0 → null + warning)
- `CV_final_pct = bw_final_sd_g / bw_final_mean_g * 100` (if SD exists)

Economics (if costs provided):
- `feed_cost_total = feed_consumed_kg * diet_cost_per_kg`
- `total_cost = feed_cost_total + additive_cost_total + chick_cost_per_bird*birds_placed + other_variable_costs_total`
- `kg_sold_est = (bw_final/1000) * birds_end`
- `cost_per_kg_sold = total_cost / kg_sold_est`
- `cost_per_bird_sold = total_cost / birds_end`
- `cost_per_kg_gain = total_cost / total_liveweight_gain_kg`

## Statistical safety rule (critical)

PCTA **must not** output p-values when there is no replication:
- If **any treatment** has fewer than **2 units**, inferential stats are disabled.
- PCTA will still provide descriptive summaries and an explicit warning.

When replication exists, PCTA:
- checks normality (Shapiro) and equal variances (Levene)
- selects ANOVA / Welch ANOVA / Kruskal-Wallis
- runs posthoc tests when enabled (Tukey; Games-Howell approximation; Dunn approximation)
- reports effect sizes when feasible

## Excel export

The app exports a report with sheets:
- `CLEANED_INPUT`
- `UNIT_KPIS`
- `TREATMENT_SUMMARY`
- `STATS`
- `WARNINGS`

## Tests

```bash
pytest -q
```
