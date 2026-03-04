# Sample data

Place sample input files in this folder (not included by default).

Recommended:
- `sample_template.xlsx` using the Excel template format:
  - `HOUSE_SUMMARY` (required)
  - `WEIGH_SAMPLES` (optional)
  - `COSTS` (optional)

Or:
- `sample_house_summary.csv` as a minimal single-table input equivalent to `HOUSE_SUMMARY`.

## Minimal HOUSE_SUMMARY example (CSV)

```csv
trial_id,unit_type,unit_id,treatment,days,birds_placed,mortality_total,birds_sold,feed_delivered_kg,feed_refusals_kg,bw_initial_mean_g,bw_final_mean_g,bw_final_sd_g
T1,house,H1,A,35,20000,200,,65000,500,42,2500,250
T1,house,H2,A,35,20000,180,,64500,450,42,2480,240
T1,house,H3,B,35,20000,240,,66000,520,42,2575,260
T1,house,H4,B,35,20000,210,,65800,480,42,2590,255
```

Notes:
- `birds_sold` can be left blank; PCTA will use `birds_placed - mortality_total`.
- If `bw_final_sd_g` is blank, uniformity (CV) will not be available.
- If `WEIGH_SAMPLES` are provided, PCTA can derive initial and final BW if missing in HOUSE_SUMMARY.
