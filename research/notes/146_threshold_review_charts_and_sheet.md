# 146 Threshold review: large charts + user threshold sheet (2026-10-15)

User asked (after 1.1.0 build, note 144) to review thresholds by hand: bigger charts with explicit old vs new
limits, and a sheet with all limits + % flagged + columns to set his own. Nothing in src/ changed.

Data: 1.1.0 package output on h24_b (dc_work/s144/out144_h24b.parquet: measures, limits used per detector, per-check
scores s8_*) merged with the note-143 simulation (dc_work/s143/sim143.parquet: 1.0.1 limits lim_cur_* and scores cur_*).
Limit per type = median over that type's detectors (model-unsure detectors carry looser limits).

Charts (research/code/health146/h146_charts.py -> dc_work/s146/charts, copied to review/thresholds_v8_charts):
111 = 8 measures x function x lanes (types with >= 5 values). Histogram (log count), grey = 1.0.1 suspect/bad,
red = 1.1.0 suspect/bad; legend and title carry the values and % flagged (score >= .35). Artifact
MZ1SeZHeepwiYKAuqnrM4L v3 = tables (note 145) + these charts with old/new limits as text above each.

Sheet review/thresholds_review_v8.xlsx (h146_sheet.py): one row per check x type: n, measure, flagged above/below,
1.0.1 suspect / bad / % suspect / % bad, 1.1.0 same, YOUR suspect / bad / comment (yellow), chart link.
% suspect / % bad = this check's own score (suspect >= .35, bad >= 1; busy at night bad >= .8 = level "bad");
the final status can still be raised by two findings / 4-h persistence.
Bad rules shown: stuck max(60 min, 2x) (1.0.1 60 min); silent x100/30 (1.0.1 100); drops and night ratio /3; too
fast, too many, erratic 2x; busy at night no number (needs mates). Too fast 1.0.1 column = note-141 cut (1.0.1 had
the 5-in-a-row rule).
Caveat: same limit can give a different % (e.g. night Presence 1 lane .58 -> .38 %) because 1.1.0's slimmer
stage 1 changes which detectors are phase-mate yardsticks (note 144).
Next: user fills the sheet -> limits into weights/health/health_limits.json -> re-run note-144 parity / check ->
v1.1.0 release.
