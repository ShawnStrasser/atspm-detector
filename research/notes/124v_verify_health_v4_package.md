# 124v Adversarial check of package v7 after notes 120 + 124 (`dc_work/final_v7_prod`, 7.0.0) (2026-10-07)
Nothing in the package changed. CPU, 4 threads. Work `%DC_WORK%/s124v`, scripts `research/code/final124v/`.
Locked_v2 asserted absent everywhere. 10 NEW signals (seed 20261124, 2 per channel quintile, 5-42 channels; not in
115 / 115v / 115dv / bench71 / the 24 health-review signals), Sun 2026-09-20: 30 min (17:00), 3 h (10:00), 24 h (06:00)
= 30 cases, 604 detectors, 3,729 candidate rows, 155 phase rows.

| item | result | evidence |
|---|---|---|
| non-health columns vs backup `final_v7_prod_bak124` | PASS | 29 columns + candidate + phase tables: 0 differing cells (floats ==); health moved on 13 / 604 (`cmp_old_new.json`) |
| shipped wheel (fresh venv, pandas 3.0.6) vs src (pandas 2.3.3) | PASS | all 40 columns, 0 cells (`cmp_new_whl.json`); 69 wheel files = src = installed (sha256) |
| health = research scorer (resolved_v4d), 5 review signals x 8 windows incl. 4 x 30 min (not in note 124) | PASS | research stage 1 + OOF refs: 1,011 / 1,011 statuses; findings, watch, config, rules, severity, reason text all 0 diffs (`health5_research.json`). 23 `tod_z` values differ; diagnostic only, not used in any decision or text. Shipped refs: 1,010 / 1,011 (be117825 h24_a det 48 = note 124's listed night-model case) |
| predict() -> health wiring, same 5 signals x 5 windows | PASS | events handed to health = the raw log (same multiset, same sorted detector arrays); predict health columns = assess re-run, 0 diffs on 634 detectors (`wiring.json`). Info: with the package's own phase / function, 624 / 634 statuses = research |
| health text consistency | PASS | "det N: Pp Function" matches Detector / phase_pred / function_pred on 604 / 604; every bad period inside its window |
| channel + phase bijection, 7 signals x 3 lengths x 2 seeds (21 runs), health text translated back | FAIL (narrow, pre-existing, not a leak) | 20 / 21 identical incl. health (max float diff 2e-16). v08 (seed 11): dets 23 / 26 are EXACT twins (1,020 / 1,020 identical events); their Count / Presence answers swap (24 h), probabilities / lanes shift (30 min, 3 h). Cause: 5e-8 network float noise flips the stacker's within-phase rank (stacker.py ctx_X `rk`). Backup bak124 gives the same. Nothing else differs (`perm/`) |
| robustness battery 115v / 115dv on the wheel (v08 + new v05) | PASS | empty df / parquet / window, 1 detector, no begin-green, no 7-11, no 43/44, duplicates, >64, shuffled, column spellings (df / csv / parquet), string ts, tz (8 variants), bad rows, apostrophes (paths, temp dir, DeviceId), batching / chunking / glob, disallowed codes + 83-88 added, channel 0, stuck ON (never-off, one OFF), 1 / 5 min, 90-min hole, start > end, one event: all identical / sensible |
| df vs path with start / end | PASS | robust_d flagged 9 columns under pandas 3 only: test artefact (`astype(str)` keeps NaN, NaN != NaN); `DataFrame.equals` = True |
| 7 days, 1 thread | PASS | n08 (38 ch, 2.0 M events) 12.6 s, peak 862 MB; v09 (42 ch, 2.2 M events) 16.2 s, peak 981 MB (< 1 GB, little headroom) |
| banned imports | PASS | fresh venv, torch / lightgbm / scipy / sklearn / pyarrow / matplotlib blocked, all warnings = errors: sample + 24 h predict run, charts module imports, health_chart -> clear ImportError. Dev venv: pyarrow only via `import pandas` itself (bak124 the same) |
| fresh-venv install + check.py | PASS | pip install of the shipped wheel (pandas 3.0.6, numpy 2.5.3, duckdb 1.5.6, ort 1.30): check 7 / 7; src check (dev venv) 7 / 7 |
| profile leftovers | PASS | `profile` / DC_FAST_PROFILE only in the deprecation shim, its check and docs; no fast.json, decode_trees, nonet / single, siba_members, cProfile / timing code; no stray wheel files |

**Twin case:** the two channels cannot be told apart by behaviour. The one-ATSPM-per-lane rule makes the model give them
different classes, so any choice between them is arbitrary. Making it permutation-proof would mean giving both
channels the same class, which breaks that rule. It also changes function answers, so it was not fixed.
Observations (not defects): `health_v4_stats.colour_stats` opens its own DuckDB (1 thread, default memory limit, tiny
table); `charts.health_chart(path=...)` switches matplotlib to Agg for the whole process; model_card names
`dc_work/official/locked_v2.csv` (a relative mention only).
