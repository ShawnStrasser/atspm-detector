# 72 — Label review sheet v1 for the user (likely-wrong labels only) (2026-10-01)
Code `research/code/evaluation/review72.py` (built on review64's `build_items`, `--arm fj`, fj on all six folds asserted).
Output `review/function_label_review_v1.xlsx` + `review/function_label_review_v1_charts/` (38 PNG). Work files
`%DC_WORK%/cand64/review72_dryrun.json`, `review72_rows.parquet`, chart data `review_data72/`. locked_v2 asserted absent.
CPU, DuckDB 4 threads; 52 s end to end. Nothing trained or re-scored; charts drawn from saved data.
## Selection (Sept 2026 log, samples >= 30 min, candidate = trees + fj, decoded)
Start: review64 items (detector wrong after review64's exclusions in >= half its samples): 645 (596 function, 49 phase).
Dropped: not confident enough 298; definitional pending the user's open question 276 (note-70 B1-B5 rules on every wrong
window, not first-match, share >= .5: Other subtypes acting like a PM class, radar long Advance called Other, YR/Count
twins, Mid vs Advance / Presence, stacked extra lane); unhealthy 6 (A2 / health bad share >= .5); fewer than 3 samples 5;
user already ruled 1. Review64's own exclusions (misconfigured, field issue, switch / additional call, overlap phase,
known print-phase mislabels, YR/Count twin swaps) apply first.
Kept (59 items): config-only label contradicted (note-70 A3, or A4 with how-sure >= .7) 35 function; confident
contradiction (mean p on own answer over wrong samples >= .9) 16 function + 8 phase; collapsed by (signal, kind, label,
model answer) -> **38 rows on 30 signals** (20 config-only, 18 confident; 31 function, 7 phase; 15 multi-detector;
31 with a print link; tech loop 14, radar 10, unknown 14; label source config 20, print_high 15, whole-int Other 2, none 1).
Confidence of rows: >= .95 14, >= .9 30, >= .8 35, all >= .79.
Widening confident contradictions to p >= .8 would add 29 items.
Top patterns: Advance -> Presence 9, Advance -> Count 5, Presence -> Other 4 (radar, print_high), YR -> Count 3 (radar),
Presence -> Advance 3; phase rows: 2C009 det 19 / det 5 (radar count, print_high), 10073 det 61, 04014 det 32,
14076 det 33 and 14037 det 22, 23 (P2 -> P6), 14016 det 36.
## Sheet
One-line question at the top (L / M / ?, optional correct class or phase); 9 columns: #, signal, detector(s), label
(with phase), model says, how sure (max over the row's detectors), chart, print, answer. Signals ordered by their most
confident row. Chart per row (lead detector; group mates dashed): top = 15-min counts vs the labelled phase's detectors
(legend names their labels) or, for phase rows, share of actuations starting in each phase's green; middle = % of cycles
ON and actuation starts vs seconds from begin of green (label phase; + model phase on phase rows); bottom = 10 busy
minutes with green / yellow / red and an ON-length histogram (log bins).
## Spot-check (5 rows: 3, 6, 13, 20, 35)
Detector numbers, labels, phases and model answers match `function_rows` / the v3s label table (e.g. 2B054 det 9 P4 and
det 23 P8 Advance, model Presence 14 / 14 at p .95 / .96; 2B060 det 8 Advance -> YR 13 / 14); chart files exist and
match the row; print links resolve to the right signal's PDF (05999 has no print on file).
## For the orchestrator
* 38 rows, below the ~120 target: the brief's rules (bucket a + p >= .9, definitional excluded) are the limit. Second
  batch (v1b) not built — there is nothing left under these rules; p >= .8 would give ~29 more items if wanted.
* Answers feed the label table (v3s -> next) before the next retrain; M answers on config-only rows are relabels.
