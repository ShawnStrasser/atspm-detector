# 42 — Lane output: n_lanes per phase, lane(s) per detector (research code, not in `model/`)
Code `research/code/lanes/` (`lane_output.py` = `lanes(events, predictions) -> (phase table, detector table)`,
numpy/pandas only; `ln1…ln4`) → `%DC_WORK%/lanes/`, model `%DC_WORK%/lanes/model/` (3 LightGBM text models + `lane_model.json`).
Runs after phase/function; hi-res log only; predicted phase = grouping key only. Locked_v2 asserted absent; released 71 in.
## Definition
* Lanes counted per phase in TOTAL (all approaches). **Numbering: lane 1 = busiest lane of the phase** (largest actuation
  rate among its single-lane A/P/C/YR detectors), then 2, 3 … — not geometry. Spanning detector lists all its lanes ("1,2").
* Detector table: `lanes`, `n_lanes_spanned`, `lane_conf`, `lane_note` (no lane when < 10 actuations, Bike, or no phase).
  Phase table: `n_lanes`, `n_lanes_conf`, `lane_volumes_per_hour`.
* Pair model: note-30 cues re-implemented in numpy (identical to the lr2 SQL on a test signal, max diff 4e-16) + volumes,
  window hours, predicted-function pair type; LightGBM binary, 300 trees, 3 seeds; numpy booster parity 0.0.
* Decode per predicted phase, L = 1…4: max Σ log P(same/diff) + log P(span | predicted function) − 3·L; ≤ 1 single-lane
  A/P/C/YR per lane (penalty 4), every lane needs a single-lane A/P/C/YR anchor; Mid/Other join lanes, may span; any lane
  subset allowed. Exhaustive over anchor partitions (≤ 9) + coordinate ascent. (λ, β) grid picked per fold on the other five:
  (3, 0) on all six folds (first grid's corner (2, .5) → extended; surface flat: n_lanes exact .836–.847 over λ 1–4, β 0–.5).
## Truth / data
Print labels (`function_labels_v3`, lp1 rules): 2,303 phases / 476 signals (1 lane 1,295, 2 779, 3 217, 4 11, 5 1); 9,373
print-high vehicle pairs, 63 % same lane. Sept-2026 windows m30 ×4, h6 ×2, h24 ×2, full66; predicted phase / function = frame v6 OOF
(first.all.wi, 3 seeds). 72,702 labelled pair-windows. Folds folds_v4 (phase OOF inherits the 22-signal mismatch, note 37).
## Pair model, OOF AUC (all / ≥ 2-lane phases)
| window | full model | − pair type | cues only | + noise | shuffled |
|---|---|---|---|---|---|
| 30 min | .949 / **.935** | .893 / .875 | .877 / .865 | .935 | .54 |
| 6 h | .969 / **.958** | .949 / .936 | .947 / .936 | .958 | .51 |
| 24 h | .975 / **.966** | .964 / .954 | .964 / .955 | .966 | .54 |
| full 66 h | .976 / **.968** | .966 / .957 | .967 / .958 | .968 | .56 |
Seed spread ≤ .0006. Note 30 (full, no pair type, with A3 cue): .965 / .955. Predicted-function pair type is the big gain at 30 min.
## End to end, OOF (predicted phase + function; truth phase P# vs predicted phase #)
| | 30 min | 6 h | 24 h | full |
|---|---|---|---|---|
| truth phases / predicted phase found | 9,212 / .866 | 4,606 / .987 | 4,606 / .992 | 2,303 / .994 |
| **n_lanes exact / ±1** | **.801 / .987** | **.874 / .992** | **.880 / .993** | **.883 / .994** |
| exact on ≥ 2-lane phases | .652 | .762 | .774 | .782 |
| baseline always 1 / max(#A,#P,#C) predicted | .518 / .757 | .560 / .821 | .562 / .818 | .563 / .812 |
| truth pairs, both assigned (coverage) | .850 | .982 | .991 | .995 |
| same-lane precision / recall (all) | .933 / .887 | .953 / .923 | .958 / .938 | .959 / .938 |
| pairwise accuracy all / ≥ 2-lane / ≥ 2-lane & phase right | .888 / .872 / .883 | .922 / .907 / .916 | .934 / .921 / .929 | .935 / .922 / .929 |
| spanning detectors: precision / recall | .796 / .604 | .836 / .681 | .859 / .709 | .860 / .705 |
| detector lane set exact after best relabel (all / ≥ 2-lane) | .884 / .837 | .919 / .879 | .927 / .889 | .929 / .890 |
Full-window confusion true→pred: 1 lane 1,237 right / 50 over; 2 lanes 637 right / 101 → 1 / 34 → 3–4; 3 lanes 138 right / 66 under
/ 13 over; 4 lanes 8 / 11. Errors ~all ±1. Detector coverage .858 at 30 min (< 10 actuations → no lane), ≥ .987 from 6 h.
## Confidence (all windows, ≥ 2-lane phases)
lane_conf bin → lane set exact: < .5 .40 · .5–.7 .60 · .7–.8 .70 · .8–.9 .80 · .9–.95 .87 · ≥ .95 .93 (monotone, ~calibrated).
n_lanes_conf → exact: .5–.7 .53 · .7–.8 .59 · .8–.9 .71 · .9–.95 .72 · ≥ .95 .88 (over-confident above .8; use as a ranking).
## Package check (`ln4_final.py smoke`, 20 signals, raw Sept-2026 events, lightgbm/torch/sklearn/scipy blocked)
Runs; 0.12 / 0.15 / 0.20 / 0.34 s per signal at 30 min / 6 h / 24 h / 66 h. Cues from raw events vs the interval cache:
equal up to one window-edge actuation (ONs are the 82s followed by an 81, as in predict.py's intervals).
## Caveats / open
* Truth = print-complete signals, Sept 2026 only. Numbering by volume is a convention: across windows lanes of near-equal volume can swap numbers; only
  "which detectors share a lane" and n_lanes are stable claims. Not wired into `model/`; nothing shipped.
