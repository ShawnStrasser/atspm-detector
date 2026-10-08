# 30 — Lane structure with real lane labels (cabinet prints)
Code `research/code/trackA/lp1…lp5_*.py` → `%DC_WORK%/trackA/lp*`. Unlocked signals only (asserted),
folds `folds_v3.csv`, staging window (66 h) only (prints = current wiring).
## 1. Truth tables (lp1, from `function_labels_v3.parquet`, read only)
Rows: high confidence, complete_high/mixed, not unusual, diagram phase = timing phase, vehicle lane.
* `nlanes_labels.parquet` (DeviceId, target, n_lanes, …) — the file note 17 / `lr4_decode.py`
  expected. Phase kept only if every print row agrees on n_lanes_phase and none sits beyond it
  (dropped: 22 disagree, 73 lane > n_lanes, 51 diagram≠timing). **2,141 phases / 441 signals:
  1 lane 1,205, 2 725, 3 199, 4 11, 5 1.**
* `lane_pairs_labels.parquet` — pairs of high rows on a kept phase; same_lane = lane intervals
  [lane_index, +lanes_spanned−1] overlap (spanning = same lane as every lane it spans). 8,699 pairs.
  With cues (both >= 20 actuations): **8,178 pairs / 425 signals / 1,829 phases** (text set: 1,010
  / 47). "multi" = >= 2-lane phases: 6,118 pairs, 50.8 % same; a-s 1,458, s-s 2,863, a-a 555, other 1,242.
## 2. Cues on the print truth (lp2, lp3; univariate, fixed sign; multi / adv-stop / stop-stop)
| cue | all | multi | a-s | s-s | text set (note 17) |
|---|---|---|---|---|---|
| (a) zero-lag sharpness | .344 | .377 | .371 | .467 | .36–.47 (inverted) |
| (b) 2–8 s lead excess, night | .865 | .873 | .871 | .803 | .841 |
| (c) 15-min corr off-peak − peak | .584 | .605 | .606 | .525 | .525 |
| (c′) 15-min corr off-peak | .867 | .866 | .867 | .889 | .834 |
| (d) 1-min high-pass corr off-peak | **.947** | **.934** | .912 | .953 | .895 |
| A3 correlogram cue | .878 | .835 | .780 | .897 | .842 |
Same ordering as note 17, every useful cue higher; (a) still inverted, (c) useless, (d) best.
## 3. Supervised pair model (OOF, six signal-grouped folds)
| model | all | multi | a-s | s-s | a-a |
|---|---|---|---|---|---|
| note-17 compact logistic, trained on the text pairs, applied | .934 | .917 | .895 | .921 | .727 |
| logistic, all cues, print-trained | .956 | .943 | .905 | .952 | .906 |
| **GBM, all cues + A3 + volumes / phase size (best)** | **.965** | **.955** | .919 | .966 | .920 |
| + predicted-function pair type (67 % covered; same subset without it .962 / .953) | .971 | .963 | .933 | .966 | .844 |
Controls: + noise column .9556; labels shuffled in fold .54. GBM deterministic (seed spread 0). On
the RL/CL/LL text pairs (other folds) the print model scores **.910** vs note 17's OOF **.883**
(s-s .917 vs .855, a-s .902 vs .918). Materially better: 1−AUC ~halves vs note 17 on prints.
## 4. n_lanes per phase vs the print (lp4; timing-phase grouping, OOF, th picked on other folds)
Estimator: largest set of pairwise different-lane detectors (P(same) < .2–.3; spanning ones drop
out). Exact / ±1 (exact on >= 2-lane phases):
| detectors on the phase | n | always 1 | max(#A,#P,#C) predicted | note-17 model | **print model** |
|---|---|---|---|---|---|
| every actuating channel | 2,012 | .543 / .896 | .697 / .944 | .699 / .940 | **.809 / .982** (.698) |
| predicted A/P/C/YR only | 1,277 | .553 / .882 | .697 / .944 | .787 / .969 | **.865 / .990** (.760) |
| print members (oracle) | 1,996 | .546 / .897 | .807 / .981 | .817 / .984 | .875 / .989 (.743) |
Clustering (note 17's method) is worse: .743 / .825 / .850 exact. Note 17's full pipeline (`lr4_decode.py`
re-run, reads the new table) scores **.650** exact (2,461 phase-periods). Errors ~all ±1, both ways.
## 5. Does lane structure help FUNCTION? (lp5) — No
Note-25 b7 OOF (Sept full window); truth = v3 print_high label (being revalidated: a screen only).
complete_high 1,246 rows / 75 signals (both tiers 4,385 / 270). Seed means (3), 5-class / core-four:
* b7 argmax **.8911 / .9108**; per-lane constraint decode th .5/.6/.7 **−1.5 to −2.3 pt** (.8684–.8756).
* Stacker on b7 probs only .9176 / .9363; + 14 lane features (same-lane partners, phase n_lanes,
  lanes spanned, upstream score, lane-mates' class mass, best hp_off) **.9168 / .9348** (−0.1);
  shuffled lane features .9162 / .9317; both tiers .9056 → .9046. **No lane effect.** (The stacker's
  own +2.7 pt is recalibration of v2-trained b7 onto print labels — the note-28 retrain's job.)
## Verdict
Truth 8× the text set. Pair model materially better (multi .917 → .955; text .883 → .910); n_lanes
.81 / .98 ±1 on every channel (.87 / .99 on predicted PM detectors), up from .65–.70. Function: decode
costs, lane features add 0 → **dropped for function** (third time: 14, 17, 30). Keep pair model +
n_lanes as a review / data-quality aid only.
