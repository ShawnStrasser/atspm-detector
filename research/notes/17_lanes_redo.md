# 17 — Lane structure, redone with the per-lane rule
A3 (note 14) called two Presence zones on a phase a label violation; the rule is per LANE (1–3
lanes; ≤ 1 Advance / Presence / Count per lane; Yellow_Red free; rest Other). Unlocked signals,
`?` dropped, phase = timing target. Code `research/code/trackA/lr*.py` (+ A3's `a3_lanes.py`).

## 1. The labels obey the rule — A3's conclusion does not stand
Lower bound on lanes L_min = max(#A, #P, #C) per (signal, phase), 384 signals / 2,076 phases
with an actuating A/P/C detector: **1 lane 55.2 %, 2 34.0 %, 3 9.4 %, > 3 1.4 % (29 phases)**
(all labelled: 1.7 %; the old ≥ 2-of-a-class reading flags 44.8 %). The 29 are mostly Advance-
driven (15; two setbacks per lane?), listed in `lr1_gt3_phases.csv` for review. Lane text (RL/CL/LL,
50 signals): labels break ≤ 1-per-lane on **4 of 202 true lanes (2.0 %)**.
A3's "25 %" treated a sensor unit ("Rad A") as a lane; a unit watches an approach. Per unit
the old reading gives 48.5 % here (42.9 % on all 115 unit-text signals); the correct reading
(≤ 3 of each per unit) gives **0.4 %** (5.9 % if capped at the lanes the text names).

## 2. Lane grouping: from .736 to ~.88 AUC
Truth pairs: both lane-tagged, same timing phase, actuating, full window; same lane = same
token and same unit (ambiguous two-unit cases dropped). 1,010 pairs, 47 signals, 26 % same,
both periods. Univariate AUC (fixed sign, no fit), all / stop-stop / adv-stop:

| cue | AUC | s-s | a-s |
|---|---|---|---|
| (a) zero-lag co-location (excess, sharpness) | **.36–.47** | .39–.51 | .44–.55 |
| (b) 2–8 s lead, excess per ON, night hours | .841 | .817 | .925 |
| (c) 15-min count corr off-peak minus peak | .525 | .623 | .525 |
| (c′) 15-min count corr off-peak alone | .834 | .786 | .899 |
| (d) new: 1-min counts minus 15-min mean, corr off-peak | **.895** (93.6 % cov.) | .884 | .944 |

(a) is **inverted**: parallel lanes discharge together at green start; a lane's count and
presence zones do not switch on at the same instant. (c) as a divergence fails; the off-peak
level carries it. A3's cue (correlogram excess) scores .842 here (84 % cov.) vs .736 on A3's
pair set: most of that gap was a cleaner truth (unit-aware, timing phase). OOF logistic, six signal-grouped folds: (a) .611,
(b) .832, (c) .816, (d) .859, all .868 (GBM .873), compact + predicted-function pair type
**.875**, + A3 cue **.883** (stop-stop .855, adv-stop .918).

## 3. Constrained decode: still does not pay
P(same lane) from the compact model, out-of-fold on every unlocked signal (21,603 pairs),
average linkage, A3's exact assignment; full window only (the cues need a day). Argmax:
folds 1–5 **.8138** 5-class / **.8518** A/P/C, fold 0 .8428 / .8862.

| decode (threshold picked on folds 1–5) | f1–5 5-cl | f1–5 APC | f0 5-cl | f0 APC | Other rec. |
|---|---|---|---|---|---|
| per lane, no lane cap, th .7 (best) | .8219 | .8422 | .8445 | .8776 | .663 → .736 |
| per lane, ≤ 3 lanes, th .5 (the physical rule) | .7888 | .7901 | .8026 | .8212 | .766 |

Best: +0.8 pt 5-class (f1–5), +0.2 (f0), bought by turning A/P/C into Other (APC −1.0 / −0.9):
below the 1-pt bar → **dropped**; ≤ 3 lanes costs 2.5 pt; the phase-level cap tops at .8157. With
TRUE lanes (36 fully tagged phases, 142 detectors, 0 label breaks): .7465 → .7535 5-class,
.7939 → .8015 A/P/C — one detector; the model already rarely doubles a role in a lane.

**n_lanes** (≤ 3-lane setting, 4,009 phase-periods): 1 lane 43.6 %, 2 30.1 %, 3 26.3 %;
plausible (1–3 and ≥ L_min) **94.6 %**; = L_min 67.5 % (A3: 34 %); exact on the 36 fully
lane-tagged phases **66.7 %** (32 are 2-lane; 3 inferred on 13 — still over-splits); pairwise
agreement with the text 79.7 % (923 pairs). Table `lr4_nlanes.parquet`; a cabinet-print table
at `%DC_WORK%/trackA/nlanes_labels.parquet` (DeviceId, target "P2", n_lanes) is scored by
`lr4_decode.py` automatically — validation only, never an input.

**Verdict.** Labels: fine under the right rule (1.4 % of phases > 3 lanes). Grouping: materially better
(OOF .875–.883; off-peak high-pass count correlation .895 alone). Decode: no — keep argmax; keep
n_lanes as a review aid. Open for the user: is ≤ 1 Advance per lane right with two setbacks?
