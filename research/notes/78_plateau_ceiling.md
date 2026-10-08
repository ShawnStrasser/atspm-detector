# 78 — Why function accuracy has plateaued, how much is labels, and the realistic ceiling (2026-10-03)
Champion = note-77 OOF (trees arm c 3 seeds + siba 3 seeds -> context stacker f77 3 seeds -> D lanes, gate .9, pick, twin
decode; truth_v3s; ATSPM stack-aware score). Reproduced: >= 30 min .9171 E / .9282 R. Code `evaluation/err78.py`
(rows / attrib / labels / lc) + `evaluation/lc78.py` -> `%DC_WORK%/err78/` (rows.parquet, attrib78.json, labels78.json,
lc78.json, errors_*.parquet). Note-70 rules unchanged (err70.attribute). locked_v2 asserted absent. CPU only, <= 6 threads.
## 1. Decomposition on the champion (pt of rows; share of errors) — note 70 (cand64) -> now
| >= 30 min | realistic (183,024 rows) | everything (187,980) | R 5 min / 10 min |
|---|---|---|---|
| total error | 8.26 -> **7.18** | 9.52 -> 8.29 | 9.62 / 8.72 |
| (a) label likely wrong | 1.52 -> **1.46 (20 %)** | 2.97 -> 2.72 (33 %) | 1.58 / 1.62 |
| (b) ambiguous by definition | 3.21 -> **3.24 (45 %)** | 3.10 -> 3.14 (38 %) | 4.96 / 4.40 |
| (c) model-fixable | 3.30 -> **2.26 (31 %)** | 3.23 -> 2.22 (27 %) | 2.99 / 2.58 |
| (d) other | 0.23 -> 0.22 (3 %) | 0.22 -> 0.21 | 0.09 / 0.13 |
All of the 1.1-pt gain since note 70 (stacker + gate + siba) came out of (c); (a) and (b) did not move. R sub-causes: (a) config-only
confident .59, unhealthy .44, Dec stale .26, config-only unsure .17; (b) Other subtypes acting as PM 1.76 (advance_presence .55,
long_zone .47, presence_20_75 .21, superseded loops .18+.07, departure .13), Mid vs Adv/Pres .61, YR/Count twins .57, radar long
Advance .18, stack extra .12; (c) PM<->PM .87, short-sample .69, ->/from Other .58 (Pres->Other .47), phase .11.
## 2. Mislabel rate, four angles (non-locked labels; realistic >= 30 min unless said)
* (a) Reviews: locked-143 round (note 14, config labels): 37 of 79 model misses were label errors (47 %), 38 confirmed, 4 '?'.
  Sheet v1 (note 72): 4 rows answered so far: 2 function rows label wrong (2B091), 2 phase rows label right (2C009).
* (b) Print (high) vs config, both present, >= 5 act.: disagree 10.3 % at ATSPM level (n 5,215; medium 30.8 %, low 40.3 %);
  top Other->Advance 80, Advance->Presence 77, Presence->Count 54. Config-only labels = 8.4 % of scored labels; their error rate
  12.2 % vs 6.3 % on print_high -> excess ~0.5 pt of rows, i.e. ~0.5-0.9 % of rows carry a wrong config label. User rulings
  disagree with config on 13.4 %. print_high error itself is not measurable this way.
* (c) Behaviour check: fail + misconfigured = 3.0 % of checked labels (10 % not checkable); these are already out of
  "realistic" and are 1.29 pt of the everything-set error.
* (d) All six families (trees, fj, siba, sibm, cw2, stacker) agree, each p >= .9, against the label: 0.59 pt of rows (139 dets;
  p >= .8 1.21 pt / 254; p >= .95 0.29), only 52 of 9,743 detectors in most windows. Of the .9 rows: (a) .32, (b) .23, (c) .08 -
  half of "everyone disagrees" is definitional, not a wrong label. Caveat: all families learn from the same labels.
* Reconciled: label-caused error in the realistic score **0.4-1.5 pt (6-20 % of errors), central ~0.9**; ~1-2 % of realistic
  labels wrong; with the behaviour-check fails (everything set) ~3-5 % of all labels questionable.
## 3. Irreducible / definitional (R >= 30, 3.24 pt)
* Open user question (radar / long zones labelled Other: advance_presence + long_zone + presence_20_75 + radar long Advance):
  ~1.4 pt; "score as nearest class" moves ~1.4 pt (up to 1.9 with the rest of B1/B2); "keep Other" moves nothing (some of it is
  learnable: 47 % of (b) rows have at least one family right, but a model that calls them Other loses real Advance / Presence).
* Unidentifiable from the log whatever the answer: YR/Count twins .57, superseded loops .25, departure .13 -> ~1.0 pt floor;
  Mid vs Advance / Presence .61 is half definition (series loops).
## 4. Plateau diagnosis
* Learning curve (trees arm, seed 0, signals subsampled within each training fold, 2 draws, decoded, R >= 30): 25 % .8883,
  50 % .9031, 75 % .9092, 100 % .9144 (vs 100 %: -2.6 / -1.1 / -0.5 pt, CIs < 0); ~+1.1-1.2 pt per doubling of signals, not
  flattening yet. Note 66: more days per detector added nothing -> it is more SIGNALS (layouts), not more data per detector.
  Expect ~+0.5-1 pt on the champion per doubling of labelled signals (stacker partly substitutes).
* Oracle-of-models: at least one family right on 3.26 pt of the 7.18 (45 %): (c) 56 %, (b) 47 %, (a) 26 %. Post-hoc pick, so
  an upper bound; per family 1.5-1.9 pt. Family alone (R >= 30, decoded): trees .9149, siba .9089, cw2 .9073, sibm .9053, fj
  .8845; stacker argmax .9280 = champion (the decode/gate now adds ~0). GRU-siba seed 0 fold 0 .8722 vs TCN siba 3 seeds .8907.
* Seed spread: champion stacker seeds .9172 / .9169 / .9168 E (+-0.02 pt); trees seeds +-0.03 pt. The last five model changes
  (sibm, cw2, specialist decider, order-free features, order-free lanes) were all within -0.04..+0.03 pt.
## 5. Ceiling (realistic, >= 30 min, today .928)
* Current labels: + half to all of the learnable part of (c) (0.6-1.3 pt) -> **.935-.94**.
* + label fixes (0.4-1.5 pt measured): **.94-.955**. + definition answered "nearest class" (+1.4-1.9): **.955-.97**.
* Floor even then: twins / superseded / departure ~1 pt + Mid ~.3 + residual model error -> ~.97-.975 is the hard top.
## User summary (8 lines)
Today the function model is right on about 92.8 of 100 detector-samples (30 min or longer; 90 at 5 min).
About 1 in 100 labels we score against is wrong (range 0.5-1.5); 3 % more fail the behaviour check and are not scored.
Of the 7 misses per 100: ~1 wrong label, ~3 detectors that by definition act like another class, ~2-3 real model errors.
Realistic ceiling on today's labels: about 93.5-94 %; with the wrong labels fixed: 94-95.5 %.
If long radar / advance-presence zones were scored as their nearest class: 95.5-97 %; ~97.5 % is the hard top.
The last five model changes each moved less than 0.05 points: model tuning has reached its limit.
What still pays: your answers on the review sheet and the open definition question (1-2 pt), and more signals with prints (~1 pt per doubling).
Further network / tuning work is unlikely to add more than a few tenths of a point.
