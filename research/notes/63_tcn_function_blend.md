# 63 — TCN function head + trees: honest blend weight, stackers, error analysis (2026-10-01)
Fold 0 only (folds_v4, 91 signals, 27,295 E / 26,669 R scoring rows), note-59 harness (stack-aware ATSPM score, D-lane
stack-pick decode, truth_v3s). Trees = note-57 229-feature 3-seed arm (.8810 E / .8870 R fold 0, reproduced exactly).
Nets (note 53): `fj` joint phase+function head, `ff` function-only head; coverage 1.0. CPU 6 threads; locked_v2 asserted
absent. Code `neural/tcn63_blend.py` (run / err / ctl / new62); work `%DC_WORK%/tcn53/blend63_fjff.json`, `blend63_on62.json`,
`blend63_treectl.json`, `err63_{fj,ff}.json` + `_rows.parquet`. Technology = analysis only. Headline = samples >= 30 min
(user 2026-10-01: 30 min / 1 h / 3 h / 6 h / 24 h / full pooled); 5 / 10 min secondary.
## Nested set-up
Fold-0 signals split into 5 seeded signal groups; for each outer group the weight / stacker is chosen / fitted on the other 4
and applied to it; 3 partition seeds (63/64/65). Decode / stack groups never cross a window, so assembling rows from
per-weight decodes is exact. Weight chosen on all E rows of inner signals, applied to both sets. Stackers: fixed shallow
settings set up front (LGB 7 leaves, depth 3, min leaf 300, L2 10, 150 rounds; logistic C 0.05), no tuning.
## 1. Blend weight (tree weight w; pt vs trees, signal-bootstrap 95 % CI; p63 unless a range is given)
| arm | >= 30 min E (.8922) | >= 30 min R (.8987) | 5/10 min E (.8514) | all E |
|---|---|---|---|---|
| fj nested fixed w (0.6 in 14/15 splits) | +0.89 [+0.19,+1.71] (p64 +0.86) | +0.76 [+0.14,+1.50] | +0.76 [+0.16,+1.41] | +0.85 [+0.25,+1.57] |
| fj nested per-window w | +0.91 / +0.75 / +0.92 | +0.78 / +0.63 / +0.79 | +0.65 / +0.57 / +0.62 | +0.83 / +0.70 / +0.83 |
| ff nested fixed w (0.6-0.65) | +0.49 [+0.02,+1.04] (+0.52, +0.48) | +0.39 [+0.00,+0.88] | +0.49 [-0.06,+1.08] | +0.49 [+0.04,+1.00] |
| fj alone / ff alone | -2.00 / -3.21 | -2.20 / -3.42 | -4.42 / -5.36 | -2.67 / -3.81 |
* Full-fold curve (fj, all E): w .3 .8793, .5 .8886, .6 .8895 (top), .7 .8872, .9 .8830 — flat 0.5-0.65.
* Per-window weights (m5/m10 0.6-0.65, h1-h24 0.5-0.55, full 0.45) add nothing over one weight.
* On top of the new note-62 reference (routed short head + cy twin decode; fold 0 .8827 E / .8887 R): fj nested
  >= 30 min +0.84 [+0.13,+1.69] E / +0.71 R (unchanged pool), 5/10 min +0.74..+0.86 [+0.10,+1.55] E; ff +0.49 / +0.21..+0.33.
  Additive to note 62.
## 2. Stackers (nested; >= 30 min E pt, CI for p63; then p64 / p65)
| stacker | fj | ff |
|---|---|---|
| LGB on both probs + log minutes + entropies + margins + agree | +0.39 [-0.60,+1.61] / +0.50 / +0.48 | +0.06 / +0.17 / +0.25 |
| logistic, same inputs | +0.23 [-0.83,+1.51] / +0.51 / +0.75 | +0.36 / +0.30 / +0.63 |
| LGB + context (user idea: health core + stack, pick inputs, D lanes, lane- / phase-mates' probs; 39 cols) | +0.69 [-0.54,+1.97] / +1.10 [+0.11,+1.99] / +0.98 [-0.01,+1.81] | +0.43 / +0.75 / +0.89 |
| control: context row-shuffled | +0.34 | -0.06 |
| control: net row-shuffled | -0.27 | -0.24 |
| trees' probs only (recalibration) / trees + context, no net | -0.26 / -0.19 | same |
* A learned stacker on ~73 signals costs ~0.3 pt (>= 30 min; -1.6 on 5/10 min) by itself; context pays +0.35..+0.5 over its
  shuffled control; the context stacker matches but does not beat w 0.6 and its CI is 1.5x wider. Judge it again only with
  six-fold training rows.
## Control: is it just ensembling?
Trees blended with another TREE arm (note-57 drop arms, seed 0), all E: drop_px .5 +0.07 [-0.16,+0.30], .6 +0.01; drop_cond
.5 -0.11; drop_sib .5 -0.30. Tree-tree blends gain nothing: the +0.85 is information the network adds, not averaging.
## 3. Error analysis (all E rows; fj w 0.6: 476 fixed, 244 broken; ff w 0.65: 363 / 220)
* Fixed (trees wrong -> blend right): Presence->Other 96 (74 radar), Count->Other 48 (radar 28 / video 20), Presence->Advance
  42 (26 loop), Advance->Other 37, Count->Yellow_Red 32 (radar, mostly pulse), Count->Presence 31 (radar / video).
* Broken: Advance->Other 43 and Other->Advance 41 (video 42 of 84), Yellow_Red->Count 34 (radar pulse, 27 at 5/10 min),
  Mid->Advance 18 (loop / unknown, short windows), Other->Presence 17. ff: same pairs, fewer of each.
* ATSPM error by true class (trees -> fj blend): Count 8.2 -> 5.8 %, Presence 6.5 -> 4.8 %, Advance 8.0 -> 8.0 %,
  Yellow_Red 44.0 -> 45.9 %, Other 30.1 -> 30.3 %, Mid 16.8 -> 15.7 %.
* Net pt inside group: radar +1.06, loop +0.80, video +0.64; normal (non-pulse) +1.10, pulse -0.21; volume high +1.01 /
  mid +0.64 / low +0.81. Top 5 signals hold 37 % of fixes and 39 % of breaks (fixes on 57 of 91 signals).
## Verdict
Winner = fixed blend, tree weight 0.6, with fj (joint head beats function-only ff by ~0.4 pt). Headline >= 30 min:
+0.89 [+0.19,+1.71] E / +0.76 [+0.14,+1.50] R on fold 0; 5/10 min +0.76; additive to note 62. Stackers (with or without
context) do not beat it on one fold. Plain terms: the network rescues stop-bar zones with normal (non-pulse) output that the
trees wrote off as Other or as the wrong stop-bar class — mostly radar Presence and radar / video Count zones; it breaks
Advance-vs-Other on video and radar pulse Yellow_Red zones on 5-10 min samples.
Six-fold: >= 30-min point estimate +0.89 is just under the 1-pt function screening bar but clear of noise (CI > 0 on all
three partitions, tree-blend control ~0); only six folds can settle it.
