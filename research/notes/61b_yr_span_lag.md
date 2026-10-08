# 61b — User cues for Yellow_Red vs Count: lane spanning and downstream lag (2026-10-01)
Follow-up to note 61. The user ruled that YR<->Count swaps get no scoring credit. The headline is now samples >= 30 min
(user decision); 5/10 min are secondary. Same set-up, rows and scoring as note 61 (base .8914 E / .9042 R; seed 0 .8905 /
.9034). Code `trackA/s61_atspm_conf.py` (`feats` now also writes sp_ / lg_; `target` reports all / ge30 / short scopes);
work `%DC_WORK%/s61/` (feats.parquet, fit/{splg, splg_shuf, sp, lg}, score_screen2.json, target_screen2.json,
target_tw3_scoped.json). locked_v2 asserted absent. CPU only, 6 threads. Technology = analysis only.
## Features (hi-res only, phase-anonymous, NaN when absent; siblings = same predicted phase, >= 5 ONs, match = ON within 0.5 s)
* sp (6), cue 1 "YR spans lanes, Count is single-lane": siblings this detector covers (chance-corrected >= .5 of THEIR ONs
  matched), how many of those are mutually independent (< .3 either way = separate lanes), volume / SUM of the independent
  covered zones, union cover, side-by-side share (>= 2 covered zones fire for one ON = counted once), siblings that cover it.
* lg (9), cue 2 "YR sits on top of or downstream of Count, never upstream": per matched vehicle, signed lag self minus best
  twin for ON and OFF (median, share < -0.05 s, share > 0.1 s); ON lag pooled over the covered zones and over the zones that
  cover it. Coverage: sp 85 % of rows, lg 28 % (needs a twin; YR/C truth rows 51-98 %).
## The cues are in the log
* Lag sign, YR truth vs Count truth (median ON lag to the twin): YR later 56-57 %, earlier 13-14 %; Count earlier 49-51 %,
  later 16-22 %; the same at 5 min, 30 min and 66 h. OFF lag likewise (YR later 58-60 %, Count 12-16 %). Overall AUC
  YR vs Count: lg_off_med .79, lg_on_med .73, sp_ncov .80, sp_union_cov .68.
* Inside the error cells (YR called Count / Count called YR) they do not separate: AUC .41-.56 (lag reversed), sp .48-.51.
## Decode check (does a lane-spanning YR take "Count" from a single-lane zone?)
Yes, in principle: the greedy decode books (lane, Count) on every lane a detector spans. In practice it is rare: of 2,003
YR->Count rows (realistic) only 33 are on a lane-spanning YR (27 of them block a single-lane true Count) = 0.01 pt.
Lane-spanning YR truth is called YR 96 % of the time (3,162 / 3,284); the errors are single-lane YR (849 called Count of 5,765)
and 5/10-min windows, which have no lanes (1,121 of 2,003).
## Screen (six folds, seed 0 vs base seed 0; realistic; >= 30 min first, CI = paired signal bootstrap, pt)
| cfg | ATSPM >= 30 min (base 91.283) | YR<->C errors >= 30 min (base 1.059) | ATSPM all (E dE) | 5 min (E) |
|---|---|---|---|---|
| sp + lg | 91.289 [-0.20,+0.19] | 1.061 [-0.17,+0.15] | 90.412 (+0.11 [-0.06,+0.27]) | +0.40 [+0.12,+0.65] |
| sp + lg shuffled | 91.253 [-0.11,+0.05] | 1.083 [-0.07,+0.02] | 90.299 (-0.03) | +0.02 |
| sp | 91.338 [-0.05,+0.16] | 1.022 [-0.03,+0.12] | 90.345 (+0.02) | +0.00 (m10 -0.22 [-0.37,-0.06]) |
| lg | 91.282 [-0.18,+0.16] | 1.094 [-0.19,+0.10] | 90.397 (+0.07) | +0.33 [+0.08,+0.58] |
All-windows YR<->C errors (base 1.335): sp+lg 1.254 [-0.06,+0.22], sp 1.320, lg 1.293, control 1.369. Adv<->Pres flat.
Note-61 tw re-judged on the new headline (3 seeds vs 3 seeds): >= 30 min ATSPM 91.366 -> 91.342 [-0.13,+0.08], YR<->C 1.071 ->
1.026 [-0.02,+0.12] n.s., Adv<->Pres 0.832 -> 0.857 [-0.05,-0.00]; short windows ATSPM +0.17 [+0.03,+0.32], YR<->C 2.012 ->
1.802 [+0.11,+0.33]. So tw's gain is short-window only.
## Verdict
* Nothing qualifies on the >= 30-min headline (no overall gain, no clear YR<->C gain) -> no 3-seed promotion. sp, lg and sp+lg
  dropped by the rule; tw from note 61 also fails the new headline (short-window gain only).
* The user's two cues are physically right and visible in the log (lag sign and lane coverage separate YR from Count with
  AUC ~.8), but the trees already reach the same information through the existing twin / lag / sibling features: the
  remaining >= 30-min errors are the detectors where the cue is absent or reversed (YR level with or ahead of its Count).
* Where they do help is 5-min samples (sp+lg +0.40, lg +0.33, tw +0.28 at 5 min; CI > 0), i.e. the twin pair decision without
  lanes — input for the note-62 short-window twin rule (files: `%DC_WORK%/s61/feats.parquet`, columns tw_ / sp_ / lg_).
* Decode: the spanning-YR-takes-Count case exists but is worth 0.01 pt; no change proposed.
