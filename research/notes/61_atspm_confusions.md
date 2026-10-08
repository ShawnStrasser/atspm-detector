# 61 — Confusions among the ATSPM classes: diagnosis + targeted features (2026-10-01)
Set-up = note 59 baseline (229-feature function arm, 3 seeds, D lanes, stack-pick decode, stack_loser nonatspm_ap):
ATSPM .8914 E / .9042 R, reproduced exactly. Six folds OOF, paired signal bootstrap (pt). CPU, 6 threads. locked_v2 asserted
absent. Code `trackA/s61_atspm_conf.py` (diag / feats / fit / score / target); work `%DC_WORK%/s61/` (rows_base.parquet,
feats.parquet, fit/, score_*.json, target_*.json, an*.py = the analysis scripts). Technology / print subtype = analysis only.
## Diagnosis (realistic, 254,540 rows; A->wrongA = 8,622 errors = 3.39 pt of the 9.58 pt ATSPM error)
| true -> pred | pt | radar / loop / video / ? | short (m5+m10) share | from dets wrong >= 80 % of windows |
|---|---|---|---|---|
| Yellow_Red -> Count | .79 | 1767 / 0 / 168 / 68 | .51 | .24 (both YR<->C directions) |
| Advance -> Presence | .66 | 135 / 894 / 440 / 212 | .43 | .34 (both A<->P directions) |
| Count -> Yellow_Red | .55 | 1371 / 0 / 14 / 11 | .31 | |
| Presence -> Advance | .43 | 365 / 364 / 213 / 145 | .48 | |
| Count -> Presence / Count -> Advance / Advance -> Count | .29 / .21 / .21 | mostly radar | | |
* YR <-> Count (1.34 pt, 92 % radar): YR recall .67 at 5 min vs .87 at 66 h. 631 of 2,003 YR->Count rows are SWAPS (the
  phase's Count is called YR in the same window). At 5/10 min there are no lanes, so YR and its Count twin can both be Count.
  334 of 1,396 Count->YR come from the decode (argmax Count, lane already holds a Count -> next free class); the decode's
  Count->YR moves are net positive (903 right vs 467 wrong).
* Event-level twins (66 h, 485 YR zones with a Count twin): YR and Count see the same vehicles: volume ratio .99-1.00,
  75-80 % of ONs matched both ways within 0.5 s, same first-ON-in-green lag (4.5-4.7 s). The user's speed-filter signature
  (YR misses the first car, fewer counts) is NOT visible in this radar data; only a weak one (YR fewer non-green ONs:
  green share .915 vs .876) that is REVERSED in the error cases (.870 vs .898).
* Advance <-> Presence (1.09 pt, loop / video): the top existing features separate A from P with AUC .93-.97 overall but
  point the OTHER way inside the errors (dur_mean_red AUC .85 reversed; all top-30 features reversed): the misses are zones
  that behave as the other class (advance under a standing queue / video advance zones held in red; 396 of 1,681 A->P are
  phases whose only sibling is another Advance = no stop-bar partner for the lag family). Not a missing-feature problem.
* Existing features already carry the physics: twin coincidence (lag family) is the top YR/C separator overall (AUC .77),
  but no existing or new feature separates the YR/C error cells (best within-error AUC .61, mostly reversed).
## New features (hi-res only, phase-anonymous, NaN when absent; 456,033 det-windows, 62 s)
tw (15): best stop-bar twin on the predicted phase (max chance-corrected share of ONs matched within 0.5 s): match both ways
+ asymmetry, log volume ratio, unmatched non-green ONs per cycle (self / twin / diff), non-green share diff, twin's first
green ON hit by self and reverse, first-ON lag diff, n twins >= .5, cover by any sibling, summed twin volume (spanning).
rl (7): occupied at green start, time to first OFF after green start (median / q75; start-up wave), red onset of the holding
ON (share of red), both relative to the phase's minimum, first-OFF minus twin. Coverage tw 63 % of YR/C rows, rl 72 % of A/P.
## Screen (seed 0 vs base seed 0; bar 1 pt overall, or CI > 0 on the targeted confusion with no overall loss)
| cfg | ATSPM E (R) | dE [CI] (dR) | YR<->C errors pt (base 1.335) | A<->P errors pt (base 1.084) | by window E |
|---|---|---|---|---|---|
| tw+rl | .8906 (.9033) | +0.01 [-0.10,+0.12] (-0.01) | 1.254, gain [0.00,0.15] | 1.104 | m5 +0.21, h3 -0.14 |
| tw+rl shuffled | .8902 (.9031) | -0.03 [-0.10,+0.04] | 1.355 [-0.05,+0.01] | 1.086 | full -0.20 |
| tw | .8912 (.9039) | +0.06 [-0.04,+0.17] (+0.05) | 1.261 [-0.01,+0.15] | 1.095 | m5 +0.34 [+0.11,+0.57] |
| rl | .8898 (.9025) | -0.08 [-0.16,+0.01] (-0.09 [-0.18,-0.00]) | 1.369 | 1.082 [-0.02,+0.03] | m30 -0.15 |
tw promoted to 3 seeds (vs base 3 seeds): ATSPM **.8919 E / .9045 R, +0.05 [-0.04,+0.14] / +0.03 [-0.07,+0.13]**; YR<->C errors
1.335 -> 1.244 pt, **gain +0.09 [+0.02,+0.16]** (per seed s0/s1/s2: +0.07 / +0.11 [+0.04,+0.18] / +0.09 [+0.00,+0.16]);
A<->P errors 1.087 -> 1.117 (-0.03 [-0.05,-0.01]); all A->wrongA 3.387 -> 3.331 (+0.06 [-0.01,+0.13]); by window m5
+0.28 [+0.08,+0.48], m10 +0.12, h3 -0.16 [-0.33,-0.01], full +0.18 n.s.
## Verdict
* rl dropped (no A/P gain, overall slightly negative). tw: a real but small targeted gain (YR<->C -0.09 pt of errors, CI clear,
  three seeds agree) with no overall loss (+0.05 n.s.), paid partly by +0.03 pt more A<->P swaps. Far under the 1-pt bar;
  by the brief's second criterion it qualifies -> orchestrator's call (keep as "beyond noise on the target", or drop by the bar).
* Ceiling: most of the 3.4 pt of ATSPM-among-ATSPM error is not model-fixable from the log: radar YR and Count zones log the
  same vehicles (no speed-filter signature), and A<->P misses behave as the other class. Remaining fixable levers are
  short-sample (5/10 min: half the YR/C errors; no lanes there) and labels (A<->P reversed cases worth a label look).
* Not tried (needs the user's OK, note 56b): changing the non-stack loser rule (Count -> next free class); its Count->YR moves
  are net positive anyway, Count->Other / Count->Advance moves lose ~0.1 pt.
