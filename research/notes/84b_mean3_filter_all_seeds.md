# 84b — addendum to note 84: mean3 + filter with all three siba seeds filtered (2026-10-04) — locked untouched
Closes note 84's open item "a filtered OOF for seeds 1 / 2 would make the mean3+filter number fully clean". Inference only;
no model, stacker recipe or package changed (candidate v4b and `model/` untouched).
## What ran
* LOCAL A1000, detached queue (`dc_work/tcn53/q74/queue74c.ps1`, 12 INFER lines appended to `queue74.txt`), 03:58 -> 05:45,
  8-10 min per fold, all 12 ok first try: `x74_sibaflt_s1` / `_s2` = the x69_siba_s1 / _s2 fold checkpoints
  (`tcn53/cloud69/models`), `tcn69_func.py infer --keep cand64/phase_oof.parquet --keep_thr 0.01`, snapshot snap74b
  (post-note-76 inputs) = exactly the seed-0 x74_sibaflt set-up. Keep masks identical across seeds (fold 0: 374,932 of
  681,904 pairs run; 73,594 of 113,850 detector-pieces had a tree row). Coverage .9998 as seed 0.
* `research/code/final84/oof84.py` extended (no other change): arms single_flt1 / single_flt2, 'single_flt' (correctness
  averaged over the three filtered seeds) and **mean3_flt3** = mean3 stacker on mean(flt0, flt1, flt2); new contrasts
  and by-class blocks. Stackers re-fitted as before: every note-84 arm reproduces to 2.98e-8 (unchanged numbers).
  Output `dc_work/final_v3_work/f84/oof/oof84.json` (note-84 version kept as `oof84_note84.json`), log `oof84b.log`.
## Headline (six folds, v4l truth, gate .9 decode, paired signal bootstrap 2,000; 95 % CI)
| pool (E rows) | single+flt0 | single+flt (3 seeds) | mean3+flt0 (note 84) | **mean3+flt3** | champion (note 81) |
|---|---|---|---|---|---|
| >= 30 min E (187,444) | .9174 | .9179 | .9191 | **.9192 [.9081,.9292]** | .9190 |
| >= 30 min R (182,828) | .9273 | .9276 | .9290 | **.9291 [.9184,.9386]** | .9289 |
| 10 min E (37,504) | .8998 | .8989 | .9036 | **.9034** | .9030 |
| 5 min E (35,809) | .8909 | .8898 | .8940 | **.8933** | .8936 |
| contrast (mean3+flt3 minus) | >= 30 E | >= 30 R | 10 min E | 5 min E |
|---|---|---|---|---|
| single+flt0 | **+0.19 [+0.08,+0.30]** | **+0.18 [+0.08,+0.28]** | **+0.36 [+0.18,+0.56]** | **+0.24 [+0.06,+0.42]** |
| single+flt (3-seed mean) | **+0.14 [+0.06,+0.21]** | **+0.14 [+0.08,+0.21]** | **+0.45 [+0.31,+0.58]** | **+0.35 [+0.21,+0.49]** |
| mean3+flt0 (note 84) | +0.01 [-0.04,+0.08] | +0.01 [-0.04,+0.07] | -0.02 [-0.09,+0.06] | -0.07 [-0.15,+0.01] |
| mean3 unfiltered | +0.01 [-0.05,+0.08] | +0.00 [-0.06,+0.08] | +0.00 [-0.08,+0.09] | **-0.09 [-0.18,-0.01]** |
| champion | +0.02 [-0.05,+0.10] | +0.02 [-0.06,+0.11] | +0.04 [-0.08,+0.15] | -0.03 [-0.15,+0.08] |
Filter alone, single member averaged over seeds (single+flt minus single): >= 30 E -0.02 [-0.07,+0.03], 10 min -0.04
[-0.10,+0.02], 5 min **-0.09 [-0.16,-0.02]**.
## By class, >= 30 min E (delta in pt [95 % CI]; n rows)
| class (n) | vs single+flt0 | vs single+flt | vs champion |
|---|---|---|---|
| Advance (59,206) | **+0.28 [+0.10,+0.48]** | **+0.17 [+0.04,+0.31]** | -0.04 [-0.12,+0.05] |
| Presence (57,859) | +0.12 [0.00,+0.26] | **+0.09 [0.00,+0.18]** | +0.12 [-0.04,+0.35] |
| Count (31,639) | **+0.34 [+0.06,+0.66]** | **+0.21 [+0.05,+0.39]** | **-0.19 [-0.36,-0.04]** |
| Yellow_Red (9,491) | 0.00 [-0.54,+0.45] | -0.10 [-0.46,+0.21] | +0.11 [-0.16,+0.38] |
| non-ATSPM (29,249) | +0.01 [-0.28,+0.31] | +0.15 [-0.07,+0.40] | +0.13 [-0.07,+0.32] |
R set: same signs (Count vs champion -0.15 [-0.27,-0.04]).
## Reading
* The fully filtered mean3 equals note 84's half-filtered number (.9192 vs .9191 E at >= 30 min) and the champion
  (+0.02, CI includes 0). Note 84's verdict stands and is now clean: mean3 + filter = v4b default; its gain over a single
  filtered member is real at every length (+0.14 .. +0.45 pt vs the 3-seed single mean, CIs > 0).
* Small cost of the filter on short windows: 5 min -0.07 / -0.09 pt vs half-filtered / unfiltered mean3 (the second CI
  just excludes 0), and -0.09* for a single member. Inside the 5-min seed noise of the network (0.1-0.45 pt); the speed
  gain is the reason for the filter, so no change proposed. Count is -0.19* vs the champion at >= 30 min, offset by
  Presence / YR / non-ATSPM (overall +0.02); worth watching, not acting on.
* Caveats unchanged from note 84: OOF nets v3s-trained (production members are full-data v4l), stackers trained on
  unfiltered OOF (as the package), filter coverage only where tree rows exist (59 % of detector-pieces).
GPU idle again; queue empty.
