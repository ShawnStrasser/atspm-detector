# 43 — Detector health v3: the user's spot-check ideas, phase/function-aware rules (2026-09-29)
Code `research/code/health/` (`health_core` + `h3_oof/explore/eval_build/recovery/final/final_eval/baseline/spotcheck`);
artefacts `%DC_WORK%/health3/`. Training signals only (folds_v4 minus locked_v2); no fault events; note 38's harness. **Rules vs learned:** the scorer is still expert RULES (fixed limits = p99.5 of 8,666 presumed-healthy detectors); the
learned synthetic-fault model (note 38) stays dropped. The only learned parts are its OPTIONAL inputs: OOF predicted
phase (stage-12 frame `pred_phase`) and OOF function (note 28 first.all.wi, frame_v6, 3-seed mean) - nothing in-sample.
## What was tried (user feedback 2026-09-28) -> kept / dropped
* **Choppiness** `chop` = Pearson dispersion of a detector's counts around e_t = s_t * r_t, r = its reference, s_t = its
  share of r in the +-2 h around t excluding t (Poisson on both ~1). Reference = other live detectors on its predicted
  phase (>= 2, twins = same zone on two inputs excluded), else rest of signal. Healthy AUC vs real problems by
  resolution (66 h): 5 min .69, **15 min .72**, 30 min .73, per signal cycle .69 (cycle best on "share fell > 5x",
  .62 vs .53, but a heavy healthy tail: p99.5 10.4). At equal FA vs loosening the old rules: every resolution within
  +-2 pt (1-2 detectors). KEPT 15-min, replacing note 38's `erratic` (same catches, clearer meaning: spikes the phase
  shares are traffic). One-sibling reference rejected: when that sibling fails the detector looks choppy (2B113 d2).
* **Correlation with the phase's traffic, flat-line (elasticity) for Advance/Count, per-function limits** (chop, corr,
  flat): no gain at equal FA; skipping `corr` for Presence/Other or `night_day` for Bike loses the user's own
  05032 d52 (Other) and 03024 d4 (Bike), both "yes, bad". DROPPED.
* **Silence expected from the phase** (>= 2 phase siblings): KEPT - a lull the whole phase shares is not a fault.
* **Bike-aware silence** (predicted Bike: expected x .02, median share .06 % vs 3.6 %): KEPT; -> not_enough_data.
* **Stuck-on with no OFF logged** (ON -> ON, held to the next ON; counts only if >= 30 ONs expected meanwhile: 1.9 %
  of healthy have such a gap at 66 h) KEPT; silent run inside it reported once, as stuck.
* **Shared stuck-on** (>= 3 other detectors ON >= 5 min through half the episode): capped at suspect, period listed.
  Zeroing it (first try) lost 15-35 pt of stuck-defined positives (whole cabinets ON 30-40 min).
* **Bad periods + recovery check**: `bad_periods` = [{start, end, what, recovered}]; recovered = share of the phase
  reference in the 6 h after / 6 h before within 0.5-2 (>= 3 h each side); clean -> capped at suspect.
## Is data after a stuck / silent episode good? (`h3_recovery.py`, share after / before, phase reference)
| event (Sept 2026, 66 h) | n | within 1.5x | off > 2x | fell < 1/2 |
|---|---|---|---|---|
| stuck-on, recovered | 671 | 50 % | 33 % | 17 % |
| silent, recovered | 571 | 63 % | 21 % | 10 % |
| healthy, random split (null) | 7,939 | 74 % | 10 % | 5 % |
254 of 797 detectors had 2+ episodes in 66 h. Dec 2024 -> Sept 2026: detectors with a recovered episode in Dec are bad
in Sept 15.0 % vs 4.6 % (dead 5.5 vs 2.9 %; n 379 / 6,910). => Often usable, not always: check per episode (as built).
03033 d8 (user's example): after its 2.2-h stuck ON it counted 46 % of its share for 6 h -> stays bad.
## Result, % flagged (bad or suspect), Sept 2026 windows (v2 = note 38/40 scorer)
| group | v2 | v3 log only | v3 + phase | **v3 + phase + function** | same, short-window preds | CONTROL shuffled preds |
|---|---|---|---|---|---|---|
| presumed healthy = FA 2h / 6h / 24h / 66h | 1.4/1.3/2.3/4.2 | 1.7/1.5/2.5/4.5 | 1.7/1.4/2.4/3.9 | **1.0/1.1/2.3/3.9** | 1.2/1.2/2.3/3.9 | 1.7/1.4/3.1/5.6 |
| dq health fail (87) | 24.5/27.6/62.6/96.6 | 24.9/27.6/63.2/96.6 | 25.7/28.2/63.2/97.7 | **24.1/27.0/63.2/97.7** | 24.5/27.0/63.2/97.7 | 24.9/27.6/63.2/97.7 |
| dead on print / dead since Dec / card dead | 92-100 | same | same | -0.2..-0.9 (Bike loops) | | |
(label-check health (74) moves like dq: 21.6/25.7/62.2/95.9 -> 22.5/24.3/62.2/97.3; share fell > 5x since Dec: unchanged.)
% **bad** on healthy (what training exclusion uses): v2 0.8/0.6/0.9/1.2 -> v3+pf **0.4/0.4/0.7/0.9**; on dq positives at 66 h
48 -> 35 % bad (the rest suspect: shared or cleanly recovered episodes, periods listed). Shuffled predictions give no
gain (FA 3.1 / 5.6 at 24 / 66 h, worse than v2): the gain comes from real predictions. Log-only v3 is +0.1-0.3 pt FA vs
v2 (choppy + no-OFF stuck); the phase / function inputs more than pay for it. Chop limits were set on all healthy rows,
so its FA share is in-sample (note 40's cross-fit put that optimism at +0.3-0.4 pt). Positives are few (87 / 74): +-1.1 pt = one detector.
## The user's 15 answers as a mini test (66 h; "maybe" rows unscored)
v2 12/13, every v3 variant 12/13; the miss is 2B044 d14 in all (he said "no, but maybe": 15-min ON, only 1 other
detector overlapped; now suspect with only that 15 min listed as bad). His maybes: 2B439 d25 now ok (Presence plateau,
chop 1.8 < 5), 03090 d25 ok (d22 spikes with it), 04055 d24 ok (midnight spike shared by d22/d23), 07035 d19 still
suspect (rapid only). His remarks: 01026 d25 now bad (chop 9.7), 07027 d8 bad; misses 11042 d22 (spikes shared with
its one partner d23; no-OFF 48-min ON shared with 10 detectors) and 2B422 d8 (undercounts its stop-bar loop - a lane
pairing check, not a health rule; v2 caught it only via a borderline whole-signal silence, lam 31 vs limit 30).
## Packaged (numpy + pandas; torch/lightgbm/scipy/sklearn blocked: runs)
`health(events, start, end, detectors=None, phase=None, function=None)` -> status, score, reason, `bad_periods`, s_* .
0.18 s per signal-window (66 h ~0.5 s). Deliverable `%DC_WORK%/health/health_v3.parquet`: 26,585 rows (Sept 2026 66 h:
13,879 ok / 2,185 bad / 687 suspect / 218 n.e.d.; Dec 2024 72 h: 7,193 / 1,847 / 395 / 181), status, score, reason,
bad periods (JSON), predicted function. `model/health.py` untouched. Spot-check `review/spotcheck_health.xlsx` rebuilt:
10 rows (2 new choppy, 4 stuck-period cases, 4 of his maybes), charts `v3_*.png` with a healthy baseline (healthy
same-phase detector; IOI band = 40 healthy detectors of the predicted function, `ioi_baseline.parquet`); his earlier
answers kept on sheet 2.
