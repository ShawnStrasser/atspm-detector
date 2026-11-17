# 157d Busy at night in the package (options OFF) + whole-network side effects; 157e mostly-ON idea (2026-11-10)

User approved 157c's review; asked for unintended consequences, and whether a spiky-occupancy check is worth it.
Code: src (options below), research/code/busy157 (run_network.py, diff_network.py, debug_steps.py, chart_sig.py,
spiky_occ.py, held_flicker.py, build_review3.py). Review: review/busy_at_night_review_3.xlsx.

## Package (health_duckdb.OPTIONS, all False = today exactly; limits in health_limits.json "busy")
* busy_long_zone (tod.sql tdf/tdd/lzg/tod, assemble xx, verdict v_prof): zone with busiest-4h occupancy >= lz_occ
  0.25 is excused unless A) sqrt(night occ share) - mate_r >= mates_delta, or B) its night / evening (20-24 h) count
  ratio >= lz_k 3.5 x the MEDIAN mate's, or C) it alone brought up >= lz_ph_min 6 and >= lz_ph_share 0.2 of its phase's
  night greens (5-min bins: own ON, no mate ON; not when >= lz_recall 0.1 of them had no detector ON = recall).
  Every flagged day must be excused; never for a stuck-on detector or no daily pattern.
* night_drop_not_busy (verdict v_night_drop): no "misses vehicles at night" when busy at night (and not excused).
* flat_clean_mates (assemble flatm pm): mates keeping >= CY.own of their day level at night with >= CY.rest_min an
  hour by day leave the no-daily-pattern yardstick (independent of their own mates).
* Off = byte-identical SQL (snapshots pass), off run = original run (0 diffs); atspm-detector-check passes; pytest
  250 pass.

## Iterations (each a whole-network run, ~3 min with 12 workers)
1. B as sum of mates: stacked twins / flat mates set the yardstick -> 2B365 d36/40 cleared, false "silent" on d24/47.
   -> median mate.  2. B at 2.0-2.5 re-flags the reported long zones (04056 d33/36) along with 2B098 d24; B cannot split them
   on one complete night -> kept 3.5.  3. 06016 d24 cleared at any B (event level: 58 / 86 night greens its own) ->
   phantom-green guard C (5-min proxy matched event counts).  4. ntraf-based mate exclusion: 2B049 d51 missed (d52 not
   ntraf: its mate d51 inflates its rest) and near-silent twins at 05999 / 12072 made low-volume LT zones flat ->
   own-level rule.

## Final effect (Nov 6-8, confidence >= 0.5): 30 detectors change
* busy at night cleared 18 (bad -> ok 17); 06016 d24 kept by C; 10004 d2 loses night drop, d2/d4 -> no daily pattern;
  -> no daily pattern: 2B049 d51, 2B365 d23/42/46, 2C010 d52 (night >= day); new no daily pattern: 08019 d22,
  08056 d21/22 (already bad), 09007 d17 and 08300 d24 (ok -> bad; 08300 borderline low volume).

## 157e spiky occupancy (103 random signals, 1,814 detectors, 10-min bins, correct occupancy)
* Bursts (>= 3 x own 2-h level and mates' rise, >= +25 points): 112 detectors >= 1 a day, 80 with no other finding,
  mostly presence zones holding a waiting car -> too noisy.
* Mostly ON runs (>= 50 % for >= 3 bins, >= 3 x expected): 18 detectors, 11 already flagged (stuck on mostly), 7 new:
  01066 d41, 03020 d54, 03007 d47, 11009 d18 real; 2B382 d27 unclear; 08054 d44, 11012 d2 traffic (no phase mates).
  Needs >= 1 phase mate. ~40-50 new per network window. Pending user.
