# 143 Six agreed health checks: limits from a fitted normal bulk at 1 false alarm per 1.26 M detector-days (2026-10-14)
Orchestrator task, research only: src/ untouched, no fault events, locked_v2 asserted absent, nothing re-scored.
Sample h24_b (Mon 2026-09-28 00-24 h, 755 training signals, 14,806 detectors with a predicted function).
Inputs: s140/parity_h24b (engine internals), s141/det141 (too-fast single gaps), s143/ep143 (every ON >= 300 s with
the resolver's queue / shared flags; parity 0 diff; reused, its script h143_episodes.py is the earlier attempt).
Code research/code/health143 (h143_data, h143_fit, h143_sim, h143_report, h143_plot, h143_cases); work dc_work/s143.
## Rule (one line, every check and type)
Normal line fitted by least squares on the probit plot over the 10th-90th percentile of the type's detectors that
have a value; suspect = mu + sigma z(1 - P/share), P = 1/1,260,000, share = valued / eligible. Scales: log10 (hold s,
expected ONs, gaps per 1,000, veh/h/lane, exc_15), ln (ratios: -ln), z as is. Type = function x span 1 / 2+ (+ Count
pulse / normal for stuck on); < 150 valued -> pooled: same Count mode both spans, Count modes, function both spans,
function group same span, group both spans. Model-unsure: least strict over plausible functions (as package).
Bad = current rule relative to suspect: stuck max(1 h, 2L); silent x 100/30; drops ratio / 3; night ratio / 3.02;
busy z >= L with mates (suspect also L x 3.5/6 with mates; any finding bad by 4-h rule); too-fast / too many /
erratic 2x. Kept: hold floor 300 s (shortest hold the queue test sees), Count held ON 30 / 240 min (5 non-zero of
1,842: not fittable), physical 1800 x 1.25 veh/h/lane, gap 0.8 / 0.6 s per lane, all excuses and caps.
Simulation reproduces the package scores at current limits: stuck, count-on, level, night, volume, erratic 0
mismatches; dropout 14 (+ prof 2) = all the dropped daily-pattern interaction.
## New suspect limits (1-lane own type; full table review/thresholds_all/limits143.csv)
Stuck on (min): Advance 210, Presence 43, Count pulse 5 (floor; fit 0.6 s), Count normal 32, YR 5 (floor), Mid 22,
  Bike 1,529, Other 175; 2+: Advance / YR fit poor (R2 .89 / .81, limit > 1 yr = never). Current 15 min (5-7.5 Count/YR).
Goes silent (expected ONs): 65-178, Other 660 (2+ 5,842), Mid 2+ 31; current 30.
Count drops (ratio): .08-.21 for Advance / Presence / Count / YR / Other, Mid .39-.42, Bike .014; current .15.
Misses at night (ratio): .27-.36 most types, Mid 1 .74, Mid 2+ .58, YR 2+ .46; current .19 (new is STRICTER).
Busy at night (z): 5.0-7.0 (Presence 6.8, Advance 5.7); current 6.
Too fast (per 1,000 ONs): 3,400-67,000 = above the possible 1,000 -> never fires (current 159-208 / 17-36).
Too many: fitted 1-in-1.26M point 3,000-24,000 veh/h/lane; physical 2,250 kept (current max(p99.8, 1800)).
Erratic (exc_15): 1.8-3.1 > observed max 0.50 -> never fires (current .02-.27).
Fit-range sensitivity (25-75 / 5-95 vs 10-90): median limit ratio .75-1.22; poorly fitted stuck cells move 0.07-16,000x.
## Flagged (suspect + bad, % of 14,806) current -> new (review/thresholds_all/flagged143.csv)
1 Stuck on 1.93 -> 0.47 | 2 Counts fall 1.13 -> 0.40 (silent .93 -> .12, drops .33 -> .31) | 3 Night misses .82 -> .93
| 4 Busy night .91 -> .87 | 5 Too fast / many .33 -> .06 (too-fast .25 -> 0, too many .14 -> .06) | 6 Erratic .57 -> 0.
Status ok / suspect / bad / not enough data: package 1.0.1 research (12 checks) 94.03 / 2.13 / 2.22 / 1.62;
agreed 6 at current limits 94.41 / 1.99 / 1.97 / 1.63; at new limits 96.15 / 0.84 / 1.35 / 1.66 (325 flagged);
new but never stricter than current 96.63 / 0.57 / 1.15 / 1.66 (254). Expected false alarms at the budget: ~0.07.
## Fit quality / disagreements
- Heavy tail (obs p99 at z 4-11 instead of 2.33) almost everywhere for night misses, count drops, busy at night and
  stuck holds: the bulk line under-predicts the tail, so the flags are either real faults or far above budget.
  Night misses: 77 of 137 new flags are new (Mid 1 limit .74 flags 2.6 %; Bike 1 pooled 13.8 %) -> false-positive risk.
- Light tail (z99 1.2-2.3): too-fast, too many, erratic -> extrapolation lands beyond the observed / possible maximum.
- Poor bulk fit (R2 < .9): Count pulse hold (quantised 0.1-s ONs), YR 1 hold, stuck 2+ Advance / YR / Count, too-fast
  Advance 2+ / YR 2+, night YR 1 / Bike 1 / Other 1.
- Too many: data agree with the physical limit: 9 of 13,369 above 2,250 (max 3,360; p99.8 1,763; p99.99 2,972).
- Too-fast gap limits: no type reaches the fitted point; the per-1,000 measure has no separable faulty tail at this
  budget (p99.8 162-190 per 1,000), consistent with notes 141-142.
## Flicker cases (cases143.csv): none flagged
asclog d16 27 / 1,000 and integrator d16 85 / 1,000 vs 5,744 (Presence 1); holds 82 / 56 s, silent 11 / 22, night
1.27 / .68: ok. 11021 d2-d4 ok (current agreed set also ok; package 1.0.1 d3 suspect via chattering, now dropped).
## Outputs
review/thresholds_all/limit_1_stuck_on.png, limit_2_counts_fall.png, limit_3_misses_night.png,
limit_4_busy_night.png, limit_5_too_fast_too_many.png, limit_6_erratic.png (all opened), limits143.csv,
flagged143.csv; dc_work/s143/limits143.json (limits per check x type, definitions, fit parameters), sim143.parquet,
fit143.parquet, sens143.csv, cases143.csv.
## Open (orchestrator / user)
- Too-fast and erratic never fire under the rule; night misses becomes stricter. Variant "new, never stricter than
  current" (flagged143.csv) removes the stricter cases; choosing it, or a different bulk scale, is a user decision.
