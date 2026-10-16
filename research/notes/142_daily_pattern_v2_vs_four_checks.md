# 142 Can one rebuilt daily-pattern check replace the four time-of-day checks? 24-h samples only (2026-10-14)
Orchestrator task, research only (src untouched, no fault events, locked_v2 asserted absent). Sample = h24_b of notes
140/141 (Mon 2026-09-28 00-24 h, 755 training signals, 14,806 detectors; 14,309 scored: >= 50 ONs, daytime median
>= 2/h). Four checks = s141 det141 categories (suspect or bad): Busy at night 133, Misses vehicles at night 91, Goes
silent 75, Count drops 46; union 289 = 1.95 % of detectors (each alone 0.31-0.90 %, all above 1-in-500).
Code research/code/health142 (h142_bins, h142_cov, h142_check, h142_report, h142_plot); work dc_work/s142.
## 1 Why the current daily pattern (band_z, sqrt share of own day, z >= 4 for >= 3 h) misses (stored refs h24_b)
- Night down-moves are capped below z 4: max |z| of a SILENT hour = mc / s = 1.6-1.9 at 00-04 h (Presence / Advance
  medium, 1,000 a day), 2.2-2.8 (Count high, 2,600 a day); z 4 only from 07 h to 20 h. Misses at night and silences
  starting 21-05 h (24 of 61 missed in 141) can never fire.
- Night up-moves need ~12x the normal night count (Presence medium 03 h: 3 -> ~37 ONs/h; Count high 6.5x) for z 4;
  the per-hour spread sc 0.025-0.035 is ~half the night sqrt share itself.
- Self-normalisation: a drop lowers the day total, which lifts the share of the drop hours. Drop to 50 % from 17 h:
  z -1.9 to -0.5 (Presence medium), -3.0 to -0.9 (Count high); to 30 %: -3.2 max; only ~10 % reaches z 4, and only
  for 17-18 h, so the 3-h run fails. Hence late drops (26 of 35 missed in 141) are invisible.
- 3-h run: a 1-2 h silence (34 of 61 missed) never qualifies; an hour half-silent has z ~2.5.
- Counts only: time ON is not used, while 75 of 96 missed busy-at-night are Presence / Other / Mid.
## 2 Rebuilt check (prototype)
Per detector and hour (+ 15-min bins for counts): ONs and fraction ON; data-gap bins (no logged event at the
signal; 34 signals) = missing. Yardsticks: (a) type fingerprint fn x span x band (daytime median < 20 / < 100 / more
per h), median of N / own daytime median, leave-signal-group-out (10 seeded groups), scaled by own ROBUST level
(median ratio over hours with fingerprint >= 0.5); (b) phase mates (same predicted phase, >= 200 ONs), own robust
share of their sum. Per-bin residual: counts = signed Poisson deviance / (1 + c^2 e) (c = extra-Poisson spread per
fn x hour, robust, population); time ON = arcsine difference / MAD per fn x hour. Run evidence = max Kadane sum of
(r^2 - 4) over consecutive bins, each direction. Cut = 99.8th pct of ALL scored detectors per fn x span (pooled
< 500), bad 2x. Variants: type / mate / either / both(agree) x counts / +15 min / +time ON / log-ratio residual (21).
## 3 Coverage at 1-in-500 (best variant: hourly counts, type AND mates agree; 32 flagged = 0.22 %, 7 bad)
  Busy at night 15/133 (11 %; suspect 5/22, bad 10/111); Misses at night 8/91 (9 %); Goes silent 7/75 (9 %);
  Count drops 10/46 (22 %); current raw daily 17/77 (22 %). Union of four 25/289 (9 %). Each check's own strongest
  1-in-500 (top 30 by its score): 7/48, 5/30, 4/30, 7/30. All variants 0.20-0.23 % flagged, union 7-25 / 289;
  KAPPA 0.5-16 changes nothing (+-3).
Arithmetic cap: a 1-in-500 check flags ~30 detectors; the four flag 289. Max possible coverage 11 %.
Flag rate the new score would need to reach 90 % of each check (counts variant): Busy at night 19 % of all
detectors (1 in 5); Misses at night 59 %; Goes silent 39 %; Count drops 19 %; current daily 5.3 %. Median pct of the
category members: 97.8 / 80.4 / 92.6 / 97.2. Same flag count as the union (289): 100/289 (35 %).
With time ON (counts + 15 min + time ON, agree): 33 flagged, union 17/289; flags won by time ON 20 / 33 (61 %), by
counts 13; time-ON flags are mostly Stuck on (6 of 13 new-vs-five are stuck). Time ON lowers coverage of the four.
Reverse: counts variant flags 3 detectors none of the five flag (stuck 2, too-fast 1), 0 flagged by no check at all;
it misses 60 / 77 of the current raw daily flags.
Missed (counts variant), score / cut median and buckets (looks normal < 0.1 cut / weak 0.1-0.5 / near >= 0.5 /
cleared by mates / not scored):
  Busy at night 118: 0.077; 56 / 26 / 7 / 20 / 0. Night ONs 01-05 h of the missed median 160 vs 29 population.
  Misses at night 83: 0.007; 67 / 9 / 2 / 2 / 0. Mates predict 116 ONs 00-06 h, 19 seen (6x fewer), yet inside the
    population spread: night share vs mates varies this much across ALL detectors (a common mode, e.g. video at
    night), so a 1-in-500 tail cannot isolate it.
  Goes silent 68: 0.034; 38 / 12 / 1 / 1 / 13 not scored. Silence median 1.9 h; < 1 h 24, < 3 h 39, starts 21-05 h 24.
  Count drops 36: 0.067; 19 / 5 / 3 / 4 / 5 not scored. Day median 1,096 ONs.
## 4 Verdict (keep / replace)
Busy at night: keep. Misses vehicles at night: keep. Goes silent: keep. Count drops: keep. None reaches 90 %
coverage; what is missed is not borderline (median score 1-8 % of the cut). The dedicated checks flag targeted
patterns (short silences, night ratios vs a partner) that a whole-day population tail treats as ordinary.
Runtime (scoring from 15-min bins, refs stored, all streams): median 8.7 ms, p90 9.4 ms per signal-day = the current
busy-at-night + daily-pattern block (8.7 ms, 141b). Plot review/thresholds_all/crit18_daily_pattern_v2.png.
Tables s142/tab142_variants.csv, tab142_coverage.csv, tab142_reverse.csv, tab142_miss.csv, tab142_runtime.csv.
## Open
- The four checks flag 0.3-0.9 % each, above the 1-in-500 rule applied here; held to the same rule they would shrink
  too. Which share of their flags are real faults is unknown (no labels).
- Fingerprint built in-sample per day type (leave-signal-group-out); spreads (c, MAD) fitted on the same population.
