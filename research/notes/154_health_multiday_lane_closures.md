# 154 Health on multi-day windows: complete failures, fault times, lane closures, timing-plan switches

User go 2026-10-21 (from the ATSPM_Report session): can atspm-detector health replace the report's traffic-anomaly
(TA) detector check? Give it 2-3 days instead of a TA baseline; output fault periods (no "recovered" logic: the app
filters); detect lane closures (own phase, and with / without NEMA pairs 2-5, 1-6, 3-8, 4-7 -- a health rule, exempt
from the phase-anonymous rule); work from 30 min to a few weeks. Detectors = ODOT Model_Outputs.csv (1.2.0, 10-11..13).
Branch health-history (from ab9982f), NOT committed. Code src/atspm_detector/health_duckdb.py; research/code/health154;
work %DC_WORK%/s154 (raw days 9-26..10-13 + 10-18 pm..10-20 re-partitioned by device bucket, out/, ta/, death/).

## Changes (health_duckdb.OPTIONS; every one switchable; all off = 1.2.0, checked identical on 170 detectors)
- silent_limits: silence bad limit >= 2.44 x suspect (Mid 1, Presence 2+, Bike had suspect = bad: 12067 det 35 bad
  for a 5-min gap); a silence under 15 min is suspect at most.
- silent_merge: a quiet run broken by one daytime bin of <= 2 ONs (>= 30 min each side) is one run (03071 det 17 now
  05:50-16:10, was 08:25-16:10); a multi-day window names the day in the text.
- no_recovery_cap: removed "silence that came back to its share = suspect at most" (user: no recovered logic). Guard
  instead: a silence needs >= 1 h of data outside it and >= 5 ONs (else 1-3 ON channels gave ~160,896 expected).
- basic_periods: detectors with no classification (outf path) now get silent / stuck periods.
- signal_silent: every detector silent >= 1 daytime h while the signal serves greens -> each detector's expected =
  its ONs per green outside x greens inside; bad + signal note (08034 9-30..10-09, 08120 9-29..10-02: all inputs lost;
  invisible to phase-mate yardsticks, TA caught 08034).
- drop_end: count drops get an end (recovery = back to >= 60 % of its PRE-drop share, >= 50 expected: night hours
  cannot fake one). Periods "counts dropped" from-to; text "from Mon 08:00 to Mon 18:00".
- drop_no_mates: level check was OFF for a detector with < 2 phase mates (occ_mates NULL): 10035 det 2 (GEH 22.8,
  3/h vs 230) was watch. Now own occupancy fall, or a collapse (< 10 % of expected), confirms. -> bad.
- long_zone: no counts fall while own occupancy held/rose and stays >= 25 % (2B009 det 49 = long zone, median ON 5.5 s,
  occ 56 %, P6 twin det 51 identical; ~60 such Other zones on 2B corridors cleared per day).
- lane_closure (+ lane_pairs): a silence / count drop whose loss reappears on the group (own phase; + paired phase)
  vs a reference of the other phases' clean detectors: loss >= 100 & GEH >= 8, gain 0.5-1.5 x loss & GEH >= 3; same
  clock hours on other days when the window has them, else one like-class partner must take >= half (lane use shifts
  over a day otherwise pass). Twins count once; detectors with their own stuck / fast / spiky / busy finding and the
  signal-wide silence are never candidates. Status lane_closure (category "Lane closure (info)", period "lane closure").
- plan_switch: a silence starting AND ending within 10 min of an event-131 pattern change -> "Moves with timing plan"
  (watch): 03038 even channels 50-64 silent 09:30-18:30 = pattern 23; the odd channel takes the traffic (49+50 = det 2
  exactly every 15 min).
- MAX_DAYS 92: no dtype limit in the health SQL (bins INT32, times BIGINT ns; the 24.8-day int32 was streams.py,
  fixed in 1.0.1); memory ~15 MB per signal-day. 10 min .. 18 d all run (lengths.csv).
- check.py HEALTH += lane_closure. atspm-detector-check 7/7 PASS (no freeze), pytest 26 pass.

## Results (all signals; summary154.csv, changes154.parquet = every detector whose rating moved, per change)
- Flagged % 1.2.0 -> new (nema): 10-19 1 d 4.25 -> 4.08 (+11 lane closures), 10-20 1 d 4.85 -> 4.58 (+21),
  3 d to 10-13 5.99 -> 6.16 (+21), 3 d to 10-20 6.37 -> 6.37 (+22). Bad count per change (4 windows): limits -1..-6;
  merge + no cap + signal-silent +5..+45; drop end +10..+24; no-mates +1..+5; long zone -16..-42; closures -12..-16.
- Lane pairs: own phase 7/18/18/19, + NEMA 11/21/21/22, any phase 36/38/30/30. NEMA adds 3-4; any adds 8-25 more,
  the extras spot-checked look weak (5-min gaps, low-volume evening). Proposed default nema.
- Complete failures (warehouse 200+/day -> 0, 62 detectors, windows ending the zero day and the day after, 110
  judged): flagged 82 -> 105, with a period on the failure day 8 -> 102. 5 not flagged had 1 ON in 72 h.
- TA alerts with raw data (147: 26 channels > 64, 15 no data/class, 106 judged): flagged 60 -> 67 (period on the
  alert day 39 -> 60); 16 lane closure / plan switch (TA reports lane closures as faults: 03071 twice, 05039, 08078,
  2B389, 01010, 03066, 04081, 08080, 08024); 25 ok / watch left for the user.
- What TA still finds that we miss: a whole PHASE changing together (01074 P2 9-28: 5 of 7 detectors to ~0 from 04:00,
  det 42 flat at 60-80 % occupancy) -- the mate yardstick moves with it. A phase-level signal_silent would cover it.

## Open / limits
- One count-drop episode per detector per window (04022 det 49's daily drop shows once).
- Review sheet review/health_review_v9.xlsx (102 rows: your cases, each change, lane closures nema vs any, the 25
  TA-only alerts) waiting for the user's verdicts. README / embedding.md / CHANGELOG text after the review.

## 154b (2026-10-22, user review of health_review_v9 "Your cases" + "Lane closures")
- Lane closure rules from the verdicts: the lane must go NEAR ZERO (<= 15 % of expected, ON < 2 %; agreed closures
  0.00-0.02, rejected 0.43-0.86); a detector of the SAME kind (Advance/Mid, Count/Presence/YR; Other and Bike never)
  takes >= 55 % of the loss (agreed >= 57 %, #46 51 %); same clock hours on days of the same TYPE (weekday / weekend);
  candidates = every run of >= 2 near-zero hours (vs its pre-drop share OR the other phases' traffic: a same-lane
  yardstick closes with it, 03071 det 20) + the silent run; mirrored channels collapse only when the same kind (03071
  det 30 = det 16). Any-phase pairing dropped (user disagreed with nearly all). Result: 31/31 judged rows agree
  (#3 det 20 -> closure 05:55-15:55; #50 -> fault, nobody took P5's traffic).
- Timing plan: own status timing_plan, category "Timing plan" (user). Count drops: text "counts and occupancy both
  fell" / "nearly stopped counting"; periods = every run of >= 2 low hours (a drop each day is listed each day).
- New (user asked): whole phase dropped (phase total <= 20 % of its share of the rest, same hour + day type, >= 200
  expected, phase still served; first all-day-median version flagged 131 detectors, mostly Sunday / 07:00 starts) and
  lockstep (>= 4 non-mirror detectors of a phase with identical hourly counts, 1 + 1 %, >= 3 h: 01074 P2 from 9-28,
  each logging ~690 a day; first 5 %/3-detector version flagged 2,600 -- same-lane zones). 3-day: lockstep 0-1 signals,
  whole phase 3-4, lane closures 14-18. check 7/7, pytest 26, parity holds. Review sheet health_review_v10.xlsx (33 rows).

## 154c-e (2026-10-22, user review of health_review_v10 lane closures)
- The whole near-zero run is tested; a silent run inside it is not a candidate of its own (03055 det 2 Mon 08:55-14:55
  was a 25-min sliver). The top partner must gain in >= 2/3 of the hours and step up >= 1.25 x against the hour before
  or after (10055 det 9: one det 8 spike is not a transfer); an hour run skips the adjacent hour (2B026 det 17 closed at
  20:10, the 20:00 hour already carried part of the move).
- References: a partner's share is taken only where it is clean, against the other phases at those same bins (03071
  det 20's Sunday closure inflated its Mon gain); no same-hour day of the same type (Sunday, 1-day windows) -> the 2 h
  either side, not the whole day (01010 det 16 Sunday morning: the opposite direction's Sunday share is lower); the
  reference leaves out every phase + kind with a candidate (03071 closes lanes both ways at once: P6 16/30 and P2 3/6
  Mon/Tue 18:00, P6 17/20 and P2 2/29 Sun 18:00). Gain limit 0.5-2 x the loss (a Mid counts both lanes).
- 44 judged lane-closure rows: all agree except #101 (03071 det 30, user "?"): the data show det 16/30 at 0 while 17/20
  carry everything vs the lane-wide Mid det 18, Mon and Tue 18:00 -- a closure; Tue found, Mon 19:00 not (top 0.44).
- Charts: lanes that took the traffic drawn thick and on top. Parity 170/170, pytest 26, check pass.
- Review sheet health_review_v11.xlsx: one tab, BEFORE (1.2.0) vs AFTER per changed detector, a few per type.

## 154f-g (2026-10-22, user review of v11 rows 1-21; verdicts_v11.csv)
- all_silences: every silence >= 30 min and >= 50 expected is listed next to the worst one (periods + "also ..."):
  04081 det 32 is silent 11:05-14:00 every day (no ON at all, not stuck); only Tuesday was named.
- no_recovery_cap moved to the verdict: stage 1 keeps 1.2.0's "came back" caps, so pass-1 status (healthy mates,
  yardsticks, queue peers) is 1.2.0's; uncapping there rated mates bad and broke hold queue context (2B313 det 4 lost
  its 15-min stuck; 03066 det 22 got Tuesday presence holds). The rating itself drops the cap for silences AND holds
  (2B334 det 23: 1 h 50 min held ON -> bad); a hold the traffic came back from stays suspect under 1 h (34-min holds
  went bad otherwise). 242 holds go suspect -> bad (median 2 h 8 min).
- A silence is a second finding (-> bad) only when >= 15 min and >= 50 expected (10005 det 19, 11010 det 9: suspect).
- Timing plan within 5 min of the pattern change (user). Lockstep >= 6 h (2B349 P5 3 h at ~24/h was normal).
- Whole phase: reference hours only where the rest of the signal carried 0.5-2 x (03104 Mon P1 vs a half-down
  Tuesday); members <= 25 % of their share (03117 det 8 at 49 % is not in P4's failure).
- All 21 v11 rows now match the call except 03117 det 8 (1.2.0 busy at night), 07032 det 24 (1.2.0 queue hold) and
  03066 det 22 (now a lane closure with det 23); unsure rows left. 44 lane-closure rows unchanged.
- Tree model: ~95 judged rows over ~15 finding types, a quarter unsure -- too few to fit; kept as a regression set.

## 154h (2026-10-22, user review of v11 round 2: 30 rows; out/c7_r2 = that round's code)
- Counts fall severity is 1.2.0's again (median GEH of every hour after the drop); the drop's end bounds only the
  period / text (05049 det 50 suspect, user). 10078 det 16 (user: fine, an advance plateau vs a 2-lane mid) stays
  1.2.0's suspect -- not fixed.
- A night silence is a finding only with >= 50 expected (four "looks fine" / "before" rows -> ok; 368 night watches
  cleared). Busy at night is set aside only by a real silence (>= 0.42): 07054 det 10, 08150 det 4 back to bad.
- No function, < 20 actuations and not held ON -> not_enough_data (12057 det 36; a detector held ON 20 h keeps bad).
- all_silences lists the other silences by the detector's own suspect limit (>= 15 min; night >= 50 expected), the
  3 largest named + "and N more" (the flat 30 min / 50 of 154f was mine, not a rule).
- Not fixed: 10008 det 10 (user: sharp occupancy spikes = bad; both versions ok).
- labels.csv (research/code/health154/labels.py): every review answer, rounds v9 / v10 / v11a / v11b, normalised label.
- Sheet health_review_v11.xlsx round 3: 20 rows, only what this revision changed (1.2.0 / last sheet / now).

## 154j-m (2026-10-22, report integration rounds)
- own_history (2C026 det 44, user-approved): a silence's expected count is capped at 3 x (+10) the detector's own
  count at those clock times on the other days. 154i: a count drop while its time ON jumps >= 3 x (and 3 x its mates')
  is a fault (04019 det 4 / 18, found by the traffic-anomaly comparison).
- Proposed without the user's go and REJECTED / left off (OPTIONS False): busy_min (hid nearly-dead noise zones),
  co_silent_cap and plan_end ("fine either way"). Rule from the user: propose in the review sheet first, never switch
  on unasked.
- no_daily_pattern (approved): new finding "No daily pattern" = counts that do not follow the day while the phase
  mates (no mates: rest of signal) do (mates >= 100/h by day, night <= 30 %); detector >= 1.5/h by day and either
  night >= 80 % of day with profile corr < 0.6, or < 30/h keeping >= 30 % (and 2.5 x the mates' share) at night,
  corr < 0.8. Queue zone exception (user, 2B345 det 11): ON >= 25 % by day, >= 3 x night, counting less by day ->
  neither this nor busy at night. Period = the whole window. busy_ease (approved): busy at night bad only at
  1.5 x its limit. 3-day 10-19..21: ~90 detectors "No daily pattern" (38 newly bad).
- Labels: rounds v11b..v11e in labels.csv.

## 154n (2026-10-26, traffic-anomaly vs atspm-detector on the report run of Mon 26 Oct, window Fri 23 - Sun 25)
- Why 1.2.x missed real drops traffic-anomaly caught (12 checked by hand, clues154 / clues154b: the same-lane stop bar
  held while the detector fell; blank cycles; on-time): its count-drop check (a) is a median hourly GEH over every
  hour after the drop vs 10 -- volume-bound, a 75 % drop on a ~90/h detector rates GEH ~9; (b) compares with ONE
  partner (best 15-min correlation), blind when the partner fails with it (2C070 det 51 + 52, one sensor); (c) a
  sustained rise is not judged at all (only absolute volume, bursts, too-fast gaps, busy at night).
- Weather (weather154, Open-Meteo by curl): Sat 24 dry everywhere, Sun 25 rain 1-17 mm; at 8 of 11 signals the drop /
  rise began within an hour of the first rain (wet loops / video). 2B439 det 22 + 26 (both lane-2 loops 09-17 h,
  recovered after it dried) revised from lane traffic to a fault. Of the new drops on the full network 88 / 96 started
  Sunday.
- Prototype (proto2.py on s154/hourly7) -> package options mate_drop / count_rise (patch_154n, OFF): each detector vs
  EACH phase mate, hour by hour, yardstick = its share of that mate at the same time of day (+-1 h) over the window's
  days (>= 3 days: median incl. its own day; 2 days: the earlier day -- the change is the fault; a 2-day rise must also
  clear the day/night share over all days), plus its own count vs its own count at that time of day (<= 0.75 x drop,
  >= 1.5 x + 10 rise) -- without it a good detector whose mates got false calls / died looked low / high (2B384 det 10,
  14035 det 37, 2B497 det 4). Low hour: mates holding >= half the count weight call it <= 0.5 x; high: >= 60 % of
  mates (>= 2) call it >= 2 x (count, not weight: 2B382's busiest mate rose too). Runs of hours (unjudged hours do not
  break), scored by the run's GEH 15 (suspect) .. 20 (bad), >= 3 h; a drop also needs its time ON to fall (< 0.8 x;
  a queue holds ON) or nearly stop (<= 0.1 x). Queue rule Q2 does not apply to it; a lane closure still explains it.
- Mirror effect: with a median of the other days a Sunday fault made Saturday look like a rise; fixed by the own-day
  median / earlier-day rule above.
- Parity: options off = baseline exactly (60 signals, 1,076 detectors). All 12 missed (+ 2C070 det 51, 2B439 det 22)
  -> bad; the three mate-artefact cases stay ok.

## 154o (2026-10-27, user-approved plan: Stage A self-contained first, Stage B traffic-anomaly medians optional)
Truth set (truth154.csv: labels resolved, 59 fault / 26 fine / 26 closure / 17 not-closure / 21 unsure; eval154.py).
Network: full154.py on raw7, window Fri 23 - Sun 25 Oct (19,602 detectors), runs t_* in s154/out.
- Step 1-2 (mate_clean, mate_noise; patch_154o): best = mate_drop + count_rise + mate_noise (fixed phi 1.5): faults bad
  20 -> 39 / 59, fine kept off 25 / 26 (= today), closures 18 / 26 (= today). mate_clean adds nothing net (OFF).
  Network: bad 549 -> 606 (+57: 35 drops, 22 rises), none lost.
- Step 3 fingerprint (plaus154: PCA 10 + Gaussian mixture by function group x day type on hourly count share, share
  ON, time ON per call; fp_eval154): against the detector's own other days it does NOT separate fault from fine (AUC
  0.49-0.51: labelled fine days are unusual too -- rain, events); against its phase mates the same day AUC 0.84 vs
  0.91 for the rules, adds 2 of the 14 faults the rules miss at no false alarm. Not worth shipping; scoring is cheap
  (< 1 s / network-day) but it needs weeks of history and a fitted model.
- Step 4 confidence: a learned model is not possible on these labels (only 9 fine detectors have any period; the
  truth set is fault-enriched, so its rates are not network probabilities). Instead each period's confidence is
  anchored on its own check's limits: 0.5 at the suspect limit, 0.8 at the bad limit, on log strength (silence:
  expected count; hold: length; run: GEH; burst: GEH; whole-window checks: their score). Event checks (holds, bursts,
  count zone ON) take the detector-level check's confidence when higher. Calibration needs a random sample of
  periods labelled by the user (not yet).
- Period roll-up (option period_rollup, rollup_conf 0.7, rollup_min_day 60; patch_154o_c): malfunction time = union of
  periods with conf >= cut (lane closures / plan switches +-30 min out; only checks that judge the detector and that
  the rules kept as findings after their context); event checks weighted so their bad limit = one day's allowance (scales the
  stuck-on limit by type); bad when >= T min a day on average, or a period >= 2 h (event: >= T weighted) still runs at
  the window end; bad below that -> suspect (whole-window findings -- busy / no pattern at night, low night counts,
  too fast, too many -- keep their rating); an ok detector with a period over the cut -> suspect. A silence >= 4 h
  over the suspect limit has conf >= 0.8 (the rules call it bad, rule P).
  Labels: T 60: faults bad 39 -> 39 (+2 adding up: 2B343 det 31, 04022 det 49; -2 short: 10072 det 2 stuck 49 min/day,
  08404 det 52), closure 2C024 det 5 bad -> suspect, no new false alarms. T 30: +4 / -3 but promotes 2 closures to bad.
  Network T 60: bad 606 -> 556 (89 down, mostly stuck-on holds that recovered; 39 up), ~125 ok -> suspect (lone
  bursts). T 30 moves only ~15 more on the network; 60 kept (the user's figure; 30 promotes 2 labelled closures).
- Stage B (expected_table = traffic-anomaly's prediction, patch_154o_b): each hour vs its expected count x the median
  ratio of its phase mates (0.5-2), not judged when the whole signal is off its usual level (> 2 x or < 0.5 x:
  event / outage / holiday), not counted when its phase mates took the change (lane shift); runs as the mate runs
  (>= 3 h, GEH 15 / 20) into mhrun (source 'expected'); rises only with option expected_rise (OFF: of the 15 rises
  it added on the network the two reviewed look like event / lane-shift traffic). Network: +36 bad (21 drops, 15
  rises) over A. Labels: neutral (39 / 59; the labelled windows mostly start at
  a fault's onset). Its use: faults already present when the window opens (10035 det 2 dead since 10-19 = 'watch'
  today; dead detectors with too few actuations to judge).
- Compute (3-day network, 12 threads, single runs): today 6.7 min; + A 8.1 (+21 %); + roll-up 8.2-8.6 (~+0.2);
  + Stage B 8.5-9.1 (+0.4-1.0, mostly reading the expected table); all on 8.2-8.4 (+25 %). Fingerprint offline < 1 s
  per network-day (not in the package).
- All on vs today (network): bad 549 -> 583 (122 newly bad, 88 to suspect). Review: health_review_v11.xlsx rebuilt
  in place, 15 rows (A 4, roll-up 6, B 5), my call in each; picks s154/pick154o.json, changes s154/pick154o_changes.csv.
- Parity: options off = baseline (full network t_base vs n_base_1023: 0 of 19,602 differ; sample after the patches 0);
  pytest 26 pass, 2 skipped (test_health CHECKS += Counts too high).

## 154p (2026-10-27, user: redo the fingerprint study properly -- "typical-day profiles")
- Data: 15-min counts + time ON per detector-day (fp15_build -> s154/q15): 26 fall days (staging w40, hb, raw7), the
  report's raw store (adds 10-26) and Dec 2-4 2024 = 30 days; staging keeps ~3 days, so no older history / holidays.
  Time ON imputed as atspm's split_failures step2 (ON->ON <= 2 s: ON to the next ON, > 2 s: OFF half-way; OFF->OFF: ON
  half-way; last ON runs to the day end; leading OFF: no time; no cap); a day = >= 84 of 96 bins logged, >= 50 calls.
- Model (fp154): features in time-of-day order -- sqrt volume shape (96, normalised by the day total), sqrt occupancy
  (96), volume level (log10 per lane, x3); per function group (Advance / Stop bar / Mid / Other), no day-type split;
  PCA (90 % var, <= 30) + Gaussian mixture, k by BIC elbow (98 % of the gain); clusters chosen: Advance 8-20, Stop bar
  12-15, Other 6-20, Mid 5-10 (fold dependent); 3 % trimmed; NORMAL = within the cluster's T2, the PCA residual Q and
  a worst-hour run of bin z, one quantile tuned so 95 % of training days are normal. Out of sample by date (fit on
  one half of the days, score the other; fp_run.sh, fp_eval2.py).
- v2 (own pattern): held-out days 5.3 % abnormal (rain Sun 8 %), but Dec 2024 13-28 % (season); faults 17 / 52,
  fine 2 / 18, closures 1 / 26; adds 1 of the 16 faults the proposed rules miss.
- Why it misses: a faulty day usually lands in ANOTHER normal cluster (04006 det 51 home 4 -> 14 / 17, 05042 det 51
  6 -> 3 / 0 ...): it looks like some other detector's normal day. Usual-cluster test (fp_home): +4 faults, but
  flags 20 / 26 lane closures.
- v3rel (relative to its phase mates: 45-min share of the phase count / its day share, occupancy difference, day share
  level): weather / season cancel (Dec 2024 6-8 %, rain Sun 7 %); faults 21 / 52 (+usual clusters 22-23), fine 2 / 17,
  closures 14 / 26; adds 5 of the 16 rule misses -- at the price of ~1,250 rule-ok detectors flagged per 3-day window
  (rules: 549 bad); severity 1.5 x / 2 x: 379 / 243 extra flags, 2 / 0 of the misses. 10028 det 16: the profiles rate
  Thursday (the user's normal period) the least typical; 2B313 det 17: Saturday's bursts the least typical.
- Verdict: calibrated and weather-robust, but far less sensitive than same-day phase-mate comparison and too noisy as
  an alarm; not shipped. Revisit with a year of data (archive the daily 15-min table from now on). Partial-day
  scoring not built (the full-day version does not clear the bar). Compute: fit ~5 min per half (offline), scoring
  seconds.
- atspm-detector health's own time ON differs from atspm's (subagent review): ON->ON > 2 s counted as ON the whole gap;
  OFF->OFF no ON; a leading OFF = ON from the window start with no limit (an ON lost in a dropout can add hours to
  occupancy / count-zone checks). Fix proposed as an OFF option, awaiting the user.

## 154q occupancy (occ_strict) (2026-10-27, user ruling: health must not impute missing events)
- Option occ_strict (False / True / "held"; _strict_ons): continuous ON = an ON + ONs re-logged within 2 s (ON to the next ON: C1
  extension time kept) to its OFF; leading OFF and OFF->OFF add nothing; ON->ON > 2 s = actuation kept, time ON
  unknown; unknown / over a comms gap / open into the signal's own silence at the window end (> 120 s) carry gap =
  true -> out of 5-min time ON (all occupancy users), count-zone-ON-in-green, holds / stuck-on, per-ON stats (C1/C2).
  Unknown time is not "quiet" for a detector's own silence runs (else a lost OFF turns a hold into a silence: v1 gave
  +95 new count-fall findings); the signal-wide input-loss check keeps known time only (else one unknown input voids
  it: v2 lost 979e374f's 5-h all-inputs-lost, 17 detectors).
- Cases (network, Fri 23 - Sun 25, 19,602 dets, 131 M ONs; scratchpad measure.py on the unmodified package): leading
  OFF 4,673 dets (24 %), 102 h; OFF->OFF 4.8 M (3.7 % of ONs, 1,788 dets); ON->ON <= 2 s 4.4 M (kept); ON->ON > 2 s
  3.9 M (2.9 % of ONs, 5,502 dets = 28 %), 15,100 h = 13 % of today's time ON (2,200 dets lose > 25 % of theirs);
  over a comms gap 3,074 islands (already out today); open at window end 605, signal stopped 5.
- Stuck-on risk: 1,484 of 1,809 holds >= 15 min contain an ON->ON > 2 s (only 23 keep a known part >= 15 min).
  Of 567 stuck-on findings, 302 are lost, 275 of them built from ON->ON > 2 s; mostly ONE long ON->ON (median 1 per
  hold), 17 re-log >= 5 ONs inside the hold (stuck input re-logging ON: dd321848 det 38 Count 11 h, det 8 4 h -> ok).
- Network (mate_drop + count_rise + mate_noise on both): bad 606 -> 467, suspect 614 -> 350, ok 17,867 -> 18,259;
  444 change (430 better, 14 worse); lost Stuck on 332, Counts fall 18; gained Stuck on 35 (shared-hold cap lost:
  30feb116 det 22-25, 12-min common hold now bad), Counts fall 12. Truth (19 in window): 2 fine 2B489 det 11,
  10045 det 22 suspect -> ok; faults 15 / 15 bad kept; no label worse.
- Parity: off = unmodified (q_occ_off vs q_occ_pre, 0 of 19,602 rows differ, all columns); pytest 26 pass, 2 skip.
- Compute (4 threads, 4 GB, 3-day network): off 372-382 s, on 383-406 s (~+3 %).
- Variant "held" (coordinator, after the stuck-on result: the controller re-logs ON while an input is held): ON->ON
  > 2 s = still ON to the next OFF (today) unless a signal-wide logging gap overlaps it (then unknown); rules 1, 2,
  3, 5, 6 and both unknown-time choices as True. R4 split: 3,860,966 intervals / 5,362 dets / 14,273 h with the
  signal logging in between; only 2,291 / 441 dets / 834 h overlap a logging gap (scratchpad measure2.py).
- q_occ_held vs q_occ_off: 1 status change (168015d6 det 2, unclassified, "went silent" bad -> ok); 18 reasons
  change: 13 side findings dropped (silences / night misses inside stuck holds that a comms gap now makes unknown),
  1 gained. Examples 1-6 above all back to today's rating; labels (19) unchanged. Compute 368 s vs off 346 s.
- Recommendation: "held" (matches the controller, removes the leading-OFF / signal-stopped imputation, keeps
  stuck-on detection); True hides 302 of 567 stuck-on findings.

## 154r long pulses (long_pulse) (2026-10-27, user-approved idea from SPR 837: ON per actuation vs lane mates)
- Option long_pulse (OFF; _long_pulse, LONG_PULSE): each complete hour, median clean ON (one ON + its OFF, <= 60 s,
  the module's time ON) vs the median of its lane mates' (same phase + function; mates need >= 20 clean ONs and
  median > 0.25 s = not pulse mode), count per lane vs theirs. Hour long = >= 2 x the mates' ON AND <= 0.8 x their
  count; run of >= 3 such hours (unjudged hours do not break it). Score on the run's median ratio: suspect 2 x, bad 3 x;
  a run < 6 h stays suspect and is dropped when its own other hours are >= 0.8 x as long (nothing changed: mates /
  counts moved). Category "Long pulses", family / evidence = counts fall. Periods "long pulses" (o = 11), also in
  pcand / rollup.
- Scope from the data: Advance, Count (normal mode), Mid only. Presence / Other excluded: 2 x+ with fewer counts is
  common there (Presence 394, Other 188 of 3,075 judged dets; queue / timing dominated, mixed zone kinds).
- Not judged: hours in a lane closure / plan switch at the phase (own or gainer, +-30 min); a detector with a mate
  of about its own ON (window median within 1.25 x: a second long zone, e.g. det 49-52 advance zones next to loops);
  model top-class probability < 0.7 (long zones called Advance with p_other 0.3-0.46). All mates long = ratio ~1.
- Network distribution (Fri 16:00 - Sun, raw7; scratchpad lp_hourly / lp_dist; detector window median ratio, mates
  non-pulse, >= 6 judged hours): Advance 2,523 judged: < 1.25 x 2,193, 1.25-1.5 186, 1.5-2 65, 2-3 42, 3-5 15, 5+ 22;
  with counts <= 0.8 x: 2-3 25, 3-5 6, 5+ 18 (no mate twin: 21 / 6 / 11). Count 274 (pulse mates leave 1,199 ->
  274): 2 x+ 8; Mid 141: 7.
- Network (mate_drop + count_rise + mate_noise on both): bad 606 -> 620, suspect 614 -> 625, ok 17,867 -> 17,844,
  watch 142 -> 140. Long pulses on 33 dets (Advance 13 bad / 17 suspect, Count 3 suspect); 25 newly flagged
  (Advance 24, Count 1; 12 bad, some by combining with a borderline silence), 8 already suspect / bad (stuck on 3,
  counts fall 3). Mostly new faults, not re-finds.
- Truth (149 rows, all windows): no status change (faults 39 / 59 bad, fine 25 / 26, closures 18 / 26). A first
  version (no own-history rule) turned closure 2B389 det 16 into suspect (3-h run, ON as in its other hours).
  Labelled dets with the finding: 10028 det 16 (fault, already bad), 2B316 det 6 (unsure, already bad).
- Examples (8, clear and doubtful): s154/long_pulse_examples.csv. Doubtful kinds: 13 x ratios (long zone called
  Advance?), steady long ON in own other hours (design?), normal-mode Count zones.
- Parity: off = unmodified (r_lp_off vs r_lp_pre, 0 of 19,602 rows differ, all columns); pytest 26 pass, 2 skip.
- Compute (4 threads, 4 GB, 3-day network, machine shared): off 5.6-7.9 min, on 5.4-8.3 min (within the noise).

## 154s (2026-10-27) median profiles, SPR 837 review
- Typical-day profiles redone with MEDIANS (user's idea; fpm154.py / fpm_eval.py / fpm_counts.py, out s154/fpm): k-medians
  centres (k 5-12 per group, distance curve flat after 4-6), median-of-bins matching, robust (median-ratio) day scale,
  own vs relative-to-mates, 18 variants, held-out calibrated to 5 % (Dec 2024 ~5 % in every new variant, was 21 %).
  Best (rel, k-medians, sum-of-squares matching): faults 31 / 52, fine 4 / 17, closures 21 / 26, adds 6 of the 16 rule
  misses, ~760 rule-ok detectors flagged per 3-day window. Median MATCHING adds nothing; k-medians centres +4-5 faults;
  robust scale only matters for partial days (keeps them at 4-5 %). Not a health check on its own.
- As support for the count checks (counts dropped / too high, 36 labelled findings): profile-abnormal bins rarely
  overlap the rule's period (median share 0-6 %); gating the rule on profile agreement keeps only 9-15 of 22 faults.
  10028 det 16: Thursday = its most typical day (the user's reading), the abnormal days are Tue / Wed / Sat -- the rule's
  "too high Thu" period gets 0 abnormal bins. 2B313 det 17: normal every day. Use, if any: a second opinion on WHEN
  (shown, not rated); needs stored 15-min profiles the package lacks. Scoring 0.6 s / network-day.
- SPR 837 (S:\Research_Reports_Papers\Detection Health Research Project; dc_work/spr837, spr837_method): the Greenshields
  method does not separate faulty from healthy zones on the study's own data (re-run: 1 / 7 failed vs 1 / 23 healthy
  flagged; their 42 % vs 10.6 % gap is 17 % vs 9.9 %); not used. Drone labels accurate (0.2 % missed, 0.9 % extra on 55
  matched zones) but tiny (7.7 h daytime weekend, minutes per zone), the study's channel map has swaps; the one real fault
  type is adjacent-lane pickup (2C024). Borrowed: ON per actuation vs lane mates -> 154r.

## 154t (2026-10-27) clean yardsticks (clean_yardsticks)
- Problem: 2C046 det 12 (P4 Advance, ~380-400 calls an hour day and night, rated bad "No daily pattern") is the night
  yardstick of its healthy mate det 8 ("Misses vehicles at night" in 3 of 9 windows) and its long_pulse lane mate.
  Stage-1 pass 2 drops pass-1 bad mates only; "no daily pattern" is decided later, so it stays a yardstick.
- Option clean_yardsticks (OFF; health_duckdb CLEAN_YARD, _no_traffic -> ntm / ntraf): from the 5-min counts only
  (before any yardstick check), a detector "does not look like traffic" when its night (01-05) per hour >= 0.5 x its
  day (07-19) per hour, while its phase mates (their day < 20 an hour or none: the rest of the signal) keep <= 0.3 x
  (the no-daily-pattern limit; 0.25 missed det 12 because P4 det 11 also keeps 37 % at night), it counts >= 1.5 an hour
  by day, >= 2 covered hours of each. Such a detector is left out of: the stage-1 night count (partner, phase mates,
  rest of the signal; fallback phase -> signal as before), the hourly night GEH (ghn / ghnn, only for detectors whose
  yardstick held one, others keep theirs exactly), long_pulse's lane mates (lpw twin test and lpf). The partner
  ("tracks") is ignored only for the night checks; count drops keep it (a first version filtered it everywhere and
  added "Counts fall" to 08130 dets 2 / 3, both busy at night).
- Network window 10-23..26: 141 dets flagged at 108 signals (Presence 47, Other 35, unclassified 26, Advance 22, Bike 8,
  Count 2, Mid 1); of them 82 already bad (76 No daily pattern / Busy at night), 51 ok (mostly unclassified high-count
  channels and presence zones that count more at night: bad yardsticks either way).
- Base (mate_drop + count_rise + mate_noise) -> + clean: bad 606 -> 605, suspect 614 -> 613, watch 142 -> 141, ok
  17,867 -> 17,870; 6 rows change, 4 statuses, no new finding. "Misses vehicles at night" 30 -> 28: 08052 det 9
  (mate det 10 Presence busy at night 95 %, bad; 275 vs ~1,461 expected) and 08119 det 16 (silent 25 min, rest of
  signal held P4 det 8 / P7 det 10 presence zones busier at night). Kept, expectation smaller: 01028 det 8 (mate det 9
  No daily pattern; 28,255 -> 13,692 expected, GEH 33.9 -> 11.9, bad for other findings too), 08130 det 4 (mates 2 / 3
  both out -> rest of signal; 4,110 -> 3,928, stuck-on zone, a real miss). Also 11021 det 5 watch -> ok, a det 10 with
  no TSSU in Model_Outputs bad -> suspect (night part gone, silence kept). Of the 4 night findings with a broken mate, 1 goes
  (2 keep a real miss, 1 is 03049 det 53 whose mate det 57 keeps only 34 % at night: not flagged, silence-type finding).
- With long_pulse: bad 620 -> 618 (2C046 det 8 bad -> ok: its only mate was det 12), the same 6 others; Long pulses
  33 -> 32 dets.
- Truth (149 rows, eval154, cy_on / cy_lp vs 154r lp_off / lp_on): no status change (faults 39 / 59 bad, 47 / 59 bad or
  suspect, fine 25 / 26, closures 18 / 26). Two reason texts differ in the order of equal periods only (seen in 154r too).
- Parity: option off = unmodified (y_off vs r_lp_off and y_lp vs r_lp_on: 0 of 19,602 rows differ, all columns; the
  off-path SQL is byte-identical to the pre-154t copy); pytest 26 pass, 2 skip.
- Compute (4 threads, 4 GB, 3-day network, shared machine): off 5.8 min, on 5.6; long_pulse 5.3, + clean 5.2-5.5 (noise).

## 154u (2026-10-27 night) report-side CUSUM of bad minutes, report-side history, long_pulse verdict
- Data: the network on 9 back-to-back 3-day windows 09-26..10-27 (27 days, gap 10-14..17), count options on, frozen
  package s154/src_154q (cusum_runs.py -> s154/cusum_k; CU_PALL=0 = only the periods the rules keep as findings).
  periods_all (every candidate) made the CUSUM alert on fine detectors (silences the rules' context had dismissed):
  fine 6-9 / 24 -> the CUSUM must read the kept periods only.
- Speed: the w40 staging export mixes every signal in every row group; the same network window ran 35-46 min vs 7 min
  on the report's store (sorted by signal). sort_days.py (sorted copies, s154/sorted) -> 5-8 min. atspm-detector input
  should be sorted by DeviceId (worth a line in its README).
- Daily bad minutes x (cusum154.py): union of kept periods with conf >= 0.7, lane closures / plan switches out, event
  checks weighted (a hold at its type's bad limit = 60 min). S = min(cap, max(0, S + x - k)); alert while S > h; days
  without data leave S as it was. Grid (k 15-60, h 0-480, caps; s154/cusum_k/grid*.csv). Without a cap alerts linger
  4-8 days after repair; cap = h + 2k clears an alert after 2 clean days.
- Chosen: k = 30 min/day, h = 120, cap = 180 (alert at once on a day with >= 150 bad min; after ~4 days averaging an
  hour a day; one 2.5 h hold does not). Whole-window findings without a real time period (busy at night -- one 4 h period
  per window --, no daily pattern, misses vehicles at night, or no period at all) are reported directly (cusum_report.py).
  Per report run: rules bad 656 -> reported 440 (CUSUM 393 + direct ~47), nothing added beyond the rules' bad;
  dropped ~257 / run: one-off holds (stuck on 176 / run), silences days ago, unclassified detectors' single short
  holds. Labels: faults 32 / 56 (rules 31; adds 2B343 det 31 and 08034 det 52, loses 2B334 det 23 -- 1-2 h holds every
  few days, ~15 min / day, S peaks 110), fine 0 / 24 (rules 1), closures 2 / 26 (rules 3). Charts of 8 dropped and the
  disagreements: s154/cusum_k/charts (cusum_chart.py). Compute: seconds per run on stored daily minutes.
- History in the report (hist154.py; per detector-day 06-22 h its share of the phase mates' calls and its time ON per
  call vs a TRAILING baseline = median of its earlier unflagged days of that kind, last 14, >= 4; a median of all days
  takes a long fault as normal): share <= 0.5 x, share >= 2 x, ON per call >= 2 x with share <= 0.8 x, mates steady.
  ~26 detectors in a month the CUSUM never alerts on; of the top 8 charted: 2 clear new faults (08071 det 3 merged
  calls since 10-10 -- ON per call 1 -> 20 s, a third of the calls; 2B082 det 50 dead since ~10-14, rated ok because it
  is low the whole window), 2 chronic near-dead (12067 det 25, 08036 det 6), 3 lane-share swings that look like traffic
  (10047 det 12, 04057 det 49, 01074 det 17). Worth an OFF option in the report (needs the daily per-detector table).
- Sensitivity knob = which periods count: conf >= 0.5 (suspect-level too) -> 489 reported per run (+49, all of them
  detectors the rules rate suspect whose periods add up), faults 36 / 56 (+2B439 det 26, 04058 det 16, 04007 det 36,
  03096 det 50), fine 0 / 24, closures 2 / 26, not-closures 4 / 17; charted samples plausible (03078 det 49 Monday
  afternoon at half its usual, 04119 det 5 hour-long hold). k 60 / cap 240: 397 per run, faults 31. h 90 / cap 150:
  468, faults 33, closures 3 (catches 2B334 det 23's recurring holds).
- Month charts revise four v11 calls: 2B127 det 1 (the same weekday 1-2 h peak all month = ramp queue, not a fault;
  rated bad 8 / 9 windows today, ~1,400 bad min a day with the count options), 04024 det 49 (steady low lane, Sundays
  lower), 10004 det 2 (tracks its mate on a congested approach), 05069 det 52 (rain-weekend dip, normal Monday).
  16 detectors have count-check periods on >= half their days (3.6 % of CUSUM alert-days): real chronic faults (2B049
  det 52, ~2,000 calls/h day and night) mixed with recurring traffic (2B127 det 1), presence queue zones (11021 det 19,
  bad in all 9 windows: spiky / no daily pattern) and broken yardsticks (2B009 det 50 vs its flat mate det 49).
  141 of 938 detector-windows with count-check periods (15 %) have a same-phase detector rated bad for no daily
  pattern / busy at night -> clean_yardsticks extended to the count checks (154t follow-up).
- Review sheet health_review_v11.xlsx rebuilt in place (sheet154u.py, picks s154/pick154u.json, 21 rows: C 6, H 3,
  Y 3, L 2, A 4, B 3; month charts; the 154o version backed up as s154/health_review_v11_before_154u.xlsx).
- long_pulse (154r) charted: all 8 examples are steady for the whole month (zone design / lane use: 2B402 det 22,
  03011 det 22, 08007 det 16, 04070 det 51, 04010 det 52, 04066 det 27; 2B010 det 22 equal ON per call) or a broken mate
  (2C046 det 8 vs det 12, fixed by clean_yardsticks 154t). A 3-day lane-mate ratio cannot tell design from a fault ->
  keep OFF; the history version above (a CHANGE in ON per call) is the useful form.

## 154v design proposal (for the user's approval; nothing built)
atspm-detector (finds conditions and when they happened):
- New `health_periods` output (a second DataFrame / table, one row per period): DeviceId, Detector, start, end,
  condition (stuck on, silent, counts dropped, counts too high, spiky, too fast, count zone held ON, busy at night, no
  daily pattern, misses vehicles at night, long pulses ...), severity 0-1 (the period's own strength on the check's
  suspect -> bad scale, today's `conf`), whole_window flag (the finding has no real time of its own), context (lane
  closure / plan switch) -- only periods the rules keep as findings. Replaces parsing the JSON `health_bad_periods`
  (kept one release for compatibility). No `confidence` column (user 2026-10-27).
- Every finding gets a row: whole-window findings span the window with whole_window = true.
- The detector table stays (status, score, reason) for standalone use; period_rollup and periods_all stay research
  options (OFF) and are not needed by the report.
atspm-report (decides what to alert and when):
- Store per run, per detector-day, the weighted bad minutes from the periods table (latest run wins for a day) in the
  report's DuckDB warehouse next to the existing timeline tables; also a daily per-detector stats row (calls 06-22 h,
  share of phase mates' calls, ON per call) for the history checks.
- CUSUM per detector on the stored days: k 30 min/day, h 120, cap 180, severity >= 0.7; event checks weighted (a hold
  at its type's bad limit = 60 min). Alert while S > h. A whole-window finding the rules rate bad alerts at once.
  The existing ongoing-issue suppression then handles repeats.
- Optional (OFF): history checks from the daily stats (share / ON per call vs the detector's trailing baseline).
- Compute: the CUSUM and history steps are seconds per run; storage ~20k rows a day.
- clean_yardsticks="all" (follow-up): everything True does, plus the count checks: the hourly yardstick gh (count drop
  GEH, night GEH; partner / phase mates / rest of the signal without the not-traffic ones) and the mate_drop /
  count_rise / mate_noise mate pairs (mhp). Partner: dropped too, but a detector that is itself not traffic keeps its
  own yardsticks (08130 dets 2 / 3 track each other: no new "Counts fall" there, the clean form of the exception).
  True output unchanged (y_on2 = y_on); off SQL byte-identical.
- Network (base -> "all"): bad 606 -> 603, suspect 614 -> 613, watch 142 -> 141, ok 17,867 -> 17,872; 6 status changes
  (the 4 of True + 2B365 dets 36 / 40 P8 Presence bad -> ok: "Counts too high 3.3 x Sun 10-14" against P8 Other dets
  23 / 42 / 46, 62 % of their day level at night), 12 reasons; no new finding. Counts fall / too high 243 -> 240
  (also 01028 det 8 "too high 2.7 x Sun 20-23" gone, mate det 9 No daily pattern). Of 28 count-finding dets with a
  bad NDP / busy mate: 15 have that mate flagged, 3 findings go, the rest stay with smaller numbers (2B439 dets 22 / 26
  fall 30 -> 31 %, 2B382 det 22 rise 4.9 -> 6.1 x, both labelled faults) or the mate is one of many (03038 P2: 12
  mates, det 54 ~5 an hour); 13 broken mates are not flagged (night 0.27-0.46 x day, or the judged det is busy too).
- 2B009 det 50: in this window no count finding, only "Misses vehicles at night 4,069 vs ~6,964" (suspect, kept):
  mate det 49 keeps 26 % of its day level at night here (340 / 88 an hour), under the 0.5 limit -> not flagged. A
  time-ON test (60 % ON) would be needed for it; not tried.
- With long_pulse: bad 620 -> 616 (+ 2C046 det 8); truth (cy_all, cy_all_lp): no status change, 4 labelled reasons
  move numbers only (03104 det 3 6.5 -> 5.9 x). Compute: y_all 6.1 min, y_all_lp 5.4 min (off 5.8 / 5.3).
- User ruling 2026-10-28: lane closures are informational, not faults -- the report lists every closure >= 1 h on
  every run (repeats too), no CUSUM, no alert suppression. atspm-detector finds closures of >= 2 judged hours today;
  a 1 h minimum needs a detector change (to test). The report-side history checks are the stand-in for
  traffic-anomaly's knowledge of a detector's own normal (154o Stage B's only use was faults present when the window
  opens -- the same cases). Speed: health() reads the input once per group of signals; unsorted parquet makes every
  group scan the whole file -> fix in the package: copy the input once into a temp table ordered by DeviceId.
- The CUSUM can only use what the rules find: of 56 labelled faults the rules miss 24 in the month windows (14 rated
  ok, 8 suspect, 1 bad without a kept period, 1 not enough data) -- next work item.

## 154w (2026-10-28, user rulings on 154u-v)
- Approved: report running total (CUSUM) counting suspect-level periods too (conf >= 0.5), NOT overriding the report's
  existing 21-day suppression; repeat-rule test tuned to sit at the edge of 2B334 det 23; periods table (with
  DeviceId); occ_strict "held" ON (back-to-back ONs are normal = still ON; OFF->OFF = missing data) and
  clean_yardsticks ON; remove long_pulse and every other unused option from the package.
- Rejected: report-side history (the user's point: a baseline that absorbs a long fault alerts on the repair; and
  improvements belong in atspm-detector, as optional inputs or built from a long window). Its two examples did not
  hold: 2B082 det 50 failed ~10-14 inside the 10-14..17 hole in the month runs (daily runs would see the onset);
  08071 det 3 is rated suspect / bad by atspm-detector in all 9 windows (stuck on 25-35 min/day, under the allowance).
- Lane closures: the >= 2 h candidate minimum was mine (154b), not a rule; closures exist to keep short closures out
  of the failures -> test 1 h (15-min steps). Speed: the report's raw store is already written ORDER BY DeviceId,
  TimeStamp (odot_ingest._store_raw_events); only the research copy was unsorted. Test health() without batching.
- Review sheet v11 rebuilt without the H / L rows (pick154w.json, 16 rows).

## 154x (2026-10-28) the "Friday evening" problem (recurring traffic flagged by the count checks)
- User decisions: drop health_bad_periods JSON, atspm-detector 2.0.0; keep the expected_table (traffic-anomaly)
  option until this problem is solved; the user reviews the sheet only after brainstorming ends.
- 154s profile gating re-read: the 22 are user-labelled faults (16 counts dropped -- most on rainy Sun 10-25, wet
  loops -- and 6 counts too high). The profile rarely confirmed them because it needed single 15-min bins at |z| >= 3
  (coarse; low-volume bins never get there), NOT because long faults became "usual" (my claim to the user was wrong).
- recur154.py (first try, history only for count findings): excuse a finding when the detector's share of ALL its
  phase mates at the same weekday + clock hours matched in >= all-but-one of 2-4 earlier weeks. Excuses 2,278 / 2,539
  findings and 15 / 23 labelled faults ('clean' = earlier weeks without a finding: 543, still 13 / 23) -> fails.
  Causes: the combined-mates share is not the rule's own comparison (share 1.0 where the mates were empty); a
  one-sided "as high / as low" test passes too easily. 2B127 det 1's findings do not repeat week to week (rises and
  drops at different hours) -- a poor example of the problem.
- Next version to test: replay the rule's own comparison (same detector-vs-mate ratio, same hours) on the earlier
  weeks, two-sided match, >= 3 of 4 earlier weeks, earlier weeks not flagged; needs >= 5 weeks of input.
- recur154b.py (second try: the count checks' own detector-vs-each-mate ratio, two-sided within 1.33 x, mates holding
  >= half the calls, all of >= 3 earlier same weekdays): 36 of 141 judged findings excused, mostly window-edge days
  (Mon 10-26 05-10 h in a Sat-Mon window, Sun 10-18 evening in a Sun-Tue window). End to end (CU_EXCUSE in cusum154,
  cut 0.5): report 489 -> 488 per run, faults 36 -> 35 / 56 (loses 2C070 det 52, its mate failed with it); W 2:
  486, 34 / 56. The running total already absorbs nearly all of them -> not worth building.
- Count checks' stake (CU_DROPKIND=drop,rise): without them 443 per run (-46) and faults 24 / 56 (-12), fine 0 / 24,
  closures 0 / 26 (the 2 labelled closures that alert do so through count findings: 03071 det 3, 10090 det 16).
- 2B127 det 1 is LIKELY A FAULT (my ramp-queue call was wrong): 20-50/h vs mate det 2's 300-600/h all day, 0.1-0.3 s
  pulses vs ~0.8 s, counts properly only at the slow daily peak (misses vehicles at speed). 04024 det 49 differs
  (normal ON per call, a light lane).
- Repeat rule (repeat154.py, >= M bad min on >= N days in D days): every setting that catches 2B334 det 23 adds
  120-290 alerts per run and newly alerts 1-3 of 24 fine (gains 2-4 labelled faults) -> left out; 2B334 = known miss.
- Pulse clue (pulse154.py): the detector's ON per call vs its mates during a count finding vs the rest of the window
  changes >= 1.5 x in 4 / 14 labelled-fault findings and 187 / 602 unlabelled (31 %) -> the AVERAGE pulse length
  does not separate faults (2B439 wet loops lost calls with normal pulses). Distributions (pulse / gap quantiles from
  raw events) untested. Sheet rows updated (2B334, 2B127, 2B082).

## 154z cleanup (2026-10-28, user-approved removals and defaults)
- Removed from health_duckdb (options, code, constants, SQL, docstrings): expected_table input of health / health_sql /
  build and option expected_rise (154o Stage B: _expected_hours, EXPECTED_HOURS, mhrun source 'expected', md_src /
  mr_src); long_pulse (154r: _long_pulse, LONG_PULSE, key "long" / category "Long pulses", lp* steps and text);
  busy_min, co_silent_cap, plan_end (rejected 2026-10-22); mate_clean (mhmask); period_rollup, rollup_conf,
  rollup_min_day, periods_all (_periods_all, _rollup, pcand / rst); the occ_strict=True variant (ON->ON = unknown).
  health_duckdb 3,021 -> 2,651 lines (427 removed, 57 new).
- Defaults: occ_strict is a bool, default True (= the former "held": ON then ON = still ON unless a signal-wide
  logging gap lies between; OFF then OFF = the second OFF ignored; leading OFF dropped; nothing imputed); False = the
  1.2.0 imputation. clean_yardsticks default True ("all" kept as an option). mate_drop / count_rise / mate_noise
  stay False. OPTIONS_OFF (all False, lane_pairs None) = 1.2.0 as before.
- Parity (60 random signals, 10-11..13, 1,123 detectors, all columns): old OPTIONS_OFF = new OPTIONS_OFF 0 differ;
  old defaults = new {occ_strict False, clean_yardsticks False} 0 differ; old {held, True} vs new defaults, old
  {held, True, counts} vs new {True, True, counts}, and the "all" variant: 1-2 health_reason texts differ (a "~3"
  vs "~4 an hour at night" rounding in No daily pattern, 51971816 det 38, 86fa66df det 11) -- the old code alone
  flips the same two between two runs (DuckDB parallel sums); a re-run of the old code matched the new exactly.
- pytest 26 pass, 2 skip (test_health CHECKS -= Long pulses); atspm-detector-check 7/7 PASS. ATSPM_Report src: no
  reference to any removed name. Research scripts under research/code/health154 that pass removed options
  (cusum_runs periods_all, ...) run only against their frozen src copies.
- Follow-up (user rule: no unused code): the unused co_silent measure (cs0 / cs_1-2, drx column) removed too -> 2,644 lines; vs the post-cleanup outputs OPTIONS_OFF, defaults and defaults + count options 0 of 1,123 rows differ; pytest 26 pass, 2 skip; check 7/7 PASS.
- User review of v11 (16 rows, labels round v11u): agrees with all charted rows. Open points: 11021 det 19 (presence
  queue zone, rated bad all 9 windows) = normal per the user -> extend the queue-zone exception (2B345 det 11 rule) to
  spiky counts (to test, OFF). 10035 det 2: in the 10-18 window the healthy Sunday evening is called "counts too high"
  because 2 of 3 days are dead and the yardstick (median share over the window's days) is the dead level -- the
  mirror effect of 154n again; 37 of 395 detector-windows with "counts too high" also have "counts dropped" on other
  days -> fix to test: no rise on a day when the other days carry a drop. 08034 det 52: stuck ON 30 Sep, signal dark
  1-11 Oct, back working 12 Oct; the alert holds through days without data and ends 13 Oct (sheet text "clears by
  20 Oct" meant the score reaching 0).

## 154y (2026-10-28) the user's fingerprint idea, bounded test (pulse_build.py -> s154/pq, pulse_eval.py -> s154/pq_eval)
- Features per detector-hour (06-22 h, >= 20 pulses) from raw ON / OFF, nothing imputed: pulse-length 10 / 50 / 90 %,
  share < 0.15 s, share >= 2 s, gap 10 % / 50 %, share of gaps < 0.3 s, count; each relative to the phase mates the same
  hour (mean of the others). Signature = per detector, hour block and day type, median / MAD over the month's days
  outside the judged window; hour score = largest robust |z|; window score = 3rd largest hour (S_*), or per-day medians
  (D_*). 7,959 detectors with mates; labelled windows scored: 27 fault, 6 fine, 25 closure.
- As a check of its own: no. Shape change S_shape AUC fault vs fine 0.60 (D_shape 0.47); at a network top-2 % limit
  (~150 detectors a window) 2 / 26 faults, 1 / 11 of the faults the report misses. Volume change S_vol 0.67. Lane
  closures score high on shape (few, odd vehicles).
- As support for count findings (count_findings_shape.csv, median shape |z| over the finding hours): recurring
  weekly-pattern findings (recur154b, proxy for traffic) >= 2 in 14 / 79, >= 3 in 4 / 79; labelled-fault findings
  >= 2 in 7 / 14, >= 3 in 4 / 14 -- and the low ones are mostly the healthy-day "too high" halves (10035 det 2 10-18,
  05042 det 51 Sat) or co-failed mates (2C070 det 52). Real drops: z 7-108 (2B439 wet loops, 10035 dead, 03104).
  A modest confirm signal; not enough to excuse a finding (half the labelled ones would go).
- Flicker vs mates (share of pulses < 0.15 s >= 0.5 while the mates <= 0.1, in >= half of >= 50 hours): 34 detectors,
  most at ~70 % all month (Unknown 16, Radar 6, Loops 5, no config 7) -- steady, so a sensor's output style or a
  standing fault; the 3-day rules cannot tell which. 2B127 det 1 is not among them (it flickers only at its peak, 27 %
  of hours). Question for the user.
- 2B127 det 1 (the case that prompted it): its evening "peak" is ~90 % pulses under 0.15 s -- flicker, not vehicles.
- rise_dead_days (option, OFF; _mate_hours: mhrun0 -> mhday -> mhrun): no rise on a day when another day of the
  window has <= 10 % of its 06-22 h count while the mates kept >= half their traffic. rise_test.py (107 signals with
  a rise next to a drop or a labelled fault / fine, 9 month windows, live src, count options on): 15 detector-windows
  change, 20 rises removed, the real drop / silence kept in every one; 1 status change (10023 det 11 5-7 Oct bad ->
  suspect, the report still alerts on its silences). 10035 det 2 and 10090 det 16 lose the wrong rise. The other ~22
  rise-next-to-drop windows keep their rise (the other days not near zero). pytest 26 / 2 with it off.
- Single pass (batch_test.py, 11-13 Oct network, parquet read directly, 10 GB / 8 threads, a 5 GB job alongside):
  grouped (max_events 4 M) 5.1 min, single pass 19.5 min; 40 rows differ in health_reason / health_signal_note
  wording only (not inspected; likely the parallel-sum rounding seen in the 154z parity runs). Keep the groups.
- Review sheet health_review_v12.xlsx (sheet154z.py): 3 rise rows + the pulse-mode question (2B033 det 11).

## 154aa 2.0.0: defaults, periods table, closure_1h, pulse_mode (2026-10-28; src backup s154/src_pre_v2)
Task 1 / defaults (user-approved 2026-10-28): rise_dead_days, mate_drop, count_rise, mate_noise -> True;
clean_yardsticks -> "all". OPTIONS_OFF still = 1.2.0. pytest / check pass after each step.
Task 2 / periods table (health(..., periods=True) -> (detectors, periods); health_sql(..., periods=True) -> two
queries; predict(..., return_periods=True), CLI --out-periods; charts.health_chart(periods=)). health_bad_periods
gone from health / predict (reference parquets: column dropped with DuckDB, nothing else moved), _tstr removed.
Columns DeviceId, Detector, start, end, condition, severity, limit_s, in_closure, whole_window. severity / in_closure
= 154o's pcand conf / lc formula (_conf; silence >= 4 h over the suspect limit >= 0.8), limit_s = bad_stuck for
holds (pcand lim) and, new, 60 x bad_count_on for count zone held ON and bad bursts x 10 min for spiky counts.
Findings of a suspect / bad detector with no period get a window row (pwhole: severity from the finding's score,
>= 0.8 when the rules rate it bad); busy at night / no daily pattern / misses vehicles at night are whole_window.
Unclassified detectors (outf): silence / longest hold as before, other findings as window rows, severity from the
stage-1 score, no limit. _Q.script takes several finals (temp tables kept for both).
- Parity (v2_run.py / v2_parity.py, 60 random signals 11-13 Oct, 1,186 dets; pre = src_pre_v2 + rise_dead_days):
  (a) detector table = pre minus the column: default 0 rows differ without periods; with periods 1 reason in 1 of 3
  runs ("~26" vs "~27 at night", the parallel-sum rounding of 154z); count options 0 differ. (b) default: 352 JSON
  periods all matched (start, end, what), 0 missing, +4 window rows ("misses vehicles at night"); count options 355
  matched, +4. Every suspect / bad detector (64 / 68) has a row. severity / limit_s / in_closure vs the 154o candidates
  (src_pre_cleanup periods_all, rise_dead_days off): 352 / 352 and 355 / 355 found, max |diff| 0.000, 0 differ.
- Docs: CHANGELOG 2.0.0, README (periods, statuses, sorted input), docs/embedding.md. CITATION.cff still 1.2.0.
Task 3 / closure_1h (OFF): lcq = near-zero quarter-hours (covered, >= 2.5 expected, <= 15 %: yardstick hour spread
over its quarters as the other phases' traffic is, or the other phases' quarter traffic x the hourly share), runs of
>= 3 (any closure >= 1 h holds 3 whole quarters) -> extra candidates 'q'. They only ADD candidates: the references
(lcan of members, own other candidates, the x set) ignore them, a silence inside one stays its own candidate (plan
switch test), overlapping winners -> the longer. First versions let them into the references and lost closures
(03008 det 13, 03071 det 30, 03038 det 56 timing plan) -- fixed. Off path byte-identical in output (sample 0 differ).
- Truth (c1h_truth.py, 144 labelled rows on their windows): 1 status change, 08111 det 16 closure suspect -> closure;
  closures 18 -> 19 / 26, not_closure 0 / 17 both, no fault or fine changes status. 05035 det 11 (labelled FAULT)
  stays bad but its Monday drop becomes two closures (det 12 takes the traffic) -- flagged; 9 closures' periods move.
- Network (c1h_net.sh / c1h_net_eval.py, 9 month windows, 164,504 det-windows): closures 188 -> 213 (29 new on 28
  dets, 0 lost), 16 silence / count-drop periods explained, 21 status changes: 18 to closure (5 bad, 3 suspect, 7
  watch, 2 ok, 1 timing plan also called closure), 1 bad -> suspect (10055 det 9, doubtful), 2 ok -> bad "No daily
  pattern" not caused by the option (bad with it off too when re-run alone: a borderline flip). Bad 5,880 -> 5,876.
- Charts s154/v13_charts/c1h_*.png (c1h_chart.py: 3 days, detector vs same-kind partners on its and the paired phase).
- cusum_chart.py: % time ON (and s ON per call) now from raw events, nothing imputed (hourly_on), noted in the title.
Task 4 / pulse_mode (OFF; PULSE n 200, share 0.8, s 0.2): predicted Presence, >= 200 complete calls (one ON to its
OFF, not re-logged / open / lost), >= 80 % of them <= 0.2 s, median < 0.2 s -> suspect at least (score <= 0.69, a worse
rating kept, note added), category "Configuration (suspect)", periods row "pulse mode" (whole window, severity 0.5).
"Shorter than 0.2 s" literally (< 0.2) misses the user's own example: the log has 0.1-s ticks, so 2B033 det 11's
~0.13-s pulses log as 0.1 or 0.2 s (72-77 % under 0.2 in every window) -> the share counts <= 0.2 s (to confirm).
- Network Sat 24 - Mon 26 Oct (count options, clean "all"): <= 0.2: 10 dets (9 ok -> suspect, 1 already bad): 03104
  det 23 / 24, 04019 det 19 / 20, 08052 det 10, 08114 det 13 / 27, 2B033 det 11, 2B086 det 28, 2B489 det 4; config:
  Radar 5, Unknown 3, Loops 2; configured Presence 6, Count 3 (04019 det 19 / 20, 2B489 det 4: the model calls them
  Presence; a count zone in pulse mode is normal -> likely false flags), 1 not in the config. With < 0.2: 1 (08052 det 10).
  List s154/v2/pulse_flags_1024.csv. Sample 60 signals: off path 0 differ; tests test_pulse_mode, test_new_option_defaults.

## 154ab performance review (health_duckdb / ped_health_duckdb, DuckDB 1.5.2; code s154/perf/*.py)
- Sample 60 signals 2026-10-11..13 (29.4 M events, 8 groups of 4 M, 852 statements): 32.9 s at 1 thread, 16.0 s at
  4, 16.1 s at 10. Flat profile (4 threads): de 1.55 s (10 %), xx0 1.16 (7 %), sst 1.15 (7 %), ev 0.84 (5 %), o10s
  0.71, pcs 0.62, lcrows 0.55, g5 0.45, lcres 0.43, final detector query 0.40; no step above 10 %. Event-level steps
  scale 2.6-4.7x from 1 to 4 threads; the ~100 small steps 1.0-1.3x (one core each). Fixed cost per group ~0.45 s
  (binding / planning ~117 statements; one signal, 2 h) = ~24 % of the sample run. The machine's speed varied ~2x
  during the session: compare only runs made side by side.
- Static review: no regex / string work per event, one scan of the events per group (ev), window functions share one
  sort each; sst's ASOF join casts ph (INT -> DOUBLE, cheap). Correlated NOT EXISTS (lcrows, lcres, xf) decorrelate to
  hash joins, tens of ms. Nothing clearly slow at the SQL level -> no SQL changed. ped_health scans the events twice
  per group (ev + hours: any code counts as logging) -- by design, left.
- The ~2-core ceiling is per process, not the share: one process 1.7-2.2 cores at 10 threads; groups run at once on
  cursors of one process 2.3-2.6 cores (1.2-1.6x); max_events 16 M 1.4x (peak 3.4 GB vs 1.2). Separate PROCESSES
  scale: 300 signals 241 s in one process vs 48 s in 4 processes x 3 threads; whole network (sorted, 3 days) 13.4 min
  in one process (10 threads) vs 2.6 min in 5 processes x 2 threads x 2 GB over contiguous DeviceId shards (5.2x).
  Trivial queries on cursors of one process do use 10 cores, so the cap is contention inside one process on this
  workload (likely the Windows heap allocator: DuckDB uses jemalloc on Linux only) -- inferred, not proven.
- Shards vs one process, whole network: periods identical (4,645); 4 of 17,666 reasons differ, all run-to-run
  nondeterminism already there (repeat runs of one process flip them): the order of two lane closures in a reason
  (cats: string_agg(txt ORDER BY ki), every closure has ki 99 -> needs a tie-break) and the drop hour of 8dc2614a det
  33 (argmax over float sums, the 154z class).
- Report (odot_detector_health.py, read only): a VIEW over 24 three-hour parquet files on the S: share (sorted by
  DeviceId; join pushdown prunes row groups). 58 signals: 29.7 s from S: (ev 10.9 s = 1.36 s a group) vs 17.6 s from
  a local copy of the same files; copying the 3 days (1.7 GB) took 21.6 s. Recommendation: copy the days to local disk,
  then run health in 4-5 processes over contiguous DeviceId shards (threads 2, memory_limit 2 GB, own temp dir each),
  concatenated in shard order.
- DuckDB 2.0.0: arrow lambdas -> `lambda x:` (7, health_duckdb.py); pipeline.py relation .fetchone() and
  execute().fetchone() (2.0 leaves the result open: ResourceInUse) -> fetchall()[0] (3). Sample: 2.0 = 1.5.2 (0
  differ but the ~26/~27 flip), periods identical. Tests: 1.5.2 33 passed 2 skipped, atspm-detector-check all passed;
  2.0 16 passed, the 17 predict tests need onnxruntime (not in that venv).

## 154ac (2026-10-29, user review of v13: closure_1h approved + fault-day rule; pulse_mode presence only; DuckDB 2.x)
- Rule (user, v13 row 5, 10055 det 9): a closure on a day when the detector has faults of its own before AND after it on
  the same calendar day is part of the fault (LC_FAULT_DAY sides 2, all closures; inside closure_1h, so OPTIONS_OFF
  stays 1.2.0). Faults (lcfp): silences >= its suspect limit (silall, the main drop), mate_drop runs (>= 2 h, GEH >= 15),
  counted stuck-on holds; any overlapping a closure / switch candidate of the detector is explained and ignored.
  Before = ends <= closure start, same day as the start; after = starts >= closure end, same day as the end.
- Labelled rows (c1h2_truth.py, src_154ac; variants b2 both / e1 either / bq both, quarter-hour candidates only):
  0 changes in all three (closures 19 / 26 kept, 2 as timing plan).
- Network, 9 windows (c1h2_net.sh 3 processes x 4 threads, c1h2_net_eval.py; off / plain on reused from s154/v2/c1h:
  that code differs only in the lambda syntax): both sides: closures 188 -> 211 (27 new, 0 lost vs off), 2 removed vs
  plain closure_1h, both new 1-h ones: 10055 det 9 Sun 13:00-15:00 (suspect -> bad, = off) and 12007 det 39 Tue
  07:15-08:45 (stays suspect; silences 05:55-06:25 / 09:30-09:50). 0 existing (2 h) closures touched, so applied to all.
  Either side: 192 closures, 21 removed (10 existing, 11 new incl. the approved 03099 det 42, 2C036 det 35, 10089 det 22),
  14 status changes, bad 5,885 -> rejected. Other reason diffs vs plain on = run-to-run order / rounding (154ab).
  Bad: off 5,880, plain on 5,876, both sides 5,877. The pre-run screen on the periods tables (screen.csv) matched.
- closure_1h and pulse_mode default True (test_new_option_defaults pins them and LC_FAULT_DAY).
- pulse_mode: user ruling "count zone, yellow-red and advance zones are allowed to be pulse; presence should not be".
  Still predicted Presence; new optional detectors column configured_function (health() looks for it; health_sql /
  build configured=True): Count / Yellow_Red / Advance* (Advance_Presence too) configured -> never flagged; NULL or no
  column -> the predicted function alone. Tried first: configured function decides outright -> 28 flags on 24-26 Oct,
  21 of them configured Presence but predicted Count (04039, 04091, 04092, 04016 ... 5 signals of 04xxx) or Advance
  (2B099 det 18 / 23): likely stale configs -> not adopted (pulse_flags_1024_cfgdecides.csv), for the user.
- New list 24-26 Oct (v2_run --cfg, src_154ac2 defaults; pulse154ac.py -> s154/v2/c1h2/pulse_flags_1024_v2.csv): 7 =
  154aa's 10 minus the 3 configured Count (04019 det 19 / 20, 2B489 det 4): 08114 det 13 / 27, 03104 det 23 (bad
  already) / 24, 08052 det 10, 2B033 det 11, 2B086 det 28 (not in the config). 6 ok -> suspect; nothing else moved.
- DuckDB: pyproject duckdb>=2.0 (no BOM). 2.0.0 venv (+ onnxruntime 1.31.1): pytest 31 passed, 4 skipped (atspm not
  installable there: Windows path length; 2 of the 4 also skip on 1.5.2), atspm_detector.check all passed. 1.5.2: 33
  passed, 2 skipped, check passed.
- Sheet: review/health_review_v13.xlsx in place (backup s154/health_review_v13_before_154ac.xlsx; sheet154ac.py): row 5
  result, rows 6 (12007 det 39), 7 (03104 det 24 pulse), 8 (04019 det 19 no longer flagged). Charts c1h2_*.png, pm_*.png.

## 154ad performance (2026-10-29; DuckDB 2.0.0, pandas 3.1.0, 12 threads; harness scratchpad det/run.py, sample runs)
- Question (user): can health() be made fast in ONE call, or should the shard-processes idea of 154ab go into the
  package? Answer: into the package. `health(..., workers=N)` (default 1): N worker processes take the signal groups
  in order; the connection's threads and memory_limit are shared out (threads = N: one thread each, the most work
  per core). The window's events (allowed codes, inside each window, with their group), the windows and the detectors
  table are copied ONCE into a DuckDB file under the connection's temp_directory while the workers start; each worker
  attaches it read-only. Results come back pickled (each worker's own .df(): same dtypes) in group order. A worker
  runs health_duckdb.py as a script (python -P <file> <job dir> <k>): importing the whole package (pipeline,
  atspm_export) cost ~0.7 s more per worker, and more when 12 workers read it from S: at once (a check measured the
  60-signal sample at 13.3 s with the source on S: vs 9.7 s from a local copy, with the package import).
- Tried in one process (all on the 60-signal sample, 8 groups of 4 M): groups on 6 cursors at once 14.1 / 12.7 s vs
  processes 11.8 / 11.5 vs one call 28.8 / 23.8 -- DuckDB 2.0 uses ~6 cores this way (1.5.2: 2.3-2.6), but erratic:
  with 2 M groups 19.8 then 87.8 s, peak 11.5 GB on a 6 GB limit -> not adopted. Smaller groups (2 M) are slower in
  every mode; 8 M gains nothing with workers; max_events stays 4 M. With 30 groups of 1 M: one process 58.7 s,
  4 workers 28.3 / 20.7, 6 workers 21.0 / 19.1, 12 workers (1 thread each) 17.6 / 17.0 -> one thread a worker.
- The one-time copy: partitioned parquet (first version) 51 s of a 152 s whole-network run; a DuckDB table 12 s
  (sample, 29.6 M rows: parquet by group 1.9-3.1 s, DuckDB file 1.1-1.4 s, either sorted by group 3-6 s).
  The flat profile of 154ab is unchanged (no SQL step > 10 %): no step merged, nothing else in the SQL changed.
- Before / after (median of 2, side by side under the heavy-job lock, final code with the package source read from
  S: as the scheduler does; before = the frozen source, one call, 12 threads; after = workers=12, 12 threads, memory
  9 GB; other agents' tests ran outside the lock, so the machine was busy -- compare within the batch):
  60 signals 11-13 Oct, sorted DuckDB table: 27.4 s (29.1, 25.6) -> 10.5 s (11.2, 9.7), 2.6x; source on local disk
  10.7 s (10.8, 10.6): the same; workers=1 25.7 s (= before).
  58 signals 26-28 Oct, a view over the report's 24 three-hour files on S:: 29.8 s (31.7, 27.8) -> 10.2 s (10.0,
  10.3), 2.9x; the same files copied to local disk: 28.3 s (31.1, 25.5) -> 10.9 s (10.9, 10.8). S: = local (the files
  had been read many times: likely cached; 154ab saw 29.7 vs 17.6 s). After, workers=12 reads S: once. Peak memory
  (all processes) 5.5-6.6 GB vs 0.9-2.5 GB.
  Whole network 889 signals 11-13 Oct (parity runs, not benchmarks): 125 s with workers=12 and the source on S:
  (busy machine; 107.5 s earlier from a local copy with the package import; peak 8.6-8.9 GB) vs 7.3 min in one
  process (DuckDB 2.0, one run; 1.5.2: 10.6 min). Production estimate (not measured): 2-3 min from S: (the 3 days
  read once, ~20-60 s) instead of 15-22 min.
- ped_health: 60 signals x 2.3 days (22.7 M events) 0.6-1.1 s (6-12 threads); whole network estimate 10-30 s plus
  reading the events -> left single-process. Fixed: periods=True failed (no column lim / nothing to concat) when no
  input had a stuck-on or dead finding (it did on that sample); test_periods_without_a_limited_finding.
- Determinism (needed for exact parity): ties broken in SQL -- cats (lane closures of one reason in time order, o =
  a_ns, then text), silence runs / silmore (round(lam, 6), then earliest), phase-drop / lockstep
  / transfer / plan-switch picks (+ ph, ha / gq / m / dn), common holds (first(dur ORDER BY s, dur); text by s0, g),
  flatm own / ref day / night means in hour order (list_avg(list(... ORDER BY hod)): the "~26 / ~27" of 154z).
  Seen before on the same input: the closure order flips between runs (94046b15 det 2); the silmore pick and the
  "~26 / ~27" differ between DuckDB 1.5.2 and 2.0.
  NOT changed (kept as before): ghk / ghk2, the count-drop start / recovery hour. Hours not judged add nothing, so
  every split inside such a run ties but for the last bits of the running window sums (8dc2614a det 33: stat
  19.511564784810677 at j 31-32, ...673 at j 22-30), and the old ORDER BY stat DESC, j takes the largest of those
  bits. That pick is the same on every run seen (1-12 threads, DuckDB 1.5.2 and 2.0, sample and network, workers or
  not: "fell from Mon 08:00"), so it is not random run to run; it moves when the signals are grouped differently:
  max_events 1 M, 2 M or 8 M give "Mon 00:00" (154ab shards too). workers=N keeps the groups, so the default call
  always reads 08:00. A draft rounded stat and took the latest tied hour ("Mon 09:00"; earliest: "Sun 23:00"): a
  wording change, left for the user to decide.
- Parity (exact DataFrame equality incl. dtypes): after workers=1 = workers 4 / 12 on the 60-signal sample (final
  code; the draft also matched workers 6 / 8 and the network). Other group sizes (1 / 2 / 8 M): 1 reason differs,
  8dc2614a det 33 (above). Final code before vs after: 60-signal sample and production-style 58 signals (S: and
  local) 0 differ in both tables (the old closure order happened to be time order in these runs); whole network
  periods 4,650 identical, detectors 17,666: 41 differ, all ties -- 37 signal notes of 9d6189ae (three "held ON
  together" groups, same start, other order), 3 closure orders (8a7d2d87 det 11, 94046b15 det 2, f830bf84 det 16),
  1 silmore pick (556312dd det 25 "also ... Mon 18:15-18:40 (~25)" -> "Mon 17:50-18:10 (~25)", which old 1.5.2 also
  gives). Old 2.0 vs old 1.5.2 on the network: 6 reasons differ (the same tie class).
- Dependencies: duckdb>=2.0, numpy>=2.0, pandas>=2.2 (3.1 fine; kept low for tools still on pandas 2),
  onnxruntime>=1.31, matplotlib>=3.11, pytest>=9.2, build>=1.6 (latest 2.0.0 / 2.5.4 / 3.1.0 / 1.31.1 / 3.11.2 /
  9.2.0 / 1.6.1). Python 3.11-3.14: 3.15 is out but onnxruntime 1.31.1 has no cp315 wheel (numpy / pandas do); 3.10
  has no DuckDB 2.0 wheel. CI matrix 3.11-3.14. pandas 3.1 inplace=True drop deprecations removed (features,
  features_partner). Tests: 2.0 venv 33 passed 4 skipped, check passed; system 1.5.2 35 passed 2 skipped, check passed.
- For the report (ODOT script): a VIEW over the 3 days' parquet files where they are (S: is fine), detectors as a
  table, SET threads = 12, memory_limit = '12GB', temp_directory on local disk, health(con, 'events', 'detectors',
  start, end, periods=True, workers=12); ped_health(con, 'events', start, end, periods=True) on the same view.
