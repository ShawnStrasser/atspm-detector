# 147f Spiky counts, low-volume night silence, Count-zone undercount (dropped) (2026-10-15)
User round on top of 147e (knobs unchanged; 5 % soft). Research only. Code research/code/health147/h147f_feat.py
(per-signal 5/10/15-min step GEH + Count vs Advance per lane, 2 min), h147f_spiky.py, h147f_alloc.py (ModelF),
h147f_sheet.py. Work dc_work/s147: spiky147f.pkl, hist147f.npz, cntund147f.parquet, knobs147f.json, prop147f.csv,
flagged147f.csv, stuck147f.csv. review/thresholds_review_v12.xlsx rebuilt in place (backup _backup/..._147e.xlsx).
## A 10001 det 41 (Yellow_Red 1 lane) was bad under 147e
Not the night GEH (2.8 < 8): it was silent 22:10-00:00 with ~45 vehicles expected; goes silent (suspect 24,
bad 54 expected) scored .80, and because the silence lies at night the finding moves to night misses; one finding
above .75 = bad. The 147e cap used the whole-night expectation (84 >= 50), so it did not apply.
Fix: a silent run lying >= 80 % at night is suspect at most when fewer than 100 vehicles were expected during the
silence itself (replaces 'night expectation < 50'). 12 detectors capped; 10001 det 41 -> suspect.
## B Spiky counts (new, family 'shape')
Step = signed GEH between consecutive 10-min counts minus the same step of the yardstick scaled to the detector's
share (GEH, not first differences: differences grow with volume, GEH does not). A jump shared with the mates
(their own step |GEH| >= 3) is not excused: it is judged on what exceeds the mates' jump AND must reach 1.5 x the
bar. z = |step| / (1.4826 x the type's pooled median |step|). Suspect z >= 15; bad z >= 25 or >= 4 jumps >= 15.
Bin size: 5 / 10 / 15 min give 0.50 / 0.47 / 0.61 % flagged at z 15; 10 min keeps the user cases highest relative
to the population (z 2B087 d11 10.2, 08097 d20 15.7, 10041 d48 11.3). z 10 would catch all three but flags 2.4 %
(the cases sit at the 97-99th percentile of a heavy tail), so 15: 70 detectors (0.47 %), 16 bad; with too fast
6 / 26, with too many 3 / 12, on its own 62; 10 flagged through shared jumps only. 08097 d20 -> spiky (bad with
stuck + too many); 2B087 d11 (too fast suspect) and 10041 d48 (too many suspect) not spiky.
## C Count zone undercount: tried, dropped
Count zones (function prob >= .7) vs the median per-lane count of the same-phase Advance zones (07-19 hourly
GEH). 769 eligible: ratio bulk around 1 (log10 sd ~.2) with a long low tail: 101 (13 %) below a third, 96 of them
GEH >= 10. 13 % of model-sure Count zones failing is not a plausible fault rate: which lanes an Advance covers is
unknown, so the comparison is not clean. Dropped (04023 d24: ratio .15, GEH 10.8, but f_top .52 anyway).
## D Bad at score > .75 (unchanged): 28 detectors would move bad -> suspect if bad meant 'past the bad limit'.
## Final status % (ok / suspect / bad / n.e.d.)
Old 94.37 / 2.07 / 1.93 / 1.63 | New 96.18 / .83 / 1.34 / 1.65 | Proposed 94.37 / 2.00 / 1.97 / 1.66 (4.0 %).
Flagged (suspect + bad) Old / New / Proposed: stuck 1.93 / .52 / 2.34; counts fall .67 / .22 / .68; night .62 /
.34 / .30; busy .91 / .81 / .64; too fast + many .26 / .17 / .20; erratic .20 / .27 / 0; spiky - / - / .41.
## 147g Accumulation path (count of jumps at z 4/5/6/8 per day, or sum of squared excess z), 5/10/15-min bins:
no setting catches 2B087 d11 + 08097 d20 under ~1 %: best = 15-min sum of (z-6)^2 >= 89 -> 1.20 % (10041 missed);
15-min >= 7 jumps at z >= 6 -> 1.77 % (all three); 10-min >= 12 at z >= 5 -> 2.48 %. Not adopted; v12 stays 147f.
147g ADOPTED (user): spiky accumulation path, suspect only: 10-min, day sum of max(z - 4.5, 0) (shared jumps z/1.5)
>= 32.8 = smallest of the 3 cases; z0 / bin chosen for min % (1.70 %; 5-min z0 11 1.63 % was a single jump). Check
1.81 % (with too fast 11/26, too many 5/12, own 254); status 93.26 / 3.03 / 2.05 / 1.66; 2B087 / 10041 end bad by
two findings. h147g_alloc / h147g_sheet; prop/flagged/knobs147g, spiky_only147g.csv (6); backup _backup/..._147f.xlsx.
147h: PROPOSED bad limits shown where bad really starts (score .75: suspect + .615 x (score-1 point - suspect)),
% bad = past it; scoring unchanged -> 0 status / flag differences vs 147g (e.g. drops GEH 15 -> 13.1, night 12 ->
10.5, stuck Advance 30 -> 22.4, Bike 10 -> 8.3, too many 2,520 -> 2,358). h147h_sheet; prop/flagged/stuck147h.
147i: stuck bad starts exactly at the user's minutes (score-1 point = suspect + (min - suspect)/.615): 9 bad -> suspect; 93.26 / 3.09 / 1.99 / 1.66.
147j: spiky yardstick fix: mates' step = GEH on their own 10-min counts (>= 10 in both bins), scaled down only;
artefacts 2B388 d2 / 2B358 d4 / 2B436 d4 -> ok; path re-derived z0 2.0, threshold 31.3; spiky 1.81 -> 1.67 %,
dramatic path now fires on none; 93.41 / 2.98 / 1.95 / 1.66. h147i/j_alloc, _sheet; prop/flagged147i/j.
