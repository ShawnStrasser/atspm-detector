# 147c Proposed limits v12: user's stuck-on bad limits, Bike = 1 lane (2026-10-15)
User feedback on v11 (197 min for Advance / Mid far too long; Advance ON ~15-30 min = bad). Research only; src/,
v10, v11 untouched. Code research/code/health147/h147c_alloc.py (subclass of h147b: only the stuck-on order step
and the Bike merge change), h147c_sheet.py; work dc_work/s147 (knobs147c.json, grid147c.csv, prop147c.csv,
stuck147c.csv, flagged147c.csv). Sheet review/thresholds_review_v12.xlsx.
## Rules (on top of note 147b: physics checks fixed at New, Old-prior Dirichlet allocation alpha 5,917,
water-filling, bulk / 300-s floors)
- Stuck-on BAD limit (type median, minutes): Bike 10, Count / Yellow_Red 20, Advance 30, Mid 35 (fixed, both spans);
  Presence / Other from the allocation, never below 35, 1 lane <= 2+ (isotonic).
- Stuck-on suspect: from the allocation, <= 90 % of bad, same order (Bike <= Count, YR <= Advance <= Mid <=
  Presence <= Other; 1 lane <= 2+), >= 5 min.
- All Bike detectors = 1 lane (Bike 2+ cell, 10 detectors, merged into Bike 1).
## Result (R_s 900, R_b 3): target 3 / 2 not reached; closest with bad on target
Final status ok / suspect / bad / n.e.d.: Old 94.37 / 2.07 / 1.93 / 1.63 | New 96.18 / .83 / 1.34 / 1.65 |
Proposed 93.80 / 2.55 / 2.01 / 1.65. Grid: suspect stops at ~2.6 (2.59 / 2.09 at R_s 940); beyond, only bad grows.
Why: the fixed stuck bad limits make stuck findings bad sooner (bad floor up), stuck suspect is at its floors, and
the remaining room is busy at night (all bad) and night misses.
Flagged at final status (suspect + bad, % of all) Old / New / Proposed: stuck 1.93 / .52 / 2.28; counts fall .67 /
.22 / .44; night misses .62 / .35 / .83; busy at night .91 / .81 / 1.20; too fast + too many .26 / .17 / .18;
erratic .19 / .27 / .20.
## Stuck-on limits (minutes, suspect / bad)
Bike 5.5 / 10; Count 1 5.7 / 20, 2+ 5.7 / 20; Yellow_Red 1 5.5 / 20, 2+ 12.9 / 20; Advance 1 10.3 / 30, 2+ 12.9 / 30;
Mid 1 10.3 / 35, 2+ 12.9 / 35; Presence 1 10.3 / 572, 2+ 15.4 / 660; Other 1 14.7 / 572, 2+ 15.4 / 660.
## Caveat / open
- Presence / Other bad from the allocation lands at 9.5-11 h: almost no bad budget is left after the fixed limits,
  busy at night and two-finding cases, so the allocation puts it beyond the longest hold. The user may want a fixed
  number there too (e.g. 60 min) - that would raise bad above 2 %.
- UPDATE (orchestrator): Presence / Other stuck bad fixed at 60 min (= Old); v12 rebuilt (R_s 1050): 93.31 / 2.52 / 2.53 / 1.63; total 5 % forces bad 2.5 % (bad 2.0 only at total 4.36: R_s 850 -> 2.36 / 1.99); flagged stuck 2.31, counts fall .45, night .95, busy 1.64 (all bad), fast/many .18, erratic .19.
