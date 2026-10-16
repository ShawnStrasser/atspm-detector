# 147b Proposed limits v11: physics first, Old flag mix as prior, stuck-on order (2026-10-15)
User feedback on v10 (via orchestrator). Research only; src/ untouched; v10 left as is. Same data / rules as note 147.
Code research/code/health147/h147b_alloc.py, h147b_sheet.py; work dc_work/s147 (knobs147b.json, weights147b.csv,
grid147b.csv, prop147b.csv, stuck147b.csv, flagged147b.csv). Sheet review/thresholds_review_v11.xlsx.
## Method
1 Fixed checks at New: too fast (per-1,000 gap cuts), too many 2,250 veh/h/lane (bad 2x; the 9 above are flagged),
  erratic (New). Alone they give 0.32 % suspect + 0.26 % bad at final status; the rest of the target is shared out.
2 Remainder over stuck on, goes silent, count drops, night misses, busy at night x type (Dirichlet update):
  weight_c = (alpha x prior_c + excess_c) / (alpha + sum excess); prior_c = cell's share of the Old flags (check's
  own score >= .35 at Old limits, 733 in all); excess_c = note-147 peak of observed - expected (sum 1,972; stuck
  Count / Yellow_Red refitted per pulse / normal zone, the mix is not a fault).  alpha = 3 x sum excess = 5,917:
  the data count at one third of face value (the excess moves 11-20 % with the fit range and is skew-inflated), so
  the prior carries 75 %. (alpha = Old count 733 would let the data carry 73 % and give night misses 27 % weight.)
  Weights: stuck .383, silent .168, drops .056, night .207, busy .186 (prior .386 / .191 / .067 / .172 / .184).
  Flag budget R_s (check's own score >= .35) split by weight; each cell's limit read off its 'flagged by the check'
  curve (New per-detector limit x factor, 241 factors); never inside the bulk (>= fitted p99) nor below 300 s
  hold; a cell at that floor passes the rest on (water-filling). Bad budget R_b over the four with a bad number
  sets where the score passes .75 (one finding -> final bad); bad limit (score 1) follows; busy at night keeps
  its rule (every finding bad).
3 Stuck order on type limits (suspect and bad, minutes): Bike <= Count, Yellow_Red <= Advance <= Mid <= Presence <=
  Other (Other at least as loose as Presence), 1 lane <= 2+; Bike bad <= 10 min. Weighted geometric pooling of
  violators (isotonic, weights = detectors), suspect <= bad, then R_s / R_b re-tuned on final status.
## Result (R_s 950, R_b 10): target 3 / 2 NOT fully reachable
Final status ok / suspect / bad / n.e.d.: Old 94.37 / 2.07 / 1.93 / 1.63 | New 96.18 / .83 / 1.34 / 1.65 |
Proposed 93.64 / 2.69 / 2.02 / 1.65 (closest point; error .31 pt).
Why short: stuck on is at its floor (300 s, bulk p99, order) at 2.29 %; more flags could only come from busy at
night (every one bad) or night misses (user: implausible). Grid: suspect 3.0 needs bad >= 2.9.
Flagged at final status (suspect + bad, % of all) Old / New / Proposed: stuck 1.93 / .52 / 2.29; counts fall .67 /
.22 / .44; night misses .62 / .35 / .89; busy at night .91 / .81 / 1.36; too fast + too many .26 / .17 / .18;
erratic .19 / .27 / .19. Night misses now near its Old share (v10 2.68).
## Stuck-on limits (type median, minutes, suspect / bad)
Advance 1 10.1 / 197, 2+ 24.2 / 316; Presence 1 10.1 / 197, 2+ 24.2 / 316; Count 1 5.8 / 60, 2+ 5.8 / 60;
Yellow_Red 1 5.5 / 32, 2+ 24.2 / 207; Mid 1 10.1 / 197, 2+ 24.2 / 316; Bike 1 5.5 / 10, 2+ 5.7 / 10;
Other 1 14.7 / 197, 2+ 24.2 / 625. Ties = order constraints binding (Advance would be stricter than Presence in
the allocation; pooled). YR 2+ is pooled up to Advance 2+ by the order.
## Caveats
- Busy at night limits fall to z 3.5-4.6 (New 6) and carry 1.36 %, all bad.
- When almost no bad is allocated, the bad points of count drops / night misses land far out (shown as tiny % of
  expected, e.g. night Advance 1 bad 0.39 %): one such finding stays suspect.
- Small cells (2+ lanes, Bike 2+, Count 2+) move on 1-2 detectors (e.g. drops Count 2+ 74 %, 1 of 49).
## Open
- User: accept 2.7 / 2.0 (total 4.7 %), or allow more busy-at-night / night-miss flags to reach 3 % suspect.
