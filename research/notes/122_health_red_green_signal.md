# 122 - Red vs green as a health signal, every type (2026-10-07)
User (Oct 7): is red vs green behaviour properly considered? Analysis only; nothing adopted, packages untouched.
Data = w40 windows, 763 training signals (locked_v2 asserted absent), hi-res log only. Colors of three phases per
detector-window: predicted (production), timing truth (evaluation), placebo (random other phase). Complete gap-free
cycles only; red = after its first 2 s. Code `research/code/health122/h122_{events,study,plots}.py` (pass 215 s, 4 procs);
work `%DC_WORK%/s122/` (cyc122, stats122, lim122, rates122, base122, picks122, plots/).
## Already used (note 117 -> health v4)
Too-fast = fast ONs beyond random arrivals at each color state's own rate; too-many = green flow per lane vs
max(p99.8, 1800). Fast-ON share in red rejected (117). The classifier's phase and function are themselves chosen from
color-relative behaviour, so most red/green structure is already "spent" there.
## Healthy baseline (v4 ok, no category, phase correct; 24 h medians, base122.csv; placebo in brackets)
red ON-rate share rr: Count-1 .04 [.83], YR-1 .01 [.94], Presence-1 .27 [.57], Advance-1 .46 [.50], Mid .46, Other-1 .42.
Presence ON at begin yellow minus begin green (inv): -.51 [+.08]. Green-share excess (dep): Count .32, YR .71,
Presence .13, Advance .02 [all ~0]. -> the signal is real and strong on correct colors, gone on a random phase.
## Candidate checks (limit 1 in 500 of healthy per fn x span x length, fitted on the other signal half)
| check | scope | flags /100 | healthy /100 | excess flags | v4-flagged /100 | new | placebo healthy /100 |
|---|---|---|---|---|---|---|---|
| RR Count/YR counting in red | 14,545 | .43 | .31 | 18 | 1.5 | 61 | 67 |
| INV Presence inverted | 22,555 | .26 | .23 | 8 | .6 | 56 | 63 |
| E5 no green-start response | 35,764 | .21 | .18 | 9 | 0 | 75 | 2.5 |
| ADV hi / lo (Advance color-dependent) | 24,282 | .13 / .11 | .13 / .12 | 0 / -1 | .4 / 0 | 56 | 2.1 / 0 |
| DEP no color response (all types) | 78,233 | .28 | .25 | 22 | .5 | 211 | 14 |
| OCC hi / lo (Other/Mid/Bike red % ON) | 16,851 | .37 / .30 | .28 / .28 | 14 / 3 | 1.7 / 2.0 | 100 | 5.6 / 0 |
Union: 490 new detector-windows (.63 /100, 206 signals) vs .47 /100 on healthy = mostly the 1-in-500 tail itself.
Excess over chance ~75 flags in 78k (~.1 /100). v4-flagged detectors are hardly hit (1.35 /100): no overlap, but no
evidence these find faults either.
## Visual check (14 real new catches + 2 phase-error ones, all PNGs opened, s122/plots)
Traffic / operation 8: permissive or right-on-red movement in "red" (12052 d21, 03031 d9, 10071 d42), arrivals on red
= poor progression (2B319 d3, d4; 2B421 d6), queue over a Mid zone (2B358 d18), phase rarely served (12004 d55).
Layout 3: zone set back / not at the stop bar (2B319 d20, 08131 d17, 13022 d55). Config 2: input blind in red (2B354
d5: 0 red ONs in 2 days while Advance mates log 1,500+; 03031 d63: nothing before 20 s of green). Normal tail 1
(06027 d4). Possible fault 1: 04040 d10 Count P4 has no color response on any phase in all 8 windows.
Count limit is wide (RR p99.8 ~.61) because healthy permissive-left Count zones count in "red"; without FYA (not allowed)
this cannot be separated.
## Phase errors / phase sanity
New catches that are phase errors: 42 / 480 with truth (8.8 %, base rate 2.3 %). As a phase check: union flags 1.8 /
2.2 / 2.9 % of phase-wrong windows (30 min / 3 h / 24 h) vs .5-.7 % of correct ones -> catches < 3 %; useless.
On phase-wrong windows the TIMING phase's colors look worse than the predicted ones (RR 28 vs 2.7 /100, INV 17 vs 2.0):
the model already picks the phase whose colors fit; disagreements are often wiring (2C009 d19: acts in P4 green, timing P7).
Truth Presence zones that look inverted (64 / 18,739): 42 are called another function (21 Count, 14 Other); truth
Count/YR zones counting in red (46 / 13,052): 17 called another function. Odd color behaviour is partly absorbed into function.
## Verdict
Already used: red/green rates in too-fast and too-many (117). New and useful: none as a fault check (excess ~.1 /100,
0 clear faults in 14 opened). Possibly useful later as a config / layout note: "input blind in red" (0 red ONs while
same-phase mates count) - not measured as a rule. Not useful: RR, INV, E5, ADV, DEP, OCC as checks; phase sanity.
