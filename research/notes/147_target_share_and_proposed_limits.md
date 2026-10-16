# 147 Target flagged share (5 % = 3 % suspect + 2 % bad) and proposed limits per check x type (2026-10-15)
Orchestrator task, research only: src/ untouched, no fault events, locked signals not used, nothing re-scored.
Data = h146_sheet.py exactly (d143 + 1.1.0 display measures from out144_h24b, ep143; 755 signals x 24 h, 14,806 det).
Code research/code/health147 (h147_lib: all_scores with explicit bad points, parity 0 diffs vs h143 all_scores at
Old and New; h147_infer; h147_alloc; h147_sheet; h147_inferred_json). Work dc_work/s147.
## Outside figures (all include dead / no-data detectors, which are NOT in our population)
- FHWA Traffic Detector Handbook 3rd ed. (2006) ch.5: NY State 25 % of 15,000 signal loops not operating at any
  time; Minnesota district 24 %/yr, Cincinnati 29 %/yr failures (mostly dead; partial not split out).
- Rajagopal & Varaiya, Health of California's Loop Detector System (UCB-ITS-PRR-2007-13): ~40 % of freeway
  sensors failed on a typical 2005 day; 16-25 % always failed; mostly comm / no data; stuck / intermittent states
  exist but are not reported as a separate share.
- Miami-Dade County memo 2023-05-24 (Directive 230074): 1,388 of 12,496 loops (11 %) damaged or inoperative.
No published share of partially working (stuck / chattering / undercounting) detectors was found; all figures are
dominated by dead detectors. A 5 % partial-fault share is plausible next to 10-25 % all-cause.
## A Inferred abnormal share (fitted normal bulk per check x type, check's own population)
Probit fit 10-90 % (pooled to function / group when < 100 valued). Tail LR(t) = (observed beyond t - expected
beyond t) / expected (cumulative tail, not local density: the tails hold too few points per bin). Abnormal per cell
= peak of observed - expected beyond the fitted 90th pct. Union = detectors beyond LR >= 1 in any check.
- union LR >= 1: 12.9 % (of which ~2.8 % expected normal -> ~10 % excess); LR >= 10 ("bad-ish") 8.7 %; 1-10 4.2 %.
- per check (beyond LR1 / sum of excess peaks, % of all): stuck 5.7 / 5.4, night misses 4.3 / 4.2, busy 2.7 / 2.6,
  silent 1.7 / 1.3, drops .3 / .3, erratic .1 / .3, too fast .02 / .1, too many .07 / .05.
- fit range 25-75 / 5-95: union 17.9 / 11.0 %, LR10 15.7 / 6.8 %, excess sum 20.4 / 11.8 %.
Verdict: NOT used. Unstable (11-18 % with the fit range) and inflated by normal-but-skewed measures (Count holds mix
pulse 0.1 s and normal zones: 15 % "abnormal"; Presence holds at low volume; quiet nights). Targets 3 % + 2 %.
## B Allocation: one knob each (lam_s, lam_b) for all checks x types
limit(lam) = lowest observed value >= fitted 99th pct from which (obs - exp) / exp >= lam holds all the way out
(<= 1 / (1 + lam) of the flagged expected normal). No crossing: suspect = just beyond the observed maximum (at
least New; flags nobody), bad = none (a single finding stays suspect). Floors: hold 300 s, too many 2,250. Busy at
night uses lam_b (rule P makes every busy finding final bad). Per-detector = New x (proposed / New type median).
Grid on final status (h144 rules): lam_s = 11.0, lam_b = 1.78e6 -> suspect 3.00 %, bad 1.98 %.
Lam_b is huge because the package's linear score makes a single finding final bad at score > .75 (62 % of the way
to the bad point) and two findings / busy at night / silent >= 4 h are bad regardless; with a bad point at only
LR 1e4 bad is 2.7 %.
## Final status % of 14,806 (ok / suspect / bad / not enough data)
Old 94.37 / 2.07 / 1.93 / 1.63 | New 96.18 / .83 / 1.34 / 1.65 | Proposed 93.37 / 3.00 / 1.98 / 1.65.
Per check, flagged at final status (suspect + bad, % of all) Old / New / Proposed:
stuck 1.93 / .52 / 1.76; counts fall .67 / .22 / .25; night misses .62 / .35 / 2.68; busy at night .91 / .81 /
.63; too fast + too many .26 / .17 / 0; erratic .19 / .27 / 0 (dropout/level/night findings overlap).
## What the proposal does (prop147.csv)
- Stricter where the data show an excess tail: stuck Advance 1 210 -> 24 min, Presence 1 43 -> 11, Mid 5;
  night misses 19 % -> 41-85 % of expected (Mid 1 85 % flags 3.9 % suspect + 3.9 % bad: tight Mid bulk, sigma .06);
  goes silent 44-65 expected ONs (New 114-134).
- Looser / off where no excess tail: too fast (limits beyond observed maximum), erratic, too many (no detector
  above the proposed limits; the 9 above the physical 2,250 are no longer flagged: data say they are the normal
  tail), busy at night 7-9 z.
- Small cells (2+ lanes, Bike 2+, Count 2+) land on single detectors (e.g. silent Bike 2+ 20 %, 2 of 10).
## Outputs
review/thresholds_review_v10.xlsx (v9 layout, third row PROPOSED prefilled, summary block at top);
dc_work/s147/infer147.parquet (cells x 3 fit ranges), grid147*.csv, knobs147.json, prop147.csv, flagged147.csv.
## Open (orchestrator / user)
- Night-miss limits become far stricter (night misses = half of all flags) and too fast / erratic / too many stop
  firing: a property of the equal-LR rule on these tails, not a measured fault rate. User to judge in the sheet.
