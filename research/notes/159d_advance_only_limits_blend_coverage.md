# 159d — Advance only: what limits the distance, a controlled physics blend, weekday coverage (study only, 2026-11-16)
User ruling 2026-11-16: **only the Advance distance matters** (it feeds the performance measures); the Mid distance is
used for nothing, so note 159c's Mid gain does not count. Questions: still incorporate? anything to try differently?
Nothing in `src/` changed; nothing committed. Code `research/code/shock159/sw9_advance.py diag|blend|advonly`,
`sw10_coverage.py`; outputs `%DC_WORK%/s159/` (sw9_*.log, advance_blend.parquet / .json, sw10.log, coverage_w40.json).
Frame = the shipped model's training table (note 95), package-shaped features (159c), OOF, 15 seeds; printed Advance
5,551 detector-windows (623 detectors, 127 signals), locked_v2 absent.
**Where today's model stands.** <= 100 ft 1.6 ft median error (solved). 101-200 ft 24.6 ft. > 200 ft 53.8 ft, 47 %
within 50 ft. Misses over 100 ft: 12 % of detector-windows; 89 % of them read short; 58 % never had a held cycle;
radar is 28 % of them (11 % of rows), video 24 %, loop 48 %.
**Shrinkage: mostly no.** Median signed error by PREDICTED distance: 0 / -5 / -1 / +4 / -39 / -9 ft for <= 100 / 100-150 /
150-200 / 200-250 / 250-300 / > 300 (the 250-300 bin, 584 rows = 10 %, does read short; a recalibration of that bin was
not tried). "Reads 45 ft short beyond 200 ft" is mostly conditioning on the truth. One declared training variant,
Advance-only rows: worse (pooled median +1.3 ± 0.5 ft, within-50 -1.0 ± 0.6; 101-200 ft within-50 -4.5) — Mid rows
help the middle range. Dropped.
**Ceiling.** Give the TRUE distance to every row that passes a behaviour-only gate (>= 5 held cycles, release IQR <=
4 s, model > 100 ft): the gate covers 4 / 11 / 17 / 22 % of Advance at 30 min / 6 h / 24 h / 66 h, and pooled within-50
goes 69.3 -> 73.1 (>= 3 cycles: 74.1); at 24 h 71.2 -> 77.2; > 200 ft median 53.8 -> 43.8. So a PERFECT queue method
is worth ~4-6 points here. Advance > 200 ft with no held cycle at all: 74 / 62 / 53 / 40 %. **Coverage is the limit.**
**Controlled physics blend** (159's blend was 66 h only with after-the-fact thresholds). d_sw = wave x (release median -
lost), constants from the training folds (26.5-28.7 ft/s, 3.0-3.5 s). Rule: held >= n, IQR <= q, model > 100 ft,
optionally only upward -> a x d_sw + (1 - a) x model; n / q / a / mode picked on the training folds of each held-out
fold (picked: 3 cycles, 6 s, 0.5, mostly upward-only). Controls with the same rule: d_sw permuted among the fired
rows; a plain bump of the model on those rows.
| Advance, pooled lengths (15-seed average) | median ft | within 50 % | > 200 ft median | > 200 ft within 50 |
|---|---|---|---|---|
| today | 22.7 | 69.3 | 53.8 | 46.8 |
| today + blend (no refit) | 21.8 | 70.4 | 49.4 | 50.5 |
| tier A features | 22.1 | 70.5 | 50.2 | 49.9 |
| tier A + blend | 21.9 | 71.1 | 48.1 | 51.6 |
| today + permuted physics / + bump | 23.6 / 23.0 | 68.6 / 69.4 | 52.3 / 51.7 | 47.8 / 48.3 |
CIs vs today: blend within-50 +1.1 [+0.2, +2.2], median -0.9 [-2.1, -0.3]; A +1.2 [+0.1, +2.4]; A + blend +1.8 [+0.3,
+3.4], > 200 ft median -5.7 [-10.0, -1.8], within-50 +4.8 [+1.8, +7.9]. Single seeds (what ships; today 23.2 ft / 69.2 %):
blend 21.8 / 70.4, A + blend 22.1 / 70.9; > 200 ft 53.8 -> 49.1 / 48.5. Blend vs bump: median -1.2 [-2.1, -0.7], within-50
+1.0 [+0.2, +1.9] (the physics is real; permuted control is worse than today). Blend without refit vs tier A: tie
(within-50 -0.1 [-1.0, +0.8]). A + blend vs A: pooled within-50 +0.6 [-0.1, +1.4] (not clear), > 200 ft median -2.2
[-5.1, -0.5] (clear). On the 9 % of rows the rule fires on: 43.5 -> 28.9 ft, within-50 56 -> 68. Gain follows how often
it fires: 4 / 10 / 12 / 17 % of Advance at 30 min / 6 h / 24 h / 66 h -> within-50 +0.1 / +1.7 / +1.5 / +3.0; > 200 ft
at 24 h 50.9 -> 42.2 ft, at 66 h 52.0 -> 43.4. Loop 10 %, video 7 %, radar 3 %; video and radar gain nothing (radar 67 ft).
**Weekday coverage (count only, no model scored).** Every window of this study is Fri 16:15 - Mon 10:24; the 24-h windows
are a Saturday and a Sunday. Week-40 events already on disk (`data/staging_2026_w40`, all 127 training signals present;
Sat 26 Sep is a partial day and is not used), same detectors and predicted phases; share of printed Advance > 200 ft
(255) where the rule could fire (>= 3 held cycles, IQR <= 6 s) / same for loops > 200 ft (169):
| Sat 3, Sun 4 Oct, Sun 27 Sep (24 h each) | Tue / Wed / Thu / next Tue (24 h) | Tue 6-9 am | Tue 3-7 pm | Sat + Sun 48 h | Tue-Thu 72 h | Mon-Fri | 7 days |
|---|---|---|---|---|---|---|---|
| 26-29 / 23-31 | 35-39 / 46-50 | 23 / 27 | 25 / 31 | 32 / 36 | 47 / 62 | 48 / 64 | 48 / 64 |
A weekday gives about a third more far detectors than a weekend day (1.6-2x for far loops); three weekdays reach the
plateau (47 % vs 32 % for a whole weekend); a full week adds nothing over Tue-Thu. Every gain measured in 159b-d is a
weekend number, and the shipped distance model itself was trained on weekend-heavy windows.
**Verdict.** For Advance alone the measured gain is small: about +1 to +2 points within 50 ft, about 1 ft of median,
4-6 ft of 54 beyond 200 ft. Not worth a release by itself; harmless to carry into the next distance refit (tier A, ~6 ms;
the blend is a few lines of numpy on top). The one thing not yet tried that could change the answer: measure — and
train — the distance model on WEEKDAY windows (week 40 is on disk; needs the setback frame rebuilt there with OOF phase /
function — not scoped). Extrapolating the fire-rate table, a weekday day should look like the 66 h weekend row (about
+3 points within 50 ft, 8-9 ft beyond 200 ft) and three weekdays a little better: still modest, unmeasured. Radar (71
of 623, 67 ft) has no log-based fix found: it drops stopped vehicles. PAUSED on the user: run the weekday evaluation or stop here?
