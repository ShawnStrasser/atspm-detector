# 24 — Detector data-quality (DQ) toolkit for the cabinet-print labels
Implements `cabinet_print_guide.md` §5 -> §6 `dq_score`/`flags`. Code `research/code/cabinet/dq_core.py`, `dq_eval.py`.

## API
`dq_core.run(dets, period="auto"|"stg"|"dec")`; CLI `python dq_core.py --dets x.csv --out y.parquet`.
Input: `DeviceId, detector, phase, location` (stopbar|advance|mid|bike|other), `lane_index` (NA =
unknown -> each stop-bar zone is paired with its best-matching advance on the phase),
`lanes_spanned` (default 1), `technology`, optional `function` (Presence enables the peak test).
Output per detector: metrics (`n_on, max5, dur_max, chatter_frac, pulse_frac, n_fault, fault_s,
same_stream_as, cov_h, source`, …), scores `s_pair_sb s_span s_order s_excl s_sat s_bike s_dead
s_health` (NaN = n/a), `dq_score`, `flags`, `reasons`, `suspect_config_or_health` (dq < .5); pairs
in `out.attrs["pairs"]`. `dq = prod(dead, health, sat, bike) × (0.4 + 0.6 × mean(pair cues))`: one
hard failure -> suspect; one weak pair cue does not, two do.

## Data
Allowed codes, `DISTINCT`, `Parameter <= 64` on 81/82 (as `build_cache.py`). Source newest first:
`official/stg/cache/events` (Sept 2026, 66.4 h; excludes TEST but INCLUDES the 143 NEWTEST — the
caller must not pass locked signals), `cache/events` (Dec 2024, 72 h), then raw `data/staging` /
`data/raw/Train_Dec_*` filtered on the fly (identical output, 2.6 s vs 0.2 s). `staging_other` =
non-allowed codes only. Off-peak / peak = 10-min bins ranked by the signal's own vehicle-detector
ONs (≤ .40 / ≥ .85); bins with no event at all are comms gaps and excluded; an ON spanning a
> 120 s all-event gap is not "stuck" (else 2B062 Dec 2024 shows ten 11-h "stuck" zones).

## Checks (thresholds set once from the distributions quoted)
* pair_sb: 10-min corr off-peak (1 at ≥ .6, 0 at ≤ .2). Presence only: drop = (sb/adv peak) /
  (sb/adv off-peak), 1 at ≤ 1.3, 0 at ≥ 2.0. Presence: corr median .92 (p10 .61), drop median .95
  (p90 1.51) — "advance gains at peak" holds. Count zones do the OPPOSITE (drop median 1.36,
  n = 8): the advance undercounts at peak -> drop not tested for them.
* span: spanning loop vs SUM of single-lane advances; corr as above, off-peak ratio .5–1.5, drop
  ≤ 1.1 (0 at 1.5). 01001 mid loop 23,068 vs 24,633 on its advance pair (.94).
* order: off-peak ON cross-correlogram (lr2_pairs.py cue b) peak at +2..8 s ≥ 1.5× the 10–15 s
  baseline with forward asymmetry, OR 1-min high-pass corr off-peak ≥ .3 (note 17 cue d).
* excl: share of the quieter advance loop's off-peak ONs with the other's ON within ±0.5 s; 1 at
  ≤ .15, 0 at ≥ .35. Genuine lane pairs .00–.14; ratio-to-chance is unusable (platoons: up to ×23).
* sat: max5 ≤ 150 × lanes_spanned — never fired (max 109). bike: < 3 ONs = near-zero; > .35 ×
  phase vehicle median = too sensitive. dead: 0 ONs; near_dead: < 2 % of the phase median.
* health: faults 84–88 ≥ 10 or > 2 % of span in fault (fault -> 83), ON > 900 s, > 30 % re-triggers
  < 0.3 s. 33 of 352 detectors logged a fault; 22 only a short one -> not a failure. Pulse ONs
  (0.1 s) are not chatter (the old `detector_meta` rule `frac_dur_lt015 > .5` would flag them).

## Test: 10 pilots + 10 drawn (5 with bike, 5 with mid loops), config labels, lanes unknown
Pass rate (score ≥ .5) on presumably-good detectors (all but config "broken"); 06031 has no data.

| period | pair_sb | span | order | excl | sat | bike | dead | health | suspect |
|---|---|---|---|---|---|---|---|---|---|
| Sept 2026 (19 sig, 352 det) | .895 (105) | .750 (8) | .965 (115) | .873 (63) | 1.0 | .846 (13) | .963 | .968 | 8.2 % |
| Dec 2024 (18 sig, 320 det) | .893 | .750 | .938 | .905 | 1.0 | .818 | .966 | .974 | 7.8 % |
13 detectors suspect in both periods. Runtime ≤ 0.3 s per signal from the caches (test 6–7 s).

## Findings
* Dead detectors move: 12 dead in 2026, 9 in 2024, only 01014 d2, 2C006 d1, 01009 d5 in both.
  03099 lost d10, 11, 37, 39 by 2026; 01015 d19 near-dead (78 ONs) in 2024, dead in 2026; 2C006
  d13/d23 end in an unrestored fault 87. Print checks must use the NEWEST window.
* Same vehicles on two "side-by-side" advances, both periods: 01006 d22/23 91 %, d8/9 48 %,
  01011 d8/9 38 % -> config question for the user (same lane? series loops?).
* Bike loops like vehicle loops, both periods: 03099 d19 (3.6× phase median), 04110 d18. 2B340
  d42/d46 (Yellow_Red): 600–700 fault-88 events per 3 days in both periods.
* 2B062 d34 = d42 (Count = Yellow_Red) carry an identical ON stream: one loop on two inputs
  (`same_stream_as`, info only). 2B062 mid loops count 12–50 % of their advance pair: the
  function-proxy span is probably wrong; prints give `lanes_spanned` directly.
