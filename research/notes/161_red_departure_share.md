# 161 — red_departure_share export for atspm's Queue Model (2026-11-18, revised 2026-11-19; src changed, nothing committed)
Asked by the user (atspm session): lanes where cars leave on red (permissive left, right on red) are where atspm's
arrival-on-green Queue Model made things worse. Output `red_departure_share` (setback.py `SQL_RED` + pooling in
`features`): the OFFs of every stop-bar count zone (Count / Yellow_Red) that the pair model puts in the same lane
(p_same >= .5, with a tau), pooled per advance detector; share = OFFs >= 2 s after the phase's red start (`sb_cyc.t_red`),
before the next green; >= 50 OFFs pooled. Independent of the travel-time partner `_pick` chooses. Exported by
`to_atspm_config` as `RedDepartures` (Advance rows). atspm turns its Queue Model off above 0.25.
**Evidence (atspm study r3: 438 advance detectors with a stop-bar count zone in every lane they cover by Lanes text, 184
over 2-3 lanes; truth Nov 9/10/12; this package run on Nov 12).** Exported share vs the study's pooled share: r = .96
(2-3 lanes, 174 detectors), .95 (1 lane, 242), median |diff| <= 0.001. Queue Model off over 0.25 by the exported share:
7 detectors, 6 better, 1 worse; with the exported travel_time_s 43.3 -> 43.1 wrong per 1,000 (single lane 39.3 -> 38.9). Cutoffs 0.15-0.25 score the same; 0.25 turns it off at the fewest, fewest worse. Pooled beat the highest single zone (25 detectors off,
13 worse). Red start + 1/2/3 s and green end + 6 s scored the same; red start used so clearance lengths do not matter.
Presence zones tested and left out: share higher than the count zones' (mean .066 vs .043, r = .35); at 0.25 they turn
the Queue Model off at 28 detectors, 22 worse. Coverage: 14.5 % of ODOT Advance detectors have a stop-bar count zone
(Count / Yellow_Red) in a lane they cover (10 % of 1-lane, 73 % of 2-lane, all 3-lane).
**Checks.** Tests and atspm-detector-check: see STATUS line. References re-frozen: identical (no reference signal has a
stop-bar count partner).
