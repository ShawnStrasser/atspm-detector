# 26 — Cabinet-print consistency sweep (batches 01–12, 294 signals)

2026-09-23. Brings the finished batches 01–12, read under older rules, up to the user's answers of 2026-09-23 (guide
§3, §8; last section of `VISUAL_PASS_INSTRUCTIONS.md`). Every change is logged per field in
`%DC_WORK%/cabinet/sweep_changes.csv` (4.9k rows); each record gets a `sweep` block listing its rules.

## Tooling (`research/code/cabinet/`)
* `cab_record.py finish` now rejects: a loop as Count (hint: Advance upstream / Presence at the stop bar); Yellow_Red
  unless technology = radar; the `series_loop_pending` flag (use Mid / series_loop). A Count/YR flagged
  `position_not_drawn` may use `<DN>_d<det>_approach.png` as its crop. New signal-level
  `signal_flags: ["unusual_layout"]` (needs `signal_flags_reason`). Old records read unchanged.
* `cab_build.py` carries `unusual_layout` (+ reason) into `print_tiers.csv` / `print_labels.parquet` as its own
  column (tier unchanged): those signals are excluded from training and scored separately.

## Scripted relabel (detectors changed)
* a, loop Count (2): 2B132 stop-line loop → Presence; loop beyond the crosswalk → Other/departure.
* b, YR not on radar (6): 12067 (TrafiSense video) code-YR zones → Count, `label_disagrees`.
* c, series loop (53): → Mid / series_loop, pending flag dropped (24 had been Advance).
* d, long zone (138): radar/video "Presence" longer than 20 ft (0-75, 0-110, long zones from the bar) → Other/long_zone.
* e, undrawn code (178): null radar zones → xlsm code (CO→Count, YR→YR, 0-20→Presence, 20-75→Other,
  A*/Adv→Advance), low + `position_not_drawn`.

## Visual re-check (every page, contact sheets; 68 flagged prints + 10 pilots; five parallel workers)
* **30 of the 68** prints flagged `position_not_drawn` / `no_diagram` / null-for-no-zones had every zone on another
  sheet. All are radar or video, mostly sheet 7 or 6 where the record had used 6 or 5; 03108 is the spot-check case.
  There the tabled "Presence" zones are almost always long zones from the stop bar → Other/long_zone, and Count
  boxes sit just past the bar. 648 detectors relabelled or confirmed, most now high, with wide crops.
* None of the 22 loop prints whose record says "no intersection diagram" has one anywhere (sheet 1 "see TRS #…",
  sheet 5 a blank template). `position_not_drawn` remains on 19 signals and `no_diagram` on 8.
* **ATSPM-titled sheets:** only 13021 (sheet 6 "Data/ATSPM" = dets 33–52; sheet 7 = operating phase-call zones), and
  it was already labelled accordingly. The text layer finds no other in 01–12; the 10 no-text pilots' images show none.
* **Radar over loops** (§8): 67 loops → Other/superseded_by_radar. 35 are live loops under live radar at 10 signals;
  the rest are dead old loops (2B053, 2B054, 03028, 03075, 04034). Loops upstream of radar zones are not "under"
  them and are left alone (2B069, 2B365, 2B384, 2B552).
* **`unusual_layout`: 16 of 294.**
  * Live radar over live loops: 01017, 01030, 03019, 03032, 03033, 04024, 04033, 04070, 2B083, 2C070, 2B132.
  * Two intersections on one controller: 01010.
  * Print no longer describes the live channels (rewired): 06030, 08044, 2B054, 03071.

## Confidence
* Radar/video Count/YR/Presence/Advance → high when the reason names a zone bubble or table code, timing phase =
  diagram phase, and there is no uncertainty wording and no review flag. 147 raised, including the user's examples
  12037 d33, 2B041 d44 and 2B340 d42 (2B340 via its worker). Loops untouched.
* Print detectors in 01–12, before → after: high 2,692 → 3,382; medium 1,559 → 1,159; low 1,024 → 748.
  Unclassified 756 → 522.

## Crops
* Old Count/YR crops were located on their page by normalised cross-correlation (72 dpi, upside-down pages included)
  and re-cut ≥ 0.15 × 0.22 of the page around them at 200 dpi: 457 re-cut, 61 already wide, 251 fresh worker crops kept.
* 35 did not match (agent crops made at 600–800 dpi); a sample already shows bubble + stop bar. Old crops are backed
  up in the sweep scratch folder.

## Build (360 signals incl. 13+ so far): high 35, mixed 237 (17 unusual), incomplete 88 (2 unusual). 01–12: 22 / 194 (14 unusual) / 78 (2 unusual).

## Open
* Boxes the table calls Count/Presence but drawn set back or in a slip lane are now Other: 03057 d5/d19, 03112 d3/d17,
  10087 d5/d19. Long set-back "P" zones are Other/advance_presence (former Advance at 10031, 10071, 10079).
* 05048: the print's MT numbers predate the current channels → medium, tied by phase + description.
* 10046: YR 41/42/46/48 are drawn but not in the record. 08096: the video zone numbers do not map to channels.
