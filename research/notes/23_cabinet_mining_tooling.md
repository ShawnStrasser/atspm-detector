# 23 — Cabinet-print label mining: tooling + 10-signal pilot redo (v2)

2026-09-23. Procedure: `research/labels/cabinet_print_guide.md`. Code: `research/code/cabinet/
cab_{common,pdf,xlsm,data,build,scale,record}.py` (paths from `DC_WORK`; refuses locked signals). Outputs in `%DC_WORK%/cabinet/`: `signals/<DN>.extract.json` (scripted),
`signals/<DN>.json` (reader's record), `render/<DN>/`, `activity.parquet`, `print_labels.parquet`
(guide §6 schema), `print_tiers.csv`, `xlsm_zones.parquet`, `xlsm_agreement.csv`.

## xlsm "Detection_Configuration" workbooks
`ZoneConfigurationTable` readable for 150 non-locked signals (mostly radar/SDLC; no pilot): one row per
channel (`MT #` = detector, phase, device, slot, function text). 3,992 used channels: phase = timing on
**93.7 %** (default slot table 65.4 %); 95.8 % active; 79/150 list every active channel. Function text =
the config's own source; no geometry or lanes → channel list for radar sites, cross-check elsewhere.

## Scripted vs visual
Scripted (PyMuPDF text layer ~2 s + DuckDB counts): cabinet type, input file in three layouts (332/332S
terminal blocks, 336 front view, 332S loop table with distances), camera/radar labels, diagram-page
ranking, renders + zoom tiles, activity, tiers. After three parser fixes input file 10/10, diagram page
10/10, zone labels 6/6. Visual: lanes, loop position, function. Visual time 0.4–2.0 min, mean
**1.0 min**/signal (~11k tokens), ~95 % of wall time.

## Pilot v2 (10 signals, 163 print detectors: 131 high / 26 medium / 6 low)
Tiers: complete_high 01001, 01007; incomplete 01006 (channels 49–52 active, no description/timing
phase); the rest complete_mixed. Dead: 01014 det2 ("BROKEN"), 01009 det5 (bike); dead in 2026 only:
01011 det1, 01015 det19. 17 print functions disagree with the config (16 config-Presence → Other: tied
mid pairs at 110–150 ft, bike loops, radar/camera zones, a second stop-bar detector; 1 Advance →
departure). User spot-checks all fixed: 01001 loops 6,7 = det27; 01005 sheet 6 is the diagram, loops
3,4 → det1 (diagram Ø6 vs timing 1 → medium); 01006 det2 `shared_upstream` (det2 − det3 ≈ det15) and
diagram numbering ≠ input file; 01010 loops 11–13 = det10 (RT lane), loop 10 = departure; 01014
n_lanes Ø2 = 2, Ø4 = 3; 01015 bike loops 3/15 = det5/19, config swaps det10/11 and det24/25 (input
file + volume pairing win); 01017 Ø7 stop bar det28, Ø8 det25 Presence + det23 `stopbar_secondary`.
User decisions after the pilot (guide §3): Bike and Mid are classes (pilot records converted from
Other/bike, Other/mid); stop-bar function is read from position (Presence 0–20 ft before the bar,
Count on/past it, Yellow_Red next to the count zones) — the 10 pilots are queued for a stop-bar re-read.

## Scale-up stage 1: scripted pass on all 761 non-locked signals (2026-09-23)
`cab_scale.py extract|activity|dq|records|batches` (resumable; 3 PDF workers, DuckDB 6 threads):
extract 9 min, activity 3 min, dq 1.5 min. `cab_record.py show|zoom|crop|finish` is the visual pass's
only writer (validates vocabulary, stop-bar position, a crop per Count / Yellow_Red).
* PDF: 750 / 761 (11 none on the share); 2 signals with two versions (newest taken). xlsm: 156.
  Cabinet type: 332 500, 332S 156, 336 61, unreadable 33 (all text-less prints).
* Input file parsed with ≥ 1 detector: 455 (median 13 detectors; 332 381/500, 332S 60/156,
  336 14/61). Blank input file: 252 (41 with zone labels) — checked on renders: VIP video cards /
  radar (SDLC) sites with nothing landed, not parser misses; text hints ("VIP", "RADAR") recorded.
  No text layer: 43 (36 outlined, 7 raster): all visual. Rows only on non-detector slots (I8L, I13/I14)
  count as blank. No parser change was needed; tiles now on the two best diagram pages (1.4 GB).
* Coverage on `ok` prints: median 93 % of active non-derived channels ≤ 40 are on the input file or a
  zone label (mean 80 %; the low tail = loop sites converted to radar, channels 33–58).
* Data: 696 signals have staging (Sept 2026) events, 358 Dec 2024; **62 have none in either window**
  (38 of them 336) — held back in `batch_nodata.txt`. dq_core on config labels: 388 signals, 7,100
  channels (function labels exist only there), 9.9 % suspect.
* Visual batches (`cabinet/batches/`): 25 ordinary (24–25 signals; batch_01 = 15 + 10 pilot
  re-reads), 3 × 7 336, 7 × 8–9 complex (multi-intersection / no text / > 40 active / > 8 pages),
  snake-dealt by difficulty (active channels, zones, blank input file, several diagrams).

## Open
* 01011 camera zones 42/46 ("Cam A/C – Bikes", 0–124 actuations): stop-bar presence or bike? low.
* Tied pairs across both lanes 110–150 ft upstream, config Presence (01009 det4/18, 01015 det3/17,
  01017 det4/18): now function Mid. 01006 channels 49–52 unexplained.
* 62 no-data signals (38 of 61 336 cabinets): read them at all?
