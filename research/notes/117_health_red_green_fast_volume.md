# 117 — Health: red vs green study -> traffic-aware "too-fast" and "too many in 5 min" (2026-10-07)
User review of health_review_v3 (Oct 7, point 2): an actuation every ~0.5 s is normal in heavy traffic but only in green;
judge flow against green time per lane and saturation (~1800/h/lane); compare red vs green to separate traffic from faults.
Data = w40 windows (8 per signal), 763 training signals, locked_v2 asserted absent; hi-res log + the classifier's predicted
phase / function / lane span (phase number = join key to that phase's colour events only). CPU 4 procs, DuckDB 10 GB / 4 thr.
Code `research/code/health117/h117_{events,study,user,plots}.py`; work `%DC_WORK%/s117/`. Nothing adopted into a package.
## Event pass (h117_events, 278 s)
Vehicle ON = ON starting a continuous ON; an ON logged again without an OFF is never a fast ON (point 4 by construction:
2B068 d19, 2B058 d46 no fast finding). Colour of the detector's phase at each ON: G, Y, red (first 2 s apart), unknown;
occupancy split exactly by state; per 5-min bin and per cycle -> bins117 (10.1 M rows), det117.
## Healthy baseline per type (no v110 finding outside rapid / volume / chatter, >= 50 ONs; base117.csv; 24 h p50)
| type | ONs/h green / red | % ON green / red | busiest-5-min green flow per lane p50 [p99.8] |
|---|---|---|---|
| Count 1 | 270 / 10 | 1.7 / 0.1 | 746 [1710] |
| Yellow_Red 1 | 338 / 3 | 1.9 / 0.0 | 777 [2043] |
| Presence 1 | 114 / 36 | 25 / 22 | 427 [1581] |
| Mid 1 | 258 / 214 | 6.4 / 10 | 777 [1917] |
| Advance 1 / 2+ | 136 / 103, 280 / 302 | 6.2 / 4.0 | 488 [1830], 403 [930] |
| Other 1 | 64 / 36 | 27 / 12 | 274 [1487] |
* Count / YR are near silent in red (3-14/h) and fire in green; Presence is occupied in both; Advance / Mid see arrivals in
  both (Advance red rate ~= green rate). Healthy 1-lane busiest green flow p99.8 = 1500-2040 per lane = saturation.
* Fast ONs (< 1 s) follow traffic: at random arrivals share = 1 - exp(-rate x 1 s); 04035 d53 (2-lane Advance, 1770/h)
  has 38 % fast in red AND green = exactly the random-arrival share. Fast ONs in RED as a ratio is NOT a fault signal: healthy
  Count-1 red fast / expected p50 1.2, p90 9.7 (few red ONs, clustered: turns on red, creeping) - dropped.
## Checks v117 (limit = p99.8 healthy per fn x span x sample length, fallback span x length; cells >= 100; bad at 2x)
FAST = (a) zf: fast ONs minus the number random arrivals would give at each 5-min bin's rate per colour state (rate capped at
its centred 1-h median, so a burst cannot raise its own expectation), in Poisson SD; OR (b) burst bins: 5-min bins >= 4 SD
above that expectation with >= 5 fast ONs, more than the type limit. Replaces the raw fast-ON share (ioi < .5 / 1 s, bursts).
VOL = busiest 5 min per lane: ONs in green+yellow / green+yellow hours (bins with >= 60 s green) for stop-bar types, all
ONs per hour for Advance; limit = max(p99.8 healthy, 1800 /h/lane). Replaces max ONs in 5 min.
Not adopted: fast / expected ratio (misses 08073 d7), worst hour (limits too wide), red-only z (flags 04028 d53), no 1800 floor.
## Flag rates per 100 detector-windows (fast or volume family; rates117.csv; all types)
old (v110 rapid / volume) -> new: 24 h .84 -> .78 (fast .65 -> .72, volume .32 -> .10); 3 h .57 -> .48; 30 min .39 -> .36.
Healthy flagged by the new family .36 / .31 / .37. Changes (changes117.csv, 543 rows): 300 old flags cleared (251 lose
their only finding), 243 new (175 were ok / watch); new bad 76 / 46 / 69 (24 h / 3 h / 30 min). Per type 24 h old -> new:
Advance-1 1.09 -> .97, YR-1 .99 -> .42, Mid-2+ .54 -> .13, Count-1 .54 -> .69, Presence-1 .76 -> .82, Presence-2+ .47 -> .94.
Overall suspect + bad (v110 status, family swapped): 24 h 3.87 -> 3.79, 3 h 1.50 -> 1.37, 30 min .77 -> .73.
## User rows (user117.csv; plots s117/plots)
Heavy traffic -> ok, all 4: 2B039 d18 (zf 5.4 vs 28.8), 04028 d53 (16.5 vs 23.7; 870/h/lane), 2C028 d37 (1316 green/h/lane
< 1800), 01062 d2 (1596 < 1800). Faults: 08073 d7 fast + volume (zf 23.8 > 21.9; 2088/h/lane in one red burst) -> suspect
in this family (v110 bad stays via erratic counts); 11021 d2 fast (4 burst bins > 3; chatter unchanged) - d3 / d4 not caught
here (d3 3 = limit); 08052 d10 fast (12 burst bins > 6). 04035 d53 / d52 and 10055 d5 are NOT fast / volume faults by this
data: d53's fast ONs = its rate, at 676/h at 03:00 - a night-volume problem for the time-of-day model (point 1); 10055 d5 =
chatter (unchanged check). 04035 d53 24 h stays bad via the profile.
## Visual checks (25 PNGs opened) and open points
* Cleared samples: sustained heavy green flow (1d62f2bd d23 Presence, 1650/h/lane) look right. Judgment call: a82b3b8f d7
  / d22 (Advance-1, isolated red-time fast bursts 7 / 21 bins) - queue creeping over the loop or a fault; d22 kept (21 > 13.6),
  d7 cleared. The Advance-1 burst limit (13.6) is wide because "healthy" = leave-own-family-out still holds ~.4 % old fast flags.
* New samples: 08019 d8 Presence (bursts in red + green, bad), 6f5feefb d24 Other (7 burst bins vs 6, borderline suspect).
* Production: numpy-only (bincount per bin / state, rolling median via sliding_window_view); colour already parsed in act_stats.
