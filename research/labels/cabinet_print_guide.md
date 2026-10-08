# Mining detector labels from cabinet prints — guide for agents

Domain rules come from the user (a traffic engineer), 2026-09-23. Read `AGENTS.md` first. The print
share's path is in `%DC_WORK%/cabinet/share_path.txt` (never written into a tracked file). The share is
**strictly read-only**: copy PDFs to `%DC_WORK%/cabinet/pdf/` and work on the copies. Everything this
produces is agency-specific: it lives in `%DC_WORK%/cabinet/` and `review/`, never in tracked files.

## 0. Finding the print
* `data/detector_plans.parquet` maps all 945 `DeviceId` <-> `DeviceName`. The PDF file name starts
  with the DeviceName (e.g. `<DeviceName> …pdf`), under `Region N/…`. Use the cached listing
  `%DC_WORK%/cabinet/share_listing.csv`; if two versions exist, take the newest and record both.
* Signals analysed in isolation: identical layouts at different signals often reuse the same
  configuration and channel descriptions — that is fine, not an error.

## 1. Cabinet type and the INPUT FILE — loop numbers are NOT detector numbers
This is the step that is easy to get wrong. A **loop number** is a label on the drawing. The
**detector number** (the channel in the hi-res log, `Parameter` of events 81/82) is decided by the
input-file **slot** the loop is landed on:
1. Read the cabinet type from the print (332, 332S, 336; 336 is rare and hard to read — lower
   confidence).
2. Read the INPUT FILE drawing: per slot, the loop number(s) landed on the upper (`D`/`E`) and lower
   (`J`/`K`) inputs, and the phase marked there (`Ø2` / `◇2`). On a 332 the left block is the `I` file
   and the right block the `J` file (terminal blocks TB2… list `I1-D`, `I1-E`, … `J9-K`). Several
   loops on one slot (e.g. "3,4" or "11,12,13") are **one detector**.
3. Look the slot up in `research/notes/standard_detector_mapping.md` (`SLOT_TO_DET[cabinet][slot]`)
   to get the detector number.
4. Then find the same loop numbers on the intersection diagram and confirm the phase and read the
   lane (below). Input-file phase, diagram phase and the official timing phase can disagree: the
   **timing is the phase truth** (it comes from the controller in service; the print is only
   documentation). Record all three; any disagreement lowers confidence.
* **Three input-file drawings** (all vector text, scripted by `research/code/cabinet/cab_pdf.py`):
  332/332S terminal-block drawing on sheet 2 (TB2–TB7, "Phase / Loop No." beside `I1-D` … rows;
  loop numbers can wrap to a second line, `BIKE` written under them = bike loop); 336 front view (slot
  columns 1–14, phase + loops in the D/E and J/K boxes); 332S loop table on the last sheet (loop →
  distance ft → phase → slot; a bracket joins tied loops). The printed ◇/Ø phase markers are mostly
  template defaults (drawn on unused slots too; one pilot print had ◇6 on a phase-3 slot; 2-phase
  signals fold 6→2, 8→4) — weak evidence, never above the timing.
* **The diagram's loop numbers can differ from the input file's** (a pilot print: input file 22 loops
  matching the config and the active channels, diagram 18 loops numbered differently). Then the input
  file + config + data decide the channel and the diagram gives lanes only; within-phase lane
  assignments drop to medium.
* **Config channel descriptions can be stale or swapped** (a pilot print swapped two stop-bar pairs and
  put bike text on unused channels). The print's input file wins; confirm with the advance → stop-bar
  volume pairing (same lane ⇒ similar counts).
* **Detection-configuration workbook** (`…Detection_Configuration_332[S].xlsm`, ~150 of ~760 training
  signals, mostly radar sites; `cab_xlsm.py`): sheet `ZoneConfigurationTable`, one row per channel —
  `MT #` = detector number, phase, device letter (radar/camera unit), slot, function text (`P 0-20'`,
  `CO*`, `YR*`, `A*`, `Count`, `Loop 3,4`, `Bike Loop 5`), sometimes lane text; used rows are marked `x`
  in column A (SDLC zone) or G (loop slot). Its phase matches the timing on 94 % of used channels, its
  function text is essentially the config's own source (not an independent label), and it has no
  geometry. Use it as the channel list and cross-check on radar sites; lanes still come from the PDF.
* **Video and radar** zones are usually NOT landed in the input file (old configs may be). Their
  **detector number is written directly on the diagram** next to the zone, e.g. `CAM A 42` = camera A,
  detector 42; `2A/49` = radar zone, detector 49. Hatched areas are radar/camera zones — read their
  labels; they are detectors like any other.

## 2. The intersection diagram
* Usually one of the last sheets (sheet 5; sheet 6 on the newer 332S layout; sometimes sheet 1).
  **A print can contain more than one intersection diagram — check EVERY page and use the one with
  the loops/zones drawn** (pilot miss: one print had two; spot-check miss: a radar print read as
  "zones not drawn" had a diagram with every zone on another sheet). Never flag `position_not_drawn`
  or `no_diagram` until every page has been looked at.
* **ATSPM sheet.** Some prints split detection over two diagram sheets, one titled **ATSPM**
  (automated traffic signal performance measures). The ATSPM sheet holds exactly the detectors this
  project labels (Count, Yellow_Red, Presence, Advance); detectors drawn only on the other sheet(s)
  are operational → Other (with a subtype).
* Phases are labelled with Ø (theta/phi). Lanes carry pavement arrows. Loops: **circle = vehicle
  loop, diamond = bike loop** (a bike loop also sits in the bike lane).
* **Zoom.** Render at 150–200 dpi and crop/zoom into each approach; dense prints are unreadable at
  page scale (a pilot miss: a loop read as "on the lane line" was clearly in the right-turn lane).
* Drawings are **not to scale**: never measure distances. Record `distance_ft` (from the stop bar)
  only when it is printed (e.g. the 332S loop table).

## 3. What each detector is — location and function
Function classes: **Advance, Presence, Count, Yellow_Red, Other**, and — user decision 2026-09-23 —
**Bike** and **Mid** are real output classes too (the user wants them reported, Bike especially).
Record them as `function` = Bike / Mid, and a finer `subtype` for everything (departure, queue, …).
A detector with too few actuations may be left unclassified; low-count detectors are often bikes.
* **Technology limits (user, 2026-09-23 — hard rules):**
  * **Loops can never be stop-bar Count.** Loops count only in free flow, i.e. upstream. A loop the
    config calls "Count" is Advance if upstream, Presence if at the stop bar (config mislabel, flag
    `label_disagrees`). Loops are normally in "normal" mode (ON while the vehicle is over the loop),
    rarely pulse.
  * **Only radar and video have stop-bar Count zones. Yellow_Red is normally radar** (it needs a speed
    threshold) — a video zone next to the stop bar is Count **unless the print codes it YR**. **A zone
    the print labels YR / Yellow_Red is Yellow_Red whatever the technology** (user decision; checked on
    a thermal-video site whose YR zones detect start-of-green cars 1–2 s late, the speed-threshold
    signature).
  * **Radar/video Presence is always exactly 0–20 ft** from the stop bar. A zone labelled 0–75, 0–100
    etc. is **not** Presence → Other (subtype `long_zone`).
  * On radar, a Count zone and a Yellow_Red zone may sit **exactly on top of each other** (two bubbles,
    e.g. 41 and 42, pointing to one rectangle) or be drawn side by side — same meaning.
  * A Yellow_Red zone may span two lanes.
* **Stop-bar zones — the print DOES distinguish them by position** (user, 2026-09-23):
  * **Presence** = the zone from **0 to 20 ft before (upstream of) the stop bar**. Two or more loops
    close together there, tied to one input = one detector.
  * **Count** = zones drawn **on the stop bar or shortly past it** (downstream). Always one per lane.
  * **Yellow_Red** = zones drawn **right next to the stop-bar count zones** (often one detector across
    all lanes).
  Read the position carefully (zoom). Cross-check with the config label, the model's out-of-fold
  prediction and behaviour (pulse-mode ONs, occupancy through red, a Yellow_Red missing the first
  car of green); lower confidence when they disagree. Keep a crop of every Count / Yellow_Red zone
  you label (`dc_work/cabinet/crops/<DeviceName>_d<det>.png`) — the user will spot-check a sample.
  **Crops must be wide enough to show the zone's label bubble/number and the stop bar** — a crop
  that shows the rectangle but not its label is useless to the reviewer.
* **Confidence calibration (user, spot-check).** When the zone's number and function are written
  clearly on the print (a zone table with CO / YR / P codes, or a labelled bubble on the diagram) and
  the timing phase agrees, that is **high** — do not mark a clearly labelled zone medium or low just
  because it is video/radar or has few actuations. Record disagreements (print phase vs timing, dead
  channel) as flags, not by lowering a clear reading.
* **Advance** = advance count: lane-by-lane loops upstream, each its own detector.
* **Mid** (function Mid): loops spanning both lanes (typically a tied pair across the
  lanes on the highway, one detector). If a phase has no lane-by-lane advance loops, a mid pair may
  serve as its advance count — then label it Advance; if it also has lane-by-lane advance loops
  with different detector numbers, those are the Advance detectors and the spanning pair is Mid.
* **Bike** (function Bike): diamond symbols / loops in the bike lane. Label them.
* **Departure, queue, ETA, extension, anything that fits no category** → Other with a subtype
  (e.g. a loop on a departing lane: Other/departure). Be flexible — new kinds will turn up; name the
  subtype plainly and flag it rather than forcing a class.
* **Shared upstream loops (flag `shared_upstream`)**: an upstream loop on one phase that vehicles of
  another phase also cross (e.g. a phase-2 upstream loop that every phase-5-bound vehicle crosses,
  so phase-2 volume = loop count − phase-5 volume). Record it; not modelled yet. Check: upstream count
  − the same phase's downstream loop ≈ the turn bay's advance count.
* **Two stop-bar detectors in one lane** (e.g. a right-turn flare): the one covering the lane's main
  stop bar is the Presence; the other is Other/`stopbar_secondary`, medium confidence.
* **Camera/radar zones whose config text says "bikes"** and that count at bike level: record the
  drawn location, flag `suspect_config_or_health`, confidence low.
* Active channels described as dummy / FYA / "Pass" / ped / preempt are derived channels: they are
  accounted for automatically (`explained`), not print detectors.
* Right-turn loops wired to a phase in the timing are labelled with that phase (the model outputs a
  phase, never an overlap).

## 4. Lanes
* `n_lanes` per phase = **total lanes served by that phase over all its approaches** (the model cannot
  tell approaches apart; never try). A phase has 1–3 lanes, occasionally more on split phasing.
* Per detector: `lane_index` within the phase (1 = leftmost lane as the approaching driver sees it;
  if two approaches, number the first approach's lanes first and record the approach), and whether
  it **spans** several lanes (`lanes_spanned`).
* Standard `lane_type` vocabulary (use exactly these): `L`, `T`, `R`, `LT`, `TR`, `LR` (shared
  left/right on a T-intersection stem), `LTR`, `bike`, `departure`, `other`. Orientation trap: for a
  driver heading down the page, their left is the picture's right.
* One marked lane with two side-by-side stop-bar columns on separate detectors (L/R split arrow):
  count 2 lanes (L, R), medium confidence.
* Rules: at most one Presence, one Count, one Advance **per lane**; Yellow_Red may span lanes.
  Two count zones on a phase ⇒ two lanes. Where the stop-bar zones share lanes with advance loops,
  the timing shows it (below) — the print may be ambiguous; the data decides.

## 5. Sanity checks against the hi-res data (reusable DuckDB scripts, `research/code/cabinet/`)
The prints are not always right, things are landed wrong in the field, and detection breaks. Every
print-derived detector gets data checks; failures lower confidence or flag the detector.
Use 5- or 10-minute bins of Detector On (82) counts, de-duplicated, `Parameter <= 64`.
* **Stop bar vs advance, same lane** (minor movements, left turns with loops): counts correlate
  closely off-peak; as traffic rises the **advance counts more than the stop bar** (the stop-bar
  presence undercounts and may flatline at peak — that is the giveaway of a presence zone). If the
  pattern is absent → possible data or configuration issue. *Measured (note 24): this holds for
  stop-bar **Presence**; a stop-bar **Count** zone does the opposite (the advance undercounts it at
  peak, n = 8), so `dq_core` runs the peak test only for Presence.*
* **Toolkit**: `research/code/cabinet/dq_core.py` — `run(dets)` / CLI; use the **newest** data window
  (dead detectors change between periods). Never pass locked signals (the staging cache holds them).
* **Major movements** (setback front loops spanning lanes + lane-by-lane advance loops): the
  spanning loop tracks the **sum** of the advance loops off-peak and falls below it as traffic rises
  (two side-by-side vehicles = 2 actuations on the pair, 1 on the spanning loop).
* **Same-lane order**: an advance actuation is followed a few seconds later by the stop-bar zone in
  the same lane; a single off-peak vehicle crossing advance loop A lights A then the stop-bar zone
  while the neighbour lane's advance loop B stays off ⇒ A and B are different lanes and the stop-bar
  zone is shared or in A's lane. Needs off-peak data for the fine correlations and peak data to see
  them break down — use the full day.
* **Saturation**: ~1,800 veh/h/lane ⇒ a single-lane detector should not exceed ~**150 actuations per
  5 min**; above that only a zone spanning several lanes at peak is plausible.
* **Bike detectors**: counts must be plausible and far below the vehicle detectors'; a bike loop
  counting like a vehicle loop is set too sensitive → flag, do not train on it.
* **Zero actuations** for a detector on the print over the whole window → likely a complete failure:
  list it for maintenance (`review/dead_detectors_from_prints.csv`), exclude from training.
* **Video/radar**: the same comparisons across zones by lane/phase.
* Each check returns a score; a detector with a low data-quality score is flagged
  `suspect_config_or_health`.

## 6. Output (one row per detector channel, `%DC_WORK%/cabinet/print_labels.parquet`)
`DeviceId, DeviceName, detector, cabinet_type, pdf, diagram_page, technology (loop|video|radar),
loops (e.g. "3,4"), slot, phase_input_file, phase_diagram, phase_timing, function, subtype, lane_index,
lanes_spanned, lane_type, n_lanes_phase, distance_ft, flags, dq_score, confidence (high|medium|low),
confidence_reason`, plus one row per **active hi-res channel not on the print** (function Other,
subtype `unexplained`, flagged) and per print detector with **no** data (flag `dead`).
* **Intersection tiers**: `complete_high` = every detector on the print mapped with high confidence
  and every active channel accounted for; `complete_mixed` = all mapped, some medium/low;
  `incomplete`. Training uses `complete_high` first; medium/low go to the user for review.
* **Overrides**: where a print label is high confidence and the model's out-of-fold prediction agrees,
  it overrides the hand label. Write every overridden hand label to
  `review/print_label_overrides.xlsx` (the user asked for this list).

## 6b. Tooling (`research/code/cabinet/`, paths from `DC_WORK`)
1. `python cab_pdf.py <DN> …` — copies the newest PDF/xlsm (refuses locked signals), extracts cabinet
   type, input file, loop table, camera/radar labels, ranks diagram pages, renders pages + tiles →
   `cabinet/signals/<DN>.extract.json`, `cabinet/render/<DN>/`. `python cab_pdf.py zoom <DN> <page>
   fx0 fy0 fx1 fy1 <tag>` crops a region given as fractions of the rendered page.
2. `python cab_data.py <DN> …` — actuation counts per channel (both event windows) → `activity.parquet`.
3. `python cab_build.py sheet <DN>` — one line per channel: slot, input file, zone label, xlsm, timing,
   description, function label, counts. Read the diagram render/zooms, then write
   `cabinet/signals/<DN>.json` (schema in the `cab_build.py` docstring).
4. `python cab_build.py build` — `print_labels.parquet` (+ unexplained / dead rows, tier) and
   `print_tiers.csv`.

## 7. Known pilot mistakes — do not repeat
Loop numbers taken for detector numbers; missed the second diagram; missed camera labels
(`CAM A 42` = detector 42, a stop-bar presence zone); did not use the input file to see that loops
11-12-13 are one detector; read at page scale instead of zooming; two separate stop-bar presence
detectors on one phase missed because the input file was not read.

## 8. Radar over loops, and unusual sites (user, 2026-09-23)
* Where radar zones sit over loops, **the radar is the detection**: label the radar zones; the loops
  underneath are Other/`superseded_by_radar`.
* Signals with a rare custom set-up the model cannot be expected to read from behaviour (live radar
  over live loops, several intersections on one controller, …) are flagged `unusual_layout`. They are
  **excluded from training and scored separately**; if they turn out to be common, they become a
  research target of their own.
