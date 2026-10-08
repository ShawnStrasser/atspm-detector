# 78b — How many more signals could get print labels? (feasibility only, 2026-10-03)
Read-only count; nothing trained, nothing pulled, no database query. Share used read-only (40 PDFs COPIED to
`%DC_WORK%/tmp/cab78b/pdf/`, never written). Scratch lists only (not kept). CPU, minutes.

## 1. Signals outside the frame and outside the hold-outs
Universe = timing inventory (`data/detector_plans.parquet`, 945) + function config exports (`data/raw/detector-configs.csv`
418, `data/labels/detector_config_current.parquet` 466, statewide `all_configs.csv` 444) + every DeviceId seen in the
local event logs (staging Sept 18-21, statewide one day Feb 2025): 1,013 devices. Minus frame `folds_v4.csv` (780),
`locked_v2.csv` (115), old NEWTEST 143 / TEST 43 (all already inside one of the two): **118 left**.
* 67 are in the timing inventory: 65 have NO detector actuations in any local window (the known `batch_nodata` set,
  no training value); 1 with Sept data (2B559); 1 with Feb-2025 one-day data only (03098).
* 51 are NOT in the timing inventory (their timing export failed: `data/detector_plans_failures.csv`, 75 rows,
  mostly server "graphql" errors): 35 have Sept 18-21 detector data (8-40 channels, 16k-309k ONs each), 6 Feb-2025
  one day only, 10 nothing.
* Function config exports for the 118: 4 only. So in practice the pool = **36 signals with Sept data**
  (+6 with one Feb-2025 day, stale).

## 2. Prints on the share (same discovery as note 23: DeviceName prefix in the cached listing)
* 35 of the 36 Sept-data signals have a PDF (the 36th has no name/print); 5 of the 6 Feb-only ones too.
* Scripted probe (cab_pdf cabinet type / input-file parsers / text layer, as `cab_scale._status`) on the 40 PDFs:
  Sept-data 35: **ok 29, input_file_blank 1 (radar/video/SDLC), no text layer 5** (the 1205x/1206x Region-12 prints);
  cabinets 332 x 31, 332S x 4 (on the readable ones). 9 mention radar/camera. Prints dated 2018-2026 (20 of 35 >= 2024).
  2 xlsm workbooks.
* Training base rates by scripted status (688 signals): complete_high 18 %, complete_mixed 55 %, incomplete 27 %;
  median 78 % of print detectors high confidence, ~12-13 high detectors per signal, ~5 visual minutes per signal.
  Expected for the 35: **~26 complete-tier signals, ~400-450 high-confidence detector labels**.

## 3. Hi-res events
* The 35 already have local events: `data/staging/` Sept 18-21 (same window as the frame's stg; detector events) and
  `data/staging_other/` (other event codes). NOT in the 7-day w40 pull (that pull covered 888 devices, none of the 118)
  and not in Dec-2024. **No new event pull is needed** for the frame's window.
* The real gap is **phase truth**: these signals have no official timing export (`call_phase`), the AGENTS.md phase
  truth. Options: (a) one-time re-try of the timing export for the 35 (tiny, config only, but it is a new pull + login
  and previously failed with server errors -> needs the user's OK); (b) use them for FUNCTION only, phase from the
  print input file / diagram (94.9 % agreement with timing on old cabinets, note 34) -- cleansing layer allows it,
  but phase training/scoring on them would not follow the binding phase-truth rule.

## 4. Expected gain (note 78 learning curve, ~+1.1-1.2 pt per doubling of signals on the trees, ~0.5-1 on the champion)
Training base 772 signals (err78 rows). +35 -> log2(807/772) = 0.064 doublings -> **trees +0.07 pt, champion +0.03-0.06 pt**;
even +41 (with the stale Feb-only ones) <= 0.09 pt. Below the seed noise (+-0.02-0.07) only barely, far below the
1-pt function screening rule: **not measurable**. This agency's inventory is exhausted (895 of 945 timing signals
already in frame or locked); a real doubling (~770 more signals) can only come from another agency.

## 5. Effort if done anyway
Scripted pass + render ~5 min total; visual reading ~35 x 5 min = ~3 h agent time (5 no-text prints are slower);
records / cab_final / label check / DQ / frame rows ~2-3 h -> ~5-6 agent-hours, plus (a) above if phase labels wanted.

## 6. Side observation (not this task's count)
Inside the frame, 175 signals are `incomplete` print tier and 98 frame signals have no training print record (71 are
the released NEWTEST, read into the locked store). Finishing those raises label quality, not signal count; note 78
puts label-caused error at 0.4-1.5 pt, so that is the larger lever of the two.

## Verdict
Feasible but not worth it for accuracy (~+0.05 pt); worth it only if the user wants these 35 interchange / ramp
signals covered for their own sake, and then option (a) needs his OK.
