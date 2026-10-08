# 35 — Cabinet prints for the LOCKED signals: stage 1 (scripted pass + batches), 2026-09-24

User approval: read the 186 locked signals' prints (43 TEST + 143 NEWTEST) to build a better answer key —
**labels only**. No model scoring, no predictions, no metrics on these signals; none were computed here.

## Opt-in, separate store (`research/code/cabinet/`)
* `cab_common`: `--locked-labels-only` on any cab_* command line (or `DC_CAB_LOCKED_LABELS_ONLY=1`; the flag sets
  the env var so pool workers inherit it). In that mode `CAB = %DC_WORK%/cabinet_locked/` (signals/, pdf/, xlsm/,
  render/, crops/, batches/, activity.parquet, scripted_status.csv, xlsm_zones.parquet), only locked signals are
  accepted and non-locked ones refused, so the two stores never mix. Default mode is unchanged (refuses locked;
  761 non-locked signals, same store). The share listing is read from `cabinet/share_listing.csv` in both modes.
* `forbid_locked_mode()`: `cab_final` (label table v3, OOF agreement, user lists) and `label_check` refuse to run
  in locked mode. `fix_*` (pure print-label rules) and `cab_build` (print_labels into the locked store) are allowed.
* `cab_scale`: `todo/records` use `in_scope()`; the `dq` step is skipped (no dq_core on locked signals); no pilots.
  `cab_xlsm` likewise. `cab_data`: DuckDB caps via `DC_DUCK_MEM` / `DC_DUCK_THREADS` (run: 8GB / 4).
* `signals()`: two locked ids missing from the plans table take their name from the plans-download failure list
  (both have no controller timing in labels_official either → no timing phase in their worksheet).
* `cab_pdf.extract` now also writes a 45-dpi thumbnail of EVERY page (`thumb_p<N>.png`) and a labelled contact
  sheet (`contact.png`), because the diagram ranking misses zone sheets (note 26); `cab_record show` prints it.

## Results (186 locked signals)
* PDF found 182; **no PDF on the share 4**; xlsm 33 (31 with a readable ZoneConfigurationTable). One version each.
* Activity: all 186 have Sept 2026 (staging) events; Dec 2024 only the 43 TEST. **No-data signals: 0.**
* Cabinet type: 332 129, 332S 35, 336 8, unread 10 (+4 no PDF).
* Scripted status: ok 118 (input file parsed), input_file_blank 45 + blank_zones 9 (video / radar sites),
  no text layer 9 (6 raster, 3 outlined), unknown cabinet type 1, no_pdf 4.
* Batch class: ordinary 157, complex 17, 336 8. Batches in `cabinet_locked/batches/`: **10** —
  7 ordinary (22–23), 1 × 336 (8), 2 complex (8, 9); snake-dealt by difficulty; `batch_nodata.txt` empty.
* Wall time: extract 2 min (3 workers), activity 30 s, records + batches 10 s. Disk: pdf 207 MB, render 515 MB.
* `cabinet_locked/batches/VISUAL_PASS_INSTRUCTIONS.md`: the training copy adapted — locked-rules header (labels
  only, no model output, flag on every call, no dq), store paths, contact sheet first, pilot section dropped.

## Open
* Stage 2 = visual pass (10 batches, ~186 × 1–2 min). Consolidation for the locked store needs its own builder:
  `cab_final` is deliberately refused there.
