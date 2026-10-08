# 91 — review sheets re-laid in plain columns, answers carried (2026-10-05)
Brief (orchestrator): the user found cells like "P5, P8 stop-bar presence" ambiguous (a phase list + one function).
Fix the three open sheets without losing anything he typed. User correction mid-task: write each detector's phase and
function together, short form ("P5 Presence", "P4 Stop-bar Count", "P6 Yellow-Red", "P3 Other"), one detector per line
in grouped rows ("det 15: P5 Presence" / "det 23: P8 Presence"), same in Model says. CPU only, nothing re-scored.
## Sheets (review/, git-ignored)
* `function_label_review_v1.xlsx` (38 rows), `function_label_review_v1b.xlsx` (32), `label_sanity_20.xlsx` (20).
* Columns now: #, Signal, Detector(s), Label, Model says, How sure, What this row asks, Chart, Print, Answer
  (sanity: #, Signal, Detector, Label, What this row asks, Why it is left out, Chart, Print, Answer — no model column,
  that sheet has no model answer). Top question unchanged, chart / print links copied unchanged.
* "What this row asks", e.g. "Label says det 15 is P5 Presence and det 23 is P8 Presence; model says det 15 is P5
  Advance and det 23 is P8 Advance. Which is right?"; phase rows: "Label says det 19 is on P7; model says P4. Which
  phase is right?"; sanity: "Label says det 16 is P6 Advance, but it is left out of training (reason in the next
  column). Should it be used for training?"
## Where each value comes from (nothing re-selected)
* Rows = the saved build rows (`cand64/review72_rows.parquet`, `rev81/review81_rows.parquet`,
  `x89/v4n/label_sanity_20_rows.parquet`); asserted row-for-row equal (signal + detector list) to the sheet as it was.
* Per detector: label phase = label table phase_target (v4o; the union equals the old sheets' phase lists on every
  row); model phase = the model's most common phase over the detector's >= 30-min Sept samples (function_rows of that
  build: cand64 arm fj / rev81 arm champ). Function rows: label / model function = the row's (one per row by the
  collapse key). Phase rows: each detector's build-time label function ("function not labelled" when none) and most
  common model function.
* Only visible change of substance: v1 row 36 (05048 dets 14 / 28, Bike) shows model P6 / P2 where the old sheet
  repeated the label's P3 / P7; the phase model does not score these two (no phase rows), so it is not a phase
  question — left as the model's own answer.
## Answers preserved
* Backups, byte-identical (md5) to the originals: `review/_backup/<name>_20261005_0735xx.xlsx`.
* No file was locked (r+b open + no ~$ owner file) -> rewritten in place, no _v2 needed.
* v1: 4 answers (rows 1-4: 2B091 d21, 2B091 d9/27/28, 2C009 d19, 2C009 d5) carried to the same rows, text identical;
  v1b and sanity had none. Checked by a separate re-read: signal, detector list, answer, chart and print link equal
  on every row of all three sheets; no comments, extra sheets or cells right of the table existed. Spot-checked 5 rows
  per sheet against the backup.
## Code
* `evaluation/review72.py`: new writer (phase_txt / fn_txt / pf_txt, det_facts, row_cells, read_answers); a rebuild now
  carries answers already in the target file and refuses to drop any. `review81.py`: ARM = champ; v1 reader handles
  both layouts. `sanity89.py`: write_sheet split out, same answer guard. `evaluation/review91.py`: the re-layout (dry run
  -> %DC_WORK%/rev91/, `--write` = backup + verify + write).
