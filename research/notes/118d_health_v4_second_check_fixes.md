# 118d - health v4 sheet: second verification pass

Input: the 118c-check report (40 rows, 80 charts). Charts and sheet re-drawn from the saved plot data
(`dc_work/s118c/plot`), never re-scored. Statuses unchanged (40/40 identical to the 118c-check sheet).

## Fixed
- Whole-phase events now said in the Reason and in the chart subtitle (`phase_note` in h118c_sheet.py,
  read from day15 only; fires only when flagged and >= half the phase mates do the same, or >= 2 for a dropout):
  - row 2 (10037 det 22): all 3 P8 mates >= 95 % ON over 14:10-16:02.
  - row 4 (01074 det 4): det 2 and det 3 (P2 Advance) silent over the same 03:30-23:20. Correction to the
    check report: the six P2 Other mates did not go silent; from 03:30 they count ~805 each with near-identical
    15-min traces (49 = 51 = 53, 50 = 52 = 54) - looks like a cabinet/input change, not traffic.
  - row 7 (2B316 det 3): 6 of 7 P2 mates fall >= 70 % at 20:15 (det 6 rises).
  - row 33 (2B316 det 6): all 7 P2 mates drop at 20:15 (sharpest 30-min step in the mates' sum, +-1 h of the
    first held-ON bin).
  No other row triggers the note.
- Row 34 (08019 det 3): legend now explains points above the line without a dot (not counted: signal busy or
  phase mates ON too - the check's `quiet` / light-traffic conditions in score_v4c.occ_hi_c).
- Mates cap raised 8 -> 12 (palette 12 colors; time-of-day charts 6 -> 12). Row 4 now shows det 54; the row 40
  day chart now shows all 11 mates (3 were hidden).

## Not changed (judgment / user)
- 08073 det 7 and 11021 det 2/3/4 are suspect where the user said bad: calibration choice, not a sheet defect.
- Row 11 (08CM405 det 38 P7 Presence) bad on ~7/h night expectation from the rest of the signal: judgment.
- 2B058 det 46 is "P6 Yellow-red" in row 18 and "P6 Count" in row 38: the model's label per sample; shown as-is.
- Statuses for whole-phase rows 2/4/7/33 left at bad (scorer output; user had called them bad).

## Checks
- `python h118c_sheet.py`: 40 rows, all reasons <= 12 words (assert), 40 + 40 PNGs.
- Re-opened rows 2, 4, 7, 33, 34 charts and row 40 day chart.
