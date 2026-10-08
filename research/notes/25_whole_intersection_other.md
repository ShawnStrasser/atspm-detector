# 25_whole_intersection_other — active channels with no function label (2026-09-23)
`research/code/trackA/w1_whole_intersection.py` (`--stage audit|exp|report`) → `%DC_WORK%/trackA/w1/`.
Training/dev signals only (399), six signal-grouped folds, all out-of-fold; locked 43 + 143 asserted
absent. Active = >= 5 actuations in the period, channels 1–64.

## Part 1 — size of the problem
| period | signals | active channels | labelled | **unlabelled** | unlabelled / signal (median, mean) |
|---|---|---|---|---|---|
| Dec-2024 | 372 | 7,801 | 6,291 | **1,510 (19 %)** | 2, 4.1 — 181 signals have >= 3 |
| Sept-2026 | 384 | 8,891 | 6,739 | **2,152 (24 %)** | 4, 5.6 — 220 signals have >= 3 |
Busy, not idle (median 3,200 actuations vs 4,400 labelled). **All are in the function frame with
features (72,228 rows) and all are dropped from training** (`func5` null). Timing text:
dummy/fail-safe channels 27 %, empty 29 %, free text 33 % (zone names like "0-20P", loop
numbers, "call"/"extend" channels), text naming a PM role 10 %, pass-through 6 %, FYA 4 %, bike 2 %.
A mix: some clearly not PM detectors, some PM detectors the config export omits.
Current head (T, note 14, 442 features), OOF, full window (3,662 channel-periods, 292 signals):
**Other 49.5 %, Presence 27.1 %, Advance 12.7 %, Count 8.6 %, Yellow_Red 2.0 %**; median
confidence .75 (labelled rows .90); 24 % are a non-Other class at p >= .80; median 2 channels per
signal-period come out as a PM class. By text: dummy/fail 80 % Other, bike 88 %, pass-through 70 %,
FYA 62 %, empty 42 %, names-a-role 31 %, free text 27 %. **This is the user's concern, confirmed:
about half of the extra channels would be reported as Advance/Presence/Count/YR.**

Phase: 91.8 % have an official phase target; the ranker trains and is scored on every channel with
a timing target regardless of function label, so they are already in phase training and the phase
headline. OOF full-window top-1 **.944** vs .979 on labelled channels.

## Part 2 — one cheap experiment (six folds, CPU, existing 442 features)
Subtypes from the config string: mid (any "*mid*"), bike, advance-presence, other. Scored on
labelled rows, all windows (261,614); "unl→Other" = share of the 72,228 unlabelled active rows
predicted Other (Mid/Bike collapsed to Other).
| variant | seeds | 5-class | full | A/P/C | Other P / R | unl→Other (full) |
|---|---|---|---|---|---|---|
| a  T, trained Other (baseline) | 3 | .7911 | .8147 | .8315 | .672 / .617 | .466 (.491) |
| b  Other split into 4 subtypes, summed back | 3 | .7908 | .8125 | .8373 | .699 / .589 | .435 (.418) |
| **b7** 7-class: A,P,C,YR,**Mid,Bike**,Other (user decision) | 3 | **.7922** | .8142 | **.8363** | .693 / .602 | .449 (.445) |
| c  (b) + unlabelled active rows as "unexplained" Other | 1 | .7693 | .7998 | .7815 | .520 / .744 | **.677 (.686)** |
Seed sd 0.05–0.06 pt. b7 outputs (seed-mean): **Bike P/R .736/.672** all windows, **.816/.784** full;
**Mid .590/.516**, full .605/.490; 7-class accuracy .7843 (full .8052).

Reading. (b)/(b7): splitting Other costs nothing on 5-class (b7 +0.11 pt) and gains ~0.5 pt A/P/C
(real, 10× seed sd, but below the 1-pt function bar) — the subtypes stop pulling A/P/C rows into a
fuzzy Other. It does **not** help the whole-intersection problem: unlabelled→Other drops 2–5 pt,
likely because Other must now win as one subtype. (c) is the only thing that moves the
unlabelled channels (→ 68 % Other) but costs **2.2 pt 5-class and 5.0 pt A/P/C**: labelled Presence
→ Other goes 8 % → 18 %, Count 3 → 11 %, Advance 5 → 11 %. Even trained on them, 32 % of the
unlabelled channels still come out as PM classes out-of-fold, and text naming a role is only 51 %
Other — consistent with a pool that contains real, unlisted PM detectors. **Sensitivity run only:
without a print we do not know the config is complete at these signals, so (c) trains on labels we
cannot trust; it is not a candidate.**

## Verdict
* Problem is real and sizeable: ~20–24 % of active channels at a labelled signal carry no function
  label and the shipped-style head calls about half of them a PM class.
* b7 (Mid, Bike as outputs) is free and does what the user asked; A/P/C +0.5 pt, below the bar, so
  it rides with the round-2 retrain rather than being promoted on its own.
* The whole-intersection fix needs **print-complete signals** (AGENTS.md rule): only there is
  "unlabelled active ⇒ Other" a true label. Re-run variant (c) restricted to those signals once
  prints are read; until then add no unlabelled rows to training.
* Leaving low-activity channels unclassified (user OK) will not absorb this pool — it is as busy as
  the labelled one. Dummy/fail-safe channels (27 % of it) are the easy part, already 80 % Other.
