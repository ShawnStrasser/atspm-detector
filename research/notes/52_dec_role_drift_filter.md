# 52 — Dec-2024 role-drift filter (note 51b; re-score only, no refit) (2026-09-30)
Follows note 51 (period drift: 142 channels wrong in Dec 2024, right in Sept 2026; labels describe today's role).
Orchestrator rule (labels must describe the data they label): compare each labelled channel with the Dec-2024 config
export `%DC_WORK%/data/raw/detector-configs.csv` (dated 2024-12-19, contemporaneous with the Dec events; 5,761 rows /
418 signals; vocabulary Advance / Presence / Count only). Independent config evidence; the model's right / wrong
pattern is NOT used to select rows (only to measure overlap with note 51's list). locked_v2 asserted absent.
Code: `cabinet/cab_final.py` step 10 `dec_role_changed` / `--dec-role` -> `%DC_WORK%/cabinet/dec_role_changed.parquet`;
`trackA/v3_retrain.py --dec-role` (`dec_role_rows`, run suffix `_dr`; training uses the flag vs the training label,
scoring vs the honest-set truth); `evaluation/err51b_decrole.py` -> `%DC_WORK%/trackA/err51/decrole.txt`.
## Rule
Dec rows dropped (training AND scoring; Sept rows keep the current label) where the Dec export lists the channel with a
different function (`function`), or a phase outside {call, switch, additional call phases} of the label (`phase`), or
omits a channel labelled Advance / Presence / Count at a signal it covers (`absent`). Kept: signal not in the export
(`no_signal`), current class outside the export's vocabulary and not listed (`not_listed`: YR / Mid / Bike / Other).
Per channel (scoring truth; label table 16,759 rows): same 3,677, function 473, phase 49, absent 677, not_listed 1,298,
no_signal 4,621 -> 1,199 flagged (training label: 1,527).
## Effect on function_v3e six-fold OOF (3-seed mean, note-49 sets; signal-bootstrap 95 % CI)
| set | rows dropped | channels / sig. | acc7 before -> after | errors |
|---|---|---|---|---|
| everything | 16,261 Dec rows (6.0 %) | 942 / 194 | .8724 -> **.8851** (+1.27 [+0.87, +1.71]) | 34,303 -> 29,027 |
| realistic | 15,501 (5.9 %) | 898 / 188 | .8845 -> **.8969** (+1.24 [+0.85, +1.62]) | 30,215 -> 25,377 |
* Dropped rows by status (everything; rows, acc): function 7,346 / .496, absent 8,132 / .823, phase 783 / .828.
  Function / phase alone would give ~+1.07 (everything) / ~+1.11 (realistic); `absent` adds ~0.2 pt but costs half the
  dropped rows, and those rows are mostly right (the export presumably lists only the performance-measure detectors, not every channel).
* Dec period acc .8551 -> .8926 (Sept .8817); gain flat by window (+1.2-1.4 pt every length, m5 .8455 -> .8572).
* Note 51's drift list: 143 / 142 channels (everything / realistic), **111 / 110 caught (78 %)**, 1,788 of their 2,341
  Dec rows dropped. Not caught: 16 where the Dec export agrees with today's label, 11 not_listed, 5 no_signal.
* Training (v6e first.all.wi, h3 recipe): 283,894 -> 268,871 rows (15,023 Dec rows / 853 channels / 177 signals).
## Caveats
* Scoring gain is partly mechanical: removing 6 % of rows at acc .68. It is justified only because the removed rows
  carry a label the Dec-era configuration contradicts. The export is itself a hand config (config-only labels score .797 vs
  print-high .895 in note 51), so some `function` rows may be export errors, not role changes.
* Not retrained (orchestrator: class definitions for ambiguous detectors come first). A refit with `--dec-role` should
  add a little more (cleaner training rows); screen it against the 1-pt bar only as part of that retrain.
