# 83 — final_v3 candidate v4: every decision since note 77 in one package (2026-10-04) — not shipped, locked untouched
Package `%DC_WORK%/final_v3_candidate_v4/` (copy of candidate v3; v3 untouched; `model/` = final_v2). Code
`research/code/final83/` (fit83, export83, assemble83, oof83, parity83, bench83) + `health/h83_ticks.py`; work
`%DC_WORK%/final_v3_work/{v3fit83,f83}/`, `%DC_WORK%/health83/`. CPU <= 4 threads, GPU (q74) not touched, locked_v2 absent.
## What changed (full-data fits; phase ranker, decoder, GRU unchanged: phase truth = timing, v4l does not touch it)
* Function trees 229 x 3 on v4l (298,713 rows / 691 signals; 226 rounds = mean of the 18 note-80 v4l fold fits).
* Lanes D (both orientations) on the v4l print-lane truth (10,139 pairs vs 9,961; +335 / -157 pairs, 0 same-lane flips)
  and the v4l OOF function block; 78,423 labelled pair-windows; lam 3 (note-77 fold mode).
* Setback P50 per length group on sb7 features rebuilt from the v4l store (cabinet_v4l) + v4l OOF function (42,289 rows).
* Function net = full-data siba on v4l (`tcn53/models/x74_sibafull4l_full.pt`), re-exported as the package's pair + head
  graphs; the head now masks candidates the pair net did not run (= research forward69 keep mask). Parity vs torch on 8
  bench signal-lengths, every piece: |dlogp| 2.1e-5, |dprob| 1.4e-6; with a random candidate filter 1.9e-5 / 1.1e-6: PASS.
* Context stacker 'single' (3 x rows: the three single-seed siba OOF versions), v4l target + v4l trees, 47 columns: the
  15 health columns out (note 79b), hf_clus_n and pk_unhealthy kept; mean3 variant dropped. Package skips the
  health_core run inside the function stage (no stacker column needs it).
  BUG caught on the way: importing f76_function resets t57_function.OUT to the v3s trees; the first stacker fit had
  v3s tree inputs (check in oof83 failed: |dP| .59) -> fixed (import order + assert on S59.FUNC_DIR), refitted.
* Health (package health_core; research copy synced): OFF->ON gaps and ON->ON intervals compared in whole 0.1-s ticks
  (chatter < 3, rapid < 5 / < 10). On 234k detector-windows (stg + w40, 30 min / 3 h / 24 h; 135k presumed healthy):
  chat_frac changed on 97k rows (mean .0279 -> .0228); rapid statistics on 133 / 197k (max .02). Limits keep the
  presumed-healthy fire rate: chatter suspect .30 -> .263 (.0548 % -> .054 %), bad .60 kept (.0015 %); rapid limits
  unchanged (rates identical). Flips over all rows: chatter +53 / -33, rapid 0 / -21 (of 234k).
  "Dead channel" check removed (channel list ignored). "Undercounts vs partner" (note 79 c1, p99.5 per (own, partner)
  function and length, >= 20 actuations on both, channel-free tie-break) = INFO note in health_watch, status unchanged.
* Siba candidate filter (tree p >= .01): `predict(siba_filter=True)` / `--siba-filter` / DC_SIBA_FILTER=1, OFF by default.
## Checks
* check.py 4 / 4 PASS (stored answers; phase renumbering: every probability <= 1.6e-7; channel reversal / shift: 0.0;
  torch / lightgbm / scipy / sklearn blocked). References re-frozen (deliberate model change). ONNX trees vs text 2.0e-14.
* Parity vs the v4 OOF (parity83, 6 fold-0 signals x m5 / m10 / m30 / h3 / h24 / full66, 741 detector-windows), A =
  injected OOF inputs + the fold-0 v4 stackers: 229 features identical (169,689 values), pick inputs identical, FINAL
  FUNCTION 741 / 741 identical. 10 lane strings differ (the OOF lanes were decoded on the note-57 v3s function block, the
  injected trees are v4l) -> stacker probability <= .18 there, no answer change. B = production models: phase top-1 = OOF
  98.7 %, final function 94.3 % (different siba / full refits; v3: 96.0 %).
* Bench signals, v3 vs v4 (6 signals x 3 h / 24 h, 311 detectors): health status 0 changes, function 8 changes, 4
  partner notes (e.g. Count at 10 % of its Presence partner; healthy pairs go down to 23 %).
## OOF headline of the v4 recipe (six folds, v4l truth, paired signal bootstrap; net = each single siba seed in turn)
| v4l truth | v4 recipe | champion (note 81) | v4 - champion, pt [95 % CI] |
|---|---|---|---|
| >= 30 min E | .9181 [.9070,.9279] | .9190 | -0.09 [-0.18,0.00] |
| >= 30 min R | .9279 [.9170,.9372] | .9289 | -0.10 [-0.18,-0.01] |
| 10 min E | .8993 | .9030 | -0.37 [-0.51,-0.23] |
| 5 min E | .8907 | .8936 | -0.29 [-0.44,-0.15] |
Net seeds >= 30 E .9179 / .9186 / .9178. Health-column drop alone -0.01 [-0.04,+0.01] (>= 30 E). By class (>= 30 E):
Count -0.36 [-0.58,-0.17], Advance -0.12, Presence +0.04, YR +0.26, non-ATSPM -0.13. The loss = one siba member instead
of the 3-seed mean (note 75: -0.12 / -0.35). Caveats: the OOF net is v3s-trained (production: full-data v4l); lanes OOF
held at note 77.
## Speed / RAM (bench83: one signal per call, fresh process, 4 threads; warm median of 3 / cold; peak MB)
| filter OFF (default) | 30 min | 3 h | 24 h |
|---|---|---|---|
| typical r8 22 ch | 1.27 / 3.33 (643) | 3.25 / 3.99 (609) | 3.62 / 5.19 (632) |
| typical r11 18 ch | 1.10 / 2.34 (539) | 2.41 / 3.26 (565) | 2.70 / 3.89 (576) |
| busiest events 31 ch | 2.01 / 3.02 (690) | 4.25 / 5.47 (629) | 5.35 / 6.67 (723) |
| busiest channels 43 ch | 2.18 / 3.12 (699) | 5.19 / 6.39 (684) | 6.43 / 7.52 (787) |
Filter ON, warm 3 h: 2.29 / 1.98 typical, 2.67 / 3.33 busiest (24 h busiest 3.78 / 4.52); peak <= 0.80 GB. Import .6-.8 s.
## Open for the orchestrator
* Siba filter: switch on only if the GPU check (x74_sibaflt vs x74_sibanf) is neutral; it cuts busiest 3 h 5.2 -> 3.3 s.
* Single-member cost (-0.09 >= 30 min, -0.3 / -0.4 at 5 / 10 min) stands; a v4l six-fold siba re-run would make the
  stacker inputs consistent with the production net. pick.py's own chatter share (pk_unhealthy) still uses float gaps
  (kept for training parity).
