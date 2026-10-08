# 100c — Package v5b: v5 with three width-32 siba members (orchestrator: adopted, "ties go to the faster") (2026-10-06)
Not shipped (model/ = final_v2); locked signals untouched. Code `final100/{pkg100,assemble100,f100}.py`; fit folder
`%DC_WORK%/final_v3_work/v3fit100` (copy of v3fit95; only stacker/ refitted); packages `final_v3_candidate_v5b`,
`final_v3_candidate_v5b_fast` (v5, v5_fast, v3fit95 untouched); bench `final_v3_work/f98/bench/c100_*`.
## 1. Full-data members x100_w32full{,_s1,_s2} (local A1000, queue tcn53/q100, snap95)
s95_sibafull recipe at --width 32: every non-locked 2026 training signal (719), func_rows_v4q_2026, --accum 2, 40 epochs =
median best epoch of the 18 w32 fold runs + 1, lr = per-epoch median of their replayed schedules (`s100/lr_w32_full.json`;
written there, not over s95/lr_siba95_full.json). 20 min each. Final train loss .560 / .515 / .494 (v5 w96: .491 / .523 / .496).
## 2. Stackers (pkg100.py = pkg95 pkgsingle / pkgstacker with NET = x100_w32): 176,912 Sept-2026 rows (= v5).
## 3. Assembly (assemble100.py init / func / siba / card)
* func: ONNX from v3fit100, 24 compiled, 21 checked, max |out diff| 1.9e-14; diff vs v5 = the 7 stacker files only.
* siba: export83 pair + head graphs; parity package FuncNet vs torch on the bench extracts (120 pieces): flogp 1.5e-5,
  filtered 1.7e-5, 3-member average |dprob| 1.1e-6 -> PASS 1e-4 (`v3fit100/siba_parity100.json`). Member argmax
  disagreement on the bench pieces .105 (v5 trio .103): no outlier member. Manifest + model card `final_v5b_note100` + sha256.
* check.py --freeze, check.py: 4/4 PASS (stored answers 0.0; phase renumbering 2.6e-8; channel order 0.0; torch / lightgbm /
  scipy / sklearn blocked). Bundled sample vs v5: phase / lanes / status identical, function 1 of 17 detectors differs.
## 4. Accuracy (OOF of the same recipe, note 100b; v4q truth, Sept-2026 rows)
>= 30 min E .9307 vs v5 .9305 (+0.02 [-0.09,+0.14]); R .9397 vs .9395 (+0.02); 5 min E -0.07 n.s.; 10 min E -0.20
[-0.40,-0.01]. Phase unchanged (= v5 .9843 E). Fast profile le2h (function nets <= 2 h, nonet stacker above): .9275 E =
v5 le2h .9275 (+0.01 [-0.06,+0.07]), R +0.00; vs full v5 -0.30 (same as note 95b).
## 5. Speed (bench98: fresh process per signal, 4 threads, warm median of 3, mean of 2 passes, idle machine, one at a time)
| warm s, typical r8 / r11 / busiest ev / ch | beta | v5 | v5b | v5_fast (le2h) | v5b_fast (le2h) |
|---|---|---|---|---|---|
| 30 min | .98 / .74 / 1.39 / 1.72 | .91 / .74 / 1.11 / 1.27 | **.71 / .60 / .85 / .99** | .92 / .75 / 1.11 / 1.29 | **.72 / .61 / .87 / 1.01** |
| 3 h | .44 / .39 / .57 / .52 | 2.01 / 1.53 / 2.18 / 2.65 | **1.33 / 1.02 / 1.59 / 1.78** | .68 / .57 / .97 / .90 | .68 / .57 / .95 / .93 |
| 24 h | .82 / .55 / 1.12 / 1.01 | 2.38 / 1.70 / 3.43 / 3.85 | **1.94 / 1.32 / 2.91 / 2.94** | 1.38 / .89 / 2.28 / 1.95 | 1.38 / .88 / 2.28 / 1.95 |
| peak MB 30 min / 3 h / 24 h | 575-1575 / 246-279 / 296-399 | 508-760 / 519-786 / 558-892 | 405-531 / 400-546 / 456-670 | 504-750 / 290-328 / 334-457 | 401-545 / 293-329 / 332-454 |
* v5b vs v5: 3 h -27..-34 %, 24 h -15..-24 %, 30 min -19..-23 % time; peak RAM -19..-31 % (3 h).
* v5b vs beta: 30 min faster than beta (0.58-0.81x) with less RAM; 3 h 2.6-3.4x time, 1.6-2.0x RAM; 24 h 2.4-2.9x -> over the
  user's 2x rule above 2 h, so the fast setting is still needed there.
* v5b_fast (default le2h): above 2 h = trees-only, identical to v5_fast (same answers by construction); at <= 2 h the w32
  nets make it 19-22 % faster than v5_fast and faster than the beta at 30 min.
## 6. v5b_fast = v5_fast code + `weights_v5b` (v5b weights + v5's extras decode_trees / stacker nonet + v5b's 'single');
package.json -> weights_v5b; references `reference/weights_v5b`; check.py 5/5 PASS (incl. the no-network path). 30-min sample
answers = v5b exactly (max diff 0.0).
## Open (orchestrator / user)
* Swap v5 -> v5b (and v5_fast -> v5b_fast) as the candidate: tie on accuracy, 27-34 % faster at 3 h, 19-31 % less RAM.
* The final morning tables (note 101) still describe v5; re-run them on v5b if v5b becomes the candidate.
