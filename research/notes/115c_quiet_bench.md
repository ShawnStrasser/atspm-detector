# 115c - quiet-machine speed / RAM bench: beta vs v6b vs v7 prod (2026-10-07, CPU only)

Machine check before start: 0 % CPU load, no busy python / duckdb process (two idle python, 0 CPU-s in 5 s).
During the run: whole-machine CPU 8-17 % (our 4 threads ~ 14 % of 28 logical cores); other load only OS /
antivirus (MsMpEng <= 0.6 core, System, webview); no other agent. Pass 1 vs pass 2 warm means agree within 1.5 %.

Protocol = bench98 (copy research/code/final115c/bench115c.py; B115_THREADS sets DuckDB + DC_TREE/NET_THREADS):
fresh process per case, warm = median of 3 after a cold call, peak = process peak working set, 4 bench71 signals
(typical_r8 22 ch, typical_r11 19 ch, busiest_ev 31 ch, busiest_ch 43 ch; none locked), 2 full passes.
Packages run from local copies: beta = model/ (final_v2) copied to dc_work/s115c/beta_final_v2; v6b =
final_v3_candidate_v6b; v7 = final_v7_prod/src via shim dc_work/s115c/v7shim/predict.py.
Results dc_work/s115c/bench/<cfg>_p{1,2}/r_*.json; table dc_work/s115c/table115c.json (table115c.py).

Warm s (mean of 4 signals x 2 passes) / peak MB (mean; max):
| package / profile | 30 min        | 3 h           | 24 h          |
| beta              | 1.19 / 1076 (1558) | 0.47 / 272 (296) | 0.88 / 348 (395) |
| v6b full          | 0.67 / 331 (376)   | 1.09 / 360 (404) | 1.97 / 424 (483) |
| v6b le2h          | 0.67 / 336 (379)   | 0.76 / 252 (282) | 1.61 / 342 (401) |
| v7 full           | 0.58 / 242 (261)   | 0.89 / 263 (288) | 1.43 / 369 (442) |
| v7 le2h           | 0.59 / 239 (263)   | 0.66 / 246 (271) | 1.20 / 353 (424) |
| v7 le2h 1 thread  | 0.75 / 207 (214)   | 0.76 / 209 (229) | 1.67 / 305 (375) |

Import 0.27-0.29 s for all. Model load (cold - warm, median): beta 0.77-0.87 s, v6b 0.29-0.39, v7 0.27-0.42,
v7 1 thread 0.24-0.34.

Ratios (time = geometric mean of per-signal warm ratios; peak = ratio of means):
| v7 vs           | 30 min      | 3 h         | 24 h        |
| v6b, same prof. full | 0.87x / 0.73x | 0.83x / 0.73x | 0.74x / 0.87x |
| v6b, le2h       | 0.88x / 0.71x | 0.88x / 0.98x | 0.75x / 1.03x |
| beta, full      | 0.51x / 0.22x | 1.88x / 0.96x | 1.62x / 1.06x |
| beta, le2h      | 0.52x / 0.22x | 1.39x / 0.90x | 1.34x / 1.02x |

Read: v7 is 12-26 % faster than v6b in both profiles, and 13-29 % less RAM except le2h at >= 3 h (same).
v7 is within the 2x-of-beta rule everywhere (worst full 3 h 1.88x); at 30 min v7 is 2x faster and 4.5x
leaner than beta (beta's 30-min path peaks at 0.6-1.5 GB). 1 thread (edge proxy, le2h): +16-40 % time vs
4 threads, peak 207-305 MB mean (max 375 MB at 24 h, 31 ch) -> fits the ~1 GB edge budget with room.
Caveat: 4 signals only; times are single-call per signal, not batch throughput.
