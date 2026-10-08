# 120 Fast switch removed: one path (package v7, `dc_work/final_v7_prod`) (2026-10-07)
User decision (Oct 7): the fast switch saved little, so remove it. Backup of the earlier tree: `dc_work/final_v7_prod_bak119`.
Version stays 7.0.0 (never shipped). CPU only. Scripts `research/code/final120/`; outputs `dc_work/s120/`.

**Removed.** Profiles le2h, le2h_siba1, siba1, trees; `weights/fast.json`; `fast_profile()` (and its export);
the `n_members` / `siba_members` plumbing (pipeline, function_stage, funcnet); blend.json `cutoff_minutes` + `gb.runs_on`;
3 models: `decode_trees` (.json + .onnx), `stacker_nonet_s0`, `stacker_single_s0` (stacker.json now mean3 only);
`reference/predict_reference_nonet.parquet`; check 5 "no-network path".
**Needed for robustness? No.** A probe of the models each call loads (par120.py / edge120.py) shows the current v7
at profile full used only decode_v3 + stacker mean3 + the network in all 132 parity cases and in 13 edge cases
(1 / 3 / 5 min, no calls, no 7-10, no 131, no Begin Green, one Begin Green, one detector, no detector, two signals,
empty window). Reason: the streams are built from the same candidates and detectors as the features, so the
network always has input when there is anything to score. Kept fallbacks (code only, no extra model): a signal
without streams keeps the ranker alone in the decoder; a detector the network gives no answer for carries the
trees' probabilities in the stacker's net columns (unchanged behaviour). A model folder without the network
(blend.json / graphs) now raises FileNotFoundError; FuncNet needs all 3 members (no silent 1-member mode).
**API.** `predict(profile=...)`, `run(profile=...)`, CLI `--profile` (hidden) and env DC_FAST_PROFILE are accepted
and ignored with ONE FutureWarning ("... is ignored ... remove the argument"); answers are those of a plain call.
Chosen over raising so old scripts keep running.

**Parity** (note-115 set: 40 signals x 30 min / 3 h / 24 h + bench71 4 x 3 = 132 cases; locked_v2 asserted absent),
current v7 at profile full vs the new package (no argument), `s120/cmp_v7cur_vs_v7one.json`: 2,627 detectors,
15,705 candidate pairs, 688 phase rows; every cell of every column identical (0 differing cells, floats compared
with ==). Edge cases: identical with and without profile='le2h' (`s120/edge_*.pkl`). 7-day n08 (1 thread):
identical to the old package.

**check** (from src and in the fresh venv): 6/6 PASS, no --freeze needed (max drift 0.0). Checks: stored answers,
phase renumbering, channel order, joint channel + phase permutation, blocked imports (torch / lightgbm / scipy /
sklearn / pyarrow), NEW 5: old profile argument and env ignored (identical answers, one FutureWarning; a plain call
does not give it). Note: in the dev venv (pandas 2.3.3 + numpy 2.5.3) pandas itself emits a numpy DeprecationWarning
in pd.Timedelta(seconds=); not seen with pandas 3.0.6.

**Wheel** rebuilt; `tests/smoke_install.py` (fresh venv: pandas 3.0.6, numpy 2.5.3, duckdb 1.5.6, ort 1.30.0): wheel =
src, CLI runs, deprecated --profile gives a FutureWarning and a byte-identical CSV, predict prints no warning,
check 6/6. SMOKE OK.

**Size.** Wheel 6.02 -> 5.50 MB; files 76 -> 70; uncompressed 9.96 -> 8.98 MB; weights 9.46 -> 8.50 MB.
Models: 24 -> 21 trained models (ONNX graphs 26 -> 23; + 1 HGB json); every one now on the one path.

**One thread (edge device)** (`s120/edge1t.jsonl`; DuckDB / ORT / OMP = 1 thread, fresh process, peak working set
incl. Python + models; machine shared with other agents, times rough):
| sample | signal | cold s | warm s | peak MB |
|---|---|---|---|---|
| 30 min | typical_r8 / busiest_ch | 1.0 / 1.3 | 0.7 / 0.9 | 212 / 219 |
| 3 h | typical_r8 / busiest_ch | 1.6 / 2.1 | 1.3 / 1.7 | 218 / 229 |
| 24 h | typical_r8 / busiest_ch | 2.2 / 3.3 | 1.8 / 2.9 | 307 / 350 |
| 7 days | n08 (40 ch, 2.3 M events) | 15.3 | 14.7 | 851 (first call 826) |
7 days with predict(memory='512MB'): 15.3 s, peak 740 MB. Import alone: 0.6 s, 99 MB. Fits a ~1 GB device at every
length; for 7 days pass memory='512MB' (or split the week) to keep headroom.
