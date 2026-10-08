# 71 — Peak memory and time per signal: final_v2 vs final_v3_candidate_v2 (2026-10-01)
Question (user doubt): the beta runs fine in a 4 GB web app on the latest 3 h; are note 48's RAM figures (24 h peak 1.92 GB
candidate / 1.57 final_v2) real? Code `research/code/evaluation/bench71.py`, rows `%DC_WORK%/bench71/rows.json`. CPU, 4 threads,
numpy backend, ONE signal per predict() call, each (package, signal, length) in a fresh process; peak = process peak working
set (psutil). Signals: note-32 bench pool (non-locked, asserted): most channels (43), most events (31 ch), 4 typical (29/22/19/12
ch). Windows = the pool's fixed m30 / h3 / h6 / h24 extracts. Machine shared (5 other busy python processes): times are inflated.
## Why note 48 looked big
Note 48 timed ONE call over 20 signals; DuckDB holds all 20 signals' events, so its peak is per batch, not per signal. Per signal
the peak is set by the GRU ONNX session (pair batch 512, arena ~1 GB on busy signals). final_v2 runs the GRU only <= 120 min,
so at 3 h it is trees only — that is why the beta is small in the user's app.
## Import / load (both packages alike)
After `import predict`: 95 MB. Bundled sample (model files load inside predict(), tiny data): final_v2 0.39 GB, cand 0.37 GB.
## Peak working set per signal, GB (range over the 6 signals; busiest = 43 channels)
| package | 30 min | 3 h | 6 h | 24 h |
|---|---|---|---|---|
| final_v2 | 0.40-1.39 (busiest 1.39) | 0.15-0.18 | 0.15-0.18 | 0.18-0.28 |
| cand_v2 (GRU pair batch 512) | 0.38-1.37 (1.37) | 0.44-1.45 (1.45) | 0.44-1.45 (1.45) | 0.45-1.47 (1.47) |
| cand_v2, pair batch 128 (3 h; busiest / 22 ch) | | 0.69 / 0.67 | | |
| cand_v2, pair batch 64 (3 h; busiest / 22 ch) | | 0.43 / 0.43 | | |
A second call on the same signal in the same process raised the peak by 0.01-0.26 GB.
## Wall time per signal, s (first call, includes model load; shared machine)
| package | 30 min | 3 h | 6 h | 24 h |
|---|---|---|---|---|
| final_v2 | 3.1-5.0 | 1.9-2.2 | 2.0-2.4 | 2.1-3.1 |
| cand_v2 | 3.4-6.0 | 6.2-16.8 | 5.3-19.6 | 6.4-18.4 |
pair batch 64 at 3 h: busiest 17.2 s, 22 ch 9.8 s (vs 13.1 / 11.1 at 512: within the contention noise). Output identity at a
smaller pair batch NOT checked (batching only; expect float-level equality - verify before shipping). ~2 s of every call is
fixed overhead (DuckDB setup, text boosters); note 48's 0.1-2.5 s/signal amortised it over 20 signals.
## Not in the package: fj and D lanes (estimates)
* fj TCN function head (730 k params, `tcn53/models/fj_f0.pt`), random input of the real shape (pairs x 15 ch x 1800 1-s bins
  per 30-min piece; 1 piece at 30 min, 4 at >= 3 h). ONNX export (opset 17) + onnxruntime CPU 4 threads, batch 64: 0.23 s per
  piece at 120 pairs, 0.67 s at 344 pairs (busiest) -> +0.9 / +2.7 s per signal at >= 3 h; process peak 0.24 GB (session 0.04 GB
  + activations). torch on CPU: +0.47 GB just for `import torch`, 0.77 / 2.22 s per piece -> ship ONNX, not torch. Raster for
  the 15 channels not timed (same order as the GRU raster). Parity ONNX vs torch not checked.
* D lanes: same pair-booster shape as the shipped lane model (3 x 0.5 MB) with the function features added (already computed);
  shipped lanes cost 0.09-0.11 s/signal (note 48) -> expect ~0.1-0.2 s, < 50 MB. Not measured (D fold models not packaged).
## Conclusion
The current champion fits a 4 GB app on 3 h of data: per signal ~1.5 GB peak at worst (busiest, default GRU pair batch), ~0.45 GB
with GRU pair batch 64, plus ~0.2 GB and 1-3 s for fj + D. Conditions: score one signal (or a few) per call, ONNX not torch.
The 3-h jump vs the beta (0.18 -> 1.45 GB, 2 -> 6-17 s) is the GRU now running above 2 h, not data size.
