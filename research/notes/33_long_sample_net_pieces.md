# 33 — GRU on K pieces of a long sample (note 32's cost problem), six folds OOF (2026-09-23)

Code `research/code/neural/{pieces_infer,pieces_eval,pieces_bench}.py`; per-piece OOF `dc_work/trackB/pieces/`,
results `dc_work/trackB/eval/{pieces.json,pieces_bench.jsonl}`, logs `dc_work/logs/pieces_*.log`. Locked signals
unread. Stage-13 GRU fold models `gru2_f0..5` rerun on GPU over the long OOF windows, keeping every 30-min
piece's log-probs; K pieces chosen as production's `split_range(max_chunks=K)` does (linspace), pooled by mean
log-prob (production). The all-pieces rerun matches the stored `gru2_oof_f*` to 4e-07 (argmax identical), and
the harness reproduces `final6_gru13.json` exactly (same 1,532,182 rows, 0.5 blend before decoder).
**Decoder frozen** = fitted once on the full-length inputs, then applied to each K (what production would do).
OOF windows: h3 (6 pieces), h6 (12), h24, full 66/72 h (both capped at 32 pieces in the reference); no 2/4/12 h.

## Blend accuracy, six folds pooled (errors), frozen decoder
| | trees | K=1 | K=2 | K=3 | K=4 | K=6 | K=8 | all pieces |
|---|---|---|---|---|---|---|---|---|
| 3 h (24,671) | .97620 (587) | .97788 | .97921 (513) | .97912 | **.97969 (501)** | = all | = all | .97974 (500) |
| 6 h (24,948) | .97704 (573) | .97849 | .97977 (505) | .98005 | **.98017 (495)** | .98021 | .98013 | .98013 (496) |
| 24 h (25,361) | .97965 (516) | .97879 | .97942 | .98072 | .98111 (479) | .98111 | .98127 | .98167 (465) |
| full (12,712) | .98080 (253) | .98127 | .98227 | .98291 | .98257 (232) | .98219 | .98261 | .98280 (227) |
K=4 vs all pieces: −0.005 / +0.004 / −0.06 / −0.02 pt; K=2: −0.05 / −0.04 / −0.23 / −0.05 pt. Re-fitting the
decoder on the K-piece inputs (not what ships) gives K=4 .97989 / .98081 / .98119 / .98318 — same picture.
Per fold, K=4 vs all pieces at 6 h (errors): 71/78, 85/87, 58/53, 87/88, 90/86, 104/104 — noise both ways.
Mean-probability pooling: no better at 3–6 h (K=4 .97973 / .97985). Cost ~0.49 s per piece, flat in length.
## CPU cost, final_v3 candidate end to end, 20 non-locked signals, 4 threads (s/signal, network part)
| | K=2 | K=4 | K=6 | all pieces |
|---|---|---|---|---|
| 3 h | 1.11 (0.99) | 2.10 (1.98) | 3.09 (2.97) | 3.03 (2.91) |
| 6 h | 1.14 (1.00) | 2.11 (1.97) | 3.09 (2.96) | 6.06 (5.92) |
## Is the blend ever worse than trees alone, per fold (all pieces, i.e. final_v3 as built)?
3 h: never (6/6 folds better, −87 errors pooled). 6 h: fold 0 only, 78 vs 76 errors (−0.06 pt); 5/6 better,
−77 pooled. 24 h: fold 0 −0.20 pt (72 vs 65); full: fold 0 −0.12 pt (38 vs 36). Fold 0 is the one weak fold
past 3 h, by 2–7 errors, and the 30-signal in-sample hint in note 32 (3 and 1 detectors) is the same size.
With K=4 fold 0 at 6 h is 71 vs 76 (better). No length shows a systematic loss.
## Verdict
**`max_chunks=4` above 120 min**: accuracy within 0.01 pt of full-length at 3 h and 6 h (±1 error pooled),
still +0.35 / +0.31 pt over trees; cost 2.1 s/signal (was 3.0 / 6.1) — under the ~5 s bar, no 180-min cut-off
needed. K=2 (1.1 s) sits right at the 0.05-pt edge at 3–6 h and loses 0.23 pt at 24 h: not recommended. Option
for the orchestrator: with K=4 the network could also run above 6 h at the same 2.1 s (24 h +0.15, full +0.18 pt
over trees, OOF) — the cut-off then stops being a cost question. Single GRU seed per fold; the ≤ 0.06-pt
differences between K=4 and all pieces are inside the blend's seed sd (0.02–0.09 pt at these lengths, note 13).
