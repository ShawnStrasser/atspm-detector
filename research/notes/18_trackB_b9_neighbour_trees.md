# 18_trackB_b9_neighbour_trees — neighbour-trace summary features for the ranker (B9): **dropped**

Track B step B9 of `AGENTS.md`, a hand-built version of "sibling-detector context", screened on **fold 0** on
exactly the stage-13 rows (22 windows, official labels, scorable rule; 210,038 rows, 34,552 detector-windows).
Code `research/code/trackB/b9_{features,fit,eval}.py`, artefacts `dc_work/trackB/b9/`, logs `dc_work/logs/trackB_b9_*.log`.

## What already existed, and what is new
The **ranker** sees only the detector itself (plus its own `__pdiff` = f(c) − f(partner of c)). The **joint
decoder** sees the neighbours only through their *ranker probabilities*: phi²-weighted mean / max / top-1 share of
p0 for c over the top-8 phi neighbours (2-s "active" bins), channel adjacency, and signal-level occupancy of c.
B9 gives the **ranker** the neighbours' *raw behaviour* against c — 42 columns, all phase-anonymous (candidate,
partner and detector numbers are join keys only), built in 5 s from the cached per-window tables:
`nb_<f>` phi²-weighted neighbour mean of 17 raw pair features (green lifts, exclusive-green lifts, `pex_*`,
event-43 linkage fwd / red-restricted, first-ON latency, release, queue); `nb_<f>__dself` own − neighbourhood (6);
`nb_<f>__pc` neighbourhood value for c − for c's partner, the concurrent-pair axis (8); `nb_vote_<f>` weighted
share of neighbours whose raw argmax is c (4); `nb_pool_*` exclusive-green actuations **pooled** over the detector
and its neighbours as a log-odds against the time share, with / without self (4); neighbourhood size (3).

## Protocol
The shipped pool re-built (1,587,100 rows, 261 features, 1,539,251 labelled — identical counts), ranker re-fitted
**single-seed on all six folds** (1–2 min a fold) so the decoder is re-trained on consistent inputs, then the
stage-13 decoder recipe (train folds 2–5, stop on 1, score 0). Baseline = the same refit **without** the new
columns (same seeds, same float32 pool). Harness check: fed the stored 3-seed ranker bag it reproduces the stage-13
`_fold0` entries exactly (trees .96327, blend-before .97449 at 30 min). Network = the stage-13 GRU.

## Fold 0, mean of seeds 0 and 1 (shuffle = seed 0, new columns permuted jointly within each window)
| fold 0 | 5 min | 10 min | 30 min | 1 h | 24 h | full |
|---|---|---|---|---|---|---|
| ranker alone: base / **nb** / shuffle | .8850 / **.8859** / .8843 | .9117 / **.9130** / .9095 | .9431 / **.9463** / .9436 | .9391 / **.9446** / .9381 | .9726 / **.9761** / .9742 | .9739 / **.9753** / .9732 |
| trees (ranker → decoder): base / **nb** / shuffle | .9224 / **.9236** / .9222 | .9367 / **.9377** / .9334 | .9621 / **.9629** / .9610 | .9594 / **.9611** / .9591 | .9799 / **.9802** / .9797 | .9770 / **.9774** / .9789 |
| blend before decoder (shipped): base / **nb** / shuffle | .9520 / **.9531** / .9501 | .9617 / **.9620** / .9609 | .9737 / **.9738** / .9750 | .9748 / **.9751** / .9745 | .9794 / **.9785** / .9785 | .9778 / **.9780** / .9787 |
| blend after decoder: base / **nb** | .9492 / .9503 | .9609 / .9606 | .9728 / .9727 | .9738 / .9746 | .9810 / .9803 | .9806 / .9792 |
| **nb − base**, ranker / trees / blend-before | +0.09 / +0.12 / +0.11 | +0.13 / +0.10 / +0.03 | **+0.32 / +0.08 / +0.02** | +0.55 / +0.18 / +0.03 | +0.35 / +0.03 / −0.09 | +0.14 / +0.05 / +0.02 |

Seed spread (base s0 vs s1) at 30 min: ranker 0.22 pt, trees 0.10, blend-before 0.05; stored bag .9461 / .9633 / .9745.
Concurrent-pair errors at 30 min (6,489 detector-windows over 4 windows), seeds 0 / 1: ranker 233 / 227 → **202 /
199** (−13 %); trees 138 / 130 → 119 / 126; blend-before 92 / 92 → 82 / 90 (shuffle: 79) — all but the ranker
row inside the noise.

## Verdict: **dropped** — fails the 0.3 pt bar at 30 min (+0.02 pt for the shipped blend, +0.08 pt for trees)
* The features carry **real** information: the ranker alone gains +0.3 to +0.55 pt from 30 min to a day, both seeds
  agree, a concurrent-pair error in eight disappears, and the shuffled control gives nothing (its new-column gain
  share falls from ~3.5 % to ~0.8 % and its accuracy returns to baseline). Top columns by gain: `nb_f_on_green`,
  `nb_call43_fwd_lift__pc`, `nb_call43_red_lift__pc`, `nb_vote_on_lift_green` — neighbour event-43 linkage
  against the partner is what the ranker liked.
* But the **joint decoder already recovers almost all of it** from the neighbours' ranker probabilities (+0.08 pt
  left at 30 min, one seed-sd), and the GRU blend removes the rest (+0.02 pt, below the blend's own 0.05 pt
  spread; the shuffled control scores *higher*, .9750). Pooling raw evidence across same-approach detectors is the
  decoder's job already; doing it one stage earlier is redundant. Not promoted to six folds, no further seeds.
* Side finding: a **single-seed** ranker is 0.3 pt behind the shipped 3-seed bag on fold 0 at 30 min (.9431 vs
  .9461) — larger than the 0.02 pt pool-level seed sd of note 12 — yet only 0.12 pt after the decoder and 0.08 pt
  in the blend. The same absorption: the second stage and the blend hide first-stage differences below ~0.5 pt.
  Future ranker-side ideas should be judged after the decoder from the start.

## Cost, caveats
CPU only, 6 threads, locked signals never read: features 5 s; 30 ranker fold-fits × 1–2 min; each evaluation ~30 s. About **1 h wall**, no GPU.
Caveats: the refits read the feature tables as float32 (the stored bag was float64) — the base refit, not the stored
bag, is the comparator for that reason. The neighbour graph is the existing top-8 phi graph, which needs ≥ 3 shared
2-s bins, so at 5 min many detectors have no neighbour — a lagged cross-correlogram graph (advance → stop bar) was
**not** tried: the per-window lag table does not exist and the screen does not justify building one. No bugs found.
