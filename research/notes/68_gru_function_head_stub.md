# 68 — GRU joint function head "gj" (GRU-vs-TCN comparison for function): STOPPED by the orchestrator (2026-10-01)
* Built, not run: `neural/tcn53_func.py --arch gru` = fj recipe with only the pair backbone swapped for the p3 GRU (GRUAttnD,
  768-d -> proj 128); infer now also writes the phase head (`tcn53/ppreds/<tag>.parquet`); `runs/` mkdir fixed (note 65 bug);
  run.sh takes ARCH as 5th arg. CPU + local-GPU smoke OK. Finding: under bf16 autocast cuDNN's GRU takes a slow path (~14x
  slower, A1000 bench; 42 s GPU per job-epoch on an RTX PRO 6000) -> GRU backbone now runs in fp32 (~= TCN speed, 3x memory).
  The p3 GRU (train2) trains under the same bf16 autocast, so its ~90-min folds were likely this slow path too.
* Cloud: RunPod network volume **rvwhubuncy** ("dc-cloud-pkg", 20 GB, US-NE-1, secure cloud only; kept) holds cloud_pkg
  (code updated), pylib (pandas / pyarrow / duckdb for image runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04),
  setup68.sh, worker68.sh. Upload this time: 2.1 GB in ~1.5 min (sshd MaxStartups 10: <= 6 parallel scp).
* Ran ~10 min on 2x RTX PRO 6000 ($2.09/h) + 1 MIG upload pod, then stopped (premature: function net not optimised yet).
  No finished folds, no scores. **Spent $0.50** (balance $18.225 -> $17.725). All pods terminated; API lists none.
