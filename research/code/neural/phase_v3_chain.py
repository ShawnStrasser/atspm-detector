"""Note 37: detached, resumable GPU chain for the phase retrain with the released signals.

    GRU folds 0-5 (train, then K=4 OOF inference), final GRU on all 780 signals, ONNX export.
Each step is skipped when its output exists, so a restart resumes where it stopped.

    python phase_v3_chain.py            (log: dc_work/logs/phase_v3_chain.log)
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
sys.path.insert(0, str(HERE.parent))
import rpath  # noqa: F401,E402
from common import DC_WORK  # noqa: E402

W = DC_WORK / "final_v3_work" / "phase_v3"
LOGS = DC_WORK / "logs"


def log(m: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)


def step(name: str, args: list[str], done: Path) -> None:
    if done.exists():
        log(f"skip {name} ({done.name} exists)")
        return
    log(f"start {name}")
    t0 = time.time()
    with open(LOGS / f"phase_v3_{name}.log", "a") as f:
        r = subprocess.run([PY, str(HERE / "phase_v3_net.py")] + args, stdout=f,
                           stderr=subprocess.STDOUT, cwd=str(HERE))
    log(f"end {name}: rc={r.returncode} {(time.time()-t0)/60:.1f} min")
    if r.returncode != 0 or not done.exists():
        raise SystemExit(f"{name} failed (rc={r.returncode})")


def main() -> None:
    for k in range(6):
        step(f"train_f{k}", ["train", "--fold", str(k)], W / "work" / "gru_runs" / f"p3_f{k}.done.json")
        step(f"infer_f{k}", ["infer", "--fold", str(k)],
             W / "work" / "gru_preds" / f"p3_oof_f{k}_bywindow.parquet")
    step("train_final", ["train", "--final"], W / "work" / "gru_runs" / "p3_final.done.json")
    step("export", ["export"], W / "gru_onnx_check.json")
    log("chain done")


if __name__ == "__main__":
    main()
