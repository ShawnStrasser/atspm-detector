"""Note 53: detached, resumable GPU chain -- every variant fixed before launch (fold 0, seed 0).

Run from the LOCAL snapshot (`%DC_WORK%/tcn53/snap/research/code/neural/`), never from the share
(DataLoader workers spawned from the share stalled, note 37).  Each step is skipped when its
output exists; an interrupted training resumes from its `.last.pt`.

    python tcn53_chain.py [--only a,b]        log: %DC_WORK%/tcn53/logs/chain.log
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
sys.path.insert(0, str(HERE.parent))
import rpath  # noqa: F401,E402
from common import DC_WORK  # noqa: E402

ROOT = DC_WORK / "tcn53"
B9 = ["occ", "onrate", "g", "y", "rc", "call", "og", "oc", "coord"]


def ch(*extra, drop=None):
    c = [x for x in B9 if x != drop] + list(extra)
    return ["--chans", ",".join(c)]


VARIANTS = {
    "base": [],
    "r05": ["--bw", "500"],
    "ad_all": ch("onE", "offE", "cOn", "cOff", "pg", "pc"),
    "fx0": ["--windows", "fixed"],
    "fxA": ["--windows", "fixed", "--feats", "all"],
    "fxK": ["--windows", "fixed", "--feats", "topk"],
    "base_s1": ["--seed", "1"],
    "ad_edge": ch("onE", "offE"),
    "ad_cdrop": ch("cOn", "cOff"),
    "ad_partner": ch("pg", "pc"),
    **{f"ab_{c}": ch(drop=c) for c in B9},
    # promotion round (added after the first screen: r05 and ad_all passed on the net alone)
    "ad_all_s1": ch("onE", "offE", "cOn", "cOff", "pg", "pc") + ["--seed", "1"],
    "r05_s1": ["--bw", "500", "--seed", "1"],
    "r05_all": ["--bw", "500"] + ch("onE", "offE", "cOn", "cOff", "pg", "pc"),
    "r05_all_s1": ["--bw", "500", "--seed", "1"] + ch("onE", "offE", "cOn", "cOff", "pg", "pc"),
    "r02_all": ["--bw", "200", "--chunk", "384"] + ch("onE", "offE", "cOn", "cOff", "pg", "pc"),
    # ad_all hit the 50-epoch cap still improving: re-fit with a 100-epoch cap (same patience)
    "ad_all_e100": ch("onE", "offE", "cOn", "cOff", "pg", "pc"),
    "ad_all_e100_s1": ch("onE", "offE", "cOn", "cOff", "pg", "pc") + ["--seed", "1"],
}


def log(m: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)


def step(name: str, args: list[str], done: Path) -> bool:
    if done.exists():
        log(f"skip {name}")
        return True
    log(f"start {name}: {' '.join(args)}")
    t0 = time.time()
    with open(ROOT / "logs" / f"{name}.log", "a") as f:
        r = subprocess.run([PY, str(HERE / "tcn53.py")] + args, stdout=f,
                           stderr=subprocess.STDOUT, cwd=str(HERE))
    ok = r.returncode == 0 and done.exists()
    log(f"end {name}: rc={r.returncode} {(time.time()-t0)/60:.1f} min ok={ok}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--folds", default="0", help="e.g. 1,2,3,4,5 -- variants loop inside folds")
    ap.add_argument("--epochs", type=int, default=0, help="override the epoch cap (0 = 50)")
    a = ap.parse_args()
    names = a.only.split(",") if a.only else list(VARIANTS)
    for fold in [int(x) for x in a.folds.split(",")]:
        for v in names:
            extra = VARIANTS[v] + (["--epochs", str(a.epochs)] if a.epochs else [])
            tag = v
            fk = ["--fold", str(fold)]
            if not step(f"{v}_f{fold}_train", ["train", "--tag", tag] + fk + extra,
                        ROOT / "runs" / f"{tag}_f{fold}.done.json"):
                continue
            chunk = (["--chunk", "256"] if "200" in extra else ["--chunk", "512"] if "--bw" in extra
                     else ["--chunk", "1024"])
            step(f"{v}_f{fold}_infer", ["infer", "--tag", tag] + fk + chunk,
                 ROOT / "preds" / f"{tag}_f{fold}_bywindow.parquet")
    log("chain done")


if __name__ == "__main__":
    main()
