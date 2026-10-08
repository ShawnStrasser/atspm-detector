"""Note 65: the cand64 fj function blend scored per network seed (each seed alone, folds where that seed exists).

Re-uses cand64's function + score stages unchanged, pointing FPRED at a temp folder that holds only one seed's
fpreds (renamed fj_f{k}.parquet) and OUT at %DC_WORK%/cand65/<seed>/.  The seed-averaged run is `cand64.py all`.

    python cand65_seeds.py            -> %DC_WORK%/cand65/{s0,s1,s2}/headline_function.json + summary.json
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cand64 as C  # noqa: E402

SEEDS = {"s0": "fj_f{k}.parquet", "s1": "fj_s1_f{k}.parquet", "s2": "fj_s2_f{k}.parquet"}


def main():
    root, fp0 = C.DC_WORK / "cand65", C.FPRED
    summ = {}
    for sd, pat in SEEDS.items():
        fp = C.DC_WORK / "tmp" / "cand65" / sd
        shutil.rmtree(fp, ignore_errors=True)
        fp.mkdir(parents=True)
        n = 0
        for k in range(C.N_FOLDS):
            src = fp0 / pat.format(k=k)
            if src.exists():
                shutil.copy(src, fp / f"fj_f{k}.parquet"); n += 1
        if not n:
            continue
        C.FPRED, C.OUT = fp, root / sd
        C.OUT.mkdir(parents=True, exist_ok=True)
        C.stage_function(None)
        res = {}
        C.score_function(res)
        json.dump(res, open(C.OUT / "headline_function.json", "w"), indent=1, default=str)
        f = res["function"]
        summ[sd] = {"folds": sorted(h["fold"] for h in f["fj_folds"]),
                    **{s: {p: f["fold_subset_with_fj"][s][p]["delta_pt"] for p in ("ge30", "m5", "m10", "all")}
                       for s in ("everything", "realistic")}}
        C.log(f"{sd}: {summ[sd]}")
    json.dump(summ, open(root / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
