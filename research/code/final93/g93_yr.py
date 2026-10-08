"""Note 93 B: group test -- Yellow_Red zones on a phase with NO Count zone ("YR-no-Count") taken OUT of training.

Wraps note 89's g89_groups.py (same recipe: 229-feature trees arm c seed 0 -> x69_siba context stacker 3 seeds -> gate .9
lane decode; scoring truth FIXED = v4l truth_v3s) on the final training labels v4o (G89_SRC=v4o):
  base    v4o training labels as they are (must reproduce note 89b's not_checkable arm = v4o)
  yr_nc   the same with every YR-no-Count detector's training rows removed (train_use_train False)
Membership = (DeviceId, detector) with label Yellow_Red on a phase that has no Count, in the training label
(label_print_first_train) OR in the scoring truth (truth_v3s).  -> %DC_WORK%/s93/yr_nc_members.parquet

    set G89_SRC=v4o & set F76_ARM=c
    python g93_yr.py members
    python g93_yr.py fit --arm base|yr_nc --seeds 0 ; python g93_yr.py stack --arm ... ; python g93_yr.py score --arm ...
CPU 4 threads.  locked_v2 asserted absent (g89).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("G89_SRC", "v4o")
os.environ.setdefault("F76_ARM", "c")
os.environ.setdefault("OMP_NUM_THREADS", "4")
CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "evaluation"))
import argparse  # noqa: E402

import pandas as pd  # noqa: E402

import g89_groups as G  # noqa: E402

assert G.SRC == "v4o"
MEM = G.DCW / "s93" / "yr_nc_members.parquet"
_orig_build = G.build_tables


def members() -> pd.DataFrame:
    L = pd.read_parquet(G.V4L)

    def nc(col):
        yr = L[L[col].eq("Yellow_Red").fillna(False) & L.phase_target.notna()]
        c = set(zip(L.DeviceId[L[col].eq("Count").fillna(False)], L.phase_target[L[col].eq("Count").fillna(False)]))
        return yr[[(a, b) not in c for a, b in zip(yr.DeviceId, yr.phase_target)]]
    a, b = nc("label_print_first_train"), nc("truth_v3s")
    m = pd.concat([a.assign(in_train_set=True), b.assign(in_truth_set=True)])
    m = m.groupby(["DeviceId", "detector"], as_index=False).agg(
        DeviceName=("DeviceName", "first"), phase_target=("phase_target", "first"), technology=("technology", "first"),
        truth_v3s=("truth_v3s", "first"), train_label=("label_print_first_train", "first"),
        in_train_set=("in_train_set", "any"), in_truth_set=("in_truth_set", "any"))
    m["in_train_set"] = m.in_train_set.eq(True)
    m["in_truth_set"] = m.in_truth_set.eq(True)
    assert not m.DeviceId.str.lower().isin(G._locked()).any()
    return m


def build_tables(arm: str):
    if arm != "yr_nc":
        return _orig_build(arm)
    lp, dp, info = _orig_build("base")
    L = pd.read_parquet(lp)
    M = pd.read_parquet(MEM)
    k = set(zip(M.DeviceId.str.lower(), M.detector.astype(int)))
    m = pd.Series([(a, int(b)) in k for a, b in zip(L.DeviceId, L.detector)], index=L.index)
    L.loc[m, ["train_use", "train_use_train"]] = False
    L.loc[m, "g89_restored"] = L.loc[m, "g89_restored"].fillna("") + "yr_nc_out;"
    info["yr_nc_removed_label_rows"] = int(m.sum())
    lp2, dp2 = lp.with_name(lp.name.replace("_base", "_yr_nc")), dp.with_name(dp.name.replace("_base", "_yr_nc"))
    L.to_parquet(lp2, index=False)
    pd.read_parquet(dp).to_parquet(dp2, index=False)
    return lp2, dp2, info


G.build_tables = build_tables
_orig_groups = G.groups_of
G.groups_of = lambda arm: [] if arm == "yr_nc" else _orig_groups(arm)


if __name__ == "__main__":
    if sys.argv[1] == "members":
        MEM.parent.mkdir(parents=True, exist_ok=True)
        m = members()
        m.to_parquet(MEM, index=False)
        print(len(m), m.in_train_set.sum(), m.in_truth_set.sum(), m.groupby(["DeviceName", "phase_target"]).ngroups)
        sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fit", "stack", "score"])
    ap.add_argument("--arm", default="base")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--info-only", action="store_true", dest="info_only")
    a = ap.parse_args()
    {"fit": G.stage_fit, "stack": G.stage_stack, "score": G.stage_score}[a.stage](a)
