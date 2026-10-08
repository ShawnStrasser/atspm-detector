"""Note 80: what the v4l label pass does to the function champion (no new model family; CPU, <= 4 threads).

    F76_ARM=c python s80.py score --truth v3s|v4l --stack champ|v4l    ok arrays (E / R, gate .9, decode) -> lab80/ok_<truth>_<stack>.npz
    F76_ARM=c python s80.py fit                                         trees arm c (229 features, 3 seeds x 6 folds) on v4l
                                                                        -> f76/function_c_v4l/... (new folder, nothing reused)
    F76_ARM=c python s80.py stack                                       context stacker (siba x69_siba, 3 seeds) on the v4l trees
                                                                        and the v4l target -> f77/s74/f80v4l/
    python s80.py report                                                CIs -> lab80/s80.json

One process per (truth, stack): atspm_score / of77 cache the label table at the first load, so `atspm_score.V3S` is set
before anything is imported. Held fixed as in notes 76/77: frame, predicted phase, lanes D, pick / health inputs, twins,
siba OOF (all trained on v3s). locked_v2 is asserted absent by every loader (atspm_score.load, V.load_feats) and here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = next(p for p in Path(__file__).resolve().parents if p.name == "code")
for p in (CODE, CODE / "evaluation", CODE / "final76", CODE / "final77", CODE / "trackA"):
    sys.path.insert(0, str(p))
import rpath  # noqa: E402,F401

REPO = CODE.parents[1]
DCW = Path(os.environ.get("DC_WORK", Path.home() / "dc_work"))
LAB = {"v3s": REPO / "research/labels/function_labels_v3s.parquet",
       "v4l": REPO / "research/labels/function_labels_v4l.parquet"}
DEC = {"v4l": DCW / "cabinet_v4l" / "dec_role_changed_v4l.parquet"}
OUTD = DCW / "lab80"
F76, F77 = DCW / "final_v3_work" / "f76", DCW / "final_v3_work" / "f77"
TREES_V4L = F76 / "function_c_v4l"
STACK = {"champ": F77 / "s74" / "f77", "v4l": F77 / "s74" / "f80v4l"}
GE30 = ["m30", "h1", "h3", "h6", "h24", "full"]


def _locked() -> set[str]:
    return set(pd.read_csv(DCW / "official/locked_v2.csv").DeviceId.astype(str).str.lower())


def _set_truth(t: str):
    lab = pd.read_parquet(LAB[t], columns=["DeviceId"])
    assert not lab.DeviceId.str.lower().isin(_locked()).any(), "locked_v2 in the label table"
    import atspm_score as AS
    AS.V3S = LAB[t]


def _trees_v4l():
    import t57_function as T57F
    import f76_function as F
    import v3_retrain as V
    V.LABEL_SETS["v3s"] = (LAB["v4l"], DEC["v4l"])
    T57F.OUT = TREES_V4L
    return T57F, F, V


def stage_score(a):
    assert os.environ.get("F76_ARM") == "c"
    _set_truth(a.truth)
    if a.stack == "v4l":       # stacker inputs = the v4l trees (FUNC_DIR from T57F.OUT) - same as the stack stage
        _trees_v4l()
    import of77
    s74, S, E = of77.setup_new()
    fr = E["fr"]
    assert not fr.DeviceId.str.lower().isin(_locked()).any()
    lc = of77.lane_ctx(fr, of77.O8 / "lanes_D.func.parquet")[:, 2]
    P = np.mean([np.load(STACK[a.stack] / f"stack_ctx_s{s}.npy") for s in (0, 1, 2)], 0).astype(float)
    ok = s74.gate_ok(E, P, lc)
    seeds = {}
    for sd in (0, 1, 2):
        o = s74.gate_ok(E, np.load(STACK[a.stack] / f"stack_ctx_s{sd}.npy").astype(float), lc)
        seeds[sd] = o["E"]
    OUTD.mkdir(exist_ok=True)
    np.savez_compressed(OUTD / f"ok_{a.truth}_{a.stack}.npz", E=ok["E"], R=ok["R"], E0=seeds[0], E1=seeds[1], E2=seeds[2],
                        sig=fr.DeviceId.to_numpy(str), wgroup=fr.wgroup.to_numpy(str), period=fr.period.to_numpy(str),
                        det=fr.Detector.to_numpy(int), win=fr.win.to_numpy(str),
                        truth=fr.truth_v3s.astype("string").fillna("").to_numpy(str))
    m = fr.wgroup.isin(GE30).to_numpy()
    print(a.truth, a.stack, {s: round(float(np.nanmean(ok[s][m])), 4) for s in ("E", "R")},
          {s: int((~np.isnan(ok[s][m])).sum()) for s in ("E", "R")})


def stage_fit(a):
    assert os.environ.get("F76_ARM") == "c"
    T57F, F, V = _trees_v4l()
    T57F.setup("v6e")
    T57F.BASE_RUN = V.run_dir("exclude").name     # the v4l table's own fingerprint (stage_fit asserts it)
    TREES_V4L.mkdir(parents=True, exist_ok=True)
    T57F.stage_fit(argparse.Namespace(frame="v6e", cfg=F.CFG, seeds="0,1,2", threads=4))
    old = json.load(open(DCW / "trees57" / "function" / "v6e" / "drop_pp_xcand-pp_pdiff-pp_v2-yr-ratio-phctx" /
                         "cols.json"))["cols"]
    new = json.load(open(F.new_dir() / "cols.json"))["cols"]
    assert old == new, "feature list differs from the 229-feature arm"
    print(f"fit done -> {F.new_dir()}")


def stage_stack(a):
    assert os.environ.get("F76_ARM") == "c"
    _set_truth("v4l")
    _trees_v4l()
    import of77
    s74, S, E = of77.setup_new()
    s74.cmd_stack(argparse.Namespace(name=STACK["v4l"].name, main="x69_siba", extra=""))


def _acc_ci(ok, sig, n=2000, seed=80):
    m = ~np.isnan(ok)
    ok, sig = ok[m], sig[m]
    u, inv = np.unique(sig, return_inverse=True)
    s, c = np.bincount(inv, ok, len(u)), np.bincount(inv, minlength=len(u))
    idx = np.random.default_rng(seed).integers(0, len(u), (n, len(u)))
    r = s[idx].sum(1) / c[idx].sum(1)
    return dict(n=int(m.sum()), acc=round(float(ok.mean()), 4),
                ci=[round(float(np.quantile(r, .025)), 4), round(float(np.quantile(r, .975)), 4)])


def _delta(a, b, sig, paired, n=2000, seed=80):
    """b - a in pt, signal bootstrap. paired: same rows (both scored); else each on its own rows, same signal draw."""
    if paired:
        m = ~np.isnan(a) & ~np.isnan(b)
        a, b, sig_a, sig_b = a[m], b[m], sig[m], sig[m]
    else:
        ma, mb = ~np.isnan(a), ~np.isnan(b)
        a, b, sig_a, sig_b = a[ma], b[mb], sig[ma], sig[mb]
    u = np.unique(np.concatenate([sig_a, sig_b]))
    ia, ib = np.searchsorted(u, sig_a), np.searchsorted(u, sig_b)
    sa, ca = np.bincount(ia, a, len(u)), np.bincount(ia, minlength=len(u))
    sb, cb = np.bincount(ib, b, len(u)), np.bincount(ib, minlength=len(u))
    idx = np.random.default_rng(seed).integers(0, len(u), (n, len(u)))
    d = sb[idx].sum(1) / cb[idx].sum(1) - sa[idx].sum(1) / ca[idx].sum(1)
    return [round(100 * float(b.mean() - a.mean()), 3), round(100 * float(np.quantile(d, .025)), 3),
            round(100 * float(np.quantile(d, .975)), 3)]


def stage_report(a):
    L = {k: np.load(OUTD / f"ok_{k}.npz") for k in ("v3s_champ", "v4l_champ", "v4l_v4l", "v3s_v4l")
         if (OUTD / f"ok_{k}.npz").exists()}
    base = L["v3s_champ"]
    for k, z in L.items():
        assert (z["sig"] == base["sig"]).all() and (z["win"] == base["win"]).all() and (z["det"] == base["det"]).all()
    sig, wg = base["sig"], base["wgroup"]
    rel = set(pd.read_csv(DCW / "official/newtest_released.csv").DeviceId.str.lower())
    pools = {"ge30": np.isin(wg, GE30), "m5": wg == "m5", "m10": wg == "m10"}
    res = {}
    for k, z in L.items():
        res[k] = {f"{s}_{p}": _acc_ci(np.where(pm, z[s], np.nan), sig) for s in ("E", "R") for p, pm in pools.items()}
        res[k]["E_ge30_seeds"] = [round(float(np.nanmean(np.where(pools["ge30"], z[f"E{i}"], np.nan))), 4) for i in range(3)]
    cmp = {"rescore_champ_v3s->v4l (own rows)": ("v3s_champ", "v4l_champ", False),
           "rescore_champ_v3s->v4l (common rows)": ("v3s_champ", "v4l_champ", True),
           "retrain_under_v4l_truth (paired)": ("v4l_champ", "v4l_v4l", True),
           "retrain_under_v3s_truth (paired)": ("v3s_champ", "v3s_v4l", True),
           "total_v3s_champ->v4l_retrained (own rows)": ("v3s_champ", "v4l_v4l", False)}
    for name, (x, y, paired) in cmp.items():
        if x in L and y in L:
            res[name] = {f"{s}_{p}": _delta(np.where(pm, L[x][s], np.nan), np.where(pm, L[y][s], np.nan), sig, paired)
                         for s in ("E", "R") for p, pm in pools.items()}
    # released-signal subset and rows whose truth changed (>= 30 min, E)
    isrel = np.isin(np.char.lower(sig.astype(str)), list(rel))
    if "v4l_v4l" in L:
        g = pools["ge30"]
        res["retrain_v4l_truth_released_only_E_ge30"] = _delta(np.where(g & isrel, L["v4l_champ"]["E"], np.nan),
                                                               np.where(g & isrel, L["v4l_v4l"]["E"], np.nan), sig, True)
        res["retrain_v4l_truth_not_released_E_ge30"] = _delta(np.where(g & ~isrel, L["v4l_champ"]["E"], np.nan),
                                                              np.where(g & ~isrel, L["v4l_v4l"]["E"], np.nan), sig, True)
    tc = base["truth"] != L["v4l_champ"]["truth"]
    res["rows_truth_changed_ge30"] = int((tc & pools["ge30"]).sum())
    res["rows_scored_E_ge30"] = {k: int((~np.isnan(z["E"]) & pools["ge30"]).sum()) for k, z in L.items()}
    json.dump(res, open(OUTD / "s80.json", "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["score", "fit", "stack", "report"])
    ap.add_argument("--truth", default="v3s", choices=["v3s", "v4l"])
    ap.add_argument("--stack", default="champ", choices=["champ", "v4l"])
    a = ap.parse_args()
    {"score": stage_score, "fit": stage_fit, "stack": stage_stack, "report": stage_report}[a.stage](a)
