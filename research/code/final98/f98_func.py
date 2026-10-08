"""Note 98: function arms of the simplified (fast) variants, six-fold OOF from saved predictions only (CPU).

Recipe = note 90 (oof90 stack / tcnphase): v4o trees OOF (229 feat, 3 seeds) + siba x86_siba4l OOF (filtered, tree p
>= .01) -> 47-column context stacker (PRM0, 150 rounds, stacker seeds 0/1/2, six folds) -> gate .9 decode; truth v4l.
Phase input of the function frame: replaced by the top of a note-98 phase arm (s98/p87_q.parquet) where the phase pool
covers the row (62 %), as oof90 tcnphase (changed rows lose lane string / lane context; tree features and pick inputs held).
Stacker kinds:
  mean3   stacker trained + applied on the 3-seed MEAN of the siba OOF (= v4f)
  single  'single' recipe: trained on the three single-seed inputs (3 x rows), applied to ONE seed's input; reported per
          seed s0 / s1 / s2 (one shipped member = one of them)
  nonet   no function network: the net columns = the trees' probabilities (net_X with no net), stacker re-fitted
  trees   no network, no stacker: tree probabilities straight into the gate decode
  blend   no stacker: 0.6 trees + 0.4 mean3 net into the gate decode (reference only)

    python f98_func.py run --arms tcn_ad76:mean3,tcn_ad76:nonet,...   -> %DC_WORK%/final_v3_work/f98/oof/ok_<phase>_<kind>.npz
    python f98_func.py score                                          -> .../f98/oof/f98_func.json
    python f98_func.py pkgstacker --kind nonet|single                 -> .../f98/stacker_<kind>/ (all training rows)
locked_v2 asserted absent (fit83.stacker_inputs).  CPU 4 threads.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "final90"))
import oof90 as O90  # noqa: E402

F, F4 = O90.F, O90.F4
DCW = O90.DCW
OUT = DCW / "final_v3_work" / "f98" / "oof"
S98 = DCW / "s98"
NET = O90.NET


def _setup():
    O90.use_v4o()
    s74, S, E, Xs, y, trm, names = F.stacker_inputs()
    return s74, S, E, y, trm, names


def _phase_tops(fr, col):
    """frame-aligned replacement phase (NaN where the phase pool does not cover the row)."""
    q = pd.read_parquet(S98 / "p87_q.parquet", columns=["DeviceId", "Detector", "win", "cand_phase", col])
    k = ["DeviceId", "Detector", "win"]
    t = q.sort_values(k + [col], ascending=[True, True, True, False]).groupby(k, sort=False).first().cand_phase
    t = t.rename("top").reset_index()
    t["period"] = np.where(t.DeviceId.str.endswith("@stg"), "stg", "dec")
    t["DeviceId"] = t.DeviceId.str.replace("@stg", "", regex=False).str.lower()
    key = fr[["DeviceId", "period", "Detector", "win"]].copy()
    key["DeviceId"] = key.DeviceId.str.lower()
    m = key.merge(t.astype({"Detector": fr.Detector.dtype}), on=["DeviceId", "period", "Detector", "win"], how="left")
    assert len(m) == len(fr)
    return m.top.to_numpy(float)


def _fit_apply(X_tr_list, y, trm, fo, X_apply):
    """six-fold OOF, 3 stacker seeds; training rows = the union of X_tr_list (each restricted to trm & fold != k)."""
    import lightgbm as lgb
    for seed in (0, 1, 2):
        for k in range(6):
            tr, te = trm & (fo != k), fo == k
            Xt = np.vstack([X[tr] for X in X_tr_list])
            yt = np.concatenate([y[tr]] * len(X_tr_list))
            m = lgb.train(dict(F.PRM0, num_threads=F.THREADS, seed=seed), lgb.Dataset(Xt, yt), num_boost_round=150)
            for nm, (Xa, Ra) in X_apply.items():
                Ra[te] += m.predict(Xa[te]) / 3
    return {nm: Ra for nm, (Xa, Ra) in X_apply.items()}


def stage_run(a):
    s74, S, E, y, trm, names = _setup()
    fr = E["fr"]
    fo = fr.fold.to_numpy()
    import of77
    lc0 = of77.lane_ctx(fr, of77.O8 / "lanes_D.func.parquet")[:, 2]
    Pt = E["Pt"]
    P3 = s74.net_probs(fr, NET)
    pp0, lanes0, LN0 = fr.pred_phase.copy(), fr.lanes5g.copy(), S._CTX["ln"].copy()
    OUT.mkdir(parents=True, exist_ok=True)
    info = json.load(open(OUT / "run_info.json")) if (OUT / "run_info.json").exists() else {}
    for arm in a.arms.split(","):
        ph, kind = arm.split(":")
        fo_ = OUT / f"ok_{ph}_{kind}.npz"
        if fo_.exists() and not a.force:
            F.log(f"{arm} cached")
            continue
        t0 = time.time()
        if ph == "frame":
            ch = np.zeros(len(fr), bool)
        else:
            top = _phase_tops(fr, f"p2_{ph}")
            new = np.where(np.isfinite(top), top, pp0.to_numpy(float))
            ch = pp0.notna().to_numpy() & (new != pp0.to_numpy(float))
            fr["pred_phase"] = new
            fr["lanes5g"] = lanes0.where(~ch, None)
            S._CTX["ln"] = np.where(ch[:, None], np.nan, LN0)
        lc = np.where(ch, np.nan, lc0)
        out = {}
        if kind == "mean3":
            X = F4.net_X(s74, S, E, P3)
            R = _fit_apply([X], y, trm, fo, {"mean3": (X, np.zeros((len(y), 7)))})
            out["mean3"] = R["mean3"]
        elif kind == "single":
            Xs = [F4.net_X(s74, S, E, s74.net_probs(fr, f"{NET}:{s}")) for s in (0, 1, 2)]
            R = _fit_apply(Xs, y, trm, fo, {f"single_s{s}": (Xs[s], np.zeros((len(y), 7))) for s in (0, 1, 2)})
            out.update(R)
        elif kind == "nonet":
            X = F4.net_X(s74, S, E, np.full_like(Pt, np.nan))
            R = _fit_apply([X], y, trm, fo, {"nonet": (X, np.zeros((len(y), 7)))})
            out["nonet"] = R["nonet"]
        elif kind == "trees":
            out["trees"] = Pt.astype(float)
        elif kind == "blend":
            Pn = np.where(np.isnan(P3[:, :1]), Pt, P3)
            import cand64 as C
            out["blend"] = C.W_TREE * Pt + (1 - C.W_TREE) * Pn
        res = {}
        for nm, P in out.items():
            ok = s74.gate_ok(E, P, lc)
            res[f"{nm}_E"], res[f"{nm}_R"] = ok["E"], ok["R"]
            if kind in ("mean3", "single", "nonet"):
                np.save(OUT / f"P_{ph}_{nm}.npy", P.astype(np.float32))
        np.savez_compressed(fo_, **res)
        info[arm] = {"changed_rows": int(ch.sum()), "s": round(time.time() - t0)}
        json.dump(info, open(OUT / "run_info.json", "w"), indent=1)
        F.log(f"{arm}: phase input changed on {int(ch.sum()):,} rows, {time.time() - t0:.0f}s")
        fr["pred_phase"], fr["lanes5g"], S._CTX["ln"] = pp0, lanes0, LN0


def stage_score(a):
    import cand64 as C
    s74, S, E, y, trm, names = _setup()
    fr = E["fr"]
    sig = fr.DeviceId.to_numpy()
    ok = {}
    for f in sorted(OUT.glob("ok_*.npz")):
        z = np.load(f)
        ph = f.stem[3:].rsplit("_", 1)[0]
        for k in z.files:
            nm, st = k.rsplit("_", 1)
            ok.setdefault(f"{ph}|{nm}", {})[st] = z[k]
    ref = "tcn_ad76|mean3"
    assert ref in ok, list(ok)
    singles = [k for k in ok if "|single_s" in k]
    for ph in {k.split("|")[0] for k in singles}:                 # mean over the three single-seed arms
        ks = [k for k in singles if k.startswith(ph + "|")]
        ok[f"{ph}|single_avg"] = {st: np.mean([ok[k][st] for k in ks], axis=0) for st in ("E", "R")}
    # length-switched composites = what a fast profile does: networks on samples <= 2 h (m5 / m10 / m30 / h1), not above
    short = fr.wgroup.isin(["m5", "m10", "m30", "h1"]).to_numpy()
    comps = {"le2h_both|mean3>nonet": ("tcn_ad76|mean3", "trees_dec|nonet"),
             "le2h_phase|mean3": ("tcn_ad76|mean3", "trees_dec|mean3"),
             "le2h_both|single>nonet": ("tcn_ad76|single_avg", "trees_dec|nonet"),
             "le2h_func|mean3>nonet": ("tcn_ad76|mean3", "tcn_ad76|nonet")}
    for nm, (s_, l_) in comps.items():
        if s_ in ok and l_ in ok:
            ok[nm] = {st: np.where(short, ok[s_][st], ok[l_][st]) for st in ("E", "R")}
    tr = fr.truth_v3s.to_numpy(object)
    cl = np.where(np.isin(tr, S.ATS), tr, "nonATSPM")
    res = {}
    for st in ("E", "R"):
        for pool, fams in C.POOLS.items():
            pm = fr.wgroup.isin(fams).to_numpy()
            sc = pm & np.logical_and.reduce([~np.isnan(v[st]) for v in ok.values()])
            r = {"n": int(sc.sum()), "signals": int(len(set(sig[sc])))}
            for k, v in ok.items():
                r[k] = {"acc": C.acc_ci(v[st][sc], sig[sc])}
                if k != ref:
                    r[k]["d_vs_full"] = C.delta_ci(ok[ref][st][sc], v[st][sc], sig[sc])
            res[f"{st}_{pool}"] = r
        m = fr.wgroup.isin(C.GE30).to_numpy() & np.logical_and.reduce([~np.isnan(v[st]) for v in ok.values()])
        res[f"{st}_ge30_by_class"] = {k: {c: round(100 * float(np.mean(v[st][m & (cl == c)]) -
                                                                 np.mean(ok[ref][st][m & (cl == c)])), 2)
                                          for c in ["Advance", "Presence", "Count", "Yellow_Red", "nonATSPM"]}
                                      for k, v in ok.items() if k != ref}
    res["run_info"] = json.load(open(OUT / "run_info.json"))
    json.dump(res, open(OUT / "f98_func.json", "w"), indent=1, default=str)
    for st in ("E", "R"):
        for pool in C.POOLS:
            r = res[f"{st}_{pool}"]
            F.log(f"{st} {pool} n {r['n']}")
            for k in ok:
                F.log(f"   {k:28s} {r[k]['acc']} d {r[k].get('d_vs_full')}")


def stage_pkgstacker(a):
    """package stacker for a kind on ALL training rows (note-90 pkgstacker recipe, v4o trees OOF, no phase replacement)."""
    import lightgbm as lgb
    s74, S, E, y, trm, names = _setup()
    fr, Pt = E["fr"], E["Pt"]
    d = DCW / "final_v3_work" / "f98" / f"stacker_{a.kind}"
    d.mkdir(parents=True, exist_ok=True)
    if a.kind == "nonet":
        X = F4.net_X(s74, S, E, np.full_like(Pt, np.nan))
        Xt, yt = X[trm], y[trm]
        desc = ("nonet (note 98): no function network; the net columns carry the trees' probabilities (net_X with no "
                "net), trained on the v4o trees OOF, all training rows")
    else:
        Xs = [F4.net_X(s74, S, E, s74.net_probs(fr, f"{NET}:{s}")) for s in (0, 1, 2)]
        X = Xs[0]
        Xt, yt = np.vstack([x[trm] for x in Xs]), np.concatenate([y[trm]] * 3)
        desc = ("single (note 98): trained on the three single-seed filtered x86_siba4l OOF inputs (3 x rows), v4o "
                "trees OOF; applied to ONE full-data siba member")
    np.save(d / f"X_check_{a.kind}.npy", X[:2000].astype(np.float32))
    files = []
    for s in (0, 1, 2):
        f = d / f"stacker_{a.kind}_s{s}.txt"
        lgb.train(dict(F.PRM0, num_threads=F.THREADS, seed=s), lgb.Dataset(Xt, yt), num_boost_round=150).save_model(str(f))
        files.append(f.name)
        F.log(f"  {a.kind} seed {s} done")
    json.dump({"classes": E["C7"], "n_features": len(names), "feature_names": names, "files": files,
               "params": F.PRM0, "num_boost_round": 150, "variant": desc,
               "labels": "truth = v4l (= v4o plain columns); trees = v4o OOF (f76/function_c_v4o)",
               "trained_on": {"rows": int(trm.sum()), "signals": int(fr.DeviceId[trm].nunique())}},
              open(d / f"stacker_{a.kind}.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "score", "pkgstacker"])
    ap.add_argument("--arms", default="tcn_ad76:mean3")
    ap.add_argument("--kind", default="nonet")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    globals()[f"stage_{a.stage}"](a)
    F.log(f"== {a.stage} done ({time.time() - t0:.0f}s)")
