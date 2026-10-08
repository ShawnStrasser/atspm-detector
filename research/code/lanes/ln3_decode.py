"""Lane output (note 42), step 3 -- constrained lane decode on OOF pair probabilities, tuned and
scored out of fold against the print lane labels.

Input: ln2_pairs_oof.parquet (OOF P(same) for every pair on a predicted phase), ln1_dets (per window:
OOF predicted phase + 7-class function), ln1_truth_*.  Every signal / window of the nine Sept-2026
windows is decoded exactly as production would (lane_output.decode_groups): groups = predicted
phase, detectors >= MIN_ON actuations, Bike excluded.  Span prior P(span | predicted function) from
the TRAINING folds only.  (lam, beta) picked per held-out fold on the other five (objective: n_lanes
exact + multi-lane pairwise accuracy, all windows).

Metrics (by window group m30 / h6 / h24 / full):
  n_lanes    truth phase (DeviceId, P#) vs the decode of predicted phase # (end to end); exact, +-1;
             baselines always-1 and max(#A,#P,#C) of the predicted functions.
  pairs      truth pairs (both print-high vehicle detectors, same timing phase) with both detectors
             lane-assigned: same-lane precision / recall / accuracy; "multi" = truth phase >= 2 lanes;
             coverage = assigned share.  A pair split across predicted phases counts as "different".
  spanning   truth span > 1 vs predicted > 1 lane, detectors on >= 2-lane phases.
  lane exact detector lane set = truth after the best one-to-one lane relabelling of the phase.
  lane_conf  detector-level: all its truth pairs right, by confidence bin.

    python ln3_decode.py [--workers 4]
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: F401,E402
import argparse
import itertools
import json
import time
import numpy as np
import pandas as pd
import lane_output as LO
import ln1_cues as L1
import ln2_pairmodel as L2

OUT = L1.OUT
LAMS = [1.0, 2.0, 3.0, 4.0]   # first grid -1..2 x -0.5..0.5 picked the corner (2, 0.5): ln3_decode_grid1.json
BETAS = [0.0, 0.5, 1.0]
WGS = ["m30", "h6", "h24", "full"]
REUSE = False


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def span_prior(det, ph, D, sigs) -> dict:
    f = D[D.win == "full66"][["DeviceId", "det", "func"]]
    x = det[det.high_veh & det.phase_kept & det.DeviceId.isin(sigs)]
    x = x.merge(ph[ph.n_lanes >= 2][["DeviceId", "target"]], on=["DeviceId", "target"])
    x = x.merge(f, on=["DeviceId", "det"])
    x["k"] = x.span.clip(upper=LO.MAX_LANES)
    out = {}
    tot = x.k.value_counts()
    for fn in LO.C7:
        c = x[x.func == fn].k.value_counts()
        v = np.array([c.get(k, 0) + 2.0 * (tot.get(k, 0) + 0.5) / (len(x) + 2)
                      for k in range(1, LO.MAX_LANES + 1)], float)
        v = np.log(v / v.sum())
        out[fn] = {k: float(v[k - 1]) for k in range(1, LO.MAX_LANES + 1)}
    return out


def job(args):
    dev, Dd, Pd, cfgs = args
    probs_by_win = {w: {(int(a), int(b)): float(p) for a, b, p in zip(g.da, g.db, g.p_same)}
                    for w, g in Pd.groupby("win")}
    dets, phs = [], []
    for ci, (lam, beta, prior) in enumerate(cfgs):
        dec = LO.Decoder(prior, lam=lam, beta=beta)
        for w, g in Dd.groupby("win"):
            hours = L1.WINS[w][1] / 3600.0
            g = g.rename(columns={"det": "Detector", "pred_phase": "phase_use",
                                  "func": "func_use"})
            n_on = dict(zip(g.Detector.astype(int), g.n_on))
            pt, dt = LO.decode_groups(dev, g, n_on, hours, probs_by_win.get(w, {}), dec)
            dt["win"] = w
            dt["cfg"] = ci
            dets.append(dt)
            if len(pt):
                pt["win"] = w
                pt["cfg"] = ci
                phs.append(pt)
    return (pd.concat(dets, ignore_index=True),
            pd.concat(phs, ignore_index=True) if phs else pd.DataFrame())


def run_decode(D, P, det, ph, workers, cfgs_fn, tag):
    sig = sorted(D.DeviceId.unique())
    fold = D.groupby("DeviceId").fold.first()
    jobs = []
    for f in range(6):
        cfgs = cfgs_fn(f)
        for s in [s for s in sig if fold[s] == f]:
            jobs.append((s, D[D.DeviceId == s], P[P.DeviceId == s], cfgs))
    t0 = time.time()
    DT, PT = [], []
    from multiprocessing import Pool
    with Pool(workers) as pool:
        for i, (a, b) in enumerate(pool.imap_unordered(job, jobs, chunksize=2)):
            DT.append(a)
            PT.append(b)
            if i % 100 == 0:
                log(f"  [{tag}] {i}/{len(jobs)} ({time.time()-t0:.0f}s)")
    DT = pd.concat(DT, ignore_index=True)
    PT = pd.concat([x for x in PT if len(x)], ignore_index=True)
    DT["wg"] = DT.win.map(L2.WG)
    PT["wg"] = PT.win.map(L2.WG)
    return DT, PT


# ------------------------------------------------------------------------------ metrics
def lanes_set(s):
    return frozenset(int(x) for x in s.split(",")) if isinstance(s, str) and s else frozenset()


def eval_all(DT, PT, D, det, ph, T):
    """DT/PT for ONE config (cols win, wg).  Returns metric dict by window group + 'all'."""
    fold = D.groupby("DeviceId").fold.first()
    res = {}
    # n_lanes
    NL = []
    for w in L1.WINS:
        pt = PT[PT.win == w][["DeviceId", "phase", "n_lanes", "n_lanes_conf"]]
        x = ph.assign(ph_num=ph.target.str[1:].astype(int)).merge(
            pt, left_on=["DeviceId", "ph_num"], right_on=["DeviceId", "phase"], how="left",
            suffixes=("", "_p"))
        # lmin baseline from predicted functions of the lane-eligible detectors
        dt = DT[(DT.win == w) & (DT.lanes != "")]
        lm = dt.assign(c=dt.function.isin(["Advance", "Presence", "Count"])).groupby(
            ["DeviceId", "phase", "function"]).size().unstack(fill_value=0)
        lm = lm.reindex(columns=["Advance", "Presence", "Count"], fill_value=0).max(1).clip(
            lower=1).rename("lmin").reset_index()
        x = x.merge(lm, on=["DeviceId", "phase"], how="left")
        x["win"] = w
        NL.append(x)
    NL = pd.concat(NL, ignore_index=True)
    NL["wg"] = NL.win.map(L2.WG)
    NL["fold"] = NL.DeviceId.map(fold)
    # detector truth joined to decode
    dd = DT[["DeviceId", "Detector", "win", "wg", "phase", "lanes", "lane_conf"]].rename(
        columns={"Detector": "det"})
    # pairs
    PR = []
    for w in L1.WINS:
        d = dd[dd.win == w]
        x = T.merge(d.add_suffix("_a").rename(columns={"DeviceId_a": "DeviceId", "det_a": "da"}),
                    on=["DeviceId", "da"]).merge(
            d.add_suffix("_b").rename(columns={"DeviceId_b": "DeviceId", "det_b": "db"}),
            on=["DeviceId", "db"])
        x["win"] = w
        PR.append(x)
    PR = pd.concat(PR, ignore_index=True)
    PR["wg"] = PR.win.map(L2.WG)
    PR["fold"] = PR.DeviceId.map(fold)
    PR["covered"] = (PR.lanes_a != "") & (PR.lanes_b != "")
    sa = PR.lanes_a.map(lanes_set)
    sb = PR.lanes_b.map(lanes_set)
    PR["pred_same"] = (PR.phase_a == PR.phase_b) & np.array([len(a & b) > 0 for a, b in zip(sa, sb)])
    # detectors: spanning + lane exact + conf
    DX = []
    tv = det[det.high_veh & det.phase_kept].merge(ph, on=["DeviceId", "target"])
    for w in L1.WINS:
        x = tv.merge(dd[dd.win == w], on=["DeviceId", "det"])
        DX.append(x)
    DX = pd.concat(DX, ignore_index=True)
    DX["fold"] = DX.DeviceId.map(fold)
    DX["covered"] = DX.lanes != ""
    DX["pspan"] = DX.lanes.str.count(",") + 1
    DX["exact"] = lane_exact(DX)
    # per-detector "all pairs right"
    PRc = PR[PR.covered]
    ok = (PRc.pred_same == (PRc.same_lane == 1))
    k = pd.concat([pd.DataFrame({"DeviceId": PRc.DeviceId, "det": PRc.da, "win": PRc.win, "ok": ok}),
                   pd.DataFrame({"DeviceId": PRc.DeviceId, "det": PRc.db, "win": PRc.win, "ok": ok})])
    k = k.groupby(["DeviceId", "det", "win"]).ok.all().rename("allok").reset_index()
    DX = DX.merge(k, on=["DeviceId", "det", "win"], how="left")
    return NL, PR, DX


def lane_exact(DX) -> np.ndarray:
    """Per truth detector: lane set equal to truth after the best relabelling of its phase's lanes
    (pred phase must equal the truth phase)."""
    out = np.zeros(len(DX), bool)
    DX = DX.assign(_i=np.arange(len(DX)))
    for (dev, tgt, w), g in DX.groupby(["DeviceId", "target", "win"]):
        g = g[(g.lanes != "") & (g.phase == int(tgt[1:]))]
        if g.empty:
            continue
        ps = [lanes_set(s) for s in g.lanes]
        ts = [frozenset(range(int(a), int(e) + 1)) for a, e in zip(g.lane_index, g.end)]
        pl = sorted(set().union(*ps))
        tl = sorted(set().union(*ts)) + [-1 - i for i in range(len(pl))]
        best, bm = -1, None
        for perm in itertools.permutations(tl, len(pl)):
            mp = dict(zip(pl, perm))
            m = [frozenset(mp[x] for x in p) == t for p, t in zip(ps, ts)]
            if sum(m) > best:
                best, bm = sum(m), m
            if len(pl) > 5:
                break
        out[g._i.to_numpy()] = bm
    return out


def score(NL, PR, DX, m_nl=None, m_pr=None, m_dx=None) -> dict:
    def sub(x, m):
        return x if m is None else x[m(x)]
    NL, PR, DX = sub(NL, m_nl), sub(PR, m_pr), sub(DX, m_dx)
    r = {}
    has = NL.n_lanes_p.notna() if "n_lanes_p" in NL else NL.phase.notna()
    y = NL.n_lanes.to_numpy()
    p = NL["n_lanes_p"].to_numpy(float)
    h = has.to_numpy()
    r["n_phases"] = int(len(NL))
    r["phase_found"] = _r(h.mean())
    r["nl_exact"] = _r((p[h] == y[h]).mean())
    r["nl_pm1"] = _r((np.abs(p[h] - y[h]) <= 1).mean())
    r["nl_exact_multi"] = _r((p[h & (y >= 2)] == y[h & (y >= 2)]).mean())
    r["nl_const1"] = _r((y[h] == 1).mean())
    lm = NL.lmin.to_numpy(float)
    r["nl_lmin_pred"] = _r((lm[h] == y[h]).mean())
    r["nl_lmin_pred_pm1"] = _r((np.abs(lm[h] - y[h]) <= 1).mean())
    c = PR[PR.covered]
    r["pairs"] = int(len(PR))
    r["pair_cov"] = _r(PR.covered.mean())
    for tag, cc in (("", c), ("_multi", c[c.n_lanes >= 2])):
        t = cc.same_lane.to_numpy() == 1
        q = cc.pred_same.to_numpy()
        r[f"pair_acc{tag}"] = _r((t == q).mean())
        r[f"same_prec{tag}"] = _r(t[q].mean()) if q.any() else None
        r[f"same_rec{tag}"] = _r(q[t].mean()) if t.any() else None
        r[f"n_pairs{tag}"] = int(len(cc))
    d = DX[DX.covered & (DX.n_lanes >= 2)]
    ts_, ps_ = (d.span > 1).to_numpy(), (d.pspan > 1).to_numpy()
    r["span_true_n"] = int(ts_.sum())
    r["span_prec"] = _r(ts_[ps_].mean()) if ps_.any() else None
    r["span_rec"] = _r(ps_[ts_].mean()) if ts_.any() else None
    r["det_cov"] = _r(DX.covered.mean())
    r["lane_exact"] = _r(DX[DX.covered].exact.mean())
    r["lane_exact_multi"] = _r(DX[DX.covered & (DX.n_lanes >= 2)].exact.mean())
    return r


def _r(x, n=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def main(workers):
    P = pd.read_parquet(OUT / "ln2_pairs_oof.parquet",
                        columns=["DeviceId", "win", "da", "db", "p_same", "same_pred"])
    P = P[P.same_pred]
    D = pd.read_parquet(OUT / "ln1_dets.parquet")
    det = pd.read_parquet(OUT / "ln1_truth_det.parquet")
    ph = pd.read_parquet(OUT / "ln1_truth_phase.parquet")
    T = L2.truth_pairs(det, ph)[["DeviceId", "target", "da", "db", "same_lane", "n_lanes"]]
    fold = D.groupby("DeviceId").fold.first()
    priors = {f: span_prior(det, ph, D, set(fold.index[fold != f])) for f in range(6)}
    grid = [(lam, beta) for lam in LAMS for beta in BETAS]
    if REUSE and (OUT / "ln3_grid_phase.parquet").exists():
        DT = pd.read_parquet(OUT / "ln3_grid_det.parquet")
        PT = pd.read_parquet(OUT / "ln3_grid_phase.parquet")
    else:
        DT, PT = run_decode(D, P, det, ph, workers,
                            lambda f: [(lam, beta, priors[f]) for lam, beta in grid], "grid")
        DT.to_parquet(OUT / "ln3_grid_det.parquet", index=False)
        PT.to_parquet(OUT / "ln3_grid_phase.parquet", index=False)
    # per config, per fold objective
    ev = {}
    obj = np.zeros((len(grid), 6))
    for ci in range(len(grid)):
        NL, PR, DX = eval_all(DT[DT.cfg == ci], PT[PT.cfg == ci], D, det, ph, T)
        NL = NL.rename(columns={"n_lanes_p": "n_lanes_p"})
        ev[ci] = (NL, PR, DX)
        for f in range(6):
            s = score(NL, PR, DX, lambda x: x.fold == f, lambda x: x.fold == f,
                      lambda x: x.fold == f)
            obj[ci, f] = (s["nl_exact"] or 0) + (s["pair_acc_multi"] or 0)
        log(f"cfg {grid[ci]}: {json.dumps(score(NL, PR, DX))}")
    pick = {f: int(np.argmax(obj[:, [k for k in range(6) if k != f]].sum(1))) for f in range(6)}
    log(f"picked per fold: {[grid[pick[f]] for f in range(6)]}")
    NL = pd.concat([ev[pick[f]][0][ev[pick[f]][0].fold == f] for f in range(6)])
    PR = pd.concat([ev[pick[f]][1][ev[pick[f]][1].fold == f] for f in range(6)])
    DX = pd.concat([ev[pick[f]][2][ev[pick[f]][2].fold == f] for f in range(6)])
    res = {"grid": [list(g) for g in grid], "picked": {f: list(grid[pick[f]]) for f in range(6)},
           "grid_all": {str(grid[ci]): score(*ev[ci]) for ci in range(len(grid))}}
    res["oof"] = {"all": score(NL, PR, DX)}
    for wg in WGS:
        res["oof"][wg] = score(NL, PR, DX, lambda x: x.wg == wg, lambda x: x.wg == wg,
                               lambda x: x.wg == wg)
        log(f"OOF {wg}: {json.dumps(res['oof'][wg])}")
    # by true n_lanes (full)
    res["by_true_nlanes_full"] = {}
    for k in (1, 2, 3, 4):
        m = (NL.wg == "full") & (NL.n_lanes == k) & NL.n_lanes_p.notna()
        res["by_true_nlanes_full"][k] = {
            "n": int(m.sum()),
            "pred_dist": NL[m].n_lanes_p.value_counts().sort_index().to_dict()}
    # lane_conf calibration (detector-level all-pairs-right), multi-lane phases
    c = DX[DX.covered & DX.allok.notna() & (DX.n_lanes >= 2)]
    bins = [0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.01]
    c = c.assign(b=pd.cut(c.lane_conf, bins, right=False))
    res["conf_calibration_multi"] = {str(k): {"n": int(len(g)), "all_pairs_right": _r(g.allok.mean()),
                                              "lane_exact": _r(g.exact.mean())}
                                     for k, g in c.groupby("b", observed=True)}
    cn = NL[NL.n_lanes_p.notna()].assign(b=pd.cut(NL.n_lanes_conf, bins, right=False))
    res["nlanes_conf_calibration"] = {str(k): {"n": int(len(g)),
                                               "exact": _r((g.n_lanes == g.n_lanes_p).mean())}
                                      for k, g in cn.groupby("b", observed=True)}
    # phase correct subset: pred phase of both = truth
    ok = lambda x: (x.phase_a == x.ph_num) & (x.phase_b == x.ph_num)  # noqa: E731
    PR["ph_num"] = PR.target.str[1:].astype(int)
    for wg in WGS:
        res["oof"][wg]["pairs_phase_correct"] = score(
            NL, PR, DX, None, lambda x: (x.wg == wg) & ok(x), None)["pair_acc_multi"]
    log(json.dumps({k: v for k, v in res.items() if k not in ("grid_all",)}, default=str)[:3000])
    NL.to_parquet(OUT / "ln3_oof_nlanes.parquet", index=False)
    PR.to_parquet(OUT / "ln3_oof_pairs.parquet", index=False)
    DX.to_parquet(OUT / "ln3_oof_dets.parquet", index=False)
    json.dump(res, open(OUT / "ln3_decode.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--reuse", action="store_true", help="reuse the saved grid decode")
    REUSE = ap.parse_known_args()[0].reuse
    main(ap.parse_args().workers)
