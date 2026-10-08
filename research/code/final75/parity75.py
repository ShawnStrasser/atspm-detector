"""Note 75: parity of the integrated package (`%DC_WORK%/final_v3_candidate_v3`) with the six-fold OOF research pipeline
(cand64 / s59 / s62 / s67 / s69 / s74), on a handful of non-locked FOLD-0 signals, Sept windows.

Package run in "frame mode" (note 66's way): the signal's tables are built ONCE from its whole Sept log, every window is
cut with build_features(w0, w1) -- the research builder's definition; the production path (window-only events) differs
only by edge effects.

  A  INJECTED chain (same models / same inputs as the research OOF, so answers must agree): phase input = the frame's
     OOF phase blend; P_trees = the 229-arm OOF (3 seeds); P_net = siba OOF (3 seeds); lane pair probabilities = the
     OOF D pair model's (p_D) with the fold-0 decoder settings (lam 2, fold-0 span prior); stacker = the fold-0 stacker
     refitted here exactly as s74 (3 seeds, trained on folds 1-5).  Compared stage by stage: the 229 features, lanes,
     pick inputs, function-free health context, the 62 stacker columns, the stacker output, the final decoded function.
  B  PRODUCTION models (full-data refits; siba member = fold-0 seed-0 model): agreement with the OOF answers (phase
     top-1, function), siba vs its own OOF (same model: fold-0 seed 0, GPU bf16 there), and every ONNX tree model vs
     the numpy evaluation of its text model on the inputs captured in these runs.

    python parity75.py [--n 6] [--wins m5_a,m10_a,m30_a,h3_a,h24_a,full66]  -> %DC_WORK%/final_v3_work/parity75/
locked_v2 asserted absent. CPU only.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "6")
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: F401,E402
from common import DC_WORK  # noqa: E402

PKG = DC_WORK / "final_v3_candidate_v3"
FIT = DC_WORK / "final_v3_work" / "v3fit"
OUT = DC_WORK / "final_v3_work" / "parity75"
EVDIR = DC_WORK / "official" / "stg" / "cache" / "events"
SPAN = ("2026-09-18 00:00:00", "2026-09-22 00:00:00")
C7 = ["Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other"]
KEY = ["DeviceId", "Detector", "period", "win"]
DBG: list = []


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def locked() -> set:
    return set(pd.read_csv(DC_WORK / "official" / "locked_v2.csv").DeviceId.str.lower())


# ================================================================================================ research side
def research(sigs, wins):
    import cand64 as C
    import s67_decider as S
    import s74 as S74
    E = S74.setup("x69_siba")
    fr = E["fr"]
    Pt, Pn = E["Pt"], E["Pn"]
    Pb = C.W_TREE * Pt + (1 - C.W_TREE) * Pn
    X = np.hstack([S.stack_X(fr, Pt, Pn), S.ctx_X(E, Pb)])
    P_oof = np.mean([np.load(DC_WORK / "s69" / "on67" / f"stack_ctx_s{s}.npy") for s in (0, 1, 2)], 0).astype(float)
    lc = pd.read_parquet(C.OUT / "function_rows.parquet", columns=["DeviceId", "win", "lane_conf", "phase_n_lanes",
                                                                   "phase_n_lanes_conf"])
    assert (lc.DeviceId.to_numpy() == fr.DeviceId.to_numpy()).all()
    lanes0 = fr.lanes5g.copy()
    fr["lanes5g"] = lanes0.where(~(lc.lane_conf.to_numpy(float) < S74.GATE), None)
    pred, _ = S.run_decode(E, P_oof)
    fr["lanes5g"] = lanes0
    # fold-0 stacker, exactly s74 / s67 oof_stack (trained on folds != 0)
    y = np.array([C7.index(t) if isinstance(t, str) and t in C7 else -1 for t in fr.truth_v3s.to_numpy(object)])
    trm = (y >= 0) & ~fr.exclude_train_score.fillna(False).to_numpy(bool) & (fr.det_n_on >= 5).to_numpy()
    import lightgbm as lgb
    d0 = OUT / "stacker_f0"
    d0.mkdir(parents=True, exist_ok=True)
    for s in (0, 1, 2):
        f = d0 / f"stacker_f0_s{s}.txt"
        if not f.exists():
            prm = dict(objective="multiclass", num_class=7, learning_rate=0.05, num_leaves=7, max_depth=3,
                       min_data_in_leaf=300, lambda_l2=10.0, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                       num_threads=S.THREADS, verbose=-1, seed=s)
            tr = trm & (fr.fold.to_numpy() != 0)
            lgb.train(prm, lgb.Dataset(X[tr], y[tr]), num_boost_round=150).save_model(str(f))
    m = fr.DeviceId.isin(sigs).to_numpy() & (fr.period == "stg").to_numpy() & fr.win.isin(wins).to_numpy()
    sub = fr[m].copy()
    sub["_row"] = np.flatnonzero(m)
    return dict(fr=sub, Pt=Pt[m], Pn=Pn[m], X=X[m], P_oof=P_oof[m], pred=pred[m], lconf=lc.lane_conf.to_numpy()[m],
                LN=lc[["phase_n_lanes", "phase_n_lanes_conf", "lane_conf"]].to_numpy(float)[m], ctx_cols=X.shape[1])


def pick_signals(n):
    import cand64 as C  # noqa: F401
    import v3_retrain as V
    V.set_frame("v6e")
    k = pd.read_parquet(V.FEATS, columns=V.KEY + ["fold"])
    k["DeviceId"] = k.DeviceId.str.lower()
    k = k[(k.period == "stg") & (k.fold == 0)]
    nd = k.groupby("DeviceId").Detector.nunique()
    ok = sorted(d for d in nd[(nd >= 12) & (nd <= 30)].index
                if d not in locked() and (EVDIR / f"DeviceId={d}").is_dir())
    rng = np.random.default_rng(75)
    return sorted(rng.choice(ok, n, replace=False).tolist())


# ================================================================================================ package side
def pkg():
    sys.path.insert(0, str(PKG))
    for f in PKG.glob("*.py"):
        sys.modules.pop(f.stem, None)
    import predict as P
    assert Path(P.__file__).resolve().parent == PKG.resolve(), P.__file__
    return P


class FoldStacker:
    def __init__(self, files):
        import stacker as ST
        import trees_onnx as TO
        sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code") / "final75"))
        import onnx_trees75 as OT
        from lgbm_numpy import NumpyBooster
        on = []
        for f in files:
            o = Path(f).with_suffix(".onnx")
            OT.to_onnx_te5(NumpyBooster(f), o)
            on.append(o)
        self.bag = TO.Bag(on)
        self.ST = ST

    def predict(self, minutes, Pt, Pn, fr):
        Pb = self.ST.W_TREE * Pt + (1 - self.ST.W_TREE) * Pn
        X = np.hstack([self.ST.stack_X(minutes, Pt, Pn), self.ST.ctx_X(fr, Pb).astype(np.float64)])
        return self.bag.predict(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--wins", default="m5_a,m10_a,m30_a,h3_a,h24_a,full66")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    wins = a.wins.split(",")
    sigs = pick_signals(a.n)
    log(f"signals {sigs}")
    R = research(sigs, wins)
    log(f"research rows {len(R['fr']):,}")
    import ln8_validate as L8
    import lane_output as LO
    import a2_features as A2F
    import v3_retrain as V
    det_t, ph_t, _ = L8.truth_tabs()
    D = L8.load_dets()
    D9 = D[(D.period == "stg") & D.win.isin(L8.TRAIN_WINS) & D.DeviceId.isin(set(ph_t.DeviceId))]
    fsig = D9.groupby("DeviceId").fold.first()
    prior0 = L8.span_priors(det_t, ph_t, D9.rename(columns={"Detector": "det"}), set(fsig.index[fsig != 0]))[0]
    pk = json.load(open(L8.OUT / "decode_pick.json"))["pick"]["D.func"]
    lam0 = float(pk["0"].split("@")[1])
    pD = pd.read_parquet(L8.OUT / "p_D.parquet")
    pD = pD[pD.DeviceId.isin(sigs) & (pD.period == "stg") & pD.same_pred]
    ov = pd.read_parquet(DC_WORK / "final_v3_work" / "function_v3e" / "phase" / "phase_pred_v3blend_stg.parquet")
    ov["DeviceId"] = ov.DeviceId.str.lower()
    ov = ov[ov.DeviceId.isin(sigs)]
    cols = json.load(open(PKG / "weights" / "function" / "function229.json"))["features"]
    V.set_frame("v6e")
    FE = pd.read_parquet(V.FEATS, columns=KEY + ["pred_phase"] + cols)
    FE["DeviceId"] = FE.DeviceId.str.lower()
    FE = FE[FE.DeviceId.isin(sigs) & (FE.period == "stg") & FE.win.isin(wins)]
    ph_oof = pd.read_parquet(DC_WORK / "cand64" / "phase_oof.parquet", columns=["DeviceId", "Detector", "win", "cand_phase",
                                                                                "p2_cand"])
    ph_oof = ph_oof[ph_oof.DeviceId.str.endswith("@stg")]
    ph_oof["DeviceId"] = ph_oof.DeviceId.str.replace("@stg", "").str.lower()
    ph_oof = ph_oof[ph_oof.DeviceId.isin(sigs)]
    sib0 = pd.read_parquet(DC_WORK / "tcn53" / "fpreds" / "x69_siba_f0.parquet")
    sib0["DeviceId"] = sib0.DeviceId.str.lower()
    P = pkg()
    import function_stage as FS
    import gru_blend as gb
    import lanes as LM
    import stacker as ST
    import trees_onnx as TO
    stk = FoldStacker([OUT / "stacker_f0" / f"stacker_f0_s{s}.txt" for s in (0, 1, 2)])
    dec0 = LM.Decoder({f: {int(k): v for k, v in d.items()} for f, d in prior0.items()}, lam=lam0, beta=0.0, role_pen=4.0)
    md = PKG / "weights"
    rf = R["fr"].reset_index(drop=True)
    ridx = {k: i for i, k in enumerate(zip(rf.DeviceId, rf.Detector.astype(int), rf.win))}
    TO.CAPTURE = {}
    rows = []
    DBG.clear()
    stagecmp = {k: [] for k in ("feat", "lanes", "lane_conf", "phase_n_lanes", "pick", "hf", "ctx", "Ps", "pred_A",
                                "phase_B", "Pt_B", "Pn_B", "pred_B")}
    for dev in sigs:
        t_dev = time.time()
        con = P._connect(6, "6GB")
        ev = con.sql(f"SELECT '{dev}' AS DeviceId, Timestamp, EventId, Parameter FROM read_parquet("
                     f"'{(EVDIR / f'DeviceId={dev}' / '*.parquet').as_posix()}', hive_partitioning=false) WHERE "
                     f"Timestamp >= TIMESTAMP '{SPAN[0]}' AND Timestamp < TIMESTAMP '{SPAN[1]}'").df()
        P.load_events(con, ev)
        P.build_chunk_tables(con)
        cfg = gb.config(md)
        ep = pd.Timestamp("1970-01-01")
        for name, t0, secs in A2F.WINDOWS["stg"]:
            if name not in wins:
                continue
            w0 = (pd.Timestamp(t0) - ep).total_seconds()
            w1 = w0 + float(secs)
            minutes = round(secs / 60.0)
            streams, pieces = gb.streams_for(con, int(round(w0 * 1000)), int(round(w1 * 1000)), **gb.piece_plan(cfg, minutes))
            df, sim = P.build_features(con, w0, w1)
            if df is None or not len(df):
                continue
            # ---------------- B: production models
            dfb = P.score(df.copy(), sim, md, con, w0, w1, cfg, streams, pieces)
            capB = {}
            topB = P._function_frame(dfb)
            ex = P.fx.build(con, w0, w1, topB)
            ex["Detector"] = ex.Detector.astype(topB.Detector.dtype)
            topB = topB.merge(ex, on=["DeviceId", "Detector"], how="left")
            fnB, _ = FS.run(con, topB, md, w0, w1, streams, pieces, inject={"capture": capB})
            # ---------------- A: injected (research inputs, fold-0 models)
            o = ov[ov.win == name].drop(columns="win").copy()
            o["Detector"] = o.Detector.astype(df.Detector.dtype)
            o["cand_phase"] = o.cand_phase.astype(df.cand_phase.dtype)
            dfa = df.merge(o, on=["DeviceId", "Detector", "cand_phase"], how="inner")
            topA = P._function_frame(dfa)
            ex = P.fx.build(con, w0, w1, topA)
            ex["Detector"] = ex.Detector.astype(topA.Detector.dtype)
            topA = topA.merge(ex, on=["DeviceId", "Detector"], how="left")

            def inj(arr):
                def f(dv, dets):
                    v = np.full((len(dets), 7), np.nan)
                    for j, d in enumerate(dets):
                        i = ridx.get((dv, int(d), name))
                        if i is not None:
                            v[j] = arr[i]
                    return v
                return f
            q = pD[(pD.win == name) & (pD.DeviceId == dev)]
            lp = {dev: {(int(x), int(y)): float(z) for x, y, z in zip(q.da, q.db, q.p_same)}}
            if os.environ.get("P75_DEBUG"):
                print("DEBUG lp", dev[:8], name, len(lp[dev]), {k: v for k, v in lp[dev].items() if k[0] in (16, 17)})
            capA = {}
            FS.run(con, topA, md, w0, w1, streams, pieces,
                   inject={"Pt": inj(R["Pt"]), "Pn": inj(R["Pn"]), "lane_probs": lp, "lane_decoder": dec0,
                           "stacker": stk, "capture": capA})
            # ---------------- compare
            fe = FE[(FE.DeviceId == dev) & (FE.win == name)]
            tA = topA.assign(DeviceId=topA.DeviceId.str.lower(), Detector=topA.Detector.astype(int))
            mm = fe.assign(Detector=fe.Detector.astype(int)).merge(tA[["DeviceId", "Detector"] + cols],
                                                                   on=["DeviceId", "Detector"], suffixes=("_r", ""))
            for c in cols:
                x, y = mm[c + "_r"].to_numpy(float), mm[c].to_numpy(float)
                eq = (np.isnan(x) & np.isnan(y)) | (np.abs(x - y) <= 1e-4 * (1 + np.abs(x)))
                stagecmp["feat"].append((name, c, int(len(eq)), int((~eq).sum())))
            cA = capA.get(dev)
            if cA is None:
                continue
            fA = cA["fr"]
            dets = fA.Detector.astype(int).to_numpy()
            ri = np.array([ridx.get((dev, int(d), name), -1) for d in dets])
            hv = ri >= 0
            r = rf.iloc[ri[hv]]
            la = fA.lanes5g.to_numpy(object)[hv]
            lr = r.lanes5g.to_numpy(object)
            norm = lambda v: v if isinstance(v, str) and v else ""                          # noqa: E731
            stagecmp["lanes"].append((name, int(hv.sum()), int(sum(norm(x) != norm(y) for x, y in zip(la, lr)))))
            DBG.append(pd.DataFrame({"dev": dev, "win": name, "det": dets[hv], "lanes_pkg": [norm(x) for x in la],
                                     "lanes_res": [norm(x) for x in lr], "lc_pkg": fA.lane_conf.to_numpy(float)[hv],
                                     "lc_res": R["lconf"][ri[hv]], "pred_pkg": cA["pred"][hv], "pred_res": R["pred"][ri[hv]],
                                     "n_on": fA.det_n_on.to_numpy()[hv], "pp": fA.pred_phase.to_numpy()[hv],
                                     "pp_res": r.pred_phase.to_numpy(), "func_pkg": cA["func_t"][hv],
                                     "pt_res_argmax": R["Pt"][ri[hv]].argmax(1), "pt_pkg_argmax": cA["Pt"][hv].argmax(1)}))
            lcA, lcR = fA.lane_conf.to_numpy(float)[hv], R["lconf"][ri[hv]]
            stagecmp["lane_conf"].append((name, float(np.nanmax(np.abs(np.nan_to_num(lcA, nan=-1) -
                                                                         np.nan_to_num(lcR, nan=-1)), initial=0))))
            nlA = fA[["phase_n_lanes", "phase_n_lanes_conf"]].to_numpy(float)[hv]
            stagecmp["phase_n_lanes"].append((name, float(np.nanmax(np.abs(np.nan_to_num(nlA, nan=-1) -
                                                                              np.nan_to_num(R["LN"][ri[hv], :2], nan=-1)),
                                                                       initial=0))))
            pkd = int((fA.pk_span.to_numpy(bool)[hv] != r.pk_span.to_numpy(bool)).sum()
                      + (fA.pk_unhealthy.to_numpy(bool)[hv] != r.pk_unhealthy.to_numpy(bool)).sum()
                      + sum(norm(x) != norm(y) for x, y in zip(fA.pk_span_peers.to_numpy(object)[hv], r.pk_span_peers))
                      + sum(norm(x) != norm(y) for x, y in zip(fA.pk_coloc_peers.to_numpy(object)[hv], r.pk_coloc_peers)))
            trd = float(np.nanmax(np.abs(np.nan_to_num(fA.pk_track.to_numpy(float)[hv], nan=-9) -
                                         np.nan_to_num(r.pk_track.to_numpy(float), nan=-9)), initial=0))
            stagecmp["pick"].append((name, int(hv.sum()), pkd, trd))
            Xr = R["X"][ri[hv]]
            Pb = ST.W_TREE * cA["Pt"] + (1 - ST.W_TREE) * cA["Pn"]
            Xp = np.hstack([ST.stack_X(secs / 60.0, cA["Pt"], cA["Pn"]), ST.ctx_X(fA, Pb).astype(np.float64)])[hv]
            # research stack_X uses the nominal window minutes (WMIN) -> same as secs / 60 except full (3960 vs 3960)
            dX = np.abs(np.nan_to_num(Xp, nan=-999) - np.nan_to_num(Xr, nan=-999))
            stagecmp["hf"].append((name, float(dX[:, 20:36].max(initial=0))))
            stagecmp["ctx"].append((name, [float(v) for v in dX.max(0)]))
            stagecmp["Ps"].append((name, float(np.abs(cA["Ps"][hv] - R["P_oof"][ri[hv]]).max(initial=0))))
            stagecmp["pred_A"].append((name, int(hv.sum()), int((cA["pred"][hv] != R["pred"][ri[hv]]).sum())))
            # B: production vs OOF
            cB = capB[dev]
            fB = cB["fr"]
            detsB = fB.Detector.astype(int).to_numpy()
            riB = np.array([ridx.get((dev, int(d), name), -1) for d in detsB])
            hB = riB >= 0
            stagecmp["Pt_B"].append((name, int(hB.sum()), int((cB["Pt"][hB].argmax(1) == R["Pt"][riB[hB]].argmax(1)).sum())))
            s0 = sib0[(sib0.DeviceId == dev) & (sib0.win == name) & (sib0.period == "stg")].set_index("Detector")
            pnr = s0.reindex(detsB)[[f"P_{c}" for c in C7]].to_numpy(float)
            ok = ~np.isnan(pnr[:, 0])
            stagecmp["Pn_B"].append((name, int(ok.sum()), float(np.abs(cB["Pn"][ok] - pnr[ok]).max(initial=0)),
                                     int((cB["Pn"][ok].argmax(1) == pnr[ok].argmax(1)).sum())))
            stagecmp["pred_B"].append((name, int(hB.sum()), int((cB["pred"][hB] == R["pred"][riB[hB]]).sum()),
                                       int((cB["Ps"][hB].argmax(1) == R["P_oof"][riB[hB]].argmax(1)).sum())))
            # phase top-1: production decoded vs cand64 OOF decoded
            pb = dfb[["DeviceId", "Detector", "cand_phase", "prob"]].copy()
            pb["DeviceId"] = pb.DeviceId.str.lower()
            tb = pb.sort_values("prob").groupby("Detector").tail(1).set_index("Detector").cand_phase
            po = ph_oof[(ph_oof.DeviceId == dev) & (ph_oof.win == name)]
            to = po.sort_values("p2_cand").groupby("Detector").tail(1).set_index("Detector").cand_phase
            j = tb.index.intersection(to.index)
            stagecmp["phase_B"].append((name, int(len(j)), int((tb[j].astype(int) == to[j].astype(int)).sum())))
            rows.append(dict(dev=dev, win=name, dets=len(dets)))
        con.close()
        log(f"{dev[:8]}: {time.time()-t_dev:.0f}s")
    TO.CAPTURE, cap = None, TO.CAPTURE
    # ---- every ONNX tree model vs its text model on the captured production inputs
    from lgbm_numpy import NumpyBooster
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import onnx_trees75 as OT
    src = {"phase_lgbm_v5": DC_WORK / "final_v3_candidate_v2" / "weights", "decode_v3": FIT / "decoder",
           "function229": FIT / "function", "lane_pair_D": FIT / "lanes", "stacker_s": FIT / "stacker",
           "setback_q10": DC_WORK / "final_v3_candidate_v2" / "weights" / "setback",
           "setback_q90": DC_WORK / "final_v3_candidate_v2" / "weights" / "setback"}
    tp = {}
    for path, Xs in cap.items():
        pth = Path(path)
        if "stacker_f0" in str(pth):
            continue
        X = np.vstack(Xs)
        stem = pth.stem
        sdir = next((v for k, v in src.items() if stem.startswith(k)), FIT / "setback")
        nb = NumpyBooster(sdir / f"{stem}.txt")
        m = TO.OnnxTrees(pth)
        tp[pth.name] = {"rows": int(len(X)), "max_abs_raw": float(np.abs(OT.raw_numpy(nb, X) - m.raw(X)).max()),
                        "max_abs_out": float(np.abs(np.asarray(nb.predict(X)) - np.asarray(m.predict(X))).max())}
    if DBG:
        pd.concat(DBG, ignore_index=True).to_parquet(OUT / "debug_rows.parquet", index=False)
    res = summarise(stagecmp, cols, R["ctx_cols"])
    res["trees_on_captured_inputs"] = tp
    res["signals"], res["windows"], res["runs"] = sigs, wins, rows
    json.dump(res, open(OUT / "parity75.json", "w"), indent=1, default=str)
    log(json.dumps({k: v for k, v in res.items() if k not in ("runs", "trees_on_captured_inputs", "ctx_max_by_col")},
                   default=str))
    log("trees max raw diff " + str(max(v["max_abs_raw"] for v in tp.values())))


def summarise(sc, cols, nctx):
    r = {}
    f = pd.DataFrame(sc["feat"], columns=["win", "col", "n", "bad"])
    r["features_229"] = {"values": int(f.n.sum()), "differing": int(f.bad.sum()),
                         "cols_differing": f[f.bad > 0].groupby("col").bad.sum().sort_values(ascending=False).head(10).to_dict(),
                         "by_win_differing": f.groupby("win").bad.sum().to_dict()}
    L = pd.DataFrame(sc["lanes"], columns=["win", "n", "bad"])
    r["lanes_A"] = {"rows": int(L.n.sum()), "lane_strings_differing": int(L.bad.sum()),
                    "lane_conf_max_abs": max([x[1] for x in sc["lane_conf"]] or [0]),
                    "phase_n_lanes_max_abs": max([x[1] for x in sc["phase_n_lanes"]] or [0])}
    K = pd.DataFrame(sc["pick"], columns=["win", "n", "bad", "track"])
    r["pick_A"] = {"rows": int(K.n.sum()), "flags_or_peers_differing": int(K.bad.sum()), "track_max_abs": float(K.track.max())}
    r["hf_A_max_abs"] = max([x[1] for x in sc["hf"]] or [0])
    cm = np.max(np.array([x[1] for x in sc["ctx"]]), 0) if sc["ctx"] else np.zeros(nctx)
    r["ctx_max_by_col"] = [round(float(v), 8) for v in cm]
    r["ctx_A_max_abs"] = float(cm.max())
    r["stacker_A_max_abs"] = max([x[1] for x in sc["Ps"]] or [0])
    pa = pd.DataFrame(sc["pred_A"], columns=["win", "n", "diff"])
    r["final_function_A"] = {"rows": int(pa.n.sum()), "differing": int(pa["diff"].sum()),
                             "by_win": pa.groupby("win")["diff"].sum().to_dict()}
    pb = pd.DataFrame(sc["phase_B"], columns=["win", "n", "same"])
    r["phase_B_top1_agree"] = {"rows": int(pb.n.sum()), "agree": round(float(pb.same.sum() / max(pb.n.sum(), 1)), 4),
                               "by_win": (pb.groupby("win").same.sum() / pb.groupby("win").n.sum()).round(4).to_dict()}
    t = pd.DataFrame(sc["Pt_B"], columns=["win", "n", "same"])
    r["trees_B_argmax_agree"] = round(float(t.same.sum() / max(t.n.sum(), 1)), 4)
    n = pd.DataFrame(sc["Pn_B"], columns=["win", "n", "maxabs", "same"])
    r["siba_B_vs_own_oof"] = {"rows": int(n.n.sum()), "max_abs_prob": float(n.maxabs.max()),
                              "argmax_agree": round(float(n.same.sum() / max(n.n.sum(), 1)), 4),
                              "by_win_max_abs": n.groupby("win").maxabs.max().round(5).to_dict()}
    q = pd.DataFrame(sc["pred_B"], columns=["win", "n", "same", "same_stack_argmax"])
    r["final_function_B_agree"] = {"rows": int(q.n.sum()), "agree": round(float(q.same.sum() / max(q.n.sum(), 1)), 4),
                                   "stacker_argmax_agree": round(float(q.same_stack_argmax.sum() / max(q.n.sum(), 1)), 4),
                                   "by_win": (q.groupby("win").same.sum() / q.groupby("win").n.sum()).round(4).to_dict()}
    return r


if __name__ == "__main__":
    main()
