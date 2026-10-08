"""Note 67: (A) six-fold re-test of the note-63 decider (stacker) now that fj exists on every fold;
(B) Count / Presence regression vs final_v2 (the published beta): diagnosis + one cheap nested fix.

Current function (cand64 + note 65): trees (229 features, 3 seeds) blended with the TCN fj head (3 seeds averaged) at tree
weight 0.6 -> D-lane stack-pick decode -> short-window twin decode (cy, thr .4, 5 / 10 min only).
Scoring = cand64 / note-59 harness (ATSPM-only, stack-aware; everything E / realistic R); headline = >= 30-min pool;
95 % CI = paired signal bootstrap (cand64.delta_ci). locked_v2 asserted absent. CPU only, <= 4 threads.

    python s67_decider.py base        # cache Pt / Pn / intermediate decodes -> %DC_WORK%/s67/
    python s67_decider.py diag        # (B) Count / Presence rows final_v2 gets right and the candidate misses
    python s67_decider.py stack       # (A) probs-only / context / shuffled-context stackers, OOF over folds, 3 seeds
    python s67_decider.py fix [--base blend|ctx]     # (B) lane-confidence gate x class-weight grid, nested over folds
    python s67_decider.py demote [--base blend|ctx]  # (B, note 70) audit of true-ATSPM demotions + nested relaxations
                                                     # (stack-loser rule, unconfident-winner tau, lane-confidence gate)

Stacker OOF: the stacker applied to fold k is trained only on rows of the other five folds (their OOF base
probabilities) - no fold-k row or label. Caveat (standard OOF stacking): the base models that produced fold j's OOF
probabilities were themselves trained on folds that include k; no nested base OOF exists for the TCN.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
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

OUT = DC_WORK / "s67"
THREADS = 4
GE30 = ["m30", "h1", "h3", "h6", "h24", "full"]
WMIN = {"m5": 5, "m10": 10, "m30": 30, "h1": 60, "h3": 180, "h6": 360, "h24": 1440, "full": 3960}
ATS = ["Advance", "Presence", "Count", "Yellow_Red"]
_CTX = {}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def setup():
    import cand64 as C
    import s59_step6 as S59
    import s62_short as S62
    fr, rows = S59.scoring_frame(S59.PICK_TAG)
    fr = fr.copy()
    assert not fr.DeviceId.isin(C.locked()).any()
    Pt, seeds = S59.spec_probs("base", fr)
    assert seeds == [0, 1, 2]
    Pn, have = C.fj_probs(fr)
    assert len(have) == 6 and all(len(h["files"]) == 3 for h in have), have
    hasn = ~np.isnan(Pn[:, 0])
    Pn = np.where(hasn[:, None], Pn, Pt)
    pairs = pd.read_parquet(S62.PAIRS_F)
    short = fr.wgroup.isin(S62.SHORT).to_numpy() & (fr.det_n_on >= 5).to_numpy()
    tw = S62.twin_tokens(fr, pairs, C.TWIN_THR, 1.0, 0)
    return dict(fr=fr, rows=rows, Pt=Pt, Pn=Pn, hasn=hasn, short=short, tw=tw, C7=S59.C7)


def run_decode(E, P, steps=False):
    """full decode of probabilities P (frame order) -> (pred, credits per set) [+ intermediate preds]."""
    import atspm_score as S
    import s62_short as S62
    fr, C7 = E["fr"], E["C7"]
    for i, c in enumerate(C7):
        fr[f"P_{c}"] = P[:, i]
    pr = S.decode(fr, "lanes5g", "greedy", "strict", pick=True)
    pr2 = S62.twin_decode(P, pr, E["tw"], E["short"], ("Count", "Yellow_Red"), "cy")
    cr = {s: S.credit(fr, pr2, "truth_v3s", r, True) for s, r in E["rows"].items()}
    if not steps:
        return pr2, cr
    nop = S.decode(fr, "lanes5g", "greedy", "strict", pick=False)
    return pr2, cr, {"argmax": P.argmax(1), "lane_nopick": nop, "lane_pick": pr, "twin": pr2}


def ok_cols(E, cr):
    """frame-aligned ok arrays per set (NaN = not a scoring row)."""
    n = len(E["fr"])
    o = {}
    for s, r in E["rows"].items():
        a = np.full(n, np.nan)
        a[r] = cr[s].ok_a.to_numpy(float)
        o[s[0].upper()] = a
    return o


# ================================================================================================ base
def stage_base(a):
    import cand64 as C
    E = setup()
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "Pt.npy", E["Pt"].astype(np.float32))
    np.save(OUT / "Pn.npy", E["Pn"].astype(np.float32))
    Pb = C.W_TREE * E["Pt"] + (1 - C.W_TREE) * E["Pn"]
    _, cr, st = run_decode(E, Pb, steps=True)
    _, crt, stt = run_decode(E, E["Pt"], steps=True)
    fr = E["fr"]
    keep = ["DeviceId", "Detector", "period", "win", "wgroup", "fold", "pred_phase", "det_n_on", "truth_v3s", "validated",
            "stack_group", "stack_role", "exclude_train_score", "lanes5g", "pk_span", "pk_unhealthy", "pk_track",
            "pk_span_peers", "pk_coloc_peers"]
    d = fr[keep].copy()
    C7 = np.array(E["C7"], object)
    for k, v in st.items():
        d[f"b_{k}"] = C7[v]
    for k, v in stt.items():
        d[f"t_{k}"] = C7[v]
    for s, v in ok_cols(E, cr).items():
        d[f"ok_{s}_b"] = v
    for s, v in ok_cols(E, crt).items():
        d[f"ok_{s}_t"] = v
    # step-wise stack-aware credit for the blend (which step loses a row)
    import atspm_score as S
    for k in ("argmax", "lane_nopick"):
        for i, c in enumerate(E["C7"]):
            fr[f"P_{c}"] = Pb[:, i]
        crk = {s: S.credit(fr, st[k], "truth_v3s", r, True) for s, r in E["rows"].items()}
        for s, v in ok_cols(E, crk).items():
            d[f"ok_{s}_b_{k}"] = v
    for s, r in E["rows"].items():
        tag = np.full(len(fr), "", object)
        tag[r] = cr[s].stack_tag.to_numpy(object)
        d[f"stacktag_{s[0].upper()}"] = tag
    # twin flag (hi-res co-actuation twin at the same window; all windows, for description only)
    import s62_short as S62
    pairs = pd.read_parquet(S62.PAIRS_F)
    tw_all = S62.twin_tokens(fr, pairs, C.TWIN_THR, 1.0, 0)
    d["has_twin"] = np.isin(np.arange(len(fr)), list(tw_all.keys()))
    d.to_parquet(OUT / "base_rows.parquet", index=False)
    # check: reproduces cand64 ok_*_fj
    f = pd.read_parquet(C.OUT / "function_rows.parquet", columns=["DeviceId", "Detector", "period", "win", "ok_E_fj",
                                                                   "ok_R_fj", "ok_E_trees"])
    assert len(f) == len(d) and (f.DeviceId.to_numpy() == d.DeviceId.to_numpy()).all()
    for s in "ER":
        a1, a2 = f[f"ok_{s}_fj"].to_numpy(), d[f"ok_{s}_b"].to_numpy()
        m = ~np.isnan(a1)
        log(f"check vs cand64 fj {s}: rows {m.sum():,} identical {np.mean(a1[m] == a2[m]):.5f}")
    log(f"-> {OUT / 'base_rows.parquet'}")


def load_E():
    E = setup()
    return E


# ================================================================================================ diag (B)
def tech_and_source(d):
    import atspm_score as S
    lab = pd.read_parquet(S.V3S, columns=["DeviceId", "detector", "technology", "source", "print_subtype", "print_confidence"])
    lab["DeviceId"] = lab.DeviceId.str.lower()
    lab = lab.rename(columns={"detector": "Detector"}).astype({"Detector": d.Detector.dtype}).drop_duplicates(
        ["DeviceId", "Detector"])
    x = d[["DeviceId", "Detector"]].merge(lab, on=["DeviceId", "Detector"], how="left")
    assert len(x) == len(d)
    for c in ("technology", "source", "print_subtype", "print_confidence"):
        d[c] = x[c].fillna("?").to_numpy()
    d["tech"] = d.technology.str.lower().map(lambda t: "radar" if "radar" in t else "loop" if "loop" in t
                                            else "video" if "video" in t else "?")
    return d


def pulse(d):
    import v3_retrain as V
    V.set_frame("v6e")
    F = pd.read_parquet(V.FEATS, columns=V.KEY + ["px_pulse_frac"])
    F["DeviceId"] = F.DeviceId.str.lower()
    F = F.astype({"Detector": d.Detector.dtype}).drop_duplicates(V.KEY)
    x = d[V.KEY].merge(F, on=V.KEY, how="left")
    pf = x.px_pulse_frac.to_numpy()
    d["pulse"] = np.where(np.isnan(pf), "?", np.where(pf >= 0.8, "pulse", np.where(pf <= 0.2, "normal", "mixed")))
    return d


def vc(s, n=8):
    return {str(k): int(v) for k, v in s.value_counts().head(n).items()}


def stage_diag(a):
    import cand64 as C
    d = pd.read_parquet(OUT / "base_rows.parquet")
    f = pd.read_parquet(C.OUT / "function_rows.parquet", columns=["pred_v2", "p_pred_v2", "ok_E_v2", "ok_R_v2"])
    d = pd.concat([d, f], axis=1)
    d = tech_and_source(pulse(d))
    res = {}
    for s in "ER":
        m = d[f"ok_{s}_b"].notna() & d[f"ok_{s}_v2"].notna() & d.wgroup.isin(GE30)
        x = d[m]
        sig = x.DeviceId.to_numpy()
        per = {}
        for c in ATS + ["Mid", "Bike", "Other", "nonATSPM"]:
            mm = (~x.truth_v3s.isin(ATS)) if c == "nonATSPM" else (x.truth_v3s == c)
            mm = mm.to_numpy()
            per[c] = {"n": int(mm.sum()), "v2": round(float(x[f"ok_{s}_v2"][mm].mean()), 4),
                      "trees": round(float(x[f"ok_{s}_t"][mm].mean()), 4),
                      "cand": round(float(x[f"ok_{s}_b"][mm].mean()), 4),
                      "d_cand_v2_pt": C.delta_ci(x[f"ok_{s}_v2"].to_numpy()[mm], x[f"ok_{s}_b"].to_numpy()[mm], sig[mm])}
        tot = {"n": int(len(x)), "signals": int(x.DeviceId.nunique()),
               "d_cand_v2_pt": C.delta_ci(x[f"ok_{s}_v2"].to_numpy(), x[f"ok_{s}_b"].to_numpy(), sig)}
        res[s] = {"per_class": per, "total": tot}
        log(f"{s}: per class {json.dumps(per)}")
        # rows v2 right, candidate wrong (Count / Presence)
        for c in ("Count", "Presence"):
            xc = x[x.truth_v3s == c]
            lost = xc[(xc[f"ok_{s}_v2"] == 1) & (xc[f"ok_{s}_b"] == 0)]
            won = xc[(xc[f"ok_{s}_v2"] == 0) & (xc[f"ok_{s}_b"] == 1)]
            # decode step that loses it: argmax already wrong / lane decode w/o pick / stack pick / twin (short only)
            stp = np.where(lost[f"ok_{s}_b_argmax"] == 0,
                           np.where(lost[f"ok_{s}_t"] == 0, "argmax (trees wrong too)", "argmax (fj blend flips it)"),
                           np.where(lost[f"ok_{s}_b_lane_nopick"] == 0, "lane decode", "stack pick"))
            lost = lost.assign(step=stp)
            r = {"lost": int(len(lost)), "won": int(len(won)), "net": int(len(won) - len(lost)),
                 "lost_signals": int(lost.DeviceId.nunique()),
                 "lost_top5_signal_share": round(float(lost.DeviceId.value_counts().head(5).sum() / max(len(lost), 1)), 3),
                 "lost_top_signals": vc(lost.DeviceId, 6),
                 "lost_called": vc(lost.b_twin), "won_v2_called": vc(won.pred_v2),
                 "lost_step": vc(lost.step), "lost_tech": vc(lost.tech), "won_tech": vc(won.tech),
                 "lost_pulse": vc(lost.pulse), "won_pulse": vc(won.pulse),
                 "lost_stack": vc(lost.stack_group.notna().map({True: "stacked", False: "not stacked"})),
                 "lost_stacktag": vc(lost[f"stacktag_{s}"].replace("", "-")),
                 "lost_twin": vc(lost.has_twin.map({True: "twin", False: "no twin"})),
                 "lost_source": vc(lost.source), "won_source": vc(won.source),
                 "lost_validated": vc(lost.validated.fillna("?")),
                 "lost_wgroup": vc(lost.wgroup), "lost_v2_p_median": round(float(lost.p_pred_v2.median()), 3),
                 "lost_called_x_tech": vc(lost.b_twin + "|" + lost.tech, 10),
                 "lost_called_x_pulse": vc(lost.b_twin + "|" + lost.pulse, 10),
                 "class_tech_share": vc(xc.tech), "class_pulse_share": vc(xc.pulse)}
            # accuracy of the class by tech / pulse, v2 vs cand
            for g in ("tech", "pulse", "source"):
                r[f"acc_by_{g}"] = {str(k): [int(len(v)), round(float(v[f'ok_{s}_v2'].mean()), 4),
                                             round(float(v[f'ok_{s}_b'].mean()), 4)]
                                    for k, v in xc.groupby(g) if len(v) >= 200}
            res[s][c] = r
            log(f"{s} {c}: {json.dumps(r)}")
            if s == "E":
                lost.to_parquet(OUT / f"diag_lost_{c}.parquet", index=False)
    json.dump(res, open(OUT / "diag.json", "w"), indent=1, default=str)
    log(f"-> {OUT / 'diag.json'}")


# ================================================================================================ stack (A)
def ent(P):
    Q = np.clip(P, 1e-9, 1)
    return -(Q * np.log(Q)).sum(1)


def margin(P):
    s = np.sort(P, 1)
    return s[:, -1] - s[:, -2]


def stack_X(fr, Pt, Pn):
    lm = np.log(fr.wgroup.map(WMIN).to_numpy(float))
    lt, ln = np.log(np.clip(Pt, 1e-6, 1)), np.log(np.clip(Pn, 1e-6, 1))
    return np.column_stack([lt, ln, lm, ent(Pt), ent(Pn), margin(Pt), margin(Pn),
                            (Pt.argmax(1) == Pn.argmax(1)).astype(float)])


def grp_second_max(v, g):
    """per element: max of v over the OTHER members of its group g (NaN for singletons)."""
    o = np.lexsort((-v, g))
    gs, vs = g[o], v[o]
    first = np.r_[True, gs[1:] != gs[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(gs)), 0))
    top = vs[start]
    sec_pos = start + 1
    sec = np.where((sec_pos < len(gs)) & (gs[np.minimum(sec_pos, len(gs) - 1)] == gs), vs[np.minimum(sec_pos, len(gs) - 1)],
                   np.nan)
    out_s = np.where(np.arange(len(gs)) == start, sec, top)
    r = np.empty(len(v))
    r[o] = out_s
    return r


def ctx_X(E, Pb):
    """hi-res-only context (user idea): health core + stack-relative key, pick / stack-membership inputs, D lanes
    (n_lanes, lane, lanes spanned), lane-mates' and phase-mates' ATSPM probabilities, ranks inside the phase."""
    import s59_step6 as S59
    fr = E["fr"]
    if "H" not in _CTX:
        _CTX["H"] = S59.health_feats(fr[S59.KEY].reset_index(drop=True)).to_numpy(np.float32)
        import cand64 as C
        f = pd.read_parquet(C.OUT / "function_rows.parquet", columns=["DeviceId", "Detector", "period", "win",
                                                                       "phase_n_lanes", "phase_n_lanes_conf", "lane_conf"])
        assert (f.DeviceId.to_numpy() == fr.DeviceId.to_numpy()).all() and (f.win.to_numpy() == fr.win.to_numpy()).all()
        _CTX["ln"] = f[["phase_n_lanes", "phase_n_lanes_conf", "lane_conf"]].to_numpy(float)
    H, LN = _CTX["H"], _CTX["ln"]
    n = len(fr)
    lanes = fr.lanes5g.to_numpy(object)
    lsets = [[int(x) for x in s.split(",")] if isinstance(s, str) and s else [] for s in lanes]
    nl_self = np.array([len(s) for s in lsets], float)
    lane_min = np.array([min(s) if s else np.nan for s in lsets], float)
    npeer = lambda col: np.array([len(s.split(",")) if isinstance(s, str) and s else 0
                                  for s in fr[col].to_numpy(object)], float)
    grp = fr.groupby(["DeviceId", "period", "win", "pred_phase"], dropna=False).ngroup().to_numpy()
    n_ph = pd.Series(grp).map(pd.Series(grp).value_counts()).to_numpy(float)
    phm = np.column_stack([grp_second_max(Pb[:, c], grp) for c in range(4)])
    rk = np.column_stack([pd.Series(Pb[:, c]).groupby(grp).rank(ascending=False).to_numpy() for c in range(4)])
    e = pd.DataFrame({"g": grp, "i": np.arange(n), "lane": lsets}).explode("lane").dropna(subset=["lane"])
    n_lanes_phase = pd.Series(grp).map(e.groupby("g").lane.nunique()).fillna(0).to_numpy(float)
    m = e.merge(e, on=["g", "lane"], suffixes=("", "_m"))
    m = m[m.i != m.i_m].drop_duplicates(["i", "i_m"])
    lm = np.full((n, 5), np.nan)
    if len(m):
        for c in range(4):
            m[f"p{c}"] = Pb[m.i_m.to_numpy(int), c]
        agg = m.groupby("i").agg(nm=("i_m", "size"), p0=("p0", "max"), p1=("p1", "max"), p2=("p2", "max"),
                                 p3=("p3", "max"))
        lm[agg.index.to_numpy(int)] = agg[["nm", "p0", "p1", "p2", "p3"]].to_numpy(float)
    lm[:, 0] = np.where(nl_self > 0, np.nan_to_num(lm[:, 0]), np.nan)
    pk = np.column_stack([fr.pk_span.to_numpy(float), fr.pk_unhealthy.to_numpy(float), fr.pk_track.to_numpy(float),
                          npeer("pk_span_peers"), npeer("pk_coloc_peers")])
    ln_on = np.log1p(fr.det_n_on.to_numpy(float))
    return np.column_stack([H, pk, nl_self, lane_min, n_lanes_phase, LN, n_ph, phm, rk, lm, ln_on]).astype(np.float32)


def fit_lgb(X, y, Xte, seed):
    import lightgbm as lgb
    prm = dict(objective="multiclass", num_class=7, learning_rate=0.05, num_leaves=7, max_depth=3,
               min_data_in_leaf=300, lambda_l2=10.0, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
               num_threads=THREADS, verbose=-1, seed=seed)
    m = lgb.train(prm, lgb.Dataset(X, y), num_boost_round=150)
    return m.predict(Xte)


def oof_stack(E, X, seed):
    fr, C7 = E["fr"], E["C7"]
    y = np.array([C7.index(t) if isinstance(t, str) and t in C7 else -1 for t in fr.truth_v3s.to_numpy(object)])
    trm = (y >= 0) & ~fr.exclude_train_score.fillna(False).to_numpy(bool) & (fr.det_n_on >= 5).to_numpy()
    fo = fr.fold.to_numpy()
    P = np.zeros((len(fr), 7))
    for k in range(6):
        tr = trm & (fo != k)
        te = fo == k
        P[te] = fit_lgb(X[tr], y[tr], X[te], seed)
    return P


def compare(E, okA, okB, res, name):
    import cand64 as C
    fr = E["fr"]
    sig = fr.DeviceId.to_numpy()
    out = {}
    for s in "ER":
        a, b = okA[s], okB[s]
        sc = ~np.isnan(a)
        for pool, fams in (("ge30", GE30), ("m5", ["m5"]), ("m10", ["m10"]), ("all", list(WMIN))):
            m = sc & fr.wgroup.isin(fams).to_numpy()
            out[f"{s}_{pool}"] = [round(float(b[m].mean()), 4)] + C.delta_ci(a[m], b[m], sig[m])
        m = sc & fr.wgroup.isin(GE30).to_numpy()
        out[f"{s}_ge30_by_fold"] = [round(100 * float(b[m & (fr.fold == k).to_numpy()].mean()
                                                      - a[m & (fr.fold == k).to_numpy()].mean()), 2) for k in range(6)]
    res[name] = out
    log(f"{name}: " + " ".join(f"{k} {v}" for k, v in out.items() if "fold" not in k or k.startswith("E")))
    return out


def stage_stack(a):
    import cand64 as C
    E = setup()
    Pt, Pn = E["Pt"], E["Pn"]
    Pb = C.W_TREE * Pt + (1 - C.W_TREE) * Pn
    _, cr0 = run_decode(E, Pb)
    ok0 = ok_cols(E, cr0)
    _, crt = run_decode(E, Pt)
    okt = ok_cols(E, crt)
    res = {"ref_blend": {s: round(float(np.nanmean(v[E["fr"].wgroup.isin(GE30).to_numpy()])), 4) for s, v in ok0.items()}}
    compare(E, okt, ok0, res, "blend0.6_vs_trees")
    X = stack_X(E["fr"], Pt, Pn)
    Cx = ctx_X(E, Pb)
    log(f"stack X {X.shape}, context {Cx.shape}")
    rng = np.random.default_rng(67)
    Cs = Cx.copy()
    wg = E["fr"].wgroup.to_numpy()
    for g in WMIN:
        ix = np.flatnonzero(wg == g)
        Cs[ix] = Cx[rng.permutation(ix)]
    arms = {"probs": X, "ctx": np.hstack([X, Cx]), "ctx_shuf": np.hstack([X, Cs])}
    if a.arms:
        arms = {k: v for k, v in arms.items() if k in a.arms.split(",")}
    oks = {}
    for nm, XX in arms.items():
        Ps = []
        for seed in a.seeds:
            t0 = time.time()
            P = oof_stack(E, XX, seed)
            Ps.append(P)
            np.save(OUT / f"stack_{nm}_s{seed}.npy", P.astype(np.float32))
            _, cr = run_decode(E, P)
            oks[(nm, seed)] = ok_cols(E, cr)
            compare(E, ok0, oks[(nm, seed)], res, f"{nm}_s{seed}_vs_blend0.6")
            log(f"  ({time.time() - t0:.0f}s)")
            json.dump(res, open(OUT / "stack.json", "w"), indent=1, default=str)
        _, cr = run_decode(E, np.mean(Ps, 0))
        compare(E, ok0, ok_cols(E, cr), res, f"{nm}_seedavg_vs_blend0.6")
        json.dump(res, open(OUT / "stack.json", "w"), indent=1, default=str)
    for seed in a.seeds:
        if ("ctx", seed) in oks and ("ctx_shuf", seed) in oks:
            compare(E, oks[("ctx_shuf", seed)], oks[("ctx", seed)], res, f"ctx_vs_ctxshuf_s{seed}")
        if ("ctx", seed) in oks and ("probs", seed) in oks:
            compare(E, oks[("probs", seed)], oks[("ctx", seed)], res, f"ctx_vs_probs_s{seed}")
    json.dump(res, open(OUT / "stack.json", "w"), indent=1, default=str)
    log(f"-> {OUT / 'stack.json'}")


# ================================================================================================ fix (B)
def per_class(E, ok, s, rows_m):
    fr = E["fr"]
    t = fr.truth_v3s.to_numpy(object)
    o = {}
    for c in ATS + ["nonATSPM"]:
        mm = rows_m & ((~np.isin(t, ATS)) if c == "nonATSPM" else (t == c))
        o[c] = round(float(ok[s][mm].mean()), 4)
    return o


def stage_fix(a):
    """two cheap decode-side fixes, alone and together, chosen NESTED: per held-out fold, the grid point with the best
    >= 30-min E score on the OTHER five folds (decode groups never cross a signal, so assembling rows is exact).
      gate     the one-per-lane constraint only binds detectors whose D-lane confidence >= thr (others unconstrained)
      weights  class prior shift on the blend before the decode: P(Count) x wc, P(Presence) x wp, renormalised"""
    import cand64 as C
    E = setup()
    fr, C7 = E["fr"], E["C7"]
    Pb = C.W_TREE * E["Pt"] + (1 - C.W_TREE) * E["Pn"]
    if a.base != "blend":                    # e.g. ctx = the seed-averaged context stacker (saved OOF from `stack`)
        Pb = np.mean([np.load(f) for f in sorted(OUT.glob(f"stack_{a.base}_s*.npy"))], 0).astype(float)
        log(f"fix on base {a.base}")
    lc = pd.read_parquet(C.OUT / "function_rows.parquet", columns=["DeviceId", "win", "lane_conf"])
    assert (lc.DeviceId.to_numpy() == fr.DeviceId.to_numpy()).all() and (lc.win.to_numpy() == fr.win.to_numpy()).all()
    lconf = lc.lane_conf.to_numpy(float)
    lanes0 = fr.lanes5g.copy()
    combos = [(t, wc, wp) for t in a.gates for wc in a.grid for wp in a.grid]
    ok = {}
    g30 = fr.wgroup.isin(GE30).to_numpy()
    for t, wc, wp in combos:
        fr["lanes5g"] = lanes0.where(~(lconf < t), None) if t > 0 else lanes0
        w = np.ones(7)
        w[C7.index("Count")], w[C7.index("Presence")] = wc, wp
        P = Pb * w
        P = P / P.sum(1, keepdims=True)
        _, cr = run_decode(E, P)
        ok[(t, wc, wp)] = ok_cols(E, cr)
        log(f"gate {t} C {wc} P {wp}: E ge30 {np.nanmean(ok[(t, wc, wp)]['E'][g30]):.4f}")
    fr["lanes5g"] = lanes0
    fo = fr.fold.to_numpy()
    base = ok[(0.0, 1.0, 1.0)]
    res = {"grid_full_E_ge30": {f"{k}": round(float(np.nanmean(v["E"][g30])), 5) for k, v in ok.items()}}
    sig = fr.DeviceId.to_numpy()
    v2a = pd.read_parquet(C.OUT / "function_rows.parquet", columns=["ok_E_v2", "ok_R_v2"])
    fams = {"gate": [c for c in combos if c[1] == 1.0 and c[2] == 1.0],
            "weights": [c for c in combos if c[0] == 0.0], "both": combos}
    for fam, grid in fams.items():
        if len(grid) < 2:
            continue
        nest = {s: np.full(len(fr), np.nan) for s in "ER"}
        chosen = []
        for k in range(6):
            inner = (fo != k) & g30 & ~np.isnan(base["E"])
            best = max(grid, key=lambda g: (round(float(np.nanmean(ok[g]["E"][inner])), 6),
                                            -g[0] - abs(g[1] - 1) - abs(g[2] - 1)))
            chosen.append(best)
            for s in "ER":
                nest[s][fo == k] = ok[best][s][fo == k]
        r = {"chosen_per_fold": chosen}
        compare(E, base, nest, r, f"{fam}_nested_vs_cand")
        for s in "ER":
            v2 = v2a[f"ok_{s}_v2"].to_numpy()
            m = ~np.isnan(base[s]) & g30
            mv = m & ~np.isnan(v2)
            r[f"per_class_{s}_ge30"] = {"cand": per_class(E, base, s, m), "fixed": per_class(E, nest, s, m)}
            r[f"paired_v2_{s}_ge30"] = {"v2": per_class(E, {s: v2}, s, mv), "cand": per_class(E, base, s, mv),
                                        "fixed": per_class(E, nest, s, mv),
                                        "total_v2_cand_fixed": [round(float(v2[mv].mean()), 4),
                                                                round(float(base[s][mv].mean()), 4),
                                                                round(float(nest[s][mv].mean()), 4)]}
            for c in ATS + ["nonATSPM"]:
                t = fr.truth_v3s.to_numpy(object)
                mm = m & ((~np.isin(t, ATS)) if c == "nonATSPM" else (t == c))
                r[f"per_class_{s}_ge30"][f"d_{c}"] = C.delta_ci(base[s][mm], nest[s][mm], sig[mm])
        res[fam] = r
        log(f"{fam}: " + json.dumps(r, default=str))
        json.dump(res, open(OUT / f"fix_{a.tag}.json", "w"), indent=1, default=str)
    json.dump(res, open(OUT / f"fix_{a.tag}.json", "w"), indent=1, default=str)
    log(f"-> {OUT / f'fix_{a.tag}.json'}")


# ================================================================================================ demote (B, note 70 overlap)
def decode_group_x(P, lanes, pick, tau=0.0, stack_loser="nonatspm_ap", rec=None):
    """atspm_decode.decode_group (greedy / strict / pick) with two optional relaxations and an audit trail.
    tau          a detector that loses its FIRST-choice ATSPM class keeps it when the detector holding that class in
                 the shared lane has P(class) < tau (an unconfident winner demotes nobody); 0 = as shipped
    stack_loser  "nonatspm_ap" (default, note 56b) | "next_free" (stacked losers continue like any other loser)
    rec          list -> (i, cls, winner, p_winner, rule, span_extended, n_lanes_self) per first-choice loss"""
    import atspm_decode as AD
    P = np.asarray(P, float)
    pred = P.argmax(1)
    plan = AD._pick_lanes(lanes, pred, pick)
    keys = AD._keys(plan, "strict")
    if not AD._conflicts(pred, keys):
        return pred
    n = len(P)
    out = np.full(n, -1)
    used = {}
    order = AD._pick_order(P, pred, pick)
    sl = stack_loser == "nonatspm_ap" and pick.get("stack") is not None
    for i, c in order:
        if out[i] >= 0:
            continue
        if not AD.ATSPM[c]:
            out[i] = c
            continue
        k = keys[i]
        if k is None:
            out[i] = c
            continue
        hold = [used[(ln, c)] for ln in k if (ln, c) in used]
        if not hold:
            out[i] = c
            for ln in k:
                used[(ln, c)] = i
            continue
        w = hold[0]
        first = c == pred[i]
        if first and tau > 0 and P[w, c] < tau:
            out[i] = c
            if rec is not None:
                rec.append((i, c, w, P[w, c], "kept_tau", plan[i] != lanes[i], len(lanes[i])))
            continue
        rule = ""
        if sl and first and c in (0, 1) and any(out[j] == c for j in (pick["stack"][i] or ())):
            out[i] = AD.N_IDX[np.argmax(P[i, AD.N_IDX])]
            rule = "stack_loser_nonatspm"
        elif first:
            rule = "next_free"
        if rec is not None and first:
            rec.append((i, c, w, P[w, c], rule, plan[i] != lanes[i], len(lanes[i])))
    return out


def decode_x(fr, P, tau=0.0, stack_loser="nonatspm_ap", gate=0.0, lconf=None, audit=False):
    """frame-level twin of atspm_score.decode(fr, 'lanes5g', 'greedy', 'strict', pick=True) on decode_group_x."""
    import atspm_decode as AD
    pred = P.argmax(1)
    lanes_all = fr.lanes5g.to_numpy(object).copy()
    if gate > 0:
        lanes_all[lconf < gate] = None
    sub = fr.loc[(fr.det_n_on >= 5), ["DeviceId", "period", "win", "pred_phase"]].copy()
    sub["has"] = pd.notna(lanes_all[sub.index.to_numpy()])
    hs = sub.groupby(["DeviceId", "period", "win", "pred_phase"]).has.transform("sum")
    sub = sub[hs.to_numpy() >= 2]
    det, unh, spn = fr.Detector.to_numpy(), fr.pk_unhealthy.to_numpy(), fr.pk_span.to_numpy()
    spp, cop, trk = fr.pk_span_peers.to_numpy(object), fr.pk_coloc_peers.to_numpy(object), fr.pk_track.to_numpy(float)
    rows = []
    for _, g in sub.groupby(["DeviceId", "period", "win", "pred_phase"]).indices.items():
        ix = sub.index.to_numpy()[g]
        ls = [frozenset(int(v) for v in s.split(",")) if isinstance(s, str) and s else frozenset() for s in lanes_all[ix]]
        pk = AD.stack_pick(det[ix], ls, unh[ix], spn[ix], spp[ix], cop[ix], trk[ix])
        rec = [] if audit else None
        pred[ix] = decode_group_x(P[ix], ls, pk, tau, stack_loser, rec)
        if audit:
            for (i, c, w, pw, rule, ext, nsp) in rec:
                rows.append((ix[i], c, ix[w], pw, rule, ext, nsp, w in (pk["stack"][i] or ())))
    if audit:
        return pred, pd.DataFrame(rows, columns=["row", "cls", "winner_row", "p_winner", "rule", "span_extended",
                                                 "n_lanes_self", "winner_in_stack"])
    return pred


def run_decode_x(E, P, **kw):
    import atspm_score as S
    import s62_short as S62
    fr, C7 = E["fr"], E["C7"]
    for i, c in enumerate(C7):
        fr[f"P_{c}"] = P[:, i]
    pr = decode_x(fr, P, **kw)
    pr2 = S62.twin_decode(P, pr, E["tw"], E["short"], ("Count", "Yellow_Red"), "cy")
    return pr2, {s: S.credit(fr, pr2, "truth_v3s", r, True) for s, r in E["rows"].items()}


def stage_demote(a):
    """audit: which decode step turns a true Advance / Presence / Count (argmax right) into non-ATSPM at >= 30 min;
    then a nested grid of relaxations (stack_loser rule, unconfident-winner tau, lane-confidence gate)."""
    import cand64 as C
    import atspm_score as S
    E = setup()
    fr, C7 = E["fr"], E["C7"]
    if a.base == "blend":
        P = C.W_TREE * E["Pt"] + (1 - C.W_TREE) * E["Pn"]
    else:
        P = np.mean([np.load(f) for f in sorted(OUT.glob(f"stack_{a.base}_s*.npy"))], 0).astype(float)
    lc = pd.read_parquet(C.OUT / "function_rows.parquet", columns=["DeviceId", "win", "lane_conf"])
    assert (lc.DeviceId.to_numpy() == fr.DeviceId.to_numpy()).all()
    lconf = lc.lane_conf.to_numpy(float)
    g30 = fr.wgroup.isin(GE30).to_numpy()
    t = fr.truth_v3s.to_numpy(object)
    res = {"base": a.base}
    for i, c in enumerate(C7):
        fr[f"P_{c}"] = P[:, i]
    ref = S.decode(fr, "lanes5g", "greedy", "strict", pick=True)
    pr, au = decode_x(fr, P, audit=True)
    assert (pr == ref).all(), int((pr != ref).sum())
    am = P.argmax(1)
    C7a = np.array(C7, object)
    for s, r in E["rows"].items():
        m = np.zeros(len(fr), bool)
        m[r] = True
        m &= g30 & np.isin(t, ATS)
        dem = m & (C7a[am] == t) & ~np.isin(C7a[pr], ATS)
        x = au[au.row.isin(np.flatnonzero(dem))].drop_duplicates("row").copy()
        x["winner_truth"] = [t[j] if isinstance(t[j], str) else "unlabelled" for j in x.winner_row]
        x["same"] = x.winner_truth.to_numpy() == t[x.row.to_numpy()]
        x["wconf"] = pd.cut(x.p_winner, [0, .5, .7, .9, 1.0001]).astype(str)
        x["mech"] = np.where(x.rule == "stack_loser_nonatspm", "stack loser -> non-ATSPM (56b)",
                             np.where(x.span_extended, "lane conflict via span extension (pick b)",
                                      np.where(x.n_lanes_self > 1, "lane conflict, loser spans >1 lane",
                                               "lane conflict, one lane, next free class non-ATSPM")))
        wrong_nonA = m & ~np.isin(C7a[pr], ATS)
        res[f"audit_{s[0].upper()}"] = {
            "true_atspm_rows_ge30": int(m.sum()), "called_nonatspm": int(wrong_nonA.sum()),
            "called_nonatspm_argmax_right": int(dem.sum()),
            "share_argmax_right": round(float(dem.sum() / max(wrong_nonA.sum(), 1)), 3),
            "demoted_pt_of_true_atspm": round(100 * dem.sum() / m.sum(), 3), "audited": int(len(x)),
            "by_class": vc(pd.Series(t[x.row.to_numpy()])), "by_mech": vc(x.mech),
            "winner": vc(x.same.map({True: "winner has the same truth (lane / stack error)",
                                     False: "winner truth differs"})),
            "winner_truth": vc(x.winner_truth), "winner_in_stack": vc(x.winner_in_stack.map(
                {True: "stack peer", False: "lane-mate only"})),
            "winner_conf": vc(x.wconf), "mech_x_same": vc(x.mech + " | same=" + x.same.astype(str), 10),
            "class_x_mech": vc(pd.Series(t[x.row.to_numpy()]) + " | " + x.mech.to_numpy(), 12)}
        log(f"audit {s}: {json.dumps(res[f'audit_{s[0].upper()}'])}")
    json.dump(res, open(OUT / f"demote_{a.base}.json", "w"), indent=1, default=str)
    combos = [(sl, tau, g) for sl in ("nonatspm_ap", "next_free") for tau in a.taus for g in a.gates]
    ok = {}
    for cb in combos:
        _, cr = run_decode_x(E, P, stack_loser=cb[0], tau=cb[1], gate=cb[2], lconf=lconf)
        ok[cb] = ok_cols(E, cr)
        log(f"{cb}: E ge30 {np.nanmean(ok[cb]['E'][g30]):.4f} R ge30 {np.nanmean(ok[cb]['R'][g30]):.4f}")
    base = ok[("nonatspm_ap", 0.0, 0.0)]
    res["grid_E_R_ge30"] = {str(k): [round(float(np.nanmean(v["E"][g30])), 5), round(float(np.nanmean(v["R"][g30])), 5)]
                            for k, v in ok.items()}
    fo = fr.fold.to_numpy()
    sig = fr.DeviceId.to_numpy()
    fams = {"stack_loser": [c for c in combos if c[1] == 0 and c[2] == 0],
            "tau": [c for c in combos if c[0] == "nonatspm_ap" and c[2] == 0],
            "gate": [c for c in combos if c[0] == "nonatspm_ap" and c[1] == 0],
            "tau+gate": [c for c in combos if c[0] == "nonatspm_ap"], "all": combos}
    for fam, grid in fams.items():
        if len(grid) < 2:
            continue
        nest = {s: np.full(len(fr), np.nan) for s in "ER"}
        chosen = []
        for k in range(6):
            inner = (fo != k) & g30 & ~np.isnan(base["E"])
            best = max(grid, key=lambda g: (round(float(np.nanmean(ok[g]["E"][inner])), 6),
                                            -(g[0] != "nonatspm_ap") - g[1] - g[2]))
            chosen.append(best)
            for s in "ER":
                nest[s][fo == k] = ok[best][s][fo == k]
        r = {"chosen_per_fold": chosen}
        compare(E, base, nest, r, f"{fam}_nested_vs_shipped_decode")
        for s in "ER":
            m = ~np.isnan(base[s]) & g30
            r[f"per_class_{s}_ge30"] = {"shipped": per_class(E, base, s, m), "fixed": per_class(E, nest, s, m)}
            for c in ATS + ["nonATSPM"]:
                mm = m & ((~np.isin(t, ATS)) if c == "nonATSPM" else (t == c))
                r[f"per_class_{s}_ge30"][f"d_{c}"] = C.delta_ci(base[s][mm], nest[s][mm], sig[mm])
        res[fam] = r
        np.save(OUT / f"ok_demote_{a.base}_{fam}_E.npy", nest["E"])
        np.save(OUT / f"ok_demote_{a.base}_{fam}_R.npy", nest["R"])
        json.dump(res, open(OUT / f"demote_{a.base}.json", "w"), indent=1, default=str)
    log(f"-> {OUT / f'demote_{a.base}.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["base", "diag", "stack", "fix", "demote"])
    ap.add_argument("--taus", type=lambda s: [float(x) for x in s.split(",")], default=[0.0, 0.5, 0.7, 0.9])
    ap.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=[0, 1, 2])
    ap.add_argument("--arms", default="")
    ap.add_argument("--grid", type=lambda s: [float(x) for x in s.split(",")], default=[1.0, 1.15, 1.3, 1.5])
    ap.add_argument("--gates", type=lambda s: [float(x) for x in s.split(",")], default=[0.0, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--tag", default="gw")
    ap.add_argument("--base", default="blend")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    globals()[f"stage_{a.stage}"](a)
