"""Note 63: honest blend weight / stacker for the TCN function head + trees, error analysis (fold 0 only).

Same harness as tcn53_feval.py (note-59 scoring rows, stack-aware ATSPM score, D-lane stack-pick decode,
everything / realistic, signal-bootstrap CI vs trees). Everything is NESTED inside fold 0: the 91 fold-0
signals are split into 5 signal groups (seeded); for each outer group the blend weight (or the stacker) is
chosen / fitted on the other 4 groups only and applied to the outer group. Decode groups and stack groups
never cross a window, so per-row assembly from precomputed per-weight decodes is exact.

    python tcn63_blend.py run --cands fj,ff          # weights, per-window weights, stackers -> tcn53/blend63_*.json
    python tcn63_blend.py err --cand fj               # error analysis rows -> tcn53/err63_*.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: F401,E402
from common import DC_WORK  # noqa: E402
import s59_step6 as S59  # noqa: E402
import atspm_score as S  # noqa: E402

ROOT = DC_WORK / "tcn53"
C7 = S59.C7
PC = [f"P_{c}" for c in C7]
WG = S59.WG
WMIN = {"m5": 5, "m10": 10, "m30": 30, "h1": 60, "h3": 180, "h6": 360, "h24": 1440, "full": 3960}
WGRID = [round(w, 2) for w in np.arange(0.30, 0.901, 0.05)]
NOUT = 5
THREADS = 6


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------------------------------------ set-up
def fold0():
    fr, rows = S59.scoring_frame(S59.PICK_TAG)
    Pt, _ = S59.spec_probs("base", fr)
    f0 = fr.fold.to_numpy() == 0
    pos = np.flatnonzero(f0)
    fr0 = fr.iloc[pos].reset_index(drop=True).copy()
    rows0 = {s: np.searchsorted(pos, r[f0[r]]) for s, r in rows.items()}
    assert not fr0.DeviceId.str.lower().isin(S59.locked()).any()
    return fr0, rows0, Pt[pos]


def net_probs(fr0, nm):
    f = ROOT / "fpreds" / f"{nm}_f0.parquet"
    if not f.exists():
        return None
    n = pd.read_parquet(f).astype({"Detector": fr0.Detector.dtype})
    Pn = fr0[S59.KEY].merge(n, on=S59.KEY, how="left")[PC].to_numpy(float)
    return Pn


def score(fr0, rows0, P):
    for i, c in enumerate(C7):
        fr0[f"P_{c}"] = P[:, i]
    pred = S.decode(fr0, "lanes5g", "greedy", "strict", pick=True)
    return {s: S.credit(fr0, pred, "truth_v3s", r, True) for s, r in rows0.items()}


def outer_groups(fr0, seed=63):
    sig = np.array(sorted(fr0.DeviceId.unique()))
    rng = np.random.default_rng(seed)
    g = np.empty(len(sig), int)
    g[rng.permutation(len(sig))] = np.arange(len(sig)) % NOUT
    return dict(zip(sig, g))


def cmp(b0, d1):
    r = {"atspm": round(float(d1.ok_a.mean()), 4), "d_pt": round(100 * (d1.ok_a.mean() - b0.ok_a.mean()), 3),
         "ci": S.boot(b0, d1), "by_window": {}}
    for nm, gs in (("ge30", WG[2:]), ("short", WG[:2])):
        x0, x1 = b0[b0.wgroup.isin(gs)], d1[d1.wgroup.isin(gs)]
        r[nm] = [round(float(x0.ok_a.mean()), 4), round(float(x1.ok_a.mean()), 4),
                 round(100 * (x1.ok_a.mean() - x0.ok_a.mean()), 2)] + S.boot(x0, x1)
    for g in WG:
        x0, x1 = b0[b0.wgroup == g], d1[d1.wgroup == g]
        r["by_window"][g] = [round(float(x1.ok_a.mean()), 4), round(100 * (x1.ok_a.mean() - x0.ok_a.mean()), 2)] \
            + S.boot(x0, x1)
    return r


def ent(P):
    Q = np.clip(P, 1e-9, 1)
    return -(Q * np.log(Q)).sum(1)


def margin(P):
    s = np.sort(P, 1)
    return s[:, -1] - s[:, -2]


def stack_X(fr0, Pt, Pn):
    lm = np.log(fr0.wgroup.map(WMIN).to_numpy(float))
    lt, ln = np.log(np.clip(Pt, 1e-6, 1)), np.log(np.clip(Pn, 1e-6, 1))
    return np.column_stack([lt, ln, lm, ent(Pt), ent(Pn), margin(Pt), margin(Pn),
                            (Pt.argmax(1) == Pn.argmax(1)).astype(float)])


def ctx_X(fr0, Pt, Pn):
    """hi-res-only context: health (note 59 c), pick / stack inputs, D lanes, lane-mates' and phase-mates' probs."""
    H = S59.health_feats(fr0[S59.KEY].reset_index(drop=True)).to_numpy(np.float32)
    Pm = 0.5 * (Pt + Pn)
    n = len(fr0)
    lanes = fr0.lanes5g.to_numpy(object)
    nl_self = np.array([len(s.split(",")) if isinstance(s, str) and s else 0 for s in lanes], float)
    npeer = lambda col: np.array([len(s.split(",")) if isinstance(s, str) and s else 0
                                  for s in fr0[col].to_numpy(object)], float)
    grp = fr0.groupby(["DeviceId", "period", "win", "pred_phase"], dropna=False).ngroup().to_numpy()
    ph = pd.DataFrame({"g": grp, "i": np.arange(n)})
    n_ph = ph.groupby("g").i.transform("size").to_numpy(float)
    # phase-mates: max of each ATSPM class over the OTHER detectors of the predicted phase
    phm = np.full((n, 4), np.nan)
    for c in range(4):
        s = pd.Series(Pm[:, c])
        top1 = s.groupby(grp).transform("max").to_numpy()
        top2 = s.groupby(grp).transform(lambda v: np.sort(v.to_numpy())[-2] if len(v) > 1 else np.nan).to_numpy()
        phm[:, c] = np.where(Pm[:, c] >= top1, top2, top1)
    # rank of own prob within phase (1 = top) per ATSPM class
    rk = np.column_stack([pd.Series(Pm[:, c]).groupby(grp).rank(ascending=False).to_numpy() for c in range(4)])
    # phase lane count (distinct lane ids) and lane-mates' class maxima
    e = pd.DataFrame({"g": grp, "i": np.arange(n), "lane": [s.split(",") if isinstance(s, str) and s else []
                                                             for s in lanes]}).explode("lane").dropna(subset=["lane"])
    nl_ph = e.groupby("g").lane.nunique()
    n_lanes_phase = pd.Series(grp).map(nl_ph).fillna(0).to_numpy(float)
    m = e.merge(e, on=["g", "lane"], suffixes=("", "_m"))
    m = m[m.i != m.i_m].drop_duplicates(["i", "i_m"])
    lm = np.full((n, 5), np.nan)
    if len(m):
        for c in range(4):
            m[f"p{c}"] = Pm[m.i_m.to_numpy(int), c]
        agg = m.groupby("i").agg(nm=("i_m", "size"), p0=("p0", "max"), p1=("p1", "max"), p2=("p2", "max"),
                                 p3=("p3", "max"))
        lm[agg.index.to_numpy(int)] = agg[["nm", "p0", "p1", "p2", "p3"]].to_numpy(float)
    has_lane = (nl_self > 0).astype(float)
    lm[:, 0] = np.where(has_lane > 0, np.nan_to_num(lm[:, 0]), np.nan)
    pk = np.column_stack([fr0.pk_span.to_numpy(float), fr0.pk_unhealthy.to_numpy(float), fr0.pk_track.to_numpy(float),
                          npeer("pk_span_peers"), npeer("pk_coloc_peers")])
    ln_on = np.log1p(fr0.det_n_on.to_numpy(float))
    return np.column_stack([H, pk, nl_self, has_lane, n_lanes_phase, n_ph, phm, rk, lm, ln_on]).astype(float)


def fit_stack(kind, X, y, Xte):
    if kind == "lgb":
        import lightgbm as lgb
        prm = dict(objective="multiclass", num_class=len(C7), learning_rate=0.05, num_leaves=7, max_depth=3,
                   min_data_in_leaf=300, lambda_l2=10.0, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                   num_threads=THREADS, verbose=-1, seed=63)
        m = lgb.train(prm, lgb.Dataset(X, y), num_boost_round=150)
        return m.predict(Xte)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=0.05, max_iter=2000)
    m.fit(sc.transform(X), y)
    P = np.zeros((len(Xte), len(C7)))
    P[:, m.classes_] = m.predict_proba(sc.transform(Xte))
    return P


# ------------------------------------------------------------------------------------------------ run
def stage_run(a):
    fr0, rows0, Pt = fold0()
    sig = fr0.DeviceId.to_numpy()
    base = score(fr0, rows0, Pt)
    log("trees fold 0: " + " ".join(f"{s} {d.ok_a.mean():.4f} n {len(d)}" for s, d in base.items()))
    res = {"trees": {s: round(float(d.ok_a.mean()), 4) for s, d in base.items()}, "grid": WGRID}
    truth = fr0.truth_v3s.to_numpy(object)
    ytr = np.array([C7.index(t) if t in C7 else -1 for t in truth])
    for nm in a.cands.split(","):
        Pn = net_probs(fr0, nm)
        if Pn is None:
            log(f"{nm}: missing"); continue
        have = ~np.isnan(Pn).any(1)
        Pn = np.where(have[:, None], Pn, Pt)
        r = {"coverage": round(float(have.mean()), 4)}
        # ---- per-weight decodes (w = tree weight)
        D = {w: score(fr0, rows0, w * Pt + (1 - w) * Pn) for w in WGRID + [0.0]}
        r["full_fold_by_w_E"] = {str(w): round(float(D[w]["everything"].ok_a.mean()), 4) for w in D}
        r["full_fold_by_w_R"] = {str(w): round(float(D[w]["realistic"].ok_a.mean()), 4) for w in D}
        r["net_alone"] = {s: cmp(base[s], D[0.0][s]) for s in base}
        r["w0.5"] = {s: cmp(base[s], D[0.5][s]) for s in base}
        log(f"{nm}: full-fold E by w " + " ".join(f"{w}:{v}" for w, v in r["full_fold_by_w_E"].items()))
        if not a.no_ctx:
            C, Ct = ctx_X(fr0, Pt, Pn), ctx_X(fr0, Pt, Pt)
            log(f"context block {C.shape}")
        for pseed in a.pseeds:
            gmap = outer_groups(fr0, pseed)
            nested, nested_wg, chosen, chosen_wg = {}, {}, [], []
            for s in base:
                dg = base[s].DeviceId.map(gmap).to_numpy()
                parts, parts_wg = [], []
                for g in range(NOUT):
                    inn = dg != g
                    # choose on 'everything' rows of inner signals; the same weight is applied to both sets
                    dE = base["everything"].DeviceId.map(gmap).to_numpy() != g
                    accw = {w: D[w]["everything"].ok_a.to_numpy()[dE].mean() for w in WGRID}
                    wb = max(WGRID, key=lambda w: (round(accw[w], 6), -abs(w - 0.5)))
                    parts.append(D[wb][s][~inn])
                    if s == "everything":
                        chosen.append(wb)
                    wgE = D[WGRID[0]]["everything"].wgroup.to_numpy()
                    for wgp in WG:
                        mE = dE & (wgE == wgp)
                        acc = {w: D[w]["everything"].ok_a.to_numpy()[mE].mean() for w in WGRID}
                        wbg = max(WGRID, key=lambda w: (round(acc[w], 6), -abs(w - 0.5)))
                        d = D[wbg][s]
                        parts_wg.append(d[(~inn) & (d.wgroup.to_numpy() == wgp)])
                        if s == "everything":
                            chosen_wg.append((g, wgp, wbg))
                nested[s] = pd.concat(parts).sort_index()
                nested_wg[s] = pd.concat(parts_wg).sort_index()
                assert len(nested[s]) == len(base[s]) == len(nested_wg[s])
            r[f"nested_w_p{pseed}"] = {"chosen": chosen, **{s: cmp(base[s], nested[s]) for s in base}}
            cw = pd.DataFrame(chosen_wg, columns=["g", "wg", "w"]).groupby("wg").w.apply(list).to_dict()
            r[f"nested_wwin_p{pseed}"] = {"chosen": cw, **{s: cmp(base[s], nested_wg[s]) for s in base}}
            log(f"{nm} p{pseed} nested fixed w {chosen}: E {r[f'nested_w_p{pseed}']['everything']['d_pt']:+.2f} "
                f"{r[f'nested_w_p{pseed}']['everything']['ci']} R {r[f'nested_w_p{pseed}']['realistic']['d_pt']:+.2f}"
                f" {r[f'nested_w_p{pseed}']['realistic']['ci']}")
            log(f"{nm} p{pseed} nested per-window w: E {r[f'nested_wwin_p{pseed}']['everything']['d_pt']:+.2f} "
                f"{r[f'nested_wwin_p{pseed}']['everything']['ci']} R "
                f"{r[f'nested_wwin_p{pseed}']['realistic']['d_pt']:+.2f} {r[f'nested_wwin_p{pseed}']['realistic']['ci']}"
                f" | {cw}")
            # ---- stackers (fixed shallow hyper-parameters, chosen up front; no tuning)
            X = stack_X(fr0, Pt, Pn)
            fg = pd.Series(sig).map(gmap).to_numpy()
            for kind in ("lgb", "logit"):
                Ps = Pt.copy()
                for g in range(NOUT):
                    tr = (fg != g) & (ytr >= 0) & ~fr0.exclude_train_score.fillna(False).to_numpy(bool)
                    Ps[fg == g] = fit_stack(kind, X[tr], ytr[tr], X[fg == g])
                Ds = score(fr0, rows0, Ps)
                r[f"stack_{kind}_p{pseed}"] = {s: cmp(base[s], Ds[s]) for s in base}
                log(f"{nm} p{pseed} stack {kind}: E {r[f'stack_{kind}_p{pseed}']['everything']['d_pt']:+.2f} "
                    f"{r[f'stack_{kind}_p{pseed}']['everything']['ci']} R "
                    f"{r[f'stack_{kind}_p{pseed}']['realistic']['d_pt']:+.2f} {r[f'stack_{kind}_p{pseed}']['realistic']['ci']}")
            # ---- context stacker (user idea): probs + health + lanes + lane-/phase-mates; shuffled-context control;
            #      and context with the trees' probs only (is the gain the net or the context?)
            if not a.no_ctx:
                rng = np.random.default_rng(11)
                Cs = C.copy()
                wgv = fr0.wgroup.to_numpy()
                for wgp in WG:
                    ix = np.flatnonzero(wgv == wgp)
                    Cs[ix] = C[rng.permutation(ix)]
                Xt = stack_X(fr0, Pt, Pt)[:, [*range(7), 14, 15, 17]]     # trees probs, minutes, entropy, margin
                for tag, XX in (("lgbctx", np.hstack([X, C])), ("lgbctx_shufctx", np.hstack([X, Cs])),
                                ("lgbctx_treesonly", np.hstack([Xt, Ct])), ("lgb_treesonly", Xt)):
                    if pseed != a.pseeds[0] and tag != "lgbctx":
                        continue
                    Ps = Pt.copy()
                    for g in range(NOUT):
                        tr = (fg != g) & (ytr >= 0) & ~fr0.exclude_train_score.fillna(False).to_numpy(bool)
                        Ps[fg == g] = fit_stack("lgb", XX[tr], ytr[tr], XX[fg == g])
                    Ds = score(fr0, rows0, Ps)
                    r[f"stack_{tag}_p{pseed}"] = {s: cmp(base[s], Ds[s]) for s in base}
                    log(f"{nm} p{pseed} stack {tag}: E {r[f'stack_{tag}_p{pseed}']['everything']['d_pt']:+.2f} "
                        f"{r[f'stack_{tag}_p{pseed}']['everything']['ci']} R "
                        f"{r[f'stack_{tag}_p{pseed}']['realistic']['d_pt']:+.2f} "
                        f"{r[f'stack_{tag}_p{pseed}']['realistic']['ci']}")
            # ---- control: stacker with the net's probabilities row-shuffled within window group
            if pseed == a.pseeds[0]:
                rng = np.random.default_rng(7)
                Pc = Pn.copy()
                wgv = fr0.wgroup.to_numpy()
                for wgp in WG:
                    ix = np.flatnonzero(wgv == wgp)
                    Pc[ix] = Pn[rng.permutation(ix)]
                Xc = stack_X(fr0, Pt, Pc)
                Ps = Pt.copy()
                for g in range(NOUT):
                    tr = (fg != g) & (ytr >= 0) & ~fr0.exclude_train_score.fillna(False).to_numpy(bool)
                    Ps[fg == g] = fit_stack("lgb", Xc[tr], ytr[tr], Xc[fg == g])
                Ds = score(fr0, rows0, Ps)
                r["stack_lgb_shufnet_control"] = {s: cmp(base[s], Ds[s]) for s in base}
                log(f"{nm} control stack lgb shuffled net: E {r['stack_lgb_shufnet_control']['everything']['d_pt']:+.2f}"
                    f" {r['stack_lgb_shufnet_control']['everything']['ci']}")
            res[nm] = r
            json.dump(res, open(ROOT / f"blend63_{a.out}.json", "w"), indent=1, default=str)
    json.dump(res, open(ROOT / f"blend63_{a.out}.json", "w"), indent=1, default=str)
    log(f"-> {ROOT / f'blend63_{a.out}.json'}")


# ------------------------------------------------------------------------------------------------ errors
TRAITS = ["px_pulse_frac", "n_on_per_cycle", "det_dur_med", "det_occ_frac"]


def stage_err(a):
    fr0, rows0, Pt = fold0()
    Pn = net_probs(fr0, a.cand)
    w = a.w
    base = score(fr0, rows0, Pt)
    bl = score(fr0, rows0, w * Pt + (1 - w) * Pn)
    nt = score(fr0, rows0, Pn)
    lab = pd.read_parquet(S.V3S)
    lab["DeviceId"] = lab.DeviceId.str.lower()
    lab = lab.rename(columns={"detector": "Detector"}).astype({"Detector": fr0.Detector.dtype})
    tech = fr0[["DeviceId", "Detector"]].merge(lab[["DeviceId", "Detector", "technology"]].drop_duplicates(
        ["DeviceId", "Detector"]), on=["DeviceId", "Detector"], how="left").technology.fillna("?").to_numpy(object)
    F = pd.read_parquet(S59.V.FEATS, columns=S59.KEY + TRAITS, filters=[("DeviceId", "in", list(fr0.DeviceId.unique()))])
    F = F.astype({"Detector": fr0.Detector.dtype}).drop_duplicates(S59.KEY)
    T = fr0[S59.KEY].merge(F, on=S59.KEY, how="left")
    s = "everything"
    r = rows0[s]
    d = pd.DataFrame({"sig": base[s].DeviceId, "wg": base[s].wgroup, "t": base[s].t, "p_tree": base[s].p,
                      "p_blend": bl[s].p, "p_net": nt[s].p, "ok_tree": base[s].ok_a, "ok_blend": bl[s].ok_a,
                      "ok_net": nt[s].ok_a, "tech": tech[r]})
    d["tech"] = d.tech.str.lower().map(lambda x: "radar" if "radar" in x else "loop" if "loop" in x
                                       else "video" if ("video" in x or "camera" in x) else "other/?")
    pf = T.px_pulse_frac.to_numpy()[r]
    d["pulse"] = np.where(np.isnan(pf), "?", np.where(pf >= 0.8, "pulse", np.where(pf <= 0.2, "normal", "mixed")))
    v = T.n_on_per_cycle.to_numpy()[r]
    q = np.nanquantile(v, [1 / 3, 2 / 3])
    d["vol"] = np.where(np.isnan(v), "?", np.where(v < q[0], "low", np.where(v < q[1], "mid", "high")))
    d["short"] = np.where(d.wg.isin(["m5", "m10"]), "5-10 min", np.where(d.wg.isin(["m30", "h1"]), "30-60 min", ">=3 h"))
    d.to_parquet(ROOT / f"err63_{a.cand}_rows.parquet", index=False)
    n = len(d)
    out = {"w": w, "n": n, "vol_tercile_cuts_on_per_cycle": [round(float(x), 2) for x in q]}
    for arm in ("blend", "net"):
        fix = (~d.ok_tree) & d[f"ok_{arm}"]
        brk = d.ok_tree & ~d[f"ok_{arm}"]
        o = {"fixed": int(fix.sum()), "broken": int(brk.sum()), "net_pt": round(100 * (fix.sum() - brk.sum()) / n, 3)}
        o["fixed_pairs"] = (d[fix].t + "->" + d[fix].p_tree).value_counts().head(12).to_dict()
        o["broken_pairs"] = (d[brk].t + "->" + d[brk][f"p_{arm}"]).value_counts().head(12).to_dict()
        for c in ("tech", "pulse", "vol", "short", "wg"):
            g = d.groupby(c)
            o[f"by_{c}"] = {k: {"n": int(len(x)), "fixed": int(fix[x.index].sum()), "broken": int(brk[x.index].sum()),
                                "net_pt_in_group": round(100 * (fix[x.index].sum() - brk[x.index].sum()) / len(x), 2)}
                            for k, x in g}
        # by true class x tech for the fixed / broken
        o["fixed_true_tech"] = {"|".join(k): int(v) for k, v in d[fix].groupby(["t", "tech"]).size().sort_values(ascending=False).head(10).items()}
        o["broken_true_tech"] = {"|".join(k): int(v) for k, v in d[brk].groupby(["t", "tech"]).size().sort_values(ascending=False).head(10).items()}
        o["fixed_signals_share_top5"] = round(float(d[fix].sig.value_counts().head(5).sum() / max(fix.sum(), 1)), 3)
        o["broken_signals_share_top5"] = round(float(d[brk].sig.value_counts().head(5).sum() / max(brk.sum(), 1)), 3)
        out[arm] = o
        log(f"{arm}: fixed {o['fixed']} broken {o['broken']} net {o['net_pt']:+.2f} pt")
        log(f"  fixed pairs {o['fixed_pairs']}")
        log(f"  broken pairs {o['broken_pairs']}")
        for c in ("tech", "pulse", "vol", "short"):
            log(f"  by {c}: {o[f'by_{c}']}")
    json.dump({str(k): v for k, v in out.items()}, open(ROOT / f"err63_{a.cand}.json", "w"), indent=1,
              default=lambda x: str(x))
    log(f"-> {ROOT / f'err63_{a.cand}.json'}")


def stage_ctl(a):
    """control: blend the trees with another TREE arm (note-57 drop arms, seed 0) - is the gain just ensembling?"""
    fr, _ = S59.scoring_frame(S59.PICK_TAG)
    pos = np.flatnonzero(fr.fold.to_numpy() == 0)
    fr0, rows0, Pt = fold0()
    base = score(fr0, rows0, Pt)
    k = S59.frame_keys()
    k["_i"] = np.arange(len(k))
    k["Detector"] = k.Detector.astype(fr.Detector.dtype)
    idx = fr[S59.KEY].merge(k[S59.KEY + ["_i"]], on=S59.KEY, how="left")._i.to_numpy().astype(int)
    out = {}
    for arm in ("drop_px", "drop_sib", "drop_cond"):
        P = S59.std_probs(k, [0], S59.FUNC_DIR.parent / arm)[idx][pos]
        for w in (0.5, 0.6, 0.0):
            D = score(fr0, rows0, w * Pt + (1 - w) * P)
            out[f"{arm}@{round(w, 2)}"] = {s: cmp(base[s], D[s]) for s in base}
            log(f"{arm} w {w:.2f}: E {out[f'{arm}@{round(w, 2)}']['everything']['d_pt']:+.2f} "
                f"{out[f'{arm}@{round(w, 2)}']['everything']['ci']}")
    json.dump(out, open(ROOT / "blend63_treectl.json", "w"), indent=1, default=str)


def stage_new62(a):
    """the blend on top of the note-62 reference (route:shortw4 probs on <= 10 min + cy twin decode thr .4), nested w."""
    import s62_short as S62
    S59.OUT = S62.OUT
    fr, rows = S59.scoring_frame(S59.PICK_TAG)
    fr = fr.copy()
    P62, seeds = S62.spec_probs("route:shortw4", fr)
    pairs = pd.read_parquet(S62.PAIRS_F)
    short = fr.wgroup.isin(S62.SHORT).to_numpy() & (fr.det_n_on >= 5).to_numpy()
    tw = S62.twin_tokens(fr, pairs, 0.4, 1.0, 0)
    f0 = fr.fold.to_numpy() == 0
    rows0 = {s: r[f0[r]] for s, r in rows.items()}

    def ev(P):
        for i, c in enumerate(C7):
            fr[f"P_{c}"] = P[:, i]
        pr = S.decode(fr, "lanes5g", "greedy", "strict", pick=True)
        pr = S62.twin_decode(P, pr, tw, short, ("Count", "Yellow_Red"), "cy")
        return {s: S.credit(fr, pr, "truth_v3s", r, True) for s, r in rows0.items()}
    base = ev(P62)
    res = {"ref62_seeds": seeds, "ref62": {s: round(float(d.ok_a.mean()), 4) for s, d in base.items()}}
    log(f"note-62 reference fold 0: {res['ref62']}")
    for nm in a.cands.split(","):
        f = ROOT / "fpreds" / f"{nm}_f0.parquet"
        if not f.exists():
            continue
        n = pd.read_parquet(f).astype({"Detector": fr.Detector.dtype})
        Pn = fr[S59.KEY].merge(n, on=S59.KEY, how="left")[PC].to_numpy(float)
        Pn = np.where(np.isnan(Pn).any(1)[:, None], P62, Pn)
        D = {}
        for w in WGRID:
            D[w] = ev(np.where(f0[:, None], w * P62 + (1 - w) * Pn, P62))
        r = {"full_fold_by_w_E": {str(w): round(float(D[w]["everything"].ok_a.mean()), 4) for w in WGRID},
             "w0.6": {s: cmp(base[s], D[0.6][s]) for s in base}}
        sig0 = np.array(sorted(set(fr.DeviceId[f0])))
        for pseed in a.pseeds:
            rng = np.random.default_rng(pseed)
            g = np.empty(len(sig0), int)
            g[rng.permutation(len(sig0))] = np.arange(len(sig0)) % NOUT
            gmap = dict(zip(sig0, g))
            nested, chosen = {}, []
            dE = base["everything"].DeviceId.map(gmap).to_numpy()
            for s in base:
                dg = base[s].DeviceId.map(gmap).to_numpy()
                parts = []
                for k in range(NOUT):
                    wb = max(WGRID, key=lambda w: (round(D[w]["everything"].ok_a.to_numpy()[dE != k].mean(), 6),
                                                   -abs(w - 0.5)))
                    parts.append(D[wb][s][dg == k])
                    if s == "everything":
                        chosen.append(wb)
                nested[s] = pd.concat(parts).sort_index()
            r[f"nested_w_p{pseed}"] = {"chosen": chosen, **{s: cmp(base[s], nested[s]) for s in base}}
            e, rr = r[f"nested_w_p{pseed}"]["everything"], r[f"nested_w_p{pseed}"]["realistic"]
            log(f"{nm} vs note-62 ref p{pseed} nested w {chosen}: E {e['d_pt']:+.2f} {e['ci']} R {rr['d_pt']:+.2f} {rr['ci']}")
            log("   by window E: " + " ".join(f"{k} {v[1]:+.2f}" for k, v in e["by_window"].items()))
        res[nm] = r
        json.dump(res, open(ROOT / "blend63_on62.json", "w"), indent=1, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "err", "ctl", "new62"])
    ap.add_argument("--cands", default="fj")
    ap.add_argument("--cand", default="fj")
    ap.add_argument("--w", type=float, default=0.5)
    ap.add_argument("--pseeds", type=lambda s: [int(x) for x in s.split(",")], default=[63, 64, 65])
    ap.add_argument("--out", default="f0")
    ap.add_argument("--no_ctx", action="store_true")
    a = ap.parse_args()
    {"run": stage_run, "err": stage_err, "ctl": stage_ctl, "new62": stage_new62}[a.stage](a)


if __name__ == "__main__":
    main()
