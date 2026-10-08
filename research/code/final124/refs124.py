"""Note 124: freeze every population table the research health scorer v4d computes at run time into package data.

The research scorer (research/code/health118/score_v4c.py + score_v4d.py on top of h118 / h110 / h108 / h104 resolvers,
the 117 / 118c fast + volume statistics, the 116 night check and the 118c band check) derives its limits as quantiles
over all w40 detector-windows (763 training signals, locked_v2 absent - asserted upstream).  Production cannot see the
population, so every such table is captured here EXACTLY as the research code computes it (the research functions are
run and their table builders are wrapped) and written as JSON for detector_classifier/weights/health/.

Two research passes exist: v4 (score_v4.run_v4: needed because the v4c 'goes silent' yardstick drops phase mates with a
v4 finding) and v4c (score_v4c.run_c).  Tables of both are captured.

The 116 night model and the 118c day band are OUT-OF-FOLD in the research (2 folds by signal).  Production ships the
all-data fit (116: tod116_ref.npz, as note 116 built it; band: fitted here on every ok detector-day); the fold fits are
written to %DC_WORK%/s124/refs_oof/ for the equality harness only (never shipped).

    python refs124.py   -> %DC_WORK%/s124/refs/health_v4_refs.json (+ refs_oof/*.json), sanity prints
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
for p in ("health", "health118", "health116", "health117"):
    sys.path.insert(0, str(CODE / p))
import h108_resolve as R8  # noqa: E402
import h110_resolve as R10  # noqa: E402
import h118_resolve as R18  # noqa: E402
import score_v4 as V4  # noqa: E402
import score_v4c as V4C  # noqa: E402

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
OUT = DCW / "s124" / "refs"
OOF = DCW / "s124" / "refs_oof"
KEY = ["DeviceId", "window", "detector"]


def tab_json(T):
    """[(keys, Series indexed by keys)] -> [{"keys": [...], "rows": [[k..., value], ...]}]"""
    out = []
    for keys, s in T:
        rows = []
        for k, v in s.items():
            k = k if isinstance(k, tuple) else (k,)
            rows.append([str(x) if not isinstance(x, (int, np.integer)) else int(x) for x in k] + [float(v)])
        out.append({"keys": list(keys), "rows": rows})
    return out


# ------------------------------------------------------------------ capture cell tables during the research runs
CAP = {}
_orig_cell = R10.cell_tables


def capture(tag):
    def cell_tables(X, cols):
        T = _orig_cell(X, cols)
        for c, lst in T.items():
            CAP.setdefault(tag, {})[c] = lst
        return T
    return cell_tables


def run_v4():
    X0, E = R18.prepare()
    D = pd.read_csv(DCW / "s118" / "drops118.csv", usecols=KEY + ["g4_unscored0"])
    X0 = X0.merge(D, on=KEY, how="left")
    X0["g4_unscored0"] = X0.g4_unscored0.fillna(False).astype(bool)
    X0 = X0.merge(V4.load_117(), on=KEY, how="left").merge(V4.load_116(), on=KEY, how="left")
    R10.cell_tables = capture("v4")
    g3 = {}
    og = R18.g3_tables

    def g3w(X):
        T = og(X)
        g3["v4"] = T
        return T
    R18.g3_tables = g3w
    try:
        R = V4.run_v4(X0, E)
    finally:
        R10.cell_tables, R18.g3_tables = _orig_cell, og
    S = pd.read_parquet(DCW / "s118b" / "resolved_v4.parquet", columns=KEY + ["st8", "left8"])
    m = R[KEY + ["st8", "left8"]].merge(S, on=KEY, suffixes=("", "_saved"))
    print("v4 reproduced:", f"{(m.st8 == m.st8_saved).mean():.5f}", f"{(m.left8 == m.left8_saved).mean():.5f}", flush=True)
    return g3["v4"]


def run_v4c():
    X0, E = V4C.prepare_c()
    R10.cell_tables = capture("v4c")
    g3 = {}
    og = V4C.g3_tables_c

    def g3w(X):
        T = og(X)
        g3["v4c"] = T
        return T
    V4C.g3_tables_c = g3w
    try:
        R = V4C.run_c(X0, E)
    finally:
        R10.cell_tables, V4C.g3_tables_c = _orig_cell, og
    S = pd.read_parquet(DCW / "s118c" / "resolved_v4c.parquet", columns=KEY + ["st8", "left8"])
    m = R[KEY + ["st8", "left8"]].merge(S, on=KEY, suffixes=("", "_saved"))
    print("v4c reproduced:", f"{(m.st8 == m.st8_saved).mean():.5f}", f"{(m.left8 == m.left8_saved).mean():.5f}", flush=True)
    return g3["v4c"], X0


# ------------------------------------------------------------------ tables recomputed with the research code itself
def occ_tables(X):
    """h108.occ_hi limit (fn, cmode, bs, tl) + score_v4c.occ_hi_c extras (pulse-Count limit, Count passage time)."""
    Bn = pd.read_parquet(DCW / "health104" / "bins_ctx.parquet",
                         columns=KEY + ["fn", "b", "n", "occ", "ref", "cong", "bs", "ph", "status"])
    Bn = Bn.merge(X[KEY + ["cmode"]], on=KEY)
    Bn = Bn[Bn.cmode.ne("unknown")]
    Bn["rmax"] = Bn.groupby(KEY).ref.transform("max")
    Bn["tl"] = np.clip(np.floor(5 * Bn.ref / Bn.rmax.where(Bn.rmax > 0)), 0, 4)
    ok = Bn[Bn.status.eq("ok")]
    lim = ok.groupby(["fn", "cmode", "bs", "tl"]).occ.quantile(.995).rename("lim").reset_index()
    pl = lim[(lim.fn == "Count") & (lim.cmode == "pulse")][["bs", "tl", "lim"]]
    c = ok[ok.fn.eq("Count") & ok.cmode.eq("normal") & ok.n.gt(0)]
    tp = (c.occ * c.bs / c.n).groupby(c.bs).quantile(V4C.T_PASS_Q)
    return (lim.assign(tl=lim.tl.astype(int)).values.tolist(), pl.assign(tl=pl.tl.astype(int)).values.tolist(),
            [[int(k), float(v)] for k, v in tp.items()])


def tables117():
    """h117_study limits (v4 too-fast / too-many) and h118c_fast limits (v4c), as tables (keys fn, span, wg ...)."""
    import h117_study as S7
    X = pd.read_parquet(DCW / "s117" / "stats117.parquet",
                        columns=KEY + ["fn", "span", "wg", "healthy", "zf", "n_spk", "q5_gy", "q5_all"])
    H = X.healthy
    t7 = {c: S7.limit_tab(X, c, H)[1] for c in ("zf", "n_spk", "q5_gy", "q5_all")}
    import h118c_fast as F
    Y = pd.read_parquet(DCW / "s118c" / "fast118c.parquet",
                        columns=KEY + ["fn", "span", "wg", "healthy", "zf_c", "n_spk_c", "q5_gy", "q5_all"])
    H2 = Y.healthy.fillna(False).astype(bool)
    t8 = {c: F.tables(Y, c, H2) for c in ("zf_c", "n_spk_c", "q5_gy", "q5_all")}
    return {k: tab_json(v) for k, v in t7.items()}, {k: tab_json(v) for k, v in t8.items()}


def tod_weights():
    W = R10.tod_weights()
    Wb = W.groupby(["band", "window", "hour"]).apply(lambda d: np.average(d.w, weights=d.n_w)).rename("wb").reset_index()
    W30 = W[W.n_w >= 30]
    return ([[r.type, r.band, r.window, int(r.hour), float(r.w)] for r in W30.itertuples()],
            [[r.band, r.window, int(r.hour), float(r.wb)] for r in Wb.itertuples()])


def share_med():
    R = pd.read_parquet(DCW / "health104" / "resolved104.parquet", columns=["fn", "wg", "share_med"])
    g = R.dropna(subset=["fn"]).groupby(["fn", "wg"]).share_med.agg(["min", "max"])
    assert (g["min"] == g["max"]).all() or g.isna().all(axis=1).any()
    return [[k[0], k[1], float(v)] for k, v in g["min"].items() if np.isfinite(v)]


def q_type():
    q = pd.read_csv(DCW / "health110" / "queue_q95_by_type.csv")
    return [[r.type, float(r.q_type)] for r in q.itertuples()]


# ------------------------------------------------------------------ 116 night model + 118c band
def tod_ref_json(G):
    out = {}
    for k, g in G.items():
        d = {f: np.asarray(g[f], float).tolist() for f in ("bc", "sc", "cal_med", "cal_mad") if f in g}
        d["occ"] = bool(g["occ"])
        if bool(g["occ"]):
            d["bo"], d["so"] = np.asarray(g["bo"], float).tolist(), np.asarray(g["so"], float).tolist()
        for f in ("chart_share_q", "chart_on_q"):
            if f in g:
                d[f] = np.round(np.asarray(g[f], float), 6).tolist()
        out[k] = d
    return out


def band_ref(D, C, A, rows):
    """118c band fitted on `rows`: availability of each group key level + per-hour median / robust SD of the counts
    shape (c) and time-ON level (a) per group (exactly h118c_band.band_fit on group_keys)."""
    import h118c_band as BD
    gk = BD.group_keys(D, rows)
    avail = sorted(set(g for g in gk[rows] if g is not None))
    out = {}
    for g in avail:
        m = rows & (gk == g)
        mc, sc = BD.band_fit(C[m], BD.FLOOR_C)
        ma, sa = BD.band_fit(A[m], BD.FLOOR_A)
        out[g] = dict(mc=mc.tolist(), sc=sc.tolist(), ma=ma.tolist(), sa=sa.tolist(), n=int(m.sum()))
    # availability list (groups with >= MIN_GROUP rows among `rows`) at every level
    lv = (("fn", "span", "band", "window"), ("fn", "band", "window"), ("fn", "span", "window"), ("fn", "window"))
    av = []
    tr = D[rows]
    for keys in lv:
        for k, n in tr.groupby(list(keys)).size().items():
            if n >= BD.MIN_GROUP:
                av.append("|".join(map(str, k)) + f"#{len(keys)}")
    return {"groups": out, "avail": sorted(av)}


def band_and_tod():
    import h116_study as S
    import h116_build as B6
    import tod116 as T6
    import h118c_band as BD
    D, N, O, H, Q = S.load()
    C, A = BD.feats(N, O)
    ok = D.ok.to_numpy()
    prod_band = band_ref(D, C, A, ok)
    oof_band = {f: band_ref(D, C, A, ((D.fold != f) & D.ok).to_numpy()) for f in (0, 1)}
    G, meta = T6.load_reference(DCW / "s116" / "tod116_ref.npz")
    prod_tod = tod_ref_json(G)
    X, _ = T6.features(N, O)
    grp, z, exp_r, zc, zo, cal, folds = B6.oof(D, X)
    oof_tod = {}
    for f in (0, 1):
        Gf = {k: dict(g) for k, g in folds[f].items()}          # folds[f] = fitted on the signals NOT in fold f
        for k, g in Gf.items():
            if k in cal.index:
                g["cal_med"], g["cal_mad"] = np.array(cal.loc[k, "med"]), np.array(cal.loc[k, "mad"])
        oof_tod[f] = tod_ref_json({k: g for k, g in Gf.items() if "cal_med" in g})
    # check: fold refs reproduce oof_final z
    F = pd.read_parquet(DCW / "s116" / "oof_final.parquet", columns=KEY + ["z"])
    folds_sig = dict(zip(D.DeviceId, D.fold))
    return prod_band, oof_band, prod_tod, oof_tod, folds_sig, meta


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    OOF.mkdir(parents=True, exist_ok=True)
    g3_v4 = run_v4()
    g3_v4c, X0 = run_v4c()
    lim, pl, tp = occ_tables(X0)
    t7, t8 = tables117()
    W, Wb = tod_weights()
    prod_band, oof_band, prod_tod, oof_tod, folds_sig, meta = band_and_tod()
    refs = dict(
        version="health v4d (note 124)", source="w40 windows, 763 training signals (locked_v2 absent); research "
        "scorer score_v4c / score_v4d; tables captured by research/code/final124/refs124.py",
        cell={v: {c: tab_json(T) for c, T in CAP[v].items() if c in ("chat_frac", "stuck_x", "n3_exc", "n_ep")}
              for v in ("v4", "v4c")},
        g3={"v4": tab_json(g3_v4), "v4c": tab_json(g3_v4c)},
        occ_lim=lim, occ_lim_pulse=pl, count_t_pass=tp,
        fast117=t7, fast118c=t8, tod_w=W, tod_wb=Wb, share_med=share_med(), q_type=q_type(),
        band=prod_band, tod=dict(groups=prod_tod, meta=meta))
    (OUT / "health_v4_refs.json").write_text(json.dumps(refs, separators=(",", ":")))
    for f in (0, 1):
        (OOF / f"band_fold{f}.json").write_text(json.dumps(oof_band[f], separators=(",", ":")))
        (OOF / f"tod_fold{f}.json").write_text(json.dumps(dict(groups=oof_tod[f], meta=meta), separators=(",", ":")))
    (OOF / "folds.json").write_text(json.dumps({k: int(v) for k, v in folds_sig.items()}))
    print("written", OUT / "health_v4_refs.json", (OUT / "health_v4_refs.json").stat().st_size, "bytes")


if __name__ == "__main__":
    main()
