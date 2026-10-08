"""Note 116 study: generalised time-of-day health model - method comparison (research; sklearn allowed here only).

2-fold split BY SIGNAL (references never come from the scored signal). Methods: per-group robust z-RMS, robust
Mahalanobis (shrinkage), kNN distance to the group's trimmed references, one-class SVM, k-means nearest-centroid.
Feature sets: counts+ON (no diffs), counts+ON+diffs, counts only, ON only, 30-min. Day handling: same-day references,
pooled (both days), cross-day (the other day's references). Each method's out-of-fold score is calibrated per group
(median / MAD of log score) and flagged at ONE global threshold giving the target rate (2 per 100 detector-days).

    python h116_study.py compare      -> s116/compare.csv, s116/oof_<cfg>.parquet
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h116_methods as T  # noqa: E402

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
OUT = DCW / "s116"
MIN_GROUP = 150
TARGET = 2.0      # flags per 100 detector-days

# user's cases (signal, det, reviewed window); MUST = time-of-day cases from the Oct 7 feedback
MUST_BAD = [("12032", 6, "h24_b"), ("04035", 52, "h24_b"), ("04035", 53, "h24_b"), ("2B069", 23, "h24_b"),
            ("2B502", 4, "h24_b")]
MUST_OK = [("2B531", 35, "h24_b"), ("2B530", 60, "h24_b"), ("14003", 4, "h24_a")]
OTHER_BAD = [("01074", 4, "h24_b"), ("2B316", 3, "h24_b"), ("08073", 7, "h24_b"), ("08052", 10, "h24_b"),
             ("10028", 16, "h24_a"), ("2B368", 27, "h24_b"), ("2B019", 44, "h24_a"), ("08154", 1, "h24_a"),
             ("2B557", 24, "h24_a"), ("10037", 22, "h24_a"), ("10055", 5, "h24_b"), ("08CM405", 37, "h24_a")]
OTHER_OK = [("2B334", 13, "h24_a"), ("12061", 41, "h24_a"), ("06013", 8, "h24_a"), ("08019", 16, "h24_b"),
            ("2B039", 18, "h24_b"), ("04028", 53, "h24_b"), ("2C028", 37, "h24_b"), ("01062", 2, "h24_b"),
            ("04016", 20, "h24_b"), ("2B068", 19, "h24_b"), ("2B058", 46, "h24_b"), ("01064", 42, "h24_b")]


def load():
    D = pd.read_parquet(OUT / "day.parquet")
    N = D[[f"n{h}" for h in range(24)]].to_numpy(float)
    O = D[[f"o{h}" for h in range(24)]].to_numpy(float)
    H = D[[f"h{h}" for h in range(48)]].to_numpy(float)
    Q = D[[f"q{h}" for h in range(48)]].to_numpy(float)
    D["tot"] = np.nansum(N, 1)
    D["band"] = T.band_of(D.tot / 24)
    D["ok"] = np.isfinite(N).all(1) & np.isfinite(O).all(1) & (D.tot >= T.MIN_N)
    sig = np.array(sorted(D.DeviceId.unique()))
    rng = np.random.default_rng(116)
    fold = dict(zip(sig, rng.permutation(len(sig)) % 2))
    D["fold"] = D.DeviceId.map(fold)
    return D, N, O, H, Q


def featset(name, N, O, H, Q):
    if name == "cnt_on":
        return T.features(N, O, diffs=False)
    if name == "cnt_on_diff":
        return T.features(N, O, diffs=True)
    if name == "cnt_only":
        X = T.features(N, O, diffs=True)
        nm = T.feature_names(diffs=True)
        return X[:, [i for i, c in enumerate(nm) if c.startswith(("s", "ds")) or c == "logv"]]
    if name == "on_only":
        X = T.features(N, O, diffs=True)
        nm = T.feature_names(diffs=True)
        return X[:, [i for i, c in enumerate(nm) if c.startswith(("a", "da")) or c == "logv"]]
    if name == "blk":
        X, nm = T.features_blk(N, O, H, Q)
        COND[name] = [nm.index("o level"), nm.index("logv")]
        return X
    if name == "flat":
        COND[name] = [2, 3]
        return T.features_flat(N, O)
    if name == "night":
        return T.features_night(N, O)
    if name == "night_sqrt":
        X = T.features_night(N, O)
        X[:, 0] = np.sqrt(np.exp(X[:, 0]))
        return X
    if name == "night_raw":
        X = T.features_night(N, O)
        X[:, 0] = np.exp(X[:, 0])
        return X
    if name == "flat_v":
        COND[name] = [3]
        return T.features_flat(N, O)
    if name in ("blk_cnt", "blk_occ"):
        X, nm = T.features_blk(N, O, H, Q)
        pre = "c " if name == "blk_cnt" else "o "
        return X[:, [i for i, c in enumerate(nm) if c.startswith(pre) or c == "logv"]]
    if name == "blk_c":       # shape features only conditioned on volume
        X, nm = T.features_blk(N, O, H, Q)
        COND[name] = [nm.index("logv")]
        return X
    if name == "blk_norough":
        return T.features_blk(N, O, rough=False)[0]
    if name == "blk_hourly":
        return np.hstack([T.features_blk(N, O, H, Q)[0], T.features(N, O, diffs=False)])
    if name == "min30":
        return T.features(np.nan_to_num(H), np.nan_to_num(Q), diffs=True)
    raise KeyError(name)


def assign_groups(D, train):
    """group key per row, using groups with >= MIN_GROUP training rows (fn|span|band, fallbacks)."""
    cnt = {}
    tr = D[train]
    for keys, fmt in ((["fn", "span", "band"], lambda r: T.group_key(r[0], r[1], r[2])),
                      (["fn", "band"], lambda r: T.group_key(r[0], "*", r[1])),
                      (["fn", "span"], lambda r: T.group_key(r[0], r[1], "*")),
                      (["fn"], lambda r: T.group_key(r[0], "*", "*"))):
        for k, n in tr.groupby(keys).size().items():
            k = k if isinstance(k, tuple) else (k,)
            if n >= MIN_GROUP:
                cnt[fmt(k)] = n
    return np.array([T.resolve_group(f, s, b, cnt) for f, s, b in zip(D.fn, D.span, D.band)], object)


CUR = [None]
CURK = [None]
COND = {}   # featset -> conditioner column indices (set in featset)


def fit_score(method, Xtr, Xte, seed=0, cond=None):
    if method.startswith("cnight"):
        # counts: night / busiest 4 h, conditioned on daytime saturation + volume; time ON: absolute night % ON
        m = T.fit_cond(Xtr[:, [0, 2, 3]], [1, 2])
        rc = T.cond_z(Xte[:, [0, 2, 3]], m)[:, 0]
        if method == "cnight_c":
            return rc, False
        ms = float(method.split("_s")[1]) if "_s" in method else 0.05
        cols = [4, 2, 3] if "_od" in method else [4, 3]
        mo = T.fit_cond(Xtr[:, cols], list(range(1, len(cols))), min_scale=ms)
        ro = T.cond_z(Xte[:, cols], mo)[:, 0]
        if method.startswith("cnight_o"):
            return ro, False
        if "_f" in method and CURK[0].split("|")[0] not in ("Presence", "Other", "Mid"):
            return rc, False
        return np.fmax(rc, ro), False
    if method.startswith("cflat"):
        m = T.fit_cond(Xtr[:, [0, 2, 3]], [1, 2])
        rc = T.cond_z(Xte[:, [0, 2, 3]], m)[:, 0]
        if method == "cflat_c":
            return rc, False
        thr = float(method.split("_o")[1]) if "_o" in method else 0.05
        lo = np.arcsin(np.sqrt(thr))
        mo_rows = Xtr[:, 2] >= lo
        ro = np.full(len(Xte), -np.inf)
        if mo_rows.sum() >= 60:
            mo = T.fit_cond(Xtr[mo_rows][:, [1, 2, 3]], [1, 2])
            te = Xte[:, 2] >= lo
            ro[te] = T.cond_z(Xte[te][:, [1, 2, 3]], mo)[:, 0]
        return np.fmax(rc, ro), False
    if method.startswith(("czmax", "zmax", "cmaha")):
        cc = cond if method.startswith("c") else []
        if method.startswith("cmaha"):
            m = T.fit_cond(Xtr, cc)
            Rtr = T.cond_z(Xtr, m)
            g, _ = T.fit_group(Rtr)
            return T.maha(T.cond_z(Xte, m), g), True
        m = T.fit_cond(Xtr, cc)
        top = int(method[-1]) if method[-1].isdigit() else 1
        return T.zmax_score(T.cond_z(Xte, m), top), True
    g, keep = T.fit_group(Xtr)
    if method == "maha":
        return T.maha(Xte, g), True
    if method == "zrms":
        return T.zrms(Xte, g), True
    Ztr = np.clip((Xtr[keep] - g["med"]) / g["scale"], -T.ZCLIP, T.ZCLIP)
    rng = np.random.default_rng(seed)
    if method == "knn":
        R = Ztr[rng.permutation(len(Ztr))[:3000]]
        return T.knn(Xte, R, g, k=10), True
    if method == "kmeans":
        C, lab = T.kmeans(Ztr, k=4, seed=seed)
        Zte = np.clip((Xte - g["med"]) / g["scale"], -T.ZCLIP, T.ZCLIP)
        d = np.sqrt(((Zte[:, None, :] - C[None]) ** 2).sum(2)).min(1) / np.sqrt(Xte.shape[1])
        return d, True
    if method == "ocsvm":
        from sklearn.svm import OneClassSVM
        Zall = np.clip((Xtr - g["med"]) / g["scale"], -T.ZCLIP, T.ZCLIP)
        Zall = Zall[rng.permutation(len(Zall))[:3000]]
        m = OneClassSVM(nu=0.05, gamma="scale").fit(Zall)
        Zte = np.clip((Xte - g["med"]) / g["scale"], -T.ZCLIP, T.ZCLIP)
        return -m.decision_function(Zte), False
    raise KeyError(method)


def run(D, X, method, dayvar):
    """out-of-fold score + per-group calibrated z for every ok row."""
    sc = np.full(len(D), np.nan)
    grp = np.array([None] * len(D), object)
    logscale = True
    for f in (0, 1):
        for day in ("h24_a", "h24_b"):
            te = (D.fold == f) & (D.window == day) & D.ok
            if dayvar == "same":
                tr = (D.fold != f) & (D.window == day) & D.ok
            elif dayvar == "pooled":
                tr = (D.fold != f) & D.ok
            else:   # cross
                tr = (D.fold != f) & (D.window != day) & D.ok
            gtr = assign_groups(D, tr.to_numpy())
            # rows of the training set belong to the same resolution of groups as the test rows
            for k in set(gtr[te.to_numpy()]) - {None}:
                fn, span, band = k.split("|")
                mtr = tr.to_numpy() & (D.fn.to_numpy() == fn)
                if span != "*":
                    mtr &= D.span.to_numpy() == span
                if band != "*":
                    mtr &= D.band.to_numpy() == band
                ite = np.flatnonzero(te.to_numpy() & (gtr == k))
                CURK[0] = k
                s, logscale = fit_score(method, X[mtr], X[ite], cond=COND.get(CUR[0]))
                sc[ite] = s
                grp[ite] = k
    D2 = pd.DataFrame(dict(grp=grp, sc=sc, window=D.window))
    v = np.log(np.clip(sc, 1e-9, None)) if logscale else sc
    D2["v"] = v
    st = D2.dropna(subset=["grp"]).groupby(["grp", "window"]).v.agg(
        med="median", mad=lambda s: 1.4826 * np.median(np.abs(s - np.median(s))))
    m = D2.merge(st, left_on=["grp", "window"], right_index=True, how="left")
    z = ((m.v - m.med) / m.mad.clip(lower=1e-3)).to_numpy()
    return grp, sc, z


def case_mask(D, cases):
    out = []
    for s, d, w in cases:
        i = np.flatnonzero((D.signal == s).to_numpy() & (D.detector == d).to_numpy() & (D.window == w).to_numpy())
        out.append(i[0] if len(i) else -1)
    return np.array(out)


def summarise(D, z, thr, tag):
    flag = z >= thr
    r = dict(cfg=tag)
    ok = np.isfinite(z)
    for nm, cases in (("must_bad", MUST_BAD), ("must_ok", MUST_OK), ("oth_bad", OTHER_BAD), ("oth_ok", OTHER_OK)):
        idx = case_mask(D, cases)
        good = idx >= 0
        r[nm] = f"{int(flag[idx[good]].sum())}/{int(ok[idx[good]].sum())}"
        r[nm + "_z"] = " ".join(f"{z[i]:.1f}" if i >= 0 else "na" for i in idx)
    m69 = ((D.signal == "2B069") & (D.window == "h24_b")).to_numpy() & ok
    r["2B069"] = f"{int(flag[m69].sum())}/{int(m69.sum())}"
    for w, nm in (("h24_a", "rate_sun"), ("h24_b", "rate_mon")):
        mm = (D.window == w).to_numpy() & ok
        r[nm] = round(100 * flag[mm].mean(), 2)
    # day-to-day agreement on detectors scored both days
    A = pd.DataFrame(dict(dev=D.DeviceId, det=D.detector, w=D.window, z=z, f=flag))[ok]
    P = A.pivot_table(index=["dev", "det"], columns="w", values=["z", "f"], aggfunc="first").dropna()
    fa, fb = P[("f", "h24_a")].astype(bool), P[("f", "h24_b")].astype(bool)
    r["jaccard"] = round((fa & fb).sum() / max((fa | fb).sum(), 1), 3)
    r["p_other_day"] = round(0.5 * ((fa & fb).sum() / max(fa.sum(), 1) + (fa & fb).sum() / max(fb.sum(), 1)), 3)
    r["spearman"] = round(P[("z", "h24_a")].rank().corr(P[("z", "h24_b")].rank()), 3)
    r["scored"] = round(ok.mean(), 3)
    return r


def compare(cfgs):
    D, N, O, H, Q = load()
    print("rows", len(D), "ok", D.ok.mean().round(3))
    rows = []
    feats = {}
    for fs, method, dayvar in cfgs:
        t0 = time.time()
        if fs not in feats:
            feats[fs] = featset(fs, N, O, H, Q)
        CUR[0] = fs
        grp, sc, z = run(D, feats[fs], method, dayvar)
        thr = np.nanquantile(z, 1 - TARGET / 100)
        tag = f"{fs}|{method}|{dayvar}"
        r = summarise(D, z, thr, tag)
        r["thr"] = round(thr, 2)
        r["sec"] = round(time.time() - t0, 1)
        rows.append(r)
        print({k: r[k] for k in ("cfg", "must_bad", "must_ok", "oth_bad", "oth_ok", "2B069", "rate_sun", "rate_mon",
                                 "jaccard", "spearman", "sec")}, flush=True)
        pd.DataFrame(dict(DeviceId=D.DeviceId, window=D.window, detector=D.detector, grp=grp, sc=sc, z=z)).to_parquet(
            OUT / f"oof_{tag.replace('|', '_')}.parquet")
    C = pd.DataFrame(rows)
    f = OUT / "compare.csv"
    if f.exists():
        C = pd.concat([pd.read_csv(f), C]).drop_duplicates("cfg", keep="last")
    C.to_csv(f, index=False)
    return C


def combo(a, b, how="min"):
    """combine two out-of-fold runs (e.g. counts-only and time-ON-only): min = both must disagree with the type."""
    D, N, O, H, Q = load()
    key = ["DeviceId", "window", "detector"]
    A = pd.read_parquet(OUT / f"oof_{a.replace('|', '_')}.parquet")
    B = pd.read_parquet(OUT / f"oof_{b.replace('|', '_')}.parquet")
    M = D[key].merge(A, on=key, how="left").merge(B[key + ["z"]], on=key, how="left", suffixes=("", "_b"))
    za, zb = M.z.to_numpy(), M.z_b.to_numpy()
    z = np.fmin(za, zb) if how == "min" else (0.5 * (za + zb) if how == "mean" else np.fmax(za, zb))
    thr = np.nanquantile(z, 1 - TARGET / 100)
    tag = f"{how}({a};{b})"
    r = summarise(D, z, thr, tag)
    r["thr"] = round(thr, 2)
    print({k: r[k] for k in ("cfg", "must_bad", "must_ok", "oth_bad", "oth_ok", "2B069", "rate_sun", "rate_mon",
                             "jaccard", "spearman")}, flush=True)
    pd.DataFrame(dict(DeviceId=D.DeviceId, window=D.window, detector=D.detector, grp=M.grp, za=za, zb=zb,
                      z=z)).to_parquet(OUT / f"oof_{tag.replace('|', '_').replace(';', '__')}.parquet")
    C = pd.concat([pd.read_csv(OUT / "compare.csv"), pd.DataFrame([r])]).drop_duplicates("cfg", keep="last")
    C.to_csv(OUT / "compare.csv", index=False)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "compare"
    if what == "compare":
        cfgs = [("cnt_on_diff", m, "same") for m in ("maha", "zrms", "knn", "kmeans", "ocsvm")]
        cfgs += [("cnt_on_diff", "maha", "pooled"), ("cnt_on_diff", "maha", "cross"),
                 ("cnt_on_diff", "knn", "pooled"), ("cnt_on_diff", "knn", "cross")]
        cfgs += [(fs, "maha", "same") for fs in ("cnt_on", "cnt_only", "on_only", "min30")]
        compare(cfgs)
    elif what == "combo":
        combo(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "min")
    elif what == "one":
        compare([tuple(sys.argv[2].split("|"))])
