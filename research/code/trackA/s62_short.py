"""Note 62: function accuracy on SHORT samples (5 / 10 min), where no lanes exist and the per-lane decode does not run.

Baseline = note 59 (229-feature function arm, 3 seeds, D lanes, stack-pick decode gated >= 30 min): .8914 E / .9042 R;
5 min .8607, 10 min .8732. Screen = config seed 0 vs base seed 0 (decode-only variants also on the 3-seed probabilities).

    python s62_short.py pairs                 # co-actuation of every pair on the same predicted phase, 5 / 10-min windows
    python s62_short.py twin  --spec base@s0  # short-window twin decode variants (no refit)
    python s62_short.py gate  --spec base@s0  # per-lane decode below 30 min with the ln5 lanes (the old pair model)
    python s62_short.py fit --cfg shortw2 --seeds 0      # training-mix variants (refit)
    python s62_short.py score --cfgs shortw2@s0 --base base@s0 --tag screen
Out: %DC_WORK%/s62/. Hi-res only (event times); no technology / print fact is read by any decision rule. locked_v2
asserted absent. CPU only (<= 6 threads).
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: F401,E402
import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import s59_step6 as B  # noqa: E402
import v3_retrain as V  # noqa: E402

DCW = V.DCW
OUT = DCW / "s62"
C7 = list(V.C7)
KEY = V.KEY
log = B.log
SHORT = ("m5", "m10")
PAIRS_F = OUT / "pairs_short.parquet"
MIN_N = 3


# ------------------------------------------------------------------------------------------------ pair co-actuation
def pair_work(args):
    """per (window, predicted phase) group: chance-corrected match share both ways at the pair's best lag (+-2 s, ln6)."""
    dev, period, g = args
    import a2_features as A2F
    import pyarrow.dataset as ds
    import ln6_pick as L6
    tab = ds.dataset(str(L6.CACHE[period])).to_table(filter=ds.field("DeviceId") == dev,
                                                     columns=["Detector", "t_on", "dur"]).to_pandas().drop_duplicates()
    tab["t"] = tab.t_on.astype("datetime64[us]").astype("int64").to_numpy() / 1e6
    tab = tab.sort_values(["Detector", "t"])
    allon = {int(d): (x["t"].to_numpy(), x.dur.fillna(0).to_numpy()) for d, x in tab.groupby("Detector")}
    rows = []
    for name, t0w, secs in A2F.WINDOWS[period]:
        w = g[g.win == name]
        if w.empty:
            continue
        t0 = pd.Timestamp(t0w).value / 1e9
        t1 = t0 + secs
        for ph, gg in w.groupby("pred_phase"):
            dets = [int(d) for d in gg.Detector]
            on, du = {}, {}
            for d in dets:
                x, y = allon.get(d, (np.zeros(0), np.zeros(0)))
                m = (x >= t0) & (x < t1)
                on[d], du[d] = x[m], y[m]
            big = L6.order_dets(on, [d for d in dets if len(on[d]) >= MIN_N], t0)   # note 77: not channel order
            for i, a in enumerate(big):
                for b in big[i + 1:]:
                    mab, lab_ = L6.match_frac(on[a], on[b], secs)
                    mba, _ = L6.match_frac(on[b], on[a], secs)
                    rows.append((dev, period, name, float(ph), a, b, len(on[a]), len(on[b]), mab, mba, lab_,
                                 float(np.median(du[a])) if len(du[a]) else np.nan,
                                 float(np.median(du[b])) if len(du[b]) else np.nan))
    return pd.DataFrame(rows, columns=["DeviceId", "period", "win", "pred_phase", "da", "db", "na", "nb", "m_ab", "m_ba",
                                       "lag", "dur_a", "dur_b"])


def stage_pairs(a):
    from multiprocessing import Pool
    k = B.frame_keys()
    k = k[k.pred_phase.notna() & k.wgroup.isin(a.wgroups.split(","))]
    jobs = [(dev, per, g[["Detector", "win", "pred_phase"]].copy()) for (dev, per), g in k.groupby(["DeviceId", "period"])]
    log(f"{len(jobs)} signal-periods, {len(k):,} det-windows")
    parts, t0 = [], time.time()
    with Pool(a.workers) as pool:
        for i, f in enumerate(pool.imap_unordered(pair_work, jobs, chunksize=2)):
            parts.append(f)
            if i % 200 == 0:
                log(f"  {i}/{len(jobs)} ({time.time()-t0:.0f}s)")
    F = pd.concat(parts, ignore_index=True)
    assert not F.DeviceId.isin(V.locked_signals()).any()
    out = PAIRS_F if a.wgroups == "m5,m10" else OUT / f"pairs_{a.wgroups.replace(',', '_')}.parquet"
    F.to_parquet(out, index=False)
    log(f"wrote {len(F):,} pairs -> {out} ({time.time()-t0:.0f}s)")


# ------------------------------------------------------------------------------------------------ twin decode
def twin_tokens(fr: pd.DataFrame, pairs: pd.DataFrame, thr: float, lagmax: float, vr: float) -> dict:
    """frame row index -> set of twin partners' row indices (same group, both-way match >= thr, |lag| <= lagmax,
    volume ratio within vr)."""
    q = pairs[(np.minimum(pairs.m_ab, pairs.m_ba) >= thr) & (pairs.lag.abs() <= lagmax)]
    if vr:
        r = np.log(q.na / q.nb).abs()
        q = q[r <= np.log(vr)]
    assert (fr.index == np.arange(len(fr))).all()
    idx = fr[KEY].reset_index(drop=False).rename(columns={"index": "_i"})
    ia = q.rename(columns={"da": "Detector"})[["DeviceId", "period", "win", "Detector", "db"]]
    ia = ia.astype({"Detector": fr.Detector.dtype}).merge(idx, on=KEY, how="inner")
    ia = ia.rename(columns={"_i": "ia", "Detector": "da", "db": "Detector"}).astype({"Detector": fr.Detector.dtype})
    ia = ia.merge(idx, on=KEY, how="inner").rename(columns={"_i": "ib"})
    tw = {}
    for x, y in zip(ia.ia.to_numpy(), ia.ib.to_numpy()):
        tw.setdefault(int(x), set()).add(int(y))
        tw.setdefault(int(y), set()).add(int(x))
    return tw


def twin_decode(P: np.ndarray, pred: np.ndarray, tw: dict, rows_ok: np.ndarray, cls: tuple, mode: str) -> np.ndarray:
    """within twins, at most one detector per class in `cls`; greedy over (detector, class) pairs by probability.
    mode 'next' = the loser takes its next class; 'yr' = a Count loser becomes Yellow_Red only if YR is its next class
    (else keeps its argmax); 'joint' = pairs: the swap maximising the product of the two probabilities."""
    out = pred.copy()
    ci = [C7.index(c) for c in cls]
    seen = set()
    comps = []
    for i in tw:
        if i in seen or not rows_ok[i]:
            continue
        st, comp = [i], []
        seen.add(i)
        while st:
            j = st.pop()
            comp.append(j)
            for k in tw[j]:
                if k not in seen and rows_ok[k]:
                    seen.add(k)
                    st.append(k)
        if len(comp) >= 2:
            comps.append(comp)
    for comp in comps:
        comp = np.array(sorted(comp))
        pc = pred[comp]
        if not any(np.sum(pc == c) >= 2 and any(int(b) in tw[int(a)] for a in comp[pc == c] for b in comp[pc == c])
                   for c in ci):
            continue
        Pc = P[comp]
        if mode == "joint" and len(comp) == 2 and pc[0] == pc[1] and pc[0] == C7.index("Count"):
            c, y = C7.index("Count"), C7.index("Yellow_Red")
            a, b = Pc[0], Pc[1]
            # options: (C, next of b) or (next of a, C), with 'next' = best class other than Count
            nb = np.argsort(-b)[1]
            na = np.argsort(-a)[1]
            if a[c] * b[nb] >= a[na] * b[c]:
                out[comp[1]] = nb
            else:
                out[comp[0]] = na
            continue
        from atspm_decode import _pair_order          # note 77: ties by probability content, not row order
        order = _pair_order(Pc)
        o = np.full(len(comp), -1)
        held = {}
        for li, c in order:
            if o[li] >= 0:
                continue
            if c not in ci:
                o[li] = c
                continue
            blk = any(int(comp[m]) in tw[int(comp[li])] and o[m] == c for m in range(len(comp)))
            if not blk:
                o[li] = c
                continue
            if mode == "cy":                              # Count <-> Yellow_Red only; otherwise keep the argmax
                cc, yy = C7.index("Count"), C7.index("Yellow_Red")
                alt = yy if c == cc else cc if c == yy else None
                if alt is not None and not any(int(comp[m]) in tw[int(comp[li])] and o[m] == alt for m in range(len(comp))):
                    o[li] = alt
                else:
                    o[li] = c
                continue
            if mode == "yr":
                nxt = np.argsort(-Pc[li])
                nxt = nxt[nxt != c][0]
                o[li] = C7.index("Yellow_Red") if (c == C7.index("Count") and nxt == C7.index("Yellow_Red")) else c
        out[comp] = o
    return out


def base_pred(spec: str):
    import atspm_score as S
    fr, rows = B.scoring_frame(B.PICK_TAG)
    P, seeds = B.spec_probs(spec, fr)
    for i, c in enumerate(C7):
        fr[f"P_{c}"] = P[:, i]
    pred = S.decode(fr, "lanes5g", "greedy", "strict", pick=True)
    return fr, rows, P, pred


def creds(fr, pred, rows):
    import atspm_score as S
    return {s: S.credit(fr, pred, "truth_v3s", r, True) for s, r in rows.items()}


WG = B.WG


def compare(name, D0, D1, res):
    import atspm_score as S
    r = {}
    for sname in ("everything", "realistic"):
        b0, b1 = D0[sname], D1[sname]
        r[sname] = {"atspm": round(float(b1.ok_a.mean()), 4),
                    "d_pt": round(100 * (b1.ok_a.mean() - b0.ok_a.mean()), 3), "ci": S.boot(b0, b1)}
        for nm, gs in (("short", list(SHORT)), ("m5", ["m5"]), ("m10", ["m10"]), ("ge30", WG[2:])):
            x0, x1 = b0[b0.wgroup.isin(gs)], b1[b1.wgroup.isin(gs)]
            r[sname][nm] = [round(float(x1.ok_a.mean()), 4), round(100 * (x1.ok_a.mean() - x0.ok_a.mean()), 3)] + S.boot(x0, x1)
        yc = {("Yellow_Red", "Count"), ("Count", "Yellow_Red")}
        e0 = pd.Series(list(zip(b0.t, b0.p))).isin(yc).mean()
        e1 = pd.Series(list(zip(b1.t, b1.p))).isin(yc).mean()
        r[sname]["yrc_err_pt"] = [round(100 * e0, 3), round(100 * e1, 3)]
    res[name] = r
    e = r["everything"]
    log(f"{name:40s} E {e['atspm']:.4f} d {e['d_pt']:+.3f} {e['ci']} | short {e['short'][1]:+.3f} [{e['short'][2]:+.2f},"
        f"{e['short'][3]:+.2f}] m5 {e['m5'][1]:+.3f} m10 {e['m10'][1]:+.3f} | YR<->C err {e['yrc_err_pt']} | "
        f"R d {r['realistic']['d_pt']:+.3f} {r['realistic']['ci']}")


def stage_twin(a):
    fr, rows, P, pred0 = base_pred(a.spec)
    D0 = creds(fr, pred0, rows)
    pairs = pd.read_parquet(PAIRS_F)
    short = fr.wgroup.isin(SHORT).to_numpy() & (fr.det_n_on >= 5).to_numpy()
    res = {"spec": a.spec, "base": {s: round(float(d.ok_a.mean()), 4) for s, d in D0.items()}}
    log(f"base {a.spec}: {res['base']}")
    grid = []
    for thr in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        grid.append((f"C next thr{thr}", thr, 1.0, 0, ("Count",), "next"))
    for thr in (0.2, 0.3, 0.4, 0.5, 0.6):
        grid.append((f"all4 next thr{thr}", thr, 1.0, 0, ("Advance", "Presence", "Count", "Yellow_Red"), "next"))
        grid.append((f"C+YR next thr{thr}", thr, 1.0, 0, ("Count", "Yellow_Red"), "next"))
    grid += [("C yr thr0.7", 0.7, 1.0, 0, ("Count",), "yr"),
             ("C joint thr0.7", 0.7, 1.0, 0, ("Count",), "joint"),
             ("C next thr0.7 lag2", 0.7, 2.0, 0, ("Count",), "next"),
             ("C next thr0.7 vr1.5", 0.7, 1.0, 1.5, ("Count",), "next"),
             ("C+P next thr0.7", 0.7, 1.0, 0, ("Count", "Presence"), "next"),
             ("all4 next thr0.7", 0.7, 1.0, 0, ("Advance", "Presence", "Count", "Yellow_Red"), "next")]
    if a.grid:
        grid = [g for g in grid if g[0] in a.grid.split(";")]
    cache = {}
    preds = {}
    for name, thr, lag, vr, cls, mode in grid:
        k = (thr, lag, vr)
        if k not in cache:
            cache[k] = twin_tokens(fr, pairs, thr, lag, vr)
        pr = twin_decode(P, pred0, cache[k], short, cls, mode)
        log(f"{name}: {int((pr != pred0).sum()):,} rows changed")
        preds[name] = pr
        compare(name, D0, creds(fr, pr, rows), res)
    # nested threshold choice (no OOF tuning): per fold k, the threshold that maximises ATSPM E on the other five folds
    fo = fr.fold.to_numpy()
    rE = rows["everything"]
    fr_e = fo[rE]
    for fam in ("C next thr", "all4 next thr", "C+YR next thr"):
        thrs = [g for g in grid if g[0].startswith(fam) and "lag" not in g[0] and "vr" not in g[0]]
        if len(thrs) < 2:
            continue
        okm = {g[0]: creds(fr, preds[g[0]], {"e": rE})["e"].ok_a.to_numpy() for g in thrs}
        nest = pred0.copy()
        pick = {}
        for kf in range(6):
            best = max(thrs, key=lambda g: okm[g[0]][fr_e != kf].mean())
            pick[kf] = best[0]
            m = fo == kf
            nest[m] = preds[best[0]][m]
        res[f"nested_pick|{fam}"] = pick
        log(f"nested pick {fam}: {pick}")
        compare(f"{fam} nested", D0, creds(fr, nest, rows), res)
    # control: the same constraint on NON-twin pairs (min match < 0.2) of detectors with the same argmax ATSPM class in
    # the same predicted-phase group, as many pairs as there are same-class twin pairs at thr 0.4 -> does co-actuation matter?
    if not a.grid:
        rng = np.random.default_rng(62)
        tw = cache[(0.4, 1.0, 0)]
        lo = cache[(0.2, 1.0, 0)]
        n_pairs = sum(1 for i in tw for j in tw[i] if i < j and pred0[i] == pred0[j] and C7[pred0[i]] in
                      ("Advance", "Presence", "Count", "Yellow_Red") and short[i] and short[j])
        g = (fr.DeviceId + "|" + fr.period + "|" + fr.win + "|" + fr.pred_phase.astype(str)).to_numpy()
        cand = pd.DataFrame({"i": np.flatnonzero(short), "g": g[short], "c": pred0[short]})
        cand = cand[cand.c.isin([0, 1, 2, 3])]
        allp = []
        for _, x in cand.groupby(["g", "c"]).i:
            v = x.to_numpy()
            for u in range(len(v)):
                for w in range(u + 1, len(v)):
                    if int(v[w]) not in lo.get(int(v[u]), set()):
                        allp.append((int(v[u]), int(v[w])))
        sel = rng.choice(len(allp), min(n_pairs, len(allp)), replace=False)
        twr = {}
        for s_ in sel:
            x, y = allp[s_]
            twr.setdefault(x, set()).add(y)
            twr.setdefault(y, set()).add(x)
        log(f"control: {len(sel)} non-twin same-class pairs of {len(allp)} candidates")
        pr = twin_decode(P, pred0, twr, short, ("Advance", "Presence", "Count", "Yellow_Red"), "next")
        log(f"control non-twin pairs: {int((pr != pred0).sum()):,} rows changed")
        compare("control: non-twin same-class pairs, all4", D0, creds(fr, pr, rows), res)
    json.dump(res, open(OUT / f"twin_{a.tag}.json", "w"), indent=1, default=str)


def stage_twinlong(a):
    """the twin rule on >= 30-min rows the lane decode leaves unconstrained (no lane on the predicted phase), after the
    lane decode; optionally together with the short-window rule (--thr = short threshold, C+YR)."""
    fr, rows, P, pred0 = base_pred(a.spec)
    D0 = creds(fr, pred0, rows)
    short = fr.wgroup.isin(SHORT).to_numpy() & (fr.det_n_on >= 5).to_numpy()
    nolane = ~fr.wgroup.isin(SHORT).to_numpy() & (fr.det_n_on >= 5).to_numpy() & fr.lanes5g.isna().to_numpy()
    log(f"long rows without lane: {nolane.sum():,} of {(~fr.wgroup.isin(SHORT)).sum():,}")
    pl = pd.read_parquet(OUT / "pairs_m30_h1_h3_h6_h24_full.parquet")
    cy = ("Count", "Yellow_Red")
    res = {}
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
        tw = twin_tokens(fr, pl, thr, 1.0, 0)
        for nm, cls in (("C+YR", cy), ("all4", ("Advance", "Presence", "Count", "Yellow_Red"))):
            pr = twin_decode(P, pred0, tw, nolane, cls, "next")
            log(f"long {nm} thr{thr}: {int((pr != pred0).sum()):,} rows changed")
            compare(f"long no-lane {nm} thr{thr}", D0, creds(fr, pr, rows), res)
    json.dump(res, open(OUT / f"twinlong_{a.tag}.json", "w"), indent=1, default=str)


def stage_bias(a):
    """item 3: at 5 / 10 min the head over-calls ATSPM (nonA->A 4.0 pt vs 2.4 at >= 30 min, A->nonA 2.6 vs 3.5).
    Multiply the non-ATSPM probabilities by alpha on short rows before argmax; alpha picked per fold on the other five
    folds (nested). Optionally followed by the twin decode (--twin thr)."""
    fr, rows, P, pred0 = base_pred(a.spec)
    D0 = creds(fr, pred0, rows)
    short = fr.wgroup.isin(SHORT).to_numpy()
    nonA = [C7.index(c) for c in ("Mid", "Bike", "Other")]
    res = {}
    preds = {}
    tw = twin_tokens(fr, pd.read_parquet(PAIRS_F), a.thr, 1.0, 0) if a.thr else None
    ok5 = short & (fr.det_n_on >= 5).to_numpy()
    for al in (0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0):
        Q = P.copy()
        Q[np.ix_(short, nonA)] *= al
        pr = pred0.copy()
        pr[short] = Q[short].argmax(1)
        if tw is not None:
            pr = twin_decode(Q, pr, tw, ok5, ("Count",), "next")
        preds[al] = pr
        compare(f"nonA x{al}" + (f" + twin{a.thr}" if tw else ""), D0, creds(fr, pr, rows), res)
    fo = fr.fold.to_numpy()
    rE = rows["everything"]
    okm = {al: creds(fr, preds[al], {"e": rE})["e"].ok_a.to_numpy() for al in preds}
    fe = fo[rE]
    se = short[rE]
    nest, pick = pred0.copy(), {}
    for kf in range(6):
        best = max(preds, key=lambda al: okm[al][(fe != kf) & se].mean())
        pick[kf] = best
        m = fo == kf
        nest[m] = preds[best][m]
    res["nested_pick"] = pick
    log(f"nested pick {pick}")
    compare("nonA bias, nested alpha" + (f" + twin{a.thr}" if tw else ""), D0, creds(fr, nest, rows), res)
    json.dump(res, open(OUT / f"bias_{a.tag}.json", "w"), indent=1, default=str)


def stage_gate(a):
    """per-lane decode on 5 / 10-min windows with the ln5 lanes (D lanes do not exist there)."""
    import atspm_score as S
    fr, rows, P, pred0 = base_pred(a.spec)
    D0 = creds(fr, pred0, rows)
    res = {}
    L5 = pd.read_parquet(DCW / "lanes" / "ln5_lanes_v3s.parquet", columns=["DeviceId", "Detector", "period", "win", "phase", "lanes"])
    L5 = L5[L5.lanes != ""].astype({"Detector": fr.Detector.dtype})
    x = fr[KEY].merge(L5, on=KEY, how="left")
    assert len(x) == len(fr)
    l5 = x.lanes.where(x.phase.eq(fr.pred_phase.to_numpy()), None).to_numpy()
    for nm, gs in (("m10", ["m10"]), ("m5+m10", ["m5", "m10"])):
        col = fr.lanes5g.to_numpy(object).copy()
        m = fr.wgroup.isin(gs).to_numpy()
        col[m] = l5[m]
        fr["_lg"] = col
        pr = S.decode(fr, "_lg", "greedy", "strict", pick=True)
        compare(f"ln5 lanes on {nm}", D0, creds(fr, pr, rows), res)
    json.dump(res, open(OUT / f"gate_{a.tag}.json", "w"), indent=1, default=str)


# ------------------------------------------------------------------------------------------------ training mix
def stage_fit(a):
    """note-57 229-feature arm with a different training mix.
    shortw<k>  short-window (5 / 10 min) rows weighted k;
    shorthead  trained on 5 / 10-min rows only (predictions used for 5 / 10-min rows, base elsewhere at scoring);
    shorthead30 trained on 5 / 10 / 30-min rows."""
    fr, cols, lab, y, ok = B.load_train()
    cfg = a.cfg
    X = fr[cols].to_numpy(np.float32)
    yi = pd.Series(y).map({c: i for i, c in enumerate(C7)}).fillna(-1).astype(int).to_numpy()
    fo = fr.fold.to_numpy()
    wg = fr.wgroup.to_numpy()
    w = np.ones(len(fr))
    tr_sub = np.ones(len(fr), bool)
    if cfg.startswith("shortw"):
        w[np.isin(wg, SHORT)] = float(cfg[6:])
    elif cfg == "shorthead":
        tr_sub = np.isin(wg, SHORT)
    elif cfg == "shorthead30":
        tr_sub = np.isin(wg, SHORT + ("m30",))
    else:
        raise SystemExit(cfg)
    log(f"{cfg}: training rows {int((ok & tr_sub).sum()):,} of {int(ok.sum()):,}; short share {np.isin(wg[ok], SHORT).mean():.3f}")
    d = OUT / "fit" / cfg
    d.mkdir(parents=True, exist_ok=True)
    json.dump({"cfg": cfg, "classes": C7, "n_cols": len(cols)}, open(d / "cols.json", "w"))
    for s in [int(v) for v in a.seeds.split(",")]:
        for k in range(6):
            f = d / f"P_{B.VAR}_s{s}_f{k}.npy"
            if f.exists():
                continue
            t0 = time.time()
            inner = (k + 1) % 6
            P, nt = B.lgb_fit(X, yi, w, ok & tr_sub & (fo != k) & (fo != inner), ok & tr_sub & (fo == inner), fo == k, s,
                              a.threads, 7)
            np.save(f, P)
            log(f"{cfg} s{s} f{k}: {nt} trees, {time.time()-t0:.0f}s")


def spec_probs(spec: str, fr: pd.DataFrame):
    """cfg[@sN]; 'route:<cfg>' = <cfg> probabilities on 5 / 10-min rows, base (same seeds) elsewhere."""
    parts = spec.split("@")
    sd = "@".join(parts[1:])
    if parts[0].startswith("route:"):
        Pb, sb = B.spec_probs("base" + ("@" + sd if sd else ""), fr)
        Ps, ss = B.spec_probs(parts[0][6:] + ("@" + sd if sd else ""), fr)
        m = fr.wgroup.isin(SHORT).to_numpy()
        Pb[m] = Ps[m]
        return Pb, ss
    return B.spec_probs(spec, fr)


def stage_combo(a):
    """spec[+twin<thr>] vs base: probabilities (route: allowed) -> lane decode (>= 30 min) -> optional short-window twin
    decode on Count + Yellow_Red (`+twin0.4`)."""
    import atspm_score as S
    B.OUT = OUT
    fr, rows = B.scoring_frame(B.PICK_TAG)
    pairs = pd.read_parquet(PAIRS_F)
    short = fr.wgroup.isin(SHORT).to_numpy() & (fr.det_n_on >= 5).to_numpy()
    res = {}
    tws = {}

    def ev(spec):
        mode = "cy" if "+cy" in spec else "next"
        sp, _, thr = spec.replace("+cy", "+twin").partition("+twin")
        P, _ = spec_probs(sp, fr)
        for i, c in enumerate(C7):
            fr[f"P_{c}"] = P[:, i]
        pr = S.decode(fr, "lanes5g", "greedy", "strict", pick=True)
        if thr:
            if thr not in tws:
                tws[thr] = twin_tokens(fr, pairs, float(thr), 1.0, 0)
            pr = twin_decode(P, pr, tws[thr], short, ("Count", "Yellow_Red"), mode)
        return creds(fr, pr, rows)
    D0 = ev(a.base)
    for spec in [s for s in a.cfgs.split(",") if s]:
        compare(f"{spec} vs {a.base}", D0, ev(spec), res)
    json.dump(res, open(OUT / f"combo_{a.tag}.json", "w"), indent=1, default=str)


def stage_score(a):
    import atspm_score as S
    B.OUT = OUT
    fr, rows = B.scoring_frame(B.PICK_TAG)
    res = {}

    def ev(spec):
        P, _ = spec_probs(spec, fr)
        for i, c in enumerate(C7):
            fr[f"P_{c}"] = P[:, i]
        return creds(fr, S.decode(fr, "lanes5g", "greedy", "strict", pick=True), rows)
    D0 = ev(a.base)
    for spec in [s for s in a.cfgs.split(",") if s]:
        compare(spec, D0, ev(spec), res)
    json.dump(res, open(OUT / f"score_{a.tag}.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--wgroups", default="m5,m10")
    ap.add_argument("--spec", default="base@s0")
    ap.add_argument("--grid", default="")
    ap.add_argument("--thr", type=float, default=0.0)
    ap.add_argument("--cfg", default="")
    ap.add_argument("--cfgs", default="")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--base", default="base@s0")
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    globals()[f"stage_{a.stage}"](a)
