"""Note 61: confusions AMONG the ATSPM classes (Advance / Presence / Count / Yellow_Red) on the current set-up.

Baseline = note 59 (229-feature function arm, 3 seeds, D lanes, stack-pick decode, stack_loser nonatspm_ap): .8914 E / .9042 R.
    python s61_atspm_conf.py diag            # scored OOF rows with truth / pred / probs / technology -> s61/rows_base.parquet
    python s61_atspm_conf.py feats           # targeted hi-res features -> s61/feats.parquet
    python s61_atspm_conf.py fit --cfg new --seeds 0
    python s61_atspm_conf.py score --cfgs new@s0,new_shuf@s0 --base base@s0 --tag screen
Out: %DC_WORK%/s61/. locked_v2 asserted absent. CPU only (<= 6 threads). Technology is used for analysis only.
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
OUT = DCW / "s61"
C7 = list(V.C7)
KEY = V.KEY
log = B.log


def stage_diag(a):
    import atspm_score as S
    fr, rows = B.scoring_frame(B.PICK_TAG)
    P, seeds = B.spec_probs(a.base, fr)
    for i, c in enumerate(C7):
        fr[f"P_{c}"] = P[:, i]
    pred = S.decode(fr, "lanes5g", "greedy", "strict", pick=True)
    lab = pd.read_parquet(S.V3S)
    lab["DeviceId"] = lab.DeviceId.str.lower()
    lab = lab.rename(columns={"detector": "Detector"}).astype({"Detector": fr.Detector.dtype})
    keep = [c for c in ("technology", "print_subtype", "n_lanes_phase", "lanes_spanned") if c in lab]
    x = fr[["DeviceId", "Detector"]].merge(lab[["DeviceId", "Detector"] + keep].drop_duplicates(["DeviceId", "Detector"]),
                                           on=["DeviceId", "Detector"], how="left")
    out = {}
    for sname, r in rows.items():
        d = S.credit(fr, pred, "truth_v3s", r, True)
        d["Detector"] = fr.Detector.to_numpy()[r]
        d["pred_phase"] = fr.pred_phase.to_numpy()[r]
        d["det_n_on"] = fr.det_n_on.to_numpy()[r]
        d["fold"] = fr.fold.to_numpy()[r]
        d["argmax"] = np.array(C7, object)[P[r].argmax(1)]
        d["has_lane"] = fr.lanes5g.notna().to_numpy()[r]
        d["lanes"] = fr.lanes5g.fillna("").to_numpy()[r]
        d["pk_span"] = fr.pk_span.to_numpy()[r]
        for c in keep:
            d[c] = x[c].to_numpy()[r]
        for i, c in enumerate(C7):
            d[f"P_{c}"] = P[r, i]
        d["set"] = sname
        out[sname] = d
        log(f"{sname}: ATSPM {d.ok_a.mean():.4f} n {len(d):,}")
    R = pd.concat(out.values(), ignore_index=True)
    assert not R.DeviceId.isin(V.locked_signals()).any()
    R.to_parquet(OUT / f"rows_{a.base.replace('@', '_')}.parquet", index=False)


# ------------------------------------------------------------------------------------------------ targeted features
# tw_*  twin-relative (YR <-> Count): the sibling (same predicted phase) that sees the same vehicles at the stop bar
#       (max chance-corrected share of this detector's ONs with a sibling ON within +-0.5 s). A speed-filtered Yellow_Red
#       zone misses slow vehicles: creeping / arriving ones outside green and the first car(s) of green -> asymmetries.
# rl_*  release / onset (Advance <-> Presence): at green start an occupied stop-bar zone clears within ~2-3 s, an
#       advance zone under a spilled-back queue only when the start-up wave reaches it; in red a stop-bar zone is
#       occupied early (first arriving car), an advance zone only late (spill-back).
TW_COLS = ["tw_m_self", "tw_m_twin", "tw_m_asym", "tw_lvr", "tw_ng_unm_self", "tw_ng_unm_twin", "tw_ng_unm_diff",
           "tw_ngshare_diff", "tw_first_hit", "tw_first_hit_rev", "tw_first_hit_diff", "tw_first_lag_diff", "tw_n05",
           "tw_cover_any", "tw_sum_vr"]
RL_COLS = ["rl_on_gs", "rl_first_off_med", "rl_first_off_q75", "rl_red_onset_med", "rl_first_off_rel",
           "rl_red_onset_rel", "rl_first_off_minus_twin"]
TOL, MIN_N, TW_MIN = 0.5, 5, 0.2
FEAT_F = OUT / "feats.parquet"


def _near(x, y):
    """signed distance from each x to the nearest y (y sorted, non-empty)."""
    i = np.searchsorted(y, x)
    i0, i1 = np.clip(i - 1, 0, len(y) - 1), np.clip(i, 0, len(y) - 1)
    d0, d1 = x - y[i0], x - y[i1]
    return np.where(np.abs(d0) < np.abs(d1), d0, d1)


def _med(v, k=2):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return float(np.median(v)) if len(v) >= k else np.nan


def group_feats(dets, on, off, gs, ge, ng, secs):
    """features of every detector of one (window, predicted phase) group."""
    out = {d: {} for d in dets}
    nc = len(gs)
    prev_ge = np.full(nc, np.nan)
    if nc > 1:
        cont = np.abs(ng[:-1] - gs[1:]) < 1.0
        prev_ge[1:] = np.where(cont, ge[:-1], np.nan)
    first = {}
    for d in dets:
        t, o = on[d], off[d]
        r = out[d]
        n = len(t)
        r["_ng"] = np.zeros(n, bool)
        r["_inc"] = np.zeros(n, bool)
        if nc >= 2 and n >= 1:
            i = np.searchsorted(t, gs, side="right") - 1
            cov = (i >= 0) & (o[np.clip(i, 0, None)] > gs)
            ii = i[cov]
            r["rl_on_gs"] = float(cov.mean())
            if cov.sum() >= 2:
                fo = np.minimum(o[ii], ge[cov]) - gs[cov]
                r["rl_first_off_med"] = float(np.median(fo))
                r["rl_first_off_q75"] = float(np.quantile(fo, 0.75))
                rs = prev_ge[cov]
                ro = np.clip((t[ii] - rs) / np.maximum(gs[cov] - rs, 1.0), 0, 1)
                r["rl_red_onset_med"] = _med(ro)
        if n and nc:
            j = np.searchsorted(t, gs)
            okg = j < n
            okg[okg] &= t[j[okg]] < ge[okg]
            first[d] = (j, okg)
            k = np.searchsorted(gs, t, side="right") - 1
            kk = np.clip(k, 0, None)
            incyc = (k >= 0) & (t < ng[kk])
            r["_ng"] = incyc & (t >= ge[kk])
            r["_inc"] = incyc
    big = [d for d in dets if len(on[d]) >= MIN_N]
    for d in big:
        t, r = on[d], out[d]
        best, bx = None, TW_MIN
        cover = np.zeros(len(t), bool)
        n05, sv = 0, 0.0
        for s in big:
            if s == d:
                continue
            u = on[s]
            hit = np.abs(_near(t, u)) <= TOL
            cover |= hit
            ex = hit.mean() - (1 - np.exp(-len(u) / secs * 2 * TOL))
            hit_r = np.abs(_near(u, t)) <= TOL
            ex_r = hit_r.mean() - (1 - np.exp(-len(t) / secs * 2 * TOL))
            if ex >= 0.5:
                n05 += 1
            if ex_r >= 0.5:
                sv += len(u)
            if ex > bx:
                best, bx = s, ex
        r["tw_n05"] = float(n05)
        r["tw_cover_any"] = float(cover.mean())
        r["tw_sum_vr"] = sv / len(t)
        if best is None:
            continue
        u = on[best]
        hs = np.abs(_near(t, u)) <= TOL
        ht = np.abs(_near(u, t)) <= TOL
        r["tw_m_self"], r["tw_m_twin"] = float(hs.mean()), float(ht.mean())
        r["tw_m_asym"] = r["tw_m_self"] - r["tw_m_twin"]
        r["tw_lvr"] = float(np.log((len(t) + 1) / (len(u) + 1)))
        if nc >= 1:
            a_ng, b_ng = out[d]["_ng"], out[best]["_ng"]
            r["tw_ng_unm_self"] = float((a_ng & ~hs).sum() / nc)
            r["tw_ng_unm_twin"] = float((b_ng & ~ht).sum() / nc)
            r["tw_ng_unm_diff"] = r["tw_ng_unm_self"] - r["tw_ng_unm_twin"]
            ia, ib = out[d]["_inc"], out[best]["_inc"]
            if ia.sum() >= 3 and ib.sum() >= 3:
                r["tw_ngshare_diff"] = float(a_ng[ia].mean() - b_ng[ib].mean())
            if d in first and best in first:
                ja, oka = first[d]
                jb, okb = first[best]
                if okb.sum() >= 2:
                    r["tw_first_hit"] = float((np.abs(_near(u[jb[okb]], t)) <= TOL).mean())
                if oka.sum() >= 2:
                    r["tw_first_hit_rev"] = float((np.abs(_near(t[ja[oka]], u)) <= TOL).mean())
                if "tw_first_hit" in r and "tw_first_hit_rev" in r:
                    r["tw_first_hit_diff"] = r["tw_first_hit"] - r["tw_first_hit_rev"]
                both = oka & okb
                if both.sum() >= 2:
                    r["tw_first_lag_diff"] = float(np.median(t[ja[both]] - u[jb[both]]))
        r["_twin"] = best
    fo = {d: out[d].get("rl_first_off_med", np.nan) for d in dets}
    ro = {d: out[d].get("rl_red_onset_med", np.nan) for d in dets}
    vf = [v for v in fo.values() if not np.isnan(v)]
    vr = [v for v in ro.values() if not np.isnan(v)]
    for d in dets:
        r = out[d]
        if not np.isnan(fo[d]) and len(vf) >= 2:
            r["rl_first_off_rel"] = fo[d] - min(vf)
        if not np.isnan(ro[d]) and len(vr) >= 2:
            r["rl_red_onset_rel"] = ro[d] - min(vr)
        tw = r.get("_twin")
        if tw is not None and not np.isnan(fo[d]) and not np.isnan(fo[tw]):
            r["rl_first_off_minus_twin"] = fo[d] - fo[tw]
    return out


# note 61b (user cues): sp_* a Yellow_Red zone often spans several lanes, a Count zone is single-lane -> it covers several
# mutually independent siblings; volume vs the SUM of the covered zones; side-by-side vehicles counted once.
# lg_* a Yellow_Red zone sits on top of or just downstream of the Count zone, never upstream -> per matched vehicle the
# signed ON / OFF lag (self minus twin) is >= 0; the consistently later detector is the YR candidate.
SP_COLS = ["sp_ncov", "sp_ncov_indep", "sp_vol_vs_sum", "sp_union_cov", "sp_sbs", "sp_ncovered_by"]
LG_COLS = ["lg_on_med", "lg_on_neg", "lg_on_pos", "lg_off_med", "lg_off_neg", "lg_off_pos", "lg_cov_on_med",
           "lg_cov_on_neg", "lg_by_on_med"]
COVER_X, INDEP_X = 0.5, 0.3


def span_lag_feats(dets, on, off, secs):
    out = {d: {} for d in dets}
    big = [d for d in dets if len(on[d]) >= MIN_N]
    if len(big) < 2:
        return out
    idx = {}                                   # (a, b): (hit mask of a's ONs, index of nearest b ON)
    ex = {}                                    # (a, b): chance-corrected share of a's ONs matched by b

    def pair(a, b):
        if (a, b) not in idx:
            t, u = on[a], on[b]
            j = np.clip(np.searchsorted(u, t), 0, len(u) - 1)
            j0 = np.clip(j - 1, 0, len(u) - 1)
            jj = np.where(np.abs(t - u[j0]) < np.abs(t - u[j]), j0, j)
            hit = np.abs(t - u[jj]) <= TOL
            idx[(a, b)] = (hit, jj)
            ex[(a, b)] = hit.mean() - (1 - np.exp(-len(u) / secs * 2 * TOL))
        return idx[(a, b)]
    for a in big:
        for b in big:
            if a != b:
                pair(a, b)
    for d in big:
        r = out[d]
        t = on[d]
        cov = [s for s in big if s != d and ex[(s, d)] >= COVER_X]          # siblings whose ONs d mostly sees
        by = [s for s in big if s != d and ex[(d, s)] >= COVER_X]           # siblings that see most of d's ONs
        r["sp_ncov"] = float(len(cov))
        r["sp_ncovered_by"] = float(len(by))
        ind = []
        for s in sorted(cov, key=lambda s: -len(on[s])):
            if all(ex[(s, q)] < INDEP_X and ex[(q, s)] < INDEP_X for q in ind):
                ind.append(s)
        r["sp_ncov_indep"] = float(len(ind))
        if ind:
            r["sp_vol_vs_sum"] = len(t) / sum(len(on[s]) for s in ind)
            hits = np.vstack([pair(d, s)[0] for s in ind])
            r["sp_union_cov"] = float(hits.any(0).mean())
            if len(ind) >= 2:
                r["sp_sbs"] = float((hits.sum(0) >= 2).mean())
        # lag vs the best twin (max chance-corrected match of d's ONs)
        cand = [s for s in big if s != d and ex[(d, s)] > TW_MIN]
        if cand:
            tw = max(cand, key=lambda s: ex[(d, s)])
            hit, jj = pair(d, tw)
            if hit.sum() >= 3:
                lo = t[hit] - on[tw][jj[hit]]
                lf = off[d][hit] - off[tw][jj[hit]]
                r["lg_on_med"], r["lg_on_neg"], r["lg_on_pos"] = float(np.median(lo)), float((lo < -0.05).mean()), \
                    float((lo > 0.1).mean())
                r["lg_off_med"], r["lg_off_neg"], r["lg_off_pos"] = float(np.median(lf)), float((lf < -0.05).mean()), \
                    float((lf > 0.1).mean())
        lc = [t[pair(d, s)[0]] - on[s][pair(d, s)[1][pair(d, s)[0]]] for s in ind]
        lc = np.concatenate(lc) if lc else np.zeros(0)
        if len(lc) >= 3:
            r["lg_cov_on_med"], r["lg_cov_on_neg"] = float(np.median(lc)), float((lc < -0.05).mean())
        lb = [t[pair(d, s)[0]] - on[s][pair(d, s)[1][pair(d, s)[0]]] for s in by]
        lb = np.concatenate(lb) if lb else np.zeros(0)
        if len(lb) >= 3:
            r["lg_by_on_med"] = float(np.median(lb))
    return out


def feat_work(args):
    dev, period, g = args
    import a2_features as A2F
    import pyarrow.dataset as ds
    import ln6_pick as L6
    tab = ds.dataset(str(L6.CACHE[period])).to_table(filter=ds.field("DeviceId") == dev,
                                                     columns=["Detector", "t_on", "dur"]).to_pandas().drop_duplicates()
    tab["t"] = tab.t_on.astype("datetime64[us]").astype("int64").to_numpy() / 1e6
    tab = tab.sort_values(["Detector", "t"])
    allon = {int(d): (x["t"].to_numpy(), x["t"].to_numpy() + x.dur.fillna(0).to_numpy()) for d, x in tab.groupby("Detector")}
    pcf = (DCW / "cache" / "phase_cycles.parquet") if period == "dec" else \
        (DCW / "official" / "stg" / "cache" / "phase_cycles.parquet")
    pc = ds.dataset(str(pcf)).to_table(filter=ds.field("DeviceId") == dev).to_pandas()
    pc = pc[pc.next_green.notna()].copy()
    for c in ("green_start", "yellow_start", "red_start", "next_green"):
        pc[c] = pc[c].astype("datetime64[us]").astype("int64") / 1e6
        pc.loc[pc[c] < 0, c] = np.nan
    pc["ge"] = pc.yellow_start.fillna(pc.red_start).fillna(pc.next_green)
    pc = pc.sort_values("green_start")
    cyc = {int(p): (x["green_start"].to_numpy(), x["ge"].to_numpy(), x["next_green"].to_numpy()) for p, x in pc.groupby("Phase")}
    rows = []
    for name, t0w, secs in A2F.WINDOWS[period]:
        w = g[g.win == name]
        if w.empty:
            continue
        t0 = pd.Timestamp(t0w).value / 1e9
        t1 = t0 + secs
        for ph, gg in w.groupby("pred_phase"):
            dets = [int(d) for d in gg.Detector]
            on, off = {}, {}
            for d in dets:
                x, y = allon.get(d, (np.zeros(0), np.zeros(0)))
                m = (x >= t0) & (x < t1)
                on[d], off[d] = x[m], y[m]
            c = cyc.get(int(ph))
            if c is None:
                gs = ge = ng = np.zeros(0)
            else:
                m = (c[0] >= t0) & (c[0] < t1)
                gs, ge, ng = c[0][m], c[1][m], c[2][m]
            F = group_feats(dets, on, off, gs, ge, ng, secs)
            S2 = span_lag_feats(dets, on, off, secs)
            for d in dets:
                F[d].update(S2[d])
                r = {k: v for k, v in F[d].items() if not k.startswith("_")}
                r.update(DeviceId=dev, Detector=d, period=period, win=name)
                rows.append(r)
    return pd.DataFrame(rows)


def stage_feats(a):
    from multiprocessing import Pool
    k = B.frame_keys()
    k = k[k.pred_phase.notna()]
    jobs = [(dev, per, g[["Detector", "win", "pred_phase"]].copy()) for (dev, per), g in k.groupby(["DeviceId", "period"])]
    if a.limit:
        jobs = jobs[: a.limit]
    log(f"{len(jobs)} signal-periods")
    parts, t0 = [], time.time()
    with Pool(a.workers) as pool:
        for i, f in enumerate(pool.imap_unordered(feat_work, jobs, chunksize=1)):
            parts.append(f)
            if i % 100 == 0:
                log(f"  {i}/{len(jobs)} ({time.time()-t0:.0f}s)")
    F = pd.concat(parts, ignore_index=True)
    assert not F.DeviceId.isin(V.locked_signals()).any()
    for c in TW_COLS + RL_COLS + SP_COLS + LG_COLS:
        if c not in F:
            F[c] = np.nan
    F = F[KEY + TW_COLS + RL_COLS + SP_COLS + LG_COLS]
    F.to_parquet(FEAT_F if not a.limit else OUT / "feats_test.parquet", index=False)
    log(f"wrote {len(F):,} rows; coverage tw {F.tw_m_self.notna().mean():.3f} rl {F.rl_first_off_med.notna().mean():.3f} "
        f"sp {F.sp_ncov.notna().mean():.3f} lg {F.lg_on_med.notna().mean():.3f} ({time.time()-t0:.0f}s)")


# ------------------------------------------------------------------------------------------------ fit / score
FAMS = {"tw": TW_COLS, "rl": RL_COLS, "twrl": TW_COLS + RL_COLS, "sp": SP_COLS, "lg": LG_COLS, "splg": SP_COLS + LG_COLS,
        "twsplg": TW_COLS + SP_COLS + LG_COLS}


def stage_fit(a):
    """note-57 229-feature arm + a targeted family; `<fam>_shuf` = the family's rows permuted (control)."""
    fr, cols, lab, y, ok = B.load_train()
    cfg = a.cfg
    shuf = cfg.endswith("_shuf")
    fam = cfg[:-5] if shuf else cfg
    names = FAMS[fam]
    F = pd.read_parquet(FEAT_F).astype({"Detector": fr.Detector.dtype})
    F["DeviceId"] = F.DeviceId.str.lower()
    x = fr[KEY].assign(DeviceId=fr.DeviceId.str.lower()).merge(F[KEY + names], on=KEY, how="left")
    assert len(x) == len(fr)
    E = x[names].to_numpy(np.float32)
    log(f"{cfg}: {len(names)} columns, coverage {np.isfinite(E).any(1).mean():.3f}")
    if shuf:
        E = E[np.random.default_rng(61).permutation(len(E))]
    X = np.hstack([fr[cols].to_numpy(np.float32), E])
    yi = pd.Series(y).map({c: i for i, c in enumerate(C7)}).fillna(-1).astype(int).to_numpy()
    fo = fr.fold.to_numpy()
    w = np.ones(len(fr))
    d = OUT / "fit" / cfg
    d.mkdir(parents=True, exist_ok=True)
    json.dump({"cfg": cfg, "classes": C7, "n_cols": len(cols) + len(names), "extra": names}, open(d / "cols.json", "w"))
    for s in [int(v) for v in a.seeds.split(",")]:
        for k in range(6):
            f = d / f"P_{B.VAR}_s{s}_f{k}.npy"
            if f.exists():
                continue
            t0 = time.time()
            inner = (k + 1) % 6
            P, nt = B.lgb_fit(X, yi, w, ok & (fo != k) & (fo != inner), ok & (fo == inner), fo == k, s, a.threads, 7)
            np.save(f, P)
            log(f"{cfg} s{s} f{k}: {nt} trees, {time.time()-t0:.0f}s")


def stage_score(a):
    B.OUT = OUT                          # fit dirs of this note; 'base' still = the note-57 arm (s59 FUNC_DIR)
    B.stage_score(a)


TARGETS = {"YR<->Count": {("Yellow_Red", "Count"), ("Count", "Yellow_Red")},
           "Adv<->Pres": {("Advance", "Presence"), ("Presence", "Advance")},
           "any A->wrongA": None}


def stage_target(a):
    """targeted-confusion rates (per 100 scored rows) vs base, paired signal bootstrap."""
    import atspm_score as S
    B.OUT = OUT
    base, _ = B.evaluate(a.base)
    res = {}

    def mark(d, pairs):
        if pairs is None:
            return d.assign(ok_t=d.err.ne("A->wrongA"))
        return d.assign(ok_t=~pd.Series(list(zip(d.t, d.p)), index=d.index).isin(pairs))
    for spec in [s for s in a.cfgs.split(",") if s]:
        D, _ = B.evaluate(spec)
        res[spec] = {}
        for sname in ("everything", "realistic"):
            for scope in ("all", "ge30", "short"):
                sel0 = base[sname].wgroup.isin(["m5", "m10"])
                sel1 = D[sname].wgroup.isin(["m5", "m10"])
                if scope == "ge30":
                    sel0, sel1 = ~sel0, ~sel1
                elif scope == "all":
                    sel0, sel1 = sel0 | True, sel1 | True
                d0, d1 = base[sname][sel0], D[sname][sel1]
                res[spec][f"{sname}|{scope}|ATSPM"] = [round(100 * d0.ok_a.mean(), 3), round(100 * d1.ok_a.mean(), 3),
                                                       S.boot(d0, d1)]
                for nm, pairs in TARGETS.items():
                    b0, b1 = mark(d0, pairs), mark(d1, pairs)
                    res[spec][f"{sname}|{scope}|{nm}"] = [round(100 * (1 - b0.ok_t.mean()), 3),
                                                          round(100 * (1 - b1.ok_t.mean()), 3), S.boot(b0, b1, "ok_t")]
        for k, v in res[spec].items():
            log(f"{spec} {k}: {v[0]} -> {v[1]} {v[2]}")
    json.dump(res, open(OUT / f"target_{a.tag}.json", "w"), indent=1)


def stage_agg(a):
    B.OUT = OUT
    B.stage_agg(a)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--cfg", default="")
    ap.add_argument("--cfgs", default="")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--base", default="base")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    globals()[f"stage_{a.stage}"](a)
