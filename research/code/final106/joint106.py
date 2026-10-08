"""Note 106, idea 3: structured per-phase decode -- lanes and roles chosen TOGETHER, instead of lanes first (lanes D
decoder anchored on the tree argmax) and roles second (greedy per-lane pick, gate .9).

Per (signal, window >= 30 min, predicted phase) group of detectors with >= 10 ONs: each detector gets a role
r in {Advance, Presence, Count, Yellow_Red, N (= non-ATSPM: Mid / Bike / Other)} and, if ATSPM, a non-empty lane set
(bitmask over <= 4 lanes).  Maximise
    sum_i w_r log P_i(r_i)                                   (stacker probabilities; N = P(Mid)+P(Bike)+P(Other))
  + w_p sum_{i<j ATSPM} [overlap] log(p_ij/.5) + [disjoint] log((1-p_ij)/.5)   (lanes D pair model P(same lane))
  - mu (|lanes_i| - 1) (non-YR) - mu_y (|lanes_i| - 1) (YR)     (spanning)
  - lam * n_lanes                                            (lane count)
  - nu * [Advance shares a lane with a Presence / Count that is UPSTREAM of it by the ON-time lead-lag]
subject to <= 1 detector of each ATSPM class per lane (a spanning detector holds its class on every lane it spans;
two Count zones => two lanes) and every used lane anchored by a single-lane ATSPM detector (penalty kappa).
Search: coordinate ascent over (role, lane set) per detector with exact O(n) deltas, from two starts (the current
decode's output; argmax + greedy lane fill); the better optimum wins.  N detectors get no lane.
Outside these groups (windows < 30 min, < 10 ONs, single-detector groups) the current decode's answer is kept.
Weights picked per held-out fold on the other five folds' >= 30-min ATSPM E (nested grid).

    set F76_ARM=c & python joint106.py run --P NAME [--grid small|full] [--pairs f77|f102] [--workers 6]
    set F76_ARM=c & python joint106.py eval --P NAME --tag TAG      nested pick, ATSPM E / R vs base decode, lanes
locked_v2 asserted absent by the loaders.  CPU <= 6 workers.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("F76_ARM", "c")
import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "final106"))
import s106 as B  # noqa: E402

OUT = B.OUT / "joint"
GE30 = ["m30", "h1", "h3", "h6", "h24", "full"]
C7 = ["Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other"]
NIDX = [4, 5, 6]
MASKS = np.arange(1, 16)                       # non-empty subsets of 4 lanes
POP = np.array([bin(m).count("1") for m in range(16)])
PAIRS = {"f77": B.G.DC_WORK / "final_v3_work" / "f77" / "ln8" / "p_Dsym.parquet",
         "f102": B.G.DC_WORK / "final_v3_work" / "f102" / "ln8" / "p_base.parquet"}
KAPPA = 8.0
DELTA = 0.05


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ================================================================================================ the decoder
def decode_group(LP, Ls, Ld, Dr, role0, mask0, prm):
    """LP n x 5 role log-probs; Ls / Ld n x n pair LLR (same / different lane, 0 = unknown); Dr n x n lead-lag
    direction (Dr[i, j] > 0: i upstream of j); role0 / mask0 = start.  -> (role, mask, score)."""
    wr, wp, mu, muy, lam, nu = prm["wr"], prm["wp"], prm["mu"], prm["muy"], prm["lam"], prm["nu"]
    n = len(LP)
    Om = ((MASKS[:, None] & MASKS[None, :]) != 0)          # 15 x 15 overlap of masks
    up_wrong_A = (Dr < -DELTA)                             # i = Advance, j upstream of it
    up_wrong_S = (Dr > DELTA)                              # i = stop bar, i upstream of the advance j

    def best_move(i, role, mask):
        oth = np.arange(n) != i
        ats = oth & (role < 4)
        used = np.bitwise_or.reduce(mask[oth]) if oth.any() else 0
        single = ats & (POP[mask] == 1)
        anc = np.bitwise_or.reduce(mask[single]) if single.any() else 0
        occ = [np.bitwise_or.reduce(mask[ats & (role == c)]) if (ats & (role == c)).any() else 0 for c in range(4)]
        # pair term for every mask (vector over 15 masks)
        if ats.any():
            ov = (MASKS[:, None] & mask[ats][None, :]) != 0          # 15 x k
            pt = wp * (ov @ Ls[i, ats] + (~ov) @ Ld[i, ats])
        else:
            pt = np.zeros(15)
        Lc = POP[used | MASKS]
        unanch_ats = {}
        best = (-np.inf, 4, 0)
        # N: no lane
        un0 = POP[used & ~anc]
        s_n = wr * LP[i, 4] - lam * POP[used] - KAPPA * un0
        best = (s_n, 4, 0)
        for c in range(4):
            feas = (MASKS & occ[c]) == 0
            if not feas.any():
                continue
            sp = (muy if c == 3 else mu) * (POP[MASKS] - 1)
            anc_i = np.where(POP[MASKS] == 1, anc | MASKS, anc)
            un = POP[(used | MASKS) & ~anc_i]
            s = wr * LP[i, c] + pt - sp - lam * Lc - KAPPA * un
            if nu > 0:
                if c == 0:
                    bad = ats & (role != 0) & (role != 3) & up_wrong_A[i]
                else:
                    bad = ats & (role == 0) & up_wrong_S[i] if c in (1, 2) else np.zeros(n, bool)
                if bad.any():
                    s = s - nu * ((MASKS[:, None] & mask[bad][None, :]) != 0).sum(1)
            s = np.where(feas, s, -np.inf)
            j = int(np.argmax(s))
            if s[j] > best[0] + 1e-9:
                best = (float(s[j]), c, int(MASKS[j]))
        return best

    def total(role, mask):
        ats = role < 4
        sc = wr * LP[np.arange(n), role].sum()
        idx = np.where(ats)[0]
        for a_, b_ in itertools.combinations(idx, 2):
            sc += wp * (Ls[a_, b_] if (mask[a_] & mask[b_]) else Ld[a_, b_])
        sc -= sum((muy if role[i] == 3 else mu) * (POP[mask[i]] - 1) for i in idx)
        used = np.bitwise_or.reduce(mask[ats]) if ats.any() else 0
        single = ats & (POP[mask] == 1)
        anc = np.bitwise_or.reduce(mask[single]) if single.any() else 0
        sc -= lam * POP[used] + KAPPA * POP[used & ~anc]
        if nu > 0:
            for i in idx:
                for j in idx:
                    if i != j and (mask[i] & mask[j]):
                        if role[i] == 0 and role[j] in (1, 2) and Dr[i, j] < -DELTA:
                            sc -= nu
        return sc

    def ascend(role, mask):
        role, mask = role.copy(), mask.copy()
        for _ in range(12):
            moved = False
            for i in range(n):
                s, c, m = best_move(i, role, mask)
                cur_ok = (role[i], mask[i])
                if (c, m) != cur_ok:
                    # compare with the current state's own score for i
                    cs = _own(i, role, mask)
                    if s > cs + 1e-9:
                        role[i], mask[i] = c, m
                        moved = True
            if not moved:
                break
        return role, mask

    def _own(i, role, mask):
        r0, m0 = role[i], mask[i]
        # score of keeping detector i as it is, computed by the same formula as best_move
        oth = np.arange(n) != i
        ats = oth & (role < 4)
        used = np.bitwise_or.reduce(mask[oth]) if oth.any() else 0
        single = ats & (POP[mask] == 1)
        anc = np.bitwise_or.reduce(mask[single]) if single.any() else 0
        if r0 == 4:
            return wr * LP[i, 4] - lam * POP[used] - KAPPA * POP[used & ~anc]
        occ = np.bitwise_or.reduce(mask[ats & (role == r0)]) if (ats & (role == r0)).any() else 0
        if m0 & occ:
            return -np.inf
        pt = 0.0
        for j in np.where(ats)[0]:
            pt += Ls[i, j] if (m0 & mask[j]) else Ld[i, j]
        anc_i = anc | m0 if POP[m0] == 1 else anc
        s = wr * LP[i, r0] + wp * pt - (muy if r0 == 3 else mu) * (POP[m0] - 1) - lam * POP[used | m0] \
            - KAPPA * POP[(used | m0) & ~anc_i]
        if nu > 0:
            if r0 == 0:
                bad = ats & (role != 0) & (role != 3) & up_wrong_A[i]
            elif r0 in (1, 2):
                bad = ats & (role == 0) & up_wrong_S[i]
            else:
                bad = np.zeros(n, bool)
            s -= nu * sum(1 for j in np.where(bad)[0] if mask[j] & m0)
        return s

    # start 1: current decode
    r1, m1 = ascend(role0, mask0)
    # start 2: argmax roles, empty lanes, greedy fill in confidence order
    r2 = np.full(n, 4)
    m2 = np.zeros(n, int)
    for i in np.argsort(-LP.max(1)):
        s, c, m = best_move(i, r2, m2)
        r2[i], m2[i] = c, m
    r2, m2 = ascend(r2, m2)
    s1, s2 = total(r1, m1), total(r2, m2)
    return (r1, m1, s1) if s1 >= s2 else (r2, m2, s2)


# ================================================================================================ data
def build_groups(P, pred0, pairs_tag):
    """list of group dicts for stg windows >= 30 min, >= 10 ONs, predicted phase, >= 2 detectors."""
    k = pd.read_parquet(B.OUT / "keys.parquet")
    m = (k.period == "stg") & k.wgroup.isin(GE30) & k.pred_phase.notna() & (k.det_n_on >= 10)
    sub = k[m].copy()
    sub["_i"] = np.where(m)[0]
    q = pd.read_parquet(PAIRS[pairs_tag], columns=["DeviceId", "period", "win", "da", "db", "p_same", "same_pred"])
    q = q[q.same_pred & (q.period == "stg")]
    qd = {kk: dict(zip(zip(g.da.astype(int), g.db.astype(int)), g.p_same.astype(float)))
          for kk, g in q.groupby(["DeviceId", "win"])}
    pv = pd.read_parquet(B.OUT / "pairs_vp.parquet", columns=["DeviceId", "win", "da", "db", "dir"])
    dd = {kk: dict(zip(zip(g.da.astype(int), g.db.astype(int)), g.dir.astype(float)))
          for kk, g in pv.groupby(["DeviceId", "win"])}
    groups = []
    for (dev, w, ph), g in sub.groupby(["DeviceId", "win", "pred_phase"]):
        ix = g._i.to_numpy()
        dets = g.Detector.astype(int).to_numpy()
        pq, pdr = qd.get((dev, w), {}), dd.get((dev, w), {})
        n = len(ix)
        pp = np.full((n, n), np.nan)
        dr = np.zeros((n, n))
        for a_ in range(n):
            for b_ in range(a_ + 1, n):
                x, y = dets[a_], dets[b_]
                v = pq.get((x, y), pq.get((y, x)))
                if v is not None:
                    pp[a_, b_] = pp[b_, a_] = v
                d = pdr.get((x, y))
                if d is not None:
                    dr[a_, b_], dr[b_, a_] = d, -d
                else:
                    d = pdr.get((y, x))
                    if d is not None:
                        dr[a_, b_], dr[b_, a_] = -d, d
        lanes = [frozenset(int(v) for v in s.split(",")) if isinstance(s, str) and s else frozenset()
                 for s in g.lanes5g.to_numpy(object)]
        groups.append(dict(key=(dev, w, int(ph)), ix=ix, pp=pp, dr=dr, lanes=lanes))
    return groups


def start_state(Pg, predg, lanes):
    """role / mask from the current decode output and its lanes (lane ids remapped to bits 0..3)."""
    role = np.where(predg < 4, predg, 4)
    ids = sorted({l for s in lanes for l in s})[:4]
    bit = {l: 1 << i for i, l in enumerate(ids)}
    mask = np.array([sum(bit[l] for l in s if l in bit) for s in lanes], int)
    # ATSPM without lane: put on lane 0 (ascent repairs); N: no lane
    mask = np.where((role < 4) & (mask == 0), 1, mask)
    mask = np.where(role == 4, 0, mask)
    return role, mask


def work(args):
    groups, P, pred0, grid = args
    out = []
    for g in groups:
        Pg = P[g["ix"]]
        if len(g["ix"]) == 1:
            p0 = pred0[g["ix"]].astype(np.int8)
            out.append((g["key"], g["ix"], [(p0, int(p0[0] < 4))] * len(grid)))
            continue
        LP = np.log(np.clip(np.column_stack([Pg[:, :4], Pg[:, NIDX].sum(1)]), 1e-6, 1))
        pp = np.clip(g["pp"], 0.02, 0.98)
        Ls = np.where(np.isnan(pp), 0.0, np.log(pp / 0.5))
        Ld = np.where(np.isnan(pp), 0.0, np.log((1 - pp) / 0.5))
        np.fill_diagonal(Ls, 0)
        np.fill_diagonal(Ld, 0)
        r0, m0 = start_state(Pg, pred0[g["ix"]], g["lanes"])
        res = []
        for prm in grid:
            r, m, s = decode_group(LP, Ls, Ld, g["dr"], r0, m0, prm)
            cls = np.where(r < 4, r, np.array(NIDX)[Pg[:, NIDX].argmax(1)])
            res.append((cls.astype(np.int8), int(POP[np.bitwise_or.reduce(m[r < 4])] if (r < 4).any() else 0)))
        out.append((g["key"], g["ix"], res))
    return out


GRIDS = {
    "wide": [dict(wr=1.0, wp=wp, mu=mu, muy=0.3, lam=lam, nu=nu) for wp in (0.1, 0.25, 0.5) for lam in (0.0, 0.5, 1.0)
             for mu in (0.5, 1.5) for nu in (0.0, 1.0)],
    "small": [dict(wr=1.0, wp=wp, mu=1.0, muy=0.3, lam=lam, nu=0.0) for wp in (0.5, 1.0) for lam in (1.0, 2.0)],
    "full": [dict(wr=1.0, wp=wp, mu=mu, muy=0.3, lam=lam, nu=nu) for wp in (0.5, 1.0, 2.0) for lam in (0.5, 1.5, 3.0)
             for mu in (1.0, 2.5) for nu in (0.0, 1.0)],
}


def base_decode(P):
    """current decode (gate .9 + pick + twin) -> (pred idx, ok dict)."""
    d = B.load()
    E, S, s74 = d["E"], d["S"], d["s74"]
    fr = E["fr"]
    lc = np.load(B.OUT / "lane_ctx.npy")[:, 2]
    lanes0 = fr.lanes5g.copy()
    fr["lanes5g"] = lanes0.where(~(lc < s74.GATE), None)
    pr2, cr = S.run_decode(E, P)
    fr["lanes5g"] = lanes0
    return np.asarray(pr2), S.ok_cols(E, cr)


def ok_of(pred):
    import atspm_score as AS
    d = B.load()
    E, S = d["E"], d["S"]
    cr = {s: AS.credit(E["fr"], pred, "truth_v3s", r, True) for s, r in E["rows"].items()}
    return S.ok_cols(E, cr)


def stage_run(a):
    P = np.load(B.OUT / f"P_{a.P}.npy").astype(float)
    pred0, ok0 = base_decode(P)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / f"pred0_{a.P}.npy", pred0.astype(np.int8))
    t0 = time.time()
    groups = build_groups(P, pred0, a.pairs)
    grid = GRIDS[a.grid]
    log(f"{len(groups):,} groups (sizes: median {np.median([len(g['ix']) for g in groups])}, max "
        f"{max(len(g['ix']) for g in groups)}), {len(grid)} configs; built in {time.time()-t0:.0f}s")
    if a.limit:
        groups = groups[:a.limit]
    chunks = [groups[i::a.workers * 8] for i in range(a.workers * 8)]
    from multiprocessing import Pool
    preds = np.repeat(pred0[None, :], len(grid), 0)
    nl = []
    t0 = time.time()
    with Pool(a.workers) as pool:
        for ci, res in enumerate(pool.imap_unordered(work, [(c, P, pred0, grid) for c in chunks])):
            for key, ix, rr in res:
                for gi, (cls, nlanes) in enumerate(rr):
                    preds[gi, ix] = cls
                    nl.append((key[0], key[1], key[2], gi, nlanes))
            if ci % 8 == 0:
                log(f"  chunk {ci}/{len(chunks)} ({time.time()-t0:.0f}s)")
    tag = f"{a.P}_{a.grid}_{a.pairs}" + (a.suffix or "")
    np.save(OUT / f"preds_{tag}.npy", preds.astype(np.int8))
    pd.DataFrame(nl, columns=["DeviceId", "win", "phase", "cfg", "n_lanes"]).to_parquet(OUT / f"nl_{tag}.parquet",
                                                                                        index=False)
    json.dump(grid, open(OUT / f"grid_{tag}.json", "w"))
    log(f"decoded in {time.time()-t0:.0f}s -> {tag}")


def truth_phase():
    ph = pd.read_parquet(B.G.DC_WORK / "final_v3_work" / "f102" / "ln8" / "truth_phase.parquet")
    ph["phase"] = ph.target.str[1:].astype(int)
    return ph


def inconsistent(pred):
    """phase-windows (stg, >= 30 min, print lane count known) where the decoded output holds MORE single-class ATSPM
    detectors (Advance / Presence / Count, each) than the print has lanes; share of phase-windows."""
    k = pd.read_parquet(B.OUT / "keys.parquet", columns=["DeviceId", "win", "period", "wgroup", "pred_phase", "det_n_on"])
    m = ((k.period == "stg") & k.wgroup.isin(GE30) & (k.det_n_on >= 5)).to_numpy()
    x = k[m].assign(c=np.array(C7, object)[pred[m]])
    ph = truth_phase()
    out = {}
    g = x.groupby(["DeviceId", "win", "pred_phase"]).c.value_counts().unstack(fill_value=0)
    g = g.reset_index().merge(ph, left_on=["DeviceId", "pred_phase"], right_on=["DeviceId", "phase"])
    for c in ["Advance", "Presence", "Count"]:
        out[c] = round(float((g.get(c, 0) > g.n_lanes).mean()), 4)
    out["any"] = round(float(((g.get("Advance", 0) > g.n_lanes) | (g.get("Presence", 0) > g.n_lanes) |
                              (g.get("Count", 0) > g.n_lanes)).mean()), 4)
    out["n_phase_windows"] = int(len(g))
    return out


def joint_lanes(tag, pick):
    """n_lanes exact of the joint decode (nested config per fold) vs the current lanes (f77 D, used by v5b), same
    phase-windows (14 Sept windows >= 30 min, print lane count known), + note-102 review phases."""
    sys.path.insert(0, str(CODE / "final106"))
    import lane106 as LN
    nl = pd.read_parquet(OUT / f"nl_{tag}.parquet")
    k = pd.read_parquet(B.OUT / "keys.parquet", columns=["DeviceId", "fold"]).drop_duplicates("DeviceId")
    nl = nl.merge(k, on="DeviceId")
    nl = nl[nl.cfg == nl.fold.map(pick)]
    ph = truth_phase()
    x = ph.merge(nl, on=["DeviceId", "phase"])
    x = x[x.win.isin(LN.REVIEW_WINS)]
    x["n_lanes_p"] = x.n_lanes_y
    x["n_lanes"] = x.n_lanes_x
    x["ok"] = (x.n_lanes == x.n_lanes_p).astype(float)
    res, _ = LN.laneeval(["f77", "f102"], extra_rows={"joint": x[["DeviceId", "target", "win", "n_lanes", "n_lanes_p",
                                                                   "ok"]]})
    return res


def stage_eval(a):
    import cand64 as C
    tag = a.tag
    P = np.load(B.OUT / f"P_{a.P}.npy").astype(float)
    preds = np.load(OUT / f"preds_{tag}.npy").astype(int)
    grid = json.load(open(OUT / f"grid_{tag}.json"))
    pred0, ok0 = base_decode(P)
    d = B.load()
    fr = d["E"]["fr"]
    st = (fr.period == "stg").to_numpy()
    ge = st & fr.wgroup.isin(GE30).to_numpy()
    fo = fr.fold.to_numpy()
    sig = fr.DeviceId.to_numpy()
    oks = [ok_of(preds[g]) for g in range(len(grid))]
    accE = np.array([[np.nanmean(o["E"][ge & (fo == f)]) for f in range(6)] for o in oks])
    rowsE = np.array([[np.sum(~np.isnan(o["E"][ge & (fo == f)])) for f in range(6)] for o in oks])
    pick = {}
    for f in range(6):
        oth = [k for k in range(6) if k != f]
        sc = (accE[:, oth] * rowsE[:, oth]).sum(1) / rowsE[:, oth].sum(1)
        pick[f] = int(np.argmax(sc))
    pn = pred0.copy()
    for f in range(6):
        pn[fo == f] = preds[pick[f]][fo == f]
    okn = ok_of(pn)
    np.save(OUT / f"pred_nested_{tag}.npy", pn.astype(np.int8))
    res = {"pick": {f: grid[pick[f]] for f in range(6)},
           "full_pool_E_by_cfg": {json.dumps(grid[g]): round(float(np.nanmean(oks[g]["E"][ge])), 5) for g in range(len(grid))},
           "base_full_pool_E": round(float(np.nanmean(ok0["E"][ge])), 5)}
    tr = fr.truth_v3s.to_numpy(object)
    cl = np.where(np.isin(tr, ["Advance", "Presence", "Count", "Yellow_Red"]), tr, "nonATSPM")
    for stn in ("E", "R"):
        for pool, fams in {"ge30": GE30, "m30": ["m30"], "h1": ["h1"], "h3": ["h3"], "h6": ["h6"], "h24": ["h24"],
                           "full": ["full"], "m5": ["m5"], "m10": ["m10"]}.items():
            sc = st & fr.wgroup.isin(fams).to_numpy() & ~np.isnan(ok0[stn]) & ~np.isnan(okn[stn])
            res[f"{stn}_{pool}"] = {"n": int(sc.sum()), "base": C.acc_ci(ok0[stn][sc], sig[sc]),
                                    "joint": C.acc_ci(okn[stn][sc], sig[sc]),
                                    "joint - base": C.delta_ci(ok0[stn][sc], okn[stn][sc], sig[sc])}
        m = ge & ~np.isnan(ok0[stn]) & ~np.isnan(okn[stn])
        res[f"{stn}_ge30_by_class"] = {c: C.delta_ci(ok0[stn][m & (cl == c)], okn[stn][m & (cl == c)], sig[m & (cl == c)])
                                       for c in ["Advance", "Presence", "Count", "Yellow_Red", "nonATSPM"]}
        res[f"{stn}_ge30_by_fold"] = [round(100 * float(np.nanmean(okn[stn][m & (fo == f)]) -
                                                        np.nanmean(ok0[stn][m & (fo == f)])), 3) for f in range(6)]
    res["changed_rows_ge30"] = int(((pn != pred0) & ge).sum())
    res["lane_inconsistent"] = {"base": inconsistent(pred0), "joint": inconsistent(pn)}
    res["lanes"] = joint_lanes(tag, pick)
    f = OUT / "eval106.json"
    old = json.load(open(f)) if f.exists() else {}
    old[tag] = res
    json.dump(old, open(f, "w"), indent=1, default=str)
    for k_, v in res.items():
        if k_.startswith("E_") or k_.startswith("R_ge30") or k_ in ("pick", "base_full_pool_E", "changed_rows_ge30"):
            log(f"{k_}: {v}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "eval"])
    ap.add_argument("--P", default="base_m3")
    ap.add_argument("--grid", default="small")
    ap.add_argument("--pairs", default="f77")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    t0 = time.time()
    globals()[f"stage_{a.stage}"](a)
    log(f"== {a.stage} done ({time.time() - t0:.0f}s)")
