"""A3 -- per-lane grouping and constrained role assignment for function.

Step 1, lanes.  Inside a predicted phase, two detectors are in the SAME LANE when they
see the SAME VEHICLES: either at the same longitudinal position (cross-correlogram peak
at lag 0 with a high coincidence excess) or one after the other (peak at +2..6 s, e.g.
an advance loop and the stop-bar loop it feeds).  Two loops in *different* lanes of the
same approach still have correlated 15-minute counts -- that is demand, not a lane -- so
the correlogram excess does the work and the binned-count correlation only modulates it.
No lane label is used anywhere; `n_lanes` is an output.

Step 2, roles.  Given the lanes, the class probabilities are decoded as a constrained
assignment: at most one Presence, one Count and one Advance PER LANE; Yellow_Red is
unconstrained (one detector often spans all lanes); anything left over is Other.  Solved
exactly per lane with a rectangular Hungarian assignment over
[Advance, Presence, Count] + one free column per detector (its best of Yellow_Red/Other).

    python a3_lanes.py --stage tune     # sweep the lane threshold on folds 1-5
    python a3_lanes.py --stage run      # final numbers + n_lanes distribution
"""
from __future__ import annotations
import argparse
import json
import time
import os
from pathlib import Path
import numpy as np
import pandas as pd

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
WORK = DCW / "trackA"
CLASSES5 = ["Advance", "Presence", "Count", "Yellow_Red", "Other"]
CLASSES3 = ["Advance", "Presence", "Count"]
DUR_GROUPS = ["m5", "m10", "m30", "h1", "h3", "h6", "h24", "full"]
IA, IP, IC, IY, IO = 0, 1, 2, 3, 4
SIM_MODE = "mix"          # "mix" or "ex"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------- the data
def load_lag() -> pd.DataFrame:
    parts = []
    for per, files in (("dec", [DCW / "features" / "det_lag.parquet",
                                DCW / "features" / "det_lag_B.parquet"]),
                       ("stg", [DCW / "function_v4" / "feat_stg" /
                                "det_lag_stg.parquet"])):
        for f in files:
            if not f.exists():
                continue
            d = pd.read_parquet(f, columns=["DeviceId", "Detector", "other", "win",
                                            "lag_peak", "lag_peak_excess",
                                            "coinc_05_excess", "n_a", "n_b"])
            d["period"] = per
            parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d["Detector"] = d.Detector.astype("int16")
    d["other"] = d.other.astype("int16")
    return d


def link_table() -> pd.DataFrame:
    """One symmetric row per (signal, window, unordered detector pair) with the two
    lane cues: correlogram excess (same vehicles) and binned-count correlation."""
    lag = load_lag()
    lag["lo"] = np.minimum(lag.Detector, lag.other)
    lag["hi"] = np.maximum(lag.Detector, lag.other)
    g = lag.groupby(["DeviceId", "period", "win", "lo", "hi"], sort=False).agg(
        lag_excess=("lag_peak_excess", "max"),
        coinc=("coinc_05_excess", "max"),
        lag_peak=("lag_peak", "first"),
        n_a=("n_a", "max"), n_b=("n_b", "max")).reset_index()

    pr = []
    for per in ("dec", "stg"):
        f = WORK / f"pairs_raw_{per}.parquet"
        if f.exists():
            d = pd.read_parquet(f)
            d["period"] = per
            pr.append(d)
    if pr:
        pr = pd.concat(pr, ignore_index=True)
        pr["lo"] = np.minimum(pr.Detector, pr.other)
        pr["hi"] = np.maximum(pr.Detector, pr.other)
        pr = pr[["DeviceId", "period", "win", "lo", "hi", "c_all", "c_off",
                 "c_peak", "r_peak", "r_off"]]
        g = g.merge(pr, on=["DeviceId", "period", "win", "lo", "hi"], how="outer")
    else:
        for c in ("c_all", "c_off", "c_peak", "r_peak", "r_off"):
            g[c] = np.nan
    # same lane == the two channels see the SAME vehicles, so: their counts must be
    # nearly equal (`bal`), their binned counts must track each other even when demand
    # is low (`corr`, the engineer's "they correlate almost perfectly"), and individual
    # actuations must pair up at a consistent lag (`ex`, zero for co-located loops,
    # +2..6 s for an advance loop feeding a stop-bar loop).
    corr = g.c_off.fillna(g.c_all).fillna(0.5).clip(0, 1)
    ex = g.lag_excess.fillna(0.0).clip(0, 1)
    bal = (np.minimum(g.n_a, g.n_b) / np.maximum(g.n_a, g.n_b).clip(lower=1)).fillna(0.5)
    # measured against the 175 phases whose channel text names the lane (RL/CL/LL):
    # the correlogram excess alone separates same-lane from different-lane pairs best
    # (AUC .736), the binned-count correlation alone .681, the product .702.
    g["s"] = ex if SIM_MODE == "ex" else bal * corr * (0.3 + 0.7 * ex)
    g["bal"] = bal
    return g[["DeviceId", "period", "win", "lo", "hi", "s", "lag_excess", "coinc",
              "lag_peak", "c_all", "c_off", "r_peak", "r_off", "bal"]]


# --------------------------------------------------------------- lane grouping
def cluster(dets: np.ndarray, S: np.ndarray, th: float) -> np.ndarray:
    """Average-linkage agglomeration on similarity, cut at `th`.  Tiny groups, so a
    direct loop is faster than scipy."""
    n = len(dets)
    lab = np.arange(n)
    if n < 2:
        return lab
    S = S.copy()
    np.fill_diagonal(S, -np.inf)
    groups = {i: [i] for i in range(n)}
    while len(groups) > 1:
        keys = list(groups)
        best, bi, bj = -np.inf, None, None
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                ia, ib = groups[keys[a]], groups[keys[b]]
                v = float(np.mean(S[np.ix_(ia, ib)]))
                if v > best:
                    best, bi, bj = v, keys[a], keys[b]
        if best < th:
            break
        groups[bi] = groups[bi] + groups[bj]
        del groups[bj]
    out = np.empty(n, int)
    for k, (_, idx) in enumerate(groups.items()):
        out[idx] = k
    return out


def _solve(logp: np.ndarray, idx: np.ndarray, cap: int, out: np.ndarray) -> None:
    """Assign the detectors `idx` to roles, allowing `cap` copies of each of
    Advance / Presence / Count and an unlimited free role (Yellow_Red or Other)."""
    from scipy.optimize import linear_sum_assignment
    m = len(idx)
    free = np.where(logp[idx, IY] >= logp[idx, IO], IY, IO)
    k = 3 * cap
    cost = np.full((m, k + m), 1e6)
    for j, cls in enumerate([IA, IP, IC]):
        cost[:, j * cap:(j + 1) * cap] = -logp[idx, cls][:, None]
    for i in range(m):
        cost[i, k + i] = -logp[idx[i], free[i]]
    r, c = linear_sum_assignment(cost)
    for i, j in zip(r, c):
        out[idx[i]] = [IA, IP, IC][j // cap] if j < k else free[i]


def assign(P: np.ndarray, lanes: np.ndarray, mode: str = "lane") -> np.ndarray:
    """`lane`: <=1 Advance, <=1 Presence, <=1 Count PER LANE, each detector tied to its
    own lane.  `cap`: the same counts summed over the phase (<= n_lanes of each), which
    is the same constraint but indifferent to WHICH lane a detector is in -- much more
    forgiving when the lane grouping itself is shaky."""
    out = np.full(len(P), -1, int)
    logp = np.log(np.clip(P, 1e-9, None))
    if mode == "cap":
        _solve(logp, np.arange(len(P)), max(1, len(np.unique(lanes))), out)
        return out
    for L in np.unique(lanes):
        _solve(logp, np.flatnonzero(lanes == L), 1, out)
    return out


# ------------------------------------------------------------------- pipeline
def decode(keys: pd.DataFrame, P: np.ndarray, links: pd.DataFrame,
           th: float, mode: str = "lane") -> tuple[np.ndarray, np.ndarray]:
    """-> (decoded class index per row, n_lanes of the row's phase group)."""
    out = np.array(P).argmax(1)
    nlanes = np.ones(len(keys), np.int16)
    laneid = np.zeros(len(keys), np.int16)
    lk = {}
    for (did, per, win), g in links.groupby(["DeviceId", "period", "win"], sort=False):
        lk[(did, per, win)] = dict(zip(zip(g.lo.to_numpy(), g.hi.to_numpy()),
                                       g.s.to_numpy()))
    order = keys.groupby(["DeviceId", "period", "win", "pred_phase"], sort=False).indices
    for (did, per, win, ph), rows in order.items():
        if len(rows) == 1:
            continue
        dets = keys.Detector.to_numpy()[rows]
        sub = lk.get((did, per, win), {})
        n = len(dets)
        S = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                a, b = (dets[i], dets[j]) if dets[i] < dets[j] else (dets[j], dets[i])
                S[i, j] = S[j, i] = sub.get((a, b), 0.0)
        lanes = cluster(dets, S, th)
        out[rows] = assign(P[rows], lanes, mode)
        nlanes[rows] = len(np.unique(lanes))
        laneid[rows] = lanes
    return out, nlanes, laneid


def score(keys, pred_idx, mask=None) -> dict:
    m = np.ones(len(keys), bool) if mask is None else mask
    yt = keys.func5.to_numpy()[m]
    pred = np.array(CLASSES5)[pred_idx[m]]
    wg = keys.wgroup.to_numpy()[m]
    apc = np.isin(yt, CLASSES3)
    d = {"n": int(m.sum()), "acc5_allwin": round(float((yt == pred).mean()), 4),
         "accAPC_allwin": round(float((yt[apc] == pred[apc]).mean()), 4),
         "other_recall": round(float((pred[yt == "Other"] == "Other").mean()), 4)}
    d["acc5_by_duration"] = {g: round(float((yt[wg == g] == pred[wg == g]).mean()), 4)
                             for g in DUR_GROUPS if (wg == g).any()}
    d["accAPC_by_duration"] = {
        g: round(float((yt[(wg == g) & apc] == pred[(wg == g) & apc]).mean()), 4)
        for g in DUR_GROUPS if ((wg == g) & apc).any()}
    full = wg == "full"
    per = {}
    for i, c in enumerate(CLASSES5):
        tp = int(((yt[full] == c) & (pred[full] == c)).sum())
        fp = int(((yt[full] != c) & (pred[full] == c)).sum())
        fn = int(((yt[full] == c) & (pred[full] != c)).sum())
        per[c] = {"n": tp + fn,
                  "precision": round(tp / (tp + fp), 4) if tp + fp else None,
                  "recall": round(tp / (tp + fn), 4) if tp + fn else None}
    d["per_class_full"] = per
    return d


def load() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    keys = pd.read_parquet(WORK / "a2_oof_keys.parquet")
    P = np.load(WORK / "a2_oof_base_expert_mean.npy")
    links = link_table()
    return keys, P, links


def stage_tune(a) -> None:
    keys, P, links = load()
    dev = (keys.fold != 0).to_numpy()          # folds 1-5 only; fold 0 kept clean
    res = {}
    for mode in ("lane", "cap"):
        for th in (0.20, 0.30, 0.40, 0.50, 0.60, 0.70):
            t0 = time.time()
            key = f"{mode}/{th}"
            pred, nl, _ = decode(keys, P, links, th, mode)
            res[key] = score(keys, pred, dev)
            res[key]["n_lanes_hist"] = {str(k): int(v) for k, v in
                                        pd.Series(nl[dev]).value_counts().items()}
            res[key]["secs"] = round(time.time() - t0, 1)
            log(f"{key}: allwin {res[key]['acc5_allwin']:.4f} "
                f"APC {res[key]['accAPC_allwin']:.4f} "
                f"full {res[key]['acc5_by_duration'].get('full')} "
                f"({res[key]['secs']:.0f}s)")
    base = score(keys, np.array(P).argmax(1), dev)
    res["argmax_baseline"] = base
    log(f"argmax: allwin {base['acc5_allwin']:.4f} full "
        f"{base['acc5_by_duration'].get('full')}")
    json.dump(res, open(WORK / "a3_tune.json", "w"), indent=1, default=str)


def stage_run(a) -> None:
    keys, P, links = load()
    th = a.th
    pred, nl, lid = decode(keys, P, links, th, a.mode)
    keys["n_lanes"] = nl
    keys["lane"] = lid
    keys["pred_lane"] = np.array(CLASSES5)[pred]
    keys["pred_argmax"] = np.array(CLASSES5)[np.array(P).argmax(1)]
    keys.to_parquet(WORK / "a3_decoded.parquet", index=False)
    allm = np.ones(len(keys), bool)
    f0 = (keys.fold == 0).to_numpy()
    res = {"threshold": th, "mode": a.mode,
           "argmax": score(keys, np.array(P).argmax(1), allm),
           "lane_decoded": score(keys, pred, allm),
           "argmax_fold0": score(keys, np.array(P).argmax(1), f0),
           "lane_decoded_fold0": score(keys, pred, f0)}
    grp = keys.groupby(["DeviceId", "period", "win", "pred_phase"]).n_lanes.first()
    full = keys[keys.wgroup == "full"].groupby(
        ["DeviceId", "period", "pred_phase"]).agg(
        n_lanes=("n_lanes", "first"), n_det=("Detector", "size"))
    res["n_lanes_all_windows"] = {str(k): int(v) for k, v in
                                  grp.value_counts().sort_index().items()}
    res["n_lanes_full_window"] = {str(k): int(v) for k, v in
                                  full.n_lanes.value_counts().sort_index().items()}
    res["n_lanes_full_share"] = {
        str(k): round(float(v), 4) for k, v in
        full.n_lanes.clip(upper=3).value_counts(normalize=True).sort_index().items()}
    res["mean_detectors_per_phase_full"] = round(float(full.n_det.mean()), 2)

    # validation, never an input: the engineer's rule "two count zones on a phase means
    # two lanes, so at most two presence and two yellow_red zones".  n_count / n_presence
    # from the LABELS is therefore a lower bound on the true lane count.
    fw = keys[keys.wgroup == "full"]
    off = pd.read_parquet(DCW / "official" / "labels_official.parquet")[
        ["DeviceId", "DeviceName", "Detector", "description"]]
    off["DeviceId"] = off.DeviceId.str.lower()
    off["Detector"] = off.Detector.astype(fw.Detector.dtype)
    lb = fw.groupby(["DeviceId", "period", "pred_phase"]).agg(
        n_lanes=("n_lanes", "first"),
        n_count=("func5", lambda s: int((s == "Count").sum())),
        n_pres=("func5", lambda s: int((s == "Presence").sum())))
    lb["label_lanes"] = lb[["n_count", "n_pres"]].max(axis=1)
    ok = lb[lb.label_lanes > 0]
    res["vs_label_lower_bound"] = {
        "n_phases": int(len(ok)),
        "n_lanes_ge_label_bound": round(float((ok.n_lanes >= ok.label_lanes).mean()), 4),
        "exact_match": round(float((ok.n_lanes == ok.label_lanes).mean()), 4),
        "mean_inferred": round(float(ok.n_lanes.mean()), 3),
        "mean_label_bound": round(float(ok.label_lanes.mean()), 3),
        "crosstab": {f"lanes{int(a)}_bound{int(b)}": int(v) for (a, b), v in
                     ok.groupby([ok.n_lanes.clip(upper=4),
                                 ok.label_lanes.clip(upper=4)]).size().items()}}

    # --- an independent check the model never sees: 82 signals label their channels
    # "Rad A - Count", "Rad A - Presence", "Rad B - Count" ...  One radar unit watches
    # one approach/lane group, so two channels of the same phase that share a letter
    # should land in the same inferred lane and two that do not should not.
    sp = fw.merge(off.drop_duplicates(["DeviceId", "Detector"]),
                  on=["DeviceId", "Detector"], how="left")
    sp["rad"] = sp.description.astype(str).str.extract(r"[Rr]ad\s*([A-D])\b")[0]
    r = sp[sp.rad.notna()]
    same_ok = diff_ok = same_n = diff_n = 0
    lanes_vs_letters = []
    for (did, per, ph), g in r.groupby(["DeviceId", "period", "pred_phase"]):
        if len(g) < 2:
            continue
        lab_, lane = g.rad.to_numpy(), g.lane.to_numpy()
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                if lab_[i] == lab_[j]:
                    same_n += 1; same_ok += int(lane[i] == lane[j])
                else:
                    diff_n += 1; diff_ok += int(lane[i] != lane[j])
        lanes_vs_letters.append((int(g.n_lanes.iloc[0]), int(g.rad.nunique())))
    lv = pd.DataFrame(lanes_vs_letters, columns=["n_lanes", "n_letters"])
    res["rad_description_check"] = {
        "n_phases": int(len(lv)),
        "same_letter_pairs": same_n,
        "same_letter_same_lane": round(same_ok / same_n, 4) if same_n else None,
        "diff_letter_pairs": diff_n,
        "diff_letter_diff_lane": round(diff_ok / diff_n, 4) if diff_n else None,
        "n_lanes_equals_n_letters": round(float((lv.n_lanes == lv.n_letters).mean()), 4)
        if len(lv) else None,
        "mean_n_lanes": round(float(lv.n_lanes.mean()), 3) if len(lv) else None,
        "mean_n_letters": round(float(lv.n_letters.mean()), 3) if len(lv) else None}

    # 10 phases to eyeball, with the technician's own channel text
    spot = fw[(fw.period == "stg")].merge(
        off.drop_duplicates(["DeviceId", "Detector"]),
        on=["DeviceId", "Detector"], how="left")
    spot = spot[spot.description.astype(str).str.len() > 3]
    g = spot.groupby(["DeviceName", "pred_phase"]).filter(lambda x: len(x) >= 3)
    sel = (g.groupby(["DeviceName", "pred_phase"]).size().sort_values(ascending=False)
           .head(200).index.tolist())
    rng = np.random.default_rng(0)
    pick = [sel[i] for i in rng.choice(len(sel), size=min(10, len(sel)),
                                       replace=False)]
    rows = []
    for dn, ph in pick:
        s = g[(g.DeviceName == dn) & (g.pred_phase == ph)]
        rows.append({"signal": dn, "pred_phase": int(ph),
                     "n_lanes": int(s.n_lanes.iloc[0]),
                     "detectors": [{"det": int(r.Detector), "lane": int(r.lane),
                                    "label": r.func5,
                                    "argmax": r.pred_argmax, "lane_decoded": r.pred_lane,
                                    "desc": str(r.description)[:40]}
                                   for _, r in s.iterrows()]})
    res["spot_check"] = rows
    json.dump(res, open(WORK / "a3_run.json", "w"), indent=1, default=str)
    log(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)
                    or "acc5_allwin" in v}, indent=1, default=str))
    log("n_lanes (full window, per phase): " + json.dumps(res["n_lanes_full_share"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["tune", "run"])
    ap.add_argument("--th", type=float, default=0.15)
    ap.add_argument("--mode", default="lane", choices=["lane", "cap"])
    a = ap.parse_args()
    globals()[f"stage_{a.stage}"](a)
