"""Setback distance for every advance detector (note 41) -- research harness.

    python sb5_eval.py feats      # features for every predicted Advance / Mid (+ printed ones), 66 h
    python sb5_eval.py eval       # OOF (folds_v4) scoring, baselines, controls -> sb5_results.csv
    python sb5_eval.py final      # fit on every labelled training signal -> sb5_final.pkl

Inputs (never print labels): predicted phase = frame-v6 `pred_phase` (stage-12 ranker bag +
decoder OOF); predicted function = note-28 first.all.wi OOF (run d3f4e7ccbc, 3-seed mean, 7
classes).  Truth = printed single-number `distance_ft` of print Advance / Mid, unusual layout
out: cabinet/print_labels (training) + the released NEWTEST half (read from cabinet_locked,
filtered to newtest_released.csv).  locked_v2 signals asserted absent everywhere.
Pair model (same lane) = note-30 print-trained model, trained per fold on lp3 truth pairs of
the other folds.  Events: staging cache, 66 h (Fri 16:15 .. Mon 10:15).
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import pickle
import re
import sys
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sb5_setback as SB  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
OUT = DCW / "trackA" / "setback"
CACHE = DCW / "official" / "stg" / "cache"
FR = DCW / "trackA" / "v3" / "frame_v6"
RUN = FR / "run_d3f4e7ccbc_exclude_min5_clean_valnc"
C7 = ["Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other"]
T0 = pd.Timestamp("2026-09-18 16:15:00")
HOURS = 66.0
log = SB.log


def locked() -> set:
    return set(pd.read_csv(DCW / "official" / "locked_v2.csv").DeviceId.str.lower())


def connect():
    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'")
    con.execute("SET threads=4")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{(DCW / 'tmp').as_posix()}'")
    return con


def parse_dist(s):
    if s is None or pd.isna(s):
        return np.nan, "none"
    try:
        return float(str(s).strip()), "single"
    except ValueError:
        return np.nan, ("range" if re.fullmatch(r"-?\d+(\.\d+)?\s*-\s*\d+(\.\d+)?",
                                                 str(s).strip()) else "other")


def truth() -> pd.DataFrame:
    cols = "DeviceId, detector, function, technology, distance_ft, unusual_layout, lane_index"
    con = connect()
    rel = pd.read_csv(DCW / "official" / "newtest_released.csv")
    rel["dev"] = rel.DeviceId.str.lower()
    assert not set(rel.dev) & locked()
    con.register("rel", rel[["dev"]])
    a = con.sql(f"SELECT {cols}, 'train' AS src FROM "
                f"read_parquet('{(DCW / 'cabinet' / 'print_labels.parquet').as_posix()}')").df()
    b = con.sql(f"SELECT {cols}, 'released' AS src FROM read_parquet("
                f"'{(DCW / 'cabinet_locked' / 'print_labels.parquet').as_posix()}') "
                "WHERE lower(DeviceId) IN (SELECT dev FROM rel)").df()
    p = pd.concat([a, b], ignore_index=True)
    p["dev"] = p.DeviceId.str.lower()
    assert not p.dev.isin(locked()).any()
    p = p[p.unusual_layout != True].copy()
    d = p.distance_ft.map(parse_dist)
    p["dist"], p["dist_kind"] = [x[0] for x in d], [x[1] for x in d]
    p = p.rename(columns={"detector": "det", "function": "print_fn"})
    p["det"] = p.det.astype("int16")
    return p[["dev", "det", "print_fn", "technology", "dist", "dist_kind", "src"]]


def predictions() -> pd.DataFrame:
    fr = pd.read_parquet(FR / "feat_frame.parquet",
                         columns=["DeviceId", "Detector", "period", "wgroup", "fold",
                                  "pred_phase", "det_n_on"])
    P = np.zeros((len(fr), 7), np.float32)
    fo = fr.fold.to_numpy()
    for k in range(6):
        P[fo == k] = np.mean([np.load(RUN / f"P_first.all.wi_s{s}_f{k}.npy") for s in (0, 1, 2)], 0)
    m = ((fr.period == "stg") & (fr.wgroup == "full")).to_numpy()
    d = fr[m].reset_index(drop=True)
    P = P[m]
    d["dev"] = d.DeviceId.str.lower()
    d["fn"] = np.array(C7)[P.argmax(1)]
    d["p_mid"] = P[:, 4]
    d["p_adv"] = P[:, 0]
    d = d.rename(columns={"Detector": "det", "pred_phase": "phase", "det_n_on": "n_on"})
    d["det"] = d.det.astype("int16")
    f4 = pd.read_csv(DCW / "folds_v4.csv")
    f4["dev"] = f4.DeviceId.str.lower()
    d = d.drop(columns="fold").merge(f4[["dev", "fold"]], on="dev", how="inner")
    assert not d.dev.isin(locked()).any()
    return d[["dev", "det", "phase", "fn", "p_mid", "p_adv", "n_on", "fold"]]


def pair_model(exclude_fold=None):
    T = pd.read_parquet(DCW / "trackA" / "lp3_truth_scored.parquet")
    assert not T.DeviceId.str.lower().isin(locked()).any()
    T = T.drop(columns=["c_div", "r_shift", "lead_asym_abs"])
    T = L3derive(T)
    if exclude_fold is not None:
        T = T[T.fold != exclude_fold]
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=20, random_state=0).fit(
        T[SB.PAIR_COLS].to_numpy(float), T.same_lane.astype(int).to_numpy())


def L3derive(T):
    T = SB.L3.derive(T)
    T["lph_a"] = np.log1p(T.n_a / 66.0)
    T["lph_b"] = np.log1p(T.n_b / 66.0)
    T["log_ratio"] = np.abs(np.log1p(T.n_a) - np.log1p(T.n_b))
    return T


def stage_feats(chunk: int = 60):
    tr = truth()
    pr = predictions()
    lab = tr[tr.print_fn.isin(SB.TARGET) & (tr.dist_kind == "single")]
    pr = pr.merge(lab[["dev", "det"]].assign(lab=True), on=["dev", "det"], how="left")
    pr["is_t"] = pr.fn.isin(SB.TARGET) | pr.lab.fillna(False)
    log(f"{pr.dev.nunique()} signals, {len(pr)} detectors; predicted A/Mid "
        f"{pr.fn.isin(SB.TARGET).sum()}; targets {pr.is_t.sum()}")
    e0 = (T0 - pd.Timestamp("1970-01-01")).total_seconds()
    t1 = T0 + pd.Timedelta(hours=HOURS)
    parts = []
    for k in range(6):
        pm = pair_model(k)
        devs = sorted(pr[pr.fold == k].dev.unique())
        for i in range(0, len(devs), chunk):
            ds = devs[i:i + chunk]
            d = pr[pr.dev.isin(ds)]
            con = connect()
            con.register("dv", pd.DataFrame({"dev": ds}))
            con.execute(f"""CREATE TEMP TABLE iv AS
                SELECT lower(DeviceId) AS dev, Detector::SMALLINT AS det,
                       epoch_ms(t_on) / 1000.0 - {e0} AS t, dur,
                       (hour(t_on) >= 22 OR hour(t_on) < 6) AS quiet
                FROM read_parquet('{(CACHE / 'det_intervals.parquet').as_posix()}')
                WHERE lower(DeviceId) IN (SELECT dev FROM dv)
                  AND t_on >= TIMESTAMP '{T0}' AND t_on < TIMESTAMP '{t1}'""")
            con.execute(f"""CREATE TEMP TABLE cyc AS
                SELECT lower(DeviceId) AS dev, Phase::INT AS phase,
                       epoch_ms(green_start) / 1000.0 - {e0} AS t_g,
                       epoch_ms(coalesce(yellow_start, red_start, next_green)) / 1000.0 - {e0} AS t_y,
                       epoch_ms(coalesce(red_start, yellow_start, next_green)) / 1000.0 - {e0} AS t_red,
                       epoch_ms(next_green) / 1000.0 - {e0} AS t_next
                FROM read_parquet('{(CACHE / 'phase_cycles.parquet').as_posix()}')
                WHERE next_green IS NOT NULL AND lower(DeviceId) IN (SELECT dev FROM dv)""")
            F = SB.features(con, d[["dev", "det", "phase", "fn", "n_on", "p_mid", "is_t"]],
                            HOURS, pm)
            con.close()
            F["fold"] = k
            parts.append(F)
            log(f"fold {k} chunk {i // chunk}: {len(F)} targets, covered {F.covered.sum()}")
    F = pd.concat(parts, ignore_index=True)
    F = F.merge(tr[["dev", "det", "print_fn", "technology", "dist", "dist_kind", "src"]],
                on=["dev", "det"], how="left")
    F.to_parquet(OUT / "sb5_feat_h66.parquet", index=False)
    log(f"wrote sb5_feat_h66.parquet {F.shape}")


# ------------------------------------------------------------------------------- eval
NOPHYS = [f for f in SB.FEATS if f not in
          ("tau", "same", "p_same", "n_in", "pk_strength", "v_self", "tau_any", "pk_any",
           "p_same_any", "n_cand", "n_acc", "tau_rank", "d_user", "d_lfit", "d_any_lfit",
           "role_c")]


def metrics(y, p, lo=None, hi=None):
    ok = ~np.isnan(p)
    e = np.abs(p[ok] - y[ok])
    r = {"n": int(len(y)), "coverage": round(100 * ok.mean(), 1) if len(y) else np.nan,
         "medAE_ft": round(float(np.median(e)), 0) if len(e) else np.nan,
         "w25pct": round(100 * float(np.mean(e <= 0.25 * y[ok])), 1) if len(e) else np.nan,
         "w50ft": round(100 * float(np.mean(e <= 50)), 1) if len(e) else np.nan}
    if lo is not None and len(e):
        r["pi80_cov"] = round(100 * float(np.mean((y[ok] >= lo[ok]) & (y[ok] <= hi[ok]))), 1)
        r["pi_width_med"] = round(float(np.median(hi[ok] - lo[ok])), 0)
    return r


def stage_eval(seeds=(0, 1, 2)):
    F = pd.read_parquet(OUT / "sb5_feat_h66.parquet")
    L = F.dist_kind.eq("single") & F.print_fn.isin(SB.TARGET)
    E = pd.DataFrame(index=F.index)
    for k in range(6):
        tr = F[L & (F.fold != k)]
        te = F.fold == k
        Ft = F[te]
        E.loc[te, "median"] = tr.dist.median()
        # approach type = phase green-share tercile (training cut points)
        cuts = tr.green_share.quantile([1 / 3, 2 / 3]).to_numpy()
        at = lambda s: np.digitize(s.fillna(s.median()), cuts)
        med_at = tr.groupby(at(tr.green_share)).dist.median()
        E.loc[te, "approach_median"] = pd.Series(at(Ft.green_share), index=Ft.index).map(med_at)
        fit = SB.fit_all(tr, seeds)
        A = SB.apply_fit(Ft, fit)
        for c in ("p10", "p50", "p90", "d_lfit", "d_user", "covered"):
            E.loc[te, c] = A[c]
        E.loc[te, "L_fit"] = fit["L_fit"]
        E.loc[te, "cqr"] = fit["cqr"]
        for name, fs in (("std", SB.STD_FEATS), ("nophys", NOPHYS)):
            A2 = SB.apply_fit(Ft, SB.fit_all(tr, seeds, fs, inner=False))
            E.loc[te, f"{name}_p50"] = A2.p50
        sh = tr.copy()
        sh["dist"] = np.random.default_rng(k).permutation(sh.dist.to_numpy())
        E.loc[te, "shuffled_p50"] = SB.apply_fit(Ft, SB.fit_all(sh, (0,), inner=False)).p50
        if seeds == (0, 1, 2):
            E.loc[te, "seeds345_p50"] = SB.apply_fit(Ft, SB.fit_all(tr, (3, 4, 5), inner=False)).p50
        log(f"fold {k}: L_fit {fit['L_fit']:.1f} ft, cqr {fit['cqr']:.3f}, "
            f"{int(L[te].sum())} labelled")
    E["hybrid"] = np.where(E.covered.eq(True) & E.d_lfit.notna(), E.d_lfit, E.p50)
    F = pd.concat([F.drop(columns=["d_user", "covered"]), E], axis=1)
    F.to_parquet(OUT / "sb5_oof_h66.parquet", index=False)
    report(F)


def report(F):
    adv = (F.print_fn == "Advance") & (F.dist_kind == "single")
    # production answer: estimate if predicted A/Mid, 0 if predicted stop-bar, else none
    prod = np.where(F.fn.isin(SB.TARGET), F.p50, np.where(F.fn.isin(SB.STOPBAR), 0.0, np.nan))
    F = F.assign(prod=prod, prod_lo=np.where(F.fn.isin(SB.STOPBAR), 0.0, F.p10),
                 prod_hi=np.where(F.fn.isin(SB.STOPBAR), 0.0, F.p90))
    F["band"] = pd.cut(F.dist, [-1, 100, 200, 1e4], labels=["<=100", "101-200", ">200"])
    F["tech"] = F.technology.fillna("unknown")
    cuts = conf_cuts(F)
    F["conf"] = SB.confidence(F.p10, F.p50, F.p90, cuts)
    print("confidence cut points (log P90/P10 terciles over predicted A/Mid):", cuts)
    rows = []
    def add(mask, label, est, lo=None, hi=None):
        m = mask.to_numpy()
        r = metrics(F.dist.to_numpy()[m], F[est].to_numpy(float)[m],
                    None if lo is None else F[lo].to_numpy(float)[m],
                    None if hi is None else F[hi].to_numpy(float)[m])
        rows.append({"subset": label, "estimator": est, **r})
    for est, lo, hi in (("prod", "prod_lo", "prod_hi"), ("p50", "p10", "p90"),
                        ("hybrid", None, None), ("median", None, None),
                        ("approach_median", None, None), ("std_p50", None, None),
                        ("nophys_p50", None, None), ("shuffled_p50", None, None),
                        ("seeds345_p50", None, None), ("d_lfit", None, None),
                        ("d_user", None, None)):
        if est not in F:
            continue
        add(adv, "Advance all", est, lo, hi)
        add(adv & F.fn.isin(SB.TARGET), "Advance, predicted A/Mid", est, lo, hi)
        add(adv & F.covered.eq(True), "Advance, same-lane partner", est, lo, hi)
        add(adv & ~F.covered.eq(True), "Advance, no partner", est, lo, hi)
        for t in ("loop", "radar", "video", "unknown"):
            add(adv & (F.tech == t), f"tech {t}", est, lo, hi)
        for b in ("<=100", "101-200", ">200"):
            add(adv & (F.band == b), f"dist {b}", est, lo, hi)
        add((F.print_fn == "Mid") & (F.dist_kind == "single"), "Mid", est, lo, hi)
        for c in ("high", "medium", "low"):
            add(adv & F.fn.isin(SB.TARGET) & (F.conf == c), f"confidence {c}", est, lo, hi)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "sb5_results.csv", index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 400)
    print(R.to_string(index=False))
    print("predicted function of printed Advance:",
          F[adv].fn.value_counts().to_dict())
    print("targets predicted A/Mid:", int(F.fn.isin(SB.TARGET).sum()), "covered:",
          int((F.fn.isin(SB.TARGET) & F.covered.eq(True)).sum()))
    print("print distance top values (Advance):",
          F[adv].dist.value_counts(normalize=True).head(6).round(3).to_dict())


def conf_cuts(F) -> tuple:
    """Terciles of log(P90 / P10) over every predicted Advance / Mid (no truth used)."""
    m = F.fn.isin(SB.TARGET) & F.p50.notna()
    w = np.log(np.maximum(F.p90[m], 15.0) / np.maximum(F.p10[m], 15.0))
    return tuple(float(x) for x in np.quantile(w, [1 / 3, 2 / 3]))


def stage_final():
    F = pd.read_parquet(OUT / "sb5_feat_h66.parquet")
    L = F.dist_kind.eq("single") & F.print_fn.isin(SB.TARGET)
    fit = SB.fit_all(F[L], (0, 1, 2))
    fit["conf_cuts"] = conf_cuts(pd.read_parquet(OUT / "sb5_oof_h66.parquet"))
    pk = {"fit": fit, "pair_model": pair_model(None), "hours_trained": HOURS,
          "n_train": int(L.sum()), "note": "research/notes/41_setback_all_advance.md"}
    pickle.dump(pk, open(OUT / "sb5_final.pkl", "wb"))
    log(f"wrote sb5_final.pkl: L_fit {fit['L_fit']:.1f}, cqr {fit['cqr']:.3f}, n {L.sum()}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    st = sys.argv[1] if len(sys.argv) > 1 else "eval"
    {"feats": stage_feats, "eval": stage_eval, "final": stage_final,
     "report": lambda: report(pd.read_parquet(OUT / "sb5_oof_h66.parquet"))}[st]()
