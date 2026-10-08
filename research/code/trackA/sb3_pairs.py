"""Setback distance, physics method (note 39), step 1 -- same-lane pairing from data.

For every timing phase that carries a printed Advance / Mid detector (sb1 truth, training/dev
signals only), compute note 30's pair cues (lr2_pairs.SQL_CORR / SQL_MIN / cues) for every pair
of detectors on that phase, in the chosen window, and score P(same lane) with the note-30
print-trained gradient-boosting pair model, out-of-fold (folds_v3: a pair is scored by a model
trained on the print truth pairs of the other folds).  Counts enter per hour so a model trained
on the 66 h truth cues applies to shorter windows; the A3 cue (66 h only) is left out.

Phase grouping = official timing phase (in production: the predicted phase).  No channel
number is a feature; lane labels are used for training / validation only.

    python sb3_pairs.py [--hours 66] [--start 0]
Output: %DC_WORK%/trackA/setback/sb3_pairs_h{H}_s{S}.parquet  (dev, da, db, p_same, cues)
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import lr_common as C
import lr2_pairs as L2
import lp3_auc as L3
import sb1_setback as S1

OUT = C.DCW / "trackA" / "setback"
CUEC = L3.ALLC
CTX = ["lph_a", "lph_b", "log_ratio", "n_det_phase"]


def locked_all() -> set[str]:
    """Old hold-out lists (43 TEST + all 143 NEWTEST) plus locked_v2: every one excluded."""
    v2 = pd.read_csv(C.DCW / "official" / "locked_v2.csv").DeviceId.str.lower()
    return C.locked_ids() | set(v2)


def phase_dets() -> pd.DataFrame:
    tgt, _ = S1.truth()
    ph = tgt[["dev", "phase"]].drop_duplicates()
    w = pd.read_parquet(C.REPO / "research" / "labels" / "function_labels_v3.parquet",
                        columns=["DeviceId", "detector", "phase_target", "phase_target_type"])
    w["dev"] = w.DeviceId.str.lower()
    w = w[(w.phase_target_type == "phase")].copy()
    w["phase"] = w.phase_target.str[1:].astype(int)
    d = w.merge(ph, on=["dev", "phase"])
    d = d.rename(columns={"detector": "det", "phase": "grp"})[["dev", "det", "grp"]]
    d["det"] = d.det.astype("int16")
    d = d.drop_duplicates(["dev", "det"])
    assert not d.dev.isin(locked_all()).any()
    return d


def add_ctx(P: pd.DataFrame, hours: float) -> pd.DataFrame:
    P = L3.derive(P)
    P["lph_a"] = np.log1p(P.n_a / hours)
    P["lph_b"] = np.log1p(P.n_b / hours)
    P["log_ratio"] = np.abs(np.log1p(P.n_a) - np.log1p(P.n_b))
    return P


def cues(hours: float, start: float) -> pd.DataFrame:
    dets = phase_dets()
    C.log(f"{dets.dev.nunique()} signals, {len(dets)} detectors on target phases")
    cache = C.DCW / "official" / "stg" / "cache"
    t0 = S1.T0 + pd.Timedelta(hours=start)
    t1 = t0 + pd.Timedelta(hours=hours)
    secs = int(hours * 3600)
    e0 = (t0 - pd.Timestamp("1970-01-01")).total_seconds()
    con = S1.connect()
    con.execute("SET preserve_insertion_order=false")
    con.register("dets_df", dets)
    con.execute("CREATE TEMP TABLE dets AS SELECT * FROM dets_df")
    con.execute("""CREATE TEMP TABLE prs AS SELECT a.dev, a.det AS da, b.det AS db
                   FROM dets a JOIN dets b ON a.dev = b.dev AND a.grp = b.grp
                   AND a.det < b.det""")
    di = (cache / "det_intervals.parquet").as_posix()
    con.execute(f"""CREATE TEMP TABLE onw AS
        SELECT lower(i.DeviceId) AS dev, i.Detector::SMALLINT AS det,
               epoch_ms(i.t_on) / 1000.0 - {e0} AS t,
               (hour(i.t_on) >= 22 OR hour(i.t_on) < 6) AS quiet
        FROM read_parquet('{di}') i
        SEMI JOIN dets d ON lower(i.DeviceId) = d.dev AND i.Detector = d.det
        WHERE i.t_on >= TIMESTAMP '{t0}' AND i.t_on < TIMESTAMP '{t1}'""")
    nmin = 20 if hours >= 24 else 5
    nq = con.sql(f"""SELECT dev, det, count(*) AS n, count(*) FILTER (quiet) AS nq
                    FROM onw GROUP BY ALL HAVING count(*) >= {nmin}""").df()
    con.register("nq_df", nq)
    con.execute("CREATE TEMP TABLE nq AS SELECT * FROM nq_df")
    con.execute("DELETE FROM onw WHERE NOT EXISTS "
                "(SELECT 1 FROM nq WHERE nq.dev = onw.dev AND nq.det = onw.det)")
    corr = con.sql(L2.SQL_CORR).df()
    mm = con.sql(L2.SQL_MIN.format(nm=int(secs // 60))).df()
    P = L2.cues(corr, nq).merge(mm, on=["dev", "da", "db"], how="left")
    P = P.replace([np.inf, -np.inf], np.nan)
    act = dets.merge(nq[["dev", "det"]], on=["dev", "det"])
    nd = act.groupby(["dev", "grp"]).size().rename("n_det_phase").reset_index()
    P = P.merge(dets.rename(columns={"det": "da"}), on=["dev", "da"]).merge(
        nd, on=["dev", "grp"], how="left")
    return add_ctx(P, hours)


def main(hours: float, start: float):
    P = cues(hours, start)
    f = pd.read_csv(C.DCW / "folds_v3.csv")
    f["dev"] = f.DeviceId.str.lower()
    P = P.merge(f[["dev", "fold"]], on="dev", how="inner")
    T = pd.read_parquet(C.WORK / "lp3_truth_scored.parquet")
    T = add_ctx(T.drop(columns=["c_div", "r_shift", "lead_asym_abs"]), 66.0)
    assert not T.DeviceId.isin(locked_all()).any()
    cols = CUEC + CTX
    X, y = T[cols].to_numpy(float), T.same_lane.to_numpy().astype(int)
    P["p_same"] = np.nan
    oofT = np.full(len(T), np.nan)
    for k in sorted(T.fold.unique()):
        tr = (T.fold != k).to_numpy()
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                           max_leaf_nodes=15, min_samples_leaf=20,
                                           random_state=0).fit(X[tr], y[tr])
        oofT[~tr] = m.predict_proba(X[~tr])[:, 1]
        sel = (P.fold == k).to_numpy()
        if sel.any():
            P.loc[sel, "p_same"] = m.predict_proba(P.loc[sel, cols].to_numpy(float))[:, 1]
    C.log(f"pair model OOF AUC on the 66 h truth pairs: {roc_auc_score(y, oofT):.4f} "
          f"(multi {roc_auc_score(y[T.multi], oofT[T.multi]):.4f})")
    keep = ["dev", "da", "db", "grp", "fold", "n_a", "n_b", "p_same", "hp_off", "lead_lag_q",
            "lead_exc_q"]
    fn = OUT / f"sb3_pairs_h{hours:g}_s{start:g}.parquet"
    P[keep].to_parquet(fn, index=False)
    C.log(f"wrote {fn}: {len(P)} pairs, {P.dev.nunique()} signals")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=66)
    ap.add_argument("--start", type=float, default=0)
    a = ap.parse_args()
    main(a.hours, a.start)
