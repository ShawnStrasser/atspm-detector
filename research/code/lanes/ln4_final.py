"""Lane output (note 42), step 4 -- final pair model + package check.

fit    3-seed LightGBM on every labelled pair-window (all folds), text models + lane_model.json
       (features, span prior from all signals, lam / beta = the value most folds picked in ln3)
       -> %DC_WORK%/lanes/model/.  Numpy evaluator (model/lgbm_numpy) vs lightgbm: max |diff|.
smoke  lane_output.lanes() on raw Sept-2026 events (the staging event cache) of N training
       signals, 30 min / 6 h / 24 h / full windows, with the OOF phase / function predictions of
       frame v6 as `predictions`; runtime per signal; the cues it computes from raw events vs
       ln1's (from the interval cache); run with lightgbm blocked from import.

    python ln4_final.py fit
    python ln4_final.py smoke [--n 20]
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: F401,E402
import argparse
import collections
import json
import time
import numpy as np
import pandas as pd
import lane_output as LO
import ln1_cues as L1

OUT = L1.OUT
MD = OUT / "model"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fit():
    import lightgbm as lgb
    import ln2_pairmodel as L2
    import ln3_decode as L3
    P, T, D = L2.load()
    lab = P.labelled.to_numpy()
    X, y = P.loc[lab, LO.FEATURES], P.loc[lab, "same_lane"].astype(int)
    MD.mkdir(parents=True, exist_ok=True)
    files, maxdiff = [], 0.0
    import lgbm_numpy as LN
    for s in (0, 1, 2):
        m = lgb.LGBMClassifier(random_state=s, **L2.PARAMS).fit(X, y)
        f = f"lane_pair_s{s}.txt"
        m.booster_.save_model(str(MD / f))
        files.append(f)
        a = m.predict_proba(X.iloc[:20000])[:, 1]
        b = LN.NumpyBooster(MD / f).predict(X.iloc[:20000])
        maxdiff = max(maxdiff, float(np.abs(a - b).max()))
    det = pd.read_parquet(OUT / "ln1_truth_det.parquet")
    ph = pd.read_parquet(OUT / "ln1_truth_phase.parquet")
    prior = L3.span_prior(det, ph, D, set(D.DeviceId))
    r = json.load(open(OUT / "ln3_decode.json"))
    lam, beta = collections.Counter(tuple(v) for v in r["picked"].values()).most_common(1)[0][0]
    meta = {"pair_models": files, "features": LO.FEATURES, "span_prior": prior, "lam": lam,
            "beta": beta, "role_pen": 4.0, "min_on": LO.MIN_ON,
            "trained_on": {"pair_windows": int(lab.sum()), "signals": int(P[lab].DeviceId.nunique()),
                           "windows": list(L1.WINS)},
            "numpy_vs_lightgbm_maxdiff": maxdiff}
    json.dump(meta, open(MD / "lane_model.json", "w"), indent=1)
    log(f"fit: {lab.sum()} pair-windows, lam {lam} beta {beta}, numpy parity {maxdiff:.2e}")


def smoke(n):
    import builtins
    real = builtins.__import__

    def block(name, *a, **k):
        if name.split(".")[0] in ("lightgbm", "torch", "sklearn", "scipy"):
            raise ImportError(f"blocked: {name}")
        return real(name, *a, **k)
    builtins.__import__ = block
    import pyarrow.dataset as ds
    D = pd.read_parquet(OUT / "ln1_dets.parquet")
    C = pd.read_parquet(OUT / "ln1_pairs.parquet")
    pm = LO.PairModel(MD)
    sig = sorted(D.DeviceId.unique())[::max(1, D.DeviceId.nunique() // n)][:n]
    ev_dir = L1.DCW / "official" / "stg" / "cache" / "events"
    res = collections.defaultdict(list)
    cue_diff = []
    for s in sig:
        ev = ds.dataset(str(ev_dir / f"DeviceId={s}")).to_table().to_pandas()
        ev["DeviceId"] = s
        for w in ("m30_a", "h6_a", "h24_a", "full66"):
            t0s, secs = L1.WINS[w]
            t0 = pd.Timestamp(t0s)
            t1 = t0 + pd.Timedelta(seconds=int(secs))
            e = ev[(ev.Timestamp >= t0) & (ev.Timestamp < t1)]
            pr = D[(D.DeviceId == s) & (D.win == w)].rename(
                columns={"det": "Detector", "pred_phase": "phase_pred", "func": "function_pred"})
            tt = time.time()
            ph_t, det_t = LO.lanes(e, pr, pm, start=t0, end=t1)
            res[w].append(time.time() - tt)
            if w == "full66" and len(det_t):
                # cues from raw events vs ln1 (interval cache): recompute through the package
                on = LO.on_times(e)
                grp, func = LO.eligible(pr.assign(phase_use=pr.phase_pred, func_use=pr.function_pred),
                                        {d: len(on.get((s, d), ([],))[0]) for d in pr.Detector},
                                        pm.min_on)
                c1 = C[(C.DeviceId == s) & (C.win == w) & C.same_pred]
                if len(c1):
                    loc = {d: on[(s, d)] for d in pr.Detector.astype(int)
                           if (s, d) in on and len(on[(s, d)][0]) >= pm.min_on}
                    pairs = [(a, b) for a, b in zip(c1.da, c1.db) if a in loc and b in loc]
                    c2 = LO.signal_pair_cues(loc, sorted(loc), pairs, t0.value / 1e9,
                                             t1.value / 1e9)
                    x = c1.merge(c2, on=["da", "db"], suffixes=("", "_r"))
                    for col in ("hp_off", "z0_ratio_all", "lead_exc_q", "n_a"):
                        cue_diff.append((col, float(np.nanmax(np.abs(x[col] - x[col + "_r"])))))
    builtins.__import__ = real
    out = {"signals": len(sig), "secs_per_signal": {w: round(float(np.mean(v)), 3)
                                                    for w, v in res.items()},
           "cue_max_abs_diff_raw_vs_cache": {c: max(v for k, v in cue_diff if k == c) for c in {k for k, _ in cue_diff}},
           "example_phase_table": ph_t.head(8).to_dict("records"),
           "example_detector_table": det_t.head(12).to_dict("records")}
    json.dump(out, open(OUT / "ln4_smoke.json", "w"), indent=1, default=str)
    log(json.dumps(out, default=str)[:2500])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fit", "smoke"])
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    fit() if a.stage == "fit" else smoke(a.n)
