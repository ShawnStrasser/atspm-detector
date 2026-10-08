"""Note 115d parity: compare two runs (det / cand / ph parquet triples).
    python cmp115d.py <prefixA> <prefixB> <out.json>      prefix = .../<tag>_<profile>  (files <prefix>_det.parquet ...)
Decisions must be identical; every differing detector is listed with the columns that differ (a -> b)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, duckdb
DECISION = ["phase_pred", "phase_2nd", "function_pred", "status", "review_flag", "review_reason", "lanes",
            "phase_n_lanes", "distance_ft", "setback_confidence", "night_speed_mph", "night_speed_vehicles",
            "health_status", "health_reason", "health_bad_periods", "health_watch", "phase_guess", "function_guess",
            "n_actuations", "minutes_of_data", "n_candidate_phases"]
PROB = ["phase_prob", "phase_2nd_prob", "function_prob", "phase_guess_prob", "function_guess_prob", "phase_margin",
        "p_advance", "p_presence", "p_count", "p_yellow_red", "p_other", "p_mid", "p_bike", "lane_conf",
        "phase_n_lanes_conf", "health_score"]


def load(prefix):
    return {k: duckdb.sql(f"select * from '{Path(prefix + f'_{k}.parquet').as_posix()}'").df() for k in ("det", "cand", "ph")}


def canon(s):
    def f(v):
        if v is None or v is pd.NA or (isinstance(v, float) and v != v):
            return "<NA>"
        if isinstance(v, (bool, np.bool_)):
            return str(bool(v))
        try:
            return repr(round(float(v), 9))
        except (TypeError, ValueError):
            return str(v)
    return s.map(f)


def main():
    A, B, outp = sys.argv[1:4]
    a, b = load(A), load(B)
    common = set(a["det"].case) & set(b["det"].case)
    a = {k: v[v.case.isin(common)].copy() for k, v in a.items()}
    b = {k: v[v.case.isin(common)].copy() for k, v in b.items()}
    k = ["case", "DeviceId", "Detector"]
    for x in (a, b):
        x["det"]["Detector"] = x["det"].Detector.astype("int64")
    m = a["det"].merge(b["det"], on=k, how="outer", suffixes=("_a", "_b"), indicator=True)
    res = {"A": A, "B": B, "n_cases": len(common), "det_rows": [len(a["det"]), len(b["det"])],
           "det_unmatched": m.loc[m._merge != "both", k + ["_merge"]].astype(str).values.tolist()}
    m = m[m._merge == "both"]
    dec, rows = {}, {}
    for c in DECISION:
        ok = canon(m[c + "_a"]) == canon(m[c + "_b"])
        dec[c] = int((~ok).sum())
        for _, r in m[~ok].iterrows():
            rows.setdefault((r.case, r.DeviceId, int(r.Detector)), {})[c] = [str(r[c + "_a"])[:160], str(r[c + "_b"])[:160]]
    pr = {}
    for c in PROB:
        x = pd.to_numeric(m[c + "_a"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(m[c + "_b"], errors="coerce").to_numpy(float)
        d = np.abs(x - y)
        pr[c] = {"max": float(np.nanmax(d)) if np.isfinite(d).any() else 0.0,
                 "nan_mismatch": int((np.isnan(x) != np.isnan(y)).sum()), "n_gt_1e6": int((d > 1e-6).sum())}
    kc = ["case", "DeviceId", "Detector", "cand_phase"]
    ca = a["cand"].astype({"Detector": "int64", "cand_phase": "int64"})
    cb = b["cand"].astype({"Detector": "int64", "cand_phase": "int64"})
    mc = ca.merge(cb, on=kc, how="outer", suffixes=("_a", "_b"), indicator=True)
    res["cand_rows"] = [len(ca), len(cb)]
    res["cand_unmatched"] = int((mc._merge != "both").sum())
    mc = mc[mc._merge == "both"]
    for c in ("p0", "prob"):
        d = np.abs(mc[c + "_a"].to_numpy(float) - mc[c + "_b"].to_numpy(float))
        pr["cand_" + c] = {"max": float(np.nanmax(d)), "n_gt_1e6": int((d > 1e-6).sum())}
    kp = ["case", "DeviceId", "phase"]
    for x in (a, b):
        x["ph"]["phase"] = x["ph"].phase.astype("int64")
    mp = a["ph"].merge(b["ph"], on=kp, how="outer", suffixes=("_a", "_b"), indicator=True)
    res["ph_unmatched"] = int((mp._merge != "both").sum())
    mp = mp[mp._merge == "both"]
    res["phase_table_differ"] = {c: int((canon(mp[c + "_a"]) != canon(mp[c + "_b"])).sum())
                                 for c in ("n_detectors", "n_lanes", "n_lanes_conf", "lane_volumes_per_hour",
                                           "night_speed_mph", "night_speed_vehicles")}
    res.update(n_det=int(len(m)), decisions_differ=dec, prob=pr, n_det_changed=len(rows),
               changed=[{"case": c, "DeviceId": d, "Detector": t, "diff": v} for (c, d, t), v in sorted(rows.items())])
    Path(outp).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k_: res[k_] for k_ in ("A", "B", "n_cases", "n_det", "det_unmatched", "cand_unmatched",
                                             "ph_unmatched", "decisions_differ", "n_det_changed", "phase_table_differ")},
                     default=str))
    print("prob max:", {c: f"{v['max']:.1e}" for c, v in pr.items()})
    for r in res["changed"]:
        print(r["case"], r["DeviceId"][:8], r["Detector"], {c: v for c, v in r["diff"].items()})


if __name__ == "__main__":
    main()
