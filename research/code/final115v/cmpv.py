"""Compare two verifier parity runs: python cmpv.py <tagA> <tagB> [--json out]"""
import os as _os
from pathlib import Path as _P
DCW = _os.environ.get("DC_WORK", str(_P.home() / "dc_work"))
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, duckdb
OUT = Path(DCW + r"\s115v\out")
DECISION = ["phase_pred", "phase_2nd", "function_pred", "status", "review_flag", "review_reason", "lanes",
            "phase_n_lanes", "distance_ft", "setback_confidence", "night_speed_mph", "night_speed_vehicles",
            "health_status", "health_reason", "health_bad_periods", "health_watch", "phase_guess", "function_guess",
            "n_actuations", "minutes_of_data", "n_candidate_phases"]
PROB = ["phase_prob", "phase_2nd_prob", "function_prob", "phase_guess_prob", "function_guess_prob", "phase_margin",
        "p_advance", "p_presence", "p_count", "p_yellow_red", "p_other", "p_mid", "p_bike", "lane_conf",
        "phase_n_lanes_conf", "health_score"]


def load(tag):
    return {k: duckdb.sql(f"select * from '{(OUT / f'{tag}_{k}.parquet').as_posix()}'").df() for k in ("det", "cand", "ph")}


def canon(s):
    def f(v):
        if v is None or v is pd.NA or (isinstance(v, float) and v != v):
            return "<NA>"
        try:
            x = float(v)
            return repr(round(x, 9))
        except (TypeError, ValueError):
            return str(v)
    return s.map(f)


def main():
    A, B = sys.argv[1], sys.argv[2]
    a, b = load(A), load(B)
    res = {"A": A, "B": B}
    k = ["case", "DeviceId", "Detector"]
    for x in (a, b):
        x["det"]["Detector"] = x["det"].Detector.astype("int64")
    m = a["det"].merge(b["det"], on=k, how="outer", suffixes=("_a", "_b"), indicator=True)
    res["det_rows"] = [len(a["det"]), len(b["det"])]
    res["det_unmatched"] = int((m._merge != "both").sum())
    m = m[m._merge == "both"]
    dec, cases, ex = {}, {}, {}
    for c in DECISION:
        ok = canon(m[c + "_a"]) == canon(m[c + "_b"])
        dec[c] = int((~ok).sum())
        if (~ok).any():
            r = m[~ok].iloc[0]
            ex[c] = [r.case, int(r.Detector), str(r[c + "_a"])[:100], str(r[c + "_b"])[:100]]
        for cs in m.loc[~ok, "case"].unique():
            cases.setdefault(cs, []).append(c)
    pr = {}
    for c in PROB:
        x, y = pd.to_numeric(m[c + "_a"], errors="coerce").to_numpy(float), pd.to_numeric(m[c + "_b"], errors="coerce").to_numpy(float)
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
    phd = {c: int((canon(mp[c + "_a"]) != canon(mp[c + "_b"])).sum())
           for c in ("n_detectors", "n_lanes", "n_lanes_conf", "lane_volumes_per_hour", "night_speed_mph", "night_speed_vehicles")}
    res.update(n_cases=int(m.case.nunique()), n_det=int(len(m)), decisions_differ=dec, examples=ex, prob=pr,
               phase_table_differ=phd, cases_with_diff=cases)
    s = json.dumps(res, indent=1, default=str)
    print(s)
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(s)


if __name__ == "__main__":
    main()
