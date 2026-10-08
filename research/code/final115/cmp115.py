"""Note 115 parity comparison of two parity runs (par115.py outputs).

    python cmp115.py <runA_profile> <runB_profile> [--json out.json]
Decisions (phase / function / lanes / setback / health / status text) must be identical; probabilities are compared by
max |diff|.  Reports per column and the cases where anything differs."""
import json, sys
import os
from pathlib import Path
W = Path(os.environ.get("DC_WORK") or Path.home() / "dc_work")
import numpy as np, pandas as pd
OUT = Path(str(W / r"s115\parity\out"))
DECISION = ["phase_pred", "phase_2nd", "function_pred", "status", "review_flag", "review_reason", "lanes",
            "phase_n_lanes", "distance_ft", "setback_confidence", "night_speed_mph", "night_speed_vehicles",
            "health_status", "health_reason", "health_bad_periods", "health_watch", "phase_guess", "function_guess",
            "n_actuations", "minutes_of_data", "n_candidate_phases"]
PROB = ["phase_prob", "phase_2nd_prob", "function_prob", "phase_guess_prob", "function_guess_prob", "phase_margin",
        "p_advance", "p_presence", "p_count", "p_yellow_red", "p_other", "p_mid", "p_bike", "lane_conf",
        "phase_n_lanes_conf", "health_score"]


def load(tag):
    return {k: pd.read_parquet(OUT / f"{tag}_{k}.parquet") for k in ("det", "cand", "ph")}


def same_obj(a, b):
    na, nb = a.isna(), b.isna()
    return (na & nb) | (~na & ~nb & (a.astype(str) == b.astype(str)))


def main():
    A, B = sys.argv[1], sys.argv[2]
    a, b = load(A), load(B)
    common = set(a["det"].case) & set(b["det"].case)
    a = {k: v[v.case.isin(common)] for k, v in a.items()}
    b = {k: v[v.case.isin(common)] for k, v in b.items()}
    res = {"A": A, "B": B}
    k = ["case", "DeviceId", "Detector"]
    m = a["det"].merge(b["det"], on=k, how="outer", suffixes=("_a", "_b"), indicator=True)
    res["det_rows"] = [int(len(a["det"])), int(len(b["det"]))]
    res["det_unmatched"] = int((m._merge != "both").sum())
    m = m[m._merge == "both"]
    dec, cases = {}, {}
    for c in DECISION:
        ok = same_obj(m[c + "_a"], m[c + "_b"])
        dec[c] = int((~ok).sum())
        for cs in m.loc[~ok, "case"].unique():
            cases.setdefault(cs, []).append(c)
    pr = {}
    for c in PROB:
        x, y = m[c + "_a"].astype(float).to_numpy(), m[c + "_b"].astype(float).to_numpy()
        nan_mis = int((np.isnan(x) != np.isnan(y)).sum())
        d = np.abs(x - y)
        pr[c] = {"max": float(np.nanmax(d)) if np.isfinite(d).any() else 0.0, "nan_mismatch": nan_mis,
                 "n_gt_1e6": int((d > 1e-6).sum())}
    kc = ["case", "DeviceId", "Detector", "cand_phase"]
    ca, cb = a["cand"].astype({"Detector": "int64", "cand_phase": "int64"}), b["cand"].astype({"Detector": "int64", "cand_phase": "int64"})
    mc = ca.merge(cb, on=kc, how="outer", suffixes=("_a", "_b"), indicator=True)
    res["cand_rows"] = [int(len(ca)), int(len(cb))]
    res["cand_unmatched"] = int((mc._merge != "both").sum())
    mc = mc[mc._merge == "both"]
    for c in ("p0", "prob"):
        d = np.abs(mc[c + "_a"].to_numpy(float) - mc[c + "_b"].to_numpy(float))
        pr["cand_" + c] = {"max": float(np.nanmax(d)), "n_gt_1e6": int((d > 1e-6).sum())}
    kp = ["case", "DeviceId", "phase"]
    mp = a["ph"].merge(b["ph"], on=kp, how="outer", suffixes=("_a", "_b"), indicator=True)
    res["ph_unmatched"] = int((mp._merge != "both").sum())
    mp = mp[mp._merge == "both"]
    phd = {}
    for c in ("n_detectors", "n_lanes", "n_lanes_conf", "lane_volumes_per_hour", "night_speed_mph", "night_speed_vehicles"):
        phd[c] = int((~same_obj(mp[c + "_a"], mp[c + "_b"])).sum())
    res.update(decisions_differ=dec, prob=pr, phase_table_differ=phd, cases_with_diff=cases,
               n_cases=int(m.case.nunique()), n_det=int(len(m)))
    s = json.dumps(res, indent=1)
    print(s)
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(s)


if __name__ == "__main__":
    main()
