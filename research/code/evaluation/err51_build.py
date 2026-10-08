"""Note 51 step 1: one row per scored detector-window of function_v3e's six-fold OOF (frame v6e, 3-seed mean),
with the truth, both scoring sets (note 49: everything / realistic) and every attribute the error review breaks
down by (analysis only: technology, print tier, label source, health, permissive phase, phase correctness).

    python err51_build.py -> %DC_WORK%/trackA/err51/rows.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trackA"))
import v3_retrain as V  # noqa: E402

RUN = "run_ad6ea6959f_exclude_min5_clean_valnc_h3"
OUTD = V.DCW / "trackA" / "err51"
BEH = ["det_on_per_hour", "det_occ_frac", "det_dur_med", "det_dur_q90", "px_pulse_frac", "px_short_frac",
       "px_occ_red_all", "px_span_to_green", "px_hold_through_red", "px_g_first_med", "px_frac_on_first5_green",
       "on_lift_green", "occ_lift_green", "release_frac", "first_on_med", "yr_lift_yr", "yr_hit_yr",
       "px_corr_sib_max", "lagsib_best_lag", "lagsib_best_coinc", "sib_n", "cycle_mean", "green_share", "top_prob"]


def main():
    V.MIN_ON, V.CLEAN, V.REQ_VAL, V.NC_HIGH = 5, True, True, True
    V.set_frame("v6e")
    fr = pd.read_parquet(V.FEATS, columns=V.KEY + ["wgroup", "fold", "pred_phase", "det_n_on"] + BEH)
    fr["DeviceId"] = fr.DeviceId.str.lower()
    lk = set(pd.read_csv(V.DCW / "official" / "locked_v2.csv").DeviceId.str.lower())
    assert not fr.DeviceId.isin(lk).any()
    C7 = np.array(V.C7, object)
    Ps = []
    for s in (0, 1, 2):
        P, hv, got = V.load_oof(fr, V.OUT / RUN, "first.all.wi", s, range(6))
        assert len(got) == 6 and hv.all()
        Ps.append(P)
    P = np.mean(Ps, 0)
    fr["pred"] = C7[P.argmax(1)]
    fr["p_max"] = P.max(1)
    for i, c in enumerate(V.C7):
        fr[f"P_{c}"] = P[:, i]
    # truth exactly as eval49
    v3 = pd.read_parquet(V.LABELS_V3)
    v3["DeviceId"] = v3.DeviceId.str.lower()
    pf, src = v3.print_function, v3.source.astype("string")
    user = src.eq("user_ruling").fillna(False)
    high = v3.print_source.eq("print") & v3.print_confidence.eq("high") & pf.isin(V.C7)
    noprint = ~v3.print_source.eq("print").fillna(False) | pf.isna()
    truth = np.where(user, v3.function, np.where(high, pf, np.where(noprint & v3.func7_v2.notna(), v3.func7_v2, None)))
    tsrc = np.where(user, "user_ruling", np.where(high, "print_high", np.where(noprint & v3.func7_v2.notna(), "config", None)))
    t = pd.DataFrame({"DeviceId": v3.DeviceId, "Detector": v3.detector.astype(fr.Detector.dtype), "truth": truth,
                      "truth_src": tsrc, "validated": v3.validated.astype(str), "v3_source": v3.source,
                      "print_conf": v3.print_confidence, "print_function": pf, "config_function": v3.config_function,
                      "func7_v2": v3.func7_v2, "print_subtype": v3.print_subtype, "technology": v3.technology,
                      "lane_type": v3.lane_type, "tier": v3.tier, "unusual": v3.unusual_layout,
                      "phase_target": v3.phase_target, "phase_target_type": v3.phase_target_type,
                      "failed_checks": v3.failed_checks, "field_issue": v3.field_issue, "description": v3.description,
                      "DeviceName": v3.DeviceName})
    fr = fr.merge(t, on=["DeviceId", "Detector"], how="left")
    y = fr.truth.to_numpy(object)
    fr["setA"] = pd.notna(y) & np.isin(y, V.C7) & (fr.det_n_on >= 5).to_numpy()
    fr["setR"] = fr.setA & ~fr.validated.isin(["fail", "misconfigured"])
    fr = fr[fr.setA].reset_index(drop=True)
    fr["ok"] = fr.pred == fr.truth
    # phase correctness vs official timing (call phase, switch phase or an additional call phase = right)
    off = pd.read_parquet(V.DCW / "official" / "labels_official.parquet",
                          columns=["DeviceId", "Detector", "target_type", "target_num", "switch_phase",
                                   "additional_call_phases"])
    off["DeviceId"] = off.DeviceId.str.lower()
    off["Detector"] = off.Detector.astype(fr.Detector.dtype)
    off = off.drop_duplicates(["DeviceId", "Detector"])
    fr = fr.merge(off, on=["DeviceId", "Detector"], how="left")

    def okset(r):
        if r.target_type != "phase":
            return None
        s = {int(r.target_num)}
        if pd.notna(r.switch_phase) and r.switch_phase > 0:
            s.add(int(r.switch_phase))
        a = r.additional_call_phases
        if a is not None and not (isinstance(a, float) and np.isnan(a)):
            for x in (list(a) if not isinstance(a, str) else a.replace(";", ",").split(",")):
                try:
                    s.add(int(x))
                except (TypeError, ValueError):
                    pass
        return s
    sets = fr.apply(okset, axis=1)
    pp = fr.pred_phase.to_numpy()
    fr["phase_ok"] = [None if s is None or pd.isna(p) else (int(p) in s) for s, p in zip(sets, pp)]
    fr["phase_ok_strict"] = [None if s is None or pd.isna(p) else int(p) == int(t_)
                             for s, p, t_ in zip(sets, pp, fr.target_num)]
    # permissive / right turn on the LABELLED phase (label-check v4 table)
    pl = pd.read_parquet(V.DCW / "cabinet" / "label_check_pplt.parquet", columns=["sid", "p", "pplt", "how"])
    pl = pl[pl.pplt].assign(sid=lambda d: d.sid.str.lower()).drop_duplicates(["sid", "p"])
    lab_ph = pd.to_numeric(fr.phase_target.astype(str).str.extract(r"^P(\d+)$")[0], errors="coerce")
    key = fr.DeviceId + "|" + lab_ph.astype("Int64").astype(str)
    fr["permissive"] = key.isin(set(pl.sid + "|" + pl.p.astype(str)))
    fr["right_turn"] = fr.lane_type.eq("R")
    # health: v5 for Sept rows (stg), v3 for Dec rows (v5 was never run on Dec)
    h5 = pd.read_parquet(V.DCW / "health" / "health_v5.parquet", columns=["period", "DeviceId", "detector", "status"])
    h3 = pd.read_parquet(V.DCW / "health" / "health_v3.parquet", columns=["period", "DeviceId", "detector", "status"])
    h = pd.concat([h5[h5.period == "stg"], h3[h3.period == "dec"]])
    h = h.assign(DeviceId=h.DeviceId.str.lower(), Detector=h.detector.astype(fr.Detector.dtype)).drop_duplicates(
        ["period", "DeviceId", "Detector"])
    fr = fr.merge(h[["period", "DeviceId", "Detector", "status"]].rename(columns={"status": "health"}),
                  on=["period", "DeviceId", "Detector"], how="left")
    fr["health"] = fr.health.fillna("no_row")
    OUTD.mkdir(parents=True, exist_ok=True)
    fr.to_parquet(OUTD / "rows.parquet", index=False)
    print(len(fr), fr.setR.sum(), fr.DeviceId.nunique(), "acc A", fr.ok.mean(), "acc R", fr.ok[fr.setR].mean())


if __name__ == "__main__":
    main()
