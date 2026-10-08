"""Note 93 B: Yellow_Red zones on a phase with no Count zone -- do they matter, are they real?

Reads the saved v4f OOF rows (s93/func_rows.parquet), the feature frame (f76/frame_v6e_c, behaviour features of the
same detector-windows), labels v4o and the member list (g93_yr.py members).  Windows >= 30 min only.  No model scoring
beyond the saved OOF.  -> %DC_WORK%/s93/b93.json + s93/b93_phases.parquet (per YR-no-Count phase: every detector on it)

    python b93_yr.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import rpath  # noqa: F401,E402
import cand64 as C  # noqa: E402

DCW = Path(os.environ.get("DC_WORK", Path.home() / "dc_work"))
S93 = DCW / "s93"
REPO = CODE.parents[1]
FEAT = DCW / "final_v3_work" / "f76" / "frame_v6e_c" / "feat_frame.parquet"
BEH = {  # feature -> plain meaning
    "px_pulse_frac": "share of ONs lasting one 0.1-s tick (pulse)",
    "det_dur_med": "median ON length (s)",
    "first_on_med": "median lag of the first ON after begin-green (s)",
    "px_g_first_le2": "share of greens with an ON in the first 2 s",
    "px_g_first_ge6": "share of greens whose first ON is >= 6 s in",
    "f_on_red": "share of ONs during red",
    "px_hold_through_red": "held ON through red",
    "yr_f_on_yr": "share of ONs during yellow + red clearance",
    "det_on_per_hour": "ONs per hour",
}
K = ["DeviceId", "Detector", "period", "win"]


def boot_med(v, sig, n=1000, seed=93):
    """median over detectors with a signal-bootstrap 95 % CI."""
    v, sig = np.asarray(v, float), np.asarray(sig)
    ok = ~np.isnan(v)
    v, sig = v[ok], sig[ok]
    if len(v) == 0:
        return [None, None, None, 0]
    u = np.unique(sig)
    idx = {s: np.where(sig == s)[0] for s in u}
    rng = np.random.default_rng(seed)
    bs = [np.median(np.concatenate([v[idx[s]] for s in rng.choice(u, len(u))])) for _ in range(n)]
    return [round(float(np.median(v)), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3),
            int(len(v))]


def main():
    L = pd.read_parquet(REPO / "research/labels/function_labels_v4o.parquet")
    L["k"] = L.DeviceId.str.lower()
    M = pd.read_parquet(S93 / "yr_nc_members.parquet")
    M["k"] = M.DeviceId.str.lower()
    fr = pd.read_parquet(S93 / "func_rows.parquet")
    fr["k"] = fr.DeviceId.str.lower()
    ge = fr.wgroup.isin(C.GE30).to_numpy()
    cols = list(BEH)
    F = pd.read_parquet(FEAT, columns=K + cols)
    assert len(F) == len(fr) and (F.Detector.to_numpy() == fr.Detector.to_numpy()).all() and \
        (F.win.to_numpy() == fr.win.to_numpy()).all()
    for c in cols:
        fr[c] = F[c].to_numpy()
    lab = L[["k", "detector", "phase_target", "truth_v3s", "technology", "DeviceName"]].rename(
        columns={"detector": "Detector", "truth_v3s": "truth_lab"})
    fr = fr.merge(lab.astype({"Detector": fr.Detector.dtype}), on=["k", "Detector"], how="left")
    mk = set(zip(M.k, M.detector.astype(int)))
    fr["yrnc"] = [(a, int(b)) in mk for a, b in zip(fr.k, fr.Detector)]
    # groups (label view: truth where it exists, else the member's training label)
    tr = fr.truth_v3s.astype("string").fillna("")
    fr["grp"] = np.select([fr.yrnc & tr.eq("Yellow_Red"), fr.yrnc & tr.eq(""), tr.eq("Yellow_Red"), tr.eq("Count"),
                           tr.eq("Presence"), tr.eq("Advance")],
                          ["YR_noCount", "YR_noCount_untruthed", "YR_withCount", "Count", "Presence", "Advance"], "")
    res = {}
    # ---------------------------------------------------------------- accuracy (>= 30 min, E and R), what the model says
    acc = {}
    for g in ("YR_noCount", "YR_withCount"):
        m = ge & fr.grp.eq(g).to_numpy()
        acc[g] = {s: C.acc_ci(fr[f"ok{s}"].to_numpy()[m & fr[f"ok{s}"].notna().to_numpy()],
                              fr.DeviceId.to_numpy()[m & fr[f"ok{s}"].notna().to_numpy()]) + [int((m & fr[f"ok{s}"].notna()).sum())]
                  for s in ("E", "R")}
        acc[g]["signals"] = int(fr.DeviceId[m & fr.okE.notna()].nunique())
        acc[g]["pred"] = fr.pred_func[m & fr.okE.notna().to_numpy()].value_counts(normalize=True).round(3).to_dict()
    m = ge & fr.grp.eq("YR_noCount_untruthed").to_numpy()
    acc["YR_noCount_untruthed_pred"] = fr.pred_func[m].value_counts(normalize=True).round(3).to_dict()
    acc["YR_noCount_untruthed_rows"] = int(m.sum())
    # by technology for the YR groups
    for g in ("YR_noCount", "YR_withCount"):
        for t in ("video", "radar", "loop"):
            mm = ge & fr.grp.eq(g).to_numpy() & fr.technology.eq(t).fillna(False).to_numpy() & fr.okE.notna().to_numpy()
            if mm.sum():
                acc[f"{g}_{t}"] = C.acc_ci(fr.okE.to_numpy()[mm], fr.DeviceId.to_numpy()[mm]) + [int(mm.sum()),
                                                                                               int(fr.DeviceId[mm].nunique())]
    res["accuracy_ge30"] = acc
    # share of the YR scoring rows and the headline effect of dropping them from scoring
    sc = ge & fr.okE.notna().to_numpy()
    res["share_of_E_rows"] = {"YR_noCount": int((sc & fr.grp.eq("YR_noCount").to_numpy()).sum()), "all": int(sc.sum())}
    res["headline_E"] = {"all": C.acc_ci(fr.okE.to_numpy()[sc], fr.DeviceId.to_numpy()[sc]),
                         "without_YR_noCount": C.acc_ci(fr.okE.to_numpy()[sc & ~fr.yrnc.to_numpy()],
                                                        fr.DeviceId.to_numpy()[sc & ~fr.yrnc.to_numpy()])}
    # ---------------------------------------------------------------- behaviour: per detector median over >= 30 min windows
    d = fr[ge & fr.grp.ne("").to_numpy()].groupby(["k", "Detector", "grp"], as_index=False)[cols].median()
    sig = d.k.to_numpy()
    beh = {}
    for c in cols:
        beh[c] = {g: boot_med(d[c][d.grp == g], sig[d.grp == g]) for g in
                  ("YR_noCount", "YR_noCount_untruthed", "YR_withCount", "Count", "Presence")}
    res["behaviour_median_per_detector"] = beh
    # by technology (YR groups and Count)
    dt = d.merge(lab[["k", "Detector", "technology"]].drop_duplicates(["k", "Detector"]).astype({"Detector": d.Detector.dtype}),
                 on=["k", "Detector"], how="left")
    beht = {}
    for t in ("video", "radar"):
        for c in ("px_pulse_frac", "first_on_med", "px_g_first_le2", "f_on_red", "yr_f_on_yr"):
            beht[f"{t}:{c}"] = {g: boot_med(dt[c][(dt.grp == g) & (dt.technology == t)],
                                            dt.k[(dt.grp == g) & (dt.technology == t)])
                                for g in ("YR_noCount", "YR_noCount_untruthed", "YR_withCount", "Count")}
    res["behaviour_by_tech"] = beht
    # ---------------------------------------------------------------- volume vs phase-mates (labelled phase, same window)
    w = fr[ge].copy()
    w["phk"] = w.k + "|" + w.phase_target.astype("string").fillna("?")
    keyw = ["phk", "period", "win"]
    for role in ("Count", "Presence", "Advance"):
        s = w[w.truth_lab.eq(role)].groupby(keyw).det_n_on.sum().rename(f"sum_{role}")
        w = w.merge(s, left_on=keyw, right_index=True, how="left")
    vol = {}
    for g in ("YR_noCount", "YR_noCount_untruthed", "YR_withCount", "Count"):
        x = w[w.grp.eq(g)]
        for role in ("Count", "Presence", "Advance"):
            r = (x.det_n_on / x[f"sum_{role}"]).where(x[f"sum_{role}"] > 20)
            dd = pd.DataFrame({"k": x.k, "Detector": x.Detector, "r": r}).groupby(["k", "Detector"]).r.median()
            vol[f"{g}/{role}"] = boot_med(dd.to_numpy(), dd.index.get_level_values(0).to_numpy())
    res["volume_ratio_to_phase_sum"] = vol
    # ---------------------------------------------------------------- every detector on each YR-no-Count phase
    ph = M[["k", "DeviceName", "phase_target"]].drop_duplicates()
    pkeys = set(zip(ph.k, ph.phase_target))
    x = fr[ge].copy()
    x["lab_phase"] = x.phase_target
    # unlabelled detectors: assign them to a phase by their predicted phase (majority over windows)
    x["php"] = "P" + x.pred_phase.astype("Int64").astype("string")
    x["phase_use"] = x.lab_phase.where(x.lab_phase.notna(), x.php)
    x = x[[(a, b) in pkeys for a, b in zip(x.k, x.phase_use)]]
    agg = x.groupby(["k", "Detector"], as_index=False).agg(
        DeviceName=("DeviceName", "first"), phase=("phase_use", lambda s: s.mode().iat[0]),
        lab_phase=("lab_phase", "first"), truth=("truth_v3s", "first"), technology=("technology", "first"),
        yrnc=("yrnc", "first"), n_win=("win", "size"), on_per_h=("det_on_per_hour", "median"),
        p_count=("p_Count", "mean"), p_yr=("p_Yellow_Red", "mean"), p_pres=("p_Presence", "mean"),
        p_adv=("p_Advance", "mean"), p_other=("p_Other", "mean"),
        pred_top=("pred_func", lambda s: s.value_counts().index[0]),
        pred_count_share=("pred_func", lambda s: float((s == "Count").mean())),
        pulse=("px_pulse_frac", "median"), dur_med=("det_dur_med", "median"), first_on=("first_on_med", "median"),
        g_le2=("px_g_first_le2", "median"), on_red=("f_on_red", "median"), on_yr=("yr_f_on_yr", "median"))
    agg["DeviceName"] = agg.DeviceName.fillna(agg.k.map(dict(zip(L.k, L.DeviceName))))
    lt = L[["k", "detector", "function", "print_function", "print_subtype", "print_confidence", "description", "source"]]
    agg = agg.merge(lt.rename(columns={"detector": "Detector"}).astype({"Detector": agg.Detector.dtype}),
                    on=["k", "Detector"], how="left")
    agg = agg.sort_values(["DeviceName", "phase", "Detector"])
    agg.to_parquet(S93 / "b93_phases.parquet", index=False)
    cand = agg[~agg.yrnc & ((agg.pred_count_share >= .5) | (agg.p_count >= .4))]
    res["unlabelled_count_candidates"] = cand[["DeviceName", "phase", "Detector", "truth", "function", "print_function",
                                               "description", "p_count", "pred_count_share", "pulse", "first_on",
                                               "on_red"]].round(3).to_dict("records")
    res["phases_seen"] = int(agg.groupby(["k", "phase"]).ngroups)
    # model's own view per YR-no-Count detector (does it call them YR / Count / other?)
    yy = agg[agg.yrnc]
    res["yrnc_detectors"] = {"n": int(len(yy)), "pred_top": yy.pred_top.value_counts().to_dict(),
                             "mean_p_yr": round(float(yy.p_yr.mean()), 3), "mean_p_count": round(float(yy.p_count.mean()), 3)}
    json.dump(res, open(S93 / "b93.json", "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
