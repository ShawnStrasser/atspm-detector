"""Note 48: function head of the final_v3 candidate v2 (= function_v3d recipe) vs the final_v2 head family, six folds OOF,
TWO numbers (user rule 2026-09-29): everything and realistic.

  everything  every labelled detector-window of frame v6 (truth = user ruling > high-confidence print > config label
              where there is no print reading; note 45's set A), only the production rule >= 5 actuations.
  realistic   everything minus detectors the label check calls `fail` (known-wrong / doubtful label) or
              `misconfigured` (zone set up wrong); detectors in poor health are KEPT and reported by health status.
Candidate = function_v3d's OOF (run_ad6ea6959f_exclude_min5_clean_valnc_h3, first.all.wi, 3 seeds; the shipped
model is that recipe refitted on all training signals).  final_v2 head = note-25 b7 OOF (the final_v2 recipe with Mid /
Bike split out, 7 classes; compared on the rows it has).  Phase input of both = the frame's OOF predicted phase.
Health breakdown: health v5 (`health/health_v5.parquet`, Sept-2026 period only, whole 66 h).
Locked_v2 asserted absent.

    python eval48_function.py -> %DC_WORK%/final_v3_work/eval48/function.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trackA"))
import v3_retrain as V  # noqa: E402

RUN = "run_ad6ea6959f_exclude_min5_clean_valnc_h3"
OUTD = V.DCW / "final_v3_work" / "eval48"


def boot(ok_a, ok_b, sig, m, n_boot=2000, seed=0):
    s = sig[m]; a = ok_a[m].astype(float); b = ok_b[m].astype(float)
    u, inv = np.unique(s, return_inverse=True); n = len(u)
    sa, sb, c = np.bincount(inv, a, n), np.bincount(inv, b, n), np.bincount(inv, minlength=n)
    idx = np.random.default_rng(seed).integers(0, n, (n_boot, n))
    d = (sb[idx].sum(1) - sa[idx].sum(1)) / c[idx].sum(1)
    return [round((b.mean() - a.mean()) * 100, 2), round(float(np.quantile(d, .025)) * 100, 2),
            round(float(np.quantile(d, .975)) * 100, 2)]


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    V.set_frame("v6")
    V.MIN_ON, V.CLEAN, V.REQ_VAL, V.NC_HIGH = 5, True, True, True
    fr, _ = V.load_feats()
    fr = fr[V.KEY + ["wgroup", "det_n_on", "fold"]].copy()
    fr["DeviceId"] = fr.DeviceId.str.lower()
    lk = set(pd.read_csv(V.DCW / "official" / "locked_v2.csv").DeviceId.str.lower())
    assert not fr.DeviceId.isin(lk).any()
    v3 = pd.read_parquet(V.LABELS_V3)
    v3["DeviceId"] = v3.DeviceId.str.lower()
    pf, src = v3.print_function, v3.source.astype("string")
    user = src.eq("user_ruling").fillna(False)
    high = v3.print_source.eq("print") & v3.print_confidence.eq("high") & pf.isin(V.C7)
    noprint = ~v3.print_source.eq("print").fillna(False) | pf.isna()
    truth = np.where(user, v3.function, np.where(high, pf, np.where(noprint & v3.func7_v2.notna(), v3.func7_v2, None)))
    t = pd.DataFrame({"DeviceId": v3.DeviceId, "Detector": v3.detector.astype(fr.Detector.dtype), "truth": truth,
                      "validated": v3.validated.astype(str)})
    T = fr[V.KEY].merge(t, on=["DeviceId", "Detector"], how="left")
    assert len(T) == len(fr)
    y = T.truth.to_numpy(object)
    mA = pd.notna(y) & np.isin(y, V.C7) & (fr.det_n_on >= 5).to_numpy()
    bad_lab = T.validated.isin(["fail", "misconfigured"]).to_numpy()
    mR = mA & ~bad_lab
    # health v5 per detector (Sept 2026 = frame period 'stg')
    h = pd.read_parquet(V.DCW / "health" / "health_v5.parquet", columns=["period", "DeviceId", "detector", "status"])
    h = h[h.period == "stg"].assign(DeviceId=lambda x: x.DeviceId.str.lower()).rename(columns={"detector": "Detector"})
    h["Detector"] = h.Detector.astype(fr.Detector.dtype)
    hs = fr[["DeviceId", "Detector", "period"]].merge(h[["DeviceId", "Detector", "period", "status"]],
                                                        on=["DeviceId", "Detector", "period"], how="left").status
    hs = hs.fillna("none").to_numpy(object)
    # predictions
    C7 = np.array(V.C7, object)
    Ps, have = [], np.ones(len(fr), bool)
    for s in (0, 1, 2):
        P, hv, got = V.load_oof(fr, V.OUT / RUN, "first.all.wi", s, range(6))
        assert len(got) == 6, got
        Ps.append(P)
        have &= hv
    pn = C7[np.mean(Ps, 0).argmax(1)]
    Pb, hb = V.load_baseline(fr)
    pb = C7[Pb.argmax(1)]
    to5 = np.vectorize(lambda c: "Other" if c in ("Mid", "Bike") else c, otypes=[object])
    y5, pn5, pb5 = to5(np.where(pd.isna(y), "none", y)), to5(pn), to5(pb)
    core = np.isin(y, V.CORE4)
    wg, sig = fr.wgroup.to_numpy(), fr.DeviceId.to_numpy()
    res = {"n": {}, "sets": {}}
    for sn, m0 in (("everything", mA & have), ("realistic", mR & have)):
        r = {"rows": int(m0.sum()), "signals": int(pd.unique(sig[m0]).size),
             "detectors": int(fr[m0][["DeviceId", "Detector"]].drop_duplicates().shape[0])}
        okn, ok5n = pn == y, pn5 == y5
        r["candidate"] = {"acc7": okn[m0].mean(), "acc5": ok5n[m0].mean(), "core4": okn[m0 & core].mean(),
                          **{g: okn[m0 & (wg == g)].mean() for g in ("m30", "h6", "full")}}
        mb = m0 & hb
        okb, ok5b = pb == y, pb5 == y5
        r["b7_rows"] = {"rows": int(mb.sum()), "signals": int(pd.unique(sig[mb]).size),
                        "final_v2_head_b7": {"acc7": okb[mb].mean(), "acc5": ok5b[mb].mean(),
                                             "core4": okb[mb & core].mean(),
                                             **{g: okb[mb & (wg == g)].mean() for g in ("m30", "h6", "full")}},
                        "candidate": {"acc7": okn[mb].mean(), "acc5": ok5n[mb].mean(), "core4": okn[mb & core].mean(),
                                      **{g: okn[mb & (wg == g)].mean() for g in ("m30", "h6", "full")}},
                        "delta_ci": {"acc7": boot(okb, okn, sig, mb), "acc5": boot(ok5b, ok5n, sig, mb),
                                     "core4": boot(okb, okn, sig, mb & core),
                                     **{g: boot(okb, okn, sig, mb & (wg == g)) for g in ("m30", "h6", "full")}}}
        st = {}
        for k in ("ok", "suspect", "bad", "not_enough_data", "none"):
            mm = m0 & (hs == k)
            if mm.sum():
                st[k] = {"rows": int(mm.sum()), "acc7": okn[mm].mean(), "acc5": ok5n[mm].mean(),
                         "b7_acc7_on_b7_rows": okb[mm & hb].mean() if (mm & hb).any() else None}
        r["by_health_v5_status_sept2026"] = st
        res["sets"][sn] = r
    res["n"]["excluded_label_fail_or_misconfigured_rows"] = int((mA & have & bad_lab).sum())
    res["n"]["everything_rows_lt5_actuations_dropped"] = int((pd.notna(y) & np.isin(y, V.C7) &
                                                              (fr.det_n_on < 5).to_numpy()).sum())
    json.dump(res, open(OUTD / "function.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=lambda x: round(float(x), 4)))


if __name__ == "__main__":
    main()
