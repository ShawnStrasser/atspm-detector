"""Note 49: the function head on the phase input the final_v3 candidate ships (phase_v3 blend OOF, frame v6e)
and on the trees-only phase_v3 OOF (frame v6t = what candidate v2's
predict.py actually hands its function model, `prob_lgbm`) vs note 45 (same recipe, frame v6 = older tree-only phase OOF) vs the final_v2 head (note-25 b7 OOF).

Both runs: first.all.wi, --min-on 5 --clean --require-validated --allow-not-checkable-high --health3, 6 folds x
3 seeds.  Sets (eval48_function): everything (truth = user ruling > high print > config; >= 5 actuations) and
realistic (minus label-check fail / misconfigured).  Frames v6 and v6e have the same rows in the same order
(asserted), so every comparison is paired on identical rows; bootstrap = 2,000 signal-grouped draws.

    python eval49_function.py -> %DC_WORK%/final_v3_work/function_v3e/eval49.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trackA"))
import v3_retrain as V  # noqa: E402

RUNS = {"note45_v6": ("v6", "run_ad6ea6959f_exclude_min5_clean_valnc_h3"),
        "v6e": ("v6e", "run_ad6ea6959f_exclude_min5_clean_valnc_h3"),
        "v6t": ("v6t", "run_ad6ea6959f_exclude_min5_clean_valnc_h3")}   # trees-only phase_v3 OOF
OUTD = V.DCW / "final_v3_work" / "function_v3e"
WG = ["m5", "m10", "m30", "h1", "h3", "h6", "h24", "full"]


def boot(ok_a, ok_b, sig, m, n_boot=2000, seed=0):
    s = sig[m]; a = ok_a[m].astype(float); b = ok_b[m].astype(float)
    u, inv = np.unique(s, return_inverse=True); n = len(u)
    sa, sb, c = np.bincount(inv, a, n), np.bincount(inv, b, n), np.bincount(inv, minlength=n)
    idx = np.random.default_rng(seed).integers(0, n, (n_boot, n))
    d = (sb[idx].sum(1) - sa[idx].sum(1)) / c[idx].sum(1)
    return [round((b.mean() - a.mean()) * 100, 3), round(float(np.quantile(d, .025)) * 100, 3),
            round(float(np.quantile(d, .975)) * 100, 3)]


def main():
    V.MIN_ON, V.CLEAN, V.REQ_VAL, V.NC_HIGH = 5, True, True, True
    C7 = np.array(V.C7, object)
    frs, preds, seedp = {}, {}, {}
    for name, (tag, run) in RUNS.items():
        V.set_frame(tag)
        fr, _ = V.load_feats()
        fr = fr[V.KEY + ["wgroup", "det_n_on", "fold", "pred_phase"]].copy()
        fr["DeviceId"] = fr.DeviceId.str.lower()
        Ps, have = [], np.ones(len(fr), bool)
        for s in (0, 1, 2):
            P, hv, got = V.load_oof(fr, V.OUT / run, "first.all.wi", s, range(6))
            assert len(got) == 6, (name, got)
            Ps.append(P)
            have &= hv
        assert have.all()
        frs[name] = fr
        preds[name] = C7[np.mean(Ps, 0).argmax(1)]
        seedp[name] = [C7[p.argmax(1)] for p in Ps]
    f0, f1 = frs["note45_v6"], frs["v6e"]
    assert (f0[V.KEY].astype(str).values == f1[V.KEY].astype(str).values).all()
    assert (f0.det_n_on.values == f1.det_n_on.values).all()
    assert (f0[V.KEY].astype(str).values == frs["v6t"][V.KEY].astype(str).values).all()
    fr = f0
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
    y = T.truth.to_numpy(object)
    mA = pd.notna(y) & np.isin(y, V.C7) & (fr.det_n_on >= 5).to_numpy()
    mR = mA & ~T.validated.isin(["fail", "misconfigured"]).to_numpy()
    Pb, hb = V.load_baseline(fr)
    pb = C7[Pb.argmax(1)]
    to5 = np.vectorize(lambda c: "Other" if c in ("Mid", "Bike") else c, otypes=[object])
    y5 = to5(np.where(pd.isna(y), "none", y))
    core = np.isin(y, V.CORE4)
    wg, sig = fr.wgroup.to_numpy(), fr.DeviceId.to_numpy()
    ok = {k: preds[k] == y for k in preds}
    ok5 = {k: to5(preds[k]) == y5 for k in preds}
    okb, ok5b = pb == y, to5(pb) == y5

    def summ(o, o5, m):
        return {"acc7": round(float(o[m].mean()), 5), "acc5": round(float(o5[m].mean()), 5),
                "core4": round(float(o[m & core].mean()), 5),
                "by_window_acc7": {g: round(float(o[m & (wg == g)].mean()), 5) for g in WG}}
    res = {"sets": {}}
    for sn, m in (("everything", mA), ("realistic", mR)):
        r = {"rows": int(m.sum()), "signals": int(pd.unique(sig[m]).size)}
        for k in preds:
            r[k] = summ(ok[k], ok5[k], m)
            r[k]["per_seed_acc7"] = [round(float((sp == y)[m].mean()), 5) for sp in seedp[k]]
            r[k]["seed_sd_acc7_pt"] = round(float(np.std(r[k]["per_seed_acc7"], ddof=1) * 100), 3)
        for a_, b_ in (("note45_v6", "v6e"), ("note45_v6", "v6t"), ("v6t", "v6e")):
            r[f"{b_}_minus_{a_}"] = {"acc7": boot(ok[a_], ok[b_], sig, m), "acc5": boot(ok5[a_], ok5[b_], sig, m),
                                     "core4": boot(ok[a_], ok[b_], sig, m & core),
                                     **{g: boot(ok[a_], ok[b_], sig, m & (wg == g)) for g in WG}}
        # single-seed pairs within a frame: how far two equal-recipe fits differ (noise scale)
        r["seed_pair_diffs_pt"] = {k: [round(100 * float(((seedp[k][i] == y)[m].mean()) - ((seedp[k][j] == y)[m].mean())), 3)
                                       for i, j in ((0, 1), (0, 2), (1, 2))] for k in preds}
        mb = m & hb
        r["b7_rows"] = {"rows": int(mb.sum()), "signals": int(pd.unique(sig[mb]).size),
                        "final_v2_head_b7": summ(okb, ok5b, mb),
                        **{k: summ(ok[k], ok5[k], mb) for k in preds}}
        for k in preds:
            r["b7_rows"][f"{k}_minus_b7"] = {"acc7": boot(okb, ok[k], sig, mb), "acc5": boot(ok5b, ok5[k], sig, mb),
                                              "core4": boot(okb, ok[k], sig, mb & core),
                                              **{g: boot(okb, ok[k], sig, mb & (wg == g)) for g in ("m30", "h6", "full")}}
        res["sets"][sn] = r
    # how different the phase input is (frame pred_phase vs official timing, labelled rows with >= 5 actuations)
    off = pd.read_parquet(V.DCW / "official" / "labels_official.parquet",
                          columns=["DeviceId", "Detector", "target_type", "target_num"])
    off = off[off.target_type == "phase"].assign(DeviceId=lambda d: d.DeviceId.str.lower())
    off["Detector"] = off.Detector.astype(fr.Detector.dtype)
    ph = fr[V.KEY].merge(off[["DeviceId", "Detector", "target_num"]], on=["DeviceId", "Detector"], how="left").target_num
    mp = ph.notna().to_numpy() & (fr.det_n_on >= 5).to_numpy()
    res["phase_input"] = {"rows_with_timing_label": int(mp.sum()),
                          "pred_phase_changed_share_all_rows": round(float((f0.pred_phase.values != f1.pred_phase.values).mean()), 5),
                          "top1_vs_timing_v6": round(float((f0.pred_phase.values[mp] == ph.values[mp]).mean()), 5),
                          "top1_vs_timing_v6e": round(float((f1.pred_phase.values[mp] == ph.values[mp]).mean()), 5),
                          "top1_vs_timing_v6t": round(float((frs["v6t"].pred_phase.values[mp] == ph.values[mp]).mean()), 5),
                          "by_window_v6_v6e": {g: [round(float((f0.pred_phase.values[mp & (wg == g)] == ph.values[mp & (wg == g)]).mean()), 4),
                                                   round(float((f1.pred_phase.values[mp & (wg == g)] == ph.values[mp & (wg == g)]).mean()), 4)]
                                               for g in WG}}
    OUTD.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUTD / "eval49.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
