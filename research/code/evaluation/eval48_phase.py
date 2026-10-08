"""Note 48: phase of the final_v3 candidate v2 vs final_v2, six folds OOF, TWO numbers (user rule 2026-09-29).

Harness = note 37's (`neural/phase_v3_eval.py`: 0.5 blend of ranker bag and GRU before the joint decoder, decoder
re-fitted out of fold on the blended inputs, K = 4 pieces for the net above 120 min).  Arms, all OOF:
  candidate  phase_v3 trees OOF + phase_v3 GRU OOF, blended at EVERY length (what candidate v2 ships).
  final_v2   final_v1 trees OOF + stage-13 GRU OOF, blended up to 1 h (final_v2's 120-min cut-off), trees only
             (final_v1 trees + decoder) from 3 h on.  On the 71 released signals the old arm uses models that never
             saw them (note 37).
Row sets (detector-windows; the labelled phase turns green in the window; note 37's `all` rows):
  everything  official timing label (call_phase), >= 5 actuations in the window.
  realistic   everything minus detectors whose label is known to be doubtful: label check `fail` / `misconfigured`
              (the same list as the function realistic set) or a high-confidence cabinet-print phase that differs from
              the timing phase; and a prediction of the timing's switch phase / additional call phase counts as right
              (AGENTS A1 scorer rule).  Detectors in poor health are KEPT; accuracy by health v5 status is reported
              (Sept-2026 period only).
Paired signal-grouped bootstrap (2,000) of candidate - final_v2.  Locked_v2 asserted absent.

    python eval48_phase.py -> %DC_WORK%/final_v3_work/eval48/phase.json (+ phase_rows.parquet)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
import phase_v3_eval as PE  # noqa: E402  (sets up rpath / blend_v2 / trackb_eval)

B, BP, TE = PE.B, PE.BP, PE.TE
KEY, DET = PE.KEY, PE.DET
DCW = PE.DC_WORK
OUTD = DCW / "final_v3_work" / "eval48"
FAMS = ["m5", "m30", "h1", "h3", "h6", "h24", "full"]
NET_V2 = ("m5", "m10", "m30", "h1")          # final_v2's network runs up to 120 min


def top1(q: pd.DataFrame, keep: pd.DataFrame, col: str) -> pd.Series:
    d = keep.merge(q[KEY + ["Phase", col]], on=KEY, how="left")
    d[col] = d[col].fillna(0.0)
    d = d.sort_values(DET + [col, "cand_phase"], ascending=[1, 1, 1, 0, 1])
    t = d.groupby(DET, sort=False).first()
    return t["cand_phase"].rename(col)


def boot(a, b, sig, n=2000, seed=0):
    u, inv = np.unique(sig, return_inverse=True)
    k = len(u)
    sa, sb, c = np.bincount(inv, a, k), np.bincount(inv, b, k), np.bincount(inv, minlength=k)
    idx = np.random.default_rng(seed).integers(0, k, (n, k))
    d = (sb[idx].sum(1) - sa[idx].sum(1)) / c[idx].sum(1)
    return [round(100 * (b.mean() - a.mean()), 3), round(100 * float(np.quantile(d, .025)), 3),
            round(100 * float(np.quantile(d, .975)), 3)]


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    rows_f = OUTD / "phase_rows.parquet"
    lk = set(pd.read_csv(PE.LOCKED).DeviceId.str.lower())
    if not rows_f.exists():
        rel_stg = {d + "@stg" for d in PE.released()}
        keep_same = B.common_frame(TE._norm(B.gru_oof()))[KEY].copy()
        tn = PE.trees_new()
        assert not PE.plain(tn.DeviceId).isin(lk).any()
        rel_keys = tn.loc[tn.DeviceId.isin(rel_stg), KEY]
        folds = PE.fold_map()
        lab = B.labels()
        ctx, sim = BP.load_context(set(folds.DeviceId))
        nn_o, nn_n = PE.net_old(), PE.net_new()
        assert not PE.plain(nn_n.DeviceId).isin(lk).any() and not PE.plain(nn_o.DeviceId).isin(lk).any()
        r = rel_keys.merge(ctx[["DeviceId", "Detector", "win", "cand_phase", "det_n_on"]], on=KEY, how="left")
        r["dev_plain"] = PE.plain(r.DeviceId)
        r = r.merge(lab.rename(columns={"DeviceId": "dev_plain"}), on=["dev_plain", "Detector"])
        has = (r.cand_phase == r.Phase).groupby([r[c] for c in DET]).transform("max")
        act = r.groupby(DET)["det_n_on"].transform("max")
        r = r[has & (act >= 1)]
        cov_o = nn_o[nn_o.DeviceId.isin(rel_stg)][DET].drop_duplicates()
        cov_n = nn_n[nn_n.DeviceId.isin(rel_stg)][DET].drop_duplicates()
        r = r.merge(cov_o, on=DET).merge(cov_n, on=DET)
        keep = pd.concat([keep_same, r[KEY].drop_duplicates()], ignore_index=True)
        q_old = PE.run_arm("old", PE.trees_old(rel_keys), nn_o, folds, lab, ctx, sim, rel_stg)
        q_new = PE.run_arm("new", tn[KEY + ["prob", "p0"]], nn_n, folds, lab, ctx, sim, set())
        out = pd.concat([top1(q_old, keep, "prob").rename("old_trees"), top1(q_old, keep, "p2").rename("old_blend"),
                         top1(q_new, keep, "prob").rename("new_trees"), top1(q_new, keep, "p2").rename("new_blend")],
                        axis=1).reset_index()
        info = keep.merge(ctx[KEY + ["det_n_on"]], on=KEY, how="left").groupby(DET).det_n_on.max().reset_index()
        out = out.merge(info, on=DET, how="left")
        out["dev_plain"] = PE.plain(out.DeviceId)
        out = out.merge(lab.rename(columns={"DeviceId": "dev_plain"}), on=["dev_plain", "Detector"], how="left")
        out = out.merge(folds, on="DeviceId", how="left")
        out.to_parquet(rows_f, index=False)
    t = pd.read_parquet(rows_f)
    assert not t.dev_plain.isin(lk).any()
    t["fam"] = t.win.map(B.fam_of)
    t["final_v2"] = np.where(t.fam.isin(NET_V2), t.old_blend, t.old_trees)
    t["candidate"] = t.new_blend
    # ---- label-quality facts (labels v3) and health v5
    v3 = pd.read_parquet(Path(__file__).resolve().parents[2] / "labels" / "function_labels_v3.parquet")
    v3["DeviceId"] = v3.DeviceId.str.lower()
    tp = v3.phase_target.str.extract(r"P(\d+)")[0].astype(float)
    hi = v3.print_source.eq("print") & v3.print_confidence.eq("high") & v3.phase_diagram.notna() & tp.notna()
    v3["print_phase_disagrees"] = hi & (v3.phase_diagram.astype(float) != tp)
    v3["lab_bad"] = v3.validated.isin(["fail", "misconfigured"]).fillna(False)
    v3["alt"] = [({int(s)} if pd.notna(s) and s > 0 else set()) |
                 {int(x) for x in str(a).split(",") if str(x).strip().isdigit()}
                 for s, a in zip(v3.switch_phase, v3.additional_call_phases.fillna(""))]
    L = v3[["DeviceId", "detector", "print_phase_disagrees", "lab_bad", "alt"]].rename(
        columns={"DeviceId": "dev_plain", "detector": "Detector"})
    t = t.merge(L, on=["dev_plain", "Detector"], how="left")
    t["print_phase_disagrees"] = t.print_phase_disagrees.fillna(False).astype(bool)
    t["lab_bad"] = t.lab_bad.fillna(False).astype(bool)
    h = pd.read_parquet(DCW / "health" / "health_v5.parquet", columns=["period", "DeviceId", "detector", "status"])
    h = h[h.period == "stg"]
    h = pd.DataFrame({"DeviceId": h.DeviceId.str.lower() + "@stg", "Detector": h.detector.astype(int),
                      "hstatus": h.status})
    t = t.merge(h, on=["DeviceId", "Detector"], how="left")
    t["hstatus"] = np.where(t.DeviceId.str.endswith("@stg"), t.hstatus.fillna("none"), "dec_period")
    ev = (t.det_n_on >= 5) & t.Phase.notna()
    real = ev & ~t.lab_bad & ~t.print_phase_disagrees

    def ok(col, lenient):
        o = (t[col] == t.Phase).to_numpy()
        if lenient:
            o = o | np.array([p in a if isinstance(a, set) else False for p, a in zip(t[col], t.alt)])
        return o.astype(float)
    res = {"rows": {}, "sets": {}}
    for sn, m, len_ in (("everything", ev, False), ("realistic", real, True)):
        oc, o2 = ok("candidate", len_), ok("final_v2", len_)
        ont, oot = ok("new_trees", len_), ok("old_trees", len_)
        R = {}
        for fam in FAMS:
            mm = (m & (t.fam == fam)).to_numpy()
            if not mm.any():
                continue
            per = lambda o: float(pd.Series(o[mm]).groupby(t.win.to_numpy()[mm]).mean().mean())  # noqa: E731
            R[fam] = {"n_detwin": int(mm.sum()), "n_signals": int(t.DeviceId[mm].nunique()),
                      "candidate": round(per(oc), 5), "final_v2": round(per(o2), 5),
                      "candidate_trees_only": round(per(ont), 5), "final_v1_trees": round(per(oot), 5),
                      "delta_pt_ci": boot(o2[mm], oc[mm], t.DeviceId.to_numpy()[mm]),
                      "err_candidate": int((1 - oc[mm]).sum()), "err_final_v2": int((1 - o2[mm]).sum())}
            hs = {}
            for k in ("ok", "suspect", "bad", "not_enough_data", "none", "dec_period"):
                mk = mm & (t.hstatus == k).to_numpy()
                if mk.any():
                    hs[k] = {"n": int(mk.sum()), "candidate": round(float(oc[mk].mean()), 4),
                             "final_v2": round(float(o2[mk].mean()), 4)}
            R[fam]["by_health_v5"] = hs
        res["sets"][sn] = R
    res["rows"] = {"all_detwin": int(len(t)), "everything": int(ev.sum()), "realistic": int(real.sum()),
                   "dropped_lt5_actuations": int(((t.det_n_on < 5) & t.Phase.notna()).sum()),
                   "realistic_excluded_label_check": int((ev & t.lab_bad).sum()),
                   "realistic_excluded_print_phase": int((ev & t.print_phase_disagrees & ~t.lab_bad).sum()),
                   "detectors_with_alt_phase": int(t[ev & t.alt.map(lambda a: bool(a) if isinstance(a, set) else False)]
                                                   [["DeviceId", "Detector"]].drop_duplicates().shape[0])}
    json.dump(res, open(OUTD / "phase.json", "w"), indent=1)
    for sn, R in res["sets"].items():
        for fam, v in R.items():
            print(f"{sn:10s} {fam:4s} n {v['n_detwin']:6d} cand {v['candidate']:.4f} v2 {v['final_v2']:.4f} "
                  f"d {v['delta_pt_ci']} trees new/old {v['candidate_trees_only']:.4f}/{v['final_v1_trees']:.4f}")
    print(json.dumps(res["rows"]))


if __name__ == "__main__":
    main()
