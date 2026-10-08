"""Note 101: final comparison table for the user's morning page -- beta (final_v2) vs v5 full vs v5 fast (le2h).
Saved OOF only, nothing trained, CPU <= 4 threads, locked_v2 asserted absent.

  phase     rows = s95/phase/rows95.parquet (2026 pool, timing truth, everything / realistic, alt phases right in R);
            v5 arms from s95/phase/q95.parquet (trees ranker p0t, TCN p95_ad alone, 0.5/0.5 average, decoder p2_p95_ad,
            le2h p2_le2h); beta = note-48 final_v2 OOF (eval48 phase_rows: GRU blend <= 2 h, trees above), joined on
            (DeviceId@stg, Detector, win).
  function  frame of func95 (v4q truth = truth_v3s, Sept-2026 'stg' rows); v5 stages: trees Pt, siba s95_siba 3-seed
            mean alone, fixed 0.6 / 0.4 blend, stacker P_v5 (argmax, stack credit), stacker through the gate .9 lane decode
            (= headline); fast = P_v5 on <= 1 h families, P_v5_nonet above, gate decode; beta = note-25 b7 5-class argmax
            (monday85 protocol).
  health    note-93 health rows (v4f package health_core on the same Sept-2026 windows) joined to the v5 function rows.
  bench     f98/bench run files (beta 4 passes, v5 2 passes, v5fast 1 pass).

    python table101.py phase | func | bench | assemble   -> %DC_WORK%/final_v3_work/final_table101.json
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "4"
os.environ.setdefault("F76_ARM", "c")
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CODE = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "final95"))
import rpath  # noqa: F401,E402
import cand64 as C  # noqa: E402

W = C.DC_WORK
OUTD = W / "final_v3_work" / "f101"
OUTD.mkdir(parents=True, exist_ok=True)
KEY4, DET = C.KEY4, C.DET
GE30 = C.GE30
CLS = ["Advance", "Presence", "Count", "Yellow_Red", "nonATSPM"]


def top(q, col):
    d = q[KEY4 + [col]].dropna()
    d = d.sort_values(DET + [col, "cand_phase"], ascending=[True, True, True, False, True])
    return d.groupby(DET, sort=False).first().reset_index()[DET + ["cand_phase"]]


def blend(q, p0, nn, thr=0.01):
    grp = [q[c] for c in DET]
    nn = np.where(p0 >= thr, nn, np.nan)
    tot = pd.Series(np.nan_to_num(nn)).groupby(grp).transform("sum").to_numpy()
    nn = np.where(~np.isnan(nn) & (tot > 0), nn / np.where(tot > 0, tot, 1.0), np.nan)
    p = np.where(np.isnan(nn), p0, 0.5 * p0 + 0.5 * nn)
    return p / pd.Series(p).groupby(grp).transform("sum").to_numpy()


def phase():
    rows = pd.read_parquet(W / "s95" / "phase" / "rows95.parquet")
    assert not rows.sig.isin(C.locked()).any()
    q = pd.read_parquet(W / "s95" / "phase" / "q95.parquet")
    assert not C.plain(q.DeviceId).isin(C.locked()).any()
    q["p_avg"] = blend(q, q.p0t.to_numpy(), q.pn_p95_ad.to_numpy())
    alt = [set(s.split(",")) if isinstance(s, str) and s else set() for s in rows.alt_s]
    arms = {"trees": "p0t", "tcn": "pn_p95_ad", "avg": "p_avg", "trees_dec": "p2_trees", "decider": "p2_p95_ad",
            "le2h": "p2_le2h"}
    for nm, col in arms.items():
        t = top(q, col)
        pred = rows[DET].merge(t, on=DET, how="left").cand_phase.to_numpy()
        has = ~pd.isna(pred)
        okE = has & (pred == rows.Phase.to_numpy())
        okR = okE | np.array([h and str(int(p)) in a for h, p, a in zip(has, np.nan_to_num(pred.astype(float)), alt)])
        rows[f"E_{nm}"] = np.where(has, okE, np.nan)
        rows[f"R_{nm}"] = np.where(has, okR, np.nan)
    # consistency with note 95 (rows95 columns)
    for a, b in (("decider", "p2_p95_ad"), ("le2h", "p2_le2h"), ("trees_dec", "p2_trees")):
        x, y = rows[f"E_{a}"].to_numpy(), rows[f"E_{b}"].to_numpy()
        m = ~np.isnan(x) & ~np.isnan(y)
        assert np.mean(x[m] == y[m]) > 0.9999, (a, np.mean(x[m] == y[m]))
    e = pd.read_parquet(W / "final_v3_work" / "eval48" / "phase_rows.parquet",
                        columns=DET + ["old_trees", "old_blend"]).astype({"Detector": rows.Detector.dtype})
    assert not e.duplicated(DET).any()
    n0 = len(rows)
    rows = rows.merge(e, on=DET, how="left")
    assert len(rows) == n0
    bp = np.where(rows.fam.isin(C.V2_NET), rows.old_blend, rows.old_trees)
    hb = ~pd.isna(bp)
    okE = hb & (bp == rows.Phase.to_numpy())
    okR = okE | np.array([h and str(int(p)) in a for h, p, a in zip(hb, np.nan_to_num(bp.astype(float)), alt)])
    rows["E_beta"], rows["R_beta"] = np.where(hb, okE, np.nan), np.where(hb, okR, np.nan)
    sig = rows.sig.to_numpy()
    ge = rows.fam.isin(GE30).to_numpy()
    res = {}
    for st, sname in (("E", "everything"), ("R", "realistic")):
        base = rows[sname].to_numpy() & ge
        full = base & ~np.isnan(rows[f"{st}_decider"].to_numpy())
        m = full & ~np.isnan(rows[f"{st}_beta"].to_numpy()) & ~np.isnan(rows[f"{st}_le2h"].to_numpy())
        b, v, f = (rows[f"{st}_{k}"].to_numpy()[m] for k in ("beta", "decider", "le2h"))
        res[f"main_{st}"] = {"n": int(m.sum()), "signals": int(len(set(sig[m]))), "v5_rows": int(full.sum()),
                             "v5_signals": int(len(set(sig[full]))), "coverage_of_v5_rows": round(float(m.sum() / full.sum()), 4),
                             "by_src_paired": rows[m].src.value_counts().to_dict(),
                             "beta": C.acc_ci(b, sig[m]), "v5": C.acc_ci(v, sig[m]), "v5_fast": C.acc_ci(f, sig[m]),
                             "v5_minus_beta": C.delta_ci(b, v, sig[m]), "fast_minus_beta": C.delta_ci(b, f, sig[m]),
                             "fast_minus_v5": C.delta_ci(v, f, sig[m]),
                             "v5_all_rows_unpaired": C.acc_ci(rows[f"{st}_decider"].to_numpy()[full], sig[full])}
        # stage table: rows where every stage answers (TCN coverage)
        stg = ["trees", "tcn", "avg", "trees_dec", "decider", "le2h"]
        ms = full & np.logical_and.reduce([~np.isnan(rows[f"{st}_{k}"].to_numpy()) for k in stg])
        res[f"stages_{st}"] = {"n": int(ms.sum()), "coverage_of_v5_rows": round(float(ms.sum() / full.sum()), 4),
                               **{k: C.acc_ci(rows[f"{st}_{k}"].to_numpy()[ms], sig[ms]) for k in stg},
                               "decider_minus_avg": C.delta_ci(rows[f"{st}_avg"].to_numpy()[ms], rows[f"{st}_decider"].to_numpy()[ms], sig[ms]),
                               "avg_minus_trees": C.delta_ci(rows[f"{st}_trees"].to_numpy()[ms], rows[f"{st}_avg"].to_numpy()[ms], sig[ms])}
        print(st, json.dumps(res[f"main_{st}"]), "\n", json.dumps(res[f"stages_{st}"]), flush=True)
    json.dump(res, open(OUTD / "phase101.json", "w"), indent=1, default=str)


def func():
    import func95 as FN
    import atspm_score as AS
    import of77
    F, F4 = FN.fsetup()
    F.THREADS = 4
    s74, S, E, Xs, y, trm, names = F.stacker_inputs()
    fr = E["fr"]
    assert not fr.DeviceId.str.lower().isin(C.locked()).any()
    C7 = list(E["C7"])
    assert C7 == list(AS.C7), (C7, AS.C7)
    st = (fr.period == "stg").to_numpy()
    Pt = E["Pt"].astype(float)
    Pt0 = np.load(FN.OOF95 / "Pt_v4q26.npy").astype(float)
    assert np.allclose(Pt, Pt0, atol=1e-5)
    Pf = s74.net_probs(fr, FN.NET95)
    hasn = ~np.isnan(Pf[:, 0])
    Pb = C.W_TREE * Pt + (1 - C.W_TREE) * np.where(hasn[:, None], Pf, Pt)
    P5 = np.load(FN.OOF95 / "P_v5.npy").astype(float)
    Rn = np.load(FN.OOF95 / "P_v5_nonet.npy").astype(float)
    le = fr.wgroup.isin(["m5", "m10", "m30", "h1"]).to_numpy()
    Pl = np.where(le[:, None], P5, Rn)
    lc = of77.lane_ctx(fr, of77.O8 / "lanes_D.func.parquet")[:, 2]
    ok = {"v5": s74.gate_ok(E, P5, lc), "fast": s74.gate_ok(E, Pl, lc)}
    key = ["DeviceId", "Detector", "period", "win"]
    base = fr[key + ["wgroup", "truth_v3s", "stack_group", "stack_role"]].copy()
    B5 = C.final_v2_probs(base)["v2"]
    hasb = ~np.isnan(B5[:, 0])
    Bb = np.zeros((len(base), 7))
    for j, c in enumerate(C.C5):
        Bb[:, C7.index(c)] = np.nan_to_num(B5[:, j])

    def argmax_ok(P, valid, beta=False):
        b = base.copy()
        for i, c in enumerate(C7):
            b[f"P_{c}"] = P[:, i]
        if beta:
            pred = np.where(valid, np.array([C7.index(c) for c in C.C5])[np.nan_to_num(B5, nan=-1).argmax(1)], C7.index("Other"))
        else:
            pred = np.nan_to_num(P, nan=-1).argmax(1)
        out = {}
        for stn in ("E", "R"):
            r = np.flatnonzero(~np.isnan(ok["v5"][stn]) & valid)
            d = AS.credit(b, pred, "truth_v3s", r, True)
            o = np.full(len(b), np.nan)
            o[r] = d.ok_a.to_numpy(float)
            out[stn] = o
        return out
    allv = np.ones(len(fr), bool)
    ok["beta"] = argmax_ok(Bb, hasb, beta=True)
    ok["trees"] = argmax_ok(Pt, allv)
    ok["tcn"] = argmax_ok(Pf, hasn)
    ok["blend"] = argmax_ok(Pb, allv)
    ok["decider"] = argmax_ok(P5, allv)
    np.savez_compressed(OUTD / "func_ok101.npz", **{f"{k}_{s}": v[s] for k, v in ok.items() for s in ("E", "R")})
    sig = fr.DeviceId.to_numpy()
    tr = fr.truth_v3s.to_numpy(object)
    cl = np.where(np.isin(tr, S.ATS), tr, "nonATSPM")
    ge = fr.wgroup.isin(GE30).to_numpy() & st
    res = {}
    for stn in ("E", "R"):
        full = ge & ~np.isnan(ok["v5"][stn])
        m = full & ~np.isnan(ok["beta"][stn]) & ~np.isnan(ok["fast"][stn])
        blk = {"n": int(m.sum()), "signals": int(len(set(sig[m]))), "v5_rows": int(full.sum()),
               "v5_signals": int(len(set(sig[full]))), "coverage_of_v5_rows": round(float(m.sum() / full.sum()), 4),
               "v5_all_rows_unpaired": C.acc_ci(ok["v5"][stn][full], sig[full])}
        for c in ["overall"] + CLS:
            mc = m if c == "overall" else m & (cl == c)
            b, v, f = (ok[k][stn][mc] for k in ("beta", "v5", "fast"))
            blk[c] = {"n": int(mc.sum()), "beta": C.acc_ci(b, sig[mc]), "v5": C.acc_ci(v, sig[mc]),
                      "v5_fast": C.acc_ci(f, sig[mc]), "v5_minus_beta": C.delta_ci(b, v, sig[mc]),
                      "fast_minus_beta": C.delta_ci(b, f, sig[mc]), "fast_minus_v5": C.delta_ci(v, f, sig[mc])}
        res[f"main_{stn}"] = blk
        stg = ["trees", "tcn", "blend", "decider", "v5"]
        ms = full & np.logical_and.reduce([~np.isnan(ok[k][stn]) for k in stg])
        res[f"stages_{stn}"] = {"n": int(ms.sum()), "coverage_of_v5_rows": round(float(ms.sum() / full.sum()), 4),
                                **{k: C.acc_ci(ok[k][stn][ms], sig[ms]) for k in stg},
                                "full_rows_trees_decider_v5": {k: C.acc_ci(ok[k][stn][full], sig[full]) for k in ("trees", "blend", "decider", "v5")}}
        print(stn, json.dumps(res[f"main_{stn}"]), json.dumps(res[f"stages_{stn}"]), flush=True)
    # health (note-93 rows, v4f package health_core on the same Sept-2026 windows)
    H = pd.read_parquet(W / "s93" / "health_rows.parquet", columns=["DeviceId", "period", "win", "detector", "status", "health_score"])
    H = H.rename(columns={"detector": "Detector"})
    H["DeviceId"] = H.DeviceId.str.lower()
    k = fr[key].copy()
    k["DeviceId"] = k.DeviceId.str.lower()
    H = H.astype({"Detector": k.Detector.dtype}).drop_duplicates(key)
    hs = k.merge(H, on=key, how="left").status.to_numpy(object)
    hres = {}
    for stn in ("E", "R"):
        full = ge & ~np.isnan(ok["v5"][stn])
        blk = {"rows_with_health": int((full & pd.notna(hs)).sum()), "rows": int(full.sum())}
        for s_ in ("ok", "not_enough_data", "suspect", "bad"):
            mm = full & (hs == s_)
            blk[s_] = {"n": int(mm.sum()), "v5": C.acc_ci(ok["v5"][stn][mm], sig[mm]), "v5_fast": C.acc_ci(ok["fast"][stn][mm], sig[mm])}
        fl = full & np.isin(hs, ["suspect", "bad"])
        ok_ = full & (hs == "ok")
        blk["flagged_n"] = int(fl.sum())
        blk["flagged"] = C.acc_ci(ok["v5"][stn][fl], sig[fl])
        hres[stn] = blk
        print("health", stn, json.dumps(blk), flush=True)
    res["health"] = hres
    json.dump(res, open(OUTD / "func101.json", "w"), indent=1, default=str)


def bench():
    B = W / "final_v3_work" / "f98" / "bench"
    groups = {"beta": ["beta_p1", "beta_p2", "beta_p3", "beta_p4"], "v5_full": ["v5", "v5_p2"], "v5_fast": ["v5fast"]}
    res = {}
    for g, tags in groups.items():
        for sg in ("typical_r8", "typical_r11", "busiest_ev", "busiest_ch"):
            for L in ("m30", "h3", "h24"):
                rs = [json.load(open(B / t / f"r_{sg}_{L}.json")) for t in tags if (B / t / f"r_{sg}_{L}.json").exists()]
                res.setdefault(g, {})[f"{sg}|{L}"] = {
                    "passes": len(rs), "channels": rs[0]["channels"],
                    "warm_s": round(float(np.mean([r["warm_s"] for r in rs])), 2),
                    "cold_s": round(float(np.mean([r["cold_s"] for r in rs])), 2),
                    "peak_mb": round(float(np.mean([r["peak_all_mb"] for r in rs]))),
                    "profile": rs[0].get("profile"), "pkg": rs[0].get("pkg")}
    json.dump(res, open(OUTD / "bench101.json", "w"), indent=1, default=str)
    for g in res:
        print(g, {k: (v["warm_s"], v["peak_mb"], v["passes"]) for k, v in res[g].items()})


SIDE = {
    "lanes": {"source": "no v5 lane OOF exists (note 95 held the lanes / pick / twin OOF chain fixed); latest measured = "
                        "lanes D six-fold OOF of note 77",
              "n_lanes_exact_per_sample_note99": {"all_ge30": 0.844, "m30": 0.813, "h1": 0.786, "h3": 0.871, "h6": 0.880,
                                                  "h24": 0.883, "full66": 0.884, "truth": "print lanes on v4q (2,515 phases / 517 signals)"},
              "note77_sept_scope_m30_full": {"n_lanes_exact": 0.854, "n_lanes_within_1": 0.991, "multi_lane_phases": 0.726,
                                             "detector_lane_set_exact": 0.922, "same_lane_pair_acc_multi": 0.924}},
    "setback": {"source": "note 77 sb_eval 'new' (printed Advance, OOF); v5 P50 refit not OOF-evaluated",
                "median_error_ft": {"m30": 25.0, "h6": 22.5, "h24": 21.7, "full66": 20.4},
                "within_50ft_pct": {"m30": 69.1, "h6": 68.5, "h24": 70.4, "full66": 70.2}},
}


def assemble():
    ph = json.load(open(OUTD / "phase101.json"))
    fu = json.load(open(OUTD / "func101.json"))
    be = json.load(open(OUTD / "bench101.json"))
    res = {"scope": "six-fold OOF, Sept-2026 windows, samples >= 30 min, paired rows (beta, v5 full and v5 fast all "
                    "score the row), signal-bootstrap 95 % CI; E = everything, R = realistic; function truth v4q "
                    "(ATSPM score, stack credit), phase truth = official timing (R also accepts switch / additional "
                    "call phases). Beta OOF exists only on the STG + REL signals (405 phase / 373 function signals); "
                    "the Dec-pool signals' Sept-2026 rows have no beta OOF.",
           "phase": ph, "function": fu, "bench": be, **SIDE}
    json.dump(res, open(W / "final_v3_work" / "final_table101.json", "w"), indent=1, default=str)
    print("->", W / "final_v3_work" / "final_table101.json")


if __name__ == "__main__":
    {"phase": phase, "func": func, "bench": bench, "assemble": assemble}[sys.argv[1]]()
