"""Note 109: a SIMPLER architecture -- PHASE side.  Can the siba function net's own phase head replace the separate
phase TCN (p95_ad)?  2026-only pool (note 95), folds_v4, timing truth, CPU <= 6 threads, saved / local-GPU OOF only.

    python p109_phase.py nets                       siba phase-head OOF aligned to the pool -> s109/phase/nets.parquet
    python p109_phase.py decode --net NAME          trees bag 0.5 / net 0.5 (net on tree p >= .01) -> decoder six-fold OOF
    python p109_phase.py ctx --net NAME             decoder-style columns on the net's probabilities (all candidates)
    python p109_phase.py fit --net NAME [--seeds]   ONE LightGBM (note-103 A2 recipe: ranker features + A1 context + net
                                                    columns), six folds x seeds -> s109/phase/A2_NAME
    python p109_phase.py score                      per length E / R with CIs vs full v5c (= v5 phase p2_p95_ad)
Nets: w32m3 / w32s0 / w32s1 / w32s2 = x100_w32 fold nets, inference WITH the tree filter (as packaged);
      nf_* = the same checkpoints re-run WITHOUT the filter (s109/q/infer109.ps1, otag x109_w32nf).
locked_v2 asserted absent (pool loader).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "6")
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "final95"))
sys.path.insert(0, str(CODE / "final103"))
import phase95 as P  # noqa: E402
import p103_phase as P103  # noqa: E402

C, T57, TE = P.C, P.T57, P.TE
NJ = int(os.environ.get("P109_THREADS", "5"))
P.NJ = C.NJ = T57.NJ = P103.NJ = NJ
OUT = C.DC_WORK / "s109" / "phase"
PP = C.DC_WORK / "tcn53" / "ppreds"
KEY4, DET = P.KEY4, P.DET
FAMS = {"m5": ["m5"], "m10": ["m10"], "m30": ["m30"], "h1": ["h1"], "h3": ["h3"], "h24": ["h24"],
        "ge30": ["m30", "h1", "h3", "h6", "h24", "full"]}
NETS = {"w32s0": ["x100_w32"], "w32s1": ["x100_w32_s1"], "w32s2": ["x100_w32_s2"],
        "w32m3": ["x100_w32", "x100_w32_s1", "x100_w32_s2"],
        "nf_s0": ["x109_w32nf"], "nf_s1": ["x109_w32nf_s1"], "nf_s2": ["x109_w32nf_s2"],
        "nf_m3": ["x109_w32nf", "x109_w32nf_s1", "x109_w32nf_s2"]}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def one_tag(comb, tag):
    fs = [PP / f"{tag}_f{k}.parquet" for k in range(6)]
    t = pd.concat([TE._norm(pd.read_parquet(f))[KEY4 + ["prob"]] for f in fs], ignore_index=True)
    assert not t.duplicated(KEY4).any()
    k = comb[KEY4].copy()
    k["DeviceId"] = k.DeviceId.str.lower()
    t = t.astype({"Detector": k.Detector.dtype, "cand_phase": k.cand_phase.dtype})
    return k.merge(t, on=KEY4, how="left").prob.to_numpy()


def stage_nets(a):
    OUT.mkdir(parents=True, exist_ok=True)
    comb, _ = P.pool(KEY4 + ["fold"])
    f = OUT / "nets.parquet"
    out = pd.read_parquet(f) if f.exists() else comb[KEY4].copy()
    grp = [comb[c] for c in DET]
    for nm, tags in NETS.items():
        if nm in out.columns or not all((PP / f"{t}_f5.parquet").exists() for t in tags):
            continue
        ps = [one_tag(comb, t) for t in tags]
        p = np.mean(ps, 0)   # NaN where any member lacks the pair (same pieces for every member)
        s = pd.Series(np.nan_to_num(p)).groupby(grp).transform("sum").to_numpy()
        out[nm] = np.where(np.isnan(p) | (s <= 0), np.nan, p / np.where(s > 0, s, 1))
        log(f"{nm}: coverage {np.mean(~np.isnan(out[nm])):.4f} of pool pairs")
    out.to_parquet(f, index=False)


def net_col(comb, nm):
    n = pd.read_parquet(OUT / "nets.parquet")
    for c in KEY4:
        assert (n[c].to_numpy() == comb[c].to_numpy()).all()
    return n[nm].to_numpy()


def stage_decode(a):
    import p90_phase as P90
    P90.C.NJ = NJ
    comb, simc = P.pool()
    t = P.trees_bag(comb)
    fq, fi = OUT / "q109.parquet", OUT / "dec_iters.json"
    done = pd.read_parquet(fq) if fq.exists() else comb[KEY4].copy()
    its = json.load(open(fi)) if fi.exists() else {}
    nn = net_col(comb, a.net)
    if a.ranker == "bag":
        p0, nm = t.p0_bag.to_numpy(), a.net
    else:                      # one ranker seed (p0_s0 ...) -> 1-seed design
        p0 = pd.read_parquet(P.P95 / "full" / "oof.parquet", columns=[f"p0_{a.ranker}"])[f"p0_{a.ranker}"].to_numpy()
        nm = f"{a.net}_r{a.ranker}"
    done[f"p2_{nm}"], its[nm] = P90.decode_iters(comb, simc, P.blend(comb, p0, nn))
    done.to_parquet(fq, index=False)
    json.dump(its, open(fi, "w"), indent=1)
    log(f"decoded {nm}: iterations {its[nm]}")


def stage_ctx(a):
    """decoder-style columns on the NET's probabilities over all candidates (note-103 A2 recipe, prefix tn_)."""
    import decode_train as dec
    comb, simc = P.pool()
    dcols = [c for c in dec.BASE_COLS + dec.SIM_COLS + dec.ADJ_COLS
             if c not in ("cand_green_share", "call43_per_cycle", "det_n_on", "log_win_hours")]
    nn = net_col(comb, a.net)
    grp = [comb[c] for c in DET]
    ncand = pd.Series(np.ones(len(comb))).groupby(grp).transform("sum").to_numpy()
    nnf = np.where(np.isnan(nn), 1.0 / ncand, nn)
    nnf = nnf / pd.Series(nnf).groupby(grp).transform("sum").to_numpy()
    pr = comb[KEY4].copy()
    pr["p0"] = nnf
    D = dec.assemble(pr, pairs=comb, sim=simc)
    D = comb[KEY4].merge(D[KEY4 + dcols], on=KEY4, how="left")
    D = D[dcols].rename(columns={c: "tn_" + c for c in dcols}).astype(np.float32)
    D["tn_missing"] = np.isnan(nn).astype(np.float32)
    D.to_parquet(OUT / f"ctx_{a.net}.parquet", index=False)
    log(f"ctx {a.net} {D.shape}")


def stage_fit(a):
    import train_official as T
    comb, simc, fc = T57.load_pool()
    assert not C.plain(comb.DeviceId).isin(C.locked()).any()
    new = []
    for D in (pd.read_parquet(C.DC_WORK / "s103" / "phase" / "ctx_A1.parquet"), pd.read_parquet(OUT / f"ctx_{a.net}.parquet")):
        assert len(D) == len(comb)
        for c in D.columns:
            comb[c] = D[c].to_numpy()
            new.append(c)
    cols = list(fc) + new
    d = OUT / f"A2_{a.net}"
    d.mkdir(parents=True, exist_ok=True)
    json.dump({"net": a.net, "n_cols": len(cols), "cols": cols}, open(d / "cols.json", "w"), indent=1)
    lab = comb.Phase.notna().to_numpy()
    fold = comb.fold.to_numpy()
    for sd in [int(x) for x in a.seeds.split(",")]:
        for k in range(6):
            f = d / f"S_f{k}_s{sd}.npy"
            if f.exists():
                continue
            t1 = time.time()
            inner = (k + 1) % 6
            base = (fold != k) & lab
            P_ = T57.rp(sd)
            P_["n_jobs"] = NJ
            m = T._fit_rank(comb[base & (fold != inner)], comb[base & (fold == inner)], cols, P_)
            np.save(f, m.predict(comb.loc[fold == k, cols], num_threads=NJ))
            with open(d / "timing.jsonl", "a") as fh:
                fh.write(json.dumps({"fold": k, "seed": sd, "trees": int(m.best_iteration_ or 0),
                                     "secs": round(time.time() - t1, 1)}) + "\n")
            log(f"  A2_{a.net} fold {k} seed {sd}: {m.best_iteration_} trees ({time.time() - t1:.0f}s)")
            if k == 0 and sd == 0:
                imp = pd.Series(m.booster_.feature_importance("gain"), index=cols)
                (imp / imp.sum()).sort_values(ascending=False).head(40).to_json(d / "gain_f0_s0.json", indent=1)


def a2_prob(comb, d, seeds=None):
    import train_official as T
    fold = comb.fold.to_numpy()
    seeds = seeds or sorted({int(p.stem.split("_s")[1]) for p in d.glob("S_f5_s*.npy")})
    Ps = []
    for sd in seeds:
        S = np.zeros(len(comb))
        for k in range(6):
            S[fold == k] = np.load(d / f"S_f{k}_s{sd}.npy")
        Ps.append(T.to_prob(comb, S))
    return np.mean(Ps, 0), seeds


def stage_score(a):
    comb, _ = P.pool(KEY4 + ["fold"])
    q = pd.read_parquet(P.P95 / "q95.parquet")
    for c in KEY4:
        assert (q[c].to_numpy() == comb[c].to_numpy()).all()
    arms = {"full_v5c": q.p2_p95_ad.to_numpy(), "no_nets": q.p2_trees.to_numpy(), "fast_le2h": q.p2_le2h.to_numpy(),
            "trees_alone": q.p0t.to_numpy()}
    nets = pd.read_parquet(OUT / "nets.parquet")
    arms["phaseTCN_alone"] = q.pn_p95_ad.to_numpy()
    arms["avg_phaseTCN"] = P.blend(comb, q.p0t.to_numpy(), q.pn_p95_ad.to_numpy())
    for nm in NETS:
        if nm in nets:
            arms[f"{nm}_alone"] = nets[nm].to_numpy()
            arms[f"avg_{nm}"] = P.blend(comb, q.p0t.to_numpy(), nets[nm].to_numpy())   # no decoder
    if (OUT / "q109.parquet").exists():
        q9 = pd.read_parquet(OUT / "q109.parquet")
        for c in q9.columns:
            if c.startswith("p2_"):
                arms["dec_" + c[3:]] = q9[c].to_numpy()
    for d in sorted(OUT.glob("A2_*")):
        if (d / "S_f5_s0.npy").exists():
            arms[d.name], sds = a2_prob(comb, d)
            if len(sds) > 1:
                for sd in sds:
                    arms[f"{d.name}@s{sd}"], _ = a2_prob(comb, d, [sd])
    for d in sorted((C.DC_WORK / "s103" / "phase").glob("A2")):
        arms["A2_tcn103"], _ = a2_prob(comb, d)
    rows = P.rows_()
    alt = rows.alt.to_numpy()
    sig = rows.sig.to_numpy()
    ok = {}
    for nm, p in arms.items():
        d = comb[KEY4].copy()
        d["p"] = p
        d = d.dropna(subset=["p"]).sort_values(DET + ["p", "cand_phase"], ascending=[True, True, True, False, True])
        t = d.groupby(DET, sort=False).first().reset_index()
        pred = rows[DET].merge(t[DET + ["cand_phase"]], on=DET, how="left").cand_phase.to_numpy()
        okE = (pred == rows.Phase.to_numpy()).astype(float)
        okR = np.maximum(okE, np.array([p_ in s for p_, s in zip(pred, alt)], float))
        ok[nm] = {"E": okE, "R": okR}
    res = {"n": {}, "acc": {}, "delta_vs_full": {}}
    for sname, pre in (("everything", "E"), ("realistic", "R")):
        mset = rows[sname].to_numpy()
        for pn, fams in FAMS.items():
            m = mset & rows.fam.isin(fams).to_numpy()
            res["n"].setdefault(pre, {})[pn] = [int(m.sum()), int(len(set(sig[m])))]
            for nm in ok:
                res["acc"].setdefault(nm, {}).setdefault(pre, {})[pn] = C.acc_ci(ok[nm][pre][m], sig[m])
                if nm != "full_v5c":
                    res["delta_vs_full"].setdefault(nm, {}).setdefault(pre, {})[pn] = C.delta_ci(
                        ok["full_v5c"][pre][m], ok[nm][pre][m], sig[m])
    json.dump(res, open(OUT / "score109.json", "w"), indent=1)
    for nm in ok:
        log(f"{nm:18s} E " + " ".join(f"{pn} {res['acc'][nm]['E'][pn][0]}" for pn in FAMS))
        if nm != "full_v5c":
            log(f"{'':18s} dE " + " ".join(f"{pn} {res['delta_vs_full'][nm]['E'][pn]}" for pn in FAMS))
            log(f"{'':18s} dR ge30 {res['delta_vs_full'][nm]['R']['ge30']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["nets", "decode", "ctx", "fit", "score"])
    ap.add_argument("--net", default="w32m3")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--ranker", default="bag", help="bag (3 seeds) or s0 / s1 / s2")
    a = ap.parse_args()
    t0 = time.time()
    globals()[f"stage_{a.stage}"](a)
    log(f"== {a.stage} done ({time.time() - t0:.0f}s)")
