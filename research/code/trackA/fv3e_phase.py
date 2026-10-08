"""Note 49: phase_v3 six-fold OOF phase probabilities (note 37 blend, K = 4 pieces) for EVERY row of the
function frame v6, so the function head can be re-fitted and re-scored on the phase input the final_v3
candidate actually ships (note 48 open item).

The note-37 OOF covers the phase pool only: Dec-2024 DEV (all channels), Sept-2026 NEWTRAIN + released
(channels with a timing label).  The frame also holds the Sept-2026 rows of the DEV signals (354 signals),
the Dec-2024 rows of 6 NEWTRAIN signals and a few unlabelled channels of pool signals ("extras").  They
are scored here by the phase_v3 FOLD models of their own signal's phase fold (never trained on them):
  trees   the note-37 ranker bag re-fitted per fold, exactly `phase_v3_trees.stage_oof` (the fold models
          were not saved; the refit's pool predictions are compared with the saved OOF as a check);
  GRU     the saved note-37 fold models `gru_models/p3_f{k}.pt`, K = 4 pieces above 120 min (rasters for
          the Sept-2026 DEV signals built into a private folder, the shared cache is not touched);
  blend   note 37's `run_arm`: 0.5 blend before the joint decoder, decoder re-fitted out of fold.  Pool rows
          keep the exact note-37 blend (arm A, verified); extras come from a second decode (arm B) over
          pool + extras, extras never trained on.
Locked_v2 asserted absent.  Outputs under %DC_WORK%/final_v3_work/function_v3e/phase/.

    python fv3e_phase.py extras
    python fv3e_phase.py trees --threads 6
    python fv3e_phase.py rasters
    python fv3e_phase.py gru
    python fv3e_phase.py blend [--variant trees]   # trees: decoder on the ranker bag only (prob_lgbm)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "neural"))
import rpath  # noqa: F401,E402
from common import DC_WORK  # noqa: E402

OUT = DC_WORK / "final_v3_work" / "function_v3e" / "phase"
SIGDIR_STG = DC_WORK / "final_v3_work" / "function_v3e" / "sig_stg_dev"
FRAME_V6 = DC_WORK / "function_v4" / "funcframe_v6.parquet"
LOCKED = DC_WORK / "official" / "locked_v2.csv"
P3 = DC_WORK / "final_v3_work" / "phase_v3" / "work"
KEY = ["DeviceId", "Detector", "win", "cand_phase"]
DET = ["DeviceId", "Detector", "win"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def locked() -> set:
    return set(pd.read_csv(LOCKED).DeviceId.str.lower())


def phase_folds() -> pd.DataFrame:
    """Phase fold of every signal-period key (DeviceId lower, '@stg' for Sept 2026): note 37's map (pool),
    DEV signals' Sept-2026 rows = their folds.csv fold, NEWTRAIN signals' Dec-2024 rows = newtrain_folds."""
    f = pd.read_csv(DC_WORK / "folds.csv")
    nt = pd.read_csv(DC_WORK / "official" / "newtrain_folds.csv")
    rel = pd.read_csv(DC_WORK / "folds_v4.csv")
    rs = set(pd.read_csv(DC_WORK / "official" / "newtest_released.csv").DeviceId.str.lower())
    rel = rel[rel.DeviceId.str.lower().isin(rs)]
    rows = []
    for d, k in zip(f.DeviceId.str.lower(), f.fold):
        rows += [(d, k), (d + "@stg", k)]
    for d, k in zip(nt.DeviceId.str.lower(), nt.fold):
        rows += [(d + "@stg", k), (d, k)]
    for d, k in zip(rel.DeviceId.str.lower(), rel.fold):
        rows += [(d + "@stg", k)]
    out = pd.DataFrame(rows, columns=["DeviceId", "fold"]).drop_duplicates()
    assert not out.DeviceId.duplicated().any()
    out["fold"] = out.fold.astype(int)
    return out


def frame_keys() -> pd.DataFrame:
    fr = pd.read_parquet(FRAME_V6, columns=["DeviceId", "Detector", "period", "win"])
    fr["DeviceId"] = fr.DeviceId.str.lower() + np.where(fr.period == "stg", "@stg", "")
    fr["Detector"] = fr.Detector.astype(int)
    assert not fr.DeviceId.str.replace("@stg", "").isin(locked()).any()
    return fr[DET].drop_duplicates()


# ------------------------------------------------------------------ extras
def stage_extras(a) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fk = frame_keys()
    tn = pd.read_parquet(P3 / "trees_oof_bywindow.parquet", columns=DET)
    tn["DeviceId"] = tn.DeviceId.str.lower()
    tn["Detector"] = tn.Detector.astype(int)
    tn = tn.drop_duplicates()
    m = fk.merge(tn.assign(_in=1), on=DET, how="left")
    ex = m[m._in.isna()][DET].reset_index(drop=True)
    fm = phase_folds()
    ex = ex.merge(fm, on="DeviceId", how="left")
    assert ex.fold.notna().all(), ex[ex.fold.isna()].DeviceId.unique()[:5]
    per = np.where(ex.DeviceId.str.endswith("@stg"), "stg", "dec")
    pool_sig = set(tn.DeviceId)
    ex["kind"] = np.where(ex.DeviceId.isin(pool_sig), "pool_signal_extra_channel", "new_signal_period")
    ex.to_parquet(OUT / "extras_detwin.parquet", index=False)
    s = {"frame_detwin": len(fk), "covered_by_note37_oof": int(len(fk) - len(ex)), "extras_detwin": len(ex),
         "by_period_kind": pd.crosstab(per, ex.kind).to_dict(),
         "extra_signals_by_period": pd.Series(per).groupby(per).size().to_dict(),
         "extra_signal_count": int(ex.DeviceId.nunique())}
    json.dump(s, open(OUT / "extras.json", "w"), indent=1, default=str)
    log(json.dumps(s, default=str))


def load_extra_pairs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair rows (every candidate phase) of the extra detector-windows, both periods, with the tree features."""
    import train_official as T
    ex = pd.read_parquet(OUT / "extras_detwin.parquet")
    parts, sims = [], []
    for per in ("dec", "stg"):
        e = ex[ex.DeviceId.str.endswith("@stg") == (per == "stg")]
        plain = set(e.DeviceId.str.replace("@stg", "", regex=False))
        import duckdb
        con = duckdb.connect()
        f = (T.SFEAT / "pair_features_stg.parquet") if per == "stg" else None
        if per == "stg":
            ids = con.execute(f"SELECT DISTINCT DeviceId FROM '{f.as_posix()}'").df().DeviceId
        else:
            ids = pd.concat([con.execute(f"SELECT DISTINCT DeviceId FROM '{g.as_posix()}'").df().DeviceId
                             for g in (T.FEAT / "pair_features_windows.parquet",
                                       T.FEAT / "pair_features_windows_B.parquet") if g.exists()])
        keep = {i for i in ids if str(i).lower() in plain}
        df, sim = (T.load_stg if per == "stg" else T.load_dec)(keep=keep)
        df["DeviceId"] = df.DeviceId.str.lower()
        sim["DeviceId"] = sim.DeviceId.str.lower()
        df["Detector"] = df.Detector.astype(int)
        df = df.merge(e[DET + ["fold"]], on=DET, how="inner")
        parts.append(df)
        sims.append(sim)
        log(f"[{per}] extra pair rows {df.shape}, {df.DeviceId.nunique()} signals")
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
    return df, pd.concat(sims, ignore_index=True)


# ------------------------------------------------------------------ trees
def stage_trees(a) -> None:
    import phase_v3_trees as PT
    import train_official as T
    import fit_final_v1 as F
    T.RANK_PARAMS["n_jobs"] = a.threads
    T.BIN_PARAMS["n_jobs"] = a.threads
    ck = OUT / "trees_ckpt"
    ck.mkdir(parents=True, exist_ok=True)
    comb, simc, fc = PT.build_pool_v3()
    k0 = pd.read_parquet(PT.CKPT / "rows.parquet")
    assert len(k0) == len(comb) and (k0.DeviceId.values == comb.DeviceId.values).all()
    ex, _ = load_extra_pairs()
    for c in fc:
        assert c in ex.columns, c
    ex[fc] = ex[fc].astype(np.float32)
    lab = comb.Phase.notna().to_numpy()
    chk = {}
    t0 = time.time()
    for k in range(6):
        te = (comb.fold == k).to_numpy()
        tex = (ex.fold == k).to_numpy()
        inner = (k + 1) % 6
        base = (~te) & lab
        for sd in PT.SEEDS:
            f = ck / f"Sx_f{k}_s{sd}.npy"
            if f.exists():
                continue
            tr = comb[base & (comb.fold != inner).to_numpy()]
            va = comb[base & (comb.fold == inner).to_numpy()]
            m = T._fit_rank(tr, va, fc, PT.rp(sd))
            del tr, va
            sp = m.predict(comb.loc[te, fc])
            ref = np.load(PT.CKPT / f"S_f{k}_s{sd}.npy")
            chk[f"f{k}_s{sd}"] = {"trees": int(m.best_iteration_), "max_abs_vs_saved": float(np.max(np.abs(sp - ref))),
                                  "corr": float(np.corrcoef(sp, ref)[0, 1])}
            np.save(f, m.predict(ex.loc[tex, fc]) if tex.any() else np.zeros(0))
            json.dump(chk, open(ck / f"check_f{k}_s{sd}.json", "w"))
            log(f"  fold {k} seed {sd}: {m.best_iteration_} trees, refit vs saved pool OOF "
                f"max|d| {chk[f'f{k}_s{sd}']['max_abs_vs_saved']:.2e} ({time.time()-t0:.0f}s)")
    S = np.zeros((len(PT.SEEDS), len(ex)))
    for k in range(6):
        tex = (ex.fold == k).to_numpy()
        for j, sd in enumerate(PT.SEEDS):
            S[j, tex] = np.load(ck / f"Sx_f{k}_s{sd}.npy")
    o = ex[KEY + ["fold"]].copy()
    o["p0"] = F.bag_prob(ex, list(S))
    o["prob"] = o.p0            # the trees' own decoder is not used by the blend
    o.to_parquet(OUT / "trees_extra_bywindow.parquet", index=False)
    allchk = {}
    for f in sorted(ck.glob("check_*.json")):
        allchk.update(json.load(open(f)))
    json.dump(allchk, open(OUT / "trees_refit_check.json", "w"), indent=1)
    log(f"wrote {OUT / 'trees_extra_bywindow.parquet'} ({len(o):,} rows)")


# ------------------------------------------------------------------ GRU
def new_gru_signals() -> pd.DataFrame:
    """Extra signal-periods that are not in the note-37 GRU map (key 'period|dev')."""
    ex = pd.read_parquet(OUT / "extras_detwin.parquet")
    ex = ex[ex.kind == "new_signal_period"]
    s = ex.groupby("DeviceId").fold.first().reset_index()
    s["period"] = np.where(s.DeviceId.str.endswith("@stg"), "stg", "dec")
    s["DeviceId"] = s.DeviceId.str.replace("@stg", "", regex=False)
    s["key"] = s.period + "|" + s.DeviceId
    return s.sort_values("key").reset_index(drop=True)


def stage_rasters(a) -> None:
    from neural import ncache2 as NC
    s = new_gru_signals()
    stg = set(s[s.period == "stg"].DeviceId)
    meta = pd.read_parquet(DC_WORK / "official" / "stg" / "cache" / "signal_meta.parquet", columns=["DeviceId"])
    devs = sorted(d for d in meta.DeviceId if str(d).lower() in stg)
    log(f"stg rasters for {len(devs)} of {len(stg)} signals -> {SIGDIR_STG}")
    NC.PERIODS["stg"]["sigdir"] = SIGDIR_STG
    NC.build("stg", devices=devs)
    dec = s[s.period == "dec"].DeviceId
    miss = [d for d in dec if not (NC.PERIODS["dec"]["sigdir"] / f"{d}.npz").exists()]
    assert not miss, miss


def stage_gru(a) -> None:
    import torch
    import phase_v3_net as PN  # noqa: F401  -- patches the GRU signal map (not used for scoring here)
    from neural import data2 as D2
    from neural import infer2 as I2
    from pieces_infer import subset
    D2.PERIODS["stg"]["sigdir"] = SIGDIR_STG     # only the new Sept-2026 DEV signals are scored here
    orig = I2.pieces

    def pieces_k4(period, start, secs, max_chunks):
        out = orig(period, start, secs, PN.GRID)
        if secs > PN.LONG_MIN * 60:
            out = [out[i] for i in subset(len(out), PN.K_LONG)]
        return out
    I2.pieces = pieces_k4
    s = new_gru_signals()
    assert not s.DeviceId.isin(locked()).any()
    table = D2.load_table(s, labelled_only=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parts = []
    for k in range(6):
        dest = OUT / f"gru_extra_f{k}.parquet"
        if dest.exists():
            parts.append(pd.read_parquet(dest))
            continue
        keys = [x for x in s.loc[s.fold == k, "key"] if x in table]
        if not keys:
            continue
        model = I2.load_model(str(PN.MODELDIR / f"p3_f{k}.pt"), device)
        t0 = time.time()
        df = I2.score_windows(model, table, keys, device, D2.Stores(), PN.GRID, 180, 6, 1024)
        df.to_parquet(dest, index=False)
        parts.append(df)
        log(f"fold {k}: {len(keys)} signals, {len(df):,} rows ({time.time()-t0:.0f}s)")
        del model
        torch.cuda.empty_cache()
    g = pd.concat(parts, ignore_index=True)
    g.to_parquet(OUT / "gru_extra_bywindow.parquet", index=False)
    log(f"GRU extras {len(g):,} rows, {g.DeviceId.nunique()} signal-periods")


# ------------------------------------------------------------------ blend
def stage_blend(a) -> None:
    import phase_v3_eval as PE
    B, BP, TE = PE.B, PE.BP, PE.TE
    _dec = BP.decode_oof
    BP.decode_oof = lambda pr, ctx, sim, meta, n_jobs=a.threads: _dec(pr, ctx, sim, meta, n_jobs=n_jobs)  # shared CPU
    lk = locked()
    fm = phase_folds()
    tn = PE.trees_new()
    tx = TE._norm(pd.read_parquet(OUT / "trees_extra_bywindow.parquet"))
    nn = PE.net_new()
    nx = TE._norm(pd.read_parquet(OUT / "gru_extra_bywindow.parquet"))
    tag, chk = ("v3blend", "new_blend") if a.variant == "blend" else ("v3trees", "new_trees")
    if a.variant == "trees":        # what candidate v2's predict.py feeds the function model (prob_lgbm)
        nn, nx = nn.iloc[:0], nx.iloc[:0]
    for d in (tn, tx, nn, nx):
        assert not PE.plain(d.DeviceId).isin(lk).any()
    assert not tx.merge(tn[KEY], on=KEY).shape[0], "extra rows overlap the note-37 OOF"
    folds_a = PE.fold_map()
    new_sp = set(tx.DeviceId) - set(tn.DeviceId)
    folds_b = pd.concat([folds_a, fm[fm.DeviceId.isin(new_sp)]], ignore_index=True)
    assert not folds_b.DeviceId.duplicated().any()
    lab = B.labels()
    ctx, sim = BP.load_context(set(folds_b.DeviceId))
    qa = PE.run_arm("A note37", tn[KEY + ["prob", "p0"]], nn, folds_a, lab, ctx, sim, set())
    lg_b = pd.concat([tn[KEY + ["prob", "p0"]], tx[KEY + ["prob", "p0"]]], ignore_index=True)
    qb = PE.run_arm("B +extras", lg_b, pd.concat([nn, nx], ignore_index=True), folds_b, lab, ctx, sim, new_sp)
    # check: arm A == the note-48 candidate blend top-1 on its rows
    rows = pd.read_parquet(DC_WORK / "final_v3_work" / "eval48" / "phase_rows.parquet",
                           columns=DET + [chk])
    t = qa.sort_values(DET + ["p2", "cand_phase"], ascending=[1, 1, 1, 0, 1]).groupby(DET, sort=False).first()
    j = rows.merge(t.cand_phase.rename("a").reset_index(), on=DET, how="left")
    same48 = float((j.a == j[chk]).mean())
    xk = tx[KEY]
    pa = qa[KEY + ["p2"]]
    pb = qb.merge(xk, on=KEY, how="inner")[KEY + ["p2"]]
    p = pd.concat([pa, pb], ignore_index=True)
    fk = frame_keys()
    cov = fk.merge(p[DET].drop_duplicates().assign(_h=1), on=DET, how="left")._h.notna().mean()
    # net coverage of the extras (the GRU only scores channels with a timing label, as in note 37)
    nb = qb.merge(xk, on=KEY)
    res = {f"arm_A_top1_equals_note48_{chk}": same48, "frame_detwin_covered": float(cov),
           "extras_rows": len(pb), "extras_net_coverage": float(nb.p_nn.notna().mean())}
    # accuracy vs official timing on the extras (labelled, phase greens) -- a pipeline check
    q = nb.copy()
    q = q[q.Phase.notna()]
    has = (q.cand_phase == q.Phase).groupby([q[c] for c in DET]).transform("max")
    q = q[has.astype(bool)]
    for col, nm in (("p2", "blend"), ("prob", "trees")):
        tt = q.sort_values(DET + [col, "cand_phase"], ascending=[1, 1, 1, 0, 1]).groupby(DET, sort=False).first()
        res[f"extras_top1_vs_timing_{nm}"] = {"n": len(tt), "acc": float((tt.cand_phase == tt.Phase).mean())}
    p["period"] = np.where(p.DeviceId.str.endswith("@stg"), "stg", "dec")
    p["DeviceId"] = p.DeviceId.str.replace("@stg", "", regex=False)
    for per in ("dec", "stg"):
        o = p[p.period == per].rename(columns={"p2": "prob"})[["DeviceId", "Detector", "win", "cand_phase", "prob"]]
        o["Detector"] = o.Detector.astype(np.int16)
        o["cand_phase"] = o.cand_phase.astype(np.int16)
        o.to_parquet(OUT / f"phase_pred_{tag}_{per}.parquet", index=False)
    json.dump(res, open(OUT / f"blend_{a.variant}.json", "w"), indent=1)
    log(json.dumps(res))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extras", "trees", "rasters", "gru", "blend"])
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--variant", default="blend", choices=["blend", "trees"],
                    help="trees = no GRU (the phase candidate v2 hands to its function model)")
    a = ap.parse_args()
    globals()[f"stage_{a.stage}"](a)
