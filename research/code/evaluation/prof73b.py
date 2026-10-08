"""Note 73b: six-fold OOF accuracy check of speed idea 1 (note 73) -- the GRU scores only the candidate phases whose
TREE probability is >= thr; the other candidates keep the tree probability alone (exactly production's `gru_blend.mix`
for rows the network has no opinion on), GRU renormalised over the kept candidates.

Saved OOF only (cand64 phase pipeline): trees = note-57 3-seed bag (`p0_bag`), GRU = p3 OOF (K = 4 past 2 h), 0.5 / 0.5
blend BEFORE the joint decoder, decoder re-fitted out of fold on folds_v4 exactly as `cand64.decode` (threads 4).
Scored with cand64's phase rows (everything / realistic), >= 30-min pool + 5 / 10 min, paired signal bootstrap vs the
current candidate (no filter).  Function: the function frame's phase input is not rebuilt here; reported = share of
detector-windows whose decoded top phase changes (an upper bound on the function rows that can move).
Locked_v2 asserted absent (cand64).  CPU, <= 4 threads.

    python prof73b.py [--thr 0.003,0.01,0.03]     -> %DC_WORK%/prof73/oof73b.json, oof73b_q.parquet
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: F401,E402
import cand64 as C  # noqa: E402

C.NJ = 4
OUT = C.DC_WORK / "prof73"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def blend(q: pd.DataFrame, keep: np.ndarray) -> np.ndarray:
    grp = [q[c] for c in C.DET]
    nn = np.where(keep, q.p_nn.to_numpy(), np.nan)
    tot = pd.Series(np.nan_to_num(nn)).groupby(grp).transform("sum").to_numpy()
    nn = np.where(~np.isnan(nn) & (tot > 0), nn / np.where(tot > 0, tot, 1.0), np.nan)
    base = q.p0_bag.to_numpy()
    p0 = np.where(np.isnan(nn), base, 0.5 * base + 0.5 * nn)
    return p0 / pd.Series(p0).groupby(grp).transform("sum").to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", default="0.003,0.01,0.03")
    a = ap.parse_args()
    thrs = [float(x) for x in a.thr.split(",")]
    lk = C.locked()
    need = C.KEY4 + ["fold", "Phase", "y", "cand_green_share", "call43_per_cycle", "det_n_on", "log_win_hours"]
    comb = pd.read_parquet(C.T57P / "pool.parquet", columns=need)
    simc = pd.read_parquet(C.T57P / "pool_sim.parquet")
    assert not C.plain(comb.DeviceId).isin(lk).any()
    q = pd.read_parquet(C.OUT / "phase_oof.parquet")
    for c in C.KEY4:
        assert (q[c].to_numpy() == comb[c].to_numpy()).all(), c
    s = q.groupby(C.DET).p0_bag.transform("sum")
    log(f"rows {len(q):,}; p0_bag per-detector sum {s.min():.4f}..{s.max():.4f}; GRU covers {q.p_nn.notna().mean():.3f}")
    p_ref = blend(q, np.ones(len(q), bool))
    log(f"reference blend reproduces p0_blend: max |diff| {np.abs(p_ref - q.p0_blend.to_numpy()).max():.2e}")
    res = {"cover": {}}
    fq = OUT / "oof73b_q.parquet"
    done = pd.read_parquet(fq) if fq.exists() else q[C.KEY4].copy()
    if "p2_ref4" not in done:                        # decoder determinism at 4 threads vs cand64's 6
        done["p2_ref4"] = C.decode(comb, simc, q.p0_blend.to_numpy())
        done.to_parquet(fq, index=False)
    res["ref4_vs_cand_max_abs"] = float(np.abs(done.p2_ref4.to_numpy() - q.p2_cand.to_numpy()).max())
    log(f"re-decode at 4 threads vs p2_cand: max |diff| {res['ref4_vs_cand_max_abs']:.2e}")
    has = q.p_nn.notna().to_numpy()
    for t in thrs:
        col = f"p2_thr{t:g}"
        keep = q.p0_bag.to_numpy() >= t
        res["cover"][col] = {"pairs_kept_of_gru_rows": float((keep & has).sum() / has.sum())}
        if col not in done:
            log(f"thr {t}: GRU on {res['cover'][col]['pairs_kept_of_gru_rows']:.3f} of its pair rows; decoding")
            done[col] = C.decode(comb, simc, blend(q, keep))
            done.to_parquet(fq, index=False)
    q = q.join(done.drop(columns=C.KEY4))
    rows = C.phase_rows()
    sig = rows.dev_plain.to_numpy()
    arms = {"ref4": "p2_ref4", **{f"thr{t:g}": f"p2_thr{t:g}" for t in thrs}}
    for nm, col in arms.items():
        rows[f"pred_{nm}"], _ = C.top1(q, col, rows)
        rows[f"okE_{nm}"] = (rows[f"pred_{nm}"] == rows.Phase).astype(float)
    # realistic credit as cand64.phase_rows: switch / additional-call phases count as right
    import t57_phase as T57
    alt = T57.score_rows().set_index(C.DET).alt.reindex(rows.set_index(C.DET).index).to_numpy()
    for nm in arms:
        rows[f"okR_{nm}"] = (rows[f"okE_{nm}"].astype(bool) |
                             np.array([p in s_ for p, s_ in zip(rows[f"pred_{nm}"], alt)])).astype(float)
    res["acc"], res["delta_vs_cand"], res["ref_check"] = {}, {}, {}
    for sname, okp in (("everything", "okE"), ("realistic", "okR")):
        mset = rows[sname].to_numpy()
        for pool, fams in C.POOLS.items():
            m = mset & rows.fam.isin(fams).to_numpy()
            base = rows[f"{okp}_cand"].to_numpy()[m]
            res["ref_check"].setdefault(sname, {})[pool] = int((rows[f"{okp}_ref4"].to_numpy()[m] != base).sum())
            res["acc"].setdefault("cand", {}).setdefault(sname, {})[pool] = C.acc_ci(base, sig[m])
            for nm in arms:
                if nm == "ref4":
                    continue
                v = rows[f"{okp}_{nm}"].to_numpy()[m]
                res["acc"].setdefault(nm, {}).setdefault(sname, {})[pool] = C.acc_ci(v, sig[m])
                res["delta_vs_cand"].setdefault(nm, {}).setdefault(sname, {})[pool] = C.delta_ci(base, v, sig[m])
    # top-phase change on every detector-window of the pool (function's phase input), by pool
    tops = {}
    for nm, col in {"cand": "p2_cand", **arms}.items():
        d = q[C.KEY4 + [col]].sort_values(C.DET + [col], ascending=[True, True, True, False])
        tops[nm] = d.groupby(C.DET, sort=True).first().cand_phase
    fam = tops["cand"].index.get_level_values("win").map(C.fam_of)
    res["top_change_share"] = {nm: {pool: float((tops[nm] != tops["cand"]).to_numpy()[fam.isin(f)].mean())
                                    for pool, f in C.POOLS.items()} for nm in arms}
    json.dump(res, open(OUT / "oof73b.json", "w"), indent=1)
    for nm in arms:
        if nm == "ref4":
            continue
        for sname in ("everything", "realistic"):
            log(f"{nm} {sname[:4]}: " + " | ".join(f"{p} {res['acc'][nm][sname][p]} d {res['delta_vs_cand'][nm][sname][p]}"
                                                   for p in C.POOLS))
    log(f"cand: {json.dumps(res['acc']['cand'])}")
    log(f"ref check (rows differing): {res['ref_check']}; top change: {json.dumps(res['top_change_share'])}")
    log(f"cover: {json.dumps(res['cover'])}")


if __name__ == "__main__":
    main()
