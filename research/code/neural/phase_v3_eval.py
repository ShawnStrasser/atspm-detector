"""Note 37: stage 13 vs the phase retrain with the 71 released NEWTEST signals, six-fold OOF.

Two arms, one harness (the stage-13 / note-33 one: 0.5 blend of ranker bag and GRU BEFORE
the joint decoder; decoder re-fitted out of fold on the blended inputs; K = 4 pieces for
the net above 120 min):

  old  final_v1 trees OOF (`official/final_v1/oof_bywindow.parquet`) + stage-13 GRU OOF
       (`preds/gru2/gru2_oof_f*`, m5..h1; `trackB/pieces` lp_k4 for h3..full).  On the
       released 71 (which it never saw): the final_v1 trees (`trackA/v6/phase_pred_new_stg`)
       and the stage-13 final GRU (K = 4); those rows are scored but never trained on by the
       old arm's OOF decoder.
  new  `final_v3_work/phase_v3/work/trees_oof_bywindow.parquet` + `gru_preds/p3_oof_f*`.

Row sets: `same` = stage 13's own rows (blend_v2.common_frame: 1,532,182 rows, 701 signals);
`released` = labelled, phase greens in the window, detector actuated (det_n_on >= 1), both
nets scored it; `all` = same + released.  Paired signal-grouped bootstrap (2,000 draws) of
new - old on pooled detector-window accuracy.  Locked signals are asserted absent.

    python phase_v3_eval.py            -> final_v3_work/phase_v3/work/eval.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402
from common import DC_WORK  # noqa: E402
import blend_v2 as B  # noqa: E402
import blend_v2_predecode as BP  # noqa: E402
import trackb_eval as TE  # noqa: E402

W = DC_WORK / "final_v3_work" / "phase_v3" / "work"
KEY, DET, GRP = B.KEY, B.DET, BP.GRP
FAMS = TE.FAMS
LONGF = ("h3", "h6", "h24", "full")
REL = DC_WORK / "official" / "newtest_released.csv"
LOCKED = DC_WORK / "official" / "locked_v2.csv"
F4 = DC_WORK / "folds_v4.csv"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def plain(s):
    return s.str.replace("@stg", "", regex=False)


def released() -> set:
    return set(pd.read_csv(REL).DeviceId.str.lower())


def fold_map() -> pd.DataFrame:
    old = B.folds()
    f4 = pd.read_csv(F4)
    f4["DeviceId"] = f4.DeviceId.str.lower()
    r = f4[f4.DeviceId.isin(released())].copy()
    r["DeviceId"] = r.DeviceId + "@stg"
    return pd.concat([old, r[["DeviceId", "fold"]]], ignore_index=True)


def net_old() -> pd.DataFrame:
    s = TE._norm(B.gru_oof())
    s = s[~s.win.map(B.fam_of).isin(LONGF)][KEY + ["prob"]]
    p = TE._norm(pd.concat([pd.read_parquet(f) for f in sorted(
        (DC_WORK / "trackB" / "pieces").glob("gru2_pieces_f*.parquet"))], ignore_index=True))
    p = p[p.variant == "lp_k4"][KEY + ["prob"]]
    r = TE._norm(pd.read_parquet(W / "gru_preds" / "s13final_released_bywindow.parquet"))
    return pd.concat([s, p, r[KEY + ["prob"]]], ignore_index=True)


def net_new() -> pd.DataFrame:
    return pd.concat([TE._norm(pd.read_parquet(f))[KEY + ["prob"]] for f in sorted(
        (W / "gru_preds").glob("p3_oof_f*_bywindow.parquet"))], ignore_index=True)


def trees_new() -> pd.DataFrame:
    return TE._norm(pd.read_parquet(W / "trees_oof_bywindow.parquet"))


def trees_old(rel_keys: pd.DataFrame) -> pd.DataFrame:
    lg = TE._norm(pd.read_parquet(B.LG_OOF))[KEY + ["prob", "p0"]]
    v6 = pd.read_parquet(DC_WORK / "trackA" / "v6" / "phase_pred_new_stg.parquet")
    v6["DeviceId"] = v6.DeviceId.str.lower() + "@stg"
    v6 = TE._norm(v6).merge(rel_keys, on=KEY, how="inner")    # same detectors as the new pool
    for c in ("prob", "p0"):
        v6[c] = v6[c] / v6.groupby(DET)[c].transform("sum")
    return pd.concat([lg, v6[KEY + ["prob", "p0"]]], ignore_index=True)


def run_arm(name, lg, nn, folds, lab, ctx, sim, no_train: set) -> pd.DataFrame:
    lg = lg.merge(folds, on="DeviceId", how="inner")
    lg["dev_plain"] = plain(lg.DeviceId)
    lg = lg.merge(lab.rename(columns={"DeviceId": "dev_plain"}), on=["dev_plain", "Detector"],
                  how="left")
    lg["y"] = np.where(lg.Phase.notna(), (lg.cand_phase == lg.Phase).astype(float), np.nan)
    lg = lg.merge(nn.rename(columns={"prob": "p_nn"}), on=KEY, how="left")
    tot = lg.groupby(GRP)["p_nn"].transform("sum")
    lg["p_nn"] = np.where(tot > 0, lg.p_nn / tot.replace(0, np.nan), np.nan)
    meta = lg[KEY + ["Phase", "fold", "y"]].copy()
    meta.loc[lg.DeviceId.isin(no_train).to_numpy(), "y"] = np.nan   # scored, never trained on
    pr = lg[KEY].copy()
    pr["p0"] = np.where(lg.p_nn.notna(), 0.5 * lg.p0 + 0.5 * lg.p_nn, lg.p0)
    pr["p0"] = pr.p0 / pr.groupby(GRP)["p0"].transform("sum")
    log(f"[{name}] decoding {len(pr):,} rows, net covers {lg.p_nn.notna().mean():.1%}")
    out = BP.decode_oof(pr, ctx, sim, meta)
    q = lg[KEY + ["Phase", "fold", "prob", "p_nn"]].merge(out, on=KEY, how="left")
    q["p2"] = q.p2.fillna(0.0)
    return q


def score(q: pd.DataFrame, keep: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    d = keep.merge(q, on=KEY, how="left")
    for c in ("prob", "p_nn"):
        d[c] = d[c] / d.groupby(DET)[c].transform("sum")
    res, oks = {}, {}
    for col, nm in (("prob", "trees"), ("p_nn", "net"), ("p2", "blend")):
        dd = d if col != "p_nn" else d[d.groupby(DET)["p_nn"].transform("count") > 0]
        res[nm] = TE.flat(B.acc_by_fam(dd, col))
        t = B.top1(dd, col)
        oks[nm] = t.set_index(DET).ok
    ok = pd.DataFrame(oks)
    ok["fam"] = ok.index.get_level_values("win").map(B.fam_of)
    return res, ok


def boot(ok_old, ok_new, n=2000, seed=0) -> dict:
    out = {}
    for m in ("trees", "net", "blend"):
        j = pd.concat([ok_old[m].rename("o"), ok_new[m].rename("n")], axis=1).dropna()
        j["fam"] = j.index.get_level_values("win").map(B.fam_of)
        j["sig"] = j.index.get_level_values("DeviceId")
        out[m] = {}
        for fam in FAMS:
            g = j[j.fam == fam]
            if not len(g):
                continue
            s = g.groupby("sig").agg(d=("n", "sum"), o=("o", "sum"), k=("n", "size"))
            dd = (s.d - s.o).to_numpy(float)
            kk = s.k.to_numpy(float)
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, len(s), (n, len(s)))
            bs = dd[idx].sum(1) / kk[idx].sum(1)
            out[m][fam] = {"delta_pt": round(100 * dd.sum() / kk.sum(), 3),
                           "ci95_pt": [round(100 * float(np.quantile(bs, .025)), 3),
                                       round(100 * float(np.quantile(bs, .975)), 3)],
                           "n_detwin": int(kk.sum()), "n_signals": int(len(s)),
                           "err_old": int((1 - g.o).sum()), "err_new": int((1 - g.n).sum())}
    return out


def main() -> None:
    lk = set(pd.read_csv(LOCKED).DeviceId.str.lower())
    rel = released()
    rel_stg = {d + "@stg" for d in rel}
    keep_same = B.common_frame(TE._norm(B.gru_oof()))[KEY].copy()
    log(f"same rows {len(keep_same):,}")
    tn = trees_new()
    assert not plain(tn.DeviceId).isin(lk).any()
    rel_keys = tn.loc[tn.DeviceId.isin(rel_stg), KEY]
    folds = fold_map()
    lab = B.labels()
    ctx, sim = BP.load_context(set(folds.DeviceId))
    nn_o, nn_n = net_old(), net_new()
    assert not plain(nn_n.DeviceId).isin(lk).any() and not plain(nn_o.DeviceId).isin(lk).any()

    # released rows: labelled, labelled phase greens, detector actuated, both nets scored it
    r = rel_keys.merge(ctx[["DeviceId", "Detector", "win", "cand_phase", "det_n_on"]],
                       on=KEY, how="left")
    r["dev_plain"] = plain(r.DeviceId)
    r = r.merge(lab.rename(columns={"DeviceId": "dev_plain"}), on=["dev_plain", "Detector"])
    has = (r.cand_phase == r.Phase).groupby([r[c] for c in DET]).transform("max")
    act = r.groupby(DET)["det_n_on"].transform("max")
    r = r[has & (act >= 1)]
    cov_o = nn_o[nn_o.DeviceId.isin(rel_stg)][DET].drop_duplicates()
    cov_n = nn_n[nn_n.DeviceId.isin(rel_stg)][DET].drop_duplicates()
    r = r.merge(cov_o, on=DET).merge(cov_n, on=DET)
    keep_rel = r[KEY].drop_duplicates()
    keep_all = pd.concat([keep_same, keep_rel], ignore_index=True)
    log(f"released rows {len(keep_rel):,} ({keep_rel.groupby(DET).ngroups:,} det-windows, "
        f"{keep_rel.DeviceId.nunique()} signals)")

    # rule check: the det_n_on rule on the old rows vs stage 13's GRU n_act rule
    res = {"n_rows": {"same": len(keep_same), "released": len(keep_rel), "all": len(keep_all)}}
    q_old = run_arm("old", trees_old(rel_keys), nn_o, folds, lab, ctx, sim, rel_stg)
    q_new = run_arm("new", tn[KEY + ["prob", "p0"]], nn_n, folds, lab, ctx, sim, set())
    for rs, keep in (("same", keep_same), ("released", keep_rel), ("all", keep_all)):
        ro, oo = score(q_old, keep)
        rn, on_ = score(q_new, keep)
        res[rs] = {"old": ro, "new": rn, "bootstrap_new_minus_old": boot(oo, on_)}
        for arm, rr in (("old", ro), ("new", rn)):
            for m in ("trees", "net", "blend"):
                log(f"{rs:8s} {arm} {m:5s} " + json.dumps(rr[m]))
        json.dump(res, open(W / "eval.json", "w"), indent=1)
    # recipe-noise control: the same tree code on the 709 old signals only, + the stage-13 GRU
    ctlf = W / "trees_oof_control_bywindow.parquet"
    if ctlf.exists():
        tc = TE._norm(pd.read_parquet(ctlf))[KEY + ["prob", "p0"]]
        q_ctl = run_arm("ctl", tc, nn_o, folds, lab, ctx, sim, set())
        rc, oc = score(q_ctl, keep_same)
        ro, oo = score(q_old, keep_same)
        rn, on_ = score(q_new, keep_same)
        res["same_control"] = {"ctl": rc, "ctl_minus_old": boot(oo, oc),
                               "new_minus_ctl": boot(oc, on_)}
        for m in ("trees", "blend"):
            log(f"same     ctl {m:5s} " + json.dumps(rc[m]))
        json.dump(res, open(W / "eval.json", "w"), indent=1)
    # per-fold blend at 30 min, same rows
    for arm, q in (("old", q_old), ("new", q_new)):
        d = keep_same.merge(q, on=KEY, how="left")
        res[f"per_fold_m30_blend_{arm}"] = {
            int(k): round(float(B.acc_by_fam(d[d.fold == k], "p2")["m30"]["acc"]), 5)
            for k in range(6)}
    # sanity: old arm, same rows, long windows == note 33's lp_k4 refit
    try:
        pz = json.load(open(DC_WORK / "trackB" / "eval" / "pieces.json"))
        res["sanity_note33_lp_k4_refit"] = {f: v["acc"] for f, v in
                                            pz["variants"]["lp_k4"]["refit"]["all"].items()}
    except Exception as e:  # noqa: BLE001
        res["sanity_note33_lp_k4_refit"] = str(e)
    json.dump(res, open(W / "eval.json", "w"), indent=1)
    log(f"wrote {W / 'eval.json'}")


if __name__ == "__main__":
    main()
