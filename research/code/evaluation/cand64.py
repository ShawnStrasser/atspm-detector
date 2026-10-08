"""Note 64: ONE integrated six-fold OOF scorer for the current candidate (every later change is judged on the whole pipeline).

Candidate (all OOF, six folds of folds_v4; locked_v2 asserted absent everywhere; CPU, <= 6 threads):
  phase     note-57 trees (261 features, 3-seed bag; `trees57/phase/full`) blended 0.5 / 0.5 with the current GRU (p3 OOF,
            K = 4 pieces past 2 h; note 37) BEFORE the joint decoder; decoder re-fitted out of fold on the blended inputs
            (note-57 recipe, folds_v4).  Rows the GRU did not score keep the trees alone (as in note 37).
  function  note-57 229-feature arm, 3 seeds (blend phase input = frame v6e) -> per-lane decode on D lanes (note 58), stack
            pick, stack_loser nonatspm_ap (notes 56 / 56b) -> short-window twin decode (cy, thr .4; note 62; NO routed short
            head).  OPTIONAL: TCN fj head blended at tree weight 0.6 BEFORE the decode (note 63) for every fold whose
            `tcn53/fpreds/fj[_sN]_f{k}.parquet` exists (seeds averaged); reported on that fold subset with and without.
  side      lanes D (`lanes/ln8/lanes_D.func`), setback sb7 `all` P50 (note 58), health v5 (note 47), night speed sp1
            (note 60) - attached per row as they are (their own OOF; they read the 229-arm function, not the fj blend).
Scoring: phase = note-57 two numbers (everything = timing label, >= 5 actuations; realistic = minus label-check fail /
misconfigured and high-confidence print-phase disagreements, switch / additional-call phase credited); function = ATSPM-only
stack-aware score (note 54/55; everything / realistic = setA_s / setR_s minus stale Dec roles). Headline = >= 30-min pool
(30 min / 1 h / 3 h / 6 h / 24 h / full, rows pooled; user rule 2026-10-01); 5 / 10 min secondary. 95 % CIs = signal
bootstrap (2,000).

    python cand64.py all            # phase (cached after the first run) + function + score  -> %DC_WORK%/cand64/
    python cand64.py phase [--force] [--check]   # --check also re-decodes the trees alone (must reproduce note 57)
    python cand64.py function | score
    python cand64.py v2ref          # (re)attach the final_v2 reference arm to function_rows, then score
Re-run `all` whenever new fj folds land: only the function and score stages redo (~3 min).

Reference arm final_v2 (= the published beta, `model/`; added 2026-10-01, nothing re-trained, saved OOF only):
  phase     note 48's final_v2 arm (`final_v3_work/eval48/phase_rows.parquet`): final_v1 trees + stage-13 GRU blended up
            to 1 h, final_v1 trees + decoder alone from 3 h.  Its own fold map (note 37; differs from folds_v4 on 22
            signals; every row is still out of fold for its model).  Scored on the cand64 rows it covers (99.94 %).
  function  final_v2 ships a 5-class head and no lanes / decode: argmax as it ships.  Main = note 48's final_v2 head
            family, note-25 b7 OOF collapsed to 5 classes (`trackA/w1/P_b7_s*.npy`, seed mean; folds = folds_v4);
            sensitivity = the stage-11 OOF of the shipped booster itself (`function_v4/function_oof_v4_bywindow`,
            labelled channels only).  Same ATSPM stack-aware credit as the candidate, on the rows each arm covers.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: F401,E402
from common import DC_WORK, N_FOLDS  # noqa: E402

OUT = DC_WORK / "cand64"
T57P = DC_WORK / "trees57" / "phase"
FPRED = DC_WORK / "tcn53" / "fpreds"
KEY4 = ["DeviceId", "Detector", "win", "cand_phase"]
DET = ["DeviceId", "Detector", "win"]
NJ = 6
W_TREE = 0.6                       # note 63: fixed tree weight of the fj blend
TWIN_THR = 0.4                     # note 62
GE30 = ["m30", "h1", "h3", "h6", "h24", "full"]
POOLS = {"ge30": GE30, "m5": ["m5"], "m10": ["m10"]}
FAMS = ["m5", "m10", "m30", "h1", "h3", "h6", "h24", "full"]
V2_PHASE = DC_WORK / "final_v3_work" / "eval48" / "phase_rows.parquet"
V2_NET = ("m5", "m10", "m30", "h1")       # final_v2's network runs up to 120 min (eval48_phase.NET_V2)
V2_B7 = DC_WORK / "trackA" / "w1"
V2_SHIP = DC_WORK / "function_v4" / "function_oof_v4_bywindow.parquet"
C5 = ["Advance", "Presence", "Count", "Yellow_Red", "Other"]
V2_ARMS = ("v2", "v2ship")                # function: b7 -> 5 classes (note 48) / stage-11 shipped booster OOF


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def locked() -> set:
    return set(pd.read_csv(DC_WORK / "official" / "locked_v2.csv").DeviceId.str.lower())


def plain(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("@stg", "", regex=False).str.lower()


def fam_of(w: str) -> str:
    return "full" if w.startswith("full") else w.split("_")[0]


def acc_ci(ok: np.ndarray, sig: np.ndarray, n=2000, seed=64) -> list:
    """pooled accuracy and its signal-bootstrap 95 % CI (fractions)."""
    if len(ok) == 0:
        return [None, None, None]
    u, inv = np.unique(sig, return_inverse=True)
    s, c = np.bincount(inv, ok, len(u)), np.bincount(inv, minlength=len(u))
    idx = np.random.default_rng(seed).integers(0, len(u), (n, len(u)))
    b = s[idx].sum(1) / c[idx].sum(1)
    return [round(float(ok.mean()), 4), round(float(np.quantile(b, .025)), 4), round(float(np.quantile(b, .975)), 4)]


def delta_ci(a: np.ndarray, b: np.ndarray, sig: np.ndarray, n=2000, seed=64) -> list:
    """b - a in pt, paired signal bootstrap 95 % CI."""
    if len(a) == 0:
        return [None, None, None]
    u, inv = np.unique(sig, return_inverse=True)
    sa, sb, c = np.bincount(inv, a, len(u)), np.bincount(inv, b, len(u)), np.bincount(inv, minlength=len(u))
    idx = np.random.default_rng(seed).integers(0, len(u), (n, len(u)))
    d = (sb[idx].sum(1) - sa[idx].sum(1)) / c[idx].sum(1)
    return [round(100 * float(b.mean() - a.mean()), 3), round(100 * float(np.quantile(d, .025)), 3),
            round(100 * float(np.quantile(d, .975)), 3)]


# ================================================================================================ phase
def decode(comb: pd.DataFrame, simc: pd.DataFrame, p0: np.ndarray) -> np.ndarray:
    """note-57 decoder recipe (OOF over folds_v4), on first-stage probabilities p0 (pool row order)."""
    import lightgbm as lgb
    import train_official as T
    import decode_train as dec
    cols2 = dec.BASE_COLS + dec.SIM_COLS + dec.ADJ_COLS
    pr = comb[KEY4].copy()
    pr["p0"] = p0
    X = dec.assemble(pr, pairs=comb, sim=simc)
    X = X.merge(comb[KEY4 + ["Phase", "fold", "y"]], on=KEY4, how="left")
    X = X.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
    s2 = np.zeros(len(X))
    labX = X.Phase.notna().to_numpy()
    fx = X.fold.to_numpy()
    for k in range(N_FOLDS):
        te = fx == k
        inner = (k + 1) % N_FOLDS
        base = (~te) & labX
        tr, va = X[base & (fx != inner)], X[base & (fx == inner)]
        P = dict(T.BIN_PARAMS, n_jobs=NJ, bagging_seed=0, feature_fraction_seed=100, seed=0)
        n = P.pop("n_estimators")
        m = lgb.LGBMClassifier(n_estimators=n, **P)
        m.fit(tr[cols2], tr.y, eval_set=[(va[cols2], va.y)], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
        s2[te] = m.predict_proba(X.loc[te, cols2])[:, 1]
        log(f"    decoder fold {k}: {m.best_iteration_} trees")
    X["p2"] = T.norm_prob(X, s2)
    return comb[KEY4].merge(X[KEY4 + ["p2"]], on=KEY4, how="left").p2.to_numpy()


def stage_phase(a):
    f = OUT / "phase_oof.parquet"
    if f.exists() and not a.force:
        log(f"phase OOF cached: {f}")
        return
    import phase_v3_eval as PE
    lk = locked()
    need = KEY4 + ["fold", "Phase", "y", "cand_green_share", "call43_per_cycle", "det_n_on", "log_win_hours"]
    comb = pd.read_parquet(T57P / "pool.parquet", columns=need)
    simc = pd.read_parquet(T57P / "pool_sim.parquet")
    assert not plain(comb.DeviceId).isin(lk).any()
    oof = pd.read_parquet(T57P / "full" / "oof.parquet", columns=KEY4 + ["p0_bag", "p2_bag"])
    q = comb[KEY4].merge(oof, on=KEY4, how="left")
    assert q.p0_bag.notna().all()
    nn = PE.net_new().rename(columns={"prob": "p_nn"})
    assert not plain(nn.DeviceId).isin(lk).any()
    nn = nn.astype({"Detector": comb.Detector.dtype, "cand_phase": comb.cand_phase.dtype})
    q = q.merge(nn, on=KEY4, how="left")
    tot = q.groupby(DET)["p_nn"].transform("sum")
    q["p_nn"] = np.where(tot > 0, q.p_nn / tot.replace(0, np.nan), np.nan)
    p0 = np.where(q.p_nn.notna(), 0.5 * q.p0_bag + 0.5 * q.p_nn, q.p0_bag)
    p0 = p0 / pd.Series(p0).groupby([q[c] for c in DET]).transform("sum").to_numpy()
    log(f"blend: net covers {q.p_nn.notna().mean():.3f} of {len(q):,} pair rows")
    q["p0_blend"] = p0
    q["p2_cand"] = decode(comb, simc, p0)
    if a.check:
        q["p2_trees_redecode"] = decode(comb, simc, q.p0_bag.to_numpy())
    OUT.mkdir(parents=True, exist_ok=True)
    q.rename(columns={"p2_bag": "p2_trees"}).to_parquet(f, index=False)
    log(f"wrote {f}")


def top1(q: pd.DataFrame, col: str, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    d = q[KEY4 + [col]].merge(rows[DET], on=DET, how="inner")
    d = d.sort_values(DET + [col, "cand_phase"], ascending=[True, True, True, False, True])
    t = d.groupby(DET, sort=False).first().reset_index()
    m = rows[DET].merge(t[DET + ["cand_phase", col]], on=DET, how="left")
    return m.cand_phase.to_numpy(), m[col].to_numpy()


def phase_rows() -> pd.DataFrame:
    import t57_phase as T57
    rows = T57.score_rows().copy()
    q = pd.read_parquet(OUT / "phase_oof.parquet")
    q = q.astype({"Detector": rows.Detector.dtype})
    for col, nm in (("p2_cand", "cand"), ("p2_trees", "trees")) + \
            ((("p2_trees_redecode", "trees_redecode"),) if "p2_trees_redecode" in q else ()):
        rows[f"pred_{nm}"], rows[f"p_{nm}"] = top1(q, col, rows)
    # runner-up of the candidate (for the review sheet)
    d = q[KEY4 + ["p2_cand"]].merge(rows[DET], on=DET, how="inner")
    d = d.merge(rows[DET + ["Phase"]], on=DET, how="left")
    lab_p = d[d.cand_phase == d.Phase].set_index(DET).p2_cand
    rows["p_cand_label"] = rows.set_index(DET).index.map(lab_p).to_numpy()
    # reference arm final_v2 (note 48's saved OOF; not a `pred_` arm because it does not cover every row)
    e = pd.read_parquet(V2_PHASE, columns=DET + ["old_trees", "old_blend"]).astype({"Detector": rows.Detector.dtype})
    assert not e.duplicated(DET).any()
    n0 = len(rows)
    rows = rows.merge(e, on=DET, how="left")
    assert len(rows) == n0
    rows["v2_pred"] = np.where(rows.fam.isin(V2_NET), rows.old_blend, rows.old_trees)
    rows["has_v2"] = rows.v2_pred.notna()
    rows = rows.drop(columns=["old_trees", "old_blend"])
    for nm, col in [(c[5:], c) for c in rows.columns if c.startswith("pred_")] + [("v2", "v2_pred")]:
        rows[f"okE_{nm}"] = (rows[col] == rows.Phase).astype(float)
        rows[f"okR_{nm}"] = (rows[f"okE_{nm}"].astype(bool) |
                             np.array([p in s for p, s in zip(rows[col], rows.alt)])).astype(float)
    return rows.drop(columns=["alt"])


def score_phase(res: dict):
    r = phase_rows()
    r.to_parquet(OUT / "phase_rows.parquet", index=False)
    sig = r.dev_plain.to_numpy()
    arms = [c[5:] for c in r.columns if c.startswith("pred_")]
    out = {"n": {}, "acc": {}, "delta_cand_vs_trees": {}, "by_family_mean_of_windows": {}}
    for sname, ok_p in (("everything", "okE"), ("realistic", "okR")):
        mset = r[sname].to_numpy()
        for pool, fams in POOLS.items():
            m = mset & r.fam.isin(fams).to_numpy()
            out["n"].setdefault(sname, {})[pool] = [int(m.sum()), int(len(set(sig[m])))]
            for arm in arms:
                out["acc"].setdefault(arm, {}).setdefault(sname, {})[pool] = acc_ci(r[f"{ok_p}_{arm}"].to_numpy()[m], sig[m])
            out["delta_cand_vs_trees"].setdefault(sname, {})[pool] = delta_ci(
                r[f"{ok_p}_trees"].to_numpy()[m], r[f"{ok_p}_cand"].to_numpy()[m], sig[m])
            mv = m & r.has_v2.to_numpy()
            ov, oc = r[f"{ok_p}_v2"].to_numpy()[mv], r[f"{ok_p}_cand"].to_numpy()[mv]
            out.setdefault("final_v2_ref", {}).setdefault(sname, {})[pool] = {
                "n": int(mv.sum()), "signals": int(len(set(sig[mv]))), "coverage": round(float(mv.sum() / m.sum()), 4),
                "final_v2": acc_ci(ov, sig[mv]), "cand": acc_ci(oc, sig[mv]),
                "delta_pt_cand_minus_v2": delta_ci(ov, oc, sig[mv])}
        for arm in arms:
            o = r[f"{ok_p}_{arm}"].to_numpy()
            out["by_family_mean_of_windows"].setdefault(arm, {})[sname] = {
                f: round(float(pd.Series(o[mset & (r.fam == f).to_numpy()]).groupby(
                    r.win.to_numpy()[mset & (r.fam == f).to_numpy()]).mean().mean()), 5)
                for f in FAMS if (mset & (r.fam == f).to_numpy()).any()}
    res["phase"] = out
    for arm in arms:
        for s in ("everything", "realistic"):
            log(f"phase {arm:15s} {s[:4]} " + " ".join(f"{p} {v}" for p, v in out["acc"][arm][s].items()))
    for s in ("everything", "realistic"):
        log(f"phase cand - trees {s[:4]} " + " ".join(f"{p} {v}" for p, v in out["delta_cand_vs_trees"][s].items()))
    for s in ("everything", "realistic"):
        for p, v in out["final_v2_ref"][s].items():
            log(f"phase final_v2 ref {s[:4]} {p}: v2 {v['final_v2']} cand {v['cand']} d {v['delta_pt_cand_minus_v2']} "
                f"(n {v['n']}, cov {v['coverage']})")


# ================================================================================================ function
def fj_probs(fr: pd.DataFrame) -> tuple[np.ndarray, list]:
    """frame-aligned fj probabilities (NaN where no fj prediction), seeds averaged per fold; folds present."""
    pc = [f"P_{c}" for c in ("Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other")]
    Pn = np.full((len(fr), 7), np.nan)
    have = []
    key = ["DeviceId", "Detector", "period", "win"]
    for k in range(N_FOLDS):
        fs = sorted(set(FPRED.glob(f"fj_f{k}.parquet")) | set(FPRED.glob(f"fj_s*_f{k}.parquet")))
        if not fs:
            continue
        parts = []
        for f in fs:
            n = pd.read_parquet(f)
            n["DeviceId"] = n.DeviceId.str.lower()
            parts.append(n.astype({"Detector": fr.Detector.dtype}).set_index(key)[pc])
        n = sum(parts) / len(parts)
        m = fr[key].merge(n.reset_index(), on=key, how="left")[pc].to_numpy(float)
        fk = fr.fold.to_numpy() == k
        Pn[fk] = m[fk]
        have.append({"fold": k, "files": [f.name for f in fs], "coverage": round(float((~np.isnan(m[fk, 0])).mean()), 4)})
    return Pn, have


def stage_function(a):
    import s59_step6 as S59
    import s62_short as S62
    import atspm_score as S
    C7 = S59.C7
    fr, rows = S59.scoring_frame(S59.PICK_TAG)
    fr = fr.copy()
    lk = locked()
    assert not fr.DeviceId.isin(lk).any()
    Pt, seeds = S59.spec_probs("base", fr)
    assert seeds == [0, 1, 2], seeds
    Pn, have = fj_probs(fr)
    hasn = ~np.isnan(Pn[:, 0])
    Pb = np.where(hasn[:, None], W_TREE * Pt + (1 - W_TREE) * np.nan_to_num(Pn), Pt)
    pairs = pd.read_parquet(S62.PAIRS_F)
    short = fr.wgroup.isin(S62.SHORT).to_numpy() & (fr.det_n_on >= 5).to_numpy()
    tw = S62.twin_tokens(fr, pairs, TWIN_THR, 1.0, 0)

    def run(P):
        for i, c in enumerate(C7):
            fr[f"P_{c}"] = P[:, i]
        pr = S.decode(fr, "lanes5g", "greedy", "strict", pick=True)
        pr = S62.twin_decode(P, pr, tw, short, ("Count", "Yellow_Red"), "cy")
        cr = {s: S.credit(fr, pr, "truth_v3s", r, True) for s, r in rows.items()}
        return pr, cr
    pred_t, cr_t = run(Pt)
    pred_b, cr_b = run(Pb) if hasn.any() else (pred_t, cr_t)
    # per-row table (every frame row; scoring flags per set)
    out = fr[["DeviceId", "Detector", "period", "win", "wgroup", "fold", "pred_phase", "det_n_on", "truth_v3s",
              "validated", "stack_group", "stack_role", "lanes5g"]].copy()
    out["pred_trees"] = np.array(C7, object)[pred_t]
    out["pred_fj"] = np.array(C7, object)[pred_b]
    out["has_fj"] = hasn
    out["p_pred_trees"] = Pt[np.arange(len(fr)), pred_t]
    out["p_pred_fj"] = Pb[np.arange(len(fr)), pred_b]
    out["p_truth_trees"] = [Pt[i, C7.index(t)] if isinstance(t, str) and t in C7 else np.nan
                            for i, t in enumerate(fr.truth_v3s.to_numpy(object))]
    for i, c in enumerate(C7):
        out[f"Pt_{c}"] = Pt[:, i].astype(np.float32)
    for arm, cr in (("trees", cr_t), ("fj", cr_b)):
        for s, r in rows.items():
            d = cr[s]
            col_ok, col_err = np.full(len(fr), np.nan), np.full(len(fr), None, object)
            col_ok[r] = d.ok_a.to_numpy(float)
            col_err[r] = d.err.to_numpy(object)
            out[f"ok_{s[0].upper()}_{arm}"] = col_ok
            out[f"err_{s[0].upper()}_{arm}"] = col_err
    out = attach_side(out)
    out = final_v2_function(out)
    assert not out.DeviceId.isin(lk).any()
    out.to_parquet(OUT / "function_rows.parquet", index=False)
    json.dump({"fj_folds": have, "n_rows": len(out), "built": time.strftime("%Y-%m-%d %H:%M")},
              open(OUT / "function_meta.json", "w"), indent=1)
    log(f"function rows {len(out):,}; fj folds {[h['fold'] for h in have]}")


def final_v2_probs(out: pd.DataFrame) -> dict:
    """{arm: (P5 aligned to out rows, NaN where the arm has no OOF)} - final_v2's 5-class head, saved OOF only."""
    key = ["DeviceId", "Detector", "period", "win"]
    res = {}
    k = pd.read_parquet(V2_B7 / "keys.parquet", columns=key)
    fs = sorted(V2_B7.glob("P_b7_s*.npy"))
    assert len(fs) == 3, fs
    P = np.mean([np.load(f) for f in fs], axis=0)          # b7 collapsed to CLASSES5 (w1 `collapse`)
    assert P.shape == (len(k), 5)
    k["DeviceId"] = k.DeviceId.str.lower()
    k["_i"] = np.arange(len(k))
    idx = out[key].merge(k.astype({"Detector": out.Detector.dtype}), on=key, how="left")._i.to_numpy()
    Pa = np.full((len(out), 5), np.nan)
    h = ~np.isnan(idx)
    Pa[h] = P[idx[h].astype(int)]
    res["v2"] = Pa
    o = pd.read_parquet(V2_SHIP, columns=key + ["p_advance", "p_presence", "p_count", "p_yellow_red", "p_other"])
    o["DeviceId"] = o.DeviceId.str.lower()
    m = out[key].merge(o.astype({"Detector": out.Detector.dtype}), on=key, how="left")
    res["v2ship"] = m[["p_advance", "p_presence", "p_count", "p_yellow_red", "p_other"]].to_numpy(float)
    return res


def final_v2_function(out: pd.DataFrame) -> pd.DataFrame:
    """final_v2 reference arms: argmax of the 5-class head (no lanes / decode), the candidate's ATSPM stack-aware credit
    (atspm_score.credit) on the rows each arm covers; rows it does not cover stay NaN."""
    import atspm_score as S
    C7 = list(S.C7)
    out = out.drop(columns=[c for c in out.columns if any(c.endswith(f"_{a}") for a in V2_ARMS)
                            and c.split("_")[0] in ("ok", "err", "pred", "p")], errors="ignore")
    for arm, P5 in final_v2_probs(out).items():
        has = ~np.isnan(P5[:, 0])
        P7 = np.zeros((len(out), 7))
        for j, c in enumerate(C5):
            P7[:, C7.index(c)] = np.nan_to_num(P5[:, j])
        pred = np.where(has, np.array([C7.index(c) for c in C5])[np.nan_to_num(P5, nan=-1).argmax(1)], C7.index("Other"))
        fr = out[["DeviceId", "period", "win", "wgroup", "truth_v3s", "stack_group", "stack_role"]].copy()
        for i, c in enumerate(C7):
            fr[f"P_{c}"] = P7[:, i]
        out[f"pred_{arm}"] = np.where(has, np.array(C7, object)[pred], None)
        out[f"p_pred_{arm}"] = np.where(has, P7[np.arange(len(out)), pred], np.nan)
        for s in ("E", "R"):
            r = np.flatnonzero(out[f"ok_{s}_trees"].notna().to_numpy() & has)
            d = S.credit(fr, pred, "truth_v3s", r, True)
            col_ok, col_err = np.full(len(out), np.nan), np.full(len(out), None, object)
            col_ok[r] = d.ok_a.to_numpy(float)
            col_err[r] = d.err.to_numpy(object)
            out[f"ok_{s}_{arm}"], out[f"err_{s}_{arm}"] = col_ok, col_err
        log(f"final_v2 function arm {arm}: OOF on {has.mean():.3f} of rows, "
            f"{np.isfinite(out[f'ok_E_{arm}']).sum():,} scored (E)")
    return out


def stage_v2ref(a):
    f = OUT / "function_rows.parquet"
    out = final_v2_function(pd.read_parquet(f))
    assert not out.DeviceId.isin(locked()).any()
    out.to_parquet(f, index=False)
    stage_score(a)


def attach_side(out: pd.DataFrame) -> pd.DataFrame:
    """lanes D, setback sb7, health v5, night speed sp1 - as their notes left them (per detector-window)."""
    L = pd.read_parquet(DC_WORK / "lanes" / "ln8" / "lanes_D.func.parquet",
                        columns=["DeviceId", "Detector", "period", "win", "phase", "lanes", "lane_conf"])
    L = L.rename(columns={"phase": "lane_phase", "lanes": "lanes_D"}).astype({"Detector": out.Detector.dtype})
    out = out.merge(L, on=["DeviceId", "Detector", "period", "win"], how="left")
    Lp = pd.read_parquet(DC_WORK / "lanes" / "ln8" / "lanes_D.func_ph.parquet",
                         columns=["DeviceId", "period", "win", "phase", "n_lanes", "n_lanes_conf"])
    Lp = Lp.rename(columns={"phase": "lane_phase", "n_lanes": "phase_n_lanes", "n_lanes_conf": "phase_n_lanes_conf"})
    out = out.merge(Lp, on=["DeviceId", "period", "win", "lane_phase"], how="left")
    sb = pd.read_parquet(DC_WORK / "trackA" / "setback" / "sb7" / "oof.parquet", columns=["dev", "det", "win", "all"])
    sb = sb.rename(columns={"dev": "DeviceId", "det": "Detector", "all": "setback_ft"}).assign(period="stg")
    sb = sb.astype({"Detector": out.Detector.dtype}).drop_duplicates(["DeviceId", "Detector", "win"])
    out = out.merge(sb, on=["DeviceId", "Detector", "period", "win"], how="left")
    h = pd.read_parquet(DC_WORK / "health" / "health_v5.parquet", columns=["period", "DeviceId", "detector", "status",
                                                                           "health_score"])
    h = h.rename(columns={"detector": "Detector", "status": "health_status"})
    h["DeviceId"] = h.DeviceId.str.lower()
    out = out.merge(h.astype({"Detector": out.Detector.dtype}).drop_duplicates(["period", "DeviceId", "Detector"]),
                    on=["period", "DeviceId", "Detector"], how="left")
    sp = pd.read_parquet(DC_WORK / "trackA" / "speed" / "sp1" / "det.parquet")
    sp = sp[sp.variant == "pred"].rename(columns={"dev": "DeviceId", "det": "Detector", "speed_mph": "night_speed_mph",
                                                  "reason": "night_speed_reason"})
    sp = sp[["DeviceId", "Detector", "win", "night_speed_mph", "night_speed_reason"]].assign(period="stg")
    out = out.merge(sp.astype({"Detector": out.Detector.dtype}).drop_duplicates(["DeviceId", "Detector", "win"]),
                    on=["DeviceId", "Detector", "period", "win"], how="left")
    return out


def score_function(res: dict):
    f = pd.read_parquet(OUT / "function_rows.parquet")
    meta = json.load(open(OUT / "function_meta.json"))
    sig = f.DeviceId.to_numpy()
    out = {"fj_folds": meta["fj_folds"], "n": {}, "acc_trees": {}, "fold_subset_with_fj": {}}
    fjf = sorted(h["fold"] for h in meta["fj_folds"])
    for s in ("E", "R"):
        sname = "everything" if s == "E" else "realistic"
        ok_t, ok_b = f[f"ok_{s}_trees"].to_numpy(), f[f"ok_{s}_fj"].to_numpy()
        sc = ~np.isnan(ok_t)
        allp = {**POOLS, "all": FAMS}
        for pool, fams in allp.items():
            m = sc & f.wgroup.isin(fams).to_numpy()
            out["n"].setdefault(sname, {})[pool] = [int(m.sum()), int(len(set(sig[m])))]
            out["acc_trees"].setdefault(sname, {})[pool] = acc_ci(ok_t[m], sig[m])
            if fjf:
                mm = m & f.fold.isin(fjf).to_numpy()
                out["fold_subset_with_fj"].setdefault(sname, {})[pool] = {
                    "n": int(mm.sum()), "signals": int(len(set(sig[mm]))),
                    "without_fj": acc_ci(ok_t[mm], sig[mm]), "with_fj": acc_ci(ok_b[mm], sig[mm]),
                    "delta_pt": delta_ci(ok_t[mm], ok_b[mm], sig[mm])}
        for arm in V2_ARMS:
            if f"ok_{s}_{arm}" not in f:
                continue
            ok_v = f[f"ok_{s}_{arm}"].to_numpy()
            for pool, fams in allp.items():
                m = sc & ~np.isnan(ok_v) & f.wgroup.isin(fams).to_numpy()
                n_all = int((sc & f.wgroup.isin(fams).to_numpy()).sum())
                ref = {"n": int(m.sum()), "signals": int(len(set(sig[m]))), "coverage": round(m.sum() / max(n_all, 1), 4),
                       "final_v2": acc_ci(ok_v[m], sig[m]), "cand_trees": acc_ci(ok_t[m], sig[m]),
                       "delta_pt_cand_trees_minus_v2": delta_ci(ok_v[m], ok_t[m], sig[m])}
                if fjf:
                    mm = m & f.fold.isin(fjf).to_numpy()
                    ref["fj_folds"] = {"folds": fjf, "n": int(mm.sum()), "signals": int(len(set(sig[mm]))),
                                       "final_v2": acc_ci(ok_v[mm], sig[mm]), "cand_fj": acc_ci(ok_b[mm], sig[mm]),
                                       "cand_trees": acc_ci(ok_t[mm], sig[mm]),
                                       "delta_pt_cand_fj_minus_v2": delta_ci(ok_v[mm], ok_b[mm], sig[mm]),
                                       "delta_pt_cand_trees_minus_v2": delta_ci(ok_v[mm], ok_t[mm], sig[mm])}
                out.setdefault(f"final_v2_ref_{arm}", {}).setdefault(sname, {})[pool] = ref
        out["acc_trees"][sname]["by_window"] = {g: round(float(np.nanmean(ok_t[sc & (f.wgroup == g).to_numpy()])), 4)
                                                for g in FAMS}
        # error mix on >= 30 min (pt of rows)
        e = f[f"err_{s}_trees"].to_numpy(object)
        m = sc & f.wgroup.isin(GE30).to_numpy()
        out.setdefault("err_mix_ge30_pt", {})[sname] = {k: round(100 * float(np.mean(e[m] == k)), 3)
                                                       for k in ("A->wrongA", "A->nonA", "nonA->A", "stack_extra")}
    res["function"] = out
    for sname in ("everything", "realistic"):
        log(f"function trees {sname[:4]} " + " ".join(f"{p} {v}" for p, v in out["acc_trees"][sname].items()
                                                       if p != "by_window"))
        if fjf:
            for p, v in out["fold_subset_with_fj"][sname].items():
                log(f"  folds {fjf} {sname[:4]} {p}: without {v['without_fj']} with {v['with_fj']} d {v['delta_pt']}")
        for arm in V2_ARMS:
            for p, v in out.get(f"final_v2_ref_{arm}", {}).get(sname, {}).items():
                log(f"  final_v2 ref {arm} {sname[:4]} {p}: v2 {v['final_v2']} cand {v['cand_trees']} "
                    f"d {v['delta_pt_cand_trees_minus_v2']} (n {v['n']}, cov {v['coverage']})"
                    + (f" | fj folds n {v['fj_folds']['n']}: v2 {v['fj_folds']['final_v2']} cand+fj "
                       f"{v['fj_folds']['cand_fj']} d {v['fj_folds']['delta_pt_cand_fj_minus_v2']}" if "fj_folds" in v else ""))


def phase_input_agreement(res: dict):
    """how often the function frame's phase input (v6e) equals the candidate phase (same detector-window)."""
    f = pd.read_parquet(OUT / "function_rows.parquet", columns=["DeviceId", "Detector", "period", "win", "pred_phase",
                                                                 "ok_E_trees", "wgroup"])
    p = pd.read_parquet(OUT / "phase_rows.parquet", columns=["DeviceId", "Detector", "win", "pred_cand"])
    p["period"] = np.where(p.DeviceId.str.endswith("@stg"), "stg", "dec")
    p["DeviceId"] = plain(p.DeviceId)
    m = f[f.ok_E_trees.notna()].merge(p.astype({"Detector": f.Detector.dtype}), on=["DeviceId", "Detector", "period", "win"],
                                      how="inner")
    res["function_phase_input_vs_candidate_phase"] = {
        "rows_matched": int(len(m)), "agree": round(float((m.pred_phase == m.pred_cand).mean()), 4),
        "agree_ge30": round(float((m.pred_phase == m.pred_cand)[m.wgroup.isin(GE30)].mean()), 4)}
    log(f"function phase input = candidate phase on {res['function_phase_input_vs_candidate_phase']}")


def side_summary(res: dict):
    """the side outputs' own OOF numbers (unchanged components; notes 58 / 60 / 47), copied for the one table."""
    s = {}
    try:
        ev = json.load(open(DC_WORK / "lanes" / "ln8" / "eval.json"))
        s["lanes_D_eval_json"] = {k: v for k, v in ev.items() if "D.func" in k} or "see note 58"
    except Exception as e:  # noqa: BLE001
        s["lanes_D_eval_json"] = str(e)
    f = pd.read_parquet(OUT / "function_rows.parquet", columns=["wgroup", "period", "lanes_D", "setback_ft",
                                                                 "health_status", "night_speed_mph", "pred_trees"])
    st = f[f.period == "stg"]
    adv = st.pred_trees == "Advance"
    s["coverage_stg"] = {
        "lanes_given_ge30": round(float((st.lanes_D.fillna("") != "")[st.wgroup.isin(GE30)].mean()), 4),
        "setback_on_pred_advance": round(float(st.setback_ft.notna()[adv].mean()), 4),
        "health_status": st.health_status.value_counts(normalize=True).round(4).to_dict(),
        "night_speed_on_pred_advance_full": round(float(st.night_speed_mph.notna()[adv & (st.wgroup == "full")].mean()), 4)}
    s["from_notes"] = {"lanes_D_note58": "n_lanes exact .854 (m30 .811 / full .886), lane set .922, pair acc .923",
                       "setback_sb7_note58": "medAE 25.7 / 22.4 / 22.6 / 20.4 ft at m30 / h6 / h24 / full; ~70 % within 50 ft",
                       "health_v5_note47": "presumed-healthy flagged 1.0-4.8 %, known problems caught 27-99 % (2 h-66 h)",
                       "night_speed_note60": "answers 19-23 % of phases with an Advance at >= 24 h; median 30 mph"}
    res["side_outputs"] = s


def stage_score(a):
    res = {"built": time.strftime("%Y-%m-%d %H:%M")}
    score_phase(res)
    score_function(res)
    phase_input_agreement(res)
    side_summary(res)
    json.dump(res, open(OUT / "headline.json", "w"), indent=1, default=str)
    log(f"-> {OUT / 'headline.json'}")


def stage_all(a):
    stage_phase(a)
    stage_function(a)
    stage_score(a)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["all", "phase", "function", "score", "v2ref"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    globals()[f"stage_{a.stage}"](a)
