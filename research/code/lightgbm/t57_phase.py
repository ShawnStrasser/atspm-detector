"""Note 57 (validation plan step 2): the PHASE trees on their own -- LightGBM pair ranker (+ joint decoder), no network.

Pool = note 37's 780-signal phase pool (`phase_v3_trees.build_pool_v3`: DEV Dec-2024 + NEWTRAIN Sept-2026 + the 71
released NEWTEST, official timing labels, 22-window mix), but with the SIX FOLDS OF `folds_v4.csv` for every signal
(note 37 kept the stage-13 phase map, which differs on 22 signals).  Recipe = note 37 / fit_final_v1 (same
LambdaRank params, inner fold (k+1)%6 for early stopping, decoder = binary LightGBM on the OOF first-stage probs).

    python t57_phase.py pool                               # cache the pool (features float32) -> trees57/phase/pool.*
    python t57_phase.py run --cfg full --seeds 0,1,2       # ranker OOF per seed (+ checkpoints), then decode
    python t57_phase.py run --cfg drop:call --seeds 0      # leave one feature group out
    python t57_phase.py run --cfg noise --seeds 0          # + as many shuffled copies of real columns as the
                                                             biggest group has (noise-column control)
    python t57_phase.py score --cfgs full,drop:call ...    # two-number scoring, paired signal bootstrap vs --base

Outputs per config `%DC_WORK%/trees57/phase/<cfg>/`: S_f{k}_s{sd}.npy (ranker scores), oof.parquet (KEY4 + p0 per seed,
p0 bag, p2 per seed decoded, p2 bag decoded).  Locked_v2 asserted absent.  threads <= 6 (shared machine).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: F401,E402
from common import DC_WORK, N_FOLDS  # noqa: E402

OUT = DC_WORK / "trees57" / "phase"
KEY4 = ["DeviceId", "Detector", "win", "cand_phase"]
DET = ["DeviceId", "Detector", "win"]
NJ = 6
FAMS = ["m5", "m10", "m30", "h1", "h3", "h6", "h24", "full"]
REPO = Path(__file__).resolve().parents[3]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def plain(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("@stg", "", regex=False).str.lower()


def fam_of(w: str) -> str:
    return "full" if w.startswith("full") else w.split("_")[0]


# ------------------------------------------------------------------ feature groups (leave-one-group-out)
COND = ("on_lift_green_coord", "on_lift_green_free")          # condition-dependent (coord / free periods)


def group_of(c: str) -> set:
    """A column can sit in several groups (content x transform); dropping a group drops every column in it."""
    g = set()
    base = c.split("__")[0]
    tr = c.split("__")[1] if "__" in c else ""
    if tr in ("rank", "z", "mgap", "argmax"):
        g.add("xcand")                       # cross-candidate normalisations of a per-pair feature
    if tr == "pdiff":
        g.add("pdiff")                       # difference to the concurrent partner phase's value
    if any(s in base for s in ("call43", "call44", "n43_per", "looks_recall")):
        g.add("call")                        # phase-call events 43 / 44
    if base.startswith(("excl_", "pex_", "solo_", "cogreen", "partner_lead", "n_excl", "n_on_pair", "anygreen")) \
            or base in ("excl_secs_p", "excl_secs_q", "excl_secs_min"):
        g.add("excl")                        # exclusivity vs other phases / partner phase
    if base in COND:
        g.add("cond")
    if base.startswith(("dtg_", "dtr_", "tog_")):
        g.add("timing_hist")                 # actuation timing histograms relative to green / red
    if base.startswith(("dur_", "det_dur", "det_frac", "burst", "long_on", "n_long", "late_green", "late2")) \
            or base in ("dur_ratio_red_green",):
        g.add("duration")
    if base.startswith(("first_on", "release", "queue", "cyc_hit", "straddle", "ext_")):
        g.add("queue_release")
    if base in ("n_cycles", "green_mean", "green_sd", "green_med", "cycle_mean", "cand_green_share",
                "cand_cycle_ratio", "win_secs", "log_win_hours", "n_cycles_per_hour", "n_cand", "cyc_active_frac",
                "green_share"):
        g.add("signal_ctx")                  # candidate / window context, not the detector itself
    if not g:
        g.add("core")
    return g


# ------------------------------------------------------------------ pool
def stage_pool(a):
    import phase_v3_trees as P3
    P3.NJ = NJ
    comb, simc, fc = P3.build_pool_v3()
    f4 = pd.read_csv(DC_WORK / "folds_v4.csv")
    fm = dict(zip(f4.DeviceId.str.lower(), f4.fold))
    newf = plain(comb.DeviceId).map(fm)
    assert newf.notna().all(), f"{plain(comb.DeviceId)[newf.isna()].unique()[:5]} not in folds_v4"
    moved = (newf != comb.fold)
    log(f"folds_v4: {plain(comb.DeviceId)[moved].nunique()} signals change fold vs the note-37 phase map")
    comb["fold_p37"] = comb.fold
    comb["fold"] = newf.astype(int)
    lk = set(pd.read_csv(DC_WORK / "official" / "locked_v2.csv").DeviceId.str.lower())
    assert not plain(comb.DeviceId).isin(lk).any() and not plain(simc.DeviceId).isin(lk).any()
    keep = KEY4 + ["src", "fold", "fold_p37", "Phase", "y", "scorable", "det_n_on"] + \
        [c for c in fc if c not in ("det_n_on",)]
    OUT.mkdir(parents=True, exist_ok=True)
    comb[keep].to_parquet(OUT / "pool.parquet", index=False)
    simc.to_parquet(OUT / "pool_sim.parquet", index=False)
    json.dump({"features": fc, "rows": len(comb), "signals": int(comb.DeviceId.nunique()),
               "labelled_rows": int(comb.Phase.notna().sum()),
               "signals_moved_vs_p37": int(plain(comb.DeviceId)[moved].nunique()),
               "groups": {c: sorted(group_of(c)) for c in fc}}, open(OUT / "pool_meta.json", "w"), indent=1)
    log(f"pool {comb.shape} cached")


def load_pool():
    meta = json.load(open(OUT / "pool_meta.json"))
    comb = pd.read_parquet(OUT / "pool.parquet")
    sim = pd.read_parquet(OUT / "pool_sim.parquet")
    return comb, sim, meta["features"]


def cfg_cols(cfg: str, fc: list[str]) -> list[str]:
    if cfg in ("full", "noise", "nodec"):
        return list(fc)
    if cfg.startswith("drop:"):
        gs = set(cfg[5:].split("+"))
        return [c for c in fc if not (group_of(c) & gs)]
    if cfg.startswith("only:"):
        gs = set(cfg[5:].split("+"))
        return [c for c in fc if group_of(c) <= gs]
    raise ValueError(cfg)


def rp(sd: int) -> dict:
    import train_official as T
    return dict(T.RANK_PARAMS, n_jobs=NJ, bagging_seed=sd, feature_fraction_seed=sd + 100,
                data_random_seed=sd + 200, seed=sd)


def add_noise(comb, fc, n, seed=57):
    """n shuffled copies of randomly chosen real feature columns (marginals kept, link to the label destroyed;
    permuted across whole signals' rows so the column stays non-informative at every level)."""
    rng = np.random.default_rng(seed)
    src = rng.choice(fc, n, replace=False)
    new = []
    for i, c in enumerate(src):
        nm = f"noise_{i}_{c}"
        comb[nm] = comb[c].to_numpy()[rng.permutation(len(comb))]
        new.append(nm)
    return new


def stage_run(a):
    import lightgbm as lgb
    import train_official as T
    import decode_train as dec
    comb, simc, fc = load_pool()
    cols = cfg_cols(a.cfg, fc)
    if a.cfg == "noise":
        cols = cols + add_noise(comb, fc, a.n_noise)
    d = OUT / a.cfg.replace(":", "_").replace("+", "-")
    d.mkdir(parents=True, exist_ok=True)
    json.dump({"cfg": a.cfg, "n_cols": len(cols), "cols": cols}, open(d / "cols.json", "w"), indent=1)
    log(f"cfg {a.cfg}: {len(cols)} ranker features -> {d}")
    seeds = [int(x) for x in a.seeds.split(",")]
    lab = comb.Phase.notna().to_numpy()
    fold = comb.fold.to_numpy()
    t0 = time.time()
    for k in range(N_FOLDS):
        te = fold == k
        inner = (k + 1) % N_FOLDS
        base = (~te) & lab
        for sd in seeds:
            f = d / f"S_f{k}_s{sd}.npy"
            if f.exists():
                continue
            src = d if a.cfg != "nodec" else OUT / "full"
            tr = comb[base & (fold != inner)]
            va = comb[base & (fold == inner)]
            m = T._fit_rank(tr, va, cols, rp(sd))
            del tr, va
            np.save(f, m.predict(comb.loc[te, cols], num_threads=NJ))
            with open(d / "timing.jsonl", "a") as fh:
                fh.write(json.dumps({"fold": k, "seed": sd, "trees": int(m.best_iteration_ or 0),
                                     "secs": round(time.time() - t0, 1)}) + "\n")
            log(f"  fold {k} seed {sd}: {m.best_iteration_} trees ({time.time()-t0:.0f}s)")
    # first-stage probabilities per seed and bagged
    have = sorted({int(p.stem.split("_s")[1]) for p in d.glob("S_f0_s*.npy")
                   if all((d / f"S_f{k}_s{p.stem.split('_s')[1]}.npy").exists() for k in range(N_FOLDS))})
    ob = comb[KEY4].copy()
    P0 = {}
    for sd in have:
        S = np.zeros(len(comb))
        for k in range(N_FOLDS):
            S[fold == k] = np.load(d / f"S_f{k}_s{sd}.npy")
        P0[sd] = T.to_prob(comb, S)
        ob[f"p0_s{sd}"] = P0[sd]
    ob["p0_bag"] = np.mean([P0[s] for s in have], axis=0)
    # decoder (OOF) on the bag and, if asked, on each seed alone
    targets = ["bag"] + ([f"s{s}" for s in have] if a.decode_seeds and len(have) > 1 else [])
    if len(have) == 1:
        targets = ["bag"]
    cols2 = dec.BASE_COLS + dec.SIM_COLS + dec.ADJ_COLS
    for tname in targets:
        pcol = "p0_bag" if tname == "bag" else f"p0_{tname}"
        pr = comb[KEY4].copy()
        pr["p0"] = ob[pcol].to_numpy()
        X = dec.assemble(pr, pairs=comb, sim=simc)
        X = X.merge(comb[KEY4 + ["Phase", "fold", "y"]], on=KEY4, how="left")
        X = X.sort_values(KEY4[2:3] + ["DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
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
        X["p2"] = T.norm_prob(X, s2)
        back = comb[KEY4].merge(X[KEY4 + ["p2"]], on=KEY4, how="left")
        ob[f"p2_{tname}"] = back.p2.to_numpy()
        log(f"  decoded {tname}")
    ob.to_parquet(d / "oof.parquet", index=False)
    log(f"wrote {d / 'oof.parquet'} seeds {have} ({time.time()-t0:.0f}s)")


# ------------------------------------------------------------------ scoring (two numbers)
_ROWS = {}


def score_rows() -> pd.DataFrame:
    """Scorable detector-windows of the pool with the label-quality facts (note 48's sets, labels v3s)."""
    if "r" in _ROWS:
        return _ROWS["r"]
    f = OUT / "score_rows.parquet"
    if f.exists():
        r = pd.read_parquet(f)
        r["alt"] = r.alt_s.map(lambda s: {int(x) for x in s.split(",") if x})
        _ROWS["r"] = r
        return r
    comb = pd.read_parquet(OUT / "pool.parquet", columns=KEY4 + ["src", "fold", "Phase", "scorable", "det_n_on"])
    r = comb[comb.scorable].groupby(DET, sort=False).agg(Phase=("Phase", "first"), det_n_on=("det_n_on", "max"),
                                                          fold=("fold", "first"), src=("src", "first")).reset_index()
    r["dev_plain"] = plain(r.DeviceId)
    v = pd.read_parquet(rpath.LABELS_CURRENT)  # note 81: v4l
    v["DeviceId"] = v.DeviceId.str.lower()
    tp = v.phase_target.str.extract(r"P(\d+)")[0].astype(float)
    hi = v.print_source.eq("print") & v.print_confidence.eq("high") & v.phase_diagram.notna() & tp.notna()
    v["print_phase_disagrees"] = hi & (v.phase_diagram.astype(float) != tp)
    v["lab_bad"] = v.validated.isin(["fail", "misconfigured"]).fillna(False)
    v["alt_s"] = [",".join(sorted(({str(int(s))} if pd.notna(s) and s > 0 else set()) |
                                  {str(int(x)) for x in str(q).split(",") if str(x).strip().isdigit()}))
                  for s, q in zip(v.switch_phase, v.additional_call_phases.fillna(""))]
    L = v[["DeviceId", "detector", "print_phase_disagrees", "lab_bad", "alt_s"]].rename(
        columns={"DeviceId": "dev_plain", "detector": "Detector"}).drop_duplicates(["dev_plain", "Detector"])
    L["Detector"] = L.Detector.astype(r.Detector.dtype)
    r = r.merge(L, on=["dev_plain", "Detector"], how="left")
    r["print_phase_disagrees"] = r.print_phase_disagrees.fillna(False).astype(bool)
    r["lab_bad"] = r.lab_bad.fillna(False).astype(bool)
    r["alt_s"] = r.alt_s.fillna("")
    r["fam"] = r.win.map(fam_of)
    r["everything"] = r.det_n_on >= 5
    r["realistic"] = r.everything & ~r.lab_bad & ~r.print_phase_disagrees
    r.to_parquet(f, index=False)
    r["alt"] = r.alt_s.map(lambda s: {int(x) for x in s.split(",") if x})
    _ROWS["r"] = r
    return r


def top1(oof: pd.DataFrame, col: str, rows: pd.DataFrame) -> np.ndarray:
    d = oof[KEY4 + [col]]
    d = d.merge(rows[DET], on=DET, how="inner")
    d = d.sort_values(DET + [col, "cand_phase"], ascending=[True, True, True, False, True])
    t = d.groupby(DET, sort=False).first().reset_index()
    return rows[DET].merge(t[DET + ["cand_phase"]], on=DET, how="left").cand_phase.to_numpy()


def boot(a, b, sig, n=2000, seed=0):
    u, inv = np.unique(sig, return_inverse=True)
    k = len(u)
    sa, sb, c = np.bincount(inv, a, k), np.bincount(inv, b, k), np.bincount(inv, minlength=k)
    idx = np.random.default_rng(seed).integers(0, k, (n, k))
    dd = (sb[idx].sum(1) - sa[idx].sum(1)) / c[idx].sum(1)
    return [round(100 * (b.mean() - a.mean()), 3), round(100 * float(np.quantile(dd, .025)), 3),
            round(100 * float(np.quantile(dd, .975)), 3)]


def ok_vec(rows, pred, lenient):
    o = pred == rows.Phase.to_numpy()
    if lenient:
        o = o | np.array([p in s for p, s in zip(pred, rows.alt)])
    return o.astype(float)


def summarise(rows, ok, mset):
    out = {}
    for fam in FAMS:
        m = (mset & (rows.fam == fam)).to_numpy()
        if m.any():
            out[fam] = round(float(pd.Series(ok[m]).groupby(rows.win.to_numpy()[m]).mean().mean()), 5)
    return out


def stage_score(a):
    rows = score_rows()
    preds = {}
    for spec in a.cfgs.split(","):
        cfg, col = spec.split("@") if "@" in spec else (spec, "p2_bag")
        d = OUT / cfg.replace(":", "_").replace("+", "-")
        oof = pd.read_parquet(d / "oof.parquet", columns=KEY4 + [col])
        oof = oof.astype({"Detector": rows.Detector.dtype})
        preds[f"{cfg}@{col}"] = top1(oof, col, rows)
    base = a.base if "@" in a.base else a.base + "@p2_bag"
    if base not in preds:
        cfg, col = base.split("@")
        oof = pd.read_parquet(OUT / cfg.replace(":", "_").replace("+", "-") / "oof.parquet", columns=KEY4 + [col])
        preds[base] = top1(oof.astype({"Detector": rows.Detector.dtype}), col, rows)
    sig = plain(rows.DeviceId).to_numpy()
    res = {"base": base, "n": {}, "acc": {}, "delta_vs_base": {}}
    for sname, len_ in (("everything", False), ("realistic", True)):
        mset = rows[sname]
        res["n"][sname] = {f: int((mset & (rows.fam == f)).sum()) for f in FAMS}
        ob = ok_vec(rows, preds[base], len_)
        for k, p in preds.items():
            o = ok_vec(rows, p, len_)
            res["acc"].setdefault(k, {})[sname] = summarise(rows, o, mset)
            if k == base:
                continue
            dl = {}
            for fam in FAMS:
                m = (mset & (rows.fam == fam)).to_numpy()
                dl[fam] = boot(ob[m], o[m], sig[m])
            res["delta_vs_base"].setdefault(k, {})[sname] = dl
    out = OUT / f"score_{a.tag}.json"
    json.dump(res, open(out, "w"), indent=1)
    for k in preds:
        e, r = res["acc"][k]["everything"], res["acc"][k]["realistic"]
        dE = res["delta_vs_base"].get(k, {}).get("everything", {})
        print(f"{k:40s} E " + " ".join(f"{f} {e.get(f, float('nan')):.4f}" for f in FAMS))
        print(f"{'':40s} R " + " ".join(f"{f} {r.get(f, float('nan')):.4f}" for f in FAMS))
        if dE:
            print(f"{'':40s} dE m5 {dE['m5']} m30 {dE['m30']} h6 {dE['h6']} full {dE['full']}")
            dR = res["delta_vs_base"][k]["realistic"]
            print(f"{'':40s} dR m5 {dR['m5']} m30 {dR['m30']} h6 {dR['h6']} full {dR['full']}")
    log(f"-> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pool", "run", "score"])
    ap.add_argument("--cfg", default="full")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--n-noise", type=int, default=40, dest="n_noise")
    ap.add_argument("--decode-seeds", action="store_true", dest="decode_seeds")
    ap.add_argument("--cfgs", default="full")
    ap.add_argument("--base", default="full")
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    globals()[f"stage_{a.stage}"](a)
