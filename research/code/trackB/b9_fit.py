"""Track B step B9: refit the pair ranker with / without the neighbour-trace features.

Same pool, labels, folds, parameters and feature set as the shipped ranker
(`research/code/lightgbm/fit_final_v1.py`: DEV Dec-2024 all active channels + NEWTRAIN
Sept-2026 labelled channels, official labels, 22 windows, `train_official.RANK_PARAMS`),
single seed per call, threads capped at 6.  For fold k the ranker is trained on the other
folds minus the inner fold (k+1)%6, early-stopped on the inner fold, and scores every
fold-k row; the output is the per-detector softmax `p0`, the input the decoder expects.

    --variant base      the shipped feature set (261 columns)
    --variant nb        + the `nb_*` columns of b9_features.py
    --variant nbshuf    + the `nb_*` columns permuted jointly across rows within each window
                          (the shuffled-feature control)

    python research/code/trackB/b9_fit.py --variant nb --folds 0 --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402
from common import DC_WORK, N_FOLDS  # noqa: E402
import train_official as T  # noqa: E402
from features_partner import PDIFF_FEATS, add_partner_diffs  # noqa: E402
from neural.data2 import training_signals  # noqa: E402

B9 = DC_WORK / "trackB" / "b9"
KEY = ["DeviceId", "Detector", "win", "cand_phase"]
PDIFF = PDIFF_FEATS + ["on_lift_green", "occ_lift_green", "f_on_green", "excl_diff_min",
                       "release_frac_long", "call43_fwd_lift"]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _read(files, ids) -> pd.DataFrame:
    parts = []
    for f in files:
        if not Path(f).exists():
            continue
        d = pd.read_parquet(f, filters=[("DeviceId", "in", list(ids))])
        for c in d.columns:
            if d[c].dtype == np.float64:
                d[c] = d[c].astype(np.float32)
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def _side(base, v2, ids, tag) -> pd.DataFrame:
    df = _read(base, ids)
    ex = _read(v2, ids)
    df = df.merge(ex, on=T.PAIR_KEY, how="left")
    del ex
    df = add_partner_diffs(df, PDIFF)
    df["src"] = tag
    return df


def build_pool() -> tuple[pd.DataFrame, list[str]]:
    """fit_final_v1.build_pool, lean: filtered reads, float32."""
    s = training_signals()
    fmap = {("stg" if p == "stg" else "dec", d): f
            for d, p, f in zip(s.DeviceId, s.period, s.fold)}
    dev = sorted(s[s.period == "dec"].DeviceId)
    nt = sorted(s[s.period == "stg"].DeviceId)
    ev = _side([T.FEAT / "pair_features_windows.parquet",
                T.FEAT / "pair_features_windows_B.parquet"],
               [T.FEAT / "pair_features_v2_extra.parquet",
                T.FEAT / "pair_features_v2_extra_B.parquet"], dev, "DEC")
    ev = T.attach_labels(ev, T._official_phase())
    ev["fold"] = [fmap[("dec", d)] for d in ev.DeviceId]
    ev = T.add_scorable(ev)
    ex = _side([T.SFEAT / "pair_features_stg.parquet"],
               [T.SFEAT / "pair_features_v2_stg.parquet"], nt, "STG")
    ex["fold"] = [fmap[("stg", d)] for d in ex.DeviceId]
    ex["DeviceId"] = ex.DeviceId + "@stg"
    off = T._official_phase().copy()
    off["DeviceId"] = off.DeviceId + "@stg"
    ex = T.attach_labels(ex, off)
    ex = ex[ex.Phase.notna()].reset_index(drop=True)
    ex = T.add_scorable(ex)
    fc = [c for c in T.feature_cols(ev) if c in ex.columns]
    comb = pd.concat([ev, ex], ignore_index=True)
    del ev, ex
    comb["DeviceId"] = comb.DeviceId.str.lower()
    comb["Detector"] = comb.Detector.astype(int)
    comb["cand_phase"] = comb.cand_phase.astype(int)
    comb = comb.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
    log(f"pool {comb.shape}; {comb.DeviceId.nunique()} signals; {len(fc)} base features; "
        f"{int(comb.Phase.notna().sum()):,} labelled rows")
    return comb, fc


def add_nb(comb: pd.DataFrame, shuffle: bool, seed: int) -> tuple[pd.DataFrame, list[str]]:
    nb = pd.read_parquet(B9 / "nb_features.parquet")
    nb["DeviceId"] = nb.DeviceId.str.lower()
    cols = [c for c in nb.columns if c.startswith("nb_")]
    comb = comb.merge(nb, on=KEY, how="left")
    log(f"nb features: {len(cols)}; rows with any neighbour "
        f"{comb[cols].notna().any(axis=1).mean():.3f}")
    if shuffle:
        rng = np.random.default_rng(1000 + seed)
        block = comb[cols].to_numpy()
        for _, idx in comb.groupby("win").indices.items():
            block[idx] = block[rng.permutation(idx)]
        comb[cols] = block
        log("nb features SHUFFLED within windows")
    return comb, cols


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["base", "nb", "nbshuf"])
    ap.add_argument("--folds", default="0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    B9.mkdir(parents=True, exist_ok=True)
    comb, fc = build_pool()
    new = []
    if a.variant != "base":
        comb, new = add_nb(comb, a.variant == "nbshuf", a.seed)
    feats = fc + new
    sd = a.seed
    rp = dict(T.RANK_PARAMS, bagging_seed=sd, feature_fraction_seed=sd + 100,
              data_random_seed=sd + 200, seed=sd, n_jobs=a.threads)
    lab = comb.Phase.notna().to_numpy()
    for k in [int(x) for x in a.folds.split(",")]:
        tag = f"{a.variant}_s{sd}_f{k}"
        out = B9 / f"p0_{tag}.parquet"
        if out.exists():
            log(f"{tag} exists, skipped")
            continue
        t0 = time.time()
        te = (comb.fold == k).to_numpy()
        inner = (k + 1) % N_FOLDS
        base = (~te) & lab
        tr = comb[base & (comb.fold != inner).to_numpy()]
        va = comb[base & (comb.fold == inner).to_numpy()]
        m = T._fit_rank(tr, va, feats, rp)
        del tr, va
        sub = comb.loc[te, KEY].copy()
        sub["p0"] = T.to_prob(sub, m.predict(comb.loc[te, feats]))
        sub.to_parquet(out, index=False)
        imp = pd.Series(m.booster_.feature_importance("gain"), index=feats)
        info = {"tag": tag, "n_features": len(feats), "n_new": len(new),
                "best_iter": int(m.best_iteration_), "secs": round(time.time() - t0),
                "gain_share_new": float(imp[new].sum() / imp.sum()) if new else 0.0,
                "top_new": {c: round(float(v / imp.sum()), 5)
                            for c, v in imp[new].sort_values(ascending=False)[:12].items()}
                if new else {}}
        json.dump(info, open(B9 / f"fit_{tag}.json", "w"), indent=1)
        log(json.dumps(info))


if __name__ == "__main__":
    main()
