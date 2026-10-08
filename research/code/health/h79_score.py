"""Note 79: score the current health output (h79_run.py) on the evaluation set (h79_evalset.py), then judge the
candidate relative / self-consistency checks on a held-out split by signal.

    python h79_score.py -> prints; %DC_WORK%/health79/tables.csv, cand.csv
"""
from __future__ import annotations

import hashlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hb_data as H  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
OUT = H.DCW / "health79"
CLS = ["dead", "stuck", "chatter", "intermittent", "degraded"]
SGRP = {"dead": ["s_dead"], "stuck": ["s_stuck"], "chatter": ["s_chatter", "s_rapid", "s_volume", "s_short_on"],
        "intermittent": ["s_dropout", "s_choppy", "s_corr", "s_night_day"], "degraded": ["s_level", "s_night_drop"]}
LEN = ["m30", "h3", "h24"]


def half(dev, seed=0):
    return int(hashlib.md5(f"{seed}:{dev}".encode()).hexdigest(), 16) % 2


def load():
    R = pd.read_parquet(OUT / (sys.argv[1] if len(sys.argv) > 1 else "run.parquet"))
    E = pd.read_parquet(OUT / "evalset.parquet")
    R["wlen"] = R.window.str.split("_").str[0]
    R["flag"] = R.status.isin(["bad", "suspect"])
    s = R[[c for c in R if c.startswith("s_")]].fillna(0)
    best = pd.DataFrame({k: s[[c for c in v if c in s]].max(1) for k, v in SGRP.items()})
    R["pred_cls"] = np.where(R.flag, best.idxmax(1), "")
    # every (eval row x window) in BOTH modes; a missing row = the channel did not appear in the window's log
    W = R[["DeviceId", "period", "window", "wlen"]].drop_duplicates()
    rows = []
    for mode in ("prod", "listed"):
        x = W.merge(E, on="DeviceId").merge(R[R["mode"] == mode].drop(columns=["wlen", "mode"]),
                                            on=["DeviceId", "period", "window", "detector"], how="left")
        x["mode"] = mode
        x["seen"] = x.status.notna()
        x["flag"] = x.flag.fillna(False).astype(bool)
        x["pred_cls"] = x.pred_cls.fillna("")
        rows.append(x)
    return R, E, pd.concat(rows, ignore_index=True)


def table(M, by):
    pos = M[M.cls != "healthy"]
    g = pos.groupby(by + ["cls"])
    t = pd.DataFrame({"n": g.size(), "seen%": 100 * g.seen.mean(), "recall%": 100 * g.flag.mean(),
                      "bad%": 100 * g.status.apply(lambda s: s.eq("bad").mean()),
                      "ned%": 100 * g.status.apply(lambda s: s.eq("not_enough_data").mean())})
    return t.round(1)


def precision(M, by):
    out = []
    for k, x in M[M.flag].groupby(by):
        k = k if isinstance(k, tuple) else (k,)
        r = dict(zip(by, k))
        r["flagged"] = len(x)
        r["prec_any%"] = round(100 * (x.cls != "healthy").mean(), 1)
        for c in CLS:
            y = x[x.pred_cls == c]
            r[f"{c}: n / prec%"] = f"{len(y)} / {100 * (y.cls == c).mean():.0f}" if len(y) else "-"
        out.append(r)
    return pd.DataFrame(out)


def main():
    R, E, M = load()
    print("eval rows:", E.cls.value_counts().to_dict())
    H_ = M[M.cls == "healthy"]
    for per in ("w40", "stg"):
        P = M[M.period == per]
        print(f"\n===== period {per} ({'independent re-observation' if per == 'w40' else 'SAME period the C labels came from'})")
        for mode in ("prod", "listed"):
            X = P[P["mode"] == mode]
            t = table(X, ["wlen"]).reset_index()
            fa = X[X.cls == "healthy"].groupby("wlen").agg(n=("flag", "size"), fa=("flag", "mean"), seen=("seen", "mean"))
            print(f"\n-- mode {mode}: false alarm on presumed healthy %:",
                  {w: round(100 * fa.fa[w], 2) for w in LEN if w in fa.index})
            print(t.pivot_table(index="cls", columns="wlen", values=["recall%", "seen%"])[[(a, w) for a in ("recall%", "seen%") for w in LEN]].to_string())
        # by tier (prod)
        X = P[P["mode"] == "prod"]
        t = X[X.cls != "healthy"].groupby(["cls", "tier", "wlen"]).flag.mean().unstack()[LEN].mul(100).round(1)
        n = X[(X.cls != "healthy") & (X.wlen == "h24")].drop_duplicates(["DeviceId", "detector"]).groupby(["cls", "tier"]).size()
        print("\nrecall % by tier (prod):\n", t.assign(n=n).to_string())
        print("\nprecision (prod, labelled universe = positives + presumed healthy):")
        print(precision(X, ["wlen"]).to_string(index=False))
        # technology (prod)
        tt = X.groupby(["technology", "wlen"]).apply(lambda x: pd.Series({
            "FA%": 100 * x[x.cls == "healthy"].flag.mean(), "recall_all%": 100 * x[x.cls != "healthy"].flag.mean(),
            "recall_nondead%": 100 * x[~x.cls.isin(["healthy", "dead"])].flag.mean(),
            "n_pos": x[x.cls != "healthy"].drop_duplicates(["DeviceId", "detector"]).shape[0]})).round(1)
        print("\nby technology (prod):\n", tt.unstack("wlen").to_string())
        # dead detail: are the dead still dead in this period (listed mode n_on)?
        D = P[(P["mode"] == "listed") & (P.cls == "dead") & (P.wlen == "h24")]
        print("\ndead labels: share with 0 actuations in this period's 24 h windows:",
              round(100 * D.n_on.fillna(0).eq(0).mean(), 1), "% ; n", len(D))
    # the user's answered rows on the independent period, per window
    U = M[(M.tier == "U") & (M["mode"] == "prod")]
    print("\nuser answers (prod): flagged share by period x wlen (positives) / ok rows")
    print(U.assign(pos=U.cls != "healthy").groupby(["period", "pos", "wlen"]).flag.mean().unstack().reindex(columns=LEN).round(2).to_string())
    M.to_parquet(OUT / "scored.parquet", index=False)


if __name__ == "__main__":
    main()
