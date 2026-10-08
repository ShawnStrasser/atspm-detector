"""Note 121: health v4d = v4c (score_v4c.py) + a general severity rule (no per-row tuning; research only).

Severity rule (applied to the final v4c status, so it can sit at the end of any resolver):
  A  two independent findings -> bad.  v4c already made 'suspect' + findings from 2 FAMILIES bad; A drops the family
     condition (chatter + too-fast, erratic counts + too many are two different checks).  Not counted (as before):
     a stuck / silent finding exactly at its limit (.35 = capped).  One quiet stretch seen by several checks
     (goes silent, count drops, misses at night, a QUIET unusual day) counts once.
  C  a Count zone held ON >= 30 min in light traffic is a counting fault even when ONs logged again without an OFF
     (extension) explain it: watch / ok -> suspect (v4c: watch + config note).
  P  (proposed, OFF by default) a finding that persists >= 4 h of the day -> bad: busy at night (01-05 h by
     construction), unusual daily pattern run >= 4 h, silent >= 4 h, unexplained ON in >= 16 15-min periods.

    python score_v4d.py [--persist]   -> %DC_WORK%/s121/{resolved_v4d.parquet, rates121.csv, rows121.csv,
                                                         cases121.csv}
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
IN = DCW / "s118c"
OUT = DCW / "s121"
KEY = ["DeviceId", "window", "detector"]
QUIET = {"dropout", "level", "night_drop"}
PERSIST_H = 4.0
FL = ("suspect", "bad")
RANK = {"ok": 0, "not_enough_data": 0, "watch": 1, "suspect": 2, "bad": 3}


def _findings(r):
    out = []
    for k in str(r.left8).split(","):
        c = f"s8_{k}"
        if k and c in r.index and np.isfinite(r[c]) and r[c] >= .35:
            out.append((k, float(r[c])))
    return out


def n_evidence(R):
    """number of independent findings per detector-window (rule A)."""
    n = []
    for _, r in R.iterrows():
        ev = set()
        for k, v in _findings(r):
            if k in ("stuck", "dropout") and v == .35:
                continue
            quiet = k in QUIET or (k == "shape24" and r.bd_ev_dir < 0)
            ev.add("quiet" if quiet else k)
        n.append(len(ev))
    return np.array(n)


def persists(R):
    has = lambda k: R.left8.fillna("").str.split(",").apply(lambda l: k in l)  # noqa: E731
    silent_h = (R.drop_b1 - R.drop_b0) * 300 / 3600
    return (has("prof") | (has("shape24") & (R.bd_ev_h >= PERSIST_H)) | (has("dropout") & (silent_h >= PERSIST_H)) |
            (has("occspk") & (R.n3_n >= 4 * PERSIST_H))).to_numpy()


def count_held_by_extension(R):
    return (R.fn.eq("Count") & R.c1_pre.astype(str).eq("True") & (R.hi_min_c >= 30)).to_numpy()


def severity_v4d(R, persist=False):
    st = R.st8.copy()
    sus = st.eq("suspect").to_numpy()
    R["n_ev"] = n_evidence(R)
    rule = np.full(len(R), "", object)
    a = sus & (R.n_ev.to_numpy() >= 2)
    st[a], rule[a] = "bad", "A"
    if persist:
        p = sus & ~a & persists(R)
        st[p], rule[p] = "bad", "P"
    c = count_held_by_extension(R) & st.isin(["ok", "watch"]).to_numpy()
    st[c], rule[c] = "suspect", "C"
    R["st_v4d"], R["sev_rule"] = st.to_numpy(), rule
    return R


def rates(R, cols):
    rows = []
    for c in cols:
        for wg in ("m30", "h3", "h24"):
            x = R[R.wg == wg][c]
            rows.append(dict(status=c, window=wg, n=len(x), bad=100 * x.eq("bad").mean(),
                             suspect=100 * x.eq("suspect").mean(), flagged=100 * x.isin(FL).mean()))
    T = pd.DataFrame(rows)
    b = T[T.status == "st8"].set_index("window")
    T["bad_change_pct"] = 100 * (T.bad / T.window.map(b.bad) - 1)
    T["flagged_change_pct"] = 100 * (T.flagged / T.window.map(b.flagged) - 1)
    return T.round(3)


def user_cases(R, cols):
    """the user's earlier verdicts (v3 / v4 reviews: 20 bad, 12 fine detectors; s118c/cases_v4c.csv), worst status
    over their 3 h / 24 h windows."""
    C = pd.read_csv(IN / "cases_v4c.csv", dtype={"sig": str})
    x = R.merge(C[["grp", "sig", "det", "win"]].rename(columns={"sig": "signal", "det": "detector", "win": "window"}),
                on=["signal", "detector", "window"])
    out = []
    for c in cols:
        w = x.groupby(["grp", "signal", "detector"])[c].agg(lambda s: max(s, key=lambda v: RANK[v])).reset_index()
        for g, t in w.groupby("grp"):
            out.append(dict(status=c, group=g, **t[c].value_counts().to_dict()))
    return pd.DataFrame(out).fillna(0)


def main(persist=False):
    OUT.mkdir(parents=True, exist_ok=True)
    R = pd.read_parquet(IN / "resolved_v4c.parquet")
    sig = pd.read_parquet(IN / "review_v4c.parquet", columns=["DeviceId", "signal"]).drop_duplicates()
    R = R.merge(sig, on="DeviceId", how="left")
    R = severity_v4d(R, persist=False)
    Rp = severity_v4d(R.copy(), persist=True)
    R["st_v4d_persist"] = Rp.st_v4d.to_numpy()
    R["sev_rule_persist"] = Rp.sev_rule.to_numpy()
    R.to_parquet(OUT / "resolved_v4d.parquet")
    cols = ["st8", "st_v4d", "st_v4d_persist"]
    T = rates(R, cols)
    T.to_csv(OUT / "rates121.csv", index=False)
    print(T.to_string(index=False))
    U = user_cases(R, cols)
    U.to_csv(OUT / "cases121.csv", index=False)
    print(U.to_string(index=False))
    # the 40 rows of review/health_review_v4.xlsx
    rows = json.loads((IN / "plot" / "rows.json").read_text())
    k = pd.DataFrame([{c: r[c] for c in ("row", "signal", "label", "category", "DeviceId", "window", "detector")}
                      for r in rows])
    k = k.merge(R[KEY + ["st8", "st_v4d", "sev_rule", "st_v4d_persist", "n_ev", "left8"]], on=KEY, how="left")
    xl = pd.read_excel(Path(__file__).resolve().parents[3] / "review" / "health_review_v4.xlsx", header=None)
    h = xl.index[xl[0].astype(str).eq("#")][0]
    sh = xl.iloc[h + 1:h + 41, [0, 4, 9]].rename(columns={0: "row", 4: "sheet_status", 9: "user_comment"})
    sh["row"] = sh.row.astype(int)
    k = k.merge(sh, on="row", how="left")
    # old = the status shown on the sheet (config notes shown as such); new = v4d where the rule changed it
    k["old_status"] = k.sheet_status
    k["new_status"] = np.where(k.st_v4d != k.st8, k.st_v4d, k.sheet_status)
    k["with_persist_rule"] = np.where(k.st_v4d_persist != k.st8, k.st_v4d_persist, k.sheet_status)
    keep = ["row", "signal", "label", "category", "window", "old_status", "new_status", "sev_rule", "with_persist_rule",
            "n_ev", "left8", "user_comment"]
    k[keep].to_csv(OUT / "rows121.csv", index=False)
    print(k[keep].to_string(index=False))


if __name__ == "__main__":
    main(persist="--persist" in sys.argv)
