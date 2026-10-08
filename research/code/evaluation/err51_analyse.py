"""Note 51 step 2: break down function_v3e OOF errors (rows from err51_build.py) on both scoring sets.

    python err51_analyse.py > %DC_WORK%/trackA/err51/breakdown.txt
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

D = os.path.join(os.environ.get("DC_WORK", os.path.expanduser("~/dc_work")), "trackA/err51/")
C7 = ["Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other"]
pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200)


def by(df, col, name=None):
    g = df.groupby(col, dropna=False).agg(rows=("ok", "size"), acc=("ok", "mean"), err=("ok", lambda s: (~s).sum()))
    g["err_share"] = g.err / (~df.ok).sum()
    g["acc"] = g.acc.round(4); g["err_share"] = g.err_share.round(3)
    print(f"\n-- by {name or col}\n{g.sort_values('rows', ascending=False).to_string()}")


def main():
    fr = pd.read_parquet(D + "rows.parquet")
    for sn, m in (("EVERYTHING", fr.setA), ("REALISTIC", fr.setR)):
        df = fr[m].copy()
        e = df[~df.ok]
        print(f"\n\n========== {sn}: rows {len(df):,}, signals {df.DeviceId.nunique()}, acc7 {df.ok.mean():.4f}, "
              f"errors {len(e):,}")
        cm = pd.crosstab(df.truth, df.pred).reindex(index=C7, columns=C7, fill_value=0)
        print("\nconfusion (rows = truth, cols = pred)\n", cm.to_string())
        rec = pd.Series({c: cm.loc[c, c] / cm.loc[c].sum() for c in C7}).round(3)
        prec = pd.Series({c: cm.loc[c, c] / max(cm[c].sum(), 1) for c in C7}).round(3)
        print("recall", rec.to_dict(), "\nprecision", prec.to_dict())
        pairs = e.groupby(["truth", "pred"]).size().sort_values(ascending=False)
        print("\ntop confusion pairs (share of errors)\n", (pairs / len(e)).round(3).head(15).to_string())
        # unordered pairs
        up = e.assign(pair=[" <-> ".join(sorted([a, b])) for a, b in zip(e.truth, e.pred)]).groupby("pair").size()
        print("\nunordered pairs\n", (up / len(e)).sort_values(ascending=False).round(3).head(10).to_string())
        by(df, "wgroup", "window")
        by(df, "technology")
        by(df, "health")
        df["perm_rt"] = np.where(df.right_turn, "right_turn_lane", np.where(df.permissive, "permissive_phase", "other"))
        by(df, "perm_rt", "permissive / right-turn")
        by(df, "truth_src", "label source")
        by(df, "v3_source", "v3 table source")
        by(df, "validated", "label check status")
        by(df, "tier", "print tier")
        by(df, "unusual", "unusual layout")
        df["phase_state"] = np.where(df.phase_ok.isna(), "no timing phase",
                                     np.where(df.phase_ok.astype(bool), "phase right", "phase WRONG"))
        by(df, "phase_state", "phase prediction")
        # confident vs not
        df["pbin"] = pd.cut(df.p_max, [0, .5, .7, .9, .97, 1.01])
        by(df, "pbin", "model confidence")
        # signal concentration
        s = df.groupby("DeviceId").agg(rows=("ok", "size"), err=("ok", lambda x: (~x).sum()))
        s["rate"] = s.err / s.rows
        n10 = max(1, int(round(len(s) * .1)))
        top_cnt = s.sort_values("err", ascending=False).head(n10).err.sum() / s.err.sum()
        top_rate = s.sort_values("rate", ascending=False).head(n10)
        print(f"\n-- signal concentration: {len(s)} signals; worst 10 % ({n10}) by error COUNT hold "
              f"{top_cnt:.3f} of errors; worst 10 % by error RATE hold {top_rate.err.sum() / s.err.sum():.3f} of "
              f"errors on {top_rate.rows.sum() / s.rows.sum():.3f} of rows (their acc {1 - top_rate.err.sum() / top_rate.rows.sum():.3f})")
        print("share of signals with 0 errors", round((s.err == 0).mean(), 3), "; median signal acc",
              round(1 - s.rate.median(), 3), "; acc without worst 10 % by rate",
              round(1 - (s.err.sum() - top_rate.err.sum()) / (s.rows.sum() - top_rate.rows.sum()), 4))
        # detector concentration: detectors wrong in every window vs occasionally
        d = df.groupby(["DeviceId", "Detector"]).ok.agg(["mean", "size"])
        det_err = df.assign(dacc=df.set_index(["DeviceId", "Detector"]).index.map(d["mean"]))
        cls = pd.cut(det_err.dacc, [-.01, .0001, .5, .9999, 1.01], labels=["always wrong", "mostly wrong",
                                                                            "sometimes wrong", "always right"])
        print("\n-- errors by detector consistency (share of this set's errors)\n",
              (det_err[~det_err.ok].groupby(cls[~det_err.ok], observed=False).size() / len(e)).round(3).to_string())
        print("detectors:", (d["mean"] == 0).sum(), "always wrong of", len(d), "; mostly wrong", ((d["mean"] > 0) & (d["mean"] <= .5)).sum())
        # phase-error x pair
        ph = e.assign(phase_state=df.loc[e.index, "phase_state"])
        print("\n-- phase wrong share within top pairs\n",
              ph.groupby(["truth", "pred"]).phase_state.apply(lambda x: (x == "phase WRONG").mean()).loc[pairs.index[:8]].round(3).to_string())
        # window x top pairs
        print("\n-- top pairs by window (errors per 1000 rows)")
        top = list(pairs.index[:6])
        tab = {}
        for g, dg in df.groupby("wgroup"):
            eg = dg[~dg.ok]
            tab[g] = {f"{a}->{b}": round(1000 * ((eg.truth == a) & (eg.pred == b)).sum() / len(dg), 1) for a, b in top}
        print(pd.DataFrame(tab).to_string())
        # technology x top pairs
        print("\n-- top pairs by technology (errors per 1000 rows of that truth class)")
        tab = {}
        for g, dg in df.groupby(df.technology.fillna("none")):
            tab[g] = {f"{a}->{b}": round(1000 * ((dg.truth == a) & (dg.pred == b)).sum() / max((dg.truth == a).sum(), 1), 1)
                      for a, b in top}
            tab[g]["rows"] = len(dg)
        print(pd.DataFrame(tab).to_string())


if __name__ == "__main__":
    main()
