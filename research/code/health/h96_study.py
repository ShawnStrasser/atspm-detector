"""Note 96 study: per-type normal patterns of occupancy / traffic-following, and whether the context separates the
user's health_review_v1 answers (Y vs N / ?).  Reads h96_occ outputs + note-79 run (package health on w40, prod) +
note-82 chosen rows + the user's answers (review/health_review_v1.xlsx, READ ONLY).

    python h96_study.py types|answers
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import openpyxl

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 200)
DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
REPO = Path(__file__).resolve().parents[3]
OUT = DCW / "health96"
CHK = ["dropout", "stuck", "chatter", "rapid", "volume", "level", "choppy", "night_drop", "night_day", "corr"]


def run_w40():
    r = pd.read_parquet(DCW / "health79" / "run.parquet")
    r = r[(r.period == "w40") & (r["mode"] == "prod")].copy()
    r["DeviceId"] = r.DeviceId.str.lower()
    return r


def answers():
    wb = openpyxl.load_workbook(REPO / "review" / "health_review_v1.xlsx", read_only=True)
    ws = wb["Examples"]
    rows = [r for r in ws.iter_rows(min_row=4, values_only=True) if r[0] is not None]
    a = pd.DataFrame([dict(row=r[0], check_name=r[1], signal=r[2], det=r[3], model=r[4], sample=r[5], ans=r[10],
                           comment=r[11]) for r in rows])
    s = a.ans.fillna("").astype(str).str.strip()
    a["answer"] = np.where(s.str[:1].str.upper() == "Y", "Y", np.where(s.str[:1].str.upper() == "N", "N",
                           np.where(s == "", "", "?")))
    a["text"] = s
    c = pd.read_csv(DCW / "health82" / "rows.csv")
    c = c.drop(columns="check").rename(columns={"n": "row", "dev": "DeviceId", "det": "detector", "key": "check"})
    c["DeviceId"] = c.DeviceId.str.lower()
    a = a.merge(c[["row", "DeviceId", "window", "detector", "check", "signal", "sensor"]].rename(
        columns={"signal": "sig2"}), on="row")
    assert (a.sig2.astype(str).str.zfill(5) == a.signal.astype(str).str.zfill(5)).all() and (a.det == a.detector).all()
    return a


def types():
    F = pd.read_parquet(OUT / "feat.parquet")
    r = run_w40()
    F = F.merge(r[["DeviceId", "window", "detector", "status"] + [f"s_{k}" for k in CHK]],
                on=["DeviceId", "window", "detector"], how="left")
    x = F[F.window.str.startswith("h24") & (F.n_on >= 96)]
    ok = x[x.status == "ok"]
    print("h24 rows", len(x), "ok", len(ok))
    q = lambda s, p: s.quantile(p)
    T = ok.groupby("fn").agg(n=("n_on", "size"), occ_med=("occ", "median"), occ_p95=("occ", lambda s: q(s, .95)),
                             occmax_p95=("occ_max", lambda s: q(s, .95)), on_s_med=("mean_on_s", "median"),
                             c_cnt_med=("c_cnt_ref", "median"), c_cnt_p05=("c_cnt_ref", lambda s: q(s, .05)),
                             c_occ_med=("c_occ_ref", "median"), c_occ_p05=("c_occ_ref", lambda s: q(s, .05)),
                             c_occcong_med=("c_occ_cong", "median"), max5_p99=("max5", lambda s: q(s, .99)),
                             share_med=("share_ref", "median"))
    print(T.round(3))
    # per lanes for max5
    print(ok.groupby(["fn", "lanes"]).max5.describe(percentiles=[.5, .99, .995])[["count", "50%", "99%", "99.5%"]]
          .round(0))
    # occupancy vs traffic level and congestion (15-min bins, ok detectors)
    Bn = pd.read_parquet(OUT / "bins_h24.parquet")
    Bn = Bn.merge(ok[["DeviceId", "window", "detector"]], on=["DeviceId", "window", "detector"])
    Bn["rmax"] = Bn.groupby(["DeviceId", "window", "detector"]).ref.transform("max")
    Bn["tl"] = pd.cut(Bn.ref / Bn.rmax, [-.01, .1, .3, .5, .7, .9, 1.0], labels=["<10", "10-30", "30-50", "50-70",
                                                                                "70-90", "90+"])
    Bn["cg"] = pd.cut(Bn.cong, [0, .5, 1, 1.5, 2, 3, 99], labels=["<.5", ".5-1", "1-1.5", "1.5-2", "2-3", "3+"])
    print("median 15-min occupancy % by traffic level (share of the detector's own peak reference)")
    print((100 * Bn.pivot_table(index="fn", columns="tl", values="occ", aggfunc="median")).round(1))
    print("p95 15-min occupancy % by traffic level")
    print((100 * Bn.pivot_table(index="fn", columns="tl", values="occ", aggfunc=lambda s: s.quantile(.95))).round(1))
    print("median 15-min occupancy % by signal congestion index")
    print((100 * Bn.pivot_table(index="fn", columns="cg", values="occ", aggfunc="median")).round(1))
    print("share of bins per congestion class"); print(Bn.cg.value_counts(normalize=True).sort_index().round(3))
    # 'follows traffic in occupancy but not in counts'
    for fnm, g in ok.groupby("fn"):
        lo = g.c_cnt_ref < .5
        print(f"{fnm:11s} counts corr<.5: {lo.mean():.3f}; of those occ corr>=.5: {(g[lo].c_occ_ref >= .5).mean():.2f}")
    # flagged corr / choppy rows by type: does occupancy follow?
    fl = x[x.s_corr >= .35]
    print("corr-flagged h24 rows by fn", fl.fn.value_counts().to_dict())
    print(fl.groupby("fn")[["c_cnt_ref", "c_occ_ref", "c_occ_cong", "occ", "mean_on_s"]].median().round(2))


def ans():
    a = answers()
    F = pd.read_parquet(OUT / "feat.parquet")
    r = run_w40()
    a = a.merge(F, on=["DeviceId", "window", "detector"], how="left").merge(
        r[["DeviceId", "window", "detector", "status", "c_ratio", "c_partner"] + [f"s_{k}" for k in CHK]],
        on=["DeviceId", "window", "detector"], how="left")
    a.to_parquet(OUT / "answers.parquet")
    cols = ["row", "check", "answer", "fn", "lanes", "hours", "n_on", "occ", "occ_max", "mean_on_s", "max5",
            "c_cnt_ref", "c_occ_ref", "c_occ_cong", "cong_max", "share_ref", "ref_kind", "run_min", "run_phocc",
            "run_phocc_all", "run_sgocc", "run_sgocc_all", "run_refn", "run_refn_all", "run_nheld"]
    print(a[cols].round(2).to_string())


if __name__ == "__main__":
    {"types": types, "answers": ans}[sys.argv[1]]()
