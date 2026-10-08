"""Note 117: the user's rows (Oct 7 point 2 + the faults that must stay caught) through the old and new fast / volume
checks.  Reads %DC_WORK%/s117/stats117.parquet (h117_study) and the v110 review rows (signal -> DeviceId).

    python h117_user.py   -> %DC_WORK%/s117/user117.csv
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
S = DCW / "s117"
USER = [("2B039", [18], "h24_b", "heavy"), ("04028", [53], "h24_b", "heavy"), ("2C028", [37], "h3_b", "heavy"),
        ("01062", [2], "h3_b", "heavy"), ("04035", [52, 53], "h3_b", "fault?"), ("10055", [5], "h3_b", "fault?"),
        ("08073", [7], "h24_b", "fault?"), ("11021", None, "h3_b", "fault?")]


def main():
    X = pd.read_parquet(S / "stats117.parquet")
    dev = pd.read_csv(DCW / "health110" / "review" / "rows_v110.csv").drop_duplicates("signal").set_index("signal").dev
    rows = []
    for sig, dets, w, exp in USER:
        for ww in ([w] if w.startswith("m30") else [w, w.replace("_b", "_a")] if w.startswith("h") else [w]):
            x = X[(X.DeviceId == dev[sig]) & (X.window == ww)]
            x = x[x.phase.eq(2)] if dets is None else x[x.detector.isin(dets)]
            for _, r in x.iterrows():
                rows.append(dict(signal=sig, det=int(r.detector), window=ww, user=exp, type=r.type, lanes=r.ln,
                                 st_v110=r.st8, old=",".join(k for k, v in (("fast", r.old_fast), ("vol", r.old_vol),
                                                                          ("chatter", r.old_chat)) if v),
                                 new=",".join(k for k, v in (("fast", r.new_fast), ("vol", r.new_vol)) if v),
                                 lvl_new=r.lvl_new, n_on=r.n_on, gfrac=round(r.gfrac, 2), r_gy=round(r.r_gy), r_r=round(r.r_r),
                                 o_gy=round(r.o_gy, 3), o_r=round(r.o_r, 3), fs_gy=round(r.fs_gy, 3), fs_r=round(r.fs_r, 3),
                                 fo_all=r.fo_all, fe_all=round(r.fe_all), fx_all=round(r.fx_all, 2), lim_fx_all=round(r.lim_fx_all, 2),
                                 fo_r=r.fo_r, fe_r=round(r.fe_r, 1), fx_r=round(r.fx_r, 2), lim_fx_r=round(r.lim_fx_r, 2),
                                 zf=round(r.zf, 1), lim_zf=round(r.lim_zf, 1), zf_r=round(r.zf_r, 1), lim_zf_r=round(r.lim_zf_r, 1), q5_gy=round(r.q5_gy), q5_all=round(r.q5_all), lim_vol=round(r.lim_vol),
                                 ch_gy=round(r.ch_gy, 3), ch_r=round(r.ch_r, 3), max5=r.max5_117))
    U = pd.DataFrame(rows)
    U.to_csv(S / "user117.csv", index=False)
    print(U.to_string(index=False))


if __name__ == "__main__":
    main()
