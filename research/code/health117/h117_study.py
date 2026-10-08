"""Note 117: RED vs GREEN study per detector type (healthy baseline) -> green-time-aware fast / volume checks.

Inputs: %DC_WORK%/s117/{det117,bins117}.parquet (h117_events) + health110/resolved110.parquet (type, lanes, v110
status and findings).  Healthy for this family = no v110 finding or watch outside {rapid, volume, chatter} (leave-own-
family-out), >= 50 vehicle ONs, color known for its predicted phase.

Statistics per detector-window (vehicle ONs only; extension re-ONs never count):
  r_gy, r_r       ONs per hour of green+yellow / of red (red after the first 2 s)
  o_gy, o_r       % ON during green+yellow / red
  q_gy            ONs per green+yellow hour per lane spanned (lanes = model lane span, >= 1)
  q5_gy           busiest 5 min of green flow per lane: max over 5-min bins with >= 60 s green+yellow of
                  ONs in green+yellow / green+yellow hours / lanes  (saturation ~1800-2000)
  q5_all          busiest 5 min, all ONs per hour per lane (the old 'too many in 5 min', per lane)
  fs_gy, fs_r     share of ONs that come < 1 s after the previous ON, in green+yellow / in red
  fr_h            fast ONs in red per red hour;  fr_n  fast ONs in red (count)
  ch_gy, ch_r     chatter share (OFF -> ON < 0.3 s) in green+yellow / red

New checks (limit = p99.8 of healthy per cell = fn x span x sample length, fallback span x length, length; cells >= 100):
  FAST  fast ONs during RED: fr_h > limit AND fr_n >= 5 (stop-bar types and Advance alike; the limit is per type)
        OR green flow beyond what queue discharge can give: q5_gy > limit.
        A fast-ON share during green alone is no longer a fault (queue discharge every ~0.5 s is normal).
  VOL   'too many in 5 min' = q5_gy (green-time and lane normalised) for stop-bar types; Advance (sees arrivals
        regardless of color) keeps q5_all per lane.
  Severity as the package: suspect at the limit, bad at 2x.

    python h117_study.py            -> %DC_WORK%/s117/{stats117.parquet, base117.csv, lim117.csv, rates117.csv,
                                       changes117.csv, user117.csv}
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 400)
DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
S = DCW / "s117"
Q, MIN_CELL = 0.998, 100
SAT = 1800.0                       # saturation flow, veh per green hour per lane (user: 1800, sometimes more)
KEYS = (["fn", "span", "wg"], ["span", "wg"], ["wg"])
FAM = {"rapid", "volume", "chatter"}
STOPBAR = {"Count", "Yellow_Red", "Presence", "Mid", "Other", "Bike"}
USER = [("2B039", 18, "heavy"), ("04028", 53, "heavy"), ("2C028", 37, "heavy"), ("01062", 2, "heavy"),
        ("04035", 53, "fault?"), ("04035", 52, "fault?"), ("10055", 5, "fault?"), ("08073", 7, "fault?"),
        ("11021", 2, "fault?"), ("11021", 3, "fault?"), ("11021", 4, "fault?")]


def stats():
    D = pd.read_parquet(S / "det117.parquet")
    R = pd.read_parquet(DCW / "health110" / "resolved110.parquet",
                        columns=["DeviceId", "window", "detector", "fn", "type", "span", "lanes", "band", "cmode", "wg",
                                 "st8", "left8", "watch8", "f_top", "technology", "ioi_lt1", "max5", "lim_ioi_lt1",
                                 "lim_max5", "n_on"]).rename(columns={"n_on": "n_on_pkg"})
    X = D.merge(R, on=["DeviceId", "window", "detector"], how="inner")
    X["ln"] = np.fmax(pd.to_numeric(X.lanes, errors="coerce").fillna(1), 1)
    sgy = X.s_G + X.s_Y
    with np.errstate(divide="ignore", invalid="ignore"):
        X["r_gy"] = X.n_GY / sgy * 3600
        X["r_r"] = X.n_R / X.s_R * 3600
        X["o_gy"] = (X.o_G + X.o_Y) / sgy
        X["o_r"] = X.o_R / X.s_R
        X["q_gy"] = X.r_gy / X.ln
        X["fs_gy"] = X.f1_GY / X.n_GY
        X["fs_r"] = X.f1_R / X.n_R
        X["fr_h"] = X.f1_R / X.s_R * 3600
        X["fr_n"] = X.f1_R
        X["ch_gy"] = X.c_GY / X.n_GY
        X["ch_r"] = X.c_R / X.n_R
        X["br_n"] = X.b_R
        X["gfrac"] = sgy / (sgy + X.s_R + X.s_Rg)
    # 5-min bins, aggregated in DuckDB (10 M rows): busiest green flow per lane, busiest 5 min per lane, and the
    # fast ONs EXPECTED if the ONs of each color state in each bin arrived at random at that bin's rate
    # (Poisson: share of intervals < 1 s = 1 - exp(-rate x 1 s)); observed / expected = fast excess.
    import duckdb
    con = duckdb.connect()
    con.execute("SET memory_limit='10GB'; SET threads=4")
    con.register("L", X[["DeviceId", "window", "detector", "ln"]])
    A = con.execute(f"""
        WITH b AS (SELECT b.*, L.ln, greatest(300 - sGY - sR - sRg, 0) AS sU, nG + nY AS nGY
                   FROM read_parquet('{(S / "bins117.parquet").as_posix()}') b JOIN L USING (DeviceId, "window", detector))
        SELECT DeviceId, "window", detector,
               max(CASE WHEN sGY >= 60 THEN nGY / sGY * 3600 / ln END) AS q5_gy,
               max(n / 300 * 3600 / ln) AS q5_all, max(n) AS max5,
               sum(fGY) AS fo_gy, sum(fR) AS fo_r, sum(fGY + fR + fU) AS fo_all,
               sum(CASE WHEN sGY > 0 THEN nGY * (1 - exp(-nGY / sGY)) ELSE 0 END) AS fe_gy,
               sum(CASE WHEN sR > 0 THEN nR * (1 - exp(-nR / sR)) ELSE 0 END) AS fe_r,
               sum(CASE WHEN sU > 0 THEN nU * (1 - exp(-least(nU / sU, 50))) ELSE nU END) AS fe_u
        FROM b GROUP BY ALL""").df()
    # robust variant: each bin's rate per color state capped at its centred 1-h median (+-6 bins), so a burst of
    # fault ONs cannot raise its own expectation; sustained heavy traffic keeps its rate
    Hh = con.execute(f"""
        WITH b AS (SELECT DeviceId, "window", detector, b, fGY, fR, fU, nG + nY AS nGY, nR, nU, sGY, sR,
                          greatest(300 - sGY - sR - sRg, 0) AS sU
                   FROM read_parquet('{(S / "bins117.parquet").as_posix()}')),
        r AS (SELECT *, CASE WHEN sGY >= 30 THEN nGY / sGY END AS rg, CASE WHEN sR >= 30 THEN nR / sR END AS rr,
                        CASE WHEN sU >= 30 THEN nU / sU END AS ru FROM b),
        m AS (SELECT *, median(rg) OVER w AS mg, median(rr) OVER w AS mr, median(ru) OVER w AS mu FROM r
              WINDOW w AS (PARTITION BY DeviceId, "window", detector ORDER BY b ROWS BETWEEN 6 PRECEDING AND 6 FOLLOWING)),
        e AS (SELECT *, CASE WHEN sGY > 0 THEN nGY * (1 - exp(-least(nGY / sGY, coalesce(mg, nGY / sGY)))) ELSE 0 END AS eg,
                        CASE WHEN sR > 0 THEN nR * (1 - exp(-least(nR / sR, coalesce(mr, nR / sR)))) ELSE 0 END AS er,
                        CASE WHEN sU > 0 THEN nU * (1 - exp(-least(nU / sU, coalesce(mu, nU / sU), 50))) ELSE nU END AS eu
              FROM m)
        SELECT DeviceId, "window", detector, sum(eg + er + eu) AS fem_all, sum(er) AS fem_r,
               max((fGY + fR + fU - eg - er - eu) / sqrt(eg + er + eu + 1)) AS zb_max,
               sum(CASE WHEN (fGY + fR + fU - eg - er - eu) / sqrt(eg + er + eu + 1) >= 4
                         AND fGY + fR + fU >= 5 THEN 1 ELSE 0 END) AS n_spk
        FROM e GROUP BY ALL""").df()
    A = A.merge(Hh, on=["DeviceId", "window", "detector"], how="left")
    A["fe_all"] = A.fe_gy + A.fe_r + A.fe_u
    A["fx_all"] = np.where(A.fo_all >= 20, (A.fo_all + 1) / (A.fe_all + 1), np.nan)
    A["fx_r"] = np.where(A.fo_r >= 10, (A.fo_r + 1) / (A.fe_r + 1), np.nan)
    A["fxm_all"] = np.where(A.fo_all >= 20, (A.fo_all + 1) / (A.fem_all + 1), np.nan)
    A["fxm_r"] = np.where(A.fo_r >= 10, (A.fo_r + 1) / (A.fem_r + 1), np.nan)
    # adopted statistic: excess fast ONs over the robust random-arrival expectation, in Poisson standard deviations
    A["zf"] = (A.fo_all - A.fem_all) / np.sqrt(A.fem_all + 1)
    A["zf_r"] = (A.fo_r - A.fem_r) / np.sqrt(A.fem_r + 1)
    X = X.merge(A.rename(columns={"max5": "max5_117"}), on=["DeviceId", "window", "detector"], how="left")
    old = (X.left8.fillna("") + "," + X.watch8.fillna("")).str.split(",")
    X["old_fast"] = X.left8.fillna("").str.split(",").apply(lambda l: "rapid" in l)
    X["old_vol"] = X.left8.fillna("").str.split(",").apply(lambda l: "volume" in l)
    X["old_chat"] = X.left8.fillna("").str.split(",").apply(lambda l: "chatter" in l)
    X["healthy"] = (old.apply(lambda l: set(x for x in l if x) <= FAM) & X.st8.ne("not_enough_data")
                    & (X.n_on >= 50) & X.has_colour & (X.s_R > 600) & (sgy > 600))
    X.to_parquet(S / "stats117.parquet")
    return X


def limit_tab(X, col, hl):
    T = []
    for keys in KEYS:
        gg = X[hl & X[col].notna()].groupby(keys)[col].agg(["count", lambda s: s.quantile(Q)])
        gg.columns = ["cnt", "lim"]
        T.append((keys, gg[gg.cnt >= MIN_CELL].lim))
    lim = pd.Series(np.nan, index=X.index)
    for keys, tab in T:
        m = X[keys].merge(tab.rename("lim"), left_on=keys, right_index=True, how="left").lim.to_numpy()
        fill = lim.isna().to_numpy() & np.isfinite(m)
        lim[fill] = m[fill]
    return lim, T


def main():
    X = stats() if not (S / "stats117.parquet").exists() or os.environ.get("REBUILD") else pd.read_parquet(S / "stats117.parquet")
    H = X.healthy
    # ---------------- baseline per type (24 h and 3 h, healthy)
    cols = ["r_gy", "r_r", "o_gy", "o_r", "q_gy", "q5_gy", "q5_all", "fs_gy", "fs_r", "fr_h", "ch_gy", "ch_r", "gfrac", "fx_all", "fx_r", "fxm_all", "fxm_r", "zf", "zf_r"]
    rows = []
    for (wg, ty), gg in X[H].groupby(["wg", "type"]):
        if len(gg) < 30:
            continue
        r = dict(wg=wg, type=ty, n=len(gg))
        for c in cols:
            v = gg[c].replace([np.inf, -np.inf], np.nan).dropna()
            r[f"{c}_p50"] = v.median()
            r[f"{c}_p99.8"] = v.quantile(Q) if len(v) else np.nan
        rows.append(r)
    base = pd.DataFrame(rows)
    base.to_csv(S / "base117.csv", index=False)
    show = ["wg", "type", "n", "r_gy_p50", "r_r_p50", "o_gy_p50", "o_r_p50", "q5_gy_p50", "q5_gy_p99.8",
            "q5_all_p99.8", "fs_gy_p50", "fs_gy_p99.8", "fx_all_p50", "fx_all_p99.8", "fx_r_p50", "fx_r_p99.8", "zf_p50", "zf_p99.8", "zf_r_p99.8"]
    print(base[base.wg != "m30"][show].round(3).to_string(index=False))
    # ---------------- limits + new checks
    Xc = X.copy()
    Xc["fr_h"] = Xc.fr_h.where(Xc.s_R > 600)
    for c, hl in (("fr_h", H & (Xc.n_R >= 20)), ("fx_all", H), ("fx_r", H), ("fxm_all", H), ("fxm_r", H), ("zf", H), ("zf_r", H), ("zb_max", H), ("n_spk", H), ("q5_gy", H), ("q5_all", H)):
        X[f"lim_{c}"] = limit_tab(Xc, c, hl)[0]
    adv = X.fn.eq("Advance")
    sc = lambda v, l: np.where(np.isfinite(v) & np.isfinite(l) & (l > 0), v / l, np.nan)  # noqa: E731
    enough = X.n_on >= 50
    # FAST (adopted) = fast ONs above what random arrivals at the robust local rate of each color state give, in
    # Poisson SDs (zf); variants kept for the note: ratio with bin rate (fx), ratio robust (fxm), red only (zf_r)
    X["x_fx"] = np.fmax(sc(X.fx_all, X.lim_fx_all), sc(X.fx_r, X.lim_fx_r))
    X["x_fxm"] = np.fmax(sc(X.fxm_all, X.lim_fxm_all), sc(X.fxm_r, X.lim_fxm_r))
    X["x_zf_r"] = sc(X.zf_r, X.lim_zf_r)
    X["x_zf"] = sc(X.zf, X.lim_zf)
    X["v_fx"], X["v_fxm"], X["v_zf_r"] = enough & (X.x_fx >= 1), enough & (X.x_fxm >= 1), enough & (X.x_zf_r >= 1)
    X["x_zb"] = sc(X.zb_max, X.lim_zb_max)
    X["x_spk"] = np.where(X.n_spk > X.lim_n_spk, X.n_spk / np.fmax(X.lim_n_spk, 1), 0)
    X["v_zb"], X["v_spk"] = enough & (X.x_zb >= 1), enough & (X.n_spk > X.lim_n_spk)
    # adopted FAST: whole-sample excess (zf) OR more 5-min bursts (bins >= 4 SD above the robust expectation, >= 5
    # fast ONs) than 1 in 500 healthy detectors of the type have
    X["new_fast_x"] = np.fmax(np.nan_to_num(X.x_zf), X.x_spk)
    X["new_fast"] = enough & ((X.x_zf >= 1) | X.v_spk)
    # VOL = busiest 5 min per lane against max(healthy p99.8, saturation); green time for stop-bar types, all time
    # for Advance (arrivals regardless of color)
    lim_g = np.fmax(X.lim_q5_gy, SAT)
    lim_a = np.fmax(X.lim_q5_all, SAT)
    X["lim_vol"] = np.where(adv, lim_a, lim_g)
    X["new_vol_x"] = np.where(adv, sc(X.q5_all, lim_a), sc(X.q5_gy, lim_g))
    X["new_vol"] = X.new_vol_x >= 1
    X["new_vol_nofloor"] = np.where(adv, sc(X.q5_all, X.lim_q5_all), sc(X.q5_gy, X.lim_q5_gy)) >= 1
    X["new_fast_red"] = X.new_fast                        # (name kept for h117_user)
    X["new_any"] = X.new_fast | X.new_vol
    X["old_any"] = X.old_fast | X.old_vol
    X["lvl_new"] = np.where(np.fmax(np.where(X.new_fast, X.new_fast_x, 0), np.where(X.new_vol, X.new_vol_x, 0)) >= 2, "bad",
                            np.where(X.new_any, "suspect", "ok"))
    X.to_parquet(S / "stats117.parquet")
    lt = X.groupby(["fn", "span", "wg"])[["lim_fr_h", "lim_zf", "lim_q5_gy", "lim_q5_all"]].first().round(1)
    lt.to_csv(S / "lim117.csv")
    print(lt[lt.index.get_level_values(2) != "m30"].to_string())
    # ---------------- rates per type
    out = []
    for (wg, ty), gg in X[X.st8.ne("not_enough_data")].groupby(["wg", "type"]):
        out.append(dict(wg=wg, type=ty, n=len(gg),
                        old_fast=100 * gg.old_fast.mean(), old_vol=100 * gg.old_vol.mean(), old_any=100 * gg.old_any.mean(),
                        new_fast=100 * gg.new_fast.mean(), v_fx=100 * gg.v_fx.mean(), v_fxm=100 * gg.v_fxm.mean(), v_zf_r=100 * gg.v_zf_r.mean(), v_zb=100 * gg.v_zb.mean(), v_spk=100 * gg.v_spk.mean(), v_or=100 * (gg.new_fast | gg.v_spk).mean(), new_vol=100 * gg.new_vol.mean(), vol_nofloor=100 * gg.new_vol_nofloor.mean(), new_any=100 * gg.new_any.mean(),
                        healthy_new=100 * gg[gg.healthy].new_any.mean()))
    rt = pd.DataFrame(out)
    for wg, gg in X[X.st8.ne("not_enough_data")].groupby("wg"):
        out.append(dict(wg=wg, type="ALL", n=len(gg), old_fast=100 * gg.old_fast.mean(), old_vol=100 * gg.old_vol.mean(),
                        old_any=100 * gg.old_any.mean(), new_fast=100 * gg.new_fast.mean(), v_fx=100 * gg.v_fx.mean(), v_fxm=100 * gg.v_fxm.mean(), v_zf_r=100 * gg.v_zf_r.mean(), v_zb=100 * gg.v_zb.mean(), v_spk=100 * gg.v_spk.mean(), v_or=100 * (gg.new_fast | gg.v_spk).mean(), vol_nofloor=100 * gg.new_vol_nofloor.mean(),
                        new_vol=100 * gg.new_vol.mean(), new_any=100 * gg.new_any.mean(),
                        healthy_new=100 * gg[gg.healthy].new_any.mean()))
    rt = pd.DataFrame(out)
    rt.to_csv(S / "rates117.csv", index=False)
    print(rt.round(2).to_string(index=False))
    # ---------------- rows that change
    ch = X[X.old_any != X.new_any]
    ch[["DeviceId", "window", "detector", "type", "ln", "st8", "left8", "old_fast", "old_vol", "new_fast", "new_vol",
        "new_fast_x", "new_vol_x", "zf", "lim_zf", "fx_all", "lim_vol", "fr_n", "fr_h", "q5_gy", "q5_all", "fs_gy", "fs_r"]].to_csv(S / "changes117.csv", index=False)
    print("changes", len(ch), "old->ok", int((ch.old_any & ~ch.new_any).sum()), "new flags", int((~ch.old_any & ch.new_any).sum()))


if __name__ == "__main__":
    main()
