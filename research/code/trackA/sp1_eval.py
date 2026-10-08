"""Note 60 -- evaluate the deterministic night-time free-flow speed (sp1_night_speed.py) on the Sept-2026 staging
windows, OOF inputs only: function / phase / lanes = ln8 D.func lanes table (note 58), setback = sb7 'all' P50 OOF
(note 58).  Print distance and technology are used for EVALUATION only (sensitivity, per technology).  locked_v2 is
asserted absent.  Windows: note 58's nine (m30 a-d, h6 a/b, h24 a/b, full66) + three night-anchored short windows
(n30 Sat 01:00-01:30, n2h Sat 01:00-03:00, n6 Sun 00:00-06:00) that borrow m30_c's predictions (Sat 21:30, the
nearest 30-min sample; disclosed approximation).

    python sp1_eval.py run | report
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sp1_night_speed as NS  # noqa: E402

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
OUT = DCW / "trackA" / "speed" / "sp1"
CACHE = DCW / "official" / "stg" / "cache"
WINS = {"m30_a": ("2026-09-21 07:30:00", 1800), "m30_b": ("2026-09-19 12:00:00", 1800),
        "m30_c": ("2026-09-19 21:30:00", 1800), "m30_d": ("2026-09-18 17:00:00", 1800),
        "h6_a": ("2026-09-20 06:00:00", 6 * 3600), "h6_b": ("2026-09-19 12:00:00", 6 * 3600),
        "h24_a": ("2026-09-19 00:00:00", 24 * 3600), "h24_b": ("2026-09-20 00:00:00", 24 * 3600),
        "full66": ("2026-09-18 16:15:00", 66 * 3600),
        "n30": ("2026-09-19 01:00:00", 1800), "n2h": ("2026-09-19 01:00:00", 2 * 3600),
        "n6": ("2026-09-20 00:00:00", 6 * 3600)}
PRED_WIN = {"n30": "m30_c", "n2h": "m30_c", "n6": "m30_c"}
E0 = pd.Timestamp("1970-01-01")
SHIFT = 47.0


def ts(s):
    return (pd.Timestamp(s) - E0).total_seconds()


def locked() -> set:
    return set(pd.read_csv(DCW / "official" / "locked_v2.csv").DeviceId.str.lower())


def inputs():
    sb = pd.read_parquet(DCW / "trackA" / "setback" / "sb7" / "oof.parquet",
                         columns=["dev", "det", "win", "phase", "fn", "all", "print_fn", "technology", "dist",
                                  "dist_kind", "fold"])
    L = pd.read_parquet(DCW / "lanes" / "ln8" / "lanes_D.func.parquet",
                        columns=["DeviceId", "Detector", "phase", "function", "lanes", "period", "win"])
    L = L[(L.period == "stg") & L.win.isin(set(WINS) | set(PRED_WIN.values()))].copy()
    L["dev"] = L.DeviceId.str.lower()
    L = L.rename(columns={"Detector": "det"}).drop(columns=["DeviceId", "period"])
    L["det"] = L.det.astype("int16")
    devs = set(sb.dev)
    L = L[L.dev.isin(devs)]
    lk = locked()
    assert not (devs & lk), "locked signal in inputs"
    return sb, L


def run(chunk=80):
    OUT.mkdir(parents=True, exist_ok=True)
    sb, L = inputs()
    devs = sorted(sb.dev.unique())
    con = duckdb.connect()
    con.execute("SET memory_limit='10GB'; SET threads=4; SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{(DCW / 'tmp').as_posix()}'")
    det_out, ph_out = [], []
    for i in range(0, len(devs), chunk):
        ds = devs[i:i + chunk]
        con.register("dv", pd.DataFrame({"dev": ds}))
        iv = con.sql(f"""SELECT lower(DeviceId) AS dev, Detector::SMALLINT AS det,
                epoch_ms(t_on) / 1000.0 AS t_on, epoch_ms(t_off) / 1000.0 AS t_off
            FROM read_parquet('{(CACHE / 'det_intervals.parquet').as_posix()}')
            WHERE lower(DeviceId) IN (SELECT dev FROM dv) AND (hour(t_on) < 6 OR hour(t_on) >= 23)""").df()
        cy = con.sql(f"""SELECT lower(DeviceId) AS dev, Phase::INT AS phase,
                epoch_ms(green_start) / 1000.0 AS gs,
                epoch_ms(coalesce(yellow_start, red_start, TIMESTAMP '2026-09-21 10:24:00')) / 1000.0 AS ge
            FROM read_parquet('{(CACHE / 'phase_cycles.parquet').as_posix()}')
            WHERE lower(DeviceId) IN (SELECT dev FROM dv) AND green_start IS NOT NULL""").df()
        ivg = {k: g for k, g in iv.groupby("dev")}
        cyg = {k: g for k, g in cy.groupby("dev")}
        for dev in ds:
            e = ivg.get(dev, iv.iloc[:0])
            c = cyg.get(dev, cy.iloc[:0])
            for w, (s0, dur) in WINS.items():
                t0 = ts(s0)
                t1 = t0 + dur
                pw = PRED_WIN.get(w, w)
                lw = L[(L.dev == dev) & (L.win == pw)]
                if lw.empty:
                    continue
                sw = sb[(sb.dev == dev) & (sb.win == pw)].set_index("det")
                ew = e[(e.t_on >= t0) & (e.t_on < t1)]
                on = {k: g.t_on.to_numpy() for k, g in ew.groupby("det")}
                off = {k: g.t_off.to_numpy() for k, g in ew.groupby("det")}
                cw = c[(c["gs"] < t1) & (c["ge"] > t0)]
                greens = {k: (np.maximum(g["gs"].to_numpy(), t0), np.minimum(g["ge"].to_numpy(), t1))
                          for k, g in cw.groupby("phase")}
                for variant in ("pred", "print", "shift"):
                    if variant == "shift" and w not in ("h24_a", "full66"):
                        continue
                    dets = []
                    for r in lw.itertuples():
                        lanes = tuple(int(x) for x in str(r.lanes).split(",") if x.strip().isdigit())
                        s = np.nan
                        if r.function == "Advance" and r.det in sw.index:
                            q = sw.loc[r.det]
                            if variant in ("pred", "shift"):
                                s = q["all"]
                            elif q.print_fn == "Advance" and q.dist_kind == "single":
                                s = q.dist
                        dets.append(dict(det=int(r.det), phase=int(r.phase), function=r.function, lanes=lanes,
                                         setback_ft=float(s)))
                    if variant == "print" and not any(np.isfinite(x["setback_ft"]) for x in dets):
                        continue
                    o1, f1 = on, off
                    if variant == "shift":   # control: stop-bar zones moved SHIFT s later (same rhythm, no same-vehicle link)
                        adv = {x["det"] for x in dets if x["function"] == "Advance"}
                        o1 = {k: (v if k in adv else v + SHIFT) for k, v in on.items()}
                        f1 = {k: (v if k in adv else v + SHIFT) for k, v in off.items()}
                    dr, pr = NS.night_speed(t0, t1, dets, o1, f1, greens)
                    for x in dr:
                        x.update(dev=dev, win=w, variant=variant, partner=str(x["partner"]))
                    for x in pr:
                        x.update(dev=dev, win=w, variant=variant)
                    det_out += dr
                    ph_out += pr
        print(f"chunk {i // chunk}: {len(ds)} signals, det rows {len(det_out)}", flush=True)
    D = pd.DataFrame(det_out)
    P = pd.DataFrame(ph_out)
    D.to_parquet(OUT / "det.parquet", index=False)
    P.to_parquet(OUT / "phase.parquet", index=False)
    print(D.shape, P.shape)


def q(x, ps=(10, 25, 50, 75, 90)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [round(float(v), 1) for v in np.percentile(x, ps)] if len(x) else []


def report():
    D = pd.read_parquet(OUT / "det.parquet")
    P = pd.read_parquet(OUT / "phase.parquet")
    sb, _ = inputs()
    res = {}
    # coverage by window (pred variant): phases with >= 1 predicted Advance
    pp = P[P.variant == "pred"]
    cov = pp.groupby("win").apply(lambda g: pd.Series(dict(
        phases=len(g), answered=int(g.speed_mph.notna().sum()), share=round(g.speed_mph.notna().mean(), 3),
        n_med=float(g.n[g.speed_mph.notna()].median()) if g.speed_mph.notna().any() else np.nan,
        reasons=g.reason_ph.value_counts().to_dict())), include_groups=False)
    res["coverage"] = cov.reset_index().to_dict("records")
    dp = D[D.variant == "pred"]
    res["det_coverage"] = dp.groupby("win").apply(lambda g: dict(n=len(g), share=round(g.speed_mph.notna().mean(), 3),
                                                                 reasons=g.reason.value_counts().to_dict()),
                                                  include_groups=False).to_dict()
    # distribution (pred) on full66, h24
    for w in ("full66", "h24_a", "h24_b", "n6", "n2h"):
        v = pp[(pp.win == w)].speed_mph.dropna()
        res[f"dist_{w}"] = dict(n=len(v), pct=q(v), in25_55=round(v.between(25, 55).mean(), 3),
                                in20_60=round(v.between(20, 60).mean(), 3), lt15=round((v < 15).mean(), 3),
                                gt65=round((v > 65).mean(), 3),
                                iqr_rel_med=round(float(((pp[pp.win == w].p75 - pp[pp.win == w].p25)
                                                         / pp[pp.win == w].speed_mph).median()), 3))
    # by phase number group (evaluation only): 2/6 vs 4/8 vs left turns
    f = pp[pp.win == "full66"].dropna(subset=["speed_mph"])
    grp = np.where(f.phase.isin([2, 6]), "2/6", np.where(f.phase.isin([4, 8]), "4/8", "other"))
    res["by_phase_group_full"] = {g: dict(n=int((grp == g).sum()), pct=q(f.speed_mph[grp == g])) for g in
                                  ("2/6", "4/8", "other")}
    # opposing pairs 2-6 and 4-8 at full66
    piv = f.pivot_table(index="dev", columns="phase", values="speed_mph")
    opp = []
    for a, b in ((2, 6), (4, 8)):
        if a in piv and b in piv:
            m = piv[[a, b]].dropna()
            opp.append(dict(pair=f"{a}-{b}", n=len(m), corr=round(float(np.corrcoef(np.log(m[a]), np.log(m[b]))[0, 1]), 3),
                            med_abs_diff=round(float((m[a] - m[b]).abs().median()), 1)))
    res["opposing"] = opp
    # stability across windows: h24_a vs h24_b, h24 vs full
    piv2 = pp.pivot_table(index=["dev", "phase"], columns="win", values="speed_mph")
    st = {}
    for a, b in (("h24_a", "h24_b"), ("h24_a", "full66"), ("n6", "full66"), ("n2h", "full66"), ("n30", "full66")):
        if a in piv2 and b in piv2:
            m = piv2[[a, b]].dropna()
            st[f"{a}_vs_{b}"] = dict(n=len(m), med_abs_diff=round(float((m[a] - m[b]).abs().median()), 2),
                                     p90_abs_diff=round(float((m[a] - m[b]).abs().quantile(.9)), 2))
    res["stability"] = st
    # shifted-stop-bar control and coverage by phase group (evaluation only)
    ctl = {}
    for v in ("pred", "shift"):
        for w in ("h24_a", "full66"):
            x = P[(P.variant == v) & (P.win == w)]
            a = x.speed_mph.notna()
            ctl[f"{v}_{w}"] = dict(phases=len(x), answered=int(a.sum()), share=round(float(a.mean()), 3),
                                   n_med=float(x.n[a].median()) if a.any() else None,
                                   n_bg_med=float(x.n_bg[a].median()) if a.any() else None, pct=q(x.speed_mph),
                                   reasons=x.reason_ph.value_counts().to_dict())
    res["control"] = ctl
    cg = {}
    for w in ("n30", "n2h", "n6", "h24_a", "full66"):
        x = pp[pp.win == w]
        g = np.where(x.phase.isin([2, 6]), "2/6", np.where(x.phase.isin([4, 8]), "4/8", "other"))
        cg[w] = {k: dict(phases=int((g == k).sum()), share=round(float(x.speed_mph[g == k].notna().mean()), 3),
                         share_with_partner=round(float(x.speed_mph[(g == k) & (x.reason_ph != "no_partner")]
                                                        .notna().mean()), 3)) for k in ("2/6", "4/8", "other")}
    res["coverage_by_group"] = cg
    # sensitivity to setback error: detectors with a print distance (pred vs print variant, same window)
    k = ["dev", "det", "win"]
    a = D[D.variant == "pred"][k + ["speed_mph", "n"]]
    b = D[D.variant == "print"][k + ["speed_mph", "n"]]
    m = a.merge(b, on=k, suffixes=("_pr", "_pt"))
    meta = sb[["dev", "det", "win", "all", "dist", "dist_kind", "print_fn", "technology"]]
    m = m.merge(meta, on=["dev", "det", "win"], how="left")
    sens = {}
    for w in ("full66", "h24_a", "h24_b"):
        x = m[(m.win == w) & m.speed_mph_pr.notna() & m.speed_mph_pt.notna()]
        rel = (x.speed_mph_pr / x.speed_mph_pt - 1)
        se = (x["all"] / x.dist - 1)
        sens[w] = dict(n=len(x), speed_rel_err_abs_med=round(float(rel.abs().median()), 3),
                       within10pct=round(float((rel.abs() <= .10).mean()), 3),
                       within5mph=round(float(((x.speed_mph_pr - x.speed_mph_pt).abs() <= 5).mean()), 3),
                       setback_rel_err_abs_med=round(float(se.abs().median()), 3),
                       corr_rel=round(float(np.corrcoef(rel, se)[0, 1]), 3),
                       print_speed_pct=q(x.speed_mph_pt), pred_speed_pct=q(x.speed_mph_pr),
                       print_in25_55=round(float(x.speed_mph_pt.between(25, 55).mean()), 3))
        by_t = {}
        for t, g in x.groupby(x.technology.fillna("none")):
            r = (g.speed_mph_pr / g.speed_mph_pt - 1)
            by_t[t] = dict(n=len(g), rel_abs_med=round(float(r.abs().median()), 3),
                           within10pct=round(float((r.abs() <= .1).mean()), 3), print_speed=q(g.speed_mph_pt),
                           pred_speed=q(g.speed_mph_pr), print_in25_55=round(float(g.speed_mph_pt.between(25, 55).mean()), 3))
        sens[w]["by_tech"] = by_t
    res["sensitivity"] = sens
    # per technology (print technology of the advance, evaluation only) on full66, pred variant, detector level
    dd = dp[dp.win == "full66"].merge(meta[meta.win == "full66"].drop(columns="win"), on=["dev", "det"], how="left")
    res["tech_full66"] = {t: dict(n=len(g), answered=round(float(g.speed_mph.notna().mean()), 3), pct=q(g.speed_mph),
                                  in25_55=round(float(g.speed_mph.dropna().between(25, 55).mean()), 3),
                                  reasons=g.reason.value_counts().to_dict())
                          for t, g in dd.groupby(dd.technology.fillna("unlabelled"))}
    res["role_full66"] = {r: dict(n=len(g), pct=q(g.speed_mph)) for r, g in dd.groupby(dd.role.fillna("none"))}
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
