"""Setback distance, physics method (note 39), step 2 -- the user's rule, no machine learning.

  1. Free-flow speed from stop-bar zone ON times.  Isolated vehicles (>= 3 s clear before and
     after on the zone) crossing >= 15 s after begin-green, in the detector's low-flow hours
     (hourly count <= its median hour).  speed = (20 ft vehicle + zone length) / ON time, the
     median over vehicles.  Zone length prior: Count / Yellow_Red 6 ft loop, Presence 20 ft.
     Pulse-mode zones (>= 50 % of ONs one 0.1 s tick) give no speed.  Valid 15-60 mph.
     Speed source, first that exists: the travel-time partner itself > median of the phase's
     Count / Yellow_Red zones > median of its Presence zones > the advance zone's own ON time
     (20 + 6 ft) > 40 mph.
  2. Travel time from the advance zone to a same-lane stop-bar zone.  Advance ONs >= 10 s (note 39: 4 s) into
     green and >= 10 s (4 s) before yellow (--since / --toy), >= 4 s clear before / 2 s after, low-flow hours; each is
     matched to the NEXT ON of the partner 0.2-15 s later.  tau = median of the matches within
     [0.7, 1.4] x the smoothed histogram mode (random matches are uniform, the mode is the
     same-vehicle travel time); accepted when >= 5 matches sit in that window and the mode
     stands >= 1.5x above the uniform background.
     Same lane = note-30 pair model P(same) >= 0.5 (sb3_pairs.py, OOF, same window).  Partner
     preference: Count > Yellow_Red > Presence, then P(same).  No accepted same-lane partner ->
     "no partner" (fold-median fallback in the all-detector table, as in note 31).
  3. distance = speed x tau + stop-bar zone upstream edge (Count / YR 0 ft, Presence 20 ft);
     < 15 ft -> 0 (stop-bar zone).
Partner role = v3 label function (print, else config; in production: the predicted function).
Also reported: the same with the PRINT lane (oracle pairing) and with a fixed 40 mph.

    python sb3_physics.py [--hours 66] [--start 0]
Output: %DC_WORK%/trackA/setback/sb3_phys_h{H}_s{S}.parquet (one row per Advance / Mid target).
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import lr_common as C
import sb1_setback as S1
import sb3_pairs as SP

OUT = C.DCW / "trackA" / "setback"
L_VEH = 20.0
ZONE = {"Count": 6.0, "Yellow_Red": 6.0, "Presence": 20.0}
EDGE = {"Count": 0.0, "Yellow_Red": 0.0, "Presence": 20.0}
ROLE_RANK = {"Count": 0, "Yellow_Red": 1, "Presence": 2}
V40 = 58.7
VMIN, VMAX = 22.0, 88.0          # 15-60 mph
CLIP0 = 15.0

SQL_BASE = """
CREATE TEMP TABLE onc AS
WITH o AS (
  SELECT lower(i.DeviceId) AS dev, i.Detector::SMALLINT AS det, ad.phase,
         epoch_ms(i.t_on) / 1000.0 - {e0} AS t, i.dur,
         floor((epoch_ms(i.t_on) / 1000.0 - {e0}) / 3600)::INT AS hr
  FROM read_parquet('{di}') i
  JOIN ad ON lower(i.DeviceId) = ad.dev AND i.Detector = ad.det
  WHERE i.t_on >= TIMESTAMP '{t0}' AND i.t_on < TIMESTAMP '{t1}'
), g AS (
  SELECT *, t - lag(t + dur) OVER w AS gap_prev, lead(t) OVER w - (t + dur) AS gap_next
  FROM o WINDOW w AS (PARTITION BY dev, det ORDER BY t)
), h AS (SELECT dev, det, hr, count(*) AS nh FROM o GROUP BY ALL),
hm AS (SELECT dev, det, median(nh) AS nh_med FROM h GROUP BY ALL)
SELECT g.*, (h.nh <= hm.nh_med) AS low, t - c.t_g AS since, c.t_y - t AS to_y
FROM g JOIN h USING (dev, det, hr) JOIN hm USING (dev, det)
ASOF JOIN cyc c ON g.dev = c.dev AND g.phase = c.phase AND g.t >= c.t_g
WHERE g.t < c.t_next
"""

SQL_SPEED = """
SELECT dev, det, count(*) AS n_iso, avg((dur <= 0.15)::INT) AS pulse,
       count(*) FILTER (low) AS n_low,
       median(dur) FILTER (dur > 0.15) AS dur50_all,
       median(dur) FILTER (dur > 0.15 AND low) AS dur50_low
FROM onc
WHERE to_y > 0 AND since >= 15 AND gap_prev >= 3 AND gap_next >= 3
GROUP BY ALL
"""

SQL_MATCH = """
WITH a AS (
  SELECT o.dev, o.det AS da, o.t, o.dur AS dur_a, o.low
  FROM onc o SEMI JOIN tg ON o.dev = tg.dev AND o.det = tg.det
  WHERE to_y >= {toy} AND since >= {since} AND gap_prev >= 4 AND gap_next >= 2
), ap AS (SELECT a.*, p.db, a.t + 0.2 AS t2 FROM a JOIN ptab p ON a.dev = p.dev AND a.da = p.da),
b AS (SELECT dev, det, t FROM onc)
SELECT ap.dev, ap.da, ap.db, ap.low, ap.dur_a, b.t - ap.t AS dt
FROM ap ASOF JOIN b ON ap.dev = b.dev AND ap.db = b.det AND b.t >= ap.t2
WHERE b.t - ap.t <= 15
"""


def robust_tau(dt: np.ndarray, dur: np.ndarray | None = None) -> dict:
    """Mode-anchored median of advance -> stop-bar delays (0.2-15 s).  With the advance ON
    times `dur`, also the per-vehicle distance (20 ft + 6 ft loop) / dur_i x dt_i, median over
    the in-window vehicles with a non-pulse ON (d_veh)."""
    ok = (dt > 0.2) & (dt <= 15)
    dt = dt[ok]
    dur = None if dur is None else dur[ok]
    r = {"n_match": len(dt), "tau": np.nan, "n_in": 0, "pk_strength": np.nan,
         "tau_med_raw": float(np.median(dt)) if len(dt) else np.nan}
    if len(dt) < 5:
        return r
    h, e = np.histogram(dt, bins=np.arange(0.2, 15.2, 0.2))
    hs = np.convolve(h, np.ones(5) / 5, "same")
    k = hs.argmax()
    mode = (e[k] + e[k + 1]) / 2
    lo, hi = 0.7 * mode, 1.4 * mode
    inw = dt[(dt >= lo) & (dt <= hi)]
    bg = len(dt) / 14.8 * (hi - lo)          # uniform background expected in the window
    r.update(n_in=len(inw), pk_strength=len(inw) / max(bg, 0.5), mode=mode)
    if len(inw) >= 5 and r["pk_strength"] >= 1.5:
        r["tau"] = float(np.median(inw))
        if dur is not None:
            w = (dt >= lo) & (dt <= hi) & (dur > 0.15)
            v = (L_VEH + 6.0) / dur[w]
            w2 = (v >= VMIN) & (v <= VMAX)
            if w2.sum() >= 5:
                r["d_veh"] = float(np.median(v[w2] * dt[w][w2]))
                r["v_veh"] = float(np.median(v[w2]))
    return r


def lanes_overlap(li_a, sp_a, li_b, sp_b):
    ok = li_a.notna() & li_b.notna()
    ea, eb = li_a + sp_a.fillna(1) - 1, li_b + sp_b.fillna(1) - 1
    return np.where(ok, (li_a <= eb) & (li_b <= ea), np.nan)


def speed_of(sp_row, role) -> float:
    if sp_row is None or pd.isna(sp_row.get("dur")) or sp_row["pulse"] >= 0.5:
        return np.nan
    v = (L_VEH + ZONE.get(role, 6.0)) / sp_row["dur"]
    return v if VMIN <= v <= VMAX else np.nan


def run(hours: float, start: float, since: float = 10, toy: float = 10) -> pd.DataFrame:
    tgt, sb = S1.truth()
    lock = SP.locked_all()
    assert not tgt.dev.isin(lock).any() and not sb.dev.isin(lock).any()
    tgt = tgt[tgt.dist_kind == "single"].copy()
    sbz = sb[sb.function.isin(list(ZONE))].copy()
    pairs = pd.read_parquet(OUT / f"sb3_pairs_h{hours:g}_s{start:g}.parquet")
    ps = pd.concat([pairs[["dev", "da", "db", "p_same"]],
                    pairs[["dev", "db", "da", "p_same"]].rename(columns={"db": "da", "da": "db"})])
    # candidate partners: stop-bar zones on the same timing phase
    cand = tgt[["dev", "detector", "phase", "lane_index", "lanes_spanned"]].merge(
        sbz.rename(columns={"detector": "db", "lane_index": "li_b", "lanes_spanned": "sp_b"}),
        on=["dev", "phase"])
    cand = cand.rename(columns={"detector": "da"})
    cand["da"] = cand.da.astype("int16")
    cand["db"] = cand.db.astype("int16")
    cand = cand[cand.da != cand.db].merge(ps, on=["dev", "da", "db"], how="left")
    cand["lane_print"] = lanes_overlap(cand.lane_index.astype(float),
                                       cand.lanes_spanned.astype(float), cand.li_b, cand.sp_b)
    # --- events
    t0 = S1.T0 + pd.Timedelta(hours=float(start))
    t1 = t0 + pd.Timedelta(hours=float(hours))
    e0 = (t0 - pd.Timestamp("1970-01-01")).total_seconds()
    con = S1.connect()
    ad = pd.concat([tgt[["dev", "detector", "phase"]],
                    sbz[sbz.dev.isin(set(tgt.dev))][["dev", "detector", "phase"]]]
                   ).drop_duplicates(["dev", "detector"]).rename(columns={"detector": "det"})
    ad["det"] = ad.det.astype("int16")
    con.register("ad_df", ad)
    con.execute("CREATE TEMP TABLE ad AS SELECT * FROM ad_df")
    con.register("tg_df", tgt[["dev", "detector"]].rename(columns={"detector": "det"}))
    con.execute("CREATE TEMP TABLE tg AS SELECT * FROM tg_df")
    con.register("pt_df", cand[["dev", "da", "db"]])
    con.execute("CREATE TEMP TABLE ptab AS SELECT * FROM pt_df")
    cache = C.DCW / "official" / "stg" / "cache"
    pc = (cache / "phase_cycles.parquet").as_posix()
    con.execute(f"""CREATE TEMP TABLE cyc AS
        SELECT lower(DeviceId) AS dev, Phase::INT AS phase,
               epoch_ms(green_start) / 1000.0 - {e0} AS t_g,
               epoch_ms(yellow_start) / 1000.0 - {e0} AS t_y,
               epoch_ms(next_green) / 1000.0 - {e0} AS t_next
        FROM read_parquet('{pc}') WHERE next_green IS NOT NULL
          AND lower(DeviceId) IN (SELECT DISTINCT dev FROM ad)""")
    con.execute(SQL_BASE.format(e0=e0, t0=t0, t1=t1,
                                di=(cache / "det_intervals.parquet").as_posix()))
    spd = con.sql(SQL_SPEED).df()
    spd["dur"] = np.where(spd.n_low >= 20, spd.dur50_low, spd.dur50_all)
    spd = spd.set_index(["dev", "det"])
    M = con.sql(SQL_MATCH.format(since=since, toy=toy)).df()
    C.log(f"[h{hours:g} s{start:g}] matches {len(M):,}; speed rows {len(spd)}")
    # --- tau per candidate pair (low-flow hours when >= 30 matches there, else all)
    rows = []
    for (dev, da, db), g in M.groupby(["dev", "da", "db"]):
        lw = g.low.to_numpy()
        use = lw if lw.sum() >= 30 else np.ones(len(g), bool)
        r = robust_tau(g.dt.to_numpy()[use], g.dur_a.to_numpy()[use])
        r.update(dev=dev, da=da, db=db, used_low=lw.sum() >= 30)
        rows.append(r)
    TAU = pd.DataFrame(rows)
    cand = cand.merge(TAU, on=["dev", "da", "db"], how="left")
    cand["v_self"] = [speed_of(spd.loc[(d, b)].to_dict() if (d, b) in spd.index else None, f)
                      for d, b, f in zip(cand.dev, cand.db, cand.function)]
    # phase-level speeds (every stop-bar zone on the phase)
    sbz2 = sbz.copy()
    sbz2["v"] = [speed_of(spd.loc[(d, int(b))].to_dict() if (d, int(b)) in spd.index else None, f)
                 for d, b, f in zip(sbz2.dev, sbz2.detector, sbz2.function)]
    vc = (sbz2[sbz2.function.isin(["Count", "Yellow_Red"])].groupby(["dev", "phase"]).v.median()
          .rename("v_phase_count"))
    vp = sbz2[sbz2.function == "Presence"].groupby(["dev", "phase"]).v.median().rename(
        "v_phase_pres")
    out = []
    for key, g in cand.groupby(["dev", "da"]):
        out.append(pick(g, "model", g.p_same >= 0.5))
        out.append(pick(g, "print", g.lane_print == 1))
    R = pd.DataFrame([o for o in out if o is not None])
    W = R.pivot_table(index=["dev", "da"], columns="pairing", aggfunc="first")
    W.columns = [f"{a}_{b}" for a, b in W.columns]
    W = W.reset_index().rename(columns={"da": "detector"})
    F = tgt.merge(W, on=["dev", "detector"], how="left")
    F = F.merge(vc.reset_index(), on=["dev", "phase"], how="left").merge(
        vp.reset_index(), on=["dev", "phase"], how="left")
    adv = spd.reset_index().rename(columns={"det": "detector"})
    adv["v_adv"] = [speed_of(r, "Count") for r in adv.to_dict("records")]
    F = F.merge(adv[["dev", "detector", "v_adv", "pulse", "dur"]].rename(
        columns={"pulse": "adv_pulse", "dur": "adv_dur"}), on=["dev", "detector"], how="left")
    for pr in ("model", "print"):
        vsrc = np.select([F[f"v_self_{pr}"].notna(), F.v_phase_count.notna(),
                          F.v_phase_pres.notna(), F.v_adv.notna()],
                         ["partner", "phase_count", "phase_presence", "advance_self"], "40mph")
        v = (F[f"v_self_{pr}"].fillna(F.v_phase_count).fillna(F.v_phase_pres)
             .fillna(F.v_adv).fillna(V40))
        F[f"vsrc_{pr}"] = np.where(F[f"tau_{pr}"].notna(), vsrc, None)
        F[f"v_{pr}"] = v
        edge = F[f"role_{pr}"].map(EDGE).fillna(0)
        d = F[f"tau_{pr}"] * v + edge
        F[f"d_phys_{pr}"] = np.where(d < CLIP0, 0.0, d)
        d40 = F[f"tau_{pr}"] * V40 + edge
        F[f"d_40mph_{pr}"] = np.where(d40 < CLIP0, 0.0, d40)
        dv = F[f"d_veh_{pr}"] + edge
        F[f"d_veh_{pr}"] = np.where(dv < CLIP0, 0.0, dv)
    F["hours"], F["start"] = hours, start
    tag = "" if (since, toy) == (10, 10) else f"_g{since:g}_{toy:g}"
    fn = OUT / f"sb3_phys_h{hours:g}_s{start:g}{tag}.parquet"
    keep = [c for c in F.columns if F[c].dtype != object or c in
            ("dev", "function", "technology", "dist_kind", "role_model", "role_print",
             "vsrc_model", "vsrc_print", "lane_model", "lane_print")]
    F[keep].to_parquet(fn, index=False)
    C.log(f"wrote {fn}: {len(F)} targets; model-paired tau {F.tau_model.notna().sum()}, "
          f"print-paired tau {F.tau_print.notna().sum()}")
    return F


def pick(g: pd.DataFrame, tag: str, same: pd.Series):
    """Best accepted partner: same lane first (Count > YR > Presence, then P(same));
    otherwise the best accepted other-lane zone, flagged."""
    g = g.assign(rank=g.function.map(ROLE_RANK), ok=g.tau.notna(), same=same.fillna(False))
    acc = g[g.ok].sort_values(["same", "rank", "p_same"], ascending=[False, True, False])
    if acc.empty:
        bs = g.sort_values(["same", "n_match"], ascending=False).iloc[0]
        return {"dev": g.dev.iloc[0], "da": g.da.iloc[0], "pairing": tag,
                "lane": "none", "n_cand": len(g), "n_match": bs.n_match,
                "n_in": bs.n_in, "pk_strength": bs.pk_strength,
                "n_same_count": int((g.same & (g.function == "Count")).sum())}
    b = acc.iloc[0]
    return {"dev": b.dev, "da": b.da, "pairing": tag, "partner": int(b.db),
            "role": b.function, "lane": "same" if b.same else "other",
            "p_same": b.p_same, "lane_truth": b.lane_print, "tau": b.tau,
            "tau_raw": b.tau_med_raw, "n_match": b.n_match, "n_in": b.n_in,
            "pk_strength": b.pk_strength, "v_self": b.v_self, "n_cand": len(g),
            "d_veh": b.get("d_veh", np.nan), "v_veh": b.get("v_veh", np.nan),
            "n_same_count": int((g.same & (g.function == "Count")).sum())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=66)
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--since", type=float, default=10, help="advance ON >= s into green")
    ap.add_argument("--toy", type=float, default=10, help="advance ON >= s before yellow")
    a = ap.parse_args()
    run(a.hours, a.start, a.since, a.toy)
