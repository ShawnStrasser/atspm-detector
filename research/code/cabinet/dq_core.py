"""Detector data-quality (DQ) checks against the hi-res events (cabinet_print_guide.md §5).

Input: one row per detector to check --
    DeviceId, detector, phase, location (stopbar|advance|mid|bike|other),
    lane_index (int or NA = unknown), lanes_spanned (int, default 1), technology (optional),
    function (optional; "Presence" switches on the peak-undercount test of pair_sb)
Output (`run_signal` / `run`): one row per input detector with the raw metrics, one score
per check in [0, 1] (NaN = not applicable), `dq_score`, `flags`, `reasons` and
`suspect_config_or_health`.

Data: de-duplicated events, allowed codes only, `Parameter <= 64` on 81/82 (AGENTS.md).
Source, newest first (`period="auto"`): the staging event cache (`official/stg/cache/events`,
Sept 2026, ~66 h), the Dec-2024 event cache (`cache/events`, 72 h), then the raw staging /
Dec-2024 pulls filtered on the fly.  Only 5-/10-/1-minute bin counts and a correlogram
histogram ever leave DuckDB.

Off-peak / peak come from the signal's OWN volume: 10-min bins ranked by the summed ON
count of the input vehicle detectors (bins with no events at all = comms gap, excluded);
off = rank <= OFF_Q, peak = rank >= PEAK_Q.

Checks (one score each; details in research/notes/24_detector_dq_toolkit.md):
  pair_sb   stop bar vs same-lane advance: 10-min counts correlate off-peak; for a
            Presence zone the stop-bar / advance ratio must not RISE at peak (it undercounts).
  span      spanning (lanes_spanned >= 2) loop vs the SUM of the single-lane advance loops
            it covers: correlate off-peak, ratio near 1 off-peak, not above it at peak.
  order     same-lane advance -> downstream lead: off-peak ON cross-correlogram peak at
            +2..8 s, or off-peak 1-min high-pass count correlation (note 17 cue d).
  excl      side-by-side advance loops (different lanes): few ONs within +-0.5 s of the
            other loop's ON, off-peak ("A fires, B stays off").
  sat       max ONs per 5 min <= SAT_PER_LANE * lanes_spanned.
  bike      bike loop: not ~zero, and far below the phase's vehicle detectors.
  dead      zero ONs over the covered span (near_dead: < 2 % of the phase's median).
  health    stuck-on (ON > STUCK_S), chatter (re-trigger < 0.3 s): ACTUATIONS only.
            Detector fault events 83-88 are never read (user ban 2026-09-28; note 88 removed the
            former 'faults' part and their use as comms-coverage evidence).

    python dq_core.py --dets dets.csv --out dq.parquet [--period auto|stg|dec]
"""
from __future__ import annotations

import argparse
import glob
import os
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
ALLOWED = (1, 7, 8, 9, 10, 11, 43, 44, 81, 82, 131, 150, 173)   # no detector fault events 83-88 (note 88)
MAXCH = 64
SOURCES = {   # name -> (kind, path template); per-device caches are already filtered + deduped
    "stg": [("dev", DCW / "official" / "stg" / "cache" / "events" / "DeviceId={d}" / "*.parquet"),
            ("raw", DCW / "data" / "staging" / "date=*" / "part_*.parquet")],
    "dec": [("dev", DCW / "cache" / "events" / "DeviceId={d}" / "*.parquet"),
            ("raw", DCW / "data" / "raw" / "Train_Dec_*_2024.parquet")],
}

# ---- thresholds: set once from presumably-good detectors (note 24), not tuned per signal
OFF_Q, PEAK_Q = 0.40, 0.85        # 10-min bin volume rank: off-peak <= .40, peak >= .85
MIN_ON = 20                       # a detector needs this many ONs for pair checks
C_OFF_OK, C_OFF_BAD = 0.6, 0.2    # 10-min count corr off-peak: score 1 above OK, 0 below BAD
DROP_OK, DROP_BAD = 1.30, 2.00    # presence: (sb/adv at peak) / (sb/adv off-peak); >1 = sb gains
SPAN_DROP_OK, SPAN_DROP_BAD = 1.10, 1.50   # spanning / sum(advance): must not gain at peak
SPAN_R_LO, SPAN_R_HI = 0.5, 1.5   # spanning / sum(advance) off-peak ratio band
LEAD_OK = np.log(1.5)             # correlogram +2..8 s peak / baseline (log)
HP_OK = 0.30                      # 1-min high-pass corr off-peak (same-lane evidence)
EXCL_OK, EXCL_BAD = 0.15, 0.35    # share of the quieter loop's off-peak ONs with the other
                                  # loop's ON within +-0.5 s (side-by-side advance loops)
SAT_PER_LANE = 150                # ONs per 5 min per lane (1,800 veh/h/lane)
BIKE_NEAR_ZERO = 3                # ONs over the whole span
BIKE_REL_BAD = 0.35               # bike ONs / median vehicle-detector ONs on the phase
NEAR_DEAD_REL = 0.02              # vehicle detector with < 2 % of its phase's median ONs
STUCK_S = 900.0
GAP_S = 120.0                     # no event of any code for this long = comms gap                   # one ON longer than this = stuck
CHATTER_GAP, CHATTER_FRAC = 0.3, 0.30   # share of OFF->ON gaps < 0.3 s
SUSPECT = 0.5                     # dq_score below this -> suspect_config_or_health
VEH = ("stopbar", "advance", "mid")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def connect(threads: int = 6) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET memory_limit='10GB'")
    con.execute(f"SET threads={threads}")
    con.execute("SET preserve_insertion_order=false")
    (DCW / "tmp").mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{(DCW / 'tmp').as_posix()}'")
    return con


def _lin(x, bad, ok):
    """1 at/above ok, 0 at/below bad, linear between (works for bad > ok too)."""
    return float(np.clip((x - bad) / (ok - bad), 0, 1)) if np.isfinite(x) else np.nan


# ----------------------------------------------------------------------------- data
def load_events(con, device_id: str, period: str = "auto") -> str | None:
    """Materialise the device's filtered, de-duplicated events as temp table `ev`
    (ts TIMESTAMP, e SMALLINT, p SMALLINT). Returns the source used, or None."""
    order = ["stg", "dec"] if period == "auto" else [period]
    codes = ",".join(map(str, ALLOWED))
    for per in order:
        for kind, tmpl in SOURCES[per]:
            if kind == "dev":
                g = Path(str(tmpl).format(d=device_id))
                if not g.parent.is_dir():
                    g = Path(str(tmpl).format(d=device_id.lower()))
                    if not g.parent.is_dir():
                        continue
                src, where = g.as_posix(), "TRUE"
            else:
                if not glob.glob(str(tmpl)):
                    continue
                src, where = tmpl.as_posix(), f"lower(DeviceId) = '{device_id.lower()}'"
            con.execute(f"""CREATE OR REPLACE TEMP TABLE ev AS
                SELECT DISTINCT Timestamp AS ts, EventId::SMALLINT AS e, Parameter::SMALLINT AS p
                FROM read_parquet('{src}') WHERE {where} AND EventId IN ({codes})
                  AND NOT (EventId IN (81, 82) AND Parameter > {MAXCH})""")
            if con.sql("SELECT count(*) FROM ev").fetchone()[0] > 0:
                return f"{per}:{kind}"
    return None


SQL_IV = """
CREATE OR REPLACE TEMP TABLE iv AS
WITH d AS (
  SELECT p AS det, ts, e, LEAD(ts) OVER w AS nts, LEAD(e) OVER w AS ne,
         LAG(ts) OVER w AS pts, LAG(e) OVER w AS pe
  FROM ev WHERE e IN (81, 82)
  WINDOW w AS (PARTITION BY p ORDER BY ts, CASE WHEN e = 82 THEN 0 ELSE 1 END))
SELECT det, epoch_ms(ts) / 1000.0 - {e0} AS t,
       CASE WHEN ne = 81 THEN epoch_ms(nts - ts) / 1000.0
            WHEN nts IS NULL THEN epoch_ms(TIMESTAMP '{t1}' - ts) / 1000.0 END AS dur,
       CASE WHEN pe = 81 THEN epoch_ms(ts - pts) / 1000.0 END AS gap
FROM d WHERE e = 82
"""

SQL_CORR = """
WITH a AS (SELECT o.det AS da, pr.db, o.t, floor(o.t / 20)::INT AS bk
           FROM ivl o JOIN prs pr ON o.det = pr.da WHERE o.lv = 0),
b AS (SELECT det AS db, t, unnest([floor(t / 20)::INT - 1, floor(t / 20)::INT,
                                   floor(t / 20)::INT + 1]) AS bk
      FROM ivl WHERE det IN (SELECT db FROM prs))
SELECT a.da, a.db, floor((b.t - a.t) / 0.5)::INT AS lb, count(*) AS n
FROM a JOIN b USING (db, bk) WHERE abs(b.t - a.t) < 15 GROUP BY ALL
"""


def _corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 6 or x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


# ------------------------------------------------------------------------ one signal
def run_signal(con, device_id: str, dets: pd.DataFrame, period: str = "auto") -> pd.DataFrame:
    t_start = time.time()
    d = dets.copy()
    d["detector"] = d.detector.astype(int)
    d["location"] = d.location.fillna("other").str.lower()
    d["function"] = d["function"] if "function" in d else pd.NA   # optional; Presence tightens pair_sb
    d["lanes_spanned"] = pd.to_numeric(d.get("lanes_spanned", 1), errors="coerce").fillna(1).astype(int)
    d["lane_index"] = pd.to_numeric(d.get("lane_index"), errors="coerce") if "lane_index" in d else np.nan
    d = d.drop_duplicates("detector").set_index("detector")
    src = load_events(con, device_id, period)
    if src is None:
        out = d.reset_index()
        out["source"], out["dq_score"], out["flags"] = None, np.nan, "no_data"
        out["suspect_config_or_health"], out["secs"] = False, 0.0
        return out
    e0, t0, t1 = con.sql("SELECT epoch_ms(date_trunc('hour', min(ts))) / 1000.0, "
                         "date_trunc('hour', min(ts)), max(ts) FROM ev").fetchone()
    con.execute(SQL_IV.format(e0=e0, t1=t1))
    # ---- coverage: 5-min bins with any event at all
    cov = con.sql(f"SELECT DISTINCT floor((epoch_ms(ts) / 1000.0 - {e0}) / 300)::INT AS b "
                  "FROM ev").df().b.to_numpy()
    cov_h = len(cov) / 12
    # ---- per-detector stats + 5-min, 1-min counts
    # comms gaps (no event of any code for > GAP_S): an ON spanning one is not "stuck"
    con.execute(f"""CREATE OR REPLACE TEMP TABLE gaps AS SELECT * FROM (
        SELECT epoch_ms(ts) / 1000.0 - {e0} AS g0,
               epoch_ms(LEAD(ts) OVER (ORDER BY ts) - ts) / 1000.0 AS gl
        FROM (SELECT DISTINCT ts FROM ev)) WHERE gl > {GAP_S}""")
    con.execute("""UPDATE iv SET dur = NULL WHERE dur > 60 AND EXISTS
                   (SELECT 1 FROM gaps WHERE g0 >= iv.t AND g0 < iv.t + iv.dur)""")
    st = con.sql(f"""SELECT det, count(*) AS n_on, max(dur) AS dur_max,
                  sum(dur) AS occ_s, avg((dur <= 0.15)::INT) AS pulse_frac,
                  avg((gap < {CHATTER_GAP})::INT) AS chatter_frac
                  FROM iv GROUP BY det""").df().set_index("det")
    c1 = con.sql("SELECT det, floor(t / 60)::INT AS m, count(*) AS n FROM iv GROUP BY ALL").df()
    nm = int(np.ceil((pd.Timestamp(t1) - pd.Timestamp(t0)).total_seconds() / 60)) + 1
    M1 = np.zeros((nm, 65))
    M1[c1.m.clip(0, nm - 1).to_numpy(), c1.det.to_numpy()] = c1.n.to_numpy()
    covm = np.zeros(nm, bool)
    for b in cov:                           # minutes of covered 5-min bins
        covm[b * 5:min(nm, b * 5 + 5)] = True
    M5 = M1[: nm // 5 * 5].reshape(-1, 5, 65).sum(1)
    cov5 = covm[: nm // 5 * 5].reshape(-1, 5).all(1)
    M10 = M1[: nm // 10 * 10].reshape(-1, 10, 65).sum(1)
    cov10 = covm[: nm // 10 * 10].reshape(-1, 10).all(1)
    veh = [x for x in d.index if d.at[x, "location"] in VEH and 1 <= x <= MAXCH]
    vol = M10[:, veh].sum(1) if veh else M10.sum(1)
    rk = pd.Series(vol[cov10]).rank(pct=True).to_numpy()
    lv10 = np.full(len(vol), -1)
    lv10[cov10] = np.where(rk <= OFF_Q, 0, np.where(rk >= PEAK_Q, 2, 1))
    off, peak = lv10 == 0, lv10 == 2
    # 1-min high-pass (minus 15-min block mean), 15-min block level by volume rank
    n15 = nm // 15
    B = M1[: n15 * 15].reshape(n15, 15, 65)
    R = (B - B.mean(1, keepdims=True)).reshape(-1, 65)
    cov15 = covm[: n15 * 15].reshape(n15, 15).all(1)
    v15 = B[:, :, veh].sum((1, 2)) if veh else B.sum((1, 2))
    rk15 = pd.Series(v15[cov15]).rank(pct=True).to_numpy()
    lv15 = np.full(n15, -1)
    lv15[cov15] = np.where(rk15 <= 1 / 3, 0, np.where(rk15 >= 2 / 3, 2, 1))
    hp_off_mask = np.repeat(lv15 == 0, 15)

    res = d.copy()
    res["source"], res["span_h"], res["cov_h"] = src, round((pd.Timestamp(t1) - pd.Timestamp(t0)).total_seconds() / 3600, 1), round(cov_h, 1)
    res = res.join(st, how="left")
    for c in ("n_on", "occ_s"):
        res[c] = res[c].fillna(0)
    res["on_per_h"] = res.n_on / max(cov_h, 1e-9)
    ok_det = [x for x in res.index if 1 <= x <= MAXCH]
    res["max5"] = [M5[cov5, x].max() if x in ok_det and cov5.any() else np.nan for x in res.index]
    res["p99_5"] = [np.quantile(M5[cov5, x], .99) if x in ok_det and cov5.any() else np.nan
                    for x in res.index]
    res["occ_frac"] = res.occ_s / max(cov_h * 3600, 1e-9)
    # info: channels carrying the identical ON stream (one loop wired to several inputs)
    same = {}
    for x in ok_det:
        if res.at[x, "n_on"] >= MIN_ON:
            same.setdefault(M1[:, x].tobytes(), []).append(x)
    res["same_stream_as"] = ""
    for grp in same.values():
        for x in grp:
            res.at[x, "same_stream_as"] = ",".join(str(y) for y in grp if y != x)
    act = set(res.index[res.n_on >= MIN_ON])

    # ------------------------------------------------------------- pairs
    pairs = []          # (a, b, kind); a upstream / partner, b tested detector
    for ph, g in res.groupby("phase"):
        adv1 = [x for x in g.index if g.at[x, "location"] == "advance" and g.at[x, "lanes_spanned"] == 1 and x in act]
        sb = [x for x in g.index if g.at[x, "location"] == "stopbar" and g.at[x, "lanes_spanned"] == 1 and x in act]
        span = [x for x in g.index if g.at[x, "location"] in VEH and g.at[x, "lanes_spanned"] >= 2 and x in act]
        for s in sb:
            li = g.at[s, "lane_index"]
            # same lane when lanes are known, else every single-lane advance (best one wins)
            cand = [a for a in adv1 if pd.notna(li) and g.at[a, "lane_index"] == li] or adv1
            for a in cand:
                pairs.append((a, s, "adv_sb"))
        for i, a in enumerate(adv1):
            for b in adv1[i + 1:]:
                la, lb = g.at[a, "lane_index"], g.at[b, "lane_index"]
                if pd.isna(la) or pd.isna(lb) or la != lb:
                    pairs.append((a, b, "adv_adv"))
        for s in span:
            li, ns = g.at[s, "lane_index"], g.at[s, "lanes_spanned"]
            cov_adv = [a for a in adv1 if pd.notna(li) and pd.notna(g.at[a, "lane_index"])
                       and li <= g.at[a, "lane_index"] < li + ns] or adv1
            if len(cov_adv) >= 2:
                pairs.append((tuple(cov_adv), s, "span"))
            for a in cov_adv:
                if g.at[s, "location"] != "advance":
                    pairs.append((a, s, "adv_down"))
    P = []
    # correlogram for single-detector pairs
    sp = [(a, b) for a, b, k in pairs if k in ("adv_sb", "adv_adv", "adv_down")]
    H = {}
    if sp:
        lvb = pd.DataFrame({"b10": np.arange(len(lv10)), "lv": lv10})
        con.register("lvb_df", lvb)
        con.execute("CREATE OR REPLACE TEMP TABLE ivl AS SELECT i.det, i.t, l.lv FROM iv i "
                    "JOIN lvb_df l ON floor(i.t / 600)::INT = l.b10")
        con.register("prs_df", pd.DataFrame(sorted(set(sp)), columns=["da", "db"]))
        con.execute("CREATE OR REPLACE TEMP TABLE prs AS SELECT da::SMALLINT da, db::SMALLINT db FROM prs_df")
        cg = con.sql(SQL_CORR).df()
        noff = con.sql("SELECT det, count(*) n FROM ivl WHERE lv = 0 GROUP BY det").df().set_index("det").n
        lags = np.arange(-30, 30)
        mid = lags * 0.5 + 0.25
        for (a, b), grp in cg.groupby(["da", "db"]):
            h = grp.set_index("lb").n.reindex(lags, fill_value=0).to_numpy(float)
            H[(a, b)] = (h, float(noff.get(a, 0)))
    else:
        noff = pd.Series(dtype=float)
    for a, b, kind in pairs:
        r = {"a": a, "b": b, "kind": kind}
        xa = M10[:, list(a)].sum(1) if isinstance(a, tuple) else M10[:, a]
        xb = M10[:, b]
        r["c_off"] = _corr(xa[off], xb[off])
        r["c_peak"] = _corr(xa[peak], xb[peak])
        so, sp_ = xa[off].sum(), xa[peak].sum()
        r["r_off"] = xb[off].sum() / so if so else np.nan
        r["r_peak"] = xb[peak].sum() / sp_ if sp_ else np.nan
        r["drop"] = r["r_peak"] / r["r_off"] if r["r_off"] else np.nan
        if not isinstance(a, tuple):
            m = hp_off_mask[: len(R)] & covm[: len(R)]
            r["hp_off"] = _corr(R[m, a], R[m, b])
            h, na = H.get((a, b), (np.zeros(60), 0.0))
            base = max(h[np.abs(mid) >= 10].mean(), 0.5)
            fwd = h[(mid >= 1.5) & (mid <= 8.5)]
            run = fwd[:-1] + fwd[1:]
            r["lead_ratio"] = float(np.log(max(run.max(), 0.5) / (2 * base)))
            r["lead_s"] = float(1.5 + 0.5 * run.argmax() + 0.5)
            bwd = h[(mid <= -1.5) & (mid >= -8.5)]
            r["lead_asym"] = float(np.log((fwd.sum() + 1) / (bwd.sum() + 1)))
            z = h[np.abs(mid) < 0.5].sum()
            r["excl_ratio"] = float(z / (2 * base))
            nmin = min(na, float(noff.get(b, 0)))
            r["coinc_frac"] = float(z / nmin) if nmin else np.nan
        P.append(r)
    P = pd.DataFrame(P)

    # ------------------------------------------------------------- scores
    for c in ("s_pair_sb", "s_span", "s_order", "s_excl", "s_sat", "s_bike", "s_dead", "s_health"):
        res[c] = np.nan
    reasons = {x: [] for x in res.index}
    info = {x: {} for x in res.index}

    def put(x, col, s, why, inf=None):
        if x not in res.index:
            return
        cur = res.at[x, col]
        if np.isnan(cur) or s > cur:          # best supporting partner wins
            res.at[x, col] = s
            if inf is not None:
                info[x][col] = inf
        if s < 0.5 and why:
            reasons[x].append(why)

    if len(P):
        for _, r in P.iterrows():
            if r.kind == "adv_sb":
                pres = str(res.at[r.b, "function"]).lower() == "presence"
                sc = min(_lin(r.c_off, C_OFF_BAD, C_OFF_OK),
                         _lin(r["drop"], DROP_BAD, DROP_OK) if pres and np.isfinite(r["drop"]) else 1.0)
                put(r.b, "s_pair_sb", sc, None, f"adv{r.a}:c{r.c_off:.2f},drop{r['drop']:.2f}")
            if r.kind == "span":
                sc = min(_lin(r.c_off, C_OFF_BAD, C_OFF_OK),
                         1.0 if SPAN_R_LO <= r.r_off <= SPAN_R_HI else 0.3,
                         _lin(r["drop"], SPAN_DROP_BAD, SPAN_DROP_OK) if np.isfinite(r["drop"]) else 1.0)
                put(r.b, "s_span", sc, None, f"sum{list(r.a)}:c{r.c_off:.2f},r{r.r_off:.2f},drop{r['drop']:.2f}")
            if r.kind in ("adv_sb", "adv_down"):
                sc = max(1.0 if (r.lead_ratio >= LEAD_OK and r.lead_asym > 0) else 0.0,
                         _lin(r.hp_off, 0.0, HP_OK))
                put(r.b, "s_order", sc, None, f"adv{r.a}:lead{r.lead_ratio:.2f}@{r.lead_s:.1f}s,hp{r.hp_off:.2f}")
                put(r.a, "s_order", sc, None, f"->{r.b}")
            if r.kind == "adv_adv":
                sc = _lin(r.coinc_frac, EXCL_BAD, EXCL_OK) if np.isfinite(r.coinc_frac) else np.nan
                if np.isfinite(sc):
                    for x, y in ((r.a, r.b), (r.b, r.a)):
                        cur = res.at[x, "s_excl"]
                        res.at[x, "s_excl"] = sc if np.isnan(cur) else min(cur, sc)  # worst neighbour
                        if sc < 0.5:
                            reasons[x].append(f"excl: {r.coinc_frac:.0%} of ONs coincide with adv{y} "
                                              f"(x{r.excl_ratio:.0f} chance)")
        # a failing best pair -> reason
        for x in res.index:
            if res.at[x, "s_pair_sb"] < 0.5:
                reasons[x].append(f"pair_sb {info[x].get('s_pair_sb', '')}")
            if res.at[x, "s_span"] < 0.5:
                reasons[x].append(f"span {info[x].get('s_span', '')}")
            if res.at[x, "s_order"] < 0.5 and res.at[x, "location"] != "advance":
                reasons[x].append(f"order {info[x].get('s_order', '')}")
    # single-detector checks
    vehmed = {ph: np.median(res.loc[(res.phase == ph) & res.location.isin(VEH) & (res.n_on > 0), "n_on"])
              if ((res.phase == ph) & res.location.isin(VEH) & (res.n_on > 0)).any() else np.nan
              for ph in res.phase.unique()}
    for x in res.index:
        n, loc = res.at[x, "n_on"], res.at[x, "location"]
        vm = vehmed.get(res.at[x, "phase"], np.nan)
        low = loc in VEH and np.isfinite(vm) and 0 < n < NEAR_DEAD_REL * vm
        res.at[x, "s_dead"] = 0.0 if n == 0 else (0.3 if low else 1.0)
        if n == 0:
            reasons[x].append("dead: 0 ONs")
        elif low:
            reasons[x].append(f"near-dead: {n:.0f} ONs vs phase median {vm:.0f}")
        if n > 0 and loc in VEH:
            cap = SAT_PER_LANE * max(res.at[x, "lanes_spanned"], 1)
            res.at[x, "s_sat"] = 1.0 if res.at[x, "max5"] <= cap else 0.3
            if res.at[x, "max5"] > cap:
                reasons[x].append(f"sat: max {res.at[x, 'max5']:.0f}/5min > {cap}")
        if loc == "bike":
            vm = vehmed.get(res.at[x, "phase"], np.nan)
            rel = n / vm if vm and np.isfinite(vm) else np.nan
            res.at[x, "bike_rel"] = rel
            if n < BIKE_NEAR_ZERO:
                res.at[x, "s_bike"] = 0.4
                reasons[x].append(f"bike: near-zero ({n:.0f} ONs)")
            elif np.isfinite(rel) and rel > BIKE_REL_BAD:
                res.at[x, "s_bike"] = 0.2
                reasons[x].append(f"bike: counts like a vehicle loop ({rel:.2f} of phase median)")
            else:
                res.at[x, "s_bike"] = 1.0
        if n > 0:
            h = []
            if res.at[x, "dur_max"] > STUCK_S:
                h.append(f"stuck-on {res.at[x, 'dur_max'] / 60:.0f} min")
            if res.at[x, "chatter_frac"] > CHATTER_FRAC and n >= MIN_ON:
                h.append(f"chatter {res.at[x, 'chatter_frac']:.0%} gaps<{CHATTER_GAP}s")
            res.at[x, "s_health"] = 0.2 if h else 1.0
            if h:
                reasons[x].append("health: " + "; ".join(h))
    S = res[["s_pair_sb", "s_span", "s_order", "s_excl", "s_sat", "s_bike", "s_dead", "s_health"]]
    # hard checks multiply, soft (pair/lane) checks averaged so one weak cue is not fatal
    hard = S[["s_dead", "s_health", "s_sat", "s_bike"]].fillna(1).prod(1)
    soft = S[["s_pair_sb", "s_span", "s_order", "s_excl"]].mean(1, skipna=True).fillna(1)
    res["dq_score"] = (hard * (0.4 + 0.6 * soft)).clip(0, 1).round(3)
    flags = []
    for x in res.index:
        f = []
        if res.at[x, "s_dead"] == 0:
            f.append("dead")
        elif res.at[x, "s_dead"] < 0.5:
            f.append("near_dead")
        if res.at[x, "s_health"] < 0.5:
            f.append("health")
        for c in ("pair_sb", "span", "order", "excl", "sat", "bike"):
            if res.at[x, f"s_{c}"] < 0.5 and not (c == "order" and res.at[x, "location"] == "advance"):
                f.append(c)
        flags.append(",".join(f))
    res["flags"] = flags
    res["reasons"] = ["; ".join(dict.fromkeys(reasons[x])) for x in res.index]
    res["checks_n"] = S.notna().sum(1)
    res["suspect_config_or_health"] = res.dq_score < SUSPECT
    res["secs"] = round(time.time() - t_start, 1)
    out = res.reset_index().rename(columns={"index": "detector"})
    out.attrs["pairs"] = P
    return out


def run(dets: pd.DataFrame, period: str = "auto", threads: int = 6) -> pd.DataFrame:
    con = connect(threads)
    outs, pairs = [], []
    for dev, g in dets.groupby("DeviceId", sort=False):
        o = run_signal(con, dev, g.drop(columns="DeviceId"), period)
        o.insert(0, "DeviceId", dev)
        outs.append(o)
        p = o.attrs.pop("pairs", None)  # attrs holding DataFrames break pd.concat's attrs comparison
        if p is not None and len(p):
            pairs.append(p.assign(DeviceId=dev))
        log(f"{dev}: {len(o)} detectors, source {o.source.iloc[0]}, {o.secs.iloc[0]}s, "
            f"suspect {int(o.suspect_config_or_health.sum())}")
    out = pd.concat(outs, ignore_index=True)
    out.attrs["pairs"] = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dets", required=True, help="csv/parquet: DeviceId, detector, phase, location, "
                    "lane_index, lanes_spanned, technology")
    ap.add_argument("--out", required=True, help="output parquet (per detector); pairs -> *_pairs.parquet")
    ap.add_argument("--period", default="auto", choices=["auto", "stg", "dec"])
    a = ap.parse_args()
    p = Path(a.dets)
    dets = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, dtype={"DeviceId": str})
    out = run(dets, a.period)
    out.to_parquet(a.out, index=False)
    pr = out.attrs["pairs"]
    if len(pr):
        pr["a"] = pr.a.astype(str)
        pr.to_parquet(Path(a.out).with_name(Path(a.out).stem + "_pairs.parquet"), index=False)
    log(f"wrote {a.out}: {len(out)} detectors, {int(out.suspect_config_or_health.sum())} suspect")


if __name__ == "__main__":
    main()
