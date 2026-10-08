"""Label-behaviour validation (data cleansing of training labels): does each labelled detector behave like its label?

Version 2 (note 29), rebuilt after the traffic engineer's review of version 1. The rules may use the cabinet print
(technology, lanes, zone codes): they clean labels, they are never model inputs. Each rule is one number, calibrated
ONCE on the most trustworthy rows (print_high at complete_high signals, not unusual_layout, not dead), per technology
where it matters, and kept only if it separates (AUC >= .75). A rule that cannot be evaluated never fails a detector.

    python label_check.py                 stats (cached) -> thresholds (frozen once calibrated) -> v3 columns + review
    python label_check.py --restat        recompute the per-detector statistics
    python label_check.py --recalibrate   recompute the thresholds (only when a rule's definition changes)
    python label_check.py --write         also write the columns into the v3 table (cab_final.py step 7 does this)
    python label_check.py --spotcheck     v2-layout spot-check (the v3 follow-up round is label_spotcheck.py)

Version 3 (engineer's spot-check of v2, 2026-09-28): 'label wrong' (validated = fail, removes the label) is kept apart
from 'detector / configuration problem' (label kept; column field_issue; review/field_issues_for_staff.xlsx):
presence zone set to pulse, count zone not in pulse where vehicles stop (when order / occupancy confirm Count), advance
loop missing detections. Health from ACTUATIONS only (stuck on / chatter / saturation / erratic card) -> validated =
unhealthy; detector fault events 83-88 are never used. Protected-permissive (FYA) left-turn phases (FYA events 32 in the
raw pull, or a detector also calling the through phase in the timing: cleansing only) are exempt from the presence
holds-through-red and count off-in-red rules. New cross-check: of a stop-bar Presence / Count pair the Presence is more
occupied and counts fewer (support = a passed check; the reverse = fail); advance-before-presence = a passed check.
Version 4 (engineer's spot-check of v3, 2026-09-29): FYA events are a clue, not the definition -- permissive left
turns are also found from behaviour (vehicles waiting on red leave before green, leave_red_stats); a stop-bar zone in
a right-turn-only lane (print lane R, right turn on red) is exempt from the red-time rules like FYA; a configuration
field issue (pulse-set presence, count zone not in pulse where vehicles stop) -> validated = misconfigured (label kept,
NOT trained); an advance loop missing vehicles, the user's 'unhealthy' rulings (final_rulings.csv) and the health
agent's 'bad' detectors (HEALTH_STATUS, when present) -> unhealthy.

Data: each signal's NEWEST window (Sept 2026 staging cache, else Dec 2024), full span; de-duplicated detector
intervals (channels 1-64), phase cycles, coordination state (event 131). Aggregated in DuckDB; only per-detector /
per-pair sums reach pandas. Off-peak / peak = 10-min bins ranked by the signal's own labelled vehicle volume
(<= .40 / >= .85). Quiet = free-running 10-min bins (no coordination pattern) where the signal has >= 2 h of them.

Rules (engineer, 2026-09-23/24):
  presence_holds_red     Presence holds the call through red at peak: ON when its green starts (peak cycles).
                         A radar presence with a programmed delay may undercount: never failed for low counts.
  phase_presence_missing SIGNAL level: the print shows a stop-bar presence zone on a phase but no channel of that
                         phase holds a call through red at peak (field misconfiguration).
  count_off_in_red / count_short_ons   stop-bar Count: off during red, short (pulse) ONs.
  advance_red_arrivals   Advance gets arrivals on red as on green, judged ONLY in free-running periods
                         (coordination may bunch arrivals into green: a tendency, not a rule).
  advance_leads_lane     off-peak single vehicles light the advance first, then the SAME-LANE stop-bar zone 1.5-9 s
                         later: share of the stop-bar zone's off-peak ONs preceded by the advance, minus chance.
  advance_loop_counts    LOOPS only: a loop advance counts >= its same-lane stop-bar loop off-peak. Radar is not
                         count-compared (occlusion: a far zone undercounts, a spanning zone counts < the lane sum).
  loop_not_count         the engineer's hard rule: a loop is never a stop-bar Count.
  mid_tracks_sum, bike_far_below   as version 1.
  info only: presence undercounts the same-lane advance at peak; radar Advance held long like an ETA zone.
  removed: Yellow_Red off in red (YR exists to catch vehicles entering on yellow / red), YR first-car signature
           (too weak), count grows at peak, the any-stop-bar-zone lead.
Same-lane partners: the print's lane_index / lanes_spanned; else a one-lane phase; else the note-30 pair model
(P(same lane) >= .5, lp3_pairs_oof); else (Advance only) every stop-bar zone of the phase, said so in the text.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from cab_common import CAB, DC_WORK, REPO, STORE, forbid_locked_mode

forbid_locked_mode(__name__)  # training-label / model-facing: never on the locked-label store

V3 = REPO / "research/labels/function_labels_v3.parquet"
STATS = CAB / "label_check_stats.parquet"
PAIRS = CAB / "label_check_pairs.parquet"
HEALTH = CAB / "label_check_health.parquet"
THRESH = CAB / "label_check_thresholds.json"
CARDS = CAB / "card_channels.parquet"   # note 34 (VD audit): both outputs of one detector card dead / erratic
LANE_P = DC_WORK / "trackA/lp3_pairs_oof.parquet"
REVIEW = (CAB / "lists") if STORE else REPO / "review"  # note 80: a store copy never writes review/
PERIODS = {"stg": DC_WORK / "official/stg/cache", "dec": DC_WORK / "cache"}
VEHICLE = ["Advance", "Presence", "Count", "Yellow_Red", "Mid"]
STOPBAR = ["Presence", "Count", "Yellow_Red"]
ATSPM = ["Advance", "Presence", "Count", "Yellow_Red"]
TECHS = ["loop", "radar", "video"]
OFF_Q, PEAK_Q = 0.40, 0.85          # as dq_core
MIN_ON = 100                        # ONs for any behaviour statistic
MIN_RED_ON = 20                     # red-start ONs for hold_frac
MIN_CYC_PEAK = 20                   # peak cycles for on_green_peak
MIN_QUIET_H = 2.0                   # free-running hours for the red-arrival rule
MIN_QUIET_ON = 50                   # ONs in the free-running bins
MAX_CYC_S = 900                     # a "cycle" longer than this spans a gap / flash: dropped
SAMPLE_Y = 1500                     # off-peak ONs of the stop-bar zone sampled for the order statistic
LEAD_WIN = (1.5, 9.0)               # advance ON this long before the stop-bar ON = "leads"
CHANCE_WIN = (37.5, 45.0)           # same-width window far away = chance
BUSY_CHANCE = 0.60                  # partner so busy that chance explains everything: order not checkable
P_SAME = 0.5                        # note-30 pair model: same lane
TOL = 0.03                          # threshold = 3rd / 97th percentile of the trusted rows of the class
AUC_KEEP = 0.75
AUC_TIE = 0.01                      # a later candidate statistic must beat the first by this much
MIN_TRUST = 20
MIN_TECH = 50                       # trusted rows of a technology for its own limit (else the pooled one)
DB_MEM, DB_THREADS = "8GB", (4 if STORE else 6)  # note 80: <= 4 threads on the shared machine
# v3 (engineer's spot-check of v2, 2026-09-28)
EXTRA = CAB / "label_check_extra.parquet"   # per detector: occupancy, pulse share (actuations only)
PPLT = CAB / "label_check_pplt.parquet"     # protected-permissive (FYA) phases: cleansing only, never a model input
RAW_STG_OTHER = DC_WORK / "data/staging_other"   # Sept 2026 pull of the non-model event codes (incl. FYA 32/33)
RAW_DEC = DC_WORK / "data/raw"                   # Dec 2024 pull (all codes)
PULSE_S, PULSE_SHARE = 0.3, 0.80    # a zone whose ONs are <= 0.3 s this often is set to pulse
OCC_RATIO = 1.2                     # "more occupied" in the order / occupancy role check
FYA_MIN = 5                         # FYA begin-permissive events (32) on the phase in the window
THROUGH = {1: 2, 3: 4, 5: 6, 7: 8}  # left-turn phase -> its through phase (agency convention; timing 'additional call')
# v4 (engineer's spot-check of v3, 2026-09-29)
# FYA events are logged only at SOME signals: a clue, not the definition. Permissive left turns are also found from
# behaviour: vehicles that WAIT on a detector of the phase during its red (ON >= LEAVE_DUR s) and then LEAVE before
# the phase turns green (the detector goes off > 2 s before green and stays off). On the phase's STOP-BAR zone
# (label Presence / Count / Yellow_Red, not a right-turn lane: an upstream loop also "empties" as the queue moves up)
# with the most such red waits (>= LEAVE_MIN): share leaving >= LEAVE_THR. AUC .95 for FYA-event phases vs left-turn
# phases without FYA events at signals that do log them (228 / 48 phases; recall .89, 8 % of those flagged at .20).
LEAVE_DUR, LEAVE_MIN, LEAVE_THR = 3.0, 20, 0.20
RIGHT_LANE = {"R"}                  # print lane_type of a right-turn-only lane: right turn on red (permissive)
RULINGS = CAB / "final_rulings.csv"  # rule 'unhealthy' = the user's own health verdicts (label right, detector poor)
HEALTH_STATUS = DC_WORK / "health" / "health_status.parquet"  # health agent (optional): DeviceId|DeviceName, detector, status


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def connect():
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{DB_MEM}'")
    con.execute(f"SET threads={DB_THREADS}")
    con.execute("SET preserve_insertion_order=false")
    (DC_WORK / "tmp").mkdir(exist_ok=True)
    con.execute(f"SET temp_directory='{(DC_WORK / 'tmp').as_posix()}'")
    return con


def tech_of(x) -> str:
    x = x if isinstance(x, str) else ""
    return "radar" if x.startswith("radar") else x if x in TECHS else "unknown"


def targets(v3: pd.DataFrame) -> pd.DataFrame:
    t = v3[["DeviceId", "DeviceName", "detector", "function", "phase_target", "phase_diagram", "lane_index",
            "lanes_spanned", "technology", "n_lanes_phase"]].copy()
    t["DeviceId"] = t.DeviceId.str.lower()
    ph = pd.to_numeric(t.phase_target.astype("string").str.extract(r"^P(\d+)$")[0], errors="coerce")
    t["p"] = ph.fillna(pd.to_numeric(t.phase_diagram, errors="coerce")).astype("Int16")
    for c in ("lane_index", "lanes_spanned", "n_lanes_phase"):
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t["tech"] = t.technology.map(tech_of)
    return t


# ============================================================================ statistics (DuckDB)
SQL_DET = """SELECT dev, det, count(*) AS n_on, median(dur) AS dur_med, avg((dur <= 0.15)::INT) AS pulse_frac
             FROM iv GROUP BY 1, 2"""
SQL_BINS = "SELECT dev, det, floor(t_on / 600)::BIGINT AS b, count(*) AS n FROM iv GROUP BY ALL"
SQL_BINFREE = """SELECT l.dev, l.b, CASE WHEN c.is_coord IS NULL THEN -1 WHEN c.is_coord THEN 0 ELSE 1 END AS free
                 FROM lvl0 l ASOF LEFT JOIN coord c ON l.dev = c.dev AND l.b * 600 + 300 >= c.ts"""
SQL_COLOUR = """
WITH o AS (SELECT iv.*, tg.p FROM iv JOIN tg USING (dev, det)),
j AS (SELECT o.*, c.gs, c.ge, c.ng FROM o ASOF JOIN cyc c ON o.dev = c.dev AND o.p = c.p AND o.t_on >= c.gs),
k AS (SELECT j.*, t_on < ge AS in_g, t_on >= ge AS in_r, l.lv, l.free
      FROM j LEFT JOIN lvl l ON j.dev = l.dev AND floor(j.t_on / 600)::BIGINT = l.b WHERE t_on < ng)
SELECT dev, det, count(*) AS n_col, sum(in_g::INT) AS n_g, sum(in_r::INT) AS n_r,
       avg(CASE WHEN in_r THEN (t_off >= ng - 0.05)::INT END) AS hold_frac,
       avg(CASE WHEN in_r AND lv = 2 THEN (t_off >= ng - 0.05)::INT END) AS hold_frac_peak,
       count(*) FILTER (in_r AND lv = 2) AS n_r_peak,
       sum(greatest(least(t_off, ng) - greatest(t_on, ge), 0)) AS ov_red,
       count(*) FILTER (in_g AND free = 1) AS n_g_q, count(*) FILTER (in_r AND free = 1) AS n_r_q
FROM k GROUP BY 1, 2
"""
SQL_REDSECS = """SELECT c.dev, c.p, sum(ng - ge) AS red_s, sum(ge - gs) AS green_s,
                        sum(CASE WHEN l.free = 1 THEN ng - ge END) AS red_s_q, sum(CASE WHEN l.free = 1 THEN ge - gs END) AS green_s_q,
                        count(*) FILTER (l.free = 1) AS n_cyc_q
                 FROM cyc c LEFT JOIN lvl l ON c.dev = l.dev AND floor(c.gs / 600)::BIGINT = l.b GROUP BY 1, 2"""
SQL_ONGREEN = """
WITH cg AS (SELECT c.dev, c.p, c.gs, l.lv FROM cyc c JOIN lvl l ON c.dev = l.dev AND floor(c.gs / 600)::BIGINT = l.b),
dg AS (SELECT tg.dev, tg.det, cg.gs, cg.lv FROM tg JOIN cg ON tg.dev = cg.dev AND tg.p = cg.p),
a AS (SELECT dg.*, iv.t_off FROM dg ASOF LEFT JOIN iv ON dg.dev = iv.dev AND dg.det = iv.det AND dg.gs >= iv.t_on)
SELECT dev, det, avg((coalesce(t_off, 0) > gs)::INT) AS on_green,
       avg(CASE WHEN lv = 2 THEN (coalesce(t_off, 0) > gs)::INT END) AS on_green_peak,
       count(*) FILTER (lv = 2) AS n_cyc_peak
FROM a GROUP BY 1, 2
"""
SQL_ORDER = f"""
WITH q AS (SELECT prs.dev, prs.x, prs.y, ys.t FROM prs JOIN ys ON prs.dev = ys.dev AND prs.y = ys.det),
a AS (SELECT q.*, xs.t_on AS tx1 FROM q ASOF LEFT JOIN xs ON q.dev = xs.dev AND q.x = xs.det AND q.t - {LEAD_WIN[0]} >= xs.t_on),
b AS (SELECT a.*, xs.t_on AS tx2 FROM a ASOF LEFT JOIN xs ON a.dev = xs.dev AND a.x = xs.det AND a.t - {CHANCE_WIN[0]} >= xs.t_on)
SELECT dev, x, y, count(*) AS n_y,
       avg(coalesce(t - tx1 <= {LEAD_WIN[1]}, false)::INT) AS sh_before,
       avg(coalesce(t - tx2 <= {CHANCE_WIN[1]}, false)::INT) AS sh_chance,
       median(CASE WHEN t - tx1 <= {LEAD_WIN[1]} THEN t - tx1 END) AS lag_med
FROM b GROUP BY ALL
"""


def _corr(x, y):
    if len(x) < 6 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _levels(M: np.ndarray, veh_cols: list[int]):
    """10-min bin levels from the signal's own vehicle volume: 0 off-peak, 2 peak, 1 between, -1 no data."""
    cov = M.sum(1) > 0
    vol = M[:, veh_cols].sum(1) if veh_cols else M.sum(1)
    lv = np.full(len(vol), -1)
    rk = pd.Series(vol[cov]).rank(pct=True).to_numpy()
    lv[cov] = np.where(rk <= OFF_Q, 0, np.where(rk >= PEAK_Q, 2, 1))
    return lv


def _lane_rel(a, b, n_lanes, p_same):
    """Same-lane relation of two detectors of a phase: (same: bool | None, how)."""
    if pd.notna(a.lane_index) and pd.notna(b.lane_index):
        sa = a.lanes_spanned if pd.notna(a.lanes_spanned) and a.lanes_spanned >= 1 else 1
        sb = b.lanes_spanned if pd.notna(b.lanes_spanned) and b.lanes_spanned >= 1 else 1
        ov = a.lane_index <= b.lane_index + sb - 1 and b.lane_index <= a.lane_index + sa - 1
        return bool(ov), "print"
    if n_lanes == 1:
        return True, "one-lane phase"
    if p_same is not None and np.isfinite(p_same):
        return bool(p_same >= P_SAME), f"data (P={p_same:.2f})"
    return None, "unknown"


def build_pairs(t: pd.DataFrame, M_by_dev: dict, lanep: dict) -> pd.DataFrame:
    """Ordered pairs (x -> y) on one phase, y a stop-bar zone, with their lane relation and count statistics."""
    rows = []
    for (dev, p), h in t[t.function.isin(VEHICLE) & t.p.notna() & (t.n_on >= MIN_ON)].groupby(["DeviceId", "p"]):
        if dev not in M_by_dev or len(h) < 2:
            continue
        M, lv = M_by_dev[dev]
        off, peak = lv == 0, lv == 2
        nl = h.n_lanes_phase.dropna()
        n_lanes = int(nl.mode().iloc[0]) if len(nl) else None
        recs = list(h.itertuples(index=False))
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                a, b = recs[i], recs[j]
                if a.function not in STOPBAR and b.function not in STOPBAR:
                    continue
                lo, hi = sorted((int(a.detector), int(b.detector)))
                same, how = _lane_rel(a, b, n_lanes, lanep.get((dev, f"P{int(p)}", lo, hi)))
                c_off = _corr(M[off, int(a.detector)], M[off, int(b.detector)])
                for x, y in ((a, b), (b, a)):  # both orders: the reverse is the swap control of the order rule
                    xo, yo = M[off, int(x.detector)].sum(), M[off, int(y.detector)].sum()
                    xp, yp = M[peak, int(x.detector)].sum(), M[peak, int(y.detector)].sum()
                    r_off = xo / yo if yo else np.nan
                    xa, ya = M[:, int(x.detector)].sum(), M[:, int(y.detector)].sum()
                    r_peak = xp / yp if yp else np.nan
                    rows.append(dict(DeviceId=dev, p=int(p), x=int(x.detector), y=int(y.detector), fx=x.function,
                                     fy=y.function, tx=x.tech, ty=y.tech, same=same, lane_how=how, c_off=c_off,
                                     x_off=xo, y_off=yo, x_peak=xp, y_peak=yp, r_off=r_off, x_all=xa, y_all=ya,
                                     r_all=xa / ya if ya else np.nan,
                                     # (y / x at peak) / (y / x off-peak): < 1 = the stop-bar zone y falls behind
                                     y_drop=(r_off / r_peak) if (r_peak and np.isfinite(r_peak)) else np.nan))
    return pd.DataFrame(rows)


def span_stats(t: pd.DataFrame, M_by_dev: dict) -> pd.DataFrame:
    """Mid / Advance vs the SUM of the other single-lane advance loops of its phase (off-peak)."""
    rows = []
    for (dev, p), h in t[t.p.notna() & (t.n_on >= MIN_ON)].groupby(["DeviceId", "p"]):
        if dev not in M_by_dev:
            continue
        M, lv = M_by_dev[dev]
        off = lv == 0
        h = h.set_index("detector")
        adv1 = h[(h.function == "Advance") & (h.lanes_spanned.isna() | (h.lanes_spanned <= 1))]
        for x, r in h[h.function.isin(["Mid", "Advance"])].iterrows():
            oth = adv1.drop(index=x, errors="ignore")
            if pd.notna(r.lane_index) and pd.notna(r.lanes_spanned):
                inl = oth[oth.lane_index.between(r.lane_index, r.lane_index + r.lanes_spanned - 1)]
                oth = inl if len(inl) >= 2 else oth
            if len(oth) < 2:
                continue
            s = M[:, [int(i) for i in oth.index]].sum(1)
            rows.append(dict(DeviceId=dev, detector=int(x), span_c_off=_corr(s[off], M[off, int(x)]),
                             span_with=",".join(str(int(i)) for i in oth.index),
                             span_x_off=M[off, int(x)].sum(), span_sum_off=s[off].sum()))
    return pd.DataFrame(rows)


def compute_stats(v3: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = targets(v3)
    lp = pd.read_parquet(LANE_P) if LANE_P.exists() else pd.DataFrame(columns=["DeviceId", "target", "da", "db", "p_print"])
    lanep = {(d.lower(), tg, int(a), int(b)): float(pp) for d, tg, a, b, pp in
             zip(lp.DeviceId, lp.target, lp.da, lp.db, lp.p_print)}
    con = connect()
    have = {per: set(con.sql(f"SELECT DISTINCT lower(DeviceId) AS d FROM '{(d / 'det_intervals.parquet').as_posix()}'")
                     .df().d) for per, d in PERIODS.items()}
    t["period"] = np.where(t.DeviceId.isin(have["stg"]), "stg", np.where(t.DeviceId.isin(have["dec"]), "dec", None))
    parts, pparts = [], []
    for per, d in PERIODS.items():
        tp = t[t.period == per].copy()
        if tp.empty:
            continue
        tw = time.time()
        con.register("devs_df", pd.DataFrame({"dev": tp.DeviceId.unique()}))
        con.execute(f"""CREATE OR REPLACE TEMP TABLE iv AS
            SELECT lower(DeviceId) AS dev, Detector::SMALLINT AS det, epoch_ms(t_on) / 1000.0 AS t_on,
                   epoch_ms(t_off) / 1000.0 AS t_off, dur::DOUBLE AS dur
            FROM read_parquet('{(d / 'det_intervals.parquet').as_posix()}')
            WHERE lower(DeviceId) IN (SELECT dev FROM devs_df) AND Detector BETWEEN 1 AND 64 AND dur IS NOT NULL""")
        con.execute(f"""CREATE OR REPLACE TEMP TABLE cyc AS
            SELECT lower(DeviceId) AS dev, Phase::SMALLINT AS p, epoch_ms(green_start) / 1000.0 AS gs,
                   epoch_ms(coalesce(yellow_start, red_start, next_green)) / 1000.0 AS ge,
                   epoch_ms(next_green) / 1000.0 AS ng
            FROM read_parquet('{(d / 'phase_cycles.parquet').as_posix()}')
            WHERE lower(DeviceId) IN (SELECT dev FROM devs_df) AND next_green IS NOT NULL
              AND epoch_ms(next_green - green_start) / 1000.0 <= {MAX_CYC_S}""")
        con.execute(f"""CREATE OR REPLACE TEMP TABLE coord AS
            SELECT lower(DeviceId) AS dev, epoch_ms(t_start) / 1000.0 AS ts, is_coord
            FROM read_parquet('{(d / 'coord_state.parquet').as_posix()}') WHERE lower(DeviceId) IN (SELECT dev FROM devs_df)""")
        tg = tp[tp.p.notna()][["DeviceId", "detector", "p"]].rename(columns={"DeviceId": "dev", "detector": "det"})
        con.register("tg_df", tg.astype({"det": "int16", "p": "int16"}))
        con.execute("CREATE OR REPLACE TEMP TABLE tg AS SELECT dev::VARCHAR AS dev, det::SMALLINT AS det, p::SMALLINT AS p FROM tg_df")
        det = con.sql(SQL_DET).df()
        bins = con.sql(SQL_BINS).df()
        tp = tp.merge(det.rename(columns={"dev": "DeviceId", "det": "detector"}), on=["DeviceId", "detector"], how="left")
        tp["n_on"] = tp.n_on.fillna(0)
        # 10-min levels per device (own labelled vehicle volume) -> lvl table (dev, b, lv, free)
        M_by_dev, lv_rows = {}, []
        for dev, bd in bins.groupby("dev"):
            b0 = int(bd.b.min())
            M = np.zeros((int(bd.b.max()) - b0 + 1, 65))
            M[(bd.b - b0).to_numpy(), bd.det.to_numpy()] = bd.n.to_numpy()
            g = tp[(tp.DeviceId == dev) & tp.function.isin(VEHICLE) & tp.detector.between(1, 64)]
            lv = _levels(M, [int(x) for x in g.detector])
            M_by_dev[dev] = (M, lv)
            lv_rows.append(pd.DataFrame({"dev": dev, "b": np.arange(len(lv)) + b0, "lv": lv}))
        lvl0 = pd.concat(lv_rows, ignore_index=True)
        con.register("lvl0", lvl0)
        fr = con.sql(SQL_BINFREE).df()
        lvl = lvl0.merge(fr, on=["dev", "b"], how="left")
        lvl["free"] = lvl.free.fillna(-1).astype(int)
        lvl.loc[lvl.lv < 0, "free"] = -1
        con.register("lvl_df", lvl)
        con.execute("CREATE OR REPLACE TEMP TABLE lvl AS SELECT dev, b::BIGINT AS b, lv::INT AS lv, free::INT AS free FROM lvl_df")
        qh = lvl[lvl.free == 1].groupby("dev").size().rename("quiet_h") / 6
        col = con.sql(SQL_COLOUR).df()
        red = con.sql(SQL_REDSECS).df()
        ong = con.sql(SQL_ONGREEN).df()
        log(f"{per}: {tp.DeviceId.nunique()} signals, det {len(det)}, colour {len(col)}, on-green {len(ong)} "
            f"({time.time() - tw:.0f}s)")
        R = lambda x: x.rename(columns={"dev": "DeviceId", "det": "detector"})
        s = tp.merge(R(col), on=["DeviceId", "detector"], how="left").merge(R(ong), on=["DeviceId", "detector"], how="left")
        s = s.merge(R(red).astype({"p": "Int16"}), on=["DeviceId", "p"], how="left")
        s = s.merge(qh.reset_index().rename(columns={"dev": "DeviceId"}), on="DeviceId", how="left")
        s["quiet_h"] = s.quiet_h.fillna(0)
        # per-detector off-peak / peak counts
        offc, peakc = [], []
        for r in s.itertuples(index=False):
            M, lv = M_by_dev.get(r.DeviceId, (None, None))
            ok = M is not None and 1 <= r.detector <= 64
            offc.append(M[lv == 0, int(r.detector)].sum() if ok else np.nan)
            peakc.append(M[lv == 2, int(r.detector)].sum() if ok else np.nan)
        s["n_off"], s["n_peak"] = offc, peakc
        # pairs + order statistic
        P = build_pairs(s, M_by_dev, lanep)
        if len(P):
            P["period"] = per
            keep = P.same.eq(True) | P.same.isna()
            prs = P[keep][["DeviceId", "x", "y"]].drop_duplicates().rename(columns={"DeviceId": "dev"})
            con.register("prs_df", prs.astype({"x": "int16", "y": "int16"}))
            con.execute("CREATE OR REPLACE TEMP TABLE prs AS SELECT dev::VARCHAR AS dev, x::SMALLINT AS x, y::SMALLINT AS y FROM prs_df")
            con.execute(f"""CREATE OR REPLACE TEMP TABLE ys AS
                SELECT iv.dev, iv.det, iv.t_on AS t FROM iv JOIN lvl l ON iv.dev = l.dev AND floor(iv.t_on / 600)::BIGINT = l.b
                WHERE l.lv = 0 AND (iv.dev, iv.det) IN (SELECT DISTINCT dev, y FROM prs)
                QUALIFY row_number() OVER (PARTITION BY iv.dev, iv.det ORDER BY hash(iv.t_on)) <= {SAMPLE_Y}""")
            con.execute("""CREATE OR REPLACE TEMP TABLE xs AS SELECT dev, det, t_on FROM iv
                           WHERE (dev, det) IN (SELECT DISTINCT dev, x FROM prs)""")
            O = con.sql(SQL_ORDER).df().rename(columns={"dev": "DeviceId"})
            P = P.merge(O.astype({"x": int, "y": int}), on=["DeviceId", "x", "y"], how="left")
            log(f"{per}: pairs {len(P)}, order {len(O)} ({time.time() - tw:.0f}s)")
            pparts.append(P)
        sp = span_stats(s, M_by_dev)
        if len(sp):
            s = s.merge(sp, on=["DeviceId", "detector"], how="left")
        parts.append(s)
    s = pd.concat(parts + [t[t.period.isna()]], ignore_index=True)
    P = pd.concat(pparts, ignore_index=True) if pparts else pd.DataFrame()
    s["n_on"] = s.n_on.fillna(0)
    rs = s.red_s / (s.red_s + s.green_s)
    s["red_occ"] = s.ov_red / s.red_s
    s["red_arr"] = (s.n_r / (s.n_g + s.n_r)) / rs
    rq = s.red_s_q / (s.red_s_q + s.green_s_q)
    s["red_share_q"] = rq
    s["red_arr_q"] = ((s.n_r_q / (s.n_g_q + s.n_r_q)) / rq).where(
        (s.quiet_h >= MIN_QUIET_H) & ((s.n_g_q + s.n_r_q) >= MIN_QUIET_ON))
    s["hold_frac"] = s.hold_frac.where(s.n_r >= MIN_RED_ON)
    s["hold_frac_peak"] = s.hold_frac_peak.where(s.n_r_peak >= MIN_RED_ON)
    s["on_green_peak"] = s.on_green_peak.where(s.n_cyc_peak >= MIN_CYC_PEAK)
    veh = s[s.function.isin(VEHICLE) & (s.n_on > 0)].groupby(["DeviceId", "p"]).n_on.median().rename("phase_veh_med")
    s = s.merge(veh.reset_index(), on=["DeviceId", "p"], how="left")
    s["bike_rel"] = s.n_on / s.phase_veh_med
    few = s.n_on < MIN_ON
    for c in ("hold_frac", "hold_frac_peak", "red_occ", "red_arr", "red_arr_q", "dur_med", "pulse_frac", "on_green", "on_green_peak"):
        s.loc[few, c] = np.nan
    s = s.merge(pair_features(s, P), on=["DeviceId", "detector"], how="left")
    return s, P


def extra_stats(stats: pd.DataFrame) -> pd.DataFrame:
    """Per detector, newest window (the period compute_stats chose): occupancy (share of the window ON), share of
    ONs <= PULSE_S (pulse setting), 90th-percentile ON. Actuations only."""
    con = connect()
    out = []
    per = stats.drop_duplicates("DeviceId").set_index("DeviceId").period.dropna()
    for p, d in PERIODS.items():
        devs = per.index[per == p]
        if not len(devs):
            continue
        con.register("devs_df", pd.DataFrame({"dev": list(devs)}))
        out.append(con.sql(f"""
            WITH iv AS (SELECT lower(DeviceId) AS dev, Detector::SMALLINT AS det, epoch_ms(t_on) / 1000.0 AS t_on,
                               epoch_ms(t_off) / 1000.0 AS t_off, dur::DOUBLE AS dur
                        FROM read_parquet('{(d / 'det_intervals.parquet').as_posix()}')
                        WHERE lower(DeviceId) IN (SELECT dev FROM devs_df) AND Detector BETWEEN 1 AND 64
                          AND dur IS NOT NULL),
            sp AS (SELECT dev, max(t_off) - min(t_on) AS span_s FROM iv GROUP BY 1)
            SELECT iv.dev AS DeviceId, det AS detector, sum(dur) / any_value(span_s) AS occ,
                   avg((dur <= {PULSE_S})::INT) AS short_frac, quantile_cont(dur, 0.9) AS dur_p90,
                   any_value(span_s) / 3600 AS span_h
            FROM iv JOIN sp USING (dev) GROUP BY 1, 2""").df())
    return pd.concat(out, ignore_index=True)


def pplt_table(v3: pd.DataFrame) -> pd.DataFrame:
    """Protected-permissive (FYA) left-turn phases, for CLEANSING only (never a model input): the phase's FYA
    begin-permissive events (32, parameter = FYA # = the left-turn phase) in the raw pulls, or the timing: a detector
    of left-turn phase p that also calls p's through phase (additional call p -> p+1)."""
    con = connect()
    ev = []
    for src, glob in (("stg", (RAW_STG_OTHER / "*/*.parquet").as_posix()),
                      ("dec", (RAW_DEC / "Train_Dec_*.parquet").as_posix())):
        try:
            ev.append(con.sql(f"""SELECT lower(DeviceId) AS sid, Parameter::INT AS p, count(*) AS n_fya, '{src}' AS win
                                  FROM read_parquet('{glob}') WHERE EventId = 32 GROUP BY 1, 2""").df())
        except duckdb.Error as e:  # a pull that is not on this machine
            log(f"pplt: no raw events for {src}: {e}")
    E = pd.concat(ev, ignore_index=True) if ev else pd.DataFrame(columns=["sid", "p", "n_fya", "win"])
    E = E.sort_values("n_fya", ascending=False).drop_duplicates(["sid", "p"])
    t = targets(v3)
    ac = v3.additional_call_phases.astype("string").fillna("")
    t["calls_through"] = [str(THROUGH.get(int(p), -1)) in [x.strip() for x in a.split(",")] if pd.notna(p) else False
                          for p, a in zip(t.p, ac)]
    C = t[t.calls_through].groupby(["DeviceId", "p"]).detector.apply(lambda x: ",".join(map(str, sorted(x))))
    C = C.reset_index().rename(columns={"DeviceId": "sid", "detector": "cfg_dets"})
    C["p"] = C.p.astype(int)
    T = E.merge(C, on=["sid", "p"], how="outer")
    B = leave_red_stats(t.assign(lane_type=v3.lane_type), con)
    T = T.merge(B, on=["sid", "p"], how="outer")
    # left-turn phases: the agency's odd phases, or a phase whose print lanes are all left-turn lanes
    lt = v3.assign(sid=v3.DeviceId.str.lower(), p=t.p)
    lt = lt[lt.p.notna() & lt.lane_type.notna()].groupby(["sid", "p"]).lane_type.agg(lambda s: set(s) == {"L"})
    left_print = {(a, int(b)) for (a, b), v in lt.items() if v}
    T["left"] = T.p.isin(list(THROUGH)) | pd.Series([(a, int(b)) in left_print for a, b in zip(T.sid, T.p)],
                                                      index=T.index)
    T = T[T.left & T.sid.isin(set(t.DeviceId))]   # training-store signals only
    ev = T.n_fya.fillna(0).ge(FYA_MIN)
    beh = T.leave_red.fillna(0).ge(LEAVE_THR) & T.n_wait.fillna(0).ge(LEAVE_MIN)
    T["pplt"] = ev | T.cfg_dets.notna() | beh
    T["by_events"], T["by_timing"], T["by_behaviour"] = ev, T.cfg_dets.notna(), beh
    T["how"] = ["; ".join(x for x in (
        f"FYA events: {int(n)} begin-permissive" if e else "",
        f"timing: det {c} also call phase {THROUGH.get(int(p), '?')}" if isinstance(c, str) else "",
        f"behaviour: {lr:.0%} of {int(nw)} red waits on det {int(dd)} left before green" if b else "") if x)
        for n, e, c, p, b, lr, nw, dd in zip(T.n_fya.fillna(0), ev, T.cfg_dets, T.p, beh, T.leave_red.fillna(0),
                                              T.n_wait.fillna(0), T.leave_det.fillna(0))]
    return T[T.pplt].reset_index(drop=True)


def leave_red_stats(t: pd.DataFrame, con) -> pd.DataFrame:
    """Per (signal, phase), newest window: of the vehicles that wait on a detector of the phase during its red
    (ON starting in red, lasting >= LEAVE_DUR s), the share that LEAVE before the phase's green (OFF > 2 s before
    green and no new ON within 3 s). Taken on the phase's detector with the most red waits. Phases from the timing
    target (cleansing only)."""
    out = []
    sb = t.function.isin(STOPBAR) & ~t.lane_type.astype("string").isin(list(RIGHT_LANE)).fillna(False)
    tg = t[t.p.notna() & sb][["DeviceId", "detector", "p"]].rename(columns={"DeviceId": "dev", "detector": "det"})
    for per, d in PERIODS.items():
        con.register("lr_tg", tg.astype({"det": "int16", "p": "int16"}))
        try:
            out.append(con.sql(f"""
            WITH iv AS (SELECT lower(DeviceId) AS dev, Detector::SMALLINT AS det, epoch_ms(t_on) / 1000.0 AS t_on,
                               epoch_ms(t_off) / 1000.0 AS t_off, dur::DOUBLE AS dur
                        FROM read_parquet('{(d / 'det_intervals.parquet').as_posix()}')
                        WHERE lower(DeviceId) IN (SELECT DISTINCT dev FROM lr_tg) AND Detector BETWEEN 1 AND 64
                          AND dur IS NOT NULL),
            iv2 AS (SELECT *, lead(t_on) OVER (PARTITION BY dev, det ORDER BY t_on) AS nxt FROM iv),
            cyc AS (SELECT lower(DeviceId) AS dev, Phase::SMALLINT AS p, epoch_ms(green_start) / 1000.0 AS gs,
                           epoch_ms(coalesce(yellow_start, red_start, next_green)) / 1000.0 AS ge,
                           epoch_ms(next_green) / 1000.0 AS ng
                    FROM read_parquet('{(d / 'phase_cycles.parquet').as_posix()}')
                    WHERE lower(DeviceId) IN (SELECT DISTINCT dev FROM lr_tg) AND next_green IS NOT NULL
                      AND epoch_ms(next_green - green_start) / 1000.0 <= {MAX_CYC_S}),
            o AS (SELECT iv2.*, lr_tg.p FROM iv2 JOIN lr_tg USING (dev, det)),
            j AS (SELECT o.*, c.ge, c.ng FROM o ASOF JOIN cyc c ON o.dev = c.dev AND o.p = c.p AND o.t_on >= c.gs)
            SELECT dev AS sid, det, p, '{per}' AS win, count(*) AS n_wait,
                   avg((t_off < ng - 2 AND coalesce(nxt, 1e12) > least(t_off + 3, ng))::INT) AS leave_red
            FROM j WHERE t_on >= ge AND t_on < ng AND dur >= {LEAVE_DUR} GROUP BY ALL""").df())
        except duckdb.Error as e:
            log(f"leave_red: {per}: {e}")
    if not out:
        return pd.DataFrame(columns=["sid", "p", "n_wait", "leave_red", "leave_det"])
    B = pd.concat(out, ignore_index=True)
    # newest window per signal (stg over dec), then the phase's detector with the most red waits
    B["_w"] = B.win.map({"stg": 0, "dec": 1})
    B = B[B._w == B.groupby("sid")._w.transform("min")]
    B = B.sort_values(["n_wait", "det"], ascending=[False, True]).groupby(["sid", "p"]).head(1)
    B = B.rename(columns={"det": "leave_det"})[["sid", "p", "n_wait", "leave_red", "leave_det"]]
    B["p"] = B.p.astype(int)
    return B


def role_check(d: pd.DataFrame, P: pd.DataFrame, lead_thr: float) -> pd.DataFrame:
    """The engineer's order + occupancy cross-check (2026-09-28), label-for-label on one phase (same lane or lane
    unknown): of a stop-bar Presence / Count pair the Presence is MORE occupied and counts FEWER; an advance fires
    first and the presence zone behind it is more occupied. Support confirms the label; the reverse on a Presence /
    Count pair (the Count looks like the presence zone) contradicts both. A zone set to pulse is left out."""
    if P.empty:
        return pd.DataFrame(columns=["sid", "detector", "role_support", "role_contra", "role_pair"])
    key = list(zip(d.sid, d.detector.astype(int)))
    occ = dict(zip(key, d.occ.astype(float)))
    n = dict(zip(key, d.n_on.astype(float)))
    pulse = dict(zip(key, d.short_frac.astype(float).fillna(0) >= PULSE_SHARE))
    Q = P[P.same.ne(False)].copy()
    Q["lead"] = _lead_stat(Q)
    sup, con_, pair = {}, {}, set()
    for r in Q.itertuples(index=False):
        a, b = (r.DeviceId, int(r.x)), (r.DeviceId, int(r.y))
        oa, ob, na, nb = occ.get(a, np.nan), occ.get(b, np.nan), n.get(a, np.nan), n.get(b, np.nan)
        if not (np.isfinite(oa) and np.isfinite(ob)) or min(na, nb) < MIN_ON or pulse.get(a) or pulse.get(b):
            continue
        if r.fx == "Presence" and r.fy == "Count":   # each unordered pair once (x = the Presence)
            num = f"det {r.x} (Presence) ON {oa:.1%} of the time, {int(na):,} actuations; det {r.y} (Count) {ob:.1%}, {int(nb):,}"
            if oa >= OCC_RATIO * ob and na < nb:
                sup.setdefault(a, []).append(num + f": presence more occupied, fewer counts (as expected)")
                sup.setdefault(b, []).append(num + f": presence more occupied, fewer counts (as expected)")
                pair |= {a, b}
            elif ob >= OCC_RATIO * oa and nb < na:
                t = num + ": the Count zone is the more occupied one with fewer counts - the two look swapped"
                con_.setdefault(a, []).append(t)
                con_.setdefault(b, []).append(t)
        elif r.fx == "Advance" and r.fy == "Presence" and np.isfinite(_num(r.lead)) and r.lead >= lead_thr \
                and ob >= OCC_RATIO * oa:
            sup.setdefault(b, []).append(
                f"det {r.x} (Advance) fires first ({r.sh_before:.0%} of det {r.y}'s off-peak actuations 1.5-9 s after it, "
                f"chance {r.sh_chance:.0%}) and det {r.y} is more occupied ({ob:.1%} vs {oa:.1%}): downstream presence zone")
    ks = set(sup) | set(con_)
    return pd.DataFrame({"sid": [k[0] for k in ks], "detector": [k[1] for k in ks],
                         "role_support": [" | ".join(sup[k]) if k in sup else None for k in ks],
                         "role_contra": [" | ".join(con_[k]) if k in con_ else None for k in ks],
                         # the Presence / Count pair evidence is specific (trusted pairs: support 53 %, swapped 0 %);
                         # advance-before-presence is not (fires on 90 % of trusted Presence but 31 % of Count):
                         # only the pair evidence may excuse a failed behaviour rule
                         "role_pair": [k in pair for k in ks]})


# ============================================================================ per-detector features from the pairs
def _lead_stat(P: pd.DataFrame) -> pd.Series:
    """Order statistic of a pair x -> y: share of y's off-peak ONs preceded 1.5-9 s by x, minus chance."""
    v = P.sh_before - P.sh_chance
    return v.where((P.n_y >= 30) & (P.sh_chance <= BUSY_CHANCE))


def pair_features(s: pd.DataFrame, P: pd.DataFrame) -> pd.DataFrame:
    cols = ["DeviceId", "detector"]
    if P.empty:
        return pd.DataFrame(columns=cols)
    P = P.copy()
    P["lead"] = _lead_stat(P)
    out = []
    # order: x (any vehicle class) -> its best same-lane stop-bar zone; lane-unknown partners only if none known
    Q = P[P.lead.notna() & P.fy.isin(STOPBAR)].copy()
    Q["known"] = Q.same.eq(True)
    Q = Q[Q.known | Q.same.isna()]
    Q = Q.sort_values(["known", "lead"], ascending=False)
    best = Q.groupby(["DeviceId", "x"]).head(1)
    best = best.assign(lead_known=best.known)
    o = best[["DeviceId", "x", "y", "lead", "lead_known", "sh_before", "sh_chance", "n_y", "lag_med", "lane_how"]]
    o = o.rename(columns={"x": "detector", "y": "lead_to", "sh_before": "lead_before", "sh_chance": "lead_chance",
                          "n_y": "lead_n", "lag_med": "lead_lag", "lane_how": "lead_how"})
    # busy partner(s) only -> remembered for the text
    busy = P[P.fy.isin(STOPBAR) & (P.sh_chance > BUSY_CHANCE)].groupby(["DeviceId", "x"]).size().rename("lead_busy")
    o = o.merge(busy.reset_index().rename(columns={"x": "detector"}), on=cols, how="outer")
    out.append(o)
    # loops: advance loop vs same-lane stop-bar Presence loop, off-peak counts (single-lane both)
    ls = s.set_index(["DeviceId", "detector"]).lanes_spanned
    L = P[(P.fx == "Advance") & (P.tx == "loop") & (P.ty == "loop") & (P.fy == "Presence") & P.same.eq(True)
          & (P.y_all >= 100)].copy()
    if len(L):
        spx = ls.reindex(list(zip(L.DeviceId, L.x))).to_numpy()
        spy = ls.reindex(list(zip(L.DeviceId, L.y))).to_numpy()
        L = L[~(spx > 1) & ~(spy > 1)]
        L = L.assign(pr=L.lane_how.eq("print")).sort_values(["pr", "c_off"], ascending=False)
        lb = L.groupby(["DeviceId", "x"]).head(1)
        lb = lb[["DeviceId", "x", "y", "r_all", "x_all", "y_all", "r_off", "lane_how"]].rename(
            columns={"x": "detector", "y": "cnt_to", "r_all": "cnt_ratio", "x_all": "cnt_x", "y_all": "cnt_y",
                     "r_off": "cnt_ratio_off", "lane_how": "cnt_how"})
        out.append(lb)
    # presence (info): falls behind its best same-lane advance at peak
    A = P[(P.fx == "Advance") & P.fy.isin(STOPBAR) & P.same.eq(True) & (P.y_off >= 30) & (P.y_peak >= 30)]
    A = A.sort_values("c_off", ascending=False).groupby(["DeviceId", "y"]).head(1)
    out.append(A[["DeviceId", "y", "x", "y_drop"]].rename(columns={"y": "detector", "x": "drop_vs", "y_drop": "sb_drop"}))
    f = out[0]
    for o in out[1:]:
        f = f.merge(o, on=cols, how="outer")
    return f


# ============================================================================ health (dq_core)
LOC = {"Presence": "stopbar", "Count": "stopbar", "Yellow_Red": "stopbar", "Advance": "advance", "Mid": "mid",
       "Bike": "bike"}


def health(v3: pd.DataFrame) -> pd.DataFrame:
    """dq_core with the v3 label's location, newest window: dead / stuck-on / chatter / saturation (no fault events)."""
    import dq_core
    _c = dq_core.connect

    def _con(threads: int = DB_THREADS):
        con = _c(min(threads, DB_THREADS))
        con.execute(f"SET memory_limit='{DB_MEM}'")
        return con
    dq_core.connect = _con
    t = targets(v3)
    nl = t.n_lanes_phase.fillna(3).to_numpy()
    ls = t.lanes_spanned.where(t.lanes_spanned.notna(), pd.Series(
        np.where(t.function.eq("Mid"), 2, np.where(t.function.eq("Yellow_Red"), nl, 1)), index=t.index))
    d = pd.DataFrame(dict(DeviceId=t.DeviceId, detector=t.detector.astype(int), phase=t.p.astype("string").fillna("?"),
                          location=t.function.map(LOC).fillna("other"), lane_index=t.lane_index.astype(float),
                          lanes_spanned=ls.astype(float), function=t.function))
    h = dq_core.run(d, period="auto", threads=DB_THREADS)
    h.attrs.pop("pairs", None)
    keep = ["DeviceId", "detector", "source", "n_on", "max5", "dur_max", "chatter_frac", "cov_h", "s_dead", "s_health", "s_sat", "lanes_spanned", "reasons"]
    h = h[[c for c in keep if c in h]].rename(columns={"source": "dq_src", "n_on": "dq_n_on", "reasons": "dq_reasons",
                                                       "lanes_spanned": "dq_lanes"})
    h["DeviceId"] = h.DeviceId.str.lower()
    return h


# ============================================================================ rules and calibration
# (id, class, engineer's rule, candidate statistics [(stat, direction)], per-technology?, kind)
# direction +1: a high value is class-like (fail BELOW the 3rd percentile of the trusted class); -1 the reverse.
# kind "class": separation = AUC class vs the other classes (trusted rows);
#      "swap":  separation = AUC of the statistic on the true direction (advance -> stop-bar) vs the swapped one.
RULES = [
    ("presence_holds_red", "Presence", "holds the call through red at peak (a vehicle arriving on red keeps it ON until green)",
     [("hold_frac_peak", 1), ("on_green_peak", 1), ("hold_frac", 1), ("on_green", 1)], True, "class"),
    ("count_off_in_red", "Count", "mostly off during red", [("red_occ", -1), ("red_arr", -1)], True, "class"),
    ("count_short_ons", "Count", "short / pulse actuations", [("dur_med", -1), ("pulse_frac", 1)], True, "class"),
    ("advance_red_arrivals", "Advance", "gets arrivals on red as on green while the signal runs free",
     [("red_arr_q", 1)], True, "class"),
    ("advance_leads_lane", "Advance", "fires 1.5-9 s before the same-lane stop-bar zone (off-peak single vehicles)",
     [("lead", 1)], True, "swap"),
    ("advance_loop_counts", "Advance", "a loop advance counts at least as much as its same-lane stop-bar loop",
     [("cnt_ratio", 1)], False, "lane"),
    ("mid_tracks_sum", "Mid", "tracks the sum of the lane-by-lane advance loops off-peak", [("span_c_off", 1)], False,
     "class"),
    ("bike_far_below", "Bike", "counts far below the vehicle detectors of its phase", [("bike_rel", -1)], False, "class"),
    ("presence_undercounts_peak", "Presence", "falls behind the same-lane advance at peak (information only)",
     [("sb_drop", -1)], False, "class"),
]
INFO_ONLY = {"presence_undercounts_peak"}


def auc(pos, neg) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = pd.Series(np.r_[pos, neg]).rank().to_numpy()
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def trusted_mask(d: pd.DataFrame) -> pd.Series:
    """High-confidence print labels at complete_high signals, not unusual_layout, not dead."""
    return (d.source.eq("print_high") & d.tier.eq("complete_high") & ~d.unusual_layout.astype(bool)
            & ~d.dead.astype(bool))


def _swap_values(P: pd.DataFrame, stat: str, tr_keys: set) -> tuple[np.ndarray, np.ndarray]:
    """Trusted same-lane (print) Advance / stop-bar pairs: the statistic in the true direction and swapped."""
    if P.empty:
        return np.array([]), np.array([])
    Q = P[P.same.eq(True) & P.lane_how.eq("print")].copy()
    Q = Q[[(a, x) in tr_keys and (a, y) in tr_keys for a, x, y in zip(Q.DeviceId, Q.x, Q.y)]]
    if stat == "lead":
        Q["v"] = _lead_stat(Q)
        fwd = Q[(Q.fx == "Advance") & Q.fy.isin(STOPBAR)].v.dropna()     # advance -> stop-bar zone
        rev = Q[(Q.fy == "Advance") & Q.fx.isin(STOPBAR)].v.dropna()     # stop-bar zone -> advance
        return fwd.to_numpy(), rev.to_numpy()
    if stat == "cnt_ratio":
        # the one-sided swap (advance >= stop-bar vs the reverse) does not separate: same-lane loops count the SAME
        # (median ratio 1.00). What separates is the pairing: |log ratio| on same-lane vs different-lane print pairs.
        L = P[P.lane_how.eq("print") & (P.fx == "Advance") & (P.fy == "Presence") & (P.tx == "loop") & (P.ty == "loop")
              & (P.y_all >= 100) & (P.x_all >= 100)]
        L = L[[(a, x) in tr_keys and (a, y) in tr_keys for a, x, y in zip(L.DeviceId, L.x, L.y)]]
        v = -np.abs(np.log(L.r_all))
        return v[L.same.eq(True)].dropna().to_numpy(), v[L.same.eq(False)].dropna().to_numpy()
    raise KeyError(stat)


def calibrate(d: pd.DataFrame, P: pd.DataFrame) -> dict:
    tm = trusted_mask(d)
    tr = d[tm]
    tr_keys = set(zip(tr.sid, tr.detector.astype(int)))
    out = {"meta": dict(date=time.strftime("%Y-%m-%d"), version=2, trusted="source print_high, tier complete_high, "
                        "not unusual_layout, not dead", n_trusted=int(len(tr)), tol=TOL, auc_keep=AUC_KEEP,
                        min_on=MIN_ON, min_trust=MIN_TRUST), "rules": {}}
    for rid, cls, text, cands, per_tech, kind in RULES:
        rr = dict(cls=cls, text=text, kind=kind, per_tech=per_tech, candidates={}, kept=False)
        best = None
        for stat, dr in cands:
            x = tr[[stat, "function", "tech"]].dropna(subset=[stat])
            vals = x[x.function == cls][stat]
            if kind == "class":
                pos, neg = x[x.function == cls][stat] * dr, x[x.function != cls][stat] * dr
            else:
                pos, neg = _swap_values(P, stat, tr_keys)
            a = auc(pos, neg)
            rr["candidates"][stat] = round(a, 3) if np.isfinite(a) else None
            # the engineer's own formulation is listed first; another candidate replaces it only if clearly better
            if len(vals) >= MIN_TRUST and np.isfinite(a) and (best is None or a > best[2] + AUC_TIE):
                best = (stat, dr, a, len(pos), len(neg), vals)
        if best:
            stat, dr, a, npos, nneg, vals = best
            thr = float(np.quantile(vals, TOL if dr > 0 else 1 - TOL))
            rr.update(stat=stat, dir=dr, auc=round(a, 3), n_class=int(len(vals)), n_other=int(nneg), threshold=thr,
                      kept=bool(a >= AUC_KEEP),
                      trusted_pass=float(((vals >= thr) if dr > 0 else (vals <= thr)).mean()))
            if kind == "class":
                x = tr[[stat, "function"]].dropna()
                rr["auc_by_class"] = {c: round(auc(x[x.function == cls][stat] * dr, x[x.function == c][stat] * dr), 3)
                                      for c in sorted(x.function.unique()) if c != cls}
            else:  # also the plain class AUC, for information
                x = tr[[stat, "function"]].dropna()
                rr["class_auc"] = round(auc(x[x.function == cls][stat] * dr, x[x.function != cls][stat] * dr), 3)
            if per_tech:
                rr["by_tech"] = {}
                for tech in TECHS:
                    xt = tr[(tr.tech == tech)][[stat, "function"]].dropna()
                    vt = xt[xt.function == cls][stat]
                    if len(vt) < MIN_TECH:
                        rr["by_tech"][tech] = dict(n_class=int(len(vt)), note="too few trusted rows: pooled limit")
                        continue
                    if kind == "class":
                        nt = xt[xt.function != cls][stat]
                        nt = nt if len(nt) >= MIN_TRUST else tr[[stat, "function"]].dropna().query("function != @cls")[stat]
                        at = auc(vt * dr, nt * dr)
                    else:
                        f, r = _swap_values(P[P.tx == tech] if len(P) else P, stat, tr_keys)
                        at = auc(f, r)
                    tt = float(np.quantile(vt, TOL if dr > 0 else 1 - TOL))
                    rr["by_tech"][tech] = dict(n_class=int(len(vt)), auc=round(at, 3) if np.isfinite(at) else None,
                                               threshold=tt, kept=bool(np.isfinite(at) and at >= AUC_KEEP),
                                               trusted_pass=float(((vt >= tt) if dr > 0 else (vt <= tt)).mean()))
        out["rules"][rid] = rr
    return out


def limit_for(c: dict, tech: str):
    """(threshold, applies) for a technology: its own limit if calibrated, pooled otherwise."""
    bt = (c.get("by_tech") or {}).get(tech)
    if bt and "threshold" in bt:
        return bt["threshold"], bt["kept"]
    return c.get("threshold"), c.get("kept", False)


# ============================================================================ plain-English text
SRC_PLAIN = {"print_high": "cabinet print (clear)", "user_ruling": "your ruling", "print_medium": "cabinet print (medium)",
             "print_low": "cabinet print (unsure)", "config": "config export", "review_round1": "your review (round 1)",
             "whole_intersection_other": "not on the print (Other by rule)"}


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def pct(x):
    return f"{x:.0%}" if np.isfinite(_num(x)) else "n/a"


def lane_txt(li, ls):
    li, ls = _num(li), _num(ls)
    if not np.isfinite(li):
        return "lane ?"
    if np.isfinite(ls) and ls > 1:
        return f"lanes {int(li)}-{int(li + ls - 1)}"
    return f"lane {int(li)}"


STUCK_S, CHATTER_FRAC = 900.0, 0.30   # dq_core's limits


def _health(r) -> tuple[list[str], str]:
    """Health from ACTUATIONS only (engineer, 2026-09-28): detector fault events 83-88 are set per practitioner in
    the timing and prove nothing, so they are never used here. Stuck on / chattering only."""
    out = []
    if _num(r.dur_max) > STUCK_S:
        out.append(f"stuck on for {_num(r.dur_max) / 60:.0f} min")
    if _num(r.chatter_frac) > CHATTER_FRAC and _num(r.n_on) >= 20:
        out.append(f"chatters ({_num(r.chatter_frac):.0%} of re-triggers within 0.3 s)")
    return out, ""


def explain(rid: str, r, thr: float, look) -> dict:
    """What the detector was compared with, both numbers, the expectation and the result, in plain words."""
    f, det = r.function, int(r.detector)
    e = dict(check=rid, compared_with="", this_value="", partner_value="", expectation="", result="")
    if rid == "presence_holds_red":
        e.update(compared_with="its own phase's red periods (peak hours)",
                 this_value=f"{pct(r.hold_frac_peak)} of {int(_num(r.n_r_peak)):,} vehicles arriving on red at peak "
                            f"kept it ON until green" + (f" (ON at {pct(r.on_green_peak)} of peak green starts)"
                                                         if np.isfinite(_num(r.on_green_peak)) else ""),
                 expectation=f"a stop-bar presence zone holds the call through red at peak: at least {pct(thr)} of "
                             f"red arrivals held until green (clear prints)",
                 result="vehicles waiting at the stop bar do not keep it on: behaves like a count / advance zone "
                        "(or the zone is not where the print shows it)")
    elif rid == "count_off_in_red":
        e.update(compared_with="its own phase's red time", this_value=f"occupied {pct(r.red_occ)} of red",
                 expectation=f"a stop-bar count zone is mostly off during red: at most {pct(thr)} of red occupied",
                 result="occupied on red: behaves like a presence zone")
    elif rid == "count_short_ons":
        e.update(compared_with="-", this_value=f"typical actuation {_num(r.dur_med):.1f} s",
                 expectation=f"a count zone gives short / pulse actuations: typical at most {_num(thr):.1f} s",
                 result="actuations too long: behaves like a presence zone")
    elif rid == "advance_red_arrivals":
        e.update(compared_with=f"its own phase's colours during {_num(r.quiet_h):.0f} h of free running",
                 this_value=f"{pct(_num(r.n_r_q) / max(_num(r.n_r_q) + _num(r.n_g_q), 1))} of actuations on red "
                            f"while red was {pct(r.red_share_q)} of the time (ratio {_num(r.red_arr_q):.2f})",
                 expectation=f"with the signal running free, an advance zone sees vehicles on red about as often as "
                             f"red lasts: ratio at least {_num(thr):.2f}",
                 result="quiet on red even while running free: behaves like a stop-bar count zone")
    elif rid == "advance_leads_lane":
        y = int(_num(r.lead_to))
        pl = look(y)
        where = "same lane" if bool(r.lead_known) else "lane unknown, best stop-bar zone of the phase"
        e.update(compared_with=f"det {y} ({pl}; {where}, {r.lead_how})",
                 this_value=f"fired 1.5-9 s before {pct(r.lead_before)} of det {y}'s off-peak actuations",
                 partner_value=f"chance level {pct(r.lead_chance)} (of {int(_num(r.lead_n)):,} actuations sampled)",
                 expectation=f"off-peak, a single vehicle lights the advance first and the same-lane stop-bar zone a "
                             f"few seconds later: at least {_num(thr) * 100:.0f} points above chance",
                 result=f"only {(_num(r.lead_before) - _num(r.lead_chance)) * 100:.0f} points above chance: the "
                        f"advance does not lead this stop-bar zone (not upstream of it, or not in its lane)")
    elif rid == "advance_loop_counts":
        y = int(_num(r.cnt_to))
        e.update(compared_with=f"det {y} ({look(y)}; same lane, {r.cnt_how})",
                 this_value=f"{int(_num(r.cnt_x)):,} actuations (off-peak ratio {_num(r.cnt_ratio_off):.2f})",
                 partner_value=f"{int(_num(r.cnt_y)):,} actuations",
                 expectation=f"a loop advance counts at least as much as its same-lane stop-bar loop: ratio at least "
                             f"{_num(thr):.2f} (clear prints)",
                 result=f"ratio {_num(r.cnt_ratio):.2f}: the advance loop undercounts the stop-bar loop (wrong lane "
                        f"pairing, a missing / weak loop, or not an advance)")
    elif rid == "mid_tracks_sum":
        e.update(compared_with=f"sum of advance loops {r.span_with}",
                 this_value=f"{int(_num(r.span_x_off)):,} off-peak", partner_value=f"{int(_num(r.span_sum_off)):,} off-peak",
                 expectation=f"a spanning (mid) loop tracks the sum of the lane-by-lane advance loops off-peak: "
                             f"10-min correlation at least {_num(thr):.2f}",
                 result=f"correlation {_num(r.span_c_off):.2f}: does not track them")
    elif rid == "bike_far_below":
        e.update(compared_with="the typical vehicle detector of its phase",
                 this_value=f"{int(_num(r.n_on)):,} actuations", partner_value=f"{int(_num(r.phase_veh_med)):,} actuations",
                 expectation=f"a bike detector counts far below the vehicle detectors: at most {pct(thr)} of them",
                 result=f"{pct(r.bike_rel)}: counts like a vehicle detector (too sensitive, or not a bike loop)")
    elif rid == "presence_undercounts_peak":
        y = int(_num(r.drop_vs))
        e.update(compared_with=f"det {y} ({look(y)})", this_value=f"peak / off-peak share vs det {y}: {_num(r.sb_drop):.2f}x",
                 expectation=f"a presence zone falls behind its same-lane advance at peak (at most {_num(thr):.2f}x)",
                 result="does not fall behind at peak (information only)")
    elif rid == "loop_not_count":
        e.update(compared_with="the print (technology)", this_value="loop",
                 expectation="a loop is never a stop-bar Count (your rule): loops count only upstream",
                 result="labelled Count on a loop" + (f"; the print reads it as {r.print_subtype}"
                                                      if isinstance(r.print_subtype, str) else ""))
    return e


# ============================================================================ validation
def ruled_unhealthy() -> dict:
    """(DeviceName or lower-case DeviceId, detector) -> plain text: detectors to keep out of training for poor health,
    label kept. Sources: the user's rulings (final_rulings.csv, rule 'unhealthy') and, when the health agent's
    per-detector output exists (HEALTH_STATUS, status 'bad'), those too. Never a model input."""
    out = {}
    if RULINGS.exists():
        R = pd.read_csv(RULINGS, dtype=str, keep_default_na=False)
        for x in R[R.rule == "unhealthy"].itertuples(index=False):
            for det in str(x.detector).split("|"):
                if det.strip():
                    out[(x.DeviceName, int(det))] = "your ruling: " + x.reason.split(": ", 1)[-1]
    if HEALTH_STATUS.exists():
        h = pd.read_parquet(HEALTH_STATUS)
        key = "DeviceName" if "DeviceName" in h else "DeviceId"
        h = h[h.status.astype(str).eq("bad")]
        for a, b, why in zip(h[key].astype(str), h.detector.astype(int),
                             h.reason if "reason" in h else [""] * len(h)):
            out.setdefault((a.lower() if key == "DeviceId" else a, int(b)), f"health model: bad ({why})")
    return out


def card_faults() -> dict:
    """(DeviceName, detector) -> plain text, for channels on a card whose two outputs are both dead or both erratic
    (note 34): a failing card takes both its detectors down the same way."""
    if not CARDS.exists():
        return {}
    c = pd.read_parquet(CARDS)
    n_on = dict(zip(zip(c.DeviceName, c.detector), c.n_on))
    # "erratic" must come from actuations (stuck-on / chatter), never from fault events 83-88 (engineer, 2026-09-28)
    act = c.reasons.fillna("").str.contains("stuck-on|chatter")
    ok_act = dict(zip(zip(c.DeviceName, c.detector), act))
    out = {}
    for r in c[c.card_suspect].itertuples(index=False):
        mate = int(r.card_mate) if pd.notna(r.card_mate) else None
        if not r.dead and not (ok_act.get((r.DeviceName, int(r.detector))) and ok_act.get((r.DeviceName, mate))):
            continue
        how = "dead" if r.dead else "erratic"
        mn = n_on.get((r.DeviceName, mate), np.nan) if mate is not None else np.nan
        num = f"{int(_num(r.n_on) or 0):,} / {int(mn) if np.isfinite(_num(mn)) else 0:,} actuations"
        out[(r.DeviceName, int(r.detector))] = dict(
            slot=r.slot_no, mate=mate, how=how,
            text=f"card fault: both outputs of slot {r.slot_no} {how} (det {int(r.detector)} and det {mate}: {num}"
                 + ("; " + "; ".join(x.strip() for x in str(r.reasons).replace("health:", "").split(";")
                                     if "stuck" in x or "chatter" in x) if not r.dead else "") + ")")
    return out


def validate(v3: pd.DataFrame, stats: pd.DataFrame, hl: pd.DataFrame, cal: dict, P: pd.DataFrame | None = None,
             extra: pd.DataFrame | None = None, pplt: pd.DataFrame | None = None
             ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (d = v3 + stats + validation columns, detail = one row per failed / health / field-issue check,
    phase flags).

    validated (v3 of this module, engineer's spot-check 2026-09-28): 'label wrong' is kept apart from 'detector /
    configuration problem'. Only the former removes a label:
      fail           the detector does not behave like its label (label wrong)
      unhealthy      dead-alive but stuck on / chattering / saturated / erratic card, by ACTUATIONS only (no fault
                     events): excluded from training, not a label verdict
      misconfigured  (v4) a configuration field issue: label kept, never trained on
      pass           at least one behaviour check passed (a field issue may still be listed, label kept)
      not_checkable  no check could be evaluated (a field issue may still be listed)
      no_data        no hi-res data / no actuations
    field_issue: configuration / field problems for review/field_issues_for_staff.xlsx (pulse-set presence zone, count
    zone not in pulse where vehicles stop, advance loop missing detections, ...)."""
    drop = [c for c in ("DeviceName", "function", "phase_target", "phase_diagram", "lane_index", "lanes_spanned",
                        "technology", "n_lanes_phase") if c in stats]
    s = stats.drop(columns=drop).rename(columns={"DeviceId": "sid"})
    d = (v3.assign(sid=v3.DeviceId.str.lower()).merge(s, on=["sid", "detector"], how="left")
         .merge(hl.rename(columns={"DeviceId": "sid"}), on=["sid", "detector"], how="left"))
    if extra is not None and len(extra):
        d = d.merge(extra.rename(columns={"DeviceId": "sid"}), on=["sid", "detector"], how="left")
    for c in ("occ", "short_frac", "dur_p90"):
        if c not in d:
            d[c] = np.nan
    assert len(d) == len(v3)
    rules = cal["rules"]
    pres = rules["presence_holds_red"]
    # protected-permissive (FYA) phases: most left turns leave during the permissive interval, which the phase's own
    # colour sees as red -> the presence "holds through red" and count "off in red" rules do not apply there
    pp = {} if pplt is None or pplt.empty else {(a, int(b)): h for a, b, h in zip(pplt.sid, pplt.p, pplt.how)}
    d["pplt"] = [pp.get((a, int(p))) if pd.notna(p) else None for a, p in zip(d.sid, d.p)]
    d["pulse_set"] = (d.short_frac.astype(float) >= PULSE_SHARE) & (d.n_on.fillna(0) >= MIN_ON)
    # right-turn-only lane (print): right turn on red, so a stop-bar zone there does not hold through red (2026-09-29)
    d["right_lane"] = d.lane_type.astype("string").isin(list(RIGHT_LANE)).fillna(False).astype(bool) \
        if "lane_type" in d else False
    ruled = ruled_unhealthy()
    lead_thr = limit_for(rules["advance_leads_lane"], "loop")[0] if rules["advance_leads_lane"].get("stat") else 0.05
    R = role_check(d, P if P is not None else pd.DataFrame(), lead_thr)
    d = d.merge(R, on=["sid", "detector"], how="left")
    # ---------------- rule 6: phase with a print presence zone but no channel holding a call through red at peak
    pstat = pres.get("stat", "on_green_peak")

    def holds(row):
        thr, ok = limit_for(pres, row.tech)
        x = _num(getattr(row, pstat))
        return (x >= thr) if (ok and np.isfinite(x)) else None
    d["_holds"] = [holds(r) for r in d.itertuples(index=False)]
    flags = []
    for (dev, p), g in d[d.p.notna()].groupby(["sid", "p"]):
        if (dev, int(p)) in pp:  # permissive left turn: queued vehicles leave on the FYA, nothing holds through "red"
            continue
        pz = g[g.print_subtype.eq("stopbar_presence") & g.source.astype("string").str.startswith("print").fillna(False)]
        if pz.empty:
            continue
        ev = g[g._holds.notna()]
        if ev.empty or ev._holds.astype(bool).any():
            continue
        top = ev.assign(_v=ev[pstat].astype(float)).sort_values("_v", ascending=False).iloc[0]
        pzt = [f"det {int(q.detector)} " + ("set to pulse (ONs ~0.2 s)" if bool(q.pulse_set) else
                                             f"held {_num(getattr(q, pstat)):.0%}" if np.isfinite(_num(getattr(q, pstat)))
                                             else f"{int(_num(q.n_on) if np.isfinite(_num(q.n_on)) else 0):,} actuations, "
                                                  f"too few to judge") for q in pz.itertuples(index=False)]
        flags.append(dict(sid=dev, DeviceName=g.DeviceName.iloc[0], p=int(p), phase=f"P{int(p)}",
                          print_presence="; ".join(pzt), all_pulse=bool(pz.pulse_set.all()),
                          n_checked=len(ev), best_det=int(top.detector), best_label=top.function,
                          best_value=float(top._v), unusual_layout=bool(g.unusual_layout.astype(bool).any())))
    F = pd.DataFrame(flags)
    flagged = set(zip(F.sid, F.p)) if len(F) else set()
    # ---------------- per detector
    by_dev = {k: g.set_index("detector") for k, g in d.groupby("sid")}

    def looker(dev):
        g = by_dev[dev]

        def look(y):
            if y not in g.index:
                return f"det {y}"
            q = g.loc[y]
            q = q.iloc[0] if isinstance(q, pd.DataFrame) else q
            return f"{q.function if isinstance(q.function, str) else 'unlabelled'}, {q.tech}, {lane_txt(q.lane_index, q.lanes_spanned)}"
        return look
    val, fails, why, nums, fis, detail = [], [], [], [], [], []
    cards = card_faults()
    d["card_fault"] = [cards.get((a, int(b)), {}).get("text") for a, b in zip(d.DeviceName, d.detector)]

    def card_row(r, kind):
        cf = cards[(r.DeviceName, int(r.detector))]
        detail.append(dict(sid=r.sid, DeviceName=r.DeviceName, detector=int(r.detector), function=r.function,
                           tech=r.tech, lane=lane_txt(r.lane_index, r.lanes_spanned), phase=r.phase_target,
                           source=r.source, check="card_fault",
                           compared_with=f"det {cf['mate']} (the other output of the same card, slot {cf['slot']})",
                           this_value=cf["text"].split("(", 1)[1].rstrip(")"), partner_value="",
                           expectation="the two outputs of one detector card fail independently",
                           result=f"both outputs {cf['how']}: the card itself is suspect - do not train on either",
                           info=False, kind=kind))
    for r in d.itertuples(index=False):
        f = r.function
        if not isinstance(f, str):
            val.append(None); fails.append(None); why.append(None); nums.append(None); fis.append(None)
            continue
        if r.card_fault and (not isinstance(r.period, str) or not _num(r.n_on) > 0):
            card_row(r, "health")
            val.append("no_data"); fails.append("card_fault"); why.append(r.card_fault); nums.append("0 actuations")
            fis.append(r.card_fault)
            continue
        n = _num(r.n_on)
        n = 0 if not np.isfinite(n) else n
        if not isinstance(r.period, str):
            val.append("no_data"); fails.append(""); why.append("no hi-res data for this signal"); nums.append("")
            fis.append(None)
            continue
        if n == 0:
            val.append("no_data"); fails.append(""); why.append("no actuations in the newest window (dead?)")
            nums.append("0 actuations"); fis.append(None)
            continue
        look = looker(r.sid)
        fl, hl_, fi, rs, nm, ok, info = [], [], [], [], [f"{int(n):,} actuations"], 0, []
        base = dict(sid=r.sid, DeviceName=r.DeviceName, detector=int(r.detector), function=f, tech=r.tech,
                    lane=lane_txt(r.lane_index, r.lanes_spanned), phase=r.phase_target, source=r.source)
        # ---- health: actuations only (stuck / chatter / saturation / erratic card) -> unhealthy, not a label verdict
        hp, _ = _health(r)
        mis = []   # configuration problems (label kept, not trained: validated = misconfigured)
        ru = ruled.get((r.DeviceName, int(r.detector))) or ruled.get((r.sid, int(r.detector)))
        if ru:
            hp = hp + [ru]
        if hp:
            hl_.append("health"); rs.append("detector health: " + "; ".join(hp))
            detail.append({**base, "check": "health", "compared_with": "-", "this_value": "; ".join(hp),
                           "partner_value": "", "expectation": "not stuck on (> 15 min), not chattering (actuations only; "
                           "fault events are not used)", "result": "detector health problem", "info": False,
                           "kind": "health"})
        if r.card_fault:
            hl_.append("card_fault"); rs.append(r.card_fault)
            card_row(r, "health")
        if f in VEHICLE and _num(r.s_sat) < 0.5:
            lanes = max(int(_num(r.dq_lanes)) if np.isfinite(_num(r.dq_lanes)) else 1, 1)
            txt = f"{_num(r.max5):.0f} actuations in 5 minutes"
            hl_.append("saturation"); rs.append(f"{txt}, more than {lanes} lane(s) can carry (limit {150 * lanes})")
            nm.append(f"max {_num(r.max5):.0f} per 5 min")
            detail.append({**base, "check": "saturation", "compared_with": "-", "this_value": txt, "partner_value": "",
                           "expectation": f"at most ~150 actuations per 5 min per lane ({lanes} lane(s))",
                           "result": "more than the lanes can carry (double counting / chatter / wrong lanes)",
                           "info": False, "kind": "health"})
        if f == "Count" and r.tech == "loop":
            e = explain("loop_not_count", r, np.nan, look)
            fl.append("loop_not_count"); rs.append(f"{e['expectation']}; {e['result']}")
            detail.append({**base, **e, "info": False, "kind": "label"})
        sup = r.role_support if isinstance(r.role_support, str) else None
        con_ = r.role_contra if isinstance(r.role_contra, str) and not sup else None
        pulse = bool(r.pulse_set)
        has_rule = False
        for rid, c in rules.items():
            if c["cls"] != f or not c.get("stat"):
                continue
            thr, applies = limit_for(c, r.tech)
            if not applies:
                continue
            if rid in ("presence_holds_red", "count_off_in_red") and r.pplt:
                info.append(f"{CHECK_PLAIN[rid]} not applied: protected-permissive left turn ({r.pplt}) - most "
                            f"vehicles leave on the flashing yellow arrow, which the phase's own colour logs as red")
                continue
            if rid in ("presence_holds_red", "count_off_in_red") and r.right_lane:
                info.append(f"{CHECK_PLAIN[rid]} not applied: right-turn lane (print) - vehicles turn right on red, "
                            f"so waiting vehicles leave before the phase turns green")
                continue
            has_rule = has_rule or rid not in INFO_ONLY
            x = _num(getattr(r, c["stat"]))
            if not np.isfinite(x):
                continue
            passed = x >= thr if c["dir"] > 0 else x <= thr
            weak = rid in INFO_ONLY
            ok += bool(passed) and not weak
            if passed:
                continue
            e = explain(rid, r, thr, look)
            txt = f"{c['text']}: {e['this_value']}" + (f" vs {e['partner_value']}" if e["partner_value"] else "") + \
                  f" [{e['compared_with']}] - {e['result']}"
            if weak:
                info.append("information only: " + txt)
                nm.append(f"{c['stat']} {x:.2f} (info)")
                detail.append({**base, **e, "info": True, "kind": "info"})
                continue
            # ---- a configuration / field problem, not a wrong label -> field issue, label kept
            issue = None
            if rid == "presence_holds_red" and pulse:
                issue = (f"presence zone set to PULSE: {r.short_frac:.0%} of its ONs last <= {PULSE_S} s (90 % <= "
                         f"{_num(r.dur_p90):.1f} s), so it cannot hold a call; should be set to normal / presence")
            elif rid == "advance_loop_counts":
                issue = (f"advance loop misses detections: {int(_num(r.cnt_x)):,} actuations vs {int(_num(r.cnt_y)):,} "
                         f"on its same-lane stop-bar loop det {int(_num(r.cnt_to))} (ratio {_num(r.cnt_ratio):.2f})")
            elif rid in ("count_off_in_red", "count_short_ons", "presence_holds_red") and sup and r.role_pair == True:  # noqa: E712 (NaN-safe)
                issue = (f"{CHECK_PLAIN[rid]} fails ({e['this_value']}) but order / occupancy on the phase confirm the "
                         f"{f} label ({sup})" + ("; count zone not set to pulse and placed where vehicles stop on it"
                                                  if f == "Count" else ""))
            if issue:
                fi.append(issue); nm.append(f"{c['stat']} {x:.2f} (field)")
                detail.append({**base, **e, "result": issue, "info": False, "kind": "field"})
                # 2026-09-29: a detector missing vehicles is a HEALTH problem; the others are CONFIGURATION problems.
                # Either way the label is kept but the detector is not trained on.
                (hl_ if rid == "advance_loop_counts" else mis).append(
                    "advance_misses" if rid == "advance_loop_counts" else f"misconfigured:{rid}")
            else:
                fl.append(rid); rs.append(txt); nm.append(f"{c['stat']} {x:.2f}")
                detail.append({**base, **e, "info": False, "kind": "label"})
        # ---- the engineer's order + occupancy cross-check (Presence / Count / Advance-before-Presence)
        if f in ("Presence", "Count"):
            if sup:
                ok += 1; has_rule = True
                info.append("order / occupancy confirm the label: " + sup)
            elif con_:
                has_rule = True
                fl.append("role_order_occupancy"); rs.append("order / occupancy: " + con_)
                detail.append({**base, "check": "role_order_occupancy", "compared_with": "the other stop-bar zone of "
                               "its phase / lane", "this_value": con_, "partner_value": "",
                               "expectation": "of a stop-bar Presence / Count pair the Presence zone is more occupied "
                                              "and counts fewer vehicles", "result": "the two look swapped",
                               "info": False, "kind": "label"})
        if f == "Presence" and pulse and not any("PULSE" in x for x in fi):
            fi.append(f"presence zone set to PULSE: {r.short_frac:.0%} of its ONs last <= {PULSE_S} s; should be "
                      f"normal / presence")
        if f == "Presence" and pulse and not any(m.startswith("misconfigured:pulse") for m in mis):
            mis = [m for m in mis if m != "misconfigured:presence_holds_red"] + ["misconfigured:pulse_presence"]
        # radar Advance held long like an ETA zone: information (rule 7)
        if f == "Advance" and r.tech == "radar" and _num(r.dur_med) >= 2.5:
            info.append(f"information only: held {_num(r.dur_med):.1f} s typically, like a radar ETA zone")
        if f == "Advance" and not np.isfinite(_num(getattr(r, "lead", np.nan))) and _num(getattr(r, "lead_busy", np.nan)) > 0:
            info.append(f"order not checkable: the advance fires so often ({int(n):,} actuations) that its "
                        f"lead over the stop-bar zones cannot be told from chance")
        if f == "Presence" and (r.sid, r.p) in flagged:
            rs.append(f"phase {r.phase_target}: no detector on this phase holds a call through red at peak")
        if fi:
            rs.append("field issue (label kept): " + " | ".join(fi))
        rs += info
        if hl_:
            val.append("unhealthy")
            ht = "; ".join(x for x in rs if x.startswith(("detector health", "card fault")) or "actuations in 5 minutes" in x)
            if ht:
                fi.append(ht)
        elif fl:
            val.append("fail")
        elif mis:   # 2026-09-29 (engineer): a misconfigured zone is never trained on, even with the right label
            val.append("misconfigured")
            rs.append("misconfigured: label kept, not used for training")
        elif ok or not has_rule:
            val.append("pass")
        else:
            val.append("not_checkable")
            rs.append("too few actuations for the behaviour checks" if n < MIN_ON else
                      "no behaviour check could be evaluated (no peak cycles, no free-running period, or no same-lane "
                      "partner)")
        fails.append(",".join(hl_ + fl + mis)); why.append(" | ".join(rs)); nums.append("; ".join(nm))
        fis.append(" || ".join(fi) if fi else None)
    d["validated"], d["failed_checks"], d["validation_reason"], d["validation_numbers"] = val, fails, why, nums
    d["field_issue"] = fis
    D = pd.DataFrame(detail)
    if len(D) and "kind" not in D:
        D["kind"] = "label"
    return d, D, F


# ============================================================================ summary + review workbooks
def summary(d: pd.DataFrame, cal: dict) -> pd.DataFrame:
    lab = d[d.function.notna()]
    fc = lab.failed_checks.fillna("").str.split(",")
    rows = []
    alive = lab.validated.isin(["pass", "fail", "not_checkable", "unhealthy"])
    for rid, c in cal["rules"].items():
        m = lab.function.eq(c["cls"]) & alive
        st = c.get("stat")
        nchk = int((m & lab[st].notna()).sum()) if st else 0
        nf = int((m & fc.map(lambda l: rid in l)).sum())
        bt = "; ".join(f"{k}: limit {v['threshold']:.2f}, AUC {v.get('auc')}" + ("" if v.get("kept") else " (not used)")
                       for k, v in (c.get("by_tech") or {}).items() if "threshold" in v)
        rows.append({"check": rid, "for label": c["cls"], "engineer's rule": c["text"], "measured as": st or "",
                     "separates (AUC)": c.get("auc"),
                     "used": ("information only" if rid in INFO_ONLY else "yes" if c["kept"] else "no - does not separate"),
                     "limit (clear prints)": round(c["threshold"], 3) if c.get("threshold") is not None else None,
                     "per technology": bt, "detectors checked": nchk, "failed": nf,
                     "fail rate": round(nf / nchk, 3) if nchk else None})
    fi = lab.field_issue.fillna("") if "field_issue" in lab else pd.Series("", index=lab.index)
    for rid, text, m in (("role_order_occupancy", "of a stop-bar Presence / Count pair the Presence is more occupied and "
                          "counts fewer (engineer, 2026-09-28); support = a passed check, the reverse = fail",
                          alive & lab.function.isin(["Presence", "Count"])),
                         ("health", "not stuck on, not chattering (actuations only; fault events not used) -> unhealthy",
                          alive),
                         ("saturation", "at most ~150 actuations per 5 min per lane", alive & lab.function.isin(VEHICLE)),
                         ("loop_not_count", "a loop is never a stop-bar Count (your rule)", alive & lab.function.eq("Count")),
                         ("card_fault", "both outputs of one detector card dead / erratic (note 34)",
                          lab.validated.isin(["pass", "fail", "not_checkable", "no_data"]))):
        nf = int((m & fc.map(lambda l: rid in l)).sum())
        if rid == "role_order_occupancy":
            nchk = int((m & (lab.role_support.notna() | lab.role_contra.notna())).sum())
            rows.append({"check": rid, "for label": "Presence / Count", "engineer's rule": text, "measured as": "occ, n_on",
                         "separates (AUC)": None, "used": "yes", "limit (clear prints)": OCC_RATIO, "per technology": "",
                         "detectors checked": nchk, "failed": nf, "fail rate": round(nf / nchk, 3) if nchk else None})
            continue
        rows.append({"check": rid, "for label": "all", "engineer's rule": text, "measured as": "", "separates (AUC)": None,
                     "used": "yes (physical limit / your rule)", "limit (clear prints)": None, "per technology": "",
                     "detectors checked": int(m.sum()), "failed": nf, "fail rate": round(nf / m.sum(), 3) if m.sum() else None})
    nfi = int(fi.ne("").sum())
    npp = int(lab.pplt.notna().sum()) if "pplt" in lab else 0
    rows.append({"check": "field issues (label kept)", "for label": "all", "engineer's rule": "pulse-set presence, count "
                 "zone not in pulse where vehicles stop, advance loop missing detections, health: listed in "
                 "review/field_issues_for_staff.xlsx", "measured as": "", "separates (AUC)": None, "used": "no label removed",
                 "limit (clear prints)": None, "per technology": "", "detectors checked": int(len(lab)), "failed": nfi,
                 "fail rate": None})
    rows.append({"check": "protected-permissive phases", "for label": "Presence / Count", "engineer's rule": "presence "
                 "holds through red / count off in red NOT applied on FYA / permissive left-turn phases",
                 "measured as": "FYA events 32 (a clue: not every signal logs them), timing additional call, or "
                 f"behaviour (>= {LEAVE_THR:.0%} of red waits leave before green)", "separates (AUC)": 0.82,
                 "used": "exemption", "limit (clear prints)": LEAVE_THR, "per technology": "", "detectors checked": npp,
                 "failed": 0, "fail rate": None})
    nrl = int(lab.right_lane.sum()) if "right_lane" in lab else 0
    rows.append({"check": "right-turn lanes", "for label": "Presence / Count", "engineer's rule": "right turn on red: "
                 "the red-time rules are NOT applied to a stop-bar zone in a right-turn-only lane (print lane R)",
                 "measured as": "print lane_type R", "separates (AUC)": None, "used": "exemption",
                 "limit (clear prints)": None, "per technology": "", "detectors checked": nrl, "failed": 0,
                 "fail rate": None})
    nmc = int(lab.validated.eq("misconfigured").sum())
    rows.append({"check": "misconfigured (label kept, not trained)", "for label": "all", "engineer's rule": "a zone "
                 "set up wrong (presence set to pulse, count zone not in pulse where vehicles stop) is never trained on",
                 "measured as": "field issue of the configuration kind", "separates (AUC)": None, "used": "yes",
                 "limit (clear prints)": None, "per technology": "", "detectors checked": int(len(lab)), "failed": nmc,
                 "fail rate": None})
    return pd.DataFrame(rows)


def _priority(f: pd.DataFrame) -> pd.Series:
    """Top = 100: ATSPM class x2; source clear print / user x1.5, medium x1.2, config x1, unsure x0.9, whole-int x0.7;
    busy x(1 + log10(1 + ONs) / 4); health / saturation only x0.5; unusual_layout x0.3."""
    n = pd.to_numeric(f.n_on, errors="coerce").fillna(0)
    srcw = f.source.map({"print_high": 1.5, "user_ruling": 1.5, "print_medium": 1.2, "config": 1.0,
                         "review_round1": 1.0, "print_low": 0.9, "whole_intersection_other": 0.7}).fillna(1.0)
    ho = f.failed_checks.fillna("").str.split(",").map(lambda l: set(l) <= {"health", "saturation", "card_fault"})
    w = (np.where(f.function.isin(ATSPM), 2.0, 1.0) * srcw * (1 + np.log10(1 + n) / 4)
         * np.where(ho, 0.5, 1.0) * np.where(f.unusual_layout.astype(bool), 0.3, 1.0))
    return pd.Series(100 * w / w.max() if len(f) else w, index=f.index).round(1)


CHECK_PLAIN = {"presence_holds_red": "presence: holds the call through red at peak",
               "count_off_in_red": "count: off during red", "count_short_ons": "count: short actuations",
               "advance_red_arrivals": "advance: arrivals on red (free running)",
               "advance_leads_lane": "advance: fires before the same-lane stop-bar zone",
               "advance_loop_counts": "advance loop: counts like its same-lane stop-bar loop (not far less)",
               "mid_tracks_sum": "mid: tracks the advance-loop sum", "bike_far_below": "bike: far below vehicles",
               "presence_undercounts_peak": "presence: falls behind the advance at peak (info)",
               "loop_not_count": "a loop cannot be Count", "health": "detector health", "saturation": "saturation",
               "card_fault": "card fault (both outputs of one card)",
               "role_order_occupancy": "order / occupancy: presence vs count on the phase",
               "phase_presence_missing": "PHASE: no detector holds a call through red"}


def _crop(dn, det, c):
    if isinstance(c, str) and c:
        return Path(c).name
    g = f"{dn}_d{det}.png"
    return g if (CAB / "crops" / g).exists() else ""


def detail_table(d: pd.DataFrame, detail: pd.DataFrame, F: pd.DataFrame, pres_thr: dict) -> pd.DataFrame:
    """One row per failed check (plus phase flags), grouped by signal (signal order = its top priority)."""
    f = d[d.validated.isin(["fail", "unhealthy"]) | (d.validated.eq("no_data") & d.card_fault.notna())].copy()
    f["priority"] = _priority(f)
    D = detail[~detail["info"] & detail["kind"].isin(["label", "health"])].copy() if len(detail) else detail
    D = D.merge(f[["sid", "detector", "priority", "crop", "n_on", "unusual_layout"]], on=["sid", "detector"], how="inner")
    D["crop file"] = [_crop(a, b, c) for a, b, c in zip(D.DeviceName, D.detector, D.crop)]
    rows = D.assign(kind="detector", type=np.where(D["kind"].eq("health"), "detector health (not a label verdict)",
                                                     "label looks wrong"))
    if len(F):
        pmax = f.groupby("sid").priority.max()
        G = F.assign(detector="", function="(phase)", tech="", lane="", source="print", kind="phase",
                     type="phase flag (field)",
                     check="phase_presence_missing",
                     compared_with=[f"every channel on {p} ({n} checked); best det {b} ({l})"
                                    for p, n, b, l in zip(F.phase, F.n_checked, F.best_det, F.best_label)],
                     this_value=[f"best: det {b} held {v:.0%} of peak red arrivals until green"
                                 for b, v in zip(F.best_det, F.best_value)],
                     partner_value=[f"print presence zone(s): {x}" for x in F.print_presence],
                     expectation="a phase whose print shows a stop-bar presence zone has a detector holding the call "
                                 "through red at peak",
                     result="no presence-like detector on this phase: the zone may be misconfigured in the field or "
                            "wired to another channel",
                     priority=[max(pmax.get(i, 0), 60.0) * (0.3 if u else 1.0) for i, u in zip(F.sid, F.unusual_layout)],
                     **{"crop file": ""})
        rows = pd.concat([G, rows], ignore_index=True)
    rows["sig_pri"] = rows.groupby("sid").priority.transform("max")
    rows = rows.sort_values(["sig_pri", "DeviceName", "kind", "priority", "detector"],
                            ascending=[False, True, False, False, True], key=None)
    out = pd.DataFrame({"signal": rows.DeviceName, "phase": rows.phase.astype("string").fillna(""),
                        "detector": rows.detector, "label": rows.function, "technology": rows.tech, "lane (print)": rows.lane,
                        "label source": rows.source.map(SRC_PLAIN).fillna(rows.source),
                        "type": rows["type"], "check": rows.check.map(CHECK_PLAIN).fillna(rows.check),
                        "compared with": rows.compared_with, "this detector": rows.this_value,
                        "partner / reference": rows.partner_value, "expected": rows.expectation,
                        "what it means": rows.result, "priority": rows.priority.round(1),
                        "unusual site (not trained on)": np.where(rows.unusual_layout.astype("boolean").fillna(False), "yes", ""),
                        "crop file": rows["crop file"].fillna("")})
    return out.reset_index(drop=True)


COLW = {"signal": 8, "phase": 6, "detector": 8, "label": 10, "technology": 9, "lane (print)": 9, "label source": 18,
        "check": 30, "compared with": 38, "this detector": 40, "partner / reference": 34, "expected": 55,
        "what it means": 50, "priority": 7, "unusual site (not trained on)": 9, "crop file": 18, "note": 40,
        "verdict": 12, "why chosen": 34, "your answer (right / wrong / ?)": 18, "your comment": 30}


def write_grouped(path: Path, sheets: dict[str, pd.DataFrame], header_col: str = "signal"):
    """Workbook with a bold header row before each signal's rows."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    wb = Workbook()
    wb.remove(wb.active)
    for name, df in sheets.items():
        ws = wb.create_sheet(name)
        cols = list(df.columns)
        ws.append(cols)
        for c in ws[1]:
            c.font = Font(bold=True)
        grouped = header_col in cols and name != "checks"
        last = None
        for rec in df.itertuples(index=False):
            rec = list(rec)
            if grouped and rec[cols.index(header_col)] != last:
                last = rec[cols.index(header_col)]
                g = df[df[header_col] == last]
                nd = g[g.label != "(phase)"].detector.nunique() if "label" in g else len(g)
                npf = int((g.label == "(phase)").sum()) if "label" in g else 0
                ws.append([f"Signal {last}" if name == "spot check" else
                           f"Signal {last}: {nd} detector(s) failed" + (f", {npf} phase flag(s)" if npf else "")])
                for c in ws[ws.max_row]:
                    c.font = Font(bold=True, color="FFFFFF")
                    c.fill = PatternFill("solid", fgColor="44546A")
            ws.append([None if (isinstance(v, float) and not np.isfinite(v)) else v for v in rec])
        for i, c in enumerate(cols):
            ws.column_dimensions[ws.cell(1, i + 1).column_letter].width = COLW.get(c, 14)
        for row in ws.iter_rows(min_row=2):
            for c in row:
                c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.freeze_panes = "A2"
    path.parent.mkdir(exist_ok=True)
    try:
        wb.save(path)
    except PermissionError:  # the reviewer has the old workbook open in Excel: never lose the new one
        alt = path.with_name(path.stem + '_new' + path.suffix)
        wb.save(alt)
        log(f'{path.name} is open elsewhere -> wrote {alt.name}')


def review_list(d, detail, F, cal, path: Path | None = None) -> pd.DataFrame:
    path = path or REVIEW / "label_check_failures.xlsx"
    out = detail_table(d, detail, F, cal["rules"]["presence_holds_red"])
    write_grouped(path, {"failed checks by signal": out, "checks": summary(d, cal)})
    return out


# ============================================================================ spot-check workbook for the user
NOT_CHK = {"presence_holds_red": "fewer than 20 peak cycles / 100 actuations",
           "count_off_in_red": "too few actuations", "count_short_ons": "too few actuations",
           "advance_red_arrivals": "the signal is coordinated all window (no free-running period to judge)",
           "advance_leads_lane": "no same-lane stop-bar zone with data, or the advance fires too often to see the order",
           "advance_loop_counts": "not a loop advance with a same-lane stop-bar presence loop",
           "mid_tracks_sum": "fewer than two lane-by-lane advance loops on the phase",
           "bike_far_below": "no vehicle detector on the phase"}


def all_checks(r, rules: dict, look) -> list[dict]:
    """Every check of the detector's class with its verdict (for controls and the user's earlier examples)."""
    out = []
    for rid, c in rules.items():
        if c["cls"] != r.function or not c.get("stat") or rid in INFO_ONLY:
            continue
        thr, applies = limit_for(c, r.tech)
        e = dict(check=rid, compared_with="", this_value="", partner_value="", expectation=c["text"], result="")
        if not applies:
            out.append({**e, "verdict": "not used", "result": f"rule not used for {r.tech} (does not separate there)"})
            continue
        x = _num(getattr(r, c["stat"]))
        if not np.isfinite(x):
            out.append({**e, "verdict": "not checkable", "result": NOT_CHK.get(rid, "not evaluated")})
            continue
        passed = x >= thr if c["dir"] > 0 else x <= thr
        e = explain(rid, r, thr, look)
        if passed:
            e["result"] = "as expected"
        out.append({**e, "verdict": "pass" if passed else "FAIL", "margin": (x - thr) / (abs(thr) + 1e-9) * c["dir"]})
    return out


def _looker(d: pd.DataFrame):
    by_dev = {k: g.set_index("detector") for k, g in d.groupby("sid")}

    def look_for(dev):
        g = by_dev[dev]

        def look(y):
            if y not in g.index:
                return f"det {y}"
            q = g.loc[y]
            q = q.iloc[0] if isinstance(q, pd.DataFrame) else q
            f = q.function if isinstance(q.function, str) else "unlabelled"
            return f"{f}, {q.tech}, {lane_txt(q.lane_index, q.lanes_spanned)}"
        return look
    return look_for


def spotcheck(d: pd.DataFrame, detail: pd.DataFrame, F: pd.DataFrame, cal: dict, path: Path | None = None,
              examples: list | None = None, n_fail: int = 15, seed: int = 7) -> pd.DataFrame:
    """~15 diverse failed detectors (checks x technologies x classes, some borderline), 3 passing controls and the
    user's earlier examples, each with the comparison spelled out."""
    path = path or REVIEW / "spotcheck_label_checks.xlsx"
    rules = cal["rules"]
    rng = np.random.default_rng(seed)
    look_for = _looker(d)
    rows = []

    def add(r, why, only=None):
        look = look_for(r.sid)
        cs = all_checks(r, rules, look)
        hp, _ = _health(r)
        if hp:
            cs = [dict(check="health", compared_with="-", this_value="; ".join(hp), partner_value="",
                       expectation="not stuck on, not chattering", verdict="FAIL",
                       result="detector health problem")] + cs
        if r.function == "Count" and r.tech == "loop":
            e = explain("loop_not_count", r, np.nan, look)
            cs = [{**e, "verdict": "FAIL"}] + cs
        if only:
            cs = [c for c in cs if c["check"] in only] or cs
        if not cs:
            cs = [dict(check="-", verdict=r.validated, result=r.validation_reason or
                       f"{r.function}: only detector health is checked, and it is fine")]
        for c in cs:
            rows.append({"why chosen": why, "signal": r.DeviceName, "phase": r.phase_target, "detector": int(r.detector),
                         "label": r.function, "technology": r.tech, "lane (print)": lane_txt(r.lane_index, r.lanes_spanned),
                         "label source": SRC_PLAIN.get(r.source, r.source), "overall": r.validated,
                         "check": CHECK_PLAIN.get(c["check"], c["check"]), "verdict": c.get("verdict", ""),
                         "compared with": c.get("compared_with", ""), "this detector": c.get("this_value", ""),
                         "partner / reference": c.get("partner_value", ""), "expected": c.get("expectation", ""),
                         "what it means": c.get("result", ""), "crop file": _crop(r.DeviceName, int(r.detector), r.crop),
                         "your answer (right / wrong / ?)": "", "your comment": ""})
    used = set()
    for dn, det, why in (examples or []):
        x = d[(d.DeviceName == dn) & (d.detector == det)]
        if len(x):
            add(next(x.itertuples(index=False)), why)
            used.add((dn, det))
    # diverse failures: one per (check, technology) cell, clear-print labels preferred; then borderline ones
    D = detail[~detail["info"]].merge(d[["sid", "detector", "unusual_layout"]], on=["sid", "detector"])
    D = D[~D.unusual_layout.astype(bool)]
    D["clear"] = D.source.isin(["print_high", "user_ruling"])
    picks = []
    for (chk, tech), _ in D.groupby(["check", "tech"]).size().sort_values(ascending=False).items():
        if len(picks) >= n_fail - 4:
            break
        c = D[(D.check == chk) & (D.tech == tech)]
        c = c[[(a, b) not in used for a, b in zip(c.DeviceName, c.detector)]]
        if c.empty:
            continue
        c = c[c.clear] if c.clear.any() else c
        q = c.iloc[int(rng.integers(len(c)))]
        picks.append((q.DeviceName, int(q.detector), f"a typical failure: {CHECK_PLAIN.get(chk, chk)} ({tech})", chk))
        used.add((q.DeviceName, int(q.detector)))
    bl = []
    for r in d[d.validated.eq("fail") & ~d.unusual_layout.astype(bool)].itertuples(index=False):
        for c in all_checks(r, rules, lambda y: ""):
            if c.get("verdict") == "FAIL" and np.isfinite(c.get("margin", np.nan)):
                bl.append((abs(c["margin"]), r.DeviceName, int(r.detector), c["check"]))
    seen = set()
    for m, dn, det, chk in sorted(bl, key=lambda q: q[0]):
        if len(seen) >= 4:
            break
        if (dn, det) in used or chk in seen:
            continue
        picks.append((dn, det, f"borderline: just outside the limit ({CHECK_PLAIN.get(chk, chk)})", chk))
        used.add((dn, det)); seen.add(chk)
    for dn, det, why, chk in picks:
        x = d[(d.DeviceName == dn) & (d.detector == det)]
        add(next(x.itertuples(index=False)), why, only={chk})
    # controls: clear-print detectors passing every check of their class
    P = d[d.validated.eq("pass") & trusted_mask(d) & d.failed_checks.fillna("").eq("")]
    for cls, tech, why in (("Advance", "loop", "control (passes): loop advance"),
                           ("Presence", "radar", "control (passes): radar presence"),
                           ("Count", "radar", "control (passes): radar count")):
        c = P[(P.function == cls) & (P.tech == tech)]
        if cls == "Advance" and (c.lead.notna() & c.cnt_ratio.notna()).any():
            c = c[c.lead.notna() & c.cnt_ratio.notna()]
        if len(c):
            add(next(c.iloc[[int(rng.integers(len(c)))]].itertuples(index=False)), why)
    S = pd.DataFrame(rows)
    G = []
    if len(F):
        ex = [dn for dn, _, _ in (examples or [])]
        rest = F[~F.DeviceName.isin(ex) & ~F.unusual_layout]
        FF = pd.concat([F[F.DeviceName.isin(ex)], rest.sample(min(2, len(rest)), random_state=seed)])
        for f in FF.itertuples(index=False):
            G.append({"why chosen": "your earlier example (phase with no presence-like detector)" if f.DeviceName in ex
                      else "a phase flag (new signal-level check)", "signal": f.DeviceName, "phase": f.phase,
                      "detector": "", "label": "(phase)", "technology": "", "lane (print)": "", "label source": "print",
                      "overall": "flag", "check": CHECK_PLAIN["phase_presence_missing"], "verdict": "FLAG",
                      "compared with": f"every channel on {f.phase} ({f.n_checked} checked)",
                      "this detector": f"best: det {f.best_det} ({f.best_label}) held {f.best_value:.0%} of peak red "
                                       f"arrivals until green",
                      "partner / reference": f"print presence zone(s): {f.print_presence}",
                      "expected": "a phase whose print shows a stop-bar presence zone has a detector holding the call "
                                  "through red at peak", "what it means": "no presence-like detector on this phase",
                      "crop file": "", "your answer (right / wrong / ?)": "", "your comment": ""})
    S = pd.concat([pd.DataFrame(G), S], ignore_index=True)
    order = {sg: i for i, sg in enumerate(dict.fromkeys(list(S.signal[S.label != "(phase)"]) + list(S.signal)))}
    S = S.assign(_o=S.signal.map(order), _k=(S.label != "(phase)").astype(int)).sort_values(
        ["_o", "_k"], kind="stable").drop(columns=["_o", "_k"]).reset_index(drop=True)
    guide = pd.DataFrame({"read me": [
        "One block per detector; a detector can have several checks. 'verdict' is the new check's answer.",
        "Please mark 'your answer': right = the check is right about this detector, wrong = it is not, ? = can't tell.",
        "Controls pass every check; your earlier examples are shown with every check of their label.",
        "Limits were set once on the clearest print labels (complete, high-confidence prints), per technology."]})
    write_grouped(path, {"spot check": S, "read me": guide}, header_col="signal")
    return S


# ============================================================================ field issues for staff (engineer, 2026-09-28)
FIELD_XLSX = REVIEW / "field_issues_for_staff.xlsx"
SPOT_CONFIRMED = {("2B108", 18), ("2B108", 4), ("2B066", 11), ("2B349", 44), ("2B422", 8),  # his spot-check answers
                  ("01080", 16), ("04034", 4), ("11042", 22)}


def _fix_for(issue: str) -> str:
    i = issue.lower()
    if "not set to pulse" in i:
        return "Set the count zone to pulse, and move it just past the stop bar so stopped vehicles do not sit on it."
    if "pulse" in i and "presence" in i:
        return "Set the zone's output to normal / presence mode (not pulse) so it holds the call while a vehicle waits."
    if "misses detections" in i:
        return "Check the advance loop (sensitivity, lead-in, splices): it misses vehicles its stop-bar loop sees."
    if "poor detector health" in i or "health model" in i:
        return "Check the detector (loop / radar zone, card, sensitivity): its counts are not believable."
    if "stuck on" in i or "chatter" in i or "5 minutes" in i or "erratic" in i:
        return "Check the detector card / loop or radar zone: it stays on or retriggers far faster than traffic can."
    return "Check the zone setup against the print."


def field_issues(d: pd.DataFrame, F: pd.DataFrame, dead: pd.DataFrame | None = None,
                 path: Path | None = None) -> pd.DataFrame:
    """review/field_issues_for_staff.xlsx: one row per suspected configuration / field problem (never a model input;
    the label table keeps the label unless the label itself looks wrong). Sorted by signal."""
    path = path or FIELD_XLSX
    rows = []
    lab = d[d.function.notna()]

    def add(sig, dets, issue, ev, fix, src):
        rows.append({"signal": sig, "detector(s)": dets, "issue": issue, "evidence": ev, "suggested fix": fix,
                     "source": src})
    # 1. configuration / health found by the label check (label kept)
    for r in lab[lab.field_issue.notna() & ~lab.validated.eq("no_data")].itertuples(index=False):
        src = ("your spot-check 2026-09-28 + label check" if (r.DeviceName, int(r.detector)) in SPOT_CONFIRMED
               else "label check (hi-res actuations)")
        for part in str(r.field_issue).split(" || "):
            p = part.lower()
            if "not set to pulse" in p:
                iss = f"Count zone on {r.phase_target} not set to pulse, and vehicles stop on it"
            elif "pulse" in p and "presence" in p:
                iss = f"{r.function} zone on {r.phase_target} set to pulse instead of presence"
            elif "misses detections" in p:
                iss = f"Advance loop on {r.phase_target} misses vehicles (possible detector problem)"
            elif "confirm the" in p:
                iss = f"{r.function} zone on {r.phase_target}: right role (order / occupancy), but set up so it misbehaves"
            elif "card fault" in p:
                iss = f"Detector card erratic (both outputs), {r.phase_target}"
            elif "stuck" in p or "chatter" in p or "5 minutes" in p:
                iss = f"{r.function} detector on {r.phase_target} stuck on / chattering / over-counting"
            elif "poor detector health" in p or "health model" in p:
                iss = f"{r.function} detector on {r.phase_target}: poor detector health (label right)"
            else:
                iss = f"{r.function} detector on {r.phase_target}: behaviour does not fit its setup"
            add(r.DeviceName, str(int(r.detector)), iss, part, _fix_for(part), src)
    # 2. dead: the print / timing uses the channel but it never actuates; both outputs of one card dead -> card first
    cf = card_faults()
    deadset = {}
    if dead is not None and len(dead):
        for r in dead.itertuples(index=False):
            deadset.setdefault(str(r.signal), set()).add(int(r.detector))
    for r in lab[lab.validated.eq("no_data") & lab.card_fault.notna()].itertuples(index=False):
        deadset.setdefault(r.DeviceName, set()).add(int(r.detector))
    for sig, dets in deadset.items():
        slots = sorted({(v["slot"], min(k[1], v["mate"]), max(k[1], v["mate"])) for k, v in cf.items()
                        if k[0] == sig and v["how"] == "dead" and k[1] in dets})
        ev = f"0 actuations in the Sept 2026 window on detector(s) {', '.join(map(str, sorted(dets)))}"
        if slots:
            ev += "; both outputs of card slot(s) " + ", ".join(f"{s} (det {a} + {b})" for s, a, b in slots) + " dead"
        add(sig, ", ".join(map(str, sorted(dets))), "Detector(s) on the print / in the timing with no actuations",
            ev, "Check the detector card(s) first where both outputs are dead, then the loop / zone and its wiring.",
            "dead-detector list + card check")
    # 3. phase with a print presence zone but nothing holding a call through red (not on permissive left turns)
    for f in (F.itertuples(index=False) if len(F) else []):
        if f.all_pulse:
            continue  # the pulse rows above already say it
        add(f.DeviceName, "phase " + f.phase, f"No detector on {f.phase} holds a call while vehicles wait at the stop bar",
            f"print presence zone(s): {f.print_presence}; best on the phase: det {f.best_det} ({f.best_label}) held "
            f"{f.best_value:.0%} of peak red arrivals until green",
            "Check the stop-bar presence zone: output mode, position and which channel it is wired to.",
            "label check (hi-res actuations)")
    # 4. clear-print label that the detector does not behave like (print or field setup out of step)
    x = lab[lab.validated.eq("fail") & lab.source.eq("print_high") & lab.function.isin(ATSPM)]
    for r in x.itertuples(index=False):
        fc = [c for c in str(r.failed_checks).split(",") if c and c not in ("health", "card_fault", "saturation")]
        if not fc:
            continue
        add(r.DeviceName, str(int(r.detector)),
            f"Print says {r.function} on {r.phase_target}, but the detector does not behave like one",
            "; ".join(CHECK_PLAIN.get(c, c) for c in fc) + " - " + str(r.validation_reason).split(" | ")[0][:300],
            "Check the zone's position and output mode in the field against the print; update whichever is wrong.",
            ("your spot-check 2026-09-28 + " if (r.DeviceName, int(r.detector)) in SPOT_CONFIRMED else "")
            + "label check vs cabinet print")
    # 4b. the user's rulings that correct a wrong controller-configuration label (2026-09-29)
    if RULINGS.exists():
        R = pd.read_csv(RULINGS, dtype=str, keep_default_na=False)
        R = R[(R.rule == "relabel") & R.reason.str.contains("config labels wrong", case=False)]
        for x in R.itertuples(index=False):
            g = lab[(lab.DeviceName == x.DeviceName) & lab.detector.isin([int(v) for v in x.detector.split("|")])]
            for q in g.itertuples(index=False):
                cfg = q.config_function if isinstance(q.config_function, str) and q.config_function else q.func7_v2
                if isinstance(q.func7_v2, str) and q.func7_v2 == x.value:
                    continue   # the configuration already says so
                add(q.DeviceName, str(int(q.detector)),
                    f"Controller configuration labels det {int(q.detector)} as {cfg or 'nothing'}; it is {x.value}",
                    x.reason.split(": ", 1)[-1], "Correct the detector's function / description in the controller "
                    "configuration.", "your spot-check 2026-09-29")
    # 5. print vs controller timing
    pf = lab if "print_flags" not in d else d
    for flag, iss, fix in (("phase_print_vs_timing", "Cabinet print and controller timing disagree on the phase",
                            "Update the print (or the detector-to-phase assignment if the print is right)."),
                           ("print_channel_mismatch", "Print's zone numbers do not match the controller channels",
                            "Re-issue the print with the channel numbers the controller uses."),
                           ("print_numbering_stale", "Print's zone numbering is stale (timing grouped differently)",
                            "Re-issue the print with the in-service channel numbering."),
                           ("print_outdated", "Cabinet print is out of date (detection changed since)",
                            "Re-issue the print for the current detection.")):
        m = pf[pf.print_flags.fillna("").str.contains(flag)]
        for sig, g in m.groupby("DeviceName"):
            if flag == "phase_print_vs_timing":
                ev = "; ".join(f"det {int(q.detector)}: print {q.phase_diagram}, timing {q.phase_target}"
                               for q in g.itertuples(index=False))
            else:
                ev = str(g.confidence_reason.dropna().iloc[0])[:300] if g.confidence_reason.notna().any() else ""
            add(sig, ", ".join(str(int(v)) for v in sorted(g.detector)), iss, ev, fix, "cabinet print vs controller timing")
    out = pd.DataFrame(rows, columns=["signal", "detector(s)", "issue", "evidence", "suggested fix", "source"])
    out = out.sort_values(["signal", "issue"], kind="stable").reset_index(drop=True)
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    wb = Workbook()
    ws = wb.active
    ws.title = "field issues"
    ws.append(list(out.columns))
    for c in ws[1]:
        c.font = Font(bold=True)
    for rec in out.itertuples(index=False):
        ws.append(list(rec))
    for col, w in zip("ABCDEF", (8, 12, 45, 70, 50, 26)):
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A2"
    try:
        wb.save(path)
    except PermissionError:
        wb.save(path.with_name(path.stem + "_new" + path.suffix))
        log(f"{path.name} is open elsewhere -> wrote {path.stem}_new")
    return out


# the user's examples from his review of version 1 (review/label_check_failures.xlsx of 2026-09-23, ranks 1, 4, 5, 6)
USER_EXAMPLES = [
    ("10062", 18, "your earlier example (v1 rank 1): radar advance that failed the old lead check"),
    ("14037", 33, "your earlier example (v1 rank 4): radar advance det 33"),
    ("2B108", 18, "your earlier example (v1 rank 5): phase 6 presence, no presence-like detector on the phase"),
    ("2B108", 4, "same signal, phase 2: presence det 4 does not hold, count det 2 does (swapped?)"),
    ("04051", 50, "your earlier example (v1 rank 6): radar advance 50"),
    ("04051", 49, "your earlier example (v1 rank 6): radar ETA-type zone 49 (Other)"),
    ("04051", 2, "your earlier example (v1 rank 6): radar presence 2"),
]


# ============================================================================ driver
OUT_COLS = ["validated", "failed_checks", "validation_reason", "validation_numbers", "field_issue"]


def _stale(v3: pd.DataFrame) -> bool:
    cols = ["DeviceId", "detector", "function", "p", "lane_index", "lanes_spanned"]

    def norm(x):
        x = x.copy()
        x["DeviceId"], x["function"] = x.DeviceId.astype(str), x.function.astype("object").where(x.function.notna(), "")
        for c in cols[1:]:
            if c != "function":
                x[c] = pd.to_numeric(x[c], errors="coerce").astype(float).fillna(-1)
        return x[cols].sort_values(cols[:2]).reset_index(drop=True)
    try:
        return not norm(targets(v3)).equals(norm(pd.read_parquet(STATS, columns=cols)))
    except Exception:
        return True


def run(v3: pd.DataFrame, restat: bool = False, recalibrate: bool = False, write_review: bool = True):
    """v3 table -> (v3 + validation columns, working frame, calibration). Statistics cached (recomputed when a label,
    phase or lane changes); thresholds calibrated once and then read from THRESH."""
    if not restat and STATS.exists() and PAIRS.exists() and _stale(v3):
        log("labels / phases / lanes changed since the cached statistics -> recomputing")
        restat = True
    if restat or not STATS.exists() or not PAIRS.exists():
        s, P = compute_stats(v3)
        s.to_parquet(STATS, index=False)
        P.to_parquet(PAIRS, index=False)
    if restat or not HEALTH.exists():
        health(v3).to_parquet(HEALTH, index=False)
    st, P, hl = pd.read_parquet(STATS), pd.read_parquet(PAIRS), pd.read_parquet(HEALTH)
    cal = json.loads(THRESH.read_text(encoding="utf-8")) if THRESH.exists() else {}
    if recalibrate or cal.get("meta", {}).get("version") != 2:
        d0 = v3.assign(sid=v3.DeviceId.str.lower()).merge(
            st.drop(columns=["DeviceName", "function", "phase_target", "phase_diagram", "lane_index", "lanes_spanned",
                             "technology", "n_lanes_phase"], errors="ignore").rename(columns={"DeviceId": "sid"}),
            on=["sid", "detector"], how="left")
        cal = calibrate(d0, P)
        THRESH.write_text(json.dumps(cal, indent=1, default=float), encoding="utf-8")
        log(f"thresholds calibrated -> {THRESH}")
    if restat or not EXTRA.exists():
        extra_stats(st).to_parquet(EXTRA, index=False)
    pl = pplt_table(v3)
    pl.to_parquet(PPLT, index=False)
    d, detail, F = validate(v3, st, hl, cal, P=P, extra=pd.read_parquet(EXTRA), pplt=pl)
    d.attrs["detail"], d.attrs["phase_flags"] = None, None
    if write_review:
        review_list(d, detail, F, cal)
    out = v3.drop(columns=[c for c in OUT_COLS + ["train_use_validated"] if c in v3]).copy()
    for c in OUT_COLS:
        out[c] = pd.Series(d[c].to_numpy(), index=out.index).astype("string")
    out["train_use_validated"] = out.train_use & out.validated.eq("pass").fillna(False).astype(bool)
    run.last = dict(detail=detail, flags=F, pairs=P, pplt=pl, d=d)
    return out, d, cal


def report(d: pd.DataFrame, cal: dict):
    pd.set_option("display.width", 250)
    for rid, c in cal["rules"].items():
        print(f"{rid:26s} {c['cls']:10s} kept={c['kept']!s:5s} stat={c.get('stat')} auc={c.get('auc')} "
              f"thr={c.get('threshold') or float('nan'):.3g} n={c.get('n_class')} cands={c['candidates']} "
              f"tech={ {k: (v.get('auc'), round(v['threshold'], 3) if 'threshold' in v else None) for k, v in (c.get('by_tech') or {}).items()} }")
    lab = d[d.function.notna()].copy()
    lab["trusted"] = trusted_mask(lab)
    print(lab.validated.value_counts(dropna=False).to_dict())
    chk = lab[lab.validated.isin(["pass", "fail"])]
    print("fail rate among checked:", round((chk.validated == "fail").mean(), 4), "of", len(chk),
          "| among all labelled:", round((lab.validated == "fail").mean(), 4), "of", len(lab))
    for by in ("function", "tech", "source", "trusted", "unusual_layout", "tier"):
        t = lab.groupby(by, dropna=False).validated.value_counts().unstack(fill_value=0)
        t["fail_of_checked"] = (t.get("fail", 0) / (t.get("fail", 0) + t.get("pass", 0))).round(3)
        print(t.to_string())
    fc = lab.failed_checks.fillna("").str.split(",").explode()
    print(fc[fc != ""].value_counts().to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--restat", action="store_true")
    ap.add_argument("--recalibrate", action="store_true")
    ap.add_argument("--write", action="store_true", help="write the columns into the v3 table (cab_final does this)")
    ap.add_argument("--spotcheck", action="store_true", help="also write review/spotcheck_label_checks.xlsx")
    a = ap.parse_args()
    v3 = pd.read_parquet(V3)
    out, d, cal = run(v3, a.restat, a.recalibrate)
    if a.spotcheck:
        spotcheck(d, run.last["detail"], run.last["flags"], cal, examples=USER_EXAMPLES)
        log("wrote review/spotcheck_label_checks.xlsx")
    if a.write:
        out.to_parquet(V3, index=False)
        log(f"wrote {V3}")
    report(d, cal)


if __name__ == "__main__":
    main()
