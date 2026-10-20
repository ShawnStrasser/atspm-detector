"""Detector health in DuckDB SQL (note 153): the six health checks of 1.1.0 as plain SQL (ported from the numpy modules
health_core / health_v4 / health_v4_stats / health_counts, removed in 1.1.1; names below refer to them), vectorised
across signals (the signal key in every GROUP BY / PARTITION BY).  predict() runs it on its loaded events.

    health_sql(events_table="events", detectors_table="detectors", start=None, end=None) -> str   the query text
    health(con, events_table="events", detectors_table="detectors", start=None, end=None) -> DataFrame

The query reads the caller's tables where they are (no copy):
  events     DeviceId, Timestamp, EventId, Parameter (the controller log, local time)
  detectors  DeviceId, Detector, phase, function, lanes [, p_advance, p_presence, p_count, p_yellow_red, p_other,
             p_mid, p_bike] -- the classifier's answers (predicted phase, function, lane span, class probabilities)
and returns one row per detector with actuations in the window: DeviceId, Detector, health_status, health_score,
health_reason, health_bad_periods, health_categories, health_config, health_watch, health_signal_note.

Same checks, same limits (weights/health/*.json, injected as VALUES tables), same order of decisions as the numpy
version had; sums and medians are DuckDB's, so a measure may differ from it in the last bits (accepted, user 2026-10-20).
health_sql gives one WITH query (steps named hq_*); health() runs the same steps as temp tables (a step used once is
written into the step that uses it), which saves DuckDB's half second of planning the whole query.
Layout (each block a few steps):
  inputs     windows, events (allowed codes, de-duplicated), comms gaps, covered 5-min bins
  per ON     detector events in (time, ON before OFF) order with their neighbours (lag / lead); continuous ONs
  bins       5-min counts / time ON per detector (dense grid) with running sums for window sums
  stage 1    sudden silence, stuck-on, night count, level shift -> pass-1 status (twice: pass 2 without the
             detectors pass 1 calls bad as yardsticks)
  measures   15-min occupancy context, Count zone held ON, per-ON statistics, too fast / too many, hourly GEH,
             spiky bursts, holds with their queue context, busy at night
  verdict    limits, scores, resolver / severity (CASE logic), categories, reason text and bad periods
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

WDIR = Path(__file__).resolve().parent / "weights" / "health"
ALLOWED = (1, 7, 8, 9, 10, 11, 43, 44, 81, 82, 131, 150, 173)
FNS = ("Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other")
PROB = {"Advance": "p_advance", "Presence": "p_presence", "Count": "p_count", "Yellow_Red": "p_yellow_red",
        "Other": "p_other", "Mid": "p_mid", "Bike": "p_bike"}
NS5 = 300_000_000_000                    # 5 min in ns
NSH = 3_600_000_000_000                  # 1 h in ns


@lru_cache(maxsize=2)
def _json(name: str) -> dict:
    with open(WDIR / name, encoding="utf-8") as f:
        return json.load(f)


def _f(x) -> str:
    """a float as a DOUBLE literal."""
    return f"CAST({float(x)!r} AS DOUBLE)" if x == x else "CAST('nan' AS DOUBLE)"


def _s(x) -> str:
    return "'" + str(x).replace("'", "''") + "'"


def _values(cols: str, rows: list) -> str:
    """a VALUES table (cols: 'a, b, c'); rows of SQL literal text."""
    if not rows:
        n = len(cols.split(","))
        return f"SELECT * FROM (VALUES ({', '.join(['NULL'] * n)})) t({cols}) WHERE false"
    return f"SELECT * FROM (VALUES {', '.join('(' + ', '.join(r) + ')' for r in rows)}) t({cols})"


PREFIX = "hq_"


class _Q:
    """an ordered list of named steps (CTEs) -> one WITH query, or a script of temp tables.  Step names are prefixed
    with PREFIX when rendered, so they never meet a table of the caller's."""

    def __init__(self):
        self.ctes: list = []
        self.subs: dict = {}          # @EVENTS@ etc. -> the caller's table names, put in after the renaming

    def add(self, name: str, sql: str, mat: bool = False):
        self.ctes.append((name, sql.strip(), mat))

    def _renamer(self):
        names = sorted((n for n, _, _ in self.ctes), key=len, reverse=True)
        rx = re.compile(r"\b(" + "|".join(names) + r")\b")

        def rn(sql):
            sql = rx.sub(lambda m: PREFIX + m.group(1), sql)
            for k, v in self.subs.items():
                sql = sql.replace(k, v)
            return sql
        return rn

    def text(self, final: str) -> str:
        rn = self._renamer()
        body = ",\n".join(f"{PREFIX}{n} AS {'MATERIALIZED ' if m else ''}(\n{rn(s)}\n)" for n, s, m in self.ctes)
        return f"WITH {body}\n{rn(final)}"

    def script(self, final: str) -> tuple:
        """(statements, drops): a step used by more than one later step becomes a temp table, a step used once is
        written into the statement that uses it; drops[i] = the temp tables no longer needed after statement i."""
        rn = self._renamer()
        names = [n for n, _, _ in self.ctes]
        body = {n: s for n, s, _ in self.ctes}
        words = {n: set(re.findall(r"\w+", s)) for n, s in body.items()}
        words["__final"] = set(re.findall(r"\w+", final))
        uses = {n: [] for n in names}
        for n in names + ["__final"]:
            for o in names:
                if o != n and o in words[n]:
                    uses[o].append(n)
        table = {n for n in names if len(uses[n]) != 1}

        def deps(n, seen):
            out = []
            for o in names:
                if o in table or o in seen or o not in words[n]:
                    continue
                seen.add(o)
                out += deps(o, seen) + [o]
            return out
        stmts, made = [], []
        for n in names:
            if n not in table:
                continue
            inl = sorted(set(deps(n, set())), key=names.index)
            w = ("WITH " + ",\n".join(f"{PREFIX}{o} AS (\n{rn(body[o])}\n)" for o in inl) + "\n") if inl else ""
            stmts.append(f"CREATE OR REPLACE TEMP TABLE {PREFIX}{n} AS {w}{rn(body[n])}")
            made.append(n)
        inl = sorted(set(deps("__final", set())), key=names.index)
        w = ("WITH " + ",\n".join(f"{PREFIX}{o} AS (\n{rn(body[o])}\n)" for o in inl) + "\n") if inl else ""
        stmts.append(w + rn(final))
        # last statement that reads each table (directly, or through a step written into it)
        last = {}
        for i, st in enumerate(stmts):
            w = set(re.findall(r"\w+", st.split(" AS ", 1)[1] if i < len(made) else st))
            for n in made:
                if PREFIX + n in w:
                    last[n] = i
        drops = [[PREFIX + n for n in made if last.get(n, made.index(n)) == i] for i in range(len(stmts))]
        return stmts, drops


# ============================================================================ inputs and per-ON layer
def _nn(x: str) -> str:
    """a DOUBLE with NaN as NULL."""
    return f"(CASE WHEN isnan(({x})::DOUBLE) THEN NULL ELSE ({x})::DOUBLE END)"


def _inputs(q: _Q, events: str, detectors: str, windows: str | None, start, end, probabilities: bool = True,
            distinct: bool = True):
    codes = ",".join(str(c) for c in ALLOWED)
    good = f"EventId IN ({codes}) AND (EventId NOT IN (81, 82) OR Parameter BETWEEN 1 AND 64)"
    q.subs.update({"@EVENTS@": events, "@DETECTORS@": detectors, "@DISTINCT@": "DISTINCT" if distinct else ""})
    events, detectors = "@EVENTS@", "@DETECTORS@"
    if windows is not None:
        q.subs["@WINDOWS@"] = windows
        windows = "@WINDOWS@"
    if windows is None:
        # the sample window of each signal: start / end where given, else its first event and one second past its
        # last; a bound more than 30 min outside the signal's data is treated as not given (pipeline._windows)
        b0 = "w0" if start is None else str(_ns(start))
        b1 = "w1" if end is None else str(_ns(end))
        rng = []
        if start is not None:
            rng.append(f"Timestamp >= {_ts_lit(start)}")
        if end is not None:
            rng.append(f"Timestamp < {_ts_lit(end)}")
        where = " AND ".join([good, "DeviceId IS NOT NULL", "Timestamp IS NOT NULL"] + rng)
        q.add("wraw", f"""
SELECT DeviceId, epoch_ns(min(Timestamp)) AS w0, epoch_ns(max(Timestamp)) + 1000000000 AS w1, count(*) AS n
FROM {events} WHERE {where} GROUP BY DeviceId""")
        q.add("wbin", f"""
SELECT DeviceId, w0, w1, CASE WHEN {b1} > {b0} THEN {b0} ELSE w0 END AS b0,
       CASE WHEN {b1} > {b0} THEN {b1} ELSE w1 END AS b1 FROM wraw""")
        q.add("wsel", """
SELECT DeviceId, CASE WHEN w0 - b0 > 1800000000000 THEN w0 ELSE b0 END AS s_ns,
       CASE WHEN b1 - w1 > 1800000000000 THEN w1 ELSE b1 END AS e_ns FROM wbin""")
        windows = "wsel"
    # one row per signal: its window [s_ns, e_ns) (epoch ns), length T (s), 5-min bins nb5, length class wg and the
    # occupancy bin size bs (15 min, 5 min for a sample under ~1.2 h)
    q.add("win", f"""
SELECT dev, DeviceId, s_ns, e_ns, T, T / 3600.0 AS hours, ceil(T / 300.0)::INT AS nb5, wg,
       CASE WHEN wg = 'm30' THEN 300 ELSE 900 END AS bs,
       ceil(T / CASE WHEN wg = 'm30' THEN 300.0 ELSE 900.0 END)::INT AS nbb,
       make_timestamp(s_ns // 1000) AS lo, make_timestamp(e_ns // 1000 + 1) AS hi,
       s_ns % 86400000000000 AS sod
FROM (SELECT (dense_rank() OVER (ORDER BY CAST(DeviceId AS VARCHAR)))::INT AS dev, DeviceId,
             s_ns::BIGINT AS s_ns, e_ns::BIGINT AS e_ns, (e_ns - s_ns) / 1e9 AS T,
             CASE WHEN (e_ns - s_ns) / 3.6e12 < sqrt(1.5) THEN 'm30'
                  WHEN (e_ns - s_ns) / 3.6e12 < sqrt(72.0) THEN 'h3' ELSE 'h24' END AS wg
      FROM {windows})""", mat=True)
    # allowed events inside the window, de-duplicated; dn = ns from the window start
    q.add("ev", f"""
SELECT @DISTINCT@ w.dev, epoch_ns(e.Timestamp) - w.s_ns AS dn, e.EventId::INT AS eid, e.Parameter::INT AS par
FROM {events} e JOIN win w ON e.DeviceId = w.DeviceId
WHERE e.{good.replace(' Parameter', ' e.Parameter').replace('(EventId', '(e.EventId')}
  AND e.Timestamp >= w.lo AND e.Timestamp <= w.hi
  AND epoch_ns(e.Timestamp) >= w.s_ns AND epoch_ns(e.Timestamp) < w.e_ns""", mat=True)
    # the classifier's answers per detector (phase, function, lane span, class probabilities)
    pc = ", ".join((_nn(f"d.{c}") if probabilities else "NULL::DOUBLE") + f" AS {c}" for c in PROB.values())
    q.add("dcl", f"""
SELECT w.dev, d.Detector::INT AS det, {_nn('d.phase')} AS ph, CAST(d.function AS VARCHAR) AS fn,
       {_nn('d.lanes')} AS lanes, {pc}
FROM {detectors} d JOIN win w ON CAST(d.DeviceId AS VARCHAR) = CAST(w.DeviceId AS VARCHAR)""", mat=True)
    # covered 5-min bins (any event) and comms gaps (no event of any code for > 120 s)
    q.add("evm", "SELECT dev, dn // 60000000000 AS m, min(dn) AS a, max(dn) AS z FROM ev GROUP BY dev, m", mat=True)
    q.add("cov5", "SELECT DISTINCT dev, (m // 5)::INT AS b FROM evm", mat=True)
    q.add("gaps", """
SELECT dev, t AS g0 FROM (SELECT dev, z / 1e9 AS t, lead(a) OVER (PARTITION BY dev ORDER BY m) / 1e9 AS nt FROM evm)
WHERE nt - t > 120.0""", mat=True)
    # detector events in (time, ON before OFF) order; isl = OFFs before the event: a continuous ON (an ON, further
    # ONs logged without an OFF, the OFF that ends it) is one island
    q.add("de", """
SELECT dev, par AS det, dn, eid, lag(eid) OVER w AS pe, lag(dn) OVER w AS pdn, lead(eid) OVER w AS ne,
       lead(dn) OVER w AS ndn, lag(eid, 2) OVER w AS ppe, lag(dn, 2) OVER w AS p2dn, lag(eid, 3) OVER w AS p3e,
       lead(eid, 2) OVER w AS nne, lead(dn, 2) OVER w AS n2dn
FROM ev WHERE eid IN (81, 82) WINDOW w AS (PARTITION BY dev, par ORDER BY dn, eid DESC)""", mat=True)
    q.add("dets", "SELECT DISTINCT dev, det FROM de", mat=True)
    # continuous ONs (an ON after a non-ON, further ONs logged without an OFF, the OFF that ends it): start s, end f
    # (the OFF, else the window end), open (no OFF), multi (ONs logged again inside it).  Single-ON ones straight from
    # the event order; the few with re-logged ONs are numbered among their own events.
    q.add("mst", "SELECT dev, det, dn FROM de WHERE eid = 82 AND pe IS DISTINCT FROM 82 AND ne = 82", mat=True)
    q.add("multis", """
SELECT s.dev, s.det, s.dn AS on_dn, e.dn AS off_dn
FROM mst s LEFT JOIN (SELECT e.dev, e.det, e.dn, m.dn AS sdn
                      FROM (SELECT dev, det, dn FROM de WHERE eid = 81 AND pe = 82 AND ppe = 82) e
                      ASOF JOIN mst m ON m.dev = e.dev AND m.det = e.det AND e.dn >= m.dn) e
  ON e.dev = s.dev AND e.det = s.det AND e.sdn = s.dn""")
    q.add("isl0", """
SELECT d.dev, d.det, d.dn / 1e9 AS s, coalesce(d.ndn / 1e9, w.T) AS f, d.ne IS NULL AS open_end, false AS multi
FROM de d JOIN win w USING (dev) WHERE d.eid = 82 AND d.pe IS DISTINCT FROM 82 AND d.ne IS DISTINCT FROM 82
UNION ALL
SELECT m.dev, m.det, m.on_dn / 1e9, coalesce(m.off_dn / 1e9, w.T), m.off_dn IS NULL, true
FROM multis m JOIN win w USING (dev)""")
    q.add("ons", """
SELECT c.dev, c.det, c.s, c.f, c.open_end, c.multi, c.f - c.s AS dur, false AS gap FROM isl0 c WHERE c.f - c.s <= 60
UNION ALL
SELECT c.dev, c.det, c.s, c.f, c.open_end, c.multi, c.f - c.s, coalesce(g.g0 < c.f, false)
FROM (SELECT * FROM isl0 WHERE f - s > 60) c ASOF LEFT JOIN gaps g ON c.dev = g.dev AND g.g0 >= c.s""", mat=True)
    # a channel whose first event is an OFF was ON when the window opened
    q.add("lead0", "SELECT dev, det, dn / 1e9 AS f FROM de WHERE eid = 81 AND pe IS NULL")


def _ns(x) -> int:
    t = pd.Timestamp(x)
    t = t.tz_localize(None) if t.tzinfo is not None else t
    return int(t.value)


def _ts_lit(x) -> str:
    t = pd.Timestamp(x)
    t = t.tz_localize(None) if t.tzinfo is not None else t
    return f"TIMESTAMP '{t.isoformat(sep=' ')}'"


# ============================================================================ 5-min bins
def _bins(q: _Q):
    # time ON split across 5-min bins: continuous ONs not over a comms gap + the ON at the window start
    q.add("occ5", """
WITH o AS (SELECT o.dev, o.det, o.s, o.f, floor(o.s / 300.0)::INT AS b0, least(floor(o.f / 300.0)::INT, w.nb5 - 1) AS b1
           FROM (SELECT dev, det, s, f FROM ons WHERE NOT gap UNION ALL SELECT dev, det, 0.0, f FROM lead0) o
           JOIN win w USING (dev))
SELECT dev, det, b, sum(occ) AS occ FROM (
  SELECT dev, det, b0 AS b, greatest(least(f, (b0 + 1) * 300.0) - s, 0.0) AS occ FROM o WHERE b1 = b0
  UNION ALL
  SELECT dev, det, b, greatest(least(f, (b + 1) * 300.0) - greatest(s, b * 300.0), 0.0)
  FROM (SELECT dev, det, s, f, unnest(range(b0, b1 + 1)) AS b FROM o WHERE b1 > b0))
GROUP BY dev, det, b""")
    q.add("n5", f"SELECT dev, det, (dn // {NS5})::INT AS b, count(*) AS n FROM de WHERE eid = 82 GROUP BY dev, det, b")
    q.add("bins5", f"""
SELECT b.dev, b.b, c.b IS NOT NULL AS cov, ((b.s_ns + b.b * {NS5}) // {NSH}) % 24 AS hr
FROM (SELECT dev, s_ns, unnest(range(nb5)) AS b FROM win) b LEFT JOIN cov5 c ON c.dev = b.dev AND c.b = b.b""",
          mat=True)
    # dense detector x 5-min grid: count n, time ON occ (s), covered
    q.add("g5", """
SELECT d.dev, d.det, b.b, b.cov, b.hr, coalesce(n.n, 0)::DOUBLE AS n, coalesce(o.occ, 0.0) AS occ
FROM dets d JOIN bins5 b USING (dev) LEFT JOIN n5 n ON n.dev = d.dev AND n.det = d.det AND n.b = b.b
LEFT JOIN occ5 o ON o.dev = d.dev AND o.det = d.det AND o.b = b.b""", mat=True)
    # running sums at the bin boundaries k = 0 .. nb5 (sum over bins < k): counts cn, quiet bins cq (no ON, < 1 s ON);
    # looked up one boundary per CTE (_at), so no join order can pair two lookups of the same detector
    q.add("k5", """
SELECT dev, det, b + 1 AS k, sum(n) OVER w AS cn, sum((n = 0 AND occ < 1)::INT) OVER w AS cq
FROM g5 WINDOW w AS (PARTITION BY dev, det ORDER BY b ROWS UNBOUNDED PRECEDING)
UNION ALL SELECT dev, det, 0, 0.0, 0 FROM dets""", mat=True)
    # per detector: ONs on covered bins, active, live, day / night ONs
    q.add("dstat", """
SELECT g.dev, g.det, sum(g.n * g.cov::INT) AS n_on, (sum(g.n) > 0 OR sum(g.occ) > 0) AS active,
       sum(g.n * g.cov::INT) > 0 AS live,
       sum(g.n * (g.cov AND g.hr >= 7 AND g.hr < 19)::INT) AS dy, sum(g.n * (g.cov AND (g.hr >= 21 OR g.hr < 5))::INT) AS nt
FROM g5 g GROUP BY g.dev, g.det""", mat=True)
    q.add("sig0", """
SELECT s.dev, s.n_cov, s.last_cov, s.ok_night, a.n_active
FROM (SELECT dev, count(*) FILTER (cov) AS n_cov, coalesce(max(b) FILTER (cov), -1) AS last_cov,
             count(*) FILTER (cov AND (hr >= 21 OR hr < 5)) >= 24 AND count(*) FILTER (cov AND hr >= 7 AND hr < 19) >= 24
             AS ok_night FROM bins5 GROUP BY dev) s
JOIN (SELECT dev, sum(active::INT) AS n_active FROM dstat GROUP BY dev) a USING (dev)""", mat=True)
    # twins: two channels carrying (almost) the same 5-min counts = one zone wired to two inputs
    q.add("twin", """
SELECT a.dev, a.det AS i, b.det AS j
FROM dstat a JOIN dstat b ON a.dev = b.dev AND a.det <> b.det AND a.n_on > 0 AND b.n_on > 0
     AND abs(a.n_on - b.n_on) < 0.1 * greatest(a.n_on + b.n_on, 1)
JOIN g5 x ON x.dev = a.dev AND x.det = a.det JOIN g5 y ON y.dev = a.dev AND y.det = b.det AND y.b = x.b
GROUP BY a.dev, a.det, b.det, a.n_on, b.n_on
HAVING sum(abs(x.n * x.cov::INT - y.n * y.cov::INT)) / greatest(a.n_on + b.n_on, 1) < 0.1""", mat=True)
    # 15-min bins of whole 5-min triples: counts, all three covered (partner correlation, level shift)
    q.add("a15", """
SELECT g.dev, g.det, g.b // 3 AS k, sum(g.n) AS n15, bool_and(g.cov) AS c15
FROM g5 g JOIN win w USING (dev) WHERE g.b < w.nb5 // 3 * 3 GROUP BY g.dev, g.det, g.b // 3""", mat=True)
    # quiet runs (no ON, < 1 s ON, covered) of every detector with ONs: [a, z) in 5-min bins
    q.add("runs", """
SELECT dev, det, min(b) AS a, max(b) + 1 AS z FROM (
  SELECT g.dev, g.det, g.b, g.b - row_number() OVER (PARTITION BY g.dev, g.det ORDER BY g.b) AS grp
  FROM g5 g JOIN dstat s USING (dev, det) WHERE s.n_on > 0 AND g.n = 0 AND g.occ < 1 AND g.cov)
GROUP BY dev, det, grp""", mat=True)
    # level shift of the share of the signal's traffic (15-min, best single change point, Poisson LLR)
    q.add("lvl0", """
SELECT dev, det, k, n15, S15, c15,
       sum(n15) OVER w AS cx, sum(S15) OVER w AS cs, row_number() OVER w AS pos,
       count(*) OVER (PARTITION BY dev, det) AS nl, sum(n15) OVER (PARTITION BY dev, det) AS X,
       sum(S15) OVER (PARTITION BY dev, det) AS SS
FROM (SELECT dev, det, k, n15, sum(n15) OVER (PARTITION BY dev, k) - n15 AS S15, c15 FROM a15) WHERE c15
WINDOW w AS (PARTITION BY dev, det ORDER BY k ROWS UNBOUNDED PRECEDING)""")
    q.add("lvlshift", """
SELECT dev, det, first(llr ORDER BY llr DESC, pos) AS level_llr, first(ratio ORDER BY llr DESC, pos) AS level_ratio,
       first(pos * 3 ORDER BY llr DESC, pos) AS level_b
FROM (SELECT dev, det, pos,
        (CASE WHEN x1 > 0 THEN x1 * ln(greatest(p1, 1e-12)) ELSE 0 END - p1 * s1)
        + (CASE WHEN x2 > 0 THEN x2 * ln(greatest(p2, 1e-12)) ELSE 0 END - p2 * s2)
        - (X * ln(greatest(p0, 1e-12)) - p0 * SS) AS llr,
        ((x2 + 0.5) / greatest(s2, 1)) / ((x1 + 0.5) / greatest(s1, 1)) AS ratio
      FROM (SELECT *, cx AS x1, cs AS s1, X - cx AS x2, SS - cs AS s2,
                   CASE WHEN cs > 0 THEN cx / greatest(cs, 1e-9) ELSE 0 END AS p1,
                   CASE WHEN SS - cs > 0 THEN (X - cx) / greatest(SS - cs, 1e-9) ELSE 0 END AS p2,
                   X / greatest(SS, 1) AS p0
            FROM lvl0 WHERE nl >= 8 AND X >= 20 AND pos >= 2 AND pos <= nl - 2))
GROUP BY dev, det""")
    # continuous ONs held >= 5 min, not over a comms gap (stuck-on candidates); co = other detectors of the signal
    # held ON (>= 5 min) through at least half of it
    q.add("hold", """
SELECT h.*, row_number() OVER (ORDER BY h.dev, h.det, h.s) AS hid FROM (
  SELECT dev, det, s, f, open_end, dur FROM ons WHERE dur >= 300 AND NOT gap) h""", mat=True)
    q.add("hold_co", """
SELECT a.hid, count(DISTINCT b.det) FILTER (least(a.f, b.f) - greatest(a.s, b.s) >= 0.5 * a.dur) AS co
FROM hold a LEFT JOIN hold b ON a.dev = b.dev AND a.det <> b.det GROUP BY a.hid""", mat=True)


# ============================================================================ stage 1 (health_core.health)
def _sc1(x: str, lo: str, hi: str) -> str:
    """health_core._score: 0 below lo, .35 at lo rising to 1 at hi; NULL stays NULL."""
    return (f"(CASE WHEN ({x}) IS NULL OR isnan({x}) THEN NULL WHEN ({x}) < {lo} THEN 0.0 "
            f"ELSE 0.35 + 0.65 * least(greatest((({x}) - {lo}) / ({hi} - {lo}), 0.0), 1.0) END)")


def _at(q: _Q, name: str, src: str, who: str, at: tuple, col: str):
    """{name}_1 .. {name}_n: `src` with k5's running sum `col` of detector `who` at each boundary column of `at` added as
    {col}_{boundary} -- one lookup per CTE."""
    prev = src
    for i, b in enumerate(at, 1):
        q.add(f"{name}_{i}", f"""
SELECT x.*, k.{col} AS {col}_{b} FROM {prev} x JOIN k5 k ON k.dev = x.dev AND k.det = x.{who} AND k.k = x.{b}""",
              mat=True)
        prev = f"{name}_{i}"


def _stage1(q: _Q, p: str):
    """the four stage-1 rules for the detectors of {p}ph (dev, det, ph): references, silent runs, night count against
    the detector it tracks, stuck-on holds with their recovery -> {p}st (one row per detector)."""
    # reference candidates: the other live detectors of the signal, never a twin; same = on the same phase
    q.add(f"{p}rf", f"""
SELECT a.dev, a.det, b.det AS r, coalesce(a.ph = b.ph, false) AS same
FROM {p}ph a JOIN {p}ph b ON a.dev = b.dev AND a.det <> b.det
JOIN dstat sb ON sb.dev = b.dev AND sb.det = b.det AND sb.live
LEFT JOIN twin t ON t.dev = a.dev AND t.i = a.det AND t.j = b.det WHERE t.i IS NULL""", mat=True)
    # yardstick kind: the other detectors on its phase when there are >= 2, else the rest of the signal
    q.add(f"{p}kd", f"""
SELECT a.dev, a.det, a.ph, coalesce(r.nsame, 0) >= 2 AS isph
FROM {p}ph a LEFT JOIN (SELECT dev, det, count(*) FILTER (same) AS nsame FROM {p}rf GROUP BY dev, det) r
USING (dev, det)""", mat=True)
    q.add(f"{p}pref", f"""
SELECT f.dev, f.det, f.r FROM {p}rf f JOIN {p}kd k USING (dev, det) WHERE k.isph AND f.same""", mat=True)
    # recovery partner candidates: every other live detector on the predicted phase, twins included
    q.add(f"{p}pc", f"""
SELECT a.dev, a.det, b.det AS c FROM {p}ph a JOIN {p}ph b ON a.dev = b.dev AND a.det <> b.det AND a.ph = b.ph
JOIN dstat sb ON sb.dev = b.dev AND sb.det = b.det AND sb.live""", mat=True)
    # reference members of every detector for window sums: phase refs, else the rest of the signal
    q.add(f"{p}rm", f"""
SELECT dev, det, r FROM {p}pref
UNION ALL SELECT k.dev, k.det, d.det FROM {p}kd k JOIN dets d ON d.dev = k.dev AND d.det <> k.det WHERE NOT k.isph""",
          mat=True)
    # silent runs: expected ONs lam = n_on / (reference ONs outside the run) x reference ONs inside it
    q.add(f"{p}rs0", f"""
SELECT u.dev, u.det, u.a, u.z, m.r, w.nb5 FROM runs u JOIN {p}rm m ON m.dev = u.dev AND m.det = u.det
JOIN win w ON w.dev = u.dev""", mat=True)
    _at(q, f"{p}rs", f"{p}rs0", "r", ("a", "z", "nb5"), "cn")
    q.add(f"{p}rs", f"""
SELECT dev, det, a, z, sum(cn_z - cn_a) AS s_in, sum(cn_nb5) AS s_tot FROM {p}rs_3 GROUP BY dev, det, a, z""")
    q.add(f"{p}dr", f"""
SELECT k.dev, k.det, coalesce(d.lam, 0.0) AS drop_lam, coalesce(d.b0, -1) AS drop_b0, coalesce(d.b1, -1) AS drop_b1
FROM {p}kd k LEFT JOIN (
  SELECT dev, det, max(lam) AS lam, first(a ORDER BY lam DESC, a) AS b0, first(z ORDER BY lam DESC, a) AS b1
  FROM (SELECT r.*, s.n_on / greatest(r.s_tot - r.s_in, 1) * r.s_in AS lam FROM {p}rs r JOIN dstat s USING (dev, det))
  WHERE lam > 0 GROUP BY dev, det) d USING (dev, det)""", mat=True)
    q.add(f"{p}cs0", f"""
SELECT d.dev, d.det, d.drop_b0, d.drop_b1, x.det AS r, x.active FROM {p}dr d JOIN dstat x ON x.dev = d.dev
WHERE d.drop_b1 - d.drop_b0 >= 3""", mat=True)
    _at(q, f"{p}cs", f"{p}cs0", "r", ("drop_b0", "drop_b1"), "cq")
    q.add(f"{p}drx", f"""
SELECT d.*, s.n_on, (d.drop_b1 = w.nb5 OR (d.drop_b1 > 0 AND g.last_cov < d.drop_b1)) AS drop_to_end,
       coalesce(c.co_silent, 0) AS co_silent
FROM {p}dr d JOIN dstat s USING (dev, det) JOIN win w ON w.dev = d.dev JOIN sig0 g ON g.dev = d.dev
LEFT JOIN (SELECT dev, det, count(*) FILTER (active AND (cq_drop_b1 - cq_drop_b0) / (drop_b1 - drop_b0) >= 0.8) - 1
                  AS co_silent FROM {p}cs_2 GROUP BY dev, det) c ON c.dev = d.dev AND c.det = d.det""", mat=True)
    # partner requests: the whole sample (night), each stuck-on hold >= 15 min and each silent run (left out);
    # [xa, xb) = 15-min bins left out
    q.add(f"{p}rq", f"""
SELECT 'n' AS kind, dev * 128 + det AS id, dev, det, 0 AS xa, 0 AS xb FROM {p}kd
UNION ALL
SELECT 'e', h.hid, h.dev, h.det, floor(h.s / 300.0)::INT // 3, ceil(least(ceil(h.f / 300.0), w.nb5) / 3.0)::INT
FROM hold h JOIN {p}kd k USING (dev, det) JOIN win w ON w.dev = h.dev WHERE h.dur >= 900
UNION ALL
SELECT 'd', dev * 128 + det, dev, det, drop_b0 // 3, ceil(drop_b1 / 3.0)::INT FROM {p}drx
WHERE n_on > 0 AND drop_lam >= 30 AND drop_b1 > drop_b0""", mat=True)
    q.add(f"{p}rc", f"""
SELECT r.kind, r.id, c.c FROM {p}rq r JOIN {p}pc c ON c.dev = r.dev AND c.det = r.det
WHERE r.kind <> 'e' OR NOT EXISTS (
  SELECT 1 FROM hold e JOIN hold o ON o.dev = e.dev AND o.det = c.c AND o.dur >= 900 AND o.s < e.f AND o.f > e.s
  WHERE e.hid = r.id)""")
    q.add(f"{p}pxy", f"""
SELECT r.kind, r.id, c.c, x.n15 AS x, y.n15 AS y, x.k + 1 AS w
FROM {p}rq r JOIN {p}rc c ON c.kind = r.kind AND c.id = r.id
JOIN a15 x ON x.dev = r.dev AND x.det = r.det AND x.c15 AND NOT (x.k >= r.xa AND x.k < r.xb)
JOIN a15 y ON y.dev = r.dev AND y.det = c.c AND y.k = x.k""")
    # the partner: best 15-min correlation (ties: its own counts, never its channel order), kept from 0.7
    q.add(f"{p}part", f"""
SELECT kind, id, p FROM (
  SELECT kind, id, first(c ORDER BY round(cr, 12) DESC, ys DESC, yw DESC, c) AS p,
         first(cr ORDER BY round(cr, 12) DESC, ys DESC, yw DESC, c) AS pcr
  FROM (SELECT kind, id, c, corr(y, x) AS cr, sum(y) AS ys, sum(y * w) AS yw, count(*) AS m,
               min(x) < max(x) AS xv, min(y) < max(y) AS yv FROM {p}pxy GROUP BY kind, id, c)
  WHERE m >= 16 AND xv AND yv GROUP BY kind, id) WHERE pcr >= 0.7""", mat=True)
    # stuck-on holds >= 15 min and silent runs: share after / before (6 h each side, >= 3 h of data)
    q.add(f"{p}rr", f"""
SELECT 'e' AS kind, h.hid AS id, h.dev, h.det, floor(h.s / 300.0)::INT AS a, least(ceil(h.f / 300.0)::INT, w.nb5) AS b
FROM hold h JOIN {p}kd k USING (dev, det) JOIN win w ON w.dev = h.dev WHERE h.dur >= 900 AND NOT h.open_end
UNION ALL
SELECT 'd', dev * 128 + det, dev, det, drop_b0, drop_b1 FROM {p}drx
WHERE n_on > 0 AND drop_lam >= 30 AND drop_b1 > drop_b0 AND NOT drop_to_end""")
    q.add(f"{p}rr2", f"""
SELECT r.*, greatest(r.a - 72, 0) AS lo0, least(r.b + 72, w.nb5) AS hi1, pa.p,
       (r.a >= 36 AND w.nb5 - r.b >= 36) AS can
FROM {p}rr r JOIN win w ON w.dev = r.dev LEFT JOIN {p}part pa ON pa.kind = r.kind AND pa.id = r.id""", mat=True)
    q.add(f"{p}rec", f"""
SELECT kind, id, CASE WHEN can AND c0 >= 36 AND c1 >= 36 THEN ((x1 + 0.5) / greatest(s1, 1.0)) /
                      ((x0 + 0.5) / greatest(s0, 1.0)) END AS rec
FROM (
  SELECT r.kind, r.id, any_value(r.can) AS can,
         count(*) FILTER (mm.own AND g.cov AND g.b < r.a) AS c0, count(*) FILTER (mm.own AND g.cov AND g.b >= r.b) AS c1,
         coalesce(sum(g.n) FILTER (mm.own AND g.b < r.a), 0) AS x0, coalesce(sum(g.n) FILTER (mm.own AND g.b >= r.b), 0) AS x1,
         coalesce(sum(g.n) FILTER (NOT mm.own AND g.b < r.a), 0) AS s0,
         coalesce(sum(g.n) FILTER (NOT mm.own AND g.b >= r.b), 0) AS s1
  FROM {p}rr2 r
  JOIN (SELECT r.kind, r.id, r.det AS m, true AS own FROM {p}rr2 r
        UNION ALL SELECT r.kind, r.id, r.p, false FROM {p}rr2 r WHERE r.p IS NOT NULL
        UNION ALL SELECT r.kind, r.id, m.r, false FROM {p}rr2 r JOIN {p}rm m ON m.dev = r.dev AND m.det = r.det
        WHERE r.p IS NULL) mm ON mm.kind = r.kind AND mm.id = r.id
  JOIN g5 g ON g.dev = r.dev AND g.det = mm.m AND ((g.b >= r.lo0 AND g.b < r.a) OR (g.b >= r.b AND g.b < r.hi1))
  GROUP BY r.kind, r.id)""", mat=True)
    # stuck-on per detector: the longest hold >= 15 min (first in time), its shared count and recovery
    q.add(f"{p}ep", f"""
SELECT h.dev, h.det, max(h.dur) AS ep_dur, first(h.s ORDER BY h.dur DESC, h.s) AS ep_t0,
       first(h.f ORDER BY h.dur DESC, h.s) AS ep_t1, first(c.co ORDER BY h.dur DESC, h.s) AS ep_co,
       first(r.rec ORDER BY h.dur DESC, h.s) AS ep_rec
FROM hold h JOIN {p}kd k USING (dev, det) JOIN hold_co c ON c.hid = h.hid
LEFT JOIN {p}rec r ON r.kind = 'e' AND r.id = h.hid WHERE h.dur >= 900 GROUP BY h.dev, h.det""")
    # silent run inside a stuck-on hold (>= half of it): not a silence
    q.add(f"{p}dis", f"""
SELECT d.dev, d.det, coalesce(sum(greatest(least(h.f, d.drop_b1 * 300.0) - greatest(h.s, d.drop_b0 * 300.0), 0.0))
         FILTER (h.hid IS NOT NULL), 0)
       >= 0.5 * (d.drop_b1 - d.drop_b0) * 300.0 AS drop_in_stuck
FROM {p}drx d LEFT JOIN hold h ON h.dev = d.dev AND h.det = d.det AND h.dur >= 900
WHERE d.n_on > 0 AND d.drop_lam >= 30 AND d.drop_b1 > d.drop_b0 GROUP BY d.dev, d.det, d.drop_b0, d.drop_b1""")
    # night count (21-05) against the detector it tracks, else its phase, else the rest of the signal
    q.add(f"{p}rsum", f"""
SELECT m.dev, m.det, sum(x.dy) AS rd, sum(x.nt) AS rn FROM {p}rm m JOIN dstat x ON x.dev = m.dev AND x.det = m.r
GROUP BY m.dev, m.det""")
    q.add(f"{p}nt", f"""
SELECT dev, det, nt AS night_n, ex AS night_exp, (nt + 0.5) / (ex + 0.5) AS night_ratio, who FROM (
  SELECT k.dev, k.det, s.nt, s.dy / greatest(coalesce(ps.dy, r.rd, 0), 1) * coalesce(ps.nt, r.rn, 0) AS ex,
         CASE WHEN pa.p IS NOT NULL THEN 'd' || pa.p || ', which it tracks'
              WHEN k.isph THEN 'the other detectors on its phase' ELSE 'the rest of the signal' END AS who
  FROM {p}kd k JOIN dstat s USING (dev, det) JOIN sig0 g ON g.dev = k.dev
  LEFT JOIN {p}part pa ON pa.kind = 'n' AND pa.id = k.dev * 128 + k.det
  LEFT JOIN dstat ps ON ps.dev = k.dev AND ps.det = pa.p LEFT JOIN {p}rsum r ON r.dev = k.dev AND r.det = k.det
  WHERE g.ok_night AND s.dy >= 20) WHERE ex >= 30""")
    # the four rules -> pass status
    sd = _sc1("d.drop_lam", "30.0", "100.0")
    ss = _sc1("coalesce(e.ep_dur, 0)", "900.0", "3600.0")
    q.add(f"{p}st", f"""
SELECT *, CASE WHEN hs < 0.25 THEN 'bad' WHEN hs < 0.70 THEN 'suspect' WHEN n_on < 20 THEN 'not_enough_data'
               ELSE 'ok' END AS status
FROM (SELECT *, (1 - coalesce(s_dropout, 0)) * (1 - s_stuck) * (1 - coalesce(s_night_drop, 0))
                * (1 - coalesce(s_level, 0)) AS hs
  FROM (SELECT d.dev, d.det, d.n_on, d.drop_lam, d.drop_b0, d.drop_b1, d.drop_to_end, d.co_silent,
          coalesce(i.drop_in_stuck, false) AS drop_in_stuck, rd.rec AS drop_rec, e.ep_dur, e.ep_t0, e.ep_t1, e.ep_co,
          e.ep_rec, n.night_ratio, n.night_n, n.night_exp, n.who AS night_ref, l.level_ratio,
          coalesce(l.level_llr, 0.0) AS level_llr, l.level_b,
          CASE WHEN d.n_on > 0 THEN (
            CASE WHEN coalesce(i.drop_in_stuck, false) THEN 0.0
                 WHEN {sd} > 0 AND NOT d.drop_to_end AND rd.rec BETWEEN 0.5 AND 2.0 THEN least({sd}, 0.35)
                 ELSE {sd} END) END AS s_dropout,
          CASE WHEN coalesce(e.ep_dur, 0) >= 900 AND (e.ep_co >= 3 OR e.ep_rec BETWEEN 0.5 AND 2.0)
               THEN least({ss}, 0.35) ELSE {ss} END AS s_stuck,
          CASE WHEN n.night_ratio IS NOT NULL THEN
               {_sc1('-ln(greatest(n.night_ratio, 1e-6))', '-ln(0.19)', '-ln(0.063)')} END AS s_night_drop,
          CASE WHEN l.level_ratio IS NOT NULL THEN CASE WHEN l.level_llr <= 50 THEN 0.0
               ELSE {_sc1('-ln(l.level_ratio)', '-ln(0.15)', '-ln(0.05)')} END END AS s_level
        FROM {p}drx d LEFT JOIN {p}dis i USING (dev, det)
        LEFT JOIN {p}rec rd ON rd.kind = 'd' AND rd.id = d.dev * 128 + d.det
        LEFT JOIN {p}ep e USING (dev, det) LEFT JOIN {p}nt n USING (dev, det) LEFT JOIN lvlshift l USING (dev, det)))""",
          mat=True)


def _stage1_both(q: _Q):
    """pass 1 on every detector; pass 2 (signals with a detector pass 1 calls bad, and a predicted phase) without the
    bad detectors as anybody's phase mate; a bad detector keeps its pass-1 row."""
    q.add("p1ph", "SELECT d.dev, d.det, c.ph FROM dets d LEFT JOIN dcl c USING (dev, det)", mat=True)
    _stage1(q, "p1")
    q.add("p2sig", """
SELECT DISTINCT s.dev FROM p1st s WHERE s.status = 'bad'
  AND s.dev IN (SELECT dev FROM dcl WHERE ph IS NOT NULL AND NOT isnan(ph))""", mat=True)
    q.add("p2ph", """
SELECT p.dev, p.det, CASE WHEN s.status = 'bad' THEN NULL ELSE p.ph END AS ph
FROM p1ph p JOIN p1st s USING (dev, det) WHERE p.dev IN (SELECT dev FROM p2sig)""", mat=True)
    _stage1(q, "p2")
    q.add("stage1", """
SELECT x.*, a.status AS status1, a.hs AS score1 FROM (
  SELECT * FROM p1st WHERE dev NOT IN (SELECT dev FROM p2sig) OR status = 'bad'
  UNION ALL SELECT b.* FROM p2st b JOIN p1st a USING (dev, det) WHERE a.status <> 'bad') x
JOIN p1st a USING (dev, det)""", mat=True)


# ============================================================================ measures (health_v4_stats, health_counts)
C7 = ("Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other")


def _in(xs) -> str:
    return "(" + ", ".join(_s(x) for x in xs) + ")"


def _sgeh(m: str, c: str) -> str:
    """signed GEH: + when m < c (under-counting)."""
    return (f"(CASE WHEN ({m}) < ({c}) THEN 1 ELSE -1 END * sqrt(2 * (({m}) - ({c})) ^ 2 / "
            f"greatest(({m}) + ({c}), 1e-9)))")


def _measures(q: _Q, LJ: dict, R: dict):
    """the per-detector measures the six checks read (the scored table X) and the holds (E)."""
    alts = ", ".join(f"CASE WHEN c.{PROB[f]} >= 0.15 THEN {_s(f)} END" for f in C7)
    top = "greatest(" + ", ".join(f"c.{PROB[f]}" for f in C7) + ")"
    # ---- the scored table: detectors with a predicted function the checks know
    q.add("xbase", f"""
SELECT s.*, c.fn, c.ph, c.lanes, w.hours, w.wg, w.bs, CASE WHEN c.lanes >= 2 THEN '2+' ELSE '1' END AS span,
       {top} AS f_top,
       CASE WHEN {top} < 0.7 THEN list_filter([{alts}], x -> x IS NOT NULL) ELSE []::VARCHAR[] END AS alts
FROM stage1 s JOIN dcl c USING (dev, det) JOIN win w USING (dev) WHERE c.fn IN {_in(FNS)}""", mat=True)
    # every detector with its answers (phase, function, traffic zone, lanes) and pass-1 status
    q.add("dz", f"""
SELECT d.dev, d.det, c.ph, c.fn, coalesce(c.fn IN {_in(('Advance', 'Count'))}, false) AS tr,
       greatest(coalesce(c.lanes, 1.0), 1.0) AS ln1, c.lanes, s.status1
FROM dets d LEFT JOIN dcl c USING (dev, det) LEFT JOIN stage1 s USING (dev, det)""", mat=True)
    # ---- occupancy bins of bs seconds (15 min; 5 min under ~1.2 h): count n, fraction ON occ (NULL = not covered)
    q.add("gb1", """
SELECT g.dev, g.det, g.k, CASE WHEN c.k IS NOT NULL THEN g.n END AS n,
       CASE WHEN c.k IS NOT NULL THEN g.occs / w.bs END AS occ, c.k IS NOT NULL AS cov
FROM (SELECT g.dev, g.det, g.b // (w.bs // 300) AS k, sum(g.n) AS n, sum(g.occ) AS occs
      FROM g5 g JOIN win w USING (dev) GROUP BY g.dev, g.det, g.b // (w.bs // 300)) g JOIN win w USING (dev)
LEFT JOIN (SELECT DISTINCT c.dev, c.b // (w.bs // 300) AS k FROM cov5 c JOIN win w USING (dev)) c
  ON c.dev = g.dev AND c.k = g.k""")
    # traffic yardstick of each detector: its phase's Advance / Count zones, else the signal's, else all others
    q.add("rmm", """
SELECT dev, det, r FROM (
  SELECT a.dev, a.det, b.det AS r, coalesce(a.ph = b.ph, false) AND b.tr AS s1, b.tr AS s2,
         bool_or(coalesce(a.ph = b.ph, false) AND b.tr) OVER (PARTITION BY a.dev, a.det) AS h1,
         bool_or(b.tr) OVER (PARTITION BY a.dev, a.det) AS h2
  FROM dz a JOIN dz b ON a.dev = b.dev AND a.det <> b.det)
WHERE (h1 AND s1) OR (NOT h1 AND h2 AND s2) OR (NOT h1 AND NOT h2)""", mat=True)
    q.add("gb", """
SELECT g.*, CASE WHEN g.cov THEN r.ref END AS ref, CASE WHEN p.mocc > 0.01 THEN g.occ / p.mocc END AS rel, p.mocc,
       z.ph, z.fn, z.status1, (z.status1 IS DISTINCT FROM 'bad' AND g.occ IS NOT NULL) AS hl
FROM gb1 g JOIN (SELECT dev, det, avg(occ) AS mocc FROM gb1 GROUP BY dev, det) p USING (dev, det)
JOIN dz z USING (dev, det)
LEFT JOIN (SELECT m.dev, m.det, g.k, sum(coalesce(g.n, 0)) AS ref FROM rmm m JOIN gb1 g ON g.dev = m.dev AND g.det = m.r
           GROUP BY m.dev, m.det, g.k) r USING (dev, det, k)""", mat=True)
    # congestion: median of the other detectors' ON fraction relative to their own mean (signals with >= 4)
    q.add("gcong", """
SELECT dev, det, k, CASE WHEN m > 0 THEN (L[(CASE WHEN lo < p THEN lo ELSE lo + 1 END) + 1]
                                          + L[(CASE WHEN h < p THEN h ELSE h + 1 END) + 1]) / 2.0 END AS cong
FROM (SELECT *, m // 2 AS h, CASE WHEN m % 2 = 1 THEN m // 2 ELSE m // 2 - 1 END AS lo FROM (
  SELECT g.dev, g.det, g.k, l.L, CASE WHEN g.rel IS NULL THEN len(l.L) ELSE len(list_filter(l.L, x -> x < g.rel)) END AS p,
         len(l.L) - (g.rel IS NOT NULL)::INT AS m
  FROM gb g JOIN (SELECT dev, k, list_sort(list(rel) FILTER (rel IS NOT NULL)) AS L FROM gb GROUP BY dev, k) l
  USING (dev, k) WHERE g.dev IN (SELECT dev FROM dets GROUP BY dev HAVING count(*) >= 4)))""")
    # peer: mean ON fraction of the healthy phase mates; same: of the healthy phase mates of the same function
    q.add("gctx", """
SELECT g.*, c.cong,
  CASE WHEN g.ph IS NOT NULL THEN (a.so - coalesce(CASE WHEN g.hl THEN g.occ END, 0)) / nullif(a.sh - g.hl::INT, 0)
       END AS peer,
  CASE WHEN g.ph IS NOT NULL AND g.fn IS NOT NULL THEN (b.so - coalesce(CASE WHEN g.hl THEN g.occ END, 0))
       / nullif(b.sh - g.hl::INT, 0) END AS same
FROM gb g LEFT JOIN gcong c USING (dev, det, k)
LEFT JOIN (SELECT dev, ph, k, sum(occ) FILTER (hl) AS so, count(*) FILTER (hl) AS sh FROM gb WHERE ph IS NOT NULL
           GROUP BY dev, ph, k) a ON a.dev = g.dev AND a.ph = g.ph AND a.k = g.k
LEFT JOIN (SELECT dev, ph, fn, k, sum(occ) FILTER (hl) AS so, count(*) FILTER (hl) AS sh FROM gb
           WHERE ph IS NOT NULL AND fn IS NOT NULL GROUP BY dev, ph, fn, k) b
  ON b.dev = g.dev AND b.ph = g.ph AND b.fn = g.fn AND b.k = g.k""", mat=True)
    # count / time-ON elasticity vs the traffic yardstick on its busier half (queue pattern)
    q.add("slp", """
SELECT dev, det, CASE WHEN one >= 6 AND vx > 0.5 THEN bn END AS elhi_cnt,
       CASE WHEN one >= 6 AND vx > 0.5 THEN bo END AS elhi_occ
FROM (SELECT dev, det, count(*) AS one, regr_slope(lnn, lr) AS bn, regr_slope(lo, lr) AS bo,
             var_pop(lr) * count(*) AS vx
      FROM (SELECT g.dev, g.det, ln(g.ref + 1) AS lr, ln(g.n + 1) AS lnn, ln(g.occ + 0.005) AS lo
            FROM gb g JOIN win w USING (dev) JOIN (SELECT dev, det, median(ref) FILTER (n IS NOT NULL AND ref IS NOT NULL)
                                                   AS rmed FROM gb GROUP BY dev, det) m USING (dev, det)
            WHERE w.wg <> 'm30' AND g.n IS NOT NULL AND g.ref IS NOT NULL AND g.ref >= m.rmed)
      GROUP BY dev, det)""")
    q.add("lkc", """
SELECT dev, det, CASE WHEN c1 IS NOT NULL AND NOT isnan(c1) THEN c1 WHEN NOT isnan(c2) THEN c2 END AS c_occ_like
FROM (SELECT g.dev, g.det,
        CASE WHEN count(*) FILTER (g.occ IS NOT NULL AND g.same IS NOT NULL) >= 6
             THEN corr(g.same, g.occ) FILTER (g.occ IS NOT NULL AND g.same IS NOT NULL) END AS c1,
        CASE WHEN count(*) FILTER (g.occ IS NOT NULL AND g.peer IS NOT NULL) >= 6
             THEN corr(g.peer, g.occ) FILTER (g.occ IS NOT NULL AND g.peer IS NOT NULL) END AS c2
      FROM gctx g JOIN win w USING (dev) WHERE w.wg <> 'm30' GROUP BY g.dev, g.det)""")
    # ---- per-ON statistics: ONs logged again without an OFF, holds, the median clean ON (Count mode)
    q.add("act", """
SELECT o.dev, o.det, a.n_on_a, a.n_rep, a.n_rep / a.n_on_a AS rep_frac, count(*) AS n_start,
       max(CASE WHEN o.gap THEN 0.0 ELSE o.dur END) FILTER (NOT o.open_end) AS max_on_s,
       coalesce(sum(CASE WHEN o.gap THEN 0.0 ELSE o.dur END) FILTER (NOT o.open_end AND o.multi), 0.0) AS rep_time_s,
       CASE WHEN count(*) FILTER (NOT o.multi AND NOT o.open_end AND NOT o.gap) >= 5
            THEN median(o.dur) FILTER (NOT o.multi AND NOT o.open_end AND NOT o.gap) END AS dur_p50,
       count(*) FILTER (NOT o.gap AND o.dur >= 5) AS n_ge5, count(*) FILTER (NOT o.gap AND o.dur >= 60) AS n_ge60,
       max(o.dur) FILTER (NOT o.gap) AS long_max_s
FROM ons o JOIN (SELECT dev, det, count(*) AS n_on_a, count(*) FILTER (pe = 82) AS n_rep FROM de WHERE eid = 82
                 GROUP BY dev, det) a USING (dev, det)
GROUP BY o.dev, o.det, a.n_on_a, a.n_rep""", mat=True)
    # ---- too fast: every ON-to-ON gap under 0.8 s / lanes (stop-bar zones) or 0.6 s / lanes, in whole 0.1-s ticks
    tf = LJ["too_fast"]
    q.add("onx", """
WITH r AS (SELECT dev, det, dn, pe = 81 AND ppe = 81 AS rp, ne = 81 AND nne = 81 AS rn FROM de
           WHERE eid = 82 AND ((pe = 81 AND ppe = 81) OR (ne = 81 AND nne = 81))),
o AS (SELECT d.dev, d.det, d.dn FROM de d WHERE d.eid = 82 AND (d.dev, d.det) IN (SELECT (dev, det) FROM r))
SELECT r.dev, r.det, r.dn, CASE WHEN r.rp THEN p.dn END AS prv, CASE WHEN r.rn THEN n.dn END AS nxt
FROM r ASOF LEFT JOIN o p ON p.dev = r.dev AND p.det = r.det AND r.dn > p.dn
ASOF LEFT JOIN o n ON n.dev = r.dev AND n.det = r.det AND r.dn < n.dn""")
    q.add("tfo", f"""
SELECT d.dev, d.det, d.dn, z.lim,
       coalesce(round_even((d.dn - CASE WHEN d.pe = 82 THEN d.pdn WHEN d.pe = 81 AND d.ppe = 82 THEN d.p2dn
                                        ELSE x.prv END) / 1e8, 0) < z.lim * 10, false) AS f,
       CASE WHEN d.ne = 82 THEN d.ndn WHEN d.ne = 81 AND d.nne = 82 THEN d.n2dn ELSE x.nxt END IS NOT NULL AS nx,
       coalesce(round_even((CASE WHEN d.ne = 82 THEN d.ndn WHEN d.ne = 81 AND d.nne = 82 THEN d.n2dn ELSE x.nxt END
                            - d.dn) / 1e8, 0) < z.lim * 10, false) AS fnx
FROM de d JOIN (SELECT dev, det, (CASE WHEN fn IN {_in(tf['stopbar'])} THEN {_f(tf['gap_s_stopbar'])}
                                        ELSE {_f(tf['gap_s_other'])} END) / ln1 AS lim FROM dz) z USING (dev, det)
LEFT JOIN onx x ON x.dev = d.dev AND x.det = d.det AND x.dn = d.dn
WHERE d.eid = 82""", mat=True)
    q.add("tf", """
SELECT dev, det, count(*) AS N, count(*) FILTER (f) AS n_fast, 1000.0 * count(*) FILTER (f) / greatest(count(*), 1)
       AS fast_rate, any_value(lim) AS gap_lim,
       CASE WHEN count(*) FILTER (f AND nx) > 0 THEN count(*) FILTER (f AND nx AND NOT fnx) / count(*) FILTER (f AND nx)
       END AS pair_share
FROM tfo GROUP BY dev, det""")
    # ---- vehicle starts (an ON after a non-ON) with their colour state; phase timelines (>= 3 greens)
    q.add("st0r", """
WITH r AS (SELECT dev, det, dn FROM de WHERE eid = 82 AND pe = 81 AND ppe = 81),
o AS (SELECT d.dev, d.det, d.dn FROM de d WHERE d.eid = 82 AND d.pe IS DISTINCT FROM 82
      AND (d.dev, d.det) IN (SELECT (dev, det) FROM r))
SELECT r.dev, r.det, r.dn, p.dn AS prv FROM r ASOF LEFT JOIN o p ON p.dev = r.dev AND p.det = r.det AND r.dn > p.dn
UNION ALL
SELECT d.dev, d.det, d.dn, m.on_dn FROM de d JOIN multis m ON m.dev = d.dev AND m.det = d.det AND m.off_dn = d.pdn
WHERE d.eid = 82 AND d.pe = 81 AND d.ppe = 82 AND d.p3e = 82""")
    q.add("starts", f"""
SELECT d.dev, d.det, d.dn, d.dn / 1e9 AS x, (d.dn // {NS5})::INT AS b,
       coalesce(round_even((d.dn - CASE WHEN d.pe = 81 AND d.ppe = 82 AND d.p3e IS DISTINCT FROM 82 THEN d.p2dn
                                        ELSE r.prv END) / 1e8, 0) < 10, false) AS f1,
       coalesce(d.pe = 81 AND round_even((d.dn - d.pdn) / 1e8, 0) < 3, false) AS chat
FROM de d LEFT JOIN st0r r ON r.dev = d.dev AND r.det = d.det AND r.dn = d.dn
WHERE d.eid = 82 AND d.pe IS DISTINCT FROM 82""")
    q.add("seg", """
SELECT c.dev, c.ph, c.t AS a, coalesce(lead(c.t) OVER (PARTITION BY c.dev, c.ph ORDER BY c.t, c.eid), w.T) AS z, c.st
FROM (SELECT dev, par AS ph, dn / 1e9 AS t, eid, CASE eid WHEN 1 THEN 0 WHEN 8 THEN 1 ELSE 2 END AS st,
             lag(CASE eid WHEN 1 THEN 0 WHEN 8 THEN 1 ELSE 2 END) OVER (PARTITION BY dev, par ORDER BY dn, eid) AS pst
      FROM ev WHERE eid IN (1, 8, 9, 10)) c JOIN win w USING (dev)
WHERE c.pst IS DISTINCT FROM c.st""", mat=True)
    q.add("phs3", "SELECT dev, par AS ph FROM ev WHERE eid = 1 GROUP BY dev, par HAVING count(*) >= 3", mat=True)
    # pieces: green 0, yellow 1, red split into its first 2 s (2) and the rest (3); valid = no comms gap starts in it
    q.add("pcs", """
SELECT p.*, NOT coalesce(g.g0 < p.z, false) AS valid FROM (
  SELECT * FROM (SELECT dev, ph, a, z, st FROM seg WHERE st < 2
                 UNION ALL SELECT dev, ph, a, least(a + 2.0, z), 2 FROM seg WHERE st = 2
                 UNION ALL SELECT dev, ph, least(a + 2.0, z), z, 3 FROM seg WHERE st = 2) WHERE z > a) p
ASOF LEFT JOIN gaps g ON p.dev = g.dev AND g.g0 >= p.a""", mat=True)
    q.add("sst", """
SELECT s.*, CASE WHEN p.valid THEN p.st ELSE 4 END AS state
FROM (SELECT s.*, z.ph, z.ln1 FROM starts s JOIN dz z USING (dev, det)) s
ASOF LEFT JOIN (SELECT p.* FROM pcs p JOIN phs3 USING (dev, ph)) p ON s.dev = p.dev AND s.ph = p.ph AND s.x >= p.a""",
          mat=True)
    q.add("sgyb", """
SELECT dev, ph, b, sum(least(z, (b + 1) * 300.0) - greatest(a, b * 300.0)) AS sgy
FROM (SELECT p.dev, p.ph, p.a, p.z, unnest(range(floor(p.a / 300.0)::INT, least(floor(p.z / 300.0)::INT, w.nb5 - 1) + 1)) AS b
      FROM pcs p JOIN phs3 USING (dev, ph) JOIN win w USING (dev) WHERE p.valid AND p.st < 2)
GROUP BY dev, ph, b""", mat=True)
    # too many for the traffic: busiest 5 min in veh/h/lane, starts in green + yellow per green + yellow second
    # (>= 60 s of it in the bin) and all starts (in float32, as the research)
    q.add("q5b", """
SELECT s.dev, s.det, s.b, count(*) AS n, count(*) FILTER (s.state < 2) AS ngy, any_value(s.ln1) AS ln1
FROM sst s GROUP BY s.dev, s.det, s.b""", mat=True)
    q.add("q5gyb", """
SELECT z.dev, z.det, y.b, ((coalesce(q.ngy, 0)::FLOAT / y.sgy::FLOAT) * 3600::FLOAT)::DOUBLE / z.ln1 AS q5_gy
FROM dz z JOIN phs3 p ON p.dev = z.dev AND p.ph = z.ph JOIN sgyb y ON y.dev = z.dev AND y.ph = z.ph
LEFT JOIN q5b q ON q.dev = z.dev AND q.det = z.det AND q.b = y.b WHERE y.sgy >= 60""", mat=True)
    q.add("q5", """
SELECT a.dev, a.det, g.q5_gy, a.q5_all FROM (SELECT dev, det, max(((n::FLOAT / 300::FLOAT) * 3600::FLOAT)::DOUBLE / ln1)
                                                AS q5_all FROM q5b GROUP BY dev, det) a
LEFT JOIN (SELECT dev, det, max(q5_gy) AS q5_gy FROM q5gyb GROUP BY dev, det) g USING (dev, det)""")
    # rule F: are the fast starts the same share of the starts in every 15-min period?
    q.add("fe", """
SELECT dev, det, count(*) AS fe_n, any_value(p) AS fe_p,
       CASE WHEN count(*) >= 2 AND any_value(p) > 0 AND any_value(p) < 1
            THEN sum((x - p * n) ^ 2 / (p * (1 - p) * n)) / greatest(count(*) - 1, 1) END AS fe_phi
FROM (SELECT *, sum(x) OVER (PARTITION BY dev, det) / sum(n) OVER (PARTITION BY dev, det) AS p FROM (
        SELECT dev, det, b // 3 AS g, count(*) FILTER (f1 AND NOT chat AND state <> 2) AS x,
               count(*) - count(*) FILTER (chat AND state <> 2) AS n
        FROM sst GROUP BY dev, det, b // 3) WHERE n > 0)
GROUP BY dev, det""")
    _held_on(q, R)
    _hourly(q, LJ)
    _spiky(q, LJ)
    _holds(q, LJ, R)


def _held_on(q: _Q, R: dict):
    """Count zone held ON in light traffic (score_v4c.occ_hi_c + note 135 green test) -> ohi, hib (the bins)."""
    olp = _values("bs, tl, lp", [[_f(a), _f(b), _f(c)] for a, b, c in R["occ_lim_pulse"]])
    ctp = _values("bs, tp", [[_f(a), _f(b)] for a, b in R["count_t_pass"]])
    q.add("xm", """
SELECT x.*, CASE WHEN x.fn <> 'Count' THEN '' WHEN a.dur_p50 IS NULL THEN 'unknown' WHEN a.dur_p50 <= 0.25 THEN 'pulse'
                 ELSE 'normal' END AS cmode
FROM xbase x LEFT JOIN act a USING (dev, det)""", mat=True)
    q.add("ohr", f"""
SELECT h.*, CASE WHEN h.xfn = 'Count' AND h.occ > greatest(l.lp, h.n * t.tp / h.bs, 0.20) AND h.cong < 1.5
                  AND NOT coalesce(h.pocc >= 0.40, false) AND h.tl <= 1 THEN true ELSE false END AS hic
FROM (SELECT g.dev, g.det, g.k, g.n, g.occ, g.cong, x.ph AS xph, x.fn AS xfn, w.bs,
             CASE WHEN g.ph IS NOT NULL THEN (a.so - coalesce(g.occ, 0)) / CASE WHEN a.co > 1 THEN a.co - 1 END END AS pocc,
             CASE WHEN m.rmax > 0 THEN least(greatest(floor(5 * g.ref / m.rmax), 0), 4) END AS tl
      FROM gctx g JOIN xm x USING (dev, det) JOIN win w USING (dev)
      JOIN (SELECT dev, det, max(ref) AS rmax FROM gb GROUP BY dev, det) m USING (dev, det)
      LEFT JOIN (SELECT g.dev, g.ph, g.k, sum(g.occ) AS so, count(g.occ) AS co FROM gb g JOIN xm x USING (dev, det)
                 WHERE x.cmode <> 'unknown' AND g.ph IS NOT NULL GROUP BY g.dev, g.ph, g.k) a
        ON a.dev = g.dev AND a.ph = g.ph AND a.k = g.k
      WHERE x.cmode <> 'unknown') h
LEFT JOIN ({olp}) l ON l.bs = h.bs AND l.tl = h.tl LEFT JOIN ({ctp}) t ON t.bs = h.bs""", mat=True)
    # note 135: the zone ON for >= 20 % of its phase's green after the first 5 s (>= 60 s of such green in the bin)
    q.add("hicd", "SELECT DISTINCT dev, det, xph AS ph FROM ohr WHERE hic AND xph IS NOT NULL")
    q.add("grn", """
SELECT p.dev, p.ph, p.a + 5.0 AS a, p.z FROM pcs p JOIN (SELECT DISTINCT dev, ph FROM hicd) USING (dev, ph)
WHERE p.st = 0 AND p.valid AND p.z > p.a + 5.0""")
    q.add("gonb", """
SELECT dev, det, k, sum(least(z, (k + 1) * bs) - greatest(a, k * bs)) FILTER (gon) AS gon,
       sum(least(z, (k + 1) * bs) - greatest(a, k * bs)) FILTER (NOT gon) AS gav
FROM (SELECT i.*, w.bs, unnest(range(floor(i.a / w.bs)::INT, least(floor(i.z / w.bs)::INT, w.nbb - 1) + 1)) AS k
      FROM (SELECT h.dev, h.det, greatest(o.s, g.a) AS a, least(o.f, g.z) AS z, true AS gon
            FROM (SELECT h.dev, h.det, h.ph, o.s, o.f FROM hicd h
                  JOIN (SELECT dev, det, s, f FROM ons WHERE NOT gap UNION ALL SELECT dev, det, 0.0, f FROM lead0) o
                  ON o.dev = h.dev AND o.det = h.det) o
            JOIN grn g ON g.dev = o.dev AND g.ph = o.ph AND o.s < g.z AND o.f > g.a JOIN hicd h ON h.dev = o.dev
              AND h.det = o.det
            UNION ALL SELECT h.dev, h.det, g.a, g.z, false FROM hicd h JOIN grn g USING (dev, ph)) i
      JOIN win w USING (dev))
GROUP BY dev, det, k""")
    q.add("hib", """
SELECT h.dev, h.det, h.k, h.occ FROM ohr h JOIN gonb g USING (dev, det, k)
WHERE h.hic AND h.xph IS NOT NULL AND g.gav >= 60 AND coalesce(g.gon, 0) >= 0.20 * g.gav""", mat=True)
    q.add("ohi", """
SELECT o.dev, o.det, count(h.k) * any_value(o.bs) / 60.0 AS hi_min_c, max(h.occ) AS hi_occ_max
FROM (SELECT DISTINCT dev, det, bs FROM ohr) o LEFT JOIN hib h USING (dev, det) GROUP BY o.dev, o.det""")


def _hourly(q: _Q, LJ: dict):
    """hourly GEH against the yardstick (health_counts.hourly_geh): count drops and misses at night -> geh."""
    q.add("hix", f"""
SELECT dev, hid, hid - min(hid) OVER (PARTITION BY dev) AS j, hid % 24 AS hod, nb_ = 12 AND nc = 12 AS ch,
       count(*) OVER (PARTITION BY dev) AS H
FROM (SELECT b.dev, (w.s_ns + b.b * {NS5}) // {NSH} AS hid, count(*) AS nb_, count(*) FILTER (b.cov) AS nc
      FROM bins5 b JOIN win w USING (dev) WHERE w.hours >= {_f(LJ['min_hours_counts'])} GROUP BY b.dev, hid)""",
          mat=True)
    q.add("mh", f"""
SELECT g.dev, g.det, (w.s_ns + g.b * {NS5}) // {NSH} AS hid, sum(g.n) AS M, sum(g.occ) AS O
FROM g5 g JOIN win w USING (dev) WHERE w.hours >= {_f(LJ['min_hours_counts'])} GROUP BY g.dev, g.det, hid""",
          mat=True)
    # yardstick: the detector it tracks (partner), else its phase mates (>= 2), else the rest of the signal
    q.add("yk", """
SELECT k.dev, k.det, pa.p, k.isph FROM p1kd k LEFT JOIN p1part pa ON pa.kind = 'n' AND pa.id = k.dev * 128 + k.det""",
          mat=True)
    q.add("gh", """
SELECT m.dev, m.det, x.j, x.hod, x.ch, x.H, m.M, m.O, y.p IS NOT NULL OR y.isph AS has_m,
       CASE WHEN y.p IS NULL AND NOT y.isph THEN s.SM - m.M ELSE coalesce(r.RM, 0) END AS R, coalesce(r.RO, 0) AS Om
FROM mh m JOIN hix x USING (dev, hid) JOIN yk y USING (dev, det)
JOIN (SELECT dev, hid, sum(M) AS SM FROM mh GROUP BY dev, hid) s USING (dev, hid)
LEFT JOIN (SELECT a.dev, a.det, b.hid, sum(b.M) AS RM, sum(b.O) AS RO
           FROM (SELECT dev, det, p AS r FROM yk WHERE p IS NOT NULL
                 UNION ALL SELECT f.dev, f.det, f.r FROM p1pref f JOIN yk y USING (dev, det) WHERE y.p IS NULL) a
           JOIN mh b ON b.dev = a.dev AND b.det = a.r GROUP BY a.dev, a.det, b.hid) r USING (dev, det, hid)""",
          mat=True)

    def share(mask: str) -> str:
        return (f"CASE WHEN count(*) FILTER ({mask} AND R >= 30) >= 4 THEN median(M / R) FILTER ({mask} AND R >= 30) "
                f"ELSE coalesce(sum(M) FILTER ({mask}), 0) / greatest(coalesce(sum(R) FILTER ({mask}), 0), 1) END")
    day, nt = "ch AND hod >= 7 AND hod < 19", "ch AND (hod >= 21 OR hod < 5)"
    q.add("gh1", f"""
SELECT dev, det, coalesce(sum(M) FILTER ({day}), 0) >= 20 AND coalesce(sum(R) FILTER ({day}), 0) > 0 AS elig,
       {share(day)} AS s, count(*) FILTER ({nt}) AS nn
FROM gh GROUP BY dev, det""", mat=True)
    q.add("gh2", f"""
SELECT g.*, a.s, a.elig, CASE WHEN g.ch THEN {_sgeh('g.M', 'a.s * g.R')} ELSE 0.0 END AS sg,
       sum(CASE WHEN g.ch THEN {_sgeh('g.M', 'a.s * g.R')} ELSE 0.0 END) OVER w AS cs, sum(g.ch::INT) OVER w AS cc,
       sum(CASE WHEN g.ch THEN {_sgeh('g.M', 'a.s * g.R')} ELSE 0.0 END) OVER (PARTITION BY g.dev, g.det) AS cs_t,
       sum(g.ch::INT) OVER (PARTITION BY g.dev, g.det) AS cc_t
FROM gh g JOIN gh1 a USING (dev, det) WINDOW w AS (PARTITION BY g.dev, g.det ORDER BY g.j ROWS UNBOUNDED PRECEDING)""",
          mat=True)
    # count drops: the hour with the largest step up in under-counting (>= 3 covered hours each side)
    q.add("ghk", """
SELECT a.dev, a.det, coalesce(k.k, 1) AS k, a.elig AND k.k IS NOT NULL AS has_k FROM gh1 a LEFT JOIN (
  SELECT dev, det, first(j + 1 ORDER BY stat DESC, j) AS k FROM (
    SELECT dev, det, j, ((cs_t - cs) / (cc_t - cc) - cs / cc) * sqrt(cc * (cc_t - cc) / cc_t) AS stat
    FROM gh2 WHERE j <= H - 2 AND cc >= 3 AND cc_t - cc >= 3) GROUP BY dev, det) k USING (dev, det)""")
    q.add("gh3", f"""
SELECT g.dev, g.det, k.k, k.has_k, g.j < k.k AND g.ch AS pre, g.j >= k.k AND g.ch AS post, g.M, g.R, g.O, g.Om, g.ch,
       g.sg, g.s, g.hod, g.H, g.has_m, g.elig
FROM gh2 g JOIN ghk k USING (dev, det)""")
    q.add("gh4", f"""
SELECT dev, det, any_value(k) AS k, any_value(has_k) AS has_k, any_value(H) AS H, any_value(has_m) AS has_m,
       any_value(elig) AS elig, any_value(s) AS s, {share('pre')} AS s2,
       count(*) FILTER (pre AND R >= 30) >= 3 AS re, greatest(count(*) FILTER (pre), 1) AS npre,
       greatest(count(*) FILTER (post), 1) AS npost, coalesce(sum(M) FILTER (pre), 0) AS Mpre,
       coalesce(sum(M) FILTER (post), 0) AS Mpost, coalesce(sum(R) FILTER (post), 0) AS Rpost,
       coalesce(sum(O) FILTER (pre), 0) AS Opre, coalesce(sum(O) FILTER (post), 0) AS Opost,
       coalesce(sum(Om) FILTER (pre), 0) AS Ompre, coalesce(sum(Om) FILTER (post), 0) AS Ompost,
       count(*) FILTER ({nt}) AS nn, coalesce(sum(M) FILTER ({nt}), 0) AS Mn, coalesce(sum(s * R) FILTER ({nt}), 0) AS Cn
FROM gh3 GROUP BY dev, det""")
    q.add("geh", f"""
SELECT a.dev, a.det, a.elig AS geh_ok,
       CASE WHEN a.elig AND a.nn >= 4 THEN {_sgeh('a.Mn / greatest(a.nn, 1)', 'a.Cn / greatest(a.nn, 1)')} END AS geh_night,
       a.Mn AS geh_night_n0, a.Cn AS geh_night_exp0,
       CASE WHEN a.has_k AND b.med > 0 THEN b.med END AS geh_drop,
       CASE WHEN a.has_k THEN make_timestamp(((x.hid0 + least(a.k, a.H - 1)) * {NSH}) // 1000) END AS drop_t,
       a.Mpre / a.npre AS own_before, a.Mpost / a.npost AS own_after,
       CASE WHEN a.re THEN a.s2 ELSE a.s END * a.Rpost / a.npost AS exp_after,
       CASE WHEN a.has_k THEN (a.Opost / a.npost) / greatest(a.Opre / a.npre, 1e-6) END AS occ_own,
       CASE WHEN a.has_k AND a.has_m THEN (a.Ompost / a.npost) / greatest(a.Ompre / a.npre, 1e-6) END AS occ_mates
FROM gh4 a JOIN (SELECT dev, min(hid) AS hid0 FROM hix GROUP BY dev) x USING (dev)
LEFT JOIN (SELECT g.dev, g.det, median(CASE WHEN a.re THEN CASE WHEN g.ch THEN {_sgeh('g.M', 'a.s2 * g.R')} ELSE 0.0 END
                                           ELSE g.sg END) FILTER (g.post) AS med
           FROM gh3 g JOIN gh4 a USING (dev, det) GROUP BY g.dev, g.det) b USING (dev, det)""")


def _spiky(q: _Q, LJ: dict):
    """bursts in 10-min counts against its own level of the surrounding 2 h and the phase mates' own rise
    (health_counts.spiky) -> spk (per detector), spk_runs (the bursts)."""
    P = LJ["spiky"]
    k, wb = int(P["bin_min"] * 60 // 300), int(P["window_bins"])
    q.add("o10", f"""
SELECT g.dev, g.det, g.b // {k} AS t, sum(g.n) AS own, bool_and(g.cov) AS c
FROM g5 g JOIN win w USING (dev)
WHERE w.hours >= {_f(LJ['min_hours_counts'])} AND w.nb5 // {k} >= {2 * wb} AND g.b < w.nb5 // {k} * {k}
GROUP BY g.dev, g.det, g.b // {k}""", mat=True)
    lvl = ("CASE WHEN len(list_filter(coalesce({a}, []) || coalesce({b}, []), x -> x IS NOT NULL)) >= 3 THEN "
           "list_median(list_filter(coalesce({a}, []) || coalesce({b}, []), x -> x IS NOT NULL)) END")
    q.add("o10l", f"""
SELECT dev, det, t, own, c, m,
       list(CASE WHEN c THEN own END) OVER wl AS ol, list(CASE WHEN c THEN own END) OVER wr AS orr,
       list(CASE WHEN c THEN m END) OVER wl AS ml, list(CASE WHEN c THEN m END) OVER wr AS mr
FROM (SELECT o.*, coalesce(m.m, 0.0) AS m FROM o10 o LEFT JOIN (
        SELECT f.dev, f.det, o.t, sum(o.own) AS m FROM p1rf f JOIN o10 o ON o.dev = f.dev AND o.det = f.r
        WHERE f.same GROUP BY f.dev, f.det, o.t) m USING (dev, det, t))
WINDOW wl AS (PARTITION BY dev, det ORDER BY t ROWS BETWEEN {wb} PRECEDING AND 2 PRECEDING),
       wr AS (PARTITION BY dev, det ORDER BY t ROWS BETWEEN 2 FOLLOWING AND {wb} FOLLOWING)""")
    q.add("o10s", f"""
SELECT *, b IS NOT NULL AND c AND own / ex >= {_f(P['ratio'])} AND g >= {_f(P['geh'])} AND own > ex AS sp FROM (
  SELECT *, sqrt(2 * (own - ex) ^ 2 / greatest(own + ex, 1e-9)) AS g FROM (
    SELECT *, greatest(b, 1.0) * greatest(CASE WHEN m >= {_f(P['mates_min'])} AND bm IS NOT NULL
                                               THEN m / greatest(bm, 1.0) ELSE 1.0 END, 1.0) AS ex FROM (
      SELECT dev, det, t, own, c, m, {lvl.format(a='ol', b='orr')} AS b, {lvl.format(a='ml', b='mr')} AS bm FROM o10l)))""",
          mat=True)
    q.add("spk_runs", """
SELECT dev, det, min(t) AS a, max(t) + 1 AS z FROM (
  SELECT dev, det, t, t - row_number() OVER (PARTITION BY dev, det ORDER BY t) AS grp FROM o10s WHERE sp)
GROUP BY dev, det, grp""", mat=True)
    q.add("spk", f"""
SELECT s.*, CASE WHEN spk_daily < {_f(P['daily_none'])} THEN 0
                 WHEN spk_daily < {_f(P['daily_low'])} AND lv = 2 AND spk_gmax < {_f(P['geh_big'])} THEN 1 ELSE lv END
            AS spk_lvl
FROM (SELECT s.*, CASE WHEN spk_bursts >= {_f(P['bad'])} THEN 2 WHEN spk_bursts >= {_f(P['suspect'])} THEN 1 ELSE 0 END AS lv
      FROM (SELECT o.dev, o.det, coalesce(r.nr, 0) / greatest(w.hours / 24.0, 1.0) AS spk_bursts,
                   coalesce(max(o.g) FILTER (o.sp), 0.0) AS spk_gmax, sum(o.own) * 24.0 / greatest(w.hours, 1e-9) AS spk_daily
            FROM o10s o JOIN win w USING (dev)
            LEFT JOIN (SELECT dev, det, count(*) AS nr FROM spk_runs GROUP BY dev, det) r USING (dev, det)
            GROUP BY o.dev, o.det, r.nr, w.hours) s) s""")


def _holds(q: _Q, LJ: dict, R: dict):
    """holds (continuous ONs >= 5 min): held ON together with other detectors (common), and the queue context of each
    (health_v4_stats.episodes_ctx) -> ectx (one row per hold)."""
    tol = _f(LJ["common_hold_s"])
    q.add("hpair", f"""
SELECT a.hid, b.det AS dj, abs(b.s - a.s) <= {tol} AND abs(b.f - a.f) <= {tol} AS near
FROM hold a JOIN hold b ON a.dev = b.dev AND a.det <> b.det AND least(a.f, b.f) > greatest(a.s, b.s)
LEFT JOIN twin t ON t.dev = a.dev AND t.i = a.det AND t.j = b.det WHERE t.i IS NULL""")
    q.add("hcom", """
SELECT h.hid, coalesce(p.anyov, false) AS n_ov_any, coalesce(p.anyov AND p.allnear, false) AS common, p.mates
FROM hold h LEFT JOIN (SELECT hid, true AS anyov, bool_and(near) AS allnear, list(DISTINCT dj ORDER BY dj) AS mates
                       FROM hpair GROUP BY hid) p USING (hid)""", mat=True)
    qt = _values("type, qt", [[_s(a), _f(b)] for a, b in R["q_type"]])
    # the hold in bs-bins: [ia, iz) touched, full bins [fa, fz)
    q.add("hb", """
SELECT h.hid, h.dev, h.det, h.s, h.f, h.dur, x.ph AS p, w.bs, w.nbb, floor(h.s / w.bs)::INT AS ia,
       ceil(h.f / w.bs)::INT AS iz, ceil(h.s / w.bs)::INT AS fa, floor(h.f / w.bs)::INT AS fz
FROM hold h JOIN win w USING (dev) LEFT JOIN xbase x USING (dev, det)""", mat=True)
    # healthy phase mates not themselves held >= 99 % ON for >= 80 % of the hold
    q.add("hmate", """
SELECT h.hid, z.det AS r FROM hb h JOIN dz z ON z.dev = h.dev AND z.ph = h.p AND z.det <> h.det
WHERE z.status1 IS DISTINCT FROM 'bad' AND NOT (least(h.iz, h.nbb) > h.ia AND (
  SELECT count(*) FILTER (g.occ >= 0.99) FROM gb g WHERE g.dev = h.dev AND g.det = z.det
  AND g.k >= h.ia AND g.k < least(h.iz, h.nbb)) / (least(h.iz, h.nbb) - h.ia) >= 0.8)""", mat=True)
    q.add("hpm", f"""
SELECT m.hid, g.k, avg(g.occ) AS pm, bool_or(g.occ >= t.qt) AS hit
FROM hmate m JOIN hb h USING (hid) JOIN gb g ON g.dev = h.dev AND g.det = m.r
LEFT JOIN xm x ON x.dev = h.dev AND x.det = m.r LEFT JOIN ({qt}) t ON t.type = x.fn || ' ' || x.span
GROUP BY m.hid, g.k""")
    q.add("htr", """
SELECT h.hid, g.k, sum(coalesce(g.n, 0)) AS tot FROM hb h
JOIN (SELECT h.hid, z.det AS r, bool_or(z.tr) OVER (PARTITION BY h.hid) AS anytr, z.tr
      FROM hb h JOIN dz z ON z.dev = h.dev AND z.ph = h.p AND z.det <> h.det) m ON m.hid = h.hid AND (m.tr OR NOT m.anytr)
JOIN gb g ON g.dev = h.dev AND g.det = m.r GROUP BY h.hid, g.k""")
    q.add("hrh", """
SELECT h.hid, bool_or(g.occ >= 0.95) AS ref_held_e FROM hb h JOIN rmm m ON m.dev = h.dev AND m.det = h.det
JOIN gb g ON g.dev = h.dev AND g.det = m.r AND g.k >= h.ia AND g.k < least(h.iz, h.nbb) GROUP BY h.hid""")
    q.add("ectx", """
SELECT h.hid, h.dev, h.det, h.s, h.f, h.dur, h.p,
  coalesce((SELECT count(*) FROM hmate m WHERE m.hid = h.hid), 0) AS n_hpeer_e, b.phx_h_e, b.corr_h_e, b.cover_e,
  b.refx_e, b.light_e, b.busy_min_e, b.busy_occ_e, coalesce(r.ref_held_e, false) AS ref_held_e, t.trafx_e, c.co AS co5,
  m.n_ov_any, m.common, m.mates
FROM hb h JOIN hold_co c USING (hid) JOIN hcom m USING (hid) LEFT JOIN hrh r USING (hid)
LEFT JOIN (
  SELECT h.hid,
    CASE WHEN any_value(np.hid) IS NOT NULL AND count(*) FILTER (g.k >= h.ia AND g.k < h.iz) > 0 THEN
      avg(p.pm) FILTER (g.k >= h.ia AND g.k < h.iz) / greatest(avg(p.pm), 1e-9) END AS phx_h_e,
    CASE WHEN any_value(np.hid) IS NOT NULL AND count(*) FILTER (NOT (g.k >= h.ia AND g.k < h.iz) AND g.occ IS NOT NULL
         AND p.pm IS NOT NULL) >= 4 THEN nullif(corr(p.pm, g.occ) FILTER (NOT (g.k >= h.ia AND g.k < h.iz)
         AND g.occ IS NOT NULL AND p.pm IS NOT NULL), 'nan'::DOUBLE) END AS corr_h_e,
    CASE WHEN any_value(np.hid) IS NOT NULL AND count(*) FILTER (g.k >= h.ia AND g.k < h.iz) > 0 THEN
      count(*) FILTER (g.k >= h.ia AND g.k < h.iz AND p.hit) / count(*) FILTER (g.k >= h.ia AND g.k < h.iz) END AS cover_e,
    CASE WHEN count(*) FILTER (g.k >= h.ia AND g.k < h.iz) > 0 THEN
      avg(g.ref) FILTER (g.k >= h.ia AND g.k < h.iz) / greatest(avg(g.ref), 1e-9) END AS refx_e,
    count(*) FILTER (NOT (g.k >= h.ia - 1 AND g.k <= h.iz) AND g.occ >= 0.9 AND g.ref <= 0.3 * mr.rm
                     AND (mr.mpo IS NULL OR NOT coalesce(g.peer > mr.mpo, false))) AS light_e,
    CASE WHEN count(g.ref) FILTER (g.k >= h.ia AND g.k < h.iz) = 0 THEN 0.0
         ELSE count(*) FILTER ((g.k < h.ia - 1 OR g.k > h.iz) AND g.ref IS NOT NULL AND g.occ IS NOT NULL
                               AND g.ref >= mr.rh) * any_value(h.bs) / 60.0 END AS busy_min_e,
    CASE WHEN count(g.ref) FILTER (g.k >= h.ia AND g.k < h.iz) > 0 THEN
      median(g.occ) FILTER ((g.k < h.ia - 1 OR g.k > h.iz) AND g.ref IS NOT NULL AND g.occ IS NOT NULL AND g.ref >= mr.rh)
      END AS busy_occ_e
  FROM hb h JOIN gctx g ON g.dev = h.dev AND g.det = h.det
  JOIN (SELECT h.hid, avg(g.ref) AS rm, avg(g.peer) AS mpo, avg(g.ref) FILTER (g.k >= h.ia AND g.k < h.iz) AS rh
        FROM hb h JOIN gctx g ON g.dev = h.dev AND g.det = h.det GROUP BY h.hid) mr ON mr.hid = h.hid
  LEFT JOIN hpm p ON p.hid = h.hid AND p.k = g.k
  LEFT JOIN (SELECT DISTINCT hid FROM hmate) np ON np.hid = h.hid
  GROUP BY h.hid) b USING (hid)
LEFT JOIN (
  SELECT h.hid, CASE WHEN count(*) FILTER (t.k >= greatest(h.ja, 0) AND t.k < least(h.jz, h.nbb)) > 0 THEN
    avg(t.tot) FILTER (t.k >= greatest(h.ja, 0) AND t.k < least(h.jz, h.nbb)) / greatest(avg(t.tot), 1e-9) END AS trafx_e
  FROM (SELECT *, CASE WHEN fz > fa THEN fa ELSE ia END AS ja, CASE WHEN fz > fa THEN fz ELSE iz END AS jz FROM hb) h
  JOIN htr t USING (hid) GROUP BY h.hid) t USING (hid)""", mat=True)
# ============================================================================ limits, scores, verdict (health_v4)
KEYS = ("stuck", "count_on", "dropout", "level", "night_drop", "prof", "rapid", "volume", "spiky")
STAT = ("dropout", "rapid", "level", "night_drop", "prof", "spiky")
FAMILY = {"rapid": "fast", "dropout": "silent", "stuck": "stuck", "level": "level", "night_drop": "level",
          "volume": "shape", "prof": "shape", "count_on": "stuck", "spiky": "shape"}
EVID = {"stuck": "stuck", "count_on": "stuck", "dropout": "quiet", "level": "quiet", "night_drop": "quiet",
        "rapid": "fast", "volume": "fast", "prof": "prof", "spiky": "spiky"}
CATEGORY = {"stuck": "Stuck on", "count_on": "Stuck on", "dropout": "Counts fall", "level": "Counts fall",
            "night_drop": "Misses vehicles at night", "prof": "Busy at night", "rapid": "Too-fast actuations",
            "volume": "Too many for the traffic", "spiky": "Spiky counts"}
NO_SHARED = ("Advance", "Count", "Yellow_Red")
DAYNS = 86_400_000_000_000


def _sc(x: str, lo: str, bad: str) -> str:
    """health_v4.score: 0 below the suspect limit, .35 at it, .75 at the bad limit (linear), 1 at most; NULL = not
    checked."""
    return (f"(CASE WHEN ({x}) IS NULL OR isnan({x}) OR ({lo}) IS NULL OR isnan({lo}) THEN NULL WHEN ({x}) < ({lo}) "
            f"THEN 0.0 ELSE least(0.35 + 0.40 * greatest((({x}) - ({lo})) / greatest(({bad}) - ({lo}), 1e-9), 0.0), 1.0) END)")


def _limits(q: _Q, LJ: dict, R: dict):
    """per-detector [suspect, bad] limits of every check (health_v4.limits); the least strict plausible function when
    the model is unsure."""
    h, cur, sh = LJ["h24"], LJ["current"], LJ["short"]
    rows = []
    for t in ("stuck_s", "silent_exp"):
        for k, (a, b) in h[t].items():
            rows.append([_s(t), _s(k), _f(a), _f(b)])
    for k, v in h["busy_z"].items():
        rows.append(["'busy_z'", _s(k), _f(v), "NULL"])
    for k, v in LJ["too_fast"]["per_1000"].items():
        rows.append(["'per_1000'", _s(k), _f(v), "NULL"])
    q.add("ltab", _values("tab, key, lo, hi", rows), mat=True)
    lv = []
    for tab, levels in (("stuck_x", R["cell"]["v4c"]["stuck_x"]), ("n_ep", R["cell"]["v4c"]["n_ep"]),
                        ("q5_gy", R["fast118c"]["q5_gy"]), ("q5_all", R["fast118c"]["q5_all"])):
        for i, lev in enumerate(levels):
            for r in lev["rows"]:
                v = r[-1]
                if v is None or v != v:
                    continue
                key = dict(zip(lev["keys"], (str(x) for x in r[:-1])))
                lv.append([_s(tab), str(i)] + [_s(key[c]) if c in key else "NULL"
                                                for c in ("fn", "span", "cmode", "wg", "band")] + [_f(v)])
    q.add("lref", _values("tab, lvl, fn, span, cmode, wg, band, v", lv), mat=True)
    # the functions whose limits apply: its own, and when the model is unsure (top < .7) every plausible one
    q.add("lcand", """
SELECT x.dev, x.det, x.fn AS fa, true AS own, x.span, x.cmode, x.wg, x.band,
       CASE WHEN x.fn = 'Count' THEN '|' || CASE WHEN x.cmode = 'normal' THEN 'normal' ELSE 'pulse' END ELSE '' END AS cms,
       x.cmode AS cmr
FROM xx0 x
UNION ALL
SELECT x.dev, x.det, a.fa, false, x.span, x.cmode, x.wg, x.band,
       CASE WHEN a.fa = 'Count' THEN '|' || CASE WHEN x.dur_p50 <= 0.25 THEN 'pulse' ELSE 'normal' END ELSE '' END,
       CASE WHEN a.fa <> 'Count' THEN '' WHEN x.dur_p50 IS NULL THEN 'unknown' WHEN x.dur_p50 <= 0.25 THEN 'pulse'
            ELSE 'normal' END
FROM xx0 x, (SELECT unnest(x2.alts) AS fa, x2.dev, x2.det FROM xx0 x2) a
WHERE a.dev = x.dev AND a.det = x.det AND a.fa <> x.fn""", mat=True)

    def tab(t, col="lo", split=False):
        key = "c.fa || '|' || c.span" + (" || c.cms" if split else "")
        return f"(SELECT max(l.{col}) FROM ltab l WHERE l.tab = '{t}' AND l.key = {key})"

    def ref(t, cm_fn=False):
        cm = "c.cmr" if cm_fn else "c.cmode"
        return (f"(SELECT first(l.v ORDER BY l.lvl) FROM lref l WHERE l.tab = '{t}' AND (l.fn IS NULL OR l.fn = c.fa) "
                f"AND (l.span IS NULL OR l.span = c.span) AND (l.cmode IS NULL OR l.cmode = {cm}) "
                f"AND (l.wg IS NULL OR l.wg = c.wg) AND (l.band IS NULL OR l.band = c.band))")
    q.add("lval", f"""
SELECT c.dev, c.det, c.own, {ref('n_ep')} AS n_ep, {tab('per_1000')} AS rapid,
       {tab('stuck_s', 'lo', True)} AS st_lo, {tab('stuck_s', 'hi', True)} AS st_hi,
       {tab('silent_exp', 'lo')} AS si_lo, {tab('silent_exp', 'hi')} AS si_hi,
       {ref('stuck_x', True)} AS stuck_x, {ref('q5_gy')} AS q5gy, {ref('q5_all')} AS q5all
FROM lcand c""")
    tf, a, b, c = LJ["too_fast"], *sh["stuck_bad"]
    q.add("limits", f"""
SELECT x.dev, x.det,
  v.n_ep + 1 AS lim_n_ep, (v.n_ep + 1) * {_f(sh['n_holds_bad_x'])} AS bad_n_ep,
  v.rapid AS lim_rapid, v.rapid * {_f(tf['bad_x'])} AS bad_rapid,
  {_f(LJ['count_drops_geh'][0])} AS lim_level, {_f(LJ['count_drops_geh'][1])} AS bad_level,
  {_f(LJ['night_geh'][0])} AS lim_night_drop, {_f(LJ['night_geh'][1])} AS bad_night_drop,
  {_f(LJ['count_held_on_min'][0])} AS lim_count_on, {_f(LJ['count_held_on_min'][1])} AS bad_count_on,
  CASE WHEN x.wg = 'h24' THEN greatest(v.st_lo, {_f(LJ['hold_floor_s'])})
       ELSE least(greatest(v.sx, 300.0), 900.0) END AS lim_stuck,
  CASE WHEN x.wg = 'h24' THEN greatest(v.st_hi, v.st_lo, {_f(LJ['hold_floor_s'])})
       ELSE greatest({_f(a)} + {_f(b)} * least(greatest(v.sx, 300.0), 900.0), {_f(c)} * least(greatest(v.sx, 300.0), 900.0))
       END AS bad_stuck,
  CASE WHEN x.wg = 'h24' THEN v.si_lo ELSE {_f(cur['silent_exp'])} END AS lim_dropout,
  CASE WHEN x.wg = 'h24' THEN v.si_hi ELSE {_f(cur['silent_exp'] * sh['silent_bad_x'])} END AS bad_dropout,
  CASE WHEN x.wg = 'h24' THEN (SELECT max(l.lo) FROM ltab l WHERE l.tab = 'busy_z' AND l.key = x.fn || '|' || x.span)
       ELSE {_f(cur['busy_z'])} END AS lim_prof,
  CASE WHEN x.wg = 'h24' THEN {_f(LJ['too_many_h24_veh_h_lane'][0])}
       ELSE greatest(CASE WHEN x.fn = 'Advance' THEN v.q5all ELSE v.q5gy END, {_f(cur['too_many_veh_h_lane'])}) END
       AS lim_volume,
  CASE WHEN x.wg = 'h24' THEN {_f(LJ['too_many_h24_veh_h_lane'][1])}
       ELSE greatest(CASE WHEN x.fn = 'Advance' THEN v.q5all ELSE v.q5gy END, {_f(cur['too_many_veh_h_lane'])})
            * {_f(sh['too_many_bad_x'])} END AS bad_volume
FROM xx0 x JOIN (
  SELECT dev, det, max(n_ep) AS n_ep, max(rapid) AS rapid, max(st_lo) AS st_lo, max(st_hi) AS st_hi,
         max(si_lo) AS si_lo, max(si_hi) AS si_hi, max(q5gy) AS q5gy, max(q5all) AS q5all,
         CASE WHEN bool_or(own AND stuck_x IS NOT NULL) THEN max(stuck_x) END AS sx
  FROM lval GROUP BY dev, det) v USING (dev, det)""", mat=True)


def _tod(q: _Q, LJ: dict, R: dict):
    """busy at night (24-h samples, h116): per whole day the night level against its type's model and its phase mates;
    the day with the strongest finding is kept; a signal note when many detectors had an unusual night."""
    g = R["tod"]["groups"]
    rows = [[_s(k), _f(v["bc"][0]), _f(v["bc"][1]), _f(v["bc"][2]), _f(v["sc"]), "true" if v["occ"] else "false",
             _f(v["bo"][0]) if v["occ"] else "NULL", _f(v["bo"][1]) if v["occ"] else "NULL",
             _f(v["so"]) if v["occ"] else "NULL", _f(v["cal_med"]), _f(v["cal_mad"])] for k, v in g.items()]
    q.add("todg", _values("key, b0, b1, b2, sc, occ, o0, o1, so, med, mad", rows), mat=True)
    P = LJ["busy"]
    # the whole days (00:00-24:00) of each 24-h window; one not starting at midnight is scored by clock hour as one day
    q.add("tdays", f"""
WITH w AS (SELECT dev, s_ns, e_ns, nbb, T, s_ns - sod + CASE WHEN sod > 0 THEN {DAYNS} ELSE 0 END AS f0
           FROM win WHERE wg = 'h24' AND bs = 900),
d AS (SELECT dev, s_ns, nbb, f0 + unnest(range((e_ns - f0) // {DAYNS})) * {DAYNS} AS d0 FROM w
      WHERE e_ns >= f0 + {DAYNS}
      UNION ALL SELECT dev, s_ns, nbb, s_ns FROM w WHERE e_ns < f0 + {DAYNS} AND T >= 86400)
SELECT dev, s_ns, nbb, d0, round_even((d0 - s_ns) / 9e11, 0)::INT AS j0,
       ((d0 % {DAYNS}) // 1000000000 // 900) % 96 AS sh
FROM d WHERE round_even((d0 - s_ns) / 9e11, 0) >= 0 AND round_even((d0 - s_ns) / 9e11, 0) + 96 <= nbb""", mat=True)
    q.add("tdh", """
SELECT dev, det, d0, hr, CASE WHEN count(H) = 2 THEN sum(H) END AS N, avg(Q) AS O FROM (
  SELECT g.dev, g.det, t.d0, ((g.k - t.j0 + t.sh) % 96) // 2 AS hh, ((g.k - t.j0 + t.sh) % 96) // 4 AS hr,
         CASE WHEN count(g.n) > 0 THEN sum(g.n) END AS H, avg(g.occ) AS Q
  FROM tdays t JOIN gb g ON g.dev = t.dev AND g.k >= t.j0 AND g.k < t.j0 + 96
  JOIN xbase x ON x.dev = g.dev AND x.det = g.det GROUP BY g.dev, g.det, t.d0, hh, hr)
GROUP BY dev, det, d0, hr""")
    q.add("tdf", f"""
SELECT a.*, sqrt((a.nn + 0.5) / (b.b4n + 0.5)) AS r, asin(sqrt(a.onm)) AS onv,
       asin(sqrt(least(greatest(b.b4o, 0.0), 1.0))) AS od, log10(1 + a.tot) AS lv,
       CASE WHEN a.tot / 24.0 < 20 THEN 'low' WHEN a.tot / 24.0 < 100 THEN 'medium' ELSE 'high' END AS band
FROM (SELECT dev, det, d0, sum(N) AS tot, avg(N) FILTER (hr >= 1 AND hr < 5) AS nn, sum(N) FILTER (hr >= 1 AND hr < 5)
             AS n_night, avg(least(greatest(O, 0.0), 1.0)) FILTER (hr >= 1 AND hr < 5) AS onm
      FROM tdh GROUP BY dev, det, d0 HAVING count(N) = 24 AND count(O) = 24 AND sum(N) >= 50) a
JOIN (SELECT a.dev, a.det, a.d0, max(s4n) / 4 AS b4n, max(s4o) / 4 AS b4o FROM (
        SELECT a.dev, a.det, a.d0, a.hr, sum(b.N) AS s4n, sum(least(greatest(b.O, 0.0), 1.0)) AS s4o
        FROM tdh a JOIN tdh b ON b.dev = a.dev AND b.det = a.det AND b.d0 = a.d0 AND (b.hr - a.hr + 24) % 24 < 4
        GROUP BY a.dev, a.det, a.d0, a.hr) a GROUP BY a.dev, a.det, a.d0) b USING (dev, det, d0)""")
    # z of the detector's own type, and the smallest over the plausible functions when the model is unsure
    q.add("tdz", """
SELECT f.dev, f.det, f.d0, c.own, ((greatest((f.r - (g.b0 + g.b1 * f.od + g.b2 * f.lv)) / g.sc,
       CASE WHEN g.occ THEN (f.onv - (g.o0 + g.o1 * f.lv)) / g.so END)) - g.med) / g.mad AS z
FROM tdf f JOIN (SELECT dev, det, fa, own FROM lcand) c USING (dev, det) JOIN xbase x USING (dev, det)
JOIN (SELECT f.dev, f.det, f.d0, c.fa, first(g.key ORDER BY o.o) AS key
      FROM tdf f JOIN lcand c USING (dev, det) JOIN xbase x USING (dev, det),
      (SELECT 1 AS o UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4) o, todg g
      WHERE g.key = CASE o.o WHEN 1 THEN c.fa || '|' || x.span || '|' || f.band WHEN 2 THEN c.fa || '|*|' || f.band
                             WHEN 3 THEN c.fa || '|' || x.span || '|*' ELSE c.fa || '|*|*' END
      GROUP BY f.dev, f.det, f.d0, c.fa) k ON k.dev = f.dev AND k.det = f.det AND k.d0 = f.d0 AND k.fa = c.fa
JOIN todg g ON g.key = k.key""")
    q.add("tdd", f"""
SELECT f.*, z.z_own AS tod_z0, CASE WHEN z.z_own IS NOT NULL THEN least(z.z_own, z.z_alt) END AS tod_z,
       f.r - m.mate_r AS tod_dph, m.mate_r AS tod_mate_r, l.lim_prof
FROM tdf f JOIN (SELECT dev, det, d0, max(z) FILTER (own) AS z_own, min(z) FILTER (NOT own) AS z_alt FROM tdz
                 GROUP BY dev, det, d0) z USING (dev, det, d0)
JOIN xbase x USING (dev, det) JOIN limits l USING (dev, det)
LEFT JOIN (SELECT a.dev, a.det, a.d0, median(b.r) AS mate_r FROM tdf a JOIN xbase xa USING (dev, det)
           JOIN xbase xb2 ON xb2.dev = a.dev AND xb2.ph = xa.ph AND xb2.det <> a.det
           JOIN tdf b ON b.dev = a.dev AND b.det = xb2.det AND b.d0 = a.d0
           JOIN (SELECT dev, det, d0, max(z) FILTER (own) AS zo FROM tdz GROUP BY dev, det, d0) bz
             ON bz.dev = b.dev AND bz.det = b.det AND bz.d0 = b.d0
           WHERE bz.zo IS NOT NULL AND b.tot >= 200 GROUP BY a.dev, a.det, a.d0) m USING (dev, det, d0)""")
    q.add("tdl", f"""
SELECT *, CASE WHEN tod_z IS NULL THEN NULL
               WHEN n_night >= {_f(P['min_night'])} AND coalesce(tod_dph, -1) >= {_f(P['mates_delta'])}
                    AND tod_z >= lim_prof * {_f(P['z_x'])} AND tod_z >= lim_prof THEN 0.8
               WHEN n_night >= {_f(P['min_night'])} AND coalesce(tod_dph, -1) >= {_f(P['mates_delta'])}
                    AND tod_z >= lim_prof * {_f(P['z_x'])} THEN 0.5 ELSE 0.0 END AS lvl
FROM tdd""")
    q.add("tod", """
SELECT dev, det, first(tod_z ORDER BY coalesce(lvl, -1) DESC, d0) AS tod_z,
       first(tod_dph ORDER BY coalesce(lvl, -1) DESC, d0) AS tod_dph,
       first(n_night ORDER BY coalesce(lvl, -1) DESC, d0) AS tod_n_night,
       first(r ORDER BY coalesce(lvl, -1) DESC, d0) AS tod_r, first(tod_mate_r ORDER BY coalesce(lvl, -1) DESC, d0) AS tod_mate_r,
       first(d0 ORDER BY coalesce(lvl, -1) DESC, d0) AS tod_d0
FROM tdl GROUP BY dev, det""")
    q.add("todnote", """
SELECT dev, first(d0 ORDER BY n_odd / greatest(n_ok, 1) DESC, d0) AS d0, first(n_odd ORDER BY n_odd / greatest(n_ok, 1)
       DESC, d0) AS n_odd, first(n_ok ORDER BY n_odd / greatest(n_ok, 1) DESC, d0) AS n_ok
FROM (SELECT dev, d0, count(*) FILTER (tod_z >= 3.5) AS n_odd, count(*) AS n_ok FROM tdd GROUP BY dev, d0)
WHERE n_odd >= 5 AND n_odd >= 0.20 * n_ok GROUP BY dev""")


def _verdict(q: _Q, p: str, src: str, LJ: dict):
    """the check scores and the resolver / severity (health_v4.scores + status) for the rows of `src` -> {p}V."""
    P = LJ["busy"]
    sd = _sc("x.drop_lam", "l.lim_dropout", "l.bad_dropout")
    q.add(f"{p}S", f"""
SELECT x.dev, x.det, s.s_stuck_v AS v_stuck,
  CASE WHEN x.count_on_ok THEN {_sc('coalesce(x.hi_min_c, 0)', 'l.lim_count_on', 'l.bad_count_on')} END AS v_count_on,
  CASE WHEN x.fn = 'Bike' THEN NULL WHEN x.silent THEN 0.0 ELSE x.sd END AS v_dropout,
  CASE WHEN x.fn = 'Bike' OR NOT x.geh_ok THEN NULL
       WHEN x.geh_drop IS NOT NULL AND x.occ_mates IS NOT NULL AND x.occ_own / greatest(x.occ_mates, 1e-9) < 0.8
       THEN {_sc('x.geh_drop', 'l.lim_level', 'l.bad_level')} ELSE 0.0 END AS v_level,
  CASE WHEN x.fn = 'Bike' THEN NULL WHEN x.moved AND coalesce(x.drop_lam, 0) < 100 THEN least(x.sd, 0.5)
       WHEN x.moved THEN x.sd ELSE x.nd END AS v_night_drop,
  CASE WHEN x.fn <> 'Bike' AND coalesce(CASE WHEN x.silent THEN 0.0 ELSE x.sd END, 0) >= 0.35
       THEN CASE WHEN x.pv IS NOT NULL THEN 0.0 END ELSE x.pv END AS v_prof,
  CASE WHEN x.N >= 50 THEN {_sc('x.xr + 1e-9 * (x.xr > 0)::INT', '1.0', 'l.bad_rapid / l.lim_rapid')} END AS v_rapid,
  CASE WHEN x.q5_use IS NOT NULL THEN {_sc('x.q5_use / l.lim_volume', '1.0', 'l.bad_volume / l.lim_volume')} END
       AS v_volume,
  CASE WHEN x.spk_lvl >= 2 THEN 1.0 WHEN x.spk_lvl >= 1 THEN 0.5 WHEN x.spk_lvl IS NOT NULL THEN 0.0 END AS v_spiky,
  x.silent AS silent_night
FROM (SELECT x.*, coalesce(x.sd0, 0) >= 0.35 AND x.nf >= 0.8 AS silent,
             coalesce(x.sd0, 0) >= 0.35 AND x.nf >= 0.8 AND coalesce(x.sd0, 0) > coalesce(x.nd, 0) + 1e-9 AS moved,
             x.sd0 AS sd
      FROM (SELECT x.*, CASE WHEN x.n_on > 0 THEN CASE WHEN x.drop_in_stuck THEN 0.0
                 WHEN x.s_dropout = 0.35 AND x.drop_lam > 30.001 AND {sd} > 0.35 THEN 0.35 ELSE {sd} END END AS sd0,
                   {_sc('x.geh_night', 'l.lim_night_drop', 'l.bad_night_drop')} AS nd,
                   CASE WHEN x.tod_z IS NULL THEN NULL
                        WHEN x.tod_n_night >= {_f(P['min_night'])} AND coalesce(x.tod_dph, -1) >= {_f(P['mates_delta'])}
                             AND x.tod_z >= l.lim_prof * {_f(P['z_x'])} AND x.tod_z >= l.lim_prof THEN 0.8
                        WHEN x.tod_n_night >= {_f(P['min_night'])} AND coalesce(x.tod_dph, -1) >= {_f(P['mates_delta'])}
                             AND x.tod_z >= l.lim_prof * {_f(P['z_x'])} THEN 0.5 ELSE 0.0 END AS pv,
                   CASE WHEN x.n_fast >= 3 THEN x.fast_rate / l.lim_rapid ELSE 0.0 END AS xr,
                   CASE WHEN x.drop_b1 > x.drop_b0 THEN (SELECT count(*) FILTER (b.hr >= 21 OR b.hr < 5) FROM bins5 b
                        WHERE b.dev = x.dev AND b.b >= x.drop_b0 AND b.b < x.drop_b1) / (x.drop_b1 - x.drop_b0)
                        ELSE 0.0 END AS nf
            FROM {src} x JOIN limits l USING (dev, det)) x) x
JOIN limits l USING (dev, det) JOIN stk s USING (dev, det)""", mat=True)
    fk = ", ".join(f"coalesce(v_{k} >= 0.35, false) AS f_{k}" for k in KEYS)
    q.add(f"{p}F0", f"""
SELECT s.*, {fk}, x.queue_pat8, x.c_occ_like, x.hours, x.status1, x.status AS status2, x.n_hmates, x.n_on,
       x.drop_b0, x.drop_b1, x.fn, x.c1_pre, x.hi_min_c, x.fe_n, x.fe_phi
FROM {p}S s JOIN {src} x USING (dev, det)""")
    # Q2 (queue pattern explains a count drop), R6 (silence in a sample under 1 h: watch), counts fall = one finding
    q.add(f"{p}F1", f"""
SELECT * REPLACE (
    CASE WHEN f_dropout AND f_level AND NOT (v_dropout >= v_level) THEN false ELSE f_dropout END AS f_dropout,
    CASE WHEN f_dropout AND f_level AND v_dropout >= v_level THEN false ELSE f_level END AS f_level)
FROM (SELECT * REPLACE (f_level AND NOT (queue_pat8 AND coalesce(c_occ_like >= 0.8, false)) AS f_level,
                        f_dropout AND NOT (hours < 1) AS f_dropout),
             f_dropout AND hours < 1 AS w_dropout_short,
             status2 <> status1 AND n_hmates = 0 AND status1 IN ('suspect', 'bad') AS w_no_yardstick
      FROM {p}F0)""")
    stat_border = " OR ".join(f"(f_{k} AND v_{k} < 0.42)" for k in STAT)
    nf = " + ".join(f"f_{k}::INT" for k in KEYS)
    q.add(f"{p}F2", f"""
SELECT *, ({nf}) = 1 AND ({stat_border}) AS r7 FROM {p}F1""")
    keep = ", ".join(f"f_{k} AND NOT r7 AS f_{k}" for k in KEYS)
    bord = ", ".join(f"f_{k} AND r7 AS b_{k}" for k in KEYS)
    q.add(f"{p}F3", f"SELECT * REPLACE ({keep}), {bord} FROM {p}F2")
    prod = " * ".join(f"(1 - CASE WHEN f_{k} THEN v_{k} ELSE 0 END)" for k in KEYS)
    anyfin = " OR ".join(f"v_{k} IS NOT NULL" for k in KEYS)

    def real(k):
        return f"(f_{k} AND NOT coalesce(v_{k} = 0.35, false))" if k in ("stuck", "dropout") else f"f_{k}"
    fams = sorted(set(FAMILY.values()))
    nfam = " + ".join("(" + " OR ".join(real(k) for k in KEYS if FAMILY[k] == f) + ")::INT" for f in fams)
    evs = sorted(set(EVID.values()))
    nev = " + ".join("(" + " OR ".join(real(k) for k in KEYS if EVID[k] == e) + ")::INT" for e in evs)
    q.add(f"{p}F4", f"""
SELECT *, CASE WHEN {anyfin} THEN {prod} END AS score, {nev} AS n_ev, {nfam} AS n_fam,
       (drop_b1 - drop_b0) * 300 / 3600.0 AS silent_h,
       CASE WHEN {prod} < 0.25 AND ({anyfin}) THEN 'bad' WHEN {prod} < 0.70 AND ({anyfin}) THEN 'suspect'
            WHEN r7 OR w_dropout_short OR w_no_yardstick THEN 'watch' WHEN n_on < 20 THEN 'not_enough_data'
            ELSE 'ok' END AS st0
FROM {p}F3""")
    q.add(f"{p}V", f"""
SELECT * EXCLUDE (st0, st1, st2, st3),
       CASE WHEN st3 = 'bad' AND f_rapid AND ({nf}) = 1 AND fe_n >= 8 AND fe_phi <= 2.0
                 AND fn IN ('Presence', 'Other', 'Mid') THEN 'suspect' ELSE st3 END AS health_st,
       CASE WHEN st3 = 'bad' AND f_rapid AND ({nf}) = 1 AND fe_n >= 8 AND fe_phi <= 2.0
                 AND fn IN ('Presence', 'Other', 'Mid') THEN 'F'
            WHEN st3 <> st2 THEN 'C' WHEN st2 <> st1 THEN 'P' WHEN st1 <> st0 AND n_fam < 2 THEN 'A' ELSE '' END AS sev_rule
FROM (SELECT *, CASE WHEN fn = 'Count' AND c1_pre AND hi_min_c >= 30 AND st2 IN ('ok', 'watch') THEN 'suspect'
                     ELSE st2 END AS st3
      FROM (SELECT *, CASE WHEN st1 = 'suspect' AND (f_prof OR (f_dropout AND silent_h >= 4.0)) THEN 'bad' ELSE st1 END
                   AS st2
            FROM (SELECT *, CASE WHEN st0 = 'suspect' AND (n_fam >= 2 OR n_ev >= 2) THEN 'bad' ELSE st0 END AS st1
                  FROM {p}F4)))""", mat=True)


def _assemble(q: _Q, LJ: dict, R: dict):
    """the scored table X, holds with their flags, limits, the pre-pass and the clean silence yardstick -> xf, vf."""
    q.add("xx0", f"""
SELECT x.*, a.rep_frac, a.max_on_s, a.rep_time_s, a.dur_p50, a.n_on_a, a.n_rep, a.n_start, a.n_ge5, a.n_ge60,
       a.long_max_s, s.elhi_cnt, s.elhi_occ, k.c_occ_like, q.q5_gy, q.q5_all, f.fe_n, f.fe_p, f.fe_phi, t.N, t.n_fast,
       t.fast_rate, t.gap_lim, t.pair_share, x.n_on / x.hours AS rate,
       CASE WHEN x.n_on / x.hours < 20 THEN 'low' WHEN x.n_on / x.hours < 100 THEN 'medium' ELSE 'high' END AS band,
       x.fn || ' ' || x.span AS type, greatest(x.ep_dur, a.max_on_s) AS stuck_x,
       coalesce(s.elhi_occ > 0 AND s.elhi_occ > s.elhi_cnt, false) AS queue_pat8,
       CASE WHEN x.ph IS NOT NULL THEN hm.nh - (x.status1 <> 'bad')::INT END AS n_hmates,
       o.hi_min_c, o.hi_occ_max, coalesce(g.geh_ok, false) AS geh_ok, g.geh_night,
       CASE WHEN g.geh_night IS NOT NULL THEN g.geh_night_n0 END AS geh_night_n,
       CASE WHEN g.geh_night IS NOT NULL THEN g.geh_night_exp0 END AS geh_night_exp, g.geh_drop, g.drop_t,
       g.own_before, g.own_after, g.exp_after, g.occ_own, g.occ_mates, p.spk_bursts, p.spk_gmax, p.spk_daily, p.spk_lvl,
       m.mocc AS occ
FROM xm x LEFT JOIN act a USING (dev, det) LEFT JOIN slp s USING (dev, det) LEFT JOIN lkc k USING (dev, det)
LEFT JOIN q5 q USING (dev, det) LEFT JOIN fe f USING (dev, det) LEFT JOIN tf t USING (dev, det)
LEFT JOIN ohi o USING (dev, det) LEFT JOIN geh g USING (dev, det) LEFT JOIN spk p USING (dev, det)
LEFT JOIN (SELECT DISTINCT dev, det, mocc FROM gb) m USING (dev, det)
LEFT JOIN (SELECT dev, ph, sum((status1 <> 'bad')::INT) AS nh FROM xm GROUP BY dev, ph) hm
  ON hm.dev = x.dev AND hm.ph = x.ph""", mat=True)
    _limits(q, LJ, R)
    _tod(q, LJ, R)
    # Count zone held ON / extension; the queue excuse is never for a counting zone
    q.add("xx", f"""
SELECT x.*, t.tod_z, t.tod_dph, t.tod_n_night, t.tod_r, t.tod_mate_r, t.tod_d0,
       x.fn = 'Count' AND (len(x.alts) = 0 OR list_has_all(['Count'], x.alts)) AND NOT x.c1 AS count_on_ok,
       x.c1 AS c1_pre, CASE WHEN x.c1 THEN 'C1' WHEN x.c2 THEN 'C2' ELSE '' END AS cfg,
       x.fn IN {_in(NO_SHARED)} AND NOT list_has_any(x.alts, ['Presence', 'Other', 'Mid']) AS noq,
       CASE WHEN x.fn = 'Advance' THEN x.q5_all ELSE x.q5_gy END AS q5_use
FROM (SELECT x.*, coalesce(x.fn = 'Count' AND ((x.rep_time_s >= 600 AND x.rep_time_s >= 0.70 * x.occ * x.hours * 3600)
                           OR (x.n_rep >= 20 AND x.n_rep >= 0.05 * x.n_on)), false) AS c1,
             coalesce(x.fn = 'Count' AND x.cmode = 'pulse' AND (x.n_ge60 >= 1 OR x.n_ge5 >= 3), false) AS c2
      FROM xx0 x) x
LEFT JOIN tod t USING (dev, det)""", mat=True)
    # holds with the queue / traffic test (episode_flags)
    q.add("ef", f"""
SELECT e.*, e.q_ok AND e.dur < 3600 AND NOT e.common AS q1,
       e.q_ok AND e.dur >= 3600 AND coalesce(e.cover >= 0.75, false) AND NOT e.lifted0 AND NOT e.common AS q1b,
       e.q_ok AND e.lifted0 AS lifted,
       CASE WHEN e.fn IN {_in(NO_SHARED)} THEN 0 ELSE e.co5 END >= 3 AND NOT e.common AS shared
FROM (SELECT e.*, coalesce(e.q_ok0 AND e.dur >= 3600 AND e.cover >= 0.75 AND e.busy_min_e >= 60 AND e.busy_occ_e < 0.5
                           AND NOT e.ref_held_e, false) AS lifted0, coalesce(e.q_ok0, false) AS q_ok
      FROM (SELECT e.*, x.fn, coalesce(CASE WHEN x.noq THEN 0 ELSE e.n_hpeer_e END, 0) > 0 AND e.phx >= 1.5
                   AND e.corr_h_e >= 0.70 AND e.refx_e >= 0.5 AND coalesce(e.light_e, 1) = 0 AND e.trafx_e >= 0.5 AS q_ok0
            FROM (SELECT e.*, e.cover_e AS cover, CASE WHEN coalesce(e.cover_e, 0) >= 0.75
                         THEN greatest(coalesce(e.phx_h_e, 0), 1.5) ELSE e.phx_h_e END AS phx FROM ectx e) e
            LEFT JOIN xx x USING (dev, det)) e) e""", mat=True)
    q.add("efl", """
SELECT e.*, l.lim_stuck, CASE WHEN e.common THEN 'together' WHEN e.q1 OR e.q1b THEN 'queue'
       WHEN e.shared THEN 'shared' WHEN e.lifted THEN 'not_queue' ELSE 'counted' END AS label,
       NOT e.q1 AND NOT e.q1b AND NOT e.shared AS counted
FROM ef e JOIN limits l USING (dev, det) WHERE e.dur >= l.lim_stuck""", mat=True)
    # stuck on: counted holds over the limit; total time and number against the limits; caps (health_v4._stuck)
    q.add("stk", f"""
SELECT x.dev, x.det, CASE WHEN coalesce(e.n_common, 0) > 0 THEN 1.0
  WHEN coalesce(e.n_ep, 0) > 0 AND e.n_ep = e.n_lift AND sv > {_f(0.35)} THEN 0.35
  WHEN coalesce(e.n_over, 0) > 0 AND coalesce(e.n_ep, 0) = 0 AND coalesce(e.n_sh, 0) = 0 AND e.n_q > 0 THEN 0.0
  WHEN coalesce(e.n_ep, 0) = 0 AND coalesce(e.n_sh, 0) > 0 THEN 0.35 ELSE sv END AS s_stuck_v,
  coalesce(e.n_ep, 0) AS st_n_ep, coalesce(e.tot, 0) AS st_tot_s
FROM (SELECT x.dev, x.det, CASE WHEN x.s_stuck = 0.35 AND coalesce(e.n_ep, 0) <= 1 AND s0 > 0.35 THEN 0.35 ELSE s0 END AS sv
      FROM (SELECT x.*, CASE WHEN st IS NULL THEN coalesce(sn, 0) ELSE greatest(st, coalesce(sn, 0)) END AS s0 FROM (
              SELECT x.dev, x.det, x.s_stuck, {_sc('coalesce(e.tot, 0.0)', 'l.lim_stuck', 'l.bad_stuck')} AS st,
                     {_sc('coalesce(e.n_ep, 0)::DOUBLE', 'l.lim_n_ep', 'l.bad_n_ep')} AS sn
              FROM xx x JOIN limits l USING (dev, det) LEFT JOIN (
                SELECT dev, det, count(*) FILTER (counted) AS n_ep, sum(dur) FILTER (counted) AS tot FROM efl
                GROUP BY dev, det) e USING (dev, det)) x) x
      LEFT JOIN (SELECT dev, det, count(*) FILTER (counted) AS n_ep FROM efl GROUP BY dev, det) e USING (dev, det)) x
LEFT JOIN (SELECT dev, det, count(*) AS n_over, count(*) FILTER (counted) AS n_ep, sum(dur) FILTER (counted) AS tot,
                  count(*) FILTER (q1 OR q1b) AS n_q, count(*) FILTER (shared AND NOT q1 AND NOT q1b) AS n_sh,
                  count(*) FILTER (counted AND lifted) AS n_lift, count(*) FILTER (common) AS n_common
           FROM efl GROUP BY dev, det) e USING (dev, det)""", mat=True)
    # pre-pass: phase mates with a finding of their own (not a silence) leave the silence yardstick
    _verdict(q, "pre", "xx", LJ)
    q.add("flag", """
SELECT dev, det, f_stuck OR f_count_on OR f_prof OR f_rapid OR f_volume OR f_spiky AS flagged FROM preV""")
    # goes silent, clean yardstick (h118c_dropout): references by the phase of the scored detectors only
    q.add("dcph", "SELECT d.dev, d.det, x.ph FROM dets d LEFT JOIN xbase x USING (dev, det)", mat=True)
    q.add("dcrf", """
SELECT a.dev, a.det, b.det AS r, coalesce(a.ph = b.ph, false) AS same, coalesce(f.flagged, false) AS fl
FROM dcph a JOIN dcph b ON a.dev = b.dev AND a.det <> b.det JOIN dstat sb ON sb.dev = b.dev AND sb.det = b.det AND sb.live
LEFT JOIN twin t ON t.dev = a.dev AND t.i = a.det AND t.j = b.det LEFT JOIN flag f ON f.dev = b.dev AND f.det = b.det
WHERE t.i IS NULL""")
    q.add("dccand", """
SELECT x.dev, x.det, coalesce(k.nsame, 0) >= 2 AS isph, coalesce(k.nkeep, 0) >= 2 AS keep2
FROM xx x LEFT JOIN (SELECT dev, det, count(*) FILTER (same) AS nsame, count(*) FILTER (same AND NOT fl) AS nkeep
                     FROM dcrf GROUP BY dev, det) k USING (dev, det)
WHERE x.drop_lam >= 30 AND x.n_on > 0""", mat=True)
    # members of S0 (yardstick as in stage 1) and S1 (without the flagged detectors); dropped = flagged ones left out
    q.add("dcm", """
SELECT c.dev, c.det, 0 AS v, f.r FROM dccand c JOIN dcrf f USING (dev, det) WHERE c.isph AND f.same
UNION ALL SELECT c.dev, c.det, 0, d.det FROM dccand c JOIN dets d ON d.dev = c.dev AND d.det <> c.det WHERE NOT c.isph
UNION ALL SELECT c.dev, c.det, 1, f.r FROM dccand c JOIN dcrf f USING (dev, det) WHERE c.isph AND c.keep2 AND f.same
          AND NOT f.fl
UNION ALL SELECT c.dev, c.det, 1, d.det FROM dccand c JOIN dets d ON d.dev = c.dev AND d.det <> c.det
          LEFT JOIN flag f ON f.dev = d.dev AND f.det = d.det WHERE NOT (c.isph AND c.keep2)
          AND NOT coalesce(f.flagged, false)""")
    q.add("dcnd", """
SELECT c.dev, c.det, CASE WHEN c.isph AND c.keep2 THEN count(*) FILTER (f.same AND f.fl)
                          ELSE count(*) FILTER (fl2) END AS n_dropped
FROM dccand c JOIN (SELECT a.dev, a.det, a.same, a.fl, coalesce(g.flagged, false) AS fl2 FROM
                      (SELECT d.dev, c.det, d.det AS r, coalesce(f.same, false) AS same, coalesce(f.fl, false) AS fl
                       FROM dccand c JOIN dets d ON d.dev = c.dev AND d.det <> c.det
                       LEFT JOIN dcrf f ON f.dev = c.dev AND f.det = c.det AND f.r = d.det) a
                    LEFT JOIN flag g ON g.dev = a.dev AND g.det = a.r) f USING (dev, det)
GROUP BY c.dev, c.det, c.isph, c.keep2""")
    q.add("dcrs0", """
SELECT u.dev, u.det, m.v, u.a, u.z, m.r, w.nb5 FROM runs u JOIN dcm m ON m.dev = u.dev AND m.det = u.det
JOIN win w ON w.dev = u.dev""", mat=True)
    _at(q, "dcrs", "dcrs0", "r", ("a", "z", "nb5"), "cn")
    q.add("dcrs", """
SELECT dev, det, v, a, z, sum(cn_z - cn_a) AS s_in, sum(cn_nb5) AS s_tot FROM dcrs_3 GROUP BY dev, det, v, a, z""")
    q.add("dc", """
SELECT c.dev, c.det, coalesce(r0.lam, 0.0) AS drop_lam_chk, coalesce(r1.lam, 0.0) AS drop_lam_c,
       coalesce(r1.b0, -1) AS drop_b0_c, coalesce(r1.b1, -1) AS drop_b1_c, n.n_dropped
FROM dccand c JOIN dcnd n USING (dev, det)
LEFT JOIN (SELECT dev, det, v, max(lam) AS lam, first(a ORDER BY lam DESC, a) AS b0, first(z ORDER BY lam DESC, a) AS b1
           FROM (SELECT r.*, s.n_on / greatest(r.s_tot - r.s_in, 1) * r.s_in AS lam FROM dcrs r JOIN dstat s USING (dev, det))
           WHERE lam > 0 GROUP BY dev, det, v) r0 ON r0.dev = c.dev AND r0.det = c.det AND r0.v = 0
LEFT JOIN (SELECT dev, det, v, max(lam) AS lam, first(a ORDER BY lam DESC, a) AS b0, first(z ORDER BY lam DESC, a) AS b1
           FROM (SELECT r.*, s.n_on / greatest(r.s_tot - r.s_in, 1) * r.s_in AS lam FROM dcrs r JOIN dstat s USING (dev, det))
           WHERE lam > 0 GROUP BY dev, det, v) r1 ON r1.dev = c.dev AND r1.det = c.det AND r1.v = 1""")
    q.add("xf", """
SELECT x.* REPLACE (
    CASE WHEN m THEN x.drop_lam * c.drop_lam_c / c.drop_lam_chk ELSE x.drop_lam END AS drop_lam,
    CASE WHEN m THEN c.drop_b0_c ELSE x.drop_b0 END AS drop_b0,
    CASE WHEN m THEN c.drop_b1_c ELSE x.drop_b1 END AS drop_b1,
    CASE WHEN m THEN CASE WHEN x.drop_in_stuck THEN 0.0 WHEN x.drop_lam * c.drop_lam_c / c.drop_lam_chk < 30 THEN 0.0
                          ELSE 0.35 + 0.65 * least((x.drop_lam * c.drop_lam_c / c.drop_lam_chk - 30) / 70, 1) END
         ELSE x.s_dropout END AS s_dropout)
FROM (SELECT x.*, coalesce(c.n_dropped > 0 AND c.drop_lam_chk > 0, false) AS m FROM xx x LEFT JOIN dc c USING (dev, det)) x
LEFT JOIN dc c USING (dev, det)""", mat=True)
    _verdict(q, "fin", "xf", LJ)
    _outputs(q)


def _dur(x: str) -> str:
    """health_v4._dur: '45 s', '12 min', '2 h 5 min'."""
    m = f"round_even(({x}) / 60.0, 0)::BIGINT"
    return (f"(CASE WHEN ({x}) < 300 THEN format('{{:.0f}} s', ({x})) WHEN {m} < 60 THEN {m} || ' min' "
            f"ELSE ({m} // 60) || ' h' || CASE WHEN {m} % 60 > 0 THEN ' ' || ({m} % 60) || ' min' ELSE '' END END)")


def _tstr(ns: str) -> str:
    """str(pd.Timestamp) of epoch ns: seconds, then microseconds or nanoseconds when not zero."""
    return (f"(strftime(make_timestamp(({ns}) // 1000), '%Y-%m-%d %H:%M:%S') || CASE "
            f"WHEN ({ns}) % 1000 <> 0 THEN '.' || lpad((({ns}) % 1000000000)::VARCHAR, 9, '0') "
            f"WHEN ({ns}) % 1000000000 <> 0 THEN '.' || lpad(((({ns}) % 1000000000) // 1000)::VARCHAR, 6, '0') "
            f"ELSE '' END)")


def _hm(ns: str, fmt: str = "%a %H:%M") -> str:
    return f"strftime(make_timestamp(({ns}) // 1000), '{fmt}')"


def _outputs(q: _Q):
    """categories with severity, the plain reason, configuration notes, watch notes, bad periods (JSON), signal notes."""
    kv = _values("key, ki, cat", [[_s(k), str(i), _s(CATEGORY[k])] for i, k in enumerate(KEYS)])
    fsel = " ".join(f"WHEN '{k}' THEN v.f_{k}" for k in KEYS)
    vsel = " ".join(f"WHEN '{k}' THEN v.v_{k}" for k in KEYS)
    # the reported findings: the kept ones, plus the Count zone held ON of rule C
    q.add("fnd", f"""
SELECT v.dev, v.det, k.key, k.ki, k.cat, coalesce(CASE k.key {vsel} END, 0.0) AS sc,
       CASE WHEN (v.sev_rule = 'C' AND k.key = 'count_on') OR (v.sev_rule = 'F' AND k.key = 'rapid') THEN 'suspect'
            WHEN v.sev_rule = 'P' AND (k.key = 'prof' OR (k.key = 'dropout' AND v.silent_h >= 4.0)) THEN 'bad'
            WHEN coalesce(CASE k.key {vsel} END, 0.0) >= 0.75 THEN 'bad' ELSE 'suspect' END AS sev
FROM finV v, ({kv}) k WHERE CASE k.key {fsel} END OR (k.key = 'count_on' AND v.sev_rule = 'C')""", mat=True)
    # stuck-on holds over the detector's limit, labelled (health_v4._stuck_eps)
    q.add("eps", """
SELECT e.dev, e.det, count(*) FILTER (e.label <> 'queue') AS n_cnt, sum(e.dur) FILTER (e.label <> 'queue') AS tot,
       min(e.s) FILTER (e.label <> 'queue') AS t0, count(*) FILTER (e.label = 'not_queue') AS n_nq,
       min(e.busy_occ_e) FILTER (e.label = 'not_queue') AS nq_occ, count(*) FILTER (e.label = 'together') AS n_tog,
       sum(e.dur) FILTER (e.label = 'together') AS tog_tot, min(e.s) FILTER (e.label = 'together') AS tog_t0,
       arg_min(e.dur, e.s) FILTER (e.label = 'together') AS tog_dur,
       list_sort(list_distinct(flatten(list(e.mates) FILTER (e.label = 'together')))) AS tog_mates
FROM efl e GROUP BY e.dev, e.det""")
    ns = "w.s_ns + round(({t}) * 1e9)::BIGINT"
    why = ("CASE WHEN e.n_nq > 0 AND e.n_nq = e.n_cnt THEN format('; not a queue: at the same traffic it was ON {:.0f} % "
           "of the time', 100 * e.nq_occ) ELSE '' END")
    stuck = f"""CASE WHEN e.n_tog > 0 THEN 'held ON ' || CASE WHEN e.n_tog = 1
        THEN {_dur('e.tog_dur')} || ' from ' || {_hm(ns.format(t='e.tog_t0'))}
        ELSE e.n_tog || ' times from ' || {_hm(ns.format(t='e.tog_t0'))} || ', ' || {_dur('e.tog_tot')} || ' in total' END
        || ' together with det ' || array_to_string(e.tog_mates, ', ') || ' (same start and end: a common failure)'
      WHEN e.n_cnt > 1 THEN 'held ON ' || e.n_cnt || ' times, ' || {_dur('e.tot')} || ' in total' || {why}
      WHEN e.n_cnt = 1 THEN 'held ON ' || {_dur('e.tot')} || ' from ' || {_hm(ns.format(t='e.t0'))} || {why}
      ELSE 'held ON up to ' || {_dur('x.stuck_x')} END"""
    silent = (f"'no actuations ' || {_hm('w.s_ns + x.drop_b0 * ' + str(NS5), '%H:%M')} || '-' || "
              f"{_hm('w.s_ns + x.drop_b1 * ' + str(NS5), '%H:%M')} || format('; ~{{:,.0f}} expected', x.drop_lam)")
    q.add("evid", f"""
SELECT f.dev, f.det, f.key, f.ki, f.cat, f.sc, f.sev, CASE f.key
  WHEN 'stuck' THEN {stuck}
  WHEN 'count_on' THEN format('count zone ON up to {{:.0f}} % in light traffic for {{:.0f}} min', 100 * x.hi_occ_max,
                              x.hi_min_c)
  WHEN 'dropout' THEN {silent}
  WHEN 'night_drop' THEN CASE WHEN v.silent_night THEN {silent} WHEN x.geh_night_exp IS NOT NULL
       THEN format('night {{:,.0f}} actuations vs ~{{:,.0f}} expected (GEH {{:.1f}})', x.geh_night_n, x.geh_night_exp,
                   x.geh_night) ELSE 'fewer actuations at night' END
  WHEN 'level' THEN 'counts dropped' || CASE WHEN x.drop_t IS NOT NULL THEN ' from ' || strftime(x.drop_t, '%a %H:%M')
       ELSE '' END || format(': {{:.0f}} per hour where its phase mates'' traffic gives ~{{:.0f}} (GEH {{:.1f}}); '
                             || 'its occupancy fell too', x.own_after, x.exp_after, x.geh_drop)
  WHEN 'rapid' THEN format('{{:,}} of {{:,}} ONs follow the previous ON within {{:.2g}} s ({{:.0f}} per 1,000; '
                           || 'limit {{:.0f}})', x.n_fast, x.N, x.gap_lim, x.fast_rate, l.lim_rapid)
       || CASE WHEN x.pair_share >= 0.9 THEN '; counts each vehicle twice' ELSE '' END
       || CASE WHEN v.sev_rule = 'F' THEN ', spread evenly with traffic - flickers as vehicles pass, check zone setup'
               ELSE '' END
  WHEN 'spiky' THEN format('{{:.0f}} bursts{{}} in 10-min counts at 3 x its usual level or more (largest GEH {{:.0f}}); '
                           || '~{{:,.0f}} vehicles a day', x.spk_bursts, CASE WHEN x.hours > 24.5 THEN ' a day' ELSE '' END,
                           x.spk_gmax, x.spk_daily)
  WHEN 'volume' THEN format('busiest 5 min {{:,.0f}} veh/h/lane vs limit {{:,.0f}}', x.q5_use, l.lim_volume)
  WHEN 'prof' THEN format('night level {{:.0f}} % of its busiest hours; phase mates ', 100 * x.tod_r ^ 2)
       || CASE WHEN x.tod_mate_r IS NOT NULL THEN format('{{:.0f}} %', 100 * x.tod_mate_r ^ 2) ELSE 'none' END
  END AS txt
FROM fnd f JOIN xf x USING (dev, det) JOIN finV v USING (dev, det) JOIN limits l USING (dev, det) JOIN win w USING (dev)
LEFT JOIN eps e USING (dev, det)""")
    q.add("cats", """
SELECT dev, det, string_agg(cat || ' (' || sev || ')', '; ' ORDER BY s DESC, ki0) AS health_categories,
       string_agg(cat || ': ' || txt, '; ' ORDER BY s DESC, ki0) AS parts, bool_or(lane) AS lane_cat
FROM (SELECT dev, det, cat, max(sc) AS s, min(ki) AS ki0, CASE WHEN bool_or(sev = 'bad') THEN 'bad' ELSE 'suspect' END AS sev,
             string_agg(txt, '; ' ORDER BY ki) AS txt, bool_or(key IN ('stuck', 'rapid', 'volume', 'prof', 'spiky')) AS lane
      FROM evid GROUP BY dev, det, cat)
GROUP BY dev, det""")
    # bad periods: where each check saw the problem (suspect / bad only)
    q.add("perb", f"""
WITH r AS (SELECT * FROM finV WHERE health_st IN ('suspect', 'bad')),
runs_hi AS (SELECT dev, det, min(k) AS a, max(k) + 1 AS z FROM (SELECT dev, det, k, k - row_number() OVER
            (PARTITION BY dev, det ORDER BY k) AS g FROM hib) GROUP BY dev, det, g),
fh AS (SELECT dev, det, dn // {NSH} AS h, count(*) FILTER (f) AS cf, count(*) AS ca FROM tfo GROUP BY dev, det, h),
runs_fast AS (SELECT dev, det, min(h) AS a, max(h) + 1 AS z FROM (
              SELECT f.dev, f.det, f.h, f.h - row_number() OVER (PARTITION BY f.dev, f.det ORDER BY f.h) AS g
              FROM fh f JOIN limits l USING (dev, det)
              WHERE f.cf >= 3 AND 1000.0 * f.cf / greatest(f.ca, 1) >= l.lim_rapid) GROUP BY dev, det, g),
qv AS (SELECT q.dev, q.det, q.b FROM q5b q JOIN xf x USING (dev, det) JOIN limits l USING (dev, det)
       WHERE x.fn = 'Advance' AND ((q.n::FLOAT / 300::FLOAT) * 3600::FLOAT)::DOUBLE / q.ln1 > l.lim_volume
       UNION ALL SELECT q.dev, q.det, q.b FROM q5gyb q JOIN xf x USING (dev, det) JOIN limits l USING (dev, det)
       WHERE x.fn <> 'Advance' AND q.q5_gy > l.lim_volume),
runs_vol AS (SELECT dev, det, min(b) AS a, max(b) + 1 AS z FROM (SELECT dev, det, b, b - row_number() OVER
             (PARTITION BY dev, det ORDER BY b) AS g FROM qv) GROUP BY dev, det, g),
p AS (
  SELECT r.dev, r.det, 0 AS o, w.s_ns + round(e.s * 1e9)::BIGINT AS a, w.s_ns + round(e.f * 1e9)::BIGINT AS z,
         'stuck on' AS what FROM r JOIN efl e USING (dev, det) JOIN win w USING (dev) WHERE r.f_stuck AND e.label <> 'queue'
  UNION ALL SELECT r.dev, r.det, 1, w.s_ns + u.a * w.bs * 1000000000, least(w.s_ns + u.z * w.bs * 1000000000, w.e_ns),
         'count zone held ON' FROM r JOIN runs_hi u USING (dev, det) JOIN win w USING (dev)
         WHERE r.f_count_on OR r.sev_rule = 'C'
  UNION ALL SELECT r.dev, r.det, 2, w.s_ns + r.drop_b0 * {NS5}, w.s_ns + r.drop_b1 * {NS5}, 'silent'
         FROM r JOIN win w USING (dev) WHERE r.f_dropout
  UNION ALL SELECT r.dev, r.det, 3, w.s_ns + r.drop_b0 * {NS5}, w.s_ns + r.drop_b1 * {NS5}, 'silent at night'
         FROM r JOIN win w USING (dev) WHERE r.f_night_drop AND r.silent_night
  UNION ALL SELECT r.dev, r.det, 4, epoch_ns(x.drop_t), w.e_ns, 'counts dropped'
         FROM r JOIN xf x USING (dev, det) JOIN win w USING (dev) WHERE r.f_level AND x.drop_t IS NOT NULL
  UNION ALL SELECT r.dev, r.det, 5, w.s_ns + u.a * {NSH}, least(w.s_ns + u.z * {NSH}, w.e_ns), 'too-fast actuations'
         FROM r JOIN runs_fast u USING (dev, det) JOIN win w USING (dev) WHERE r.f_rapid
  UNION ALL SELECT r.dev, r.det, 6, w.s_ns + u.a * {NS5}, least(w.s_ns + u.z * {NS5}, w.e_ns), 'too many for the traffic'
         FROM r JOIN runs_vol u USING (dev, det) JOIN win w USING (dev) WHERE r.f_volume
  UNION ALL SELECT r.dev, r.det, 7, x.tod_d0 + {NSH}, x.tod_d0 + 5 * {NSH}, 'busy at night'
         FROM r JOIN xf x USING (dev, det) WHERE r.f_prof AND x.tod_d0 IS NOT NULL
  UNION ALL SELECT r.dev, r.det, 8, w.s_ns + u.a * 600000000000, least(w.s_ns + u.z * 600000000000, w.e_ns),
         'spiky counts' FROM r JOIN spk_runs u USING (dev, det) JOIN win w USING (dev) WHERE r.f_spiky)
SELECT dev, det, '[' || string_agg('{{"start": "' || sa || '", "end": "' || sz || '", "what": "' || what
                                    || '", "recovered": null}}', ', ' ORDER BY sa, o, a) || ']' AS health_bad_periods
FROM (SELECT *, {_tstr('a')} AS sa, {_tstr('z')} AS sz FROM p) GROUP BY dev, det""")
    lab = ("'det ' || x.det || ': ' || CASE WHEN x.ph IS NOT NULL THEN 'P' || trunc(x.ph)::BIGINT || ' ' ELSE '' END "
           "|| replace(x.fn, 'Yellow_Red', 'Yellow-red')")
    nl = "greatest(round_even(x.lanes, 0)::BIGINT, 1)"
    border = " ".join(f"WHEN v.b_{k} THEN {_s('borderline ' + CATEGORY[k].lower())}" for k in KEYS)
    q.add("outx", f"""
SELECT v.dev, v.det, v.health_st AS health_status,
       CASE WHEN v.health_st <> 'not_enough_data' AND NOT isnan(v.score) THEN v.score END AS health_score,
       CASE WHEN v.health_st IN ('suspect', 'bad') THEN b.lb || ' - ' || coalesce(c.parts, v.health_st)
            WHEN v.health_st = 'watch' THEN b.lb || ' - watch' || CASE WHEN v.wt <> '' THEN ': ' || v.wt ELSE '' END
            WHEN v.health_st = 'not_enough_data' THEN b.lb || ' - too few actuations to judge'
            ELSE b.lb || ' - nothing unusual' END AS health_reason,
       coalesce(p.health_bad_periods, '') AS health_bad_periods, coalesce(c.health_categories, '') AS health_categories,
       CASE x.cfg WHEN 'C1' THEN format('Extension time on a count zone: {{:,}} of {{:,}} ONs logged again without an OFF',
                                        x.n_rep, trunc(x.n_on)::BIGINT)
                  WHEN 'C2' THEN 'Set to pulse but holds ON: ' || CASE WHEN x.long_max_s IS NOT NULL
                       THEN x.n_ge5 || ' ONs over 5 s, longest ' || {_dur('x.long_max_s')}
                       ELSE 'holds ON although set to pulse' END ELSE '' END AS health_config,
       v.wt AS health_watch
FROM (SELECT v.*, array_to_string(list_filter([
        CASE WHEN v.w_no_yardstick THEN 'no healthy phase mate to compare with' END,
        CASE {border} END,
        CASE WHEN v.w_dropout_short AND NOT (v.b_dropout OR v.b_level) THEN 'borderline counts fall' END],
        x -> x IS NOT NULL), '; ') AS wt FROM finV v) v
JOIN xf x USING (dev, det) LEFT JOIN cats c USING (dev, det) LEFT JOIN perb p USING (dev, det)
JOIN (SELECT x.dev, x.det, {lab} || CASE WHEN x.lanes IS NOT NULL AND coalesce(c.lane_cat, false) THEN ' (covers '
             || {nl} || ' lane' || CASE WHEN {nl} > 1 THEN 's' ELSE '' END || ')' ELSE '' END AS lb
      FROM xf x LEFT JOIN cats c USING (dev, det)) b USING (dev, det)""")
    # detectors without a known predicted function: the stage-1 checks as they are (pass-2 status)
    q.add("outf", """
SELECT s.dev, s.det, s.status AS health_status, CASE WHEN s.status <> 'not_enough_data' THEN s.hs END AS health_score,
       'det ' || s.det || ': ' || CASE s.status
         WHEN 'ok' THEN format('{:,.0f} actuations in {:.1f} h, nothing unusual', s.n_on, g.n_cov * 300 / 3600.0)
         WHEN 'not_enough_data' THEN format('only {:.0f} actuations in {:.1f} h - too few to judge', s.n_on,
                                            g.n_cov * 300 / 3600.0)
         ELSE array_to_string(list_filter([CASE WHEN s.s_stuck > 0 THEN 'stayed ON continuously for '
                                                  || round(s.ep_dur / 60)::BIGINT || ' min' END,
                                             CASE WHEN s.s_dropout > 0 THEN 'went silent' END,
                                             CASE WHEN s.s_night_drop > 0 THEN 'misses vehicles at night' END,
                                             CASE WHEN s.s_level > 0 THEN 'its share of the traffic dropped' END],
                                            x -> x IS NOT NULL), '; ')
         END AS health_reason, '' AS health_bad_periods, '' AS health_categories, '' AS health_config, '' AS health_watch
FROM stage1 s JOIN sig0 g USING (dev) ANTI JOIN xbase x USING (dev, det)""")
    # signal notes: many detectors with an unusual night; holds shared by several detectors (common failure)
    q.add("cgrp", """
SELECT dev, g, min(s) AS s0, arg_min(dur, s) AS dur0, count(DISTINCT round_even(s / 60, 0)) AS k
FROM (SELECT e.dev, e.s, e.dur, list_sort(list_distinct(list_append(coalesce(e.mates, []), e.det))) AS g
      FROM efl e WHERE e.common)
GROUP BY dev, g""")
    q.add("snote", f"""
SELECT w.dev, array_to_string(list_filter([
  CASE WHEN t.dev IS NOT NULL THEN t.n_odd || ' of ' || t.n_ok || ' detectors at this signal had an unusual night on '
       || {_hm('t.d0', '%a %d %b')} || ': a signal-wide event or an input change, not one bad detector' END,
  c.txt], x -> x IS NOT NULL), '; ') AS health_signal_note
FROM win w LEFT JOIN todnote t USING (dev)
LEFT JOIN (SELECT g.dev, string_agg(len(g.g) || ' detectors (det ' || array_to_string(g.g, ', ') || ') held ON together '
                  || CASE WHEN g.k = 1 THEN 'from ' || {_hm('w.s_ns + round(g.s0 * 1e9)::BIGINT')} || ' for '
                          || {_dur('g.dur0')} ELSE g.k || ' times from ' || {_hm('w.s_ns + round(g.s0 * 1e9)::BIGINT')} END
                  || ' (same start and end): one common failure, check the cabinet first', '; ' ORDER BY g.s0) AS txt
           FROM cgrp g JOIN win w USING (dev) GROUP BY g.dev) c USING (dev)""")


FINAL = """
SELECT w.DeviceId, o.det AS Detector, o.health_status, o.health_score, o.health_reason, o.health_bad_periods,
       o.health_categories, o.health_config, o.health_watch, n.health_signal_note
FROM (SELECT * FROM outx UNION ALL BY NAME SELECT * FROM outf) o JOIN win w USING (dev) JOIN snote n USING (dev)
ORDER BY CAST(w.DeviceId AS VARCHAR), o.det"""
OUT_COLS = ["DeviceId", "Detector", "health_status", "health_score", "health_reason", "health_bad_periods",
            "health_categories", "health_config", "health_watch", "health_signal_note"]


def build(events: str = "events", detectors: str = "detectors", windows: str | None = None, start=None, end=None,
          upto: str | None = None, probabilities: bool = True, distinct: bool = True) -> _Q:
    """the health steps; `windows` (optional): a table DeviceId, s_ns, e_ns (epoch ns) of the sample windows (predict
    passes its own); `upto`: stop after that step (testing)."""
    q = _Q()
    _inputs(q, events, detectors, windows, start, end, probabilities, distinct)
    _bins(q)
    _stage1_both(q)
    LJ, R = _json("health_limits.json"), _json("health_v4_refs.json")
    _measures(q, LJ, R)
    _assemble(q, LJ, R)
    if upto is not None:
        q.ctes = q.ctes[:[n for n, _, _ in q.ctes].index(upto) + 1]
    return q


def health_sql(events_table: str = "events", detectors_table: str = "detectors", start=None, end=None,
               probabilities: bool = True) -> str:
    """The detector health of every signal in `events_table` as ONE DuckDB query (text), to run on the connection that
    holds the tables -- nothing is copied out.

    events_table     DeviceId, Timestamp, EventId, Parameter (the hi-res controller log, local time)
    detectors_table  DeviceId, Detector, phase, function, lanes [, p_advance, p_presence, p_count, p_yellow_red,
                     p_other, p_mid, p_bike] -- each detector's predicted phase, function, lane span and (optional,
                     probabilities=True) class probabilities, e.g. the columns of predict()'s output
                     (phase_pred, function_pred, ...) renamed
    start / end      the sample window (end exclusive); default: each signal's first event to one second past its last
    Returns one row per detector with actuations in the window: DeviceId, Detector, health_status, health_score,
    health_reason, health_bad_periods, health_categories, health_config, health_watch, health_signal_note.
    Table names are inserted as given (quote them yourself if they need it)."""
    return build(events_table, detectors_table, None, start, end, probabilities=probabilities).text(FINAL)


@lru_cache(maxsize=8)
def _script(events_table: str, detectors_table: str, probs: bool, distinct: bool):
    q = build(events_table, detectors_table, f"{PREFIX}wpart", probabilities=probs, distinct=distinct)
    stmts, drops = q.script(FINAL)
    return q, stmts, drops


def _cols(con, table: str) -> set:
    return {str(r[0]).lower() for r in con.execute(f"DESCRIBE SELECT * FROM {table}").fetchall()}


def health(con, events_table: str = "events", detectors_table: str = "detectors", start=None, end=None,
           windows_table: str | None = None, max_events: int | None = 4_000_000,
           distinct: bool = True) -> pd.DataFrame:
    """Run the detector health on an existing DuckDB connection (the tables stay where they are) and return a DataFrame
    (columns as health_sql).  Signals are processed in groups of at most `max_events` events (memory); the steps run
    as temp tables (prefix hq_), each dropped as soon as it is no longer needed.  `windows_table` (optional): DeviceId,
    s_ns, e_ns (epoch ns) of each signal's window, in place of start / end (max_events None: no grouping);
    `distinct` False when the events hold no duplicate rows (skips the de-duplication)."""
    probs = set(PROB.values()) <= _cols(con, detectors_table)
    if windows_table is None:
        q = _Q()
        _inputs(q, events_table, detectors_table, None, start, end, probs)
        wq = _Q()
        wq.subs = q.subs
        wq.ctes = [c for c in q.ctes if c[0] in ("wraw", "wbin", "wsel")]
        con.execute(f"CREATE OR REPLACE TEMP TABLE {PREFIX}wall AS " + wq.text(
            "SELECT w.DeviceId, w.s_ns, w.e_ns, r.n FROM wsel w JOIN wraw r USING (DeviceId)"))
    elif max_events is None:
        con.execute(f"CREATE OR REPLACE TEMP TABLE {PREFIX}wall AS SELECT DeviceId, s_ns, e_ns, 0 AS n FROM {windows_table}")
    else:
        codes = ",".join(str(c) for c in ALLOWED)
        con.execute(f"""CREATE OR REPLACE TEMP TABLE {PREFIX}wall AS
            SELECT w.DeviceId, w.s_ns, w.e_ns, count(e.EventId) AS n FROM {windows_table} w LEFT JOIN {events_table} e
            ON e.DeviceId = w.DeviceId AND e.EventId IN ({codes}) AND epoch_ns(e.Timestamp) >= w.s_ns
            AND epoch_ns(e.Timestamp) < w.e_ns GROUP BY w.DeviceId, w.s_ns, w.e_ns""")
    sig = con.execute(f"SELECT DeviceId, n FROM {PREFIX}wall ORDER BY CAST(DeviceId AS VARCHAR)").fetchall()
    groups, cur, size = [], [], 0
    for d, n in sig:
        if cur and max_events is not None and size + n > max_events:
            groups.append(cur)
            cur, size = [], 0
        cur.append(d)
        size += n
    if cur:
        groups.append(cur)
    q, stmts, drops = _script(events_table, detectors_table, probs, distinct)
    out = []
    try:
        for g in groups:
            con.execute(f"CREATE OR REPLACE TEMP TABLE {PREFIX}wpart AS SELECT DeviceId, s_ns, e_ns FROM {PREFIX}wall "
                        f"WHERE DeviceId IN (SELECT unnest(?))", [list(g)])
            for i, st in enumerate(stmts):
                if i < len(stmts) - 1:
                    con.execute(st)
                else:
                    out.append(con.execute(st).df())
                for t in drops[i]:
                    con.execute(f"DROP TABLE IF EXISTS {t}")
    finally:
        for n, _, _ in q.ctes:
            con.execute(f"DROP TABLE IF EXISTS {PREFIX}{n}")
        con.execute(f"DROP TABLE IF EXISTS {PREFIX}wpart")
        con.execute(f"DROP TABLE IF EXISTS {PREFIX}wall")
    if not out:
        return pd.DataFrame(columns=OUT_COLS)
    return pd.concat(out, ignore_index=True)[OUT_COLS]
