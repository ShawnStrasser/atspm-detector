"""Sudden-stop test for ped buttons: did this button go silent for longer than its own traffic can explain?

Works from 15-minute press counts of one window, whatever their source (the controller log or a counts table).  Inferred
only from the window: no list of buttons, no settings.  For each input with presses in the window:

  run        its longest run of consecutive logged 15-min bins with zero presses (bins the signal did not log are skipped)
  expected   presses that the input's own level would have brought in the run: for each bin of the run, the mean presses
             per bin at that hour of day (+- 1 h, weights 1-2-1) outside the run.  An hour never seen outside the run adds
             nothing, so a run with little traffic around it expects little.
  clump      presses come in clumps: squared deviations of its hourly counts outside the run from the level at that hour,
             over the presses that level expects; pulled toward the fleet's typical CLUMP0 by CLUMP_K hours of weight, so a
             short baseline cannot claim no clumps; at least 1.  expected / clump = the independent clumps expected.
  chance     (1 + PHI * clumps) ** (-1 / PHI): the chance of no clump at all when the level itself varies from day to day by
             a gamma factor of squared variation PHI (fixed; plain Poisson is exp(-expected), PHI -> 0, clump 1).
  chance_cap the same with each baseline hour counted at most CAP x the button's median busy hour: one burst sets no level.
  simple     the run is >= SIMPLE_RUN_H logged hours and its level over the day (the level at each hour of day seen outside the
             run, summed) is >= SIMPLE_LEVEL presses a day.
  rating     flagged when the statistical test (chance <= P_SUSPECT) OR the simple one says so: suspect; bad: chance_cap <=
             P_BAD.  A run must be >= MIN_RUN_H logged hours and the input needs >= MIN_PRESSES presses in >= MIN_ACTIVE_H
             separate hours outside the run: low-volume buttons cannot flag.
  next_b     the bin of the first press after the run; empty while the button is silent at the window's end.  A period that
             ended just has an end time: there is no 'recovered' logic.

stop_scores(con, counts, logged) -> one row per input: dev, ch, n_out, bins_out, active_h, run_h, run_start, run_end, next_b,
starts_window, expected, expected_cap, clump, level, chance, chance_cap, chance_poisson, simple, rating.  counts: table / view
(dev, ch, b, n), b the bin start; logged: (dev, b).  Both already limited to the window.

walk_change(con, counts, logged, since) -> one row per (input, day) where walks jumped and presses per walk fell against the
button's own days: the existing report check's idea (walks vs presses) on its own baseline, with no comparison to the region.  A
note, never a fault: ped recall looks the same.  counts: (dev, ch, b, n, sv), sv the walks; hand it the report's 21-day read.

last_button_events(con, events, start, end, carry) -> DeviceId, Input, last_ev ('on' / 'off'), last_ts.  open_ons(...) ->
DeviceId, Input, t_on, Phase: the ONs with no OFF, to carry into the next chunk (ped_health(carry=)).  An open ON alone proves nothing
(lost data): stuck ON needs the phase walking every cycle from that ON (hq/sql/ped/judge.sql, f_stuck).

Tuned on 10 user-labelled stops (2026-11-02), sliding 3-day windows of 15-min counts, Sep 18 - Oct 31 2026; final limits from the
user's review of 2026-11-02."""
from __future__ import annotations

PHI = 0.25
P_SUSPECT = 1e-3   # 1 in 1,000
P_BAD = 1e-3       # 1 in 1,000, and it must hold with bursts capped (chance_cap)
MIN_RUN_H = 6.0
MIN_PRESSES = 20
MIN_ACTIVE_H = 6   # presses in at least this many separate hours outside the run: one burst sets no level
BIN_MIN = 15
CLUMP0 = 6.0       # typical clump size (fleet median), counted as CLUMP_K hours of evidence
CLUMP_K = 12
CAP = 3.0          # chance_cap: an hour counts at most 3 x the button's median busy hour
SIMPLE_RUN_H = 24.0
SIMPLE_LEVEL = 30.0
WC_RATIO = 3.0     # walk_change: walks per logged hour >= 3 x the button's lower-quartile day
WC_DROP = 0.5      # and presses per walk <= half its upper-quartile day
WC_MIN_WALKS_H = 2.0
WC_MIN_DAYS = 3
WC_MIN_HOURS = 4.0
# unique presses (hq/sql/ped/judge.sql, step uniq; only the walks-without-presses check reads them): a press counts when it opens a new UNIQ_S-second slot of its run; a run is
# presses each within UNIQ_S of the one before.  One person pressing 8 times in 8 s counts 1; a button repeating every 10 s for an hour counts 240, never 1.  The window's presses
# only, so any cutting of the log into updates gives the same counts.  Sudden stop and every other check stay on raw presses (user review 2026-11-03)
UNIQ_S = 15
# walks without presses (option walks_no_press): a zero-press run of >= NP_H logged hours in which the walks are >= NP_WALKS_H an hour, >= NP_JUMP x what those hours
# of day brought in the NP_LOOK_H before the run, and imply >= NP_IMPLIED presses at the button's own earlier walks per unique press; ped calls <= NP_CALLS of the walks.
# A look-back with < NP_MIN_U0 unique presses says little about walks per press: then the walks must be >= NP_LOW_WALKS_H an hour and >= NP_LOW_JUMP x the usual
NP_H = 1.0
NP_WALKS_H = 8.0
NP_JUMP = 3.0
NP_IMPLIED = 15.0
NP_CALLS = 0.1
NP_LOOK_H = 72
NP_MIN_U0 = 10
NP_LOW_WALKS_H = 12.0
NP_LOW_JUMP = 10.0
# tuned on 3-day windows: a run needs >= NP_MIN_LOOK_H logged hours before it in the window, so every hour of day has a usual level (with less, a morning reads against the night
# just before it); its walks in the latest hour >= NP_GREEN_SHARE of its phase's greens, as the stuck on check: a held button or recall brings a walk at nearly every green, a few
# people on another button do not; another button of the signal firing within NP_OTHER_Q quarter-hours of it is a signal-wide recall or plan
NP_MIN_LOOK_H = 24
NP_GREEN_SHARE = 0.8
NP_OTHER_Q = 2
# rated bad (option walks_bad) once the run holds >= NP_BAD_H logged hours of walks at >= NP_WALKS_H an hour and >= NP_GREEN_SHARE of a phase's greens: the report then counts it at once (150 bad
# minutes in a day alert), and one short of it stays suspect, which never adds up: the flag drops off the later windows.  Set from the five buttons the user marked real (review of 2026-11-03);
# the shortest ran 2.75 and 3.5 such hours
NP_BAD_H = 2.5
# sustained pressing (option sustained_press, on; hq/sql/ped/judge.sql, step f_press): a day with >= SP_PRESSES presses, >= SP_TIMES x its usual (the median of the window's other logged
# days, itself >= SP_USUAL) and >= SP_BINS 15-minute periods of >= SP_BIN presses and >= SP_PER_WALK presses per walk (no walk counts as one).  Raw presses.  Rated suspect, however
# strong.  From the review of 2026-11-03: presses per walk alone finds people (13 of 13 flagged days seen in events were not faults); 43 days of counts found the two real cases
SP_PRESSES = 1000
SP_TIMES = 5.0
SP_USUAL = 20.0
SP_BINS = 5
SP_BIN = 30
SP_PER_WALK = 10


def stop_sql(counts: str, logged: str, phi: float = PHI, cap: float = CAP) -> str:
    return f"""
WITH inp AS (SELECT dev, ch, sum(n) AS tot FROM {counts} GROUP BY ALL HAVING sum(n) > 0),
grid AS (SELECT l.dev, i.ch, l.b, coalesce(c.n, 0) AS n,
                row_number() OVER (PARTITION BY l.dev, i.ch ORDER BY l.b) AS k
         FROM (SELECT DISTINCT dev, b FROM {logged}) l JOIN inp i USING (dev)
         LEFT JOIN {counts} c ON c.dev = l.dev AND c.ch = i.ch AND c.b = l.b),
runs AS (SELECT dev, ch, min(b) AS r0, max(b) AS r1, count(*) AS bins
         FROM (SELECT dev, ch, b, k - row_number() OVER (PARTITION BY dev, ch ORDER BY b) AS g FROM grid WHERE n = 0)
         GROUP BY dev, ch, g),
first_ AS (SELECT dev, ch, min(b) AS firstb FROM grid GROUP BY ALL),
best AS (SELECT * FROM runs QUALIFY row_number() OVER (PARTITION BY dev, ch ORDER BY bins DESC, r0 DESC) = 1),
nxt AS (SELECT g.dev, g.ch, min(g.b) AS next_b FROM grid g JOIN best b USING (dev, ch) WHERE g.b > b.r1 AND g.n > 0 GROUP BY ALL),
out0 AS (SELECT g.dev, g.ch, g.b, g.n, hour(g.b) AS hh FROM grid g JOIN best b USING (dev, ch) WHERE g.b < b.r0 OR g.b > b.r1),
hours AS (SELECT dev, ch, date_trunc('hour', b) AS h, hour(b) AS hh, sum(n) AS n, count(*) AS m FROM out0 GROUP BY ALL),
capv AS (SELECT dev, ch, {cap} * quantile_cont(n, 0.5) FILTER (n > 0) AS cap FROM hours GROUP BY ALL),
hod AS (SELECT h.dev, h.ch, h.hh, sum(h.n) AS s, sum(least(h.n, coalesce(c.cap, h.n))) AS sc, sum(h.m) AS m
        FROM hours h LEFT JOIN capv c USING (dev, ch) GROUP BY ALL),
-- level per bin at each hour of day, smoothed 1-2-1 over the neighbouring hours (circular); only over hours seen
lvl AS (SELECT a.dev, a.ch, x.hh, sum(a.s * (2 - x.d)) / sum(a.m * (2 - x.d)) AS lam, sum(a.sc * (2 - x.d)) / sum(a.m * (2 - x.d)) AS lamc
        FROM hod a, LATERAL (SELECT (a.hh + o + 24) % 24 AS hh, abs(o) AS d FROM (VALUES (-1), (0), (1)) t(o)) x
        WHERE x.hh IN (SELECT hh FROM hod h WHERE h.dev = a.dev AND h.ch = a.ch)
        GROUP BY ALL),
disp AS (SELECT h.dev, h.ch, sum(pow(h.n - h.m * l.lam, 2)) AS ss, sum(h.m * l.lam) AS mu,
                count(*) AS nh, count(DISTINCT h.hh) AS nhh
         FROM hours h JOIN lvl l USING (dev, ch, hh) WHERE l.lam > 0 GROUP BY ALL),
runb AS (SELECT g.dev, g.ch, sum(coalesce(l.lam, 0)) AS expected, sum(coalesce(l.lamc, 0)) AS expected_cap
         FROM grid g JOIN best b USING (dev, ch)
         LEFT JOIN lvl l ON l.dev = g.dev AND l.ch = g.ch AND l.hh = hour(g.b)
         WHERE g.b BETWEEN b.r0 AND b.r1 GROUP BY ALL),
dayl AS (SELECT dev, ch, 4 * sum(lam) AS level FROM lvl GROUP BY ALL),
outn AS (SELECT dev, ch, sum(n) AS n_out, count(*) AS bins_out,
                count(DISTINCT date_trunc('hour', b)) FILTER (n > 0) AS active_h FROM out0 GROUP BY ALL),
s AS (SELECT b.dev, b.ch, coalesce(o.n_out, 0) AS n_out, coalesce(o.bins_out, 0) AS bins_out, coalesce(o.active_h, 0) AS active_h,
             b.bins * {BIN_MIN} / 60.0 AS run_h, b.r0 AS run_start, b.r1 + INTERVAL {BIN_MIN} MINUTE AS run_end, x.next_b,
             b.r0 = z.firstb AS starts_window, coalesce(y.level, 0) AS level,
             coalesce(r.expected, 0) AS expected, coalesce(r.expected_cap, 0) AS expected_cap,
             greatest(1.0, (coalesce(d.ss / nullif(d.mu, 0), 0) * greatest(coalesce(d.nh - d.nhh, 0), 0) + {CLUMP0} * {CLUMP_K}) / (greatest(coalesce(d.nh - d.nhh, 0), 0) + {CLUMP_K})) AS clump
      FROM best b JOIN first_ z USING (dev, ch) LEFT JOIN nxt x USING (dev, ch) LEFT JOIN runb r USING (dev, ch)
      LEFT JOIN outn o USING (dev, ch) LEFT JOIN disp d USING (dev, ch) LEFT JOIN dayl y USING (dev, ch))
SELECT *, pow(1 + {phi} * expected / clump, -1.0 / {phi}) AS chance, pow(1 + {phi} * expected_cap / clump, -1.0 / {phi}) AS chance_cap,
       exp(-expected) AS chance_poisson, level, coalesce(run_h >= {SIMPLE_RUN_H} AND level >= {SIMPLE_LEVEL}, false) AS simple FROM s"""


def rate(chance, chance_cap, run_h, n_out, active_h, simple=False):
    """'bad' / 'suspect' / None for one input: the statistical test OR the simple one."""
    if run_h < MIN_RUN_H or n_out < MIN_PRESSES or active_h < MIN_ACTIVE_H or (chance > P_SUSPECT and not simple):
        return None
    return "bad" if chance_cap <= P_BAD else "suspect"


def stop_scores(con, counts: str, logged: str):
    df = con.execute(stop_sql(counts, logged)).df()
    df["rating"] = [rate(c, k, r, n, h, q) for c, k, r, n, h, q in
                    zip(df.chance, df.chance_cap, df.run_h, df.n_out, df.active_h, df.simple)]
    return df


def walk_change_sql(counts: str, logged: str, since=None) -> str:
    keep = f"AND x.d >= DATE '{since}'" if since else ""
    return f"""
WITH h AS (SELECT dev, b::DATE AS d, count(*) * {BIN_MIN} / 60.0 AS hrs FROM (SELECT DISTINCT dev, b FROM {logged}) GROUP BY ALL),
p AS (SELECT c.dev, c.ch, c.b::DATE AS d, sum(c.n) AS a, sum(c.sv) AS s FROM {counts} c GROUP BY ALL),
x AS (SELECT p.*, h.hrs, p.s / h.hrs AS wr, CASE WHEN p.s > 0 THEN p.a / p.s END AS aps FROM p JOIN h USING (dev, d) WHERE h.hrs >= {WC_MIN_HOURS}),
b AS (SELECT dev, ch, count(*) AS n, quantile_cont(wr, 0.25) AS wb, quantile_cont(aps, 0.75) AS ab FROM x GROUP BY ALL)
SELECT x.dev, x.ch, x.d, x.a, x.s, x.hrs, b.wb, b.ab, x.wr / greatest(b.wb, 0.2) AS ratio FROM x JOIN b USING (dev, ch)
WHERE b.n >= {WC_MIN_DAYS} AND x.wr >= {WC_MIN_WALKS_H} AND x.wr / greatest(b.wb, 0.2) >= {WC_RATIO}
  AND x.aps <= {WC_DROP} * greatest(b.ab, 0.01) {keep} ORDER BY x.d, x.dev, x.ch"""


def walk_change(con, counts: str, logged: str, since=None):
    return con.execute(walk_change_sql(counts, logged, since)).df()


def last_button_events(con, events: str, start=None, end=None, carry: str | None = None):
    """the last 89 / 90 of each input in the window, a carried ON (DeviceId, Input, t_on) counting as an earlier 90."""
    w = " AND ".join(x for x in (f"Timestamp >= TIMESTAMP '{start}'" if start else "", f"Timestamp < TIMESTAMP '{end}'" if end else "") if x)
    src = f"SELECT DeviceId, Parameter::INT AS ch, Timestamp AS ts, EventId::INT AS ev FROM {events} WHERE EventId IN (89, 90)" + (f" AND {w}" if w else "")
    if carry:
        src += f" UNION ALL SELECT DeviceId, Input::INT, t_on, 90 FROM {carry}"
    return con.execute(f"""SELECT DeviceId, ch AS "Input", CASE arg_max(ev, (ts, ev)) WHEN 90 THEN 'on' ELSE 'off' END AS last_ev, max(ts) AS last_ts
                           FROM ({src}) GROUP BY ALL""").df()


def open_ons(con, events: str, start=None, end=None, carry: str | None = None):
    """the ONs still open at the end of this chunk, with the phase their ped call named (45 within 0.25 s; empty when none): hand them
    to ped_health(carry=...) of the next chunk.  The phase has to travel with the ON: the next chunk may hold no press to tell it."""
    d = last_button_events(con, events, start, end, carry)
    d = d[d.last_ev == "on"].rename(columns={"last_ts": "t_on"})[["DeviceId", "Input", "t_on"]].reset_index(drop=True)
    con.register("pq_open_", d)
    old = f"(SELECT c.Phase FROM {carry} c WHERE c.DeviceId = o.DeviceId AND c.Input = o.Input AND c.t_on = o.t_on)" if carry else "NULL"
    out = con.execute(f"""SELECT o.DeviceId, o."Input", o.t_on, coalesce((SELECT arg_min(e.Parameter::INT, e.Timestamp) FROM {events} e
                          WHERE e.DeviceId = o.DeviceId AND e.EventId = 45 AND e.Timestamp >= o.t_on
                          AND e.Timestamp <= o.t_on + INTERVAL 250 MILLISECOND), {old}) AS Phase FROM pq_open_ o""").df()
    con.unregister("pq_open_")
    return out
