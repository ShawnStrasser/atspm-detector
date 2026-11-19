"""Flash: no data from a signal in flash is used for anything (user ruling 2026-11-18).

A signal is in flash
  * from a unit flash status change (173) with Parameter other than 2 (2 = not in flash) to its next 173 with 2, and
  * from a coordination pattern change (131) to pattern 255 (flash) to its next 131 to another pattern;
  a flash also ends at the signal's next begin green (1): a signal timing greens is not in flash, and the closing event
  is sometimes missing from the log (seen at two signals whose night flash never logged its end, 2026-11-15 to 17).
  With neither, the flash lasts to the end of the log.  When a source's first event in the log closes a flash
  (173 = 2, or 131 to another pattern) and the signal logged no begin green (1) before it, the signal was in flash from
  the start of the log to that event (a window that opens mid-flash).  A status logged again while nothing changed (a
  173 = 3 at 23:59:59.9 during a night flash) only repeats the state.

Every event of the signal inside such a period is dropped, so the period reads as time the signal logged nothing: no
check judges it and no model learns from it.

    intervals_sql(E)                SELECT DeviceId, s, e (TIMESTAMP, e exclusive; open ends -infinity / infinity) of `E`'s flash periods
    without_flash(con, E, name)     a TEMP VIEW `name` of `E` without them, or `E` itself when the log holds none
    drop_flash(con, table)          DELETE them from a table in place (DeviceId, Timestamp, EventId, Parameter)
"""
from __future__ import annotations

from .common import EV_BEGIN_GREEN, EV_COORD_PATTERN, EV_FLASH

FLASH_PATTERN = 255
NOT_FLASH = 2


def intervals_sql(E: str) -> str:
    """the flash periods of every signal of events table / view `E`: DeviceId, s, e (overlapping periods merged; an open
    end is -infinity / infinity)"""
    return f"""
WITH m AS (SELECT DeviceId, Timestamp AS t, EventId AS src,
                  CASE WHEN EventId = {EV_FLASH} THEN Parameter <> {NOT_FLASH} ELSE Parameter = {FLASH_PATTERN} END AS on_
           FROM {E} WHERE EventId IN ({EV_FLASH}, {EV_COORD_PATTERN}) AND Timestamp IS NOT NULL),
gr AS (SELECT DeviceId, Timestamp AS tg FROM {E} WHERE EventId = {EV_BEGIN_GREEN} AND Timestamp IS NOT NULL
       AND DeviceId IN (SELECT DeviceId FROM m)),
g AS (SELECT DeviceId, min(tg) AS g0 FROM gr GROUP BY ALL),
o0 AS (SELECT m.*, min(t) FILTER (NOT on_) OVER (PARTITION BY DeviceId, src ORDER BY t
                  ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING) AS nxt_off,
             row_number() OVER (PARTITION BY DeviceId, src ORDER BY t) AS k
      FROM m),
o AS (SELECT o0.*, gr.tg AS nxt_green FROM o0 ASOF LEFT JOIN gr ON o0.DeviceId = gr.DeviceId AND gr.tg > o0.t),
p AS (SELECT DeviceId, t AS s, least(coalesce(nxt_off, 'infinity'::TIMESTAMP), coalesce(nxt_green, 'infinity'::TIMESTAMP)) AS e
      FROM o WHERE on_
      UNION ALL
      SELECT o.DeviceId, '-infinity'::TIMESTAMP, o.t FROM o LEFT JOIN g USING (DeviceId)
      WHERE o.k = 1 AND NOT o.on_ AND (g.g0 IS NULL OR g.g0 >= o.t)),
b AS (SELECT *, max(e) OVER (PARTITION BY DeviceId ORDER BY s, e ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS pe
      FROM p),
c AS (SELECT *, sum(CASE WHEN pe IS NULL OR s > pe THEN 1 ELSE 0 END) OVER (PARTITION BY DeviceId ORDER BY s, e) AS isl
      FROM b)
SELECT DeviceId, min(s) AS s, max(e) AS e FROM c GROUP BY DeviceId, isl"""


def _cond(alias: str, iv: str) -> str:
    return (f"EXISTS (SELECT 1 FROM {iv} f WHERE f.DeviceId = {alias}.DeviceId "
            f"AND {alias}.Timestamp >= f.s AND {alias}.Timestamp < f.e)")


def _intervals(con, E: str, iv: str) -> int:
    con.execute(f"CREATE OR REPLACE TEMP TABLE {iv} AS {intervals_sql(E)}")
    return con.execute(f"SELECT count(*) FROM {iv}").fetchone()[0]


def without_flash(con, E: str, name: str = "fl_events") -> str:
    """the name of `E` without the events of signals in flash: a TEMP VIEW `name` (over the TEMP TABLE of the
    periods `<name>_iv`, kept for the view), or `E` unchanged when its log holds no flash period"""
    iv = f"{name}_iv"
    if not _intervals(con, E, iv):
        con.execute(f"DROP TABLE IF EXISTS {iv}")
        return E
    con.execute(f"CREATE OR REPLACE TEMP VIEW {name} AS SELECT x.* FROM {E} x ANTI JOIN {iv} f "
                f"ON f.DeviceId = x.DeviceId AND x.Timestamp >= f.s AND x.Timestamp < f.e")
    return name


def drop_flash(con, table: str) -> int:
    """delete the events of signals in flash from `table` in place -> rows deleted"""
    iv = "fl_drop_iv"
    n = 0
    if _intervals(con, table, iv):
        n = con.execute(f"SELECT count(*) FROM {table} x WHERE {_cond('x', iv)}").fetchone()[0]
        con.execute(f"DELETE FROM {table} x WHERE {_cond('x', iv)}")
    con.execute(f"DROP TABLE IF EXISTS {iv}")
    return int(n)
