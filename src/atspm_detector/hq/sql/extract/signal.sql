{# The extractor, signal part (design 4.1 health_signal_5min, 4.2 comms_gap / plan): per minute and per 5-min bin the
   allowed events, the signal's last allowed event (its time cut, design 3.2 rule 1), comms gaps (consecutive allowed
   events more than 120 s apart: today's per-minute test, the same doubles) and the plan changes (event 131).  The rows
   of xevm count this chunk's events only. #}
-- @step xevm mat
SELECT dev, dn // 60000000000 AS m, min(dn) AS a, max(dn) AS z, count(*) AS n FROM xev GROUP BY dev, m
{# health_signal_5min: events, first and last allowed event of each 5-min bin #}
-- @step xs5
SELECT dev, (m // 5)::INT AS b, sum(n)::BIGINT AS n, min(a) AS a, max(z) AS z FROM xevm GROUP BY dev, (m // 5)::INT
{# the signal's last allowed event (carry 9402) and the previous one's, l0: this chunk writes the time from l0 to lz
   (design 3.2 rule 1); NULL for a signal with nothing carried in (greatest skips it) #}
-- @step xlast mat
SELECT e.dev, e.lz, c.dn AS l0
FROM (SELECT dev, max(z) AS lz FROM xevm GROUP BY dev) e LEFT JOIN (SELECT dev, dn FROM xcar WHERE cid = 9402) c USING (dev)
{# comms gaps: g0 the last event before, g1 the first after (g0d: g0 in seconds, as today's gaps.g0); the last event of
   the previous chunk counts, so a gap across the cut is found by the event ending it.  The carry's markers (9403, first
   gap at or after the start of what is still open) are in the list too, g1 NULL: the look-ups below find them, only
   rows with g1 are written #}
-- @step xgap mat
SELECT dev, z AS g0, nt AS g1, z / 1e9 AS g0d
FROM (SELECT dev, z, lead(a) OVER (PARTITION BY dev ORDER BY a) AS nt
      FROM (SELECT dev, a, z FROM xevm UNION ALL SELECT dev, dn, dn FROM xcar WHERE cid = 9402))
WHERE nt / 1e9 - z / 1e9 > 120.0
UNION ALL SELECT dev, dn, NULL, dn / 1e9 FROM xcar WHERE cid = 9403
-- @step xplan
SELECT dev, dn, par FROM xev WHERE eid = 131
