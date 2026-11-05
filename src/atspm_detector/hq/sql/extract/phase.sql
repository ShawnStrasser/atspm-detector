{# The extractor, phase part (design 4.1 health_phase_5min, health_phase_15min): colour pieces, greens per 5-min bin and
   per clock hour.  The double expressions (piece starts and ends in seconds, the 2-s red split, the 5-s green start) are
   today's (seg, pcs, grn), so every colour state and validity comes out as today. #}
{# colour changes per phase: a piece starts at each colour event (1 green, 8 yellow, 9 / 10 red) whose state differs
   from the one before, the piece carried in (9300-9302) coming first; z = the next change, NULL while the piece is still
   open at the chunk end #}
-- @step xseg mat
SELECT dev, ph, a, lead(a) OVER (PARTITION BY dev, ph ORDER BY a, eid) AS z, st
FROM (SELECT dev, ph, a, eid, st, lag(st) OVER (PARTITION BY dev, ph ORDER BY a, eid) AS pst
      FROM (SELECT dev, par AS ph, dn AS a, eid, CASE eid WHEN 1 THEN 0 WHEN 8 THEN 1 ELSE 2 END AS st
            FROM xev WHERE eid IN (1, 8, 9, 10)
            UNION ALL SELECT dev, par, dn, 0, cid - 9300 FROM xcar WHERE cid BETWEEN 9300 AND 9302))
WHERE pst IS DISTINCT FROM st
{# pieces: green 0, yellow 1, the red's first 2 s 2 and the rest of it 3 (today's pcs); tag = the colour piece's start
   in ns (its label while open), z its end, ad / zd the piece's start / end in seconds (zd NULL while open);
   valid = no comms gap starts in it (NULL while open: finalize decides, from mk = the first gap at or after it) #}
-- @step xpc mat
WITH s AS (SELECT dev, ph, a, z, st, a / 1e9 AS ad, z / 1e9 AS zd FROM xseg),
p AS (SELECT dev, ph, st, a AS tag, z, ad, zd, z IS NULL AS op FROM s WHERE st < 2
      UNION ALL
      SELECT dev, ph, 2, a, z, ad, CASE WHEN z IS NOT NULL THEN least(ad + 2.0, zd) END, z IS NULL FROM s WHERE st = 2
      UNION ALL
      SELECT dev, ph, 3, a, z, CASE WHEN z IS NULL THEN ad + 2.0 ELSE least(ad + 2.0, zd) END, zd, z IS NULL
      FROM s WHERE st = 2)
SELECT p.*, g.g0 AS mk, CASE WHEN NOT p.op THEN NOT coalesce(g.g0d < p.zd, false) END AS valid
FROM (SELECT * FROM p WHERE op OR zd > ad) p ASOF LEFT JOIN xgap g ON p.dev = g.dev AND g.g0d >= p.ad
{# greens after their first 5 s (today's grn): closed valid ones longer than 5 s, and the open ones; k numbers them per
   phase, a5n = the next one's start #}
-- @step xg5 mat
SELECT dev, ph, tag, tag + 5000000000 AS a5, z, op, (row_number() OVER w)::INT AS k, lead(tag + 5000000000) OVER w AS a5n
FROM xpc WHERE st = 0 AND (op OR (valid AND zd > ad + 5.0)) WINDOW w AS (PARTITION BY dev, ph ORDER BY tag)
{# green events (1) per 5-min bin and per clock quarter-hour (a store's window may start on any quarter-hour, so rows
   per hour could hold greens from before it) #}
-- @step xgr
SELECT dev, par AS ph, (dn // {{ NS5 }})::INT AS b, count(*) AS ngreen FROM xev WHERE eid = 1 GROUP BY ALL
-- @step xgq
SELECT e.dev, e.par AS ph, ({{ ORG }} + e.dn) // {{ NSQ }} AS q, count(*) AS ngreen
FROM xev e JOIN win w USING (dev) WHERE e.eid = 1 GROUP BY ALL
