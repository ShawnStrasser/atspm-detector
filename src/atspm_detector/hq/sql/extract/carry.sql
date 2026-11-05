{# What an update carries to the next one for each signal of its chunk (design 4.4), all computed from this chunk's rows
   and the carry it read, so carrying twice equals carrying once (T4.4).
   9201 / 9202 a detector's history as far as a later event can need it: an OFF / an ON; 9221 / 9222 its last ON, followed
   by an OFF / its last event (a chain still open).  Kept: everything from the last chain's start on, and when the last
   chain is a single ON the OFF and the ON before it (its too-fast ticks); of a run of ONs its first and last two, of a run
   of OFFs its last (a later event's look-backs reach the ON before it either way): at most 4 rows a detector.  no = ONs
   from the end, ns = chain starts from the end.
   9300-9302 each phase's colour piece.  9402 the signal's last allowed event.  9403 the first comms gap at or after the
   start of each chain or piece still open. #}
-- @step ytail
WITH a AS (SELECT dev, par AS det, dn, eid, lag(eid) OVER w AS p1, lead(eid) OVER w AS n1, lead(eid, 2) OVER w AS n2
           FROM xdi WINDOW w AS (PARTITION BY dev, par ORDER BY dn, eid DESC)),
b AS (SELECT *, sum((eid = 82)::INT) OVER r AS no, sum((eid = 82 AND p1 IS DISTINCT FROM 82)::INT) OVER r AS ns
      FROM a WINDOW r AS (PARTITION BY dev, det ORDER BY dn DESC, eid ASC ROWS UNBOUNDED PRECEDING))
SELECT dev, det, dn, CASE WHEN eid = 81 THEN 9201 WHEN no = 1 AND n1 IS NULL THEN 9222 WHEN no = 1 THEN 9221 ELSE 9202 END AS cid
FROM b
WHERE (ns = 0 OR (eid = 82 AND p1 IS DISTINCT FROM 82 AND ns = 1) OR no <= 1 OR (no = 2 AND eid = 82))
  AND CASE WHEN eid = 82 THEN p1 IS DISTINCT FROM 82 OR n1 IS DISTINCT FROM 82 OR n2 IS DISTINCT FROM 82
           ELSE n1 IS DISTINCT FROM 81 END
-- @step health_carry
-- @emit health_carry
SELECT make_timestamp_ns({{ ORG }} + r.dn) AS TimeStamp, w.DeviceId, r.cid::SMALLINT AS EventId, r.par::INTEGER AS Parameter, true AS IsValid
FROM (SELECT dev, dn, cid, det AS par FROM ytail
      UNION ALL SELECT dev, a, 9300 + st, ph FROM xseg WHERE z IS NULL
      UNION ALL SELECT dev, lz, 9402, 0 FROM xlast
      UNION ALL SELECT DISTINCT dev, mk, 9403, 0
                FROM (SELECT dev, mk FROM xoc UNION ALL SELECT dev, mk FROM xpc WHERE op AND st < 3) WHERE mk IS NOT NULL) r
JOIN win w USING (dev)
{# a detector's phase as this update used it, when it differs from the last one recorded (design 4.5) #}
-- @step health_detector_phase
-- @emit health_detector_phase
SELECT w.DeviceId, d.det::SMALLINT AS Detector, d.ph::INTEGER AS Phase, make_timestamp_ns({{ ORG }} + f.a) AS ValidFrom
FROM dcl d JOIN win w USING (dev) JOIN (SELECT dev, min(a) AS a FROM xevm GROUP BY dev) f USING (dev)
LEFT JOIN (SELECT DeviceId, Detector, last(Phase ORDER BY ValidFrom) AS Phase FROM @PHASES@ GROUP BY DeviceId, Detector) o
  ON o.DeviceId = w.DeviceId AND o.Detector = d.det
WHERE o.Detector IS NULL OR o.Phase IS DISTINCT FROM d.ph::INTEGER
