{# The extractor, detector part (design 3.2 "the event logic, restated", 4.1 health_detector_5min / health_tick_counts,
   4.2 hold): detector events with their neighbours, chains, starts with their colour, too-fast ticks.  Each decision
   uses today's double expression on the same ns offsets (chain longer than 60 s, 5 s, 300 s; the 0.1-s ticks), so it
   comes out as today's; times are kept in ns (integers). #}
{# this chunk's detector events (old 0) after the end of each detector's history as the carry keeps it (design 4.4): the
   last OFF of a run, a run of ONs as its first and last two, the last chain, and when that is a single ON the OFF and ON
   before it.  old: 1 = carried, 2 = carried and the detector's last ON, followed by a carried OFF, 3 = carried,
   the last ON and the detector's last event (a chain still open).  A fact is written only when its owner is new
   (design 3.2 rule 2): nothing below writes from a row with old 1; a chain or a too-fast pair of a carried ON is
   completed by a new event. #}
-- @step xdi
SELECT dev, par, dn, eid, 0 AS old FROM xev WHERE eid IN (81, 82)
UNION ALL
SELECT dev, par, dn, CASE WHEN cid = 9201 THEN 81 ELSE 82 END, CASE cid WHEN 9221 THEN 2 WHEN 9222 THEN 3 ELSE 1 END
FROM xcar WHERE cid IN (9201, 9202, 9221, 9222)
{# detector events in (time, ON before OFF) order with the events around them (today's de), kept: every ON, and the
   OFFs that do not close a chain (a second OFF, a leading OFF) -- an OFF after an ON is that ON's ndn #}
-- @step xde mat
SELECT * FROM (
  SELECT dev, par AS det, dn, eid, old, lag(eid) OVER w AS pe, lag(dn) OVER w AS pdn, lead(eid) OVER w AS ne,
         lead(dn) OVER w AS ndn, lag(eid, 2) OVER w AS ppe, lag(dn, 2) OVER w AS p2dn, lag(eid, 3) OVER w AS p3e,
         lead(eid, 2) OVER w AS nne, lead(dn, 2) OVER w AS n2dn
  FROM xdi WINDOW w AS (PARTITION BY dev, par ORDER BY dn, eid DESC))
WHERE eid = 82 OR pe IS DISTINCT FROM 82
{# starts of chains with ONs re-logged inside (an ON after a non-ON, an ON next: today's cst) #}
-- @step xmst mat
SELECT dev, det, dn, old FROM xde WHERE eid = 82 AND pe IS DISTINCT FROM 82 AND ne = 82
{# those chains: start s, end f = the event after their last ON (an OFF; NULL while open); cd: carried in, it started
   in an earlier chunk #}
-- @step xmc mat
SELECT m.dev, m.det, m.dn AS s, e.ndn AS f, m.old <> 0 AS cd
FROM (SELECT dev, det, dn, ndn FROM xde WHERE eid = 82 AND pe = 82 AND ne IS DISTINCT FROM 82 AND old IN (0, 3)) e
ASOF JOIN xmst m ON m.dev = e.dev AND m.det = e.det AND e.dn >= m.dn
{# a start right after a re-logged chain: that chain's start (today's multis), the start before it #}
-- @step xpms
SELECT d.dev, d.det, d.dn, m.dn AS pst
FROM (SELECT dev, det, dn FROM xde WHERE eid = 82 AND pe = 81 AND ppe = 82 AND p3e = 82) d
ASOF JOIN xmst m ON m.dev = d.dev AND m.det = d.det AND d.dn > m.dn
{# the rare ONs after two OFFs or before two OFFs (today's onx / st0r): the ON before (prv) and after (nxt), and the
   start before (pst).  The ON on the other side of two OFFs or more is such an ON too, so these rows alone hold prv and
   nxt; pst = prv, or when prv was re-logged inside a chain that chain's start (xmst). #}
-- @step xrare mat
SELECT r.dev, r.det, r.dn, r.prv, r.nxt, CASE WHEN r.prvp = 82 THEN m.dn ELSE r.prv END AS pst
FROM (SELECT dev, det, dn, lag(dn) OVER w AS prv, lag(pe) OVER w AS prvp, lead(dn) OVER w AS nxt
      FROM xde WHERE eid = 82 AND ((pe = 81 AND ppe = 81) OR (ne = 81 AND nne = 81))
      WINDOW w AS (PARTITION BY dev, det ORDER BY dn)) r
ASOF LEFT JOIN xmst m ON m.dev = r.dev AND m.det = r.det AND r.dn > m.dn
{# chains (note 154q, occ_strict): a single ON to the event after it, or a re-logged chain (multi); f NULL = still open
   at the chunk end #}
{% macro chains() %}SELECT dev, det, dn AS s, CASE WHEN ne = 81 THEN ndn END AS f, false AS multi, old <> 0 AS cd FROM xde
   WHERE eid = 82 AND pe IS DISTINCT FROM 82 AND ne IS DISTINCT FROM 82 AND old IN (0, 3)
   UNION ALL SELECT dev, det, s, f, true, cd FROM xmc{% endmacro %}
{# closed chains: gap = longer than 60 s with a comms gap starting inside (today's ons) #}
-- @step xcc mat
WITH c AS (SELECT dev, det, s, f, multi, cd, f / 1e9 - s / 1e9 AS dur FROM ({{ chains() }}) WHERE f IS NOT NULL)
SELECT dev, det, s, f, multi, cd, dur, false AS gap FROM c WHERE dur <= 60
UNION ALL
SELECT c.dev, c.det, c.s, c.f, c.multi, c.cd, c.dur, coalesce(g.g0 < c.f, false)
FROM (SELECT * FROM c WHERE dur > 60) c ASOF LEFT JOIN xgap g ON c.dev = g.dev AND g.g0 >= c.s
{# the chain still open at the chunk end (no OFF yet; carry 9222): its start, and mk = the first comms gap at or after its
   start (carry 9403) #}
-- @step xoc mat
SELECT c.dev, c.det, c.s, g.g0 AS mk
FROM (SELECT * FROM ({{ chains() }}) WHERE f IS NULL) c ASOF LEFT JOIN xgap g ON c.dev = g.dev AND g.g0 >= c.s
{# starts per 5-min bin with their rule-F flags and the colour piece of the detector's phase each falls in (today's
   starts / sst, all phases: the judge keeps phases with >= 3 greens).  f1: within 1 s of the start before (today's
   case analysis: the ON before the OFF before it, or a re-logged chain's start, or after two OFFs the start before);
   chat: < 0.3 s after an OFF.  Starts in a valid green / yellow (gy) or in the first 2 s of a valid red (r2); a start
   in an open green / yellow or red's first 2 s counts provisionally under its piece's label (pph, ptag, pk) #}
-- @step xsa mat
SELECT dev, det, b, pph, ptag, pk, count(*) AS nst, count(*) FILTER (f1 AND NOT chat) AS nfst,
       count(*) FILTER (chat) AS nchat, count(*) FILTER (ok AND st < 2) AS nst_gy,
       count(*) FILTER (ok AND st = 2 AND f1 AND NOT chat) AS nfst_r2, count(*) FILTER (ok AND st = 2 AND chat) AS nchat_r2
FROM (SELECT s.dev, s.det, s.b, s.f1, s.chat, p.st, coalesce(p.op OR p.valid, false) AS ok,
             CASE WHEN p.op AND p.st < 3 THEN p.ph END AS pph, CASE WHEN p.op AND p.st < 3 THEN p.tag END AS ptag,
             CASE WHEN p.op AND p.st < 3 THEN CASE WHEN p.st < 2 THEN 0 ELSE 2 END END AS pk
      FROM (SELECT d.dev, d.det, (d.dn // {{ NS5 }})::INT AS b, d.dn / 1e9 AS xd, c.ph,
                   coalesce(round_even((d.dn - CASE WHEN d.pe = 81 AND d.ppe = 82 AND d.p3e IS DISTINCT FROM 82 THEN d.p2dn
                                                    WHEN d.pe = 81 AND d.ppe = 82 THEN m.pst
                                                    WHEN d.pe = 81 AND d.ppe = 81 THEN r.pst END) / 1e8, 0) < 10, false) AS f1,
                   coalesce(d.pe = 81 AND round_even((d.dn - d.pdn) / 1e8, 0) < 3, false) AS chat
            FROM (SELECT * FROM xde WHERE eid = 82 AND pe IS DISTINCT FROM 82 AND old = 0) d
            LEFT JOIN xrare r ON r.dev = d.dev AND r.det = d.det AND r.dn = d.dn
            LEFT JOIN xpms m ON m.dev = d.dev AND m.det = d.det AND m.dn = d.dn
            LEFT JOIN dcl c ON c.dev = d.dev AND c.det = d.det) s
      ASOF LEFT JOIN xpc p ON s.dev = p.dev AND s.ph = p.ph AND s.xd >= p.ad)
GROUP BY ALL
{# too fast: every ON's 0.1-s ticks to the ON before (ta, rows only under the largest limit) and to the ON after (tb,
   capped; NULL: none yet -- pending, finalize adds it as "no next ON"); the ONs around it as today's tfo finds them.
   The detector's last carried ON (old 2, 3) is here for its next ON. #}
-- @step xtf
SELECT dev, det, dn, ta, CASE WHEN nx IS NOT NULL THEN least(round_even((nx - dn) / 1e8, 0), {{ TFC }})::INT END AS tb
FROM (SELECT d.dev, d.det, d.dn,
             round_even((d.dn - CASE WHEN d.pe = 82 THEN d.pdn WHEN d.pe = 81 AND d.ppe = 82 THEN d.p2dn
                                     WHEN d.pe = 81 AND d.ppe = 81 THEN r.prv END) / 1e8, 0)::INT AS ta,
             CASE WHEN d.ne = 82 THEN d.ndn WHEN d.ne = 81 AND d.nne = 82 THEN d.n2dn
                  WHEN d.ne = 81 AND d.nne = 81 THEN r.nxt END AS nx
      FROM xde d LEFT JOIN xrare r ON r.dev = d.dev AND r.det = d.det AND r.dn = d.dn WHERE d.eid = 82 AND d.old <> 1)
WHERE ta < {{ TFC }}
