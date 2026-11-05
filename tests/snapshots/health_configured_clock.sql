WITH hq_wraw AS (
SELECT DeviceId, epoch_ns(min(Timestamp)) AS w0, epoch_ns(max(Timestamp)) + 1000000000 AS w1, count(*) AS n
FROM events WHERE EventId IN (1,7,8,9,10,11,43,44,81,82,131,150,173) AND (EventId NOT IN (81, 82) OR Parameter BETWEEN 1 AND 64) AND DeviceId IS NOT NULL AND Timestamp IS NOT NULL GROUP BY DeviceId
),
hq_wbin AS (
SELECT DeviceId, w0, w1, CASE WHEN w1 > w0 THEN w0 ELSE w0 END AS b0,
       CASE WHEN w1 > w0 THEN w1 ELSE w1 END AS b1 FROM hq_wraw
),
hq_wsel AS (
SELECT DeviceId, CASE WHEN w0 - b0 > 1800000000000 THEN w0 ELSE b0 END AS s_ns,
       CASE WHEN b1 - w1 > 1800000000000 THEN w1 ELSE b1 END AS e_ns FROM hq_wbin
),
hq_win AS MATERIALIZED (
SELECT dev, DeviceId, s_ns, e_ns, T, T / 3600.0 AS hours, ceil(T / 300.0)::INT AS nb5, wg,
       CASE WHEN wg = 'm30' THEN 300 ELSE 900 END AS bs,
       ceil(T / CASE WHEN wg = 'm30' THEN 300.0 ELSE 900.0 END)::INT AS nbb,
       make_timestamp(s_ns // 1000) AS lo, make_timestamp(e_ns // 1000 + 1) AS hi,
       s_ns % 86400000000000 AS sod
FROM (SELECT (dense_rank() OVER (ORDER BY CAST(DeviceId AS VARCHAR)))::INT AS dev, DeviceId,
             s_ns::BIGINT AS s_ns, e_ns::BIGINT AS e_ns, (e_ns - s_ns) / 1e9 AS T,
             CASE WHEN (e_ns - s_ns) / 3.6e12 < sqrt(1.5) THEN 'm30'
                  WHEN (e_ns - s_ns) / 3.6e12 < sqrt(72.0) THEN 'h3' ELSE 'h24' END AS wg
      FROM (SELECT DeviceId, s_ns::BIGINT - s_ns::BIGINT % 900000000000 AS s_ns, e_ns FROM hq_wsel))
),
hq_dcl AS MATERIALIZED (
SELECT w.dev, d.Detector::INT AS det, (CASE WHEN isnan((d.phase)::DOUBLE) THEN NULL ELSE (d.phase)::DOUBLE END) AS ph, CAST(d.function AS VARCHAR) AS fn,
       (CASE WHEN isnan((d.lanes)::DOUBLE) THEN NULL ELSE (d.lanes)::DOUBLE END) AS lanes, (CASE WHEN isnan((d.p_advance)::DOUBLE) THEN NULL ELSE (d.p_advance)::DOUBLE END) AS p_advance, (CASE WHEN isnan((d.p_presence)::DOUBLE) THEN NULL ELSE (d.p_presence)::DOUBLE END) AS p_presence, (CASE WHEN isnan((d.p_count)::DOUBLE) THEN NULL ELSE (d.p_count)::DOUBLE END) AS p_count, (CASE WHEN isnan((d.p_yellow_red)::DOUBLE) THEN NULL ELSE (d.p_yellow_red)::DOUBLE END) AS p_yellow_red, (CASE WHEN isnan((d.p_other)::DOUBLE) THEN NULL ELSE (d.p_other)::DOUBLE END) AS p_other, (CASE WHEN isnan((d.p_mid)::DOUBLE) THEN NULL ELSE (d.p_mid)::DOUBLE END) AS p_mid, (CASE WHEN isnan((d.p_bike)::DOUBLE) THEN NULL ELSE (d.p_bike)::DOUBLE END) AS p_bike, NULLIF(trim(CAST(d.configured_function AS VARCHAR)), '') AS cfn
FROM detectors d JOIN hq_win w ON CAST(d.DeviceId AS VARCHAR) = CAST(w.DeviceId AS VARCHAR)
),
hq_xovd AS MATERIALIZED (
SELECT dev, det, ph FROM hq_dcl WHERE ph IS NOT NULL AND fn IN ('Count')
),
hq_xev AS MATERIALIZED (
SELECT DISTINCT w.dev, epoch_ns(e.Timestamp) - 0 AS dn, e.EventId::INT AS eid, e.Parameter::INT AS par
FROM events e JOIN hq_win w ON e.DeviceId = w.DeviceId
WHERE e.EventId IN (1,7,8,9,10,11,43,44,81,82,131,150,173) AND (e.EventId NOT IN (81, 82) OR e.Parameter BETWEEN 1 AND 64)
  AND e.Timestamp >= w.lo AND e.Timestamp <= w.hi
  AND epoch_ns(e.Timestamp) >= w.s_ns AND epoch_ns(e.Timestamp) < w.e_ns
),
hq_xcar AS MATERIALIZED (
SELECT w.dev, epoch_ns(c.TimeStamp) - 0 AS dn, c.EventId::INT AS cid, c.Parameter::INT AS par
FROM (SELECT DeviceId, NULL::TIMESTAMP_NS AS TimeStamp, NULL::INTEGER AS EventId, NULL::INTEGER AS Parameter FROM hq_win WHERE false) c JOIN hq_win w ON c.DeviceId = w.DeviceId
),
hq_xevm AS MATERIALIZED (
SELECT dev, dn // 60000000000 AS m, min(dn) AS a, max(dn) AS z, count(*) AS n FROM hq_xev GROUP BY dev, m
),
hq_xs5 AS (
SELECT dev, (m // 5)::INT AS b, sum(n)::BIGINT AS n, min(a) AS a, max(z) AS z FROM hq_xevm GROUP BY dev, (m // 5)::INT
),
hq_xlast AS MATERIALIZED (
SELECT e.dev, e.lz, c.dn AS l0
FROM (SELECT dev, max(z) AS lz FROM hq_xevm GROUP BY dev) e LEFT JOIN (SELECT dev, dn FROM hq_xcar WHERE cid = 9402) c USING (dev)
),
hq_xgap AS MATERIALIZED (
SELECT dev, z AS g0, nt AS g1, z / 1e9 AS g0d
FROM (SELECT dev, z, lead(a) OVER (PARTITION BY dev ORDER BY a) AS nt
      FROM (SELECT dev, a, z FROM hq_xevm UNION ALL SELECT dev, dn, dn FROM hq_xcar WHERE cid = 9402))
WHERE nt / 1e9 - z / 1e9 > 120.0
UNION ALL SELECT dev, dn, NULL, dn / 1e9 FROM hq_xcar WHERE cid = 9403
),
hq_xplan AS (
SELECT dev, dn, par FROM hq_xev WHERE eid = 131
),
hq_xseg AS MATERIALIZED (
SELECT dev, ph, a, lead(a) OVER (PARTITION BY dev, ph ORDER BY a, eid) AS z, st
FROM (SELECT dev, ph, a, eid, st, lag(st) OVER (PARTITION BY dev, ph ORDER BY a, eid) AS pst
      FROM (SELECT dev, par AS ph, dn AS a, eid, CASE eid WHEN 1 THEN 0 WHEN 8 THEN 1 ELSE 2 END AS st
            FROM hq_xev WHERE eid IN (1, 8, 9, 10)
            UNION ALL SELECT dev, par, dn, 0, cid - 9300 FROM hq_xcar WHERE cid BETWEEN 9300 AND 9302))
WHERE pst IS DISTINCT FROM st
),
hq_xpc AS MATERIALIZED (
WITH s AS (SELECT dev, ph, a, z, st, a / 1e9 AS ad, z / 1e9 AS zd FROM hq_xseg),
p AS (SELECT dev, ph, st, a AS tag, z, ad, zd, z IS NULL AS op FROM s WHERE st < 2
      UNION ALL
      SELECT dev, ph, 2, a, z, ad, CASE WHEN z IS NOT NULL THEN least(ad + 2.0, zd) END, z IS NULL FROM s WHERE st = 2
      UNION ALL
      SELECT dev, ph, 3, a, z, CASE WHEN z IS NULL THEN ad + 2.0 ELSE least(ad + 2.0, zd) END, zd, z IS NULL
      FROM s WHERE st = 2)
SELECT p.*, g.g0 AS mk, CASE WHEN NOT p.op THEN NOT coalesce(g.g0d < p.zd, false) END AS valid
FROM (SELECT * FROM p WHERE op OR zd > ad) p ASOF LEFT JOIN hq_xgap g ON p.dev = g.dev AND g.g0d >= p.ad
),
hq_xg5 AS MATERIALIZED (
SELECT dev, ph, tag, tag + 5000000000 AS a5, z, op, (row_number() OVER w)::INT AS k, lead(tag + 5000000000) OVER w AS a5n
FROM hq_xpc WHERE st = 0 AND (op OR (valid AND zd > ad + 5.0)) WINDOW w AS (PARTITION BY dev, ph ORDER BY tag)
),
hq_xgr AS (
SELECT dev, par AS ph, (dn // 300000000000)::INT AS b, count(*) AS ngreen FROM hq_xev WHERE eid = 1 GROUP BY ALL
),
hq_xgq AS (
SELECT e.dev, e.par AS ph, (0 + e.dn) // 900000000000 AS q, count(*) AS ngreen
FROM hq_xev e JOIN hq_win w USING (dev) WHERE e.eid = 1 GROUP BY ALL
),
hq_xdi AS (
SELECT dev, par, dn, eid, 0 AS old FROM hq_xev WHERE eid IN (81, 82)
UNION ALL
SELECT dev, par, dn, CASE WHEN cid = 9201 THEN 81 ELSE 82 END, CASE cid WHEN 9221 THEN 2 WHEN 9222 THEN 3 ELSE 1 END
FROM hq_xcar WHERE cid IN (9201, 9202, 9221, 9222)
),
hq_xde AS MATERIALIZED (
SELECT * FROM (
  SELECT dev, par AS det, dn, eid, old, lag(eid) OVER w AS pe, lag(dn) OVER w AS pdn, lead(eid) OVER w AS ne,
         lead(dn) OVER w AS ndn, lag(eid, 2) OVER w AS ppe, lag(dn, 2) OVER w AS p2dn, lag(eid, 3) OVER w AS p3e,
         lead(eid, 2) OVER w AS nne, lead(dn, 2) OVER w AS n2dn
  FROM hq_xdi WINDOW w AS (PARTITION BY dev, par ORDER BY dn, eid DESC))
WHERE eid = 82 OR pe IS DISTINCT FROM 82
),
hq_xmst AS MATERIALIZED (
SELECT dev, det, dn, old FROM hq_xde WHERE eid = 82 AND pe IS DISTINCT FROM 82 AND ne = 82
),
hq_xmc AS MATERIALIZED (
SELECT m.dev, m.det, m.dn AS s, e.ndn AS f, m.old <> 0 AS cd
FROM (SELECT dev, det, dn, ndn FROM hq_xde WHERE eid = 82 AND pe = 82 AND ne IS DISTINCT FROM 82 AND old IN (0, 3)) e
ASOF JOIN hq_xmst m ON m.dev = e.dev AND m.det = e.det AND e.dn >= m.dn
),
hq_xpms AS (
SELECT d.dev, d.det, d.dn, m.dn AS pst
FROM (SELECT dev, det, dn FROM hq_xde WHERE eid = 82 AND pe = 81 AND ppe = 82 AND p3e = 82) d
ASOF JOIN hq_xmst m ON m.dev = d.dev AND m.det = d.det AND d.dn > m.dn
),
hq_xrare AS MATERIALIZED (
SELECT r.dev, r.det, r.dn, r.prv, r.nxt, CASE WHEN r.prvp = 82 THEN m.dn ELSE r.prv END AS pst
FROM (SELECT dev, det, dn, lag(dn) OVER w AS prv, lag(pe) OVER w AS prvp, lead(dn) OVER w AS nxt
      FROM hq_xde WHERE eid = 82 AND ((pe = 81 AND ppe = 81) OR (ne = 81 AND nne = 81))
      WINDOW w AS (PARTITION BY dev, det ORDER BY dn)) r
ASOF LEFT JOIN hq_xmst m ON m.dev = r.dev AND m.det = r.det AND r.dn > m.dn
),
hq_xcc AS MATERIALIZED (
WITH c AS (SELECT dev, det, s, f, multi, cd, f / 1e9 - s / 1e9 AS dur FROM (SELECT dev, det, dn AS s, CASE WHEN ne = 81 THEN ndn END AS f, false AS multi, old <> 0 AS cd FROM hq_xde
   WHERE eid = 82 AND pe IS DISTINCT FROM 82 AND ne IS DISTINCT FROM 82 AND old IN (0, 3)
   UNION ALL SELECT dev, det, s, f, true, cd FROM hq_xmc) WHERE f IS NOT NULL)
SELECT dev, det, s, f, multi, cd, dur, false AS gap FROM c WHERE dur <= 60
UNION ALL
SELECT c.dev, c.det, c.s, c.f, c.multi, c.cd, c.dur, coalesce(g.g0 < c.f, false)
FROM (SELECT * FROM c WHERE dur > 60) c ASOF LEFT JOIN hq_xgap g ON c.dev = g.dev AND g.g0 >= c.s
),
hq_xoc AS MATERIALIZED (
SELECT c.dev, c.det, c.s, g.g0 AS mk
FROM (SELECT * FROM (SELECT dev, det, dn AS s, CASE WHEN ne = 81 THEN ndn END AS f, false AS multi, old <> 0 AS cd FROM hq_xde
   WHERE eid = 82 AND pe IS DISTINCT FROM 82 AND ne IS DISTINCT FROM 82 AND old IN (0, 3)
   UNION ALL SELECT dev, det, s, f, true, cd FROM hq_xmc) WHERE f IS NULL) c ASOF LEFT JOIN hq_xgap g ON c.dev = g.dev AND g.g0 >= c.s
),
hq_xsa AS MATERIALIZED (
SELECT dev, det, b, pph, ptag, pk, count(*) AS nst, count(*) FILTER (f1 AND NOT chat) AS nfst,
       count(*) FILTER (chat) AS nchat, count(*) FILTER (ok AND st < 2) AS nst_gy,
       count(*) FILTER (ok AND st = 2 AND f1 AND NOT chat) AS nfst_r2, count(*) FILTER (ok AND st = 2 AND chat) AS nchat_r2
FROM (SELECT s.dev, s.det, s.b, s.f1, s.chat, p.st, coalesce(p.op OR p.valid, false) AS ok,
             CASE WHEN p.op AND p.st < 3 THEN p.ph END AS pph, CASE WHEN p.op AND p.st < 3 THEN p.tag END AS ptag,
             CASE WHEN p.op AND p.st < 3 THEN CASE WHEN p.st < 2 THEN 0 ELSE 2 END END AS pk
      FROM (SELECT d.dev, d.det, (d.dn // 300000000000)::INT AS b, d.dn / 1e9 AS xd, c.ph,
                   coalesce(round_even((d.dn - CASE WHEN d.pe = 81 AND d.ppe = 82 AND d.p3e IS DISTINCT FROM 82 THEN d.p2dn
                                                    WHEN d.pe = 81 AND d.ppe = 82 THEN m.pst
                                                    WHEN d.pe = 81 AND d.ppe = 81 THEN r.pst END) / 1e8, 0) < 10, false) AS f1,
                   coalesce(d.pe = 81 AND round_even((d.dn - d.pdn) / 1e8, 0) < 3, false) AS chat
            FROM (SELECT * FROM hq_xde WHERE eid = 82 AND pe IS DISTINCT FROM 82 AND old = 0) d
            LEFT JOIN hq_xrare r ON r.dev = d.dev AND r.det = d.det AND r.dn = d.dn
            LEFT JOIN hq_xpms m ON m.dev = d.dev AND m.det = d.det AND m.dn = d.dn
            LEFT JOIN hq_dcl c ON c.dev = d.dev AND c.det = d.det) s
      ASOF LEFT JOIN hq_xpc p ON s.dev = p.dev AND s.ph = p.ph AND s.xd >= p.ad)
GROUP BY ALL
),
hq_xtf AS (
SELECT dev, det, dn, ta, CASE WHEN nx IS NOT NULL THEN least(round_even((nx - dn) / 1e8, 0), 8)::INT END AS tb
FROM (SELECT d.dev, d.det, d.dn,
             round_even((d.dn - CASE WHEN d.pe = 82 THEN d.pdn WHEN d.pe = 81 AND d.ppe = 82 THEN d.p2dn
                                     WHEN d.pe = 81 AND d.ppe = 81 THEN r.prv END) / 1e8, 0)::INT AS ta,
             CASE WHEN d.ne = 82 THEN d.ndn WHEN d.ne = 81 AND d.nne = 82 THEN d.n2dn
                  WHEN d.ne = 81 AND d.nne = 81 THEN r.nxt END AS nx
      FROM hq_xde d LEFT JOIN hq_xrare r ON r.dev = d.dev AND r.det = d.det AND r.dn = d.dn WHERE d.eid = 82 AND d.old <> 1)
WHERE ta < 8
),
hq_xov AS MATERIALIZED (
WITH c AS (SELECT c.dev, c.det, greatest(c.s, l.l0) AS s, c.f, c.ctag, d.ph, l.lz
           FROM (SELECT dev, det, s, f, NULL::BIGINT AS ctag FROM hq_xcc WHERE NOT gap
                 UNION ALL SELECT o.dev, o.det, o.s, l.lz, o.s FROM hq_xoc o JOIN hq_xlast l USING (dev)) c
           JOIN hq_xovd d USING (dev, det) JOIN hq_xlast l USING (dev)),
lo AS (SELECT c.*, p.k AS k0, p.a5n, p.a5, p.z, p.op, p.ph AS gph, p.tag
       FROM c ASOF LEFT JOIN hq_xg5 p ON c.dev = p.dev AND c.ph = p.ph AND c.s >= p.a5),
one AS (SELECT dev, det, ctag, gph, op, tag, greatest(s, a5) AS s, least(f, coalesce(z, lz)) AS f,
               lo.s = lo.f AND lo.s > a5 AND (z IS NULL OR lo.s < z) AS inst
        FROM lo WHERE k0 IS NOT NULL AND (a5n IS NULL OR f <= a5n)),
many AS (SELECT r.dev, r.det, r.ctag, p.ph AS gph, p.op, p.tag, greatest(r.s, p.a5) AS s, least(r.f, coalesce(p.z, r.lz)) AS f, false AS inst
         FROM (SELECT l.dev, l.det, l.s, l.f, l.ctag, l.ph, l.lz, unnest(range(coalesce(l.k0, 1), p.k + 1))::INT AS k
               FROM (SELECT * FROM lo WHERE NOT (k0 IS NOT NULL AND (a5n IS NULL OR f <= a5n))) l
               ASOF JOIN hq_xg5 p ON l.dev = p.dev AND l.ph = p.ph AND l.f > p.a5) r
         JOIN hq_xg5 p ON p.dev = r.dev AND p.ph = r.ph AND p.k = r.k)
SELECT dev, det, ctag, CASE WHEN op THEN gph END AS pph, CASE WHEN op THEN tag END AS ptag, s, f
FROM (SELECT * FROM one UNION ALL SELECT * FROM many) WHERE f > s OR inst
),
hq_xd5 AS (
SELECT dev, det, b, sum(nst) AS nst, sum(nfst) AS nfst, sum(nchat) AS nchat, sum(nst_gy) AS nst_gy,
       sum(nfst_r2) AS nfst_r2, sum(nchat_r2) AS nchat_r2, sum(nrel) AS nrel, sum(nxoff) AS nxoff, sum(on_ns) AS on_ns,
       sum(unk_ns) AS unk_ns, sum(nclose) AS nclose, sum(nclean) AS nclean, max(max_on) AS max_on,
       sum(relog_ns) AS relog_ns, sum(n5s) AS n5s, sum(n60s) AS n60s, max(lmax) AS lmax, sum(ong5_ns) AS ong5_ns
FROM (SELECT dev, det, b, nst, nfst, nchat, CASE WHEN ptag IS NULL THEN nst_gy END AS nst_gy,
       CASE WHEN ptag IS NULL THEN nfst_r2 END AS nfst_r2, CASE WHEN ptag IS NULL THEN nchat_r2 END AS nchat_r2
FROM hq_xsa
UNION ALL BY NAME
SELECT dev, det, (dn // 300000000000)::INT AS b, count(*) FILTER (eid = 82) AS nrel, count(*) FILTER (eid = 81) AS nxoff
FROM hq_xde WHERE (pe = 82 OR eid = 81) AND old = 0 GROUP BY ALL
UNION ALL BY NAME
SELECT dev, det, b, sum(t) FILTER (NOT gap) AS on_ns, sum(t) FILTER (gap) AS unk_ns, count(*) FILTER (b = b1) AS nclose,
       count(*) FILTER (b = b0 AND NOT multi AND NOT gap) AS nclean,
       max(CASE WHEN gap THEN 0.0 ELSE dur END) FILTER (b = b0) AS max_on,
       sum(f - s) FILTER (b = b0 AND multi AND NOT gap) AS relog_ns,
       count(*) FILTER (b = b0 AND NOT gap AND dur >= 5) AS n5s, count(*) FILTER (b = b0 AND NOT gap AND dur >= 60) AS n60s,
       max(dur) FILTER (b = b0 AND NOT gap) AS lmax
FROM (WITH o AS (SELECT o.*, (o.s // 300000000000)::INT AS b0, (o.f // 300000000000)::INT AS b1
           FROM (SELECT dev, det, s, f, gap, multi, dur FROM hq_xcc WHERE NOT cd) o)
SELECT dev, det, gap, multi, dur, s, f, b0, b1, b0 AS b, least(f, (b0 + 1) * 300000000000) - s AS t FROM o WHERE b1 = b0
UNION ALL
SELECT dev, det, gap, multi, dur, s, f, b0, b1, b, least(f, (b + 1) * 300000000000) - greatest(s, b * 300000000000)
FROM (SELECT *, unnest(range(b0, b1 + 1))::INT AS b FROM o WHERE b1 > b0))
GROUP BY ALL
UNION ALL BY NAME
SELECT dev, det, b, sum(t) FILTER (NOT gap) AS on_ns, sum(t) FILTER (gap) AS unk_ns, count(*) FILTER (b = b1) AS nclose
FROM (WITH o AS (SELECT o.*, (o.s // 300000000000)::INT AS b0, (o.f // 300000000000)::INT AS b1
           FROM (SELECT c.dev, c.det, l.l0 AS s, c.f, c.gap FROM hq_xcc c JOIN hq_xlast l USING (dev) WHERE c.cd) o)
SELECT dev, det, gap, b1, b0 AS b, least(f, (b0 + 1) * 300000000000) - s AS t FROM o WHERE b1 = b0
UNION ALL
SELECT dev, det, gap, b1, b, least(f, (b + 1) * 300000000000) - greatest(s, b * 300000000000)
FROM (SELECT *, unnest(range(b0, b1 + 1))::INT AS b FROM o WHERE b1 > b0))
GROUP BY ALL
UNION ALL BY NAME
SELECT dev, det, (s // 300000000000)::INT AS b, (NOT multi AND NOT gap)::INT AS nclean,
       CASE WHEN gap THEN 0.0 ELSE dur END AS max_on, CASE WHEN multi AND NOT gap THEN f - s END AS relog_ns,
       (NOT gap AND dur >= 5)::INT AS n5s, (NOT gap AND dur >= 60)::INT AS n60s, CASE WHEN NOT gap THEN dur END AS lmax
FROM hq_xcc WHERE cd
UNION ALL BY NAME
SELECT dev, det, b, t AS ong5_ns
FROM (WITH o AS (SELECT o.*, (o.s // 300000000000)::INT AS b0, (o.f // 300000000000)::INT AS b1
           FROM (SELECT dev, det, s, f FROM hq_xov WHERE ctag IS NULL AND ptag IS NULL) o)
SELECT dev, det, b0 AS b, least(f, (b0 + 1) * 300000000000) - s AS t FROM o WHERE b1 = b0
UNION ALL
SELECT dev, det, b, least(f, (b + 1) * 300000000000) - greatest(s, b * 300000000000)
FROM (SELECT *, unnest(range(b0, b1 + 1))::INT AS b FROM o WHERE b1 > b0)))
GROUP BY dev, det, b
),
hq_xd5l AS (
SELECT dev, det, b, ctag, t AS on_ns
FROM (WITH o AS (SELECT o.*, (o.s // 300000000000)::INT AS b0, (o.f // 300000000000)::INT AS b1
           FROM (SELECT o.dev, o.det, greatest(o.s, l.l0) AS s, l.lz AS f, o.s AS ctag FROM hq_xoc o JOIN hq_xlast l USING (dev)) o)
SELECT dev, det, ctag, b0 AS b, least(f, (b0 + 1) * 300000000000) - s AS t FROM o WHERE b1 = b0
UNION ALL
SELECT dev, det, ctag, b, least(f, (b + 1) * 300000000000) - greatest(s, b * 300000000000)
FROM (SELECT *, unnest(range(b0, b1 + 1))::INT AS b FROM o WHERE b1 > b0))
UNION ALL BY NAME
SELECT dev, det, b, pph, ptag, pk, nst_gy, nfst_r2, nchat_r2 FROM hq_xsa WHERE ptag IS NOT NULL
UNION ALL BY NAME
SELECT dev, det, b, ctag, pph, ptag, 0 AS pk, t AS ong5_ns
FROM (WITH o AS (SELECT o.*, (o.s // 300000000000)::INT AS b0, (o.f // 300000000000)::INT AS b1
           FROM (SELECT * FROM hq_xov WHERE ctag IS NOT NULL OR ptag IS NOT NULL) o)
SELECT dev, det, ctag, pph, ptag, b0 AS b, least(f, (b0 + 1) * 300000000000) - s AS t FROM o WHERE b1 = b0
UNION ALL
SELECT dev, det, ctag, pph, ptag, b, least(f, (b + 1) * 300000000000) - greatest(s, b * 300000000000)
FROM (SELECT *, unnest(range(b0, b1 + 1))::INT AS b FROM o WHERE b1 > b0))
),
hq_xp5 AS (
SELECT dev, ph, b, sum(ngreen) AS ngreen, sum(gy_ns) AS gy_ns, sum(g5_ns) AS g5_ns
FROM (SELECT dev, ph, b, ngreen FROM hq_xgr
UNION ALL BY NAME
SELECT dev, ph, b, CASE WHEN kind = 0 THEN t END AS gy_ns, CASE WHEN kind = 1 THEN t END AS g5_ns
FROM (WITH o AS (SELECT o.*, (o.s // 300000000000)::INT AS b0, (o.f // 300000000000)::INT AS b1
           FROM (SELECT p.dev, p.ph, greatest(p.tag, l.l0) AS s, p.z AS f, 0 AS kind FROM hq_xpc p JOIN hq_xlast l USING (dev) WHERE p.st < 2 AND NOT p.op AND p.valid
   UNION ALL SELECT p.dev, p.ph, greatest(p.a5, l.l0), p.z, 1 FROM hq_xg5 p JOIN hq_xlast l USING (dev) WHERE NOT p.op) o)
SELECT dev, ph, kind, b0 AS b, least(f, (b0 + 1) * 300000000000) - s AS t FROM o WHERE b1 = b0
UNION ALL
SELECT dev, ph, kind, b, least(f, (b + 1) * 300000000000) - greatest(s, b * 300000000000)
FROM (SELECT *, unnest(range(b0, b1 + 1))::INT AS b FROM o WHERE b1 > b0)))
GROUP BY dev, ph, b
),
hq_xp5l AS (
SELECT dev, ph, b, ptag, CASE WHEN kind = 0 THEN t END AS gy_ns, CASE WHEN kind = 1 THEN t END AS g5_ns
FROM (WITH o AS (SELECT o.*, (o.s // 300000000000)::INT AS b0, (o.f // 300000000000)::INT AS b1
           FROM (SELECT p.dev, p.ph, greatest(p.tag, l.l0) AS s, l.lz AS f, 0 AS kind, p.tag AS ptag FROM hq_xpc p JOIN hq_xlast l USING (dev) WHERE p.st < 2 AND p.op
   UNION ALL SELECT p.dev, p.ph, greatest(p.a5, l.l0), l.lz, 1, p.tag FROM hq_xg5 p JOIN hq_xlast l USING (dev) WHERE p.op AND l.lz > p.a5) o)
SELECT dev, ph, kind, ptag, b0 AS b, least(f, (b0 + 1) * 300000000000) - s AS t FROM o WHERE b1 = b0
UNION ALL
SELECT dev, ph, kind, ptag, b, least(f, (b + 1) * 300000000000) - greatest(s, b * 300000000000)
FROM (SELECT *, unnest(range(b0, b1 + 1))::INT AS b FROM o WHERE b1 > b0))
),
hq_xtko AS (
SELECT dev, det, s // 900000000000 AS q, least((f - s) // 1000, 1000001) AS a, count(*) AS n FROM hq_xcc
WHERE NOT multi AND NOT gap GROUP BY ALL
),
hq_xtkg AS (
SELECT dev, det, dn // 900000000000 AS q, ta, tb, count(*) AS n FROM hq_xtf WHERE tb IS NOT NULL GROUP BY ALL
),
hq_xhold AS (
SELECT dev, det, s, f, multi FROM hq_xcc WHERE dur >= 300 AND NOT gap
),
hq_xbad AS (
SELECT dev, 'bad_chain' AS Kind, det AS ch, s, f, NULL::INT AS val FROM hq_xcc WHERE cd AND gap
UNION ALL
SELECT p.dev, 'bad_piece', p.ph, p.tag, p.z, CASE WHEN p.st < 2 THEN 0 ELSE 2 END
FROM hq_xpc p JOIN hq_xlast l USING (dev) WHERE NOT p.op AND p.st < 3 AND NOT p.valid AND p.tag <= l.l0
),
hq_health_signal_5min AS (
SELECT w.DeviceId, make_timestamp_ns(0 + r.b * 300000000000) AS TimeStamp, r.n::INTEGER AS Events, make_timestamp_ns(0 + r.a) AS First_Event,
       make_timestamp_ns(0 + r.z) AS Last_Event
FROM hq_xs5 r JOIN hq_win w USING (dev)
),
hq_health_detector_5min AS MATERIALIZED (
SELECT w.DeviceId, r.det::SMALLINT AS Detector, make_timestamp_ns(0 + r.b * 300000000000) AS TimeStamp, make_timestamp_ns(0 + r.ctag) AS Chain_Tag,
       r.pph AS Piece_Phase, make_timestamp_ns(0 + r.ptag) AS Piece_Tag, r.pk::SMALLINT AS Piece_Kind,
       r.nst::INTEGER AS Starts, r.nfst::INTEGER AS Fast_Starts, r.nchat::INTEGER AS Chatter_Starts,
       r.nst_gy::INTEGER AS Starts_GY, r.nfst_r2::INTEGER AS Fast_Starts_R2, r.nchat_r2::INTEGER AS Chatter_Starts_R2,
       r.nrel::INTEGER AS Relogged_Ons, r.nxoff::INTEGER AS Stray_Offs, r.nclose::INTEGER AS Closing_Offs,
       r.on_ns::BIGINT AS On_Time_ns, r.unk_ns::BIGINT AS Unknown_Time_ns, r.ong5_ns::BIGINT AS On_Green5_ns,
       r.nclean::INTEGER AS Clean_Ons, r.max_on AS Max_On_s, r.relog_ns::BIGINT AS Relog_Time_ns, r.n5s::INTEGER AS Ons_5s,
       r.n60s::INTEGER AS Ons_60s, r.lmax AS Long_Max_s
FROM (SELECT * FROM hq_xd5 UNION ALL BY NAME SELECT * FROM hq_xd5l) r JOIN hq_win w USING (dev)
),
hq_health_phase_5min AS MATERIALIZED (
SELECT w.DeviceId, r.ph AS Phase, make_timestamp_ns(0 + r.b * 300000000000) AS TimeStamp, make_timestamp_ns(0 + r.ptag) AS Piece_Tag,
       r.ngreen::INTEGER AS Greens, r.gy_ns::BIGINT AS Green_Yellow_ns, r.g5_ns::BIGINT AS Green5_ns
FROM (SELECT * FROM hq_xp5 UNION ALL BY NAME SELECT * FROM hq_xp5l) r JOIN hq_win w USING (dev)
),
hq_health_phase_15min AS (
SELECT w.DeviceId, r.ph AS Phase, make_timestamp_ns(r.q * 900000000000) AS TimeStamp, r.ngreen::INTEGER AS Greens
FROM hq_xgq r JOIN hq_win w USING (dev)
),
hq_health_tick_counts AS MATERIALIZED (
SELECT w.DeviceId, r.det::SMALLINT AS Detector, make_timestamp_ns(0 + r.q * 900000000000) AS TimeStamp, r.Kind, r.a::INTEGER AS A,
       r.tb::SMALLINT AS B, r.n::INTEGER AS Count
FROM (SELECT dev, det, q, 'on_us' AS Kind, a, NULL::INT AS tb, n FROM hq_xtko
      UNION ALL SELECT dev, det, q, 'gap_pair', ta, tb, n FROM hq_xtkg) r JOIN hq_win w USING (dev)
),
hq_health_intervals AS MATERIALIZED (
SELECT w.DeviceId, r.Kind, r.ch::INTEGER AS Channel, make_timestamp_ns(0 + r.s) AS StartTime, make_timestamp_ns(0 + r.f) AS EndTime,
       r.multi AS Relogged, r.val::INTEGER AS Value
FROM (SELECT dev, 'long_on' AS Kind, det AS ch, s, f, multi, NULL::INT AS val FROM hq_xhold
      UNION ALL SELECT dev, 'comms_gap', NULL, g0, g1, NULL, NULL FROM hq_xgap WHERE g1 IS NOT NULL
      UNION ALL SELECT dev, 'plan', NULL, dn, dn, NULL, par FROM hq_xplan
      UNION ALL SELECT dev, Kind, ch, s, f, NULL, val FROM hq_xbad) r JOIN hq_win w USING (dev)
),
hq_oxc AS (
SELECT c.dev, c.det, (c.s + 0 - w.s_ns) AS s, c.s / 1e9 AS sd, (c.mk + 0 - w.s_ns) AS mk FROM hq_xoc c JOIN hq_win w USING (dev)
),
hq_oxp AS (
SELECT p.dev, p.ph, p.st, (p.tag + 0 - w.s_ns) AS tag, p.ad, p.mk / 1e9 AS mkd FROM hq_xpc p JOIN hq_win w USING (dev) WHERE p.op AND p.st < 3
),
hq_oxl AS MATERIALIZED (
SELECT l.dev, (l.lz + 0 - w.s_ns) AS lz, l.lz / 1e9 AS lzd FROM hq_xlast l JOIN hq_win w USING (dev)
),
hq_oxt AS (
SELECT p.dev, p.det, (p.dn + 0 - w.s_ns) AS dn, p.ta FROM hq_xtf p JOIN hq_win w USING (dev) WHERE p.tb IS NULL
),
hq_rs5 AS (
SELECT w.dev, ((epoch_ns(r.TimeStamp) - w.s_ns) // 300000000000)::INT AS b FROM hq_health_signal_5min r JOIN hq_win w USING (DeviceId)
WHERE r.TimeStamp >= make_timestamp_ns(w.s_ns) AND r.TimeStamp < make_timestamp_ns(w.e_ns) GROUP BY ALL
),
hq_rd5 AS (
SELECT w.dev, r.Detector::INT AS det, ((epoch_ns(r.TimeStamp) - w.s_ns) // 300000000000)::INT AS b, r.Starts AS nst,
       r.Fast_Starts AS nfst, r.Chatter_Starts AS nchat, r.Starts_GY AS nst_gy, r.Fast_Starts_R2 AS nfst_r2,
       r.Chatter_Starts_R2 AS nchat_r2, r.Relogged_Ons AS nrel, r.Stray_Offs AS nxoff, r.Closing_Offs AS nclose,
       r.On_Time_ns AS on_ns, r.Unknown_Time_ns AS unk_ns, r.On_Green5_ns AS ong5_ns, r.Clean_Ons AS nclean,
       r.Max_On_s AS max_on, r.Relog_Time_ns AS relog_ns, r.Ons_5s AS n5s, r.Ons_60s AS n60s, r.Long_Max_s AS lmax
FROM hq_health_detector_5min r JOIN hq_win w USING (DeviceId)
WHERE r.Chain_Tag IS NULL AND r.Piece_Tag IS NULL AND r.TimeStamp >= make_timestamp_ns(w.s_ns) AND r.TimeStamp < make_timestamp_ns(w.e_ns)
),
hq_rd5l AS (
SELECT w.dev, r.Detector::INT AS det, ((epoch_ns(r.TimeStamp) - w.s_ns) // 300000000000)::INT AS b, (epoch_ns(r.Chain_Tag) - w.s_ns) AS ctag,
       r.Piece_Phase AS pph, (epoch_ns(r.Piece_Tag) - w.s_ns) AS ptag, r.Piece_Kind::INT AS pk, r.On_Time_ns AS on_ns,
       r.On_Green5_ns AS ong5_ns, r.Starts_GY AS nst_gy, r.Fast_Starts_R2 AS nfst_r2, r.Chatter_Starts_R2 AS nchat_r2
FROM hq_health_detector_5min r JOIN hq_win w USING (DeviceId)
WHERE (r.Chain_Tag IS NOT NULL OR r.Piece_Tag IS NOT NULL) AND r.TimeStamp >= make_timestamp_ns(w.s_ns) AND r.TimeStamp < make_timestamp_ns(w.e_ns)
),
hq_rp5 AS (
SELECT w.dev, r.Phase AS ph, ((epoch_ns(r.TimeStamp) - w.s_ns) // 300000000000)::INT AS b, r.Greens AS ngreen, r.Green_Yellow_ns AS gy_ns,
       r.Green5_ns AS g5_ns
FROM hq_health_phase_5min r JOIN hq_win w USING (DeviceId) WHERE r.Piece_Tag IS NULL AND r.TimeStamp >= make_timestamp_ns(w.s_ns) AND r.TimeStamp < make_timestamp_ns(w.e_ns)
),
hq_rp5l AS (
SELECT w.dev, r.Phase AS ph, ((epoch_ns(r.TimeStamp) - w.s_ns) // 300000000000)::INT AS b, (epoch_ns(r.Piece_Tag) - w.s_ns) AS ptag,
       r.Green_Yellow_ns AS gy_ns, r.Green5_ns AS g5_ns
FROM hq_health_phase_5min r JOIN hq_win w USING (DeviceId) WHERE r.Piece_Tag IS NOT NULL AND r.TimeStamp >= make_timestamp_ns(w.s_ns) AND r.TimeStamp < make_timestamp_ns(w.e_ns)
),
hq_rgq AS (
SELECT w.dev, r.Phase AS ph, epoch_ns(r.TimeStamp) // 3600000000000 AS h FROM hq_health_phase_15min r JOIN hq_win w USING (DeviceId)
WHERE r.TimeStamp >= make_timestamp_ns(w.s_ns - w.s_ns % 900000000000) AND r.TimeStamp < make_timestamp_ns(w.e_ns) GROUP BY ALL
),
hq_rtko AS (
SELECT w.dev, r.Detector::INT AS det, r.A AS a, r.Count AS n FROM hq_health_tick_counts r JOIN hq_win w USING (DeviceId)
WHERE r.Kind = 'on_us' AND r.TimeStamp >= make_timestamp_ns(w.s_ns) AND r.TimeStamp < make_timestamp_ns(w.e_ns)
),
hq_rtkg AS (
SELECT w.dev, r.Detector::INT AS det, (epoch_ns(r.TimeStamp) - w.s_ns) // 900000000000 AS q, r.A AS ta, r.B::INT AS tb, r.Count AS n
FROM hq_health_tick_counts r JOIN hq_win w USING (DeviceId) WHERE r.Kind = 'gap_pair' AND r.TimeStamp >= make_timestamp_ns(w.s_ns) AND r.TimeStamp < make_timestamp_ns(w.e_ns)
),
hq_rhold AS (
SELECT w.dev, r.Channel AS det, (epoch_ns(r.StartTime) - w.s_ns) AS s, (epoch_ns(r.EndTime) - w.s_ns) AS f
FROM hq_health_intervals r JOIN hq_win w USING (DeviceId)
WHERE r.Kind = 'long_on' AND r.EndTime >= make_timestamp_ns(w.s_ns) AND r.StartTime < make_timestamp_ns(w.e_ns)
),
hq_rplan AS (
SELECT w.dev, (epoch_ns(r.StartTime) - w.s_ns) AS dn, r.Value AS par FROM hq_health_intervals r JOIN hq_win w USING (DeviceId)
WHERE r.Kind = 'plan' AND r.StartTime >= make_timestamp_ns(w.s_ns) AND r.StartTime < make_timestamp_ns(w.e_ns)
),
hq_rbad AS MATERIALIZED (
SELECT w.dev, r.Kind AS k, r.Channel AS ch, (epoch_ns(r.StartTime) - w.s_ns) AS s, r.Value AS pk
FROM hq_health_intervals r JOIN hq_win w USING (DeviceId)
WHERE r.Kind IN ('bad_chain', 'bad_piece') AND r.EndTime >= make_timestamp_ns(w.s_ns)
),
hq_fch AS MATERIALIZED (
SELECT c.dev, c.det, c.s, (w.e_ns / 1e9) - c.sd AS dur,
       (w.e_ns / 1e9) - l.lzd > 120.0 OR ((w.e_ns / 1e9) - c.sd > 60 AND c.mk IS NOT NULL) AS gap
FROM hq_oxc c JOIN hq_win w USING (dev) JOIN hq_oxl l USING (dev)
),
hq_fpc AS MATERIALIZED (
SELECT p.dev, p.ph, p.st, p.tag, CASE WHEN p.st < 2 THEN 0 ELSE 2 END AS pk,
       NOT coalesce(p.mkd < CASE WHEN p.st = 2 THEN least(p.ad + 2.0, (w.e_ns / 1e9)) ELSE (w.e_ns / 1e9) END, false) AS valid
FROM hq_oxp p JOIN hq_win w USING (dev)
),
hq_fdl AS (
SELECT dev, det, b, CASE WHEN NOT gap THEN on_ns END AS on_ns, CASE WHEN gap THEN on_ns END AS unk_ns,
       CASE WHEN ok THEN nst_gy END AS nst_gy, CASE WHEN ok THEN nfst_r2 END AS nfst_r2,
       CASE WHEN ok THEN nchat_r2 END AS nchat_r2, CASE WHEN NOT gap AND ok THEN ong5_ns END AS ong5_ns
FROM (SELECT l.*, coalesce(c.gap, false) OR bc.s IS NOT NULL AS gap, coalesce(p.valid, true) AND bp.s IS NULL AS ok
      FROM hq_rd5l l
      LEFT JOIN hq_fch c ON c.dev = l.dev AND c.det = l.det AND c.s = l.ctag
      LEFT JOIN hq_fpc p ON p.dev = l.dev AND p.ph = l.pph AND p.tag = l.ptag AND p.pk = l.pk
      LEFT JOIN (SELECT * FROM hq_rbad WHERE k = 'bad_chain') bc ON bc.dev = l.dev AND bc.ch = l.det AND bc.s = l.ctag
      LEFT JOIN (SELECT * FROM hq_rbad WHERE k = 'bad_piece') bp ON bp.dev = l.dev AND bp.ch = l.pph AND bp.s = l.ptag AND bp.pk = l.pk)
),
hq_fct AS (
WITH o AS (SELECT o.*, (o.s // 300000000000)::INT AS b0, least(o.f // 300000000000, w.nb5 - 1)::INT AS b1
           FROM (SELECT c.dev, c.det, l.lz AS s, w.e_ns - w.s_ns AS f, c.gap FROM hq_fch c JOIN hq_oxl l USING (dev) JOIN hq_win w USING (dev)) o JOIN hq_win w USING (dev))
SELECT dev, det, gap, b0 AS b, least(f, (b0 + 1) * 300000000000) - s AS t FROM o WHERE b1 = b0
UNION ALL
SELECT dev, det, gap, b, least(f, (b + 1) * 300000000000) - greatest(s, b * 300000000000)
FROM (SELECT *, unnest(range(b0, b1 + 1))::INT AS b FROM o WHERE b1 > b0)
),
hq_fpt AS (
WITH o AS (SELECT o.*, (o.s // 300000000000)::INT AS b0, least(o.f // 300000000000, w.nb5 - 1)::INT AS b1
           FROM (SELECT p.dev, p.ph, l.lz AS s, w.e_ns - w.s_ns AS f, 0 AS kind FROM hq_fpc p JOIN hq_oxl l USING (dev) JOIN hq_win w USING (dev) WHERE p.st < 2 AND p.valid
   UNION ALL SELECT p.dev, p.ph, greatest(l.lz, p.tag + 5000000000), w.e_ns - w.s_ns, 1 FROM hq_fpc p JOIN hq_oxl l USING (dev) JOIN hq_win w USING (dev)
   WHERE p.st = 0 AND p.valid AND greatest(l.lz, p.tag + 5000000000) < w.e_ns - w.s_ns) o JOIN hq_win w USING (dev))
SELECT dev, ph, kind, b0 AS b, least(f, (b0 + 1) * 300000000000) - s AS t FROM o WHERE b1 = b0
UNION ALL
SELECT dev, ph, kind, b, least(f, (b + 1) * 300000000000) - greatest(s, b * 300000000000)
FROM (SELECT *, unnest(range(b0, b1 + 1))::INT AS b FROM o WHERE b1 > b0)
),
hq_fov AS (
WITH o AS (SELECT o.*, (o.s // 300000000000)::INT AS b0, least(o.f // 300000000000, w.nb5 - 1)::INT AS b1
           FROM (SELECT c.dev, c.det, greatest(l.lz, p.tag + 5000000000) AS s, w.e_ns - w.s_ns AS f FROM hq_fch c JOIN hq_xovd d USING (dev, det)
   JOIN hq_fpc p ON p.dev = c.dev AND p.ph = d.ph JOIN hq_oxl l ON l.dev = c.dev JOIN hq_win w ON w.dev = c.dev
   WHERE NOT c.gap AND p.st = 0 AND p.valid AND greatest(l.lz, p.tag + 5000000000) < w.e_ns - w.s_ns) o JOIN hq_win w USING (dev))
SELECT dev, det, b0 AS b, least(f, (b0 + 1) * 300000000000) - s AS t FROM o WHERE b1 = b0
UNION ALL
SELECT dev, det, b, least(f, (b + 1) * 300000000000) - greatest(s, b * 300000000000)
FROM (SELECT *, unnest(range(b0, b1 + 1))::INT AS b FROM o WHERE b1 > b0)
),
hq_d5 AS MATERIALIZED (
SELECT dev, det, b, (coalesce(sum(nst), 0) + coalesce(sum(nrel), 0) + coalesce(sum(nclose), 0)
                     + coalesce(sum(nxoff), 0))::BIGINT AS nev,
       (coalesce(sum(nst), 0) + coalesce(sum(nrel), 0))::BIGINT AS non, sum(nrel)::BIGINT AS nrel,
       sum(nst)::BIGINT AS nst, sum(nfst)::BIGINT AS nfst, sum(nchat)::BIGINT AS nchat, sum(nst_gy)::BIGINT AS nst_gy,
       sum(nfst_r2)::BIGINT AS nfst_r2, sum(nchat_r2)::BIGINT AS nchat_r2, sum(on_ns)::BIGINT AS on_ns,
       sum(unk_ns)::BIGINT AS unk_ns, sum(ong5_ns)::BIGINT AS ong5_ns, sum(nclean)::BIGINT AS nclean,
       max(max_on) AS max_on, sum(relog_ns)::BIGINT AS relog_ns, sum(n5s)::BIGINT AS n5s, sum(n60s)::BIGINT AS n60s,
       max(lmax) AS lmax
FROM (SELECT * FROM hq_rd5
      UNION ALL BY NAME SELECT * FROM hq_fdl
      UNION ALL BY NAME
      SELECT dev, det, b, CASE WHEN NOT gap THEN t END AS on_ns, CASE WHEN gap THEN t END AS unk_ns FROM hq_fct
      UNION ALL BY NAME
      SELECT dev, det, (greatest(s, 0) // 300000000000)::INT AS b, (NOT gap AND dur >= 5)::INT AS n5s, (NOT gap AND dur >= 60)::INT AS n60s,
             CASE WHEN NOT gap THEN dur END AS lmax FROM hq_fch
      UNION ALL BY NAME
      SELECT dev, det, b, t AS ong5_ns FROM hq_fov)
GROUP BY dev, det, b
),
hq_p5 AS MATERIALIZED (
SELECT dev, ph, b, sum(ngreen)::BIGINT AS ngreen, sum(gy_ns)::BIGINT AS gy_ns, sum(g5_ns)::BIGINT AS g5_ns
FROM (SELECT * FROM hq_rp5
      UNION ALL BY NAME
      SELECT l.dev, l.ph, l.b, l.gy_ns, l.g5_ns FROM hq_rp5l l
      LEFT JOIN hq_fpc p ON p.dev = l.dev AND p.ph = l.ph AND p.tag = l.ptag AND p.pk = 0
      LEFT JOIN (SELECT * FROM hq_rbad WHERE k = 'bad_piece' AND pk = 0) bp ON bp.dev = l.dev AND bp.ch = l.ph AND bp.s = l.ptag
      WHERE coalesce(p.valid, true) AND bp.s IS NULL
      UNION ALL BY NAME
      SELECT dev, ph, b, CASE WHEN kind = 0 THEN t END AS gy_ns, CASE WHEN kind = 1 THEN t END AS g5_ns FROM hq_fpt)
GROUP BY dev, ph, b
),
hq_tkg AS MATERIALIZED (
SELECT dev, det, q, ta, tb, n FROM hq_rtkg
UNION ALL SELECT dev, det, dn // 900000000000, ta, 9, 1 FROM hq_oxt
),
hq_cov5 AS MATERIALIZED (
SELECT dev, b FROM hq_rs5
),
hq_dets AS MATERIALIZED (
SELECT dev, det FROM hq_d5 GROUP BY dev, det HAVING sum(nev) > 0
),
hq_occ5 AS (
SELECT dev, det, b::BIGINT AS b, on_ns / 1e9 AS occ FROM hq_d5 WHERE on_ns IS NOT NULL
),
hq_unk5 AS (
SELECT dev, det, b::BIGINT AS b, unk_ns / 1e9 AS occ FROM hq_d5 WHERE unk_ns IS NOT NULL
),
hq_n5 AS (
SELECT dev, det, b, non AS n FROM hq_d5 WHERE non > 0
),
hq_bins5 AS MATERIALIZED (
SELECT b.dev, b.b, c.b IS NOT NULL AS cov, ((b.s_ns + b.b * 300000000000) // 3600000000000) % 24 AS hr
FROM (SELECT dev, s_ns, unnest(range(nb5)) AS b FROM hq_win) b LEFT JOIN hq_cov5 c ON c.dev = b.dev AND c.b = b.b
),
hq_g5 AS MATERIALIZED (
SELECT d.dev, d.det, b.b, b.cov, b.hr, coalesce(n.n, 0)::DOUBLE AS n, coalesce(o.occ, 0.0) AS occ, coalesce(u.occ, 0.0) AS unk
FROM hq_dets d JOIN hq_bins5 b USING (dev) LEFT JOIN hq_n5 n ON n.dev = d.dev AND n.det = d.det AND n.b = b.b
LEFT JOIN hq_occ5 o ON o.dev = d.dev AND o.det = d.det AND o.b = b.b LEFT JOIN hq_unk5 u ON u.dev = d.dev AND u.det = d.det AND u.b = b.b
),
hq_k5 AS MATERIALIZED (
SELECT dev, det, b + 1 AS k, sum(n) OVER w AS cn, sum((n = 0 AND (occ + unk) < 1)::INT) OVER w AS cq
FROM hq_g5 WINDOW w AS (PARTITION BY dev, det ORDER BY b ROWS UNBOUNDED PRECEDING)
UNION ALL SELECT dev, det, 0, 0.0, 0 FROM hq_dets
),
hq_dstat AS MATERIALIZED (
SELECT g.dev, g.det, sum(g.n * g.cov::INT) AS n_on, (sum(g.n) > 0 OR sum(g.occ) > 0) AS active,
       sum(g.n * g.cov::INT) > 0 AS live,
       sum(g.n * (g.cov AND g.hr >= 7 AND g.hr < 19)::INT) AS dy, sum(g.n * (g.cov AND (g.hr >= 21 OR g.hr < 5))::INT) AS nt
FROM hq_g5 g GROUP BY g.dev, g.det
),
hq_sig0 AS MATERIALIZED (
SELECT s.dev, s.n_cov, s.last_cov, s.ok_night, a.n_active
FROM (SELECT dev, count(*) FILTER (cov) AS n_cov, coalesce(max(b) FILTER (cov), -1) AS last_cov,
             count(*) FILTER (cov AND (hr >= 21 OR hr < 5)) >= 24 AND count(*) FILTER (cov AND hr >= 7 AND hr < 19) >= 24
             AS ok_night FROM hq_bins5 GROUP BY dev) s
JOIN (SELECT dev, sum(active::INT) AS n_active FROM hq_dstat GROUP BY dev) a USING (dev)
),
hq_twin AS MATERIALIZED (
SELECT a.dev, a.det AS i, b.det AS j
FROM hq_dstat a JOIN hq_dstat b ON a.dev = b.dev AND a.det <> b.det AND a.n_on > 0 AND b.n_on > 0
     AND abs(a.n_on - b.n_on) < 0.1 * greatest(a.n_on + b.n_on, 1)
JOIN hq_g5 x ON x.dev = a.dev AND x.det = a.det JOIN hq_g5 y ON y.dev = a.dev AND y.det = b.det AND y.b = x.b
GROUP BY a.dev, a.det, b.det, a.n_on, b.n_on
HAVING sum(abs(x.n * x.cov::INT - y.n * y.cov::INT)) / greatest(a.n_on + b.n_on, 1) < 0.1
),
hq_a15 AS MATERIALIZED (
SELECT g.dev, g.det, g.b // 3 AS k, sum(g.n) AS n15, bool_and(g.cov) AS c15
FROM hq_g5 g JOIN hq_win w USING (dev) WHERE g.b < w.nb5 // 3 * 3 GROUP BY g.dev, g.det, g.b // 3
),
hq_runs0 AS MATERIALIZED (
SELECT dev, det, min(b) AS a, max(b) + 1 AS z FROM (
  SELECT g.dev, g.det, g.b, g.b - row_number() OVER (PARTITION BY g.dev, g.det ORDER BY g.b) AS grp
  FROM hq_g5 g JOIN hq_dstat s USING (dev, det) WHERE s.n_on > 0 AND g.n = 0 AND (g.occ + g.unk) < 1 AND g.cov)
GROUP BY dev, det, grp
),
hq_runs AS MATERIALIZED (
SELECT dev, det, min(a) AS a, max(z) AS z FROM (
  SELECT *, sum(new::INT) OVER (PARTITION BY dev, det ORDER BY a) AS grp FROM (
    SELECT r.*, NOT coalesce(r.a - lag(r.z) OVER w = 1 AND lag(r.z - r.a) OVER w >= 6 AND r.z - r.a >= 6
                             AND g.n <= 2 AND g.cov AND g.hr >= 5 AND g.hr < 21, false) AS new
    FROM hq_runs0 r LEFT JOIN hq_g5 g ON g.dev = r.dev AND g.det = r.det AND g.b = r.a - 1
    WINDOW w AS (PARTITION BY r.dev, r.det ORDER BY r.a)))
GROUP BY dev, det, grp
),
hq_lvl0 AS (
SELECT dev, det, k, n15, S15, c15,
       sum(n15) OVER w AS cx, sum(S15) OVER w AS cs, row_number() OVER w AS pos,
       count(*) OVER (PARTITION BY dev, det) AS nl, sum(n15) OVER (PARTITION BY dev, det) AS X,
       sum(S15) OVER (PARTITION BY dev, det) AS SS
FROM (SELECT dev, det, k, n15, sum(n15) OVER (PARTITION BY dev, k) - n15 AS S15, c15 FROM hq_a15) WHERE c15
WINDOW w AS (PARTITION BY dev, det ORDER BY k ROWS UNBOUNDED PRECEDING)
),
hq_lvlshift AS (
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
            FROM hq_lvl0 WHERE nl >= 8 AND X >= 20 AND pos >= 2 AND pos <= nl - 2))
GROUP BY dev, det
),
hq_hold AS MATERIALIZED (
SELECT h.*, row_number() OVER (ORDER BY h.dev, h.det, h.s) AS hid FROM (
  SELECT dev, det, s / 1e9 AS s, f / 1e9 AS f, false AS open_end, f / 1e9 - s / 1e9 AS dur FROM hq_rhold
  UNION ALL
  SELECT c.dev, c.det, c.s / 1e9, w.T, true, c.dur FROM hq_fch c JOIN hq_win w USING (dev) WHERE NOT c.gap AND c.dur >= 300) h
),
hq_hold_co AS MATERIALIZED (
SELECT a.hid, count(DISTINCT b.det) FILTER (least(a.f, b.f) - greatest(a.s, b.s) >= 0.5 * a.dur) AS co
FROM hq_hold a LEFT JOIN hq_hold b ON a.dev = b.dev AND a.det <> b.det GROUP BY a.hid
),
hq_p1ph AS MATERIALIZED (
SELECT d.dev, d.det, c.ph FROM hq_dets d LEFT JOIN hq_dcl c USING (dev, det)
),
hq_ntm AS MATERIALIZED (
WITH h AS (SELECT dev, count(*) FILTER (cov AND hr >= 7 AND hr < 19) / 12.0 AS hd,
                  count(*) FILTER (cov AND hr >= 1 AND hr < 5) / 12.0 AS hn FROM hq_bins5 GROUP BY dev),
o AS (SELECT g.dev, g.det, coalesce(sum(g.n) FILTER (g.cov AND g.hr >= 7 AND g.hr < 19), 0) AS d,
             coalesce(sum(g.n) FILTER (g.cov AND g.hr >= 1 AND g.hr < 5), 0) AS n FROM hq_g5 g GROUP BY g.dev, g.det),
r AS (SELECT a.dev, a.det, coalesce(sum(b.d) FILTER (a.ph = b.ph), 0) AS pd, coalesce(sum(b.n) FILTER (a.ph = b.ph), 0) AS pn,
             sum(b.d) AS sd, sum(b.n) AS sn
      FROM (SELECT o.*, p.ph FROM o JOIN hq_p1ph p USING (dev, det)) a
      JOIN (SELECT o.*, p.ph FROM o JOIN hq_p1ph p USING (dev, det)) b ON b.dev = a.dev AND b.det <> a.det
      GROUP BY a.dev, a.det)
SELECT o.dev, o.det, h.hd, h.hn, o.d / h.hd AS own_d, o.n / h.hn AS own_n, x.ph_rest,
       CASE WHEN x.ph_rest THEN r.pd ELSE r.sd END / h.hd AS rest_d, CASE WHEN x.ph_rest THEN r.pn ELSE r.sn END / h.hn AS rest_n
FROM o JOIN h USING (dev) JOIN r USING (dev, det)
CROSS JOIN LATERAL (SELECT r.pd / greatest(h.hd, 1e-9) >= CAST(20.0 AS DOUBLE) AS ph_rest) x
WHERE h.hd >= 2 AND h.hn >= 2
),
hq_ntraf AS MATERIALIZED (
SELECT dev, det FROM hq_ntm WHERE own_d >= CAST(1.5 AS DOUBLE) AND rest_d >= CAST(20.0 AS DOUBLE)
  AND own_n >= CAST(0.5 AS DOUBLE) * own_d AND rest_n <= CAST(0.3 AS DOUBLE) * rest_d
),
hq_p1rf AS MATERIALIZED (
SELECT a.dev, a.det, b.det AS r, coalesce(a.ph = b.ph, false) AS same
FROM hq_p1ph a JOIN hq_p1ph b ON a.dev = b.dev AND a.det <> b.det
JOIN hq_dstat sb ON sb.dev = b.dev AND sb.det = b.det AND sb.live
LEFT JOIN hq_twin t ON t.dev = a.dev AND t.i = a.det AND t.j = b.det WHERE t.i IS NULL
),
hq_p1kd AS MATERIALIZED (
SELECT a.dev, a.det, a.ph, coalesce(r.nsame, 0) >= 2 AS isph
FROM hq_p1ph a LEFT JOIN (SELECT dev, det, count(*) FILTER (same) AS nsame FROM hq_p1rf GROUP BY dev, det) r
USING (dev, det)
),
hq_p1pref AS MATERIALIZED (
SELECT f.dev, f.det, f.r FROM hq_p1rf f JOIN hq_p1kd k USING (dev, det) WHERE k.isph AND f.same
),
hq_p1pc AS MATERIALIZED (
SELECT a.dev, a.det, b.det AS c FROM hq_p1ph a JOIN hq_p1ph b ON a.dev = b.dev AND a.det <> b.det AND a.ph = b.ph
JOIN hq_dstat sb ON sb.dev = b.dev AND sb.det = b.det AND sb.live
),
hq_p1rm AS MATERIALIZED (
SELECT dev, det, r FROM hq_p1pref
UNION ALL SELECT k.dev, k.det, d.det FROM hq_p1kd k JOIN hq_dets d ON d.dev = k.dev AND d.det <> k.det WHERE NOT k.isph
),
hq_p1rs0 AS MATERIALIZED (
SELECT u.dev, u.det, u.a, u.z, m.r, w.nb5 FROM hq_runs u JOIN hq_p1rm m ON m.dev = u.dev AND m.det = u.det
JOIN hq_win w ON w.dev = u.dev
),
hq_p1rs_1 AS MATERIALIZED (
SELECT x.*, k.cn AS cn_a FROM hq_p1rs0 x JOIN hq_k5 k ON k.dev = x.dev AND k.det = x.r AND k.k = x.a
),
hq_p1rs_2 AS MATERIALIZED (
SELECT x.*, k.cn AS cn_z FROM hq_p1rs_1 x JOIN hq_k5 k ON k.dev = x.dev AND k.det = x.r AND k.k = x.z
),
hq_p1rs_3 AS MATERIALIZED (
SELECT x.*, k.cn AS cn_nb5 FROM hq_p1rs_2 x JOIN hq_k5 k ON k.dev = x.dev AND k.det = x.r AND k.k = x.nb5
),
hq_p1rs AS (
SELECT dev, det, a, z, sum(cn_z - cn_a) AS s_in, sum(cn_nb5) AS s_tot FROM hq_p1rs_3 GROUP BY dev, det, a, z
),
hq_p1dr AS MATERIALIZED (
SELECT k.dev, k.det, coalesce(d.lam, 0.0) AS drop_lam, coalesce(d.b0, -1) AS drop_b0, coalesce(d.b1, -1) AS drop_b1
FROM hq_p1kd k LEFT JOIN (
  SELECT dev, det, max(lam) AS lam, first(a ORDER BY round(lam, 6) DESC, a) AS b0, first(z ORDER BY round(lam, 6) DESC, a) AS b1
  FROM (SELECT r.*, s.n_on / greatest(r.s_tot - r.s_in, 1) * r.s_in AS lam FROM hq_p1rs r JOIN hq_dstat s USING (dev, det)
        JOIN hq_win w ON w.dev = r.dev WHERE r.z - r.a <= w.nb5 - 12 AND s.n_on >= 5)
  WHERE lam > 0 GROUP BY dev, det) d USING (dev, det)
),
hq_p1dra AS MATERIALIZED (
SELECT r.dev, r.det, r.a, r.z, s.n_on / greatest(r.s_tot - r.s_in, 1) * r.s_in AS lam
FROM hq_p1rs r JOIN hq_dstat s USING (dev, det) JOIN hq_win w ON w.dev = r.dev
WHERE r.z - r.a >= 3 AND r.z - r.a <= w.nb5 - 12 AND s.n_on >= 5
),
hq_p1drx AS MATERIALIZED (
SELECT d.*, s.n_on, (d.drop_b1 = w.nb5 OR (d.drop_b1 > 0 AND g.last_cov < d.drop_b1)) AS drop_to_end
FROM hq_p1dr d JOIN hq_dstat s USING (dev, det) JOIN hq_win w ON w.dev = d.dev JOIN hq_sig0 g ON g.dev = d.dev
),
hq_p1rq AS MATERIALIZED (
SELECT 'n' AS kind, dev * 128 + det AS id, dev, det, 0 AS xa, 0 AS xb FROM hq_p1kd
UNION ALL
SELECT 'e', h.hid, h.dev, h.det, floor(h.s / 300.0)::INT // 3, ceil(least(ceil(h.f / 300.0), w.nb5) / 3.0)::INT
FROM hq_hold h JOIN hq_p1kd k USING (dev, det) JOIN hq_win w ON w.dev = h.dev WHERE h.dur >= 900
UNION ALL
SELECT 'd', dev * 128 + det, dev, det, drop_b0 // 3, ceil(drop_b1 / 3.0)::INT FROM hq_p1drx
WHERE n_on > 0 AND drop_lam >= 30 AND drop_b1 > drop_b0
),
hq_p1rc AS (
SELECT r.kind, r.id, c.c FROM hq_p1rq r JOIN hq_p1pc c ON c.dev = r.dev AND c.det = r.det
WHERE r.kind <> 'e' OR NOT EXISTS (
  SELECT 1 FROM hq_hold e JOIN hq_hold o ON o.dev = e.dev AND o.det = c.c AND o.dur >= 900 AND o.s < e.f AND o.f > e.s
  WHERE e.hid = r.id)
),
hq_p1pxy AS (
SELECT r.kind, r.id, c.c, x.n15 AS x, y.n15 AS y, x.k + 1 AS w
FROM hq_p1rq r JOIN hq_p1rc c ON c.kind = r.kind AND c.id = r.id
JOIN hq_a15 x ON x.dev = r.dev AND x.det = r.det AND x.c15 AND NOT (x.k >= r.xa AND x.k < r.xb)
JOIN hq_a15 y ON y.dev = r.dev AND y.det = c.c AND y.k = x.k
),
hq_p1part AS MATERIALIZED (
SELECT kind, id, p FROM (
  SELECT kind, id, first(c ORDER BY round(cr, 12) DESC, ys DESC, yw DESC, c) AS p,
         first(cr ORDER BY round(cr, 12) DESC, ys DESC, yw DESC, c) AS pcr
  FROM (SELECT kind, id, c, corr(y, x) AS cr, sum(y) AS ys, sum(y * w) AS yw, count(*) AS m,
               min(x) < max(x) AS xv, min(y) < max(y) AS yv FROM hq_p1pxy GROUP BY kind, id, c)
  WHERE m >= 16 AND xv AND yv GROUP BY kind, id) WHERE pcr >= 0.7
),
hq_p1rr AS (
SELECT 'e' AS kind, h.hid AS id, h.dev, h.det, floor(h.s / 300.0)::INT AS a, least(ceil(h.f / 300.0)::INT, w.nb5) AS b
FROM hq_hold h JOIN hq_p1kd k USING (dev, det) JOIN hq_win w ON w.dev = h.dev WHERE h.dur >= 900 AND NOT h.open_end
UNION ALL
SELECT 'd', dev * 128 + det, dev, det, drop_b0, drop_b1 FROM hq_p1drx
WHERE n_on > 0 AND drop_lam >= 30 AND drop_b1 > drop_b0 AND NOT drop_to_end
),
hq_p1rr2 AS MATERIALIZED (
SELECT r.*, greatest(r.a - 72, 0) AS lo0, least(r.b + 72, w.nb5) AS hi1, pa.p,
       (r.a >= 36 AND w.nb5 - r.b >= 36) AS can
FROM hq_p1rr r JOIN hq_win w ON w.dev = r.dev LEFT JOIN hq_p1part pa ON pa.kind = r.kind AND pa.id = r.id
),
hq_p1rec AS MATERIALIZED (
SELECT kind, id, CASE WHEN can AND c0 >= 36 AND c1 >= 36 THEN ((x1 + 0.5) / greatest(s1, 1.0)) /
                      ((x0 + 0.5) / greatest(s0, 1.0)) END AS rec
FROM (
  SELECT r.kind, r.id, any_value(r.can) AS can,
         count(*) FILTER (mm.own AND g.cov AND g.b < r.a) AS c0, count(*) FILTER (mm.own AND g.cov AND g.b >= r.b) AS c1,
         coalesce(sum(g.n) FILTER (mm.own AND g.b < r.a), 0) AS x0, coalesce(sum(g.n) FILTER (mm.own AND g.b >= r.b), 0) AS x1,
         coalesce(sum(g.n) FILTER (NOT mm.own AND g.b < r.a), 0) AS s0,
         coalesce(sum(g.n) FILTER (NOT mm.own AND g.b >= r.b), 0) AS s1
  FROM hq_p1rr2 r
  JOIN (SELECT r.kind, r.id, r.det AS m, true AS own FROM hq_p1rr2 r
        UNION ALL SELECT r.kind, r.id, r.p, false FROM hq_p1rr2 r WHERE r.p IS NOT NULL
        UNION ALL SELECT r.kind, r.id, m.r, false FROM hq_p1rr2 r JOIN hq_p1rm m ON m.dev = r.dev AND m.det = r.det
        WHERE r.p IS NULL) mm ON mm.kind = r.kind AND mm.id = r.id
  JOIN hq_g5 g ON g.dev = r.dev AND g.det = mm.m AND ((g.b >= r.lo0 AND g.b < r.a) OR (g.b >= r.b AND g.b < r.hi1))
  GROUP BY r.kind, r.id)
),
hq_p1ep AS (
SELECT h.dev, h.det, max(h.dur) AS ep_dur, first(h.s ORDER BY h.dur DESC, h.s) AS ep_t0,
       first(h.f ORDER BY h.dur DESC, h.s) AS ep_t1, first(c.co ORDER BY h.dur DESC, h.s) AS ep_co,
       first(r.rec ORDER BY h.dur DESC, h.s) AS ep_rec
FROM hq_hold h JOIN hq_p1kd k USING (dev, det) JOIN hq_hold_co c ON c.hid = h.hid
LEFT JOIN hq_p1rec r ON r.kind = 'e' AND r.id = h.hid WHERE h.dur >= 900 GROUP BY h.dev, h.det
),
hq_p1dis AS (
SELECT d.dev, d.det, coalesce(sum(greatest(least(h.f, d.drop_b1 * 300.0) - greatest(h.s, d.drop_b0 * 300.0), 0.0))
         FILTER (h.hid IS NOT NULL), 0)
       >= 0.5 * (d.drop_b1 - d.drop_b0) * 300.0 AS drop_in_stuck
FROM hq_p1drx d LEFT JOIN hq_hold h ON h.dev = d.dev AND h.det = d.det AND h.dur >= 900
WHERE d.n_on > 0 AND d.drop_lam >= 30 AND d.drop_b1 > d.drop_b0 GROUP BY d.dev, d.det, d.drop_b0, d.drop_b1
),
hq_p1nkd AS MATERIALIZED (
SELECT a.dev, a.det, a.ph, coalesce(r.nsame, 0) >= 2 AS isph
FROM hq_p1ph a LEFT JOIN (SELECT dev, det, count(*) FILTER (same) AS nsame FROM hq_p1rf f
                        WHERE NOT EXISTS (SELECT 1 FROM hq_ntraf t WHERE t.dev = f.dev AND t.det = f.r) GROUP BY dev, det) r
USING (dev, det)
),
hq_p1nrm AS MATERIALIZED (
SELECT f.dev, f.det, f.r FROM hq_p1rf f JOIN hq_p1nkd k USING (dev, det)
WHERE k.isph AND f.same AND NOT EXISTS (SELECT 1 FROM hq_ntraf t WHERE t.dev = f.dev AND t.det = f.r)
UNION ALL SELECT k.dev, k.det, d.det FROM hq_p1nkd k JOIN hq_dets d ON d.dev = k.dev AND d.det <> k.det
WHERE NOT k.isph AND NOT EXISTS (SELECT 1 FROM hq_ntraf t WHERE t.dev = d.dev AND t.det = d.det)
),
hq_p1rsum AS (
SELECT m.dev, m.det, sum(x.dy) AS rd, sum(x.nt) AS rn FROM hq_p1nrm m JOIN hq_dstat x ON x.dev = m.dev AND x.det = m.r
GROUP BY m.dev, m.det
),
hq_p1nt AS (
SELECT dev, det, nt AS night_n, ex AS night_exp, (nt + 0.5) / (ex + 0.5) AS night_ratio, who FROM (
  SELECT k.dev, k.det, s.nt, s.dy / greatest(coalesce(ps.dy, r.rd, 0), 1) * coalesce(ps.nt, r.rn, 0) AS ex,
         CASE WHEN pa.p IS NOT NULL THEN 'd' || pa.p || ', which it tracks'
              WHEN k.isph THEN 'the other detectors on its phase' ELSE 'the rest of the signal' END AS who
  FROM hq_p1nkd k JOIN hq_dstat s USING (dev, det) JOIN hq_sig0 g ON g.dev = k.dev
  LEFT JOIN (SELECT * FROM hq_p1part x WHERE NOT EXISTS (SELECT 1 FROM hq_ntraf t WHERE t.dev = x.id // 128 AND t.det = x.p)) pa ON pa.kind = 'n' AND pa.id = k.dev * 128 + k.det
  LEFT JOIN hq_dstat ps ON ps.dev = k.dev AND ps.det = pa.p LEFT JOIN hq_p1rsum r ON r.dev = k.dev AND r.det = k.det
  WHERE g.ok_night AND s.dy >= 20) WHERE ex >= 30
),
hq_p1st AS MATERIALIZED (
SELECT *, CASE WHEN n_on < 20 AND s_stuck_u = 0 THEN 'not_enough_data' WHEN hs_u < 0.25 THEN 'bad' WHEN hs_u < 0.70 THEN 'suspect'
               ELSE 'ok' END AS status_u, CASE WHEN hs < 0.25 THEN 'bad' WHEN hs < 0.70 THEN 'suspect' WHEN n_on < 20 THEN 'not_enough_data'
               ELSE 'ok' END AS status
FROM (SELECT *, (1 - coalesce(s_dropout, 0)) * (1 - s_stuck) * (1 - coalesce(s_night_drop, 0))
                * (1 - coalesce(s_level, 0)) AS hs, (1 - coalesce(s_dropout_u, 0)) * (1 - s_stuck_u) * (1 - coalesce(s_night_drop, 0))
                * (1 - coalesce(s_level, 0)) AS hs_u
  FROM (SELECT d.dev, d.det, d.n_on, d.drop_lam, d.drop_b0, d.drop_b1, d.drop_to_end,
          coalesce(i.drop_in_stuck, false) AS drop_in_stuck, rd.rec AS drop_rec, e.ep_dur, e.ep_t0, e.ep_t1, e.ep_co,
          e.ep_rec, n.night_ratio, n.night_n, n.night_exp, n.who AS night_ref, l.level_ratio,
          coalesce(l.level_llr, 0.0) AS level_llr, l.level_b,
          CASE WHEN d.n_on > 0 THEN (
            CASE WHEN coalesce(i.drop_in_stuck, false) THEN 0.0
                 WHEN (CASE WHEN (d.drop_lam) IS NULL OR isnan(d.drop_lam) THEN NULL WHEN (d.drop_lam) < 30.0 THEN 0.0 ELSE 0.35 + 0.65 * least(greatest(((d.drop_lam) - 30.0) / (100.0 - 30.0), 0.0), 1.0) END) > 0 AND NOT d.drop_to_end AND rd.rec BETWEEN 0.5 AND 2.0 THEN least((CASE WHEN (d.drop_lam) IS NULL OR isnan(d.drop_lam) THEN NULL WHEN (d.drop_lam) < 30.0 THEN 0.0 ELSE 0.35 + 0.65 * least(greatest(((d.drop_lam) - 30.0) / (100.0 - 30.0), 0.0), 1.0) END), 0.35)
                 ELSE (CASE WHEN (d.drop_lam) IS NULL OR isnan(d.drop_lam) THEN NULL WHEN (d.drop_lam) < 30.0 THEN 0.0 ELSE 0.35 + 0.65 * least(greatest(((d.drop_lam) - 30.0) / (100.0 - 30.0), 0.0), 1.0) END) END) END AS s_dropout,
          CASE WHEN coalesce(e.ep_dur, 0) >= 900 AND (e.ep_co >= 3 OR e.ep_rec BETWEEN 0.5 AND 2.0)
               THEN least((CASE WHEN (coalesce(e.ep_dur, 0)) IS NULL OR isnan(coalesce(e.ep_dur, 0)) THEN NULL WHEN (coalesce(e.ep_dur, 0)) < 900.0 THEN 0.0 ELSE 0.35 + 0.65 * least(greatest(((coalesce(e.ep_dur, 0)) - 900.0) / (3600.0 - 900.0), 0.0), 1.0) END), 0.35) ELSE (CASE WHEN (coalesce(e.ep_dur, 0)) IS NULL OR isnan(coalesce(e.ep_dur, 0)) THEN NULL WHEN (coalesce(e.ep_dur, 0)) < 900.0 THEN 0.0 ELSE 0.35 + 0.65 * least(greatest(((coalesce(e.ep_dur, 0)) - 900.0) / (3600.0 - 900.0), 0.0), 1.0) END) END AS s_stuck,
CASE WHEN d.n_on > 0 THEN CASE WHEN coalesce(i.drop_in_stuck, false) THEN 0.0
                  ELSE (CASE WHEN (d.drop_lam) IS NULL OR isnan(d.drop_lam) THEN NULL WHEN (d.drop_lam) < 30.0 THEN 0.0 ELSE 0.35 + 0.65 * least(greatest(((d.drop_lam) - 30.0) / (100.0 - 30.0), 0.0), 1.0) END) END END
            AS s_dropout_u,
          CASE WHEN coalesce(e.ep_dur, 0) >= 900 AND (e.ep_co >= 3 OR (e.ep_rec BETWEEN 0.5 AND 2.0 AND e.ep_dur < 3600))
               THEN least((CASE WHEN (coalesce(e.ep_dur, 0)) IS NULL OR isnan(coalesce(e.ep_dur, 0)) THEN NULL WHEN (coalesce(e.ep_dur, 0)) < 900.0 THEN 0.0 ELSE 0.35 + 0.65 * least(greatest(((coalesce(e.ep_dur, 0)) - 900.0) / (3600.0 - 900.0), 0.0), 1.0) END), 0.35) ELSE (CASE WHEN (coalesce(e.ep_dur, 0)) IS NULL OR isnan(coalesce(e.ep_dur, 0)) THEN NULL WHEN (coalesce(e.ep_dur, 0)) < 900.0 THEN 0.0 ELSE 0.35 + 0.65 * least(greatest(((coalesce(e.ep_dur, 0)) - 900.0) / (3600.0 - 900.0), 0.0), 1.0) END) END AS s_stuck_u,

          CASE WHEN n.night_ratio IS NOT NULL THEN
               (CASE WHEN (-ln(greatest(n.night_ratio, 1e-6))) IS NULL OR isnan(-ln(greatest(n.night_ratio, 1e-6))) THEN NULL WHEN (-ln(greatest(n.night_ratio, 1e-6))) < -ln(0.19) THEN 0.0 ELSE 0.35 + 0.65 * least(greatest(((-ln(greatest(n.night_ratio, 1e-6))) - -ln(0.19)) / (-ln(0.063) - -ln(0.19)), 0.0), 1.0) END) END AS s_night_drop,
          CASE WHEN l.level_ratio IS NOT NULL THEN CASE WHEN l.level_llr <= 50 THEN 0.0
               ELSE (CASE WHEN (-ln(l.level_ratio)) IS NULL OR isnan(-ln(l.level_ratio)) THEN NULL WHEN (-ln(l.level_ratio)) < -ln(0.15) THEN 0.0 ELSE 0.35 + 0.65 * least(greatest(((-ln(l.level_ratio)) - -ln(0.15)) / (-ln(0.05) - -ln(0.15)), 0.0), 1.0) END) END END AS s_level
        FROM hq_p1drx d LEFT JOIN hq_p1dis i USING (dev, det)
        LEFT JOIN hq_p1rec rd ON rd.kind = 'd' AND rd.id = d.dev * 128 + d.det
        LEFT JOIN hq_p1ep e USING (dev, det) LEFT JOIN hq_p1nt n USING (dev, det) LEFT JOIN hq_lvlshift l USING (dev, det)))
),
hq_p2sig AS MATERIALIZED (
SELECT DISTINCT s.dev FROM hq_p1st s WHERE s.status = 'bad'
  AND s.dev IN (SELECT dev FROM hq_dcl WHERE ph IS NOT NULL AND NOT isnan(ph))
),
hq_p2ph AS MATERIALIZED (
SELECT p.dev, p.det, CASE WHEN s.status = 'bad' THEN NULL ELSE p.ph END AS ph
FROM hq_p1ph p JOIN hq_p1st s USING (dev, det) WHERE p.dev IN (SELECT dev FROM hq_p2sig)
),
hq_p2rf AS MATERIALIZED (
SELECT a.dev, a.det, b.det AS r, coalesce(a.ph = b.ph, false) AS same
FROM hq_p2ph a JOIN hq_p2ph b ON a.dev = b.dev AND a.det <> b.det
JOIN hq_dstat sb ON sb.dev = b.dev AND sb.det = b.det AND sb.live
LEFT JOIN hq_twin t ON t.dev = a.dev AND t.i = a.det AND t.j = b.det WHERE t.i IS NULL
),
hq_p2kd AS MATERIALIZED (
SELECT a.dev, a.det, a.ph, coalesce(r.nsame, 0) >= 2 AS isph
FROM hq_p2ph a LEFT JOIN (SELECT dev, det, count(*) FILTER (same) AS nsame FROM hq_p2rf GROUP BY dev, det) r
USING (dev, det)
),
hq_p2pref AS MATERIALIZED (
SELECT f.dev, f.det, f.r FROM hq_p2rf f JOIN hq_p2kd k USING (dev, det) WHERE k.isph AND f.same
),
hq_p2pc AS MATERIALIZED (
SELECT a.dev, a.det, b.det AS c FROM hq_p2ph a JOIN hq_p2ph b ON a.dev = b.dev AND a.det <> b.det AND a.ph = b.ph
JOIN hq_dstat sb ON sb.dev = b.dev AND sb.det = b.det AND sb.live
),
hq_p2rm AS MATERIALIZED (
SELECT dev, det, r FROM hq_p2pref
UNION ALL SELECT k.dev, k.det, d.det FROM hq_p2kd k JOIN hq_dets d ON d.dev = k.dev AND d.det <> k.det WHERE NOT k.isph
),
hq_p2rs0 AS MATERIALIZED (
SELECT u.dev, u.det, u.a, u.z, m.r, w.nb5 FROM hq_runs u JOIN hq_p2rm m ON m.dev = u.dev AND m.det = u.det
JOIN hq_win w ON w.dev = u.dev
),
hq_p2rs_1 AS MATERIALIZED (
SELECT x.*, k.cn AS cn_a FROM hq_p2rs0 x JOIN hq_k5 k ON k.dev = x.dev AND k.det = x.r AND k.k = x.a
),
hq_p2rs_2 AS MATERIALIZED (
SELECT x.*, k.cn AS cn_z FROM hq_p2rs_1 x JOIN hq_k5 k ON k.dev = x.dev AND k.det = x.r AND k.k = x.z
),
hq_p2rs_3 AS MATERIALIZED (
SELECT x.*, k.cn AS cn_nb5 FROM hq_p2rs_2 x JOIN hq_k5 k ON k.dev = x.dev AND k.det = x.r AND k.k = x.nb5
),
hq_p2rs AS (
SELECT dev, det, a, z, sum(cn_z - cn_a) AS s_in, sum(cn_nb5) AS s_tot FROM hq_p2rs_3 GROUP BY dev, det, a, z
),
hq_p2dr AS MATERIALIZED (
SELECT k.dev, k.det, coalesce(d.lam, 0.0) AS drop_lam, coalesce(d.b0, -1) AS drop_b0, coalesce(d.b1, -1) AS drop_b1
FROM hq_p2kd k LEFT JOIN (
  SELECT dev, det, max(lam) AS lam, first(a ORDER BY round(lam, 6) DESC, a) AS b0, first(z ORDER BY round(lam, 6) DESC, a) AS b1
  FROM (SELECT r.*, s.n_on / greatest(r.s_tot - r.s_in, 1) * r.s_in AS lam FROM hq_p2rs r JOIN hq_dstat s USING (dev, det)
        JOIN hq_win w ON w.dev = r.dev WHERE r.z - r.a <= w.nb5 - 12 AND s.n_on >= 5)
  WHERE lam > 0 GROUP BY dev, det) d USING (dev, det)
),
hq_p2dra AS MATERIALIZED (
SELECT r.dev, r.det, r.a, r.z, s.n_on / greatest(r.s_tot - r.s_in, 1) * r.s_in AS lam
FROM hq_p2rs r JOIN hq_dstat s USING (dev, det) JOIN hq_win w ON w.dev = r.dev
WHERE r.z - r.a >= 3 AND r.z - r.a <= w.nb5 - 12 AND s.n_on >= 5
),
hq_p2drx AS MATERIALIZED (
SELECT d.*, s.n_on, (d.drop_b1 = w.nb5 OR (d.drop_b1 > 0 AND g.last_cov < d.drop_b1)) AS drop_to_end
FROM hq_p2dr d JOIN hq_dstat s USING (dev, det) JOIN hq_win w ON w.dev = d.dev JOIN hq_sig0 g ON g.dev = d.dev
),
hq_p2rq AS MATERIALIZED (
SELECT 'n' AS kind, dev * 128 + det AS id, dev, det, 0 AS xa, 0 AS xb FROM hq_p2kd
UNION ALL
SELECT 'e', h.hid, h.dev, h.det, floor(h.s / 300.0)::INT // 3, ceil(least(ceil(h.f / 300.0), w.nb5) / 3.0)::INT
FROM hq_hold h JOIN hq_p2kd k USING (dev, det) JOIN hq_win w ON w.dev = h.dev WHERE h.dur >= 900
UNION ALL
SELECT 'd', dev * 128 + det, dev, det, drop_b0 // 3, ceil(drop_b1 / 3.0)::INT FROM hq_p2drx
WHERE n_on > 0 AND drop_lam >= 30 AND drop_b1 > drop_b0
),
hq_p2rc AS (
SELECT r.kind, r.id, c.c FROM hq_p2rq r JOIN hq_p2pc c ON c.dev = r.dev AND c.det = r.det
WHERE r.kind <> 'e' OR NOT EXISTS (
  SELECT 1 FROM hq_hold e JOIN hq_hold o ON o.dev = e.dev AND o.det = c.c AND o.dur >= 900 AND o.s < e.f AND o.f > e.s
  WHERE e.hid = r.id)
),
hq_p2pxy AS (
SELECT r.kind, r.id, c.c, x.n15 AS x, y.n15 AS y, x.k + 1 AS w
FROM hq_p2rq r JOIN hq_p2rc c ON c.kind = r.kind AND c.id = r.id
JOIN hq_a15 x ON x.dev = r.dev AND x.det = r.det AND x.c15 AND NOT (x.k >= r.xa AND x.k < r.xb)
JOIN hq_a15 y ON y.dev = r.dev AND y.det = c.c AND y.k = x.k
),
hq_p2part AS MATERIALIZED (
SELECT kind, id, p FROM (
  SELECT kind, id, first(c ORDER BY round(cr, 12) DESC, ys DESC, yw DESC, c) AS p,
         first(cr ORDER BY round(cr, 12) DESC, ys DESC, yw DESC, c) AS pcr
  FROM (SELECT kind, id, c, corr(y, x) AS cr, sum(y) AS ys, sum(y * w) AS yw, count(*) AS m,
               min(x) < max(x) AS xv, min(y) < max(y) AS yv FROM hq_p2pxy GROUP BY kind, id, c)
  WHERE m >= 16 AND xv AND yv GROUP BY kind, id) WHERE pcr >= 0.7
),
hq_p2rr AS (
SELECT 'e' AS kind, h.hid AS id, h.dev, h.det, floor(h.s / 300.0)::INT AS a, least(ceil(h.f / 300.0)::INT, w.nb5) AS b
FROM hq_hold h JOIN hq_p2kd k USING (dev, det) JOIN hq_win w ON w.dev = h.dev WHERE h.dur >= 900 AND NOT h.open_end
UNION ALL
SELECT 'd', dev * 128 + det, dev, det, drop_b0, drop_b1 FROM hq_p2drx
WHERE n_on > 0 AND drop_lam >= 30 AND drop_b1 > drop_b0 AND NOT drop_to_end
),
hq_p2rr2 AS MATERIALIZED (
SELECT r.*, greatest(r.a - 72, 0) AS lo0, least(r.b + 72, w.nb5) AS hi1, pa.p,
       (r.a >= 36 AND w.nb5 - r.b >= 36) AS can
FROM hq_p2rr r JOIN hq_win w ON w.dev = r.dev LEFT JOIN hq_p2part pa ON pa.kind = r.kind AND pa.id = r.id
),
hq_p2rec AS MATERIALIZED (
SELECT kind, id, CASE WHEN can AND c0 >= 36 AND c1 >= 36 THEN ((x1 + 0.5) / greatest(s1, 1.0)) /
                      ((x0 + 0.5) / greatest(s0, 1.0)) END AS rec
FROM (
  SELECT r.kind, r.id, any_value(r.can) AS can,
         count(*) FILTER (mm.own AND g.cov AND g.b < r.a) AS c0, count(*) FILTER (mm.own AND g.cov AND g.b >= r.b) AS c1,
         coalesce(sum(g.n) FILTER (mm.own AND g.b < r.a), 0) AS x0, coalesce(sum(g.n) FILTER (mm.own AND g.b >= r.b), 0) AS x1,
         coalesce(sum(g.n) FILTER (NOT mm.own AND g.b < r.a), 0) AS s0,
         coalesce(sum(g.n) FILTER (NOT mm.own AND g.b >= r.b), 0) AS s1
  FROM hq_p2rr2 r
  JOIN (SELECT r.kind, r.id, r.det AS m, true AS own FROM hq_p2rr2 r
        UNION ALL SELECT r.kind, r.id, r.p, false FROM hq_p2rr2 r WHERE r.p IS NOT NULL
        UNION ALL SELECT r.kind, r.id, m.r, false FROM hq_p2rr2 r JOIN hq_p2rm m ON m.dev = r.dev AND m.det = r.det
        WHERE r.p IS NULL) mm ON mm.kind = r.kind AND mm.id = r.id
  JOIN hq_g5 g ON g.dev = r.dev AND g.det = mm.m AND ((g.b >= r.lo0 AND g.b < r.a) OR (g.b >= r.b AND g.b < r.hi1))
  GROUP BY r.kind, r.id)
),
hq_p2ep AS (
SELECT h.dev, h.det, max(h.dur) AS ep_dur, first(h.s ORDER BY h.dur DESC, h.s) AS ep_t0,
       first(h.f ORDER BY h.dur DESC, h.s) AS ep_t1, first(c.co ORDER BY h.dur DESC, h.s) AS ep_co,
       first(r.rec ORDER BY h.dur DESC, h.s) AS ep_rec
FROM hq_hold h JOIN hq_p2kd k USING (dev, det) JOIN hq_hold_co c ON c.hid = h.hid
LEFT JOIN hq_p2rec r ON r.kind = 'e' AND r.id = h.hid WHERE h.dur >= 900 GROUP BY h.dev, h.det
),
hq_p2dis AS (
SELECT d.dev, d.det, coalesce(sum(greatest(least(h.f, d.drop_b1 * 300.0) - greatest(h.s, d.drop_b0 * 300.0), 0.0))
         FILTER (h.hid IS NOT NULL), 0)
       >= 0.5 * (d.drop_b1 - d.drop_b0) * 300.0 AS drop_in_stuck
FROM hq_p2drx d LEFT JOIN hq_hold h ON h.dev = d.dev AND h.det = d.det AND h.dur >= 900
WHERE d.n_on > 0 AND d.drop_lam >= 30 AND d.drop_b1 > d.drop_b0 GROUP BY d.dev, d.det, d.drop_b0, d.drop_b1
),
hq_p2nkd AS MATERIALIZED (
SELECT a.dev, a.det, a.ph, coalesce(r.nsame, 0) >= 2 AS isph
FROM hq_p2ph a LEFT JOIN (SELECT dev, det, count(*) FILTER (same) AS nsame FROM hq_p2rf f
                        WHERE NOT EXISTS (SELECT 1 FROM hq_ntraf t WHERE t.dev = f.dev AND t.det = f.r) GROUP BY dev, det) r
USING (dev, det)
),
hq_p2nrm AS MATERIALIZED (
SELECT f.dev, f.det, f.r FROM hq_p2rf f JOIN hq_p2nkd k USING (dev, det)
WHERE k.isph AND f.same AND NOT EXISTS (SELECT 1 FROM hq_ntraf t WHERE t.dev = f.dev AND t.det = f.r)
UNION ALL SELECT k.dev, k.det, d.det FROM hq_p2nkd k JOIN hq_dets d ON d.dev = k.dev AND d.det <> k.det
WHERE NOT k.isph AND NOT EXISTS (SELECT 1 FROM hq_ntraf t WHERE t.dev = d.dev AND t.det = d.det)
),
hq_p2rsum AS (
SELECT m.dev, m.det, sum(x.dy) AS rd, sum(x.nt) AS rn FROM hq_p2nrm m JOIN hq_dstat x ON x.dev = m.dev AND x.det = m.r
GROUP BY m.dev, m.det
),
hq_p2nt AS (
SELECT dev, det, nt AS night_n, ex AS night_exp, (nt + 0.5) / (ex + 0.5) AS night_ratio, who FROM (
  SELECT k.dev, k.det, s.nt, s.dy / greatest(coalesce(ps.dy, r.rd, 0), 1) * coalesce(ps.nt, r.rn, 0) AS ex,
         CASE WHEN pa.p IS NOT NULL THEN 'd' || pa.p || ', which it tracks'
              WHEN k.isph THEN 'the other detectors on its phase' ELSE 'the rest of the signal' END AS who
  FROM hq_p2nkd k JOIN hq_dstat s USING (dev, det) JOIN hq_sig0 g ON g.dev = k.dev
  LEFT JOIN (SELECT * FROM hq_p2part x WHERE NOT EXISTS (SELECT 1 FROM hq_ntraf t WHERE t.dev = x.id // 128 AND t.det = x.p)) pa ON pa.kind = 'n' AND pa.id = k.dev * 128 + k.det
  LEFT JOIN hq_dstat ps ON ps.dev = k.dev AND ps.det = pa.p LEFT JOIN hq_p2rsum r ON r.dev = k.dev AND r.det = k.det
  WHERE g.ok_night AND s.dy >= 20) WHERE ex >= 30
),
hq_p2st AS MATERIALIZED (
SELECT *, CASE WHEN n_on < 20 AND s_stuck_u = 0 THEN 'not_enough_data' WHEN hs_u < 0.25 THEN 'bad' WHEN hs_u < 0.70 THEN 'suspect'
               ELSE 'ok' END AS status_u, CASE WHEN hs < 0.25 THEN 'bad' WHEN hs < 0.70 THEN 'suspect' WHEN n_on < 20 THEN 'not_enough_data'
               ELSE 'ok' END AS status
FROM (SELECT *, (1 - coalesce(s_dropout, 0)) * (1 - s_stuck) * (1 - coalesce(s_night_drop, 0))
                * (1 - coalesce(s_level, 0)) AS hs, (1 - coalesce(s_dropout_u, 0)) * (1 - s_stuck_u) * (1 - coalesce(s_night_drop, 0))
                * (1 - coalesce(s_level, 0)) AS hs_u
  FROM (SELECT d.dev, d.det, d.n_on, d.drop_lam, d.drop_b0, d.drop_b1, d.drop_to_end,
          coalesce(i.drop_in_stuck, false) AS drop_in_stuck, rd.rec AS drop_rec, e.ep_dur, e.ep_t0, e.ep_t1, e.ep_co,
          e.ep_rec, n.night_ratio, n.night_n, n.night_exp, n.who AS night_ref, l.level_ratio,
          coalesce(l.level_llr, 0.0) AS level_llr, l.level_b,
          CASE WHEN d.n_on > 0 THEN (
            CASE WHEN coalesce(i.drop_in_stuck, false) THEN 0.0
                 WHEN (CASE WHEN (d.drop_lam) IS NULL OR isnan(d.drop_lam) THEN NULL WHEN (d.drop_lam) < 30.0 THEN 0.0 ELSE 0.35 + 0.65 * least(greatest(((d.drop_lam) - 30.0) / (100.0 - 30.0), 0.0), 1.0) END) > 0 AND NOT d.drop_to_end AND rd.rec BETWEEN 0.5 AND 2.0 THEN least((CASE WHEN (d.drop_lam) IS NULL OR isnan(d.drop_lam) THEN NULL WHEN (d.drop_lam) < 30.0 THEN 0.0 ELSE 0.35 + 0.65 * least(greatest(((d.drop_lam) - 30.0) / (100.0 - 30.0), 0.0), 1.0) END), 0.35)
                 ELSE (CASE WHEN (d.drop_lam) IS NULL OR isnan(d.drop_lam) THEN NULL WHEN (d.drop_lam) < 30.0 THEN 0.0 ELSE 0.35 + 0.65 * least(greatest(((d.drop_lam) - 30.0) / (100.0 - 30.0), 0.0), 1.0) END) END) END AS s_dropout,
          CASE WHEN coalesce(e.ep_dur, 0) >= 900 AND (e.ep_co >= 3 OR e.ep_rec BETWEEN 0.5 AND 2.0)
               THEN least((CASE WHEN (coalesce(e.ep_dur, 0)) IS NULL OR isnan(coalesce(e.ep_dur, 0)) THEN NULL WHEN (coalesce(e.ep_dur, 0)) < 900.0 THEN 0.0 ELSE 0.35 + 0.65 * least(greatest(((coalesce(e.ep_dur, 0)) - 900.0) / (3600.0 - 900.0), 0.0), 1.0) END), 0.35) ELSE (CASE WHEN (coalesce(e.ep_dur, 0)) IS NULL OR isnan(coalesce(e.ep_dur, 0)) THEN NULL WHEN (coalesce(e.ep_dur, 0)) < 900.0 THEN 0.0 ELSE 0.35 + 0.65 * least(greatest(((coalesce(e.ep_dur, 0)) - 900.0) / (3600.0 - 900.0), 0.0), 1.0) END) END AS s_stuck,
CASE WHEN d.n_on > 0 THEN CASE WHEN coalesce(i.drop_in_stuck, false) THEN 0.0
                  ELSE (CASE WHEN (d.drop_lam) IS NULL OR isnan(d.drop_lam) THEN NULL WHEN (d.drop_lam) < 30.0 THEN 0.0 ELSE 0.35 + 0.65 * least(greatest(((d.drop_lam) - 30.0) / (100.0 - 30.0), 0.0), 1.0) END) END END
            AS s_dropout_u,
          CASE WHEN coalesce(e.ep_dur, 0) >= 900 AND (e.ep_co >= 3 OR (e.ep_rec BETWEEN 0.5 AND 2.0 AND e.ep_dur < 3600))
               THEN least((CASE WHEN (coalesce(e.ep_dur, 0)) IS NULL OR isnan(coalesce(e.ep_dur, 0)) THEN NULL WHEN (coalesce(e.ep_dur, 0)) < 900.0 THEN 0.0 ELSE 0.35 + 0.65 * least(greatest(((coalesce(e.ep_dur, 0)) - 900.0) / (3600.0 - 900.0), 0.0), 1.0) END), 0.35) ELSE (CASE WHEN (coalesce(e.ep_dur, 0)) IS NULL OR isnan(coalesce(e.ep_dur, 0)) THEN NULL WHEN (coalesce(e.ep_dur, 0)) < 900.0 THEN 0.0 ELSE 0.35 + 0.65 * least(greatest(((coalesce(e.ep_dur, 0)) - 900.0) / (3600.0 - 900.0), 0.0), 1.0) END) END AS s_stuck_u,

          CASE WHEN n.night_ratio IS NOT NULL THEN
               (CASE WHEN (-ln(greatest(n.night_ratio, 1e-6))) IS NULL OR isnan(-ln(greatest(n.night_ratio, 1e-6))) THEN NULL WHEN (-ln(greatest(n.night_ratio, 1e-6))) < -ln(0.19) THEN 0.0 ELSE 0.35 + 0.65 * least(greatest(((-ln(greatest(n.night_ratio, 1e-6))) - -ln(0.19)) / (-ln(0.063) - -ln(0.19)), 0.0), 1.0) END) END AS s_night_drop,
          CASE WHEN l.level_ratio IS NOT NULL THEN CASE WHEN l.level_llr <= 50 THEN 0.0
               ELSE (CASE WHEN (-ln(l.level_ratio)) IS NULL OR isnan(-ln(l.level_ratio)) THEN NULL WHEN (-ln(l.level_ratio)) < -ln(0.15) THEN 0.0 ELSE 0.35 + 0.65 * least(greatest(((-ln(l.level_ratio)) - -ln(0.15)) / (-ln(0.05) - -ln(0.15)), 0.0), 1.0) END) END END AS s_level
        FROM hq_p2drx d LEFT JOIN hq_p2dis i USING (dev, det)
        LEFT JOIN hq_p2rec rd ON rd.kind = 'd' AND rd.id = d.dev * 128 + d.det
        LEFT JOIN hq_p2ep e USING (dev, det) LEFT JOIN hq_p2nt n USING (dev, det) LEFT JOIN hq_lvlshift l USING (dev, det)))
),
hq_stage1 AS MATERIALIZED (
SELECT x.*, a.status AS status1, a.hs AS score1 FROM (
  SELECT * FROM hq_p1st WHERE dev NOT IN (SELECT dev FROM hq_p2sig) OR status = 'bad'
  UNION ALL SELECT b.* FROM hq_p2st b JOIN hq_p1st a USING (dev, det) WHERE a.status <> 'bad') x
JOIN hq_p1st a USING (dev, det)
),
hq_silall AS MATERIALIZED (
SELECT a.* FROM hq_p1dra a JOIN hq_p1st s USING (dev, det) WHERE a.dev NOT IN (SELECT dev FROM hq_p2sig) OR s.status = 'bad'
UNION ALL SELECT b.* FROM hq_p2dra b JOIN hq_p1st s USING (dev, det) WHERE s.status <> 'bad'
),
hq_xbase AS MATERIALIZED (
SELECT s.*, c.fn, c.ph, c.lanes, w.hours, w.wg, w.bs, CASE WHEN c.lanes >= 2 THEN '2+' ELSE '1' END AS span,
       greatest(c.p_advance, c.p_presence, c.p_count, c.p_yellow_red, c.p_mid, c.p_bike, c.p_other) AS f_top,
       CASE WHEN greatest(c.p_advance, c.p_presence, c.p_count, c.p_yellow_red, c.p_mid, c.p_bike, c.p_other) < 0.7 THEN list_filter([CASE WHEN c.p_advance >= 0.15 THEN 'Advance' END, CASE WHEN c.p_presence >= 0.15 THEN 'Presence' END, CASE WHEN c.p_count >= 0.15 THEN 'Count' END, CASE WHEN c.p_yellow_red >= 0.15 THEN 'Yellow_Red' END, CASE WHEN c.p_mid >= 0.15 THEN 'Mid' END, CASE WHEN c.p_bike >= 0.15 THEN 'Bike' END, CASE WHEN c.p_other >= 0.15 THEN 'Other' END], lambda x: x IS NOT NULL) ELSE []::VARCHAR[] END AS alts
FROM hq_stage1 s JOIN hq_dcl c USING (dev, det) JOIN hq_win w USING (dev) WHERE c.fn IN ('Advance', 'Presence', 'Count', 'Yellow_Red', 'Mid', 'Bike', 'Other')
),
hq_dz AS MATERIALIZED (
SELECT d.dev, d.det, c.ph, c.fn, coalesce(c.fn IN ('Advance', 'Count'), false) AS tr,
       greatest(coalesce(c.lanes, 1.0), 1.0) AS ln1, c.lanes, s.status1
FROM hq_dets d LEFT JOIN hq_dcl c USING (dev, det) LEFT JOIN hq_stage1 s USING (dev, det)
),
hq_gb1 AS (
SELECT g.dev, g.det, g.k, CASE WHEN c.k IS NOT NULL THEN g.n END AS n,
       CASE WHEN c.k IS NOT NULL THEN g.occs / w.bs END AS occ, c.k IS NOT NULL AS cov
FROM (SELECT g.dev, g.det, g.b // (w.bs // 300) AS k, sum(g.n) AS n, sum(g.occ) AS occs
      FROM hq_g5 g JOIN hq_win w USING (dev) GROUP BY g.dev, g.det, g.b // (w.bs // 300)) g JOIN hq_win w USING (dev)
LEFT JOIN (SELECT DISTINCT c.dev, c.b // (w.bs // 300) AS k FROM hq_cov5 c JOIN hq_win w USING (dev)) c
  ON c.dev = g.dev AND c.k = g.k
),
hq_rmm AS MATERIALIZED (
SELECT dev, det, r FROM (
  SELECT a.dev, a.det, b.det AS r, coalesce(a.ph = b.ph, false) AND b.tr AS s1, b.tr AS s2,
         bool_or(coalesce(a.ph = b.ph, false) AND b.tr) OVER (PARTITION BY a.dev, a.det) AS h1,
         bool_or(b.tr) OVER (PARTITION BY a.dev, a.det) AS h2
  FROM hq_dz a JOIN hq_dz b ON a.dev = b.dev AND a.det <> b.det)
WHERE (h1 AND s1) OR (NOT h1 AND h2 AND s2) OR (NOT h1 AND NOT h2)
),
hq_gb AS MATERIALIZED (
SELECT g.*, CASE WHEN g.cov THEN r.ref END AS ref, CASE WHEN p.mocc > 0.01 THEN g.occ / p.mocc END AS rel, p.mocc,
       z.ph, z.fn, z.status1, (z.status1 IS DISTINCT FROM 'bad' AND g.occ IS NOT NULL) AS hl
FROM hq_gb1 g JOIN (SELECT dev, det, avg(occ) AS mocc FROM hq_gb1 GROUP BY dev, det) p USING (dev, det)
JOIN hq_dz z USING (dev, det)
LEFT JOIN (SELECT m.dev, m.det, g.k, sum(coalesce(g.n, 0)) AS ref FROM hq_rmm m JOIN hq_gb1 g ON g.dev = m.dev AND g.det = m.r
           GROUP BY m.dev, m.det, g.k) r USING (dev, det, k)
),
hq_gcong AS (
SELECT dev, det, k, CASE WHEN m > 0 THEN (L[(CASE WHEN lo < p THEN lo ELSE lo + 1 END) + 1]
                                          + L[(CASE WHEN h < p THEN h ELSE h + 1 END) + 1]) / 2.0 END AS cong
FROM (SELECT *, m // 2 AS h, CASE WHEN m % 2 = 1 THEN m // 2 ELSE m // 2 - 1 END AS lo FROM (
  SELECT g.dev, g.det, g.k, l.L, CASE WHEN g.rel IS NULL THEN len(l.L) ELSE len(list_filter(l.L, lambda x: x < g.rel)) END AS p,
         len(l.L) - (g.rel IS NOT NULL)::INT AS m
  FROM hq_gb g JOIN (SELECT dev, k, list_sort(list(rel) FILTER (rel IS NOT NULL)) AS L FROM hq_gb GROUP BY dev, k) l
  USING (dev, k) WHERE g.dev IN (SELECT dev FROM hq_dets GROUP BY dev HAVING count(*) >= 4)))
),
hq_gctx AS MATERIALIZED (
SELECT g.*, c.cong,
  CASE WHEN g.ph IS NOT NULL THEN (a.so - coalesce(CASE WHEN g.hl THEN g.occ END, 0)) / nullif(a.sh - g.hl::INT, 0)
       END AS peer,
  CASE WHEN g.ph IS NOT NULL AND g.fn IS NOT NULL THEN (b.so - coalesce(CASE WHEN g.hl THEN g.occ END, 0))
       / nullif(b.sh - g.hl::INT, 0) END AS same
FROM hq_gb g LEFT JOIN hq_gcong c USING (dev, det, k)
LEFT JOIN (SELECT dev, ph, k, sum(occ) FILTER (hl) AS so, count(*) FILTER (hl) AS sh FROM hq_gb WHERE ph IS NOT NULL
           GROUP BY dev, ph, k) a ON a.dev = g.dev AND a.ph = g.ph AND a.k = g.k
LEFT JOIN (SELECT dev, ph, fn, k, sum(occ) FILTER (hl) AS so, count(*) FILTER (hl) AS sh FROM hq_gb
           WHERE ph IS NOT NULL AND fn IS NOT NULL GROUP BY dev, ph, fn, k) b
  ON b.dev = g.dev AND b.ph = g.ph AND b.fn = g.fn AND b.k = g.k
),
hq_slp AS (
SELECT dev, det, CASE WHEN one >= 6 AND vx > 0.5 THEN bn END AS elhi_cnt,
       CASE WHEN one >= 6 AND vx > 0.5 THEN bo END AS elhi_occ
FROM (SELECT dev, det, count(*) AS one, regr_slope(lnn, lr) AS bn, regr_slope(lo, lr) AS bo,
             var_pop(lr) * count(*) AS vx
      FROM (SELECT g.dev, g.det, ln(g.ref + 1) AS lr, ln(g.n + 1) AS lnn, ln(g.occ + 0.005) AS lo
            FROM hq_gb g JOIN hq_win w USING (dev) JOIN (SELECT dev, det, median(ref) FILTER (n IS NOT NULL AND ref IS NOT NULL)
                                                   AS rmed FROM hq_gb GROUP BY dev, det) m USING (dev, det)
            WHERE w.wg <> 'm30' AND g.n IS NOT NULL AND g.ref IS NOT NULL AND g.ref >= m.rmed)
      GROUP BY dev, det)
),
hq_lkc AS (
SELECT dev, det, CASE WHEN c1 IS NOT NULL AND NOT isnan(c1) THEN c1 WHEN NOT isnan(c2) THEN c2 END AS c_occ_like
FROM (SELECT g.dev, g.det,
        CASE WHEN count(*) FILTER (g.occ IS NOT NULL AND g.same IS NOT NULL) >= 6
             THEN corr(g.same, g.occ) FILTER (g.occ IS NOT NULL AND g.same IS NOT NULL) END AS c1,
        CASE WHEN count(*) FILTER (g.occ IS NOT NULL AND g.peer IS NOT NULL) >= 6
             THEN corr(g.peer, g.occ) FILTER (g.occ IS NOT NULL AND g.peer IS NOT NULL) END AS c2
      FROM hq_gctx g JOIN hq_win w USING (dev) WHERE w.wg <> 'm30' GROUP BY g.dev, g.det)
),
hq_onmed AS MATERIALIZED (
SELECT dev, det, sum(n)::BIGINT AS np, (sum(n) FILTER (round(a / 1e6, 3) <= CAST(0.2 AS DOUBLE)))::BIGINT AS nps,
       (min(a) FILTER (cn > (nn - 1) // 2) + min(a) FILTER (cn > nn // 2)) / 2e6 AS med
FROM (SELECT dev, det, a, n, sum(n) OVER (PARTITION BY dev, det ORDER BY a ROWS UNBOUNDED PRECEDING) AS cn,
             sum(n) OVER (PARTITION BY dev, det) AS nn
      FROM (SELECT dev, det, a, sum(n) AS n FROM hq_rtko GROUP BY dev, det, a))
GROUP BY dev, det
),
hq_act AS MATERIALIZED (
SELECT a.dev, a.det, a.n_on_a, a.n_rep, a.n_rep / a.n_on_a AS rep_frac, a.n_start, a.max_on_s, a.rep_time_s,
       CASE WHEN m.np >= 5 THEN m.med END AS dur_p50, a.n_ge5, a.n_ge60, a.long_max_s
FROM (SELECT dev, det, sum(non)::BIGINT AS n_on_a, coalesce(sum(nrel), 0)::BIGINT AS n_rep, sum(nst)::BIGINT AS n_start,
             max(max_on) AS max_on_s, coalesce(sum(relog_ns), 0) / 1e9 AS rep_time_s,
             coalesce(sum(n5s), 0)::BIGINT AS n_ge5, coalesce(sum(n60s), 0)::BIGINT AS n_ge60, max(lmax) AS long_max_s
      FROM hq_d5 GROUP BY dev, det HAVING sum(non) > 0) a
LEFT JOIN hq_onmed m USING (dev, det)
),
hq_tflim AS MATERIALIZED (
SELECT dev, det, (CASE WHEN fn IN ('Count', 'Presence', 'Yellow_Red') THEN CAST(0.8 AS DOUBLE)
                       ELSE CAST(0.6 AS DOUBLE) END) / ln1 AS lim FROM hq_dz
),
hq_tf AS (
SELECT o.dev, o.det, o.N, coalesce(k.nf, 0)::BIGINT AS n_fast, 1000.0 * coalesce(k.nf, 0)::BIGINT / greatest(o.N, 1) AS fast_rate,
       z.lim AS gap_lim, CASE WHEN k.fx > 0 THEN coalesce(k.px, 0)::BIGINT / k.fx::BIGINT END AS pair_share
FROM (SELECT dev, det, sum(non)::BIGINT AS N FROM hq_d5 GROUP BY dev, det HAVING sum(non) > 0) o
JOIN hq_tflim z USING (dev, det)
LEFT JOIN (SELECT k.dev, k.det, sum(k.n) FILTER (k.ta < z.lim * 10) AS nf,
                  sum(k.n) FILTER (k.ta < z.lim * 10 AND k.tb < 9) AS fx,
                  sum(k.n) FILTER (k.ta < z.lim * 10 AND k.tb < 9 AND NOT k.tb < z.lim * 10) AS px
           FROM hq_tkg k JOIN hq_tflim z USING (dev, det) GROUP BY k.dev, k.det) k USING (dev, det)
),
hq_phs3 AS MATERIALIZED (
SELECT dev, ph FROM hq_p5 GROUP BY dev, ph HAVING sum(ngreen) >= 3
),
hq_sgyb AS MATERIALIZED (
SELECT p.dev, p.ph, p.b::BIGINT AS b, p.gy_ns / 1e9 AS sgy FROM hq_p5 p JOIN hq_phs3 USING (dev, ph) WHERE p.gy_ns IS NOT NULL
),
hq_q5b AS MATERIALIZED (
SELECT d.dev, d.det, d.b, d.nst AS n, CASE WHEN p.ph IS NOT NULL THEN coalesce(d.nst_gy, 0) ELSE 0 END::BIGINT AS ngy, z.ln1
FROM hq_d5 d JOIN hq_dz z USING (dev, det) LEFT JOIN hq_phs3 p ON p.dev = z.dev AND p.ph = z.ph WHERE d.nst > 0
),
hq_q5gyb AS MATERIALIZED (
SELECT z.dev, z.det, y.b, ((coalesce(q.ngy, 0)::FLOAT / y.sgy::FLOAT) * 3600::FLOAT)::DOUBLE / z.ln1 AS q5_gy
FROM hq_dz z JOIN hq_phs3 p ON p.dev = z.dev AND p.ph = z.ph JOIN hq_sgyb y ON y.dev = z.dev AND y.ph = z.ph
LEFT JOIN hq_q5b q ON q.dev = z.dev AND q.det = z.det AND q.b = y.b WHERE y.sgy >= 60
),
hq_q5 AS (
SELECT a.dev, a.det, g.q5_gy, a.q5_all FROM (SELECT dev, det, max(((n::FLOAT / 300::FLOAT) * 3600::FLOAT)::DOUBLE / ln1)
                                                AS q5_all FROM hq_q5b GROUP BY dev, det) a
LEFT JOIN (SELECT dev, det, max(q5_gy) AS q5_gy FROM hq_q5gyb GROUP BY dev, det) g USING (dev, det)
),
hq_fe15 AS (
SELECT d.dev, d.det, d.b // 3 AS g,
       (sum(d.nfst) - CASE WHEN any_value(p.ph) IS NOT NULL THEN coalesce(sum(d.nfst_r2), 0) ELSE 0 END)::BIGINT AS x,
       (sum(d.nst) - (coalesce(sum(d.nchat), 0) - CASE WHEN any_value(p.ph) IS NOT NULL THEN coalesce(sum(d.nchat_r2), 0) ELSE 0 END))::BIGINT AS n
FROM hq_d5 d JOIN hq_dz z USING (dev, det) LEFT JOIN hq_phs3 p ON p.dev = z.dev AND p.ph = z.ph WHERE d.nst > 0
GROUP BY d.dev, d.det, d.b // 3
),
hq_fe AS (
SELECT dev, det, count(*) AS fe_n, any_value(p) AS fe_p,
       CASE WHEN count(*) >= 2 AND any_value(p) > 0 AND any_value(p) < 1
            THEN sum((x - p * n) ^ 2 / (p * (1 - p) * n)) / greatest(count(*) - 1, 1) END AS fe_phi
FROM (SELECT *, sum(x) OVER (PARTITION BY dev, det) / sum(n) OVER (PARTITION BY dev, det) AS p FROM hq_fe15 WHERE n > 0)
GROUP BY dev, det
),
hq_xm AS MATERIALIZED (
SELECT x.*, CASE WHEN x.fn <> 'Count' THEN '' WHEN a.dur_p50 IS NULL THEN 'unknown' WHEN a.dur_p50 <= 0.25 THEN 'pulse'
                 ELSE 'normal' END AS cmode
FROM hq_xbase x LEFT JOIN hq_act a USING (dev, det)
),
hq_ohr AS MATERIALIZED (
SELECT h.*, CASE WHEN h.xfn = 'Count' AND h.occ > greatest(l.lp, h.n * t.tp / h.bs, 0.20) AND h.cong < 1.5
                  AND NOT coalesce(h.pocc >= 0.40, false) AND h.tl <= 1 THEN true ELSE false END AS hic
FROM (SELECT g.dev, g.det, g.k, g.n, g.occ, g.cong, x.ph AS xph, x.fn AS xfn, w.bs,
             CASE WHEN g.ph IS NOT NULL THEN (a.so - coalesce(g.occ, 0)) / CASE WHEN a.co > 1 THEN a.co - 1 END END AS pocc,
             CASE WHEN m.rmax > 0 THEN least(greatest(floor(5 * g.ref / m.rmax), 0), 4) END AS tl
      FROM hq_gctx g JOIN hq_xm x USING (dev, det) JOIN hq_win w USING (dev)
      JOIN (SELECT dev, det, max(ref) AS rmax FROM hq_gb GROUP BY dev, det) m USING (dev, det)
      LEFT JOIN (SELECT g.dev, g.ph, g.k, sum(g.occ) AS so, count(g.occ) AS co FROM hq_gb g JOIN hq_xm x USING (dev, det)
                 WHERE x.cmode <> 'unknown' AND g.ph IS NOT NULL GROUP BY g.dev, g.ph, g.k) a
        ON a.dev = g.dev AND a.ph = g.ph AND a.k = g.k
      WHERE x.cmode <> 'unknown') h
LEFT JOIN (SELECT * FROM (VALUES (CAST(300.0 AS DOUBLE), CAST(0.0 AS DOUBLE), CAST(0.03160000031789146 AS DOUBLE)), (CAST(300.0 AS DOUBLE), CAST(1.0 AS DOUBLE), CAST(0.03391665379206339 AS DOUBLE)), (CAST(300.0 AS DOUBLE), CAST(2.0 AS DOUBLE), CAST(0.02933332125345866 AS DOUBLE)), (CAST(300.0 AS DOUBLE), CAST(3.0 AS DOUBLE), CAST(0.04465337702433239 AS DOUBLE)), (CAST(300.0 AS DOUBLE), CAST(4.0 AS DOUBLE), CAST(0.054898310852050686 AS DOUBLE)), (CAST(900.0 AS DOUBLE), CAST(0.0 AS DOUBLE), CAST(0.010222217771742079 AS DOUBLE)), (CAST(900.0 AS DOUBLE), CAST(1.0 AS DOUBLE), CAST(0.022888895670572916 AS DOUBLE)), (CAST(900.0 AS DOUBLE), CAST(2.0 AS DOUBLE), CAST(0.0360000560760498 AS DOUBLE)), (CAST(900.0 AS DOUBLE), CAST(3.0 AS DOUBLE), CAST(0.06100013715955946 AS DOUBLE)), (CAST(900.0 AS DOUBLE), CAST(4.0 AS DOUBLE), CAST(0.12279038043552033 AS DOUBLE))) t(bs, tl, lp)) l ON l.bs = h.bs AND l.tl = h.tl LEFT JOIN (SELECT * FROM (VALUES (CAST(300.0 AS DOUBLE), CAST(3.6639998092651394 AS DOUBLE)), (CAST(900.0 AS DOUBLE), CAST(3.9841893535413244 AS DOUBLE))) t(bs, tp)) t ON t.bs = h.bs
),
hq_hicd AS (
SELECT DISTINCT dev, det, xph AS ph FROM hq_ohr WHERE hic AND xph IS NOT NULL
),
hq_gonb AS (
SELECT h.dev, h.det, g.k::BIGINT AS k, o.gon, g.gav
FROM hq_hicd h JOIN (SELECT p.dev, p.ph, p.b // (w.bs // 300) AS k, sum(p.g5_ns) / 1e9 AS gav FROM hq_p5 p JOIN hq_win w USING (dev)
                  WHERE p.g5_ns IS NOT NULL GROUP BY p.dev, p.ph, p.b // (w.bs // 300)) g ON g.dev = h.dev AND g.ph = h.ph
LEFT JOIN (SELECT d.dev, d.det, d.b // (w.bs // 300) AS k, sum(d.ong5_ns) / 1e9 AS gon FROM hq_d5 d JOIN hq_win w USING (dev)
           WHERE d.ong5_ns IS NOT NULL GROUP BY d.dev, d.det, d.b // (w.bs // 300)) o
  ON o.dev = h.dev AND o.det = h.det AND o.k = g.k
),
hq_hib AS MATERIALIZED (
SELECT h.dev, h.det, h.k, h.occ FROM hq_ohr h JOIN hq_gonb g USING (dev, det, k)
WHERE h.hic AND h.xph IS NOT NULL AND g.gav >= 60 AND coalesce(g.gon, 0) >= 0.20 * g.gav
),
hq_ohi AS (
SELECT o.dev, o.det, count(h.k) * any_value(o.bs) / 60.0 AS hi_min_c, max(h.occ) AS hi_occ_max
FROM (SELECT DISTINCT dev, det, bs FROM hq_ohr) o LEFT JOIN hq_hib h USING (dev, det) GROUP BY o.dev, o.det
),
hq_hix AS MATERIALIZED (
SELECT dev, hid, hid - min(hid) OVER (PARTITION BY dev) AS j, hid % 24 AS hod, nb_ = 12 AND nc = 12 AS ch,
       count(*) OVER (PARTITION BY dev) AS H
FROM (SELECT b.dev, (w.s_ns + b.b * 300000000000) // 3600000000000 AS hid, count(*) AS nb_, count(*) FILTER (b.cov) AS nc
      FROM hq_bins5 b JOIN hq_win w USING (dev) WHERE w.hours >= CAST(3.0 AS DOUBLE) GROUP BY b.dev, hid)
),
hq_mh AS MATERIALIZED (
SELECT g.dev, g.det, (w.s_ns + g.b * 300000000000) // 3600000000000 AS hid, sum(g.n) AS M, sum(g.occ) AS O
FROM hq_g5 g JOIN hq_win w USING (dev) WHERE w.hours >= CAST(3.0 AS DOUBLE) GROUP BY g.dev, g.det, hid
),
hq_yk AS MATERIALIZED (
SELECT k.dev, k.det, pa.p, k.isph FROM hq_p1kd k LEFT JOIN hq_p1part pa ON pa.kind = 'n' AND pa.id = k.dev * 128 + k.det
),
hq_gh AS MATERIALIZED (
SELECT m.dev, m.det, x.j, x.hod, x.ch, x.H, m.M, m.O, y.p IS NOT NULL OR y.isph AS has_m,
       CASE WHEN y.p IS NULL AND NOT y.isph THEN s.SM - m.M ELSE coalesce(r.RM, 0) END AS R, coalesce(r.RO, 0) AS Om
FROM hq_mh m JOIN hq_hix x USING (dev, hid) JOIN hq_yk y USING (dev, det)
JOIN (SELECT dev, hid, sum(M) AS SM FROM hq_mh GROUP BY dev, hid) s USING (dev, hid)
LEFT JOIN (SELECT a.dev, a.det, b.hid, sum(b.M) AS RM, sum(b.O) AS RO
           FROM (SELECT dev, det, p AS r FROM hq_yk WHERE p IS NOT NULL
                 UNION ALL SELECT f.dev, f.det, f.r FROM hq_p1pref f JOIN hq_yk y USING (dev, det) WHERE y.p IS NULL) a
           JOIN hq_mh b ON b.dev = a.dev AND b.det = a.r GROUP BY a.dev, a.det, b.hid) r USING (dev, det, hid)
),
hq_gh1 AS MATERIALIZED (
SELECT dev, det, coalesce(sum(M) FILTER (ch AND hod >= 7 AND hod < 19), 0) >= 20 AND coalesce(sum(R) FILTER (ch AND hod >= 7 AND hod < 19), 0) > 0 AS elig,
       CASE WHEN count(*) FILTER (ch AND hod >= 7 AND hod < 19 AND R >= 30) >= 4 THEN median(M / R) FILTER (ch AND hod >= 7 AND hod < 19 AND R >= 30) ELSE coalesce(sum(M) FILTER (ch AND hod >= 7 AND hod < 19), 0) / greatest(coalesce(sum(R) FILTER (ch AND hod >= 7 AND hod < 19), 0), 1) END AS s, count(*) FILTER (ch AND (hod >= 21 OR hod < 5)) AS nn
FROM hq_gh GROUP BY dev, det
),
hq_gh2 AS MATERIALIZED (
SELECT g.*, a.s, a.elig, CASE WHEN g.ch THEN (CASE WHEN (g.M) < (a.s * g.R) THEN 1 ELSE -1 END * sqrt(2 * ((g.M) - (a.s * g.R)) ^ 2 / greatest((g.M) + (a.s * g.R), 1e-9))) ELSE 0.0 END AS sg,
       sum(CASE WHEN g.ch THEN (CASE WHEN (g.M) < (a.s * g.R) THEN 1 ELSE -1 END * sqrt(2 * ((g.M) - (a.s * g.R)) ^ 2 / greatest((g.M) + (a.s * g.R), 1e-9))) ELSE 0.0 END) OVER w AS cs, sum(g.ch::INT) OVER w AS cc,
       sum(CASE WHEN g.ch THEN (CASE WHEN (g.M) < (a.s * g.R) THEN 1 ELSE -1 END * sqrt(2 * ((g.M) - (a.s * g.R)) ^ 2 / greatest((g.M) + (a.s * g.R), 1e-9))) ELSE 0.0 END) OVER (PARTITION BY g.dev, g.det) AS cs_t,
       sum(g.ch::INT) OVER (PARTITION BY g.dev, g.det) AS cc_t,
       sum(CASE WHEN g.ch AND g.R >= 30 THEN g.M ELSE 0.0 END) OVER w AS cm, sum(CASE WHEN g.ch AND g.R >= 30 THEN g.R ELSE 0.0 END) OVER w AS cr,
       sum(CASE WHEN g.ch AND g.R >= 30 THEN g.M ELSE 0.0 END) OVER (PARTITION BY g.dev, g.det) AS cm_t,
       sum(CASE WHEN g.ch AND g.R >= 30 THEN g.R ELSE 0.0 END) OVER (PARTITION BY g.dev, g.det) AS cr_t
FROM hq_gh g JOIN hq_gh1 a USING (dev, det) WINDOW w AS (PARTITION BY g.dev, g.det ORDER BY g.j ROWS UNBOUNDED PRECEDING)
),
hq_ghk AS (
SELECT a.dev, a.det, coalesce(k.k, 1) AS k, a.elig AND k.k IS NOT NULL AS has_k FROM hq_gh1 a LEFT JOIN (
  SELECT dev, det, first(j + 1 ORDER BY stat DESC, j) AS k FROM (
    SELECT dev, det, j, ((cs_t - cs) / (cc_t - cc) - cs / cc) * sqrt(cc * (cc_t - cc) / cc_t) AS stat
    FROM hq_gh2 WHERE j <= H - 2 AND cc >= 3 AND cc_t - cc >= 3) GROUP BY dev, det) k USING (dev, det)
),
hq_ghk2 AS (
SELECT dev, det, first(j + 1 ORDER BY stat DESC, j) AS k2 FROM (
  SELECT g.dev, g.det, g.j, ((g.cs - b.cs) / (g.cc - b.cc) - (g.cs_t - g.cs) / (g.cc_t - g.cc))
         * sqrt((g.cc - b.cc) * (g.cc_t - g.cc) / (g.cc_t - b.cc)) AS stat,
         (g.cs - b.cs) / (g.cc - b.cc) AS m_in, (g.cs_t - g.cs) / (g.cc_t - g.cc) AS m_rec
  FROM hq_gh2 g JOIN hq_ghk k USING (dev, det) JOIN hq_gh2 b ON b.dev = g.dev AND b.det = g.det AND b.j = k.k - 1
  WHERE k.has_k AND g.j >= k.k AND g.j <= g.H - 2 AND g.cc - b.cc >= 3 AND g.cc_t - g.cc >= 3 AND b.cr > 0
    AND (g.cr_t - g.cr) * b.cm / b.cr >= 50 AND g.cm_t - g.cm >= 0.6 * (g.cr_t - g.cr) * b.cm / b.cr)
WHERE m_in > 0 AND m_rec <= 0.35 * m_in AND m_rec < CAST(5.0 AS DOUBLE) GROUP BY dev, det
),
hq_gh3 AS (
SELECT g.dev, g.det, k.k, k.has_k, r.k2, g.j < k.k AND g.ch AS pre,
       g.j >= k.k AND (r.k2 IS NULL OR g.j < r.k2) AND g.ch AS post, g.j >= k.k AND g.ch AS post0, g.M, g.R, g.O,
       g.Om, g.ch, g.sg, g.s, g.hod, g.H, g.has_m, g.elig
FROM hq_gh2 g JOIN hq_ghk k USING (dev, det) LEFT JOIN hq_ghk2 r USING (dev, det)
),
hq_gh4 AS (
SELECT dev, det, any_value(k) AS k, any_value(has_k) AS has_k, any_value(k2) AS k2, any_value(H) AS H,
       any_value(has_m) AS has_m,
       any_value(elig) AS elig, any_value(s) AS s, CASE WHEN count(*) FILTER (pre AND R >= 30) >= 4 THEN median(M / R) FILTER (pre AND R >= 30) ELSE coalesce(sum(M) FILTER (pre), 0) / greatest(coalesce(sum(R) FILTER (pre), 0), 1) END AS s2,
       count(*) FILTER (pre AND R >= 30) >= 3 AS re, greatest(count(*) FILTER (pre), 1) AS npre,
       greatest(count(*) FILTER (post), 1) AS npost, coalesce(sum(M) FILTER (pre), 0) AS Mpre,
       coalesce(sum(M) FILTER (post), 0) AS Mpost, coalesce(sum(R) FILTER (post), 0) AS Rpost,
       coalesce(sum(O) FILTER (pre), 0) AS Opre, coalesce(sum(O) FILTER (post), 0) AS Opost,
       coalesce(sum(Om) FILTER (pre), 0) AS Ompre, coalesce(sum(Om) FILTER (post), 0) AS Ompost,
       count(*) FILTER (ch AND (hod >= 21 OR hod < 5)) AS nn, coalesce(sum(M) FILTER (ch AND (hod >= 21 OR hod < 5)), 0) AS Mn, coalesce(sum(s * R) FILTER (ch AND (hod >= 21 OR hod < 5)), 0) AS Cn
FROM hq_gh3 GROUP BY dev, det
),
hq_ykn AS MATERIALIZED (
SELECT k.dev, k.det, pa.p, k.isph FROM hq_p1nkd k
LEFT JOIN (SELECT * FROM hq_p1part x WHERE NOT EXISTS (SELECT 1 FROM hq_ntraf t WHERE t.dev = x.id // 128 AND t.det = x.p)) pa
  ON pa.kind = 'n' AND pa.id = k.dev * 128 + k.det
),
hq_ykaff AS MATERIALIZED (
SELECT DISTINCT y.dev, y.det FROM hq_yk y JOIN hq_ntraf t ON t.dev = y.dev AND t.det <> y.det
WHERE t.det = y.p OR (y.p IS NULL AND NOT y.isph)
   OR (y.p IS NULL AND y.isph AND EXISTS (SELECT 1 FROM hq_p1pref f WHERE f.dev = y.dev AND f.det = y.det AND f.r = t.det))
),
hq_ghn AS MATERIALIZED (
SELECT m.dev, m.det, x.hod, x.ch, m.M,
       CASE WHEN y.p IS NULL AND NOT y.isph THEN s.SM - CASE WHEN t.det IS NULL THEN m.M ELSE 0 END
            ELSE coalesce(r.RM, 0) END AS R
FROM hq_mh m JOIN hq_ykaff USING (dev, det) JOIN hq_hix x USING (dev, hid) JOIN hq_ykn y USING (dev, det)
LEFT JOIN hq_ntraf t USING (dev, det)
JOIN (SELECT m.dev, m.hid, sum(m.M) FILTER (t.det IS NULL) AS SM FROM hq_mh m LEFT JOIN hq_ntraf t USING (dev, det)
      GROUP BY m.dev, m.hid) s USING (dev, hid)
LEFT JOIN (SELECT a.dev, a.det, b.hid, sum(b.M) AS RM
           FROM (SELECT dev, det, p AS r FROM hq_ykn WHERE p IS NOT NULL
                 UNION ALL SELECT f.dev, f.det, f.r FROM hq_p1nrm f JOIN hq_ykn y USING (dev, det) WHERE y.p IS NULL AND y.isph) a
           JOIN hq_mh b ON b.dev = a.dev AND b.det = a.r GROUP BY a.dev, a.det, b.hid) r USING (dev, det, hid)
),
hq_ghnn AS (
SELECT g.dev, g.det, a.elig, a.nn, coalesce(sum(g.M) FILTER (ch AND (hod >= 21 OR hod < 5)), 0) AS Mn, coalesce(sum(a.s * g.R) FILTER (ch AND (hod >= 21 OR hod < 5)), 0) AS Cn
FROM hq_ghn g JOIN (SELECT dev, det, coalesce(sum(M) FILTER (ch AND hod >= 7 AND hod < 19), 0) >= 20 AND coalesce(sum(R) FILTER (ch AND hod >= 7 AND hod < 19), 0) > 0
                        AS elig, CASE WHEN count(*) FILTER (ch AND hod >= 7 AND hod < 19 AND R >= 30) >= 4 THEN median(M / R) FILTER (ch AND hod >= 7 AND hod < 19 AND R >= 30) ELSE coalesce(sum(M) FILTER (ch AND hod >= 7 AND hod < 19), 0) / greatest(coalesce(sum(R) FILTER (ch AND hod >= 7 AND hod < 19), 0), 1) END AS s, count(*) FILTER (ch AND (hod >= 21 OR hod < 5)) AS nn FROM hq_ghn GROUP BY dev, det) a
  USING (dev, det)
GROUP BY g.dev, g.det, a.elig, a.nn
),
hq_geh AS (
SELECT a.dev, a.det, a.elig AS geh_ok,
       CASE WHEN n.det IS NOT NULL THEN CASE WHEN n.elig AND n.nn >= 4 THEN (CASE WHEN (n.Mn / greatest(n.nn, 1)) < (n.Cn / greatest(n.nn, 1)) THEN 1 ELSE -1 END * sqrt(2 * ((n.Mn / greatest(n.nn, 1)) - (n.Cn / greatest(n.nn, 1))) ^ 2 / greatest((n.Mn / greatest(n.nn, 1)) + (n.Cn / greatest(n.nn, 1)), 1e-9))) END ELSE CASE WHEN a.elig AND a.nn >= 4 THEN (CASE WHEN (a.Mn / greatest(a.nn, 1)) < (a.Cn / greatest(a.nn, 1)) THEN 1 ELSE -1 END * sqrt(2 * ((a.Mn / greatest(a.nn, 1)) - (a.Cn / greatest(a.nn, 1))) ^ 2 / greatest((a.Mn / greatest(a.nn, 1)) + (a.Cn / greatest(a.nn, 1)), 1e-9))) END END AS geh_night,
       CASE WHEN n.det IS NOT NULL THEN n.Mn ELSE a.Mn END AS geh_night_n0, CASE WHEN n.det IS NOT NULL THEN n.Cn ELSE a.Cn END AS geh_night_exp0,
       CASE WHEN a.has_k AND b.med > 0 THEN b.med END AS geh_drop,
       CASE WHEN a.has_k THEN make_timestamp(((x.hid0 + least(a.k, a.H - 1)) * 3600000000000) // 1000) END AS drop_t,
       CASE WHEN a.has_k AND a.k2 IS NOT NULL THEN make_timestamp(((x.hid0 + a.k2) * 3600000000000) // 1000) END AS drop_t2,
       a.Opost / a.npost / 3600.0 AS occ_after,
       a.Mpre / a.npre AS own_before, a.Mpost / a.npost AS own_after,
       CASE WHEN a.re THEN a.s2 ELSE a.s END * a.Rpost / a.npost AS exp_after,
       CASE WHEN a.has_k THEN (a.Opost / a.npost) / greatest(a.Opre / a.npre, 1e-6) END AS occ_own,
       CASE WHEN a.has_k AND a.has_m THEN (a.Ompost / a.npost) / greatest(a.Ompre / a.npre, 1e-6) END AS occ_mates
FROM hq_gh4 a JOIN (SELECT dev, min(hid) AS hid0 FROM hq_hix GROUP BY dev) x USING (dev)
LEFT JOIN hq_ghnn n USING (dev, det)
-- note 154h: the severity (geh_drop) over every hour after the drop, as 1.2.0; its end bounds only the period / text
LEFT JOIN (SELECT g.dev, g.det, median(CASE WHEN a.re THEN CASE WHEN g.ch THEN (CASE WHEN (g.M) < (a.s2 * g.R) THEN 1 ELSE -1 END * sqrt(2 * ((g.M) - (a.s2 * g.R)) ^ 2 / greatest((g.M) + (a.s2 * g.R), 1e-9))) ELSE 0.0 END
                                           ELSE g.sg END) FILTER (g.post0) AS med
           FROM hq_gh3 g JOIN hq_gh4 a USING (dev, det) GROUP BY g.dev, g.det) b USING (dev, det)
),
hq_lvr AS MATERIALIZED (
SELECT * FROM (
WITH h AS (SELECT g.dev, g.det, g.j, g.ch AND CASE WHEN a.re THEN a.s2 ELSE a.s END * g.R >= 10 AS judged,
                  g.ch AND CASE WHEN a.re THEN a.s2 ELSE a.s END * g.R >= 10
                  AND g.M <= 0.5 * CASE WHEN a.re THEN a.s2 ELSE a.s END * g.R AS low
           FROM hq_gh2 g JOIN hq_gh4 a USING (dev, det)),
r AS (SELECT *, sum((judged AND NOT low)::INT) OVER (PARTITION BY dev, det ORDER BY j) AS grp FROM h)
SELECT dev, det, min(j) FILTER (low) AS ja, max(j) FILTER (low) + 1 AS jz FROM r
GROUP BY dev, det, grp HAVING count(*) FILTER (low) >= 2)
),
hq_o10 AS MATERIALIZED (
SELECT g.dev, g.det, g.b // 2 AS t, sum(g.n) AS own, bool_and(g.cov) AS c
FROM hq_g5 g JOIN hq_win w USING (dev)
WHERE w.hours >= CAST(3.0 AS DOUBLE) AND w.nb5 // 2 >= 12 AND g.b < w.nb5 // 2 * 2
GROUP BY g.dev, g.det, g.b // 2
),
hq_o10l AS (
SELECT dev, det, t, own, c, m,
       list(CASE WHEN c THEN own END) OVER wl AS ol, list(CASE WHEN c THEN own END) OVER wr AS orr,
       list(CASE WHEN c THEN m END) OVER wl AS ml, list(CASE WHEN c THEN m END) OVER wr AS mr
FROM (SELECT o.*, coalesce(m.m, 0.0) AS m FROM hq_o10 o LEFT JOIN (
        SELECT f.dev, f.det, o.t, sum(o.own) AS m FROM hq_p1rf f JOIN hq_o10 o ON o.dev = f.dev AND o.det = f.r
        WHERE f.same GROUP BY f.dev, f.det, o.t) m USING (dev, det, t))
WINDOW wl AS (PARTITION BY dev, det ORDER BY t ROWS BETWEEN 6 PRECEDING AND 2 PRECEDING),
       wr AS (PARTITION BY dev, det ORDER BY t ROWS BETWEEN 2 FOLLOWING AND 6 FOLLOWING)
),
hq_o10s AS MATERIALIZED (
SELECT *, b IS NOT NULL AND c AND own / ex >= CAST(3.0 AS DOUBLE) AND g >= CAST(5.0 AS DOUBLE) AND own > ex AS sp FROM (
  SELECT *, sqrt(2 * (own - ex) ^ 2 / greatest(own + ex, 1e-9)) AS g FROM (
    SELECT *, greatest(b, 1.0) * greatest(CASE WHEN m >= CAST(10.0 AS DOUBLE) AND bm IS NOT NULL
                                               THEN m / greatest(bm, 1.0) ELSE 1.0 END, 1.0) AS ex FROM (
      SELECT dev, det, t, own, c, m, CASE WHEN len(list_filter(coalesce(ol, []) || coalesce(orr, []), lambda x: x IS NOT NULL)) >= 3 THEN list_median(list_filter(coalesce(ol, []) || coalesce(orr, []), lambda x: x IS NOT NULL)) END AS b, CASE WHEN len(list_filter(coalesce(ml, []) || coalesce(mr, []), lambda x: x IS NOT NULL)) >= 3 THEN list_median(list_filter(coalesce(ml, []) || coalesce(mr, []), lambda x: x IS NOT NULL)) END AS bm FROM hq_o10l)))
),
hq_spk_runs AS MATERIALIZED (
SELECT dev, det, min(t) AS a, max(t) + 1 AS z FROM (
  SELECT dev, det, t, t - row_number() OVER (PARTITION BY dev, det ORDER BY t) AS grp FROM hq_o10s WHERE sp)
GROUP BY dev, det, grp
),
hq_spk AS (
SELECT s.*, CASE WHEN spk_daily < CAST(200.0 AS DOUBLE) THEN 0
                 WHEN spk_daily < CAST(500.0 AS DOUBLE) AND lv = 2 AND spk_gmax < CAST(10.0 AS DOUBLE) THEN 1 ELSE lv END
            AS spk_lvl
FROM (SELECT s.*, CASE WHEN spk_bursts >= CAST(3.0 AS DOUBLE) THEN 2 WHEN spk_bursts >= CAST(2.0 AS DOUBLE) THEN 1 ELSE 0 END AS lv
      FROM (SELECT o.dev, o.det, coalesce(r.nr, 0) / greatest(w.hours / 24.0, 1.0) AS spk_bursts,
                   coalesce(max(o.g) FILTER (o.sp), 0.0) AS spk_gmax, sum(o.own) * 24.0 / greatest(w.hours, 1e-9) AS spk_daily
            FROM hq_o10s o JOIN hq_win w USING (dev)
            LEFT JOIN (SELECT dev, det, count(*) AS nr FROM hq_spk_runs GROUP BY dev, det) r USING (dev, det)
            GROUP BY o.dev, o.det, r.nr, w.hours) s) s
),
hq_hpair AS (
SELECT a.hid, b.det AS dj, abs(b.s - a.s) <= CAST(5.0 AS DOUBLE) AND abs(b.f - a.f) <= CAST(5.0 AS DOUBLE) AS near
FROM hq_hold a JOIN hq_hold b ON a.dev = b.dev AND a.det <> b.det AND least(a.f, b.f) > greatest(a.s, b.s)
LEFT JOIN hq_twin t ON t.dev = a.dev AND t.i = a.det AND t.j = b.det WHERE t.i IS NULL
),
hq_hcom AS MATERIALIZED (
SELECT h.hid, coalesce(p.anyov, false) AS n_ov_any, coalesce(p.anyov AND p.allnear, false) AS common, p.mates
FROM hq_hold h LEFT JOIN (SELECT hid, true AS anyov, bool_and(near) AS allnear, list(DISTINCT dj ORDER BY dj) AS mates
                       FROM hq_hpair GROUP BY hid) p USING (hid)
),
hq_hb AS MATERIALIZED (
SELECT h.hid, h.dev, h.det, h.s, h.f, h.dur, x.ph AS p, w.bs, w.nbb, floor(h.s / w.bs)::INT AS ia,
       ceil(h.f / w.bs)::INT AS iz, ceil(h.s / w.bs)::INT AS fa, floor(h.f / w.bs)::INT AS fz
FROM hq_hold h JOIN hq_win w USING (dev) LEFT JOIN hq_xbase x USING (dev, det)
),
hq_hmate AS MATERIALIZED (
SELECT h.hid, z.det AS r FROM hq_hb h JOIN hq_dz z ON z.dev = h.dev AND z.ph = h.p AND z.det <> h.det
WHERE z.status1 IS DISTINCT FROM 'bad' AND NOT (least(h.iz, h.nbb) > h.ia AND (
  SELECT count(*) FILTER (g.occ >= 0.99) FROM hq_gb g WHERE g.dev = h.dev AND g.det = z.det
  AND g.k >= h.ia AND g.k < least(h.iz, h.nbb)) / (least(h.iz, h.nbb) - h.ia) >= 0.8)
),
hq_hpm AS (
SELECT m.hid, g.k, avg(g.occ) AS pm, bool_or(g.occ >= t.qt) AS hit
FROM hq_hmate m JOIN hq_hb h USING (hid) JOIN hq_gb g ON g.dev = h.dev AND g.det = m.r
LEFT JOIN hq_xm x ON x.dev = h.dev AND x.det = m.r LEFT JOIN (SELECT * FROM (VALUES ('Advance 1', CAST(0.3038888888888889 AS DOUBLE)), ('Advance 2+', CAST(0.3742222086588542 AS DOUBLE)), ('Bike 1', CAST(0.0377222230699327 AS DOUBLE)), ('Bike 2+', CAST(0.0866666624281141 AS DOUBLE)), ('Count 1', CAST(0.0603333320617675 AS DOUBLE)), ('Count 2+', CAST(0.1705444191826712 AS DOUBLE)), ('Mid 1', CAST(0.3430610639784066 AS DOUBLE)), ('Mid 2+', CAST(0.4015555063883463 AS DOUBLE)), ('Other 1', CAST(0.5927777777777777 AS DOUBLE)), ('Other 2+', CAST(0.6406667073567708 AS DOUBLE)), ('Presence 1', CAST(0.6725556098090277 AS DOUBLE)), ('Presence 2+', CAST(0.7850777825249563 AS DOUBLE)), ('Yellow_Red 1', CAST(0.034333332909478 AS DOUBLE)), ('Yellow_Red 2+', CAST(0.1325550011528863 AS DOUBLE))) t(type, qt)) t ON t.type = x.fn || ' ' || x.span
GROUP BY m.hid, g.k
),
hq_htr AS (
SELECT h.hid, g.k, sum(coalesce(g.n, 0)) AS tot FROM hq_hb h
JOIN (SELECT h.hid, z.det AS r, bool_or(z.tr) OVER (PARTITION BY h.hid) AS anytr, z.tr
      FROM hq_hb h JOIN hq_dz z ON z.dev = h.dev AND z.ph = h.p AND z.det <> h.det) m ON m.hid = h.hid AND (m.tr OR NOT m.anytr)
JOIN hq_gb g ON g.dev = h.dev AND g.det = m.r GROUP BY h.hid, g.k
),
hq_hrh AS (
SELECT h.hid, bool_or(g.occ >= 0.95) AS ref_held_e FROM hq_hb h JOIN hq_rmm m ON m.dev = h.dev AND m.det = h.det
JOIN hq_gb g ON g.dev = h.dev AND g.det = m.r AND g.k >= h.ia AND g.k < least(h.iz, h.nbb) GROUP BY h.hid
),
hq_ectx AS MATERIALIZED (
SELECT h.hid, h.dev, h.det, h.s, h.f, h.dur, h.p,
  coalesce((SELECT count(*) FROM hq_hmate m WHERE m.hid = h.hid), 0) AS n_hpeer_e, b.phx_h_e, b.corr_h_e, b.cover_e,
  b.refx_e, b.light_e, b.busy_min_e, b.busy_occ_e, coalesce(r.ref_held_e, false) AS ref_held_e, t.trafx_e, c.co AS co5,
  m.n_ov_any, m.common, m.mates
FROM hq_hb h JOIN hq_hold_co c USING (hid) JOIN hq_hcom m USING (hid) LEFT JOIN hq_hrh r USING (hid)
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
  FROM hq_hb h JOIN hq_gctx g ON g.dev = h.dev AND g.det = h.det
  JOIN (SELECT h.hid, avg(g.ref) AS rm, avg(g.peer) AS mpo, avg(g.ref) FILTER (g.k >= h.ia AND g.k < h.iz) AS rh
        FROM hq_hb h JOIN hq_gctx g ON g.dev = h.dev AND g.det = h.det GROUP BY h.hid) mr ON mr.hid = h.hid
  LEFT JOIN hq_hpm p ON p.hid = h.hid AND p.k = g.k
  LEFT JOIN (SELECT DISTINCT hid FROM hq_hmate) np ON np.hid = h.hid
  GROUP BY h.hid) b USING (hid)
LEFT JOIN (
  SELECT h.hid, CASE WHEN count(*) FILTER (t.k >= greatest(h.ja, 0) AND t.k < least(h.jz, h.nbb)) > 0 THEN
    avg(t.tot) FILTER (t.k >= greatest(h.ja, 0) AND t.k < least(h.jz, h.nbb)) / greatest(avg(t.tot), 1e-9) END AS trafx_e
  FROM (SELECT *, CASE WHEN fz > fa THEN fa ELSE ia END AS ja, CASE WHEN fz > fa THEN fz ELSE iz END AS jz FROM hq_hb) h
  JOIN hq_htr t USING (hid) GROUP BY h.hid) t USING (hid)
),
hq_mhg AS MATERIALIZED (
SELECT m.dev, m.det, m.hid, x.hod, m.hid // 24 AS dy, m.M, m.O, z.ph
FROM hq_mh m JOIN hq_hix x USING (dev, hid) JOIN hq_dz z USING (dev, det)
WHERE x.ch AND z.ph IS NOT NULL AND coalesce(z.fn, '') <> 'Bike'
),
hq_mhp AS MATERIALIZED (
SELECT a.dev, a.det, b.det AS mate, a.hid, a.hod, a.dy, a.M, b.M AS Mm
FROM hq_mhg a JOIN hq_mhg b ON b.dev = a.dev AND b.hid = a.hid AND b.ph = a.ph AND b.det <> a.det
),
hq_mhqa AS MATERIALIZED (
SELECT dev, det, mate, hod >= 7 AND hod < 19 AS dband, median(M / Mm) AS q_band
FROM hq_mhp x WHERE Mm >= CAST(10.0 AS DOUBLE) GROUP BY ALL
),
hq_mhqh AS MATERIALIZED (
SELECT k.dev, k.det, k.mate, k.dy, k.hod,
       CASE WHEN count(DISTINCT s.dy) >= 3 THEN median(s.q)
            WHEN count(DISTINCT s.dy) = 2 AND min(s.dy) < k.dy THEN median(s.q) FILTER (s.dy < k.dy) END AS q_hod,
       count(DISTINCT s.dy) < 3 AS amb
FROM (SELECT DISTINCT dev, det, mate, dy, hod FROM hq_mhp) k
JOIN (SELECT dev, det, mate, dy, hod, M / Mm AS q FROM hq_mhp x WHERE Mm >= CAST(10.0 AS DOUBLE)) s
  ON s.dev = k.dev AND s.det = k.det AND s.mate = k.mate AND (abs(s.hod - k.hod) <= 1 OR abs(s.hod - k.hod) = 23)
GROUP BY k.dev, k.det, k.mate, k.dy, k.hod
),
hq_mhoh AS MATERIALIZED (
SELECT k.dev, k.det, k.dy, k.hod,
       CASE WHEN count(DISTINCT s.dy) >= 3 THEN median(s.M)
            WHEN count(DISTINCT s.dy) = 2 AND min(s.dy) < k.dy THEN median(s.M) FILTER (s.dy < k.dy) END AS own_ref,
       CASE WHEN count(DISTINCT s.dy) >= 3 THEN median(s.O)
            WHEN count(DISTINCT s.dy) = 2 AND min(s.dy) < k.dy THEN median(s.O) FILTER (s.dy < k.dy) END AS occ_ref
FROM (SELECT DISTINCT dev, det, dy, hod FROM hq_mhg) k
JOIN hq_mhg s ON s.dev = k.dev AND s.det = k.det AND (abs(s.hod - k.hod) <= 1 OR abs(s.hod - k.hod) = 23)
GROUP BY k.dev, k.det, k.dy, k.hod
),
hq_mhv AS MATERIALIZED (
SELECT p.dev, p.det, p.hid, count(*) AS nv, median(h.q_hod * p.Mm) AS E,
       sum(p.Mm) FILTER (p.M <= CAST(0.5 AS DOUBLE) * h.q_hod * p.Mm AND (p.M - h.q_hod * p.Mm) / sqrt(CAST(1.5 AS DOUBLE) * h.q_hod * p.Mm + 1) <= -CAST(2.5 AS DOUBLE)) / sum(p.Mm) AS wlo,
       count(*) FILTER (p.M >= CAST(2.0 AS DOUBLE) * h.q_hod * p.Mm AND (NOT h.amb OR p.M >= CAST(2.0 AS DOUBLE) * a.q_band * p.Mm)
                        AND p.M - h.q_hod * p.Mm >= CAST(15.0 AS DOUBLE) AND (p.M - h.q_hod * p.Mm) / sqrt(CAST(1.5 AS DOUBLE) * h.q_hod * p.Mm + 1) >= CAST(2.5 AS DOUBLE)) AS nhi
FROM hq_mhp p JOIN hq_mhqh h USING (dev, det, mate, dy, hod)
JOIN hq_mhqa a ON a.dev = p.dev AND a.det = p.det AND a.mate = p.mate AND a.dband = (p.hod >= 7 AND p.hod < 19)
WHERE h.q_hod IS NOT NULL AND p.Mm >= 5 GROUP BY p.dev, p.det, p.hid
),
hq_mhf AS MATERIALIZED (
SELECT g.dev, g.det, g.hid, g.M, g.O, v.E, o.occ_ref,
       coalesce(v.nv >= 1 AND v.E >= CAST(8.0 AS DOUBLE), false) AS jl, coalesce(v.nv >= 1, false) AS jh,
       coalesce(v.nv >= 1 AND v.E >= CAST(8.0 AS DOUBLE) AND v.wlo >= 0.5 AND g.M <= CAST(0.75 AS DOUBLE) * o.own_ref AND (g.M - o.own_ref) / sqrt(CAST(1.5 AS DOUBLE) * o.own_ref + 1) <= -CAST(2.5 AS DOUBLE),
                false) AS lo,
       coalesce(v.nv >= 1 AND v.nhi >= least(2, v.nv) AND v.nhi >= CAST(0.6 AS DOUBLE) * v.nv
                AND g.M >= CAST(1.5 AS DOUBLE) * o.own_ref + 10 AND (g.M - o.own_ref) / sqrt(CAST(1.5 AS DOUBLE) * o.own_ref + 1) >= CAST(2.5 AS DOUBLE), false) AS hi
FROM hq_mhg g LEFT JOIN hq_mhv v USING (dev, det, hid) LEFT JOIN hq_mhoh o USING (dev, det, dy, hod)
),
hq_mhrun0 AS MATERIALIZED (
SELECT * , sqrt(2 * (M - E) ^ 2 / greatest(M + E, 1e-9)) AS rgeh FROM (
  SELECT dev, det, 'drop' AS kind, min(hid) FILTER (lo) AS a, max(hid) FILTER (lo) + 1 AS z, count(*) FILTER (lo) AS hours,
         sum(M) FILTER (lo) AS M, sum(E) FILTER (lo) AS E, sum(O) FILTER (lo) AS O, sum(occ_ref) FILTER (lo) AS Oref
  FROM (SELECT *, sum((jl AND NOT lo)::INT) OVER (PARTITION BY dev, det ORDER BY hid) AS grp FROM hq_mhf)
  GROUP BY dev, det, grp HAVING count(*) FILTER (lo) >= 1
  UNION ALL
  SELECT dev, det, 'rise', min(hid) FILTER (hi), max(hid) FILTER (hi) + 1, count(*) FILTER (hi),
         sum(M) FILTER (hi), sum(E) FILTER (hi), sum(O) FILTER (hi), sum(occ_ref) FILTER (hi)
  FROM (SELECT *, sum((jh AND NOT hi)::INT) OVER (PARTITION BY dev, det ORDER BY hid) AS grp FROM hq_mhf)
  GROUP BY dev, det, grp HAVING count(*) FILTER (hi) >= 1)
),
hq_mhday AS MATERIALIZED (
SELECT g.dev, g.det, g.dy, g.Md, m.Mmd
FROM (SELECT dev, det, dy, sum(M) AS Md FROM hq_mhg WHERE hod >= 6 AND hod < 22 GROUP BY ALL) g
JOIN (SELECT dev, det, dy, sum(Mm) AS Mmd FROM hq_mhp WHERE hod >= 6 AND hod < 22 GROUP BY ALL) m USING (dev, det, dy)
),
hq_mhrun AS MATERIALIZED (
SELECT * FROM hq_mhrun0 u
WHERE u.kind <> 'rise' OR NOT EXISTS (
  SELECT 1 FROM hq_mhday r JOIN hq_mhday d ON d.dev = r.dev AND d.det = r.det AND d.dy <> r.dy
  WHERE r.dev = u.dev AND r.det = u.det AND r.dy = u.a // 24
    AND d.Md <= 0.1 * r.Md AND d.Mmd >= 0.5 * r.Mmd AND r.Mmd > 0)
),
hq_mhbest AS MATERIALIZED (
SELECT d.dev, d.det, true AS mh_ok, r.hours AS md_h, r.rgeh AS md_geh, r.M / greatest(r.E, 1e-9) AS md_r, r.a AS md_a,
       r.z AS md_z, r.M / (r.z - r.a) AS md_own, r.E / (r.z - r.a) AS md_exp,
       u.hours AS mr_h, u.rgeh AS mr_geh, u.M / greatest(u.E, 1e-9) AS mr_r, u.a AS mr_a, u.z AS mr_z
FROM (SELECT DISTINCT dev, det FROM hq_mhv) d
LEFT JOIN (SELECT * FROM hq_mhrun WHERE kind = 'drop' AND hours >= 3
             AND (O < CAST(0.8 AS DOUBLE) * Oref OR M <= 0.1 * E)
           QUALIFY row_number() OVER (PARTITION BY dev, det ORDER BY rgeh DESC, a) = 1) r USING (dev, det)
LEFT JOIN (SELECT * FROM hq_mhrun WHERE kind = 'rise' AND hours >= 3
           QUALIFY row_number() OVER (PARTITION BY dev, det ORDER BY rgeh DESC, a) = 1) u USING (dev, det)
),
hq_xx0 AS MATERIALIZED (
SELECT x.*, a.rep_frac, a.max_on_s, a.rep_time_s, a.dur_p50, a.n_on_a, a.n_rep, a.n_start, a.n_ge5, a.n_ge60,
       a.long_max_s, s.elhi_cnt, s.elhi_occ, k.c_occ_like, q.q5_gy, q.q5_all, f.fe_n, f.fe_p, f.fe_phi, t.N, t.n_fast,
       t.fast_rate, t.gap_lim, t.pair_share, x.n_on / x.hours AS rate,
       CASE WHEN x.n_on / x.hours < 20 THEN 'low' WHEN x.n_on / x.hours < 100 THEN 'medium' ELSE 'high' END AS band,
       x.fn || ' ' || x.span AS type, greatest(x.ep_dur, a.max_on_s) AS stuck_x,
       coalesce(s.elhi_occ > 0 AND s.elhi_occ > s.elhi_cnt, false) AS queue_pat8,
       CASE WHEN x.ph IS NOT NULL THEN hm.nh - (x.status1 <> 'bad')::INT END AS n_hmates,
       o.hi_min_c, o.hi_occ_max, coalesce(g.geh_ok, false) AS geh_ok, g.geh_night,
       CASE WHEN g.geh_night IS NOT NULL THEN g.geh_night_n0 END AS geh_night_n,
       CASE WHEN g.geh_night IS NOT NULL THEN g.geh_night_exp0 END AS geh_night_exp, g.geh_drop, g.drop_t, g.drop_t2,
       g.occ_after,
       g.own_before, g.own_after, g.exp_after, g.occ_own, g.occ_mates, p.spk_bursts, p.spk_gmax, p.spk_daily, p.spk_lvl,
       m.mocc AS occ, coalesce(mb.mh_ok, false) AS mh_ok, mb.md_h, mb.md_geh, mb.md_r, mb.md_a, mb.md_z, mb.md_own, mb.md_exp, mb.mr_h, mb.mr_geh, mb.mr_r, mb.mr_a, mb.mr_z
FROM hq_xm x LEFT JOIN hq_act a USING (dev, det) LEFT JOIN hq_slp s USING (dev, det) LEFT JOIN hq_lkc k USING (dev, det)
LEFT JOIN hq_q5 q USING (dev, det) LEFT JOIN hq_fe f USING (dev, det) LEFT JOIN hq_tf t USING (dev, det)
LEFT JOIN hq_ohi o USING (dev, det) LEFT JOIN hq_geh g USING (dev, det) LEFT JOIN hq_spk p USING (dev, det)
LEFT JOIN (SELECT DISTINCT dev, det, mocc FROM hq_gb) m USING (dev, det)
LEFT JOIN hq_mhbest mb USING (dev, det)
LEFT JOIN (SELECT dev, ph, sum((status1 <> 'bad')::INT) AS nh FROM hq_xm GROUP BY dev, ph) hm
  ON hm.dev = x.dev AND hm.ph = x.ph
),
hq_ltab AS MATERIALIZED (
SELECT * FROM (VALUES ('stuck_s', 'Advance|1', CAST(606.819377627 AS DOUBLE), CAST(1800.0 AS DOUBLE)), ('stuck_s', 'Advance|2+', CAST(710.635480449 AS DOUBLE), CAST(1800.0 AS DOUBLE)), ('stuck_s', 'Presence|1', CAST(606.819377048 AS DOUBLE), CAST(3600.0 AS DOUBLE)), ('stuck_s', 'Presence|2+', CAST(924.222127102 AS DOUBLE), CAST(3600.0 AS DOUBLE)), ('stuck_s', 'Count|1|pulse', CAST(342.090105353 AS DOUBLE), CAST(1200.0 AS DOUBLE)), ('stuck_s', 'Count|1|normal', CAST(2159.337067371 AS DOUBLE), CAST(2159.469950268 AS DOUBLE)), ('stuck_s', 'Count|2+|pulse', CAST(342.090105353 AS DOUBLE), CAST(1200.0 AS DOUBLE)), ('stuck_s', 'Count|2+|normal', CAST(749.462875665 AS DOUBLE), CAST(1458.628300179 AS DOUBLE)), ('stuck_s', 'Yellow_Red|1', CAST(328.333559032 AS DOUBLE), CAST(1200.0 AS DOUBLE)), ('stuck_s', 'Yellow_Red|2+', CAST(710.635480449 AS DOUBLE), CAST(1200.0 AS DOUBLE)), ('stuck_s', 'Mid|1', CAST(606.819377048 AS DOUBLE), CAST(2100.0 AS DOUBLE)), ('stuck_s', 'Mid|2+', CAST(710.635480161 AS DOUBLE), CAST(2100.0 AS DOUBLE)), ('stuck_s', 'Bike|1', CAST(328.333559032 AS DOUBLE), CAST(600.0 AS DOUBLE)), ('stuck_s', 'Bike|2+', CAST(328.333559032 AS DOUBLE), CAST(600.0 AS DOUBLE)), ('stuck_s', 'Other|1', CAST(881.624355227 AS DOUBLE), CAST(3600.0 AS DOUBLE)), ('stuck_s', 'Other|2+', CAST(924.222127102 AS DOUBLE), CAST(3600.0 AS DOUBLE)), ('silent_exp', 'Advance|1', CAST(23.589250055 AS DOUBLE), CAST(756.704318903 AS DOUBLE)), ('silent_exp', 'Advance|2+', CAST(11.653101112 AS DOUBLE), CAST(77.503311673 AS DOUBLE)), ('silent_exp', 'Presence|1', CAST(24.836852771 AS DOUBLE), CAST(147.222879645 AS DOUBLE)), ('silent_exp', 'Presence|2+', CAST(25.44293518 AS DOUBLE), CAST(25.444501514 AS DOUBLE)), ('silent_exp', 'Count|1', CAST(22.851456978 AS DOUBLE), CAST(177.198200529 AS DOUBLE)), ('silent_exp', 'Count|2+', CAST(16.181775084 AS DOUBLE), CAST(51.908471294 AS DOUBLE)), ('silent_exp', 'Yellow_Red|1', CAST(24.489525225 AS DOUBLE), CAST(42.513393988 AS DOUBLE)), ('silent_exp', 'Yellow_Red|2+', CAST(17.803387272 AS DOUBLE), CAST(42.012776556 AS DOUBLE)), ('silent_exp', 'Mid|1', CAST(18.609944848 AS DOUBLE), CAST(18.61109069 AS DOUBLE)), ('silent_exp', 'Mid|2+', CAST(6.688239183 AS DOUBLE), CAST(1253.69109513 AS DOUBLE)), ('silent_exp', 'Bike|1', CAST(19.143709744 AS DOUBLE), CAST(19.144888434 AS DOUBLE)), ('silent_exp', 'Bike|2+', CAST(19.143709744 AS DOUBLE), CAST(19.144888434 AS DOUBLE)), ('silent_exp', 'Other|1', CAST(54.589934263 AS DOUBLE), CAST(150.19586738 AS DOUBLE)), ('silent_exp', 'Other|2+', CAST(124.399294409 AS DOUBLE), CAST(585.728912925 AS DOUBLE)), ('busy_z', 'Advance|1', CAST(2.900778577 AS DOUBLE), NULL), ('busy_z', 'Advance|2+', CAST(2.394314778 AS DOUBLE), NULL), ('busy_z', 'Presence|1', CAST(3.711525417 AS DOUBLE), NULL), ('busy_z', 'Presence|2+', CAST(3.667360721 AS DOUBLE), NULL), ('busy_z', 'Count|1', CAST(2.900778577 AS DOUBLE), NULL), ('busy_z', 'Count|2+', CAST(2.900778577 AS DOUBLE), NULL), ('busy_z', 'Yellow_Red|1', CAST(2.900778577 AS DOUBLE), NULL), ('busy_z', 'Yellow_Red|2+', CAST(2.686465366 AS DOUBLE), NULL), ('busy_z', 'Mid|1', CAST(3.161702698 AS DOUBLE), NULL), ('busy_z', 'Mid|2+', CAST(2.791566081 AS DOUBLE), NULL), ('busy_z', 'Bike|1', CAST(3.674402063 AS DOUBLE), NULL), ('busy_z', 'Bike|2+', CAST(3.674402063 AS DOUBLE), NULL), ('busy_z', 'Other|1', CAST(3.724573474 AS DOUBLE), NULL), ('busy_z', 'Other|2+', CAST(2.686465366 AS DOUBLE), NULL), ('per_1000', 'Advance|1', CAST(188.1303618616211 AS DOUBLE), NULL), ('per_1000', 'Advance|2+', CAST(188.1303618616211 AS DOUBLE), NULL), ('per_1000', 'Presence|1', CAST(196.97746920883776 AS DOUBLE), NULL), ('per_1000', 'Presence|2+', CAST(196.97746920883776 AS DOUBLE), NULL), ('per_1000', 'Count|1', CAST(158.6942935224567 AS DOUBLE), NULL), ('per_1000', 'Count|2+', CAST(158.6942935224567 AS DOUBLE), NULL), ('per_1000', 'Yellow_Red|1', CAST(191.8147153598292 AS DOUBLE), NULL), ('per_1000', 'Yellow_Red|2+', CAST(191.8147153598292 AS DOUBLE), NULL), ('per_1000', 'Mid|1', CAST(170.27296207664892 AS DOUBLE), NULL), ('per_1000', 'Mid|2+', CAST(170.27296207664892 AS DOUBLE), NULL), ('per_1000', 'Bike|1', CAST(207.61904761904648 AS DOUBLE), NULL), ('per_1000', 'Bike|2+', CAST(207.61904761904648 AS DOUBLE), NULL), ('per_1000', 'Other|1', CAST(190.16502704672212 AS DOUBLE), NULL), ('per_1000', 'Other|2+', CAST(190.16502704672212 AS DOUBLE), NULL)) t(tab, key, lo, hi)
),
hq_lref AS MATERIALIZED (
SELECT * FROM (VALUES ('stuck_x', 0, 'Advance', '1', '', 'h24', NULL, CAST(8151.436000000009 AS DOUBLE)), ('stuck_x', 0, 'Advance', '1', '', 'h3', NULL, CAST(1271.8773999999996 AS DOUBLE)), ('stuck_x', 0, 'Advance', '1', '', 'm30', NULL, CAST(238.28799999999765 AS DOUBLE)), ('stuck_x', 0, 'Advance', '2+', '', 'h24', NULL, CAST(434.7639999999967 AS DOUBLE)), ('stuck_x', 0, 'Advance', '2+', '', 'h3', NULL, CAST(293.135999999999 AS DOUBLE)), ('stuck_x', 0, 'Advance', '2+', '', 'm30', NULL, CAST(107.09899999999976 AS DOUBLE)), ('stuck_x', 0, 'Bike', '1', '', 'h24', NULL, CAST(3674.885199999973 AS DOUBLE)), ('stuck_x', 0, 'Bike', '1', '', 'h3', NULL, CAST(224.3440000000003 AS DOUBLE)), ('stuck_x', 0, 'Bike', '1', '', 'm30', NULL, CAST(121.16000000000001 AS DOUBLE)), ('stuck_x', 0, 'Count', '1', 'normal', 'h24', NULL, CAST(1982.6100000000088 AS DOUBLE)), ('stuck_x', 0, 'Count', '1', 'normal', 'h3', NULL, CAST(415.2726000000002 AS DOUBLE)), ('stuck_x', 0, 'Count', '1', 'normal', 'm30', NULL, CAST(117.32659999999964 AS DOUBLE)), ('stuck_x', 0, 'Count', '1', 'pulse', 'h24', NULL, CAST(258.36480000000364 AS DOUBLE)), ('stuck_x', 0, 'Count', '1', 'pulse', 'h3', NULL, CAST(98.6648000000003 AS DOUBLE)), ('stuck_x', 0, 'Count', '1', 'pulse', 'm30', NULL, CAST(24.744199999999733 AS DOUBLE)), ('stuck_x', 0, 'Mid', '1', '', 'h24', NULL, CAST(7984.704799999996 AS DOUBLE)), ('stuck_x', 0, 'Mid', '1', '', 'h3', NULL, CAST(839.1480000000181 AS DOUBLE)), ('stuck_x', 0, 'Mid', '1', '', 'm30', NULL, CAST(117.73020000000058 AS DOUBLE)), ('stuck_x', 0, 'Mid', '2+', '', 'h24', NULL, CAST(7665.438399999997 AS DOUBLE)), ('stuck_x', 0, 'Mid', '2+', '', 'h3', NULL, CAST(604.6307999999999 AS DOUBLE)), ('stuck_x', 0, 'Mid', '2+', '', 'm30', NULL, CAST(144.56360000000058 AS DOUBLE)), ('stuck_x', 0, 'Other', '1', '', 'h24', NULL, CAST(3536.1683999999213 AS DOUBLE)), ('stuck_x', 0, 'Other', '1', '', 'h3', NULL, CAST(729.7349999999974 AS DOUBLE)), ('stuck_x', 0, 'Other', '1', '', 'm30', NULL, CAST(340.67579999999975 AS DOUBLE)), ('stuck_x', 0, 'Other', '2+', '', 'h24', NULL, CAST(22211.02920000112 AS DOUBLE)), ('stuck_x', 0, 'Other', '2+', '', 'h3', NULL, CAST(776.7320000000138 AS DOUBLE)), ('stuck_x', 0, 'Other', '2+', '', 'm30', NULL, CAST(300.0475999999893 AS DOUBLE)), ('stuck_x', 0, 'Presence', '1', '', 'h24', NULL, CAST(30252.799999999996 AS DOUBLE)), ('stuck_x', 0, 'Presence', '1', '', 'h3', NULL, CAST(1636.7886000000124 AS DOUBLE)), ('stuck_x', 0, 'Presence', '1', '', 'm30', NULL, CAST(385.4715999999961 AS DOUBLE)), ('stuck_x', 0, 'Presence', '2+', '', 'h24', NULL, CAST(1942.4346000000046 AS DOUBLE)), ('stuck_x', 0, 'Presence', '2+', '', 'h3', NULL, CAST(601.856000000002 AS DOUBLE)), ('stuck_x', 0, 'Presence', '2+', '', 'm30', NULL, CAST(321.00400000000036 AS DOUBLE)), ('stuck_x', 0, 'Yellow_Red', '1', '', 'h24', NULL, CAST(447.2599999999825 AS DOUBLE)), ('stuck_x', 0, 'Yellow_Red', '1', '', 'h3', NULL, CAST(174.67380000000037 AS DOUBLE)), ('stuck_x', 0, 'Yellow_Red', '1', '', 'm30', NULL, CAST(90.41839999999961 AS DOUBLE)), ('stuck_x', 0, 'Yellow_Red', '2+', '', 'h24', NULL, CAST(178.24000000000333 AS DOUBLE)), ('stuck_x', 0, 'Yellow_Red', '2+', '', 'h3', NULL, CAST(139.8615999999992 AS DOUBLE)), ('stuck_x', 0, 'Yellow_Red', '2+', '', 'm30', NULL, CAST(114.11919999999994 AS DOUBLE)), ('stuck_x', 1, 'Advance', '1', NULL, 'h24', NULL, CAST(8151.436000000009 AS DOUBLE)), ('stuck_x', 1, 'Advance', '1', NULL, 'h3', NULL, CAST(1271.8773999999996 AS DOUBLE)), ('stuck_x', 1, 'Advance', '1', NULL, 'm30', NULL, CAST(238.28799999999765 AS DOUBLE)), ('stuck_x', 1, 'Advance', '2+', NULL, 'h24', NULL, CAST(434.7639999999967 AS DOUBLE)), ('stuck_x', 1, 'Advance', '2+', NULL, 'h3', NULL, CAST(293.135999999999 AS DOUBLE)), ('stuck_x', 1, 'Advance', '2+', NULL, 'm30', NULL, CAST(107.09899999999976 AS DOUBLE)), ('stuck_x', 1, 'Bike', '1', NULL, 'h24', NULL, CAST(3674.885199999973 AS DOUBLE)), ('stuck_x', 1, 'Bike', '1', NULL, 'h3', NULL, CAST(224.3440000000003 AS DOUBLE)), ('stuck_x', 1, 'Bike', '1', NULL, 'm30', NULL, CAST(121.16000000000001 AS DOUBLE)), ('stuck_x', 1, 'Count', '1', NULL, 'h24', NULL, CAST(412.5087999999919 AS DOUBLE)), ('stuck_x', 1, 'Count', '1', NULL, 'h3', NULL, CAST(136.31599999999946 AS DOUBLE)), ('stuck_x', 1, 'Count', '1', NULL, 'm30', NULL, CAST(93.4553999999998 AS DOUBLE)), ('stuck_x', 1, 'Count', '2+', NULL, 'm30', NULL, CAST(103.43499999999983 AS DOUBLE)), ('stuck_x', 1, 'Mid', '1', NULL, 'h24', NULL, CAST(7984.704799999996 AS DOUBLE)), ('stuck_x', 1, 'Mid', '1', NULL, 'h3', NULL, CAST(839.1480000000181 AS DOUBLE)), ('stuck_x', 1, 'Mid', '1', NULL, 'm30', NULL, CAST(117.73020000000058 AS DOUBLE)), ('stuck_x', 1, 'Mid', '2+', NULL, 'h24', NULL, CAST(7665.438399999997 AS DOUBLE)), ('stuck_x', 1, 'Mid', '2+', NULL, 'h3', NULL, CAST(604.6307999999999 AS DOUBLE)), ('stuck_x', 1, 'Mid', '2+', NULL, 'm30', NULL, CAST(144.56360000000058 AS DOUBLE)), ('stuck_x', 1, 'Other', '1', NULL, 'h24', NULL, CAST(3536.1683999999213 AS DOUBLE)), ('stuck_x', 1, 'Other', '1', NULL, 'h3', NULL, CAST(729.7349999999974 AS DOUBLE)), ('stuck_x', 1, 'Other', '1', NULL, 'm30', NULL, CAST(340.67579999999975 AS DOUBLE)), ('stuck_x', 1, 'Other', '2+', NULL, 'h24', NULL, CAST(22211.02920000112 AS DOUBLE)), ('stuck_x', 1, 'Other', '2+', NULL, 'h3', NULL, CAST(776.7320000000138 AS DOUBLE)), ('stuck_x', 1, 'Other', '2+', NULL, 'm30', NULL, CAST(300.0475999999893 AS DOUBLE)), ('stuck_x', 1, 'Presence', '1', NULL, 'h24', NULL, CAST(30252.799999999996 AS DOUBLE)), ('stuck_x', 1, 'Presence', '1', NULL, 'h3', NULL, CAST(1636.7886000000124 AS DOUBLE)), ('stuck_x', 1, 'Presence', '1', NULL, 'm30', NULL, CAST(385.4715999999961 AS DOUBLE)), ('stuck_x', 1, 'Presence', '2+', NULL, 'h24', NULL, CAST(1942.4346000000046 AS DOUBLE)), ('stuck_x', 1, 'Presence', '2+', NULL, 'h3', NULL, CAST(601.856000000002 AS DOUBLE)), ('stuck_x', 1, 'Presence', '2+', NULL, 'm30', NULL, CAST(321.00400000000036 AS DOUBLE)), ('stuck_x', 1, 'Yellow_Red', '1', NULL, 'h24', NULL, CAST(447.2599999999825 AS DOUBLE)), ('stuck_x', 1, 'Yellow_Red', '1', NULL, 'h3', NULL, CAST(174.67380000000037 AS DOUBLE)), ('stuck_x', 1, 'Yellow_Red', '1', NULL, 'm30', NULL, CAST(90.41839999999961 AS DOUBLE)), ('stuck_x', 1, 'Yellow_Red', '2+', NULL, 'h24', NULL, CAST(178.24000000000333 AS DOUBLE)), ('stuck_x', 1, 'Yellow_Red', '2+', NULL, 'h3', NULL, CAST(139.8615999999992 AS DOUBLE)), ('stuck_x', 1, 'Yellow_Red', '2+', NULL, 'm30', NULL, CAST(114.11919999999994 AS DOUBLE)), ('stuck_x', 2, NULL, '1', NULL, 'h24', NULL, CAST(7756.599599999983 AS DOUBLE)), ('stuck_x', 2, NULL, '1', NULL, 'h3', NULL, CAST(1208.4951999999844 AS DOUBLE)), ('stuck_x', 2, NULL, '1', NULL, 'm30', NULL, CAST(308.75720000000223 AS DOUBLE)), ('stuck_x', 2, NULL, '2+', NULL, 'h24', NULL, CAST(10772.759999999998 AS DOUBLE)), ('stuck_x', 2, NULL, '2+', NULL, 'h3', NULL, CAST(602.947200000013 AS DOUBLE)), ('stuck_x', 2, NULL, '2+', NULL, 'm30', NULL, CAST(258.2580000000003 AS DOUBLE)), ('stuck_x', 3, NULL, NULL, NULL, 'h24', NULL, CAST(7909.660800000041 AS DOUBLE)), ('stuck_x', 3, NULL, NULL, NULL, 'h3', NULL, CAST(1120.5854000000018 AS DOUBLE)), ('stuck_x', 3, NULL, NULL, NULL, 'm30', NULL, CAST(307.30000000000007 AS DOUBLE)), ('n_ep', 0, 'Advance', '1', '', 'h24', NULL, CAST(1.0 AS DOUBLE)), ('n_ep', 0, 'Advance', '1', '', 'h3', NULL, CAST(1.0 AS DOUBLE)), ('n_ep', 0, 'Advance', '1', '', 'm30', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 0, 'Advance', '2+', '', 'h24', NULL, CAST(0.8780000000000427 AS DOUBLE)), ('n_ep', 0, 'Advance', '2+', '', 'h3', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 0, 'Advance', '2+', '', 'm30', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 0, 'Bike', '1', '', 'h24', NULL, CAST(1.0 AS DOUBLE)), ('n_ep', 0, 'Bike', '1', '', 'h3', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 0, 'Bike', '1', '', 'm30', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 0, 'Count', '1', 'normal', 'h24', NULL, CAST(1.0 AS DOUBLE)), ('n_ep', 0, 'Count', '1', 'normal', 'h3', NULL, CAST(0.7259999999999991 AS DOUBLE)), ('n_ep', 0, 'Count', '1', 'normal', 'm30', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 0, 'Count', '1', 'pulse', 'h24', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 0, 'Count', '1', 'pulse', 'h3', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 0, 'Count', '1', 'pulse', 'm30', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 0, 'Mid', '1', '', 'h24', NULL, CAST(1.0 AS DOUBLE)), ('n_ep', 0, 'Mid', '1', '', 'h3', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 0, 'Mid', '1', '', 'm30', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 0, 'Mid', '2+', '', 'h24', NULL, CAST(1.0 AS DOUBLE)), ('n_ep', 0, 'Mid', '2+', '', 'h3', NULL, CAST(0.51400000000001 AS DOUBLE)), ('n_ep', 0, 'Mid', '2+', '', 'm30', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 0, 'Other', '1', '', 'h24', NULL, CAST(1.0 AS DOUBLE)), ('n_ep', 0, 'Other', '1', '', 'h3', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 0, 'Other', '1', '', 'm30', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 0, 'Other', '2+', '', 'h24', NULL, CAST(28.85600000000204 AS DOUBLE)), ('n_ep', 0, 'Other', '2+', '', 'h3', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 0, 'Other', '2+', '', 'm30', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 0, 'Presence', '1', '', 'h24', NULL, CAST(2.0 AS DOUBLE)), ('n_ep', 0, 'Presence', '1', '', 'h3', NULL, CAST(1.0 AS DOUBLE)), ('n_ep', 0, 'Presence', '1', '', 'm30', NULL, CAST(1.0 AS DOUBLE)), ('n_ep', 0, 'Presence', '2+', '', 'h24', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 0, 'Presence', '2+', '', 'h3', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 0, 'Presence', '2+', '', 'm30', NULL, CAST(0.2400000000000091 AS DOUBLE)), ('n_ep', 0, 'Yellow_Red', '1', '', 'h24', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 0, 'Yellow_Red', '1', '', 'h3', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 0, 'Yellow_Red', '1', '', 'm30', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 0, 'Yellow_Red', '2+', '', 'h24', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 0, 'Yellow_Red', '2+', '', 'h3', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 0, 'Yellow_Red', '2+', '', 'm30', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 1, 'Advance', '1', NULL, 'h24', NULL, CAST(1.0 AS DOUBLE)), ('n_ep', 1, 'Advance', '1', NULL, 'h3', NULL, CAST(1.0 AS DOUBLE)), ('n_ep', 1, 'Advance', '1', NULL, 'm30', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 1, 'Advance', '2+', NULL, 'h24', NULL, CAST(0.8780000000000427 AS DOUBLE)), ('n_ep', 1, 'Advance', '2+', NULL, 'h3', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 1, 'Advance', '2+', NULL, 'm30', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 1, 'Bike', '1', NULL, 'h24', NULL, CAST(1.0 AS DOUBLE)), ('n_ep', 1, 'Bike', '1', NULL, 'h3', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 1, 'Bike', '1', NULL, 'm30', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 1, 'Count', '1', NULL, 'h24', NULL, CAST(0.023999999999887223 AS DOUBLE)), ('n_ep', 1, 'Count', '1', NULL, 'h3', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 1, 'Count', '1', NULL, 'm30', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 1, 'Count', '2+', NULL, 'm30', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 1, 'Mid', '1', NULL, 'h24', NULL, CAST(1.0 AS DOUBLE)), ('n_ep', 1, 'Mid', '1', NULL, 'h3', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 1, 'Mid', '1', NULL, 'm30', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 1, 'Mid', '2+', NULL, 'h24', NULL, CAST(1.0 AS DOUBLE)), ('n_ep', 1, 'Mid', '2+', NULL, 'h3', NULL, CAST(0.51400000000001 AS DOUBLE)), ('n_ep', 1, 'Mid', '2+', NULL, 'm30', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 1, 'Other', '1', NULL, 'h24', NULL, CAST(1.0 AS DOUBLE)), ('n_ep', 1, 'Other', '1', NULL, 'h3', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 1, 'Other', '1', NULL, 'm30', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 1, 'Other', '2+', NULL, 'h24', NULL, CAST(28.85600000000204 AS DOUBLE)), ('n_ep', 1, 'Other', '2+', NULL, 'h3', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 1, 'Other', '2+', NULL, 'm30', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 1, 'Presence', '1', NULL, 'h24', NULL, CAST(2.0 AS DOUBLE)), ('n_ep', 1, 'Presence', '1', NULL, 'h3', NULL, CAST(1.0 AS DOUBLE)), ('n_ep', 1, 'Presence', '1', NULL, 'm30', NULL, CAST(1.0 AS DOUBLE)), ('n_ep', 1, 'Presence', '2+', NULL, 'h24', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 1, 'Presence', '2+', NULL, 'h3', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 1, 'Presence', '2+', NULL, 'm30', NULL, CAST(0.2400000000000091 AS DOUBLE)), ('n_ep', 1, 'Yellow_Red', '1', NULL, 'h24', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 1, 'Yellow_Red', '1', NULL, 'h3', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 1, 'Yellow_Red', '1', NULL, 'm30', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 1, 'Yellow_Red', '2+', NULL, 'h24', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 1, 'Yellow_Red', '2+', NULL, 'h3', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 1, 'Yellow_Red', '2+', NULL, 'm30', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 2, NULL, '1', NULL, 'h24', NULL, CAST(1.0 AS DOUBLE)), ('n_ep', 2, NULL, '1', NULL, 'h3', NULL, CAST(1.0 AS DOUBLE)), ('n_ep', 2, NULL, '1', NULL, 'm30', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 2, NULL, '2+', NULL, 'h24', NULL, CAST(1.1199999999998909 AS DOUBLE)), ('n_ep', 2, NULL, '2+', NULL, 'h3', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 2, NULL, '2+', NULL, 'm30', NULL, CAST(0.0 AS DOUBLE)), ('n_ep', 3, NULL, NULL, NULL, 'h24', NULL, CAST(1.0 AS DOUBLE)), ('n_ep', 3, NULL, NULL, NULL, 'h3', NULL, CAST(1.0 AS DOUBLE)), ('n_ep', 3, NULL, NULL, NULL, 'm30', NULL, CAST(0.0 AS DOUBLE)), ('q5_gy', 0, 'Advance', '1', NULL, 'h24', NULL, CAST(1830.0604140625005 AS DOUBLE)), ('q5_gy', 0, 'Advance', '1', NULL, 'h3', NULL, CAST(1613.7195373535055 AS DOUBLE)), ('q5_gy', 0, 'Advance', '1', NULL, 'm30', NULL, CAST(2539.6475976562883 AS DOUBLE)), ('q5_gy', 0, 'Advance', '2+', NULL, 'h24', NULL, CAST(929.6164270019551 AS DOUBLE)), ('q5_gy', 0, 'Advance', '2+', NULL, 'h3', NULL, CAST(1041.3407031249951 AS DOUBLE)), ('q5_gy', 0, 'Advance', '2+', NULL, 'm30', NULL, CAST(973.082352539064 AS DOUBLE)), ('q5_gy', 0, 'Bike', '1', NULL, 'h24', NULL, CAST(582.3916625976603 AS DOUBLE)), ('q5_gy', 0, 'Bike', '1', NULL, 'h3', NULL, CAST(436.8933553466803 AS DOUBLE)), ('q5_gy', 0, 'Count', '1', NULL, 'h24', NULL, CAST(1709.686094726558 AS DOUBLE)), ('q5_gy', 0, 'Count', '1', NULL, 'h3', NULL, CAST(1660.5506767578113 AS DOUBLE)), ('q5_gy', 0, 'Count', '1', NULL, 'm30', NULL, CAST(1676.7677001953125 AS DOUBLE)), ('q5_gy', 0, 'Mid', '1', NULL, 'h24', NULL, CAST(1917.3225000000007 AS DOUBLE)), ('q5_gy', 0, 'Mid', '1', NULL, 'h3', NULL, CAST(1781.9296923828078 AS DOUBLE)), ('q5_gy', 0, 'Mid', '1', NULL, 'm30', NULL, CAST(1681.4054882812504 AS DOUBLE)), ('q5_gy', 0, 'Mid', '2+', NULL, 'h24', NULL, CAST(1107.5753173828155 AS DOUBLE)), ('q5_gy', 0, 'Mid', '2+', NULL, 'h3', NULL, CAST(1021.2781213378897 AS DOUBLE)), ('q5_gy', 0, 'Mid', '2+', NULL, 'm30', NULL, CAST(971.9941563720695 AS DOUBLE)), ('q5_gy', 0, 'Other', '1', NULL, 'h24', NULL, CAST(1487.0499648437528 AS DOUBLE)), ('q5_gy', 0, 'Other', '1', NULL, 'h3', NULL, CAST(1331.8145961914056 AS DOUBLE)), ('q5_gy', 0, 'Other', '1', NULL, 'm30', NULL, CAST(1460.109050048828 AS DOUBLE)), ('q5_gy', 0, 'Other', '2+', NULL, 'h24', NULL, CAST(633.4109650878887 AS DOUBLE)), ('q5_gy', 0, 'Other', '2+', NULL, 'h3', NULL, CAST(584.288161499023 AS DOUBLE)), ('q5_gy', 0, 'Other', '2+', NULL, 'm30', NULL, CAST(500.9896003417969 AS DOUBLE)), ('q5_gy', 0, 'Presence', '1', NULL, 'h24', NULL, CAST(1581.395137207025 AS DOUBLE)), ('q5_gy', 0, 'Presence', '1', NULL, 'h3', NULL, CAST(1493.0760800781252 AS DOUBLE)), ('q5_gy', 0, 'Presence', '1', NULL, 'm30', NULL, CAST(1521.1233603515616 AS DOUBLE)), ('q5_gy', 0, 'Presence', '2+', NULL, 'h24', NULL, CAST(839.4874870605458 AS DOUBLE)), ('q5_gy', 0, 'Presence', '2+', NULL, 'h3', NULL, CAST(776.8595728759767 AS DOUBLE)), ('q5_gy', 0, 'Yellow_Red', '1', NULL, 'h24', NULL, CAST(2043.4089277343728 AS DOUBLE)), ('q5_gy', 0, 'Yellow_Red', '1', NULL, 'h3', NULL, CAST(1631.4227895507881 AS DOUBLE)), ('q5_gy', 0, 'Yellow_Red', '1', NULL, 'm30', NULL, CAST(1629.9925717773435 AS DOUBLE)), ('q5_gy', 0, 'Yellow_Red', '2+', NULL, 'h24', NULL, CAST(1555.3581311035161 AS DOUBLE)), ('q5_gy', 0, 'Yellow_Red', '2+', NULL, 'h3', NULL, CAST(1430.7180546874986 AS DOUBLE)), ('q5_gy', 0, 'Yellow_Red', '2+', NULL, 'm30', NULL, CAST(1403.4441943359366 AS DOUBLE)), ('q5_gy', 1, NULL, '1', NULL, 'h24', NULL, CAST(1704.6516347656236 AS DOUBLE)), ('q5_gy', 1, NULL, '1', NULL, 'h3', NULL, CAST(1597.8785180664051 AS DOUBLE)), ('q5_gy', 1, NULL, '1', NULL, 'm30', NULL, CAST(1701.9509140625003 AS DOUBLE)), ('q5_gy', 1, NULL, '2+', NULL, 'h24', NULL, CAST(1330.5578662109385 AS DOUBLE)), ('q5_gy', 1, NULL, '2+', NULL, 'h3', NULL, CAST(1239.125662109385 AS DOUBLE)), ('q5_gy', 1, NULL, '2+', NULL, 'm30', NULL, CAST(1270.588183593752 AS DOUBLE)), ('q5_gy', 2, NULL, NULL, NULL, 'h24', NULL, CAST(1690.2927121581897 AS DOUBLE)), ('q5_gy', 2, NULL, NULL, NULL, 'h3', NULL, CAST(1580.9374658203126 AS DOUBLE)), ('q5_gy', 2, NULL, NULL, NULL, 'm30', NULL, CAST(1673.854343749984 AS DOUBLE)), ('q5_all', 0, 'Advance', '1', NULL, 'h24', NULL, CAST(1435.5600000000013 AS DOUBLE)), ('q5_all', 0, 'Advance', '1', NULL, 'h3', NULL, CAST(1375.7999999999738 AS DOUBLE)), ('q5_all', 0, 'Advance', '1', NULL, 'm30', NULL, CAST(2448.0 AS DOUBLE)), ('q5_all', 0, 'Advance', '2+', NULL, 'h24', NULL, CAST(755.0400000000009 AS DOUBLE)), ('q5_all', 0, 'Advance', '2+', NULL, 'h3', NULL, CAST(856.2240000000011 AS DOUBLE)), ('q5_all', 0, 'Advance', '2+', NULL, 'm30', NULL, CAST(728.4000000000019 AS DOUBLE)), ('q5_all', 0, 'Bike', '1', NULL, 'h24', NULL, CAST(348.0 AS DOUBLE)), ('q5_all', 0, 'Bike', '1', NULL, 'h3', NULL, CAST(338.3039999999987 AS DOUBLE)), ('q5_all', 0, 'Count', '1', NULL, 'h24', NULL, CAST(1181.4239999999972 AS DOUBLE)), ('q5_all', 0, 'Count', '1', NULL, 'h3', NULL, CAST(1152.0 AS DOUBLE)), ('q5_all', 0, 'Count', '1', NULL, 'm30', NULL, CAST(1092.0959999999977 AS DOUBLE)), ('q5_all', 0, 'Mid', '1', NULL, 'h24', NULL, CAST(1536.0 AS DOUBLE)), ('q5_all', 0, 'Mid', '1', NULL, 'h3', NULL, CAST(1536.0 AS DOUBLE)), ('q5_all', 0, 'Mid', '1', NULL, 'm30', NULL, CAST(1177.9200000000003 AS DOUBLE)), ('q5_all', 0, 'Mid', '2+', NULL, 'h24', NULL, CAST(845.1000000000015 AS DOUBLE)), ('q5_all', 0, 'Mid', '2+', NULL, 'h3', NULL, CAST(814.8600000000004 AS DOUBLE)), ('q5_all', 0, 'Mid', '2+', NULL, 'm30', NULL, CAST(647.9759999999999 AS DOUBLE)), ('q5_all', 0, 'Other', '1', NULL, 'h24', NULL, CAST(1152.2160000000003 AS DOUBLE)), ('q5_all', 0, 'Other', '1', NULL, 'h3', NULL, CAST(1164.0239999999994 AS DOUBLE)), ('q5_all', 0, 'Other', '1', NULL, 'm30', NULL, CAST(1202.712 AS DOUBLE)), ('q5_all', 0, 'Other', '2+', NULL, 'h24', NULL, CAST(528.0 AS DOUBLE)), ('q5_all', 0, 'Other', '2+', NULL, 'h3', NULL, CAST(444.0 AS DOUBLE)), ('q5_all', 0, 'Other', '2+', NULL, 'm30', NULL, CAST(429.432 AS DOUBLE)), ('q5_all', 0, 'Presence', '1', NULL, 'h24', NULL, CAST(1077.648000000001 AS DOUBLE)), ('q5_all', 0, 'Presence', '1', NULL, 'h3', NULL, CAST(1014.6000000000022 AS DOUBLE)), ('q5_all', 0, 'Presence', '1', NULL, 'm30', NULL, CAST(916.5119999999997 AS DOUBLE)), ('q5_all', 0, 'Presence', '2+', NULL, 'h24', NULL, CAST(533.0159999999998 AS DOUBLE)), ('q5_all', 0, 'Presence', '2+', NULL, 'h3', NULL, CAST(548.2079999999999 AS DOUBLE)), ('q5_all', 0, 'Yellow_Red', '1', NULL, 'h24', NULL, CAST(929.544043823238 AS DOUBLE)), ('q5_all', 0, 'Yellow_Red', '1', NULL, 'h3', NULL, CAST(1134.2400000000052 AS DOUBLE)), ('q5_all', 0, 'Yellow_Red', '1', NULL, 'm30', NULL, CAST(1131.504 AS DOUBLE)), ('q5_all', 0, 'Yellow_Red', '2+', NULL, 'h24', NULL, CAST(897.7680000000003 AS DOUBLE)), ('q5_all', 0, 'Yellow_Red', '2+', NULL, 'h3', NULL, CAST(807.3600000000024 AS DOUBLE)), ('q5_all', 0, 'Yellow_Red', '2+', NULL, 'm30', NULL, CAST(753.2999999999997 AS DOUBLE)), ('q5_all', 1, NULL, '1', NULL, 'h24', NULL, CAST(1272.0 AS DOUBLE)), ('q5_all', 1, NULL, '1', NULL, 'h3', NULL, CAST(1217.4719999999797 AS DOUBLE)), ('q5_all', 1, NULL, '1', NULL, 'm30', NULL, CAST(1253.4720000000016 AS DOUBLE)), ('q5_all', 1, NULL, '2+', NULL, 'h24', NULL, CAST(822.7800000000007 AS DOUBLE)), ('q5_all', 1, NULL, '2+', NULL, 'h3', NULL, CAST(770.3040000000001 AS DOUBLE)), ('q5_all', 1, NULL, '2+', NULL, 'm30', NULL, CAST(694.8000000000025 AS DOUBLE)), ('q5_all', 2, NULL, NULL, NULL, 'h24', NULL, CAST(1253.7359999999899 AS DOUBLE)), ('q5_all', 2, NULL, NULL, NULL, 'h3', NULL, CAST(1188.0 AS DOUBLE)), ('q5_all', 2, NULL, NULL, NULL, 'm30', NULL, CAST(1200.0 AS DOUBLE))) t(tab, lvl, fn, span, cmode, wg, band, v)
),
hq_lcand AS MATERIALIZED (
SELECT x.dev, x.det, x.fn AS fa, true AS own, x.span, x.cmode, x.wg, x.band,
       CASE WHEN x.fn = 'Count' THEN '|' || CASE WHEN x.cmode = 'normal' THEN 'normal' ELSE 'pulse' END ELSE '' END AS cms,
       x.cmode AS cmr
FROM hq_xx0 x
UNION ALL
SELECT x.dev, x.det, a.fa, false, x.span, x.cmode, x.wg, x.band,
       CASE WHEN a.fa = 'Count' THEN '|' || CASE WHEN x.dur_p50 <= 0.25 THEN 'pulse' ELSE 'normal' END ELSE '' END,
       CASE WHEN a.fa <> 'Count' THEN '' WHEN x.dur_p50 IS NULL THEN 'unknown' WHEN x.dur_p50 <= 0.25 THEN 'pulse'
            ELSE 'normal' END
FROM hq_xx0 x, (SELECT unnest(x2.alts) AS fa, x2.dev, x2.det FROM hq_xx0 x2) a
WHERE a.dev = x.dev AND a.det = x.det AND a.fa <> x.fn
),
hq_lval AS (
SELECT c.dev, c.det, c.own, (SELECT first(l.v ORDER BY l.lvl) FROM hq_lref l WHERE l.tab = 'n_ep' AND (l.fn IS NULL OR l.fn = c.fa) AND (l.span IS NULL OR l.span = c.span) AND (l.cmode IS NULL OR l.cmode = c.cmode) AND (l.wg IS NULL OR l.wg = c.wg) AND (l.band IS NULL OR l.band = c.band)) AS n_ep, (SELECT max(l.lo) FROM hq_ltab l WHERE l.tab = 'per_1000' AND l.key = c.fa || '|' || c.span) AS rapid,
       (SELECT max(l.lo) FROM hq_ltab l WHERE l.tab = 'stuck_s' AND l.key = c.fa || '|' || c.span || c.cms) AS st_lo, (SELECT max(l.hi) FROM hq_ltab l WHERE l.tab = 'stuck_s' AND l.key = c.fa || '|' || c.span || c.cms) AS st_hi,
       (SELECT max(l.lo) FROM hq_ltab l WHERE l.tab = 'silent_exp' AND l.key = c.fa || '|' || c.span) AS si_lo, (SELECT max(l.hi) FROM hq_ltab l WHERE l.tab = 'silent_exp' AND l.key = c.fa || '|' || c.span) AS si_hi,
       (SELECT first(l.v ORDER BY l.lvl) FROM hq_lref l WHERE l.tab = 'stuck_x' AND (l.fn IS NULL OR l.fn = c.fa) AND (l.span IS NULL OR l.span = c.span) AND (l.cmode IS NULL OR l.cmode = c.cmr) AND (l.wg IS NULL OR l.wg = c.wg) AND (l.band IS NULL OR l.band = c.band)) AS stuck_x, (SELECT first(l.v ORDER BY l.lvl) FROM hq_lref l WHERE l.tab = 'q5_gy' AND (l.fn IS NULL OR l.fn = c.fa) AND (l.span IS NULL OR l.span = c.span) AND (l.cmode IS NULL OR l.cmode = c.cmode) AND (l.wg IS NULL OR l.wg = c.wg) AND (l.band IS NULL OR l.band = c.band)) AS q5gy, (SELECT first(l.v ORDER BY l.lvl) FROM hq_lref l WHERE l.tab = 'q5_all' AND (l.fn IS NULL OR l.fn = c.fa) AND (l.span IS NULL OR l.span = c.span) AND (l.cmode IS NULL OR l.cmode = c.cmode) AND (l.wg IS NULL OR l.wg = c.wg) AND (l.band IS NULL OR l.band = c.band)) AS q5all
FROM hq_lcand c
),
hq_limits AS MATERIALIZED (
SELECT x.dev, x.det,
  v.n_ep + 1 AS lim_n_ep, (v.n_ep + 1) * CAST(1.6153846153846154 AS DOUBLE) AS bad_n_ep,
  v.rapid AS lim_rapid, v.rapid * CAST(1.1678321678321677 AS DOUBLE) AS bad_rapid,
  CAST(10.0 AS DOUBLE) AS lim_level, CAST(13.076923076923077 AS DOUBLE) AS bad_level,
  CAST(8.0 AS DOUBLE) AS lim_night_drop, CAST(10.461538461538462 AS DOUBLE) AS bad_night_drop,
  CAST(30.0 AS DOUBLE) AS lim_count_on, CAST(159.23076923076923 AS DOUBLE) AS bad_count_on,
  CASE WHEN x.wg = 'h24' THEN greatest(v.st_lo, CAST(300.0 AS DOUBLE))
       ELSE least(greatest(v.sx, 300.0), 900.0) END AS lim_stuck,
  CASE WHEN x.wg = 'h24' THEN greatest(v.st_hi, v.st_lo, CAST(300.0 AS DOUBLE))
       ELSE greatest(CAST(2215.3846153846157 AS DOUBLE) + CAST(0.3846153846153846 AS DOUBLE) * least(greatest(v.sx, 300.0), 900.0), CAST(1.6153846153846154 AS DOUBLE) * least(greatest(v.sx, 300.0), 900.0))
       END AS bad_stuck,
  CASE WHEN x.wg = 'h24' THEN v.si_lo ELSE CAST(30.0 AS DOUBLE) END AS lim_dropout,
  CASE WHEN x.wg = 'h24' THEN greatest(v.si_hi, v.si_lo * CAST(2.435897435897436 AS DOUBLE))
       ELSE CAST(73.07692307692308 AS DOUBLE) END AS bad_dropout,
  CASE WHEN x.wg = 'h24' THEN (SELECT max(l.lo) FROM hq_ltab l WHERE l.tab = 'busy_z' AND l.key = x.fn || '|' || x.span)
       ELSE CAST(6.0 AS DOUBLE) END AS lim_prof,
  CASE WHEN x.wg = 'h24' THEN CAST(2100.0 AS DOUBLE)
       ELSE greatest(CASE WHEN x.fn = 'Advance' THEN v.q5all ELSE v.q5gy END, CAST(1800.0 AS DOUBLE)) END
       AS lim_volume,
  CASE WHEN x.wg = 'h24' THEN CAST(2358.4615384615386 AS DOUBLE)
       ELSE greatest(CASE WHEN x.fn = 'Advance' THEN v.q5all ELSE v.q5gy END, CAST(1800.0 AS DOUBLE))
            * CAST(1.6153846153846154 AS DOUBLE) END AS bad_volume
FROM hq_xx0 x JOIN (
  SELECT dev, det, max(n_ep) AS n_ep, max(rapid) AS rapid, max(st_lo) AS st_lo, max(st_hi) AS st_hi,
         max(si_lo) AS si_lo, max(si_hi) AS si_hi, max(q5gy) AS q5gy, max(q5all) AS q5all,
         CASE WHEN bool_or(own AND stuck_x IS NOT NULL) THEN max(stuck_x) END AS sx
  FROM hq_lval GROUP BY dev, det) v USING (dev, det)
),
hq_todg AS MATERIALIZED (
SELECT * FROM (VALUES ('Advance|*|*', CAST(0.15173982078624734 AS DOUBLE), CAST(-0.008950267020532691 AS DOUBLE), CAST(0.023754368805614715 AS DOUBLE), CAST(0.06262142589183162 AS DOUBLE), false, NULL, NULL, NULL, CAST(0.03174572771301481 AS DOUBLE), CAST(1.145460757596919 AS DOUBLE)), ('Advance|*|high', CAST(-0.060085622633160175 AS DOUBLE), CAST(0.02179707075553361 AS DOUBLE), CAST(0.0787964366937161 AS DOUBLE), CAST(0.059336584971327495 AS DOUBLE), false, NULL, NULL, NULL, CAST(0.05818211040564282 AS DOUBLE), CAST(1.1514647554661972 AS DOUBLE)), ('Advance|*|low', CAST(0.429843513513465 AS DOUBLE), CAST(-0.09257335624294993 AS DOUBLE), CAST(-0.08018886431120373 AS DOUBLE), CAST(0.06899250986662991 AS DOUBLE), false, NULL, NULL, NULL, CAST(0.633891983013079 AS DOUBLE), CAST(1.6513520963758777 AS DOUBLE)), ('Advance|*|medium', CAST(0.08788892766294283 AS DOUBLE), CAST(-0.037390763089143235 AS DOUBLE), CAST(0.04290539696126859 AS DOUBLE), CAST(0.06115243284171432 AS DOUBLE), false, NULL, NULL, NULL, CAST(-0.10502611251391737 AS DOUBLE), CAST(1.1545303633003394 AS DOUBLE)), ('Advance|1|*', CAST(0.16645400681582453 AS DOUBLE), CAST(-0.002720466463951623 AS DOUBLE), CAST(0.01811159041680023 AS DOUBLE), CAST(0.06212616656041871 AS DOUBLE), false, NULL, NULL, NULL, CAST(0.0334730601677664 AS DOUBLE), CAST(1.146549714879889 AS DOUBLE)), ('Advance|1|high', CAST(-0.018242798407099248 AS DOUBLE), CAST(0.03078244616740705 AS DOUBLE), CAST(0.06562238529319123 AS DOUBLE), CAST(0.05839492966521462 AS DOUBLE), false, NULL, NULL, NULL, CAST(0.0698226215758878 AS DOUBLE), CAST(1.1676178229219478 AS DOUBLE)), ('Advance|1|low', CAST(0.4207941785842385 AS DOUBLE), CAST(-0.15413554992801823 AS DOUBLE), CAST(-0.07330254168560896 AS DOUBLE), CAST(0.06857701773815165 AS DOUBLE), false, NULL, NULL, NULL, CAST(0.012137589123372617 AS DOUBLE), CAST(1.117973348903616 AS DOUBLE)), ('Advance|1|medium', CAST(0.08788333523598578 AS DOUBLE), CAST(-0.03702117224048157 AS DOUBLE), CAST(0.04284579069052866 AS DOUBLE), CAST(0.061304516167530146 AS DOUBLE), false, NULL, NULL, NULL, CAST(0.024320741392222768 AS DOUBLE), CAST(1.144480076526912 AS DOUBLE)), ('Advance|2+|*', CAST(0.09144633698832426 AS DOUBLE), CAST(-0.004424479161990791 AS DOUBLE), CAST(0.04741037936817662 AS DOUBLE), CAST(0.07020597063146743 AS DOUBLE), false, NULL, NULL, NULL, CAST(0.04512549842072321 AS DOUBLE), CAST(1.134281855621958 AS DOUBLE)), ('Advance|2+|high', CAST(-0.026715749099756096 AS DOUBLE), CAST(0.011716449974157522 AS DOUBLE), CAST(0.07756425487073167 AS DOUBLE), CAST(0.06835513819663923 AS DOUBLE), false, NULL, NULL, NULL, CAST(0.07520447400912164 AS DOUBLE), CAST(1.20082131837731 AS DOUBLE)), ('Bike|*|*', CAST(0.40781224568091523 AS DOUBLE), CAST(-0.09358035564461113 AS DOUBLE), CAST(-0.060524342987840915 AS DOUBLE), CAST(0.07154913582645438 AS DOUBLE), false, NULL, NULL, NULL, CAST(-0.04352509338698135 AS DOUBLE), CAST(0.5864112993180862 AS DOUBLE)), ('Bike|*|low', CAST(0.5087124372878915 AS DOUBLE), CAST(-0.08887613250204417 AS DOUBLE), CAST(-0.11001817364017681 AS DOUBLE), CAST(0.06622648315697278 AS DOUBLE), false, NULL, NULL, NULL, CAST(-0.25560267278049587 AS DOUBLE), CAST(0.9352518180769596 AS DOUBLE)), ('Bike|1|*', CAST(0.4017589476101453 AS DOUBLE), CAST(-0.10204623178246111 AS DOUBLE), CAST(-0.05684427879028285 AS DOUBLE), CAST(0.07231803822852216 AS DOUBLE), false, NULL, NULL, NULL, CAST(0.595580334897188 AS DOUBLE), CAST(1.2249209483393768 AS DOUBLE)), ('Bike|1|low', CAST(0.4946895376283688 AS DOUBLE), CAST(-0.09212588609752928 AS DOUBLE), CAST(-0.10313248211548881 AS DOUBLE), CAST(0.06632880509086524 AS DOUBLE), false, NULL, NULL, NULL, CAST(-0.03988659881492339 AS DOUBLE), CAST(1.2286655090846559 AS DOUBLE)), ('Count|*|*', CAST(0.19003958030099657 AS DOUBLE), CAST(0.0529178780526209 AS DOUBLE), CAST(0.00673620135189941 AS DOUBLE), CAST(0.061059042453687006 AS DOUBLE), false, NULL, NULL, NULL, CAST(0.027844449758927463 AS DOUBLE), CAST(1.122399992104414 AS DOUBLE)), ('Count|*|high', CAST(0.035745769192944886 AS DOUBLE), CAST(0.03251209376786404 AS DOUBLE), CAST(0.05103636109966034 AS DOUBLE), CAST(0.0580178737676503 AS DOUBLE), false, NULL, NULL, NULL, CAST(0.3966202799981108 AS DOUBLE), CAST(1.112764122929836 AS DOUBLE)), ('Count|*|low', CAST(0.4986516078145814 AS DOUBLE), CAST(-0.015866001205219388 AS DOUBLE), CAST(-0.11803680429156718 AS DOUBLE), CAST(0.06115701340406455 AS DOUBLE), false, NULL, NULL, NULL, CAST(0.8904667245122947 AS DOUBLE), CAST(0.9571649335916316 AS DOUBLE)), ('Count|*|medium', CAST(0.07827920986485343 AS DOUBLE), CAST(0.03577075226882645 AS DOUBLE), CAST(0.03959103481705019 AS DOUBLE), CAST(0.05880862159488926 AS DOUBLE), false, NULL, NULL, NULL, CAST(0.6424614302817576 AS DOUBLE), CAST(2.1290930564999293 AS DOUBLE)), ('Count|1|*', CAST(0.19218625691880395 AS DOUBLE), CAST(0.04484065791121792 AS DOUBLE), CAST(0.006172368490202106 AS DOUBLE), CAST(0.06060351008415702 AS DOUBLE), false, NULL, NULL, NULL, CAST(0.034564889641272196 AS DOUBLE), CAST(1.1294063366737581 AS DOUBLE)), ('Count|1|high', CAST(0.06858230779554558 AS DOUBLE), CAST(0.01288461466233038 AS DOUBLE), CAST(0.042805277042036655 AS DOUBLE), CAST(0.05728856301287724 AS DOUBLE), false, NULL, NULL, NULL, CAST(0.06856639318219858 AS DOUBLE), CAST(1.1574763714949825 AS DOUBLE)), ('Count|1|low', CAST(0.5005560150889997 AS DOUBLE), CAST(-0.01134950572594666 AS DOUBLE), CAST(-0.1191515048255099 AS DOUBLE), CAST(0.060904052961774036 AS DOUBLE), false, NULL, NULL, NULL, CAST(0.07886333989164888 AS DOUBLE), CAST(1.2740093672205983 AS DOUBLE)), ('Count|1|medium', CAST(0.09153376470814327 AS DOUBLE), CAST(0.04224629754348347 AS DOUBLE), CAST(0.034945129392464795 AS DOUBLE), CAST(0.05808075796805089 AS DOUBLE), false, NULL, NULL, NULL, CAST(0.07625773012682 AS DOUBLE), CAST(1.179716329025291 AS DOUBLE)), ('Mid|*|*', CAST(0.006558658183266311 AS DOUBLE), CAST(0.03411398104886831 AS DOUBLE), CAST(0.05808731511934386 AS DOUBLE), CAST(0.05041317739343219 AS DOUBLE), true, CAST(-0.15682224342970982 AS DOUBLE), CAST(0.059211108349187455 AS DOUBLE), CAST(0.05 AS DOUBLE), CAST(0.2656441909637829 AS DOUBLE), CAST(0.7517298154211713 AS DOUBLE)), ('Mid|*|high', CAST(-0.0298259567754991 AS DOUBLE), CAST(0.03236841776987502 AS DOUBLE), CAST(0.06757538713015904 AS DOUBLE), CAST(0.049534958057547186 AS DOUBLE), true, CAST(-0.18182116806767376 AS DOUBLE), CAST(0.06563215569842767 AS DOUBLE), CAST(0.05 AS DOUBLE), CAST(0.27669148743518845 AS DOUBLE), CAST(0.755255964725556 AS DOUBLE)), ('Mid|1|*', CAST(0.11543422902692706 AS DOUBLE), CAST(-0.017096483476999443 AS DOUBLE), CAST(0.03411849689248281 AS DOUBLE), CAST(0.0577177923276541 AS DOUBLE), true, CAST(-0.10696029960690268 AS DOUBLE), CAST(0.04550505540903713 AS DOUBLE), CAST(0.05 AS DOUBLE), CAST(0.22078654959354718 AS DOUBLE), CAST(0.7758848062652253 AS DOUBLE)), ('Mid|1|high', CAST(0.10272865381382125 AS DOUBLE), CAST(-0.02653903096598352 AS DOUBLE), CAST(0.03864877378771238 AS DOUBLE), CAST(0.056220253496234385 AS DOUBLE), true, CAST(-0.10009278236442494 AS DOUBLE), CAST(0.043683017966673746 AS DOUBLE), CAST(0.05 AS DOUBLE), CAST(0.21725537113976226 AS DOUBLE), CAST(0.74003479244995 AS DOUBLE)), ('Mid|2+|*', CAST(-0.038272624433221276 AS DOUBLE), CAST(0.0693999251807908 AS DOUBLE), CAST(0.06610749409601777 AS DOUBLE), CAST(0.04336151893373361 AS DOUBLE), true, CAST(-0.20391371377671944 AS DOUBLE), CAST(0.07153644201891585 AS DOUBLE), CAST(0.05 AS DOUBLE), CAST(0.6686926231145837 AS DOUBLE), CAST(1.0913411254495915 AS DOUBLE)), ('Mid|2+|high', CAST(-0.0831817214274114 AS DOUBLE), CAST(0.06168095932427581 AS DOUBLE), CAST(0.078319541549683 AS DOUBLE), CAST(0.04359517227920162 AS DOUBLE), true, CAST(-0.23114589513623035 AS DOUBLE), CAST(0.07846695067983339 AS DOUBLE), CAST(0.05 AS DOUBLE), CAST(0.3091091060802901 AS DOUBLE), CAST(0.7808458054869823 AS DOUBLE)), ('Other|*|*', CAST(0.187220666442133 AS DOUBLE), CAST(0.06817212047080998 AS DOUBLE), CAST(0.008690191529372684 AS DOUBLE), CAST(0.09422118440510326 AS DOUBLE), true, CAST(-0.16390386603053075 AS DOUBLE), CAST(0.0806533135080866 AS DOUBLE), CAST(0.05 AS DOUBLE), CAST(0.33523674479477494 AS DOUBLE), CAST(1.0881003534015585 AS DOUBLE)), ('Other|*|high', CAST(-0.007365842666773347 AS DOUBLE), CAST(0.17051390198792343 AS DOUBLE), CAST(0.04702571910426542 AS DOUBLE), CAST(0.09803938263243396 AS DOUBLE), true, CAST(-0.6496120741994246 AS DOUBLE), CAST(0.2139543402902668 AS DOUBLE), CAST(0.07368288595376557 AS DOUBLE), CAST(0.39265733347723386 AS DOUBLE), CAST(1.0643836352596387 AS DOUBLE)), ('Other|*|low', CAST(0.39534567400549636 AS DOUBLE), CAST(-0.13599977601395186 AS DOUBLE), CAST(-0.04806344810330898 AS DOUBLE), CAST(0.07550489959734984 AS DOUBLE), true, CAST(-0.05418108153064735 AS DOUBLE), CAST(0.034163683407457 AS DOUBLE), CAST(0.05 AS DOUBLE), CAST(0.3493639807106918 AS DOUBLE), CAST(1.060084425744398 AS DOUBLE)), ('Other|*|medium', CAST(0.1162505314275135 AS DOUBLE), CAST(0.06871292528589065 AS DOUBLE), CAST(0.026843190147239446 AS DOUBLE), CAST(0.09336403910760406 AS DOUBLE), true, CAST(-0.1939226302114991 AS DOUBLE), CAST(0.08840114685529643 AS DOUBLE), CAST(0.05 AS DOUBLE), CAST(0.357028963338099 AS DOUBLE), CAST(1.0690546165166426 AS DOUBLE)), ('Other|1|*', CAST(0.26386577046447907 AS DOUBLE), CAST(0.033958884416966734 AS DOUBLE), CAST(-0.01668689242640284 AS DOUBLE), CAST(0.08453628134463449 AS DOUBLE), true, CAST(-0.11839833408212518 AS DOUBLE), CAST(0.062306707854075195 AS DOUBLE), CAST(0.05 AS DOUBLE), CAST(0.35281174554362715 AS DOUBLE), CAST(1.00711473209805 AS DOUBLE)), ('Other|1|high', CAST(-0.08189313336890199 AS DOUBLE), CAST(0.04862888951710985 AS DOUBLE), CAST(0.07884139314879832 AS DOUBLE), CAST(0.08510306449537527 AS DOUBLE), true, CAST(-0.46099126476020436 AS DOUBLE), CAST(0.15459334980079953 AS DOUBLE), CAST(0.061437693993489646 AS DOUBLE), CAST(0.45325226066622515 AS DOUBLE), CAST(1.0473394357793815 AS DOUBLE)), ('Other|1|low', CAST(0.3810127098751035 AS DOUBLE), CAST(-0.13911461391725508 AS DOUBLE), CAST(-0.04238952861572917 AS DOUBLE), CAST(0.0726445927849767 AS DOUBLE), true, CAST(-0.056702349327102254 AS DOUBLE), CAST(0.03501621581220388 AS DOUBLE), CAST(0.05 AS DOUBLE), CAST(0.15665500634554713 AS DOUBLE), CAST(0.7500097653604548 AS DOUBLE)), ('Other|1|medium', CAST(0.18234656069318178 AS DOUBLE), CAST(0.06521404749264113 AS DOUBLE), CAST(0.0020179608536568944 AS DOUBLE), CAST(0.08502791110203098 AS DOUBLE), true, CAST(-0.16473735745641505 AS DOUBLE), CAST(0.07750522057597516 AS DOUBLE), CAST(0.05 AS DOUBLE), CAST(0.40355690969695207 AS DOUBLE), CAST(1.065518757373279 AS DOUBLE)), ('Other|2+|*', CAST(0.23080195991659355 AS DOUBLE), CAST(0.1650920615949319 AS DOUBLE), CAST(-0.011814078199009287 AS DOUBLE), CAST(0.10004788851074539 AS DOUBLE), true, CAST(-0.2296726884198655 AS DOUBLE), CAST(0.10582724473068829 AS DOUBLE), CAST(0.058698382913588684 AS DOUBLE), CAST(0.42348291218222417 AS DOUBLE), CAST(1.1935454337548608 AS DOUBLE)), ('Other|2+|high', CAST(0.1752461452834641 AS DOUBLE), CAST(0.26184774969281654 AS DOUBLE), CAST(-0.013508127427021316 AS DOUBLE), CAST(0.09687189014248929 AS DOUBLE), true, CAST(-0.6759119026801298 AS DOUBLE), CAST(0.22610958749639995 AS DOUBLE), CAST(0.07743443412139556 AS DOUBLE), CAST(0.42380136459072126 AS DOUBLE), CAST(1.175513069296313 AS DOUBLE)), ('Other|2+|medium', CAST(0.07629371541035773 AS DOUBLE), CAST(0.09161216119323154 AS DOUBLE), CAST(0.04944513679434158 AS DOUBLE), CAST(0.1033039334985789 AS DOUBLE), true, CAST(-0.22410239534179155 AS DOUBLE), CAST(0.10194675594083617 AS DOUBLE), CAST(0.05 AS DOUBLE), CAST(0.3991930634250946 AS DOUBLE), CAST(1.255360001459963 AS DOUBLE)), ('Presence|*|*', CAST(0.2502192460685377 AS DOUBLE), CAST(0.05949433572946355 AS DOUBLE), CAST(-0.016424414984183254 AS DOUBLE), CAST(0.07656837149675651 AS DOUBLE), true, CAST(-0.11480989031697753 AS DOUBLE), CAST(0.06395390578843343 AS DOUBLE), CAST(0.05 AS DOUBLE), CAST(0.3717995058282743 AS DOUBLE), CAST(1.109015968878733 AS DOUBLE)), ('Presence|*|high', CAST(-0.0891896573268105 AS DOUBLE), CAST(0.07491452362726315 AS DOUBLE), CAST(0.07753733435743412 AS DOUBLE), CAST(0.07055519062882146 AS DOUBLE), true, CAST(-0.005048039428338359 AS DOUBLE), CAST(0.03383245000690121 AS DOUBLE), CAST(0.05211057214991916 AS DOUBLE), CAST(0.9343452183582355 AS DOUBLE), CAST(1.993039701797683 AS DOUBLE)), ('Presence|*|low', CAST(0.48449222091579347 AS DOUBLE), CAST(-0.0006847493792134507 AS DOUBLE), CAST(-0.10318911473755264 AS DOUBLE), CAST(0.07359993970700757 AS DOUBLE), true, CAST(-0.05679138307519776 AS DOUBLE), CAST(0.03647523365382952 AS DOUBLE), CAST(0.05 AS DOUBLE), CAST(2.4571791978344595 AS DOUBLE), CAST(2.0361413579602585 AS DOUBLE)), ('Presence|*|medium', CAST(0.1981207029915537 AS DOUBLE), CAST(0.10278162806220706 AS DOUBLE), CAST(-0.011980581646386615 AS DOUBLE), CAST(0.0797580240272935 AS DOUBLE), true, CAST(-0.2238191357107874 AS DOUBLE), CAST(0.1012469159264263 AS DOUBLE), CAST(0.05 AS DOUBLE), CAST(0.9726734841315352 AS DOUBLE), CAST(1.7189869404778266 AS DOUBLE)), ('Presence|1|*', CAST(0.25151978721689705 AS DOUBLE), CAST(0.05523050610820004 AS DOUBLE), CAST(-0.016268604944829332 AS DOUBLE), CAST(0.07582749054151008 AS DOUBLE), true, CAST(-0.11496129930777109 AS DOUBLE), CAST(0.06388466592142579 AS DOUBLE), CAST(0.05 AS DOUBLE), CAST(0.3672684800161549 AS DOUBLE), CAST(1.10021702867134 AS DOUBLE)), ('Presence|1|high', CAST(-0.09314527574961404 AS DOUBLE), CAST(0.0693237347919046 AS DOUBLE), CAST(0.07949526586755248 AS DOUBLE), CAST(0.06980665842207774 AS DOUBLE), true, CAST(-0.019214637528821053 AS DOUBLE), CAST(0.03744885498760467 AS DOUBLE), CAST(0.05126933719991199 AS DOUBLE), CAST(0.45862511917246995 AS DOUBLE), CAST(1.2610781918820217 AS DOUBLE)), ('Presence|1|low', CAST(0.4819006980211764 AS DOUBLE), CAST(-0.0010525269636757345 AS DOUBLE), CAST(-0.10225213704833289 AS DOUBLE), CAST(0.07322103612231183 AS DOUBLE), true, CAST(-0.05656734936548297 AS DOUBLE), CAST(0.03634458337613739 AS DOUBLE), CAST(0.05 AS DOUBLE), CAST(0.23826639701202298 AS DOUBLE), CAST(0.7964806912807875 AS DOUBLE)), ('Presence|1|medium', CAST(0.20632793332648308 AS DOUBLE), CAST(0.09653017040493016 AS DOUBLE), CAST(-0.013561830590275673 AS DOUBLE), CAST(0.07852763657870865 AS DOUBLE), true, CAST(-0.22063535441698146 AS DOUBLE), CAST(0.10010186832837967 AS DOUBLE), CAST(0.05 AS DOUBLE), CAST(0.3840904782943305 AS DOUBLE), CAST(1.1006648230200338 AS DOUBLE)), ('Presence|2+|*', CAST(0.25804999830432424 AS DOUBLE), CAST(0.2515138425849993 AS DOUBLE), CAST(-0.04593662706232321 AS DOUBLE), CAST(0.08885289693364809 AS DOUBLE), true, CAST(-0.08183533664437737 AS DOUBLE), CAST(0.060864180389172885 AS DOUBLE), CAST(0.07919519362514602 AS DOUBLE), CAST(0.46819829803677104 AS DOUBLE), CAST(1.2118945265307697 AS DOUBLE)), ('Yellow_Red|*|*', CAST(0.1620860117538036 AS DOUBLE), CAST(0.14260713757282656 AS DOUBLE), CAST(0.008742034016688327 AS DOUBLE), CAST(0.060872020889432096 AS DOUBLE), false, NULL, NULL, NULL, CAST(0.024020935979068545 AS DOUBLE), CAST(1.1444026816231958 AS DOUBLE)), ('Yellow_Red|*|high', CAST(-0.28055057982114545 AS DOUBLE), CAST(0.014061645341303136 AS DOUBLE), CAST(0.13741111105314674 AS DOUBLE), CAST(0.05109358452526777 AS DOUBLE), false, NULL, NULL, NULL, CAST(0.012131706376501914 AS DOUBLE), CAST(1.1913616612915927 AS DOUBLE)), ('Yellow_Red|*|low', CAST(0.5584818693503538 AS DOUBLE), CAST(-0.1556080871997609 AS DOUBLE), CAST(-0.14668832905130327 AS DOUBLE), CAST(0.060294388585322156 AS DOUBLE), false, NULL, NULL, NULL, CAST(0.1328589149619169 AS DOUBLE), CAST(1.0536148135579575 AS DOUBLE)), ('Yellow_Red|*|medium', CAST(0.115193053472864 AS DOUBLE), CAST(0.09962478808329078 AS DOUBLE), CAST(0.017158496964970613 AS DOUBLE), CAST(0.0541634696435394 AS DOUBLE), false, NULL, NULL, NULL, CAST(0.3738867105163734 AS DOUBLE), CAST(1.029276799992392 AS DOUBLE)), ('Yellow_Red|1|*', CAST(0.280531630837403 AS DOUBLE), CAST(0.09799321347990052 AS DOUBLE), CAST(-0.034704258979101524 AS DOUBLE), CAST(0.058688512766720775 AS DOUBLE), false, NULL, NULL, NULL, CAST(0.1793421737061065 AS DOUBLE), CAST(1.261764120412862 AS DOUBLE)), ('Yellow_Red|1|low', CAST(0.5423644152087175 AS DOUBLE), CAST(-0.15514409074458313 AS DOUBLE), CAST(-0.13983882024768357 AS DOUBLE), CAST(0.06278813855920973 AS DOUBLE), false, NULL, NULL, NULL, CAST(0.1477945676443005 AS DOUBLE), CAST(0.9961628067951139 AS DOUBLE)), ('Yellow_Red|1|medium', CAST(0.16091211885297177 AS DOUBLE), CAST(0.12210175897292902 AS DOUBLE), CAST(0.00022199154857840964 AS DOUBLE), CAST(0.055249290027506344 AS DOUBLE), false, NULL, NULL, NULL, CAST(0.13261934320341195 AS DOUBLE), CAST(1.1710440957243746 AS DOUBLE)), ('Yellow_Red|2+|*', CAST(0.006532708658519865 AS DOUBLE), CAST(0.03421208191762244 AS DOUBLE), CAST(0.06259064876126087 AS DOUBLE), CAST(0.05078077076247053 AS DOUBLE), false, NULL, NULL, NULL, CAST(1.2772542901051334 AS DOUBLE), CAST(2.3016582445115974 AS DOUBLE)), ('Yellow_Red|2+|high', CAST(-0.20339571657906885 AS DOUBLE), CAST(0.003958543133712578 AS DOUBLE), CAST(0.1195732911272396 AS DOUBLE), CAST(0.04862105447190484 AS DOUBLE), false, NULL, NULL, NULL, CAST(0.01200146514527289 AS DOUBLE), CAST(1.1347814874782092 AS DOUBLE))) t(key, b0, b1, b2, sc, occ, o0, o1, so, med, mad)
),
hq_tdays AS MATERIALIZED (
WITH w AS (SELECT dev, s_ns, e_ns, nbb, T, s_ns - sod + CASE WHEN sod > 0 THEN 86400000000000 ELSE 0 END AS f0
           FROM hq_win WHERE wg = 'h24' AND bs = 900),
d AS (SELECT dev, s_ns, nbb, f0 + unnest(range((e_ns - f0) // 86400000000000)) * 86400000000000 AS d0 FROM w
      WHERE e_ns >= f0 + 86400000000000
      UNION ALL SELECT dev, s_ns, nbb, s_ns FROM w WHERE e_ns < f0 + 86400000000000 AND T >= 86400)
SELECT dev, s_ns, nbb, d0, round_even((d0 - s_ns) / 9e11, 0)::INT AS j0,
       ((d0 % 86400000000000) // 1000000000 // 900) % 96 AS sh
FROM d WHERE round_even((d0 - s_ns) / 9e11, 0) >= 0 AND round_even((d0 - s_ns) / 9e11, 0) + 96 <= nbb
),
hq_tdh AS (
SELECT dev, det, d0, hr, CASE WHEN count(H) = 2 THEN sum(H) END AS N, avg(Q) AS O FROM (
  SELECT g.dev, g.det, t.d0, ((g.k - t.j0 + t.sh) % 96) // 2 AS hh, ((g.k - t.j0 + t.sh) % 96) // 4 AS hr,
         CASE WHEN count(g.n) > 0 THEN sum(g.n) END AS H, avg(g.occ) AS Q
  FROM hq_tdays t JOIN hq_gb g ON g.dev = t.dev AND g.k >= t.j0 AND g.k < t.j0 + 96
  JOIN hq_xbase x ON x.dev = g.dev AND x.det = g.det GROUP BY g.dev, g.det, t.d0, hh, hr)
GROUP BY dev, det, d0, hr
),
hq_tdf AS (
SELECT a.*, sqrt((a.nn + 0.5) / (b.b4n + 0.5)) AS r, asin(sqrt(a.onm)) AS onv,
       asin(sqrt(least(greatest(b.b4o, 0.0), 1.0))) AS od, log10(1 + a.tot) AS lv,
       CASE WHEN a.tot / 24.0 < 20 THEN 'low' WHEN a.tot / 24.0 < 100 THEN 'medium' ELSE 'high' END AS band
FROM (SELECT dev, det, d0, sum(N) AS tot, avg(N) FILTER (hr >= 1 AND hr < 5) AS nn, sum(N) FILTER (hr >= 1 AND hr < 5)
             AS n_night, avg(least(greatest(O, 0.0), 1.0)) FILTER (hr >= 1 AND hr < 5) AS onm
      FROM hq_tdh GROUP BY dev, det, d0 HAVING count(N) = 24 AND count(O) = 24 AND sum(N) >= 50) a
JOIN (SELECT a.dev, a.det, a.d0, max(s4n) / 4 AS b4n, max(s4o) / 4 AS b4o FROM (
        SELECT a.dev, a.det, a.d0, a.hr, sum(b.N) AS s4n, sum(least(greatest(b.O, 0.0), 1.0)) AS s4o
        FROM hq_tdh a JOIN hq_tdh b ON b.dev = a.dev AND b.det = a.det AND b.d0 = a.d0 AND (b.hr - a.hr + 24) % 24 < 4
        GROUP BY a.dev, a.det, a.d0, a.hr) a GROUP BY a.dev, a.det, a.d0) b USING (dev, det, d0)
),
hq_tdz AS (
SELECT f.dev, f.det, f.d0, c.own, ((greatest((f.r - (g.b0 + g.b1 * f.od + g.b2 * f.lv)) / g.sc,
       CASE WHEN g.occ THEN (f.onv - (g.o0 + g.o1 * f.lv)) / g.so END)) - g.med) / g.mad AS z
FROM hq_tdf f JOIN (SELECT dev, det, fa, own FROM hq_lcand) c USING (dev, det) JOIN hq_xbase x USING (dev, det)
JOIN (SELECT f.dev, f.det, f.d0, c.fa, first(g.key ORDER BY o.o) AS key
      FROM hq_tdf f JOIN hq_lcand c USING (dev, det) JOIN hq_xbase x USING (dev, det),
      (SELECT 1 AS o UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4) o, hq_todg g
      WHERE g.key = CASE o.o WHEN 1 THEN c.fa || '|' || x.span || '|' || f.band WHEN 2 THEN c.fa || '|*|' || f.band
                             WHEN 3 THEN c.fa || '|' || x.span || '|*' ELSE c.fa || '|*|*' END
      GROUP BY f.dev, f.det, f.d0, c.fa) k ON k.dev = f.dev AND k.det = f.det AND k.d0 = f.d0 AND k.fa = c.fa
JOIN hq_todg g ON g.key = k.key
),
hq_tdd AS (
SELECT f.*, z.z_own AS tod_z0, CASE WHEN z.z_own IS NOT NULL THEN least(z.z_own, z.z_alt) END AS tod_z,
       f.r - m.mate_r AS tod_dph, m.mate_r AS tod_mate_r, l.lim_prof
FROM hq_tdf f JOIN (SELECT dev, det, d0, max(z) FILTER (own) AS z_own, min(z) FILTER (NOT own) AS z_alt FROM hq_tdz
                 GROUP BY dev, det, d0) z USING (dev, det, d0)
JOIN hq_xbase x USING (dev, det) JOIN hq_limits l USING (dev, det)
LEFT JOIN (SELECT a.dev, a.det, a.d0, median(b.r) AS mate_r FROM hq_tdf a JOIN hq_xbase xa USING (dev, det)
           JOIN hq_xbase xb2 ON xb2.dev = a.dev AND xb2.ph = xa.ph AND xb2.det <> a.det
           JOIN hq_tdf b ON b.dev = a.dev AND b.det = xb2.det AND b.d0 = a.d0
           JOIN (SELECT dev, det, d0, max(z) FILTER (own) AS zo FROM hq_tdz GROUP BY dev, det, d0) bz
             ON bz.dev = b.dev AND bz.det = b.det AND bz.d0 = b.d0
           WHERE bz.zo IS NOT NULL AND b.tot >= 200 GROUP BY a.dev, a.det, a.d0) m USING (dev, det, d0)
),
hq_tdl AS (
SELECT *, CASE WHEN tod_z IS NULL THEN NULL
               WHEN n_night >= CAST(20.0 AS DOUBLE) AND coalesce(tod_dph, -1) >= CAST(0.3 AS DOUBLE)
                    AND tod_z >= lim_prof * CAST(0.5833333333333334 AS DOUBLE) AND tod_z >= lim_prof THEN 0.8
               WHEN n_night >= CAST(20.0 AS DOUBLE) AND coalesce(tod_dph, -1) >= CAST(0.3 AS DOUBLE)
                    AND tod_z >= lim_prof * CAST(0.5833333333333334 AS DOUBLE) THEN 0.5 ELSE 0.0 END AS lvl
FROM hq_tdd
),
hq_tod AS (
SELECT dev, det, first(tod_z ORDER BY coalesce(lvl, -1) DESC, d0) AS tod_z,
       first(tod_dph ORDER BY coalesce(lvl, -1) DESC, d0) AS tod_dph,
       first(n_night ORDER BY coalesce(lvl, -1) DESC, d0) AS tod_n_night,
       first(r ORDER BY coalesce(lvl, -1) DESC, d0) AS tod_r, first(tod_mate_r ORDER BY coalesce(lvl, -1) DESC, d0) AS tod_mate_r,
       first(d0 ORDER BY coalesce(lvl, -1) DESC, d0) AS tod_d0
FROM hq_tdl GROUP BY dev, det
),
hq_todnote AS (
SELECT dev, first(d0 ORDER BY n_odd / greatest(n_ok, 1) DESC, d0) AS d0, first(n_odd ORDER BY n_odd / greatest(n_ok, 1)
       DESC, d0) AS n_odd, first(n_ok ORDER BY n_odd / greatest(n_ok, 1) DESC, d0) AS n_ok
FROM (SELECT dev, d0, count(*) FILTER (tod_z >= 3.5) AS n_odd, count(*) AS n_ok FROM hq_tdd GROUP BY dev, d0)
WHERE n_odd >= 5 AND n_odd >= 0.20 * n_ok GROUP BY dev
),
hq_flatm AS MATERIALIZED (
SELECT *, coalesce(occ_d >= 900 AND occ_d >= 3 * occ_n AND own_d < 0.8 * own_n, false) AS qzone FROM (
WITH t AS (SELECT dev, hid, sum(M) AS S FROM hq_mh GROUP BY dev, hid),
pm AS (SELECT m.dev, z.det, m.hid, sum(m.M) AS P FROM hq_mh m JOIN hq_dz zm USING (dev, det)
       JOIN hq_dz z ON z.dev = m.dev AND z.ph = zm.ph AND z.det <> m.det WHERE z.ph IS NOT NULL GROUP BY m.dev, z.det, m.hid),
r AS (SELECT m.dev, m.det, x.hod, avg(m.M) AS own, avg(m.O) AS occ, avg(coalesce(p.P, t.S - m.M)) AS ref
      FROM hq_mh m JOIN hq_hix x USING (dev, hid) JOIN t USING (dev, hid) LEFT JOIN pm p USING (dev, det, hid)
      WHERE x.ch GROUP BY m.dev, m.det, x.hod)
SELECT dev, det, list_avg(list(own ORDER BY hod) FILTER (hod >= 7 AND hod < 19)) AS own_d,
       list_avg(list(own ORDER BY hod) FILTER (hod >= 1 AND hod < 5)) AS own_n,
       list_avg(list(ref ORDER BY hod) FILTER (hod >= 7 AND hod < 19)) AS rest_d,
       list_avg(list(ref ORDER BY hod) FILTER (hod >= 1 AND hod < 5)) AS rest_n,
       avg(occ) FILTER (hod >= 7 AND hod < 19) AS occ_d, avg(occ) FILTER (hod >= 1 AND hod < 5) AS occ_n,
       corr(own, ref) AS prof_corr, count(*) AS n_hod
FROM r GROUP BY dev, det)
),
hq_flat AS MATERIALIZED (
SELECT * FROM hq_flatm
WHERE n_hod >= 20 AND rest_d >= 100 AND rest_n <= 0.3 * rest_d AND own_d >= 1.5 AND NOT qzone
  AND ((own_n >= 0.8 * own_d AND coalesce(prof_corr, 0) < 0.6)
       OR (own_d < 30 AND own_n >= greatest(0.3, 2.5 * rest_n / rest_d) * own_d AND coalesce(prof_corr, 0) < 0.8))
),
hq_xx AS MATERIALIZED (
SELECT x.*, t.tod_z, t.tod_dph, t.tod_n_night, t.tod_r, t.tod_mate_r, t.tod_d0,
f.det IS NOT NULL AS hq_flat, f.own_d AS flat_d, f.own_n AS flat_n, f.rest_d AS flat_rd, f.rest_n AS flat_rn, coalesce(fm.qzone, false) AS qzone,
       x.fn = 'Count' AND (len(x.alts) = 0 OR list_has_all(['Count'], x.alts)) AND NOT x.c1 AS count_on_ok,
       x.c1 AS c1_pre, CASE WHEN x.c1 THEN 'C1' WHEN x.c2 THEN 'C2' ELSE '' END AS cfg,
       x.fn IN ('Advance', 'Count', 'Yellow_Red') AND NOT list_has_any(x.alts, ['Presence', 'Other', 'Mid']) AS noq,
       CASE WHEN x.fn = 'Advance' THEN x.q5_all ELSE x.q5_gy END AS q5_use
FROM (SELECT x.*, coalesce(x.fn = 'Count' AND ((x.rep_time_s >= 600 AND x.rep_time_s >= 0.70 * x.occ * x.hours * 3600)
                           OR (x.n_rep >= 20 AND x.n_rep >= 0.05 * x.n_on)), false) AS c1,
             coalesce(x.fn = 'Count' AND x.cmode = 'pulse' AND (x.n_ge60 >= 1 OR x.n_ge5 >= 3), false) AS c2
      FROM hq_xx0 x) x
LEFT JOIN hq_tod t USING (dev, det)
LEFT JOIN hq_flat f USING (dev, det) LEFT JOIN hq_flatm fm USING (dev, det)
),
hq_ef AS MATERIALIZED (
SELECT e.*, e.q_ok AND e.dur < 3600 AND NOT e.common AS q1,
       e.q_ok AND e.dur >= 3600 AND coalesce(e.cover >= 0.75, false) AND NOT e.lifted0 AND NOT e.common AS q1b,
       e.q_ok AND e.lifted0 AS lifted,
       CASE WHEN e.fn IN ('Advance', 'Count', 'Yellow_Red') THEN 0 ELSE e.co5 END >= 3 AND NOT e.common AS shared
FROM (SELECT e.*, coalesce(e.q_ok0 AND e.dur >= 3600 AND e.cover >= 0.75 AND e.busy_min_e >= 60 AND e.busy_occ_e < 0.5
                           AND NOT e.ref_held_e, false) AS lifted0, coalesce(e.q_ok0, false) AS q_ok
      FROM (SELECT e.*, x.fn, coalesce(CASE WHEN x.noq THEN 0 ELSE e.n_hpeer_e END, 0) > 0 AND e.phx >= 1.5
                   AND e.corr_h_e >= 0.70 AND e.refx_e >= 0.5 AND coalesce(e.light_e, 1) = 0 AND e.trafx_e >= 0.5 AS q_ok0
            FROM (SELECT e.*, e.cover_e AS cover, CASE WHEN coalesce(e.cover_e, 0) >= 0.75
                         THEN greatest(coalesce(e.phx_h_e, 0), 1.5) ELSE e.phx_h_e END AS phx FROM hq_ectx e) e
            LEFT JOIN hq_xx x USING (dev, det)) e) e
),
hq_efl AS MATERIALIZED (
SELECT e.*, l.lim_stuck, CASE WHEN e.common THEN 'together' WHEN e.q1 OR e.q1b THEN 'queue'
       WHEN e.shared THEN 'shared' WHEN e.lifted THEN 'not_queue' ELSE 'counted' END AS label,
       NOT e.q1 AND NOT e.q1b AND NOT e.shared AS counted
FROM hq_ef e JOIN hq_limits l USING (dev, det) WHERE e.dur >= l.lim_stuck
),
hq_stk AS MATERIALIZED (
SELECT x.dev, x.det, CASE WHEN coalesce(e.n_common, 0) > 0 THEN 1.0
  WHEN coalesce(e.n_ep, 0) > 0 AND e.n_ep = e.n_lift AND sv > CAST(0.35 AS DOUBLE) THEN 0.35
  WHEN coalesce(e.n_over, 0) > 0 AND coalesce(e.n_ep, 0) = 0 AND coalesce(e.n_sh, 0) = 0 AND e.n_q > 0 THEN 0.0
  WHEN coalesce(e.n_ep, 0) = 0 AND coalesce(e.n_sh, 0) > 0 THEN 0.35 ELSE sv END AS s_stuck_v,
  coalesce(e.n_ep, 0) AS st_n_ep, coalesce(e.tot, 0) AS st_tot_s
FROM (SELECT x.dev, x.det, CASE WHEN x.s_stuck_u = 0.35
                                    AND coalesce(e.n_ep, 0) <= 1 AND s0 > 0.35 THEN 0.35 ELSE s0 END AS sv
      FROM (SELECT x.*, CASE WHEN st IS NULL THEN coalesce(sn, 0) ELSE greatest(st, coalesce(sn, 0)) END AS s0 FROM (
              SELECT x.dev, x.det, x.s_stuck, x.s_stuck_u,
                     (CASE WHEN (coalesce(e.tot, 0.0)) IS NULL OR isnan(coalesce(e.tot, 0.0)) OR (l.lim_stuck) IS NULL OR isnan(l.lim_stuck) THEN NULL WHEN (coalesce(e.tot, 0.0)) < (l.lim_stuck) THEN 0.0 ELSE least(0.35 + 0.40 * greatest(((coalesce(e.tot, 0.0)) - (l.lim_stuck)) / greatest((l.bad_stuck) - (l.lim_stuck), 1e-9), 0.0), 1.0) END) AS st,
                     (CASE WHEN (coalesce(e.n_ep, 0)::DOUBLE) IS NULL OR isnan(coalesce(e.n_ep, 0)::DOUBLE) OR (l.lim_n_ep) IS NULL OR isnan(l.lim_n_ep) THEN NULL WHEN (coalesce(e.n_ep, 0)::DOUBLE) < (l.lim_n_ep) THEN 0.0 ELSE least(0.35 + 0.40 * greatest(((coalesce(e.n_ep, 0)::DOUBLE) - (l.lim_n_ep)) / greatest((l.bad_n_ep) - (l.lim_n_ep), 1e-9), 0.0), 1.0) END) AS sn
              FROM hq_xx x JOIN hq_limits l USING (dev, det) LEFT JOIN (
                SELECT dev, det, count(*) FILTER (counted) AS n_ep, sum(dur) FILTER (counted) AS tot FROM hq_efl
                GROUP BY dev, det) e USING (dev, det)) x) x
      LEFT JOIN (SELECT dev, det, count(*) FILTER (counted) AS n_ep FROM hq_efl GROUP BY dev, det) e USING (dev, det)) x
LEFT JOIN (SELECT dev, det, count(*) AS n_over, count(*) FILTER (counted) AS n_ep, sum(dur) FILTER (counted) AS tot,
                  count(*) FILTER (q1 OR q1b) AS n_q, count(*) FILTER (shared AND NOT q1 AND NOT q1b) AS n_sh,
                  count(*) FILTER (counted AND lifted) AS n_lift, count(*) FILTER (common) AS n_common
           FROM hq_efl GROUP BY dev, det) e USING (dev, det)
),
hq_preS AS MATERIALIZED (
SELECT x.dev, x.det, s.s_stuck_v AS v_stuck,
  CASE WHEN x.count_on_ok THEN (CASE WHEN (coalesce(x.hi_min_c, 0)) IS NULL OR isnan(coalesce(x.hi_min_c, 0)) OR (l.lim_count_on) IS NULL OR isnan(l.lim_count_on) THEN NULL WHEN (coalesce(x.hi_min_c, 0)) < (l.lim_count_on) THEN 0.0 ELSE least(0.35 + 0.40 * greatest(((coalesce(x.hi_min_c, 0)) - (l.lim_count_on)) / greatest((l.bad_count_on) - (l.lim_count_on), 1e-9), 0.0), 1.0) END) END AS v_count_on,
  CASE WHEN x.fn = 'Bike' THEN NULL WHEN x.silent THEN 0.0 
       WHEN x.drop_b1 - x.drop_b0 < 3 THEN least(x.sd, 0.5) ELSE x.sd END AS v_dropout,
  CASE WHEN x.fn = 'Bike' OR NOT x.geh_ok THEN NULL 
WHEN x.geh_drop IS NOT NULL AND x.occ_own >= 3.0 AND x.occ_own >= 3.0 * coalesce(x.occ_mates, 1.0) THEN (CASE WHEN (x.geh_drop) IS NULL OR isnan(x.geh_drop) OR (l.lim_level) IS NULL OR isnan(l.lim_level) THEN NULL WHEN (x.geh_drop) < (l.lim_level) THEN 0.0 ELSE least(0.35 + 0.40 * greatest(((x.geh_drop) - (l.lim_level)) / greatest((l.bad_level) - (l.lim_level), 1e-9), 0.0), 1.0) END) WHEN x.occ_own >= 1.0 AND coalesce(x.occ_after, 0) >= 0.25 THEN 0.0
       WHEN x.geh_drop IS NOT NULL AND ((x.occ_mates IS NOT NULL AND x.occ_own / greatest(x.occ_mates, 1e-9) < 0.8)
            OR (x.occ_mates IS NULL AND x.occ_own < 0.8) OR x.own_after < 0.1 * x.exp_after)
       THEN (CASE WHEN (x.geh_drop) IS NULL OR isnan(x.geh_drop) OR (l.lim_level) IS NULL OR isnan(l.lim_level) THEN NULL WHEN (x.geh_drop) < (l.lim_level) THEN 0.0 ELSE least(0.35 + 0.40 * greatest(((x.geh_drop) - (l.lim_level)) / greatest((l.bad_level) - (l.lim_level), 1e-9), 0.0), 1.0) END) ELSE 0.0 END AS v_lvl_old,
  CASE WHEN x.fn = 'Bike' THEN NULL  WHEN x.md_h IS NOT NULL THEN (CASE WHEN (x.md_geh) IS NULL OR isnan(x.md_geh) OR (CAST(15.0 AS DOUBLE)) IS NULL OR isnan(CAST(15.0 AS DOUBLE)) THEN NULL WHEN (x.md_geh) < (CAST(15.0 AS DOUBLE)) THEN 0.0 ELSE least(0.35 + 0.40 * greatest(((x.md_geh) - (CAST(15.0 AS DOUBLE))) / greatest((CAST(20.0 AS DOUBLE)) - (CAST(15.0 AS DOUBLE)), 1e-9), 0.0), 1.0) END) WHEN x.mh_ok THEN 0.0 END AS v_lvl_md,
  CASE WHEN v_lvl_old IS NULL AND v_lvl_md IS NULL THEN NULL
       ELSE greatest(coalesce(v_lvl_old, 0.0), coalesce(v_lvl_md, 0.0)) END AS v_level,
  CASE WHEN x.fn = 'Bike' THEN NULL  WHEN x.mr_h IS NOT NULL THEN (CASE WHEN (x.mr_geh) IS NULL OR isnan(x.mr_geh) OR (CAST(15.0 AS DOUBLE)) IS NULL OR isnan(CAST(15.0 AS DOUBLE)) THEN NULL WHEN (x.mr_geh) < (CAST(15.0 AS DOUBLE)) THEN 0.0 ELSE least(0.35 + 0.40 * greatest(((x.mr_geh) - (CAST(15.0 AS DOUBLE))) / greatest((CAST(20.0 AS DOUBLE)) - (CAST(15.0 AS DOUBLE)), 1e-9), 0.0), 1.0) END) WHEN x.mh_ok THEN 0.0 END AS v_rise,
  CASE WHEN x.fn = 'Bike' THEN NULL 
       WHEN x.moved AND coalesce(x.drop_lam, 0) < 50 THEN 0.0
       WHEN x.moved AND coalesce(x.drop_lam, 0) < 100 THEN least(x.sd, 0.5)
       WHEN x.moved THEN x.sd ELSE x.nd END AS v_night_drop,
  CASE WHEN x.fn <> 'Bike' AND coalesce(CASE WHEN x.silent THEN 0.0 ELSE x.sd END, 0)
            >= 0.42
       THEN CASE WHEN x.pv IS NOT NULL THEN 0.0 END ELSE x.pv END AS v_prof,
  CASE WHEN x.N >= 50 THEN (CASE WHEN (x.xr + 1e-9 * (x.xr > 0)::INT) IS NULL OR isnan(x.xr + 1e-9 * (x.xr > 0)::INT) OR (1.0) IS NULL OR isnan(1.0) THEN NULL WHEN (x.xr + 1e-9 * (x.xr > 0)::INT) < (1.0) THEN 0.0 ELSE least(0.35 + 0.40 * greatest(((x.xr + 1e-9 * (x.xr > 0)::INT) - (1.0)) / greatest((l.bad_rapid / l.lim_rapid) - (1.0), 1e-9), 0.0), 1.0) END) END AS v_rapid,
  CASE WHEN x.q5_use IS NOT NULL THEN (CASE WHEN (x.q5_use / l.lim_volume) IS NULL OR isnan(x.q5_use / l.lim_volume) OR (1.0) IS NULL OR isnan(1.0) THEN NULL WHEN (x.q5_use / l.lim_volume) < (1.0) THEN 0.0 ELSE least(0.35 + 0.40 * greatest(((x.q5_use / l.lim_volume) - (1.0)) / greatest((l.bad_volume / l.lim_volume) - (1.0), 1e-9), 0.0), 1.0) END) END
       AS v_volume,
  CASE WHEN x.spk_lvl >= 2 THEN 1.0 WHEN x.spk_lvl >= 1 THEN 0.5 WHEN x.spk_lvl IS NOT NULL THEN 0.0 END AS v_spiky,
  x.silent AS silent_night
FROM (SELECT x.*, coalesce(x.sd0, 0) >= 0.35 AND x.nf >= 0.8 AS silent,
             coalesce(x.sd0, 0) >= 0.35 AND x.nf >= 0.8 AND coalesce(x.sd0, 0) > coalesce(x.nd, 0) + 1e-9 AS moved,
             x.sd0 AS sd
      FROM (SELECT x.*, CASE WHEN x.n_on > 0 THEN CASE WHEN x.drop_in_stuck THEN 0.0
 ELSE (CASE WHEN (x.drop_lam) IS NULL OR isnan(x.drop_lam) OR (l.lim_dropout) IS NULL OR isnan(l.lim_dropout) THEN NULL WHEN (x.drop_lam) < (l.lim_dropout) THEN 0.0 ELSE least(0.35 + 0.40 * greatest(((x.drop_lam) - (l.lim_dropout)) / greatest((l.bad_dropout) - (l.lim_dropout), 1e-9), 0.0), 1.0) END) END END AS sd0,
                   (CASE WHEN (x.geh_night) IS NULL OR isnan(x.geh_night) OR (l.lim_night_drop) IS NULL OR isnan(l.lim_night_drop) THEN NULL WHEN (x.geh_night) < (l.lim_night_drop) THEN 0.0 ELSE least(0.35 + 0.40 * greatest(((x.geh_night) - (l.lim_night_drop)) / greatest((l.bad_night_drop) - (l.lim_night_drop), 1e-9), 0.0), 1.0) END) AS nd,
                   CASE WHEN x.hq_flat THEN 0.8 WHEN x.qzone THEN 0.0 WHEN x.tod_z IS NULL THEN NULL
                        WHEN x.tod_n_night >= CAST(20.0 AS DOUBLE) AND coalesce(x.tod_dph, -1) >= CAST(0.3 AS DOUBLE)
                             AND x.tod_z >= l.lim_prof * CAST(0.5833333333333334 AS DOUBLE) AND x.tod_z >= l.lim_prof
                             AND x.tod_z >= 1.5 * l.lim_prof THEN 0.8
                        WHEN x.tod_n_night >= CAST(20.0 AS DOUBLE) AND coalesce(x.tod_dph, -1) >= CAST(0.3 AS DOUBLE)
                             AND x.tod_z >= l.lim_prof * CAST(0.5833333333333334 AS DOUBLE) THEN 0.5 ELSE 0.0 END AS pv,
                   CASE WHEN x.n_fast >= 3 THEN x.fast_rate / l.lim_rapid ELSE 0.0 END AS xr,
                   CASE WHEN x.drop_b1 > x.drop_b0 THEN (SELECT count(*) FILTER (b.hr >= 21 OR b.hr < 5) FROM hq_bins5 b
                        WHERE b.dev = x.dev AND b.b >= x.drop_b0 AND b.b < x.drop_b1) / (x.drop_b1 - x.drop_b0)
                        ELSE 0.0 END AS nf
            FROM hq_xx x JOIN hq_limits l USING (dev, det)) x) x
JOIN hq_limits l USING (dev, det) JOIN hq_stk s USING (dev, det)
),
hq_preF0 AS (
SELECT s.*, coalesce(v_stuck >= 0.35, false) AS f_stuck, coalesce(v_count_on >= 0.35, false) AS f_count_on, coalesce(v_dropout >= 0.35, false) AS f_dropout, coalesce(v_level >= 0.35, false) AS f_level, coalesce(v_night_drop >= 0.35, false) AS f_night_drop, coalesce(v_prof >= 0.35, false) AS f_prof, coalesce(v_rapid >= 0.35, false) AS f_rapid, coalesce(v_volume >= 0.35, false) AS f_volume, coalesce(v_spiky >= 0.35, false) AS f_spiky, coalesce(v_rise >= 0.35, false) AS f_rise, x.queue_pat8, x.c_occ_like, x.hours, x.status1, x.status AS status2, x.n_hmates, x.n_on,
       x.drop_b0, x.drop_b1, x.fn, x.c1_pre, x.hi_min_c, x.fe_n, x.fe_phi, x.drop_lam
FROM hq_preS s JOIN hq_xx x USING (dev, det)
),
hq_preF1 AS (
SELECT * REPLACE (
    CASE WHEN f_dropout AND f_level AND NOT (v_dropout >= v_level) THEN false ELSE f_dropout END AS f_dropout,
    CASE WHEN f_dropout AND f_level AND v_dropout >= v_level THEN false ELSE f_level END AS f_level)
FROM (SELECT * REPLACE (f_level AND NOT (queue_pat8 AND coalesce(c_occ_like >= 0.8, false)
                                         AND NOT coalesce(v_lvl_md > v_lvl_old, false)) AS f_level,
                        f_dropout AND NOT (hours < 1) AS f_dropout),
             f_dropout AND hours < 1 AS w_dropout_short,
             status2 <> status1 AND n_hmates = 0 AND status1 IN ('suspect', 'bad') AS w_no_yardstick
      FROM hq_preF0)
),
hq_preF2 AS (
SELECT *, (f_stuck::INT + f_count_on::INT + f_dropout::INT + f_level::INT + f_night_drop::INT + f_prof::INT + f_rapid::INT + f_volume::INT + f_spiky::INT + f_rise::INT) = 1 AND ((f_dropout AND v_dropout < 0.42) OR (f_rapid AND v_rapid < 0.42) OR (f_level AND v_level < 0.42) OR (f_night_drop AND v_night_drop < 0.42) OR (f_prof AND v_prof < 0.42) OR (f_spiky AND v_spiky < 0.42) OR (f_rise AND v_rise < 0.42)) AS r7 FROM hq_preF1
),
hq_preF3 AS (
SELECT * REPLACE (f_stuck AND NOT r7 AS f_stuck, f_count_on AND NOT r7 AS f_count_on, f_dropout AND NOT r7 AS f_dropout, f_level AND NOT r7 AS f_level, f_night_drop AND NOT r7 AS f_night_drop, f_prof AND NOT r7 AS f_prof, f_rapid AND NOT r7 AS f_rapid, f_volume AND NOT r7 AS f_volume, f_spiky AND NOT r7 AS f_spiky, f_rise AND NOT r7 AS f_rise), f_stuck AND r7 AS b_stuck, f_count_on AND r7 AS b_count_on, f_dropout AND r7 AS b_dropout, f_level AND r7 AS b_level, f_night_drop AND r7 AS b_night_drop, f_prof AND r7 AS b_prof, f_rapid AND r7 AS b_rapid, f_volume AND r7 AS b_volume, f_spiky AND r7 AS b_spiky, f_rise AND r7 AS b_rise FROM hq_preF2
),
hq_preF4 AS (
SELECT *, CASE WHEN v_stuck IS NOT NULL OR v_count_on IS NOT NULL OR v_dropout IS NOT NULL OR v_level IS NOT NULL OR v_night_drop IS NOT NULL OR v_prof IS NOT NULL OR v_rapid IS NOT NULL OR v_volume IS NOT NULL OR v_spiky IS NOT NULL OR v_rise IS NOT NULL THEN (1 - CASE WHEN f_stuck THEN v_stuck ELSE 0 END) * (1 - CASE WHEN f_count_on THEN v_count_on ELSE 0 END) * (1 - CASE WHEN f_dropout THEN v_dropout ELSE 0 END) * (1 - CASE WHEN f_level THEN v_level ELSE 0 END) * (1 - CASE WHEN f_night_drop THEN v_night_drop ELSE 0 END) * (1 - CASE WHEN f_prof THEN v_prof ELSE 0 END) * (1 - CASE WHEN f_rapid THEN v_rapid ELSE 0 END) * (1 - CASE WHEN f_volume THEN v_volume ELSE 0 END) * (1 - CASE WHEN f_spiky THEN v_spiky ELSE 0 END) * (1 - CASE WHEN f_rise THEN v_rise ELSE 0 END) END AS score, (f_rapid OR f_volume)::INT + (f_prof)::INT + ((f_dropout AND NOT coalesce(v_dropout = 0.35, false) AND drop_b1 - drop_b0 >= 3 AND drop_lam >= 50) OR f_level OR f_night_drop)::INT + (f_rise)::INT + (f_spiky)::INT + ((f_stuck AND NOT coalesce(v_stuck = 0.35, false)) OR f_count_on)::INT AS n_ev, (f_rapid)::INT + (f_level OR f_night_drop)::INT + (f_rise)::INT + (f_prof OR f_volume OR f_spiky)::INT + ((f_dropout AND NOT coalesce(v_dropout = 0.35, false) AND drop_b1 - drop_b0 >= 3 AND drop_lam >= 50))::INT + ((f_stuck AND NOT coalesce(v_stuck = 0.35, false)) OR f_count_on)::INT AS n_fam,
       (drop_b1 - drop_b0) * 300 / 3600.0 AS silent_h,
       CASE WHEN (1 - CASE WHEN f_stuck THEN v_stuck ELSE 0 END) * (1 - CASE WHEN f_count_on THEN v_count_on ELSE 0 END) * (1 - CASE WHEN f_dropout THEN v_dropout ELSE 0 END) * (1 - CASE WHEN f_level THEN v_level ELSE 0 END) * (1 - CASE WHEN f_night_drop THEN v_night_drop ELSE 0 END) * (1 - CASE WHEN f_prof THEN v_prof ELSE 0 END) * (1 - CASE WHEN f_rapid THEN v_rapid ELSE 0 END) * (1 - CASE WHEN f_volume THEN v_volume ELSE 0 END) * (1 - CASE WHEN f_spiky THEN v_spiky ELSE 0 END) * (1 - CASE WHEN f_rise THEN v_rise ELSE 0 END) < 0.25 AND (v_stuck IS NOT NULL OR v_count_on IS NOT NULL OR v_dropout IS NOT NULL OR v_level IS NOT NULL OR v_night_drop IS NOT NULL OR v_prof IS NOT NULL OR v_rapid IS NOT NULL OR v_volume IS NOT NULL OR v_spiky IS NOT NULL OR v_rise IS NOT NULL) THEN 'bad' WHEN (1 - CASE WHEN f_stuck THEN v_stuck ELSE 0 END) * (1 - CASE WHEN f_count_on THEN v_count_on ELSE 0 END) * (1 - CASE WHEN f_dropout THEN v_dropout ELSE 0 END) * (1 - CASE WHEN f_level THEN v_level ELSE 0 END) * (1 - CASE WHEN f_night_drop THEN v_night_drop ELSE 0 END) * (1 - CASE WHEN f_prof THEN v_prof ELSE 0 END) * (1 - CASE WHEN f_rapid THEN v_rapid ELSE 0 END) * (1 - CASE WHEN f_volume THEN v_volume ELSE 0 END) * (1 - CASE WHEN f_spiky THEN v_spiky ELSE 0 END) * (1 - CASE WHEN f_rise THEN v_rise ELSE 0 END) < 0.70 AND (v_stuck IS NOT NULL OR v_count_on IS NOT NULL OR v_dropout IS NOT NULL OR v_level IS NOT NULL OR v_night_drop IS NOT NULL OR v_prof IS NOT NULL OR v_rapid IS NOT NULL OR v_volume IS NOT NULL OR v_spiky IS NOT NULL OR v_rise IS NOT NULL) THEN 'suspect'
            WHEN r7 OR w_dropout_short OR w_no_yardstick THEN 'watch' WHEN n_on < 20 THEN 'not_enough_data'
            ELSE 'ok' END AS st0
FROM hq_preF3
),
hq_preV AS MATERIALIZED (
SELECT * EXCLUDE (st0, st1, st2, st3),
       CASE WHEN st3 = 'bad' AND f_rapid AND (f_stuck::INT + f_count_on::INT + f_dropout::INT + f_level::INT + f_night_drop::INT + f_prof::INT + f_rapid::INT + f_volume::INT + f_spiky::INT + f_rise::INT) = 1 AND fe_n >= 8 AND fe_phi <= 2.0
                 AND fn IN ('Presence', 'Other', 'Mid') THEN 'suspect' ELSE st3 END AS health_st,
       CASE WHEN st3 = 'bad' AND f_rapid AND (f_stuck::INT + f_count_on::INT + f_dropout::INT + f_level::INT + f_night_drop::INT + f_prof::INT + f_rapid::INT + f_volume::INT + f_spiky::INT + f_rise::INT) = 1 AND fe_n >= 8 AND fe_phi <= 2.0
                 AND fn IN ('Presence', 'Other', 'Mid') THEN 'F'
            WHEN st3 <> st2 THEN 'C' WHEN st2 <> st1 THEN 'P' WHEN st1 <> st0 AND n_fam < 2 THEN 'A' ELSE '' END AS sev_rule
FROM (SELECT *, CASE WHEN fn = 'Count' AND c1_pre AND hi_min_c >= 30 AND st2 IN ('ok', 'watch') THEN 'suspect'
                     ELSE st2 END AS st3
      FROM (SELECT *, CASE WHEN st1 = 'suspect' AND (f_prof OR (f_dropout AND silent_h >= 4.0)) THEN 'bad' ELSE st1 END
                   AS st2
            FROM (SELECT *, CASE WHEN st0 = 'suspect' AND (n_fam >= 2 OR n_ev >= 2) THEN 'bad' ELSE st0 END AS st1
                  FROM hq_preF4)))
),
hq_flag AS (
SELECT dev, det, f_stuck OR f_count_on OR f_prof OR f_rapid OR f_volume OR f_spiky AS flagged FROM hq_preV
),
hq_dcph AS MATERIALIZED (
SELECT d.dev, d.det, x.ph FROM hq_dets d LEFT JOIN hq_xbase x USING (dev, det)
),
hq_dcrf AS (
SELECT a.dev, a.det, b.det AS r, coalesce(a.ph = b.ph, false) AS same, coalesce(f.flagged, false) AS fl
FROM hq_dcph a JOIN hq_dcph b ON a.dev = b.dev AND a.det <> b.det JOIN hq_dstat sb ON sb.dev = b.dev AND sb.det = b.det AND sb.live
LEFT JOIN hq_twin t ON t.dev = a.dev AND t.i = a.det AND t.j = b.det LEFT JOIN hq_flag f ON f.dev = b.dev AND f.det = b.det
WHERE t.i IS NULL
),
hq_dccand AS MATERIALIZED (
SELECT x.dev, x.det, coalesce(k.nsame, 0) >= 2 AS isph, coalesce(k.nkeep, 0) >= 2 AS keep2
FROM hq_xx x LEFT JOIN (SELECT dev, det, count(*) FILTER (same) AS nsame, count(*) FILTER (same AND NOT fl) AS nkeep
                     FROM hq_dcrf GROUP BY dev, det) k USING (dev, det)
WHERE x.drop_lam >= 30 AND x.n_on > 0
),
hq_dcm AS (
SELECT c.dev, c.det, 0 AS v, f.r FROM hq_dccand c JOIN hq_dcrf f USING (dev, det) WHERE c.isph AND f.same
UNION ALL SELECT c.dev, c.det, 0, d.det FROM hq_dccand c JOIN hq_dets d ON d.dev = c.dev AND d.det <> c.det WHERE NOT c.isph
UNION ALL SELECT c.dev, c.det, 1, f.r FROM hq_dccand c JOIN hq_dcrf f USING (dev, det) WHERE c.isph AND c.keep2 AND f.same
          AND NOT f.fl
UNION ALL SELECT c.dev, c.det, 1, d.det FROM hq_dccand c JOIN hq_dets d ON d.dev = c.dev AND d.det <> c.det
          LEFT JOIN hq_flag f ON f.dev = d.dev AND f.det = d.det WHERE NOT (c.isph AND c.keep2)
          AND NOT coalesce(f.flagged, false)
),
hq_dcnd AS (
SELECT c.dev, c.det, CASE WHEN c.isph AND c.keep2 THEN count(*) FILTER (f.same AND f.fl)
                          ELSE count(*) FILTER (fl2) END AS n_dropped
FROM hq_dccand c JOIN (SELECT a.dev, a.det, a.same, a.fl, coalesce(g.flagged, false) AS fl2 FROM
                      (SELECT d.dev, c.det, d.det AS r, coalesce(f.same, false) AS same, coalesce(f.fl, false) AS fl
                       FROM hq_dccand c JOIN hq_dets d ON d.dev = c.dev AND d.det <> c.det
                       LEFT JOIN hq_dcrf f ON f.dev = c.dev AND f.det = c.det AND f.r = d.det) a
                    LEFT JOIN hq_flag g ON g.dev = a.dev AND g.det = a.r) f USING (dev, det)
GROUP BY c.dev, c.det, c.isph, c.keep2
),
hq_dcrs0 AS MATERIALIZED (
SELECT u.dev, u.det, m.v, u.a, u.z, m.r, w.nb5 FROM hq_runs u JOIN hq_dcm m ON m.dev = u.dev AND m.det = u.det
JOIN hq_win w ON w.dev = u.dev
),
hq_dcrs_1 AS MATERIALIZED (
SELECT x.*, k.cn AS cn_a FROM hq_dcrs0 x JOIN hq_k5 k ON k.dev = x.dev AND k.det = x.r AND k.k = x.a
),
hq_dcrs_2 AS MATERIALIZED (
SELECT x.*, k.cn AS cn_z FROM hq_dcrs_1 x JOIN hq_k5 k ON k.dev = x.dev AND k.det = x.r AND k.k = x.z
),
hq_dcrs_3 AS MATERIALIZED (
SELECT x.*, k.cn AS cn_nb5 FROM hq_dcrs_2 x JOIN hq_k5 k ON k.dev = x.dev AND k.det = x.r AND k.k = x.nb5
),
hq_dcrs AS (
SELECT dev, det, v, a, z, sum(cn_z - cn_a) AS s_in, sum(cn_nb5) AS s_tot FROM hq_dcrs_3 GROUP BY dev, det, v, a, z
),
hq_dc AS (
SELECT c.dev, c.det, coalesce(r0.lam, 0.0) AS drop_lam_chk, coalesce(r1.lam, 0.0) AS drop_lam_c,
       coalesce(r1.b0, -1) AS drop_b0_c, coalesce(r1.b1, -1) AS drop_b1_c, n.n_dropped
FROM hq_dccand c JOIN hq_dcnd n USING (dev, det)
LEFT JOIN (SELECT dev, det, v, max(lam) AS lam, first(a ORDER BY round(lam, 6) DESC, a) AS b0, first(z ORDER BY round(lam, 6) DESC, a) AS b1
           FROM (SELECT r.*, s.n_on / greatest(r.s_tot - r.s_in, 1) * r.s_in AS lam FROM hq_dcrs r JOIN hq_dstat s USING (dev, det)
                 JOIN hq_win w ON w.dev = r.dev WHERE r.z - r.a <= w.nb5 - 12 AND s.n_on >= 5)
           WHERE lam > 0 GROUP BY dev, det, v) r0 ON r0.dev = c.dev AND r0.det = c.det AND r0.v = 0
LEFT JOIN (SELECT dev, det, v, max(lam) AS lam, first(a ORDER BY round(lam, 6) DESC, a) AS b0, first(z ORDER BY round(lam, 6) DESC, a) AS b1
           FROM (SELECT r.*, s.n_on / greatest(r.s_tot - r.s_in, 1) * r.s_in AS lam FROM hq_dcrs r JOIN hq_dstat s USING (dev, det)
                 JOIN hq_win w ON w.dev = r.dev WHERE r.z - r.a <= w.nb5 - 12 AND s.n_on >= 5)
           WHERE lam > 0 GROUP BY dev, det, v) r1 ON r1.dev = c.dev AND r1.det = c.det AND r1.v = 1
),
hq_xfa0 AS MATERIALIZED (
SELECT x.* REPLACE (
    CASE WHEN m THEN x.drop_lam * c.drop_lam_c / c.drop_lam_chk ELSE x.drop_lam END AS drop_lam,
    CASE WHEN m THEN c.drop_b0_c ELSE x.drop_b0 END AS drop_b0,
    CASE WHEN m THEN c.drop_b1_c ELSE x.drop_b1 END AS drop_b1,
    CASE WHEN m THEN CASE WHEN x.drop_in_stuck THEN 0.0 WHEN x.drop_lam * c.drop_lam_c / c.drop_lam_chk < 30 THEN 0.0
                          ELSE 0.35 + 0.65 * least((x.drop_lam * c.drop_lam_c / c.drop_lam_chk - 30) / 70, 1) END
         ELSE x.s_dropout END AS s_dropout)
FROM (SELECT x.*, coalesce(c.n_dropped > 0 AND c.drop_lam_chk > 0, false) AS m FROM hq_xx x LEFT JOIN hq_dc c USING (dev, det)) x
LEFT JOIN hq_dc c USING (dev, det)
),
hq_ownh AS MATERIALIZED (
SELECT * FROM (
WITH r AS (SELECT dev, det, drop_b0, drop_b1 FROM hq_xfa0 WHERE drop_b1 > drop_b0 AND drop_lam > 0),
g AS (SELECT g.dev, g.det, g.b, g.n, g.cov, (w.s_ns // 300000000000 + g.b) % 288 AS bod, r.drop_b0, r.drop_b1
      FROM hq_g5 g JOIN r USING (dev, det) JOIN hq_win w USING (dev)),
ref AS (SELECT dev, det, bod, avg(n) AS mn FROM g
        WHERE cov AND NOT (b >= drop_b0 AND b < drop_b1)
          AND NOT EXISTS (SELECT 1 FROM hq_silall u
                          WHERE u.dev = g.dev AND u.det = g.det AND u.lam >= 30
                          AND g.b >= u.a AND g.b < u.z)
        GROUP BY dev, det, bod)
SELECT g.dev, g.det, sum(f.mn) AS own_typ, count(f.mn) AS nref, count(*) AS nb
FROM g LEFT JOIN ref f USING (dev, det, bod) WHERE g.b >= g.drop_b0 AND g.b < g.drop_b1 GROUP BY g.dev, g.det)
),
hq_xfa AS MATERIALIZED (
SELECT x.* REPLACE (CASE WHEN o.nref >= 0.8 * o.nb AND x.drop_lam > 3 * o.own_typ + 10 THEN 3 * o.own_typ + 10
                         ELSE x.drop_lam END AS drop_lam)
FROM hq_xfa0 x LEFT JOIN hq_ownh o USING (dev, det)
),
hq_greens5 AS (
SELECT dev, b::BIGINT AS b, sum(ngreen)::BIGINT AS k FROM hq_p5 GROUP BY dev, b HAVING sum(ngreen) > 0
),
hq_sgb AS MATERIALIZED (
SELECT b.dev, b.b, b.cov, b.hr, coalesce(n.n, 0) AS n, coalesce(n.occ, 0) AS occ, coalesce(g.k, 0) AS greens
FROM hq_bins5 b LEFT JOIN (SELECT dev, b, sum(n) AS n, sum(occ) AS occ FROM hq_g5 GROUP BY dev, b) n USING (dev, b)
LEFT JOIN hq_greens5 g USING (dev, b)
),
hq_sgr AS MATERIALIZED (
SELECT dev, a, z, nday FROM (
  SELECT dev, min(b) AS a, max(b) + 1 AS z, count(*) FILTER (hr >= 7 AND hr < 19) AS nday,
         row_number() OVER (PARTITION BY dev ORDER BY count(*) FILTER (hr >= 7 AND hr < 19) DESC, min(b)) AS rk
  FROM (SELECT *, n = 0 AND occ < 1 AND greens > 0 AS quiet,
               sum((NOT (n = 0 AND occ < 1 AND greens > 0))::INT) OVER (PARTITION BY dev ORDER BY b) AS grp
        FROM hq_sgb WHERE cov)
  WHERE quiet
  GROUP BY dev, grp)
WHERE rk = 1 AND nday >= 12
),
hq_sgl AS MATERIALIZED (
SELECT g.dev, g.det, r.a, r.z,
       sum(g.n) FILTER (NOT (g.b >= r.a AND g.b < r.z)) / greatest(sum(s.greens) FILTER (NOT (g.b >= r.a AND g.b < r.z)), 1)
       * sum(s.greens) FILTER (g.b >= r.a AND g.b < r.z) AS lam_sig
FROM hq_g5 g JOIN hq_sgr r USING (dev) JOIN hq_sgb s ON s.dev = g.dev AND s.b = g.b
WHERE g.cov AND g.hr >= 7 AND g.hr < 19
GROUP BY g.dev, g.det, r.a, r.z
HAVING sum(g.n) FILTER (NOT (g.b >= r.a AND g.b < r.z)) >= 20
),
hq_xfs AS MATERIALIZED (
SELECT x.* EXCLUDE (msg) REPLACE (CASE WHEN msg THEN l.lam_sig ELSE x.drop_lam END AS drop_lam,
         CASE WHEN msg THEN l.a ELSE x.drop_b0 END AS drop_b0, CASE WHEN msg THEN l.z ELSE x.drop_b1 END AS drop_b1,
         CASE WHEN msg THEN 0.35 + 0.65 * least((l.lam_sig - 30) / 70, 1) ELSE x.s_dropout END AS s_dropout),
       coalesce(msg, false) AS sig_silent
FROM (SELECT x.*, l.lam_sig >= 30 AND l.lam_sig > coalesce(x.drop_lam, 0) AS msg FROM hq_xfa x LEFT JOIN hq_sgl l USING (dev, det)) x
LEFT JOIN hq_sgl l USING (dev, det)
),
hq_pdm AS MATERIALIZED (
SELECT m.dev, p.ph, m.hid, x.j, x.hod, x.ch, sum(m.M) FILTER (z.ph = p.ph) AS mp, sum(m.M) FILTER (z.ph <> p.ph) AS rr
FROM hq_mh m JOIN hq_dz z USING (dev, det) JOIN hq_hix x USING (dev, hid)
JOIN (SELECT dev, ph FROM hq_dz WHERE ph IS NOT NULL AND fn IS DISTINCT FROM 'Bike' GROUP BY dev, ph HAVING count(*) >= 2) p
  ON p.dev = m.dev
WHERE z.ph IS NOT NULL AND z.fn IS DISTINCT FROM 'Bike'
GROUP BY m.dev, p.ph, m.hid, x.j, x.hod, x.ch
),
hq_pdr AS MATERIALIZED (
SELECT * FROM (
WITH p AS (SELECT *, hid // 24 AS day, isodow(make_timestamp(hid * 3600000000000 // 1000)) >= 6 AS wk FROM hq_pdm),
h AS (SELECT a.dev, a.ph, a.hid, a.j, a.mp, a.rr, median(b.mp / b.rr) AS sp
      FROM p a JOIN p b ON b.dev = a.dev AND b.ph = a.ph AND b.hod = a.hod AND b.wk = a.wk AND b.day <> a.day
                        AND b.ch AND b.rr >= 30 AND b.rr BETWEEN 0.5 * a.rr AND 2 * a.rr
      WHERE a.ch AND a.hod >= 6 AND a.hod < 21 AND a.rr >= 30 GROUP BY a.dev, a.ph, a.hid, a.j, a.mp, a.rr),
l AS (SELECT *, j - row_number() OVER (PARTITION BY dev, ph ORDER BY j) AS g FROM h
      WHERE sp > 0 AND sp * rr >= 30 AND mp <= 0.2 * sp * rr)
SELECT dev, ph, min(hid) AS ha, max(hid) + 1 AS hz FROM l GROUP BY dev, ph, g
HAVING count(*) >= 2 AND sum(sp * rr) >= 200)
),
hq_greens_hour AS (
SELECT dev, ph, h FROM hq_rgq
),
hq_pdg AS MATERIALIZED (
SELECT r.dev, r.ph, r.ha, r.hz FROM hq_pdr r JOIN hq_win w USING (dev)
WHERE EXISTS (SELECT 1 FROM hq_greens_hour e WHERE e.dev = r.dev AND e.ph = r.ph AND e.h >= r.ha AND e.h < r.hz)
),
hq_pdd AS MATERIALIZED (
SELECT z.dev, z.det, g.ph, g.ha, g.hz,
       sum(m.M) FILTER (m.hid >= g.ha AND m.hid < g.hz) AS obs,
       sum(m.M) FILTER (NOT (m.hid >= g.ha AND m.hid < g.hz) AND x.ch AND x.hod >= 7 AND x.hod < 19 AND p.rr >= 30)
       / nullif(sum(p.rr) FILTER (NOT (m.hid >= g.ha AND m.hid < g.hz) AND x.ch AND x.hod >= 7 AND x.hod < 19
                                 AND p.rr >= 30), 0)
       * sum(p.rr) FILTER (m.hid >= g.ha AND m.hid < g.hz) AS ex
FROM hq_pdg g JOIN hq_dz z ON z.dev = g.dev AND z.ph = g.ph AND z.fn IS DISTINCT FROM 'Bike'
JOIN hq_mh m ON m.dev = z.dev AND m.det = z.det JOIN hq_hix x ON x.dev = m.dev AND x.hid = m.hid
JOIN hq_pdm p ON p.dev = m.dev AND p.ph = g.ph AND p.hid = m.hid
GROUP BY z.dev, z.det, g.ph, g.ha, g.hz
),
hq_pdk AS MATERIALIZED (
SELECT * FROM (
  SELECT d.*, count(*) FILTER (d.ex >= 30 AND d.obs <= 0.25 * d.ex) OVER (PARTITION BY d.dev, d.ph, d.ha) AS k,
         count(*) OVER (PARTITION BY d.dev, d.ph, d.ha) AS m FROM hq_pdd d)
QUALIFY row_number() OVER (PARTITION BY dev, det ORDER BY ex - obs DESC, ph, ha) = 1
),
hq_lkh AS MATERIALIZED (
SELECT m.dev, z.ph, m.hid, x.j, x.ch, m.det, m.M, median(m.M) OVER (PARTITION BY m.dev, z.ph, m.hid) AS md
FROM hq_mh m JOIN hq_dz z USING (dev, det) JOIN hq_hix x USING (dev, hid) JOIN hq_dstat s USING (dev, det)
WHERE z.ph IS NOT NULL AND z.fn IS DISTINCT FROM 'Bike' AND s.live
  AND NOT EXISTS (SELECT 1 FROM hq_twin t WHERE t.dev = m.dev AND t.i = m.det AND t.j < m.det)
),
hq_lkr AS MATERIALIZED (
SELECT dev, ph, min(hid) AS ha, max(hid) + 1 AS hz, median(md) AS md FROM (
  SELECT *, j - row_number() OVER (PARTITION BY dev, ph ORDER BY j) AS g FROM (
    SELECT dev, ph, hid, j, any_value(md) AS md FROM hq_lkh WHERE ch AND md >= 20
    GROUP BY dev, ph, hid, j HAVING count(*) FILTER (abs(M - md) <= 0.01 * md + 1) >= 4))
GROUP BY dev, ph, g HAVING count(*) >= 6
),
hq_lkd AS MATERIALIZED (
SELECT * FROM (
  SELECT h.dev, h.det, r.ph, r.ha, r.hz, r.md, count(*) OVER (PARTITION BY r.dev, r.ph, r.ha) AS nk FROM hq_lkr r
  JOIN hq_lkh h ON h.dev = r.dev AND h.ph = r.ph AND h.hid >= r.ha AND h.hid < r.hz
  GROUP BY h.dev, h.det, r.dev, r.ph, r.ha, r.hz, r.md HAVING count(*) FILTER (abs(h.M - h.md) <= 0.01 * h.md + 1) >= 3)
QUALIFY row_number() OVER (PARTITION BY dev, det ORDER BY hz - ha DESC, ph, ha) = 1
),
hq_xf AS MATERIALIZED (
SELECT x.* EXCLUDE (mpd, mlk) REPLACE (
         CASE WHEN mlk THEN 1e6 WHEN mpd THEN k.ex - k.obs ELSE x.drop_lam END AS drop_lam,
         CASE WHEN mlk THEN (l.ha * 3600000000000 - w.s_ns) // 300000000000 WHEN mpd THEN (k.ha * 3600000000000 - w.s_ns) // 300000000000
              ELSE x.drop_b0 END AS drop_b0,
         CASE WHEN mlk THEN least((l.hz * 3600000000000 - w.s_ns) // 300000000000, w.nb5)
              WHEN mpd THEN least((k.hz * 3600000000000 - w.s_ns) // 300000000000, w.nb5) ELSE x.drop_b1 END AS drop_b1,
         CASE WHEN mlk THEN 1.0 WHEN mpd THEN 0.35 + 0.65 * least((k.ex - k.obs - 30) / 70, 1) ELSE x.s_dropout END
           AS s_dropout),
       mpd AND NOT mlk AS ph_drop, mlk AS ph_lock, k.k AS ph_k, k.m AS ph_m, k.obs AS ph_obs, k.ex AS ph_exp,
       l.nk AS lk_n, l.md AS lk_md
FROM (SELECT x.*, coalesce(k.ex >= 30 AND k.obs <= 0.25 * k.ex AND k.ex - k.obs >= 30
                           AND k.ex - k.obs > coalesce(x.drop_lam, 0), false) AS mpd, l.det IS NOT NULL AS mlk
      FROM hq_xfs x LEFT JOIN hq_pdk k USING (dev, det) LEFT JOIN hq_lkd l USING (dev, det)) x
LEFT JOIN hq_pdk k USING (dev, det) LEFT JOIN hq_lkd l USING (dev, det) JOIN hq_win w USING (dev)
),
hq_lcz AS MATERIALIZED (
SELECT * FROM (
WITH o AS (SELECT m.dev, m.det, m.hid, m.M, sum(m2.M) AS ro FROM hq_mh m JOIN hq_dz z USING (dev, det)
           JOIN hq_mh m2 ON m2.dev = m.dev AND m2.hid = m.hid JOIN hq_dz z2 ON z2.dev = m2.dev AND z2.det = m2.det
           WHERE z.ph IS NOT NULL AND z2.ph IS NOT NULL AND z2.ph <> z.ph GROUP BY m.dev, m.det, m.hid, m.M),
os AS (SELECT o.*, x.j, x.ch, median(o.M / o.ro) FILTER (x.ch AND o.ro >= 30) OVER (PARTITION BY o.dev, o.det) AS so
       FROM o JOIN hq_hix x USING (dev, hid)),
h AS (SELECT g.dev, g.det, g.j,
             (g.ch AND CASE WHEN a.re THEN a.s2 ELSE a.s END * g.R >= 10) OR coalesce(b.ch AND b.so * b.ro >= 10, false)
             AS judged,
             (g.ch AND CASE WHEN a.re THEN a.s2 ELSE a.s END * g.R >= 10
              AND g.M <= 0.15 * CASE WHEN a.re THEN a.s2 ELSE a.s END * g.R)
             OR coalesce(b.ch AND b.so * b.ro >= 10 AND b.M <= 0.15 * b.so * b.ro, false) AS z
      FROM hq_gh2 g JOIN hq_gh4 a USING (dev, det) LEFT JOIN os b ON b.dev = g.dev AND b.det = g.det AND b.j = g.j),
r AS (SELECT *, sum((judged AND NOT z)::INT) OVER (PARTITION BY dev, det ORDER BY j) AS grp FROM h)
SELECT dev, det, min(j) FILTER (z) AS ja, max(j) FILTER (z) + 1 AS jz FROM r
GROUP BY dev, det, grp HAVING count(*) FILTER (z) >= 2)
),
hq_lcq AS MATERIALIZED (
SELECT * FROM (
WITH m AS (SELECT g.dev, g.det, (w.s_ns + g.b * 300000000000) // 900000000000 AS qid, sum(g.n) AS M, count(*) FILTER (g.cov) = 3 AS cq
           FROM hq_g5 g JOIN hq_win w USING (dev) JOIN hq_dz z ON z.dev = g.dev AND z.det = g.det
           WHERE w.hours >= CAST(3.0 AS DOUBLE) AND z.ph IS NOT NULL GROUP BY g.dev, g.det, qid),
o AS (SELECT m.dev, m.det, m.qid, m.qid // 4 AS hid, m.M, m.cq, coalesce(sum(m2.M), 0) AS ro
      FROM m JOIN hq_dz z USING (dev, det) LEFT JOIN (m m2 JOIN hq_dz z2 ON z2.dev = m2.dev AND z2.det = m2.det)
        ON m2.dev = m.dev AND m2.qid = m.qid AND z2.ph IS NOT NULL AND z2.ph <> z.ph
      GROUP BY m.dev, m.det, m.qid, m.M, m.cq),
oh AS (SELECT o.*, sum(o.ro) OVER (PARTITION BY o.dev, o.det, o.hid) AS roh,
              sum(o.M) OVER (PARTITION BY o.dev, o.det, o.hid) AS Mh,
              bool_and(o.cq) OVER (PARTITION BY o.dev, o.det, o.hid) AND count(*) OVER (PARTITION BY o.dev, o.det, o.hid) = 4
              AS chq FROM o),
os AS (SELECT oh.*, median(Mh / roh) FILTER (chq AND roh >= 30) OVER (PARTITION BY dev, det) AS so FROM oh),
e AS (SELECT s.dev, s.det, s.qid, s.M, s.cq,
             CASE WHEN g.ch THEN CASE WHEN a.re THEN a.s2 ELSE a.s END * g.R
                  * CASE WHEN s.roh > 0 THEN s.ro / s.roh ELSE 0.25 END END AS ey,
             s.so * s.ro AS eo
      FROM os s JOIN hq_hix x ON x.dev = s.dev AND x.hid = s.hid
      LEFT JOIN hq_gh2 g ON g.dev = s.dev AND g.det = s.det AND g.j = x.j LEFT JOIN hq_gh4 a ON a.dev = s.dev AND a.det = s.det),
h AS (SELECT dev, det, qid, coalesce(cq AND ey >= 2.5, false) OR coalesce(cq AND eo >= 2.5, false) AS judged,
             coalesce(cq AND ey >= 2.5 AND M <= 0.15 * ey, false) OR coalesce(cq AND eo >= 2.5 AND M <= 0.15 * eo, false) AS z
      FROM e),
r AS (SELECT *, sum((judged AND NOT z)::INT) OVER (PARTITION BY dev, det ORDER BY qid) AS grp FROM h)
SELECT dev, det, min(qid) FILTER (z) AS ja, max(qid) FILTER (z) + 1 AS jz FROM r
GROUP BY dev, det, grp HAVING count(*) FILTER (z) >= 3)
),
hq_lcan AS MATERIALIZED (
SELECT z.dev, z.det, greatest(((h.hid0 + z.ja) * 3600000000000 - w.s_ns) // 300000000000, 0) AS b0,
       least(((h.hid0 + z.jz) * 3600000000000 - w.s_ns) // 300000000000, w.nb5) AS b1, false AS qr
FROM hq_lcz z JOIN hq_win w USING (dev) JOIN (SELECT dev, min(hid) AS hid0 FROM hq_hix GROUP BY dev) h USING (dev)
UNION ALL
SELECT z.dev, z.det, greatest((z.ja * 900000000000 - w.s_ns) // 300000000000, 0), least((z.jz * 900000000000 - w.s_ns) // 300000000000, w.nb5), true FROM hq_lcq z JOIN hq_win w USING (dev)
UNION ALL
SELECT dev, det, drop_b0, drop_b1, false FROM hq_xf WHERE drop_b1 - drop_b0 >= 3 AND drop_lam >= 30
),
hq_lcc AS MATERIALIZED (
SELECT x.dev, x.det, x.ph, x.fn, 'd' AS kind, x.drop_b0::BIGINT AS b0, x.drop_b1::BIGINT AS b1
FROM hq_xf x JOIN hq_limits l USING (dev, det)
WHERE x.n_on > 0 AND x.ph IS NOT NULL AND x.drop_b1 - x.drop_b0 >= 3 AND x.drop_lam >= l.lim_dropout AND NOT x.sig_silent AND NOT x.ph_drop AND NOT x.ph_lock AND (x.dev, x.det) NOT IN (SELECT (dev, det) FROM hq_flag WHERE flagged)
  AND NOT EXISTS (SELECT 1 FROM hq_lcan a JOIN hq_lcz z ON z.dev = a.dev AND z.det = a.det
                  WHERE a.dev = x.dev AND a.det = x.det AND a.b0 <= x.drop_b0 AND a.b1 >= x.drop_b1
                  AND a.b1 - a.b0 > x.drop_b1 - x.drop_b0 AND NOT a.qr)
UNION ALL
SELECT x.dev, x.det, x.ph, x.fn, 'z' || z.ja, greatest(((h.hid0 + z.ja) * 3600000000000 - w.s_ns) // 300000000000, 0),
       least(((h.hid0 + z.jz) * 3600000000000 - w.s_ns) // 300000000000, w.nb5)
FROM hq_lcz z JOIN hq_xf x USING (dev, det) JOIN hq_win w USING (dev)
JOIN (SELECT dev, min(hid) AS hid0 FROM hq_hix GROUP BY dev) h USING (dev)
WHERE x.ph IS NOT NULL AND NOT x.sig_silent AND NOT x.ph_drop AND NOT x.ph_lock AND (x.dev, x.det) NOT IN (SELECT (dev, det) FROM hq_flag WHERE flagged)
UNION ALL
SELECT x.dev, x.det, x.ph, x.fn, 'q' || z.ja, greatest((z.ja * 900000000000 - w.s_ns) // 300000000000, 0), least((z.jz * 900000000000 - w.s_ns) // 300000000000, w.nb5)
FROM hq_lcq z JOIN hq_xf x USING (dev, det) JOIN hq_win w USING (dev)
WHERE x.ph IS NOT NULL AND NOT x.sig_silent AND NOT x.ph_drop AND NOT x.ph_lock AND (x.dev, x.det) NOT IN (SELECT (dev, det) FROM hq_flag WHERE flagged)
),
hq_lcg AS MATERIALIZED (
SELECT dev, det, kind, b0, b1, ph, fn, -1.0 AS gq FROM hq_lcc
UNION ALL
SELECT c.dev, c.det, c.kind, c.b0, c.b1, c.ph, c.fn, p.ph FROM hq_lcc c
JOIN (SELECT DISTINCT dev, ph FROM hq_dz WHERE ph IS NOT NULL) p ON p.dev = c.dev AND p.ph <> c.ph AND ((least(c.ph, p.ph) = 2 AND greatest(c.ph, p.ph) = 5) OR (least(c.ph, p.ph) = 1 AND greatest(c.ph, p.ph) = 6) OR (least(c.ph, p.ph) = 3 AND greatest(c.ph, p.ph) = 8) OR (least(c.ph, p.ph) = 4 AND greatest(c.ph, p.ph) = 7))
),
hq_lcm AS MATERIALIZED (
SELECT g.dev, g.det, g.kind, g.gq, g.det AS m, 'd' AS role FROM hq_lcg g JOIN (SELECT * FROM (VALUES ('Advance', 'up'), ('Mid', 'up'), ('Count', 'bar'), ('Presence', 'bar'), ('Yellow_Red', 'bar')) t(fn, cl)) a ON a.fn = g.fn
UNION ALL
SELECT g.dev, g.det, g.kind, g.gq, z.det, 'p'
FROM hq_lcg g JOIN hq_dz z ON z.dev = g.dev AND z.det <> g.det AND (z.ph = g.ph OR z.ph = g.gq)
JOIN (SELECT * FROM (VALUES ('Advance', 'up'), ('Mid', 'up'), ('Count', 'bar'), ('Presence', 'bar'), ('Yellow_Red', 'bar')) t(fn, cl)) a ON a.fn = g.fn JOIN (SELECT * FROM (VALUES ('Advance', 'up'), ('Mid', 'up'), ('Count', 'bar'), ('Presence', 'bar'), ('Yellow_Red', 'bar')) t(fn, cl)) b ON b.fn = z.fn AND b.cl = a.cl
JOIN hq_dstat s ON s.dev = z.dev AND s.det = z.det AND s.live
LEFT JOIN hq_twin t ON t.dev = g.dev AND t.i = g.det AND t.j = z.det
WHERE t.i IS NULL AND NOT EXISTS (SELECT 1 FROM hq_twin u JOIN hq_dz w ON w.dev = u.dev AND w.det = u.j
                                   WHERE u.dev = z.dev AND u.i = z.det AND u.j < z.det AND w.fn = z.fn)
UNION ALL
SELECT g.dev, g.det, g.kind, g.gq, z.det, 'x'
FROM hq_lcg g JOIN hq_dz z ON z.dev = g.dev AND z.ph IS NOT NULL AND z.ph <> g.ph AND z.ph <> g.gq
  AND z.status1 IS DISTINCT FROM 'bad'
JOIN hq_dstat s ON s.dev = z.dev AND s.det = z.det AND s.live
LEFT JOIN (SELECT * FROM (VALUES ('Advance', 'up'), ('Mid', 'up'), ('Count', 'bar'), ('Presence', 'bar'), ('Yellow_Red', 'bar')) t(fn, cl)) k ON k.fn = z.fn
WHERE NOT EXISTS (SELECT 1 FROM hq_lcc o LEFT JOIN (SELECT * FROM (VALUES ('Advance', 'up'), ('Mid', 'up'), ('Count', 'bar'), ('Presence', 'bar'), ('Yellow_Red', 'bar')) t(fn, cl)) ko ON ko.fn = o.fn
                  WHERE o.dev = z.dev AND (o.det = z.det OR (o.ph = z.ph AND ko.cl = k.cl)) AND o.kind NOT LIKE 'q%')
),
hq_lcrows AS MATERIALIZED (
SELECT * FROM (
WITH bw AS (SELECT b.dev, b.b, b.hr, isodow(make_timestamp((w.s_ns + b.b * 300000000000) // 1000)) >= 6 AS wk FROM hq_bins5 b JOIN hq_win w USING (dev))
SELECT m.dev, m.det, m.kind, m.gq, m.m, m.role, c.b0, c.b1, n.b, n.n, n.occ, n.hr, bw.wk,
       n.b >= c.b0 AND n.b < c.b1 AS inp,
       NOT EXISTS (SELECT 1 FROM hq_lcan a WHERE a.dev = m.dev AND a.det = m.m AND n.b >= a.b0 AND n.b < a.b1 AND NOT a.qr)
       AND NOT EXISTS (SELECT 1 FROM hq_lcc o WHERE o.dev = m.dev AND o.det = m.det AND n.b >= o.b0 AND n.b < o.b1 AND o.kind NOT LIKE 'q%') AS ref
FROM hq_lcm m JOIN hq_lcc c ON c.dev = m.dev AND c.det = m.det AND c.kind = m.kind
JOIN hq_g5 n ON n.dev = m.dev AND n.det = m.m AND n.cov JOIN bw ON bw.dev = n.dev AND bw.b = n.b)
),
hq_lcs AS (
SELECT * FROM (
WITH hp AS (SELECT DISTINCT dev, det, kind, hr, wk FROM hq_lcrows WHERE role = 'd' AND inp),
xb AS (SELECT dev, det, kind, gq, b, sum(n) AS xn FROM hq_lcrows WHERE role = 'x' AND ref GROUP BY ALL),
r AS (SELECT r.*, h.hr IS NOT NULL AS th, xb.xn,
             CASE WHEN r.kind = 'd' THEN 0 WHEN r.kind LIKE 'q%' THEN 3 ELSE 12 END AS g
      FROM hq_lcrows r LEFT JOIN hp h ON h.dev = r.dev AND h.det = r.det AND h.kind = r.kind AND h.hr = r.hr AND h.wk = r.wk
      LEFT JOIN xb ON xb.dev = r.dev AND xb.det = r.det AND xb.kind = r.kind AND xb.gq = r.gq AND xb.b = r.b)
SELECT dev, det, kind, gq, m, role,
       sum(n) FILTER (inp) AS sp, sum(occ) FILTER (inp) AS op, count(*) FILTER (inp) AS np,
       sum(n) FILTER (ref AND NOT inp) AS sra, sum(xn) FILTER (ref AND NOT inp) AS xra,
       sum(n) FILTER (ref AND NOT inp AND th) AS srt, sum(xn) FILTER (ref AND NOT inp AND th) AS xrt,
       count(*) FILTER (ref AND NOT inp AND th) AS nrt,
       sum(n) FILTER (ref AND (b >= b0 - g - 24 AND b < b0 - g OR b >= b1 + g AND b < b1 + g + 24)) AS sad,
       sum(xn) FILTER (ref AND (b >= b0 - g - 24 AND b < b0 - g OR b >= b1 + g AND b < b1 + g + 24)) AS xad,
       sum(n) FILTER (ref AND b >= b0 - g - 12 AND b < b0 - g) AS pre,
       sum(xn) FILTER (ref AND b >= b0 - g - 12 AND b < b0 - g) AS xpre,
       sum(n) FILTER (ref AND b >= b1 + g AND b < b1 + g + 12) AS post,
       sum(xn) FILTER (ref AND b >= b1 + g AND b < b1 + g + 12) AS xpost
FROM r GROUP BY dev, det, kind, gq, m, role)
),
hq_lch AS MATERIALIZED (
SELECT dev, det, kind, gq, m, role, (b - b0) // 12 AS hh, sum(n) AS n FROM hq_lcrows WHERE inp GROUP BY ALL
),
hq_lce AS (
SELECT * FROM (
WITH t AS (SELECT *, bool_and(nrt >= 0.5 * np) OVER (PARTITION BY dev, det, kind, gq) AS hq_tod FROM hq_lcs),
s AS (SELECT dev, det, kind, gq, role, m, sp, op, np, pre, post, xpre, xpost, hq_tod,
             CASE WHEN hq_tod THEN srt WHEN xad >= 100 THEN sad ELSE sra END AS sr,
             CASE WHEN hq_tod THEN xrt WHEN xad >= 100 THEN xad ELSE xra END AS xr FROM t),
x AS (SELECT dev, det, kind, gq, sum(sp) AS xp FROM s WHERE role = 'x' GROUP BY ALL),
e AS (SELECT s.dev, s.det, s.kind, s.gq, s.m, s.role, s.hq_tod, s.op / greatest(s.np * 300.0, 1) AS occ_p, s.sp AS obs,
             s.sr / s.xr * x.xp AS ex, s.sr / s.xr AS rr,
             (s.xpre >= 30 AND s.sp / x.xp >= 1.25 * coalesce(s.pre, 0) / s.xpre)
             OR (s.xpost >= 30 AND s.sp / x.xp >= 1.25 * coalesce(s.post, 0) / s.xpost)
             OR (coalesce(s.xpre, 0) < 30 AND coalesce(s.xpost, 0) < 30) AS step
      FROM s JOIN x USING (dev, det, kind, gq) WHERE s.role <> 'x' AND s.xr >= 100 AND x.xp >= 30),
xh AS (SELECT dev, det, kind, gq, hh, sum(n) AS xn FROM hq_lch WHERE role = 'x' GROUP BY ALL),
fr AS (SELECT p.dev, p.det, p.kind, p.gq, p.m,
              count(*) FILTER (p.n > e.rr * xh.xn AND e.rr * xh.xn >= 5) / nullif(count(*) FILTER (e.rr * xh.xn >= 5), 0)
              AS frac
       FROM hq_lch p JOIN xh USING (dev, det, kind, gq, hh) JOIN e ON e.dev = p.dev AND e.det = p.det AND e.kind = p.kind
            AND e.gq = p.gq AND e.m = p.m
       WHERE p.role = 'p' GROUP BY p.dev, p.det, p.kind, p.gq, p.m)
SELECT d.dev, d.det, d.kind, d.gq, d.obs AS obs_d, d.ex AS exp_d, d.ex - d.obs AS loss, d.hq_tod, d.occ_p AS occ_d,
       max(e.obs - e.ex) AS top_gain, arg_max(f.frac, e.obs - e.ex) AS top_frac, arg_max(e.step, e.obs - e.ex) AS top_step, sum(e.obs) AS obs_g, sum(e.ex) AS exp_g, sum(e.obs - e.ex) AS gain,
       list(e.m ORDER BY e.obs - e.ex DESC, e.m) FILTER (e.obs - e.ex >= 0.15 * (d.ex - d.obs)) AS gainers,
       string_agg('det ' || e.m || CASE WHEN z.ph IS NOT NULL THEN ' (P' || trunc(z.ph)::BIGINT || ')' ELSE '' END, ', '
                  ORDER BY e.obs - e.ex DESC, e.m) FILTER (e.obs - e.ex >= 0.15 * (d.ex - d.obs)) AS glab
FROM e d JOIN e ON e.dev = d.dev AND e.det = d.det AND e.kind = d.kind AND e.gq = d.gq AND e.role = 'p'
LEFT JOIN hq_dz z ON z.dev = e.dev AND z.det = e.m
LEFT JOIN fr f ON f.dev = e.dev AND f.det = e.det AND f.kind = e.kind AND f.gq = e.gq AND f.m = e.m
WHERE d.role = 'd'
GROUP BY d.dev, d.det, d.kind, d.gq, d.obs, d.ex, d.hq_tod, d.occ_p)
),
hq_lcr AS MATERIALIZED (
SELECT * FROM (
  SELECT e.*, row_number() OVER (PARTITION BY dev, det, kind ORDER BY gq >= 0, gain / loss DESC, gq) AS rk
  FROM hq_lce e WHERE obs_d <= 0.15 * exp_d AND occ_d < 0.02 AND loss >= 100 AND sqrt(2 * (exp_d - obs_d) ^ 2 / greatest(exp_d + obs_d, 1e-9)) >= 8
       AND top_gain >= 0.55 * loss AND gain >= 0.5 * loss AND gain <= 2 * loss AND sqrt(2 * (obs_g - exp_g) ^ 2 / greatest(obs_g + exp_g, 1e-9)) >= 3
       AND coalesce(top_frac, 0) >= 0.67 AND coalesce(top_step, false))
WHERE rk = 1
),
hq_ev131 AS (
SELECT dev, dn, 131 AS eid, par FROM hq_rplan
),
hq_lcsw AS (
SELECT c.dev, c.det, c.kind, first(p.par ORDER BY abs(p.dn // 300000000000 - c.b0), p.dn, p.par) AS pat
FROM hq_lcc c JOIN hq_ev131 p ON p.dev = c.dev AND p.eid = 131 AND abs(p.dn // 300000000000 - c.b0) <= 1
WHERE c.kind = 'd' AND EXISTS (SELECT 1 FROM hq_ev131 e WHERE e.dev = c.dev AND e.eid = 131 AND abs(e.dn // 300000000000 - c.b1) <= 1)
GROUP BY c.dev, c.det, c.kind
),
hq_lcfp AS MATERIALIZED (
SELECT u.dev, u.det, w.s_ns + u.a * 300000000000 AS fa, w.s_ns + u.z * 300000000000 AS fz FROM hq_silall u JOIN hq_limits l USING (dev, det) JOIN hq_win w USING (dev) WHERE u.lam >= l.lim_dropout
UNION ALL
SELECT x.dev, x.det, w.s_ns + x.drop_b0 * 300000000000, w.s_ns + x.drop_b1 * 300000000000 FROM hq_xf x JOIN hq_limits l USING (dev, det)
JOIN hq_win w USING (dev) WHERE x.drop_b1 > x.drop_b0 AND x.drop_lam >= l.lim_dropout
UNION ALL
SELECT e.dev, e.det, w.s_ns + round(e.s * 1e9)::BIGINT, w.s_ns + round(e.f * 1e9)::BIGINT FROM hq_efl e JOIN hq_win w USING (dev)
WHERE e.counted
UNION ALL
SELECT dev, det, a * 3600000000000, z * 3600000000000 FROM hq_mhrun WHERE kind = 'drop' AND hours >= 2 AND rgeh >= CAST(15.0 AS DOUBLE)
),
hq_lcres AS MATERIALIZED (
SELECT * FROM (
WITH a AS (WITH c AS (SELECT c.*, w.s_ns + c.b0 * 300000000000 AS a_ns, least(w.s_ns + c.b1 * 300000000000, w.e_ns) AS z_ns
           FROM (  SELECT c.dev, c.det, c.kind, r.gq, r.loss, r.gain, r.gainers, r.glab, c.b0, c.b1, s.pat IS NOT NULL AS switch, s.pat
  FROM hq_lcc c LEFT JOIN hq_lcr r USING (dev, det, kind) LEFT JOIN hq_lcsw s USING (dev, det, kind)
  WHERE r.dev IS NOT NULL OR s.dev IS NOT NULL) c JOIN hq_win w USING (dev)),
f AS (SELECT * FROM hq_lcfp p WHERE NOT EXISTS (SELECT 1 FROM c WHERE c.dev = p.dev AND c.det = p.det
                                              AND c.a_ns < p.fz AND c.z_ns > p.fa))
SELECT c.* EXCLUDE (a_ns, z_ns) FROM c
WHERE c.switch OR NOT (EXISTS (SELECT 1 FROM f WHERE f.dev = c.dev AND f.det = c.det AND f.fz <= c.a_ns AND (f.fz - 1) // 86400000000000 = c.a_ns // 86400000000000) AND EXISTS (SELECT 1 FROM f WHERE f.dev = c.dev AND f.det = c.det AND f.fa >= c.z_ns AND f.fa // 86400000000000 = (c.z_ns - 1) // 86400000000000)))
SELECT a.*, w.s_ns + a.b0 * 300000000000 AS a_ns, least(w.s_ns + a.b1 * 300000000000, w.e_ns) AS z_ns
FROM a JOIN hq_win w USING (dev)
WHERE NOT EXISTS (SELECT 1 FROM a o WHERE o.dev = a.dev AND o.det = a.det AND o.kind <> a.kind AND o.switch = a.switch
                  AND o.b0 < a.b1 AND o.b1 > a.b0
                  AND (o.b1 - o.b0 > a.b1 - a.b0 OR (o.b1 - o.b0 = a.b1 - a.b0 AND o.kind < a.kind))))
),
hq_xfc AS MATERIALIZED (
SELECT x.* EXCLUDE (l0, l1),
  EXISTS (SELECT 1 FROM hq_lcres r WHERE r.dev = x.dev AND r.det = x.det AND x.drop_b1 > x.drop_b0
          AND least(r.b1, x.drop_b1) - greatest(r.b0, x.drop_b0) >= 0.5 * (x.drop_b1 - x.drop_b0)) AS lc_d,
  coalesce(x.l1 > x.l0 AND (SELECT sum(greatest(least(r.b1, x.l1) - greatest(r.b0, x.l0), 0)) FROM hq_lcres r
                            WHERE r.dev = x.dev AND r.det = x.det AND NOT r.switch) >= 0.5 * (x.l1 - x.l0), false) AS lc_l,
  coalesce((SELECT coalesce(sum(greatest(least(r.z_ns, x.md_z * 3600000000000) - greatest(r.a_ns, x.md_a * 3600000000000), 0)), 0) FROM hq_lcres r WHERE r.dev = x.dev AND r.det = x.det) > 0 OR EXISTS (SELECT 1 FROM hq_lcres r WHERE r.dev = x.dev AND r.det = x.det AND r.switch), false) AS lc_md,
  coalesce((SELECT coalesce(sum(greatest(least(r.z_ns, x.mr_z * 3600000000000) - greatest(r.a_ns, x.mr_a * 3600000000000), 0)), 0) FROM hq_lcres r WHERE r.dev = x.dev AND (r.det = x.det OR list_contains(r.gainers, x.det))) > 0 OR EXISTS (SELECT 1 FROM hq_lcres r WHERE r.dev = x.dev AND r.det = x.det AND r.switch), false) AS lc_mr
FROM (SELECT x.*, (epoch_ns(x.drop_t) - w.s_ns) // 300000000000 AS l0,
             coalesce((epoch_ns(x.drop_t2) - w.s_ns) // 300000000000, w.nb5) AS l1 FROM hq_xf x JOIN hq_win w USING (dev)) x
),
hq_finS AS MATERIALIZED (
SELECT x.dev, x.det, s.s_stuck_v AS v_stuck,
  CASE WHEN x.count_on_ok THEN (CASE WHEN (coalesce(x.hi_min_c, 0)) IS NULL OR isnan(coalesce(x.hi_min_c, 0)) OR (l.lim_count_on) IS NULL OR isnan(l.lim_count_on) THEN NULL WHEN (coalesce(x.hi_min_c, 0)) < (l.lim_count_on) THEN 0.0 ELSE least(0.35 + 0.40 * greatest(((coalesce(x.hi_min_c, 0)) - (l.lim_count_on)) / greatest((l.bad_count_on) - (l.lim_count_on), 1e-9), 0.0), 1.0) END) END AS v_count_on,
  CASE WHEN x.fn = 'Bike' THEN NULL WHEN x.silent THEN 0.0 WHEN x.lc_d THEN 0.0
       WHEN x.drop_b1 - x.drop_b0 < 3 THEN least(x.sd, 0.5) ELSE x.sd END AS v_dropout,
  CASE WHEN x.fn = 'Bike' OR NOT x.geh_ok THEN NULL WHEN x.lc_l THEN 0.0
WHEN x.geh_drop IS NOT NULL AND x.occ_own >= 3.0 AND x.occ_own >= 3.0 * coalesce(x.occ_mates, 1.0) THEN (CASE WHEN (x.geh_drop) IS NULL OR isnan(x.geh_drop) OR (l.lim_level) IS NULL OR isnan(l.lim_level) THEN NULL WHEN (x.geh_drop) < (l.lim_level) THEN 0.0 ELSE least(0.35 + 0.40 * greatest(((x.geh_drop) - (l.lim_level)) / greatest((l.bad_level) - (l.lim_level), 1e-9), 0.0), 1.0) END) WHEN x.occ_own >= 1.0 AND coalesce(x.occ_after, 0) >= 0.25 THEN 0.0
       WHEN x.geh_drop IS NOT NULL AND ((x.occ_mates IS NOT NULL AND x.occ_own / greatest(x.occ_mates, 1e-9) < 0.8)
            OR (x.occ_mates IS NULL AND x.occ_own < 0.8) OR x.own_after < 0.1 * x.exp_after)
       THEN (CASE WHEN (x.geh_drop) IS NULL OR isnan(x.geh_drop) OR (l.lim_level) IS NULL OR isnan(l.lim_level) THEN NULL WHEN (x.geh_drop) < (l.lim_level) THEN 0.0 ELSE least(0.35 + 0.40 * greatest(((x.geh_drop) - (l.lim_level)) / greatest((l.bad_level) - (l.lim_level), 1e-9), 0.0), 1.0) END) ELSE 0.0 END AS v_lvl_old,
  CASE WHEN x.fn = 'Bike' THEN NULL WHEN x.lc_l OR x.lc_md THEN 0.0 WHEN x.md_h IS NOT NULL THEN (CASE WHEN (x.md_geh) IS NULL OR isnan(x.md_geh) OR (CAST(15.0 AS DOUBLE)) IS NULL OR isnan(CAST(15.0 AS DOUBLE)) THEN NULL WHEN (x.md_geh) < (CAST(15.0 AS DOUBLE)) THEN 0.0 ELSE least(0.35 + 0.40 * greatest(((x.md_geh) - (CAST(15.0 AS DOUBLE))) / greatest((CAST(20.0 AS DOUBLE)) - (CAST(15.0 AS DOUBLE)), 1e-9), 0.0), 1.0) END) WHEN x.mh_ok THEN 0.0 END AS v_lvl_md,
  CASE WHEN v_lvl_old IS NULL AND v_lvl_md IS NULL THEN NULL
       ELSE greatest(coalesce(v_lvl_old, 0.0), coalesce(v_lvl_md, 0.0)) END AS v_level,
  CASE WHEN x.fn = 'Bike' THEN NULL WHEN x.lc_mr THEN 0.0 WHEN x.mr_h IS NOT NULL THEN (CASE WHEN (x.mr_geh) IS NULL OR isnan(x.mr_geh) OR (CAST(15.0 AS DOUBLE)) IS NULL OR isnan(CAST(15.0 AS DOUBLE)) THEN NULL WHEN (x.mr_geh) < (CAST(15.0 AS DOUBLE)) THEN 0.0 ELSE least(0.35 + 0.40 * greatest(((x.mr_geh) - (CAST(15.0 AS DOUBLE))) / greatest((CAST(20.0 AS DOUBLE)) - (CAST(15.0 AS DOUBLE)), 1e-9), 0.0), 1.0) END) WHEN x.mh_ok THEN 0.0 END AS v_rise,
  CASE WHEN x.fn = 'Bike' THEN NULL WHEN x.moved AND x.lc_d THEN 0.0
       WHEN x.moved AND coalesce(x.drop_lam, 0) < 50 THEN 0.0
       WHEN x.moved AND coalesce(x.drop_lam, 0) < 100 THEN least(x.sd, 0.5)
       WHEN x.moved THEN x.sd ELSE x.nd END AS v_night_drop,
  CASE WHEN x.fn <> 'Bike' AND coalesce(CASE WHEN x.silent THEN 0.0 ELSE x.sd END, 0)
            >= 0.42
       THEN CASE WHEN x.pv IS NOT NULL THEN 0.0 END ELSE x.pv END AS v_prof,
  CASE WHEN x.N >= 50 THEN (CASE WHEN (x.xr + 1e-9 * (x.xr > 0)::INT) IS NULL OR isnan(x.xr + 1e-9 * (x.xr > 0)::INT) OR (1.0) IS NULL OR isnan(1.0) THEN NULL WHEN (x.xr + 1e-9 * (x.xr > 0)::INT) < (1.0) THEN 0.0 ELSE least(0.35 + 0.40 * greatest(((x.xr + 1e-9 * (x.xr > 0)::INT) - (1.0)) / greatest((l.bad_rapid / l.lim_rapid) - (1.0), 1e-9), 0.0), 1.0) END) END AS v_rapid,
  CASE WHEN x.q5_use IS NOT NULL THEN (CASE WHEN (x.q5_use / l.lim_volume) IS NULL OR isnan(x.q5_use / l.lim_volume) OR (1.0) IS NULL OR isnan(1.0) THEN NULL WHEN (x.q5_use / l.lim_volume) < (1.0) THEN 0.0 ELSE least(0.35 + 0.40 * greatest(((x.q5_use / l.lim_volume) - (1.0)) / greatest((l.bad_volume / l.lim_volume) - (1.0), 1e-9), 0.0), 1.0) END) END
       AS v_volume,
  CASE WHEN x.spk_lvl >= 2 THEN 1.0 WHEN x.spk_lvl >= 1 THEN 0.5 WHEN x.spk_lvl IS NOT NULL THEN 0.0 END AS v_spiky,
  x.silent AS silent_night
FROM (SELECT x.*, coalesce(x.sd0, 0) >= 0.35 AND x.nf >= 0.8 AS silent,
             coalesce(x.sd0, 0) >= 0.35 AND x.nf >= 0.8 AND coalesce(x.sd0, 0) > coalesce(x.nd, 0) + 1e-9 AS moved,
             x.sd0 AS sd
      FROM (SELECT x.*, CASE WHEN x.n_on > 0 THEN CASE WHEN x.drop_in_stuck THEN 0.0
 ELSE (CASE WHEN (x.drop_lam) IS NULL OR isnan(x.drop_lam) OR (l.lim_dropout) IS NULL OR isnan(l.lim_dropout) THEN NULL WHEN (x.drop_lam) < (l.lim_dropout) THEN 0.0 ELSE least(0.35 + 0.40 * greatest(((x.drop_lam) - (l.lim_dropout)) / greatest((l.bad_dropout) - (l.lim_dropout), 1e-9), 0.0), 1.0) END) END END AS sd0,
                   (CASE WHEN (x.geh_night) IS NULL OR isnan(x.geh_night) OR (l.lim_night_drop) IS NULL OR isnan(l.lim_night_drop) THEN NULL WHEN (x.geh_night) < (l.lim_night_drop) THEN 0.0 ELSE least(0.35 + 0.40 * greatest(((x.geh_night) - (l.lim_night_drop)) / greatest((l.bad_night_drop) - (l.lim_night_drop), 1e-9), 0.0), 1.0) END) AS nd,
                   CASE WHEN x.hq_flat THEN 0.8 WHEN x.qzone THEN 0.0 WHEN x.tod_z IS NULL THEN NULL
                        WHEN x.tod_n_night >= CAST(20.0 AS DOUBLE) AND coalesce(x.tod_dph, -1) >= CAST(0.3 AS DOUBLE)
                             AND x.tod_z >= l.lim_prof * CAST(0.5833333333333334 AS DOUBLE) AND x.tod_z >= l.lim_prof
                             AND x.tod_z >= 1.5 * l.lim_prof THEN 0.8
                        WHEN x.tod_n_night >= CAST(20.0 AS DOUBLE) AND coalesce(x.tod_dph, -1) >= CAST(0.3 AS DOUBLE)
                             AND x.tod_z >= l.lim_prof * CAST(0.5833333333333334 AS DOUBLE) THEN 0.5 ELSE 0.0 END AS pv,
                   CASE WHEN x.n_fast >= 3 THEN x.fast_rate / l.lim_rapid ELSE 0.0 END AS xr,
                   CASE WHEN x.drop_b1 > x.drop_b0 THEN (SELECT count(*) FILTER (b.hr >= 21 OR b.hr < 5) FROM hq_bins5 b
                        WHERE b.dev = x.dev AND b.b >= x.drop_b0 AND b.b < x.drop_b1) / (x.drop_b1 - x.drop_b0)
                        ELSE 0.0 END AS nf
            FROM hq_xfc x JOIN hq_limits l USING (dev, det)) x) x
JOIN hq_limits l USING (dev, det) JOIN hq_stk s USING (dev, det)
),
hq_finF0 AS (
SELECT s.*, coalesce(v_stuck >= 0.35, false) AS f_stuck, coalesce(v_count_on >= 0.35, false) AS f_count_on, coalesce(v_dropout >= 0.35, false) AS f_dropout, coalesce(v_level >= 0.35, false) AS f_level, coalesce(v_night_drop >= 0.35, false) AS f_night_drop, coalesce(v_prof >= 0.35, false) AS f_prof, coalesce(v_rapid >= 0.35, false) AS f_rapid, coalesce(v_volume >= 0.35, false) AS f_volume, coalesce(v_spiky >= 0.35, false) AS f_spiky, coalesce(v_rise >= 0.35, false) AS f_rise, x.queue_pat8, x.c_occ_like, x.hours, x.status1, x.status AS status2, x.n_hmates, x.n_on,
       x.drop_b0, x.drop_b1, x.fn, x.c1_pre, x.hi_min_c, x.fe_n, x.fe_phi, x.drop_lam
FROM hq_finS s JOIN hq_xfc x USING (dev, det)
),
hq_finF1 AS (
SELECT * REPLACE (
    CASE WHEN f_dropout AND f_level AND NOT (v_dropout >= v_level) THEN false ELSE f_dropout END AS f_dropout,
    CASE WHEN f_dropout AND f_level AND v_dropout >= v_level THEN false ELSE f_level END AS f_level)
FROM (SELECT * REPLACE (f_level AND NOT (queue_pat8 AND coalesce(c_occ_like >= 0.8, false)
                                         AND NOT coalesce(v_lvl_md > v_lvl_old, false)) AS f_level,
                        f_dropout AND NOT (hours < 1) AS f_dropout),
             f_dropout AND hours < 1 AS w_dropout_short,
             status2 <> status1 AND n_hmates = 0 AND status1 IN ('suspect', 'bad') AS w_no_yardstick
      FROM hq_finF0)
),
hq_finF2 AS (
SELECT *, (f_stuck::INT + f_count_on::INT + f_dropout::INT + f_level::INT + f_night_drop::INT + f_prof::INT + f_rapid::INT + f_volume::INT + f_spiky::INT + f_rise::INT) = 1 AND ((f_dropout AND v_dropout < 0.42) OR (f_rapid AND v_rapid < 0.42) OR (f_level AND v_level < 0.42) OR (f_night_drop AND v_night_drop < 0.42) OR (f_prof AND v_prof < 0.42) OR (f_spiky AND v_spiky < 0.42) OR (f_rise AND v_rise < 0.42)) AS r7 FROM hq_finF1
),
hq_finF3 AS (
SELECT * REPLACE (f_stuck AND NOT r7 AS f_stuck, f_count_on AND NOT r7 AS f_count_on, f_dropout AND NOT r7 AS f_dropout, f_level AND NOT r7 AS f_level, f_night_drop AND NOT r7 AS f_night_drop, f_prof AND NOT r7 AS f_prof, f_rapid AND NOT r7 AS f_rapid, f_volume AND NOT r7 AS f_volume, f_spiky AND NOT r7 AS f_spiky, f_rise AND NOT r7 AS f_rise), f_stuck AND r7 AS b_stuck, f_count_on AND r7 AS b_count_on, f_dropout AND r7 AS b_dropout, f_level AND r7 AS b_level, f_night_drop AND r7 AS b_night_drop, f_prof AND r7 AS b_prof, f_rapid AND r7 AS b_rapid, f_volume AND r7 AS b_volume, f_spiky AND r7 AS b_spiky, f_rise AND r7 AS b_rise FROM hq_finF2
),
hq_finF4 AS (
SELECT *, CASE WHEN v_stuck IS NOT NULL OR v_count_on IS NOT NULL OR v_dropout IS NOT NULL OR v_level IS NOT NULL OR v_night_drop IS NOT NULL OR v_prof IS NOT NULL OR v_rapid IS NOT NULL OR v_volume IS NOT NULL OR v_spiky IS NOT NULL OR v_rise IS NOT NULL THEN (1 - CASE WHEN f_stuck THEN v_stuck ELSE 0 END) * (1 - CASE WHEN f_count_on THEN v_count_on ELSE 0 END) * (1 - CASE WHEN f_dropout THEN v_dropout ELSE 0 END) * (1 - CASE WHEN f_level THEN v_level ELSE 0 END) * (1 - CASE WHEN f_night_drop THEN v_night_drop ELSE 0 END) * (1 - CASE WHEN f_prof THEN v_prof ELSE 0 END) * (1 - CASE WHEN f_rapid THEN v_rapid ELSE 0 END) * (1 - CASE WHEN f_volume THEN v_volume ELSE 0 END) * (1 - CASE WHEN f_spiky THEN v_spiky ELSE 0 END) * (1 - CASE WHEN f_rise THEN v_rise ELSE 0 END) END AS score, (f_rapid OR f_volume)::INT + (f_prof)::INT + ((f_dropout AND NOT coalesce(v_dropout = 0.35, false) AND drop_b1 - drop_b0 >= 3 AND drop_lam >= 50) OR f_level OR f_night_drop)::INT + (f_rise)::INT + (f_spiky)::INT + ((f_stuck AND NOT coalesce(v_stuck = 0.35, false)) OR f_count_on)::INT AS n_ev, (f_rapid)::INT + (f_level OR f_night_drop)::INT + (f_rise)::INT + (f_prof OR f_volume OR f_spiky)::INT + ((f_dropout AND NOT coalesce(v_dropout = 0.35, false) AND drop_b1 - drop_b0 >= 3 AND drop_lam >= 50))::INT + ((f_stuck AND NOT coalesce(v_stuck = 0.35, false)) OR f_count_on)::INT AS n_fam,
       (drop_b1 - drop_b0) * 300 / 3600.0 AS silent_h,
       CASE WHEN (1 - CASE WHEN f_stuck THEN v_stuck ELSE 0 END) * (1 - CASE WHEN f_count_on THEN v_count_on ELSE 0 END) * (1 - CASE WHEN f_dropout THEN v_dropout ELSE 0 END) * (1 - CASE WHEN f_level THEN v_level ELSE 0 END) * (1 - CASE WHEN f_night_drop THEN v_night_drop ELSE 0 END) * (1 - CASE WHEN f_prof THEN v_prof ELSE 0 END) * (1 - CASE WHEN f_rapid THEN v_rapid ELSE 0 END) * (1 - CASE WHEN f_volume THEN v_volume ELSE 0 END) * (1 - CASE WHEN f_spiky THEN v_spiky ELSE 0 END) * (1 - CASE WHEN f_rise THEN v_rise ELSE 0 END) < 0.25 AND (v_stuck IS NOT NULL OR v_count_on IS NOT NULL OR v_dropout IS NOT NULL OR v_level IS NOT NULL OR v_night_drop IS NOT NULL OR v_prof IS NOT NULL OR v_rapid IS NOT NULL OR v_volume IS NOT NULL OR v_spiky IS NOT NULL OR v_rise IS NOT NULL) THEN 'bad' WHEN (1 - CASE WHEN f_stuck THEN v_stuck ELSE 0 END) * (1 - CASE WHEN f_count_on THEN v_count_on ELSE 0 END) * (1 - CASE WHEN f_dropout THEN v_dropout ELSE 0 END) * (1 - CASE WHEN f_level THEN v_level ELSE 0 END) * (1 - CASE WHEN f_night_drop THEN v_night_drop ELSE 0 END) * (1 - CASE WHEN f_prof THEN v_prof ELSE 0 END) * (1 - CASE WHEN f_rapid THEN v_rapid ELSE 0 END) * (1 - CASE WHEN f_volume THEN v_volume ELSE 0 END) * (1 - CASE WHEN f_spiky THEN v_spiky ELSE 0 END) * (1 - CASE WHEN f_rise THEN v_rise ELSE 0 END) < 0.70 AND (v_stuck IS NOT NULL OR v_count_on IS NOT NULL OR v_dropout IS NOT NULL OR v_level IS NOT NULL OR v_night_drop IS NOT NULL OR v_prof IS NOT NULL OR v_rapid IS NOT NULL OR v_volume IS NOT NULL OR v_spiky IS NOT NULL OR v_rise IS NOT NULL) THEN 'suspect'
            WHEN r7 OR w_dropout_short OR w_no_yardstick THEN 'watch' WHEN n_on < 20 THEN 'not_enough_data'
            ELSE 'ok' END AS st0
FROM hq_finF3
),
hq_finV AS MATERIALIZED (
SELECT * EXCLUDE (st0, st1, st2, st3),
       CASE WHEN st3 = 'bad' AND f_rapid AND (f_stuck::INT + f_count_on::INT + f_dropout::INT + f_level::INT + f_night_drop::INT + f_prof::INT + f_rapid::INT + f_volume::INT + f_spiky::INT + f_rise::INT) = 1 AND fe_n >= 8 AND fe_phi <= 2.0
                 AND fn IN ('Presence', 'Other', 'Mid') THEN 'suspect' ELSE st3 END AS health_st,
       CASE WHEN st3 = 'bad' AND f_rapid AND (f_stuck::INT + f_count_on::INT + f_dropout::INT + f_level::INT + f_night_drop::INT + f_prof::INT + f_rapid::INT + f_volume::INT + f_spiky::INT + f_rise::INT) = 1 AND fe_n >= 8 AND fe_phi <= 2.0
                 AND fn IN ('Presence', 'Other', 'Mid') THEN 'F'
            WHEN st3 <> st2 THEN 'C' WHEN st2 <> st1 THEN 'P' WHEN st1 <> st0 AND n_fam < 2 THEN 'A' ELSE '' END AS sev_rule
FROM (SELECT *, CASE WHEN fn = 'Count' AND c1_pre AND hi_min_c >= 30 AND st2 IN ('ok', 'watch') THEN 'suspect'
                     ELSE st2 END AS st3
      FROM (SELECT *, CASE WHEN st1 = 'suspect' AND (f_prof OR (f_dropout AND silent_h >= 4.0)) THEN 'bad' ELSE st1 END
                   AS st2
            FROM (SELECT *, CASE WHEN st0 = 'suspect' AND (n_fam >= 2 OR n_ev >= 2) THEN 'bad' ELSE st0 END AS st1
                  FROM hq_finF4)))
),
hq_pmode AS MATERIALIZED (
SELECT * FROM (
  SELECT o.dev, o.det, o.np, o.nps / o.np AS ps, o.med AS pmed
  FROM hq_onmed o JOIN hq_xf x USING (dev, det) JOIN hq_dcl c USING (dev, det)
  WHERE x.fn = 'Presence' AND NOT coalesce(lower(c.cfn) IN ('count', 'yellow_red', 'yellow-red')
                                          OR lower(c.cfn) LIKE 'advance%', false))
WHERE np >= 200 AND ps >= CAST(0.8 AS DOUBLE) AND round(pmed, 3) < CAST(0.2 AS DOUBLE)
),
hq_fnd AS MATERIALIZED (
SELECT v.dev, v.det, k.key, k.ki, k.cat, coalesce(CASE k.key WHEN 'stuck' THEN v.v_stuck WHEN 'count_on' THEN v.v_count_on WHEN 'dropout' THEN v.v_dropout WHEN 'level' THEN v.v_level WHEN 'night_drop' THEN v.v_night_drop WHEN 'prof' THEN v.v_prof WHEN 'rapid' THEN v.v_rapid WHEN 'volume' THEN v.v_volume WHEN 'spiky' THEN v.v_spiky WHEN 'rise' THEN v.v_rise END, 0.0) AS sc,
       CASE WHEN (v.sev_rule = 'C' AND k.key = 'count_on') OR (v.sev_rule = 'F' AND k.key = 'rapid') THEN 'suspect'
            WHEN v.sev_rule = 'P' AND (k.key = 'prof' OR (k.key = 'dropout' AND v.silent_h >= 4.0)) THEN 'bad'
            WHEN coalesce(CASE k.key WHEN 'stuck' THEN v.v_stuck WHEN 'count_on' THEN v.v_count_on WHEN 'dropout' THEN v.v_dropout WHEN 'level' THEN v.v_level WHEN 'night_drop' THEN v.v_night_drop WHEN 'prof' THEN v.v_prof WHEN 'rapid' THEN v.v_rapid WHEN 'volume' THEN v.v_volume WHEN 'spiky' THEN v.v_spiky WHEN 'rise' THEN v.v_rise END, 0.0) >= 0.75 THEN 'bad' ELSE 'suspect' END AS sev
FROM hq_finV v, (SELECT * FROM (VALUES ('stuck', 0, 'Stuck on'), ('count_on', 1, 'Stuck on'), ('dropout', 2, 'Counts fall'), ('level', 3, 'Counts fall'), ('night_drop', 4, 'Misses vehicles at night'), ('prof', 5, 'Busy at night'), ('rapid', 6, 'Too-fast actuations'), ('volume', 7, 'Too many for the traffic'), ('spiky', 8, 'Spiky counts'), ('rise', 9, 'Counts too high')) t(key, ki, cat)) k WHERE CASE k.key WHEN 'stuck' THEN v.f_stuck WHEN 'count_on' THEN v.f_count_on WHEN 'dropout' THEN v.f_dropout WHEN 'level' THEN v.f_level WHEN 'night_drop' THEN v.f_night_drop WHEN 'prof' THEN v.f_prof WHEN 'rapid' THEN v.f_rapid WHEN 'volume' THEN v.f_volume WHEN 'spiky' THEN v.f_spiky WHEN 'rise' THEN v.f_rise END OR (k.key = 'count_on' AND v.sev_rule = 'C')
),
hq_eps AS (
SELECT e.dev, e.det, count(*) FILTER (e.label <> 'queue') AS n_cnt, sum(e.dur) FILTER (e.label <> 'queue') AS tot,
       min(e.s) FILTER (e.label <> 'queue') AS t0, count(*) FILTER (e.label = 'not_queue') AS n_nq,
       min(e.busy_occ_e) FILTER (e.label = 'not_queue') AS nq_occ, count(*) FILTER (e.label = 'together') AS n_tog,
       sum(e.dur) FILTER (e.label = 'together') AS tog_tot, min(e.s) FILTER (e.label = 'together') AS tog_t0,
       arg_min(e.dur, e.s) FILTER (e.label = 'together') AS tog_dur,
       list_sort(list_distinct(flatten(list(e.mates) FILTER (e.label = 'together')))) AS tog_mates
FROM hq_efl e GROUP BY e.dev, e.det
),
hq_silmore AS MATERIALIZED (
SELECT u.dev, u.det, u.a, u.z, u.lam FROM hq_silall u JOIN hq_xf x USING (dev, det) JOIN hq_limits l USING (dev, det)
JOIN hq_win w USING (dev)
WHERE u.lam >= l.lim_dropout AND NOT (u.a < x.drop_b1 AND u.z > x.drop_b0) AND NOT EXISTS (SELECT 1 FROM hq_lcres c WHERE c.dev = u.dev AND c.det = u.det AND least(c.z_ns, w.s_ns + u.z * 300000000000) - greatest(c.a_ns, w.s_ns + u.a * 300000000000) >= 0.5 * (u.z - u.a) * 300000000000)
  AND (u.lam >= 50 OR (SELECT count(*) FILTER (b.hr >= 21 OR b.hr < 5) FROM hq_bins5 b
                       WHERE b.dev = u.dev AND b.b >= u.a AND b.b < u.z) < 0.8 * (u.z - u.a))
),
hq_evid AS (
SELECT f.dev, f.det, f.key, f.ki,
       CASE WHEN f.key = 'prof' AND x.hq_flat THEN 'No daily pattern' ELSE f.cat END AS cat, f.sc, f.sev, CASE f.key
  WHEN 'stuck' THEN CASE WHEN e.n_tog > 0 THEN 'held ON ' || CASE WHEN e.n_tog = 1
        THEN (CASE WHEN (e.tog_dur) < 300 THEN format('{:.0f} s', (e.tog_dur)) WHEN round_even((e.tog_dur) / 60.0, 0)::BIGINT < 60 THEN round_even((e.tog_dur) / 60.0, 0)::BIGINT || ' min' ELSE (round_even((e.tog_dur) / 60.0, 0)::BIGINT // 60) || ' h' || CASE WHEN round_even((e.tog_dur) / 60.0, 0)::BIGINT % 60 > 0 THEN ' ' || (round_even((e.tog_dur) / 60.0, 0)::BIGINT % 60) || ' min' ELSE '' END END) || ' from ' || strftime(make_timestamp((w.s_ns + round((e.tog_t0) * 1e9)::BIGINT) // 1000), '%a %H:%M')
        ELSE e.n_tog || ' times from ' || strftime(make_timestamp((w.s_ns + round((e.tog_t0) * 1e9)::BIGINT) // 1000), '%a %H:%M') || ', ' || (CASE WHEN (e.tog_tot) < 300 THEN format('{:.0f} s', (e.tog_tot)) WHEN round_even((e.tog_tot) / 60.0, 0)::BIGINT < 60 THEN round_even((e.tog_tot) / 60.0, 0)::BIGINT || ' min' ELSE (round_even((e.tog_tot) / 60.0, 0)::BIGINT // 60) || ' h' || CASE WHEN round_even((e.tog_tot) / 60.0, 0)::BIGINT % 60 > 0 THEN ' ' || (round_even((e.tog_tot) / 60.0, 0)::BIGINT % 60) || ' min' ELSE '' END END) || ' in total' END
        || ' together with det ' || array_to_string(e.tog_mates, ', ') || ' (same start and end: a common failure)'
      WHEN e.n_cnt > 1 THEN 'held ON ' || e.n_cnt || ' times, ' || (CASE WHEN (e.tot) < 300 THEN format('{:.0f} s', (e.tot)) WHEN round_even((e.tot) / 60.0, 0)::BIGINT < 60 THEN round_even((e.tot) / 60.0, 0)::BIGINT || ' min' ELSE (round_even((e.tot) / 60.0, 0)::BIGINT // 60) || ' h' || CASE WHEN round_even((e.tot) / 60.0, 0)::BIGINT % 60 > 0 THEN ' ' || (round_even((e.tot) / 60.0, 0)::BIGINT % 60) || ' min' ELSE '' END END) || ' in total' || CASE WHEN e.n_nq > 0 AND e.n_nq = e.n_cnt THEN format('; not a queue: at the same traffic it was ON {:.0f} % of the time', 100 * e.nq_occ) ELSE '' END
      WHEN e.n_cnt = 1 THEN 'held ON ' || (CASE WHEN (e.tot) < 300 THEN format('{:.0f} s', (e.tot)) WHEN round_even((e.tot) / 60.0, 0)::BIGINT < 60 THEN round_even((e.tot) / 60.0, 0)::BIGINT || ' min' ELSE (round_even((e.tot) / 60.0, 0)::BIGINT // 60) || ' h' || CASE WHEN round_even((e.tot) / 60.0, 0)::BIGINT % 60 > 0 THEN ' ' || (round_even((e.tot) / 60.0, 0)::BIGINT % 60) || ' min' ELSE '' END END) || ' from ' || strftime(make_timestamp((w.s_ns + round((e.t0) * 1e9)::BIGINT) // 1000), '%a %H:%M') || CASE WHEN e.n_nq > 0 AND e.n_nq = e.n_cnt THEN format('; not a queue: at the same traffic it was ON {:.0f} % of the time', 100 * e.nq_occ) ELSE '' END
      ELSE 'held ON up to ' || (CASE WHEN (x.stuck_x) < 300 THEN format('{:.0f} s', (x.stuck_x)) WHEN round_even((x.stuck_x) / 60.0, 0)::BIGINT < 60 THEN round_even((x.stuck_x) / 60.0, 0)::BIGINT || ' min' ELSE (round_even((x.stuck_x) / 60.0, 0)::BIGINT // 60) || ' h' || CASE WHEN round_even((x.stuck_x) / 60.0, 0)::BIGINT % 60 > 0 THEN ' ' || (round_even((x.stuck_x) / 60.0, 0)::BIGINT % 60) || ' min' ELSE '' END END) END
  WHEN 'count_on' THEN format('count zone ON up to {:.0f} % in light traffic for {:.0f} min', 100 * x.hi_occ_max,
                              x.hi_min_c)
  WHEN 'dropout' THEN CASE WHEN x.ph_lock THEN 'logs the same count as ' || (x.lk_n - 1) || ' other detectors on P' || trunc(x.ph)::BIGINT || ' every hour ' || CASE WHEN w.hours > 24.5 THEN strftime(make_timestamp((w.s_ns + x.drop_b0 * 300000000000) // 1000), '%a %H:%M') ELSE strftime(make_timestamp((w.s_ns + x.drop_b0 * 300000000000) // 1000), '%H:%M') END || '-' || CASE WHEN w.hours > 24.5 THEN strftime(make_timestamp((w.s_ns + x.drop_b1 * 300000000000) // 1000), '%a %H:%M') ELSE strftime(make_timestamp((w.s_ns + x.drop_b1 * 300000000000) // 1000), '%H:%M') END || format(' (~{:,.0f} an hour each): the inputs no longer see vehicles and log the same pulses (detector card in fail-safe, or a call placed on them)', x.lk_md) WHEN x.ph_drop THEN 'whole phase dropped ' || CASE WHEN w.hours > 24.5 THEN strftime(make_timestamp((w.s_ns + x.drop_b0 * 300000000000) // 1000), '%a %H:%M') ELSE strftime(make_timestamp((w.s_ns + x.drop_b0 * 300000000000) // 1000), '%H:%M') END || '-' || CASE WHEN w.hours > 24.5 THEN strftime(make_timestamp((w.s_ns + x.drop_b1 * 300000000000) // 1000), '%a %H:%M') ELSE strftime(make_timestamp((w.s_ns + x.drop_b1 * 300000000000) // 1000), '%H:%M') END || format(': {:,.0f} actuations where ~{:,.0f} expected; ', x.ph_obs, x.ph_exp) || x.ph_k || ' of ' || x.ph_m || ' detectors on P' || trunc(x.ph)::BIGINT || ' fell together while the rest of the signal counted normally (approach closed, or a shared card / cable)' ELSE 'no actuations ' || CASE WHEN w.hours > 24.5 THEN strftime(make_timestamp((w.s_ns + x.drop_b0 * 300000000000) // 1000), '%a %H:%M') ELSE strftime(make_timestamp((w.s_ns + x.drop_b0 * 300000000000) // 1000), '%H:%M') END || '-' || CASE WHEN w.hours > 24.5 THEN strftime(make_timestamp((w.s_ns + x.drop_b1 * 300000000000) // 1000), '%a %H:%M') ELSE strftime(make_timestamp((w.s_ns + x.drop_b1 * 300000000000) // 1000), '%H:%M') END || format('; ~{:,.0f} expected', x.drop_lam) || CASE WHEN x.sig_silent THEN '; every detector at the signal was silent while it kept serving greens (all inputs lost: detector rack / cabinet)' ELSE '' END END || coalesce((SELECT '; also ' || string_agg(strftime(make_timestamp((w.s_ns + u.a * 300000000000) // 1000), '%a %H:%M') || '-' || strftime(make_timestamp((w.s_ns + u.z * 300000000000) // 1000), '%a %H:%M') || format(' (~{:,.0f})', u.lam), ', ' ORDER BY u.a) FILTER (u.rk <= 3) || CASE WHEN count(*) > 3 THEN format(' and {} more', count(*) - 3) ELSE '' END FROM (SELECT *, row_number() OVER (ORDER BY round(lam, 6) DESC, a) AS rk FROM hq_silmore WHERE dev = x.dev AND det = x.det) u), '')
  WHEN 'night_drop' THEN CASE WHEN v.silent_night THEN CASE WHEN x.ph_lock THEN 'logs the same count as ' || (x.lk_n - 1) || ' other detectors on P' || trunc(x.ph)::BIGINT || ' every hour ' || CASE WHEN w.hours > 24.5 THEN strftime(make_timestamp((w.s_ns + x.drop_b0 * 300000000000) // 1000), '%a %H:%M') ELSE strftime(make_timestamp((w.s_ns + x.drop_b0 * 300000000000) // 1000), '%H:%M') END || '-' || CASE WHEN w.hours > 24.5 THEN strftime(make_timestamp((w.s_ns + x.drop_b1 * 300000000000) // 1000), '%a %H:%M') ELSE strftime(make_timestamp((w.s_ns + x.drop_b1 * 300000000000) // 1000), '%H:%M') END || format(' (~{:,.0f} an hour each): the inputs no longer see vehicles and log the same pulses (detector card in fail-safe, or a call placed on them)', x.lk_md) WHEN x.ph_drop THEN 'whole phase dropped ' || CASE WHEN w.hours > 24.5 THEN strftime(make_timestamp((w.s_ns + x.drop_b0 * 300000000000) // 1000), '%a %H:%M') ELSE strftime(make_timestamp((w.s_ns + x.drop_b0 * 300000000000) // 1000), '%H:%M') END || '-' || CASE WHEN w.hours > 24.5 THEN strftime(make_timestamp((w.s_ns + x.drop_b1 * 300000000000) // 1000), '%a %H:%M') ELSE strftime(make_timestamp((w.s_ns + x.drop_b1 * 300000000000) // 1000), '%H:%M') END || format(': {:,.0f} actuations where ~{:,.0f} expected; ', x.ph_obs, x.ph_exp) || x.ph_k || ' of ' || x.ph_m || ' detectors on P' || trunc(x.ph)::BIGINT || ' fell together while the rest of the signal counted normally (approach closed, or a shared card / cable)' ELSE 'no actuations ' || CASE WHEN w.hours > 24.5 THEN strftime(make_timestamp((w.s_ns + x.drop_b0 * 300000000000) // 1000), '%a %H:%M') ELSE strftime(make_timestamp((w.s_ns + x.drop_b0 * 300000000000) // 1000), '%H:%M') END || '-' || CASE WHEN w.hours > 24.5 THEN strftime(make_timestamp((w.s_ns + x.drop_b1 * 300000000000) // 1000), '%a %H:%M') ELSE strftime(make_timestamp((w.s_ns + x.drop_b1 * 300000000000) // 1000), '%H:%M') END || format('; ~{:,.0f} expected', x.drop_lam) || CASE WHEN x.sig_silent THEN '; every detector at the signal was silent while it kept serving greens (all inputs lost: detector rack / cabinet)' ELSE '' END END || coalesce((SELECT '; also ' || string_agg(strftime(make_timestamp((w.s_ns + u.a * 300000000000) // 1000), '%a %H:%M') || '-' || strftime(make_timestamp((w.s_ns + u.z * 300000000000) // 1000), '%a %H:%M') || format(' (~{:,.0f})', u.lam), ', ' ORDER BY u.a) FILTER (u.rk <= 3) || CASE WHEN count(*) > 3 THEN format(' and {} more', count(*) - 3) ELSE '' END FROM (SELECT *, row_number() OVER (ORDER BY round(lam, 6) DESC, a) AS rk FROM hq_silmore WHERE dev = x.dev AND det = x.det) u), '') WHEN x.geh_night_exp IS NOT NULL
       THEN format('night {:,.0f} actuations vs ~{:,.0f} expected (GEH {:.1f})', x.geh_night_n, x.geh_night_exp,
                   x.geh_night) ELSE 'fewer actuations at night' END
  WHEN 'level' THEN CASE WHEN coalesce(v.v_lvl_md, 0) > coalesce(v.v_lvl_old, 0) THEN format('counts fell to {:.0f} % of what its phase mates gave at that time of day on the other days: ', 100 * x.md_r) || strftime(make_timestamp(x.md_a * 3600000000), '%a %H:%M') || '-' || strftime(make_timestamp(x.md_z * 3600000000), '%H:%M') || format(' ({} h, ~{:.0f} an hour where ~{:.0f} expected; GEH {:.0f})', x.md_h, x.md_own, x.md_exp, x.md_geh) ELSE CASE WHEN x.own_after < 0.1 * x.exp_after THEN 'nearly stopped counting'
       WHEN x.occ_own >= 3.0 AND x.occ_own >= 3.0 * coalesce(x.occ_mates, 1.0) THEN format('counts fell while its time ON rose {:.0f} x (holds ON longer: stuck, or a changed '
                                         || 'setting)', x.occ_own)
       ELSE 'counts and occupancy both fell' END || CASE WHEN x.drop_t IS NOT NULL THEN ' from ' || strftime(x.drop_t, '%a %H:%M') ELSE '' END
       || CASE WHEN x.drop_t2 IS NOT NULL THEN ' to ' || strftime(x.drop_t2, '%a %H:%M') ELSE '' END
       || format(': {:.0f} per hour where its phase mates'' traffic gives ~{:.0f} (GEH {:.1f})', x.own_after,
                 x.exp_after, x.geh_drop) END
  WHEN 'rapid' THEN format('{:,} of {:,} ONs follow the previous ON within {:.2g} s ({:.0f} per 1,000; '
                           || 'limit {:.0f})', x.n_fast, x.N, x.gap_lim, x.fast_rate, l.lim_rapid)
       || CASE WHEN x.pair_share >= 0.9 THEN '; counts each vehicle twice' ELSE '' END
       || CASE WHEN v.sev_rule = 'F' THEN ', spread evenly with traffic - flickers as vehicles pass, check zone setup'
               ELSE '' END
  WHEN 'spiky' THEN format('{:.0f} bursts{} in 10-min counts at 3 x its usual level or more (largest GEH {:.0f}); '
                           || '~{:,.0f} vehicles a day', x.spk_bursts, CASE WHEN x.hours > 24.5 THEN ' a day' ELSE '' END,
                           x.spk_gmax, x.spk_daily)
  WHEN 'volume' THEN format('busiest 5 min {:,.0f} veh/h/lane vs limit {:,.0f}', x.q5_use, l.lim_volume)
  WHEN 'rise' THEN CASE WHEN x.mr_r >= 100 THEN 'counts rose far above its phase mates, which gave almost nothing at that time of day on the other days: ' ELSE format('counts rose to {:.1f} x what its phase mates gave at that time of day on the other days: ', x.mr_r) END || strftime(make_timestamp(x.mr_a * 3600000000), '%a %H:%M') || '-' || strftime(make_timestamp(x.mr_z * 3600000000), '%H:%M') || format(' ({} h, GEH {:.0f}): extra calls', x.mr_h, x.mr_geh)
  WHEN 'prof' THEN CASE WHEN x.hq_flat THEN format('counts do not follow the day: ~{:.0f} an hour by day, ~{:.0f} at night, while '
       || 'the rest of the signal drops from ~{:,.0f} to ~{:,.0f} (noise, not traffic)',
       x.flat_d, x.flat_n, x.flat_rd, x.flat_rn) ELSE
       format('night level {:.0f} % of its busiest hours; phase mates ', 100 * x.tod_r ^ 2)
       || CASE WHEN x.tod_mate_r IS NOT NULL THEN format('{:.0f} %', 100 * x.tod_mate_r ^ 2) ELSE 'none' END END
  END AS txt
FROM hq_fnd f JOIN hq_xf x USING (dev, det) JOIN hq_finV v USING (dev, det) JOIN hq_limits l USING (dev, det) JOIN hq_win w USING (dev)
LEFT JOIN hq_eps e USING (dev, det)
),
hq_lctx AS (
SELECT r.dev, r.det, 'closure' AS key, 99 AS ki, CASE WHEN r.switch THEN 'Timing plan' ELSE 'Lane closure' END
       AS cat, -1.0 AS sc, 'info' AS sev,
       CASE WHEN r.switch THEN strftime(make_timestamp((r.a_ns) // 1000), '%a %H:%M') || '-' || strftime(make_timestamp((r.z_ns) // 1000), '%a %H:%M') || CASE WHEN r.kind = 'd' THEN ': silent' ELSE ': counts low' END
                               || ' while coordination pattern ' || r.pat || ' is in effect'
                               || coalesce('; its traffic moves to ' || r.glab, '')
            ELSE strftime(make_timestamp((r.a_ns) // 1000), '%a %H:%M') || '-' || strftime(make_timestamp((r.z_ns) // 1000), '%a %H:%M') || format(': ~{:,.0f} of its vehicles moved to ', r.loss) || r.glab END AS txt, r.a_ns AS o
FROM hq_lcres r
),
hq_lcx AS MATERIALIZED (
SELECT dev, det, bool_and(switch) AS all_switch FROM hq_lcres GROUP BY dev, det
),
hq_cats AS (
SELECT dev, det, string_agg(cat || ' (' || sev || ')', '; ' ORDER BY s DESC, ki0, cat) AS health_categories,
       string_agg(cat || ': ' || txt, '; ' ORDER BY s DESC, ki0, cat) AS parts, bool_or(lane) AS lane_cat
FROM (SELECT dev, det, cat, max(sc) AS s, min(ki) AS ki0,
             CASE WHEN bool_or(sev = 'bad') THEN 'bad' WHEN bool_or(sev = 'info') THEN 'info' ELSE 'suspect' END AS sev,
             string_agg(txt, '; ' ORDER BY ki, o, txt) AS txt, bool_or(key IN ('stuck', 'rapid', 'volume', 'prof', 'spiky')) AS lane
      FROM (SELECT dev, det, key, ki, cat, sc, sev, txt, 0::BIGINT AS o FROM hq_evid UNION ALL SELECT * FROM hq_lctx) GROUP BY dev, det, cat)
GROUP BY dev, det
),
hq_outx0 AS (
SELECT v.dev, v.det, CASE WHEN k.dev IS NOT NULL AND v.health_st IN ('ok', 'watch', 'not_enough_data') THEN CASE WHEN k.all_switch THEN 'timing_plan' ELSE 'lane_closure' END ELSE v.health_st END AS health_status,
       CASE WHEN v.health_st <> 'not_enough_data' AND NOT isnan(v.score) THEN v.score END AS health_score,
       CASE WHEN v.health_st IN ('suspect', 'bad') OR k.dev IS NOT NULL
            THEN b.lb || ' - ' || coalesce(c.parts, v.health_st)
            WHEN v.health_st = 'watch' THEN b.lb || ' - watch' || CASE WHEN v.wt <> '' THEN ': ' || v.wt ELSE '' END
            WHEN v.health_st = 'not_enough_data' THEN b.lb || ' - too few actuations to judge'
            ELSE b.lb || ' - nothing unusual' END AS health_reason,
       coalesce(c.health_categories, '') AS health_categories,
       CASE x.cfg WHEN 'C1' THEN format('Extension time on a count zone: {:,} of {:,} ONs logged again without an OFF',
                                        x.n_rep, trunc(x.n_on)::BIGINT)
                  WHEN 'C2' THEN 'Set to pulse but holds ON: ' || CASE WHEN x.long_max_s IS NOT NULL
                       THEN x.n_ge5 || ' ONs over 5 s, longest ' || (CASE WHEN (x.long_max_s) < 300 THEN format('{:.0f} s', (x.long_max_s)) WHEN round_even((x.long_max_s) / 60.0, 0)::BIGINT < 60 THEN round_even((x.long_max_s) / 60.0, 0)::BIGINT || ' min' ELSE (round_even((x.long_max_s) / 60.0, 0)::BIGINT // 60) || ' h' || CASE WHEN round_even((x.long_max_s) / 60.0, 0)::BIGINT % 60 > 0 THEN ' ' || (round_even((x.long_max_s) / 60.0, 0)::BIGINT % 60) || ' min' ELSE '' END END)
                       ELSE 'holds ON although set to pulse' END ELSE '' END AS health_config,
       v.wt AS health_watch
FROM (SELECT v.*, array_to_string(list_filter([
        CASE WHEN v.w_no_yardstick THEN 'no healthy phase mate to compare with' END,
        CASE WHEN v.b_stuck THEN 'borderline stuck on' WHEN v.b_count_on THEN 'borderline stuck on' WHEN v.b_dropout THEN 'borderline counts fall' WHEN v.b_level THEN 'borderline counts fall' WHEN v.b_night_drop THEN 'borderline misses vehicles at night' WHEN v.b_prof THEN 'borderline busy at night' WHEN v.b_rapid THEN 'borderline too-fast actuations' WHEN v.b_volume THEN 'borderline too many for the traffic' WHEN v.b_spiky THEN 'borderline spiky counts' WHEN v.b_rise THEN 'borderline counts too high' END,
        CASE WHEN v.w_dropout_short AND NOT (v.b_dropout OR v.b_level) THEN 'borderline counts fall' END],
        lambda x: x IS NOT NULL), '; ') AS wt FROM hq_finV v) v
JOIN hq_xf x USING (dev, det) LEFT JOIN hq_cats c USING (dev, det)
LEFT JOIN hq_lcx k USING (dev, det)
JOIN (SELECT x.dev, x.det, 'det ' || x.det || ': ' || CASE WHEN x.ph IS NOT NULL THEN 'P' || trunc(x.ph)::BIGINT || ' ' ELSE '' END || replace(x.fn, 'Yellow_Red', 'Yellow-red') || CASE WHEN x.lanes IS NOT NULL AND coalesce(c.lane_cat, false) THEN ' (covers '
             || greatest(round_even(x.lanes, 0)::BIGINT, 1) || ' lane' || CASE WHEN greatest(round_even(x.lanes, 0)::BIGINT, 1) > 1 THEN 's' ELSE '' END || ')' ELSE '' END AS lb
      FROM hq_xf x LEFT JOIN hq_cats c USING (dev, det)) b USING (dev, det)
),
hq_outx AS (
SELECT o.dev, o.det, CASE WHEN p.dev IS NOT NULL AND o.health_status IN ('ok', 'watch', 'lane_closure', 'timing_plan') THEN 'suspect' ELSE o.health_status END AS health_status,
       CASE WHEN p.dev IS NOT NULL AND o.health_status IN ('ok', 'watch', 'lane_closure', 'timing_plan') THEN least(coalesce(o.health_score, 1.0), 0.69) ELSE o.health_score END AS health_score,
       CASE WHEN p.dev IS NULL THEN o.health_reason
            WHEN o.health_status IN ('ok', 'watch') THEN split_part(o.health_reason, ' - ', 1) || ' - ' || format('Configuration: looks set to pulse mode: {:.0f} % of its {:,} calls last 0.2 s or less; a presence detector should keep its call ON while a vehicle is on it', 100 * p.ps, p.np)
            ELSE o.health_reason || '; ' || format('Configuration: looks set to pulse mode: {:.0f} % of its {:,} calls last 0.2 s or less; a presence detector should keep its call ON while a vehicle is on it', 100 * p.ps, p.np) END AS health_reason,
       CASE WHEN p.dev IS NULL THEN o.health_categories WHEN o.health_categories = '' THEN 'Configuration (suspect)'
            ELSE o.health_categories || '; Configuration (suspect)' END AS health_categories,
       o.health_config, o.health_watch
FROM hq_outx0 o LEFT JOIN hq_pmode p USING (dev, det)
),
hq_outf AS (
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
                                            lambda x: x IS NOT NULL), '; ')
         END AS health_reason, '' AS health_categories, '' AS health_config, '' AS health_watch
FROM (SELECT * REPLACE (status_u AS status, hs_u AS hs, s_dropout_u AS s_dropout, s_stuck_u AS s_stuck) FROM hq_stage1) s
JOIN hq_sig0 g USING (dev) JOIN hq_win w USING (dev) ANTI JOIN hq_xbase x USING (dev, det)
),
hq_cgrp AS (
SELECT dev, g, min(s) AS s0, first(dur ORDER BY s, dur) AS dur0, count(DISTINCT round_even(s / 60, 0)) AS k
FROM (SELECT e.dev, e.s, e.dur, list_sort(list_distinct(list_append(coalesce(e.mates, []), e.det))) AS g
      FROM hq_efl e WHERE e.common)
GROUP BY dev, g
),
hq_snote AS (
SELECT w.dev, array_to_string(list_filter([
  CASE WHEN t.dev IS NOT NULL THEN t.n_odd || ' of ' || t.n_ok || ' detectors at this signal had an unusual night on '
       || strftime(make_timestamp((t.d0) // 1000), '%a %d %b') || ': a signal-wide event or an input change, not one bad detector' END,
  c.txt, CASE WHEN sr.dev IS NOT NULL THEN 'no detector logged an actuation ' || strftime(make_timestamp((w.s_ns + sr.a * 300000000000) // 1000), '%a %H:%M') || '-' || strftime(make_timestamp((w.s_ns + sr.z * 300000000000) // 1000), '%a %H:%M') || ' while the signal kept serving greens: all detector inputs lost (detector rack / cabinet)' END], lambda x: x IS NOT NULL), '; ') AS health_signal_note
FROM hq_win w LEFT JOIN hq_todnote t USING (dev) LEFT JOIN hq_sgr sr USING (dev)
LEFT JOIN (SELECT g.dev, string_agg(len(g.g) || ' detectors (det ' || array_to_string(g.g, ', ') || ') held ON together '
                  || CASE WHEN g.k = 1 THEN 'from ' || strftime(make_timestamp((w.s_ns + round(g.s0 * 1e9)::BIGINT) // 1000), '%a %H:%M') || ' for '
                          || (CASE WHEN (g.dur0) < 300 THEN format('{:.0f} s', (g.dur0)) WHEN round_even((g.dur0) / 60.0, 0)::BIGINT < 60 THEN round_even((g.dur0) / 60.0, 0)::BIGINT || ' min' ELSE (round_even((g.dur0) / 60.0, 0)::BIGINT // 60) || ' h' || CASE WHEN round_even((g.dur0) / 60.0, 0)::BIGINT % 60 > 0 THEN ' ' || (round_even((g.dur0) / 60.0, 0)::BIGINT % 60) || ' min' ELSE '' END END) ELSE g.k || ' times from ' || strftime(make_timestamp((w.s_ns + round(g.s0 * 1e9)::BIGINT) // 1000), '%a %H:%M') END
                  || ' (same start and end): one common failure, check the cabinet first', '; ' ORDER BY g.s0, g.g) AS txt
           FROM hq_cgrp g JOIN hq_win w USING (dev) GROUP BY g.dev) c USING (dev)
)
SELECT w.DeviceId, o.det AS Detector, o.health_status, o.health_score, o.health_reason, o.health_categories,
       o.health_config, o.health_watch, n.health_signal_note
FROM (SELECT * FROM hq_outx UNION ALL BY NAME SELECT * FROM hq_outf) o JOIN hq_win w USING (dev) JOIN hq_snote n USING (dev)
ORDER BY CAST(w.DeviceId AS VARCHAR), o.det
