{# Holds (continuous ONs >= 5 min): held ON together with other detectors (common), and the queue context of each
   (health_duckdb._holds: health_v4_stats.episodes_ctx) -> ectx (one row per hold). #}
{% set tol = LJ['common_hold_s']|dbl %}
-- @step hpair
SELECT a.hid, b.det AS dj, abs(b.s - a.s) <= {{ tol }} AND abs(b.f - a.f) <= {{ tol }} AS near
FROM hold a JOIN hold b ON a.dev = b.dev AND a.det <> b.det AND least(a.f, b.f) > greatest(a.s, b.s)
LEFT JOIN twin t ON t.dev = a.dev AND t.i = a.det AND t.j = b.det WHERE t.i IS NULL
-- @step hcom mat
SELECT h.hid, coalesce(p.anyov, false) AS n_ov_any, coalesce(p.anyov AND p.allnear, false) AS common, p.mates
FROM hold h LEFT JOIN (SELECT hid, true AS anyov, bool_and(near) AS allnear, list(DISTINCT dj ORDER BY dj) AS mates
                       FROM hpair GROUP BY hid) p USING (hid)
{# the hold in bs-bins: [ia, iz) touched, full bins [fa, fz) #}
-- @step hb mat
SELECT h.hid, h.dev, h.det, h.s, h.f, h.dur, x.ph AS p, w.bs, w.nbb, floor(h.s / w.bs)::INT AS ia,
       ceil(h.f / w.bs)::INT AS iz, ceil(h.s / w.bs)::INT AS fa, floor(h.f / w.bs)::INT AS fz
FROM hold h JOIN win w USING (dev) LEFT JOIN xbase x USING (dev, det)
{# healthy phase mates not themselves held >= 99 % ON for >= 80 % of the hold #}
-- @step hmate mat
SELECT h.hid, z.det AS r FROM hb h JOIN dz z ON z.dev = h.dev AND z.ph = h.p AND z.det <> h.det
WHERE z.status1 IS DISTINCT FROM 'bad' AND NOT (least(h.iz, h.nbb) > h.ia AND (
  SELECT count(*) FILTER (g.occ >= 0.99) FROM gb g WHERE g.dev = h.dev AND g.det = z.det
  AND g.k >= h.ia AND g.k < least(h.iz, h.nbb)) / (least(h.iz, h.nbb) - h.ia) >= 0.8)
-- @step hpm
SELECT m.hid, g.k, avg(g.occ) AS pm, bool_or(g.occ >= t.qt) AS hit
FROM hmate m JOIN hb h USING (hid) JOIN gb g ON g.dev = h.dev AND g.det = m.r
LEFT JOIN xm x ON x.dev = h.dev AND x.det = m.r LEFT JOIN ({{ T['qt']|values('type, qt') }}) t ON t.type = x.fn || ' ' || x.span
GROUP BY m.hid, g.k
-- @step htr
SELECT h.hid, g.k, sum(coalesce(g.n, 0)) AS tot FROM hb h
JOIN (SELECT h.hid, z.det AS r, bool_or(z.tr) OVER (PARTITION BY h.hid) AS anytr, z.tr
      FROM hb h JOIN dz z ON z.dev = h.dev AND z.ph = h.p AND z.det <> h.det) m ON m.hid = h.hid AND (m.tr OR NOT m.anytr)
JOIN gb g ON g.dev = h.dev AND g.det = m.r GROUP BY h.hid, g.k
-- @step hrh
SELECT h.hid, bool_or(g.occ >= 0.95) AS ref_held_e FROM hb h JOIN rmm m ON m.dev = h.dev AND m.det = h.det
JOIN gb g ON g.dev = h.dev AND g.det = m.r AND g.k >= h.ia AND g.k < least(h.iz, h.nbb) GROUP BY h.hid
-- @step ectx mat
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
  JOIN htr t USING (hid) GROUP BY h.hid) t USING (hid)
