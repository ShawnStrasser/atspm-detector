{# The extractor, green overlap (design 4.1 On_Green5): the time ON of chains not over a gap inside a valid green of the
   detector's phase, after the green's first 5 s (today's gonb.gon).  Detectors: those with a phase whose function is in
   OVFN (None: every one, as the store keeps it, design decision 6; batch health() passes the functions the judge reads:
   Count zones, held_on).  The greens a chain overlaps: the last one starting at or before its start (k0), and when the
   chain runs past the next one's start, every one up to the last starting before its end (k1).  A chain of no length
   (ON and OFF at one instant) inside such a green adds no time but still makes its bin's row (inst), as today's.  A row
   still depending on an open chain or green carries its label (ctag; pph, ptag).  A chain carried in counts from the
   chunk's cut l0 (design 3.2 rule 1): what it overlapped before was written then. #}
-- @step xov mat
WITH c AS (SELECT c.dev, c.det, greatest(c.s, l.l0) AS s, c.f, c.ctag, d.ph, l.lz
           FROM (SELECT dev, det, s, f, NULL::BIGINT AS ctag FROM xcc WHERE NOT gap
                 UNION ALL SELECT o.dev, o.det, o.s, l.lz, o.s FROM xoc o JOIN xlast l USING (dev)) c
           JOIN xovd d USING (dev, det) JOIN xlast l USING (dev)),
lo AS (SELECT c.*, p.k AS k0, p.a5n, p.a5, p.z, p.op, p.ph AS gph, p.tag
       FROM c ASOF LEFT JOIN xg5 p ON c.dev = p.dev AND c.ph = p.ph AND c.s >= p.a5),
one AS (SELECT dev, det, ctag, gph, op, tag, greatest(s, a5) AS s, least(f, coalesce(z, lz)) AS f,
               lo.s = lo.f AND lo.s > a5 AND (z IS NULL OR lo.s < z) AS inst
        FROM lo WHERE k0 IS NOT NULL AND (a5n IS NULL OR f <= a5n)),
many AS (SELECT r.dev, r.det, r.ctag, p.ph AS gph, p.op, p.tag, greatest(r.s, p.a5) AS s, least(r.f, coalesce(p.z, r.lz)) AS f, false AS inst
         FROM (SELECT l.dev, l.det, l.s, l.f, l.ctag, l.ph, l.lz, unnest(range(coalesce(l.k0, 1), p.k + 1))::INT AS k
               FROM (SELECT * FROM lo WHERE NOT (k0 IS NOT NULL AND (a5n IS NULL OR f <= a5n))) l
               ASOF JOIN xg5 p ON l.dev = p.dev AND l.ph = p.ph AND l.f > p.a5) r
         JOIN xg5 p ON p.dev = r.dev AND p.ph = r.ph AND p.k = r.k)
SELECT dev, det, ctag, CASE WHEN op THEN gph END AS pph, CASE WHEN op THEN tag END AS ptag, s, f
FROM (SELECT * FROM one UNION ALL SELECT * FROM many) WHERE f > s OR inst
