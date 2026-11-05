{# Count zone held ON in light traffic (health_duckdb._held_on: score_v4c.occ_hi_c + note 135 green test) -> ohi, hib
   (the bins). #}
{% import "adapters.sql" as ad with context %}
-- @step xm mat
SELECT x.*, CASE WHEN x.fn <> 'Count' THEN '' WHEN a.dur_p50 IS NULL THEN 'unknown' WHEN a.dur_p50 <= 0.25 THEN 'pulse'
                 ELSE 'normal' END AS cmode
FROM xbase x LEFT JOIN act a USING (dev, det)
-- @step ohr mat
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
LEFT JOIN ({{ T['olp']|values('bs, tl, lp') }}) l ON l.bs = h.bs AND l.tl = h.tl LEFT JOIN ({{ T['ctp']|values('bs, tp') }}) t ON t.bs = h.bs
{# note 135: the zone ON for >= 20 % of its phase's green after the first 5 s (>= 60 s of such green in the bin) #}
-- @step hicd
SELECT DISTINCT dev, det, xph AS ph FROM ohr WHERE hic AND xph IS NOT NULL
{{ ad.gonb() }}
-- @step hib mat
SELECT h.dev, h.det, h.k, h.occ FROM ohr h JOIN gonb g USING (dev, det, k)
WHERE h.hic AND h.xph IS NOT NULL AND g.gav >= 60 AND coalesce(g.gon, 0) >= 0.20 * g.gav
-- @step ohi
SELECT o.dev, o.det, count(h.k) * any_value(o.bs) / 60.0 AS hi_min_c, max(h.occ) AS hi_occ_max
FROM (SELECT DISTINCT dev, det, bs FROM ohr) o LEFT JOIN hib h USING (dev, det) GROUP BY o.dev, o.det
