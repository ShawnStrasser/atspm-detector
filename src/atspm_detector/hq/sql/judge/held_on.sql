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
{% if opt['mostly_on'] %}
{% set MO = LJ['mostly_on'] %}
{# note 157e/f (mostly_on): a detector mostly ON -- 10-min bins (covered, < unk_s s unknown) at >= occ of the time and
   a run of >= bins such bins in a row, on average >= ratio x (median over the run) what its own level (median of the bins 20-120 min away on
   both sides, >= 3 of them) times its phase mates' rise gives, and the mates ON < mates_share of its time (else congestion); flickering, so one hold (stuck on) never reaches its limit.  mo_nm = phase
   mates seen in the run (not rated bad, not held >= 95 % ON in the bin: a mate stuck with it is no traffic; none: no way to tell a queue from a fault, the verdict caps it at suspect).  A run that holds
   (continuous ONs >= 5 min) cover half of is left to stuck on, which judges it with its queue test #}
-- @step mo10 mat
SELECT dev, det, b // 2 AS t, sum(occ) / 600.0 AS occ, bool_and(cov) AND sum(unk) < {{ MO['unk_s']|dbl }} AS c
FROM g5 GROUP BY dev, det, b // 2
-- @step mo10l mat
SELECT a.dev, a.det, a.t, a.occ, a.c, coalesce(m.mo, 0.0) AS mo, coalesce(m.nm, 0) AS nm,
       list(CASE WHEN a.c THEN a.occ END) OVER wl || list(CASE WHEN a.c THEN a.occ END) OVER wr AS ol,
       list(CASE WHEN a.c THEN coalesce(m.mo, 0.0) END) OVER wl || list(CASE WHEN a.c THEN coalesce(m.mo, 0.0) END) OVER wr AS ml
FROM mo10 a LEFT JOIN (
  SELECT z.dev, z.det, b.t, avg(b.occ) AS mo, count(*) AS nm
  FROM dz z JOIN dz y ON y.dev = z.dev AND y.ph = z.ph AND y.det <> z.det JOIN mo10 b ON b.dev = y.dev AND b.det = y.det
  WHERE z.ph IS NOT NULL AND b.c AND b.occ < 0.95 AND y.status1 IS DISTINCT FROM 'bad' GROUP BY z.dev, z.det, b.t) m USING (dev, det, t)
WINDOW wl AS (PARTITION BY a.dev, a.det ORDER BY a.t ROWS BETWEEN 12 PRECEDING AND 2 PRECEDING),
       wr AS (PARTITION BY a.dev, a.det ORDER BY a.t ROWS BETWEEN 2 FOLLOWING AND 12 FOLLOWING)
-- @step mo_runs mat
SELECT * FROM (SELECT dev, det, min(t) AS a, max(t) + 1 AS z, avg(occ) AS occ, avg(mo) AS mo, max(nm) AS nm, median(ex) AS ex FROM (
  SELECT *, t - row_number() OVER (PARTITION BY dev, det ORDER BY t) AS grp FROM (
    SELECT *, greatest(list_median(list_filter(ol, lambda v: v IS NOT NULL)), {{ MO['floor']|dbl }})
              * greatest(mo / greatest(coalesce(list_median(list_filter(ml, lambda v: v IS NOT NULL)), 0.0), {{ MO['floor']|dbl }}), 1.0) AS ex
    FROM mo10l WHERE c AND len(list_filter(ol, lambda v: v IS NOT NULL)) >= 3)
  WHERE occ >= {{ MO['occ']|dbl }})
GROUP BY dev, det, grp HAVING count(*) >= {{ MO['bins'] }} AND avg(occ) >= {{ MO['ratio']|dbl }} * median(ex)
  AND (max(nm) = 0 OR avg(mo) < {{ MO['mates_share']|dbl }} * avg(occ))) r
WHERE NOT coalesce((SELECT sum(least(h.f, r.z * 600.0) - greatest(h.s, r.a * 600.0)) FROM hold h WHERE h.dev = r.dev
                    AND h.det = r.det AND h.f > r.a * 600.0 AND h.s < r.z * 600.0) >= 0.5 * (r.z - r.a) * 600.0, false)
-- @step mon
SELECT dev, det, 10.0 * sum(z - a) AS mo_min, 10.0 * max(z - a) AS mo_long, arg_max(a, z - a) AS mo_t0,
       arg_max(occ, z - a) AS mo_occ, arg_max(mo, z - a) AS mo_mocc, max(nm) AS mo_nm, count(*) AS mo_n
FROM mo_runs GROUP BY dev, det
{% endif %}
{% if opt['silent_not_held'] or opt['drop_not_held'] %}
{# note 158: the time each detector is held ON (s, f: seconds from the window start) -- continuous ONs >= 5 min (stuck on, held
   ON together; >= 15 min, as stage 1's silence-in-a-hold rule: a shorter one is a queue, not a fault), a Count zone held ON
   (hib bins), a mostly-ON run.  A held detector sends no new actuations, so its silence and
   its count drop leave this time out: one fault, not two #}
-- @step hiv mat
SELECT dev, det, s, f FROM hold WHERE span >= 900
UNION ALL SELECT h.dev, h.det, h.k * w.bs, (h.k + 1) * w.bs FROM hib h JOIN win w USING (dev)
{% if opt['mostly_on'] %}UNION ALL SELECT dev, det, a * 600.0, z * 600.0 FROM mo_runs
{% endif %}
{% endif %}
{% if opt['drop_not_held'] %}
{# the clock hours (hid) a hold touches #}
-- @step hheld mat
SELECT DISTINCT dev, det, unnest(range((w.s_ns + round(h.s * 1e9)::BIGINT) // {{ NSH }}, (w.s_ns + round(h.f * 1e9)::BIGINT - 1) // {{ NSH }} + 1)) AS hid
FROM hiv h JOIN win w USING (dev) WHERE h.f > h.s
{% endif %}
