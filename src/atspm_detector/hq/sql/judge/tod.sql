{# Busy at night (24-h samples, h116; health_duckdb._tod): per whole day the night level against its type's model and its
   phase mates; the day with the strongest finding is kept; a signal note when many detectors had an unusual night. #}
{% set P = LJ['busy'] %}
-- @step todg mat
{{ T['todg']|values('key, b0, b1, b2, sc, occ, o0, o1, so, med, mad') }}
{# the whole days (00:00-24:00) of each 24-h window; one not starting at midnight is scored by clock hour as one day #}
-- @step tdays mat
WITH w AS (SELECT dev, s_ns, e_ns, nbb, T, s_ns - sod + CASE WHEN sod > 0 THEN {{ DAYNS }} ELSE 0 END AS f0
           FROM win WHERE wg = 'h24' AND bs = 900),
d AS (SELECT dev, s_ns, nbb, f0 + unnest(range((e_ns - f0) // {{ DAYNS }})) * {{ DAYNS }} AS d0 FROM w
      WHERE e_ns >= f0 + {{ DAYNS }}
      UNION ALL SELECT dev, s_ns, nbb, s_ns FROM w WHERE e_ns < f0 + {{ DAYNS }} AND T >= 86400)
SELECT dev, s_ns, nbb, d0, round_even((d0 - s_ns) / 9e11, 0)::INT AS j0,
       ((d0 % {{ DAYNS }}) // 1000000000 // 900) % 96 AS sh
FROM d WHERE round_even((d0 - s_ns) / 9e11, 0) >= 0 AND round_even((d0 - s_ns) / 9e11, 0) + 96 <= nbb
-- @step tdh
SELECT dev, det, d0, hr, CASE WHEN count(H) = 2 THEN sum(H) END AS N, avg(Q) AS O FROM (
  SELECT g.dev, g.det, t.d0, ((g.k - t.j0 + t.sh) % 96) // 2 AS hh, ((g.k - t.j0 + t.sh) % 96) // 4 AS hr,
         CASE WHEN count(g.n) > 0 THEN sum(g.n) END AS H, avg(g.occ) AS Q
  FROM tdays t JOIN gb g ON g.dev = t.dev AND g.k >= t.j0 AND g.k < t.j0 + 96
  JOIN xbase x ON x.dev = g.dev AND x.det = g.det GROUP BY g.dev, g.det, t.d0, hh, hr)
GROUP BY dev, det, d0, hr
-- @step tdf
SELECT a.*, sqrt((a.nn + 0.5) / (b.b4n + 0.5)) AS r, asin(sqrt(a.onm)) AS onv,
       asin(sqrt(least(greatest(b.b4o, 0.0), 1.0))) AS od, log10(1 + a.tot) AS lv,
       CASE WHEN a.tot / 24.0 < 20 THEN 'low' WHEN a.tot / 24.0 < 100 THEN 'medium' ELSE 'high' END AS band
FROM (SELECT dev, det, d0, sum(N) AS tot, avg(N) FILTER (hr >= 1 AND hr < 5) AS nn, sum(N) FILTER (hr >= 1 AND hr < 5)
             AS n_night, avg(least(greatest(O, 0.0), 1.0)) FILTER (hr >= 1 AND hr < 5) AS onm
      FROM tdh GROUP BY dev, det, d0 HAVING count(N) = 24 AND count(O) = 24 AND sum(N) >= 50) a
JOIN (SELECT a.dev, a.det, a.d0, max(s4n) / 4 AS b4n, max(s4o) / 4 AS b4o FROM (
        SELECT a.dev, a.det, a.d0, a.hr, sum(b.N) AS s4n, sum(least(greatest(b.O, 0.0), 1.0)) AS s4o
        FROM tdh a JOIN tdh b ON b.dev = a.dev AND b.det = a.det AND b.d0 = a.d0 AND (b.hr - a.hr + 24) % 24 < 4
        GROUP BY a.dev, a.det, a.d0, a.hr) a GROUP BY a.dev, a.det, a.d0) b USING (dev, det, d0)
{# z of the detector's own type, and the smallest over the plausible functions when the model is unsure #}
-- @step tdz
SELECT f.dev, f.det, f.d0, c.own, ((greatest((f.r - (g.b0 + g.b1 * f.od + g.b2 * f.lv)) / g.sc,
       CASE WHEN g.occ THEN (f.onv - (g.o0 + g.o1 * f.lv)) / g.so END)) - g.med) / g.mad AS z
FROM tdf f JOIN (SELECT dev, det, fa, own FROM lcand) c USING (dev, det) JOIN xbase x USING (dev, det)
JOIN (SELECT f.dev, f.det, f.d0, c.fa, first(g.key ORDER BY o.o) AS key
      FROM tdf f JOIN lcand c USING (dev, det) JOIN xbase x USING (dev, det),
      (SELECT 1 AS o UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4) o, todg g
      WHERE g.key = CASE o.o WHEN 1 THEN c.fa || '|' || x.span || '|' || f.band WHEN 2 THEN c.fa || '|*|' || f.band
                             WHEN 3 THEN c.fa || '|' || x.span || '|*' ELSE c.fa || '|*|*' END
      GROUP BY f.dev, f.det, f.d0, c.fa) k ON k.dev = f.dev AND k.det = f.det AND k.d0 = f.d0 AND k.fa = c.fa
JOIN todg g ON g.key = k.key
-- @step tdd
SELECT f.*, z.z_own AS tod_z0, CASE WHEN z.z_own IS NOT NULL THEN least(z.z_own, z.z_alt) END AS tod_z,
       f.r - m.mate_r AS tod_dph, m.mate_r AS tod_mate_r, l.lim_prof
FROM tdf f JOIN (SELECT dev, det, d0, max(z) FILTER (own) AS z_own, min(z) FILTER (NOT own) AS z_alt FROM tdz
                 GROUP BY dev, det, d0) z USING (dev, det, d0)
JOIN xbase x USING (dev, det) JOIN limits l USING (dev, det)
LEFT JOIN (SELECT a.dev, a.det, a.d0, median(b.r) AS mate_r FROM tdf a JOIN xbase xa USING (dev, det)
           JOIN xbase xb2 ON xb2.dev = a.dev AND xb2.ph = xa.ph AND xb2.det <> a.det
           JOIN tdf b ON b.dev = a.dev AND b.det = xb2.det AND b.d0 = a.d0
           JOIN (SELECT dev, det, d0, max(z) FILTER (own) AS zo FROM tdz GROUP BY dev, det, d0) bz
             ON bz.dev = b.dev AND bz.det = b.det AND bz.d0 = b.d0
           WHERE bz.zo IS NOT NULL AND b.tot >= 200 GROUP BY a.dev, a.det, a.d0) m USING (dev, det, d0)
-- @step tdl
SELECT *, CASE WHEN tod_z IS NULL THEN NULL
               WHEN n_night >= {{ P['min_night']|dbl }} AND coalesce(tod_dph, -1) >= {{ P['mates_delta']|dbl }}
                    AND tod_z >= lim_prof * {{ P['z_x']|dbl }} AND tod_z >= lim_prof THEN 0.8
               WHEN n_night >= {{ P['min_night']|dbl }} AND coalesce(tod_dph, -1) >= {{ P['mates_delta']|dbl }}
                    AND tod_z >= lim_prof * {{ P['z_x']|dbl }} THEN 0.5 ELSE 0.0 END AS lvl
FROM tdd
-- @step tod
SELECT dev, det, first(tod_z ORDER BY coalesce(lvl, -1) DESC, d0) AS tod_z,
       first(tod_dph ORDER BY coalesce(lvl, -1) DESC, d0) AS tod_dph,
       first(n_night ORDER BY coalesce(lvl, -1) DESC, d0) AS tod_n_night,
       first(r ORDER BY coalesce(lvl, -1) DESC, d0) AS tod_r, first(tod_mate_r ORDER BY coalesce(lvl, -1) DESC, d0) AS tod_mate_r,
       first(d0 ORDER BY coalesce(lvl, -1) DESC, d0) AS tod_d0
FROM tdl GROUP BY dev, det
-- @step todnote
SELECT dev, first(d0 ORDER BY n_odd / greatest(n_ok, 1) DESC, d0) AS d0, first(n_odd ORDER BY n_odd / greatest(n_ok, 1)
       DESC, d0) AS n_odd, first(n_ok ORDER BY n_odd / greatest(n_ok, 1) DESC, d0) AS n_ok
FROM (SELECT dev, d0, count(*) FILTER (tod_z >= 3.5) AS n_odd, count(*) AS n_ok FROM tdd GROUP BY dev, d0)
WHERE n_odd >= 5 AND n_odd >= 0.20 * n_ok GROUP BY dev
