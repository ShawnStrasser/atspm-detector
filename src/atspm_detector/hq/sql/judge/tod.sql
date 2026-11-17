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
             AS n_night, avg(least(greatest(O, 0.0), 1.0)) FILTER (hr >= 1 AND hr < 5) AS onm{% if opt['busy_long_zone'] %},
             avg(N) FILTER (hr >= 20) AS ev{% endif %}

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
       f.r - m.mate_r AS tod_dph, m.mate_r AS tod_mate_r, l.lim_prof{% if opt['busy_long_zone'] %},
       {# note 157: a zone ON >= lz_occ of its busiest 4 h has its day counts capped -> excused (lz) unless A) its night
          occupancy share beats the mates' count share by mates_delta (sqrt scale, as r) or B) its night / evening
          (20-24 h) count ratio is >= lz_k x the median phase mate's (0.25 an hour of slack; a median, so a stacked
          twin or another odd mate does not set the yardstick) #}
       sin(f.od) ^ 2 AS lz_occ, sqrt(coalesce(f.onm, 0.0) / nullif(sin(f.od) ^ 2, 0)) - m.mate_r AS lz_dph,
       ((f.nn + 0.25) / (f.ev + 0.25)) / m.m_ne AS lz_k,
       coalesce(sin(f.od) ^ 2 >= {{ P['lz_occ']|dbl }} AND m.m_ne IS NOT NULL AND f.ev IS NOT NULL
                AND sqrt(coalesce(f.onm, 0.0) / (sin(f.od) ^ 2)) - m.mate_r < {{ P['mates_delta']|dbl }}
                AND ((f.nn + 0.25) / (f.ev + 0.25)) / m.m_ne < {{ P['lz_k']|dbl }}, false) AS lz{% endif %}

FROM tdf f JOIN (SELECT dev, det, d0, max(z) FILTER (own) AS z_own, min(z) FILTER (NOT own) AS z_alt FROM tdz
                 GROUP BY dev, det, d0) z USING (dev, det, d0)
JOIN xbase x USING (dev, det) JOIN limits l USING (dev, det)
LEFT JOIN (SELECT a.dev, a.det, a.d0, median(b.r) AS mate_r{% if opt['busy_long_zone'] %} , median((b.nn + 0.25) / (b.ev + 0.25)) AS m_ne{% endif %} FROM tdf a JOIN xbase xa USING (dev, det)
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
{# note 157 (busy_long_zone): a flagged day the long-zone test does not excuse is shown first; tod_lz = every flagged
   day is excused (the verdict then clears busy at night unless the detector is stuck on) #}
{% set o = '(coalesce(lvl, -1) > 0 AND NOT lz) DESC, ' if opt['busy_long_zone'] else '' %}
{% if opt['busy_long_zone'] %}
{# note 157d: night (01-05, covered) greens of the detector's phase per 5-min bin: g_all; those only it asked for
   (ON in that bin or the one before, no phase mate ON then): g_alone; those nobody asked for (phase on recall): g_none.
   A detector that alone brings up >= lz_ph_min and >= lz_ph_share of its phase's night greens, on a phase not on
   recall (g_none <= lz_recall of them), places false calls that cost service: never excused as a long zone #}
-- @step lzg mat
WITH nb AS (SELECT dev, b FROM bins5 WHERE cov AND hr >= 1 AND hr < 5),
gr AS (SELECT p.dev, p.ph, p.b::BIGINT AS b, p.ngreen FROM p5 p JOIN nb n ON n.dev = p.dev AND n.b = p.b WHERE p.ngreen > 0),
dn AS (SELECT g.dev, g.det, g.b::BIGINT AS b, x.ph FROM g5 g JOIN xbase x USING (dev, det) WHERE g.n > 0 AND x.ph IS NOT NULL),
k AS (SELECT x.dev, x.det, gr.ngreen,
             EXISTS (SELECT 1 FROM dn WHERE dn.dev = x.dev AND dn.det = x.det AND dn.b BETWEEN gr.b - 1 AND gr.b) AS own,
             EXISTS (SELECT 1 FROM dn WHERE dn.dev = x.dev AND dn.ph = x.ph AND dn.det <> x.det AND dn.b BETWEEN gr.b - 1 AND gr.b) AS mate
      FROM xbase x JOIN gr ON gr.dev = x.dev AND gr.ph = x.ph)
SELECT dev, det, sum(ngreen) AS g_all, coalesce(sum(ngreen) FILTER (own AND NOT mate), 0) AS g_alone,
       coalesce(sum(ngreen) FILTER (NOT own AND NOT mate), 0) AS g_none
FROM k GROUP BY dev, det
{% endif %}
-- @step tod
SELECT dev, det, first(tod_z ORDER BY {{ o }}coalesce(lvl, -1) DESC, d0) AS tod_z,
       first(tod_dph ORDER BY {{ o }}coalesce(lvl, -1) DESC, d0) AS tod_dph,
       first(n_night ORDER BY {{ o }}coalesce(lvl, -1) DESC, d0) AS tod_n_night,
       first(r ORDER BY {{ o }}coalesce(lvl, -1) DESC, d0) AS tod_r, first(tod_mate_r ORDER BY {{ o }}coalesce(lvl, -1) DESC, d0) AS tod_mate_r,
       first(d0 ORDER BY {{ o }}coalesce(lvl, -1) DESC, d0) AS tod_d0{% if opt['busy_long_zone'] %},
       coalesce(bool_and(lz) FILTER (lvl > 0), false)
         AND NOT coalesce(any_value(g.g_alone) >= {{ P['lz_ph_min']|dbl }}
                          AND any_value(g.g_alone) >= {{ P['lz_ph_share']|dbl }} * any_value(g.g_all)
                          AND any_value(g.g_none) <= {{ P['lz_recall']|dbl }} * any_value(g.g_all), false) AS tod_lz,
       any_value(g.g_all) AS tod_g_all, any_value(g.g_alone) AS tod_g_alone, any_value(g.g_none) AS tod_g_none{% endif %}

FROM tdl{{ ' LEFT JOIN lzg g USING (dev, det)' if opt['busy_long_zone'] else '' }} GROUP BY dev, det
-- @step todnote
SELECT dev, first(d0 ORDER BY n_odd / greatest(n_ok, 1) DESC, d0) AS d0, first(n_odd ORDER BY n_odd / greatest(n_ok, 1)
       DESC, d0) AS n_odd, first(n_ok ORDER BY n_odd / greatest(n_ok, 1) DESC, d0) AS n_ok
FROM (SELECT dev, d0, count(*) FILTER (tod_z >= 3.5) AS n_odd, count(*) AS n_ok FROM tdd GROUP BY dev, d0)
WHERE n_odd >= 5 AND n_odd >= 0.20 * n_ok GROUP BY dev
