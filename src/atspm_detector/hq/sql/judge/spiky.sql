{# Bursts in 10-min counts against its own level of the surrounding 2 h and the phase mates' own rise
   (health_duckdb._spiky: health_counts.spiky) -> spk (per detector), spk_runs (the bursts). #}
{% set P, k, wb = LJ['spiky'], D['spk_k'], D['spk_wb'] %}
{% macro lvl(a, b) %}CASE WHEN len(list_filter(coalesce({{ a }}, []) || coalesce({{ b }}, []), lambda x: x IS NOT NULL)) >= 3 THEN list_median(list_filter(coalesce({{ a }}, []) || coalesce({{ b }}, []), lambda x: x IS NOT NULL)) END{% endmacro %}
-- @step o10 mat
SELECT g.dev, g.det, g.b // {{ k }} AS t, sum(g.n) AS own, bool_and(g.cov) AS c
FROM g5 g JOIN win w USING (dev)
WHERE w.hours >= {{ LJ['min_hours_counts']|dbl }} AND w.nb5 // {{ k }} >= {{ D['spk_2wb'] }} AND g.b < w.nb5 // {{ k }} * {{ k }}
GROUP BY g.dev, g.det, g.b // {{ k }}
-- @step o10l
SELECT dev, det, t, own, c, m,
       list(CASE WHEN c THEN own END) OVER wl AS ol, list(CASE WHEN c THEN own END) OVER wr AS orr,
       list(CASE WHEN c THEN m END) OVER wl AS ml, list(CASE WHEN c THEN m END) OVER wr AS mr
FROM (SELECT o.*, coalesce(m.m, 0.0) AS m FROM o10 o LEFT JOIN (
        SELECT f.dev, f.det, o.t, sum(o.own) AS m FROM p1rf f JOIN o10 o ON o.dev = f.dev AND o.det = f.r
        WHERE f.same GROUP BY f.dev, f.det, o.t) m USING (dev, det, t))
WINDOW wl AS (PARTITION BY dev, det ORDER BY t ROWS BETWEEN {{ wb }} PRECEDING AND 2 PRECEDING),
       wr AS (PARTITION BY dev, det ORDER BY t ROWS BETWEEN 2 FOLLOWING AND {{ wb }} FOLLOWING)
-- @step o10s mat
SELECT *, b IS NOT NULL AND c AND own / ex >= {{ P['ratio']|dbl }} AND g >= {{ P['geh']|dbl }} AND own > ex AS sp FROM (
  SELECT *, sqrt(2 * (own - ex) ^ 2 / greatest(own + ex, 1e-9)) AS g FROM (
    SELECT *, greatest(b, 1.0) * greatest(CASE WHEN m >= {{ P['mates_min']|dbl }} AND bm IS NOT NULL
                                               THEN m / greatest(bm, 1.0) ELSE 1.0 END, 1.0) AS ex FROM (
      SELECT dev, det, t, own, c, m, {{ lvl('ol', 'orr') }} AS b, {{ lvl('ml', 'mr') }} AS bm FROM o10l)))
-- @step spk_runs mat
SELECT dev, det, min(t) AS a, max(t) + 1 AS z FROM (
  SELECT dev, det, t, t - row_number() OVER (PARTITION BY dev, det ORDER BY t) AS grp FROM o10s WHERE sp)
GROUP BY dev, det, grp
-- @step spk
SELECT s.*, CASE WHEN spk_daily < {{ P['daily_none']|dbl }} THEN 0
                 WHEN spk_daily < {{ P['daily_low']|dbl }} AND lv = 2 AND spk_gmax < {{ P['geh_big']|dbl }} THEN 1 ELSE lv END
            AS spk_lvl
FROM (SELECT s.*, CASE WHEN spk_bursts >= {{ P['bad']|dbl }} THEN 2 WHEN spk_bursts >= {{ P['suspect']|dbl }} THEN 1 ELSE 0 END AS lv
      FROM (SELECT o.dev, o.det, coalesce(r.nr, 0) / greatest(w.hours / 24.0, 1.0) AS spk_bursts,
                   coalesce(max(o.g) FILTER (o.sp), 0.0) AS spk_gmax, sum(o.own) * 24.0 / greatest(w.hours, 1e-9) AS spk_daily
            FROM o10s o JOIN win w USING (dev)
            LEFT JOIN (SELECT dev, det, count(*) AS nr FROM spk_runs GROUP BY dev, det) r USING (dev, det)
            GROUP BY o.dev, o.det, r.nr, w.hours) s) s
