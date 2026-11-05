{# note 154 (option signal_silent, health_duckdb._signal_silent): every detector of the signal silent (no ON, < 1 s ON)
   for >= 1 h of daytime (07-19 h) while the signal keeps serving greens -- all inputs lost at once (detector rack,
   cabinet), which phase-mate yardsticks cannot see since the mates are silent too.  Each detector's expected count: its
   daytime ONs per green outside the run x the greens inside it; where that beats its own silence, the silence becomes
   this run -> out (src + sig_silent).  A stretch with no log at all (bins not covered) says nothing, so it does not end
   a run: the silence on both sides of a logging hole is one run, as long as no detector logged in between. #}
{% import "adapters.sql" as ad with context %}
{% macro signal_silent(src, out) %}
{{ ad.greens5() }}
-- @step sgb mat
SELECT b.dev, b.b, b.cov, b.hr, coalesce(n.n, 0) AS n, coalesce(n.occ, 0) AS occ, coalesce(g.k, 0) AS greens
FROM bins5 b LEFT JOIN (SELECT dev, b, sum(n) AS n, sum(occ) AS occ FROM g5 GROUP BY dev, b) n USING (dev, b)
LEFT JOIN greens5 g USING (dev, b)
-- @step sgr mat
SELECT dev, a, z, nday FROM (
  SELECT dev, min(b) AS a, max(b) + 1 AS z, count(*) FILTER (hr >= 7 AND hr < 19) AS nday,
         row_number() OVER (PARTITION BY dev ORDER BY count(*) FILTER (hr >= 7 AND hr < 19) DESC, min(b)) AS rk
  FROM (SELECT *, n = 0 AND occ < 1 AND greens > 0 AS quiet,
               sum((NOT (n = 0 AND occ < 1 AND greens > 0))::INT) OVER (PARTITION BY dev ORDER BY b) AS grp
        FROM sgb WHERE cov)
  WHERE quiet
  GROUP BY dev, grp)
WHERE rk = 1 AND nday >= 12
-- @step sgl mat
SELECT g.dev, g.det, r.a, r.z,
       sum(g.n) FILTER (NOT (g.b >= r.a AND g.b < r.z)) / greatest(sum(s.greens) FILTER (NOT (g.b >= r.a AND g.b < r.z)), 1)
       * sum(s.greens) FILTER (g.b >= r.a AND g.b < r.z) AS lam_sig
FROM g5 g JOIN sgr r USING (dev) JOIN sgb s ON s.dev = g.dev AND s.b = g.b
WHERE g.cov AND g.hr >= 7 AND g.hr < 19
GROUP BY g.dev, g.det, r.a, r.z
HAVING sum(g.n) FILTER (NOT (g.b >= r.a AND g.b < r.z)) >= 20
-- @step {{ out }} mat
SELECT x.* EXCLUDE (msg) REPLACE (CASE WHEN msg THEN l.lam_sig ELSE x.drop_lam END AS drop_lam,
         CASE WHEN msg THEN l.a ELSE x.drop_b0 END AS drop_b0, CASE WHEN msg THEN l.z ELSE x.drop_b1 END AS drop_b1,
         CASE WHEN msg THEN 0.35 + 0.65 * least((l.lam_sig - 30) / 70, 1) ELSE x.s_dropout END AS s_dropout),
       coalesce(msg, false) AS sig_silent
FROM (SELECT x.*, l.lam_sig >= 30 AND l.lam_sig > coalesce(x.drop_lam, 0) AS msg FROM {{ src }} x LEFT JOIN sgl l USING (dev, det)) x
LEFT JOIN sgl l USING (dev, det)
{% endmacro %}
