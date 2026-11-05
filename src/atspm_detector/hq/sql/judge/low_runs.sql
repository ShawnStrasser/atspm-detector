{# note 154b (option drop_end, health_duckdb._low_runs): every run of >= 2 low hours of a detector (count <= half of
   what its pre-drop share of the yardstick gives, >= 10 expected; hours with less expected do not break a run) -> lvr:
   the periods of a count drop, so a drop that comes back each day is listed each day. #}
-- @step lvr mat
SELECT * FROM (
WITH h AS (SELECT g.dev, g.det, g.j, g.ch AND CASE WHEN a.re THEN a.s2 ELSE a.s END * g.R >= 10 AS judged,
                  g.ch AND CASE WHEN a.re THEN a.s2 ELSE a.s END * g.R >= 10
                  AND g.M <= 0.5 * CASE WHEN a.re THEN a.s2 ELSE a.s END * g.R AS low
           FROM gh2 g JOIN gh4 a USING (dev, det)),
r AS (SELECT *, sum((judged AND NOT low)::INT) OVER (PARTITION BY dev, det ORDER BY j) AS grp FROM h)
SELECT dev, det, min(j) FILTER (low) AS ja, max(j) FILTER (low) + 1 AS jz FROM r
GROUP BY dev, det, grp HAVING count(*) FILTER (low) >= 2)
