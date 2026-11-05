{# 5-min bins (health_duckdb._bins): time ON / unknown / counts per detector and bin, the dense grid with running sums,
   per-detector stats, twins, 15-min counts, quiet runs, level shift, holds. #}
{% import "_lib.sql" as lib with context %}
{% import "adapters.sql" as ad with context %}
{# 5-min counts, time ON and time of unknown state (adapters) #}
{{ ad.bins() }}
-- @step bins5 mat
SELECT b.dev, b.b, c.b IS NOT NULL AS cov, ((b.s_ns + b.b * {{ NS5 }}) // {{ NSH }}) % 24 AS hr
FROM (SELECT dev, s_ns, unnest(range(nb5)) AS b FROM win) b LEFT JOIN cov5 c ON c.dev = b.dev AND c.b = b.b
{# dense detector x 5-min grid: count n, time ON occ (s), covered #}
-- @step g5 mat
SELECT d.dev, d.det, b.b, b.cov, b.hr, coalesce(n.n, 0)::DOUBLE AS n, coalesce(o.occ, 0.0) AS occ, coalesce(u.occ, 0.0) AS unk
FROM dets d JOIN bins5 b USING (dev) LEFT JOIN n5 n ON n.dev = d.dev AND n.det = d.det AND n.b = b.b
LEFT JOIN occ5 o ON o.dev = d.dev AND o.det = d.det AND o.b = b.b LEFT JOIN unk5 u ON u.dev = d.dev AND u.det = d.det AND u.b = b.b

{# running sums at the bin boundaries k = 0 .. nb5 (sum over bins < k): counts cn, quiet bins cq (no ON, < 1 s ON);
   looked up one boundary per CTE (at), so no join order can pair two lookups of the same detector #}
-- @step k5 mat
SELECT dev, det, b + 1 AS k, sum(n) OVER w AS cn, sum((n = 0 AND {{ lib.qocc() }} < 1)::INT) OVER w AS cq
FROM g5 WINDOW w AS (PARTITION BY dev, det ORDER BY b ROWS UNBOUNDED PRECEDING)
UNION ALL SELECT dev, det, 0, 0.0, 0 FROM dets
{# per detector: ONs on covered bins, active, live, day / night ONs #}
-- @step dstat mat
SELECT g.dev, g.det, sum(g.n * g.cov::INT) AS n_on, (sum(g.n) > 0 OR sum(g.occ) > 0) AS active,
       sum(g.n * g.cov::INT) > 0 AS live,
       sum(g.n * (g.cov AND g.hr >= 7 AND g.hr < 19)::INT) AS dy, sum(g.n * (g.cov AND (g.hr >= 21 OR g.hr < 5))::INT) AS nt
FROM g5 g GROUP BY g.dev, g.det
-- @step sig0 mat
SELECT s.dev, s.n_cov, s.last_cov, s.ok_night, a.n_active
FROM (SELECT dev, count(*) FILTER (cov) AS n_cov, coalesce(max(b) FILTER (cov), -1) AS last_cov,
             count(*) FILTER (cov AND (hr >= 21 OR hr < 5)) >= 24 AND count(*) FILTER (cov AND hr >= 7 AND hr < 19) >= 24
             AS ok_night FROM bins5 GROUP BY dev) s
JOIN (SELECT dev, sum(active::INT) AS n_active FROM dstat GROUP BY dev) a USING (dev)
{# twins: two channels carrying (almost) the same 5-min counts = one zone wired to two inputs #}
-- @step twin mat
SELECT a.dev, a.det AS i, b.det AS j
FROM dstat a JOIN dstat b ON a.dev = b.dev AND a.det <> b.det AND a.n_on > 0 AND b.n_on > 0
     AND abs(a.n_on - b.n_on) < 0.1 * greatest(a.n_on + b.n_on, 1)
JOIN g5 x ON x.dev = a.dev AND x.det = a.det JOIN g5 y ON y.dev = a.dev AND y.det = b.det AND y.b = x.b
GROUP BY a.dev, a.det, b.det, a.n_on, b.n_on
HAVING sum(abs(x.n * x.cov::INT - y.n * y.cov::INT)) / greatest(a.n_on + b.n_on, 1) < 0.1
{# 15-min bins of whole 5-min triples: counts, all three covered (partner correlation, level shift) #}
-- @step a15 mat
SELECT g.dev, g.det, g.b // 3 AS k, sum(g.n) AS n15, bool_and(g.cov) AS c15
FROM g5 g JOIN win w USING (dev) WHERE g.b < w.nb5 // 3 * 3 GROUP BY g.dev, g.det, g.b // 3
{# quiet runs (no ON, < 1 s ON, covered) of every detector with ONs: [a, z) in 5-min bins; silent_merge (note 154): a
   quiet run broken by one daytime (05-21 h) 5-min bin of at most 2 actuations (a stray ON inside an outage) is one run,
   each side at least 30 min, so the outage keeps its true start and end #}
-- @step {{ 'runs0' if opt['silent_merge'] else 'runs' }} mat
SELECT dev, det, min(b) AS a, max(b) + 1 AS z FROM (
  SELECT g.dev, g.det, g.b, g.b - row_number() OVER (PARTITION BY g.dev, g.det ORDER BY g.b) AS grp
  FROM g5 g JOIN dstat s USING (dev, det) WHERE s.n_on > 0 AND g.n = 0 AND {{ lib.qocc('g.') }} < 1 AND g.cov)
GROUP BY dev, det, grp
{% if opt['silent_merge'] %}
-- @step runs mat
SELECT dev, det, min(a) AS a, max(z) AS z FROM (
  SELECT *, sum(new::INT) OVER (PARTITION BY dev, det ORDER BY a) AS grp FROM (
    SELECT r.*, NOT coalesce(r.a - lag(r.z) OVER w = 1 AND lag(r.z - r.a) OVER w >= 6 AND r.z - r.a >= 6
                             AND g.n <= 2 AND g.cov AND g.hr >= 5 AND g.hr < 21, false) AS new
    FROM runs0 r LEFT JOIN g5 g ON g.dev = r.dev AND g.det = r.det AND g.b = r.a - 1
    WINDOW w AS (PARTITION BY r.dev, r.det ORDER BY r.a)))
GROUP BY dev, det, grp
{% endif %}
{# level shift of the share of the signal's traffic (15-min, best single change point, Poisson LLR) #}
-- @step lvl0
SELECT dev, det, k, n15, S15, c15,
       sum(n15) OVER w AS cx, sum(S15) OVER w AS cs, row_number() OVER w AS pos,
       count(*) OVER (PARTITION BY dev, det) AS nl, sum(n15) OVER (PARTITION BY dev, det) AS X,
       sum(S15) OVER (PARTITION BY dev, det) AS SS
FROM (SELECT dev, det, k, n15, sum(n15) OVER (PARTITION BY dev, k) - n15 AS S15, c15 FROM a15) WHERE c15
WINDOW w AS (PARTITION BY dev, det ORDER BY k ROWS UNBOUNDED PRECEDING)
-- @step lvlshift
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
            FROM lvl0 WHERE nl >= 8 AND X >= 20 AND pos >= 2 AND pos <= nl - 2))
GROUP BY dev, det
{# continuous ONs held >= 5 min, not over a comms gap (stuck-on candidates, adapter); co = other detectors of the signal
   held ON (>= 5 min) through at least half of it #}
{{ ad.hold() }}
-- @step hold_co mat
SELECT a.hid, count(DISTINCT b.det) FILTER (least(a.f, b.f) - greatest(a.s, b.s) >= 0.5 * a.dur) AS co
FROM hold a LEFT JOIN hold b ON a.dev = b.dev AND a.det <> b.det GROUP BY a.hid
