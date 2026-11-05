{# The per-detector measures the checks read (health_duckdb._measures, first part): the scored table, occupancy bins with
   their context, per-ON statistics, too fast, vehicle starts with their colour, phase timelines, too many, rule F. #}
{% import "_lib.sql" as lib with context %}
{% import "adapters.sql" as ad with context %}
{# ---- the scored table: detectors with a predicted function the checks know #}
{% set top %}greatest({% for f in C7 %}c.{{ PROB[f] }}{% if not loop.last %}, {% endif %}{% endfor %}){% endset %}
-- @step xbase mat
SELECT s.*, c.fn, c.ph, c.lanes, w.hours, w.wg, w.bs, CASE WHEN c.lanes >= 2 THEN '2+' ELSE '1' END AS span,
       {{ top }} AS f_top,
       CASE WHEN {{ top }} < 0.7 THEN list_filter([{% for f in C7 %}CASE WHEN c.{{ PROB[f] }} >= 0.15 THEN {{ f|str }} END{% if not loop.last %}, {% endif %}{% endfor %}], lambda x: x IS NOT NULL) ELSE []::VARCHAR[] END AS alts
FROM stage1 s JOIN dcl c USING (dev, det) JOIN win w USING (dev) WHERE c.fn IN {{ FNS|inlist }}
{# every detector with its answers (phase, function, traffic zone, lanes) and pass-1 status #}
-- @step dz mat
SELECT d.dev, d.det, c.ph, c.fn, coalesce(c.fn IN {{ ['Advance', 'Count']|inlist }}, false) AS tr,
       greatest(coalesce(c.lanes, 1.0), 1.0) AS ln1, c.lanes, s.status1
FROM dets d LEFT JOIN dcl c USING (dev, det) LEFT JOIN stage1 s USING (dev, det)
{# ---- occupancy bins of bs seconds (15 min; 5 min under ~1.2 h): count n, fraction ON occ (NULL = not covered) #}
-- @step gb1
SELECT g.dev, g.det, g.k, CASE WHEN c.k IS NOT NULL THEN g.n END AS n,
       CASE WHEN c.k IS NOT NULL THEN g.occs / w.bs END AS occ, c.k IS NOT NULL AS cov
FROM (SELECT g.dev, g.det, g.b // (w.bs // 300) AS k, sum(g.n) AS n, sum(g.occ) AS occs
      FROM g5 g JOIN win w USING (dev) GROUP BY g.dev, g.det, g.b // (w.bs // 300)) g JOIN win w USING (dev)
LEFT JOIN (SELECT DISTINCT c.dev, c.b // (w.bs // 300) AS k FROM cov5 c JOIN win w USING (dev)) c
  ON c.dev = g.dev AND c.k = g.k
{# traffic yardstick of each detector: its phase's Advance / Count zones, else the signal's, else all others #}
-- @step rmm mat
SELECT dev, det, r FROM (
  SELECT a.dev, a.det, b.det AS r, coalesce(a.ph = b.ph, false) AND b.tr AS s1, b.tr AS s2,
         bool_or(coalesce(a.ph = b.ph, false) AND b.tr) OVER (PARTITION BY a.dev, a.det) AS h1,
         bool_or(b.tr) OVER (PARTITION BY a.dev, a.det) AS h2
  FROM dz a JOIN dz b ON a.dev = b.dev AND a.det <> b.det)
WHERE (h1 AND s1) OR (NOT h1 AND h2 AND s2) OR (NOT h1 AND NOT h2)
-- @step gb mat
SELECT g.*, CASE WHEN g.cov THEN r.ref END AS ref, CASE WHEN p.mocc > 0.01 THEN g.occ / p.mocc END AS rel, p.mocc,
       z.ph, z.fn, z.status1, (z.status1 IS DISTINCT FROM 'bad' AND g.occ IS NOT NULL) AS hl
FROM gb1 g JOIN (SELECT dev, det, avg(occ) AS mocc FROM gb1 GROUP BY dev, det) p USING (dev, det)
JOIN dz z USING (dev, det)
LEFT JOIN (SELECT m.dev, m.det, g.k, sum(coalesce(g.n, 0)) AS ref FROM rmm m JOIN gb1 g ON g.dev = m.dev AND g.det = m.r
           GROUP BY m.dev, m.det, g.k) r USING (dev, det, k)
{# congestion: median of the other detectors' ON fraction relative to their own mean (signals with >= 4) #}
-- @step gcong
SELECT dev, det, k, CASE WHEN m > 0 THEN (L[(CASE WHEN lo < p THEN lo ELSE lo + 1 END) + 1]
                                          + L[(CASE WHEN h < p THEN h ELSE h + 1 END) + 1]) / 2.0 END AS cong
FROM (SELECT *, m // 2 AS h, CASE WHEN m % 2 = 1 THEN m // 2 ELSE m // 2 - 1 END AS lo FROM (
  SELECT g.dev, g.det, g.k, l.L, CASE WHEN g.rel IS NULL THEN len(l.L) ELSE len(list_filter(l.L, lambda x: x < g.rel)) END AS p,
         len(l.L) - (g.rel IS NOT NULL)::INT AS m
  FROM gb g JOIN (SELECT dev, k, list_sort(list(rel) FILTER (rel IS NOT NULL)) AS L FROM gb GROUP BY dev, k) l
  USING (dev, k) WHERE g.dev IN (SELECT dev FROM dets GROUP BY dev HAVING count(*) >= 4)))
{# peer: mean ON fraction of the healthy phase mates; same: of the healthy phase mates of the same function #}
-- @step gctx mat
SELECT g.*, c.cong,
  CASE WHEN g.ph IS NOT NULL THEN (a.so - coalesce(CASE WHEN g.hl THEN g.occ END, 0)) / nullif(a.sh - g.hl::INT, 0)
       END AS peer,
  CASE WHEN g.ph IS NOT NULL AND g.fn IS NOT NULL THEN (b.so - coalesce(CASE WHEN g.hl THEN g.occ END, 0))
       / nullif(b.sh - g.hl::INT, 0) END AS same
FROM gb g LEFT JOIN gcong c USING (dev, det, k)
LEFT JOIN (SELECT dev, ph, k, sum(occ) FILTER (hl) AS so, count(*) FILTER (hl) AS sh FROM gb WHERE ph IS NOT NULL
           GROUP BY dev, ph, k) a ON a.dev = g.dev AND a.ph = g.ph AND a.k = g.k
LEFT JOIN (SELECT dev, ph, fn, k, sum(occ) FILTER (hl) AS so, count(*) FILTER (hl) AS sh FROM gb
           WHERE ph IS NOT NULL AND fn IS NOT NULL GROUP BY dev, ph, fn, k) b
  ON b.dev = g.dev AND b.ph = g.ph AND b.fn = g.fn AND b.k = g.k
{# count / time-ON elasticity vs the traffic yardstick on its busier half (queue pattern) #}
-- @step slp
SELECT dev, det, CASE WHEN one >= 6 AND vx > 0.5 THEN bn END AS elhi_cnt,
       CASE WHEN one >= 6 AND vx > 0.5 THEN bo END AS elhi_occ
FROM (SELECT dev, det, count(*) AS one, regr_slope(lnn, lr) AS bn, regr_slope(lo, lr) AS bo,
             var_pop(lr) * count(*) AS vx
      FROM (SELECT g.dev, g.det, ln(g.ref + 1) AS lr, ln(g.n + 1) AS lnn, ln(g.occ + 0.005) AS lo
            FROM gb g JOIN win w USING (dev) JOIN (SELECT dev, det, median(ref) FILTER (n IS NOT NULL AND ref IS NOT NULL)
                                                   AS rmed FROM gb GROUP BY dev, det) m USING (dev, det)
            WHERE w.wg <> 'm30' AND g.n IS NOT NULL AND g.ref IS NOT NULL AND g.ref >= m.rmed)
      GROUP BY dev, det)
-- @step lkc
SELECT dev, det, CASE WHEN c1 IS NOT NULL AND NOT isnan(c1) THEN c1 WHEN NOT isnan(c2) THEN c2 END AS c_occ_like
FROM (SELECT g.dev, g.det,
        CASE WHEN count(*) FILTER (g.occ IS NOT NULL AND g.same IS NOT NULL) >= 6
             THEN corr(g.same, g.occ) FILTER (g.occ IS NOT NULL AND g.same IS NOT NULL) END AS c1,
        CASE WHEN count(*) FILTER (g.occ IS NOT NULL AND g.peer IS NOT NULL) >= 6
             THEN corr(g.peer, g.occ) FILTER (g.occ IS NOT NULL AND g.peer IS NOT NULL) END AS c2
      FROM gctx g JOIN win w USING (dev) WHERE w.wg <> 'm30' GROUP BY g.dev, g.det)
{# ---- per-ON statistics: ONs logged again without an OFF, holds, the median clean ON (Count mode) (adapter) #}
{{ ad.onmed() }}
{{ ad.act() }}
{# ---- too fast: every ON-to-ON gap under 0.8 s / lanes (stop-bar zones) or 0.6 s / lanes, in whole 0.1-s ticks
   (adapter) #}
{{ ad.tf() }}
{# ---- phase timelines (>= 3 greens), valid green + yellow time (adapters) #}
{{ ad.phases() }}
{# too many for the traffic: busiest 5 min in veh/h/lane, starts in green + yellow per green + yellow second (>= 60 s of
   it in the bin) and all starts (in float32, as the research) #}
{{ ad.q5b() }}
-- @step q5gyb mat
SELECT z.dev, z.det, y.b, ((coalesce(q.ngy, 0)::FLOAT / y.sgy::FLOAT) * 3600::FLOAT)::DOUBLE / z.ln1 AS q5_gy
FROM dz z JOIN phs3 p ON p.dev = z.dev AND p.ph = z.ph JOIN sgyb y ON y.dev = z.dev AND y.ph = z.ph
LEFT JOIN q5b q ON q.dev = z.dev AND q.det = z.det AND q.b = y.b WHERE y.sgy >= 60
-- @step q5
SELECT a.dev, a.det, g.q5_gy, a.q5_all FROM (SELECT dev, det, max(((n::FLOAT / 300::FLOAT) * 3600::FLOAT)::DOUBLE / ln1)
                                                AS q5_all FROM q5b GROUP BY dev, det) a
LEFT JOIN (SELECT dev, det, max(q5_gy) AS q5_gy FROM q5gyb GROUP BY dev, det) g USING (dev, det)
{# rule F: are the fast starts the same share of the starts in every 15-min period? #}
{{ ad.fe15() }}
-- @step fe
SELECT dev, det, count(*) AS fe_n, any_value(p) AS fe_p,
       CASE WHEN count(*) >= 2 AND any_value(p) > 0 AND any_value(p) < 1
            THEN sum((x - p * n) ^ 2 / (p * (1 - p) * n)) / greatest(count(*) - 1, 1) END AS fe_phi
FROM (SELECT *, sum(x) OVER (PARTITION BY dev, det) / sum(n) OVER (PARTITION BY dev, det) AS p FROM fe15 WHERE n > 0)
GROUP BY dev, det
