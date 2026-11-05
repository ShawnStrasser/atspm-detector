-- ev
SELECT DISTINCT w.dev, epoch_ns(e.Timestamp) / 1e9 AS ts, e.EventId::INT AS ev, e.Parameter::INT AS p
                   FROM events e JOIN pq_wpart w ON e.DeviceId = w.DeviceId WHERE e.Timestamp >= w.lo AND e.Timestamp < w.hi AND e.EventId IN (1, 8, 21, 22, 23, 45, 67, 89, 90, 102, 104, 105, 110, 111);

-- b10
SELECT w.dev, epoch_ns(e.Timestamp) // 10000000000 AS k, min(epoch_ns(e.Timestamp)) AS a, max(epoch_ns(e.Timestamp)) AS z FROM events e JOIN pq_wpart w ON e.DeviceId = w.DeviceId WHERE e.Timestamp >= w.lo AND e.Timestamp < w.hi GROUP BY ALL;

-- sil
SELECT dev, z AS n0, nt AS n1
FROM (SELECT dev, z, lead(a) OVER (PARTITION BY dev ORDER BY a) AS nt
      FROM (SELECT dev, a, z FROM pq_b10))
WHERE nt - z >= 29000000000;

-- logq
SELECT dev, k // 90 AS q, min(a) AS a, max(z) AS z FROM pq_b10 GROUP BY ALL;

-- gap
SELECT dev, (n0 // 60000000000 + 1) * 60.0 AS g0, (n1 // 60000000000) * 60.0 AS g1 FROM pq_sil
WHERE n1 // 60000000000 - n0 // 60000000000 > 2 AND true;

-- hole
SELECT dev, n0 / 1e9 AS g0, n1 / 1e9 AS g1 FROM pq_sil WHERE n1 / 1e9 - n0 / 1e9 >= 30 AND true;

-- press
SELECT x.dev, x.ch, x.t_on, x.t_off, x.t_off - x.t_on AS dur, CASE WHEN x.nev IS NULL THEN l.last - x.t_on END AS open_s,
       floor(x.t_on / 86400)::BIGINT AS day
FROM (SELECT dev, p AS ch, ts AS t_on, ev, lead(ev) OVER w AS nev, CASE WHEN lead(ev) OVER w = 89 THEN lead(ts) OVER w END AS t_off
      FROM pq_ev WHERE ev IN (89, 90) WINDOW w AS (PARTITION BY dev, p ORDER BY ts, ev)) x
JOIN (SELECT dev, max(z) / 1e9 AS last FROM pq_logq GROUP BY dev) l USING (dev) WHERE x.ev = 90
UNION ALL
SELECT f.dev, f.p, w.s, f.ts, f.ts - w.s, NULL, floor(w.s / 86400)::BIGINT
FROM (SELECT dev, p, arg_min(ev, (ts, ev)) AS ev0, min(ts) AS ts FROM pq_ev WHERE ev IN (89, 90) GROUP BY ALL) f JOIN pq_wpart w USING (dev)
WHERE f.ev0 = 89;

-- green
SELECT dev, p AS ph, ts, row_number() OVER (PARTITION BY dev, p ORDER BY ts) AS k FROM pq_ev WHERE ev = 1;

-- yel
SELECT dev, p AS ph, ts FROM pq_ev WHERE ev = 8;

-- walk
WITH x AS (SELECT dev, p, ts, ev, lag(ev) OVER w AS e0, lead(ev) OVER w AS e1, lead(ts) OVER w AS t1,
                  lead(ev, 2) OVER w AS e2, lead(ts, 2) OVER w AS t2
           FROM pq_ev WHERE ev IN (21, 22, 23) WINDOW w AS (PARTITION BY dev, p ORDER BY ts, ev)),
y AS (SELECT * FROM x WHERE ev = 21 OR (ev IN (22, 23) AND coalesce(e0, 23) = 23 AND true))
SELECT y.dev, y.p AS ph, CASE WHEN y.ev = 21 OR g.ts IS NULL OR y.ts - g.ts > 120
                                   OR coalesce(v.ts > g.ts, false) THEN y.ts ELSE g.ts END AS t21,
       CASE WHEN y.ev = 22 THEN y.ts WHEN y.ev = 21 AND y.e1 = 22 THEN y.t1 END AS t22,
       CASE WHEN y.ev = 23 THEN y.ts WHEN y.ev = 22 AND y.e1 = 23 THEN y.t1
            WHEN y.ev = 21 AND y.e1 = 22 AND y.e2 = 23 THEN y.t2 WHEN y.ev = 21 AND y.e1 = 23 THEN y.t1 END AS t23
FROM y ASOF LEFT JOIN pq_green g ON g.dev = y.dev AND g.ph = y.p AND y.ts >= g.ts
ASOF LEFT JOIN pq_yel v ON v.dev = y.dev AND v.ph = y.p AND y.ts - 1 > v.ts;

-- call
SELECT dev, p AS ph, ts FROM pq_ev WHERE ev = 45;

-- ovw
SELECT dev, p AS ov, ts FROM pq_ev WHERE ev = 67;

-- pre
SELECT dev, ts AS ps, least(coalesce(en, ts + 21600), ts + 21600) + 120 AS pe
FROM (SELECT dev, ts, ev, min(CASE WHEN ev IN (104, 110, 111) THEN ts END) OVER f AS en
      FROM pq_ev WHERE ev IN (102, 104, 105, 110, 111)
      WINDOW f AS (PARTITION BY dev, p ORDER BY ts, ev ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING))
WHERE ev IN (102, 105) AND true;

-- prelog
SELECT DISTINCT dev FROM pq_ev WHERE ev IN (102, 104, 105, 110, 111);

-- ovlog
SELECT DISTINCT dev FROM pq_ev WHERE ev = 67;

-- inp0
WITH n AS (SELECT dev, ch, count(*) AS presses,
                  count(*) FILTER (round(dur, 1) BETWEEN 0.6 AND 10) / count(*) AS f06,
                  quantile_cont(round(dur, 1), 0.1) AS q10, quantile_cont(round(dur, 1), 0.5) AS q50
           FROM pq_press GROUP BY ALL),
m AS (SELECT p.dev, p.ch, p.t_on, c.ts FROM pq_press p ASOF JOIN pq_call c ON c.dev = p.dev AND p.t_on <= c.ts
      WHERE c.ts - p.t_on <= 0.25),
k AS (SELECT *, row_number() OVER (PARTITION BY dev, ch ORDER BY k DESC, ph = ch DESC, ph) AS r FROM (
        SELECT m.dev, m.ch, c.ph, count(*) AS k, any_value(t.n) AS n45
        FROM m JOIN pq_call c ON c.dev = m.dev AND c.ts = m.ts    -- every phase called at that instant
        JOIN (SELECT dev, ch, count(*) AS n FROM m GROUP BY ALL) t ON t.dev = m.dev AND t.ch = m.ch GROUP BY ALL)),
a AS (SELECT dev, ch, any_value(n45) AS n45, any_value(ph) FILTER (r = 1) AS ph45,
             any_value(ph) FILTER (r = 2 AND k >= 0.3 * n45) AS ph2 FROM k GROUP BY ALL)
SELECT n.dev, n.ch, n.presses, coalesce(a.n45, 0) >= 0.01 * n.presses AS registers, a.ph45, a.ph2,
       CASE WHEN n.presses < 30 THEN 'unknown' WHEN n.q50 >= 5 THEN 'presence' WHEN n.f06 < 0.005 THEN 'pulse'
            WHEN n.q10 <= 0.2 THEN 'hold' ELSE 'hold_min' END AS type
FROM n LEFT JOIN a USING (dev, ch);

-- fb
WITH pp AS (SELECT p.dev, p.ch, p.t_on, w.ph FROM pq_press p JOIN pq_inp0 i USING (dev, ch)
            JOIN (SELECT DISTINCT dev, ph FROM pq_walk) w USING (dev) WHERE NOT i.registers),
f1 AS (SELECT pp.*, w.t21 AS nxt FROM pp ASOF LEFT JOIN pq_walk w ON w.dev = pp.dev AND w.ph = pp.ph AND pp.t_on <= w.t21),
f2 AS (SELECT f1.*, w.t22 AS pt22 FROM f1 ASOF LEFT JOIN pq_walk w ON w.dev = f1.dev AND w.ph = f1.ph AND f1.t_on >= w.t21),
s AS (SELECT dev, ch, ph, avg(CASE WHEN nxt - t_on <= 180 THEN 1 ELSE 0 END) AS share_walk,
             avg(CASE WHEN t_on < pt22 THEN 1 ELSE 0 END) AS share_inwalk FROM f2 GROUP BY ALL),
ws AS (SELECT dev, ph, sum(t22 - t21) AS wsec FROM pq_walk GROUP BY ALL),
sp AS (SELECT dev, max(t_on) - min(t_on) AS span FROM pq_press GROUP BY dev)
SELECT dev, ch, ph FROM (
  SELECT s.*, row_number() OVER (PARTITION BY s.dev, s.ch ORDER BY s.share_walk >= 0.9 DESC,
         s.share_inwalk / greatest(ws.wsec / nullif(sp.span, 0), 1e-6) ASC NULLS LAST, s.share_walk DESC, s.ph) AS r
  FROM s JOIN ws USING (dev, ph) JOIN sp USING (dev)) WHERE r = 1 AND share_walk >= 0.9;

-- ovl
WITH pp AS (SELECT p.dev, p.ch, p.t_on, o.ov FROM pq_press p JOIN pq_inp0 i USING (dev, ch)
            ANTI JOIN pq_fb f USING (dev, ch) JOIN (SELECT DISTINCT dev, ov FROM pq_ovw) o USING (dev)
            WHERE NOT i.registers),
f AS (SELECT pp.*, o.ts FROM pp ASOF LEFT JOIN pq_ovw o ON o.dev = pp.dev AND o.ov = pp.ov AND pp.t_on <= o.ts)
SELECT dev, ch, arg_max(ov, share * 1000 - ov) AS ov FROM (
  SELECT dev, ch, ov, avg(CASE WHEN ts - t_on <= 180 THEN 1 ELSE 0 END) AS share FROM f GROUP BY ALL)
WHERE share >= 0.8 GROUP BY ALL;

-- inp
SELECT i.dev, i.ch, i.presses, i.registers, i.type, CASE WHEN i.registers THEN i.ph45 ELSE f.ph END AS ph,
       CASE WHEN i.registers THEN i.ph2 END AS ph2, o.ov
FROM pq_inp0 i LEFT JOIN pq_fb f USING (dev, ch) LEFT JOIN pq_ovl o USING (dev, ch);

-- pa
WITH x AS (SELECT p.*, i.ph, i.type FROM pq_press p JOIN pq_inp i USING (dev, ch)),
y AS (SELECT x.*, w.t21, w.t22, w.t23 FROM x ASOF LEFT JOIN pq_walk w ON w.dev = x.dev AND w.ph = x.ph AND x.t_on >= w.t21),
z AS (SELECT y.*, c.ts AS c45, y.t_on - lag(y.t_on) OVER (PARTITION BY y.dev, y.ch ORDER BY y.t_on) AS gap
      FROM y ASOF LEFT JOIN pq_call c ON c.dev = y.dev AND c.ph = y.ph AND y.t_on <= c.ts),
r AS (SELECT z.*, sum(CASE WHEN gap IS NULL OR gap > 3 THEN 1 ELSE 0 END) OVER o AS run3,
             sum(CASE WHEN gap IS NULL OR gap > 120 THEN 1 ELSE 0 END) OVER o AS run120
      FROM z WINDOW o AS (PARTITION BY dev, ch ORDER BY t_on ROWS UNBOUNDED PRECEDING))
SELECT dev, ch, ph, type, t_on, t_off, dur, open_s, day, t23, gap, run120,
       CASE WHEN t21 IS NULL THEN 'nowalk' WHEN t_on < t22 THEN 'walk' WHEN t_on < t23 THEN 'fdw' ELSE 'dw' END AS state,
       coalesce(c45 - t_on <= 0.25, false) AS call45,
       dur >= 5 AND row_number() OVER (PARTITION BY dev, ch, run3 ORDER BY t_on) IN (31, 61) AS logcap
FROM r;

-- joint
WITH q AS (SELECT dev, ch, day, t_on, dur FROM pq_pa WHERE dur >= 60)
SELECT a.dev, a.ch, a.day, b.ch AS other, least(a.dur, b.dur) AS both_s FROM q a JOIN q b
  ON a.dev = b.dev AND a.ch <> b.ch AND abs(a.t_on - b.t_on) <= 5;

-- mirror
WITH a AS (SELECT dev, ch, t_on, t_off FROM pq_pa WHERE dur >= 1),
k AS (SELECT dev, ch, count(*) AS n FROM a GROUP BY ALL HAVING count(*) >= 10),
b AS (SELECT dev, ch AS o, t_on, t_off FROM pq_pa WHERE dur IS NOT NULL),
x AS (SELECT a.*, o.o FROM a JOIN k USING (dev, ch) JOIN (SELECT DISTINCT dev, o FROM b) o ON o.dev = a.dev AND o.o <> a.ch),
p AS (SELECT x.*, b.t_on AS pb, b.t_off AS pe FROM x ASOF LEFT JOIN b ON b.dev = x.dev AND b.o = x.o AND x.t_on >= b.t_on),
n AS (SELECT p.*, b.t_on AS nb, b.t_off AS ne FROM p ASOF LEFT JOIN b ON b.dev = p.dev AND b.o = p.o AND p.t_on <= b.t_on),
h AS (SELECT dev, ch, o, CASE WHEN coalesce(t_on - pb, 1e9) <= coalesce(nb - t_on, 1e9) THEN pb ELSE nb END AS mb,
             CASE WHEN coalesce(t_on - pb, 1e9) <= coalesce(nb - t_on, 1e9) THEN pe ELSE ne END AS me, t_on, t_off FROM n),
s AS (SELECT h.dev, h.ch, h.o, count(*) FILTER (abs(mb - t_on) <= 0.15 AND abs(me - t_off) <= 0.15)
             / any_value(k.n) AS share FROM h JOIN k USING (dev, ch) GROUP BY ALL)
SELECT s.dev, s.ch, s.o, s.share, greatest(s.share, coalesce(t.share, 0)) >= 0.95 AS full_pair
FROM s LEFT JOIN s t ON t.dev = s.dev AND t.ch = s.o AND t.o = s.ch;

-- f_stuck
WITH l AS (SELECT *, coalesce(dur, open_s) AS held FROM pq_pa WHERE NOT logcap AND coalesce(dur, open_s) >= 10),
n AS (SELECT l.dev, l.ch, l.t_on, count(w.t21) AS walks FROM l LEFT JOIN pq_walk w
      ON w.dev = l.dev AND w.ph = l.ph AND w.t21 > l.t_on AND w.t21 < l.t_on + l.held GROUP BY ALL),
-- ONs with a log hole (>= 30 s silent) inside: their length is not known (option log_gaps), but a stretch between holes has no lost event: it is an unbroken ON
h AS (SELECT l.dev, l.ch, l.t_on, l.held, p.g0, p.g1, coalesce(lag(p.g1) OVER (PARTITION BY l.dev, l.ch, l.t_on ORDER BY p.g0), l.t_on) AS prev FROM l JOIN pq_hole p
      ON p.dev = l.dev AND p.g0 < l.t_on + l.held AND p.g1 > l.t_on),
g AS (SELECT dev, ch, t_on, true AS gap, greatest(max(g0 - prev), max(t_on + held) - max(g1)) AS stretch FROM h GROUP BY dev, ch, t_on)
, rb AS (SELECT l.dev, l.ch, l.t_on, l.t_on <= w.s AS before,
              (SELECT count(*) FROM pq_green g WHERE g.dev = l.dev AND g.ph = l.ph AND g.ts >= l.t_on AND g.ts < l.t_on + l.held) AS g_in,
              (SELECT count(*) FROM pq_green g WHERE g.dev = l.dev AND g.ph = l.ph AND g.ts >= l.t_on - 7200 AND g.ts < l.t_on) AS g_pre,
              (SELECT count(*) FROM pq_walk k WHERE k.dev = l.dev AND k.ph = l.ph AND k.t21 >= l.t_on - 7200 AND k.t21 < l.t_on) AS w_pre,
              (SELECT count(*) FROM pq_green g WHERE g.dev = l.dev AND g.ph = l.ph AND g.ts >= l.t_on + l.held AND g.ts < l.t_on + l.held + 7200) AS g_post,
              (SELECT count(*) FROM pq_walk k WHERE k.dev = l.dev AND k.ph = l.ph AND k.t21 >= l.t_on + l.held AND k.t21 < l.t_on + l.held + 7200) AS w_post
       FROM l JOIN pq_wpart w USING (dev) WHERE l.dur IS NULL OR l.t_on <= w.s)
SELECT 'stuck on' AS kind, l.dev, l.ch, l.day, l.t_on AS a, l.t_on + l.held AS z,
       greatest(l.held / CASE WHEN l.type = 'presence' THEN 600 ELSE 300 END,
                CASE WHEN l.type = 'presence' THEN 0 ELSE n.walks / 3 END) AS x,
       n.walks AS n, (l.dur IS NULL)::INT AS other, l.held AS held, coalesce(r.before, false) AS before,
       CASE WHEN l.type = 'presence' THEN 600 ELSE 300 END AS lim
FROM l JOIN n USING (dev, ch, t_on) LEFT JOIN g USING (dev, ch, t_on) LEFT JOIN rb r USING (dev, ch, t_on)
WHERE CASE WHEN r.t_on IS NOT NULL THEN n.walks >= 3 AND n.walks >= 0.8 * r.g_in
                AND NOT (r.g_pre >= 5 AND r.w_pre >= 0.8 * r.g_pre)
                AND NOT (r.g_post >= 5 AND r.w_post >= 0.8 * r.g_post)
           WHEN coalesce(g.gap, false) AND l.type = 'presence' THEN g.stretch >= 600
           WHEN coalesce(g.gap, false) THEN n.walks >= 3
                AND NOT EXISTS (SELECT 1 FROM pq_joint j WHERE j.dev = l.dev AND j.ch = l.ch AND j.day = l.day)
           WHEN l.type = 'presence' THEN l.held >= 600
           ELSE l.held >= 300 OR (n.walks >= 3
                AND NOT EXISTS (SELECT 1 FROM pq_joint j WHERE j.dev = l.dev AND j.ch = l.ch AND j.day = l.day))
                OR (l.held >= 180 AND n.walks >= 2 AND true)
           END;

-- f_hold
WITH l AS (SELECT p.* FROM pq_pa p JOIN pq_wpart w USING (dev)
           WHERE NOT p.logcap AND p.type <> 'presence' AND p.dur >= 150 AND p.dur < 180 AND p.t_on > w.s),
n AS (SELECT l.dev, l.ch, l.t_on, count(w.t21) AS walks FROM l LEFT JOIN pq_walk w
      ON w.dev = l.dev AND w.ph = l.ph AND w.t21 > l.t_on AND w.t21 < l.t_on + l.dur GROUP BY ALL)
SELECT 'long hold' AS kind, l.dev, l.ch, l.day, l.t_on AS a, l.t_on + l.dur AS z, l.dur / 180 AS x, n.walks AS n, 0 AS other,
       l.dur AS held, 150 AS lim
FROM l JOIN n USING (dev, ch, t_on)
WHERE n.walks >= 2 AND true
  AND NOT EXISTS (SELECT 1 FROM pq_hole p WHERE p.dev = l.dev AND p.g0 < l.t_on + l.dur AND p.g1 > l.t_on)
  AND NOT EXISTS (SELECT 1 FROM pq_f_stuck f WHERE f.dev = l.dev AND f.ch = l.ch AND f.a = l.t_on);

-- f_regular
WITH g AS (SELECT dev, ch, day, gap, count(*) OVER (PARTITION BY dev, ch, day ORDER BY gap
                  RANGE BETWEEN 0.15 PRECEDING AND 0.15 FOLLOWING) AS k,
                  count(*) OVER (PARTITION BY dev, ch, day) AS n
           FROM pq_pa WHERE gap >= 5 AND type <> 'presence'),
r AS (SELECT dev, ch, day, max(k) / any_value(n) AS x, arg_max(gap, k) AS every_s, any_value(n) AS n FROM g GROUP BY ALL
      HAVING any_value(n) >= 20 AND max(k) >= 0.5 * any_value(n))
SELECT 'machine-regular presses' AS kind, r.dev, r.ch, r.day, min(p.t_on) AS a, max(p.t_on) AS z, any_value(r.x) AS x,
       any_value(r.n) AS n, any_value(r.every_s) AS every_s
FROM r JOIN pq_pa p USING (dev, ch, day) GROUP BY ALL;

-- f_never
SELECT 'never calls a walk' AS kind, p.dev, p.ch, p.day, min(p.t_on) AS a, max(p.t_on) AS z, count(*) AS x, count(*) AS n
FROM pq_pa p JOIN pq_inp i USING (dev, ch)
WHERE i.ph IS NULL AND i.ov IS NULL AND i.type <> 'presence'
  AND NOT EXISTS (SELECT 1 FROM pq_f_regular r WHERE r.dev = p.dev AND r.ch = p.ch AND r.day = p.day)
GROUP BY ALL HAVING count(*) >= 10;

-- f_drop
WITH c AS (SELECT dev, ph, ts AS tc, lead(ts) OVER (PARTITION BY dev, ph ORDER BY ts) AS tn FROM pq_call
           SEMI JOIN (SELECT DISTINCT dev, ph FROM pq_walk) USING (dev, ph)),
p AS (SELECT c.*, w.t21 AS pt21, w.t22 AS pt22 FROM c ASOF LEFT JOIN pq_walk w ON w.dev = c.dev AND w.ph = c.ph AND c.tc >= w.t21),
n AS (SELECT p.*, w.t21 AS tw FROM p ASOF LEFT JOIN pq_walk w ON w.dev = p.dev AND w.ph = p.ph AND p.tc < w.t21),
e1 AS (SELECT n.dev, n.ph, n.tc, coalesce(least(n.tw, n.tn), s.e) AS t_end FROM n
       JOIN pq_wpart s USING (dev) WHERE NOT coalesce(n.tc <= coalesce(n.pt22, n.pt21 + 7) + 0.15, false)),
-- a wait ends at the first log hole (>= 30 s silent) after the call (option log_gaps: a walk lost in it must not
-- read as a wait)
e AS (SELECT e1.dev, e1.ph, e1.tc, least(e1.t_end, coalesce(p.g0, e1.t_end)) AS t_end FROM e1
      ASOF LEFT JOIN pq_hole p ON p.dev = e1.dev AND e1.tc <= p.g0),
g0 AS (SELECT e.*, g.k AS k0, e.t_end - 10 AS te FROM e ASOF LEFT JOIN pq_green g
       ON g.dev = e.dev AND g.ph = e.ph AND e.tc >= g.ts),
g1 AS (SELECT g0.*, g.k AS k1 FROM g0 ASOF LEFT JOIN pq_green g ON g.dev = g0.dev AND g.ph = g0.ph AND g0.te > g.ts),
-- option late_call: the first green after the call is not owed when the call came after the yellow before it began
-- (the signal's last yellow start <= 15 s before that green)
ga AS (SELECT g1.*, g.ts AS gf FROM g1 ASOF LEFT JOIN pq_green g ON g.dev = g1.dev AND g.ph = g1.ph AND g1.tc < g.ts),
gy AS (SELECT ga.*, coalesce(ga.tc >= y.ts AND ga.gf - y.ts <= 15, false)::INT AS late
       FROM ga ASOF LEFT JOIN pq_yel y ON y.dev = ga.dev AND ga.gf > y.ts),
d0 AS (SELECT dev, ph, tc, t_end, greatest(coalesce(k1, 0) - coalesce(k0, 0) - late, 0) AS greens FROM gy
       WHERE coalesce(k1, 0) - coalesce(k0, 0) - late >= 2),
-- any preemption active at any time during the wait (in progress when the call came, or starting before the wait
-- ended; + recovery) excuses the whole wait (user review 2026-10-29): "omitted during preemption"
pf AS (SELECT DISTINCT d0.dev, d0.ph, d0.tc FROM d0 JOIN pq_pre p
       ON p.dev = d0.dev AND p.ps < d0.t_end AND p.pe >= d0.tc),
d AS (SELECT d0.*, pf.tc IS NOT NULL AS pre FROM d0 LEFT JOIN pf USING (dev, ph, tc)),
b AS (SELECT dev, ph, ch, t_on FROM pq_pa WHERE call45),
w AS (SELECT d.*, b.ch AS by_ch, b.t_on FROM d ASOF LEFT JOIN b ON b.dev = d.dev AND b.ph = d.ph AND d.tc >= b.t_on)
SELECT CASE WHEN w.pre THEN 'omitted during preemption' ELSE 'call dropped' END AS kind, w.dev, i.ch,
       floor(w.tc / 86400)::BIGINT AS day, w.tc AS a, w.t_end AS z, w.greens AS x, w.greens AS n, w.ph AS other,
       w.dev IN (SELECT dev FROM pq_prelog) AS pre_log
FROM w JOIN pq_inp i ON i.dev = w.dev AND i.ph = w.ph
     AND (i.ch = w.by_ch OR NOT coalesce(w.tc - w.t_on <= 0.5, false))
WHERE i.type <> 'presence';

-- stopc
SELECT p.dev, p.ch, make_timestamp((floor(p.t_on / 900) * 900 * 1000000)::BIGINT) AS b, count(*) AS n
FROM pq_press p JOIN pq_wpart w USING (dev) WHERE p.t_on > w.s GROUP BY ALL;

-- stopl
SELECT dev, make_timestamp(q * 900000000) AS b FROM pq_logq;

-- stop
WITH inp AS (SELECT dev, ch, sum(n) AS tot FROM pq_stopc GROUP BY ALL HAVING sum(n) > 0),
grid AS (SELECT l.dev, i.ch, l.b, coalesce(c.n, 0) AS n,
                row_number() OVER (PARTITION BY l.dev, i.ch ORDER BY l.b) AS k
         FROM (SELECT DISTINCT dev, b FROM pq_stopl) l JOIN inp i USING (dev)
         LEFT JOIN pq_stopc c ON c.dev = l.dev AND c.ch = i.ch AND c.b = l.b),
runs AS (SELECT dev, ch, min(b) AS r0, max(b) AS r1, count(*) AS bins
         FROM (SELECT dev, ch, b, k - row_number() OVER (PARTITION BY dev, ch ORDER BY b) AS g FROM grid WHERE n = 0)
         GROUP BY dev, ch, g),
first_ AS (SELECT dev, ch, min(b) AS firstb FROM grid GROUP BY ALL),
best AS (SELECT * FROM runs QUALIFY row_number() OVER (PARTITION BY dev, ch ORDER BY bins DESC, r0 DESC) = 1),
nxt AS (SELECT g.dev, g.ch, min(g.b) AS next_b FROM grid g JOIN best b USING (dev, ch) WHERE g.b > b.r1 AND g.n > 0 GROUP BY ALL),
out0 AS (SELECT g.dev, g.ch, g.b, g.n, hour(g.b) AS hh FROM grid g JOIN best b USING (dev, ch) WHERE g.b < b.r0 OR g.b > b.r1),
hours AS (SELECT dev, ch, date_trunc('hour', b) AS h, hour(b) AS hh, sum(n) AS n, count(*) AS m FROM out0 GROUP BY ALL),
capv AS (SELECT dev, ch, 3.0 * quantile_cont(n, 0.5) FILTER (n > 0) AS cap FROM hours GROUP BY ALL),
hod AS (SELECT h.dev, h.ch, h.hh, sum(h.n) AS s, sum(least(h.n, coalesce(c.cap, h.n))) AS sc, sum(h.m) AS m
        FROM hours h LEFT JOIN capv c USING (dev, ch) GROUP BY ALL),
-- level per bin at each hour of day, smoothed 1-2-1 over the neighbouring hours (circular); only over hours seen
lvl AS (SELECT a.dev, a.ch, x.hh, sum(a.s * (2 - x.d)) / sum(a.m * (2 - x.d)) AS lam, sum(a.sc * (2 - x.d)) / sum(a.m * (2 - x.d)) AS lamc
        FROM hod a, LATERAL (SELECT (a.hh + o + 24) % 24 AS hh, abs(o) AS d FROM (VALUES (-1), (0), (1)) t(o)) x
        WHERE x.hh IN (SELECT hh FROM hod h WHERE h.dev = a.dev AND h.ch = a.ch)
        GROUP BY ALL),
disp AS (SELECT h.dev, h.ch, sum(pow(h.n - h.m * l.lam, 2)) AS ss, sum(h.m * l.lam) AS mu,
                count(*) AS nh, count(DISTINCT h.hh) AS nhh
         FROM hours h JOIN lvl l USING (dev, ch, hh) WHERE l.lam > 0 GROUP BY ALL),
runb AS (SELECT g.dev, g.ch, sum(coalesce(l.lam, 0)) AS expected, sum(coalesce(l.lamc, 0)) AS expected_cap
         FROM grid g JOIN best b USING (dev, ch)
         LEFT JOIN lvl l ON l.dev = g.dev AND l.ch = g.ch AND l.hh = hour(g.b)
         WHERE g.b BETWEEN b.r0 AND b.r1 GROUP BY ALL),
dayl AS (SELECT dev, ch, 4 * sum(lam) AS level FROM lvl GROUP BY ALL),
outn AS (SELECT dev, ch, sum(n) AS n_out, count(*) AS bins_out,
                count(DISTINCT date_trunc('hour', b)) FILTER (n > 0) AS active_h FROM out0 GROUP BY ALL),
s AS (SELECT b.dev, b.ch, coalesce(o.n_out, 0) AS n_out, coalesce(o.bins_out, 0) AS bins_out, coalesce(o.active_h, 0) AS active_h,
             b.bins * 15 / 60.0 AS run_h, b.r0 AS run_start, b.r1 + INTERVAL 15 MINUTE AS run_end, x.next_b,
             b.r0 = z.firstb AS starts_window, coalesce(y.level, 0) AS level,
             coalesce(r.expected, 0) AS expected, coalesce(r.expected_cap, 0) AS expected_cap,
             greatest(1.0, (coalesce(d.ss / nullif(d.mu, 0), 0) * greatest(coalesce(d.nh - d.nhh, 0), 0) + 6.0 * 12) / (greatest(coalesce(d.nh - d.nhh, 0), 0) + 12)) AS clump
      FROM best b JOIN first_ z USING (dev, ch) LEFT JOIN nxt x USING (dev, ch) LEFT JOIN runb r USING (dev, ch)
      LEFT JOIN outn o USING (dev, ch) LEFT JOIN disp d USING (dev, ch) LEFT JOIN dayl y USING (dev, ch))
SELECT *, pow(1 + 0.25 * expected / clump, -1.0 / 0.25) AS chance, pow(1 + 0.25 * expected_cap / clump, -1.0 / 0.25) AS chance_cap,
       exp(-expected) AS chance_poisson, level, coalesce(run_h >= 24.0 AND level >= 30.0, false) AS simple FROM s;

-- f_stop
SELECT 'sudden stop' AS kind, s.dev, s.ch, floor(epoch(s.run_start) / 86400)::BIGINT AS day, epoch(s.run_start) AS a,
       epoch(coalesce(s.next_b, s.run_end)) AS z, s.expected / s.clump AS x, s.run_h AS n, s.chance AS every_s, s.expected AS held,
       s.chance_cap AS chance_cap, s.next_b IS NOT NULL AS ended, s.starts_window AS before,
       (SELECT CASE arg_max(e.ev, e.ts) WHEN 90 THEN 'on' ELSE 'off' END FROM pq_ev e WHERE e.dev = s.dev AND e.p = s.ch AND e.ev IN (89, 90)
        AND e.ts < epoch(coalesce(s.next_b, s.run_end))) AS last_ev,
       (SELECT count(*) FROM pq_walk w JOIN pq_inp i ON i.dev = w.dev AND i.ph = w.ph
        WHERE i.dev = s.dev AND i.ch = s.ch AND w.t21 >= epoch(s.run_start) AND w.t21 < epoch(s.run_end)) AS other
FROM pq_stop s WHERE true AND (s.chance <= 0.001 OR s.simple) AND s.run_h >= 6.0
  AND s.n_out >= 20 AND s.active_h >= 6;

-- uniq
WITH g AS (SELECT p.dev, p.ch, p.t_on, p.t_on - lag(p.t_on) OVER (PARTITION BY p.dev, p.ch ORDER BY p.t_on) AS gap
           FROM pq_press p JOIN pq_wpart w USING (dev) WHERE p.t_on > w.s),
r AS (SELECT *, sum(CASE WHEN gap IS NULL OR gap > 15 THEN 1 ELSE 0 END) OVER (PARTITION BY dev, ch ORDER BY t_on ROWS UNBOUNDED PRECEDING) AS run FROM g),
a AS (SELECT *, t_on - first_value(t_on) OVER (PARTITION BY dev, ch, run ORDER BY t_on) AS into_run FROM r)
SELECT dev, ch, t_on FROM a QUALIFY row_number() OVER (PARTITION BY dev, ch, run, floor(into_run / 15) ORDER BY t_on) = 1;

-- pbin
WITH i AS (SELECT dev, ch, ph, ov FROM pq_inp WHERE ph IS NOT NULL OR ov IS NOT NULL),
u AS (SELECT dev, ch, floor(t_on / 900)::BIGINT AS q, count(*) AS u FROM pq_uniq GROUP BY ALL),
wk AS (SELECT i.dev, i.ch, floor(w.t21 / 900)::BIGINT AS q, count(*) AS w FROM i JOIN pq_walk w ON w.dev = i.dev AND w.ph = i.ph GROUP BY ALL
       UNION ALL
       SELECT i.dev, i.ch, floor(o.ts / 900)::BIGINT, count(*) FROM i JOIN pq_ovw o ON o.dev = i.dev AND o.ov = i.ov WHERE i.ph IS NULL GROUP BY ALL),
cl AS (SELECT i.dev, i.ch, floor(c.ts / 900)::BIGINT AS q, count(*) AS c FROM i JOIN pq_call c ON c.dev = i.dev AND c.ph = i.ph GROUP BY ALL),
gr AS (SELECT i.dev, i.ch, floor(g.ts / 900)::BIGINT AS q, count(*) AS grn FROM i JOIN pq_green g ON g.dev = i.dev AND g.ph = i.ph GROUP BY ALL)
SELECT l.dev, i.ch, l.q::BIGINT AS q, ((l.q % 96) // 4)::INT AS hh, coalesce(u.u, 0) AS u, coalesce(wk.w, 0) AS w, coalesce(cl.c, 0) AS c, coalesce(gr.grn, 0) AS grn,
       row_number() OVER (PARTITION BY l.dev, i.ch ORDER BY l.q) AS k
FROM pq_logq l JOIN i USING (dev) LEFT JOIN u ON u.dev = l.dev AND u.ch = i.ch AND u.q = l.q LEFT JOIN wk ON wk.dev = l.dev AND wk.ch = i.ch AND wk.q = l.q
LEFT JOIN cl ON cl.dev = l.dev AND cl.ch = i.ch AND cl.q = l.q LEFT JOIN gr ON gr.dev = l.dev AND gr.ch = i.ch AND gr.q = l.q;

-- f_nop
WITH z AS (SELECT *, k - row_number() OVER (PARTITION BY dev, ch ORDER BY q) AS g FROM pq_pbin WHERE u = 0),
runs AS (SELECT dev, ch, g, min(q) AS q0, max(q) AS q1, count(*) AS bins FROM z GROUP BY ALL
         HAVING count(*) >= 1.0 * 4 AND sum(w) >= 8.0 * 1.0),
b AS (SELECT r.dev, r.ch, r.q0, p.hh, sum(p.u) AS u, sum(p.w) AS w, count(*) AS n
      FROM runs r JOIN pq_pbin p ON p.dev = r.dev AND p.ch = r.ch AND p.q < r.q0 AND p.q >= r.q0 - 72 * 4 GROUP BY ALL),
b0 AS (SELECT dev, ch, q0, (sum(w) + 0.5) / (sum(u) + 0.5) AS rho0, sum(u) AS u0, sum(w) * 1.0 / sum(n) AS wall FROM b GROUP BY ALL HAVING sum(n) >= 24 * 4),
x AS (SELECT r.dev, r.ch, r.q0, r.q1, r.bins, p.q, s.rho0, s.u0,
             sum(p.w) OVER o AS ww, sum(p.c) OVER o AS cc, sum(p.w) OVER l AS wl, sum(p.grn) OVER l AS gl, sum(coalesce(h.w * 1.0 / h.n, s.wall)) OVER o AS ew, row_number() OVER o AS m
      FROM runs r JOIN b0 s USING (dev, ch, q0) JOIN pq_pbin p ON p.dev = r.dev AND p.ch = r.ch AND p.q BETWEEN r.q0 AND r.q1
      LEFT JOIN b h ON h.dev = r.dev AND h.ch = r.ch AND h.q0 = r.q0 AND h.hh = p.hh
      WINDOW o AS (PARTITION BY r.dev, r.ch, r.q0 ORDER BY p.q ROWS UNBOUNDED PRECEDING), l AS (PARTITION BY r.dev, r.ch, r.q0 ORDER BY p.q ROWS 3 PRECEDING)),
f AS (SELECT dev, ch, q0, q1, any_value(bins) AS bins, any_value(rho0) AS rho0, max(ww) AS walks, min(q) AS q_fire, arg_min(ww, q) AS w_fire, arg_min(ew, q) AS e_fire, arg_min(m, q) AS m_fire
      FROM x WHERE m >= 1.0 * 4 AND ww >= CASE WHEN u0 < 10 THEN 12.0 ELSE 8.0 END * m / 4.0
                   AND ww >= CASE WHEN u0 < 10 THEN 10.0 ELSE 3.0 END * (ew + 0.5) AND ww / rho0 >= 15.0
                   AND cc <= 0.1 * ww AND wl >= 0.8 * gl GROUP BY dev, ch, q0, q1),
n AS (SELECT p.dev, p.ch, f.q0, min(p.q) AS q_next FROM f JOIN pq_pbin p ON p.dev = f.dev AND p.ch = f.ch AND p.q > f.q1 AND p.u > 0 GROUP BY ALL),
h AS (SELECT p.dev, p.ch, f.q0, count(*) / 4.0 AS walk_h FROM f JOIN pq_pbin p ON p.dev = f.dev AND p.ch = f.ch AND p.q BETWEEN f.q0 AND f.q1
      WHERE p.grn > 0 AND p.w >= 8.0 / 4.0 AND p.w >= 0.8 * p.grn GROUP BY ALL)
SELECT 'walks without presses' AS kind, f.dev, f.ch, (f.q0 // 96)::BIGINT AS day, f.q0 * 900.0 AS a, coalesce(n.q_next, f.q1 + 1) * 900.0 AS z, f.w_fire / f.rho0 AS x, f.w_fire AS n,
       f.m_fire / 4.0 AS held, f.e_fire AS every_s, f.walks AS other, n.q_next IS NOT NULL AS ended, coalesce(h.walk_h, 0.0) AS walk_h
FROM f LEFT JOIN n USING (dev, ch, q0) LEFT JOIN h USING (dev, ch, q0)
WHERE NOT EXISTS (SELECT 1 FROM f o WHERE o.dev = f.dev AND o.ch <> f.ch AND abs(o.q_fire - f.q_fire) <= 2)
  AND NOT EXISTS (SELECT 1 FROM pq_f_stuck s WHERE s.dev = f.dev AND s.ch = f.ch AND s.a < (f.q_fire + 1) * 900.0 AND s.z > f.q0 * 900.0);

-- f_press
WITH c AS (SELECT p.dev, p.ch, floor(p.t_on / 900)::BIGINT AS q, count(*) AS n FROM pq_press p JOIN pq_wpart w USING (dev) WHERE p.t_on > w.s GROUP BY ALL),
wk AS (SELECT i.dev, i.ch, floor(w.t21 / 900)::BIGINT AS q, count(*) AS w FROM pq_inp i JOIN pq_walk w ON w.dev = i.dev AND w.ph = i.ph GROUP BY ALL
       UNION ALL
       SELECT i.dev, i.ch, floor(o.ts / 900)::BIGINT, count(*) FROM pq_inp i JOIN pq_ovw o ON o.dev = i.dev AND o.ov = i.ov WHERE i.ph IS NULL GROUP BY ALL),
hot AS (SELECT c.dev, c.ch, c.q // 96 AS day, count(*) AS hot, min(c.q) AS q0, max(c.q) AS q1 FROM c LEFT JOIN wk USING (dev, ch, q)
        WHERE c.n >= 30 AND c.n >= 10 * greatest(coalesce(wk.w, 0), 1) GROUP BY ALL HAVING count(*) >= 5),
d AS (SELECT l.dev, i.ch, l.day, coalesce(sum(c.n), 0) AS n FROM (SELECT DISTINCT dev, q // 96 AS day FROM pq_logq) l JOIN (SELECT DISTINCT dev, ch FROM hot) i USING (dev)
      LEFT JOIN c ON c.dev = l.dev AND c.ch = i.ch AND c.q // 96 = l.day GROUP BY ALL),
u AS (SELECT a.dev, a.ch, a.day, any_value(a.n) AS n, median(b.n) AS usual FROM d a JOIN d b ON b.dev = a.dev AND b.ch = a.ch AND b.day <> a.day GROUP BY ALL)
SELECT 'sustained pressing' AS kind, h.dev, h.ch, h.day, h.q0 * 900.0 AS a, (h.q1 + 1) * 900.0 AS z, u.n / u.usual AS x, u.n AS n, h.hot AS other, u.usual AS every_s
FROM hot h JOIN u USING (dev, ch, day)
WHERE u.n >= 1000 AND u.usual >= 20.0 AND u.n >= 5.0 * u.usual;

-- recall
WITH w AS (SELECT dev, ph, t21 + 1 AS tq, coalesce(lag(t23) OVER (PARTITION BY dev, ph ORDER BY t21), -1e18) AS p23 FROM pq_walk),
x AS (SELECT w.*, c.ts FROM w ASOF LEFT JOIN pq_call c ON c.dev = w.dev AND c.ph = w.ph AND w.tq >= c.ts)
SELECT dev, ph FROM x GROUP BY ALL
HAVING count(*) >= 50 AND avg(CASE WHEN ts > p23 THEN 0 ELSE 1 END) >= 0.9;
