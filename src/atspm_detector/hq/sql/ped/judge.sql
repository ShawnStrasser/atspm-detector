{# Ped push-button health (ped_health_duckdb._steps): the temp tables, in order.  pq_wpart (dev, DeviceId, s, e, lo, hi)
   holds the signals of one group and their windows; times are epoch seconds (DOUBLE), days / hours count from the
   epoch (local time).  Context: E (the events table, or a store's ped log: same columns), store (empty: E is the raw log; else the
   store's name and a dot), opt (OPT), L (LIM), PRE, NOTE, CODES, PRE_CODES. #}
{% import "ped/_lib.sql" as plib %}
{% set pw = 'FROM ' ~ E ~ ' e JOIN pq_wpart w ON e.DeviceId = w.DeviceId WHERE e.Timestamp >= w.lo AND e.Timestamp < w.hi' %}
{# events, de-duplicated; minutes and hours in which the signal logged anything #}
-- @step ev
SELECT DISTINCT w.dev, epoch_ns(e.Timestamp) / 1e9 AS ts, e.EventId::INT AS ev, e.Parameter::INT AS p
                   {{ pw }} AND e.EventId IN {{ CODES|intlist }}
{% if carry %}UNION SELECT w.dev, epoch_ns(c.t_on) / 1e9, 90, c.Input::INT FROM pq_carry c JOIN pq_wpart w ON w.DeviceId = c.DeviceId WHERE c.t_on < w.lo
UNION SELECT w.dev, epoch_ns(c.t_on) / 1e9 + 0.05, 45, c.Phase::INT FROM pq_carry c JOIN pq_wpart w ON w.DeviceId = c.DeviceId WHERE c.t_on < w.lo AND c.Phase IS NOT NULL
{% endif %}{# what the signal logged, of any code, epoch ns: its silences (n0, n1) and its 15-minute bins q with the first and last event in
   them (a, z); from the raw log, or from what a store kept of it (design 6): the same rows #}
{% if store %}
-- @step sil
SELECT w.dev, epoch_ns(i.StartTime) AS n0, epoch_ns(i.EndTime) AS n1 FROM {{ store }}health_intervals i JOIN pq_wpart w ON i.DeviceId = w.DeviceId
WHERE i.Kind = 'silence' AND i.StartTime >= w.lo AND i.EndTime < w.hi
-- @step logq
SELECT w.dev, epoch_ns(l.TimeStamp) // 900000000000 AS q, min(epoch_ns(l.All_First)) AS a, max(epoch_ns(l.All_Last)) AS z
FROM {{ store }}health_log_15min l JOIN pq_wpart w ON l.DeviceId = w.DeviceId WHERE l.TimeStamp >= w.lo AND l.TimeStamp < w.hi GROUP BY ALL
{% else %}
-- @step b10
SELECT w.dev, epoch_ns(e.Timestamp) // 10000000000 AS k, min(epoch_ns(e.Timestamp)) AS a, max(epoch_ns(e.Timestamp)) AS z {{ pw }} GROUP BY ALL
-- @step sil
{{ plib.silences('pq_b10', 'dev') }}
-- @step logq
SELECT dev, k // 90 AS q, min(a) AS a, max(z) AS z FROM pq_b10 GROUP BY ALL
{% endif %}
{# log gaps (option log_gaps): >= 2 whole clock minutes without any event of the signal, [g0, g1) epoch seconds #}
-- @step gap
SELECT dev, (n0 // 60000000000 + 1) * 60.0 AS g0, (n1 // 60000000000) * 60.0 AS g1 FROM pq_sil
WHERE n1 // 60000000000 - n0 // 60000000000 > {{ L['gap_min'] }} AND {{ opt['log_gaps']|sqlbool }}
{# log holes (option log_gaps): >= 30 s without any event of the signal, [g0, g1) #}
-- @step hole
SELECT dev, n0 / 1e9 AS g0, n1 / 1e9 AS g1 FROM pq_sil WHERE n1 / 1e9 - n0 / 1e9 >= {{ L['hole_s'] }} AND {{ opt['log_gaps']|sqlbool }}
{# ONs: 90 -> the next 89 of the input (none before the next 90: no OFF); the last ON with nothing after it is ON until
   the log ends (open_s) #}
-- @step press
SELECT x.dev, x.ch, x.t_on, x.t_off, x.t_off - x.t_on AS dur, CASE WHEN x.nev IS NULL THEN l.last - x.t_on END AS open_s,
       floor(x.t_on / 86400)::BIGINT AS day
FROM (SELECT dev, p AS ch, ts AS t_on, ev, lead(ev) OVER w AS nev, CASE WHEN lead(ev) OVER w = 89 THEN lead(ts) OVER w END AS t_off
      FROM pq_ev WHERE ev IN (89, 90) WINDOW w AS (PARTITION BY dev, p ORDER BY ts, ev)) x
JOIN (SELECT dev, max(z) / 1e9 AS last FROM pq_logq GROUP BY dev) l USING (dev) WHERE x.ev = 90
UNION ALL
SELECT f.dev, f.p, w.s, f.ts, f.ts - w.s, NULL, floor(w.s / 86400)::BIGINT
FROM (SELECT dev, p, arg_min(ev, (ts, ev)) AS ev0, min(ts) AS ts FROM pq_ev WHERE ev IN (89, 90) GROUP BY ALL) f JOIN pq_wpart w USING (dev)
WHERE f.ev0 = 89
{# green starts, yellow starts, walk cycles (21, then 22, then 23), ped calls, ped overlap walks #}
-- @step green
SELECT dev, p AS ph, ts, row_number() OVER (PARTITION BY dev, p ORDER BY ts) AS k FROM pq_ev WHERE ev = 1
-- @step yel
SELECT dev, p AS ph, ts FROM pq_ev WHERE ev = 8
{# (option orphan_clearance: a clearance 22 or don't walk 23 with no walk 21 since the last 23 is a walk whose 21 (and
   22) was lost from the log; it starts at the phase's green start <= 120 s before it, else at itself; not at a green
   the phase already ended (a yellow of the phase > 1 s before the 22 / 23 came after it): that green's start was lost
   too) #}
-- @step walk
WITH x AS (SELECT dev, p, ts, ev, lag(ev) OVER w AS e0, lead(ev) OVER w AS e1, lead(ts) OVER w AS t1,
                  lead(ev, 2) OVER w AS e2, lead(ts, 2) OVER w AS t2
           FROM pq_ev WHERE ev IN (21, 22, 23) WINDOW w AS (PARTITION BY dev, p ORDER BY ts, ev)),
y AS (SELECT * FROM x WHERE ev = 21 OR (ev IN (22, 23) AND coalesce(e0, 23) = 23 AND {{ opt['orphan_clearance']|sqlbool }}))
SELECT y.dev, y.p AS ph, CASE WHEN y.ev = 21 OR g.ts IS NULL OR y.ts - g.ts > {{ L['orphan_s'] }}
                                   OR coalesce(v.ts > g.ts, false) THEN y.ts ELSE g.ts END AS t21,
       CASE WHEN y.ev = 22 THEN y.ts WHEN y.ev = 21 AND y.e1 = 22 THEN y.t1 END AS t22,
       CASE WHEN y.ev = 23 THEN y.ts WHEN y.ev = 22 AND y.e1 = 23 THEN y.t1
            WHEN y.ev = 21 AND y.e1 = 22 AND y.e2 = 23 THEN y.t2 WHEN y.ev = 21 AND y.e1 = 23 THEN y.t1 END AS t23
FROM y ASOF LEFT JOIN pq_green g ON g.dev = y.dev AND g.ph = y.p AND y.ts >= g.ts
ASOF LEFT JOIN pq_yel v ON v.dev = y.dev AND v.ph = y.p AND y.ts - 1 > v.ts
-- @step call
SELECT dev, p AS ph, ts FROM pq_ev WHERE ev = 45
-- @step ovw
SELECT dev, p AS ov, ts FROM pq_ev WHERE ev = 67
{# preemptions (any preempt number): from the call input on (102) or an entry (105) to the first input off (104), exit
   (111) or max presence (110) of that number after it; none logged: still on, at most 6 h (user review 2026-10-29);
   + 120 s recovery (as the atspm package) #}
-- @step pre
SELECT dev, ts AS ps, least(coalesce(en, ts + {{ PRE['cap_s'] }}), ts + {{ PRE['cap_s'] }}) + {{ PRE['recovery_s'] }} AS pe
FROM (SELECT dev, ts, ev, min(CASE WHEN ev IN (104, 110, 111) THEN ts END) OVER f AS en
      FROM pq_ev WHERE ev IN {{ PRE_CODES|intlist }}
      WINDOW f AS (PARTITION BY dev, p ORDER BY ts, ev ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING))
WHERE ev IN (102, 105) AND {{ opt['preempt_wait']|sqlbool }}
{# signals that logged any preempt event (none in the whole log: the preemption check could not run), any ped overlap
   walk (none in the whole log: an overlap button cannot be told apart) #}
-- @step prelog
SELECT DISTINCT dev FROM pq_ev WHERE ev IN {{ PRE_CODES|intlist }}
-- @step ovlog
SELECT DISTINCT dev FROM pq_ev WHERE ev = 67
{# type and the ped-call phase of each input #}
-- @step inp0
WITH n AS (SELECT dev, ch, count(*) AS presses,
                  count(*) FILTER (round(dur, 1) BETWEEN 0.6 AND 10) / count(*) AS f06,
                  quantile_cont(round(dur, 1), 0.1) AS q10, quantile_cont(round(dur, 1), 0.5) AS q50
           FROM pq_press GROUP BY ALL),
m AS (SELECT p.dev, p.ch, p.t_on, c.ts FROM pq_press p ASOF JOIN pq_call c ON c.dev = p.dev AND p.t_on <= c.ts
      WHERE c.ts - p.t_on <= {{ L['call_s'] }}),
k AS (SELECT *, row_number() OVER (PARTITION BY dev, ch ORDER BY k DESC, ph = ch DESC, ph) AS r FROM (
        SELECT m.dev, m.ch, c.ph, count(*) AS k, any_value(t.n) AS n45
        FROM m JOIN pq_call c ON c.dev = m.dev AND c.ts = m.ts    -- every phase called at that instant
        JOIN (SELECT dev, ch, count(*) AS n FROM m GROUP BY ALL) t ON t.dev = m.dev AND t.ch = m.ch GROUP BY ALL)),
a AS (SELECT dev, ch, any_value(n45) AS n45, any_value(ph) FILTER (r = 1) AS ph45,
             any_value(ph) FILTER (r = 2 AND k >= {{ L['second_phase'] }} * n45) AS ph2 FROM k GROUP BY ALL)
SELECT n.dev, n.ch, n.presses, coalesce(a.n45, 0) >= {{ L['registers'] }} * n.presses AS registers, a.ph45, a.ph2,
       CASE WHEN n.presses < {{ L['typed'] }} THEN 'unknown' WHEN n.q50 >= 5 THEN 'presence' WHEN n.f06 < 0.005 THEN 'pulse'
            WHEN n.q10 <= 0.2 THEN 'hold' ELSE 'hold_min' END AS type
FROM n LEFT JOIN a USING (dev, ch)
{# no ped call: the phase whose walk follows >= 90 % of its presses within 180 s (fewest presses in that walk for its
   share of time first) #}
-- @step fb
WITH pp AS (SELECT p.dev, p.ch, p.t_on, w.ph FROM pq_press p JOIN pq_inp0 i USING (dev, ch)
            JOIN (SELECT DISTINCT dev, ph FROM pq_walk) w USING (dev) WHERE NOT i.registers),
f1 AS (SELECT pp.*, w.t21 AS nxt FROM pp ASOF LEFT JOIN pq_walk w ON w.dev = pp.dev AND w.ph = pp.ph AND pp.t_on <= w.t21),
f2 AS (SELECT f1.*, w.t22 AS pt22 FROM f1 ASOF LEFT JOIN pq_walk w ON w.dev = f1.dev AND w.ph = f1.ph AND f1.t_on >= w.t21),
s AS (SELECT dev, ch, ph, avg(CASE WHEN nxt - t_on <= {{ L['follow_s'] }} THEN 1 ELSE 0 END) AS share_walk,
             avg(CASE WHEN t_on < pt22 THEN 1 ELSE 0 END) AS share_inwalk FROM f2 GROUP BY ALL),
ws AS (SELECT dev, ph, sum(t22 - t21) AS wsec FROM pq_walk GROUP BY ALL),
sp AS (SELECT dev, max(t_on) - min(t_on) AS span FROM pq_press GROUP BY dev)
SELECT dev, ch, ph FROM (
  SELECT s.*, row_number() OVER (PARTITION BY s.dev, s.ch ORDER BY s.share_walk >= {{ L['walk_share'] }} DESC,
         s.share_inwalk / greatest(ws.wsec / nullif(sp.span, 0), 1e-6) ASC NULLS LAST, s.share_walk DESC, s.ph) AS r
  FROM s JOIN ws USING (dev, ph) JOIN sp USING (dev)) WHERE r = 1 AND share_walk >= {{ L['walk_share'] }}
{# still none: the ped overlap whose walk (67) follows >= 80 % of its presses within 180 s #}
-- @step ovl
WITH pp AS (SELECT p.dev, p.ch, p.t_on, o.ov FROM pq_press p JOIN pq_inp0 i USING (dev, ch)
            ANTI JOIN pq_fb f USING (dev, ch) JOIN (SELECT DISTINCT dev, ov FROM pq_ovw) o USING (dev)
            WHERE NOT i.registers),
f AS (SELECT pp.*, o.ts FROM pp ASOF LEFT JOIN pq_ovw o ON o.dev = pp.dev AND o.ov = pp.ov AND pp.t_on <= o.ts)
SELECT dev, ch, arg_max(ov, share * 1000 - ov) AS ov FROM (
  SELECT dev, ch, ov, avg(CASE WHEN ts - t_on <= {{ L['follow_s'] }} THEN 1 ELSE 0 END) AS share FROM f GROUP BY ALL)
WHERE share >= {{ L['overlap_share'] }} GROUP BY ALL
-- @step inp
SELECT i.dev, i.ch, i.presses, i.registers, i.type, CASE WHEN i.registers THEN i.ph45 ELSE f.ph END AS ph,
       CASE WHEN i.registers THEN i.ph2 END AS ph2, o.ov
FROM pq_inp0 i LEFT JOIN pq_fb f USING (dev, ch) LEFT JOIN pq_ovl o USING (dev, ch)
{# every ON with its phase's walk state, the 45 it placed, the gap since the input's previous ON, the log-limit mark and
   the waiting-group key (don't walk: its cycle; before any walk: a new group after a 120-s gap) #}
-- @step pa
WITH x AS (SELECT p.*, i.ph, i.type FROM pq_press p JOIN pq_inp i USING (dev, ch)),
y AS (SELECT x.*, w.t21, w.t22, w.t23 FROM x ASOF LEFT JOIN pq_walk w ON w.dev = x.dev AND w.ph = x.ph AND x.t_on >= w.t21),
z AS (SELECT y.*, c.ts AS c45, y.t_on - lag(y.t_on) OVER (PARTITION BY y.dev, y.ch ORDER BY y.t_on) AS gap
      FROM y ASOF LEFT JOIN pq_call c ON c.dev = y.dev AND c.ph = y.ph AND y.t_on <= c.ts),
r AS (SELECT z.*, sum(CASE WHEN gap IS NULL OR gap > {{ L['logcap_gap_s'] }} THEN 1 ELSE 0 END) OVER o AS run3,
             sum(CASE WHEN gap IS NULL OR gap > {{ L['wait_gap_s'] }} THEN 1 ELSE 0 END) OVER o AS run120
      FROM z WINDOW o AS (PARTITION BY dev, ch ORDER BY t_on ROWS UNBOUNDED PRECEDING))
SELECT dev, ch, ph, type, t_on, t_off, dur, open_s, day, t23, gap, run120,
       CASE WHEN t21 IS NULL THEN 'nowalk' WHEN t_on < t22 THEN 'walk' WHEN t_on < t23 THEN 'fdw' ELSE 'dw' END AS state,
       coalesce(c45 - t_on <= {{ L['call_s'] }}, false) AS call45,
       dur >= {{ L['logcap_s'] }} AND row_number() OVER (PARTITION BY dev, ch, run3 ORDER BY t_on) IN (31, 61) AS logcap
FROM r
{# ON >= 60 s together with another input (starts within 5 s) #}
-- @step joint
WITH q AS (SELECT dev, ch, day, t_on, dur FROM pq_pa WHERE dur >= {{ L['joint_s'] }})
SELECT a.dev, a.ch, a.day, b.ch AS other, least(a.dur, b.dur) AS both_s FROM q a JOIN q b
  ON a.dev = b.dev AND a.ch <> b.ch AND abs(a.t_on - b.t_on) <= {{ L['joint_start_s'] }}
{# identical ONs: share of an input's ONs >= 1 s (>= 10 of them) whose nearest ON of another input starts and ends
   within 0.15 s #}
-- @step mirror
WITH a AS (SELECT dev, ch, t_on, t_off FROM pq_pa WHERE dur >= {{ L['mirror_on_s'] }}),
k AS (SELECT dev, ch, count(*) AS n FROM a GROUP BY ALL HAVING count(*) >= {{ L['mirror_ons'] }}),
b AS (SELECT dev, ch AS o, t_on, t_off FROM pq_pa WHERE dur IS NOT NULL),
x AS (SELECT a.*, o.o FROM a JOIN k USING (dev, ch) JOIN (SELECT DISTINCT dev, o FROM b) o ON o.dev = a.dev AND o.o <> a.ch),
p AS (SELECT x.*, b.t_on AS pb, b.t_off AS pe FROM x ASOF LEFT JOIN b ON b.dev = x.dev AND b.o = x.o AND x.t_on >= b.t_on),
n AS (SELECT p.*, b.t_on AS nb, b.t_off AS ne FROM p ASOF LEFT JOIN b ON b.dev = p.dev AND b.o = p.o AND p.t_on <= b.t_on),
h AS (SELECT dev, ch, o, CASE WHEN coalesce(t_on - pb, 1e9) <= coalesce(nb - t_on, 1e9) THEN pb ELSE nb END AS mb,
             CASE WHEN coalesce(t_on - pb, 1e9) <= coalesce(nb - t_on, 1e9) THEN pe ELSE ne END AS me, t_on, t_off FROM n),
s AS (SELECT h.dev, h.ch, h.o, count(*) FILTER (abs(mb - t_on) <= {{ L['mirror_s'] }} AND abs(me - t_off) <= {{ L['mirror_s'] }})
             / any_value(k.n) AS share FROM h JOIN k USING (dev, ch) GROUP BY ALL)
SELECT s.dev, s.ch, s.o, s.share, greatest(s.share, coalesce(t.share, 0)) >= {{ L['mirror_full'] }} AS full_pair
FROM s LEFT JOIN s t ON t.dev = s.dev AND t.ch = s.o AND t.o = s.ch
{# ---- findings: kind, dev, ch, day, a / z (period), x (strength), n (count), other (number for the text) #}
-- @step f_stuck
WITH l AS (SELECT *, coalesce(dur, open_s) AS held FROM pq_pa WHERE NOT logcap AND coalesce(dur, open_s) >= 10),
n AS (SELECT l.dev, l.ch, l.t_on, count(w.t21) AS walks FROM l LEFT JOIN pq_walk w
      ON w.dev = l.dev AND w.ph = l.ph AND w.t21 > l.t_on AND w.t21 < l.t_on + l.held GROUP BY ALL),
-- ONs with a log hole (>= 30 s silent) inside: their length is not known (option log_gaps), but a stretch between holes has no lost event: it is an unbroken ON
h AS (SELECT l.dev, l.ch, l.t_on, l.held, p.g0, p.g1, coalesce(lag(p.g1) OVER (PARTITION BY l.dev, l.ch, l.t_on ORDER BY p.g0), l.t_on) AS prev FROM l JOIN pq_hole p
      ON p.dev = l.dev AND p.g0 < l.t_on + l.held AND p.g1 > l.t_on),
g AS (SELECT dev, ch, t_on, true AS gap, greatest(max(g0 - prev), max(t_on + held) - max(g1)) AS stretch FROM h GROUP BY dev, ch, t_on)
, rb AS (SELECT l.dev, l.ch, l.t_on, l.t_on <= w.s AS before,
              (SELECT count(*) FROM pq_green g WHERE g.dev = l.dev AND g.ph = l.ph AND g.ts >= l.t_on AND g.ts < l.t_on + l.held) AS g_in,
              (SELECT count(*) FROM pq_green g WHERE g.dev = l.dev AND g.ph = l.ph AND g.ts >= l.t_on - {{ L['recall_look_s'] }} AND g.ts < l.t_on) AS g_pre,
              (SELECT count(*) FROM pq_walk k WHERE k.dev = l.dev AND k.ph = l.ph AND k.t21 >= l.t_on - {{ L['recall_look_s'] }} AND k.t21 < l.t_on) AS w_pre,
              (SELECT count(*) FROM pq_green g WHERE g.dev = l.dev AND g.ph = l.ph AND g.ts >= l.t_on + l.held AND g.ts < l.t_on + l.held + {{ L['recall_look_s'] }}) AS g_post,
              (SELECT count(*) FROM pq_walk k WHERE k.dev = l.dev AND k.ph = l.ph AND k.t21 >= l.t_on + l.held AND k.t21 < l.t_on + l.held + {{ L['recall_look_s'] }}) AS w_post
       FROM l JOIN pq_wpart w USING (dev) WHERE l.dur IS NULL OR l.t_on <= w.s)
SELECT 'stuck on' AS kind, l.dev, l.ch, l.day, l.t_on AS a, l.t_on + l.held AS z,
       greatest(l.held / CASE WHEN l.type = 'presence' THEN {{ L['stuck_presence_s'] }} ELSE {{ L['stuck_s'] }} END,
                CASE WHEN l.type = 'presence' THEN 0 ELSE n.walks / {{ L['stuck_walks'] }} END) AS x,
       n.walks AS n, (l.dur IS NULL)::INT AS other, l.held AS held, coalesce(r.before, false) AS before,
       CASE WHEN l.type = 'presence' THEN {{ L['stuck_presence_s'] }} ELSE {{ L['stuck_s'] }} END AS lim
FROM l JOIN n USING (dev, ch, t_on) LEFT JOIN g USING (dev, ch, t_on) LEFT JOIN rb r USING (dev, ch, t_on)
WHERE CASE WHEN r.t_on IS NOT NULL THEN n.walks >= {{ L['stuck_walks'] }} AND n.walks >= {{ L['recall_share'] }} * r.g_in
                AND NOT (r.g_pre >= {{ L['recall_min_greens'] }} AND r.w_pre >= {{ L['recall_share'] }} * r.g_pre)
                AND NOT (r.g_post >= {{ L['recall_min_greens'] }} AND r.w_post >= {{ L['recall_share'] }} * r.g_post)
           WHEN coalesce(g.gap, false) AND l.type = 'presence' THEN g.stretch >= {{ L['stuck_presence_s'] }}
           WHEN coalesce(g.gap, false) THEN n.walks >= {{ L['stuck_walks'] }}
                AND NOT EXISTS (SELECT 1 FROM pq_joint j WHERE j.dev = l.dev AND j.ch = l.ch AND j.day = l.day)
           WHEN l.type = 'presence' THEN l.held >= {{ L['stuck_presence_s'] }}
           ELSE l.held >= {{ L['stuck_s'] }} OR (n.walks >= {{ L['stuck_walks'] }}
                AND NOT EXISTS (SELECT 1 FROM pq_joint j WHERE j.dev = l.dev AND j.ch = l.ch AND j.day = l.day))
                OR (l.held >= {{ L['stuck2_s'] }} AND n.walks >= {{ L['stuck2_walks'] }} AND {{ opt['stuck_2walks']|sqlbool }})
           END
{# a long hold (user review 2026-11-02): push input ON 150-179 s through >= 2 walks, no log hole inside: suspect #}
-- @step f_hold
WITH l AS (SELECT p.* FROM pq_pa p JOIN pq_wpart w USING (dev)
           WHERE NOT p.logcap AND p.type <> 'presence' AND p.dur >= {{ L['stuck3_s'] }} AND p.dur < {{ L['stuck2_s'] }} AND p.t_on > w.s),
n AS (SELECT l.dev, l.ch, l.t_on, count(w.t21) AS walks FROM l LEFT JOIN pq_walk w
      ON w.dev = l.dev AND w.ph = l.ph AND w.t21 > l.t_on AND w.t21 < l.t_on + l.dur GROUP BY ALL)
SELECT 'long hold' AS kind, l.dev, l.ch, l.day, l.t_on AS a, l.t_on + l.dur AS z, l.dur / {{ L['stuck2_s'] }} AS x, n.walks AS n, 0 AS other,
       l.dur AS held, {{ L['stuck3_s'] }} AS lim
FROM l JOIN n USING (dev, ch, t_on)
WHERE n.walks >= {{ L['stuck2_walks'] }} AND {{ opt['stuck_2walks']|sqlbool }}
  AND NOT EXISTS (SELECT 1 FROM pq_hole p WHERE p.dev = l.dev AND p.g0 < l.t_on + l.dur AND p.g1 > l.t_on)
  AND NOT EXISTS (SELECT 1 FROM pq_f_stuck f WHERE f.dev = l.dev AND f.ch = l.ch AND f.a = l.t_on)
-- @step f_regular
WITH g AS (SELECT dev, ch, day, gap, count(*) OVER (PARTITION BY dev, ch, day ORDER BY gap
                  RANGE BETWEEN {{ L['regular_tol_s'] }} PRECEDING AND {{ L['regular_tol_s'] }} FOLLOWING) AS k,
                  count(*) OVER (PARTITION BY dev, ch, day) AS n
           FROM pq_pa WHERE gap >= {{ L['regular_gap_s'] }} AND type <> 'presence'),
r AS (SELECT dev, ch, day, max(k) / any_value(n) AS x, arg_max(gap, k) AS every_s, any_value(n) AS n FROM g GROUP BY ALL
      HAVING any_value(n) >= {{ L['regular_gaps'] }} AND max(k) >= {{ L['regular_share'] }} * any_value(n))
SELECT 'machine-regular presses' AS kind, r.dev, r.ch, r.day, min(p.t_on) AS a, max(p.t_on) AS z, any_value(r.x) AS x,
       any_value(r.n) AS n, any_value(r.every_s) AS every_s
FROM r JOIN pq_pa p USING (dev, ch, day) GROUP BY ALL
-- @step f_never
SELECT 'never calls a walk' AS kind, p.dev, p.ch, p.day, min(p.t_on) AS a, max(p.t_on) AS z, count(*) AS x, count(*) AS n
FROM pq_pa p JOIN pq_inp i USING (dev, ch)
WHERE i.ph IS NULL AND i.ov IS NULL AND i.type <> 'presence'
  AND NOT EXISTS (SELECT 1 FROM pq_f_regular r WHERE r.dev = p.dev AND r.ch = p.ch AND r.day = p.day)
GROUP BY ALL HAVING count(*) >= {{ L['no_call'] }}
{# standing calls (45 outside the walk) of phases that walk: ended by the next walk or the next 45 (dropped) #}
-- @step f_drop
WITH c AS (SELECT dev, ph, ts AS tc, lead(ts) OVER (PARTITION BY dev, ph ORDER BY ts) AS tn FROM pq_call
           SEMI JOIN (SELECT DISTINCT dev, ph FROM pq_walk) USING (dev, ph)),
p AS (SELECT c.*, w.t21 AS pt21, w.t22 AS pt22 FROM c ASOF LEFT JOIN pq_walk w ON w.dev = c.dev AND w.ph = c.ph AND c.tc >= w.t21),
n AS (SELECT p.*, w.t21 AS tw FROM p ASOF LEFT JOIN pq_walk w ON w.dev = p.dev AND w.ph = p.ph AND p.tc < w.t21),
e1 AS (SELECT n.dev, n.ph, n.tc, coalesce(least(n.tw, n.tn), s.e) AS t_end FROM n
       JOIN pq_wpart s USING (dev) WHERE NOT coalesce(n.tc <= coalesce(n.pt22, n.pt21 + 7) + {{ L['walk_tick_s'] }}, false)),
-- a wait ends at the first log hole (>= 30 s silent) after the call (option log_gaps: a walk lost in it must not
-- read as a wait)
e AS (SELECT e1.dev, e1.ph, e1.tc, least(e1.t_end, coalesce(p.g0, e1.t_end)) AS t_end FROM e1
      ASOF LEFT JOIN pq_hole p ON p.dev = e1.dev AND e1.tc <= p.g0),
g0 AS (SELECT e.*, g.k AS k0, e.t_end - {{ L['serve_s'] }} AS te FROM e ASOF LEFT JOIN pq_green g
       ON g.dev = e.dev AND g.ph = e.ph AND e.tc >= g.ts),
g1 AS (SELECT g0.*, g.k AS k1 FROM g0 ASOF LEFT JOIN pq_green g ON g.dev = g0.dev AND g.ph = g0.ph AND g0.te > g.ts),
-- option late_call: the first green after the call is not owed when the call came after the yellow before it began
-- (the signal's last yellow start <= 15 s before that green)
{% if opt['late_call'] %}
ga AS (SELECT g1.*, g.ts AS gf FROM g1 ASOF LEFT JOIN pq_green g ON g.dev = g1.dev AND g.ph = g1.ph AND g1.tc < g.ts),
gy AS (SELECT ga.*, coalesce(ga.tc >= y.ts AND ga.gf - y.ts <= {{ L['late_s'] }}, false)::INT AS late
       FROM ga ASOF LEFT JOIN pq_yel y ON y.dev = ga.dev AND ga.gf > y.ts),
{% else %}
gy AS (SELECT g1.*, 0 AS late FROM g1),
{% endif %}
d0 AS (SELECT dev, ph, tc, t_end, greatest(coalesce(k1, 0) - coalesce(k0, 0) - late, 0) AS greens FROM gy
       WHERE coalesce(k1, 0) - coalesce(k0, 0) - late >= {{ L['drop_greens'] }}),
-- any preemption active at any time during the wait (in progress when the call came, or starting before the wait
-- ended; + recovery) excuses the whole wait (user review 2026-10-29): "omitted during preemption"
pf AS (SELECT DISTINCT d0.dev, d0.ph, d0.tc FROM d0 JOIN pq_pre p
       ON p.dev = d0.dev AND p.ps < d0.t_end AND p.pe >= d0.tc),
d AS (SELECT d0.*, pf.tc IS NOT NULL AS pre FROM d0 LEFT JOIN pf USING (dev, ph, tc)),
b AS (SELECT dev, ph, ch, t_on FROM pq_pa WHERE call45),
w AS (SELECT d.*, b.ch AS by_ch, b.t_on FROM d ASOF LEFT JOIN b ON b.dev = d.dev AND b.ph = d.ph AND d.tc >= b.t_on)
SELECT CASE WHEN w.pre THEN {{ NOTE|str }} ELSE 'call dropped' END AS kind, w.dev, i.ch,
       floor(w.tc / 86400)::BIGINT AS day, w.tc AS a, w.t_end AS z, w.greens AS x, w.greens AS n, w.ph AS other,
       w.dev IN (SELECT dev FROM pq_prelog) AS pre_log
FROM w JOIN pq_inp i ON i.dev = w.dev AND i.ph = w.ph
     AND (i.ch = w.by_ch OR NOT coalesce(w.tc - w.t_on <= 0.5, false))
WHERE i.type <> 'presence'
{# sudden stop (ped_stop.py): 15-min press counts of the window (an ON from before the window not counted) and the 15-min bins
   the signal logged, the longest zero run of each input against its own level; only runs at or past a limit become findings #}
-- @step stopc
SELECT p.dev, p.ch, make_timestamp((floor(p.t_on / 900) * 900 * 1000000)::BIGINT) AS b, count(*) AS n
FROM pq_press p JOIN pq_wpart w USING (dev) WHERE p.t_on > w.s GROUP BY ALL
-- @step stopl
SELECT dev, make_timestamp(q * 900000000) AS b FROM pq_logq
-- @step stop
{{ STOP }}
-- @step f_stop
SELECT 'sudden stop' AS kind, s.dev, s.ch, floor(epoch(s.run_start) / 86400)::BIGINT AS day, epoch(s.run_start) AS a,
       epoch(coalesce(s.next_b, s.run_end)) AS z, s.expected / s.clump AS x, s.run_h AS n, s.chance AS every_s, s.expected AS held,
       s.chance_cap AS chance_cap, s.next_b IS NOT NULL AS ended, s.starts_window AS before,
       (SELECT CASE arg_max(e.ev, e.ts) WHEN 90 THEN 'on' ELSE 'off' END FROM pq_ev e WHERE e.dev = s.dev AND e.p = s.ch AND e.ev IN (89, 90)
        AND e.ts < epoch(coalesce(s.next_b, s.run_end))) AS last_ev,
       (SELECT count(*) FROM pq_walk w JOIN pq_inp i ON i.dev = w.dev AND i.ph = w.ph
        WHERE i.dev = s.dev AND i.ch = s.ch AND w.t21 >= epoch(s.run_start) AND w.t21 < epoch(s.run_end)) AS other
FROM pq_stop s WHERE {{ opt['stop_test']|sqlbool }} AND (s.chance <= {{ STOPLIM['p_suspect'] }} OR s.simple) AND s.run_h >= {{ STOPLIM['min_run_h'] }}
  AND s.n_out >= {{ STOPLIM['min_presses'] }} AND s.active_h >= {{ STOPLIM['min_active_h'] }}
{% if opt['walks_no_press'] %}
{# unique presses of the window (ped_stop.UNIQ_S): a run is presses each within UNIQ_S of the one before; a press counts when it opens a new UNIQ_S-second slot of
   its run, so a button that keeps repeating keeps counting.  Inside the judge, from the window's own presses: no store column, no carry, the same for any cutting #}
-- @step uniq
WITH g AS (SELECT p.dev, p.ch, p.t_on, p.t_on - lag(p.t_on) OVER (PARTITION BY p.dev, p.ch ORDER BY p.t_on) AS gap
           FROM pq_press p JOIN pq_wpart w USING (dev) WHERE p.t_on > w.s),
r AS (SELECT *, sum(CASE WHEN gap IS NULL OR gap > {{ UN['uniq_s'] }} THEN 1 ELSE 0 END) OVER (PARTITION BY dev, ch ORDER BY t_on ROWS UNBOUNDED PRECEDING) AS run FROM g),
a AS (SELECT *, t_on - first_value(t_on) OVER (PARTITION BY dev, ch, run ORDER BY t_on) AS into_run FROM r)
SELECT dev, ch, t_on FROM a QUALIFY row_number() OVER (PARTITION BY dev, ch, run, floor(into_run / {{ UN['uniq_s'] }}) ORDER BY t_on) = 1
{# each input with a movement x each 15 minutes the signal logged: unique presses u, walks w of its phase (else its ped overlap), ped calls c and green starts grn of its phase #}
-- @step pbin
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
LEFT JOIN cl ON cl.dev = l.dev AND cl.ch = i.ch AND cl.q = l.q LEFT JOIN gr ON gr.dev = l.dev AND gr.ch = i.ch AND gr.q = l.q
{# walks without presses: walks jump while the button's presses stop.  Needs no ON or OFF of the button, so it also works when those are missing.  A zero-press run
   against the look-back before it (never after it): walks per press from then, the usual walks at those hours of day; the first bin where the limits hold.  Not
   when another button of the signal fires about then (a signal-wide recall or plan), or a stuck ON already covers it #}
-- @step f_nop
WITH z AS (SELECT *, k - row_number() OVER (PARTITION BY dev, ch ORDER BY q) AS g FROM pq_pbin WHERE u = 0),
runs AS (SELECT dev, ch, g, min(q) AS q0, max(q) AS q1, count(*) AS bins FROM z GROUP BY ALL
         HAVING count(*) >= {{ UN['np_h'] }} * 4 AND sum(w) >= {{ UN['np_walks_h'] }} * {{ UN['np_h'] }}),
b AS (SELECT r.dev, r.ch, r.q0, p.hh, sum(p.u) AS u, sum(p.w) AS w, count(*) AS n
      FROM runs r JOIN pq_pbin p ON p.dev = r.dev AND p.ch = r.ch AND p.q < r.q0 AND p.q >= r.q0 - {{ UN['np_look_h'] }} * 4 GROUP BY ALL),
b0 AS (SELECT dev, ch, q0, (sum(w) + 0.5) / (sum(u) + 0.5) AS rho0, sum(u) AS u0, sum(w) * 1.0 / sum(n) AS wall FROM b GROUP BY ALL HAVING sum(n) >= {{ UN['np_min_look_h'] }} * 4),
x AS (SELECT r.dev, r.ch, r.q0, r.q1, r.bins, p.q, s.rho0, s.u0,
             sum(p.w) OVER o AS ww, sum(p.c) OVER o AS cc, sum(p.w) OVER l AS wl, sum(p.grn) OVER l AS gl, sum(coalesce(h.w * 1.0 / h.n, s.wall)) OVER o AS ew, row_number() OVER o AS m
      FROM runs r JOIN b0 s USING (dev, ch, q0) JOIN pq_pbin p ON p.dev = r.dev AND p.ch = r.ch AND p.q BETWEEN r.q0 AND r.q1
      LEFT JOIN b h ON h.dev = r.dev AND h.ch = r.ch AND h.q0 = r.q0 AND h.hh = p.hh
      WINDOW o AS (PARTITION BY r.dev, r.ch, r.q0 ORDER BY p.q ROWS UNBOUNDED PRECEDING), l AS (PARTITION BY r.dev, r.ch, r.q0 ORDER BY p.q ROWS 3 PRECEDING)),
f AS (SELECT dev, ch, q0, q1, any_value(bins) AS bins, any_value(rho0) AS rho0, max(ww) AS walks, min(q) AS q_fire, arg_min(ww, q) AS w_fire, arg_min(ew, q) AS e_fire, arg_min(m, q) AS m_fire
      FROM x WHERE m >= {{ UN['np_h'] }} * 4 AND ww >= CASE WHEN u0 < {{ UN['np_min_u0'] }} THEN {{ UN['np_low_walks_h'] }} ELSE {{ UN['np_walks_h'] }} END * m / 4.0
                   AND ww >= CASE WHEN u0 < {{ UN['np_min_u0'] }} THEN {{ UN['np_low_jump'] }} ELSE {{ UN['np_jump'] }} END * (ew + 0.5) AND ww / rho0 >= {{ UN['np_implied'] }}
                   AND cc <= {{ UN['np_calls'] }} * ww AND wl >= {{ UN['np_green_share'] }} * gl GROUP BY dev, ch, q0, q1),
n AS (SELECT p.dev, p.ch, f.q0, min(p.q) AS q_next FROM f JOIN pq_pbin p ON p.dev = f.dev AND p.ch = f.ch AND p.q > f.q1 AND p.u > 0 GROUP BY ALL),
{# the run's logged hours of walks at nearly every green: its rating (ped_stop.NP_BAD_H).  Quiet bins of the run do not count, and a walk needs a green to be at #}
h AS (SELECT p.dev, p.ch, f.q0, count(*) / 4.0 AS walk_h FROM f JOIN pq_pbin p ON p.dev = f.dev AND p.ch = f.ch AND p.q BETWEEN f.q0 AND f.q1
      WHERE p.grn > 0 AND p.w >= {{ UN['np_walks_h'] }} / 4.0 AND p.w >= {{ UN['np_green_share'] }} * p.grn GROUP BY ALL)
SELECT 'walks without presses' AS kind, f.dev, f.ch, (f.q0 // 96)::BIGINT AS day, f.q0 * 900.0 AS a, coalesce(n.q_next, f.q1 + 1) * 900.0 AS z, f.w_fire / f.rho0 AS x, f.w_fire AS n,
       f.m_fire / 4.0 AS held, f.e_fire AS every_s, f.walks AS other, n.q_next IS NOT NULL AS ended, coalesce(h.walk_h, 0.0) AS walk_h
FROM f LEFT JOIN n USING (dev, ch, q0) LEFT JOIN h USING (dev, ch, q0)
WHERE NOT EXISTS (SELECT 1 FROM f o WHERE o.dev = f.dev AND o.ch <> f.ch AND abs(o.q_fire - f.q_fire) <= {{ UN['np_other_q'] }})
  AND NOT EXISTS (SELECT 1 FROM pq_f_stuck s WHERE s.dev = f.dev AND s.ch = f.ch AND s.a < (f.q_fire + 1) * 900.0 AND s.z > f.q0 * 900.0)
{% endif %}
{% if opt['sustained_press'] %}
{# sustained pressing (ped_stop.SP_*): 15-minute press counts and walks of each input, its days against the window's other days it logged.  Raw presses: a repeat is the point here #}
-- @step f_press
WITH c AS (SELECT p.dev, p.ch, floor(p.t_on / 900)::BIGINT AS q, count(*) AS n FROM pq_press p JOIN pq_wpart w USING (dev) WHERE p.t_on > w.s GROUP BY ALL),
wk AS (SELECT i.dev, i.ch, floor(w.t21 / 900)::BIGINT AS q, count(*) AS w FROM pq_inp i JOIN pq_walk w ON w.dev = i.dev AND w.ph = i.ph GROUP BY ALL
       UNION ALL
       SELECT i.dev, i.ch, floor(o.ts / 900)::BIGINT, count(*) FROM pq_inp i JOIN pq_ovw o ON o.dev = i.dev AND o.ov = i.ov WHERE i.ph IS NULL GROUP BY ALL),
hot AS (SELECT c.dev, c.ch, c.q // 96 AS day, count(*) AS hot, min(c.q) AS q0, max(c.q) AS q1 FROM c LEFT JOIN wk USING (dev, ch, q)
        WHERE c.n >= {{ SP['bin'] }} AND c.n >= {{ SP['per_walk'] }} * greatest(coalesce(wk.w, 0), 1) GROUP BY ALL HAVING count(*) >= {{ SP['bins'] }}),
d AS (SELECT l.dev, i.ch, l.day, coalesce(sum(c.n), 0) AS n FROM (SELECT DISTINCT dev, q // 96 AS day FROM pq_logq) l JOIN (SELECT DISTINCT dev, ch FROM hot) i USING (dev)
      LEFT JOIN c ON c.dev = l.dev AND c.ch = i.ch AND c.q // 96 = l.day GROUP BY ALL),
u AS (SELECT a.dev, a.ch, a.day, any_value(a.n) AS n, median(b.n) AS usual FROM d a JOIN d b ON b.dev = a.dev AND b.ch = a.ch AND b.day <> a.day GROUP BY ALL)
SELECT 'sustained pressing' AS kind, h.dev, h.ch, h.day, h.q0 * 900.0 AS a, (h.q1 + 1) * 900.0 AS z, u.n / u.usual AS x, u.n AS n, h.hot AS other, u.usual AS every_s
FROM hot h JOIN u USING (dev, ch, day)
WHERE u.n >= {{ SP['presses'] }} AND u.usual >= {{ SP['usual'] }} AND u.n >= {{ SP['times'] }} * u.usual
{% endif %}
{# ped recall: >= 90 % of the phase's (>= 50) walks come with no ped call since the previous don't walk #}
-- @step recall
WITH w AS (SELECT dev, ph, t21 + 1 AS tq, coalesce(lag(t23) OVER (PARTITION BY dev, ph ORDER BY t21), -1e18) AS p23 FROM pq_walk),
x AS (SELECT w.*, c.ts FROM w ASOF LEFT JOIN pq_call c ON c.dev = w.dev AND c.ph = w.ph AND w.tq >= c.ts)
SELECT dev, ph FROM x GROUP BY ALL
HAVING count(*) >= {{ L['recall_walks'] }} AND avg(CASE WHEN ts > p23 THEN 0 ELSE 1 END) >= {{ L['recall_note_share'] }}
