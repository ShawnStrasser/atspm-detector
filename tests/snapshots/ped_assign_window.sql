-- ev
SELECT DISTINCT DeviceId AS dev, (epoch_ns(Timestamp) // 1000000) / 1000.0 AS ts, EventId::INT AS ev, Parameter::INT AS p
                   FROM events WHERE EventId IN (21, 22, 23, 45, 67, 68, 90) AND DeviceId IS NOT NULL AND Timestamp IS NOT NULL AND Timestamp >= TIMESTAMP '2026-09-20 03:17:00' AND Timestamp < TIMESTAMP '2026-09-21 00:00:00';

-- dur
SELECT dev, greatest(max(ts) - min(ts), 3600) AS dur FROM pa_ev GROUP BY dev;

-- press
SELECT dev, p AS d, ts FROM pa_ev WHERE ev = 90;

-- inp
SELECT dev, d, count(*) AS n_press FROM pa_press GROUP BY ALL;

-- mev
SELECT dev, CASE WHEN ev IN (67, 68) THEN 'overlap' ELSE 'phase' END AS kind, p AS m, ts,
                          CASE ev WHEN 45 THEN 'call' WHEN 21 THEN 'walk' WHEN 67 THEN 'walk'
                                  WHEN 22 THEN 'clr' WHEN 68 THEN 'clr' ELSE 'dw' END AS typ
                   FROM pa_ev WHERE (ev IN (21, 22, 23, 45) AND p BETWEEN 1 AND 16) OR (ev IN (67, 68) AND p BETWEEN 1 AND 32);

-- st
SELECT dev, kind, m, ts, arg_max(typ, CASE typ WHEN 'walk' THEN 4 WHEN 'clr' THEN 3 WHEN 'call' THEN 2
                   ELSE 1 END) AS typ FROM pa_mev GROUP BY ALL;

-- wst
SELECT dev, kind, m, ts, arg_max(typ, CASE typ WHEN 'walk' THEN 3 WHEN 'clr' THEN 2 ELSE 1 END) AS typ
                   FROM pa_mev WHERE typ <> 'call' GROUP BY ALL;

-- mov
SELECT dev, kind, m, count(*) FILTER (typ = 'call') AS n_calls, count(*) FILTER (typ = 'walk') AS n_walks
                   FROM pa_mev GROUP BY ALL HAVING count(*) FILTER (typ IN ('call', 'walk')) > 0;

-- calls
SELECT dev, kind, m, ts FROM pa_mev WHERE typ = 'call';

-- walks
SELECT dev, kind, m, ts, ts - lag(ts) OVER (PARTITION BY dev, kind, m ORDER BY ts) AS gap
                      FROM pa_mev WHERE typ = 'walk';

-- cg
SELECT dev, kind, m, sum(g) / 1000.0 AS s FROM (SELECT dev, kind, m,
                    round(1000 * least(1.2, coalesce(ts - lag(ts) OVER (PARTITION BY dev, kind, m ORDER BY ts), 1.2)))::BIGINT AS g
                    FROM pa_calls) GROUP BY ALL;

-- wg
SELECT dev, kind, m, sum(round(1000 * least(120, coalesce(gap, 120)))::BIGINT) / 1000.0 AS s
                  FROM pa_walks GROUP BY ALL;

-- base
SELECT o.dev, o.kind, o.m, least(1, coalesce(cg.s, 0) / d.dur) AS call_base,
                      least(1, coalesce(wg.s, 0) / d.dur) AS walk_base
                    FROM pa_mov o JOIN pa_dur d USING (dev) LEFT JOIN pa_cg cg USING (dev, kind, m)
                    LEFT JOIN pa_wg wg USING (dev, kind, m);

-- uw
SELECT dev, kind, m, ts, lag(ts) OVER (PARTITION BY dev, kind, m ORDER BY ts) AS prev
                  FROM (SELECT DISTINCT dev, kind, m, ts FROM pa_walks WHERE kind = 'phase');

-- noc
SELECT u.dev, u.kind, u.m, avg((coalesce(c.ts, -1e12) <= u.prev)::INT) AS walk_nocall
                   FROM pa_uw u ASOF LEFT JOIN pa_calls c ON u.dev = c.dev AND u.kind = c.kind AND u.m = c.m AND u.ts >= c.ts
                   WHERE u.prev IS NOT NULL GROUP BY ALL;

-- pc
SELECT p.dev, p.d, o.kind, o.m, p.ts, p.ts - 0.2 AS ts_lo FROM pa_press p JOIN pa_mov o USING (dev);

-- pcx
SELECT pc.dev, pc.d, pc.kind, pc.m, pc.ts,
                     c.ts - pc.ts AS lag_call, w.ts - pc.ts AS lag_walk, s.typ AS last_typ, ws.typ AS walk_state
                   FROM pa_pc pc
                   ASOF LEFT JOIN pa_calls c ON pc.dev = c.dev AND pc.kind = c.kind AND pc.m = c.m AND c.ts >= pc.ts_lo
                   ASOF LEFT JOIN pa_walks w ON pc.dev = w.dev AND pc.kind = w.kind AND pc.m = w.m AND w.ts >= pc.ts
                   ASOF LEFT JOIN pa_st s ON pc.dev = s.dev AND pc.kind = s.kind AND pc.m = s.m AND pc.ts_lo > s.ts
                   ASOF LEFT JOIN pa_wst ws ON pc.dev = ws.dev AND pc.kind = ws.kind AND pc.m = ws.m AND pc.ts >= ws.ts;

-- agg
SELECT dev, d, kind, m,
                     avg((coalesce(lag_call, 1e9) <= 1.0)::INT) AS call1,
                     avg((coalesce(lag_call, 1e9) <= 3.0)::INT) AS call3,
                     median(least(coalesce(lag_call, 120), 120)) AS lag_call_med,
                     count(*) FILTER (free) AS free_n,
                     avg((coalesce(lag_call, 1e9) > 3.0)::INT) FILTER (free) AS free_nocall,
                     avg((coalesce(lag_walk, 1e9) > 120)::INT) FILTER (free) AS free_nowalk,
                     avg((coalesce(lag_walk, 1e9) <= 120 OR coalesce(walk_state, '') = 'walk')::INT) AS served
                   FROM (SELECT *, CASE WHEN kind = 'phase' THEN last_typ = 'dw' ELSE last_typ = 'clr' END AS free FROM pa_pcx)
                   GROUP BY ALL;

-- cm
SELECT c.dev, i.d, c.kind, c.m, c.ts, c.ts + 0.2 AS ts_hi FROM pa_calls c JOIN pa_inp i USING (dev);

-- cov
SELECT cm.dev, cm.d, cm.kind, cm.m, avg((cm.ts - coalesce(p.ts, -1e12) <= 1.0)::INT) AS cov1
                   FROM pa_cm cm ASOF LEFT JOIN pa_press p ON cm.dev = p.dev AND cm.d = p.d AND cm.ts_hi >= p.ts
                   GROUP BY ALL;

-- wm
SELECT w.dev, i.d, w.kind, w.m, w.ts, coalesce(w.gap, 1e9) AS gap FROM pa_walks w JOIN pa_inp i USING (dev);

-- wcov
SELECT wm.dev, wm.d, wm.kind, wm.m,
                      avg((wm.ts - coalesce(p.ts, -1e12) <= least(120, wm.gap))::INT) AS walk_cov
                    FROM pa_wm wm ASOF LEFT JOIN pa_press p ON wm.dev = p.dev AND wm.d = p.d AND wm.ts >= p.ts
                    GROUP BY ALL;

-- f0
SELECT a.dev, a.d, a.kind, a.m, i.n_press, o.n_calls, o.n_walks,
                    CASE WHEN a.kind = 'phase' THEN a.call1 END AS call1,
                    CASE WHEN a.kind = 'phase' THEN a.call3 END AS call3,
                    CASE WHEN a.kind = 'phase' THEN b.call_base END AS call_base,
                    CASE WHEN a.kind = 'phase' THEN ln((a.call1 + .01) / (b.call_base + .01)) END AS call_lift,
                    CASE WHEN a.kind = 'phase' THEN a.lag_call_med END AS lag_call_med,
                    a.free_n, CASE WHEN a.kind = 'phase' THEN a.free_nocall END AS free_nocall, a.free_nowalk,
                    a.served, b.walk_base, ln((a.served + .01) / (b.walk_base + .01)) AS served_lift,
                    CASE WHEN a.kind = 'phase' THEN coalesce(c.cov1, 0) END AS cov1, coalesce(w.walk_cov, 0) AS walk_cov,
                    (a.kind = 'overlap')::INT AS is_overlap, coalesce(n.walk_nocall, 0) AS walk_nocall
                  FROM pa_agg a JOIN pa_inp i USING (dev, d) JOIN pa_mov o USING (dev, kind, m)
                  JOIN pa_base b USING (dev, kind, m)
                  LEFT JOIN pa_cov c USING (dev, d, kind, m) LEFT JOIN pa_wcov w USING (dev, d, kind, m)
                  LEFT JOIN pa_noc n USING (dev, kind, m);

-- feat
SELECT *,
                    CASE WHEN kind = 'phase' THEN rank() OVER (PARTITION BY dev, d, kind ORDER BY call_lift DESC) END AS rank_call,
                    CASE WHEN kind = 'phase' THEN call1 - coalesce(max(call1) OVER (PARTITION BY dev, d, kind ORDER BY call1 DESC
                         ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING EXCLUDE CURRENT ROW), 0) END AS margin_call,
                    rank() OVER (PARTITION BY dev, d ORDER BY served_lift DESC) AS rank_served,
                    served_lift - coalesce(max(served_lift) OVER (PARTITION BY dev, d ORDER BY served_lift DESC
                         ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING EXCLUDE CURRENT ROW), 0) AS margin_served,
                    rank() OVER (PARTITION BY dev, kind, m ORDER BY CASE WHEN kind = 'phase' THEN cov1 ELSE walk_cov END DESC) AS rank_cov,
                    count(*) OVER (PARTITION BY dev, d) AS n_cand
                  FROM pa_f0;
