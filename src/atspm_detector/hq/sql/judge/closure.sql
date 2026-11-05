{# note 154b (option lane_closure, health_duckdb._closure; rules from the user's review 2026-10-22): a lane closure = the
   lane goes to NEAR ZERO (<= 15 % of its expected count, ON < 2 % of the time) and a detector of the same kind on its
   phase (or the NEMA-paired phase) takes the lost vehicles back (>= half on one detector; 0.5-1.5 x in all), judged
   against the same clock hours on days of the same type (weekday / weekend).  Candidates: every near-zero run of >= 2
   judged hours, and the silent run; option closure_1h (note 154aa): also every near-zero run of >= 3 judged
   quarter-hours.  A silence that starts AND ends with a coordination pattern change (event 131) is a timing-plan switch.
   -> xfc (xf + lc_d, lc_l), lcres (one row per closure / switch). #}
{% import "_lib.sql" as lib with context %}
{% import "adapters.sql" as ad with context %}
{% set cls = T['cls']|values('fn, cl') %}
{% set q1h = opt['closure_1h'] %}
{% set ex = ('AND NOT x.sig_silent ' if opt['signal_silent'] else '') ~ ('AND NOT x.ph_drop AND NOT x.ph_lock ' if opt['phase_drop'] else '') ~ 'AND (x.dev, x.det) NOT IN (SELECT (dev, det) FROM flag WHERE flagged)' %}
{# near-zero hours: judged (covered, >= 10 expected), <= 15 % of what its share gives -- of its own yardstick before the
   drop, or of the other phases' traffic (a closure leaves those alone; a same-lane yardstick closes with it); unjudged
   hours do not break a run, a normal hour does #}
-- @step lcz mat
SELECT * FROM (
WITH o AS (SELECT m.dev, m.det, m.hid, m.M, sum(m2.M) AS ro FROM mh m JOIN dz z USING (dev, det)
           JOIN mh m2 ON m2.dev = m.dev AND m2.hid = m.hid JOIN dz z2 ON z2.dev = m2.dev AND z2.det = m2.det
           WHERE z.ph IS NOT NULL AND z2.ph IS NOT NULL AND z2.ph <> z.ph GROUP BY m.dev, m.det, m.hid, m.M),
os AS (SELECT o.*, x.j, x.ch, median(o.M / o.ro) FILTER (x.ch AND o.ro >= 30) OVER (PARTITION BY o.dev, o.det) AS so
       FROM o JOIN hix x USING (dev, hid)),
h AS (SELECT g.dev, g.det, g.j,
             (g.ch AND CASE WHEN a.re THEN a.s2 ELSE a.s END * g.R >= 10) OR coalesce(b.ch AND b.so * b.ro >= 10, false)
             AS judged,
             (g.ch AND CASE WHEN a.re THEN a.s2 ELSE a.s END * g.R >= 10
              AND g.M <= 0.15 * CASE WHEN a.re THEN a.s2 ELSE a.s END * g.R)
             OR coalesce(b.ch AND b.so * b.ro >= 10 AND b.M <= 0.15 * b.so * b.ro, false) AS z
      FROM gh2 g JOIN gh4 a USING (dev, det) LEFT JOIN os b ON b.dev = g.dev AND b.det = g.det AND b.j = g.j),
r AS (SELECT *, sum((judged AND NOT z)::INT) OVER (PARTITION BY dev, det ORDER BY j) AS grp FROM h)
SELECT dev, det, min(j) FILTER (z) AS ja, max(j) FILTER (z) + 1 AS jz FROM r
GROUP BY dev, det, grp HAVING count(*) FILTER (z) >= 2)
{% if q1h %}
{# note 154aa (closure_1h): the same near-zero test on quarter-hours (covered, >= 2.5 expected, <= 15 %): the
   yardstick's hour spread over its quarters as the other phases' traffic is, or the other phases' traffic of the
   quarter itself at the detector's hourly share of it; runs of >= 3 near-zero quarters #}
-- @step lcq mat
SELECT * FROM (
WITH m AS (SELECT g.dev, g.det, (w.s_ns + g.b * {{ NS5 }}) // {{ NSQ }} AS qid, sum(g.n) AS M, count(*) FILTER (g.cov) = 3 AS cq
           FROM g5 g JOIN win w USING (dev) JOIN dz z ON z.dev = g.dev AND z.det = g.det
           WHERE w.hours >= {{ LJ['min_hours_counts']|dbl }} AND z.ph IS NOT NULL GROUP BY g.dev, g.det, qid),
o AS (SELECT m.dev, m.det, m.qid, m.qid // 4 AS hid, m.M, m.cq, coalesce(sum(m2.M), 0) AS ro
      FROM m JOIN dz z USING (dev, det) LEFT JOIN (m m2 JOIN dz z2 ON z2.dev = m2.dev AND z2.det = m2.det)
        ON m2.dev = m.dev AND m2.qid = m.qid AND z2.ph IS NOT NULL AND z2.ph <> z.ph
      GROUP BY m.dev, m.det, m.qid, m.M, m.cq),
oh AS (SELECT o.*, sum(o.ro) OVER (PARTITION BY o.dev, o.det, o.hid) AS roh,
              sum(o.M) OVER (PARTITION BY o.dev, o.det, o.hid) AS Mh,
              bool_and(o.cq) OVER (PARTITION BY o.dev, o.det, o.hid) AND count(*) OVER (PARTITION BY o.dev, o.det, o.hid) = 4
              AS chq FROM o),
os AS (SELECT oh.*, median(Mh / roh) FILTER (chq AND roh >= 30) OVER (PARTITION BY dev, det) AS so FROM oh),
e AS (SELECT s.dev, s.det, s.qid, s.M, s.cq,
             CASE WHEN g.ch THEN CASE WHEN a.re THEN a.s2 ELSE a.s END * g.R
                  * CASE WHEN s.roh > 0 THEN s.ro / s.roh ELSE 0.25 END END AS ey,
             s.so * s.ro AS eo
      FROM os s JOIN hix x ON x.dev = s.dev AND x.hid = s.hid
      LEFT JOIN gh2 g ON g.dev = s.dev AND g.det = s.det AND g.j = x.j LEFT JOIN gh4 a ON a.dev = s.dev AND a.det = s.det),
h AS (SELECT dev, det, qid, coalesce(cq AND ey >= 2.5, false) OR coalesce(cq AND eo >= 2.5, false) AS judged,
             coalesce(cq AND ey >= 2.5 AND M <= 0.15 * ey, false) OR coalesce(cq AND eo >= 2.5 AND M <= 0.15 * eo, false) AS z
      FROM e),
r AS (SELECT *, sum((judged AND NOT z)::INT) OVER (PARTITION BY dev, det ORDER BY qid) AS grp FROM h)
SELECT dev, det, min(qid) FILTER (z) AS ja, max(qid) FILTER (z) + 1 AS jz FROM r
GROUP BY dev, det, grp HAVING count(*) FILTER (z) >= 3)
{% endif %}
{# closure_1h: quarter runs flagged (qr), so a silence inside one stays a candidate of its own (plan switch test) #}
-- @step lcan mat
SELECT z.dev, z.det, greatest(((h.hid0 + z.ja) * {{ NSH }} - w.s_ns) // {{ NS5 }}, 0) AS b0,
       least(((h.hid0 + z.jz) * {{ NSH }} - w.s_ns) // {{ NS5 }}, w.nb5) AS b1{{ ', false AS qr' if q1h else '' }}
FROM lcz z JOIN win w USING (dev) JOIN (SELECT dev, min(hid) AS hid0 FROM hix GROUP BY dev) h USING (dev){% if q1h %}

UNION ALL
SELECT z.dev, z.det, greatest((z.ja * {{ NSQ }} - w.s_ns) // {{ NS5 }}, 0), least((z.jz * {{ NSQ }} - w.s_ns) // {{ NS5 }}, w.nb5), true FROM lcq z JOIN win w USING (dev){% endif %}

UNION ALL
SELECT dev, det, drop_b0, drop_b1{{ ', false' if q1h else '' }} FROM xf WHERE drop_b1 - drop_b0 >= 3 AND drop_lam >= 30
-- @step lcc mat
SELECT x.dev, x.det, x.ph, x.fn, 'd' AS kind, x.drop_b0::BIGINT AS b0, x.drop_b1::BIGINT AS b1
FROM xf x JOIN limits l USING (dev, det)
WHERE x.n_on > 0 AND x.ph IS NOT NULL AND x.drop_b1 - x.drop_b0 >= 3 AND x.drop_lam >= l.lim_dropout {{ ex }}
  AND NOT EXISTS (SELECT 1 FROM lcan a JOIN lcz z ON z.dev = a.dev AND z.det = a.det
                  WHERE a.dev = x.dev AND a.det = x.det AND a.b0 <= x.drop_b0 AND a.b1 >= x.drop_b1
                  AND a.b1 - a.b0 > x.drop_b1 - x.drop_b0{{ ' AND NOT a.qr' if q1h else '' }})
UNION ALL
SELECT x.dev, x.det, x.ph, x.fn, 'z' || z.ja, greatest(((h.hid0 + z.ja) * {{ NSH }} - w.s_ns) // {{ NS5 }}, 0),
       least(((h.hid0 + z.jz) * {{ NSH }} - w.s_ns) // {{ NS5 }}, w.nb5)
FROM lcz z JOIN xf x USING (dev, det) JOIN win w USING (dev)
JOIN (SELECT dev, min(hid) AS hid0 FROM hix GROUP BY dev) h USING (dev)
WHERE x.ph IS NOT NULL {{ ex }}{% if q1h %}

UNION ALL
SELECT x.dev, x.det, x.ph, x.fn, 'q' || z.ja, greatest((z.ja * {{ NSQ }} - w.s_ns) // {{ NS5 }}, 0), least((z.jz * {{ NSQ }} - w.s_ns) // {{ NS5 }}, w.nb5)
FROM lcq z JOIN xf x USING (dev, det) JOIN win w USING (dev)
WHERE x.ph IS NOT NULL {{ ex }}{% endif %}

{# partner groups: its own phase (gq -1), and its own phase + one allowed other phase gq (lane_pairs: nema / any /
   none) #}
{% set allow %}{% if opt['lane_pairs'] == 'nema' %}{% for a, b in NEMA_PAIRS %}(least(c.ph, p.ph) = {{ a }} AND greatest(c.ph, p.ph) = {{ b }}){% if not loop.last %} OR {% endif %}{% endfor %}{% elif opt['lane_pairs'] == 'any' %}true{% else %}false{% endif %}{% endset %}
-- @step lcg mat
SELECT dev, det, kind, b0, b1, ph, fn, -1.0 AS gq FROM lcc
UNION ALL
SELECT c.dev, c.det, c.kind, c.b0, c.b1, c.ph, c.fn, p.ph FROM lcc c
JOIN (SELECT DISTINCT dev, ph FROM dz WHERE ph IS NOT NULL) p ON p.dev = c.dev AND p.ph <> c.ph AND ({{ allow }})
{# closure_1h: the quarter-hour candidates only add candidates; the references stay those of the hour runs #}
{% set nqk = " AND o.kind NOT LIKE 'q%'" if q1h else '' %}
{# members: the candidate (d), its partners of the same kind (p), the reference (x: every live detector on the other
   phases without a problem of its own, and not of a phase + kind with a candidate: its partners gain) #}
-- @step lcm mat
SELECT g.dev, g.det, g.kind, g.gq, g.det AS m, 'd' AS role FROM lcg g JOIN ({{ cls }}) a ON a.fn = g.fn
UNION ALL
SELECT g.dev, g.det, g.kind, g.gq, z.det, 'p'
FROM lcg g JOIN dz z ON z.dev = g.dev AND z.det <> g.det AND (z.ph = g.ph OR z.ph = g.gq)
JOIN ({{ cls }}) a ON a.fn = g.fn JOIN ({{ cls }}) b ON b.fn = z.fn AND b.cl = a.cl
JOIN dstat s ON s.dev = z.dev AND s.det = z.det AND s.live
LEFT JOIN twin t ON t.dev = g.dev AND t.i = g.det AND t.j = z.det
WHERE t.i IS NULL AND NOT EXISTS (SELECT 1 FROM twin u JOIN dz w ON w.dev = u.dev AND w.det = u.j
                                   WHERE u.dev = z.dev AND u.i = z.det AND u.j < z.det AND w.fn = z.fn)
UNION ALL
SELECT g.dev, g.det, g.kind, g.gq, z.det, 'x'
FROM lcg g JOIN dz z ON z.dev = g.dev AND z.ph IS NOT NULL AND z.ph <> g.ph AND z.ph <> g.gq
  AND z.status1 IS DISTINCT FROM 'bad'
JOIN dstat s ON s.dev = z.dev AND s.det = z.det AND s.live
LEFT JOIN ({{ cls }}) k ON k.fn = z.fn
WHERE NOT EXISTS (SELECT 1 FROM lcc o LEFT JOIN ({{ cls }}) ko ON ko.fn = o.fn
                  WHERE o.dev = z.dev AND (o.det = z.det OR (o.ph = z.ph AND ko.cl = k.cl)){{ nqk }})
{# sums in the period P and, outside every candidate period of the detector and the member's own anomalies, with the
   other phases' traffic at those same bins: the same clock hours on days of the same type (t), the 2 h either side
   (ad; for an hour run past the hour next to it, which may hold the start), anywhere (a); the time ON in P;
   closure_1h: a member's own quarter-hour runs (often quiet night quarters) stay in its reference #}
-- @step lcrows mat
SELECT * FROM (
WITH bw AS (SELECT b.dev, b.b, b.hr, isodow(make_timestamp((w.s_ns + b.b * {{ NS5 }}) // 1000)) >= 6 AS wk FROM bins5 b JOIN win w USING (dev))
SELECT m.dev, m.det, m.kind, m.gq, m.m, m.role, c.b0, c.b1, n.b, n.n, n.occ, n.hr, bw.wk,
       n.b >= c.b0 AND n.b < c.b1 AS inp,
       NOT EXISTS (SELECT 1 FROM lcan a WHERE a.dev = m.dev AND a.det = m.m AND n.b >= a.b0 AND n.b < a.b1{{ ' AND NOT a.qr' if q1h else '' }})
       AND NOT EXISTS (SELECT 1 FROM lcc o WHERE o.dev = m.dev AND o.det = m.det AND n.b >= o.b0 AND n.b < o.b1{{ nqk }}) AS ref
FROM lcm m JOIN lcc c ON c.dev = m.dev AND c.det = m.det AND c.kind = m.kind
JOIN g5 n ON n.dev = m.dev AND n.det = m.m AND n.cov JOIN bw ON bw.dev = n.dev AND bw.b = n.b)
-- @step lcs
SELECT * FROM (
WITH hp AS (SELECT DISTINCT dev, det, kind, hr, wk FROM lcrows WHERE role = 'd' AND inp),
xb AS (SELECT dev, det, kind, gq, b, sum(n) AS xn FROM lcrows WHERE role = 'x' AND ref GROUP BY ALL),
r AS (SELECT r.*, h.hr IS NOT NULL AS th, xb.xn,
             CASE WHEN r.kind = 'd' THEN 0 WHEN r.kind LIKE 'q%' THEN 3 ELSE 12 END AS g
      FROM lcrows r LEFT JOIN hp h ON h.dev = r.dev AND h.det = r.det AND h.kind = r.kind AND h.hr = r.hr AND h.wk = r.wk
      LEFT JOIN xb ON xb.dev = r.dev AND xb.det = r.det AND xb.kind = r.kind AND xb.gq = r.gq AND xb.b = r.b)
SELECT dev, det, kind, gq, m, role,
       sum(n) FILTER (inp) AS sp, sum(occ) FILTER (inp) AS op, count(*) FILTER (inp) AS np,
       sum(n) FILTER (ref AND NOT inp) AS sra, sum(xn) FILTER (ref AND NOT inp) AS xra,
       sum(n) FILTER (ref AND NOT inp AND th) AS srt, sum(xn) FILTER (ref AND NOT inp AND th) AS xrt,
       count(*) FILTER (ref AND NOT inp AND th) AS nrt,
       sum(n) FILTER (ref AND (b >= b0 - g - 24 AND b < b0 - g OR b >= b1 + g AND b < b1 + g + 24)) AS sad,
       sum(xn) FILTER (ref AND (b >= b0 - g - 24 AND b < b0 - g OR b >= b1 + g AND b < b1 + g + 24)) AS xad,
       sum(n) FILTER (ref AND b >= b0 - g - 12 AND b < b0 - g) AS pre,
       sum(xn) FILTER (ref AND b >= b0 - g - 12 AND b < b0 - g) AS xpre,
       sum(n) FILTER (ref AND b >= b1 + g AND b < b1 + g + 12) AS post,
       sum(xn) FILTER (ref AND b >= b1 + g AND b < b1 + g + 12) AS xpost
FROM r GROUP BY dev, det, kind, gq, m, role)
-- @step lch mat
SELECT dev, det, kind, gq, m, role, (b - b0) // 12 AS hh, sum(n) AS n FROM lcrows WHERE inp GROUP BY ALL
-- @step lce
SELECT * FROM (
WITH t AS (SELECT *, bool_and(nrt >= 0.5 * np) OVER (PARTITION BY dev, det, kind, gq) AS tod FROM lcs),
s AS (SELECT dev, det, kind, gq, role, m, sp, op, np, pre, post, xpre, xpost, tod,
             CASE WHEN tod THEN srt WHEN xad >= 100 THEN sad ELSE sra END AS sr,
             CASE WHEN tod THEN xrt WHEN xad >= 100 THEN xad ELSE xra END AS xr FROM t),
x AS (SELECT dev, det, kind, gq, sum(sp) AS xp FROM s WHERE role = 'x' GROUP BY ALL),
e AS (SELECT s.dev, s.det, s.kind, s.gq, s.m, s.role, s.tod, s.op / greatest(s.np * 300.0, 1) AS occ_p, s.sp AS obs,
             s.sr / s.xr * x.xp AS ex, s.sr / s.xr AS rr,
             (s.xpre >= 30 AND s.sp / x.xp >= 1.25 * coalesce(s.pre, 0) / s.xpre)
             OR (s.xpost >= 30 AND s.sp / x.xp >= 1.25 * coalesce(s.post, 0) / s.xpost)
             OR (coalesce(s.xpre, 0) < 30 AND coalesce(s.xpost, 0) < 30) AS step
      FROM s JOIN x USING (dev, det, kind, gq) WHERE s.role <> 'x' AND s.xr >= 100 AND x.xp >= 30),
xh AS (SELECT dev, det, kind, gq, hh, sum(n) AS xn FROM lch WHERE role = 'x' GROUP BY ALL),
fr AS (SELECT p.dev, p.det, p.kind, p.gq, p.m,
              count(*) FILTER (p.n > e.rr * xh.xn AND e.rr * xh.xn >= 5) / nullif(count(*) FILTER (e.rr * xh.xn >= 5), 0)
              AS frac
       FROM lch p JOIN xh USING (dev, det, kind, gq, hh) JOIN e ON e.dev = p.dev AND e.det = p.det AND e.kind = p.kind
            AND e.gq = p.gq AND e.m = p.m
       WHERE p.role = 'p' GROUP BY p.dev, p.det, p.kind, p.gq, p.m)
SELECT d.dev, d.det, d.kind, d.gq, d.obs AS obs_d, d.ex AS exp_d, d.ex - d.obs AS loss, d.tod, d.occ_p AS occ_d,
       max(e.obs - e.ex) AS top_gain, arg_max(f.frac, e.obs - e.ex) AS top_frac, arg_max(e.step, e.obs - e.ex) AS top_step, sum(e.obs) AS obs_g, sum(e.ex) AS exp_g, sum(e.obs - e.ex) AS gain,
       list(e.m ORDER BY e.obs - e.ex DESC, e.m) FILTER (e.obs - e.ex >= 0.15 * (d.ex - d.obs)) AS gainers,
       string_agg('det ' || e.m || CASE WHEN z.ph IS NOT NULL THEN ' (P' || trunc(z.ph)::BIGINT || ')' ELSE '' END, ', '
                  ORDER BY e.obs - e.ex DESC, e.m) FILTER (e.obs - e.ex >= 0.15 * (d.ex - d.obs)) AS glab
FROM e d JOIN e ON e.dev = d.dev AND e.det = d.det AND e.kind = d.kind AND e.gq = d.gq AND e.role = 'p'
LEFT JOIN dz z ON z.dev = e.dev AND z.det = e.m
LEFT JOIN fr f ON f.dev = e.dev AND f.det = e.det AND f.kind = e.kind AND f.gq = e.gq AND f.m = e.m
WHERE d.role = 'd'
GROUP BY d.dev, d.det, d.kind, d.gq, d.obs, d.ex, d.tod, d.occ_p)
{# a closure: near zero (<= 15 % of expected, ON < 2 % of the time), a clear loss (>= 100 vehicles, GEH >= 8), one
   partner of the same kind takes back >= 55 % of it, all of them 0.5-2 x of it, clearly (GEH >= 3), and that partner
   gains in >= 2/3 of the hours and steps up (>= 1.25 x) against the hour before or after the period (a spike or a
   normal ramp is not a transfer); its own phase first, else the paired phase taking back the most #}
-- @step lcr mat
SELECT * FROM (
  SELECT e.*, row_number() OVER (PARTITION BY dev, det, kind ORDER BY gq >= 0, gain / loss DESC, gq) AS rk
  FROM lce e WHERE obs_d <= 0.15 * exp_d AND occ_d < 0.02 AND loss >= 100 AND {{ lib.geh('exp_d', 'obs_d') }} >= 8
       AND top_gain >= 0.55 * loss AND gain >= 0.5 * loss AND gain <= 2 * loss AND {{ lib.geh('obs_g', 'exp_g') }} >= 3
       AND coalesce(top_frac, 0) >= 0.67 AND coalesce(top_step, false))
WHERE rk = 1
{# a coordination pattern change (131) within tol 5-min bins of both ends of a silence (note 154f: 5 min, user) #}
{% set tol = 1 %}
{% if opt['plan_switch'] %}
{{ ad.ev131() }}
-- @step lcsw
SELECT c.dev, c.det, c.kind, first(p.par ORDER BY abs(p.dn // {{ NS5 }} - c.b0), p.dn, p.par) AS pat
FROM lcc c JOIN ev131 p ON p.dev = c.dev AND p.eid = 131 AND abs(p.dn // {{ NS5 }} - c.b0) <= {{ tol }}
WHERE c.kind = 'd' AND EXISTS (SELECT 1 FROM ev131 e WHERE e.dev = c.dev AND e.eid = 131 AND abs(e.dn // {{ NS5 }} - c.b1) <= {{ tol }})
GROUP BY c.dev, c.det, c.kind
{% else %}
-- @step lcsw
SELECT NULL::INT AS dev, NULL::INT AS det, NULL::VARCHAR AS kind, NULL::INT AS pat WHERE false
{% endif %}
{# one row per closure / switch; a run inside a longer run of the same detector and type is the same event #}
{% set a0 %}
  SELECT c.dev, c.det, c.kind, r.gq, r.loss, r.gain, r.gainers, r.glab, c.b0, c.b1, s.pat IS NOT NULL AS switch, s.pat
  FROM lcc c LEFT JOIN lcr r USING (dev, det, kind) LEFT JOIN lcsw s USING (dev, det, kind)
  WHERE r.dev IS NOT NULL OR s.dev IS NOT NULL{% endset %}
{% if q1h %}
{# note 154ac (user, review v13 row 5): a closure on a day when the detector has faults of its own
   (LC_FAULT_DAY) before AND after it on the same calendar day is part of the fault, not a closure.  Its faults:
   silences over its suspect limit, count drops against its phase mates (>= 2 h, >= the suspect GEH), counted stuck-on
   holds -- each not overlapping any closure / switch candidate of the detector (those are explained) #}
-- @step lcfp mat
{% if opt['all_silences'] %}
SELECT u.dev, u.det, w.s_ns + u.a * {{ NS5 }} AS fa, w.s_ns + u.z * {{ NS5 }} AS fz FROM silall u JOIN limits l USING (dev, det) JOIN win w USING (dev) WHERE u.lam >= l.lim_dropout
UNION ALL
{% endif %}
SELECT x.dev, x.det, w.s_ns + x.drop_b0 * {{ NS5 }}, w.s_ns + x.drop_b1 * {{ NS5 }} FROM xf x JOIN limits l USING (dev, det)
JOIN win w USING (dev) WHERE x.drop_b1 > x.drop_b0 AND x.drop_lam >= l.lim_dropout
UNION ALL
SELECT e.dev, e.det, w.s_ns + round(e.s * 1e9)::BIGINT, w.s_ns + round(e.f * 1e9)::BIGINT FROM efl e JOIN win w USING (dev)
WHERE e.counted{% if opt['mate_drop'] %}

UNION ALL
SELECT dev, det, a * {{ NSH }}, z * {{ NSH }} FROM mhrun WHERE kind = 'drop' AND hours >= 2 AND rgeh >= {{ M['geh'][0]|dbl }}{% endif %}

{% set bf %}EXISTS (SELECT 1 FROM f WHERE f.dev = c.dev AND f.det = c.det AND f.fz <= c.a_ns AND (f.fz - 1) // {{ DAYNS }} = c.a_ns // {{ DAYNS }}){% endset %}
{% set af %}EXISTS (SELECT 1 FROM f WHERE f.dev = c.dev AND f.det = c.det AND f.fa >= c.z_ns AND f.fa // {{ DAYNS }} = (c.z_ns - 1) // {{ DAYNS }}){% endset %}
{% set a0 %}
WITH c AS (SELECT c.*, w.s_ns + c.b0 * {{ NS5 }} AS a_ns, least(w.s_ns + c.b1 * {{ NS5 }}, w.e_ns) AS z_ns
           FROM ({{ a0 }}) c JOIN win w USING (dev)),
f AS (SELECT * FROM lcfp p WHERE NOT EXISTS (SELECT 1 FROM c WHERE c.dev = p.dev AND c.det = p.det
                                              AND c.a_ns < p.fz AND c.z_ns > p.fa))
SELECT c.* EXCLUDE (a_ns, z_ns) FROM c
WHERE c.switch OR NOT ({{ '' if LCF['all'] else "c.kind LIKE 'q%' AND " }}{{ bf }} {{ 'AND' if LCF['sides'] == 2 else 'OR' }} {{ af }}){% endset %}
{% endif %}
-- @step lcres mat
SELECT * FROM (
WITH a AS ({{ a0 }})
SELECT a.*, w.s_ns + a.b0 * {{ NS5 }} AS a_ns, least(w.s_ns + a.b1 * {{ NS5 }}, w.e_ns) AS z_ns
FROM a JOIN win w USING (dev)
WHERE NOT EXISTS (SELECT 1 FROM a o WHERE o.dev = a.dev AND o.det = a.det AND o.kind <> a.kind AND o.switch = a.switch
                  AND {{ 'o.b0 < a.b1 AND o.b1 > a.b0' if q1h else 'o.b0 <= a.b0 AND o.b1 >= a.b1' }}
                  AND (o.b1 - o.b0 > a.b1 - a.b0 OR (o.b1 - o.b0 = a.b1 - a.b0 AND o.kind < a.kind))))
{# the silence / the count drop are explained when closures cover half of it; note 154n / 154o (mate_drop /
   count_rise): any overlap with a closure explains a run of hours against the phase mates (the closure is its
   near-zero core), and a detector with a timing-plan switch counts by the plan #}
{% macro ov(p, who) %}(SELECT coalesce(sum(greatest(least(r.z_ns, x.{{ p }}_z * {{ NSH }}) - greatest(r.a_ns, x.{{ p }}_a * {{ NSH }}), 0)), 0) FROM lcres r WHERE r.dev = x.dev AND {{ who }}) > 0 OR EXISTS (SELECT 1 FROM lcres r WHERE r.dev = x.dev AND r.det = x.det AND r.switch){% endmacro %}
-- @step xfc mat
SELECT x.* EXCLUDE (l0, l1),
  EXISTS (SELECT 1 FROM lcres r WHERE r.dev = x.dev AND r.det = x.det AND x.drop_b1 > x.drop_b0
          AND least(r.b1, x.drop_b1) - greatest(r.b0, x.drop_b0) >= 0.5 * (x.drop_b1 - x.drop_b0)) AS lc_d,
  coalesce(x.l1 > x.l0 AND (SELECT sum(greatest(least(r.b1, x.l1) - greatest(r.b0, x.l0), 0)) FROM lcres r
                            WHERE r.dev = x.dev AND r.det = x.det AND NOT r.switch) >= 0.5 * (x.l1 - x.l0), false) AS lc_l{% if opt['mate_drop'] or opt['count_rise'] %},
  coalesce({{ ov('md', 'r.det = x.det') }}, false) AS lc_md,
  coalesce({{ ov('mr', '(r.det = x.det OR list_contains(r.gainers, x.det))') }}, false) AS lc_mr{% endif %}

FROM (SELECT x.*, (epoch_ns(x.drop_t) - w.s_ns) // {{ NS5 }} AS l0,
             coalesce((epoch_ns(x.drop_t2) - w.s_ns) // {{ NS5 }}, w.nb5) AS l1 FROM xf x JOIN win w USING (dev)) x
