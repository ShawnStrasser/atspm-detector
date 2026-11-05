{# Stage 1 (health_duckdb._stage1_both / _stage1 / _no_traffic): pass 1 on every detector; pass 2 (signals with a
   detector pass 1 calls bad, and a predicted phase) without the bad detectors as anybody's phase mate; a bad detector
   keeps its pass-1 row. #}
{% import "_lib.sql" as lib with context %}
{# the four stage-1 rules for the detectors of {p}ph (dev, det, ph): references, silent runs, night count against the
   detector it tracks, stuck-on holds with their recovery -> {p}st (one row per detector) #}
{% macro stage1(p) %}
{# reference candidates: the other live detectors of the signal, never a twin; same = on the same phase #}
-- @step {{ p }}rf mat
SELECT a.dev, a.det, b.det AS r, coalesce(a.ph = b.ph, false) AS same
FROM {{ p }}ph a JOIN {{ p }}ph b ON a.dev = b.dev AND a.det <> b.det
JOIN dstat sb ON sb.dev = b.dev AND sb.det = b.det AND sb.live
LEFT JOIN twin t ON t.dev = a.dev AND t.i = a.det AND t.j = b.det WHERE t.i IS NULL
{# yardstick kind: the other detectors on its phase when there are >= 2, else the rest of the signal #}
-- @step {{ p }}kd mat
SELECT a.dev, a.det, a.ph, coalesce(r.nsame, 0) >= 2 AS isph
FROM {{ p }}ph a LEFT JOIN (SELECT dev, det, count(*) FILTER (same) AS nsame FROM {{ p }}rf GROUP BY dev, det) r
USING (dev, det)
-- @step {{ p }}pref mat
SELECT f.dev, f.det, f.r FROM {{ p }}rf f JOIN {{ p }}kd k USING (dev, det) WHERE k.isph AND f.same
{# recovery partner candidates: every other live detector on the predicted phase, twins included #}
-- @step {{ p }}pc mat
SELECT a.dev, a.det, b.det AS c FROM {{ p }}ph a JOIN {{ p }}ph b ON a.dev = b.dev AND a.det <> b.det AND a.ph = b.ph
JOIN dstat sb ON sb.dev = b.dev AND sb.det = b.det AND sb.live
{# reference members of every detector for window sums: phase refs, else the rest of the signal #}
-- @step {{ p }}rm mat
SELECT dev, det, r FROM {{ p }}pref
UNION ALL SELECT k.dev, k.det, d.det FROM {{ p }}kd k JOIN dets d ON d.dev = k.dev AND d.det <> k.det WHERE NOT k.isph
{# silent runs: expected ONs lam = n_on / (reference ONs outside the run) x reference ONs inside it #}
-- @step {{ p }}rs0 mat
SELECT u.dev, u.det, u.a, u.z, m.r, w.nb5 FROM runs u JOIN {{ p }}rm m ON m.dev = u.dev AND m.det = u.det
JOIN win w ON w.dev = u.dev
{{ lib.at(p ~ 'rs', p ~ 'rs0', 'r', ('a', 'z', 'nb5'), 'cn') }}
-- @step {{ p }}rs
SELECT dev, det, a, z, sum(cn_z - cn_a) AS s_in, sum(cn_nb5) AS s_tot FROM {{ p }}rs_3 GROUP BY dev, det, a, z
{# the run with the most expected ONs; equal ones (to 6 decimals: the sums' last bits depend on how DuckDB splits the
   work) -> the earliest (note 154ad) #}
-- @step {{ p }}dr mat
SELECT k.dev, k.det, coalesce(d.lam, 0.0) AS drop_lam, coalesce(d.b0, -1) AS drop_b0, coalesce(d.b1, -1) AS drop_b1
FROM {{ p }}kd k LEFT JOIN (
  SELECT dev, det, max(lam) AS lam, first(a ORDER BY round(lam, 6) DESC, a) AS b0, first(z ORDER BY round(lam, 6) DESC, a) AS b1
  FROM (SELECT r.*, s.n_on / greatest(r.s_tot - r.s_in, 1) * r.s_in AS lam FROM {{ p }}rs r JOIN dstat s USING (dev, det)
        {{ lib.cap() }})
  WHERE lam > 0 GROUP BY dev, det) d USING (dev, det)
{% if opt['all_silences'] %}
{# note 154f: every silent run of >= 15 min with its expected count (a failure that comes back each day); the ones over
   the detector's own suspect limit (a night one >= 50 expected) are listed next to the worst #}
-- @step {{ p }}dra mat
SELECT r.dev, r.det, r.a, r.z, s.n_on / greatest(r.s_tot - r.s_in, 1) * r.s_in AS lam
FROM {{ p }}rs r JOIN dstat s USING (dev, det) JOIN win w ON w.dev = r.dev
WHERE r.z - r.a >= 3 AND r.z - r.a <= w.nb5 - 12 AND s.n_on >= 5
{% endif %}
-- @step {{ p }}drx mat
SELECT d.*, s.n_on, (d.drop_b1 = w.nb5 OR (d.drop_b1 > 0 AND g.last_cov < d.drop_b1)) AS drop_to_end
FROM {{ p }}dr d JOIN dstat s USING (dev, det) JOIN win w ON w.dev = d.dev JOIN sig0 g ON g.dev = d.dev
{# partner requests: the whole sample (night), each stuck-on hold >= 15 min and each silent run (left out);
   [xa, xb) = 15-min bins left out #}
-- @step {{ p }}rq mat
SELECT 'n' AS kind, dev * 128 + det AS id, dev, det, 0 AS xa, 0 AS xb FROM {{ p }}kd
UNION ALL
SELECT 'e', h.hid, h.dev, h.det, floor(h.s / 300.0)::INT // 3, ceil(least(ceil(h.f / 300.0), w.nb5) / 3.0)::INT
FROM hold h JOIN {{ p }}kd k USING (dev, det) JOIN win w ON w.dev = h.dev WHERE h.dur >= 900
UNION ALL
SELECT 'd', dev * 128 + det, dev, det, drop_b0 // 3, ceil(drop_b1 / 3.0)::INT FROM {{ p }}drx
WHERE n_on > 0 AND drop_lam >= 30 AND drop_b1 > drop_b0
-- @step {{ p }}rc
SELECT r.kind, r.id, c.c FROM {{ p }}rq r JOIN {{ p }}pc c ON c.dev = r.dev AND c.det = r.det
WHERE r.kind <> 'e' OR NOT EXISTS (
  SELECT 1 FROM hold e JOIN hold o ON o.dev = e.dev AND o.det = c.c AND o.dur >= 900 AND o.s < e.f AND o.f > e.s
  WHERE e.hid = r.id)
-- @step {{ p }}pxy
SELECT r.kind, r.id, c.c, x.n15 AS x, y.n15 AS y, x.k + 1 AS w
FROM {{ p }}rq r JOIN {{ p }}rc c ON c.kind = r.kind AND c.id = r.id
JOIN a15 x ON x.dev = r.dev AND x.det = r.det AND x.c15 AND NOT (x.k >= r.xa AND x.k < r.xb)
JOIN a15 y ON y.dev = r.dev AND y.det = c.c AND y.k = x.k
{# the partner: best 15-min correlation (ties: its own counts, never its channel order), kept from 0.7 #}
-- @step {{ p }}part mat
SELECT kind, id, p FROM (
  SELECT kind, id, first(c ORDER BY round(cr, 12) DESC, ys DESC, yw DESC, c) AS p,
         first(cr ORDER BY round(cr, 12) DESC, ys DESC, yw DESC, c) AS pcr
  FROM (SELECT kind, id, c, corr(y, x) AS cr, sum(y) AS ys, sum(y * w) AS yw, count(*) AS m,
               min(x) < max(x) AS xv, min(y) < max(y) AS yv FROM {{ p }}pxy GROUP BY kind, id, c)
  WHERE m >= 16 AND xv AND yv GROUP BY kind, id) WHERE pcr >= 0.7
{# stuck-on holds >= 15 min and silent runs: share after / before (6 h each side, >= 3 h of data) #}
-- @step {{ p }}rr
SELECT 'e' AS kind, h.hid AS id, h.dev, h.det, floor(h.s / 300.0)::INT AS a, least(ceil(h.f / 300.0)::INT, w.nb5) AS b
FROM hold h JOIN {{ p }}kd k USING (dev, det) JOIN win w ON w.dev = h.dev WHERE h.dur >= 900 AND NOT h.open_end
UNION ALL
SELECT 'd', dev * 128 + det, dev, det, drop_b0, drop_b1 FROM {{ p }}drx
WHERE n_on > 0 AND drop_lam >= 30 AND drop_b1 > drop_b0 AND NOT drop_to_end
-- @step {{ p }}rr2 mat
SELECT r.*, greatest(r.a - 72, 0) AS lo0, least(r.b + 72, w.nb5) AS hi1, pa.p,
       (r.a >= 36 AND w.nb5 - r.b >= 36) AS can
FROM {{ p }}rr r JOIN win w ON w.dev = r.dev LEFT JOIN {{ p }}part pa ON pa.kind = r.kind AND pa.id = r.id
-- @step {{ p }}rec mat
SELECT kind, id, CASE WHEN can AND c0 >= 36 AND c1 >= 36 THEN ((x1 + 0.5) / greatest(s1, 1.0)) /
                      ((x0 + 0.5) / greatest(s0, 1.0)) END AS rec
FROM (
  SELECT r.kind, r.id, any_value(r.can) AS can,
         count(*) FILTER (mm.own AND g.cov AND g.b < r.a) AS c0, count(*) FILTER (mm.own AND g.cov AND g.b >= r.b) AS c1,
         coalesce(sum(g.n) FILTER (mm.own AND g.b < r.a), 0) AS x0, coalesce(sum(g.n) FILTER (mm.own AND g.b >= r.b), 0) AS x1,
         coalesce(sum(g.n) FILTER (NOT mm.own AND g.b < r.a), 0) AS s0,
         coalesce(sum(g.n) FILTER (NOT mm.own AND g.b >= r.b), 0) AS s1
  FROM {{ p }}rr2 r
  JOIN (SELECT r.kind, r.id, r.det AS m, true AS own FROM {{ p }}rr2 r
        UNION ALL SELECT r.kind, r.id, r.p, false FROM {{ p }}rr2 r WHERE r.p IS NOT NULL
        UNION ALL SELECT r.kind, r.id, m.r, false FROM {{ p }}rr2 r JOIN {{ p }}rm m ON m.dev = r.dev AND m.det = r.det
        WHERE r.p IS NULL) mm ON mm.kind = r.kind AND mm.id = r.id
  JOIN g5 g ON g.dev = r.dev AND g.det = mm.m AND ((g.b >= r.lo0 AND g.b < r.a) OR (g.b >= r.b AND g.b < r.hi1))
  GROUP BY r.kind, r.id)
{# stuck-on per detector: the longest hold >= 15 min (first in time), its shared count and recovery #}
-- @step {{ p }}ep
SELECT h.dev, h.det, max(h.dur) AS ep_dur, first(h.s ORDER BY h.dur DESC, h.s) AS ep_t0,
       first(h.f ORDER BY h.dur DESC, h.s) AS ep_t1, first(c.co ORDER BY h.dur DESC, h.s) AS ep_co,
       first(r.rec ORDER BY h.dur DESC, h.s) AS ep_rec
FROM hold h JOIN {{ p }}kd k USING (dev, det) JOIN hold_co c ON c.hid = h.hid
LEFT JOIN {{ p }}rec r ON r.kind = 'e' AND r.id = h.hid WHERE h.dur >= 900 GROUP BY h.dev, h.det
{# silent run inside a stuck-on hold (>= half of it): not a silence #}
-- @step {{ p }}dis
SELECT d.dev, d.det, coalesce(sum(greatest(least(h.f, d.drop_b1 * 300.0) - greatest(h.s, d.drop_b0 * 300.0), 0.0))
         FILTER (h.hid IS NOT NULL), 0)
       >= 0.5 * (d.drop_b1 - d.drop_b0) * 300.0 AS drop_in_stuck
FROM {{ p }}drx d LEFT JOIN hold h ON h.dev = d.dev AND h.det = d.det AND h.dur >= 900
WHERE d.n_on > 0 AND d.drop_lam >= 30 AND d.drop_b1 > d.drop_b0 GROUP BY d.dev, d.det, d.drop_b0, d.drop_b1
{# night count (21-05) against the detector it tracks, else its phase, else the rest of the signal; clean_yardsticks
   (note 154t): the night yardsticks without the detectors that do not look like traffic (phase mates while >= 2 are
   left, else the rest of the signal without them); a partner that does not look like traffic is left out #}
{% if opt['clean_yardsticks'] %}
{% set kd, rm = p ~ 'nkd', p ~ 'nrm' %}
{% set pn = '(SELECT * FROM ' ~ p ~ 'part x WHERE NOT EXISTS (SELECT 1 FROM ntraf t WHERE t.dev = x.id // 128 AND t.det = x.p))' %}
-- @step {{ p }}nkd mat
SELECT a.dev, a.det, a.ph, coalesce(r.nsame, 0) >= 2 AS isph
FROM {{ p }}ph a LEFT JOIN (SELECT dev, det, count(*) FILTER (same) AS nsame FROM {{ p }}rf f
                        WHERE NOT EXISTS (SELECT 1 FROM ntraf t WHERE t.dev = f.dev AND t.det = f.r) GROUP BY dev, det) r
USING (dev, det)
-- @step {{ p }}nrm mat
SELECT f.dev, f.det, f.r FROM {{ p }}rf f JOIN {{ p }}nkd k USING (dev, det)
WHERE k.isph AND f.same AND NOT EXISTS (SELECT 1 FROM ntraf t WHERE t.dev = f.dev AND t.det = f.r)
UNION ALL SELECT k.dev, k.det, d.det FROM {{ p }}nkd k JOIN dets d ON d.dev = k.dev AND d.det <> k.det
WHERE NOT k.isph AND NOT EXISTS (SELECT 1 FROM ntraf t WHERE t.dev = d.dev AND t.det = d.det)
{% else %}
{% set kd, rm, pn = p ~ 'kd', p ~ 'rm', p ~ 'part' %}
{% endif %}
-- @step {{ p }}rsum
SELECT m.dev, m.det, sum(x.dy) AS rd, sum(x.nt) AS rn FROM {{ rm }} m JOIN dstat x ON x.dev = m.dev AND x.det = m.r
GROUP BY m.dev, m.det
-- @step {{ p }}nt
SELECT dev, det, nt AS night_n, ex AS night_exp, (nt + 0.5) / (ex + 0.5) AS night_ratio, who FROM (
  SELECT k.dev, k.det, s.nt, s.dy / greatest(coalesce(ps.dy, r.rd, 0), 1) * coalesce(ps.nt, r.rn, 0) AS ex,
         CASE WHEN pa.p IS NOT NULL THEN 'd' || pa.p || ', which it tracks'
              WHEN k.isph THEN 'the other detectors on its phase' ELSE 'the rest of the signal' END AS who
  FROM {{ kd }} k JOIN dstat s USING (dev, det) JOIN sig0 g ON g.dev = k.dev
  LEFT JOIN {{ pn }} pa ON pa.kind = 'n' AND pa.id = k.dev * 128 + k.det
  LEFT JOIN dstat ps ON ps.dev = k.dev AND ps.det = pa.p LEFT JOIN {{ p }}rsum r ON r.dev = k.dev AND r.det = k.det
  WHERE g.ok_night AND s.dy >= 20) WHERE ex >= 30
{# the four rules -> pass status; no_recovery_cap (note 154g): without the "came back" caps (_u): the detector's own
   rating; the capped status stays the yardstick's #}
{% set sd = lib.sc1('d.drop_lam', '30.0', '100.0') %}
{% set ss = lib.sc1('coalesce(e.ep_dur, 0)', '900.0', '3600.0') %}
{% set nc = opt['no_recovery_cap'] %}
-- @step {{ p }}st mat
SELECT *, {% if nc %}CASE WHEN n_on < 20 AND s_stuck_u = 0 THEN 'not_enough_data' WHEN hs_u < 0.25 THEN 'bad' WHEN hs_u < 0.70 THEN 'suspect'
               ELSE 'ok' END AS status_u, {% endif %}CASE WHEN hs < 0.25 THEN 'bad' WHEN hs < 0.70 THEN 'suspect' WHEN n_on < 20 THEN 'not_enough_data'
               ELSE 'ok' END AS status
FROM (SELECT *, (1 - coalesce(s_dropout, 0)) * (1 - s_stuck) * (1 - coalesce(s_night_drop, 0))
                * (1 - coalesce(s_level, 0)) AS hs{% if nc %}, (1 - coalesce(s_dropout_u, 0)) * (1 - s_stuck_u) * (1 - coalesce(s_night_drop, 0))
                * (1 - coalesce(s_level, 0)) AS hs_u{% endif %}

  FROM (SELECT d.dev, d.det, d.n_on, d.drop_lam, d.drop_b0, d.drop_b1, d.drop_to_end,
          coalesce(i.drop_in_stuck, false) AS drop_in_stuck, rd.rec AS drop_rec, e.ep_dur, e.ep_t0, e.ep_t1, e.ep_co,
          e.ep_rec, n.night_ratio, n.night_n, n.night_exp, n.who AS night_ref, l.level_ratio,
          coalesce(l.level_llr, 0.0) AS level_llr, l.level_b,
          CASE WHEN d.n_on > 0 THEN (
            CASE WHEN coalesce(i.drop_in_stuck, false) THEN 0.0
                 WHEN {{ sd }} > 0 AND NOT d.drop_to_end AND rd.rec BETWEEN 0.5 AND 2.0 THEN least({{ sd }}, 0.35)
                 ELSE {{ sd }} END) END AS s_dropout,
          CASE WHEN coalesce(e.ep_dur, 0) >= 900 AND (e.ep_co >= 3 OR e.ep_rec BETWEEN 0.5 AND 2.0)
               THEN least({{ ss }}, 0.35) ELSE {{ ss }} END AS s_stuck,
          {% if nc %}CASE WHEN d.n_on > 0 THEN CASE WHEN coalesce(i.drop_in_stuck, false) THEN 0.0
                  ELSE {{ sd }} END END
            AS s_dropout_u,
          CASE WHEN coalesce(e.ep_dur, 0) >= 900 AND (e.ep_co >= 3 OR (e.ep_rec BETWEEN 0.5 AND 2.0 AND e.ep_dur < 3600))
               THEN least({{ ss }}, 0.35) ELSE {{ ss }} END AS s_stuck_u,
          {% endif %}

          CASE WHEN n.night_ratio IS NOT NULL THEN
               {{ lib.sc1('-ln(greatest(n.night_ratio, 1e-6))', '-ln(0.19)', '-ln(0.063)') }} END AS s_night_drop,
          CASE WHEN l.level_ratio IS NOT NULL THEN CASE WHEN l.level_llr <= 50 THEN 0.0
               ELSE {{ lib.sc1('-ln(l.level_ratio)', '-ln(0.15)', '-ln(0.05)') }} END END AS s_level
        FROM {{ p }}drx d LEFT JOIN {{ p }}dis i USING (dev, det)
        LEFT JOIN {{ p }}rec rd ON rd.kind = 'd' AND rd.id = d.dev * 128 + d.det
        LEFT JOIN {{ p }}ep e USING (dev, det) LEFT JOIN {{ p }}nt n USING (dev, det) LEFT JOIN lvlshift l USING (dev, det)))
{% endmacro %}
-- @step p1ph mat
SELECT d.dev, d.det, c.ph FROM dets d LEFT JOIN dcl c USING (dev, det)
{% if opt['clean_yardsticks'] %}
{# note 154t (option clean_yardsticks): detectors that do not look like traffic (CLEAN_YARD) -> ntm (the levels, every
   detector), ntraf (dev, det).  From the 5-min counts only, so it is known before any check that uses a yardstick. #}
-- @step ntm mat
WITH h AS (SELECT dev, count(*) FILTER (cov AND hr >= 7 AND hr < 19) / 12.0 AS hd,
                  count(*) FILTER (cov AND hr >= 1 AND hr < 5) / 12.0 AS hn FROM bins5 GROUP BY dev),
o AS (SELECT g.dev, g.det, coalesce(sum(g.n) FILTER (g.cov AND g.hr >= 7 AND g.hr < 19), 0) AS d,
             coalesce(sum(g.n) FILTER (g.cov AND g.hr >= 1 AND g.hr < 5), 0) AS n FROM g5 g GROUP BY g.dev, g.det),
r AS (SELECT a.dev, a.det, coalesce(sum(b.d) FILTER (a.ph = b.ph), 0) AS pd, coalesce(sum(b.n) FILTER (a.ph = b.ph), 0) AS pn,
             sum(b.d) AS sd, sum(b.n) AS sn
      FROM (SELECT o.*, p.ph FROM o JOIN p1ph p USING (dev, det)) a
      JOIN (SELECT o.*, p.ph FROM o JOIN p1ph p USING (dev, det)) b ON b.dev = a.dev AND b.det <> a.det
      GROUP BY a.dev, a.det)
SELECT o.dev, o.det, h.hd, h.hn, o.d / h.hd AS own_d, o.n / h.hn AS own_n, x.ph_rest,
       CASE WHEN x.ph_rest THEN r.pd ELSE r.sd END / h.hd AS rest_d, CASE WHEN x.ph_rest THEN r.pn ELSE r.sn END / h.hn AS rest_n
FROM o JOIN h USING (dev) JOIN r USING (dev, det)
CROSS JOIN LATERAL (SELECT r.pd / greatest(h.hd, 1e-9) >= {{ CY['rest_min']|dbl }} AS ph_rest) x
WHERE h.hd >= 2 AND h.hn >= 2
-- @step ntraf mat
SELECT dev, det FROM ntm WHERE own_d >= {{ CY['own_min']|dbl }} AND rest_d >= {{ CY['rest_min']|dbl }}
  AND own_n >= {{ CY['own']|dbl }} * own_d AND rest_n <= {{ CY['rest']|dbl }} * rest_d
{% endif %}
{{ stage1('p1') }}
-- @step p2sig mat
SELECT DISTINCT s.dev FROM p1st s WHERE s.status = 'bad'
  AND s.dev IN (SELECT dev FROM dcl WHERE ph IS NOT NULL AND NOT isnan(ph))
-- @step p2ph mat
SELECT p.dev, p.det, CASE WHEN s.status = 'bad' THEN NULL ELSE p.ph END AS ph
FROM p1ph p JOIN p1st s USING (dev, det) WHERE p.dev IN (SELECT dev FROM p2sig)
{{ stage1('p2') }}
-- @step stage1 mat
SELECT x.*, a.status AS status1, a.hs AS score1 FROM (
  SELECT * FROM p1st WHERE dev NOT IN (SELECT dev FROM p2sig) OR status = 'bad'
  UNION ALL SELECT b.* FROM p2st b JOIN p1st a USING (dev, det) WHERE a.status <> 'bad') x
JOIN p1st a USING (dev, det)
{% if opt['all_silences'] %}
-- @step silall mat
SELECT a.* FROM p1dra a JOIN p1st s USING (dev, det) WHERE a.dev NOT IN (SELECT dev FROM p2sig) OR s.status = 'bad'
UNION ALL SELECT b.* FROM p2dra b JOIN p1st s USING (dev, det) WHERE s.status <> 'bad'
{% endif %}
