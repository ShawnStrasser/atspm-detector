{# Hourly GEH against the yardstick (health_duckdb._hourly / _night_clean: health_counts.hourly_geh): count drops and
   misses at night -> geh. #}
{% import "_lib.sql" as lib with context %}
{% macro share(mask) %}CASE WHEN count(*) FILTER ({{ mask }} AND R >= 30) >= 4 THEN median(M / R) FILTER ({{ mask }} AND R >= 30) ELSE coalesce(sum(M) FILTER ({{ mask }}), 0) / greatest(coalesce(sum(R) FILTER ({{ mask }}), 0), 1) END{% endmacro %}
{% set day, nt = 'ch AND hod >= 7 AND hod < 19', 'ch AND (hod >= 21 OR hod < 5)' %}
-- @step hix mat
SELECT dev, hid, hid - min(hid) OVER (PARTITION BY dev) AS j, hid % 24 AS hod, nb_ = 12 AND nc = 12 AS ch,
       count(*) OVER (PARTITION BY dev) AS H
FROM (SELECT b.dev, (w.s_ns + b.b * {{ NS5 }}) // {{ NSH }} AS hid, count(*) AS nb_, count(*) FILTER (b.cov) AS nc
      FROM bins5 b JOIN win w USING (dev) WHERE w.hours >= {{ LJ['min_hours_counts']|dbl }} GROUP BY b.dev, hid)
-- @step mh mat
SELECT g.dev, g.det, (w.s_ns + g.b * {{ NS5 }}) // {{ NSH }} AS hid, sum(g.n) AS M, sum(g.occ) AS O
FROM g5 g JOIN win w USING (dev) WHERE w.hours >= {{ LJ['min_hours_counts']|dbl }} GROUP BY g.dev, g.det, hid
{# yardstick: the detector it tracks (partner), else its phase mates (>= 2), else the rest of the signal #}
{% if opt['clean_yardsticks'] == 'all' %}
{# note 154t ("all"): a detector that looks like traffic gets the yardstick without the ones that do not (its partner
   too); one that does not keeps its own (two of them tracking each other stay each other's yardstick) #}
-- @step yk mat
SELECT k.dev, k.det, pa.p, k.isph, false AS cl FROM p1kd k LEFT JOIN p1part pa ON pa.kind = 'n' AND pa.id = k.dev * 128 + k.det
WHERE EXISTS (SELECT 1 FROM ntraf t WHERE t.dev = k.dev AND t.det = k.det)
UNION ALL
SELECT k.dev, k.det, pa.p, k.isph, true FROM p1nkd k
LEFT JOIN (SELECT * FROM p1part x WHERE NOT EXISTS (SELECT 1 FROM ntraf t WHERE t.dev = x.id // 128 AND t.det = x.p)) pa
  ON pa.kind = 'n' AND pa.id = k.dev * 128 + k.det
WHERE NOT EXISTS (SELECT 1 FROM ntraf t WHERE t.dev = k.dev AND t.det = k.det)
-- @step gh mat
SELECT m.dev, m.det, x.j, x.hod, x.ch, x.H, m.M, m.O, y.p IS NOT NULL OR y.isph AS has_m,
       CASE WHEN y.p IS NULL AND NOT y.isph THEN CASE WHEN y.cl THEN s.SMc ELSE s.SM END - m.M ELSE coalesce(r.RM, 0) END
       AS R, coalesce(r.RO, 0) AS Om
FROM mh m JOIN hix x USING (dev, hid) JOIN yk y USING (dev, det)
JOIN (SELECT m.dev, m.hid, sum(m.M) AS SM, sum(m.M) FILTER (t.det IS NULL) AS SMc FROM mh m LEFT JOIN ntraf t USING (dev, det)
      GROUP BY m.dev, m.hid) s USING (dev, hid)
LEFT JOIN (SELECT a.dev, a.det, b.hid, sum(b.M) AS RM, sum(b.O) AS RO
           FROM (SELECT dev, det, p AS r FROM yk WHERE p IS NOT NULL
                 UNION ALL SELECT f.dev, f.det, f.r FROM p1pref f JOIN yk y USING (dev, det) WHERE y.p IS NULL AND NOT y.cl
                 UNION ALL SELECT f.dev, f.det, f.r FROM p1nrm f JOIN yk y USING (dev, det)
                 WHERE y.p IS NULL AND y.cl AND y.isph) a
           JOIN mh b ON b.dev = a.dev AND b.det = a.r GROUP BY a.dev, a.det, b.hid) r USING (dev, det, hid)
{% else %}
-- @step yk mat
SELECT k.dev, k.det, pa.p, k.isph FROM p1kd k LEFT JOIN p1part pa ON pa.kind = 'n' AND pa.id = k.dev * 128 + k.det
-- @step gh mat
SELECT m.dev, m.det, x.j, x.hod, x.ch, x.H, m.M, m.O, y.p IS NOT NULL OR y.isph AS has_m,
       CASE WHEN y.p IS NULL AND NOT y.isph THEN s.SM - m.M ELSE coalesce(r.RM, 0) END AS R, coalesce(r.RO, 0) AS Om
FROM mh m JOIN hix x USING (dev, hid) JOIN yk y USING (dev, det)
JOIN (SELECT dev, hid, sum(M) AS SM FROM mh GROUP BY dev, hid) s USING (dev, hid)
LEFT JOIN (SELECT a.dev, a.det, b.hid, sum(b.M) AS RM, sum(b.O) AS RO
           FROM (SELECT dev, det, p AS r FROM yk WHERE p IS NOT NULL
                 UNION ALL SELECT f.dev, f.det, f.r FROM p1pref f JOIN yk y USING (dev, det) WHERE y.p IS NULL) a
           JOIN mh b ON b.dev = a.dev AND b.det = a.r GROUP BY a.dev, a.det, b.hid) r USING (dev, det, hid)
{% endif %}
-- @step gh1 mat
SELECT dev, det, coalesce(sum(M) FILTER ({{ day }}), 0) >= 20 AND coalesce(sum(R) FILTER ({{ day }}), 0) > 0 AS elig,
       {{ share(day) }} AS s, count(*) FILTER ({{ nt }}) AS nn
FROM gh GROUP BY dev, det
{# note 154 (drop_end): running counts and expected counts, for the recovery test of a count drop #}
{% set sg = lib.sgeh('g.M', 'a.s * g.R') %}
-- @step gh2 mat
SELECT g.*, a.s, a.elig, CASE WHEN g.ch THEN {{ sg }} ELSE 0.0 END AS sg,
       sum(CASE WHEN g.ch THEN {{ sg }} ELSE 0.0 END) OVER w AS cs, sum(g.ch::INT) OVER w AS cc,
       sum(CASE WHEN g.ch THEN {{ sg }} ELSE 0.0 END) OVER (PARTITION BY g.dev, g.det) AS cs_t,
       sum(g.ch::INT) OVER (PARTITION BY g.dev, g.det) AS cc_t{% if opt['drop_end'] %},
       sum(CASE WHEN g.ch AND g.R >= 30 THEN g.M ELSE 0.0 END) OVER w AS cm, sum(CASE WHEN g.ch AND g.R >= 30 THEN g.R ELSE 0.0 END) OVER w AS cr,
       sum(CASE WHEN g.ch AND g.R >= 30 THEN g.M ELSE 0.0 END) OVER (PARTITION BY g.dev, g.det) AS cm_t,
       sum(CASE WHEN g.ch AND g.R >= 30 THEN g.R ELSE 0.0 END) OVER (PARTITION BY g.dev, g.det) AS cr_t{% endif %}

FROM gh g JOIN gh1 a USING (dev, det) WINDOW w AS (PARTITION BY g.dev, g.det ORDER BY g.j ROWS UNBOUNDED PRECEDING)
{# count drops: the hour with the largest step up in under-counting (>= 3 covered hours each side) #}
-- @step ghk
SELECT a.dev, a.det, coalesce(k.k, 1) AS k, a.elig AND k.k IS NOT NULL AS has_k FROM gh1 a LEFT JOIN (
  SELECT dev, det, first(j + 1 ORDER BY stat DESC, j) AS k FROM (
    SELECT dev, det, j, ((cs_t - cs) / (cc_t - cc) - cs / cc) * sqrt(cc * (cc_t - cc) / cc_t) AS stat
    FROM gh2 WHERE j <= H - 2 AND cc >= 3 AND cc_t - cc >= 3) GROUP BY dev, det) k USING (dev, det)
{% if opt['drop_end'] %}
{# note 154: the recovery -- the hour after the drop with the largest step back down in under-counting (>= 3 covered
   hours each side), kept when the hours after it under-count at most 0.35 x the dropped hours and below half the
   suspect GEH, and count at least 60 % of what its share BEFORE the drop gives there (>= 50 expected: quiet night
   hours cannot show a recovery); the drop then ends there, not at the window end #}
-- @step ghk2
SELECT dev, det, first(j + 1 ORDER BY stat DESC, j) AS k2 FROM (
  SELECT g.dev, g.det, g.j, ((g.cs - b.cs) / (g.cc - b.cc) - (g.cs_t - g.cs) / (g.cc_t - g.cc))
         * sqrt((g.cc - b.cc) * (g.cc_t - g.cc) / (g.cc_t - b.cc)) AS stat,
         (g.cs - b.cs) / (g.cc - b.cc) AS m_in, (g.cs_t - g.cs) / (g.cc_t - g.cc) AS m_rec
  FROM gh2 g JOIN ghk k USING (dev, det) JOIN gh2 b ON b.dev = g.dev AND b.det = g.det AND b.j = k.k - 1
  WHERE k.has_k AND g.j >= k.k AND g.j <= g.H - 2 AND g.cc - b.cc >= 3 AND g.cc_t - g.cc >= 3 AND b.cr > 0
    AND (g.cr_t - g.cr) * b.cm / b.cr >= 50 AND g.cm_t - g.cm >= 0.6 * (g.cr_t - g.cr) * b.cm / b.cr)
WHERE m_in > 0 AND m_rec <= 0.35 * m_in AND m_rec < {{ D['drop_geh_half']|dbl }} GROUP BY dev, det
-- @step gh3
SELECT g.dev, g.det, k.k, k.has_k, r.k2, g.j < k.k AND g.ch AS pre,
       g.j >= k.k AND (r.k2 IS NULL OR g.j < r.k2) AND g.ch AS post, g.j >= k.k AND g.ch AS post0, g.M, g.R, g.O,
       g.Om, g.ch, g.sg, g.s, g.hod, g.H, g.has_m, g.elig
FROM gh2 g JOIN ghk k USING (dev, det) LEFT JOIN ghk2 r USING (dev, det)
{% else %}
-- @step gh3
SELECT g.dev, g.det, k.k, k.has_k, NULL::BIGINT AS k2, g.j < k.k AND g.ch AS pre, g.j >= k.k AND g.ch AS post, g.M, g.R,
       g.O, g.Om, g.ch, g.sg, g.s, g.hod, g.H, g.has_m, g.elig
FROM gh2 g JOIN ghk k USING (dev, det)
{% endif %}
-- @step gh4
SELECT dev, det, any_value(k) AS k, any_value(has_k) AS has_k, any_value(k2) AS k2, any_value(H) AS H,
       any_value(has_m) AS has_m,
       any_value(elig) AS elig, any_value(s) AS s, {{ share('pre') }} AS s2,
       count(*) FILTER (pre AND R >= 30) >= 3 AS re, greatest(count(*) FILTER (pre), 1) AS npre,
       greatest(count(*) FILTER (post), 1) AS npost, coalesce(sum(M) FILTER (pre), 0) AS Mpre,
       coalesce(sum(M) FILTER (post), 0) AS Mpost, coalesce(sum(R) FILTER (post), 0) AS Rpost,
       coalesce(sum(O) FILTER (pre), 0) AS Opre, coalesce(sum(O) FILTER (post), 0) AS Opost,
       coalesce(sum(Om) FILTER (pre), 0) AS Ompre, coalesce(sum(Om) FILTER (post), 0) AS Ompost,
       count(*) FILTER ({{ nt }}) AS nn, coalesce(sum(M) FILTER ({{ nt }}), 0) AS Mn, coalesce(sum(s * R) FILTER ({{ nt }}), 0) AS Cn
FROM gh3 GROUP BY dev, det
{% set an = lib.sgeh('a.Mn / greatest(a.nn, 1)', 'a.Cn / greatest(a.nn, 1)') %}
{% if opt['clean_yardsticks'] is sameas true %}{# "all": gh is clean already #}
{# note 154t (option clean_yardsticks): the hourly night GEH against a yardstick without the detectors that do not look
   like traffic (ntraf) -> ghnn (Mn, Cn, nn, elig), only for the detectors whose yardstick held one (the others keep
   theirs exactly) #}
-- @step ykn mat
SELECT k.dev, k.det, pa.p, k.isph FROM p1nkd k
LEFT JOIN (SELECT * FROM p1part x WHERE NOT EXISTS (SELECT 1 FROM ntraf t WHERE t.dev = x.id // 128 AND t.det = x.p)) pa
  ON pa.kind = 'n' AND pa.id = k.dev * 128 + k.det
{# the detectors whose night yardstick (partner, else phase mates, else the rest of the signal) held one #}
-- @step ykaff mat
SELECT DISTINCT y.dev, y.det FROM yk y JOIN ntraf t ON t.dev = y.dev AND t.det <> y.det
WHERE t.det = y.p OR (y.p IS NULL AND NOT y.isph)
   OR (y.p IS NULL AND y.isph AND EXISTS (SELECT 1 FROM p1pref f WHERE f.dev = y.dev AND f.det = y.det AND f.r = t.det))
-- @step ghn mat
SELECT m.dev, m.det, x.hod, x.ch, m.M,
       CASE WHEN y.p IS NULL AND NOT y.isph THEN s.SM - CASE WHEN t.det IS NULL THEN m.M ELSE 0 END
            ELSE coalesce(r.RM, 0) END AS R
FROM mh m JOIN ykaff USING (dev, det) JOIN hix x USING (dev, hid) JOIN ykn y USING (dev, det)
LEFT JOIN ntraf t USING (dev, det)
JOIN (SELECT m.dev, m.hid, sum(m.M) FILTER (t.det IS NULL) AS SM FROM mh m LEFT JOIN ntraf t USING (dev, det)
      GROUP BY m.dev, m.hid) s USING (dev, hid)
LEFT JOIN (SELECT a.dev, a.det, b.hid, sum(b.M) AS RM
           FROM (SELECT dev, det, p AS r FROM ykn WHERE p IS NOT NULL
                 UNION ALL SELECT f.dev, f.det, f.r FROM p1nrm f JOIN ykn y USING (dev, det) WHERE y.p IS NULL AND y.isph) a
           JOIN mh b ON b.dev = a.dev AND b.det = a.r GROUP BY a.dev, a.det, b.hid) r USING (dev, det, hid)
-- @step ghnn
SELECT g.dev, g.det, a.elig, a.nn, coalesce(sum(g.M) FILTER ({{ nt }}), 0) AS Mn, coalesce(sum(a.s * g.R) FILTER ({{ nt }}), 0) AS Cn
FROM ghn g JOIN (SELECT dev, det, coalesce(sum(M) FILTER ({{ day }}), 0) >= 20 AND coalesce(sum(R) FILTER ({{ day }}), 0) > 0
                        AS elig, {{ share(day) }} AS s, count(*) FILTER ({{ nt }}) AS nn FROM ghn GROUP BY dev, det) a
  USING (dev, det)
GROUP BY g.dev, g.det, a.elig, a.nn
{% set ngeh %}CASE WHEN n.det IS NOT NULL THEN CASE WHEN n.elig AND n.nn >= 4 THEN {{ lib.sgeh('n.Mn / greatest(n.nn, 1)', 'n.Cn / greatest(n.nn, 1)') }} END ELSE CASE WHEN a.elig AND a.nn >= 4 THEN {{ an }} END END AS geh_night,
       CASE WHEN n.det IS NOT NULL THEN n.Mn ELSE a.Mn END AS geh_night_n0, CASE WHEN n.det IS NOT NULL THEN n.Cn ELSE a.Cn END AS geh_night_exp0,{% endset %}
{% set nj = '\nLEFT JOIN ghnn n USING (dev, det)' %}
{% else %}
{% set ngeh %}CASE WHEN a.elig AND a.nn >= 4 THEN {{ an }} END AS geh_night,
       a.Mn AS geh_night_n0, a.Cn AS geh_night_exp0,{% endset %}
{% set nj = '' %}
{% endif %}
-- @step geh
SELECT a.dev, a.det, a.elig AS geh_ok,
       {{ ngeh }}
       CASE WHEN a.has_k AND b.med > 0 THEN b.med END AS geh_drop,
       CASE WHEN a.has_k THEN make_timestamp(((x.hid0 + least(a.k, a.H - 1)) * {{ NSH }}) // 1000) END AS drop_t,
       CASE WHEN a.has_k AND a.k2 IS NOT NULL THEN make_timestamp(((x.hid0 + a.k2) * {{ NSH }}) // 1000) END AS drop_t2,
       a.Opost / a.npost / 3600.0 AS occ_after,
       a.Mpre / a.npre AS own_before, a.Mpost / a.npost AS own_after,
       CASE WHEN a.re THEN a.s2 ELSE a.s END * a.Rpost / a.npost AS exp_after,
       CASE WHEN a.has_k THEN (a.Opost / a.npost) / greatest(a.Opre / a.npre, 1e-6) END AS occ_own,
       CASE WHEN a.has_k AND a.has_m THEN (a.Ompost / a.npost) / greatest(a.Ompre / a.npre, 1e-6) END AS occ_mates
FROM gh4 a JOIN (SELECT dev, min(hid) AS hid0 FROM hix GROUP BY dev) x USING (dev){{ nj }}
-- note 154h: the severity (geh_drop) over every hour after the drop, as 1.2.0; its end bounds only the period / text
LEFT JOIN (SELECT g.dev, g.det, median(CASE WHEN a.re THEN CASE WHEN g.ch THEN {{ lib.sgeh('g.M', 'a.s2 * g.R') }} ELSE 0.0 END
                                           ELSE g.sg END) FILTER ({{ 'g.post0' if opt['drop_end'] else 'g.post' }}) AS med
           FROM gh3 g JOIN gh4 a USING (dev, det) GROUP BY g.dev, g.det) b USING (dev, det)
