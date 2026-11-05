{# note 154n (options mate_drop / count_rise, health_duckdb._mate_hours): hour-by-hour count drop / rise against each
   phase mate (complete hours; Bike detectors neither judged nor used) -> mhrun (every run of low / high hours) and
   mhbest (per detector its strongest drop and rise run). #}
-- @step mhg mat
SELECT m.dev, m.det, m.hid, x.hod, m.hid // 24 AS dy, m.M, m.O, z.ph
FROM mh m JOIN hix x USING (dev, hid) JOIN dz z USING (dev, det)
WHERE x.ch AND z.ph IS NOT NULL AND coalesce(z.fn, '') <> 'Bike'
{# note 154t ("all"): a mate that does not look like traffic is left out (unless the detector is one itself) #}
-- @step mhp mat
SELECT a.dev, a.det, b.det AS mate, a.hid, a.hod, a.dy, a.M, b.M AS Mm
FROM mhg a JOIN mhg b ON b.dev = a.dev AND b.hid = a.hid AND b.ph = a.ph AND b.det <> a.det{% if opt['clean_yardsticks'] == 'all' %}

WHERE NOT EXISTS (SELECT 1 FROM ntraf t WHERE t.dev = b.dev AND t.det = b.det)
   OR EXISTS (SELECT 1 FROM ntraf t WHERE t.dev = a.dev AND t.det = a.det){% endif %}

{# its share of each mate over the day (07-19) or the night hours of every day #}
-- @step mhqa mat
SELECT dev, det, mate, hod >= 7 AND hod < 19 AS dband, median(M / Mm) AS q_band
FROM mhp x WHERE Mm >= {{ M['min_mate']|dbl }} GROUP BY ALL
{# its share of each mate at the same time of day (+-1 h): >= 3 days there -> the median over all of them, else the
   earlier day (amb: a 2-day yardstick) #}
-- @step mhqh mat
SELECT k.dev, k.det, k.mate, k.dy, k.hod,
       CASE WHEN count(DISTINCT s.dy) >= 3 THEN median(s.q)
            WHEN count(DISTINCT s.dy) = 2 AND min(s.dy) < k.dy THEN median(s.q) FILTER (s.dy < k.dy) END AS q_hod,
       count(DISTINCT s.dy) < 3 AS amb
FROM (SELECT DISTINCT dev, det, mate, dy, hod FROM mhp) k
JOIN (SELECT dev, det, mate, dy, hod, M / Mm AS q FROM mhp x WHERE Mm >= {{ M['min_mate']|dbl }}) s
  ON s.dev = k.dev AND s.det = k.det AND s.mate = k.mate AND (abs(s.hod - k.hod) <= 1 OR abs(s.hod - k.hod) = 23)
GROUP BY k.dev, k.det, k.mate, k.dy, k.hod
{# its own count and time ON at the same time of day, by the same day rule #}
-- @step mhoh mat
SELECT k.dev, k.det, k.dy, k.hod,
       CASE WHEN count(DISTINCT s.dy) >= 3 THEN median(s.M)
            WHEN count(DISTINCT s.dy) = 2 AND min(s.dy) < k.dy THEN median(s.M) FILTER (s.dy < k.dy) END AS own_ref,
       CASE WHEN count(DISTINCT s.dy) >= 3 THEN median(s.O)
            WHEN count(DISTINCT s.dy) = 2 AND min(s.dy) < k.dy THEN median(s.O) FILTER (s.dy < k.dy) END AS occ_ref
FROM (SELECT DISTINCT dev, det, dy, hod FROM mhg) k
JOIN mhg s ON s.dev = k.dev AND s.det = k.det AND (abs(s.hod - k.hod) <= 1 OR abs(s.hod - k.hod) = 23)
GROUP BY k.dev, k.det, k.dy, k.hod
{% set lo, hi, zc = M['low']|dbl, M['high']|dbl, M['z']|dbl %}
{% if opt['mate_noise'] %}
{# note 154o: counting noise with a mild over-dispersion (network-wide: hour-to-hour scatter about the time-of-day
   yardstick is Poisson at low volume; the proportional part at high volume is the 0.5 x / 2 x ratio's job) #}
{% set ph = M['phi']|dbl %}
{% set zlo = ' AND (p.M - h.q_hod * p.Mm) / sqrt(' ~ ph ~ ' * h.q_hod * p.Mm + 1) <= -' ~ zc %}
{% set zhi = ' AND (p.M - h.q_hod * p.Mm) / sqrt(' ~ ph ~ ' * h.q_hod * p.Mm + 1) >= ' ~ zc %}
{% set zolo = ' AND (g.M - o.own_ref) / sqrt(' ~ ph ~ ' * o.own_ref + 1) <= -' ~ zc %}
{% set zohi = ' AND (g.M - o.own_ref) / sqrt(' ~ ph ~ ' * o.own_ref + 1) >= ' ~ zc %}
{% else %}
{% set zlo, zhi, zolo, zohi = '', '', '', '' %}
{% endif %}
-- @step mhv mat
SELECT p.dev, p.det, p.hid, count(*) AS nv, median(h.q_hod * p.Mm) AS E,
       sum(p.Mm) FILTER (p.M <= {{ lo }} * h.q_hod * p.Mm{{ zlo }}) / sum(p.Mm) AS wlo,
       count(*) FILTER (p.M >= {{ hi }} * h.q_hod * p.Mm AND (NOT h.amb OR p.M >= {{ hi }} * a.q_band * p.Mm)
                        AND p.M - h.q_hod * p.Mm >= {{ M['min_extra']|dbl }}{{ zhi }}) AS nhi
FROM mhp p JOIN mhqh h USING (dev, det, mate, dy, hod)
JOIN mhqa a ON a.dev = p.dev AND a.det = p.det AND a.mate = p.mate AND a.dband = (p.hod >= 7 AND p.hod < 19)
WHERE h.q_hod IS NOT NULL AND p.Mm >= 5 GROUP BY p.dev, p.det, p.hid
-- @step mhf mat
SELECT g.dev, g.det, g.hid, g.M, g.O, v.E, o.occ_ref,
       coalesce(v.nv >= 1 AND v.E >= {{ M['min_exp']|dbl }}, false) AS jl, coalesce(v.nv >= 1, false) AS jh,
       coalesce(v.nv >= 1 AND v.E >= {{ M['min_exp']|dbl }} AND v.wlo >= 0.5 AND g.M <= {{ M['own_low']|dbl }} * o.own_ref{{ zolo }},
                false) AS lo,
       coalesce(v.nv >= 1 AND v.nhi >= least(2, v.nv) AND v.nhi >= {{ M['high_share']|dbl }} * v.nv
                AND g.M >= {{ M['own_high']|dbl }} * o.own_ref + 10{{ zohi }}, false) AS hi
FROM mhg g LEFT JOIN mhv v USING (dev, det, hid) LEFT JOIN mhoh o USING (dev, det, dy, hod)
{# runs: hours not judged (too few expected) do not break a run #}
-- @step {{ 'mhrun0' if opt['rise_dead_days'] else 'mhrun' }} mat
SELECT * , sqrt(2 * (M - E) ^ 2 / greatest(M + E, 1e-9)) AS rgeh FROM (
  SELECT dev, det, 'drop' AS kind, min(hid) FILTER (lo) AS a, max(hid) FILTER (lo) + 1 AS z, count(*) FILTER (lo) AS hours,
         sum(M) FILTER (lo) AS M, sum(E) FILTER (lo) AS E, sum(O) FILTER (lo) AS O, sum(occ_ref) FILTER (lo) AS Oref
  FROM (SELECT *, sum((jl AND NOT lo)::INT) OVER (PARTITION BY dev, det ORDER BY hid) AS grp FROM mhf)
  GROUP BY dev, det, grp HAVING count(*) FILTER (lo) >= 1
  UNION ALL
  SELECT dev, det, 'rise', min(hid) FILTER (hi), max(hid) FILTER (hi) + 1, count(*) FILTER (hi),
         sum(M) FILTER (hi), sum(E) FILTER (hi), sum(O) FILTER (hi), sum(occ_ref) FILTER (hi)
  FROM (SELECT *, sum((jh AND NOT hi)::INT) OVER (PARTITION BY dev, det ORDER BY hid) AS grp FROM mhf)
  GROUP BY dev, det, grp HAVING count(*) FILTER (hi) >= 1)
{% if opt['rise_dead_days'] %}
{# note 154x: a day that is "too high" only because the detector nearly stopped on the window's other days (the
   time-of-day yardstick, a median over the days, is then the dead level) is not a rise -- the other days
   (<= 10 % of that day's 06-22 h count while its mates kept >= half their traffic) are the fault #}
-- @step mhday mat
SELECT g.dev, g.det, g.dy, g.Md, m.Mmd
FROM (SELECT dev, det, dy, sum(M) AS Md FROM mhg WHERE hod >= 6 AND hod < 22 GROUP BY ALL) g
JOIN (SELECT dev, det, dy, sum(Mm) AS Mmd FROM mhp WHERE hod >= 6 AND hod < 22 GROUP BY ALL) m USING (dev, det, dy)
-- @step mhrun mat
SELECT * FROM mhrun0 u
WHERE u.kind <> 'rise' OR NOT EXISTS (
  SELECT 1 FROM mhday r JOIN mhday d ON d.dev = r.dev AND d.det = r.det AND d.dy <> r.dy
  WHERE r.dev = u.dev AND r.det = u.det AND r.dy = u.a // 24
    AND d.Md <= 0.1 * r.Md AND d.Mmd >= 0.5 * r.Mmd AND r.Mmd > 0)
{% endif %}
{# the strongest run of each kind; a drop counts only when its time ON fell too (a queue over the zone holds it ON
   longer and counts fewer) or it nearly stopped counting #}
-- @step mhbest mat
SELECT d.dev, d.det, true AS mh_ok, r.hours AS md_h, r.rgeh AS md_geh, r.M / greatest(r.E, 1e-9) AS md_r, r.a AS md_a,
       r.z AS md_z, r.M / (r.z - r.a) AS md_own, r.E / (r.z - r.a) AS md_exp,
       u.hours AS mr_h, u.rgeh AS mr_geh, u.M / greatest(u.E, 1e-9) AS mr_r, u.a AS mr_a, u.z AS mr_z
FROM (SELECT DISTINCT dev, det FROM mhv) d
LEFT JOIN (SELECT * FROM mhrun WHERE kind = 'drop' AND hours >= {{ M['hours'] }}
             AND (O < {{ M['occ']|dbl }} * Oref OR M <= 0.1 * E)
           QUALIFY row_number() OVER (PARTITION BY dev, det ORDER BY rgeh DESC, a) = 1) r USING (dev, det)
LEFT JOIN (SELECT * FROM mhrun WHERE kind = 'rise' AND hours >= {{ M['hours'] }}
           QUALIFY row_number() OVER (PARTITION BY dev, det ORDER BY rgeh DESC, a) = 1) u USING (dev, det)
