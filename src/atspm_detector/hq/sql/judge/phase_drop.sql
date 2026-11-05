{# note 154b (option phase_drop, health_duckdb._phase_drop): a whole phase drops together -- its total count <= 0.2 x its
   usual share of the rest of the signal (the same clock hour on other days of the same type) for >= 2 hours (06-21),
   >= 200 vehicles expected, while the phase keeps getting greens (reference hours: the rest of the signal carried
   0.5-2 x what it carries now).  The phase-mate yardstick moves with it, so the per-detector checks cannot see it.  Each
   of its detectors whose own count in the run is <= 25 % of its share of the rest of the signal gets the run as its
   finding; lockstep: inputs of a phase logging the same hourly counts -> out (src + ph_drop, ph_lock, ph_k, ph_m,
   ph_obs, ph_exp, lk_n, lk_md). #}
{% import "adapters.sql" as ad with context %}
{% macro phase_drop(src, out) %}
-- @step pdm mat
SELECT m.dev, p.ph, m.hid, x.j, x.hod, x.ch, sum(m.M) FILTER (z.ph = p.ph) AS mp, sum(m.M) FILTER (z.ph <> p.ph) AS rr
FROM mh m JOIN dz z USING (dev, det) JOIN hix x USING (dev, hid)
JOIN (SELECT dev, ph FROM dz WHERE ph IS NOT NULL AND fn IS DISTINCT FROM 'Bike' GROUP BY dev, ph HAVING count(*) >= 2) p
  ON p.dev = m.dev
WHERE z.ph IS NOT NULL AND z.fn IS DISTINCT FROM 'Bike'
GROUP BY m.dev, p.ph, m.hid, x.j, x.hod, x.ch
{# usual share: the same clock hour on OTHER days of the same type (weekday / weekend) -- a minor phase's share of the
   signal moves with the hour and the day, so an all-day median turns every morning peak into a "drop" #}
-- @step pdr mat
SELECT * FROM (
WITH p AS (SELECT *, hid // 24 AS day, isodow(make_timestamp(hid * {{ NSH }} // 1000)) >= 6 AS wk FROM pdm),
h AS (SELECT a.dev, a.ph, a.hid, a.j, a.mp, a.rr, median(b.mp / b.rr) AS sp
      FROM p a JOIN p b ON b.dev = a.dev AND b.ph = a.ph AND b.hod = a.hod AND b.wk = a.wk AND b.day <> a.day
                        AND b.ch AND b.rr >= 30 AND b.rr BETWEEN 0.5 * a.rr AND 2 * a.rr
      WHERE a.ch AND a.hod >= 6 AND a.hod < 21 AND a.rr >= 30 GROUP BY a.dev, a.ph, a.hid, a.j, a.mp, a.rr),
l AS (SELECT *, j - row_number() OVER (PARTITION BY dev, ph ORDER BY j) AS g FROM h
      WHERE sp > 0 AND sp * rr >= 30 AND mp <= 0.2 * sp * rr)
SELECT dev, ph, min(hid) AS ha, max(hid) + 1 AS hz FROM l GROUP BY dev, ph, g
HAVING count(*) >= 2 AND sum(sp * rr) >= 200)
{{ ad.greens_hour() }}
-- @step pdg mat
SELECT r.dev, r.ph, r.ha, r.hz FROM pdr r JOIN win w USING (dev)
WHERE EXISTS (SELECT 1 FROM greens_hour e WHERE e.dev = r.dev AND e.ph = r.ph AND e.h >= r.ha AND e.h < r.hz)
-- @step pdd mat
SELECT z.dev, z.det, g.ph, g.ha, g.hz,
       sum(m.M) FILTER (m.hid >= g.ha AND m.hid < g.hz) AS obs,
       sum(m.M) FILTER (NOT (m.hid >= g.ha AND m.hid < g.hz) AND x.ch AND x.hod >= 7 AND x.hod < 19 AND p.rr >= 30)
       / nullif(sum(p.rr) FILTER (NOT (m.hid >= g.ha AND m.hid < g.hz) AND x.ch AND x.hod >= 7 AND x.hod < 19
                                 AND p.rr >= 30), 0)
       * sum(p.rr) FILTER (m.hid >= g.ha AND m.hid < g.hz) AS ex
FROM pdg g JOIN dz z ON z.dev = g.dev AND z.ph = g.ph AND z.fn IS DISTINCT FROM 'Bike'
JOIN mh m ON m.dev = z.dev AND m.det = z.det JOIN hix x ON x.dev = m.dev AND x.hid = m.hid
JOIN pdm p ON p.dev = m.dev AND p.ph = g.ph AND p.hid = m.hid
GROUP BY z.dev, z.det, g.ph, g.ha, g.hz
-- @step pdk mat
SELECT * FROM (
  SELECT d.*, count(*) FILTER (d.ex >= 30 AND d.obs <= 0.25 * d.ex) OVER (PARTITION BY d.dev, d.ph, d.ha) AS k,
         count(*) OVER (PARTITION BY d.dev, d.ph, d.ha) AS m FROM pdd d)
QUALIFY row_number() OVER (PARTITION BY dev, det ORDER BY ex - obs DESC, ph, ha) = 1
{# lockstep: >= 4 detectors of a phase (mirrored copies counted once) logging the same hourly count (within 1 + 1 %,
   >= 20 an hour) for >= 6 hours in a row (note 154f: 3 h of low traffic can match by chance) -- traffic never does
   that, not even three zones of one lane; the inputs log the same pulses (card in fail-safe, a call placed on them).
   Needs no history, so it is seen on any day of the failure, not only its first. #}
-- @step lkh mat
SELECT m.dev, z.ph, m.hid, x.j, x.ch, m.det, m.M, median(m.M) OVER (PARTITION BY m.dev, z.ph, m.hid) AS md
FROM mh m JOIN dz z USING (dev, det) JOIN hix x USING (dev, hid) JOIN dstat s USING (dev, det)
WHERE z.ph IS NOT NULL AND z.fn IS DISTINCT FROM 'Bike' AND s.live
  AND NOT EXISTS (SELECT 1 FROM twin t WHERE t.dev = m.dev AND t.i = m.det AND t.j < m.det)
-- @step lkr mat
SELECT dev, ph, min(hid) AS ha, max(hid) + 1 AS hz, median(md) AS md FROM (
  SELECT *, j - row_number() OVER (PARTITION BY dev, ph ORDER BY j) AS g FROM (
    SELECT dev, ph, hid, j, any_value(md) AS md FROM lkh WHERE ch AND md >= 20
    GROUP BY dev, ph, hid, j HAVING count(*) FILTER (abs(M - md) <= 0.01 * md + 1) >= 4))
GROUP BY dev, ph, g HAVING count(*) >= 6
-- @step lkd mat
SELECT * FROM (
  SELECT h.dev, h.det, r.ph, r.ha, r.hz, r.md, count(*) OVER (PARTITION BY r.dev, r.ph, r.ha) AS nk FROM lkr r
  JOIN lkh h ON h.dev = r.dev AND h.ph = r.ph AND h.hid >= r.ha AND h.hid < r.hz
  GROUP BY h.dev, h.det, r.dev, r.ph, r.ha, r.hz, r.md HAVING count(*) FILTER (abs(h.M - h.md) <= 0.01 * h.md + 1) >= 3)
QUALIFY row_number() OVER (PARTITION BY dev, det ORDER BY hz - ha DESC, ph, ha) = 1
-- @step {{ out }} mat
SELECT x.* EXCLUDE (mpd, mlk) REPLACE (
         CASE WHEN mlk THEN 1e6 WHEN mpd THEN k.ex - k.obs ELSE x.drop_lam END AS drop_lam,
         CASE WHEN mlk THEN (l.ha * {{ NSH }} - w.s_ns) // {{ NS5 }} WHEN mpd THEN (k.ha * {{ NSH }} - w.s_ns) // {{ NS5 }}
              ELSE x.drop_b0 END AS drop_b0,
         CASE WHEN mlk THEN least((l.hz * {{ NSH }} - w.s_ns) // {{ NS5 }}, w.nb5)
              WHEN mpd THEN least((k.hz * {{ NSH }} - w.s_ns) // {{ NS5 }}, w.nb5) ELSE x.drop_b1 END AS drop_b1,
         CASE WHEN mlk THEN 1.0 WHEN mpd THEN 0.35 + 0.65 * least((k.ex - k.obs - 30) / 70, 1) ELSE x.s_dropout END
           AS s_dropout),
       mpd AND NOT mlk AS ph_drop, mlk AS ph_lock, k.k AS ph_k, k.m AS ph_m, k.obs AS ph_obs, k.ex AS ph_exp,
       l.nk AS lk_n, l.md AS lk_md
FROM (SELECT x.*, coalesce(k.ex >= 30 AND k.obs <= 0.25 * k.ex AND k.ex - k.obs >= 30
                           AND k.ex - k.obs > coalesce(x.drop_lam, 0), false) AS mpd, l.det IS NOT NULL AS mlk
      FROM {{ src }} x LEFT JOIN pdk k USING (dev, det) LEFT JOIN lkd l USING (dev, det)) x
LEFT JOIN pdk k USING (dev, det) LEFT JOIN lkd l USING (dev, det) JOIN win w USING (dev)
{% endmacro %}
