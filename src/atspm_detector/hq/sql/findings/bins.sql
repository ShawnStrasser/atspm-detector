{# The health bins (bins.bins): one row per bin of [since, end) per problem of a detector or button; no row = no problem found.
   A problem is a rule's period on a suspect or bad detector or button (FULL lists the rules), or an ON held stuck (@HELD@, when
   held: DeviceId, DetectorType, Detector, Start, End; bins.py).  Its Score at a moment is least(1, Confidence / full); at a moment
   with several problems the worst of them is shared equally, so the rows of a bin add up to the detector's score of that bin, at
   most 1; a bin partly covered counts its covered share.  Not enough data: one row a day (midnight's bin), no Score.
   Sources: @FINDINGS@, @STATUS@, [@HELD@].  Context: since, end, B (minutes), FULL, TOO_FEW, EXPLAINED, held. #}
{% set bin %}INTERVAL ({{ B }}) MINUTE{% endset %}
-- @step bflag
SELECT DISTINCT DeviceId, DetectorType, Detector::BIGINT AS det FROM @STATUS@ WHERE Status IN ('suspect', 'bad')
{% if held %}
-- @step bheld
SELECT DeviceId, DetectorType, Detector::BIGINT AS det, "Start"::TIMESTAMP AS a, "End"::TIMESTAMP AS z FROM @HELD@
{% endif %}
{# one problem, one period, cut to [since, end).  An ON held stuck explains a detector's silence: not a second problem #}
-- @step bprob
SELECT f.DeviceId, f.DetectorType, f.Detector::BIGINT AS det, f.Label AS lab, greatest(f."Start"::TIMESTAMP, {{ since|ts }}) AS a,
       least(f."End"::TIMESTAMP, {{ end|ts }}) AS z, least(1.0, f.Confidence / m.lim) AS s
FROM @FINDINGS@ f JOIN ({{ FULL|values('lab, lim') }}) m ON m.lab = f.Label
JOIN bflag g ON g.DeviceId = f.DeviceId AND g.DetectorType = f.DetectorType AND g.det = f.Detector::BIGINT
WHERE f.Confidence > 0 AND f."End" > f."Start" AND f."Start" < {{ end|ts }} AND f."End" > {{ since|ts }}
{% if held %}  AND NOT (f.Label IN {{ EXPLAINED|inlist }} AND EXISTS (SELECT 1 FROM bheld h WHERE h.DeviceId = f.DeviceId AND h.DetectorType = f.DetectorType AND h.det = f.Detector::BIGINT))
UNION ALL
SELECT DeviceId, DetectorType, det, 'stuck on', a, z, 1.0 FROM bheld{% endif %}

{# the stretches between one detector's period ends: its problems do not change inside one #}
-- @step bseg
SELECT DeviceId, DetectorType, det, t AS a, lead(t) OVER (PARTITION BY DeviceId, DetectorType, det ORDER BY t) AS z
FROM (SELECT DeviceId, DetectorType, det, a AS t FROM bprob UNION SELECT DeviceId, DetectorType, det, z FROM bprob)
-- @step bon
SELECT g.DeviceId, g.DetectorType, g.det, g.a, g.z, p.lab, max(p.s) AS s
FROM bseg g JOIN bprob p ON p.DeviceId = g.DeviceId AND p.DetectorType = g.DetectorType AND p.det = g.det AND p.a < g.z AND p.z > g.a
WHERE g.z IS NOT NULL GROUP BY g.DeviceId, g.DetectorType, g.det, g.a, g.z, p.lab
-- @step bshare
SELECT DeviceId, DetectorType, det, a, z, lab, max(s) OVER w / count(*) OVER w AS v
FROM bon WINDOW w AS (PARTITION BY DeviceId, DetectorType, det, a)
-- @step bcell
SELECT t, DeviceId, DetectorType, det, lab,
       v * (epoch(least(z, t + {{ bin }})) - epoch(greatest(a, t))) / {{ B * 60 }} AS x
FROM (SELECT *, unnest(generate_series(time_bucket({{ bin }}, a), z - INTERVAL 1 MICROSECOND, {{ bin }})) AS t FROM bshare)
-- @step brows
SELECT t AS TimeStamp, DeviceId, DetectorType, det AS Detector, lab AS Label, round(sum(x), 6) AS Score
FROM bcell GROUP BY t, DeviceId, DetectorType, det, lab HAVING round(sum(x), 6) > 0
UNION ALL
SELECT DISTINCT time_bucket({{ bin }}, date_trunc('day', "Start"::TIMESTAMP)), DeviceId, DetectorType, Detector::BIGINT, Label, NULL::DOUBLE
FROM @FINDINGS@ WHERE Label = {{ TOO_FEW|str }} AND Detector IS NOT NULL
  AND date_trunc('day', "Start"::TIMESTAMP) >= {{ since|ts }} AND date_trunc('day', "Start"::TIMESTAMP) < {{ end|ts }}
