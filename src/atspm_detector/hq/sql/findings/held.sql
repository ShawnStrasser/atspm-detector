{# ONs held stuck, past the window if need be (bins.held, on a store).  A judge rates an ON that is still open and that the rules call stuck
   now; the rating stays in the store's carry (9223 a detector; 9232 a button, Parameter = input * 1000 + phase) for as long as the ON
   stays on, past the window and the trim.  It ends at the input's next event (a detector: its chain closing, kept in health_intervals; a
   button: its next OFF or ON in the ped log, but an OFF with an ON at most hold_gap_s after it is a brief release: it goes on, as stuck-on periods that close read as one).  Scored for each 15-minute cell of [since, end) with evidence: a detector's signal logged in the
   cell, a button's phase walked in it; no evidence, no row: nothing is known of an input while its signal is silent.
   Sources: @FINDINGS@, @STATUS@, @PEDROWS@ (DeviceId, Input, Phase).  Context: s (the store and a dot), since, end, judged, ped,
   vehicle_labels, hold_gap_s. #}
{% set cell = 'INTERVAL 15 MINUTE' %}
-- @step hraw
SELECT DeviceId, Parameter AS det, TimeStamp AS t, EventId FROM {{ s }}health_carry WHERE EventId IN (9201, 9202, 9222)
-- @step hoff
SELECT DeviceId, det, max(t) FILTER (EventId = 9201) AS off FROM hraw GROUP BY DeviceId, det HAVING count(*) FILTER (EventId = 9222) = 1
{# each detector's chain still open: its first ON after its last OFF #}
-- @step vopen
SELECT r.DeviceId, r.det, min(r.t) AS t0 FROM hraw r JOIN hoff o ON o.DeviceId = r.DeviceId AND o.det = r.det
WHERE r.EventId IN (9202, 9222) AND (o.off IS NULL OR r.t > o.off) GROUP BY r.DeviceId, r.det
{# each button's last event, when it is an ON; at one instant the ON comes after the OFF #}
-- @step popen
{% if ped %}SELECT DeviceId, det, t0 FROM (SELECT DeviceId, Parameter AS det, max_by(EventId, (TimeStamp, EventId)) AS ev, max(TimeStamp) AS t0
      FROM {{ s }}health_ped_events WHERE EventId IN (89, 90) GROUP BY DeviceId, Parameter) WHERE ev = 90
{% else %}SELECT DeviceId, det, t0 FROM vopen WHERE false{% endif %}

-- @step hflag
SELECT DISTINCT DeviceId, DetectorType, Detector::BIGINT AS det FROM @STATUS@ WHERE Status IN ('suspect', 'bad')
{# the stuck findings of this judge: Confidence at the suspect limit at least, reaching the window's last quarter-hour #}
-- @step hstuck
SELECT f.DeviceId, f.DetectorType, f.Detector::BIGINT AS det, f."End"::TIMESTAMP AS z
FROM @FINDINGS@ f JOIN hflag g ON g.DeviceId = f.DeviceId AND g.DetectorType = f.DetectorType AND g.det = f.Detector::BIGINT
WHERE f.Label IN {{ vehicle_labels|inlist }} AND f.Confidence >= 0.5 AND f."End"::TIMESTAMP >= {{ judged|ts }} - {{ cell }}
  AND (f.Label = 'stuck on' OR f.DetectorType = 'Vehicle')
-- @step hnew
SELECT DISTINCT h.DeviceId, 'Vehicle' AS DetectorType, h.det, NULL::BIGINT AS phase, v.t0
FROM hstuck h JOIN vopen v ON v.DeviceId = h.DeviceId AND v.det = h.det AND h.z > v.t0 WHERE h.DetectorType = 'Vehicle'
UNION ALL
SELECT DISTINCT h.DeviceId, 'Ped', h.det, r.Phase::BIGINT, v.t0
FROM hstuck h JOIN popen v ON v.DeviceId = h.DeviceId AND v.det = h.det AND h.z > v.t0
JOIN @PEDROWS@ r ON r.DeviceId = h.DeviceId AND r.Input::BIGINT = h.det WHERE h.DetectorType = 'Ped' AND r.Phase IS NOT NULL
-- @step hold
SELECT DeviceId, 'Vehicle' AS DetectorType, Parameter::BIGINT AS det, NULL::BIGINT AS phase, TimeStamp AS t0 FROM {{ s }}health_carry WHERE EventId = 9223
UNION ALL
SELECT DeviceId, 'Ped', (Parameter // 1000)::BIGINT, (Parameter % 1000)::BIGINT, TimeStamp FROM {{ s }}health_carry WHERE EventId = 9232
-- @step hall
SELECT DeviceId, DetectorType, det, any_value(phase) AS phase, t0, bool_or(known) AS known
FROM (SELECT *, true AS known FROM hold UNION ALL SELECT *, false FROM hnew) GROUP BY DeviceId, DetectorType, det, t0
{# a rated button's events that end an ON: an OFF with no ON soon after it, an ON with no OFF just before it #}
-- @step pends
{% if ped %}SELECT DeviceId, det, t FROM (
    SELECT e.DeviceId, e.Parameter::BIGINT AS det, e.TimeStamp AS t, e.EventId AS ev, lag(e.EventId) OVER w AS ev0, lead(e.EventId) OVER w AS ev1,
           epoch(e.TimeStamp) - epoch(lag(e.TimeStamp) OVER w) AS gap0, epoch(lead(e.TimeStamp) OVER w) - epoch(e.TimeStamp) AS gap1
    FROM {{ s }}health_ped_events e SEMI JOIN (SELECT DeviceId, det FROM hall WHERE DetectorType = 'Ped') r ON r.DeviceId = e.DeviceId AND r.det = e.Parameter
    WHERE e.EventId IN (89, 90) WINDOW w AS (PARTITION BY e.DeviceId, e.Parameter ORDER BY e.TimeStamp, e.EventId))
WHERE CASE WHEN ev = 89 THEN NOT coalesce(ev1 = 90 AND gap1 <= {{ hold_gap_s }}, false) ELSE NOT coalesce(ev0 = 89 AND gap0 <= {{ hold_gap_s }}, false) END
{% else %}SELECT DeviceId, det, t0 AS t FROM vopen WHERE false{% endif %}

{# where each rated ON stands: still open (f NULL), closed at f, or gone (no chain of it left: not scored, dropped) #}
-- @step hstate
SELECT a.*, CASE WHEN a.DetectorType = 'Vehicle' THEN (SELECT min(i.EndTime) FROM {{ s }}health_intervals i WHERE i.DeviceId = a.DeviceId AND i.Kind IN ('long_on', 'bad_chain')
                      AND i.Channel = a.det AND abs(epoch_ms(i.StartTime) - epoch_ms(a.t0)) <= 1)
{% if ped %}            ELSE (SELECT min(e.t) FROM pends e WHERE e.DeviceId = a.DeviceId AND e.det = a.det AND e.t > a.t0)
{% endif %}            END AS f,
       a.DetectorType = 'Ped' OR EXISTS (SELECT 1 FROM vopen v WHERE v.DeviceId = a.DeviceId AND v.det = a.det AND abs(epoch_ms(v.t0) - epoch_ms(a.t0)) <= 1) AS isopen
FROM hall a
{# the 15-minute cells with evidence #}
-- @step hvcell
SELECT DeviceId, time_bucket({{ cell }}, TimeStamp::TIMESTAMP) AS q FROM {{ s }}health_signal_5min WHERE Events > 0 GROUP BY 1, 2
-- @step hpcell
{% if ped %}SELECT DeviceId, Parameter::BIGINT AS phase, time_bucket({{ cell }}, TimeStamp::TIMESTAMP) AS q FROM {{ s }}health_ped_events WHERE EventId = 21 GROUP BY 1, 2, 3
{% else %}SELECT DeviceId, phase, t0::TIMESTAMP AS q FROM hold WHERE false{% endif %}

{# the stretch of each rated ON in [since, end) #}
-- @step hspan
SELECT * FROM (SELECT DeviceId, DetectorType, det, phase, greatest(t0::TIMESTAMP, {{ since|ts }}) AS a, least(coalesce(f::TIMESTAMP, {{ end|ts }}), {{ end|ts }}) AS z
               FROM hstate WHERE f IS NOT NULL OR isopen) WHERE a < z
-- @step hheld
SELECT DISTINCT s.DeviceId, s.DetectorType, s.det AS Detector, greatest(s.a, c.q) AS "Start", least(s.z, c.q + {{ cell }}) AS "End"
FROM hspan s JOIN hvcell c ON c.DeviceId = s.DeviceId AND c.q < s.z AND c.q + {{ cell }} > s.a WHERE s.DetectorType = 'Vehicle'
UNION ALL
SELECT DISTINCT s.DeviceId, s.DetectorType, s.det, greatest(s.a, c.q), least(s.z, c.q + {{ cell }})
FROM hspan s JOIN hpcell c ON c.DeviceId = s.DeviceId AND c.phase = s.phase AND c.q < s.z AND c.q + {{ cell }} > s.a WHERE s.DetectorType = 'Ped'
{# the carry after this judge: new ratings of ONs still open in, closed or gone out #}
-- @step hinsert
SELECT t0, DeviceId, CASE DetectorType WHEN 'Vehicle' THEN 9223 ELSE 9232 END AS code, CASE DetectorType WHEN 'Vehicle' THEN det ELSE det * 1000 + phase END AS par
FROM hstate WHERE NOT known AND f IS NULL AND isopen
-- @step hdelete
SELECT t0, DeviceId, CASE DetectorType WHEN 'Vehicle' THEN 9223 ELSE 9232 END AS code, CASE DetectorType WHEN 'Vehicle' THEN det ELSE det * 1000 + phase END AS par
FROM hstate WHERE known AND (f IS NOT NULL OR NOT isopen)
