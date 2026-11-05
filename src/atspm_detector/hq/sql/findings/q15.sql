{# The optional 15-minute table (findings.q15): each finding of a suspect or bad detector or button, one row per
   quarter-hour since the previous judge.  Notes and borderline rows are not faults (Confidence under 0.5, or none).
   Sources: @FINDINGS@, @STATUS@.  Context: since, end. #}
-- @step qflag
SELECT DISTINCT DeviceId, DetectorType, Detector, Status FROM @STATUS@ WHERE Status IN ('suspect', 'bad')
-- @step qbin
SELECT DeviceId, DetectorType, Detector, Label, Confidence,
       unnest(generate_series(time_bucket(INTERVAL 15 MINUTE, "Start"), "End" - INTERVAL 1 MICROSECOND, INTERVAL 15 MINUTE)) AS TimeStamp
FROM @FINDINGS@ WHERE Confidence >= 0.5 AND "End" > "Start" AND "End" > {{ since|ts }} AND "Start" < {{ end|ts }}
-- @step q15
SELECT b.TimeStamp, b.DeviceId, b.DetectorType, b.Detector, b.Label, max(b.Confidence) AS Confidence, f.Status
FROM qbin b JOIN qflag f ON f.DeviceId = b.DeviceId AND f.DetectorType = b.DetectorType AND f.Detector = b.Detector
WHERE b.TimeStamp >= {{ since|ts }} AND b.TimeStamp < {{ end|ts }}
GROUP BY b.TimeStamp, b.DeviceId, b.DetectorType, b.Detector, b.Label, f.Status
