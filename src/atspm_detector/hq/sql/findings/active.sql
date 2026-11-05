{# The optional active table (findings.active_next, off until reviewed; it adds no health rule): the store's table
   @ACTIVE@ moved on one judge ending at {{ end }}.  A finding of a suspect or bad detector or button (Confidence 0.5 or
   more) is matched to an earlier row on device, type, detector and label with overlapping times.
     new and reaching the window end -> provisional; new and already over -> confirmed;
     matched (seen in two judges running) -> confirmed;
     not matched in two judges running, or the detector back to ok -> cleared, shown once, then gone.
   Sources: @FINDINGS@, @STATUS@, @ACTIVE@. #}
-- @step aflag
SELECT DISTINCT DeviceId, DetectorType, Detector, Status FROM @STATUS@
-- @step acur
SELECT row_number() OVER () AS cid, f.DeviceId, f.DetectorType, f.Detector, f.Label, f."Start", f."End", f.Confidence
FROM @FINDINGS@ f JOIN aflag s ON s.DeviceId = f.DeviceId AND s.DetectorType = f.DetectorType AND s.Detector = f.Detector
WHERE s.Status IN ('suspect', 'bad') AND f.Confidence >= 0.5
-- @step aold
SELECT row_number() OVER () AS oid, * FROM @ACTIVE@ WHERE State <> 'cleared'
-- @step ahit
SELECT o.oid, c.cid, c."Start", c."End", c.Confidence
FROM aold o JOIN acur c ON c.DeviceId = o.DeviceId AND c.DetectorType = o.DetectorType AND c.Detector = o.Detector
     AND c.Label = o.Label AND c."Start" <= o."End" AND c."End" >= o."Start"
-- @step aseen
SELECT oid, min("Start") AS s, max("End") AS e, max(Confidence) AS q FROM ahit GROUP BY oid
-- @step anext
SELECT o.DeviceId, o.DetectorType, o.Detector, o.Label, least(o."Start", h.s) AS "Start", greatest(o."End", h.e) AS "End",
       greatest(o.Confidence, h.q) AS Confidence, 'confirmed' AS State, o.FirstSeen, {{ end|ts }} AS LastSeen, 0 AS Missed
FROM aold o JOIN aseen h ON h.oid = o.oid
UNION ALL
SELECT o.DeviceId, o.DetectorType, o.Detector, o.Label, o."Start", o."End", o.Confidence,
       CASE WHEN o.Missed + 1 >= 2 OR s.Status = 'ok' THEN 'cleared' ELSE o.State END, o.FirstSeen, o.LastSeen, o.Missed + 1
FROM aold o LEFT JOIN aflag s ON s.DeviceId = o.DeviceId AND s.DetectorType = o.DetectorType AND s.Detector = o.Detector
WHERE o.oid NOT IN (SELECT oid FROM aseen)
UNION ALL
SELECT c.DeviceId, c.DetectorType, c.Detector, c.Label, c."Start", c."End", c.Confidence,
       CASE WHEN c."End" >= {{ end|ts }} - INTERVAL 15 MINUTE THEN 'provisional' ELSE 'confirmed' END,
       {{ end|ts }}, {{ end|ts }}, 0
FROM acur c WHERE c.cid NOT IN (SELECT cid FROM ahit)
