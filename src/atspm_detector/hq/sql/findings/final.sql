{# The two tables (findings.FINDING_COLS, findings.STATUS_COLS).  An exact repeat is one finding, not two. #}
-- @step findings
SELECT DeviceId, DetectorType, Detector, "Start", "End", Label, Confidence, Detail FROM (SELECT DISTINCT * FROM frows)
ORDER BY CAST(DeviceId AS VARCHAR), DetectorType DESC, Detector NULLS FIRST, "Start", "End", Label, Detail
-- @step status
SELECT DeviceId, DetectorType, Detector, "Date", Status FROM (SELECT DISTINCT * FROM srows)
ORDER BY CAST(DeviceId AS VARCHAR), DetectorType DESC, Detector, "Date", Status
