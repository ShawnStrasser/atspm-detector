{# ped_detectors' minutes_of_data: each signal's ped log span in the window (ped_assign.ped_detectors).
   Context: E, start, end, CODES. #}
{% import "ped/_lib.sql" as plib %}
-- @step span
SELECT DeviceId, (epoch_ns(max(Timestamp)) - epoch_ns(min(Timestamp))) / 6e10 AS m
                           FROM {{ E }} WHERE {{ plib.window(start, end, CODES) }} GROUP BY DeviceId
