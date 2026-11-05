{# The extractor, ped part (design 6): the same update, the same chunk (@EVENTS@, all codes), each signal's rows after its own last event of
   any code (@PEDWIN@: DeviceId, s_ns, e_ns; wm, its last processed event, NULL for a signal starting over; carry 9401).  The vehicle part
   keeps its own cut, the last allowed event (9402).  Writes the ped log, the 15-minute bins the signal logged in and the silences; the
   ped judge (ped/judge.sql) reads those instead of the raw log, nothing else differs.  A silence is written by the event ending it. #}
{% import "ped/_lib.sql" as plib %}
-- @step xpb mat
SELECT w.DeviceId, epoch_ns(e.Timestamp) // 10000000000 AS k, min(epoch_ns(e.Timestamp)) AS a, max(epoch_ns(e.Timestamp)) AS z
FROM @EVENTS@ e JOIN @PEDWIN@ w ON e.DeviceId = w.DeviceId
WHERE epoch_ns(e.Timestamp) >= w.s_ns AND epoch_ns(e.Timestamp) < w.e_ns GROUP BY w.DeviceId, k
-- @step health_ped_events
-- @emit health_ped_events
SELECT DISTINCT e.DeviceId, make_timestamp_ns(epoch_ns(e.Timestamp)) AS TimeStamp, e.EventId::SMALLINT AS EventId, e.Parameter::INTEGER AS Parameter
FROM @EVENTS@ e JOIN @PEDWIN@ w ON e.DeviceId = w.DeviceId
WHERE epoch_ns(e.Timestamp) >= w.s_ns AND epoch_ns(e.Timestamp) < w.e_ns AND e.EventId IN {{ LOG_CODES|intlist }}
-- @step health_log_15min
-- @emit health_log_15min
SELECT DeviceId, make_timestamp_ns((k // 90) * 900000000000) AS TimeStamp, make_timestamp_ns(min(a)) AS All_First, make_timestamp_ns(max(z)) AS All_Last
FROM xpb GROUP BY DeviceId, k // 90
-- @step xpsilence
-- @emit health_intervals
SELECT DeviceId, 'silence' AS Kind, NULL::INTEGER AS Channel, make_timestamp_ns(n0) AS StartTime, make_timestamp_ns(n1) AS EndTime,
       NULL::BOOLEAN AS Relogged, NULL::INTEGER AS Value
FROM ({{ plib.silences('xpb', 'DeviceId', 'SELECT DeviceId, wm, wm FROM @PEDWIN@ WHERE wm IS NOT NULL') }})
-- @step xpcarry
-- @emit health_carry
SELECT make_timestamp_ns(max(z)) AS TimeStamp, DeviceId, 9401::SMALLINT AS EventId, 0 AS Parameter, true AS IsValid FROM xpb GROUP BY DeviceId
