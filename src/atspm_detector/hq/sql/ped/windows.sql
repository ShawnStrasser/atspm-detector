{# The ped health's signals and windows (ped_health_duckdb.ped_health -> pq_wall): signals with ped input events, their
   window [s, e) in epoch seconds (start / end where given, else the first event to one second past the last) and their
   ped-code events (for the groups).  Context: E, start, end, CODES; hist: a table of earlier press counts (option stop_history) --
   a signal whose buttons were all silent in the window is still judged. #}
{% import "ped/_lib.sql" as plib %}
-- @step wall
SELECT (dense_rank() OVER (ORDER BY CAST(DeviceId AS VARCHAR)))::INT AS dev, DeviceId, {{ start|epoch_s if start is not none else 'w0' }} AS s, {{ end|epoch_s if end is not none else 'w1' }} AS e, n
        FROM (SELECT DeviceId, epoch_ns(min(Timestamp)) / 1e9 AS w0, epoch_ns(max(Timestamp)) / 1e9 + 1 AS w1,
                     count(*) FILTER (EventId IN {{ CODES|intlist }}) AS n
              FROM {{ E }} WHERE {{ plib.window(start, end) }} GROUP BY DeviceId HAVING count(*) FILTER (EventId IN (89, 90)) > 0{% if carry %} OR DeviceId IN (SELECT DeviceId FROM pq_carry){% endif %}{% if hist %} OR CAST(DeviceId AS VARCHAR) IN (SELECT DISTINCT CAST(DeviceId AS VARCHAR) FROM {{ hist }} WHERE presses > 0){% endif %})
