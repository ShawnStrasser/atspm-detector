{# The window rules of the vehicle health (health_duckdb._inputs): each signal's window from its allowed events (a store's
   signal rows when store is set), or the caller's windows table (@WINDOWS@: DeviceId, s_ns, e_ns), then one row per
   signal (win).
   Context: windows (a windows table is bound), start / end (None or a time), store. #}
{% import "_lib.sql" as lib with context %}
{% if not windows %}
{# the sample window of each signal: start / end where given, else its first event and one second past its last; a
   bound more than 30 min outside the signal's data is treated as not given (pipeline._windows) #}
{% set b0 = start|ns if start is not none else 'w0' %}
{% set b1 = end|ns if end is not none else 'w1' %}
-- @step wraw
{% if store %}
SELECT DeviceId, epoch_ns(min(First_Event)) AS w0, epoch_ns(max(Last_Event)) + 1000000000 AS w1, sum(Events)::BIGINT AS n
FROM {{ store }}health_signal_5min WHERE TimeStamp >= {{ start|ts }} AND TimeStamp < {{ end|ts }} GROUP BY DeviceId
{% else %}
SELECT DeviceId, epoch_ns(min(Timestamp)) AS w0, epoch_ns(max(Timestamp)) + 1000000000 AS w1, count(*) AS n
FROM @EVENTS@ WHERE {{ lib.good() }} AND DeviceId IS NOT NULL AND Timestamp IS NOT NULL{% if start is not none %} AND Timestamp >= {{ start|ts }}{% endif %}{% if end is not none %} AND Timestamp < {{ end|ts }}{% endif %} GROUP BY DeviceId
{% endif %}
-- @step wbin
SELECT DeviceId, w0, w1, CASE WHEN {{ b1 }} > {{ b0 }} THEN {{ b0 }} ELSE w0 END AS b0,
       CASE WHEN {{ b1 }} > {{ b0 }} THEN {{ b1 }} ELSE w1 END AS b1 FROM wraw
-- @step wsel
SELECT DeviceId, CASE WHEN w0 - b0 > 1800000000000 THEN w0 ELSE b0 END AS s_ns,
       CASE WHEN b1 - w1 > 1800000000000 THEN w1 ELSE b1 END AS e_ns FROM wbin
{% endif %}
{# one row per signal: its window [s_ns, e_ns) (epoch ns), length T (s), 5-min bins nb5, length class wg and the
   occupancy bin size bs (15 min, 5 min for a sample under ~1.2 h).  clock (bin_origin="clock", design 3.4): the window
   starts on the quarter-hour at or before (bins, quarter-hours and hours then lie on the clock); else (bin_origin
   "window") at s_ns as given, today's origin. #}
-- @step win mat
SELECT dev, DeviceId, s_ns, e_ns, T, T / 3600.0 AS hours, ceil(T / 300.0)::INT AS nb5, wg,
       CASE WHEN wg = 'm30' THEN 300 ELSE 900 END AS bs,
       ceil(T / CASE WHEN wg = 'm30' THEN 300.0 ELSE 900.0 END)::INT AS nbb,
       make_timestamp(s_ns // 1000) AS lo, make_timestamp(e_ns // 1000 + 1) AS hi,
       s_ns % 86400000000000 AS sod
FROM (SELECT (dense_rank() OVER (ORDER BY CAST(DeviceId AS VARCHAR)))::INT AS dev, DeviceId,
             s_ns::BIGINT AS s_ns, e_ns::BIGINT AS e_ns, (e_ns - s_ns) / 1e9 AS T,
             CASE WHEN (e_ns - s_ns) / 3.6e12 < sqrt(1.5) THEN 'm30'
                  WHEN (e_ns - s_ns) / 3.6e12 < sqrt(72.0) THEN 'h3' ELSE 'h24' END AS wg
{% if clock %}
      FROM (SELECT DeviceId, s_ns::BIGINT - s_ns::BIGINT % {{ NSQ }} AS s_ns, e_ns FROM {{ '@WINDOWS@' if windows else 'wsel' }}))
{% else %}
      FROM {{ '@WINDOWS@' if windows else 'wsel' }})
{% endif %}
