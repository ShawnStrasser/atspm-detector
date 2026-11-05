{# The calendar days of each judged window (findings.tables): the source @WIN@ has DeviceId, DetectorType, start, end
   (end exclusive).  One row per signal, detector type and day the window touches, the day cut to the window.  The last
   day is the day of the window's last second: a window ending at midnight does not touch the next day, nor does one that
   ends one second past a last event just before midnight (the default windows of health() and ped_health()). #}
-- @step fdays mat
SELECT DeviceId, DetectorType, CAST(d AS DATE) AS day_date, greatest(d, s) AS day_start, least(d + INTERVAL 1 DAY, e) AS day_end
FROM (SELECT DeviceId, DetectorType, s, e,
             unnest(generate_series(CAST(s AS DATE)::TIMESTAMP, CAST(e - INTERVAL 1 SECOND AS DATE)::TIMESTAMP,
                                    INTERVAL 1 DAY)) AS d
      FROM (SELECT DeviceId, DetectorType, "start"::TIMESTAMP AS s, "end"::TIMESTAMP AS e FROM @WIN@))
