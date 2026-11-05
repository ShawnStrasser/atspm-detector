{# The final queries of the vehicle health (health_duckdb.FINAL / PERIODS_FINAL): one row per detector (OUT_COLS) and
   one row per period (PERIOD_COLS). #}
-- @step detectors
SELECT w.DeviceId, o.det AS Detector, o.health_status, o.health_score, o.health_reason, o.health_categories,
       o.health_config, o.health_watch, n.health_signal_note
FROM (SELECT * FROM outx UNION ALL BY NAME SELECT * FROM outf) o JOIN win w USING (dev) JOIN snote n USING (dev)
ORDER BY CAST(w.DeviceId AS VARCHAR), o.det
-- @step periods
SELECT w.DeviceId, p.det AS Detector, make_timestamp(p.a // 1000) AS "start", make_timestamp(p.z // 1000) AS "end",
       p.what AS condition, round(p.sev, 3) AS severity, p.lim::DOUBLE AS limit_s, coalesce(p.lc, false) AS in_closure,
       p.ww AS whole_window
FROM (SELECT * FROM pall UNION ALL SELECT * FROM pbasic) p JOIN win w USING (dev)
ORDER BY CAST(w.DeviceId AS VARCHAR), p.det, p.a, p.o, p.z
