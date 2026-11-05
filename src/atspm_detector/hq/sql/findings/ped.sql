{# The ped buttons of the findings table, from today's two outputs (@PDET@: ped_health_duckdb.COLS, @PPER@: PERIOD_COLS).
   reason is cut into one piece per check, config into its notes; a wait's note (omitted during preemption, call dropped)
   is the Detail of the wait's own rows, every other note and too few to judge one row per calendar day of the window. #}
-- @step pbody
SELECT DeviceId, Input::BIGINT AS det, status AS st, coalesce(config, '') AS cfg, coalesce(reason, '') AS body FROM @PDET@
-- @step ppiece
SELECT DeviceId, det, txt, CASE {% for rx, kind in PED_PIECES %}WHEN regexp_matches(txt, {{ rx|str }}) THEN {{ kind|str }} {% endfor %}END AS kind
FROM (SELECT DeviceId, det, unnest(string_split(regexp_replace(body, {{ PED_SPLIT|str }}, chr(31) || '\1', 'g'), chr(31))) AS txt FROM pbody)
-- @step pnote
SELECT DeviceId, det, part AS txt, CASE {% for rx, lab in PED_NOTES %}WHEN regexp_matches(part, {{ rx|str }}) THEN {{ lab|str }} {% endfor %}ELSE {{ CONFIG_LABEL|str }} END AS lab
FROM (SELECT DeviceId, det, unnest(string_split(cfg, '; ')) AS part FROM pbody WHERE cfg <> '')
-- @step pdet mat
SELECT p.DeviceId, p.Detector::BIGINT AS det, p."start"::TIMESTAMP AS a, p."end"::TIMESTAMP AS z, p.condition AS what, p.severity::DOUBLE AS sev,
       p.whole_window::BOOLEAN AS ww, coalesce(x.txt, b.body, p.condition) AS txt
FROM @PPER@ p
LEFT JOIN (SELECT DeviceId, det, kind, txt FROM ppiece WHERE kind IS NOT NULL
           UNION ALL SELECT DeviceId, det, lab, txt FROM pnote WHERE lab IN {{ PED_WAITS|inlist }}) x
  ON x.DeviceId = p.DeviceId AND x.det = p.Detector::BIGINT AND x.kind = p.condition
LEFT JOIN pbody b ON b.DeviceId = p.DeviceId AND b.det = p.Detector::BIGINT
-- @step prows
SELECT DeviceId, 'Ped' AS DetectorType, det AS Detector, a AS "Start", z AS "End", what AS Label, sev AS Confidence, txt AS Detail
FROM pdet WHERE NOT ww
UNION ALL
SELECT p.DeviceId, 'Ped', p.det, d.day_start, d.day_end, p.what, p.sev, p.txt
FROM pdet p JOIN fdays d ON d.DeviceId = p.DeviceId AND d.DetectorType = 'Ped' WHERE p.ww
UNION ALL
SELECT n.DeviceId, 'Ped', n.det, d.day_start, d.day_end, n.lab, {{ NOTE_CONF|dbl }}, n.txt
FROM pnote n JOIN fdays d ON d.DeviceId = n.DeviceId AND d.DetectorType = 'Ped' WHERE n.lab NOT IN {{ PED_WAITS|inlist }}
UNION ALL
SELECT b.DeviceId, 'Ped', b.det, d.day_start, d.day_end, {{ TOO_FEW|str }}, {{ NOTE_CONF|dbl }}, b.body
FROM pbody b JOIN fdays d ON d.DeviceId = b.DeviceId AND d.DetectorType = 'Ped' WHERE b.st = {{ TOO_FEW|str }}
-- @step pstat
SELECT b.DeviceId, 'Ped' AS DetectorType, b.det AS Detector, d.day_date AS "Date", b.st AS Status
FROM pbody b JOIN fdays d ON d.DeviceId = b.DeviceId AND d.DetectorType = 'Ped'
