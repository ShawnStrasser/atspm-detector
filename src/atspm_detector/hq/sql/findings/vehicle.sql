{# The vehicle detectors of the findings table, from today's two outputs (@VDET@: health_duckdb.OUT_COLS, @VPER@:
   PERIOD_COLS).  health_reason is cut into one piece of evidence per finding (findings.PIECES says which piece belongs
   to which condition), the periods get their piece as Detail; a period without a time of its own and every note become
   one row per calendar day of the window. #}
-- @step vbody
SELECT DeviceId, Detector::BIGINT AS det, health_status AS st, coalesce(health_watch, '') AS wt, coalesce(health_config, '') AS cfg,
       coalesce(health_signal_note, '') AS sig,
       regexp_replace(regexp_replace(coalesce(health_reason, ''), '^det [^:]+: ', ''), {{ LABEL_RX|str }}, '') AS body
FROM @VDET@
{# by category ("Stuck on: ...; Counts fall: ..."), then within a category by what each piece starts with #}
-- @step vcat
SELECT DeviceId, det, regexp_extract(c, {{ CAT_HEAD|str }}, 1) AS cat, regexp_replace(c, {{ CAT_HEAD|str }}, '') AS txt
FROM (SELECT DeviceId, det, unnest(string_split(regexp_replace(body, {{ CAT_SPLIT|str }}, chr(31) || '\1: ', 'g'), chr(31))) AS c
      FROM vbody)
-- @step vpiece mat
SELECT DeviceId, det, txt, CASE WHEN cat IN {{ CLOSURES|inlist }} THEN idx ELSE 1 END AS idx,
       CASE {% for c, rx, kind in PIECES %}WHEN cat = {{ c|str }} AND regexp_matches(txt, {{ rx|str }}) THEN {{ kind|str }} {% endfor %}END AS kind
FROM (SELECT DeviceId, det, cat, unnest(parts) AS txt, generate_subscripts(parts, 1) AS idx
      FROM (SELECT DeviceId, det, cat, string_split(regexp_replace(txt, {{ HEAD_SPLIT|str }}, chr(31) || '\1', 'g'), chr(31)) AS parts FROM vcat))
{# each period's piece: its own condition's; a silence of a detector rated only for its nights reads the night's piece.
   Lane closures and timing plans: the n-th piece of the category is the n-th period in time, as health_reason lists them #}
-- @step vper mat
SELECT row_number() OVER () AS pid, DeviceId, Detector::BIGINT AS det, "start"::TIMESTAMP AS a, "end"::TIMESTAMP AS z, condition AS what,
       severity::DOUBLE AS sev, whole_window::BOOLEAN AS ww,
       CASE WHEN condition IN {{ CLOSURE_KINDS|inlist }} THEN row_number() OVER (PARTITION BY DeviceId, Detector, condition ORDER BY "start", "end") ELSE 1 END AS idx
FROM @VPER@
-- @step vdet mat
SELECT p.pid, p.DeviceId, p.det, p.a, p.z, p.what, p.sev, p.ww, coalesce(x.txt, b.body, p.what) AS txt
FROM vper p
LEFT JOIN ({{ KINDS|values('what, kind, pref') }}) m ON m.what = p.what
LEFT JOIN vpiece x ON x.DeviceId = p.DeviceId AND x.det = p.det AND x.kind = m.kind AND x.idx = p.idx
LEFT JOIN vbody b ON b.DeviceId = p.DeviceId AND b.det = p.det
QUALIFY row_number() OVER (PARTITION BY p.pid ORDER BY x.txt IS NULL, m.pref) = 1
{# the rows with no time of their own, today's severity missing: borderline findings, config notes, too few to judge,
   signal notes (Detector empty: they speak of the signal) #}
-- @step vnote
SELECT DeviceId, det, part AS lab, {{ WATCH_CONF|dbl }} AS conf, part AS txt
FROM (SELECT DeviceId, det, unnest(string_split(wt, '; ')) AS part FROM vbody WHERE wt <> '')
UNION ALL
SELECT DeviceId, det, CASE {% for rx, lab in VEH_CONFIG %}WHEN regexp_matches(cfg, {{ rx|str }}) THEN {{ lab|str }} {% endfor %}ELSE {{ CONFIG_LABEL|str }} END,
       {{ NOTE_CONF|dbl }}, cfg FROM vbody WHERE cfg <> ''
UNION ALL
SELECT DeviceId, det, {{ TOO_FEW|str }}, {{ NOTE_CONF|dbl }}, body FROM vbody WHERE st = {{ TOO_FEW|str }}
UNION ALL
SELECT DeviceId, NULL::BIGINT, CASE {% for rx, lab in SIGNAL_NOTES %}WHEN regexp_matches(part, {{ rx|str }}) THEN {{ lab|str }} {% endfor %}ELSE {{ SIGNAL_LABEL|str }} END,
       {{ NOTE_CONF|dbl }}, part
FROM (SELECT DISTINCT DeviceId, unnest(string_split(sig, '; ')) AS part FROM vbody WHERE sig <> '')
-- @step vrows
SELECT DeviceId, 'Vehicle' AS DetectorType, det AS Detector, a AS "Start", z AS "End", what AS Label, sev AS Confidence, txt AS Detail
FROM vdet WHERE NOT ww
UNION ALL
SELECT p.DeviceId, 'Vehicle', p.det, d.day_start, d.day_end, p.what, p.sev, p.txt
FROM vdet p JOIN fdays d ON d.DeviceId = p.DeviceId AND d.DetectorType = 'Vehicle' WHERE p.ww
UNION ALL
SELECT n.DeviceId, 'Vehicle', n.det, d.day_start, d.day_end, n.lab, n.conf, n.txt
FROM vnote n JOIN fdays d ON d.DeviceId = n.DeviceId AND d.DetectorType = 'Vehicle'
-- @step vstat
SELECT b.DeviceId, 'Vehicle' AS DetectorType, b.det AS Detector, d.day_date AS "Date", b.st AS Status
FROM vbody b JOIN fdays d ON d.DeviceId = b.DeviceId AND d.DetectorType = 'Vehicle'
