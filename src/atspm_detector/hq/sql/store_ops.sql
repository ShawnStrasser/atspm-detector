{# The statements of an update, a trim and a compaction on a store (incremental.py); each `-- @step` is one statement run
   by name, not a step of a plan.  s: the store's name and a dot. #}
{% import "_lib.sql" as lib with context %}
-- @step meta
CREATE TABLE IF NOT EXISTS {{ s }}health_meta (Key VARCHAR, Value VARCHAR)
{# the chunk's usable rows, read once.  The rows left out are counted per signal (design 3.3): a stamp no nanosecond time can
   hold or none at all, whatever upto says; a good stamp at or after upto.  All codes with ped on (the ped part reads them too), the
   allowed ones alone with it off; the vehicle part and the counts take the allowed ones. #}
{% set stamp_lo, stamp_hi = "TIMESTAMP '1700-01-01'", "TIMESTAMP '2262-01-01'" %}
{% set before_upto %}{% if upto is not none %} AND Timestamp < {{ upto|ts }}{% endif %}{% endset %}
{% set bad_stamp %}coalesce(Timestamp < {{ stamp_lo }} OR Timestamp >= {{ stamp_hi }}, true){% endset %}
-- @step uev
CREATE OR REPLACE TEMP TABLE {{ P }}uev AS
SELECT DeviceId, Timestamp, EventId, Parameter FROM @EVENTS@
WHERE {% if not ped %}{{ lib.good() }} AND {% endif %}DeviceId IS NOT NULL AND Timestamp >= {{ stamp_lo }} AND Timestamp < {{ stamp_hi }}{{ before_upto }}
-- @step unusable
SELECT DeviceId, count(*) FILTER ({{ bad_stamp }}), count(*) FILTER (NOT {{ bad_stamp }}) FROM @EVENTS@
WHERE {{ lib.good() }} AND DeviceId IS NOT NULL AND ({{ bad_stamp }}{% if upto is not none %} OR Timestamp >= {{ upto|ts }}{% endif %})
GROUP BY DeviceId
{# each signal in the chunk: its new events start after its watermark (design 3.3); an event at or before that is late.  e_ns: past
   the chunk's last event.  lo: the oldest stamp still believed, keep_ns before the latest stamp known.  A row older than lo and not
   late is a wrong clock, not processed; a signal whose watermark is older than lo starts over from its first believed row (stale: its
   carry is not read; wm: the watermark to go on from, NULL then). #}
{% macro watermarks(rows, code) %}
WITH e AS ({{ rows }}),
c AS (SELECT DeviceId, epoch_ns(TimeStamp) AS wm FROM {{ s }}health_carry WHERE EventId = {{ code }}),
r AS (SELECT greatest((SELECT max(ns) FROM e), (SELECT max(wm) FROM c)) - {{ keep_ns }} AS lo)
SELECT e.DeviceId, CASE WHEN any_value(c.wm) >= any_value(r.lo) THEN any_value(c.wm) + 1 ELSE min(e.ns) FILTER (e.ns >= r.lo) END AS s_ns,
       max(e.ns) + 1 AS e_ns, count(*) FILTER (e.ns >= r.lo AND (c.wm IS NULL OR e.ns > c.wm)) AS n_new,
       count(*) FILTER (e.ns <= c.wm) AS n_late, count(*) FILTER (e.ns < r.lo AND (c.wm IS NULL OR e.ns > c.wm)) AS n_old,
       coalesce(any_value(c.wm) < any_value(r.lo), false) AS stale,
       CASE WHEN any_value(c.wm) >= any_value(r.lo) THEN any_value(c.wm) END AS wm
FROM e LEFT JOIN c USING (DeviceId) CROSS JOIN r GROUP BY e.DeviceId{% endmacro %}
{# the vehicle part: allowed events, its watermark the last allowed event it processed (9402) #}
-- @step uwall
CREATE OR REPLACE TEMP TABLE {{ P }}uwall AS
{{ watermarks('SELECT DeviceId, epoch_ns(Timestamp) AS ns FROM ' ~ P ~ 'uev WHERE ' ~ lib.good(), 9402) }}
{# the ped part: events of any code, its watermark the last it processed (9401) #}
-- @step uwp
CREATE OR REPLACE TEMP TABLE {{ P }}uwp AS
{{ watermarks('SELECT DeviceId, epoch_ns(Timestamp) AS ns FROM ' ~ P ~ 'uev', 9401) }}
{# trim at the cut C (design 4.6).  A detector's carried history is dropped when all of it lies more than a minute before
   C and no chain is open: every comparison of an event with its neighbours is under a minute (the 0.1-s ticks, 1 s
   between starts), and rows before C are trimmed anyway.  Colour pieces, the watermark and the markers stay. #}
-- @step trim_signal
DELETE FROM {{ s }}health_signal_5min WHERE TimeStamp < {{ cut|ts }}
-- @step trim_detector
DELETE FROM {{ s }}health_detector_5min WHERE TimeStamp < {{ cut|ts }}
-- @step trim_phase
DELETE FROM {{ s }}health_phase_5min WHERE TimeStamp < {{ cut|ts }}
-- @step trim_phase15
DELETE FROM {{ s }}health_phase_15min WHERE TimeStamp < {{ cut|ts }}
-- @step trim_ticks
DELETE FROM {{ s }}health_tick_counts WHERE TimeStamp < {{ cut|ts }}
-- @step trim_intervals
DELETE FROM {{ s }}health_intervals WHERE EndTime < {{ cut|ts }}
-- @step trim_ped
DELETE FROM {{ s }}health_ped_events WHERE TimeStamp < {{ cut|ts }}
-- @step trim_log
DELETE FROM {{ s }}health_log_15min WHERE TimeStamp < {{ cut|ts }}
-- @step trim_carry
DELETE FROM {{ s }}health_carry WHERE EventId IN (9201, 9202, 9221) AND (DeviceId, Parameter) IN (
  SELECT DeviceId, Parameter FROM {{ s }}health_carry WHERE EventId IN (9201, 9202, 9221, 9222) GROUP BY DeviceId, Parameter
  HAVING max(TimeStamp) < {{ cut|ts }} - INTERVAL 1 MINUTE AND count(*) FILTER (EventId = 9222) = 0)
-- @step trim_phases
DELETE FROM {{ s }}health_detector_phase WHERE ValidFrom < {{ cut|ts }} AND (DeviceId, Detector, ValidFrom) NOT IN (
  SELECT DeviceId, Detector, max(ValidFrom) FROM {{ s }}health_detector_phase GROUP BY DeviceId, Detector)
{# compaction: delta rows merged per key and label (design 4.6), integers, so no total changes #}
-- @step compact_signal
SELECT DeviceId, TimeStamp, sum(Events)::INTEGER AS Events, min(First_Event) AS First_Event, max(Last_Event) AS Last_Event
FROM {{ s }}health_signal_5min GROUP BY DeviceId, TimeStamp
-- @step compact_detector
SELECT DeviceId, Detector, TimeStamp, Chain_Tag, Piece_Phase, Piece_Tag, Piece_Kind, {% for c in ('Starts', 'Fast_Starts', 'Chatter_Starts', 'Starts_GY', 'Fast_Starts_R2', 'Chatter_Starts_R2', 'Relogged_Ons', 'Stray_Offs', 'Closing_Offs', 'Clean_Ons', 'Ons_5s', 'Ons_60s') %}sum({{ c }})::INTEGER AS {{ c }}, {% endfor %}{% for c in ('On_Time_ns', 'Unknown_Time_ns', 'On_Green5_ns', 'Relog_Time_ns') %}sum({{ c }})::BIGINT AS {{ c }}, {% endfor %}max(Max_On_s) AS Max_On_s, max(Long_Max_s) AS Long_Max_s
FROM {{ s }}health_detector_5min GROUP BY DeviceId, Detector, TimeStamp, Chain_Tag, Piece_Phase, Piece_Tag, Piece_Kind
-- @step compact_phase
SELECT DeviceId, Phase, TimeStamp, Piece_Tag, sum(Greens)::INTEGER AS Greens, sum(Green_Yellow_ns)::BIGINT AS Green_Yellow_ns, sum(Green5_ns)::BIGINT AS Green5_ns
FROM {{ s }}health_phase_5min GROUP BY DeviceId, Phase, TimeStamp, Piece_Tag
-- @step compact_log
SELECT DeviceId, TimeStamp, min(All_First) AS All_First, max(All_Last) AS All_Last FROM {{ s }}health_log_15min GROUP BY DeviceId, TimeStamp
-- @step compact_phase15
SELECT DeviceId, Phase, TimeStamp, sum(Greens)::INTEGER AS Greens FROM {{ s }}health_phase_15min GROUP BY DeviceId, Phase, TimeStamp
-- @step compact_ticks
SELECT DeviceId, Detector, TimeStamp, Kind, A, B, sum(Count)::INTEGER AS Count FROM {{ s }}health_tick_counts
GROUP BY DeviceId, Detector, TimeStamp, Kind, A, B
