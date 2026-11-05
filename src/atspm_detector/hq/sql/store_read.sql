{# The store's rows inside each signal's window, read back with the run's keys (design 5): dev, bin numbers and ns from
   its window start, for finalize (rd5, rd5l, rp5, rp5l, rtkg, rbad) and the adapters (rs5, rtko, rhold, rgq, rplan).
   store: the store's name and a dot (empty in a batch run: its temp tables). #}
{% macro at(x) %}(epoch_ns({{ x }}) - w.s_ns){% endmacro %}
{% macro inw(x) %}{{ x }} >= make_timestamp_ns(w.s_ns) AND {{ x }} < make_timestamp_ns(w.e_ns){% endmacro %}
-- @step rs5
SELECT w.dev, ({{ at('r.TimeStamp') }} // {{ NS5 }})::INT AS b FROM {{ store }}health_signal_5min r JOIN win w USING (DeviceId)
WHERE {{ inw('r.TimeStamp') }} GROUP BY ALL
-- @step rd5
SELECT w.dev, r.Detector::INT AS det, ({{ at('r.TimeStamp') }} // {{ NS5 }})::INT AS b, r.Starts AS nst,
       r.Fast_Starts AS nfst, r.Chatter_Starts AS nchat, r.Starts_GY AS nst_gy, r.Fast_Starts_R2 AS nfst_r2,
       r.Chatter_Starts_R2 AS nchat_r2, r.Relogged_Ons AS nrel, r.Stray_Offs AS nxoff, r.Closing_Offs AS nclose,
       r.On_Time_ns AS on_ns, r.Unknown_Time_ns AS unk_ns, r.On_Green5_ns AS ong5_ns, r.Clean_Ons AS nclean,
       r.Max_On_s AS max_on, r.Relog_Time_ns AS relog_ns, r.Ons_5s AS n5s, r.Ons_60s AS n60s, r.Long_Max_s AS lmax
FROM {{ store }}health_detector_5min r JOIN win w USING (DeviceId)
WHERE r.Chain_Tag IS NULL AND r.Piece_Tag IS NULL AND {{ inw('r.TimeStamp') }}
-- @step rd5l
SELECT w.dev, r.Detector::INT AS det, ({{ at('r.TimeStamp') }} // {{ NS5 }})::INT AS b, {{ at('r.Chain_Tag') }} AS ctag,
       r.Piece_Phase AS pph, {{ at('r.Piece_Tag') }} AS ptag, r.Piece_Kind::INT AS pk, r.On_Time_ns AS on_ns,
       r.On_Green5_ns AS ong5_ns, r.Starts_GY AS nst_gy, r.Fast_Starts_R2 AS nfst_r2, r.Chatter_Starts_R2 AS nchat_r2
FROM {{ store }}health_detector_5min r JOIN win w USING (DeviceId)
WHERE (r.Chain_Tag IS NOT NULL OR r.Piece_Tag IS NOT NULL) AND {{ inw('r.TimeStamp') }}
-- @step rp5
SELECT w.dev, r.Phase AS ph, ({{ at('r.TimeStamp') }} // {{ NS5 }})::INT AS b, r.Greens AS ngreen, r.Green_Yellow_ns AS gy_ns,
       r.Green5_ns AS g5_ns
FROM {{ store }}health_phase_5min r JOIN win w USING (DeviceId) WHERE r.Piece_Tag IS NULL AND {{ inw('r.TimeStamp') }}
-- @step rp5l
SELECT w.dev, r.Phase AS ph, ({{ at('r.TimeStamp') }} // {{ NS5 }})::INT AS b, {{ at('r.Piece_Tag') }} AS ptag,
       r.Green_Yellow_ns AS gy_ns, r.Green5_ns AS g5_ns
FROM {{ store }}health_phase_5min r JOIN win w USING (DeviceId) WHERE r.Piece_Tag IS NOT NULL AND {{ inw('r.TimeStamp') }}
{# the clock hours with a green; the rows are per clock quarter-hour, so the first may start before a window start that
   lies inside a quarter-hour (a batch run on today's origin; all its rows are inside the window) #}
-- @step rgq
SELECT w.dev, r.Phase AS ph, epoch_ns(r.TimeStamp) // {{ NSH }} AS h FROM {{ store }}health_phase_15min r JOIN win w USING (DeviceId)
WHERE r.TimeStamp >= make_timestamp_ns(w.s_ns - w.s_ns % {{ NSQ }}) AND r.TimeStamp < make_timestamp_ns(w.e_ns) GROUP BY ALL
-- @step rtko
SELECT w.dev, r.Detector::INT AS det, r.A AS a, r.Count AS n FROM {{ store }}health_tick_counts r JOIN win w USING (DeviceId)
WHERE r.Kind = 'on_us' AND {{ inw('r.TimeStamp') }}
-- @step rtkg
SELECT w.dev, r.Detector::INT AS det, {{ at('r.TimeStamp') }} // {{ NSQ }} AS q, r.A AS ta, r.B::INT AS tb, r.Count AS n
FROM {{ store }}health_tick_counts r JOIN win w USING (DeviceId) WHERE r.Kind = 'gap_pair' AND {{ inw('r.TimeStamp') }}
-- @step rhold
SELECT w.dev, r.Channel AS det, {{ at('r.StartTime') }} AS s, {{ at('r.EndTime') }} AS f
FROM {{ store }}health_intervals r JOIN win w USING (DeviceId)
WHERE r.Kind = 'long_on' AND r.EndTime >= make_timestamp_ns(w.s_ns) AND r.StartTime < make_timestamp_ns(w.e_ns)
-- @step rplan
SELECT w.dev, {{ at('r.StartTime') }} AS dn, r.Value AS par FROM {{ store }}health_intervals r JOIN win w USING (DeviceId)
WHERE r.Kind = 'plan' AND {{ inw('r.StartTime') }}
{# the exception rows (design 3.2 rule 3): chains and colour pieces that closed bad after labelled rows were written #}
-- @step rbad mat
SELECT w.dev, r.Kind AS k, r.Channel AS ch, {{ at('r.StartTime') }} AS s, r.Value AS pk
FROM {{ store }}health_intervals r JOIN win w USING (DeviceId)
WHERE r.Kind IN ('bad_chain', 'bad_piece') AND r.EndTime >= make_timestamp_ns(w.s_ns)
