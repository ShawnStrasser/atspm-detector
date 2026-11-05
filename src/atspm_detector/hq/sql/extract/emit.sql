{# The extractor writes the store tables (design 4; store.sql has their layout): the rows of extract/rows.sql with the run's
   keys (dev, bin numbers, ns from the origin) made the store's (DeviceId, TIMESTAMP_NS).  A step marked @emit holds rows
   of that store table: a temp table in a batch run, a CTE in health_sql(), and in an update a temp table written to the
   attached store in the update's one transaction. #}
{# time ns from the origin, as a TIMESTAMP_NS #}
{% macro t(x) %}make_timestamp_ns({{ ORG }} + {{ x }}){% endmacro %}
-- @step health_signal_5min
-- @emit health_signal_5min
SELECT w.DeviceId, {{ t('r.b * ' ~ NS5) }} AS TimeStamp, r.n::INTEGER AS Events, {{ t('r.a') }} AS First_Event,
       {{ t('r.z') }} AS Last_Event
FROM xs5 r JOIN win w USING (dev)
-- @step health_detector_5min mat
-- @emit health_detector_5min
SELECT w.DeviceId, r.det::SMALLINT AS Detector, {{ t('r.b * ' ~ NS5) }} AS TimeStamp, {{ t('r.ctag') }} AS Chain_Tag,
       r.pph AS Piece_Phase, {{ t('r.ptag') }} AS Piece_Tag, r.pk::SMALLINT AS Piece_Kind,
       r.nst::INTEGER AS Starts, r.nfst::INTEGER AS Fast_Starts, r.nchat::INTEGER AS Chatter_Starts,
       r.nst_gy::INTEGER AS Starts_GY, r.nfst_r2::INTEGER AS Fast_Starts_R2, r.nchat_r2::INTEGER AS Chatter_Starts_R2,
       r.nrel::INTEGER AS Relogged_Ons, r.nxoff::INTEGER AS Stray_Offs, r.nclose::INTEGER AS Closing_Offs,
       r.on_ns::BIGINT AS On_Time_ns, r.unk_ns::BIGINT AS Unknown_Time_ns, r.ong5_ns::BIGINT AS On_Green5_ns,
       r.nclean::INTEGER AS Clean_Ons, r.max_on AS Max_On_s, r.relog_ns::BIGINT AS Relog_Time_ns, r.n5s::INTEGER AS Ons_5s,
       r.n60s::INTEGER AS Ons_60s, r.lmax AS Long_Max_s
FROM (SELECT * FROM xd5 UNION ALL BY NAME SELECT * FROM xd5l) r JOIN win w USING (dev)
-- @step health_phase_5min mat
-- @emit health_phase_5min
SELECT w.DeviceId, r.ph AS Phase, {{ t('r.b * ' ~ NS5) }} AS TimeStamp, {{ t('r.ptag') }} AS Piece_Tag,
       r.ngreen::INTEGER AS Greens, r.gy_ns::BIGINT AS Green_Yellow_ns, r.g5_ns::BIGINT AS Green5_ns
FROM (SELECT * FROM xp5 UNION ALL BY NAME SELECT * FROM xp5l) r JOIN win w USING (dev)
-- @step health_phase_15min
-- @emit health_phase_15min
SELECT w.DeviceId, r.ph AS Phase, make_timestamp_ns(r.q * {{ NSQ }}) AS TimeStamp, r.ngreen::INTEGER AS Greens
FROM xgq r JOIN win w USING (dev)
-- @step health_tick_counts mat
-- @emit health_tick_counts
SELECT w.DeviceId, r.det::SMALLINT AS Detector, {{ t('r.q * ' ~ NSQ) }} AS TimeStamp, r.Kind, r.a::INTEGER AS A,
       r.tb::SMALLINT AS B, r.n::INTEGER AS Count
FROM (SELECT dev, det, q, 'on_us' AS Kind, a, NULL::INT AS tb, n FROM xtko
      UNION ALL SELECT dev, det, q, 'gap_pair', ta, tb, n FROM xtkg) r JOIN win w USING (dev)
-- @step health_intervals mat
-- @emit health_intervals
SELECT w.DeviceId, r.Kind, r.ch::INTEGER AS Channel, {{ t('r.s') }} AS StartTime, {{ t('r.f') }} AS EndTime,
       r.multi AS Relogged, r.val::INTEGER AS Value
FROM (SELECT dev, 'long_on' AS Kind, det AS ch, s, f, multi, NULL::INT AS val FROM xhold
      UNION ALL SELECT dev, 'comms_gap', NULL, g0, g1, NULL, NULL FROM xgap WHERE g1 IS NOT NULL
      UNION ALL SELECT dev, 'plan', NULL, dn, dn, NULL, par FROM xplan
      UNION ALL SELECT dev, Kind, ch, s, f, NULL, val FROM xbad) r JOIN win w USING (dev)
