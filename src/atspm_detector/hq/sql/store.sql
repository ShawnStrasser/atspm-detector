{# The store tables (design 4.1, 4.2, 4.3, 4.4, 4.5; a batch run keeps the first six in temp tables, extract/emit.sql).
   Rows are deltas: a key may come in several rows, readers settle the labels and add them up.  Times are TIMESTAMP_NS,
   durations BIGINT ns (_ns), counts INTEGER; only the two per-chain maxima (_s) are today's DOUBLE seconds (a maximum does
   not depend on the chunking).  Context: device_type (the input's DeviceId type, recorded in the store), s (the store's
   name and a dot, empty for the current database). #}
{% set s = s|default('') %}
{# the bins a judge has released, for the horizon: which it must not write again #}
CREATE TABLE IF NOT EXISTS {{ s }}health_bins_released (
    TimeStamp TIMESTAMP, DeviceId {{ device_type }}, DetectorType VARCHAR, Detector BIGINT, Label VARCHAR, Score DOUBLE);
CREATE TABLE IF NOT EXISTS {{ s }}health_signal_5min (
    DeviceId {{ device_type }}, TimeStamp TIMESTAMP_NS, Events INTEGER, First_Event TIMESTAMP_NS, Last_Event TIMESTAMP_NS);
CREATE TABLE IF NOT EXISTS {{ s }}health_detector_5min (
    DeviceId {{ device_type }}, Detector SMALLINT, TimeStamp TIMESTAMP_NS,
    Chain_Tag TIMESTAMP_NS, Piece_Phase INTEGER, Piece_Tag TIMESTAMP_NS, Piece_Kind SMALLINT,
    Starts INTEGER, Fast_Starts INTEGER, Chatter_Starts INTEGER, Starts_GY INTEGER, Fast_Starts_R2 INTEGER,
    Chatter_Starts_R2 INTEGER, Relogged_Ons INTEGER, Stray_Offs INTEGER, Closing_Offs INTEGER,
    On_Time_ns BIGINT, Unknown_Time_ns BIGINT, On_Green5_ns BIGINT, Clean_Ons INTEGER, Max_On_s DOUBLE,
    Relog_Time_ns BIGINT, Ons_5s INTEGER, Ons_60s INTEGER, Long_Max_s DOUBLE);
CREATE TABLE IF NOT EXISTS {{ s }}health_phase_5min (
    DeviceId {{ device_type }}, Phase INTEGER, TimeStamp TIMESTAMP_NS, Piece_Tag TIMESTAMP_NS,
    Greens INTEGER, Green_Yellow_ns BIGINT, Green5_ns BIGINT);
CREATE TABLE IF NOT EXISTS {{ s }}health_phase_15min (
    DeviceId {{ device_type }}, Phase INTEGER, TimeStamp TIMESTAMP_NS, Greens INTEGER);
CREATE TABLE IF NOT EXISTS {{ s }}health_tick_counts (
    DeviceId {{ device_type }}, Detector SMALLINT, TimeStamp TIMESTAMP_NS, Kind VARCHAR, A INTEGER, B SMALLINT, Count INTEGER);
CREATE TABLE IF NOT EXISTS {{ s }}health_intervals (
    DeviceId {{ device_type }}, Kind VARCHAR, Channel INTEGER, StartTime TIMESTAMP_NS, EndTime TIMESTAMP_NS,
    Relogged BOOLEAN, Value INTEGER);
CREATE TABLE IF NOT EXISTS {{ s }}health_carry (
    TimeStamp TIMESTAMP_NS, DeviceId {{ device_type }}, EventId SMALLINT, Parameter INTEGER, IsValid BOOLEAN);
CREATE TABLE IF NOT EXISTS {{ s }}health_detector_phase (
    DeviceId {{ device_type }}, Detector SMALLINT, Phase INTEGER, ValidFrom TIMESTAMP_NS);
CREATE TABLE IF NOT EXISTS {{ s }}health_ped_events (
    DeviceId {{ device_type }}, TimeStamp TIMESTAMP_NS, EventId SMALLINT, Parameter INTEGER);
CREATE TABLE IF NOT EXISTS {{ s }}health_log_15min (
    DeviceId {{ device_type }}, TimeStamp TIMESTAMP_NS, All_First TIMESTAMP_NS, All_Last TIMESTAMP_NS);
CREATE TABLE IF NOT EXISTS {{ s }}health_active (
    DeviceId {{ device_type }}, DetectorType VARCHAR, Detector INTEGER, Label VARCHAR, "Start" TIMESTAMP, "End" TIMESTAMP,
    Confidence DOUBLE, State VARCHAR, FirstSeen TIMESTAMP, LastSeen TIMESTAMP, Missed INTEGER);
