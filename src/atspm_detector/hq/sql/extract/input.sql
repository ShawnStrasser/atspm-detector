{# The extractor's input (design 3.1, 6): the chunk's allowed events, read once, de-duplicated, and what the previous
   chunks left for it (xcar: the carry, design 4.4).  dn = ns from the run's origin ORG: each signal's window start
   (bin_origin "window": today's origin), or the epoch (bin_origin "clock" and every store: one origin for any window
   and any chunking, so a double expression on a time always gives the same bits).
   Batch health() is one chunk per signal: its window's events, nothing carried in. #}
{% import "_lib.sql" as lib with context %}
-- @step xev mat
SELECT @DISTINCT@ w.dev, epoch_ns(e.Timestamp) - {{ ORG }} AS dn, e.EventId::INT AS eid, e.Parameter::INT AS par
FROM @EVENTS@ e JOIN win w ON e.DeviceId = w.DeviceId
WHERE {{ lib.good('e.') }}
  AND e.Timestamp >= w.lo AND e.Timestamp <= w.hi
  AND epoch_ns(e.Timestamp) >= w.s_ns AND epoch_ns(e.Timestamp) < w.e_ns
{# the carry's rows of this chunk's signals; cid: its code (design 4.4) #}
-- @step xcar mat
SELECT w.dev, epoch_ns(c.TimeStamp) - {{ ORG }} AS dn, c.EventId::INT AS cid, c.Parameter::INT AS par
FROM @CARRY@ c JOIN win w ON c.DeviceId = w.DeviceId
