{# The extractor's rows (design 4.1-4.2), keyed by the run's dev and 5-min bin b (quarter-hour q, ns dn) from the origin;
   extract/emit.sql writes them to the store tables (DeviceId / TimeStamp).  A chunk's settled rows have one row per key;
   the store's rows are deltas (design 4.0): readers add them up (finalize's d5 / p5).  Counts and ns are integers, so
   rows of any chunking add up to the same totals.  Rows still depending on an open chain (ctag = its start) or an open
   colour piece (pph, ptag = its phase and start; pk = 0 the green / yellow piece, 2 the red's first 2 s) are kept apart
   and settled by finalize. #}
{% import "_lib.sql" as lib with context %}
{# health_detector_5min: starts and their rule-F / colour counts; re-logged ONs and stray OFFs (a second or leading
   OFF); per closed chain and 5-min bin its time ON / unknown, in its start bin b0 its facts -- clean single ON, the
   longest (a gap chain 0), time of re-logged chains, >= 5 s / >= 60 s, the longest not over a gap (s, today's doubles)
   -- and in its end bin b1 the OFF closing it; green overlap.  Events = starts + re-logged ONs + closing and stray OFFs
   (every ON starts a chain or is re-logged in one, every OFF closes one or is stray).  A chain carried in and closed in
   this chunk (the two parts after the first chains' part): its time from the chunk's cut l0 to its end; its facts in its
   start bin, a key of an earlier chunk (rows are deltas, design 4.0). #}
-- @step xd5
SELECT dev, det, b, sum(nst) AS nst, sum(nfst) AS nfst, sum(nchat) AS nchat, sum(nst_gy) AS nst_gy,
       sum(nfst_r2) AS nfst_r2, sum(nchat_r2) AS nchat_r2, sum(nrel) AS nrel, sum(nxoff) AS nxoff, sum(on_ns) AS on_ns,
       sum(unk_ns) AS unk_ns, sum(nclose) AS nclose, sum(nclean) AS nclean, max(max_on) AS max_on,
       sum(relog_ns) AS relog_ns, sum(n5s) AS n5s, sum(n60s) AS n60s, max(lmax) AS lmax, sum(ong5_ns) AS ong5_ns
FROM (SELECT dev, det, b, nst, nfst, nchat, CASE WHEN ptag IS NULL THEN nst_gy END AS nst_gy,
       CASE WHEN ptag IS NULL THEN nfst_r2 END AS nfst_r2, CASE WHEN ptag IS NULL THEN nchat_r2 END AS nchat_r2
FROM xsa
UNION ALL BY NAME
SELECT dev, det, (dn // {{ NS5 }})::INT AS b, count(*) FILTER (eid = 82) AS nrel, count(*) FILTER (eid = 81) AS nxoff
FROM xde WHERE (pe = 82 OR eid = 81) AND old = 0 GROUP BY ALL
UNION ALL BY NAME
SELECT dev, det, b, sum(t) FILTER (NOT gap) AS on_ns, sum(t) FILTER (gap) AS unk_ns, count(*) FILTER (b = b1) AS nclose,
       count(*) FILTER (b = b0 AND NOT multi AND NOT gap) AS nclean,
       max(CASE WHEN gap THEN 0.0 ELSE dur END) FILTER (b = b0) AS max_on,
       sum(f - s) FILTER (b = b0 AND multi AND NOT gap) AS relog_ns,
       count(*) FILTER (b = b0 AND NOT gap AND dur >= 5) AS n5s, count(*) FILTER (b = b0 AND NOT gap AND dur >= 60) AS n60s,
       max(dur) FILTER (b = b0 AND NOT gap) AS lmax
FROM ({{ lib.split5('SELECT dev, det, s, f, gap, multi, dur FROM xcc WHERE NOT cd', 'dev, det, gap, multi, dur, s, f, b0, b1') }})
GROUP BY ALL
UNION ALL BY NAME
SELECT dev, det, b, sum(t) FILTER (NOT gap) AS on_ns, sum(t) FILTER (gap) AS unk_ns, count(*) FILTER (b = b1) AS nclose
FROM ({{ lib.split5('SELECT c.dev, c.det, l.l0 AS s, c.f, c.gap FROM xcc c JOIN xlast l USING (dev) WHERE c.cd', 'dev, det, gap, b1') }})
GROUP BY ALL
UNION ALL BY NAME
SELECT dev, det, (s // {{ NS5 }})::INT AS b, (NOT multi AND NOT gap)::INT AS nclean,
       CASE WHEN gap THEN 0.0 ELSE dur END AS max_on, CASE WHEN multi AND NOT gap THEN f - s END AS relog_ns,
       (NOT gap AND dur >= 5)::INT AS n5s, (NOT gap AND dur >= 60)::INT AS n60s, CASE WHEN NOT gap THEN dur END AS lmax
FROM xcc WHERE cd
UNION ALL BY NAME
SELECT dev, det, b, t AS ong5_ns
FROM ({{ lib.split5('SELECT dev, det, s, f FROM xov WHERE ctag IS NULL AND ptag IS NULL', 'dev, det') }}))
GROUP BY dev, det, b
{# health_detector_5min, labelled rows: the open chain's time, from the chunk's cut, up to the signal's last allowed
   event (design 3.2 rule 1); starts in an open piece; overlaps with an open chain or green #}
-- @step xd5l
SELECT dev, det, b, ctag, t AS on_ns
FROM ({{ lib.split5('SELECT o.dev, o.det, greatest(o.s, l.l0) AS s, l.lz AS f, o.s AS ctag FROM xoc o JOIN xlast l USING (dev)', 'dev, det, ctag') }})
UNION ALL BY NAME
SELECT dev, det, b, pph, ptag, pk, nst_gy, nfst_r2, nchat_r2 FROM xsa WHERE ptag IS NOT NULL
UNION ALL BY NAME
SELECT dev, det, b, ctag, pph, ptag, 0 AS pk, t AS ong5_ns
FROM ({{ lib.split5('SELECT * FROM xov WHERE ctag IS NOT NULL OR ptag IS NOT NULL', 'dev, det, ctag, pph, ptag') }})
{# health_phase_5min: greens; valid green / yellow time and valid green time after the first 5 s of the pieces closed in
   this chunk, from its cut, split at the 5-min bins #}
-- @step xp5
SELECT dev, ph, b, sum(ngreen) AS ngreen, sum(gy_ns) AS gy_ns, sum(g5_ns) AS g5_ns
FROM (SELECT dev, ph, b, ngreen FROM xgr
UNION ALL BY NAME
SELECT dev, ph, b, CASE WHEN kind = 0 THEN t END AS gy_ns, CASE WHEN kind = 1 THEN t END AS g5_ns
FROM ({{ lib.split5('SELECT p.dev, p.ph, greatest(p.tag, l.l0) AS s, p.z AS f, 0 AS kind FROM xpc p JOIN xlast l USING (dev) WHERE p.st < 2 AND NOT p.op AND p.valid
   UNION ALL SELECT p.dev, p.ph, greatest(p.a5, l.l0), p.z, 1 FROM xg5 p JOIN xlast l USING (dev) WHERE NOT p.op', 'dev, ph, kind') }}))
GROUP BY dev, ph, b
{# its labelled rows: the open piece, from the chunk's cut, up to the signal's last allowed event #}
-- @step xp5l
SELECT dev, ph, b, ptag, CASE WHEN kind = 0 THEN t END AS gy_ns, CASE WHEN kind = 1 THEN t END AS g5_ns
FROM ({{ lib.split5('SELECT p.dev, p.ph, greatest(p.tag, l.l0) AS s, l.lz AS f, 0 AS kind, p.tag AS ptag FROM xpc p JOIN xlast l USING (dev) WHERE p.st < 2 AND p.op
   UNION ALL SELECT p.dev, p.ph, greatest(p.a5, l.l0), l.lz, 1, p.tag FROM xg5 p JOIN xlast l USING (dev) WHERE p.op AND l.lz > p.a5', 'dev, ph, kind, ptag') }})
{# health_tick_counts: on_us (closed single chains not over a gap, length in us, capped at 1 s + 1 us; by the chain
   start's quarter-hour) and gap_pair (ONs with their ticks to the ON before (ta < TFC) and after (tb; TFC = TFC or
   more); the pending last ON apart) #}
-- @step xtko
SELECT dev, det, s // {{ NSQ }} AS q, least((f - s) // 1000, 1000001) AS a, count(*) AS n FROM xcc
WHERE NOT multi AND NOT gap GROUP BY ALL
-- @step xtkg
SELECT dev, det, dn // {{ NSQ }} AS q, ta, tb, count(*) AS n FROM xtf WHERE tb IS NOT NULL GROUP BY ALL
{# health_intervals hold: closed chains of >= 300 s not over a gap #}
-- @step xhold
SELECT dev, det, s, f, multi FROM xcc WHERE dur >= 300 AND NOT gap
{# health_intervals, exception rows (design 3.2 rule 3): a chain carried in that closed over a comms gap, a colour piece
   carried in (its green / yellow, or a red's first 2 s) that closed invalid: their labelled rows of earlier chunks count
   as such.  A chain or piece opened and closed in this chunk has no labelled rows. #}
-- @step xbad
SELECT dev, 'bad_chain' AS Kind, det AS ch, s, f, NULL::INT AS val FROM xcc WHERE cd AND gap
UNION ALL
SELECT p.dev, 'bad_piece', p.ph, p.tag, p.z, CASE WHEN p.st < 2 THEN 0 ELSE 2 END
FROM xpc p JOIN xlast l USING (dev) WHERE NOT p.op AND p.st < 3 AND NOT p.valid AND p.tag <= l.l0
