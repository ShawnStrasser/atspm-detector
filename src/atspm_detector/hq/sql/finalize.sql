{# Finalize (design 3.1, 5): what is still open at each signal's window end e_ns -- chains without their OFF, colour
   pieces, the last ON's too-fast pair -- closed there as today's steps close them, the labelled rows settled by those
   verdicts and the exception rows (3.2 rule 3), and the time from the signal's last allowed event to the window end
   (3.2 rule 1).  Nothing here is stored.  It reads the store's rows (store_read.sql) and the open objects: oxc, oxp, oxl,
   oxt (open_extract.sql in a batch run, open_carry.sql from a store), with ns from the window start for the keys and
   today's double seconds on the run's origin for the decisions (sd, ad, mkd, lzd, TE: the window end).  Then the settled
   rows by key: d5 (detectors), p5 (phases), tkg (gap pairs). #}
{% import "_lib.sql" as lib with context %}
{# open chains: their length to the window end (today's dur = T - s), lost (the signal silent > 120 s before the window
   end) or over a comms gap (> 60 s with a gap starting inside: mk) = gap #}
-- @step fch mat
SELECT c.dev, c.det, c.s, {{ TE }} - c.sd AS dur,
       {{ TE }} - l.lzd > 120.0 OR ({{ TE }} - c.sd > 60 AND c.mk IS NOT NULL) AS gap
FROM oxc c JOIN win w USING (dev) JOIN oxl l USING (dev)
{# open colour pieces: valid = no comms gap starting in it (the green / yellow piece to the window end; the red's
   first 2 s to least(start + 2 s, window end)).  Every row after an open green's first 5 s (below and in the extractor)
   starts before the window end, so that green is longer than 5 s; there is nothing more to check. #}
-- @step fpc mat
SELECT p.dev, p.ph, p.st, p.tag, CASE WHEN p.st < 2 THEN 0 ELSE 2 END AS pk,
       NOT coalesce(p.mkd < CASE WHEN p.st = 2 THEN least(p.ad + 2.0, {{ TE }}) ELSE {{ TE }} END, false) AS valid
FROM oxp p JOIN win w USING (dev)
{# labelled rows settled: time of a chain over a gap is unknown time; counts and time of an invalid piece drop out.  The
   chain: open at the window end (fch), closed bad (rbad), or else closed good; the piece likewise (fpc, rbad) #}
-- @step fdl
SELECT dev, det, b, CASE WHEN NOT gap THEN on_ns END AS on_ns, CASE WHEN gap THEN on_ns END AS unk_ns,
       CASE WHEN ok THEN nst_gy END AS nst_gy, CASE WHEN ok THEN nfst_r2 END AS nfst_r2,
       CASE WHEN ok THEN nchat_r2 END AS nchat_r2, CASE WHEN NOT gap AND ok THEN ong5_ns END AS ong5_ns
FROM (SELECT l.*, coalesce(c.gap, false) OR bc.s IS NOT NULL AS gap, coalesce(p.valid, true) AND bp.s IS NULL AS ok
      FROM rd5l l
      LEFT JOIN fch c ON c.dev = l.dev AND c.det = l.det AND c.s = l.ctag
      LEFT JOIN fpc p ON p.dev = l.dev AND p.ph = l.pph AND p.tag = l.ptag AND p.pk = l.pk
      LEFT JOIN (SELECT * FROM rbad WHERE k = 'bad_chain') bc ON bc.dev = l.dev AND bc.ch = l.det AND bc.s = l.ctag
      LEFT JOIN (SELECT * FROM rbad WHERE k = 'bad_piece') bp ON bp.dev = l.dev AND bp.ch = l.pph AND bp.s = l.ptag AND bp.pk = l.pk)
{# open chains from the signal's last allowed event to the window end, split at the 5-min bins #}
-- @step fct
{{ lib.split5('SELECT c.dev, c.det, l.lz AS s, w.e_ns - w.s_ns AS f, c.gap FROM fch c JOIN oxl l USING (dev) JOIN win w USING (dev)', 'dev, det, gap', true) }}
{# open greens from the last allowed event (green / yellow) or the green's 5th second (green overlap) to the window end;
   open chains not over a gap inside them (green overlap) #}
-- @step fpt
{{ lib.split5('SELECT p.dev, p.ph, l.lz AS s, w.e_ns - w.s_ns AS f, 0 AS kind FROM fpc p JOIN oxl l USING (dev) JOIN win w USING (dev) WHERE p.st < 2 AND p.valid
   UNION ALL SELECT p.dev, p.ph, greatest(l.lz, p.tag + 5000000000), w.e_ns - w.s_ns, 1 FROM fpc p JOIN oxl l USING (dev) JOIN win w USING (dev)
   WHERE p.st = 0 AND p.valid AND greatest(l.lz, p.tag + 5000000000) < w.e_ns - w.s_ns', 'dev, ph, kind', true) }}
-- @step fov
{{ lib.split5('SELECT c.dev, c.det, greatest(l.lz, p.tag + 5000000000) AS s, w.e_ns - w.s_ns AS f FROM fch c JOIN xovd d USING (dev, det)
   JOIN fpc p ON p.dev = c.dev AND p.ph = d.ph JOIN oxl l ON l.dev = c.dev JOIN win w ON w.dev = c.dev
   WHERE NOT c.gap AND p.st = 0 AND p.valid AND greatest(l.lz, p.tag + 5000000000) < w.e_ns - w.s_ns', 'dev, det', true) }}
{# settled detector rows by key (health_detector_5min after settling, plus finalize); an open chain's facts in its start
   bin, or the window's first when it started before the window #}
-- @step d5 mat
SELECT dev, det, b, (coalesce(sum(nst), 0) + coalesce(sum(nrel), 0) + coalesce(sum(nclose), 0)
                     + coalesce(sum(nxoff), 0))::BIGINT AS nev,
       (coalesce(sum(nst), 0) + coalesce(sum(nrel), 0))::BIGINT AS non, sum(nrel)::BIGINT AS nrel,
       sum(nst)::BIGINT AS nst, sum(nfst)::BIGINT AS nfst, sum(nchat)::BIGINT AS nchat, sum(nst_gy)::BIGINT AS nst_gy,
       sum(nfst_r2)::BIGINT AS nfst_r2, sum(nchat_r2)::BIGINT AS nchat_r2, sum(on_ns)::BIGINT AS on_ns,
       sum(unk_ns)::BIGINT AS unk_ns, sum(ong5_ns)::BIGINT AS ong5_ns, sum(nclean)::BIGINT AS nclean,
       max(max_on) AS max_on, sum(relog_ns)::BIGINT AS relog_ns, sum(n5s)::BIGINT AS n5s, sum(n60s)::BIGINT AS n60s,
       max(lmax) AS lmax
FROM (SELECT * FROM rd5
      UNION ALL BY NAME SELECT * FROM fdl
      UNION ALL BY NAME
      SELECT dev, det, b, CASE WHEN NOT gap THEN t END AS on_ns, CASE WHEN gap THEN t END AS unk_ns FROM fct
      UNION ALL BY NAME
      SELECT dev, det, (greatest(s, 0) // {{ NS5 }})::INT AS b, (NOT gap AND dur >= 5)::INT AS n5s, (NOT gap AND dur >= 60)::INT AS n60s,
             CASE WHEN NOT gap THEN dur END AS lmax FROM fch
      UNION ALL BY NAME
      SELECT dev, det, b, t AS ong5_ns FROM fov)
GROUP BY dev, det, b
{# settled phase rows by key #}
-- @step p5 mat
SELECT dev, ph, b, sum(ngreen)::BIGINT AS ngreen, sum(gy_ns)::BIGINT AS gy_ns, sum(g5_ns)::BIGINT AS g5_ns
FROM (SELECT * FROM rp5
      UNION ALL BY NAME
      SELECT l.dev, l.ph, l.b, l.gy_ns, l.g5_ns FROM rp5l l
      LEFT JOIN fpc p ON p.dev = l.dev AND p.ph = l.ph AND p.tag = l.ptag AND p.pk = 0
      LEFT JOIN (SELECT * FROM rbad WHERE k = 'bad_piece' AND pk = 0) bp ON bp.dev = l.dev AND bp.ch = l.ph AND bp.s = l.ptag
      WHERE coalesce(p.valid, true) AND bp.s IS NULL
      UNION ALL BY NAME
      SELECT dev, ph, b, CASE WHEN kind = 0 THEN t END AS gy_ns, CASE WHEN kind = 1 THEN t END AS g5_ns FROM fpt)
GROUP BY dev, ph, b
{# gap pairs, the pending last ON of each detector with "no next ON" (9) #}
-- @step tkg mat
SELECT dev, det, q, ta, tb, n FROM rtkg
UNION ALL SELECT dev, det, dn // {{ NSQ }}, ta, 9, 1 FROM oxt
