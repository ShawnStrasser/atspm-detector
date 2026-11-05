{# The adapters (design 5): today's step names and columns rebuilt from the settled store rows (d5, p5, tkg, rtko,
   rhold, fch, rs5, rgq, rplan) of each signal's window.  Each is a macro, called where today's step stood, so the judge
   around it is unchanged.  Integer ns become seconds here (/ 1e9), where today's steps work in seconds.
   Import with {% import "adapters.sql" as ad with context %}. #}
{# covered 5-min bins (any allowed event) #}
{% macro cov5() %}
-- @step cov5 mat
SELECT dev, b FROM rs5
{% endmacro %}
{# detectors with events (81 / 82) in the window #}
{% macro dets() %}
-- @step dets mat
SELECT dev, det FROM d5 GROUP BY dev, det HAVING sum(nev) > 0
{% endmacro %}
{# 5-min ON counts; time ON of chains not over a gap; time of unknown state (note 154q) #}
{% macro bins() %}
-- @step occ5
SELECT dev, det, b::BIGINT AS b, on_ns / 1e9 AS occ FROM d5 WHERE on_ns IS NOT NULL
-- @step unk5
SELECT dev, det, b::BIGINT AS b, unk_ns / 1e9 AS occ FROM d5 WHERE unk_ns IS NOT NULL
-- @step n5
SELECT dev, det, b, non AS n FROM d5 WHERE non > 0
{% endmacro %}
{# continuous ONs held >= 5 min, not over a comms gap: closed holds, and the open chain held to the window end
   (open_end); s, f, dur in seconds as today (the same doubles) #}
{% macro hold() %}
-- @step hold mat
SELECT h.*, row_number() OVER (ORDER BY h.dev, h.det, h.s) AS hid FROM (
  SELECT dev, det, s / 1e9 AS s, f / 1e9 AS f, false AS open_end, f / 1e9 - s / 1e9 AS dur FROM rhold
  UNION ALL
  SELECT c.dev, c.det, c.s / 1e9, w.T, true, c.dur FROM fch c JOIN win w USING (dev) WHERE NOT c.gap AND c.dur >= 300) h
{% endmacro %}
{# the median length of the closed single chains not over a gap (on_us tick counts: exact up to 1 s), their number np
   and their share lasting PULSE s or less (rounded to the ms, as today) #}
{% macro onmed() %}
-- @step onmed mat
SELECT dev, det, sum(n)::BIGINT AS np, (sum(n) FILTER (round(a / 1e6, 3) <= {{ PULSE['s']|dbl }}))::BIGINT AS nps,
       (min(a) FILTER (cn > (nn - 1) // 2) + min(a) FILTER (cn > nn // 2)) / 2e6 AS med
FROM (SELECT dev, det, a, n, sum(n) OVER (PARTITION BY dev, det ORDER BY a ROWS UNBOUNDED PRECEDING) AS cn,
             sum(n) OVER (PARTITION BY dev, det) AS nn
      FROM (SELECT dev, det, a, sum(n) AS n FROM rtko GROUP BY dev, det, a))
GROUP BY dev, det
{% endmacro %}
{# per-ON statistics (today's act): ONs, re-logged ONs, chains (n_start), the longest closed chain (a gap chain 0), time
   of closed re-logged chains, the median clean ON, chains >= 5 s / >= 60 s and the longest, open ones included #}
{% macro act() %}
-- @step act mat
SELECT a.dev, a.det, a.n_on_a, a.n_rep, a.n_rep / a.n_on_a AS rep_frac, a.n_start, a.max_on_s, a.rep_time_s,
       CASE WHEN m.np >= 5 THEN m.med END AS dur_p50, a.n_ge5, a.n_ge60, a.long_max_s
FROM (SELECT dev, det, sum(non)::BIGINT AS n_on_a, coalesce(sum(nrel), 0)::BIGINT AS n_rep, sum(nst)::BIGINT AS n_start,
             max(max_on) AS max_on_s, coalesce(sum(relog_ns), 0) / 1e9 AS rep_time_s,
             coalesce(sum(n5s), 0)::BIGINT AS n_ge5, coalesce(sum(n60s), 0)::BIGINT AS n_ge60, max(lmax) AS long_max_s
      FROM d5 GROUP BY dev, det HAVING sum(non) > 0) a
LEFT JOIN onmed m USING (dev, det)
{% endmacro %}
{# too fast (today's tfo / tf): the limit per detector (0.8 s / lanes for stop-bar zones, else 0.6 s / lanes), every
   ON (N), the ONs under it, and of those with an ON after, the share whose next gap is not under it; per hour for the
   periods (tfh, today's fh) #}
{% set TF = LJ['too_fast'] %}
{% macro tf() %}
-- @step tflim mat
SELECT dev, det, (CASE WHEN fn IN {{ TF['stopbar']|inlist }} THEN {{ TF['gap_s_stopbar']|dbl }}
                       ELSE {{ TF['gap_s_other']|dbl }} END) / ln1 AS lim FROM dz
-- @step tf
SELECT o.dev, o.det, o.N, coalesce(k.nf, 0)::BIGINT AS n_fast, 1000.0 * coalesce(k.nf, 0)::BIGINT / greatest(o.N, 1) AS fast_rate,
       z.lim AS gap_lim, CASE WHEN k.fx > 0 THEN coalesce(k.px, 0)::BIGINT / k.fx::BIGINT END AS pair_share
FROM (SELECT dev, det, sum(non)::BIGINT AS N FROM d5 GROUP BY dev, det HAVING sum(non) > 0) o
JOIN tflim z USING (dev, det)
LEFT JOIN (SELECT k.dev, k.det, sum(k.n) FILTER (k.ta < z.lim * 10) AS nf,
                  sum(k.n) FILTER (k.ta < z.lim * 10 AND k.tb < 9) AS fx,
                  sum(k.n) FILTER (k.ta < z.lim * 10 AND k.tb < 9 AND NOT k.tb < z.lim * 10) AS px
           FROM tkg k JOIN tflim z USING (dev, det) GROUP BY k.dev, k.det) k USING (dev, det)
{% endmacro %}
{% macro tfh() %}
-- @step tfh
SELECT dev, det, h, sum(cf)::BIGINT AS cf, sum(ca)::BIGINT AS ca FROM (
  SELECT k.dev, k.det, k.q // 4 AS h, sum(k.n) FILTER (k.ta < z.lim * 10) AS cf, 0 AS ca
  FROM tkg k JOIN tflim z USING (dev, det) GROUP BY k.dev, k.det, k.q // 4
  UNION ALL
  SELECT dev, det, b // 12, 0, sum(non) FROM d5 GROUP BY dev, det, b // 12)
GROUP BY dev, det, h HAVING sum(ca) > 0
{% endmacro %}
{# phases with >= 3 greens in the window; their valid green + yellow time per 5-min bin (s) #}
{% macro phases() %}
-- @step phs3 mat
SELECT dev, ph FROM p5 GROUP BY dev, ph HAVING sum(ngreen) >= 3
-- @step sgyb mat
SELECT p.dev, p.ph, p.b::BIGINT AS b, p.gy_ns / 1e9 AS sgy FROM p5 p JOIN phs3 USING (dev, ph) WHERE p.gy_ns IS NOT NULL
{% endmacro %}
{# vehicle starts per 5-min bin, those in a valid green / yellow of the detector's phase (phases with >= 3 greens only,
   today's sst) #}
{% macro q5b() %}
-- @step q5b mat
SELECT d.dev, d.det, d.b, d.nst AS n, CASE WHEN p.ph IS NOT NULL THEN coalesce(d.nst_gy, 0) ELSE 0 END::BIGINT AS ngy, z.ln1
FROM d5 d JOIN dz z USING (dev, det) LEFT JOIN phs3 p ON p.dev = z.dev AND p.ph = z.ph WHERE d.nst > 0
{% endmacro %}
{# rule F per 15 minutes: fast starts (within 1 s of the start before, not chatter) and starts less chatter, both
   without those in the first 2 s of a valid red (phases with >= 3 greens only, today's fe) #}
{% macro fe15() %}
-- @step fe15
SELECT d.dev, d.det, d.b // 3 AS g,
       (sum(d.nfst) - CASE WHEN any_value(p.ph) IS NOT NULL THEN coalesce(sum(d.nfst_r2), 0) ELSE 0 END)::BIGINT AS x,
       (sum(d.nst) - (coalesce(sum(d.nchat), 0) - CASE WHEN any_value(p.ph) IS NOT NULL THEN coalesce(sum(d.nchat_r2), 0) ELSE 0 END))::BIGINT AS n
FROM d5 d JOIN dz z USING (dev, det) LEFT JOIN phs3 p ON p.dev = z.dev AND p.ph = z.ph WHERE d.nst > 0
GROUP BY d.dev, d.det, d.b // 3
{% endmacro %}
{# note 135: time ON of the held-ON Count zones (hicd) inside their phase's valid greens after the first 5 s (gon) and
   that green time (gav), per occupancy bin k #}
{% macro gonb() %}
-- @step gonb
SELECT h.dev, h.det, g.k::BIGINT AS k, o.gon, g.gav
FROM hicd h JOIN (SELECT p.dev, p.ph, p.b // (w.bs // 300) AS k, sum(p.g5_ns) / 1e9 AS gav FROM p5 p JOIN win w USING (dev)
                  WHERE p.g5_ns IS NOT NULL GROUP BY p.dev, p.ph, p.b // (w.bs // 300)) g ON g.dev = h.dev AND g.ph = h.ph
LEFT JOIN (SELECT d.dev, d.det, d.b // (w.bs // 300) AS k, sum(d.ong5_ns) / 1e9 AS gon FROM d5 d JOIN win w USING (dev)
           WHERE d.ong5_ns IS NOT NULL GROUP BY d.dev, d.det, d.b // (w.bs // 300)) o
  ON o.dev = h.dev AND o.det = h.det AND o.k = g.k
{% endmacro %}
{# green events per 5-min bin (signal_silent) and the clock hours with a green per phase (phase_drop) #}
{% macro greens5() %}
-- @step greens5
SELECT dev, b::BIGINT AS b, sum(ngreen)::BIGINT AS k FROM p5 GROUP BY dev, b HAVING sum(ngreen) > 0
{% endmacro %}
{% macro greens_hour() %}
-- @step greens_hour
SELECT dev, ph, h FROM rgq
{% endmacro %}
{# coordination pattern changes (event 131) #}
{% macro ev131() %}
-- @step ev131
SELECT dev, dn, 131 AS eid, par FROM rplan
{% endmacro %}
