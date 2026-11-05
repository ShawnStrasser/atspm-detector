{# What is still open at the window end of a judge on a store, from its carry (design 4.4; open_extract.sql: from the
   extractor's steps in a batch run): keys in ns from the window start, decisions in the same double seconds as the
   extractor's (a store's origin is the epoch).  store: the store's name and a dot. #}
-- @step oxr mat
SELECT w.dev, w.s_ns, epoch_ns(c.TimeStamp) AS tn, c.EventId::INT AS cid, c.Parameter::INT AS par
FROM {{ store }}health_carry c JOIN win w USING (DeviceId)
{# the first comms gap at or after each open object's start (9403) #}
-- @step oxm mat
SELECT dev, tn, tn / 1e9 AS td FROM oxr WHERE cid = 9403
-- @step oxl mat
SELECT dev, tn - s_ns AS lz, tn / 1e9 AS lzd FROM oxr WHERE cid = 9402
{# chains still open (last event an ON, 9222): start = the first ON after the detector's last OFF #}
-- @step oxc
SELECT c.dev, c.det, c.s - c.s_ns AS s, c.s / 1e9 AS sd, g.tn - c.s_ns AS mk
FROM (SELECT r.dev, r.par AS det, any_value(r.s_ns) AS s_ns, min(r.tn) AS s
      FROM oxr r JOIN (SELECT dev, par, max(tn) FILTER (cid = 9201) AS off FROM oxr WHERE cid IN (9201, 9222) GROUP BY dev, par
                       HAVING count(*) FILTER (cid = 9222) = 1) o USING (dev, par)
      WHERE r.cid IN (9202, 9222) AND (o.off IS NULL OR r.tn > o.off) GROUP BY r.dev, r.par) c
ASOF LEFT JOIN oxm g ON c.dev = g.dev AND g.tn >= c.s
{# each phase's colour piece, always still open: 0 green, 1 yellow, 2 red #}
-- @step oxp
SELECT p.dev, p.par AS ph, p.cid - 9300 AS st, p.tn - p.s_ns AS tag, p.ad, g.td AS mkd
FROM (SELECT *, tn / 1e9 AS ad FROM oxr WHERE cid BETWEEN 9300 AND 9302) p ASOF LEFT JOIN oxm g ON p.dev = g.dev AND g.td >= p.ad
{# each detector's last ON (9221, 9222), its too-fast pair pending, with its ticks to the ON before (the carry's other
   ON); not one from before the window #}
-- @step oxt
SELECT dev, det, p - s_ns AS dn, ta
FROM (SELECT dev, det, s_ns, p, round_even((p - q) / 1e8, 0)::INT AS ta
      FROM (SELECT dev, par AS det, any_value(s_ns) AS s_ns, max(tn) FILTER (cid IN (9221, 9222)) AS p, max(tn) FILTER (cid = 9202) AS q
            FROM oxr WHERE cid IN (9202, 9221, 9222) GROUP BY dev, par))
WHERE ta < {{ TFC }} AND p >= s_ns
