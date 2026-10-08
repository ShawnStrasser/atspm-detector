
WITH b AS (SELECT detector, b, n, nG, nY, nR, nU, fGY, fR, fU, cGY, cR, cU, xGY, xR, xU, sGY, sR, sRg, ln,
                  nG + nY AS nGYa, greatest(300 - sGY - sR - sRg, 0) AS sU FROM h_cb),
q AS (SELECT detector, max(CASE WHEN sGY >= 60 THEN nGYa / sGY * 3600 / ln END) AS q5_gy,
             max(n / 300 * 3600 / ln) AS q5_all FROM b GROUP BY ALL),
r7 AS (SELECT detector, b, fGY, fR, fU, nGYa AS nGY, nR, nU, sGY, sR, sU,
              CASE WHEN sGY >= 30 THEN nGYa / sGY END AS rg, CASE WHEN sR >= 30 THEN nR / sR END AS rr,
              CASE WHEN sU >= 30 THEN nU / sU END AS ru FROM b),
m7 AS (SELECT *, median(rg) OVER w AS mg, median(rr) OVER w AS mr, median(ru) OVER w AS mu FROM r7
       WINDOW w AS (PARTITION BY detector ORDER BY b ROWS BETWEEN 6 PRECEDING AND 6 FOLLOWING)),
e7 AS (SELECT *, CASE WHEN sGY > 0 THEN nGY * (1 - exp(-least(nGY / sGY, coalesce(mg, nGY / sGY)))) ELSE 0 END AS eg,
                 CASE WHEN sR > 0 THEN nR * (1 - exp(-least(nR / sR, coalesce(mr, nR / sR)))) ELSE 0 END AS er,
                 CASE WHEN sU > 0 THEN nU * (1 - exp(-least(nU / sU, coalesce(mu, nU / sU), 50))) ELSE nU END AS eu
       FROM m7),
a7 AS (SELECT detector, sum(fGY + fR + fU) AS fo_all, sum(eg + er + eu) AS fem_all,
              sum(CASE WHEN (fGY + fR + fU - eg - er - eu) / sqrt(eg + er + eu + 1) >= 4
                        AND fGY + fR + fU >= 5 THEN 1 ELSE 0 END) AS n_spk FROM e7 GROUP BY ALL),
r8 AS (SELECT detector, b, xGY, xR, xU, sGY, sR, sU,
              greatest(nGYa - cGY, 0) AS nGY, greatest(nR - cR, 0) AS nR, greatest(nU - cU, 0) AS nU FROM b),
s8 AS (SELECT *, CASE WHEN sGY >= 30 THEN nGY / sGY END AS rg, CASE WHEN sR >= 30 THEN nR / sR END AS rr,
                 CASE WHEN sU >= 30 THEN nU / sU END AS ru FROM r8),
m8 AS (SELECT *, median(rg) OVER w AS mg, median(rr) OVER w AS mr, median(ru) OVER w AS mu FROM s8
       WINDOW w AS (PARTITION BY detector ORDER BY b ROWS BETWEEN 6 PRECEDING AND 6 FOLLOWING)),
e8 AS (SELECT *, CASE WHEN sGY > 0 THEN nGY * (1 - exp(-least(nGY / sGY, coalesce(mg, nGY / sGY)))) ELSE 0 END AS eg,
                 CASE WHEN sR > 0 THEN nR * (1 - exp(-least(nR / sR, coalesce(mr, nR / sR)))) ELSE 0 END AS er,
                 CASE WHEN sU > 0 THEN nU * (1 - exp(-least(nU / sU, coalesce(mu, nU / sU), 50))) ELSE nU END AS eu
       FROM m8),
a8 AS (SELECT detector, sum(xGY + xR + xU) AS fo_c, sum(eg + er + eu) AS fem_c,
              sum(CASE WHEN (xGY + xR + xU - eg - er - eu) / sqrt(eg + er + eu + 1) >= 4
                        AND xGY + xR + xU >= 5 THEN 1 ELSE 0 END) AS n_spk_c FROM e8 GROUP BY ALL)
SELECT q.detector, q5_gy, q5_all, fo_all, fem_all, n_spk, fo_c, fem_c, n_spk_c
FROM q JOIN a7 USING (detector) JOIN a8 USING (detector)
