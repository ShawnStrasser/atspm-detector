"""direct check: numpy color medians == DuckDB window medians on every capture"""
import sys, pickle, warnings, glob
sys.path.insert(0, "../final_v7_next/src"); warnings.simplefilter("ignore")
from detector_classifier import health_core as hc, health_v4_stats as hs
import numpy as np, duckdb
con = duckdb.connect(); con.execute("SET threads=1")
SQL = """WITH b AS (SELECT detector, b, nG, nY, nR, nU, cGY, cR, cU, sGY, sR, sRg, nG + nY AS nGYa,
   greatest(300 - sGY - sR - sRg, 0) AS sU FROM cb),
r7 AS (SELECT detector, b, CASE WHEN sGY >= 30 THEN nGYa / sGY END AS rg, CASE WHEN sR >= 30 THEN nR / sR END AS rr,
   CASE WHEN sU >= 30 THEN nU / sU END AS ru FROM b),
r8 AS (SELECT detector, b, sGY, sR, sU, greatest(nGYa - cGY, 0) AS g, greatest(nR - cR, 0) AS r, greatest(nU - cU, 0) AS u FROM b),
s8 AS (SELECT detector, b, CASE WHEN sGY >= 30 THEN g / sGY END AS rg, CASE WHEN sR >= 30 THEN r / sR END AS rr,
   CASE WHEN sU >= 30 THEN u / sU END AS ru FROM r8),
m7 AS (SELECT detector, b, median(rg) OVER w AS mg7, median(rr) OVER w AS mr7, median(ru) OVER w AS mu7 FROM r7
   WINDOW w AS (PARTITION BY detector ORDER BY b ROWS BETWEEN 6 PRECEDING AND 6 FOLLOWING)),
m8 AS (SELECT detector, b, median(rg) OVER w AS mg8, median(rr) OVER w AS mr8, median(ru) OVER w AS mu8 FROM s8
   WINDOW w AS (PARTITION BY detector ORDER BY b ROWS BETWEEN 6 PRECEDING AND 6 FOLLOWING))
SELECT * FROM m7 JOIN m8 USING (detector, b) ORDER BY detector, b"""
bad = n = 0
for f in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "cap/*.pkl")):
    a, k = pickle.load(open(f, "rb"))
    P = hc.prep_arrays(a[0].t, a[0].eid, a[0].par)
    CB, _ = hs.colour_bins(P, (a[2] - a[1]).total_seconds(), a[3])
    if not len(CB):
        continue
    con.register("cb", CB)
    D = con.execute(SQL).df()
    M = hs._colour_medians(CB)
    for c, v in M.items():
        n += 1
        d = D[c].to_numpy(np.float32)
        if D[c].dtype != np.float32 or not np.array_equal(d, v, equal_nan=True):
            bad += 1
            print("DIFF", f, c, D[c].dtype, np.nanmax(np.abs(d - v)))
print("median columns checked", n, "bad", bad)
