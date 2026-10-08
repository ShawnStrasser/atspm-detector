import os
p = os.path.join(os.environ.get("DC_WORK", os.path.expanduser("~/dc_work")), r"final_v7_next\src\detector_classifier\health_v4_stats.py")
s = open(p, encoding="utf-8").read()


def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:80], s.count(a))
    s = s.replace(a, b)


rep('''_SQL_FAST = """
WITH b AS (SELECT detector, b, n, nG, nY, nR, nU, fGY, fR, fU, cGY, cR, cU, xGY, xR, xU, sGY, sR, sRg, ln,
                  nG + nY AS nGYa, greatest(300 - sGY - sR - sRg, 0) AS sU FROM h_cb),''', '''# note 128: the six rolling medians (13 five-minute bins) are computed in numpy (_win_median, the median arithmetic of
# DuckDB) and passed in as columns; the rest of the research SQL is unchanged
_SQL_FAST = """
WITH b AS (SELECT detector, b, n, nG, nY, nR, nU, fGY, fR, fU, cGY, cR, cU, xGY, xR, xU, sGY, sR, sRg, ln,
                  mg7, mr7, mu7, mg8, mr8, mu8,
                  nG + nY AS nGYa, greatest(300 - sGY - sR - sRg, 0) AS sU FROM h_cb),''')
rep('''r7 AS (SELECT detector, b, fGY, fR, fU, nGYa AS nGY, nR, nU, sGY, sR, sU,
              CASE WHEN sGY >= 30 THEN nGYa / sGY END AS rg, CASE WHEN sR >= 30 THEN nR / sR END AS rr,
              CASE WHEN sU >= 30 THEN nU / sU END AS ru FROM b),
m7 AS (SELECT *, median(rg) OVER w AS mg, median(rr) OVER w AS mr, median(ru) OVER w AS mu FROM r7
       WINDOW w AS (PARTITION BY detector ORDER BY b ROWS BETWEEN 6 PRECEDING AND 6 FOLLOWING)),''', '''r7 AS (SELECT detector, b, fGY, fR, fU, nGYa AS nGY, nR, nU, sGY, sR, sU, mg7, mr7, mu7,
              CASE WHEN sGY >= 30 THEN nGYa / sGY END AS rg, CASE WHEN sR >= 30 THEN nR / sR END AS rr,
              CASE WHEN sU >= 30 THEN nU / sU END AS ru FROM b),
m7 AS (SELECT *, mg7 AS mg, mr7 AS mr, mu7 AS mu FROM r7),''')
rep('''r8 AS (SELECT detector, b, xGY, xR, xU, sGY, sR, sU,
              greatest(nGYa - cGY, 0) AS nGY, greatest(nR - cR, 0) AS nR, greatest(nU - cU, 0) AS nU FROM b),''', '''r8 AS (SELECT detector, b, xGY, xR, xU, sGY, sR, sU, mg8, mr8, mu8,
              greatest(nGYa - cGY, 0) AS nGY, greatest(nR - cR, 0) AS nR, greatest(nU - cU, 0) AS nU FROM b),''')
rep('''m8 AS (SELECT *, median(rg) OVER w AS mg, median(rr) OVER w AS mr, median(ru) OVER w AS mu FROM s8
       WINDOW w AS (PARTITION BY detector ORDER BY b ROWS BETWEEN 6 PRECEDING AND 6 FOLLOWING)),''', '''m8 AS (SELECT *, mg8 AS mg, mr8 AS mr, mu8 AS mu FROM s8),''')
rep('''def colour_stats(CB, lanes, con=None):''', '''def _win_median(R, h=6):
    """per row of the float32 matrix R (NaN = NULL): DuckDB median(x) OVER (ORDER BY column ROWS BETWEEN h
    PRECEDING AND h FOLLOWING) -- the middle value, for an even count lo + (hi - lo) * 0.5 with the difference in
    float32 and the result rounded to float32 (quantile_cont on FLOAT); NaN where the frame holds no value."""
    from numpy.lib.stride_tricks import sliding_window_view
    nd, nb = R.shape
    w = 2 * h + 1
    pad = np.full((nd, h), np.nan, np.float32)
    W = np.sort(sliding_window_view(np.concatenate([pad, R, pad], 1), w, axis=1), axis=2)
    k = np.isfinite(W).sum(2)
    half = k // 2
    hi = np.take_along_axis(W, np.clip(half, 0, w - 1)[..., None], 2)[..., 0]
    lo = np.take_along_axis(W, np.clip(half - 1, 0, w - 1)[..., None], 2)[..., 0]
    with np.errstate(invalid="ignore"):
        ev = (lo.astype(np.float64) + (hi - lo).astype(np.float64) * 0.5).astype(np.float32)
    out = np.where(k % 2 == 1, hi, ev).astype(np.float32)
    out[k == 0] = np.nan
    return out


def _colour_medians(cb):
    """the six rolling medians of the research SQL (m7 / m8: rates by colour state, raw and without chatter), from
    the float32 bins in the same float32 arithmetic as the SQL (rows: detector-major, every bin present)."""
    nd = len(np.unique(cb.detector.to_numpy()))
    nb = len(cb) // max(nd, 1)
    f = {c: cb[c].to_numpy(np.float32).reshape(nd, nb) for c in ("nG", "nY", "nR", "nU", "cGY", "cR", "cU", "sGY",
                                                                  "sR", "sRg")}
    z = np.float32(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        nGYa = f["nG"] + f["nY"]
        sU = np.maximum(np.float32(300) - f["sGY"] - f["sR"] - f["sRg"], z)
        sGY, sR = f["sGY"], f["sR"]

        def rates(g, r, u):
            return (np.where(sGY >= 30, g / sGY, np.nan).astype(np.float32),
                    np.where(sR >= 30, r / sR, np.nan).astype(np.float32),
                    np.where(sU >= 30, u / sU, np.nan).astype(np.float32))
        r7 = rates(nGYa, f["nR"], f["nU"])
        r8 = rates(np.maximum(nGYa - f["cGY"], z), np.maximum(f["nR"] - f["cR"], z), np.maximum(f["nU"] - f["cU"], z))
    out = {}
    for tag, rr in (("7", r7), ("8", r8)):
        for nm, a in zip(("mg", "mr", "mu"), rr):
            out[nm + tag] = _win_median(a).ravel()
    return out


def colour_stats(CB, lanes, con=None):''')
rep('''    cb = CB.assign(ln=np.fmax(pd.to_numeric(CB.detector.map(lanes), errors="coerce").fillna(1).to_numpy(float), 1))''',
    '''    cb = CB.assign(ln=np.fmax(pd.to_numeric(CB.detector.map(lanes), errors="coerce").fillna(1).to_numpy(float), 1),
                   **_colour_medians(CB))''')
open(p, "w", encoding="utf-8").write(s)
print("patched")
