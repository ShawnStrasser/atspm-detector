def _fast_stats(cb):
    """the research SQL of h117_study.stats + h118c_fast.fast_stats (run in DuckDB until note 128) in numpy, with the
    same arithmetic: FLOAT (float32) columns and operations where DuckDB used FLOAT, DOUBLE where it used DOUBLE, sums
    added row after row in bin order (as DuckDB on one thread), exp / sqrt the same libm calls.  cb: detector-major
    rows (every bin present) with the six rolling medians already in it.  -> one row per detector."""
    f32 = np.float32
    nd = len(np.unique(cb.detector.to_numpy()))
    nb = len(cb) // max(nd, 1)
    g = {c: cb[c].to_numpy(f32).reshape(nd, nb) for c in ("n", "nG", "nY", "nR", "nU", "fGY", "fR", "fU", "cGY",
                                                           "cR", "cU", "xGY", "xR", "xU", "sGY", "sR", "sRg", "mg7",
                                                           "mr7", "mu7", "mg8", "mr8", "mu8")}
    ln = cb.ln.to_numpy(np.float64).reshape(nd, nb)
    z = f32(0)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        nGYa = g["nG"] + g["nY"]
        sGY, sR = g["sGY"], g["sR"]
        sU = np.maximum(f32(300) - sGY - g["sR"] - g["sRg"], z)
        q5_gy = np.where(sGY >= 60, (nGYa / sGY * f32(3600)).astype(np.float64) / ln, np.nan)
        q5_all = (g["n"] / f32(300) * f32(3600)).astype(np.float64) / ln

        def expect(nGY, nR, nU, mg, mr, mu):
            def term(x, s, m, cap=None):
                r = x / s
                lo = np.fmin(r, np.where(np.isnan(m), r, m))
                if cap is not None:
                    lo = np.fmin(lo, f32(cap))
                return x.astype(np.float64) * (1.0 - np.exp(-(lo.astype(np.float64))))
            eg = np.where(sGY > 0, term(nGY, sGY, mg), 0.0)
            er = np.where(sR > 0, term(nR, sR, mr), 0.0)
            eu = np.where(sU > 0, term(nU, sU, mu, 50), nU.astype(np.float64))
            return eg, er, eu

        def agg(fa, fb, fc, eg, er, eu):
            fo = (fa + fb + fc).astype(np.float64)
            e3 = eg + er + eu
            spk = ((fo - eg - er - eu) / np.sqrt(e3 + 1) >= 4) & (fo >= 5)
            return (np.cumsum(fo, 1)[:, -1] if nb else np.zeros(nd), np.cumsum(e3, 1)[:, -1] if nb else np.zeros(nd),
                    spk.sum(1).astype(np.float64))
        e7 = expect(nGYa, g["nR"], g["nU"], g["mg7"], g["mr7"], g["mu7"])
        fo_all, fem_all, n_spk = agg(g["fGY"], g["fR"], g["fU"], *e7)
        e8 = expect(np.maximum(nGYa - g["cGY"], z), np.maximum(g["nR"] - g["cR"], z), np.maximum(g["nU"] - g["cU"], z),
                    g["mg8"], g["mr8"], g["mu8"])
        fo_c, fem_c, n_spk_c = agg(g["xGY"], g["xR"], g["xU"], *e8)
        with _quiet():
            q5g = np.nanmax(q5_gy, 1) if nb else np.full(nd, np.nan)
            q5a = np.nanmax(q5_all, 1) if nb else np.full(nd, np.nan)
    det = cb.detector.to_numpy()[::nb] if nb else np.unique(cb.detector.to_numpy())
    return pd.DataFrame({"detector": det.astype(np.int64), "q5_gy": q5g, "q5_all": q5a, "fo_all": fo_all,
                         "fem_all": fem_all, "n_spk": n_spk, "fo_c": fo_c, "fem_c": fem_c, "n_spk_c": n_spk_c})


