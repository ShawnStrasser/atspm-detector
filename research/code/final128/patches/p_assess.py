import os
p = os.path.join(os.environ.get("DC_WORK", os.path.expanduser("~/dc_work")), r"final_v7_next\src\detector_classifier\health_v4.py")
s = open(p, encoding="utf-8").read()


def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:90], s.count(a))
    s = s.replace(a, b)


rep('''def _day_type(ts) -> str:''', '''def _left(X: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """X.merge(df, on="detector", how="left") for a df with one row per detector: its columns looked up by detector
    (reindex fills a missing detector exactly as the merge does) and joined in one concat."""
    if df.detector.duplicated().any() or (set(X.columns) & set(df.columns)) - {"detector"}:
        return X.merge(df, on="detector", how="left")
    R = df.set_index("detector").reindex(X.detector.to_numpy())
    R.index = X.index
    return pd.concat([X, R], axis=1)


def _add(X: pd.DataFrame, cols: dict) -> pd.DataFrame:
    """X with new columns appended in order, one concat (scalars broadcast as a setitem would)."""
    return pd.concat([X, pd.DataFrame(cols, index=X.index)], axis=1)


def _day_type(ts) -> str:''')
# block 1
rep('''    X["fn"] = X.detector.map(fn_label)
    X["phase"] = X.detector.map(phase).astype(float)
    X["lanes"] = X.detector.map(lanes).astype(float)
    Pm = np.array([[fn_prob.get(int(d), {}).get(c, np.nan) for c in C7] for d in X.detector], float)
    for j, c in enumerate(C7):
        X[f"p_{c}"] = Pm[:, j]
    top = np.nanmax(np.where(np.isfinite(Pm), Pm, -1), 1)
    X["f_top"] = np.where(top >= 0, top, np.nan)
    alts = _alts(Pm)
    X["alt_set"] = alts
    X["alt_fns"] = [",".join(c for c in C7 if c in a) for a in alts]
    X["hours"], X["wg"], X["t_start"] = hours, wg, start
    # ---- bins, context
    O = hs.occ_bins(P, start, end, bs, fn_label, phase)
    dmap = {int(d): i for i, d in enumerate(O["dets"])}
    X["occ"] = [O["mocc"][dmap[d]] if d in dmap else np.nan for d in X.detector]
    X["share_ref"] = [O["share_ref"][dmap[d]] if d in dmap else np.nan for d in X.detector]
    M = hs.bins_ctx(O, st1)
    N3 = hs.n3_stats(M)
    A = hs.act_stats(P)
    X = X.merge(N3, on="detector", how="left").merge(A, on="detector", how="left")
    X["span"] = np.where(X.lanes >= 2, "2+", "1")
    X["rate"] = X.n_on / X.hours
    X["band"] = hs.band_of(X.rate)
    X["cmode"] = _cmode(X.fn, X.dur_p50)
    X["type"] = X.fn + " " + X.span
    X["share_med"] = [refs.share_med.get((f, wg), np.nan) for f in X.fn]
    X["share_x"] = X.share_ref / X.share_med
    X["stuck_x"] = np.fmax(pd.to_numeric(X.ep_dur, errors="coerce"), X.max_on_s)
    X = X.merge(hs.slopes(M, m30), on="detector", how="left").merge(hs.like_corr(M, m30), on="detector", how="left")''',
    '''    Pm = np.array([[fn_prob.get(int(d), {}).get(c, np.nan) for c in C7] for d in X.detector], float)
    top = np.nanmax(np.where(np.isfinite(Pm), Pm, -1), 1)
    alts = _alts(Pm)
    O = hs.occ_bins(P, start, end, bs, fn_label, phase)
    dmap = {int(d): i for i, d in enumerate(O["dets"])}
    cols = {"fn": X.detector.map(fn_label), "phase": X.detector.map(phase).astype(float),
            "lanes": X.detector.map(lanes).astype(float)}
    cols.update({f"p_{c}": Pm[:, j] for j, c in enumerate(C7)})
    cols.update(f_top=np.where(top >= 0, top, np.nan), alt_set=alts,
                alt_fns=[",".join(c for c in C7 if c in a) for a in alts], hours=hours, wg=wg, t_start=start,
                occ=[O["mocc"][dmap[d]] if d in dmap else np.nan for d in X.detector],
                share_ref=[O["share_ref"][dmap[d]] if d in dmap else np.nan for d in X.detector])
    X = _add(X, cols)
    # ---- bins, context
    M = hs.bins_ctx(O, st1)
    N3 = hs.n3_stats(M)
    A = hs.act_stats(P)
    X = _left(_left(X, N3), A)
    span = pd.Series(np.where(X.lanes >= 2, "2+", "1"), index=X.index)
    rate = X.n_on / X.hours
    share_med = pd.Series([refs.share_med.get((f, wg), np.nan) for f in X.fn], index=X.index)
    X = _add(X, {"span": span, "rate": rate, "band": hs.band_of(rate), "cmode": _cmode(X.fn, X.dur_p50),
                 "type": X.fn + " " + span, "share_med": share_med, "share_x": X.share_ref / share_med,
                 "stuck_x": np.fmax(pd.to_numeric(X.ep_dur, errors="coerce"), X.max_on_s)})
    X = _left(_left(X, hs.slopes(M, m30)), hs.like_corr(M, m30))''')
rep('''    OH = hs.occ_hi(M, dict(zip(X.detector, X.cmode)), refs)
    X = X.merge(OH, on="detector", how="left")''', '''    OH = hs.occ_hi(M, dict(zip(X.detector, X.cmode)), refs)
    X = _left(X, OH)''')
rep('''    X = X.merge(SH, on="detector", how="left")''', '''    X = _left(X, SH)''')
rep('''    X["lv_ratio110"] = [a[0] for a in lv]
    X["lv_llr110"] = [a[1] for a in lv]
    X["lv_b110"] = [a[2] for a in lv]
    EV = pd.DataFrame(ev, index=X.index)
    for c in ("n_off_15", "n_sc_15", "exc_15", "n_off_sig", "n_sc_sig", "exc_sig"):
        X[c] = EV[c] if c in EV else np.nan
    DR = pd.DataFrame(dr, index=X.index)
    for c in ("own_before", "own_after", "exp_after", "drop_at"):
        X[c] = DR[c] if c in DR else np.nan
    X["g4_unscored0"] = DR.g4_unscored0.fillna(False).astype(bool) if "g4_unscored0" in DR else False
    X["g4_unscored"] = X.g4_unscored0
    X = X.merge(hs.act118(P, T), on="detector", how="left")''', '''    cols = {"lv_ratio110": [a[0] for a in lv], "lv_llr110": [a[1] for a in lv], "lv_b110": [a[2] for a in lv]}
    EV = pd.DataFrame(ev, index=X.index)
    for c in ("n_off_15", "n_sc_15", "exc_15", "n_off_sig", "n_sc_sig", "exc_sig"):
        cols[c] = EV[c] if c in EV else np.nan
    DR = pd.DataFrame(dr, index=X.index)
    for c in ("own_before", "own_after", "exp_after", "drop_at"):
        cols[c] = DR[c] if c in DR else np.nan
    g4 = DR.g4_unscored0.fillna(False).astype(bool) if "g4_unscored0" in DR else False
    cols["g4_unscored0"] = g4
    cols["g4_unscored"] = g4
    X = _left(_add(X, cols), hs.act118(P, T))''')
rep('''    X = X.merge(CS, on="detector", how="left")
    X["n_on117"] = X.detector.map(nstart).astype(float)''', '''    X = _left(X, CS)
    X = _add(X, {"n_on117": X.detector.map(nstart).astype(float)})''')
rep('''    for c in ("drop_lam_chk", "drop_lam_c", "drop_b0_c", "drop_b1_c", "n_dropped"):
        X[c] = X.detector.map({d: v[c] for d, v in DC.items()}).astype(float)
    X["dropped"] = X.detector.map({d: v["dropped"] for d, v in DC.items()})''', '''    cols = {c: X.detector.map({d: v[c] for d, v in DC.items()}).astype(float)
            for c in ("drop_lam_chk", "drop_lam_c", "drop_b0_c", "drop_b1_c", "n_dropped")}
    cols["dropped"] = X.detector.map({d: v["dropped"] for d, v in DC.items()})
    X = _add(X, cols)''')
# time of day: initial columns in one block
rep('''    cols_bd = ["bd_flag", "bd_ev_from", "bd_ev_h", "bd_ev_dir", "bd_ev_h0", "bd_ev_h1", "bd_ev_obs", "bd_ev_exp"]
    X["tod_level_v4"] = "not scored"
    X["tod_level_c"] = "not scored"
    for c in cols_bd:
        X[c] = np.nan
    X["bd_flag"] = pd.Series([np.nan] * len(X), dtype=object)
    X["tod_day"] = pd.NaT''', '''    cols_bd = ["bd_flag", "bd_ev_from", "bd_ev_h", "bd_ev_dir", "bd_ev_h0", "bd_ev_h1", "bd_ev_obs", "bd_ev_exp"]
    cols = {"tod_level_v4": "not scored", "tod_level_c": "not scored"}
    cols.update({c: np.nan for c in cols_bd})
    cols["bd_flag"] = pd.Series([np.nan] * len(X), dtype=object, index=X.index)
    cols["tod_day"] = pd.NaT
    X = _add(X, cols)''')
open(p, "w", encoding="utf-8").write(s)
print("patched")
