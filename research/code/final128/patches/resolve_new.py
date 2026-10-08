def _limits(X, refs, v, ah=None):
    """the per-type limits of the chatter / stuck / erratic-time-ON checks, loosened to the least strict plausible type
    when the model is unsure.  Returns {column: array} (no column is written into X)."""
    base = X[["fn", "span", "cmode", "wg"]]
    lim = {col: refs.lookup(refs.cell[v][col], base) for col in ("chat_frac", "stuck_x", "n3_exc")}
    low = (X.f_top < F5_TOP).to_numpy()
    ah = ah if ah is not None else _alt_has(X)
    for fa in C7:
        m = low & ah[fa]
        if not m.any():
            continue
        D = base[m].copy()
        D["fn"] = fa
        D["cmode"] = np.where(fa != "Count", "", np.where(X.loc[m, "dur_p50"].isna(), "unknown",
                                                          np.where(X.loc[m, "dur_p50"] <= 0.25, "pulse", "normal")))
        for col in ("chat_frac", "stuck_x", "n3_exc"):
            la = refs.lookup(refs.cell[v][col], D)
            cur = lim[col][m]
            with np.errstate(invalid="ignore"):
                looser = la > cur
            lim[col][np.flatnonzero(m)[looser]] = la[looser]
    return {f"lim_{col}": a for col, a in lim.items()}


def _stuck_stats(X, E, lim):
    cols = ["n_over", "n_ep", "tot_s", "n_q1", "n_q1b", "n_shared", "n_shared_unq"]
    if not len(E):
        return pd.DataFrame(np.nan, index=np.arange(len(X)), columns=cols)
    Ex = E.merge(pd.DataFrame({"detector": X.detector.to_numpy(), "L": lim, "ri": np.arange(len(X))}),
                 on="detector", how="inner")
    Ex = Ex[Ex.dur_s >= Ex.L].copy()
    q_ok = (Ex.n_hpeer_e.fillna(0) > 0) & (Ex.phx_h_e >= 1.5) & (Ex.corr_h_e >= Q1_CORR) & (Ex.refx_e >= 0.5) & \
        (Ex.light_e.fillna(1) == 0) & (Ex.trafx_e >= TRAF_X)
    Ex["q1"] = q_ok & (Ex.dur_s < 3600)
    Ex["q1b"] = q_ok & (Ex.dur_s >= 3600) & (Ex.cover_e >= F4_COVER)
    Ex["shared"] = Ex.co5 >= 3
    Ex["count"] = ~Ex.q1 & ~Ex.q1b & ~Ex.shared
    Ex["cdur"] = np.where(Ex["count"], Ex.dur_s, 0.0)
    Ex["sh_unq"] = Ex.shared & ~Ex.q1 & ~Ex.q1b
    g = Ex.groupby("ri")
    S = pd.DataFrame({"n_over": g.size(), "n_ep": g["count"].sum(), "tot_s": g.cdur.sum(), "n_q1": g.q1.sum(),
                      "n_q1b": g.q1b.sum(), "n_shared": g.shared.sum(), "n_shared_unq": g.sh_unq.sum()})
    return S.reindex(np.arange(len(X)))


def _night_frac_v(t0, b0, b1):
    """share of the 5-min bins [b0, b1) (from t0) whose clock hour is night (21:00-05:00), one value per row
    (= _night_frac, in integer nanoseconds of the day)."""
    out = np.zeros(len(b0))
    if not len(b0):
        return out
    sod = int(pd.Timestamp(t0).value) % 86_400_000_000_000
    for i, (a, b) in enumerate(zip(b0, b1)):
        if not (np.isfinite(a) and np.isfinite(b)) or b <= a:
            continue
        hrs = ((sod + np.arange(int(a), int(b), dtype=np.int64) * 300_000_000_000) // 3_600_000_000_000) % 24
        out[i] = float(np.mean((hrs >= NIGHT_SIL[0]) | (hrs < NIGHT_SIL[1])))
    return out


def _f0(a):
    """Series.fillna(0) of a float column, as an array."""
    return np.nan_to_num(np.asarray(a, float), nan=0.0)


def _resolve(X0, E0, refs, v):
    """one research resolver pass, v = 'v4' (score_v4.run_v4) or 'v4c' (score_v4c.run_c).  Returns X with the
    scores s8_*, status st8, rules8, watch8, cleared8, left8, score8, dq8, cfg.  (note 128: the new columns are built
    as arrays and joined once; the arithmetic is the research's, step for step.)"""
    X = X0.reset_index(drop=True)
    n = len(X)
    E = E0.copy()
    if len(E):
        E.loc[E.fn.isin(NO_SHARED), "co5"] = 0
    ah = _alt_has(X)
    C = _limits(X, refs, v, ah)
    base = X[["fn", "span", "cmode", "wg"]]
    # stuck: episodes over the per-type limit; their number has its own limit
    sl = np.clip(C["lim_stuck_x"], *STUCK_CLIP)
    S0 = _stuck_stats(X, E, sl)
    C["n_ep"] = S0.n_ep.fillna(0).to_numpy()
    lim_n_ep = refs.lookup(refs.cell[v]["n_ep"], base)
    low = X.alt_fns.ne("").to_numpy()
    for fa in C7:
        m = low & ah[fa]
        if m.any():
            la = refs.lookup(refs.cell[v]["n_ep"], base[m].assign(fn=fa))
            lim_n_ep[m] = np.fmax(lim_n_ep[m], la)
    C["lim_n_ep"] = lim_n_ep
    # ---- scores (h108 scores -> h110 scores110 -> h118 scores118 -> v4 / v4c)
    S = {}
    for k in ("dropout", "level", "night_drop"):
        S[k] = pd.to_numeric(X[f"s_{k}"], errors="coerce").to_numpy()
    S["choppy"] = np.full(n, np.nan)
    n50 = X.n_on.to_numpy(float) >= 50
    lcf = C["lim_chat_frac"]
    S["chatter"] = np.where(n50, score(X.chat_frac, lcf, 2 * lcf), np.nan)
    S["rapid"] = np.full(n, np.nan)
    S["volume"] = np.full(n, np.nan)
    C["stuck_lim8"] = sl
    S["stuck"] = score(X.stuck_x.fillna(0), sl, np.fmax(STUCK_BAD, 2 * sl))
    n3ok = (X.n3_n >= 2).to_numpy()
    n3_exc = X.n3_exc.to_numpy(float)
    l3e = C["lim_n3_exc"]
    S["occspk"] = np.where(n3ok, score(n3_exc, l3e, 4 * l3e), np.where(np.isfinite(n3_exc), 0.0, np.nan))
    rep_s = X.rep_time_s.to_numpy(float)
    with np.errstate(invalid="ignore"):
        n3_dq = (S["occspk"] >= .35) & (rep_s / 60 >= 0.5 * n3_exc)
    C["n3_dq"] = n3_dq
    S["occspk"] = np.where(n3_dq, 0.0, S["occspk"])
    S["prof"] = np.full(n, np.nan)
    cap = (pd.to_numeric(X.s_stuck, errors="coerce") == 0.35).to_numpy()
    with np.errstate(invalid="ignore"):
        S["stuck"] = np.where(cap & (S["stuck"] > .35), 0.35, S["stuck"])
    # F2 level
    ratio, llr = X.lv_ratio110.to_numpy(float), X.lv_llr110.to_numpy(float)
    vv = -np.log(np.maximum(np.nan_to_num(ratio, nan=1.0), 1e-9))
    s_lv = score(vv, -np.log(0.15), -np.log(0.05))
    S["level"] = np.where(np.isfinite(ratio), np.where(llr <= 50, 0.0, s_lv), np.nan)
    # F3 / F4 stuck (the same episodes and limits as above)
    St = {c: S0[c].to_numpy() for c in S0.columns}
    for c in S0.columns:
        C[f"st_{c}"] = St[c]
    tot = _f0(St["tot_s"])
    s_tot = score(tot, sl, np.fmax(STUCK_BAD, 2 * sl))
    s_n = score(_f0(St["n_ep"]), lim_n_ep + 1, 2 * (lim_n_ep + 1))
    s_ = np.fmax(s_tot, np.nan_to_num(s_n))
    single = _f0(St["n_ep"]) <= 1
    s_ = np.where(cap & single & (s_ > .35), .35, s_)
    q1_all = ((_f0(St["n_over"]) > 0) & (_f0(St["n_ep"]) == 0) & (_f0(St["n_shared_unq"]) == 0) &
              (_f0(St["n_q1"]) + _f0(St["n_q1b"]) > 0))
    C["q1_all"] = q1_all
    shared_only = (_f0(St["n_ep"]) == 0) & (_f0(St["n_shared_unq"]) > 0)
    stk = S["stuck"]
    with np.errstate(invalid="ignore"):
        s_ = np.where(shared_only, np.where(stk >= .35, .35, stk), s_)
    s_ = np.where(q1_all, 0.0, s_)
    S["stuck"] = s_
    # G3 erratic counts
    D3 = X[["fn", "span", "band", "wg"]]
    L3 = refs.lookup(refs.g3[v], D3)
    for fa in C7:
        m = low & ah[fa]
        if not m.any():
            continue
        la = refs.lookup(refs.g3[v], D3[m].assign(fn=fa))
        with np.errstate(invalid="ignore"):
            lo_ = la > L3[m]
        ii = np.flatnonzero(m)[lo_]
        L3[ii] = la[lo_]
    if v == "v4c":
        L3 = np.fmax(L3, EXC_FLOOR)
    C["lim_exc"] = L3
    ok3 = (X.exc_15.notna() & X.n_sc_15.ge(G3_MIN_SC) & X.ref110.ne("none")).to_numpy()
    s3 = score(X.exc_15, L3, 2 * L3)
    s3 = np.where(X.n_off_15.fillna(0) >= G3_MIN_OFF, s3, 0.0)
    S["choppy"] = np.where(ok3, s3, np.nan)
    with np.errstate(invalid="ignore"):
        C["noyard_chop"] = (X.ref110.eq("none") & (X.exc_sig >= L3) & (X.n_off_sig.fillna(0) >= G3_MIN_OFF) &
                            X.n_sc_sig.ge(G3_MIN_SC)).to_numpy()
    S["level"] = np.where(X.g4_unscored.fillna(False).to_numpy(bool), np.nan, S["level"])
    # ---- too fast / too many / busy at night
    adv = X.fn.eq("Advance").to_numpy()
    enough = X.n_on117.fillna(0).to_numpy() >= 50
    with np.errstate(invalid="ignore", divide="ignore"):
        if v == "v4":
            lz = refs.lookup(refs.f117["zf"], X[["fn", "span", "wg"]])
            lk = refs.lookup(refs.f117["n_spk"], X[["fn", "span", "wg"]])
            lg = np.fmax(refs.lookup(refs.f117["q5_gy"], X[["fn", "span", "wg"]]), SAT)
            la_ = np.fmax(refs.lookup(refs.f117["q5_all"], X[["fn", "span", "wg"]]), SAT)
            sc_ = lambda a, b: np.where(np.isfinite(a) & np.isfinite(b) & (b > 0), a / b, np.nan)  # noqa: E731
            x_zf = sc_(X.zf.to_numpy(float), lz)
            nspk = X.n_spk.to_numpy(float)
            x_spk = np.where(nspk > lk, nspk / np.fmax(lk, 1), 0)
            fx = np.fmax(np.nan_to_num(x_zf), x_spk)
            vol_x = np.where(adv, sc_(X.q5_all.to_numpy(float), la_), sc_(X.q5_gy.to_numpy(float), lg))
            C["lim_zf_use"], C["lim_spk_use"], C["lim_vol_use"] = lz, lk, np.where(adv, la_, lg)
            C["fo_use"], C["fem_use"], C["zf_use"], C["n_spk_use"] = X.fo_all, X.fem_all, X.zf, X.n_spk
        else:
            lz = _least_strict(refs.f118["zf_c"], X, ZF_FLOOR, ah)
            lk = _least_strict(refs.f118["n_spk_c"], X, SPK_FLOOR, ah)
            zfc = X.zf_c.to_numpy(float)
            x_zf = np.where(np.isfinite(zfc), zfc / lz, np.nan)
            nspk = X.n_spk_c.to_numpy(float)
            x_spk = np.where(nspk > lk, nspk / lk, 0.0)
            fx = np.where(X.n_on117.to_numpy(float) >= 50, np.fmax(np.nan_to_num(x_zf), x_spk), np.nan)
            lg = _least_strict(refs.f118["q5_gy"], X, None, ah)
            la_ = _least_strict(refs.f118["q5_all"], X, None, ah)
            lv = np.where(adv, np.fmax(la_, SAT), np.fmax(lg, SAT))
            q = np.where(adv, X.q5_all, X.q5_gy).astype(float)
            vol_x = np.where(np.isfinite(q) & (lv > 0), q / lv, np.nan)
            C["lim_zf_use"], C["lim_spk_use"], C["lim_vol_use"] = lz, lk, lv
            C["fo_use"], C["fem_use"], C["zf_use"], C["n_spk_use"] = X.fo_c, X.fem_c, X.zf_c, X.n_spk_c
    C["x_zf"], C["x_spk"], C["fast_x"], C["vol_x"] = x_zf, x_spk, fx, vol_x
    S["rapid"] = np.where(enough & np.isfinite(fx), score(fx, 1.0, 2.0), np.nan)
    S["volume"] = np.where(np.isfinite(vol_x), score(vol_x, 1.0, 2.0), np.nan)
    lvc = X.tod_level.fillna("not scored")
    pv = np.where(lvc.eq("not scored"), np.nan, lvc.map(PROF_LEVEL).fillna(0.0))
    dro = _f0(S["dropout"])
    S["prof"] = np.where((dro >= .35) & np.isfinite(pv), 0.0, pv)
    occ_hi8 = X.occ_hi8.to_numpy()
    changed_hi8 = False
    if v == "v4c":
        nf = _night_frac_v(X.t_start.iat[0] if n else None, X.drop_b0.astype(float).to_numpy(),
                           X.drop_b1.astype(float).to_numpy())
        silent = (dro >= .35) & (nf >= .8)
        C["silent_night"] = silent
        nd = np.asarray(S["night_drop"], float)
        S["night_drop"] = np.where(silent, np.fmax(np.nan_to_num(nd), dro), nd)
        S["dropout"] = np.where(silent, 0.0, S["dropout"])
        run = X.bd_ev_h.to_numpy(float)
        sh = np.where(X.bd_flag.fillna(False).astype(bool), score(run, 3.0, 7.0), 0.0)
        down = X.bd_ev_dir.to_numpy(float) < 0
        with np.errstate(invalid="ignore"):
            other = (_f0(S["level"]) >= .35) | (_f0(S["night_drop"]) >= .35)
            sh = np.where(down & other, 0.0, sh)
            sh = np.where(~down & (np.nan_to_num(S["prof"]) >= .35), 0.0, sh)
            dr = _f0(S["dropout"])
            both = down & (dr >= .35) & (sh >= .35)
            S["dropout"] = np.where(both & (sh >= dr), 0.0, np.where(both, np.fmax(dr, BORDER), S["dropout"]))
            sh = np.where(both & (sh < dr), 0.0, sh)
        S["shape24"] = np.where(X.bd_flag.notna().to_numpy(), sh, np.nan)
        hm = X.hi_min_c.fillna(0).to_numpy(float)
        S["count_on"] = np.where(X.count_on_ok, score(hm, 30.0, 240.0), np.nan)
        m_hi = X.count_on_ok.to_numpy() & (hm >= 30)
        if m_hi.any():
            occ_hi8 = occ_hi8.copy()
            occ_hi8[m_hi] = False
            changed_hi8 = True
    for k, a in S.items():
        C[f"s8_{k}"] = np.asarray(a)
    C["queue_pat8"] = ((X.elhi_occ > 0) & (X.elhi_occ > X.elhi_cnt)).to_numpy()
    # ---- the rules (h110 resolve)
    rows = []
    cols = list(S)
    Sv = np.column_stack([np.asarray(S[k], float) for k in cols]) if n else np.zeros((0, len(cols)))
    L = {c: X[c].tolist() for c in ("status2", "status", "c_occ_like", "n_on", "share_x", "hours", "n_hmates",
                                    "reason", "fn", "occ", "rep_time_s")}
    qp, q1a, nyc, n3d = C["queue_pat8"].tolist(), q1_all.tolist(), C["noyard_chop"].tolist(), n3_dq.tolist()
    sq1b, oh8 = pd.Series(St["n_q1b"]).tolist(), occ_hi8.tolist()
    for i in range(n):
        s = {k: Sv[i, j] for j, k in enumerate(cols) if np.isfinite(Sv[i, j])}
        find = {k: val for k, val in s.items() if val >= .35}
        notes, watch, fired = [], [], []
        status, c_occ_like = L["status"][i], L["c_occ_like"][i]
        if L["status2"][i] != status:
            fired.append("R9")
        for k in list(find):
            rule = None
            if k in ("choppy", "level", "occspk") and bool(qp[i]) and np.isfinite(c_occ_like) and \
                    c_occ_like >= Q2_CORR:
                rule = ("Q2", "clear")
            elif k == "rapid" and L["n_on"][i] < LOW_N and np.isfinite(L["share_x"][i]) and L["share_x"][i] >= 0.1:
                rule = ("R5", "watch")
            elif k == "dropout" and L["hours"][i] < 1:
                rule = ("R6", "watch")
            if rule:
                fired.append(rule[0])
                (watch if rule[1] == "watch" else notes).append(k)
                del find[k]
        if bool(q1a[i]):
            fired.append("Q1b" if (sq1b[i] or 0) > 0 else "Q1")
            notes.append("stuck")
        nh = L["n_hmates"][i]
        if "R9" in fired and np.isfinite(nh) and nh == 0 and status in ("suspect", "bad"):
            fired.append("Y")
            watch.append("no_yardstick")
        if bool(nyc[i]):
            fired.append("Y1")
            watch.append("no_yardstick_chop")
        if len(find) == 1:
            k, val = next(iter(find.items()))
            if k in STAT and val < BORDER:
                fired.append("R7")
                watch.append(k)
                del find[k]
        if "last 0.2 s or less" in str(L["reason"][i]) and L["fn"][i] == "Presence":
            fired.append("R8")
            watch.append("short_on")
        if bool(oh8[i]) and "occspk" not in find and "stuck" not in find and \
                not (np.isfinite(c_occ_like) and c_occ_like >= .5):
            fired.append("N1")
            watch.append("occ_hi")
        dq = []
        occ, rts = L["occ"][i], L["rep_time_s"][i]
        on_s = occ * L["hours"][i] * 3600 if np.isfinite(occ) else np.nan
        if np.isfinite(rts) and rts >= D1_MIN_S and np.isfinite(on_s) and rts >= D1_SHARE * on_s:
            dq.append("D1")
        if bool(n3d[i]):
            dq.append("D1n3")
        hsc = float(np.prod([1 - val for k, val in s.items() if k in find or val < .35])) if s else np.nan
        if np.isfinite(hsc) and hsc < .25:
            st = "bad"
        elif np.isfinite(hsc) and hsc < .70:
            st = "suspect"
        elif watch:
            st = "watch"
        elif L["n_on"][i] < 20:
            st = "not_enough_data"
        else:
            st = "ok"
        fams = {FAMILY[k] for k, val in find.items() if not (k in ("stuck", "dropout") and val == .35)}
        if st == "suspect" and len(fams) >= 2:
            st = "bad"
        rows.append(dict(st8=st, rules8=",".join(fired), watch8=",".join(watch), cleared8=",".join(notes),
                         left8=",".join(sorted(find)), score8=hsc, dq8=",".join(dq)))
    Xo = X.assign(occ_hi8=occ_hi8) if changed_hi8 else X
    R = pd.concat([Xo, pd.DataFrame(C, index=X.index), pd.DataFrame(rows, index=X.index)], axis=1)
    R["cfg"] = ""
    c1 = R.fn.eq("Count") & (R.dq8.str.contains("D1") | ((R.n_rep >= C1_N) & (R.n_rep >= C1_SHARE * R.n_on)))
    R.loc[c1, "cfg"] = "C1"
    c2 = R.fn.eq("Count") & R.cmode.eq("pulse") & ((R.n_ge60 >= C2_N60) | (R.n_ge5 >= C2_N5)) & ~R.cfg.str.contains("C1")
    R.loc[c2, "cfg"] = (R.loc[c2, "cfg"] + ",C2").str.strip(",")
    return R


