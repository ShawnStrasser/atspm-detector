import os
p = os.path.join(os.environ.get("DC_WORK", os.path.expanduser("~/dc_work")), r"final_v7_next\src\detector_classifier\health_core.py")
s = open(p, encoding="utf-8").read()


def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:90], s.count(a))
    s = s.replace(a, b)


# (1) left merges on the same detectors in the same order -> one concat
rep('''def _add_cols(df: pd.DataFrame, cols: dict) -> pd.DataFrame:''', '''def _merge_same(left: pd.DataFrame, right: pd.DataFrame, how: str = "left") -> pd.DataFrame:
    """left.merge(right, on="detector", how=how) -- as one concat when both hold the same detectors in the same order
    and share no other column (then the merge adds nothing and changes no type)."""
    if (len(left) == len(right) and np.array_equal(left.detector.to_numpy(), right.detector.to_numpy())
            and not (set(left.columns) & set(right.columns)) - {"detector"}):
        r = right.drop(columns="detector")
        r.index = left.index
        return pd.concat([left, r], axis=1)
    return left.merge(right, on="detector", how=how)


def _add_cols(df: pd.DataFrame, cols: dict) -> pd.DataFrame:''')
rep('''    st = st.merge(sh.drop(columns=[c for c in ("pred_phase",) if c in sh]), on="detector", how="left")''',
    '''    st = _merge_same(st, sh.drop(columns=[c for c in ("pred_phase",) if c in sh]))''')
rep('''        st = st.merge(a[["detector"] + ACOLS], on="detector", how="left")''',
    '''        st = _merge_same(st, a[["detector"] + ACOLS])''')
rep('''    return h.merge(st, on="detector")''', '''    return _merge_same(h, st, how="inner")''')
# (2) the phase-reference silent runs written column by column (same values and types as the .loc block)
rep('''        st.loc[ph2, cols] = sp.loc[ph2, cols].to_numpy()
        st.loc[ph2, "drop_ref"] = "phase"''', '''        for c in cols:
            v = st[c].to_numpy().copy()
            v[ph2] = sp[c].to_numpy()
            st[c] = v
        v = st["drop_ref"].to_numpy().copy()
        v[ph2] = "phase"
        st["drop_ref"] = v''')
# (4) rapid group limits, vectorised
rep('''    lt1 = {int(d): _grp_lim(pf, md, bool(s2))
           for d, pf, md, s2 in zip(st.detector, st.pulse_frac.astype(float), st.med_dur.astype(float), span2)}''',
    '''    lt1 = dict(zip(st.detector.astype(int).tolist(), _grp_lims(st.pulse_frac.astype(float).to_numpy(),
                                                               st.med_dur.astype(float).to_numpy(), span2)))''')
rep('''def _grp_lim(pulse_frac, med_dur, span2: bool):''', '''def _grp_lims(pulse_frac, med_dur, span2) -> list:
    """_grp_lim for arrays."""
    g = behaviour_group(pulse_frac, med_dur)
    out = []
    for gg, md, s2 in zip(g.tolist(), med_dur, span2):
        lim = (RAPID_LIM_SPAN if bool(s2) else RAPID_LIM).get(gg if np.isfinite(md) else "")
        out.append(lim[GROUP_LT1] if lim else np.nan)
    return out


def _grp_lim(pulse_frac, med_dur, span2: bool):''')
# (3) partner notes: counts / ties / correlations cached per window (they do not depend on the phases)
rep('''    on = _cont_on_starts(ev, start, end)
    nb = max(int(T // 900), 1)
    cnt = {d: np.bincount(np.minimum((x // 900).astype(int), nb - 1), minlength=nb) for d, x in on.items()}
    n = {d: len(x) for d, x in on.items()}
    ckey = {d: int(_ticks(x).sum()) for d, x in on.items()}            # content tie-break, never the channel number
    out = {}''', '''    on = _cont_on_starts(ev, start, end)
    nb = max(int(T // 900), 1)

    def prep():
        return ({d: np.bincount(np.minimum((x // 900).astype(int), nb - 1), minlength=nb) for d, x in on.items()},
                {d: len(x) for d, x in on.items()},
                {d: int(_ticks(x).sum()) for d, x in on.items()},       # content tie-break, never the channel number
                {})
    cnt, n, ckey, cc = (_memo(ev, ("partner", start, end), prep) if isinstance(ev, Prep) else prep())
    out = {}''')
rep('''            for k in mates:
                a, b = cnt[d], cnt[k]
                c = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan''', '''            for k in mates:
                c = cc.get((d, k))
                if c is None:
                    a, b = cnt[d], cnt[k]
                    c = cc[(d, k)] = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan''')
open(p, "w", encoding="utf-8").write(s)
print("patched")
