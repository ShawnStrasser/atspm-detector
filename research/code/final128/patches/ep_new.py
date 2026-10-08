def episodes_ctx(E, M, phase_x, qtype, typ):
    """queue context per stuck ON (h110_resolve.episodes + traffic_evidence).  E: episodes (ts / tf seconds from the
    window start, dur_s), M: the bins_ctx matrices, phase_x: detector -> phase (scored table only), qtype: type -> p95
    % ON of healthy detectors, typ: detector -> type.  Per phase the mates' bins are a small matrix; the mates' mean
    % ON per bin is the research's pandas mean over the same rows (same compensated sum; cached per set of mates)."""
    cols = ["n_hpeer_e", "phx_h_e", "corr_h_e", "refx_e", "light_e", "cover_e", "trafx_e"]
    if not len(E):
        return pd.DataFrame(columns=cols)
    dets = M["dets_i"]
    bs = int(M["bs"]) if len(dets) else 900
    nb = int(M["nb"]) if len(dets) else 0
    row = {int(d): i for i, d in enumerate(dets)}
    qt_all = np.array([qtype.get(typ.get(int(d)), np.nan) if typ.get(int(d)) is not None else np.nan for d in dets],
                      float)
    piv = {}
    for p in np.unique(M["ph"]):
        mem = np.flatnonzero(M["ph"] == p)
        piv[float(p)] = (dets[mem], M["occ"][mem], M["n"][mem], M["st"][mem], M["fn"][mem], qt_all[mem], mem)
    bo = np.arange(nb)
    pm_cache, out = {}, []
    for r in E.itertuples():
        p = phase_x.get(int(r.detector), np.nan)
        res = {}
        i0 = row.get(int(r.detector))
        if i0 is None or not nb:
            out.append(res)
            continue
        ia, iz = int(r.ts // bs), int(np.ceil(r.tf / bs))
        inep = (bo >= ia) & (bo <= iz - 1)
        o, ref = M["occ"][i0], M["ref"][i0]
        keep = []
        phx_h, corr_h, cover = np.nan, np.nan, np.nan
        if np.isfinite(p) and p in piv:
            dl, occ, n, st, fn, qt, mem = piv[p]
            a0, a1 = max(ia, 0), max(min(iz, nb), 0)
            cand = (dl != r.detector) & np.array([x != "bad" for x in st])
            for j in np.flatnonzero(cand):
                seg = occ[j, a0:a1]
                if a1 > a0 and (seg >= .99).sum() / (a1 - a0) >= .8:
                    continue
                keep.append(int(dl[j]))
            if keep:
                kk = tuple(keep)
                if (p, kk) not in pm_cache:
                    s_, k_ = _ksum(occ[np.isin(dl, keep)])
                    with np.errstate(invalid="ignore", divide="ignore"):
                        pm_cache[(p, kk)] = np.where(k_ > 0, s_ / np.where(k_ > 0, k_, 1), np.nan)
                pm = pm_cache[(p, kk)]
                with _quiet():
                    phx_h = np.nanmean(pm[inep]) / max(np.nanmean(pm), 1e-9) if inep.any() else np.nan
                corr_h = _corr(o[~inep], pm[~inep])
                ki = np.isin(dl, keep)
                with np.errstate(invalid="ignore"):
                    hit = (occ[ki, a0:a1] >= qt[ki, None]).any(0)
                nb_ = int(inep.sum())
                cover = int(hit.sum()) / max(nb_, 1) if nb_ else np.nan
        with _quiet():
            refx = np.nanmean(ref[inep]) / max(np.nanmean(ref), 1e-9) if inep.any() else np.nan
            rm = np.nanmean(ref)
        near = (bo >= ia - 1) & (bo <= iz)
        po = M["peer_occ"][i0]
        with _quiet():
            quiet = ~(po > np.nanmean(po)) if np.isfinite(po).any() else np.ones(len(o), bool)
        with np.errstate(invalid="ignore"):
            light = ~near & (o >= R1_FULL) & (ref <= R1_LIGHT * rm) & quiet
        res.update(n_hpeer_e=len(keep), phx_h_e=phx_h, corr_h_e=corr_h, refx_e=refx, light_e=int(light.sum()),
                   cover_e=cover)
        # traffic evidence: the phase's Advance / Count detectors (else its other mates) during the ON
        if np.isfinite(p) and p in piv:
            dl, occ, n, st, fn, qt, mem = piv[p]
            oth = dl != r.detector
            tr = oth & np.isin(fn.astype(str), TRAF) & np.array([f is not None for f in fn])
            if not tr.any():
                tr = oth
            if tr.any():
                tot = np.nansum(n[tr], 0)                     # integer counts: exact in any order
                ja, jz = ia, iz
                fa, fz = int(np.ceil(r.ts / bs)), int(r.tf // bs)
                if fz > fa:
                    ja, jz = fa, fz
                ins = tot[max(ja, 0):max(min(jz, nb), 0)]
                res["trafx_e"] = ins.mean() / max(tot.mean(), 1e-9) if len(ins) else np.nan
        out.append(res)
    C = pd.DataFrame(out, index=E.index)
    for c in cols:
        if c not in C:
            C[c] = np.nan
    return C


