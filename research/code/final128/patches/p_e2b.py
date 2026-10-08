import os
p = os.path.join(os.environ.get("DC_WORK", os.path.expanduser("~/dc_work")), r"final_v7_next\src\detector_classifier\health_core.py")
s = open(p, encoding="utf-8").read()
a = s.index("def _events_to_bins(ev, start, end, detectors=None, bin_s: int = BIN_S) -> dict:")
b = s.index("# ======================================================================== statistics")
new = '''def _on_view(P_, T, dets):
    """the per-ON view behind events_to_bins (independent of the bin size): ON times, channel rows, continuous-ON
    durations (NaN inside a continuous ON or over a comms gap), chatter flags, the longest ON per channel and the
    channels that were ON when the window opened."""
    gaps0 = P_.g0                             # comms-gap start times
    td, ed, pd_ = P_.td, P_.ed, P_.pd_
    same_prev = np.r_[False, pd_[1:] == pd_[:-1]]
    prv_t, prv_e = np.r_[np.nan, td[:-1]], np.r_[0, ed[:-1]]
    on = ed == 82
    # continuous ON: from the ON that starts it to the next OFF (ONs inside it are extension re-calls);
    # still ON at the window end -> to the end.  Inner ONs get NaN (their time is already counted).
    c_end, c_start = continuous_on(td, ed, pd_, T)
    dur = np.where(c_start, c_end - td, np.nan)[on]
    gap = np.where(same_prev & (prv_e == 81), td - prv_t, np.nan)[on]
    ton, ci = td[on], np.searchsorted(dets, pd_[on]).astype(int)
    nd = len(dets)
    dur_max = np.zeros(nd)
    chat = np.zeros(0)
    if len(ton):
        long_ = np.where(dur > 60)[0]                     # an ON over a comms gap is not "stuck"
        if len(long_) and len(gaps0):
            k = np.searchsorted(gaps0, ton[long_])
            hit = (k < len(gaps0)) & (gaps0[np.minimum(k, len(gaps0) - 1)] < ton[long_] + dur[long_])
            dur[long_[hit]] = np.nan
        with np.errstate(invalid="ignore"):
            chat = (_ticks(gap) < CHAT_TICKS).astype(float)
        # longest ON per channel (ci is sorted: the events are ordered by channel)
        dd = np.nan_to_num(dur, nan=0.0)
        u, first = np.unique(ci, return_index=True)
        dur_max[u] = np.maximum(np.maximum.reduceat(dd, first), 0.0)
    first = np.r_[True, pd_[1:] != pd_[:-1]]
    lead = first & (ed == 81)
    return dict(ton=ton, ci=ci, dur=dur, chat=chat, dur_max=dur_max, lead_t=td[lead], lead_c=pd_[lead])


def _events_to_bins(ev, start, end, detectors=None, bin_s: int = BIN_S) -> dict:
    T = (end - start).total_seconds()
    nb = int(np.ceil(T / bin_s))
    P_ = _prep(ev, start, end)
    t, eid, par = P_.t, P_.eid, P_.par
    cov = np.bincount((t // bin_s).astype(int), minlength=nb)[:nb] > 0
    dm = np.isin(eid, (81, 82))
    chans = set(np.unique(par[dm]).tolist())
    if detectors is not None:
        chans |= {int(x) for x in detectors}
    dets = np.array(sorted(chans), dtype=int)
    nd = len(dets)
    idx = {c: i for i, c in enumerate(dets)}
    out = {k: np.zeros((nd, nb), np.float32) for k in ("n_on", "occ", "n_chat")}
    V = (_memo(P_, ("onview", start, end), lambda: _on_view(P_, T, dets)) if detectors is None
         else _on_view(P_, T, dets))
    ton, ci, dur = V["ton"], V["ci"], V["dur"]
    if len(ton):
        b0 = np.minimum((ton // bin_s).astype(int), nb - 1)
        cell = ci * nb + b0
        out["n_on"] = np.bincount(cell, minlength=nd * nb).reshape(nd, nb).astype(np.float32)
        out["n_chat"] = np.bincount(cell, weights=V["chat"], minlength=nd * nb).reshape(nd, nb).astype(np.float32)
        # occupancy, split across bins
        fin = np.isfinite(dur)
        s, f, c = ton[fin], np.minimum(ton[fin] + dur[fin], T), ci[fin]
        bs, bf = (s // bin_s).astype(int), np.minimum((f // bin_s).astype(int), nb - 1)
        nspan = bf - bs + 1
        rep = np.repeat(np.arange(len(s)), nspan)
        bb = bs[rep] + (np.arange(len(rep)) - np.repeat(np.cumsum(nspan) - nspan, nspan))
        ov = np.minimum(f[rep], (bb + 1) * bin_s) - np.maximum(s[rep], bb * bin_s)
        np.add.at(out["occ"], (c[rep], bb), np.clip(ov, 0, None))
    # a channel whose first event is an OFF was ON when the window opened: occupied until then
    for tt, cc in zip(V["lead_t"], V["lead_c"]):
        bf = min(int(tt // bin_s), nb - 1)
        i = idx[int(cc)]
        out["occ"][i, :bf] += bin_s
        out["occ"][i, bf] += tt - bf * bin_s
    # clock hour of each bin start (integer nanoseconds: = (start + k * bin_s seconds).hour)
    hrs = ((int(start.value) + np.arange(nb, dtype=np.int64) * (int(bin_s) * 1_000_000_000))
           // 3_600_000_000_000) % 24
    out.update(dets=dets, cov=cov, hour=hrs.astype(np.int32), dur_max=V["dur_max"].copy(), start=start, bin_s=bin_s,
               listed=None if detectors is None else np.isin(dets, [int(x) for x in detectors]))
    return out


'''
s = s[:a] + new + s[b:]
open(p, "w", encoding="utf-8").write(s)
print("patched")
