"""Note 55: hi-res-only inputs for the stacked-group PICK rule of the per-lane ATSPM decode (atspm_decode.py, user rule
2026-09-30). For every frame signal, both periods, every window >= 30 min (the decode is gated there, note 54):

  health   health_core v5 (shipped defaults) on the window itself: status ok / suspect / bad / not_enough_data.
           Inputs = frame predicted phase, the run's OOF class probabilities, ln5 lanes spanned (all hi-res products).
  span     a detector that behaves like ONE zone spanning several lanes of its phase (e.g. a radar advance zone over
           lane-by-lane loops): >= 2 "covered" detectors on its predicted phase - most of each one's ONs coincide with
           its ONs (chance-corrected, at the pair's best lag within +-2 s, +-0.5 s) - which are NOT co-located with
           each other (different lanes), the union of them explains clearly more of its ONs than the best one alone
           (>= SPAN_GAIN), and it carries more actuations than any of them. `span_peers` lists the covered detectors;
           the decode puts the spanning detector on all their lanes instead of a lane of its own.
  track    correlation of the detector's binned counts with the total of the phase's predicted-Count detectors
           (those sharing a lane with it when lanes are known, else all of the phase); NaN when the phase has none.

    python ln6_pick.py [--run <run dir>] [--lanes ln5_lanes_v3s] [--tag v3s] [--workers 5]
-> %DC_WORK%/lanes/ln6_pick_<tag>.parquet (DeviceId, period, win, Detector, health, health_score, span, span_gain,
   span_peers, coloc_peers, track)
  coloc    `coloc_peers`: detectors of the same predicted phase co-located with it (either one's ONs >= COVER matched
           by the other's, chance-corrected) = its hi-res "stack"; the pick keys act only inside such a stack.
No technology, print or config fact is read. locked_v2 asserted absent. CPU only.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: F401,E402
import argparse  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ln1_cues as L1  # noqa: E402

warnings.filterwarnings("ignore")
OUT = L1.OUT
DCW = L1.DCW
CACHE = {"dec": DCW / "cache" / "det_intervals.parquet",
         "stg": DCW / "official" / "stg" / "cache" / "det_intervals.parquet"}
EVR = {"stg": DCW / "official" / "stg" / "cache" / "events", "dec": DCW / "cache" / "events"}
RUN = "run_ad6ea6959f_exclude_min5_clean_valnc_h3"
C7 = ["Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other"]
MIN_ON = 10
MIN_SECS = 1800
LAGMAX, BIN, TOL = 2.0, 0.25, 0.5     # co-location: best lag within +-2 s only
COVER = 0.5           # chance-corrected share of the covered detector's ONs matched by the spanning one
COLOC = 0.5           # two covered detectors this co-located are the same lane, not two lanes
SPAN_GAIN = 0.15      # union of covered peers explains >= this much more of its ONs than the best single one


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


ORDER_FREE = True     # note 77: per-phase loops in canonical behavioural order (False = channel order, notes 55-76)


def content_key(t, t0: float) -> tuple:
    """note 77 (= package pick.content_key / lanes.content_key): (actuations, sum and a hash of the ON offsets from t0
    in 0.1 s); never the channel number."""
    r = np.rint((np.asarray(t, float) - t0) * 10.0).astype(np.int64)
    return int(len(r)), int(r.sum()), int(((r % 65521) * (r % 65519) % 1000003).sum())


def order_dets(on: dict, dets, t0: float) -> list:
    """canonical behavioural order (busiest first) when ORDER_FREE, else as given (channel order)."""
    dets = [int(d) for d in dets]
    return sorted(dets, key=lambda d: content_key(on.get(d, ()), t0), reverse=True) if ORDER_FREE else dets


def match_frac(ta: np.ndarray, tb: np.ndarray, secs: float) -> tuple[float, float]:
    """(chance-corrected share of a's ONs with a b ON within TOL of the pair's best lag, best lag)."""
    if len(ta) == 0 or len(tb) == 0:
        return 0.0, 0.0
    lo = np.searchsorted(tb, ta - LAGMAX, "left")
    hi = np.searchsorted(tb, ta + LAGMAX, "right")
    cnt = hi - lo
    tot = int(cnt.sum())
    if tot == 0:
        return 0.0, 0.0
    ia = np.repeat(np.arange(len(ta)), cnt)
    jb = np.repeat(lo - (np.cumsum(cnt) - cnt), cnt) + np.arange(tot)
    d = tb[jb] - ta[ia]
    h, e = np.histogram(d, bins=np.arange(-LAGMAX, LAGMAX + BIN, BIN))
    hs = np.convolve(h, np.ones(4), "same")            # 1-s smoothing = the +-TOL window
    lag = float(e[np.argmax(hs)] + BIN / 2)
    ok = np.abs(d - lag) <= TOL
    hit = np.zeros(len(ta), bool)
    hit[ia[ok]] = True
    raw = float(hit.mean())
    chance = 1.0 - np.exp(-len(tb) / secs * 2 * TOL)
    return max(0.0, (raw - chance) / max(1e-9, 1.0 - chance)), lag


def union_frac(ta, peers, lags, secs) -> float:
    hit = np.zeros(len(ta), bool)
    rate = 0.0
    for tb, lag in zip(peers, lags):
        lo = np.searchsorted(tb, ta + lag - TOL, "left")
        hi = np.searchsorted(tb, ta + lag + TOL, "right")
        hit |= hi > lo
        rate += len(tb) / secs
    chance = 1.0 - np.exp(-rate * 2 * TOL)
    return max(0.0, (float(hit.mean()) - chance) / max(1e-9, 1.0 - chance))


def span_feats(on: dict, dets: list, secs: float) -> dict:
    """{det: (span flag, gain, peers)} for one predicted-phase group (dets with >= MIN_ON ONs)."""
    n = len(dets)
    M = np.zeros((n, n))
    L = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            if a != b:
                M[a, b], L[a, b] = match_frac(on[dets[a]], on[dets[b]], secs)
    out, coloc = {}, {}
    for i in range(n):
        coloc[dets[i]] = ",".join(str(dets[j]) for j in range(n) if j != i and max(M[i, j], M[j, i]) >= COVER)
    for i in range(n):
        cov = [j for j in range(n) if j != i and M[j, i] >= COVER and len(on[dets[i]]) > len(on[dets[j]])]
        if len(cov) < 2:
            out[dets[i]] = (False, 0.0, "")
            continue
        # keep covered peers that are not co-located with each other (distinct lanes); greedy by own coverage
        cov = sorted(cov, key=lambda j: -M[j, i])
        keep = []
        for j in cov:
            if all(M[j, k] < COLOC and M[k, j] < COLOC for k in keep):
                keep.append(j)
        if len(keep) < 2:
            out[dets[i]] = (False, 0.0, "")
            continue
        u = union_frac(on[dets[i]], [on[dets[j]] for j in keep], [L[i, j] for j in keep], secs)
        best = max(M[i, j] for j in keep)
        g = u - best
        out[dets[i]] = (bool(g >= SPAN_GAIN), float(g), ",".join(str(dets[j]) for j in keep))
    return out, coloc


def track_feats(on: dict, dets: list, cls: dict, lanes: dict, t0: float, secs: float) -> dict:
    nb = int(min(288, max(6, round(secs / 300.0))))
    edges = t0 + np.linspace(0, secs, nb + 1)
    cnt = {d: np.histogram(on[d], bins=edges)[0].astype(float) for d in dets}
    counts = [d for d in dets if cls.get(d) == "Count"]
    out = {}
    for d in dets:
        ref = [c for c in counts if c != d]
        if not ref:
            out[d] = np.nan
            continue
        ld = lanes.get(d, frozenset())
        same = [c for c in ref if ld and lanes.get(c) and (lanes[c] & ld)]
        tot = sum(cnt[c] for c in (same or ref))
        x = cnt[d]
        out[d] = float(np.corrcoef(x, tot)[0, 1]) if x.std() > 0 and tot.std() > 0 else np.nan
    return out


def work(args):
    dev, period, g = args
    import a2_features as A2F
    import health_core as hc
    import pyarrow.dataset as ds
    tab = ds.dataset(str(CACHE[period])).to_table(filter=ds.field("DeviceId") == dev,
                                                  columns=["Detector", "t_on"]).to_pandas().drop_duplicates()
    tab["t"] = tab.t_on.astype("datetime64[us]").astype("int64").to_numpy() / 1e6
    on_all = {int(d): np.sort(x.t.to_numpy()) for d, x in tab.groupby("Detector")}
    p = EVR[period] / f"DeviceId={dev}"
    ev = ds.dataset(p).to_table(filter=ds.field("EventId").isin(list(hc.ALLOWED))).to_pandas() if p.is_dir() else None
    rows = []
    for name, t0w, secs in A2F.WINDOWS[period]:
        if secs < MIN_SECS:
            continue
        w = g[g.win == name]
        if w.empty:
            continue
        t0 = pd.Timestamp(t0w).value / 1e9
        t1 = t0 + secs
        on = {}
        for d in w.Detector.astype(int):
            t = on_all.get(d, np.zeros(0))
            on[d] = t[(t >= t0) & (t < t1)]
        lanes = {int(d): frozenset(int(v) for v in s.split(",")) if isinstance(s, str) and s else frozenset()
                 for d, s in zip(w.Detector, w.lanes)}
        cls = dict(zip(w.Detector.astype(int), w.func))
        # health on the window
        hs = {}
        if ev is not None:
            ph = {int(d): float(v) for d, v in zip(w.Detector, w.pred_phase)}
            fn = {int(d): {c: float(r[k]) for k, c in enumerate(C7)} for d, r in zip(w.Detector, w[[f"P_{c}" for c in C7]].to_numpy())}
            ln = {d: max(1, len(s)) for d, s in lanes.items() if s}
            try:
                h = hc.health(ev, pd.Timestamp(t0w), pd.Timestamp(t0w) + pd.Timedelta(seconds=secs), None, ph, fn,
                              ln or None, None)
                hs = {int(d): (s, float(sc)) for d, s, sc in zip(h.detector, h.status, h.health_score)}
            except Exception as e:                       # noqa: BLE001
                log(f"health failed {dev} {period} {name}: {e}")
        sp, tr, co = {}, {}, {}
        for ph_, gg in w.groupby("pred_phase"):
            dets = order_dets(on, [int(d) for d in gg.Detector if len(on[int(d)]) >= MIN_ON], t0)
            if len(dets) >= 2:
                a_, b_ = span_feats(on, dets, secs)
                sp.update(a_)
                co.update(b_)
                tr.update(track_feats(on, dets, cls, lanes, t0, secs))
        for d in w.Detector.astype(int):
            s, sc = hs.get(d, ("not_run", np.nan))
            f, gain, peers = sp.get(d, (False, 0.0, ""))
            rows.append((dev, period, name, d, s, sc, f, gain, peers, co.get(d, ""), tr.get(d, np.nan)))
    return pd.DataFrame(rows, columns=["DeviceId", "period", "win", "Detector", "health", "health_score", "span",
                                       "span_gain", "span_peers", "coloc_peers", "track"])


def frame(run: str, lanes_tag: str) -> pd.DataFrame:
    import v3_retrain as V
    V.set_frame("v6e")
    k = pd.read_parquet(V.FEATS, columns=V.KEY + ["fold", "pred_phase", "det_n_on"])
    k["DeviceId"] = k.DeviceId.str.lower()
    assert not k.DeviceId.isin(L1.locked()).any()
    P = np.zeros((len(k), len(C7)), np.float32)
    for s in (0, 1, 2):
        Ps, hv, got = V.load_oof(k, V.OUT / run, "first.all.wi", s, range(6))
        assert len(got) == 6
        P += Ps / 3
    for i, c in enumerate(C7):
        k[f"P_{c}"] = P[:, i]
    k["func"] = np.array(C7, object)[P.argmax(1)]
    Ln = pd.read_parquet(OUT / f"ln5_lanes_{lanes_tag}.parquet", columns=["DeviceId", "Detector", "period", "win", "phase", "lanes"])
    Ln = Ln.astype({"Detector": k.Detector.dtype})
    k = k.merge(Ln, on=["DeviceId", "Detector", "period", "win"], how="left")
    k["lanes"] = k.lanes.where(k.phase.eq(k.pred_phase), "")
    return k


def main(run, lanes_tag, tag, workers):
    import a2_features as A2F
    k = frame(run, lanes_tag)
    big = {p: {n for n, _, s in A2F.WINDOWS[p] if s >= MIN_SECS} for p in A2F.WINDOWS}
    k = k[[w in big[p] for p, w in zip(k.period, k.win)]]
    jobs = [(dev, per, g[["Detector", "win", "pred_phase", "func", "lanes"] + [f"P_{c}" for c in C7]].copy())
            for (dev, per), g in k.groupby(["DeviceId", "period"])]
    log(f"{len(jobs)} signal-periods, {len(k):,} detector-windows")
    out, t0 = [], time.time()
    from multiprocessing import Pool
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(work, jobs, chunksize=1)):
            out.append(r)
            if i % 100 == 0:
                log(f"  {i}/{len(jobs)} ({time.time()-t0:.0f}s)")
    D = pd.concat(out, ignore_index=True)
    assert not D.DeviceId.isin(L1.locked()).any()
    D.to_parquet(OUT / f"ln6_pick_{tag}.parquet", index=False)
    log(f"wrote {len(D):,}; health {D.health.value_counts().to_dict()}; span {int(D.span.sum())}; "
        f"track known {float(D.track.notna().mean()):.3f} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--lanes", default="ln5_lanes_v3e")
    ap.add_argument("--tag", default="v3e")
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    main(a.run, a.lanes, a.tag, a.workers)
