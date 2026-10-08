"""Note 104: the user's health_review_v2 answers turned into resolver changes, re-run on the note-96 data (w40 windows,
763 training signals, locked_v2 absent - asserted upstream in h96_occ / h96_health).  Hi-res log + the classifier's
outputs only (no prints / config / fault events).  The v2 answers are NOT used to set anything; they are compared at
the end (sanity check only).

Changes vs the note-96 resolver (h96_resolve):
  R1' (v2 row 3, rejected R1): 'stuck on a long zone during a queue -> ok' now also needs
      (a) the queue judged on HEALTHY phase peers only (not bad in pass 1, and not themselves held ON >= 99 % through
          the episode); (b) the detector's own 15-min occupancy correlating >= .7 with those peers outside the episode
          (with the traffic reference when there are no healthy peers); (c) no other 15-min bin in the sample >= 90 %
          ON while the phase traffic was light (<= .3x its mean count AND the healthy peers' occupancy not above its
          mean - in a queue counts fall too) - a zone held ON in light traffic is a fault, not a queue.
  R1b (v2 row 19, 'could just be high congestion, look closer'): a stuck finding on a long zone that R1' does not
      clear, of ANY length, becomes WATCH (not ok) when its time ON follows traffic (corr >= .8), it was already
      >= 80 % ON in the hour before the long ON (the top of its normal daily curve, not a jump), the traffic was at
      least its usual level (>= 1x), and condition (c) holds.
  N1' (v2 rows 16 / 18, 'two kinds of stop-bar count zone'): pulse vs normal mode decided by the median ON length
      (pulse = median ON <= 0.25 s, i.e. 1- or 2-tick pulses), not by the share of 1-tick ONs; the 'ON far longer
      than its type' limits are fitted per (type, mode).  Unknown mode (< 20 ONs) -> no N1.
  N3 (v2 row 17, 'something like erratic counts, for occupancy'): NEW check 'erratic time ON'.  Per 15-min bin
      (5-min in 30-min samples) the time ON expected from the detector's own count (count x its typical ON length);
      a SPIKE = a bin >= 3x that and >= 5 points above it, < 90 % ON (held ON is the stuck check's job), while the
      healthy phase peers are not busier than locally (< 1.5x their +-1 h median; signal congestion likewise when
      there are no peers).  Count-type zones only (a long zone's ON length follows its queue).  Statistic = minutes
      of ON time not explained in spike bins (>= 2 bins); limit = p99.5 of ok count-type detectors per sample length.
      Fires as a finding (score .35 at the limit -> 1 at 4x; family 'occ'): alone SUSPECT, with another independent
      finding BAD (package rule).  Between p99 and p99.5 -> WATCH (N3w).
  N5 (v2 rows 6 / 9, 'at night its actuations are higher than traffic on its phase'): count-type zone whose
      night / day rate is >= 3x the signal's (package night_day limit) even though its night rate stays below its
      day rate (the package check only looks when night > day), with >= 30 of its own actuations 00-05 -> WATCH.  Not on long zones (their day counts
      saturate when the zone stays occupied, which inflates the ratio: 0.83 % of ok long zones vs 0.19 % count-type).
Everything else = note 96 (R2-R9, N1 watch, scoring).

    python h104_resolve.py            -> %DC_WORK%/health104/resolved104.parquet, rates printed
    python h104_resolve.py evalset    -> note-79 evalset cross-check (circular positives; comparison only)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h96_resolve as V  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
DCW = V.DCW
IN96 = DCW / "health96"
OUT = DCW / "health104"
REPO = Path(__file__).resolve().parents[3]
LONG, COUNT_T, STAT, CHK = V.LONG, V.COUNT_T, V.STAT, V.CHK
FAMILY = dict(V.FAMILY, occspk="occ")
PULSE_MED = 0.25          # N1': pulse mode = median ON <= 0.25 s
R1_CORR, R1_FULL, R1_LIGHT = 0.70, 0.90, 0.30
R1B_PRE, R1B_FOLLOW, R1B_REF = 0.80, 0.80, 1.0
N5_REL = 3.0              # N5: package night_day limit, without its 'night rate > day rate' gate
N5_MIN = 30               # N5: >= 30 own actuations 00:00-05:00 (as the package's night checks: 30 expected)
N3_X, N3_PT, N3_FULL, N3_Q, N3_W = 3.0, 0.05, 0.90, 0.995, 0.99
RULES = dict(V.RULES)
RULES.update({
    "R1": "stuck on a long zone during a queue judged on HEALTHY peers, own occupancy follows them (>= .7), never "
          "held ON in light traffic elsewhere in the sample, < 60 min -> ok",
    "R1b": "stuck on a long zone not cleared by R1, occupancy follows traffic (>= .8), already >= 80 % ON the hour "
           "before, traffic >= its usual, never held ON in light traffic -> watch (possible congestion)",
    "N1": "count-type zone ON far longer than its type AND MODE (pulse = median ON <= 0.25 s / normal) at that "
          "traffic for >= 30 min, not congested, occupancy not following traffic -> watch",
    "N5": "count-type zone relatively busier at night: (own night/day rate) / max(signal's, .15) >= 3 even when "
          "its night rate stays below its day rate (the package check needs night > day), >= 30 actuations at night -> watch",
    "N3w": "N3 statistic between p99 and p99.5 of ok detectors -> watch",
    "N3": "NEW check 'erratic time ON': >= limit minutes of 15-min bins whose time ON is >= 3x (and >= 5 pt above) "
          "what its own count explains, while healthy phase peers are not busy; minutes of unexplained ON >= p99.5 of ok "
          "detectors -> finding (suspect alone)",
})


def wstart(w):
    return pd.Timestamp(V.S_WIN[w][0])


def load_bins(X):
    """note-96 bins + phase, pass-1 status, healthy-peer occupancy per bin (self excluded), busy mask."""
    Bn = pd.read_parquet(IN96 / "bins.parquet")
    Bn["bs"] = np.where(Bn.window.str.startswith("m30"), 300, 900)
    Bn = Bn.merge(X[["DeviceId", "window", "detector", "phase", "status"]], on=["DeviceId", "window", "detector"],
                  how="left")
    Bn["ph"] = Bn.phase.fillna(-1)
    Bn["hl"] = Bn.status.ne("bad") & Bn.occ.notna()
    Bn["ho"] = Bn.occ.where(Bn.hl)
    g = Bn.groupby(["DeviceId", "window", "ph", "b"]).ho.agg(["sum", "count"]).rename(
        columns={"sum": "hs", "count": "hc"}).reset_index()
    Bn = Bn.merge(g, on=["DeviceId", "window", "ph", "b"], how="left")
    self_h = Bn.hl.astype(int)
    Bn["peer_occ"] = ((Bn.hs - Bn.ho.fillna(0)) / (Bn.hc - self_h).where(Bn.hc - self_h > 0)).where(Bn.ph >= 0)
    Bn["peer_mean"] = Bn.groupby(["DeviceId", "window", "detector"]).peer_occ.transform("mean")
    Bn["busy"] = (Bn.peer_occ >= 1.5 * Bn.peer_mean) | (Bn.cong >= 1.5)
    Bn["hour"] = ((Bn.b * Bn.bs) // 3600 + Bn.window.map(lambda w: wstart(w).hour)) % 24
    return Bn.drop(columns=["hs", "hc", "ho"])


def n3_stats(Bn):
    """N3 'erratic time ON' per detector-window.  busy = the healthy phase peers' occupancy in the bin >= 1.5x their
    local median (+-1 h; +-20 min in 30-min samples), else the signal congestion index >= 1.5x its local median: a
    queue raises the neighbours too, a fault does not."""
    Bn = Bn.sort_values(["DeviceId", "window", "detector", "b"])
    g = Bn.groupby(["DeviceId", "window", "detector"])
    loc = lambda s: s.rolling(9, center=True, min_periods=3).median()  # noqa: E731
    pl, cl = g.peer_occ.transform(loc), g.cong.transform(loc)
    Bn["busy2"] = np.where(Bn.peer_occ.notna(), Bn.peer_occ >= 1.5 * pl.clip(lower=.01),
                           Bn.cong >= 1.5 * cl.clip(lower=.05))
    q = Bn[(Bn.n >= 3) & ~Bn.busy2 & (Bn.occ < N3_FULL)]
    dbar = (q.occ * q.bs / q.n).groupby([q.DeviceId, q.window, q.detector]).agg(["median", "count"])
    dbar = dbar[dbar["count"] >= 4]["median"].rename("dbar").reset_index()
    B = Bn.merge(dbar, on=["DeviceId", "window", "detector"])
    e = B.n * B.dbar / B.bs
    B["spk"] = (B.n >= 1) & (B.occ < N3_FULL) & (B.occ >= N3_X * e) & (B.occ - e >= N3_PT) & ~B.busy2
    B["exc"] = np.where(B.spk, (B.occ - e) * B.bs / 60, 0.0)
    B["spk_night"] = B.spk & (B.hour < 5)
    out = B.groupby(["DeviceId", "window", "detector"]).agg(
        n3_n=("spk", "sum"), n3_night=("spk_night", "sum"), n3_exc=("exc", "sum"),
        dbar=("dbar", "first")).reset_index()
    return out, B[B.spk][["DeviceId", "window", "detector", "b", "n", "occ", "exc"]]


def occ_hi_flags(X, Bn):
    """N1' = note-96 N1 with the mode from the median ON length (pulse <= 0.25 s)."""
    B = Bn.copy()
    B["rmax"] = B.groupby(["DeviceId", "window", "detector"]).ref.transform("max")
    B["tl"] = np.clip(np.floor(5 * B.ref / B.rmax.where(B.rmax > 0)), 0, 4)
    md = X[["DeviceId", "window", "detector", "med_dur"]].copy()
    m = pd.to_numeric(md.med_dur, errors="coerce")
    md["mode"] = np.where(m.isna(), "unknown", np.where(m <= PULSE_MED, "pulse", "normal"))
    B = B.merge(md[["DeviceId", "window", "detector", "mode"]], on=["DeviceId", "window", "detector"], how="left")
    ok = B[B.status.eq("ok") & B["mode"].ne("unknown")]
    lim = ok.groupby(["fn", "mode", "bs", "tl"]).occ.quantile(.995).rename("lim").reset_index()
    B = B.merge(lim, on=["fn", "mode", "bs", "tl"], how="left")
    # phase peers (all, as note 96) busy >= 40 % ON = queue
    g = B.groupby(["DeviceId", "window", "ph", "b"]).occ.agg(["sum", "count"]).rename(
        columns={"sum": "ps", "count": "pc"}).reset_index()
    B = B.merge(g, on=["DeviceId", "window", "ph", "b"], how="left")
    B["pocc_all"] = ((B.ps - B.occ.fillna(0)) / (B.pc - 1).where(B.pc > 1)).where(B.ph >= 0)
    B["hi"] = (B.occ > B.lim) & (B.cong < V.CONG_X) & ~(B.pocc_all >= V.PEER_BUSY) & B.fn.isin(COUNT_T) & \
        B["mode"].ne("unknown")
    g = B.groupby(["DeviceId", "window", "detector"]).agg(hi_n=("hi", "sum"), bs=("bs", "first"),
                                                          mode=("mode", "first")).reset_index()
    g["occ_hi"] = g.hi_n * g.bs >= 1800
    return g[["DeviceId", "window", "detector", "occ_hi", "hi_n", "mode"]], lim


def _corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def r1_context(X, Bn):
    """R1' / R1b context for every row with a stuck finding."""
    m = X.s_stuck >= .35
    keys = X.loc[m, ["DeviceId", "window"]].drop_duplicates()
    Bk = Bn.merge(keys, on=["DeviceId", "window"])
    G = {k: g for k, g in Bk.groupby(["DeviceId", "window"])}
    out = []
    for i, r in X[m].iterrows():
        b = G.get((r.DeviceId, r.window))
        a_, z_ = V._ts(r.ep_t0), V._ts(r.ep_t1)
        if b is None or pd.isna(a_) or pd.isna(z_):
            out.append(dict(i=i))
            continue
        bs = int(b.bs.iloc[0])
        t0 = wstart(r.window)
        ia, iz = int((a_ - t0).total_seconds() // bs), int(np.ceil((z_ - t0).total_seconds() / bs))
        own = b[b.detector == r.detector].set_index("b").sort_index()
        inep = own.index.to_series().between(ia, iz - 1).to_numpy()
        o, ref = own.occ.to_numpy(float), own.ref.to_numpy(float)
        # healthy peers: same phase, not bad in pass 1, not themselves held ON through the episode
        pp = b[(b.ph == r.phase) & (b.detector != r.detector) & b.status.ne("bad")] if np.isfinite(r.phase) else b.iloc[:0]
        keep = []
        for k, g in pp.groupby("detector"):
            ge = g[g.b.between(ia, iz - 1)]
            if len(ge) and (ge.occ >= .99).mean() >= .8:
                continue
            keep.append(k)
        pp = pp[pp.detector.isin(keep)]
        if len(pp):
            pm = pp.groupby("b").occ.mean().reindex(own.index).to_numpy(float)
            phx_h = np.nanmean(pm[inep]) / max(np.nanmean(pm), 1e-9) if inep.any() else np.nan
            corr_h = _corr(o[~inep], pm[~inep])
        else:
            phx_h, corr_h = np.nan, np.nan
        corr_ref = _corr(o[~inep], ref[~inep])
        rm = np.nanmean(ref)
        near = own.index.to_series().between(ia - 1, iz).to_numpy()      # episode +- one bin (partial edges)
        po = own.peer_occ.to_numpy(float)                                  # healthy peers' occupancy (queue?)
        quiet = ~(po > np.nanmean(po)) if np.isfinite(po).any() else np.ones(len(o), bool)
        light = ~near & (o >= R1_FULL) & (ref <= R1_LIGHT * rm) & quiet   # low counts AND peers not queued
        pre = own.occ[(own.index >= ia - 3600 // bs) & (own.index < ia)]
        out.append(dict(i=i, n_hpeer=len(keep), ep_phx_h=phx_h, c_occ_hpeer=corr_h, c_occ_ref_out=corr_ref,
                        light_full=int(light.sum()), pre_occ=pre.mean() if len(pre) else np.nan))
    E = pd.DataFrame(out).set_index("i")
    return X.join(E)


def resolve(X):
    rows = []
    for r in X.itertuples(index=False):
        s = {k: getattr(r, f"s_{k}") for k in CHK if np.isfinite(getattr(r, f"s_{k}", np.nan))}
        if np.isfinite(r.s_occspk):
            s["occspk"] = r.s_occspk
        find = {k: v for k, v in s.items() if v >= .35}
        notes, watch, fired = [], [], []
        if r.status2 != r.status:
            fired.append("R9")
        longz = r.fn in LONG
        follows = np.isfinite(r.c_occ_ref) and r.c_occ_ref >= V.OCC_FOLLOW
        for k in list(find):
            rule = None
            if k == "stuck" and longz:
                lf = r.light_full == 0 if np.isfinite(r.light_full) else False
                if r.n_hpeer and r.n_hpeer > 0:
                    queue = np.isfinite(r.ep_phx_h) and r.ep_phx_h >= V.CONG_X
                    corr_ok = np.isfinite(r.c_occ_hpeer) and r.c_occ_hpeer >= R1_CORR
                else:
                    queue = np.isfinite(r.ep_cong) and r.ep_cong >= V.CONG_X
                    corr_ok = np.isfinite(r.c_occ_ref_out) and r.c_occ_ref_out >= R1_CORR
                if np.isfinite(r.ep_dur) and r.ep_dur < 3600 and np.isfinite(r.ep_refx) and r.ep_refx >= V.FLOW_X \
                        and queue and corr_ok and lf:
                    rule = ("R1", "clear")
                elif follows and lf and np.isfinite(r.pre_occ) and r.pre_occ >= R1B_PRE and \
                        np.isfinite(r.ep_refx) and r.ep_refx >= R1B_REF:
                    rule = ("R1b", "watch")
            elif k in ("corr", "choppy") and longz and follows:
                rule = ("R2", "clear")
            elif k == "level" and longz and np.isfinite(r.lv_occx) and r.lv_occx >= 0.5:
                rule = ("R3", "clear")
            elif k == "volume" and np.isfinite(r.lanes) and r.lanes >= 2 and r.max5 < V.VOL2 and \
                    np.isfinite(r.c_cnt_ref) and r.c_cnt_ref >= 0.5:
                rule = ("R4", "clear")
            elif k == "rapid" and r.n_on < V.LOW_N and np.isfinite(r.share_x) and r.share_x >= 0.1:
                rule = ("R5", "watch")
            elif k == "dropout" and r.hours < 1:
                rule = ("R6", "watch")
            if rule:
                fired.append(rule[0])
                (watch if rule[1] == "watch" else notes).append(k)
                del find[k]
        if "occspk" in find:
            fired.append("N3")
        elif bool(r.n3_border):
            fired.append("N3w")
            watch.append("occspk")
        if len(find) == 1:
            k, v = next(iter(find.items()))
            if k in STAT and v < V.BORDER:
                fired.append("R7")
                watch.append(k)
                del find[k]
        if "last 0.2 s or less" in str(r.reason) and r.fn == "Presence":
            fired.append("R8")
            watch.append("short_on")
        if bool(r.occ_hi) and "occspk" not in find and not (s.get("stuck", 0) >= .35) and \
                not (np.isfinite(r.c_occ_ref) and r.c_occ_ref >= .5):
            fired.append("N1")
            watch.append("occ_hi")
        nd, snd = pd.to_numeric(r.night_day, errors="coerce"), pd.to_numeric(r.sig_night_day, errors="coerce")
        if r.fn in COUNT_T and r.night_cnt >= N5_MIN and np.isfinite(nd) and np.isfinite(snd) and nd / max(snd, .15) >= N5_REL and \
                not (s.get("night_day", 0) >= .35):
            fired.append("N5")
            watch.append("night_rel")
        hs = float(np.prod([1 - v for k, v in s.items() if k in find or v < .35])) if s else np.nan
        removed = len(notes) + len([w for w in watch if w in CHK]) > 0
        if not removed and "occspk" not in find:
            st = r.status2 if not (r.status2 in ("ok", "not_enough_data") and watch) else "watch"
            rows.append(dict(new_status=st, rules=",".join(fired), watch=",".join(watch), cleared="",
                             left=",".join(sorted(find)), new_score=r.health_score))
            continue
        if np.isfinite(hs) and hs < .25:
            st = "bad"
        elif np.isfinite(hs) and hs < .70:
            st = "suspect"
        elif watch:
            st = "watch"
        elif r.status2 == "not_enough_data":
            st = "not_enough_data"
        else:
            st = "ok"
        fams = {FAMILY[k] for k, v in find.items() if not (k in ("stuck", "dropout", "choppy") and v == .35)}
        if st == "suspect" and len(fams) >= 2:
            st = "bad"
        rows.append(dict(new_status=st, rules=",".join(fired), watch=",".join(watch), cleared=",".join(notes),
                         left=",".join(sorted(find)), new_score=hs))
    return pd.concat([X.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def rates(R, R96):
    out = []
    for wg in ("m30", "h3", "h24"):
        x, y = R[R.wg == wg], R96[R96.wg == wg]
        out.append(dict(window=wg, detectors=len(x), package=100 * x.status.isin(["suspect", "bad"]).mean(),
                        v96=100 * y.new_status.isin(["suspect", "bad"]).mean(),
                        v104=100 * x.new_status.isin(["suspect", "bad"]).mean(),
                        bad_v96=100 * (y.new_status == "bad").mean(), bad_v104=100 * (x.new_status == "bad").mean(),
                        watch_v96=100 * (y.new_status == "watch").mean(),
                        watch_v104=100 * (x.new_status == "watch").mean()))
    T = pd.DataFrame(out).round(2)
    print(T.to_string(index=False))
    for wg in ("h3", "h24"):
        x = R[R.wg == wg]
        print(wg, "rule fires per 100:", {k: round(100 * x.rules.str.split(",").apply(lambda l: k in l).mean(), 2)
                                          for k in RULES})
    return T


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    X = V.load()
    Bn = load_bins(X)
    oh, lim = occ_hi_flags(X, Bn)
    X = X.merge(oh, on=["DeviceId", "window", "detector"], how="left")
    X["occ_hi"] = X.occ_hi.fillna(False).astype(bool)
    nc = Bn[Bn.hour < 5].groupby(["DeviceId", "window", "detector"]).n.sum().rename("night_cnt").reset_index()
    X = X.merge(nc, on=["DeviceId", "window", "detector"], how="left")
    X["night_cnt"] = X.night_cnt.fillna(0)
    n3, spk = n3_stats(Bn)
    spk.to_parquet(OUT / "n3_spike_bins.parquet")
    X = X.merge(n3, on=["DeviceId", "window", "detector"], how="left")
    X["grp"] = np.where(X.fn.isin(LONG), "long", "count")
    okm = X.status.eq("ok") & X.n3_exc.notna()
    nl = X[okm].groupby(["grp", "wg"]).n3_exc.quantile([N3_W, N3_Q]).unstack().rename(
        columns={N3_W: "n3_watch", N3_Q: "n3_lim"}).reset_index()
    X = X.merge(nl, on=["grp", "wg"], how="left")
    two = (X.n3_n >= 2) & X.fn.isin(COUNT_T)     # count-type zones only: a long zone's ON length follows its queue
    X["s_occspk"] = [V_score(a, b) if t else (0.0 if (np.isfinite(a) and c) else np.nan)
                     for a, b, t, c in zip(X.n3_exc, X.n3_lim, two, X.fn.isin(COUNT_T))]
    X["n3_border"] = two & (X.n3_exc >= X.n3_watch) & (X.n3_exc < X.n3_lim)
    X["ep_dur"] = pd.to_numeric(X.ep_dur, errors="coerce")
    X["level_b"] = pd.to_numeric(X.level_b, errors="coerce")
    med = X[X.status == "ok"].groupby(["fn", "wg"]).share_ref.median().rename("share_med").reset_index()
    X = X.merge(med, on=["fn", "wg"], how="left")
    X["share_x"] = X.share_ref / X.share_med
    X = V.episode_context(X)
    X = V.level_context(X)
    X = r1_context(X, Bn)
    for c in ("n_hpeer", "ep_phx_h", "c_occ_hpeer", "c_occ_ref_out", "light_full", "pre_occ"):
        X[c] = pd.to_numeric(X[c], errors="coerce")
    R = resolve(X)
    for c in R.columns:
        if R[c].dtype == object and c not in ("DeviceId", "window"):
            R[c] = R[c].astype(str)
    R.to_parquet(OUT / "resolved104.parquet")
    lim.to_csv(OUT / "occ_hi_limits104.csv", index=False)
    nl.to_csv(OUT / "n3_limits.csv", index=False)
    print("N3 limits (min of spike bins, p99.5 of ok):\n", nl.to_string(index=False))
    R96 = pd.read_parquet(IN96 / "resolved.parquet", columns=["DeviceId", "window", "detector", "wg", "new_status",
                                                              "rules"])
    T = rates(R, R96)
    T.to_csv(OUT / "rates104.csv", index=False)
    # status transitions v96 -> v104
    J = R[["DeviceId", "window", "detector", "wg", "new_status", "rules"]].merge(
        R96, on=["DeviceId", "window", "detector", "wg"], suffixes=("", "_96"))
    for wg in ("h3", "h24"):
        j = J[J.wg == wg]
        print(wg, "v96 -> v104 (per 100):")
        print((100 * pd.crosstab(j.new_status_96, j.new_status) / len(j)).round(2))
    answers(R, R96)
    return R


def V_score(x, lim):
    if x is None or not np.isfinite(x) or not np.isfinite(lim):
        return np.nan
    if x < lim:
        return 0.0
    return float(0.35 + 0.65 * np.clip((x - lim) / (3 * lim), 0, 1))


def answers(R, R96):
    """sanity check only: how the user's v2 rows come out."""
    import openpyxl
    v = pd.read_csv(IN96 / "review_rows.csv")
    wb = openpyxl.load_workbook(REPO / "review" / "health_review_v2.xlsx", read_only=True)
    ws = wb["Cases"]
    a = pd.DataFrame([dict(n=r[0], ans=r[10], com=r[11]) for r in ws.iter_rows(min_row=4, values_only=True)
                      if r[0] is not None])
    v = v.merge(a, on="n")
    x = v.merge(R, left_on=["dev", "window", "detector"], right_on=["DeviceId", "window", "detector"]).merge(
        R96[["DeviceId", "window", "detector", "new_status", "rules"]], on=["DeviceId", "window", "detector"],
        suffixes=("", "_96"))
    cols = ["n", "rule", "ans", "status", "new_status_96", "rules_96", "new_status", "rules", "left", "watch",
            "n3_n", "n3_exc", "n3_lim", "mode", "occ_hi", "n_hpeer", "ep_phx_h", "c_occ_hpeer", "light_full", "pre_occ"]
    print(x[cols].round(2).to_string())
    x[cols].to_csv(OUT / "answers_v2_104.csv", index=False)


def evalset(nb=1000, seed=104):
    E = pd.read_parquet(DCW / "health79" / "evalset.parquet")
    E["DeviceId"] = E.DeviceId.str.lower()
    E = E[E.tier != "U"][["DeviceId", "detector", "cls"]].drop_duplicates(["DeviceId", "detector"])
    R = pd.read_parquet(OUT / "resolved104.parquet", columns=["DeviceId", "window", "detector", "wg", "new_status",
                                                               "rules"])
    R96 = pd.read_parquet(IN96 / "resolved.parquet", columns=["DeviceId", "window", "detector", "new_status"])
    R = R.merge(R96, on=["DeviceId", "window", "detector"], suffixes=("", "_96")).merge(
        E, on=["DeviceId", "detector"], how="left")
    R["old"] = R.new_status_96.isin(["suspect", "bad"])
    R["new"] = R.new_status.isin(["suspect", "bad"])
    rng = np.random.default_rng(seed)
    for wg in ("h3", "h24"):
        for nm, m in (("positives", R.cls.isin(["stuck", "chatter", "degraded"])),
                      ("presumed healthy", R.cls.eq("healthy"))):
            y = R[(R.wg == wg) & m]
            g = y.groupby("DeviceId").agg(n=("old", "size"), o=("old", "sum"), w=("new", "sum"))
            d = []
            for _ in range(nb):
                s_ = g.iloc[rng.integers(0, len(g), len(g))].sum()
                d.append(100 * (s_.w - s_.o) / s_.n)
            print(f"{wg} {nm:16s} n={len(y):5d} v96 {100 * y.old.mean():.2f} v104 {100 * y.new.mean():.2f} change "
                  f"{100 * (y.new.mean() - y.old.mean()):+.2f} [{np.quantile(d, .025):+.2f},{np.quantile(d, .975):+.2f}]")
        y = R[(R.wg == wg) & R.cls.isin(["stuck", "chatter", "degraded", "healthy"])]
        for rule in ("N3", "N3w", "R1", "R1b", "N1", "N5"):
            h = y.rules.str.split(",").apply(lambda l: rule in l)
            print(f"  {wg} {rule}: fires {h.sum()}, positives among them {100 * y[h].cls.ne('healthy').mean():.1f} % "
                  f"(base {100 * y.cls.ne('healthy').mean():.1f} %)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "evalset":
        evalset()
    else:
        build()
