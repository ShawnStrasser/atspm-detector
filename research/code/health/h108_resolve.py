"""Note 108: health checks re-run with PER-TYPE thresholds taken from the baseline of normal behaviour (h108_base),
the time-of-day profile check (h108_profile) and the user's corrections.  Saved statistics only (w40 windows, 763
training signals, locked_v2 absent - asserted upstream); hi-res log + the classifier's outputs only.

Type = model function x model lane span (1 / 2+); inside Count only, the MEASURED ON mode (pulse / normal, clean ONs).
No 'long zone' type anywhere: what notes 96 / 104 keyed on long zones is now a MEASURED pattern of the detector itself.

Limits (presumed healthy, leave-own-check-out, p99.5 per cell; cell = type [x Count mode] x sample length, fallback
function x length, then length; >= 100 healthy per cell):
  choppy      chop15 (erratic counts)                          suspect L, bad 2L
  chatter     re-ON < 0.3 s share (>= 50 ONs)                  suspect L, bad 2L
  rapid       ON->ON < 0.5 s, < 1 s, burst share (>= 50 ONs)   ratio max(v / L) - suspect 1, bad 2  (lane span in the cell)
  volume      max actuations in 5 min                          suspect L, bad 5/3 L
  stuck       longest continuous ON                            suspect clip(L, 5 min, 15 min), bad 60 min
  N3          minutes of time ON its count does not explain    suspect L (all classes; was count-type zones only)
  profile P1  time-of-day profile distance (24 h; REPLACES 'doesn't follow traffic', 'busier at night' and N5)
Unchanged package checks: goes silent, count drops, misses at night.
Resolver (replaces v104's long-zone keyed rules):
  Q1  stuck cleared as a queue (any class): healthy phase peers >= 1.5x their usual time ON, own time ON tracks them
      (r >= .7), never held ON in light traffic, < 60 min.  (R1b 'possible congestion -> watch' dropped: 08CM405 d37.)
  Q2  erratic / count-drop / N3 finding cleared (any class) only when the detector itself shows the measured queue pattern
      (busier half of the day: time ON rises with traffic faster than counts - counts flatten or fall) AND its time ON
      follows its healthy same-class phase mates' time ON (r >= .8; all healthy mates if no same class) - like with
      like.  (v104 R2 compared time ON with traffic COUNTS: 2B334 d13.)
  Y   no healthy phase mate left after R9: a finding pass 1 made against the phase cannot be confirmed or cleared ->
      watch ('no yardstick'), not ok (12032 d41).
  D1  ON logged again without an OFF: a data-quality NOTE (no status change) when >= 70 % of its time ON (and
      >= 10 min) sits in such chains; N3 minutes
      that are mostly such chains (>= 50 %) become D1, not N3 (2B058 d46, 2B068 d19).  The Count mode uses clean ONs.
  R5, R6, R7, R8, R9 as note 96; N1 (ON far longer than its class / mode at that traffic) for every class.

    python h108_resolve.py [q]  -> health108/resolved108_q<1000q>.parquet, limits108_q*.csv, rates108_q*.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h108_base as B  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 80)
pd.set_option("display.max_rows", 300)
DCW, OUT = B.DCW, B.OUT
REPO = Path(__file__).resolve().parents[3]
Q, MIN_CELL = 0.995, 100
TAG = "995"
PQ = None          # profile quantile (default = Q); the profile replaces two checks
FAMILY = {"chatter": "fast", "rapid": "fast", "dropout": "silent", "stuck": "stuck", "level": "level",
          "night_drop": "level", "choppy": "shape", "volume": "shape", "prof": "shape", "occspk": "occ"}
STAT = ("dropout", "rapid", "level", "choppy", "night_drop", "prof")
BORDER, LOW_N = 0.42, 100
STUCK_CLIP, STUCK_BAD = (300.0, 900.0), 3600.0
Q2_CORR, Q1_CORR = 0.80, 0.70
D1_SHARE, D1_MIN_S = 0.70, 600.0


def score(x, lo, hi):
    x, lo, hi = (np.asarray(v, float) for v in (x, lo, hi))
    s = 0.35 + 0.65 * np.clip((x - lo) / np.maximum(hi - lo, 1e-9), 0, 1)
    return np.where(~np.isfinite(x) | ~np.isfinite(lo), np.nan, np.where(x < lo, 0.0, s))


def cell_limit(X, col, hl, keys_list, q=Q):
    lim = pd.Series(np.nan, index=X.index)
    src = pd.Series("", index=X.index)
    for keys in keys_list:
        g = X[hl & X[col].notna()].groupby(keys)[col].agg(["count", lambda s: s.quantile(q)])
        g.columns = ["cnt", "lim"]
        g = g[g.cnt >= MIN_CELL]
        m = X[keys].merge(g, left_on=keys, right_index=True, how="left")
        fill = lim.isna().to_numpy() & m.lim.notna().to_numpy()
        lim[fill] = m.lim.to_numpy()[fill]
        src[fill] = "+".join(keys)
    return lim, src


KEYS = (["fn", "span", "cmode", "wg"], ["fn", "span", "wg"], ["span", "wg"], ["wg"])


def limits(X):
    """per-type limits from presumed-healthy (leave-own-check-out) detector-windows; returns X with lim_* columns."""
    X = X.copy()
    n50 = X.n_on >= 50
    for col, fam, extra in (("chop15", "shape", None), ("chat_frac", "fast", n50), ("ioi_lt05", "fast", n50),
                            ("ioi_lt1", "fast", n50), ("burst_frac", "fast", n50), ("max5", "shape", None),
                            ("stuck_x", "stuck", None), ("n3_exc", "occ", X.n3_n >= 2), ("rep_frac", "fast", n50)):
        hl = X[f"hl_{fam}"] & (extra if extra is not None else True)
        X[f"lim_{col}"], X[f"src_{col}"] = cell_limit(X, col, hl, KEYS, q=Q)
    return X


def context(X):
    """like-with-like occupancy context from the note-104 bins: correlation of own 15-min time ON with healthy
    same-class phase mates (else all healthy phase mates), outside nothing; number of healthy phase mates."""
    Bn = pd.read_parquet(DCW / "health104" / "bins_ctx.parquet", columns=["DeviceId", "window", "detector", "fn", "b",
                                                                          "occ", "ph", "hl", "peer_occ"])
    Bn = Bn[~Bn.window.str.startswith("m30")]
    Bn["ho"] = Bn.occ.where(Bn.hl)
    g = Bn.groupby(["DeviceId", "window", "ph", "fn", "b"]).ho.agg(["sum", "count"]).rename(
        columns={"sum": "fs", "count": "fc"}).reset_index()
    Bn = Bn.merge(g, on=["DeviceId", "window", "ph", "fn", "b"], how="left")
    selfh = Bn.hl.astype(int)
    Bn["same_occ"] = ((Bn.fs - Bn.ho.fillna(0)) / (Bn.fc - selfh).where(Bn.fc - selfh > 0)).where(Bn.ph >= 0)

    def corr(d, a, b):
        x, y = d[a], d[b]
        m = x.notna() & y.notna()
        d = d[m]
        if len(d) < 6:
            return np.nan
        return d[a].corr(d[b])
    out = []
    for k, d in Bn.groupby(["DeviceId", "window", "detector"], sort=False):
        out.append((*k, corr(d, "occ", "same_occ"), corr(d, "occ", "peer_occ")))
    C = pd.DataFrame(out, columns=["DeviceId", "window", "detector", "c_occ_same", "c_occ_all"])
    C["c_occ_like"] = C.c_occ_same.fillna(C.c_occ_all)
    return C


def build():
    X = pd.read_parquet(OUT / "base.parquet")
    X["stuck_x"] = np.fmax(pd.to_numeric(X.ep_dur, errors="coerce"), X.max_on_s)
    if (OUT / "ctx108.parquet").exists():
        C = pd.read_parquet(OUT / "ctx108.parquet")
    else:
        C = context(X)
        C.to_parquet(OUT / "ctx108.parquet")
    X = X.merge(C, on=["DeviceId", "window", "detector"], how="left")
    P24 = pd.read_parquet(OUT / "prof.parquet", columns=["DeviceId", "window", "detector", "type", "fn", "band",
                                                        "hl_prof", "d_cnt", "d_occ"])
    for c in ("d_cnt", "d_occ"):                       # profile limits at the same quantile as the other checks
        P24[c + "_lim"], _ = cell_limit(P24, c, P24.hl_prof, (["type", "band", "window"], ["fn", "band", "window"],
                                                              ["band", "window"]), q=PQ or Q)
    P24["prof_x"] = np.fmax(P24.d_cnt / P24.d_cnt_lim, P24.d_occ / P24.d_occ_lim)
    X = X.merge(P24[["DeviceId", "window", "detector", "prof_x", "d_cnt", "d_occ", "d_cnt_lim", "d_occ_lim"]],
                on=["DeviceId", "window", "detector"], how="left")
    X = limits(X)
    # healthy phase mates after pass 1 (status != bad), self excluded
    ph = X.phase.fillna(-1)
    X["hm"] = X.status.ne("bad").astype(int)
    X["n_hmates"] = X.groupby(["DeviceId", "window", ph]).hm.transform("sum") - X.hm
    X.loc[X.phase.isna(), "n_hmates"] = np.nan
    X = occ_hi(X)
    R = resolve(X)
    R.to_parquet(OUT / f"resolved108_q{TAG}.parquet")
    lim_table(R)
    rates(R)
    return R


def occ_hi(X):
    """N1 for every class: >= 30 min of bins above the p99.5 time ON of ok detectors of the same class (Count: and
    measured mode) at that traffic level, signal not congested, phase peers < 40 % ON."""
    Bn = pd.read_parquet(DCW / "health104" / "bins_ctx.parquet", columns=["DeviceId", "window", "detector", "fn", "b",
                                                                          "occ", "ref", "cong", "bs", "ph", "status"])
    Bn = Bn.merge(X[["DeviceId", "window", "detector", "cmode"]], on=["DeviceId", "window", "detector"])
    Bn = Bn[Bn.cmode.ne("unknown")]
    Bn["rmax"] = Bn.groupby(["DeviceId", "window", "detector"]).ref.transform("max")
    Bn["tl"] = np.clip(np.floor(5 * Bn.ref / Bn.rmax.where(Bn.rmax > 0)), 0, 4)
    ok = Bn[Bn.status.eq("ok")]
    lim = ok.groupby(["fn", "cmode", "bs", "tl"]).occ.quantile(.995).rename("lim").reset_index()
    Bn = Bn.merge(lim, on=["fn", "cmode", "bs", "tl"], how="left")
    g = Bn.groupby(["DeviceId", "window", "ph", "b"]).occ.agg(["sum", "count"]).rename(
        columns={"sum": "ps", "count": "pc"}).reset_index()
    Bn = Bn.merge(g, on=["DeviceId", "window", "ph", "b"], how="left")
    Bn["pocc"] = ((Bn.ps - Bn.occ.fillna(0)) / (Bn.pc - 1).where(Bn.pc > 1)).where(Bn.ph >= 0)
    Bn["hi"] = (Bn.occ > Bn.lim) & (Bn.cong < 1.5) & ~(Bn.pocc >= 0.40)
    g = Bn.groupby(["DeviceId", "window", "detector"]).agg(hi_n8=("hi", "sum"), bs8=("bs", "first")).reset_index()
    g["occ_hi8"] = g.hi_n8 * g.bs8 >= 1800
    X = X.merge(g[["DeviceId", "window", "detector", "occ_hi8"]], on=["DeviceId", "window", "detector"], how="left")
    X["occ_hi8"] = X.occ_hi8.fillna(False).astype(bool)
    return X


def scores(X):
    S = pd.DataFrame(index=X.index)
    for k in ("dropout", "level", "night_drop"):                       # package checks kept as they are
        S[k] = pd.to_numeric(X[f"s_{k}"], errors="coerce")
    S["choppy"] = score(X.chop15, X.lim_chop15, 2 * X.lim_chop15)
    n50 = X.n_on >= 50
    S["chatter"] = np.where(n50, score(X.chat_frac, X.lim_chat_frac, 2 * X.lim_chat_frac), np.nan)
    rr = np.fmax(np.fmax(X.ioi_lt05 / X.lim_ioi_lt05, X.ioi_lt1 / X.lim_ioi_lt1), X.burst_frac / X.lim_burst_frac)
    X["rapid8"] = np.where(n50, rr, np.nan)
    S["rapid"] = score(X.rapid8, 1.0, 2.0)
    S["volume"] = score(X.max5, X.lim_max5, 5 / 3 * X.lim_max5)
    sl = X.lim_stuck_x.clip(*STUCK_CLIP)
    X["stuck_lim8"] = sl
    S["stuck"] = score(X.stuck_x.fillna(0), sl, np.fmax(STUCK_BAD, 2 * sl))
    # N3: count-free; minutes mostly from ON-again-without-OFF chains -> D1 (data quality), not N3
    n3ok = (X.n3_n >= 2)
    S["occspk"] = np.where(n3ok, score(X.n3_exc, X.lim_n3_exc, 4 * X.lim_n3_exc), np.where(X.n3_exc.notna(), 0.0,
                                                                                           np.nan))
    X["n3_dq"] = (S.occspk >= .35) & (X.rep_time_s / 60 >= 0.5 * X.n3_exc)
    S.loc[X.n3_dq, "occspk"] = 0.0
    S["prof"] = np.where(X.wg.eq("h24"), score(X.prof_x, 1.0, 2.0), np.nan)
    # package-run share of the stuck episode: capped at suspect when the package capped it (co-stuck / recovery)
    cap = (pd.to_numeric(X.s_stuck, errors="coerce") == 0.35)
    S.loc[cap & (S.stuck > .35), "stuck"] = 0.35
    return S


def resolve(X):
    S = scores(X)
    for k in S.columns:
        X[f"s8_{k}"] = S[k]
    X["queue_pat8"] = (X.elhi_occ > 0) & (X.elhi_occ > X.elhi_cnt)
    rows = []
    for i, r in enumerate(X.itertuples(index=False)):
        s = {k: S.iat[i, j] for j, k in enumerate(S.columns) if np.isfinite(S.iat[i, j])}
        find = {k: v for k, v in s.items() if v >= .35}
        notes, watch, fired = [], [], []
        if r.status2 != r.status:
            fired.append("R9")
        for k in list(find):
            rule = None
            if k == "stuck":
                lf = r.light_full == 0 if np.isfinite(r.light_full) else False
                if r.n_hpeer and r.n_hpeer > 0:
                    queue = np.isfinite(r.ep_phx_h) and r.ep_phx_h >= 1.5
                    corr_ok = np.isfinite(r.c_occ_hpeer) and r.c_occ_hpeer >= Q1_CORR
                else:
                    queue, corr_ok = False, False
                ed = pd.to_numeric(r.ep_dur, errors="coerce")
                if np.isfinite(ed) and ed < 3600 and np.isfinite(r.ep_refx) and r.ep_refx >= 0.5 and queue and \
                        corr_ok and lf:
                    rule = ("Q1", "clear")
            elif k in ("choppy", "level", "occspk") and bool(r.queue_pat8) and np.isfinite(r.c_occ_like) and \
                    r.c_occ_like >= Q2_CORR:
                rule = ("Q2", "clear")
            elif k == "rapid" and r.n_on < LOW_N and np.isfinite(r.share_x) and r.share_x >= 0.1:
                rule = ("R5", "watch")
            elif k == "dropout" and r.hours < 1:
                rule = ("R6", "watch")
            if rule:
                fired.append(rule[0])
                (watch if rule[1] == "watch" else notes).append(k)
                del find[k]
        # Y: pass 1 judged it against its phase, pass 2 has no healthy phase mate left -> cannot confirm or clear
        if "R9" in fired and np.isfinite(r.n_hmates) and r.n_hmates == 0 and r.status in ("suspect", "bad"):
            fired.append("Y")
            watch.append("no_yardstick")
        if len(find) == 1:
            k, v = next(iter(find.items()))
            if k in STAT and v < BORDER:
                fired.append("R7")
                watch.append(k)
                del find[k]
        if "last 0.2 s or less" in str(r.reason) and r.fn == "Presence":
            fired.append("R8")
            watch.append("short_on")
        if bool(r.occ_hi8) and "occspk" not in find and "stuck" not in find and \
                not (np.isfinite(r.c_occ_like) and r.c_occ_like >= .5):
            fired.append("N1")
            watch.append("occ_hi")
        dq = []
        on_s = r.occ * r.hours * 3600 if np.isfinite(r.occ) else np.nan
        if np.isfinite(r.rep_time_s) and r.rep_time_s >= D1_MIN_S and np.isfinite(on_s) and \
                r.rep_time_s >= D1_SHARE * on_s:
            dq.append("D1")
        if bool(r.n3_dq):
            dq.append("D1n3")
        hs = float(np.prod([1 - v for k, v in s.items() if k in find or v < .35])) if s else np.nan
        if np.isfinite(hs) and hs < .25:
            st = "bad"
        elif np.isfinite(hs) and hs < .70:
            st = "suspect"
        elif watch:
            st = "watch"
        elif r.n_on < 20:
            st = "not_enough_data"
        else:
            st = "ok"
        fams = {FAMILY[k] for k, v in find.items() if not (k in ("stuck", "dropout") and v == .35)}
        if st == "suspect" and len(fams) >= 2:
            st = "bad"
        rows.append(dict(st8=st, rules8=",".join(fired), watch8=",".join(watch), cleared8=",".join(notes),
                         left8=",".join(sorted(find)), score8=hs, dq8=",".join(dq)))
    return pd.concat([X.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def lim_table(R):
    cols = ["chop15", "chat_frac", "ioi_lt1", "max5", "stuck_x", "n3_exc", "rep_frac"]
    t = R.groupby(["fn", "span", "cmode", "wg"])[[f"lim_{c}" for c in cols] + ["stuck_lim8"]].first()
    t.to_csv(OUT / f"limits108_q{TAG}.csv")
    print(t.round(3).to_string())


def rates(R):
    out = []
    flag = lambda s: s.isin(["suspect", "bad"])  # noqa: E731
    for wg in ("m30", "h3", "h24"):
        x = R[R.wg == wg]
        out.append(dict(window=wg, n=len(x), package=100 * flag(x.status).mean(), v104=100 * flag(x.new_status).mean(),
                        v108=100 * flag(x.st8).mean(), bad104=100 * (x.new_status == "bad").mean(),
                        bad108=100 * (x.st8 == "bad").mean(), watch104=100 * (x.new_status == "watch").mean(),
                        watch108=100 * (x.st8 == "watch").mean(), dq108=100 * x.dq8.ne("").mean()))
    T = pd.DataFrame(out).round(2)
    print(T.to_string(index=False))
    T.to_csv(OUT / f"rates108_q{TAG}.csv", index=False)
    for wg in ("h3", "h24"):
        x = R[R.wg == wg]
        print(f"\n{wg} suspect+bad per 100 by type (package / v104 / v108):")
        t = x.groupby("type").agg(n=("st8", "size"), package=("status", lambda s: 100 * flag(s).mean()),
                                  v104=("new_status", lambda s: 100 * flag(s).mean()),
                                  v108=("st8", lambda s: 100 * flag(s).mean()))
        print(t.round(2).to_string())
        print(f"{wg} finding fires per 100 (v108 score >= .35):",
              {k: round(100 * (x[f's8_{k}'] >= .35).mean(), 2) for k in ("stuck", "dropout", "chatter", "rapid",
                                                                         "volume", "level", "choppy", "night_drop",
                                                                         "occspk", "prof")})
        print(f"{wg} old package finding fires per 100:",
              {k: round(100 * (pd.to_numeric(x[f's_{k}'], errors='coerce') >= .35).mean(), 2)
               for k in ("stuck", "dropout", "chatter", "rapid", "volume", "level", "choppy", "night_drop",
                         "night_day", "corr")})
        print(f"{wg} rules per 100:", {k: round(100 * x.rules8.str.split(",").apply(lambda l: k in l).mean(), 3)
                                       for k in ("Q1", "Q2", "R5", "R6", "R7", "R8", "R9", "Y", "N1")},
              "D1", round(100 * x.dq8.str.contains("D1").mean(), 2))
        print(pd.crosstab(x.new_status, x.st8).to_string())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        Q = float(sys.argv[1])
    if len(sys.argv) > 2:
        PQ = float(sys.argv[2])
    TAG = f"{1000 * Q:.0f}" + (f"p{1000 * PQ:.0f}" if PQ else "")
    build()
