"""Note 96: proposed context-aware health logic (multi-flag + second-stage resolver), applied to the current package
health (v4f, re-run by h96_health.py) on the w40 windows, with fire rates before / after and a sanity check against the
user's health_review_v1 answers (answers are NOT used to set anything; they are only compared at the end).

Stage 1 = every package rule finding (s_k >= .35) + one new information flag (occ_hi, below).
Stage 2 = context per finding, from the hi-res log + the classifier's own outputs only:
  long zone      predicted Presence or Other (incl. ETA / long-advance zones)
  occ follows    15-min occupancy correlates >= .8 with the traffic reference (predicted Advance / Count counts on the
                 same predicted phase, else the signal)
  congestion     during a long ON: phase peers' occupancy >= 1.5x their window mean (else the signal's congestion index
                 >= 1.5) AND the traffic reference still flowing >= .5x its window mean
  lanes          predicted lanes spanned
  low volume     < 100 actuations in the window;  short sample = window < 1 h
  borderline     rule score < .42 (within ~10 % of the suspect -> bad span) on a statistical check
Resolver rules (R1 .. R8, see RULES) clear a finding (ok, with a note) or turn it into WATCH (information, no status
change); what is left is scored as the package does (health_score = prod(1 - s); < .25 bad, < .70 suspect; two
independent families >= suspect -> bad).

    python h96_resolve.py            -> %DC_WORK%/health96/resolved.parquet, rates printed
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h96_study as S  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 300)
DCW = S.DCW
OUT = DCW / "health96"
LONG = ("Presence", "Other")
COUNT_T = ("Advance", "Count", "Yellow_Red", "Mid", "Bike")
STAT = ("dropout", "rapid", "level", "choppy", "night_drop", "night_day", "corr")
CHK = ("dropout", "stuck", "chatter", "rapid", "volume", "level", "choppy", "night_drop", "night_day", "corr")
FAMILY = {"chatter": "fast", "rapid": "fast", "dropout": "silent", "stuck": "stuck", "level": "level",
          "night_drop": "level", "choppy": "shape", "corr": "shape", "volume": "shape", "night_day": "shape"}  # = package
OCC_FOLLOW, CONG_X, FLOW_X, LOW_N, BORDER = 0.80, 1.5, 0.5, 100, 0.42
PEER_BUSY = 0.40      # N1: phase peers ON >= 40 % of a bin = the phase is queued
VOL2 = 200            # 5-min limit for a detector the classifier says spans >= 2 lanes (healthy 2-lane p99.5 119-139)
RULES = {
    "R1": "stuck on a long zone during congestion (phase peers' occupancy up, traffic still flowing), < 60 min -> ok",
    "R2": "doesn't follow traffic / erratic counts on a long zone whose occupancy follows traffic -> ok",
    "R3": "count drops on a long zone whose occupancy share did not drop (>= .5 of before) -> ok",
    "R4": "too many in 5 min on a multi-lane detector (< 200) that follows traffic -> ok",
    "R5": "too-fast actuations with < 100 actuations in the window while it still sees its usual share of the traffic "
          "(>= .1x its type's median share of the reference) -> watch",
    "R6": "goes silent in a sample under 1 h -> watch",
    "R7": "single borderline statistical finding (score < .42) -> watch",
    "R8": "too-short ONs (note) kept only on Presence zones (watch); dropped elsewhere",
    "R9": "two passes: detectors called bad in pass 1 are removed from their phase mates' yardstick, the rest are "
          "judged again (pass 2)",
    "N1": "NEW information flag: count-type zone ON far longer than its type at that traffic level (>= 30 min "
          "above the p99.5 of ok detectors of the same type and pulse / normal mode, signal not congested, phase peers < 40 % ON) and its occupancy does not follow traffic "
          "(corr < .5) -> watch",
}
# Tried and dropped (note 96): "N2" = own count < 10 % of its type's median share of the traffic reference -> fired on
# ~12 per 100 detectors (healthy shares span two orders of magnitude: lane geometry), so traffic is used as a SHAPE
# reference (R2) only, never as a level.


def _ts(x):
    try:
        return pd.Timestamp(x)
    except Exception:  # noqa: BLE001
        return pd.NaT


def load():
    H1 = pd.read_parquet(OUT / "health.parquet")
    H = pd.read_parquet(OUT / "health2.parquet")              # R9: pass 2 (bad detectors are nobody's yardstick)
    H = H.merge(H1[["DeviceId", "window", "detector", "status", "health_score"]].rename(
        columns={"status": "status1", "health_score": "score1"}), on=["DeviceId", "window", "detector"])
    H = H.rename(columns={"status": "status2"}).rename(columns={"status1": "status"})
    F = pd.read_parquet(OUT / "feat.parquet")
    X = H.merge(F.drop(columns=["n_on", "max5", "mean_on_s"]), on=["DeviceId", "window", "detector"], how="left")
    X["wg"] = X.window.str.split("_").str[0]
    return X


def occ_hi_flags(X):
    """N1: per (fn, bin size, traffic level) p99.5 of 15-min occupancy over ok detectors; a count-type detector with
    >= 30 min of bins above it while the signal congestion index < 1.5 -> flag."""
    Bn = pd.read_parquet(OUT / "bins.parquet")
    Bn["bs"] = np.where(Bn.window.str.startswith("m30"), 300, 900)
    Bn["rmax"] = Bn.groupby(["DeviceId", "window", "detector"]).ref.transform("max")
    Bn["tl"] = np.clip(np.floor(5 * Bn.ref / Bn.rmax.where(Bn.rmax > 0)), 0, 4)
    okk = X.loc[X.status == "ok", ["DeviceId", "window", "detector"]]
    ok = Bn.merge(okk, on=["DeviceId", "window", "detector"])
    md = X[["DeviceId", "window", "detector", "pulse_frac"]].assign(
        mode=lambda d: np.where(pd.to_numeric(d.pulse_frac, errors="coerce") >= 0.7, "pulse", "normal"))
    Bn = Bn.merge(md[["DeviceId", "window", "detector", "mode"]], on=["DeviceId", "window", "detector"], how="left")
    ok = ok.merge(md[["DeviceId", "window", "detector", "mode"]], on=["DeviceId", "window", "detector"], how="left")
    lim = ok.groupby(["fn", "mode", "bs", "tl"]).occ.quantile(.995).rename("lim").reset_index()
    Bn = Bn.merge(lim, on=["fn", "mode", "bs", "tl"], how="left")
    # phase peers' occupancy per bin (self excluded): a busy phase (peers >= 40 % ON) = queue, not a fault
    Bn = Bn.merge(X[["DeviceId", "window", "detector", "phase"]], on=["DeviceId", "window", "detector"], how="left")
    g = Bn.groupby(["DeviceId", "window", "phase", "b"]).occ.agg(["sum", "count"]).rename(
        columns={"sum": "ps", "count": "pc"}).reset_index()
    Bn = Bn.merge(g, on=["DeviceId", "window", "phase", "b"], how="left")
    Bn["peer_occ"] = (Bn.ps - Bn.occ.fillna(0)) / (Bn.pc - 1).where(Bn.pc > 1)
    Bn["hi"] = (Bn.occ > Bn.lim) & (Bn.cong < CONG_X) & ~(Bn.peer_occ >= PEER_BUSY) & Bn.fn.isin(COUNT_T)
    g = Bn.groupby(["DeviceId", "window", "detector"]).agg(hi_n=("hi", "sum"), bs=("bs", "first")).reset_index()
    g["occ_hi"] = g.hi_n * g.bs >= 1800
    return g[["DeviceId", "window", "detector", "occ_hi", "hi_n"]], lim


def episode_context(X):
    """R1 context for rows with a stuck finding: during the package's stuck episode (ep_t0..ep_t1), the phase peers'
    occupancy and the traffic reference vs their window means (15-min bins; 5-min for 30-min windows)."""
    m = X.s_stuck >= .35
    if not m.any():
        return X.assign(ep_phx=np.nan, ep_refx=np.nan, ep_cong=np.nan)
    Bn = pd.read_parquet(OUT / "bins.parquet")
    keys = X.loc[m, ["DeviceId", "window"]].drop_duplicates()
    Bn = Bn.merge(keys, on=["DeviceId", "window"])
    Bn = Bn.merge(X[["DeviceId", "window", "detector", "phase"]], on=["DeviceId", "window", "detector"], how="left")
    out = []
    for i, r in X[m].iterrows():
        b = Bn[(Bn.DeviceId == r.DeviceId) & (Bn.window == r.window)]
        s0, h = S_WIN[r.window]
        t0 = pd.Timestamp(s0)
        bs = 300 if h < 1 else 900
        a, z = _ts(r.ep_t0), _ts(r.ep_t1)
        if pd.isna(a) or pd.isna(z):
            out.append((i, np.nan, np.nan, np.nan))
            continue
        ia, iz = int((a - t0).total_seconds() // bs), int(np.ceil((z - t0).total_seconds() / bs))
        own = b[b.detector == r.detector]
        ind = own.b.between(ia, iz - 1)
        refx = own.ref[ind].mean() / max(own.ref.mean(), 1e-9)
        cong = own.cong[ind].mean()
        pp = b[(b.phase == r.phase) & (b.detector != r.detector)] if np.isfinite(r.phase) else b.iloc[:0]
        if len(pp):
            pm = pp.groupby("b").occ.mean()
            phx = pm[pm.index.to_series().between(ia, iz - 1)].mean() / max(pm.mean(), 1e-9)
        else:
            phx = np.nan
        out.append((i, phx, refx, cong))
    E = pd.DataFrame(out, columns=["i", "ep_phx", "ep_refx", "ep_cong"]).set_index("i")
    return X.join(E)


def level_context(X):
    """R3 context: occupancy share after / before the level change point (level_b, 5-min bin) vs the reference."""
    m = X.s_level >= .35
    X["lv_occx"] = np.nan
    if not m.any():
        return X
    Bn = pd.read_parquet(OUT / "bins.parquet")
    Bn = Bn.merge(X.loc[m, ["DeviceId", "window", "detector"]], on=["DeviceId", "window", "detector"])
    for i, r in X[m].iterrows():
        b = Bn[(Bn.DeviceId == r.DeviceId) & (Bn.window == r.window) & (Bn.detector == r.detector)]
        h = S_WIN[r.window][1]
        k = int(float(r.level_b)) // (1 if h < 1 else 3)
        bef, aft = b[b.b < k], b[b.b >= k]
        if len(bef) < 2 or len(aft) < 2:
            continue
        o = (aft.occ.mean() / max(bef.occ.mean(), 1e-9)) / max(aft.ref.mean() / max(bef.ref.mean(), 1e-9), 1e-9)
        X.loc[i, "lv_occx"] = o
    return X


S_WIN = {w: v for w, v in __import__("h96_occ").WIN.items()}


def resolve(X):
    rows = []
    for r in X.itertuples(index=False):
        s = {k: getattr(r, f"s_{k}") for k in CHK if np.isfinite(getattr(r, f"s_{k}", np.nan))}
        find = {k: v for k, v in s.items() if v >= .35}
        notes, watch, fired = [], [], []
        if r.status2 != r.status:
            fired.append("R9")
        longz = r.fn in LONG
        follows = np.isfinite(r.c_occ_ref) and r.c_occ_ref >= OCC_FOLLOW
        for k in list(find):
            rule = None
            if k == "stuck" and longz and np.isfinite(r.ep_dur) and r.ep_dur < 3600 and np.isfinite(r.ep_refx) \
                    and r.ep_refx >= FLOW_X and ((np.isfinite(r.ep_phx) and r.ep_phx >= CONG_X)
                                                 or (not np.isfinite(r.ep_phx) and np.isfinite(r.ep_cong)
                                                     and r.ep_cong >= CONG_X)):
                rule = ("R1", "clear")
            elif k in ("corr", "choppy") and longz and follows:
                rule = ("R2", "clear")
            elif k == "level" and longz and np.isfinite(r.lv_occx) and r.lv_occx >= 0.5:
                rule = ("R3", "clear")
            elif k == "volume" and np.isfinite(r.lanes) and r.lanes >= 2 and r.max5 < VOL2 and \
                    np.isfinite(r.c_cnt_ref) and r.c_cnt_ref >= 0.5:
                rule = ("R4", "clear")
            elif k == "rapid" and r.n_on < LOW_N and np.isfinite(r.share_x) and r.share_x >= 0.1:
                rule = ("R5", "watch")
            elif k == "dropout" and r.hours < 1:
                rule = ("R6", "watch")
            if rule:
                fired.append(rule[0])
                (watch if rule[1] == "watch" else notes).append(k)
                del find[k]
        if len(find) == 1:
            k, v = next(iter(find.items()))
            if k in STAT and v < BORDER:
                fired.append("R7")
                watch.append(k)
                del find[k]
        short = "last 0.2 s or less" in str(r.reason)
        if short and r.fn == "Presence":
            fired.append("R8")
            watch.append("short_on")
        if bool(r.occ_hi) and not (s.get("stuck", 0) >= .35) and not (np.isfinite(r.c_occ_ref) and r.c_occ_ref >= .5):
            fired.append("N1")
            watch.append("occ_hi")
        hs = float(np.prod([1 - v for k, v in s.items() if k in find or v < .35])) if s else np.nan
        removed = len(notes) + len([w for w in watch if w in CHK]) > 0
        if not removed:
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


def rates(R):
    out = []
    for wg in ("m30", "h3", "h24"):
        x = R[R.wg == wg]
        n = len(x)
        old = x.status.isin(["suspect", "bad"])
        new = x.new_status.isin(["suspect", "bad"])
        out.append(dict(window=wg, detectors=n, before=100 * old.mean(), after=100 * new.mean(),
                        bad_before=100 * (x.status == "bad").mean(), bad_after=100 * (x.new_status == "bad").mean(),
                        watch_after=100 * (x.new_status == "watch").mean(),
                        cleared=100 * (old & ~new).mean(), added=100 * (~old & new).mean()))
    T = pd.DataFrame(out).round(2)
    print(T.to_string(index=False))
    for wg in ("h3", "h24"):
        x = R[R.wg == wg]
        print(wg, "rule fires per 100:", {k: round(100 * x.rules.str.contains(k).mean(), 2) for k in RULES})
    return T


def main():
    X = load()
    oh, lim = occ_hi_flags(X)
    X = X.merge(oh, on=["DeviceId", "window", "detector"], how="left")
    X["occ_hi"] = X.occ_hi.fillna(False)
    X["ep_dur"] = pd.to_numeric(X.ep_dur, errors="coerce")
    X["level_b"] = pd.to_numeric(X.level_b, errors="coerce")
    # N2 / R5: own count vs the traffic reference, relative to the type's median share among ok detectors
    med = X[X.status == "ok"].groupby(["fn", "wg"]).share_ref.median().rename("share_med").reset_index()
    X = X.merge(med, on=["fn", "wg"], how="left")
    X["share_x"] = X.share_ref / X.share_med
    X["exp_n"] = X.share_med * X.ref_n
    X = episode_context(X)
    X = level_context(X)
    R = resolve(X)
    R.to_parquet(OUT / "resolved.parquet")
    lim.to_csv(OUT / "occ_hi_limits.csv", index=False)
    T = rates(R)
    T.to_csv(OUT / "rates.csv", index=False)
    # sanity check vs the user's answers (never used above)
    A = S.answers()
    A = A.merge(R[["DeviceId", "window", "detector", "fn", "lanes", "status", "new_status", "rules", "left", "watch",
                   "c_occ_ref"]], on=["DeviceId", "window", "detector"], how="left")
    A.to_parquet(OUT / "answers_resolved.parquet")
    a = A[A.answer != ""]
    print(a[["row", "check", "answer", "fn", "lanes", "status", "new_status", "rules", "left", "watch"]].to_string())
    a = a.assign(old_f=a.status.isin(["suspect", "bad"]), new_f=a.new_status.isin(["suspect", "bad"]))
    print(a.groupby("answer")[["old_f", "new_f"]].agg(["sum", "count"]))


if __name__ == "__main__":
    main()


def boot(nb=1000, seed=96):
    """Cross-check against the note-79 evaluation set (positives = stuck / chatter / degraded tiers A-C, mostly
    circular; presumed healthy = tier N): flag rate before / after, signal bootstrap 95 % CI of the change."""
    E = pd.read_parquet(DCW / "health79" / "evalset.parquet")
    E["DeviceId"] = E.DeviceId.str.lower()
    E = E[E.tier != "U"][["DeviceId", "detector", "cls"]].drop_duplicates(["DeviceId", "detector"])
    R = pd.read_parquet(OUT / "resolved.parquet").merge(E, on=["DeviceId", "detector"], how="left")
    R["old"] = R.status.isin(["suspect", "bad"])
    R["new"] = R.new_status.isin(["suspect", "bad"])
    rng = np.random.default_rng(seed)
    for wg in ("h3", "h24"):
        for nm, m in (("positives", R.cls.isin(["stuck", "chatter", "degraded"])), ("presumed healthy",
                                                                                      R.cls.eq("healthy"))):
            y = R[(R.wg == wg) & m]
            g = y.groupby("DeviceId").agg(n=("old", "size"), o=("old", "sum"), w=("new", "sum"))
            d = []
            for _ in range(nb):
                s = g.iloc[rng.integers(0, len(g), len(g))].sum()
                d.append(100 * (s.w - s.o) / s.n)
            print(f"{wg} {nm:16s} n={len(y):5d} before {100 * y.old.mean():.2f} after {100 * y.new.mean():.2f} "
                  f"change {100 * (y.new.mean() - y.old.mean()):+.2f} [{np.quantile(d, .025):+.2f},{np.quantile(d, .975):+.2f}]")


def control(seed=96, reps=20):
    """Per rule: share of evalset positives among the flags it clears vs among the flags of the same checks it leaves;
    shuffled control = c_occ_ref shuffled within (fn, window group) and R2 re-applied."""
    E = pd.read_parquet(DCW / "health79" / "evalset.parquet")
    E["DeviceId"] = E.DeviceId.str.lower()
    E = E[E.tier != "U"][["DeviceId", "detector", "cls"]].drop_duplicates(["DeviceId", "detector"])
    R = pd.read_parquet(OUT / "resolved.parquet").merge(E, on=["DeviceId", "detector"], how="left")
    R = R[R.wg.isin(["h3", "h24"]) & R.cls.isin(["stuck", "chatter", "degraded", "healthy"])]
    R["pos"] = R.cls.ne("healthy")
    for rule, ks in (("R1", ["stuck"]), ("R2", ["corr", "choppy"]), ("R3", ["level"]), ("R7", list(STAT))):
        hit = R.rules.str.contains(rule)
        fl = R[[f"s_{k}" for k in ks]].ge(.35).any(axis=1)
        print(f"{rule}: flagged by {ks}: {fl.sum()} ({100 * R[fl].pos.mean():.1f} % positives); rule fired {hit.sum()} "
              f"({100 * R[hit].pos.mean():.1f} % positives); left {(fl & ~hit).sum()} ({100 * R[fl & ~hit].pos.mean():.1f} %)")
    rng = np.random.default_rng(seed)
    fl = R[["s_corr", "s_choppy"]].ge(.35).any(axis=1) & R.fn.isin(LONG)
    real = R[fl & (R.c_occ_ref >= OCC_FOLLOW)]
    sh = []
    for _ in range(reps):
        c = R.groupby(["fn", "wg"]).c_occ_ref.transform(lambda s: pd.Series(rng.permutation(s.to_numpy()), s.index))
        sh.append(R[fl & (c >= OCC_FOLLOW)].pos.mean())
    print(f"R2 real: clears {len(real)} long-zone corr/choppy flags, {100 * real.pos.mean():.1f} % positives; shuffled "
          f"occupancy correlation: {100 * np.mean(sh):.1f} % positives (mean of {reps}); all long-zone corr/choppy flags "
          f"{100 * R[fl].pos.mean():.1f} %")
