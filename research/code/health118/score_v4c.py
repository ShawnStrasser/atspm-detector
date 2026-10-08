"""Note 118c: health v4 fixed after the adversarial check (research scorer; nothing adopted into a package).

v4c = v4 (score_v4.py: v110 + 118a G1-G4 + 117 fast / volume + 116 night level) with:
  1 time of day: + 'Unusual daily pattern' (h118c_band: >= 3 consecutive hours outside the type's normal 24-h counts
    band, phase-mate and queue excuses; 24-h samples only) + a signal-wide note; the night check keeps its rule.
  2 stuck on: the queue excuse never applies to Advance, Count or Yellow_Red (unless the model is unsure and a
    queue-able type is plausible); 'queued mates' also when they sit above their type's p95 time ON (3 h = 24 h); a Count zone held ON beyond 'pulse-Count level, or its actuations x 4 s' for
    >= 30 min is a finding ('Count zone held ON'), not a watch - unless ONs logged again without OFF explain it (C1/D1).
  3 goes silent: the expected count leaves out phase mates that are themselves flagged (h118c_dropout); a silent
    stretch that lies at night (21:00-05:00) is reported as 'misses vehicles at night'.
  4 too-fast without re-triggers (h118c_fast); limit floors (too-fast excess >= 3 SD, bursts >= 2, erratic share
    >= 2 %); erratic-count type cells need >= 300 healthy detectors; model-unsure limits for the 116 / 117 / band
    checks.
    python score_v4c.py   -> %DC_WORK%/s118c/{resolved_v4c.parquet, review_v4c.parquet, rates_v4c.csv, cats_v4c.csv,
                                             stuck_eps_v4c.csv, cases_v4c.csv}
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "health"))
sys.path.insert(0, str(HERE))
import h118_resolve as R18  # noqa: E402
import h108_resolve as R8  # noqa: E402
import h110_resolve as R10  # noqa: E402
import score_v4 as V4  # noqa: E402

DCW = R18.DCW
OUT = DCW / "s118c"
KEY = R18.KEY
NO_QUEUE = ("Advance", "Count", "Yellow_Red")
QUEUE_OK = {"Presence", "Other", "Mid"}
G3_MIN_CELL, EXC_FLOOR = 300, 0.02
T_PASS_Q = 0.90                        # per-actuation ON time of healthy normal-mode Count zones: p90 (~4 s)
COUNT_ON_MIN = 0.20                    # and the 15-min bin is >= 20 % ON (a count zone ON a fifth of the time)
COUNT_ON_TL = 1                        # ... while the signal's traffic is light (< 40 % of its busiest 15 min)
NIGHT = (21, 5)
CATEGORIES = dict(V4.CATEGORIES)
CATEGORIES.update({"shape24": ("Unusual daily pattern", "finding"), "count_on": ("Count zone held ON", "finding")})
R8.FAMILY.update({"shape24": "shape", "count_on": "stuck"})
WSTART = {w: pd.Timestamp(v[0]) for w, v in R10.S_WIN.items()}


# ------------------------------------------------------------------ G3 limits: bigger cells, floor
def g3_tables_c(X):
    hl = X.hl_shape & X.n_on.ge(30) & X.ref110.ne("none") & X.exc_15.notna() & X.n_sc_15.ge(R18.G3_MIN_SC)
    T = []
    for keys in R18.G3_KEYS:
        g = X[hl].groupby(keys).exc_15.agg(["count", lambda s: s.quantile(R8.Q)])
        g.columns = ["cnt", "lim"]
        T.append((keys, g[g.cnt >= G3_MIN_CELL].lim))
    return T


_G3L = R18.g3_limit


def g3_limit_c(X, T):
    lim, src = _G3L(X, T)
    return np.fmax(lim, EXC_FLOOR), src


# ------------------------------------------------------------------ Count zone held ON (occ_hi with a Count limit)
def occ_hi_c(X):
    """N1 as v108 (>= 30 min of 15-min bins above the limit, signal not congested, phase mates < 40 % ON), but for
    Count zones the limit = max(p99.5 % ON of healthy PULSE Count zones at that traffic level, its actuations x the
    p90 ON time per actuation of healthy normal-mode Count zones / bin length, 20 % ON), counted only while the
    signal's traffic is light (a long ON in the peak can be a queue over the zone; a count zone held ON in light
    traffic cannot)."""
    Bn = pd.read_parquet(DCW / "health104" / "bins_ctx.parquet",
                         columns=KEY + ["fn", "b", "n", "occ", "ref", "cong", "bs", "ph", "status"])
    Bn = Bn.merge(X[KEY + ["cmode"]], on=KEY)
    Bn = Bn[Bn.cmode.ne("unknown")]
    Bn["rmax"] = Bn.groupby(KEY).ref.transform("max")
    Bn["tl"] = np.clip(np.floor(5 * Bn.ref / Bn.rmax.where(Bn.rmax > 0)), 0, 4)
    ok = Bn[Bn.status.eq("ok")]
    lim = ok.groupby(["fn", "cmode", "bs", "tl"]).occ.quantile(.995).rename("lim").reset_index()
    Bn = Bn.merge(lim, on=["fn", "cmode", "bs", "tl"], how="left")
    pl = lim[(lim.fn == "Count") & (lim.cmode == "pulse")][["bs", "tl", "lim"]].rename(columns={"lim": "lim_pulse"})
    c = ok[ok.fn.eq("Count") & ok.cmode.eq("normal") & ok.n.gt(0)]
    tp = (c.occ * c.bs / c.n).groupby(c.bs).quantile(T_PASS_Q).rename("t_pass").reset_index()
    Bn = Bn.merge(pl, on=["bs", "tl"], how="left").merge(tp, on="bs", how="left")
    cnt = Bn.fn.eq("Count")
    Bn["lim_c"] = np.where(cnt, np.fmax(np.fmax(Bn.lim_pulse, Bn.n * Bn.t_pass / Bn.bs), COUNT_ON_MIN), Bn.lim)
    g = Bn.groupby(KEY[:2] + ["ph", "b"]).occ.agg(["sum", "count"]).rename(columns={"sum": "ps", "count": "pc"}).reset_index()
    Bn = Bn.merge(g, on=KEY[:2] + ["ph", "b"], how="left")
    Bn["pocc"] = ((Bn.ps - Bn.occ.fillna(0)) / (Bn.pc - 1).where(Bn.pc > 1)).where(Bn.ph >= 0)
    quiet = (Bn.cong < 1.5) & ~(Bn.pocc >= 0.40)
    Bn["hi_old"] = (Bn.occ > Bn.lim) & quiet
    Bn["hi_c"] = (Bn.occ > Bn.lim_c) & quiet & np.where(cnt, Bn.tl <= COUNT_ON_TL, True)
    g = Bn.groupby(KEY).agg(hi_old=("hi_old", "sum"), hi_c=("hi_c", "sum"), bs=("bs", "first")).reset_index()
    g["hi_min_old"] = g.hi_old * g.bs / 60
    g["hi_min_c"] = g.hi_c * g.bs / 60
    print("Count passage time per actuation (p90, s):", tp.round(2).to_dict("records"))
    return g[KEY + ["hi_min_old", "hi_min_c"]], Bn[cnt][KEY + ["b", "bs", "occ", "n", "lim", "lim_c", "hi_c"]]


# ------------------------------------------------------------------ inputs
def alt_list(X):
    P = X[[f"p_{c}" for c in R10.C7]].to_numpy(float)
    top = np.nanmax(np.where(np.isfinite(P), P, -1), 1)
    return [set(c for c, v in zip(R10.C7, row) if np.isfinite(v) and v >= R10.F5_PLAUS) if 0 <= t < R10.F5_TOP else set()
            for t, row in zip(top, P)]


def prepare_c():
    X0, E = R18.prepare()
    D = pd.read_csv(DCW / "s118" / "drops118.csv", usecols=KEY + ["g4_unscored0"])
    X0 = X0.merge(D, on=KEY, how="left")
    X0["g4_unscored0"] = X0.g4_unscored0.fillna(False).astype(bool)
    X0 = X0.merge(V4.load_117(), on=KEY, how="left").merge(V4.load_116(), on=KEY, how="left")
    # 117 fast / volume (re-triggers out, floors, model-unsure limits)
    F = pd.read_parquet(OUT / "fast118c.parquet", columns=KEY + ["fo_c", "fem_c", "zf_c", "n_spk_c", "lim_zf_c",
                                                                 "lim_spk_c", "x_zf_c", "x_spk_c", "fast_x_c",
                                                                 "lim_vol_c", "vol_x_c"])
    X0 = X0.merge(F, on=KEY, how="left")
    # 116 night level with the model-unsure rule + the full-profile band check (24 h only)
    Bd = pd.read_parquet(OUT / "band_final.parquet",
                         columns=KEY + ["tod_level_c", "tod_z_c", "flag", "ev_from", "ev_h", "ev_dir", "ev_h0", "ev_h1",
                                        "ev_obs", "ev_exp", "run_h", "run_h_own", "mrun_h", "mate_z", "sig_wide",
                                        "sig_odd_n", "sig_n"])
    Bd = Bd.rename(columns={c: f"bd_{c}" for c in Bd.columns if c not in KEY})
    X0 = X0.merge(Bd, on=KEY, how="left")
    X0["tod_level"] = np.where(X0.bd_tod_level_c.notna(), X0.bd_tod_level_c, X0.tod_level)
    # goes silent with a clean yardstick
    Dr = pd.read_parquet(OUT / "dropout118c.parquet")
    X0 = X0.merge(Dr[KEY + ["drop_lam_chk", "drop_lam_c", "drop_b0_c", "drop_b1_c", "n_dropped", "dropped"]], on=KEY,
                  how="left")
    m = X0.n_dropped.fillna(0).gt(0) & X0.drop_lam_chk.gt(0)
    X0["drop_lam_pkg"] = X0.drop_lam
    X0.loc[m, "drop_lam"] = X0.loc[m, "drop_lam"] * X0.loc[m, "drop_lam_c"] / X0.loc[m, "drop_lam_chk"]
    X0.loc[m, "drop_b0"] = X0.loc[m, "drop_b0_c"]
    X0.loc[m, "drop_b1"] = X0.loc[m, "drop_b1_c"]
    sd = pd.to_numeric(X0.s_dropout, errors="coerce")
    new = np.where(X0.drop_in_stuck.astype(str).eq("True"), 0.0,
                   [np.nan if not np.isfinite(v) else (0.0 if v < 30 else .35 + .65 * min((v - 30) / 70, 1))
                    for v in X0.drop_lam.astype(float)])
    X0["s_dropout_pkg"] = sd
    X0["s_dropout"] = np.where(m, new, sd)
    # Count zone held ON
    H, _ = occ_hi_c(X0)
    X0 = X0.merge(H, on=KEY, how="left")
    alts = alt_list(X0)
    X0["alt_c"] = [",".join(sorted(a)) for a in alts]
    c1 = X0.fn.eq("Count") & ((X0.n_rep >= R18.C1_N) & (X0.n_rep >= R18.C1_SHARE * X0.n_on) |
                              ((X0.rep_time_s >= R8.D1_MIN_S) & (X0.rep_time_s >= R8.D1_SHARE * X0.occ * X0.hours * 3600)))
    sure_count = np.array([(not a) or a <= {"Count"} for a in alts])
    X0["count_on_ok"] = X0.fn.eq("Count").to_numpy() & sure_count & ~c1.fillna(False).to_numpy()
    X0["c1_pre"] = c1.fillna(False)
    # stuck: no queue excuse for Advance / Count / Yellow_Red (unless a queue-able type is plausible)
    noq = X0.fn.isin(NO_QUEUE).to_numpy() & np.array([not (a & QUEUE_OK) for a in alts])
    E = E.merge(X0[KEY].assign(noq=noq), on=KEY, how="left")
    E["n_hpeer_orig"] = E.n_hpeer_e
    E.loc[E.noq.fillna(False).astype(bool), "n_hpeer_e"] = 0
    # h3 / h24 consistency: 'mates >= 1.5x their usual time ON' compares with the mates' mean over the SAMPLE, which a
    # 3-h peak sample already is (2B530 d60, 14003 d4: queue in 24 h, 'stuck' in 3 h).  Mates above the p95 time ON
    # of healthy detectors of their type through >= 75 % of the ON (measured, sample-length free) count as queued too.
    E["phx_h_orig"] = E.phx_h_e
    qc = E.cover_e.fillna(0) >= R10.F4_COVER
    E.loc[qc, "phx_h_e"] = np.fmax(E.loc[qc, "phx_h_e"].fillna(0), 1.5)
    return X0, E


# ------------------------------------------------------------------ scores
def _night_frac(w, b0, b1):
    if not (np.isfinite(b0) and np.isfinite(b1)) or b1 <= b0:
        return 0.0
    t0 = WSTART[w]
    hrs = ((t0 + pd.to_timedelta(np.arange(int(b0), int(b1)) * 300, unit="s")).hour)
    return float(np.mean((hrs >= NIGHT[0]) | (hrs < NIGHT[1])))


def make_scores_c(G, T3):
    base = R18._make_scores_orig(G, T3)

    def scores(X, E, fixes):
        S = base(X, E, fixes)
        enough = X.n_on117.fillna(0).to_numpy() >= 50
        S["rapid"] = np.where(enough & np.isfinite(X.fast_x_c), R8.score(X.fast_x_c, 1.0, 2.0), np.nan)
        S["volume"] = np.where(np.isfinite(X.vol_x_c), R8.score(X.vol_x_c, 1.0, 2.0), np.nan)
        lv = X.tod_level.fillna("not scored")
        pv = np.where(lv.eq("not scored"), np.nan, lv.map(V4.PROF_LEVEL).fillna(0.0))
        dro = pd.to_numeric(S.get("dropout"), errors="coerce").fillna(0).to_numpy()
        S["prof"] = np.where((dro >= .35) & np.isfinite(pv), 0.0, pv)
        # a silent stretch that lies at night = misses vehicles at night
        nf = np.array([_night_frac(w, a, b) for w, a, b in zip(X.window, X.drop_b0.astype(float), X.drop_b1.astype(float))])
        X["silent_night"] = (dro >= .35) & (nf >= .8)
        nd = pd.to_numeric(S.night_drop, errors="coerce").to_numpy()
        S["night_drop"] = np.where(X.silent_night, np.fmax(np.nan_to_num(nd), dro), nd)
        S["dropout"] = np.where(X.silent_night, 0.0, S.dropout)
        # unusual daily pattern (24 h): suspect at a 3-h run, bad from ~6 h
        run = X.bd_ev_h.to_numpy(float)
        sh = np.where(X.bd_flag.fillna(False).astype(bool), R8.score(run, 3.0, 7.0), 0.0)
        down = X.bd_ev_dir.to_numpy(float) < 0
        # the same quiet stretch is reported once: the stronger of 'goes silent' and this; count drops / misses at
        # night / busy at night already describe it -> not repeated
        other = (S.level.fillna(0) >= .35) | (S.night_drop.fillna(0) >= .35)
        sh = np.where(down & other.to_numpy(), 0.0, sh)
        sh = np.where(~down & (np.nan_to_num(S.prof) >= .35), 0.0, sh)
        dr = S.dropout.fillna(0).to_numpy()
        both = down & (dr >= .35) & (sh >= .35)
        S["dropout"] = np.where(both & (sh >= dr), 0.0, np.where(both, np.fmax(dr, R8.BORDER), S.dropout))
        sh = np.where(both & (sh < dr), 0.0, sh)          # (two checks agree: the kept one is never 'borderline')
        S["shape24"] = np.where(X.bd_flag.notna().to_numpy(), sh, np.nan)
        # Count zone held ON
        hm = X.hi_min_c.fillna(0).to_numpy(float)
        S["count_on"] = np.where(X.count_on_ok, R8.score(hm, 30.0, 240.0), np.nan)
        X.loc[X.count_on_ok.to_numpy() & (hm >= 30), "occ_hi8"] = False
        return S
    return scores


def run_c(X0, E):
    X = X0.copy()
    X["prof3_x"] = np.nan
    R18._make_scores_orig = getattr(R18, "_make_scores_orig", R18.make_scores)
    saved = (R18.make_scores, R18.g3_tables, R18.g3_limit)
    R18.make_scores, R18.g3_tables, R18.g3_limit = make_scores_c, g3_tables_c, g3_limit_c
    try:
        R = R18.run(X, E, set(R18.GALL))
    finally:
        R18.make_scores, R18.g3_tables, R18.g3_limit = saved
    return R


def categories_of(r):
    out = [k for k in str(r.left8).split(",") if k and k in CATEGORIES]
    if "occ_hi" in str(r.watch8).split(","):
        out.append("occ_hi")
    out += [c for c in str(r.cfg).split(",") if c in CATEGORIES]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X0, E = prepare_c()
    R = run_c(X0, E)
    R["cats"] = [",".join(categories_of(r)) for r in R.itertuples()]
    # evidence: every stuck ON (labels with the v4c queue rule), erratic text, count-drop evidence
    Es, L = R18.stuck_list(R, E.assign(co5=np.where(E.fn.isin(R18.NO_SHARED), 0, E.co5)))
    Es[KEY + ["t0", "t1", "dur_s", "label", "also_on", "open_end", "txt"]].to_csv(OUT / "stuck_eps_v4c.csv", index=False)
    Et = R18.erratic_text(R)
    R = R.merge(L, on=KEY, how="left").merge(Et, on=KEY, how="left")
    Dd = pd.read_csv(DCW / "s118" / "drops118.csv", usecols=KEY + ["drop_at", "own_before", "own_after", "exp_after"])
    R = R.merge(Dd, on=KEY, how="left")
    V = pd.read_parquet(DCW / "s118b" / "resolved_v4.parquet", columns=KEY + ["st8", "left8", "watch8", "cfg", "cats"])
    R = R.merge(V.rename(columns={"st8": "st_v4", "left8": "left_v4", "watch8": "watch_v4", "cfg": "cfg_v4",
                                  "cats": "cats_v4"}), on=KEY, how="left")
    keep = [c for c in R.columns if not c.startswith("lsrc_") and (R[c].dtype != object or c in (
        "DeviceId", "window", "fn", "type", "span", "band", "cmode", "wg", "st8", "left8", "watch8", "rules8",
        "cleared8", "dq8", "cfg", "cats", "st_v4", "left_v4", "watch_v4", "cfg_v4", "cats_v4", "stuck_eps", "err_txt",
        "alt_fns", "alt_c", "tod_level", "tod_group", "reason", "drop_at", "night_ref", "ref110", "ref110_dets",
        "dropped", "drop_in_stuck", "bd_ev_from"))]
    R = R[keep]
    for c in R.columns:
        if R[c].dtype == object:
            R[c] = R[c].astype(str)
    R.to_parquet(OUT / "resolved_v4c.parquet")
    FL = ("suspect", "bad")
    rows = []
    for wg, x in R.groupby("wg"):
        d = dict(window=wg, n=len(x), flagged_v4=100 * x.st_v4.isin(FL).mean(), flagged_v4c=100 * x.st8.isin(FL).mean(),
                 bad_v4=100 * x.st_v4.eq("bad").mean(), bad_v4c=100 * x.st8.eq("bad").mean(),
                 watch_v4=100 * x.st_v4.eq("watch").mean(), watch_v4c=100 * x.st8.eq("watch").mean(),
                 newly=100 * (x.st8.isin(FL) & ~x.st_v4.isin(FL)).mean(),
                 cleared=100 * (~x.st8.isin(FL) & x.st_v4.isin(FL)).mean())
        for k in CATEGORIES:
            d[f"v4_{k}"] = 100 * x.cats_v4.fillna("").str.split(",").apply(lambda l, k=k: k in l).mean()
            d[f"v4c_{k}"] = 100 * x.cats.str.split(",").apply(lambda l, k=k: k in l).mean()
        rows.append(d)
    T = pd.DataFrame(rows).round(3)
    T.to_csv(OUT / "rates_v4c.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.T.to_string())
    rv = pd.read_csv(DCW / "health110" / "review" / "rows_v110.csv")[["signal", "dev"]].drop_duplicates()
    Rv = R.merge(rv.rename(columns={"dev": "DeviceId"}), on="DeviceId")
    Rv.to_parquet(OUT / "review_v4c.parquet")
    C = Rv[Rv.cats.ne("")].assign(cat=Rv.cats.str.split(",")).explode("cat")
    C = C.groupby(["cat", "wg"]).size().unstack(fill_value=0)
    C.to_csv(OUT / "cats_v4c.csv")
    print(C.to_string())


if __name__ == "__main__":
    main()
