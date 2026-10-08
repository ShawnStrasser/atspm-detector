"""Note 118a: health resolver v118 = v110 (research/code/health/h110_resolve.py, every F-fix on) + four fixes from the
user's review of health_review_v3 (Oct 7, points 3-6).  Same data (w40 windows, 763 training signals, locked_v2
absent - asserted upstream), hi-res log + classifier outputs only, rule-based, no fault events.  Each fix switchable.

  G1 stuck on: every ON over the type limit is listed (time + length + counted / queue / shared); an Advance, Count or
     Yellow_Red zone held ON is never excused as 'shared' (whole-phase holds on those zones are faults, 10037 d22);
     config note C2 'not set to pulse' when a pulse-mode Count zone holds ONs (>= 1 ON >= 60 s or >= 3 ONs >= 5 s):
     a pulse zone cannot hold ON, so it is not really pulse (04016 d20).
  G2 extension time: too-fast (ON->ON < 0.5 s / < 1 s / bursts) and too-many-in-5-min are counted on ONs that START
     a continuous ON only - an ON logged again with no OFF is never a vehicle and never 'too fast' (limits refit);
     config note C1 on a stop-bar Count zone when ONs are logged again without an OFF (the D1 rule, or >= 20 such
     ONs and >= 5 % of its ONs): misconfigured (extension time) or not a count zone (2B068 d19, 2B058 d46).
  G3 erratic counts (h118_erratic): share of its counts that sit OUTSIDE the expected range in 15-min periods
     (expected = yardstick x its share in the 30 min either side; range +- 3 sd incl. 15 % lane-share scatter, and
     >= 5 counts); finding needs >= 2 such periods; limit p99.8 of healthy per type x volume band x sample length
     (bad 2x).  What is judged is what the chart shows (band + marked periods).  No yardstick -> not scored.
  G4 count drops: unchanged decision (v110 F2); adds the plain-words evidence (drop time, own counts per 15 min
     before / after, expected after) and the expected series for the chart; a drop whose expected-after level is
     beyond any 15-min count the detector ever made x 3 is not scored (yardstick not credible).

    python h118_resolve.py   -> %DC_WORK%/s118/resolved118.parquet, rates118.csv, flags_review118.csv,
                                stuck_eps118.csv, drops118.csv, user_rows118.csv
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
import h110_resolve as R10  # noqa: E402
import h108_resolve as R8  # noqa: E402
import h118_erratic as ER  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 80)
DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
OUT = DCW / "s118"
KEY = ["DeviceId", "window", "detector"]
NO_SHARED = ("Advance", "Count", "Yellow_Red")
C1_N, C1_SHARE = 20, 0.05
C2_N60, C2_N5 = 1, 3
G3_MIN_OFF, G3_MIN_SC = 2, 6
G4_EXP_X = 3.0
GALL = frozenset({1, 2, 3, 4})
_ORIG_SCORES = R10.scores110
C7 = R10.C7
WSTART = {w: pd.Timestamp(v[0]) for w, v in R10.S_WIN.items()}


# ------------------------------------------------------------------ inputs
def prepare():
    X0, E = R10.prepare()
    A = pd.read_parquet(OUT / "act118.parquet")
    X0 = X0.merge(A[KEY + ["n_rep", "n_start", "ioi_lt05_c", "ioi_lt1_c", "burst_frac_c", "max5_c", "n_cont",
                           "n_ge2", "n_ge5", "n_ge60", "long_max_s"]], on=KEY, how="left")
    Er = pd.read_parquet(OUT / "erratic118.parquet")
    X0 = X0.merge(Er[KEY + ["n_off_15", "n_up_15", "n_sc_15", "exc_15", "n_off_sig", "exc_sig", "n_sc_sig"]],
                  on=KEY, how="left")
    fn = X0[KEY + ["fn"]]
    E = E.merge(fn, on=KEY, how="left")
    E["co5_orig"] = E.co5
    return X0, E


# ------------------------------------------------------------------ G3 limits (type x volume band x length)
G3_KEYS = (["fn", "span", "band", "wg"], ["fn", "span", "wg"], ["span", "wg"], ["wg"])


def g3_tables(X):
    hl = X.hl_shape & X.n_on.ge(30) & X.ref110.ne("none") & X.exc_15.notna() & X.n_sc_15.ge(G3_MIN_SC)
    T = []
    for keys in G3_KEYS:
        g = X[hl].groupby(keys).exc_15.agg(["count", lambda s: s.quantile(R8.Q)])
        g.columns = ["cnt", "lim"]
        T.append((keys, g[g.cnt >= R8.MIN_CELL].lim))
    return T


def g3_lookup(T, D):
    lim = pd.Series(np.nan, index=D.index)
    for keys, tab in T:
        m = D[keys].merge(tab.rename("lim"), left_on=keys, right_index=True, how="left").lim.to_numpy()
        fill = lim.isna().to_numpy() & np.isfinite(m)
        lim[fill] = m[fill]
    return lim


def g3_limit(X, T):
    D = X[["fn", "span", "band", "wg"]]
    lim = g3_lookup(T, D)
    src = pd.Series("", index=X.index)
    low = X.alt_fns.fillna("").ne("")
    for fa in C7:
        m = low & X.alt_fns.fillna("").str.split(",").apply(lambda l, f=fa: f in l) & X.fn.ne(fa)
        if not m.any():
            continue
        la = g3_lookup(T, D[m].assign(fn=fa))
        lo = la > lim[m]
        idx = lo[lo].index
        lim[idx] = la[idx]
        src[idx] = fa
    return lim, src


# ------------------------------------------------------------------ scoring wrapper
def make_scores(G, T3):
    def scores118(X, E, fixes):
        S = _ORIG_SCORES(X, E, fixes)
        if 3 in G:
            L, src = g3_limit(X, T3)
            X["lim_exc"] = L.to_numpy()
            X["lsrc_exc"] = src.to_numpy()
            ok = X.exc_15.notna() & X.n_sc_15.ge(G3_MIN_SC) & X.ref110.ne("none")
            s = R8.score(X.exc_15, L, 2 * L)
            s = np.where(X.n_off_15.fillna(0) >= G3_MIN_OFF, s, 0.0)
            S["choppy"] = np.where(ok, s, np.nan)
            X["noyard_chop"] = X.ref110.eq("none") & (X.exc_sig >= L) & (X.n_off_sig.fillna(0) >= G3_MIN_OFF) & \
                X.n_sc_sig.ge(G3_MIN_SC)
        if 4 in G:
            bad = X.g4_unscored.fillna(False).to_numpy(bool)
            S.loc[bad, "level"] = np.nan
        return S
    return scores118


def run(X0, E, G):
    X = X0.copy()
    if 2 in G:
        for c in ("ioi_lt05", "ioi_lt1", "burst_frac", "max5"):
            X[c] = X[c + "_c"].astype(float)
    E2 = E.copy()
    if 1 in G:
        E2.loc[E2.fn.isin(NO_SHARED), "co5"] = 0
    if 4 in G:
        X["g4_unscored"] = X.g4_unscored0
    T3 = g3_tables(X) if 3 in G else None
    R10.scores110 = make_scores(G, T3)
    try:
        R = R10.run(X, E2, set(R10.ALL))
    finally:
        R10.scores110 = _ORIG_SCORES
    R["cfg"] = ""
    if 2 in G:
        c1 = R.fn.eq("Count") & (R.dq8.str.contains("D1") | ((R.n_rep >= C1_N) & (R.n_rep >= C1_SHARE * R.n_on)))
        R.loc[c1, "cfg"] = "C1"
    if 1 in G:
        c2 = R.fn.eq("Count") & R.cmode.eq("pulse") & ((R.n_ge60 >= C2_N60) | (R.n_ge5 >= C2_N5)) &             ~R.cfg.str.contains("C1")                  # held ON by extension chains: C1 already says why
        R.loc[c2, "cfg"] = (R.loc[c2, "cfg"] + ",C2").str.strip(",")
    return R


# ------------------------------------------------------------------ G4: count-drop evidence
def drop_evidence(X):
    """for every detector-window with a v110 count-drop candidate (finite ratio): drop time, own counts per 15 min
    before / after, expected after (its earlier share x rest of signal x the type's hourly pattern), the credibility
    guard, and the per-period expected series (for the chart)."""
    W = R10.tod_weights()
    b = pd.read_parquet(DCW / "health110" / "b15.parquet")
    C = X[X.lv_ratio110.notna() & X.lv_b110.ge(0)][KEY + ["type", "band", "lv_b110", "lv_ratio110", "lv_llr110"]]
    b = b.merge(C, on=KEY, how="inner")
    b["day"] = np.where(b.window.isin(["h24_a", "h3_a"]), "h24_a", "h24_b")
    W2 = W.rename(columns={"window": "day"})
    b = b.merge(W2[W2.n_w >= 30][["type", "band", "day", "hour", "w"]], on=["type", "band", "day", "hour"], how="left")
    Wb = W.groupby(["band", "window", "hour"]).apply(lambda d: np.average(d.w, weights=d.n_w)).rename("wb").reset_index()
    b = b.merge(Wb.rename(columns={"window": "day"}), on=["band", "day", "hour"], how="left")
    b["w"] = b.w.fillna(b.wb).fillna(1.0)
    rows, ser = [], []
    for k, g in b.groupby(KEY, sort=False):
        g = g.sort_values("b")
        cut = int(g.lv_b110.iloc[0])
        m = g.ok.to_numpy(bool) & np.isfinite(g.S.to_numpy(float))
        x, gg, bb = g.x.to_numpy(float), (g.S * g.w).to_numpy(float), g.b.to_numpy()
        bef, aft = m & (bb < cut), m & (bb >= cut)
        x1, s1, x2, s2 = x[bef].sum(), gg[bef].sum(), x[aft].sum(), gg[aft].sum()
        sh = x1 / max(s1, 1e-9)
        exp_b = sh * gg
        exp_after = sh * s2 / max(aft.sum(), 1)
        own_max = x[m].max() if m.any() else np.nan
        t = WSTART[k[1]] + pd.Timedelta(minutes=15 * cut)
        rows.append(dict(zip(KEY, k), drop_at=t, own_before=x1 / max(bef.sum(), 1), own_after=x2 / max(aft.sum(), 1),
                         exp_after=exp_after, own_max15=own_max, pct_of_expected=100 * g.lv_ratio110.iloc[0],
                         g4_unscored0=bool(exp_after > G4_EXP_X * max(own_max, 1.0))))
        ser.append(pd.DataFrame({"DeviceId": k[0], "window": k[1], "detector": k[2], "b": bb, "x": x,
                                 "expected": np.where(m, exp_b, np.nan), "after": bb >= cut}))
    return pd.DataFrame(rows), pd.concat(ser, ignore_index=True)


# ------------------------------------------------------------------ G1: every stuck ON, labelled
def stuck_list(R, E):
    E = E.merge(R[KEY + ["stuck_lim8"]], on=KEY, how="inner")
    E = E[E.dur_s >= E.stuck_lim8].copy()
    q_ok = (E.n_hpeer_e.fillna(0) > 0) & (E.phx_h_e >= 1.5) & (E.corr_h_e >= R8.Q1_CORR) & (E.refx_e >= 0.5) & \
        (E.light_e.fillna(1) == 0) & (E.trafx_e >= R10.TRAF_X)
    q1 = q_ok & (E.dur_s < 3600)
    q1b = q_ok & (E.dur_s >= 3600) & (E.cover_e >= R10.F4_COVER)
    sh = (E.co5 >= 3) & ~q1 & ~q1b
    E["label"] = np.where(q1 | q1b, "queue", np.where(sh, "shared", "counted"))
    E["also_on"] = E.co5_orig.fillna(0).astype(int)
    E["txt"] = [f"{t.strftime('%a %H:%M')} {d / 60:.0f} min" + ("" if l == "counted" else f" [{l}]") +
                (f" ({a} other detectors ON too)" if l == "counted" and a >= 3 else "") +
                (" (still ON at the end)" if bool(o) else "")
                for t, d, l, a, o in zip(E.t0, E.dur_s, E.label, E.also_on, E.open_end)]
    L = E.sort_values("t0").groupby(KEY).agg(stuck_eps=("txt", "; ".join), n_eps=("txt", "size")).reset_index()
    return E, L


# ------------------------------------------------------------------ G3: erratic evidence text
def erratic_text(R):
    b = pd.read_parquet(DCW / "health110" / "b15.parquet")
    F = R[R.s8_choppy >= .35][KEY + ["lim_exc", "exc_15", "type", "band"]]
    b = b.merge(F, on=KEY, how="inner")
    out = []
    for k, g in b.groupby(KEY, sort=False):
        g = g.sort_values("b")
        e, hw, off, sc = ER.off_bins(g.x.to_numpy(float), g.R.to_numpy(float), g.ok.to_numpy(bool))
        x = g.x.to_numpy(float)
        ii = np.where(off)[0]
        ii = ii[np.argsort(-np.abs(x[ii] - e[ii]))][:3]
        per = "; ".join(f"{(WSTART[k[1]] + pd.Timedelta(minutes=15 * int(g.b.iloc[i]))).strftime('%a %H:%M')} "
                        f"{x[i]:.0f} vs {e[i]:.0f} expected" for i in sorted(ii))
        out.append(dict(zip(KEY, k), err_txt=f"{int(off.sum())} of {int(sc.sum())} 15-min periods outside the expected "
                        f"range; {100 * g.exc_15.iloc[0]:.0f} % of its counts outside it (limit {100 * g.lim_exc.iloc[0]:.0f} %"
                        f" for {g.type.iloc[0]}, {g.band.iloc[0]} volume); worst: {per}"))
    return pd.DataFrame(out)


# ------------------------------------------------------------------ rates
FL = ("suspect", "bad")
CHECKS = ("stuck", "dropout", "level", "choppy", "chatter", "rapid", "volume", "occspk", "prof")


def rate_rows(R, tag):
    out = []
    for wg in ("m30", "h3", "h24"):
        x = R[R.wg == wg]
        d = dict(run=tag, window=wg, n=len(x), flagged=100 * x.st8.isin(FL).mean(), bad=100 * x.st8.eq("bad").mean(),
                 watch=100 * x.st8.eq("watch").mean())
        for k in CHECKS:
            d[f"f_{k}"] = 100 * (x[f"s8_{k}"] >= .35).mean()
        d["note_C1"] = 100 * x.cfg.str.contains("C1").mean()
        d["note_C2"] = 100 * x.cfg.str.contains("C2").mean()
        d["note_D1"] = 100 * x.dq8.str.contains("D1").mean()
        out.append(d)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X0, E = prepare()
    D, Dser = drop_evidence(X0)
    X0 = X0.merge(D[KEY + ["g4_unscored0"]], on=KEY, how="left")
    X0["g4_unscored0"] = X0.g4_unscored0.fillna(False).astype(bool)
    runs = {"v110": frozenset(), "G1": {1}, "G2": {2}, "G3": {3}, "G4": {4}, "v118": GALL}
    rates, keep = [], {}
    for tag, G in runs.items():
        R = run(X0, E, set(G))
        rates += rate_rows(R, tag)
        keep[tag] = R
        if tag == "v110":
            V = pd.read_parquet(DCW / "health110" / "resolved110.parquet", columns=KEY + ["st8"])
            m = R[KEY + ["st8"]].merge(V, on=KEY, suffixes=("", "_saved"))
            print("v110 reproduced (all G off):", f"{(m.st8 == m.st8_saved).mean():.5f}", flush=True)
        print(tag, "done", flush=True)
    T = pd.DataFrame(rates).round(3)
    T.to_csv(OUT / "rates118.csv", index=False)
    print(T.to_string(index=False))
    B, A = keep["v110"], keep["v118"]
    for tag in runs:
        if tag == "v110":
            continue
        m = B[KEY + ["wg", "st8"]].merge(keep[tag][KEY + ["st8"]], on=KEY, suffixes=("_b", ""))
        for wg, g in m.groupby("wg"):
            up = (g.st8.isin(FL) & ~g.st8_b.isin(FL)).mean() * 100
            dn = (~g.st8.isin(FL) & g.st8_b.isin(FL)).mean() * 100
            print(f"{tag:5s} {wg:4s} newly flagged {up:.3f} / unflagged {dn:.3f} per 100")
    Es, L = stuck_list(A, E.assign(co5=np.where(E.fn.isin(NO_SHARED), 0, E.co5)))
    Es[KEY + ["t0", "t1", "dur_s", "label", "also_on", "open_end", "txt"]].to_csv(OUT / "stuck_eps118.csv", index=False)
    Et = erratic_text(A)
    A = A.merge(L, on=KEY, how="left").merge(Et, on=KEY, how="left").merge(
        D.drop(columns=["g4_unscored0"]), on=KEY, how="left")
    A = A.merge(B[KEY + ["st8", "left8", "watch8", "dq8", "s8_choppy", "chop15"]].rename(
        columns={"st8": "st_v110", "left8": "left_v110", "watch8": "watch_v110", "dq8": "dq_v110",
                 "s8_choppy": "s_choppy_v110", "chop15": "chop_v110"}), on=KEY, how="left")
    A.to_parquet(OUT / "resolved118.parquet")
    D.to_csv(OUT / "drops118.csv", index=False)
    Dser.to_parquet(OUT / "drop_series118.parquet")
    print("G4 guard (expected after > 3x own max 15-min count):", int(D.g4_unscored0.sum()), "of", len(D),
          "candidates;", "among flagged v110:",
          int(D.merge(B[KEY + ["s8_level"]], on=KEY).query("s8_level >= .35").g4_unscored0.sum()))


if __name__ == "__main__":
    main()


def limits_table():
    """per-type limits that changed (healthy baseline, p99.8): erratic share outside range (G3, type x band x
    length) and the start-only too-fast / 5-min limits (G2) vs v110 -> %DC_WORK%/s118/limits118.csv"""
    X0, E = prepare()
    X0["g4_unscored0"] = False
    T3 = g3_tables(X0)
    rows = []
    for keys, tab in T3[:1]:
        for k, v in tab.items():
            rows.append(dict(check="erratic_share_outside", **dict(zip(keys, k)), limit=round(float(v), 4)))
    Xc = X0.copy()
    for c in ("ioi_lt05", "ioi_lt1", "burst_frac", "max5"):
        Xc[c] = Xc[c + "_c"].astype(float)
    for tag, X in (("v110", X0), ("v118", Xc)):
        T = R10.cell_tables(X, ("ioi_lt05", "ioi_lt1", "burst_frac", "max5"))
        for col, lst in T.items():
            keys, tab = lst[1]                                # fn x span x length
            for k, v in tab.items():
                rows.append(dict(check=f"{col}_{tag}", **dict(zip(keys, k)), limit=round(float(v), 4)))
    pd.DataFrame(rows).to_csv(OUT / "limits118.csv", index=False)
