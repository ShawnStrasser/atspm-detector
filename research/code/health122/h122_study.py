"""Note 122: red vs green health checks - baselines (1 in 500), flag rates, overlap with v4, placebo, phase errors.

Input  %DC_WORK%/s122/cyc122.parquet (h122_events) + s118c/resolved_v4c.parquet (type, v4 status) + labels v4q (truth
phase incl. switch / additional call phases, evaluation only).
Healthy baseline = v4 status ok, no category / watch, predicted phase correct by timing truth, eligible.
Limits p99.8 (high side) or p0.2 (low side) per fn x span x sample length (fallback fn x length; cells >= 100),
fitted on the OTHER half of the signals (2-fold by signal hash) so no detector sets its own limit.

Candidate checks (metric on the predicted phase's colours; eligibility: >= 10 complete cycles, >= 50 vehicle ONs):
  RR   Count / Yellow_Red: red ON rate share  r_r / (r_r + r_gy)              high side (counting in red)
  INV  Presence: ON at begin yellow minus ON at begin green (share of cycles)  high side (inverted queue pattern)
  E5   Count / YR / Presence: share of demand cycles whose first ON <= 5 s     low side  (no green-start response)
       into green (needs >= 10 demand cycles)
  ADV  Advance: red ON rate share                                              both sides (colour-dependent arrivals)
  DEP  all types: share of ONs in green+yellow minus green+yellow time share   low side  (no colour response)
  OCC  Other / Mid / Bike: log ratio % ON red / % ON green                     both sides
Same metric on the true phase (t_) and on a placebo phase (x_).

    python h122_study.py   -> %DC_WORK%/s122/{stats122.parquet, lim122.csv, rates122.csv, picks122.csv}
"""
from __future__ import annotations

import os
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
S = DCW / "s122"
REPO = Path(__file__).resolve().parents[3]
QH, QL, MIN_CELL = 0.998, 0.002, 100
CHECKS = {  # name: (metric, fns or None, side)
    "RR": ("rr", {"Count", "Yellow_Red"}, "hi"),
    "INV": ("inv", {"Presence"}, "hi"),
    "E5": ("e5", {"Count", "Yellow_Red", "Presence"}, "lo"),
    "ADV_hi": ("rr", {"Advance"}, "hi"),
    "ADV_lo": ("rr", {"Advance"}, "lo"),
    "DEP": ("dep", None, "lo"),
    "OCC_hi": ("occ", {"Other", "Mid", "Bike"}, "hi"),
    "OCC_lo": ("occ", {"Other", "Mid", "Bike"}, "lo"),
}


def metrics(X, p):
    g = lambda c: X[p + c]  # noqa: E731
    with np.errstate(divide="ignore", invalid="ignore"):
        rgy = g("nGY") / g("sGY") * 3600
        rr_ = g("nR") / g("sR") * 3600
        out = pd.DataFrame({
            "rr": rr_ / (rr_ + rgy),
            "inv": g("by") - g("bg"),
            "e5": np.where(g("nd") >= 10, g("e5"), np.nan),
            "dep": g("nGY") / (g("nGY") + g("nR")) - g("sGY") / (g("sGY") + g("sR")),
            "occ": np.log((g("oR") / g("sR") + 1e-3) / (g("oGY") / g("sGY") + 1e-3)),
            "elig": (g("ncyc") >= 10) & (X.n_on >= 50),
        }, index=X.index)
    return out


def load():
    C = pd.read_parquet(S / "cyc122.parquet")
    V = pd.read_parquet(DCW / "s118c" / "resolved_v4c.parquet",
                        columns=["DeviceId", "window", "detector", "st_v4", "cats_v4", "watch_v4", "cfg_v4", "type",
                                 "fn", "span", "wg", "pred_phase", "ph_conf", "f_conf"])
    L = pd.read_parquet(REPO / "research" / "labels" / "function_labels_v4q.parquet",
                        columns=["DeviceId", "DeviceName", "detector", "phase_target_type", "switch_phase",
                                 "additional_call_phases", "function"])
    L["DeviceId"] = L.DeviceId.str.lower()
    X = C.merge(V, on=["DeviceId", "window", "detector"], how="inner").merge(L, on=["DeviceId", "detector"], how="left")
    ok_alt = []
    for pp, sw, ad in zip(X.pred, X.switch_phase, X.additional_call_phases):
        s = set()
        if pd.notna(sw) and sw:
            s.add(int(sw))
        if isinstance(ad, str) and ad.strip():
            s |= {int(float(q)) for q in ad.replace(";", ",").split(",") if q.strip()}
        ok_alt.append(pd.notna(pp) and int(pp) in s)
    X["truth_known"] = X.true_raw.notna() & (X.phase_target_type == "phase")
    X["ph_ok"] = X.truth_known & ((X.pred == X.true_raw) | np.array(ok_alt))
    X["ph_wrong"] = X.truth_known & ~X.ph_ok
    X["half"] = [zlib.crc32(d.encode()) % 2 for d in X.DeviceId]
    X["v4flag"] = X.st_v4.isin(["suspect", "bad"])
    X["healthy"] = (X.st_v4 == "ok") & (X.cats_v4.fillna("") == "") & (X.watch_v4.fillna("").astype(str).isin(["", "[]", "None"]))
    for p in ("p_", "t_", "x_"):
        M = metrics(X, p)
        for c in M:
            X[p + c] = M[c]
    return X


def limits(X):
    rows = []
    base = X[X.healthy & X.ph_ok & X.p_elig]
    for h in (0, 1):
        B = base[base.half != h]
        for name, (m, fns, side) in CHECKS.items():
            q = QH if side == "hi" else QL
            for keys in (["fn", "span", "wg"], ["fn", "wg"]):
                for k, g in B.groupby(keys):
                    v = g["p_" + m].dropna()
                    if len(v) >= MIN_CELL:
                        rows.append(dict(half=h, check=name, level=len(keys), **dict(zip(keys, k if isinstance(k, tuple) else (k,))),
                                         n=len(v), p50=v.median(), lim=v.quantile(q)))
    return pd.DataFrame(rows)


def apply(X, Lm, src):
    out = {}
    for name, (m, fns, side) in CHECKS.items():
        lim = pd.Series(np.nan, index=X.index)
        for lev, keys in ((2, ["fn", "wg"]), (3, ["fn", "span", "wg"])):     # finer overrides coarser
            l = Lm[(Lm.check == name) & (Lm.level == lev)][["half"] + keys + ["lim"]].rename(columns={"half": "lh"})
            l["half"] = 1 - l.lh
            j = X[["half"] + keys].merge(l.drop(columns="lh"), on=["half"] + keys, how="left").lim.values
            lim = lim.where(np.isnan(j), j)
        v = X[src + m]
        inscope = X[src + "elig"] & v.notna() & lim.notna() & (X.fn.isin(fns) if fns else True)
        f = (v > lim) if side == "hi" else (v < lim)
        out[name] = (inscope & f).where(inscope, other=np.nan)
        X[f"lim_{name}"] = lim
    return pd.DataFrame(out, index=X.index)


def main():
    X = load()
    Lm = limits(X)
    Lm.to_csv(S / "lim122.csv", index=False)
    Fp, Ft, Fx = apply(X, Lm, "p_"), apply(X, Lm, "t_"), apply(X, Lm, "x_")
    for c in Fp:
        X["fp_" + c], X["ft_" + c], X["fx_" + c] = Fp[c], Ft[c], Fx[c]
    rows = []
    for name in CHECKS:
        for wg in ("m30", "h3", "h24", "all"):
            m = X if wg == "all" else X[X.wg == wg]
            sc = m["fp_" + name].notna()
            f = m["fp_" + name] == 1
            H = sc & m.healthy & m.ph_ok
            new = f & ~m.v4flag
            r = dict(check=name, wg=wg, n_scope=int(sc.sum()), n_flag=int(f.sum()),
                     per100=100 * f.sum() / max(sc.sum(), 1),
                     healthy_per100=100 * (f & H).sum() / max(H.sum(), 1),
                     v4flag_per100=100 * (f & m.v4flag).sum() / max((sc & m.v4flag).sum(), 1),
                     overlap_v4=(f & m.v4flag).sum() / max(f.sum(), 1),
                     n_new=int(new.sum()),
                     new_truth_known=int((new & m.truth_known).sum()),
                     new_phase_wrong=int((new & m.ph_wrong).sum()),
                     new_real=int((new & m.ph_ok & (m["ft_" + name] == 1)).sum()),
                     wrong_per100=100 * (f & m.ph_wrong).sum() / max((sc & m.ph_wrong).sum(), 1),
                     n_wrong_scope=int((sc & m.ph_wrong).sum()),
                     wrong_true_per100=100 * ((m["ft_" + name] == 1) & m.ph_wrong).sum()
                     / max((m["ft_" + name].notna() & m.ph_wrong).sum(), 1),
                     placebo_healthy_per100=100 * ((m["fx_" + name] == 1) & H).sum()
                     / max((m["fx_" + name].notna() & H).sum(), 1))
            rows.append(r)
    R = pd.DataFrame(rows)
    R.to_csv(S / "rates122.csv", index=False)
    print(R.round(3).to_string())
    # union, phase-sanity view
    fp = X[[f"fp_{c}" for c in CHECKS]].fillna(0).max(axis=1) == 1
    sc = X[[f"fp_{c}" for c in CHECKS]].notna().any(axis=1)
    X["any_p"], X["any_scope"] = fp, sc
    for wg in ("m30", "h3", "h24"):
        m = X.wg == wg
        a, b = m & sc & X.ph_wrong, m & sc & X.ph_ok
        print(f"{wg}: union flag per100  phase-wrong {100*fp[a].mean():.1f} (n {a.sum()})  phase-ok {100*fp[b].mean():.2f}"
              f" (n {b.sum()})  phase-wrong caught share = {fp[a].mean():.3f}")
    # base table per type (p50 of each metric, healthy & correct)
    B = X[X.healthy & X.ph_ok & X.p_elig & (X.wg == "h24")]
    P = B.groupby("type")[["p_rr", "p_inv", "p_e5", "p_dep", "p_occ", "x_rr", "x_inv", "x_e5", "x_dep", "x_occ"]].median()
    P["n"] = B.groupby("type").size()
    P.to_csv(S / "base122.csv")
    print(P.round(3).to_string())
    X.to_parquet(S / "stats122.parquet")


if __name__ == "__main__":
    main()
