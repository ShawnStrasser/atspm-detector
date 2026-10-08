"""Note 78: why has function accuracy plateaued, how much is mislabels / definitions / model, and what is the ceiling?

Champion = note-77 OOF: trees (note-76 arm c, 3 seeds) + siba (3 seeds) -> context stacker (f77, 3 seeds) -> D lanes,
lane-confidence gate .9, stack pick, short twin decode; truth_v3s; ATSPM stack-aware score (of77.setup_new).

    python err78.py rows      # champion rows + every model family's probabilities / predictions -> err78/rows.parquet
    python err78.py attrib    # note-70 decomposition on the champion (>= 30 min, 5 / 10 min; both sets) + model views
    python err78.py labels    # mislabel angles: print vs config, behaviour check, all-models-agree, reviews
    python err78.py lc        # learning curve (lc78.py output) + seed spread
Analysis only (no model trained here).  locked_v2 asserted absent.  CPU only.
"""
from __future__ import annotations

import os
os.environ["F76_ARM"] = "c"
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CODE = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "evaluation"))
sys.path.insert(0, str(CODE / "final77"))
import rpath  # noqa: F401,E402
from common import DC_WORK  # noqa: E402

OUT = DC_WORK / "err78"
KEY = ["DeviceId", "Detector", "period", "win"]
GE30 = ["m30", "h1", "h3", "h6", "h24", "full"]
ATS = {"Advance", "Presence", "Count", "Yellow_Red"}
C7 = ["Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other"]
GATE = 0.9
NETS = {"fj": "fj", "siba": "x69_siba", "sibm": "x69_sibm", "cw2": "x74_sibacw2"}
_E = {}


def locked():
    return set(pd.read_csv(DC_WORK / "official" / "locked_v2.csv").DeviceId.str.lower())


def env():
    if not _E:
        import of77
        s74, S, E = of77.setup_new()
        lc = of77.lane_ctx(E["fr"], of77.O8 / "lanes_D.func.parquet")[:, 2]
        _E.update(s74=s74, S=S, E=E, lc=lc, F77=of77.F77)
    return _E


def decode(P, gate=True):
    """gated champion decode of P -> (pred idx, credit frames per set)."""
    x = env()
    S, E, fr = x["S"], x["E"], x["E"]["fr"]
    lanes0 = fr.lanes5g.copy()
    if gate:
        fr["lanes5g"] = lanes0.where(~(x["lc"] < GATE), None)
    pr, cr = S.run_decode(E, P)
    fr["lanes5g"] = lanes0
    return pr, cr


def credit_argmax(P):
    import atspm_score as AS
    x = env()
    E, fr = x["E"], x["E"]["fr"]
    for i, c in enumerate(C7):
        fr[f"P_{c}"] = P[:, i]
    pr = P.argmax(1)
    return pr, {s: AS.credit(fr, pr, "truth_v3s", r, True) for s, r in E["rows"].items()}


def frame_ok(cr):
    x = env()
    n = len(x["E"]["fr"])
    o = {}
    for s, r in x["E"]["rows"].items():
        a = np.full(n, np.nan)
        a[r] = cr[s].ok_a.to_numpy(float)
        o[s[0].upper()] = a
    return o


# ================================================================================================ rows
def stage_rows():
    x = env()
    E, fr, s74 = x["E"], x["E"]["fr"], x["s74"]
    assert not fr.DeviceId.isin(locked()).any()
    seeds = [np.load(x["F77"] / "s74" / "f77" / f"stack_ctx_s{s}.npy").astype(float) for s in (0, 1, 2)]
    Pst = np.mean(seeds, 0)
    pr, cr = decode(Pst)
    ok = frame_ok(cr)
    g30 = fr.wgroup.isin(GE30).to_numpy()
    for s in "ER":
        m = g30 & ~np.isnan(ok[s])
        print(f"champion >= 30 min {s}: {np.mean(ok[s][m]):.4f} on {m.sum():,} rows")
    d = fr[KEY + ["wgroup", "fold", "det_n_on", "validated", "truth_v3s"]].copy()
    d["validated"] = d.validated.astype(str)
    d["pred"] = np.array(C7, object)[pr]
    d["p_max"] = Pst.max(1)
    d["p_truth"] = [Pst[i, C7.index(t)] if t in C7 else np.nan for i, t in enumerate(d.truth_v3s)]
    for s in "ER":
        d[f"ok_{s}"] = ok[s]
    full = {s: cr[s] for s in cr}
    for s, r in E["rows"].items():
        e = np.full(len(fr), None, object)
        e[r] = full[s].err.to_numpy(object)
        d[f"err_{s[0].upper()}"] = e
    # model families: argmax (stack credit) and decoded (gated champion decode)
    Ps = {"stack": Pst, "trees": E["Pt"], "siba": E["Pn"]}
    for k, pref in NETS.items():
        if k == "siba":
            continue
        P = s74.net_probs(fr, pref)
        has = ~np.isnan(P[:, 0])
        print(f"{k}: coverage {has.mean():.4f}")
        Ps[k] = np.where(has[:, None], P, np.nan)
    # GRU siba backbone: fold 0, seed 0 only
    g = pd.read_parquet(DC_WORK / "tcn53" / "fpreds" / "x74_sibagru_f0.parquet")
    g["DeviceId"] = g.DeviceId.str.lower()
    g = g.astype({"Detector": fr.Detector.dtype})
    Pg = fr[KEY].merge(g[KEY + [f"P_{c}" for c in C7]], on=KEY, how="left")[[f"P_{c}" for c in C7]].to_numpy(float)
    Pg[fr.fold.to_numpy() != 0] = np.nan
    Ps["gru0"] = Pg
    for i, s in enumerate(seeds):
        Ps[f"stack_s{i}"] = s
    for k, P in Ps.items():
        has = ~np.isnan(P[:, 0])
        Pf = np.where(has[:, None], P, Pst)            # missing rows: filled, flagged by has_<k>
        pa, ca = credit_argmax(Pf)
        oa = frame_ok(ca)
        d[f"has_{k}"] = has
        d[f"am_{k}"] = np.array(C7, object)[pa]
        d[f"pam_{k}"] = Pf.max(1)
        for s in "ER":
            d[f"okam_{k}_{s}"] = np.where(has, oa[s], np.nan)
        if k in ("trees", "siba", "fj", "sibm", "cw2", "stack_s0", "stack_s1", "stack_s2"):
            pdk, cdk = decode(Pf)
            od = frame_ok(cdk)
            d[f"dec_{k}"] = np.array(C7, object)[pdk]
            for s in "ER":
                d[f"okdec_{k}_{s}"] = np.where(has, od[s], np.nan)
        print(f"  {k} done", flush=True)
    # undecoded / ungated stacker views
    pdn, cdn = decode(Pst, gate=False)
    on = frame_ok(cdn)
    for s in "ER":
        d[f"okng_stack_{s}"] = on[s]
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT / "rows.parquet", index=False)
    print(f"-> {OUT / 'rows.parquet'} {d.shape}")


# ================================================================================================ attrib
def attrib_rows(d, s):
    import err70
    r51 = pd.read_parquet(DC_WORK / "trackA" / "err51" / "rows.parquet")
    m = err70.meta()
    x = d[d[f"ok_{s}"].notna()].copy()
    x = x.rename(columns={"truth_v3s": "truth"})
    x["ok"] = x[f"ok_{s}"].astype(bool)
    x["err"] = x[f"err_{s}"].fillna("")
    x = x.merge(m, on=["DeviceId", "Detector"], how="left")
    x = err70.side(x, r51)
    df, e = err70.attribute(x)
    return df, e


FAMS = ["trees", "fj", "siba", "sibm", "cw2", "stack"]


def stage_attrib():
    d = pd.read_parquet(OUT / "rows.parquet")
    res = {}
    for s, sn in (("R", "realistic"), ("E", "everything")):
        df, e = attrib_rows(d, s)
        e = e.copy()
        e["bucket"] = e.cat.str[0]
        e.to_parquet(OUT / f"errors_{sn}.parquet", index=False)
        for pool, gs in (("ge30", GE30), ("m5", ["m5"]), ("m10", ["m10"])):
            dd, ee = df[df.wgroup.isin(gs)], e[e.wgroup.isin(gs)]
            n = len(dd)
            r = {"rows": n, "signals": int(dd.DeviceId.nunique()), "err_pt": round(100 * len(ee) / n, 2),
                 "acc": round(1 - len(ee) / n, 4)}
            for b in "ABCD":
                k = (ee.bucket == b).sum()
                r[b] = [round(100 * k / n, 2), round(k / max(len(ee), 1), 3)]
            r["cats"] = {c: round(100 * v / n, 2) for c, v in ee.cat.value_counts().items()}
            # oracle: share of each bucket's errors that at least one family gets right (argmax or decoded)
            ix = ee.set_index(KEY).index
            dk = d.set_index(KEY)
            okany = np.zeros(len(ee), bool)
            per = {}
            for f in FAMS:
                cols = [c for c in (f"okam_{f}_{s}", f"okdec_{f}_{s}") if c in dk.columns]
                v = np.zeros(len(ee), bool)
                for c in cols:
                    v |= dk.loc[ix, c].fillna(0).to_numpy(float) > 0
                per[f] = v
                okany |= v
            r["oracle_any_family_right_share"] = {b: round(float(okany[(ee.bucket == b).to_numpy()].mean()), 3)
                                                  for b in "ABCD"}
            r["oracle_any_family_right_pt"] = round(100 * okany.sum() / n, 2)
            r["oracle_by_family_pt"] = {f: round(100 * v.sum() / n, 2) for f, v in per.items()}
            res[f"{s}_{pool}"] = r
            print(s, pool, json.dumps(r))
    # model views (>= 30 min): each family argmax / decoded, champion ungated
    g30 = d.wgroup.isin(GE30)
    mv = {}
    for s in "ER":
        m = g30 & d[f"ok_{s}"].notna()
        mv[s] = {"champion": round(float(d.loc[m, f"ok_{s}"].mean()), 4),
                 "champion_nogate": round(float(d.loc[m, f"okng_stack_{s}"].mean()), 4)}
        for c in [c for c in d.columns if (c.startswith("okam_") or c.startswith("okdec_")) and c.endswith(f"_{s}")]:
            mm = m & d[c].notna()
            mv[s][c[:-2]] = [round(float(d.loc[mm, c].mean()), 4), int(mm.sum())]
        # gru0 vs siba on fold 0 (same rows)
        f0 = m & d.has_gru0
        mv[s]["fold0_gru0_vs_siba_argmax"] = [round(float(d.loc[f0, f"okam_gru0_{s}"].mean()), 4),
                                              round(float(d.loc[f0, f"okam_siba_{s}"].mean()), 4)]
    res["model_views_ge30"] = mv
    print(json.dumps(mv, indent=0))
    json.dump(res, open(OUT / "attrib78.json", "w"), indent=1)


# ================================================================================================ labels
def cfg_class(s):
    if s is None or pd.isna(s):
        return None
    s = str(s).strip().lower()
    return {"advance": "Advance", "presence": "Presence", "count": "Count", "yellow_red": "Yellow_Red",
            "mid": "Mid", "mid loop": "Mid", "bike": "Bike", "bike zone presence": "Bike"}.get(s, "Other")


def atspm_cls(c):
    return c if c in ATS else "nonATSPM"


def stage_labels():
    lk = locked()
    res = {}
    v = pd.read_parquet(rpath.LABELS_CURRENT)  # note 81: v4l
    v["DeviceId"] = v.DeviceId.str.lower()
    v = v[~v.DeviceId.isin(lk)]
    # (b) print vs config, where both exist
    b = v[v.print_function.notna() & v.config_function.notna() & v.n_on_new.fillna(0).ge(5)].copy()
    b["cc"] = b.config_function.map(cfg_class)
    for conf in ("high", "medium", "low"):
        x = b[b.print_confidence.eq(conf)]
        dis7 = (x.cc != x.print_function).mean()
        disA = (x.cc.map(atspm_cls) != x.print_function.map(atspm_cls)).mean()
        res[f"print_vs_config_{conf}"] = {"n": len(x), "disagree7": round(float(dis7), 3),
                                          "disagree_atspm": round(float(disA), 3)}
    x = b[b.print_confidence.eq("high") & (b.cc.map(atspm_cls) != b.print_function.map(atspm_cls))]
    res["print_high_vs_config_pairs"] = x.groupby(["cc", "print_function"]).size().sort_values(ascending=False)\
        .head(10).to_dict()
    res["print_high_vs_config_pairs"] = {f"{a}->{c}": int(n) for (a, c), n in res["print_high_vs_config_pairs"].items()}
    # user rulings vs what they overrode
    u = v[v.source.eq("user_ruling")]
    res["user_ruling_rows"] = int(len(u))
    uu = u[u.config_function.notna()]
    res["user_ruling_vs_config_disagree_atspm"] = round(float(
        (uu.config_function.map(cfg_class).map(atspm_cls) != uu.truth_v3s.map(atspm_cls)).mean()), 3)
    # label source mix of the scoring labels
    res["source_mix"] = v[v.truth_v3s.notna()].source.fillna("none").value_counts(normalize=True).round(3).to_dict()
    # (c) behaviour-validation status, labelled rows with data
    vv = v[v.truth_v3s.notna() & v.validated.notna()]
    res["validated_share"] = vv.validated.value_counts(normalize=True).round(4).to_dict()
    chk = vv[vv.validated.isin(["pass", "fail", "misconfigured"])]
    res["fail_or_misconf_of_checked"] = round(float(chk.validated.isin(["fail", "misconfigured"]).mean()), 4)
    # (d) all model families confident and agreeing against the label (champion rows, >= 30 min, realistic)
    d = pd.read_parquet(OUT / "rows.parquet")
    for s in "RE":
        m = d.wgroup.isin(GE30) & d[f"ok_{s}"].notna()
        x = d[m].copy()
        fams = ["trees", "fj", "siba", "sibm", "cw2", "stack"]
        am = np.stack([x[f"am_{f}"].to_numpy(object) for f in fams], 1)
        pm = np.stack([x[f"pam_{f}"].to_numpy(float) for f in fams], 1)
        same = (am == am[:, :1]).all(1)
        # ATSPM-level contradiction: the agreed class scores wrong against the label (no stack credit at this level)
        ag = am[:, 0]
        wrong = np.array([atspm_cls(a) != atspm_cls(t) for a, t in zip(ag, x.truth_v3s)])
        n = len(x)
        out = {}
        for thr in (0.8, 0.9, 0.95):
            c = same & (pm >= thr).all(1) & wrong
            # among champion errors
            err = ~x[f"ok_{s}"].astype(bool).to_numpy()
            out[thr] = {"rows_pt": round(100 * c.sum() / n, 2), "share_of_errors": round(float((c & err).sum() / err.sum()), 3),
                        "detectors": int(x[c][["DeviceId", "Detector"]].drop_duplicates().shape[0])}
        # detector-level: all-agree-against in >= half of the detector's >= 30-min windows
        x["c9"] = same & (pm >= 0.9).all(1) & wrong
        det = x.groupby(["DeviceId", "Detector"]).c9.mean()
        out["detectors_majority_c9"] = int((det >= .5).sum())
        out["detectors_total"] = int(len(det))
        # split the .9 rows by note-78 bucket (definitional B vs the rest)
        e = pd.read_parquet(OUT / f"errors_{'realistic' if s == 'R' else 'everything'}.parquet")
        e = e[e.wgroup.isin(GE30)]
        ek = e.set_index(KEY).cat
        xi = x[x.c9].set_index(KEY).index
        cats = ek.reindex(xi)
        out["c9_by_bucket_pt"] = {k: round(100 * v / n, 2) for k, v in cats.fillna("credited/ok").str[:2].value_counts().items()}
        res[f"all_agree_{s}"] = out
    print(json.dumps(res, indent=1, default=str))
    json.dump(res, open(OUT / "labels78.json", "w"), indent=1, default=str)


# ================================================================================================ lc / seeds
def stage_lc():
    import t57_function as T57F
    import v3_retrain as V
    sys.path.insert(0, str(CODE / "final76"))
    import f76_function as F  # noqa: F401
    x = env()
    fr = x["E"]["fr"]
    T57F.setup("v6e")
    fk = pd.read_parquet(V.FEATS, columns=V.KEY + ["fold"])
    fk["DeviceId"] = fk.DeviceId.str.lower()
    fk["_i"] = np.arange(len(fk))
    fk["Detector"] = fk.Detector.astype(fr.Detector.dtype)
    idx = fr[V.KEY].merge(fk[V.KEY + ["_i"]], on=V.KEY, how="left")._i.to_numpy().astype(int)
    fo = fk.fold.to_numpy()
    d = DC_WORK / "err78" / "lc"
    g30 = fr.wgroup.isin(GE30).to_numpy()
    m5 = fr.wgroup.isin(["m5", "m10"]).to_numpy()
    sig = fr.DeviceId.to_numpy()
    import cand64 as C

    def P_of(fn):
        P = np.zeros((len(fk), 7), np.float32)
        for k in range(6):
            P[fo == k] = np.load(fn(k))
        return P[idx].astype(float)

    full = T57F.cfg_dir("v6e", F.CFG)
    arms = {"1.00_s0": P_of(lambda k: full / f"P_first.all.wi_s0_f{k}.npy")}
    for f in ("0.25", "0.50", "0.75"):
        for dr in (78, 79):
            if all((d / f"P_{f}_d{dr}_f{k}.npy").exists() for k in range(6)):
                arms[f"{f}_d{dr}"] = P_of(lambda k, f=f, dr=dr: d / f"P_{f}_d{dr}_f{k}.npy")
    for sd in (1, 2):
        arms[f"1.00_s{sd}"] = P_of(lambda k, sd=sd: full / f"P_first.all.wi_s{sd}_f{k}.npy")
    res = {}
    ref = {}
    for name, P in arms.items():
        _, ca = credit_argmax(P)
        _, cd = decode(P)
        oa, od = frame_ok(ca), frame_ok(cd)
        r = {}
        for s in "ER":
            for lab, o in (("argmax", oa[s]), ("decoded", od[s])):
                mm = g30 & ~np.isnan(o)
                r[f"{s}_ge30_{lab}"] = round(float(np.mean(o[mm])), 4)
                ms = m5 & ~np.isnan(o)
                r[f"{s}_m5m10_{lab}"] = round(float(np.mean(o[ms])), 4)
                if name == "1.00_s0":
                    ref[(s, lab)] = o
                else:
                    a = ref[(s, lab)]
                    mm2 = g30 & ~np.isnan(o) & ~np.isnan(a)
                    r[f"{s}_ge30_{lab}_vs_full_s0"] = C.delta_ci(a[mm2], o[mm2], sig[mm2])
        res[name] = r
        print(name, json.dumps(r, default=str), flush=True)
    # stacker seed spread inside the champion
    dd = pd.read_parquet(OUT / "rows.parquet")
    m = dd.wgroup.isin(GE30)
    res["stack_seed_spread"] = {s: [round(float(dd.loc[m & dd[f"okdec_stack_s{i}_{s}"].notna(), f"okdec_stack_s{i}_{s}"]
                                           .mean()), 4) for i in range(3)] for s in "ER"}
    print(res["stack_seed_spread"])
    json.dump(res, open(OUT / "lc78.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    globals()[f"stage_{sys.argv[1]}"]()
