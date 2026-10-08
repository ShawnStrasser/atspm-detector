"""Extra analysis for note 28: symmetric evaluation sets, 95% paired bootstrap, compact tables.

    python v3_analyse.py                                   # note 28 original: frame v5, validated == pass
    python v3_analyse.py --frame v6 --nc-high --variants first.all.wi,first.mixed.wi,v2.all.nowi \
        --extra first.all.wi --out ana28_lc3             # note 28 rerun on label-check v3
--nc-high: rows = pass + not_checkable high-print rows (v3_retrain --allow-not-checkable-high).
--extra V: variant V's OOF from the pass-only run directory, added as "V[pass]" (scored on the same rows).
Also reports the pass-only subset of FIX (FIXp), per-class P/R, and detectors on protected-permissive (FYA) phases.
"""
import sys, json, argparse
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v3_retrain as V

ap = argparse.ArgumentParser()
ap.add_argument("--frame", default="v5")
ap.add_argument("--nc-high", action="store_true", dest="nc")
ap.add_argument("--variants", default="first.high.wi,first.high.nowi,first.mixed.wi,first.mixed.nowi,first.all.wi,v2.all.nowi")
ap.add_argument("--extra", default="")
ap.add_argument("--out", default="ana28")
ap.add_argument("--also", default="", help="NAME=SUFFIX:VARIANT,... OOF from run directories <run>SUFFIX (e.g. "
                "lc=_lc:first.all.wi); bootstrapped vs --ref too")
ap.add_argument("--ref", default="", help="variant the --also runs are compared with (paired bootstrap)")
A = ap.parse_args()
V.set_frame(A.frame)
fr, _ = V.load_feats()
VARS = A.variants.split(",")
# exclusion accounting: step by step, none -> +min5 -> +dq -> +extra unusual (== CLEAN)
def state(min_on, clean, req_val=False, nc=False):
    V.MIN_ON, V.CLEAN, V.REQ_VAL, V.NC_HIGH = min_on, clean, req_val, nc
    lab = V.load_labels(fr, "exclude")
    return lab, V.eval_sets(fr, lab), {v: V.variant_target(lab, fr, v)[1] for v in VARS}
lab0, es0, tr0 = state(0, False)
_, esm, trm = state(5, False)
labc, esc, trc = state(5, True)
labp, esp, trp = state(5, True, True)          # validated == pass only (note 28 final run)
lab, es1, tr1 = state(5, True, True, A.nc)     # the run's rule (pass [+ not_checkable high print])
# dq only (no extra unusual): recompute with dq but original unusual flags
dqrow = lab.dq.to_numpy()
nvrow = lab.notval.to_numpy()
low = (fr.det_n_on < 5).to_numpy()
print("frame rows:", len(fr), "zero-actuation:", int((fr.det_n_on == 0).sum()), "1-4 actuations:", int(low.sum()))
for k in es0:
    a, b, cc, c = es0[k][0], esm[k][0], esc[k][0], es1[k][0]
    print(f"eval {k}: {int(a.sum())} -> <5 act -{int(a.sum()-b.sum())} -> dq/extra-unusual -{int(b.sum()-cc.sum())}"
          f" (of which dq {int((b & dqrow).sum())}) -> not validated -{int(cc.sum()-c.sum())}"
          f" = {int(c.sum())} rows, {pd.unique(fr.DeviceId[c]).size} signals;"
          f" <5 by window {dict((g, int((a & low & (fr.wgroup == g).to_numpy()).sum())) for g in ('m5','m10','m30','h1','h6','full'))}")
for v in VARS:
    a, b, cc, c = tr0[v], trm[v], trc[v], tr1[v]
    print(f"train {v}: {int(a.sum())} -> <5 -{int(a.sum()-b.sum())} -> dq+extra -{int(b.sum()-cc.sum())}"
          f" (dq {int((b & dqrow).sum())}) -> not validated -{int(cc.sum()-c.sum())} = {int(c.sum())}")
print("dead-labelled rows in frame:", int(lab.dead.sum()), "in any FIX/train set:",
      int((lab.dead.to_numpy() & (es1['FIX'][0] | tr1[VARS[0]])).sum()))
rd = V.run_dir("exclude")
V.NC_HIGH = False
rd_pass = V.run_dir("exclude")
V.NC_HIGH = A.nc
C7 = np.array(V.C7, object)
TO5 = V.TO5
wg = fr.wgroup.to_numpy()
sig = fr.DeviceId.to_numpy()
es = V.eval_sets(fr, lab)
fix = es["FIX"][0]
yp = lab.print_function.to_numpy(object)
yv2 = lab.func7_v2.to_numpy(object)
agr = fix & (yp == yv2)
nov2 = fix & pd.isna(yv2)
dis = fix & pd.notna(yv2) & (yp != yv2)
Pb, hb = V.load_baseline(fr)

def to5(a):
    return np.array([TO5.get(c, c) for c in a], object)

def mf1(yt, p, cls):
    f = []
    for c in cls:
        tp = ((yt == c) & (p == c)).sum(); npred = (p == c).sum(); nt = (yt == c).sum()
        if nt == 0: continue
        pr = tp / npred if npred else 0; rc = tp / nt
        f.append(2 * pr * rc / (pr + rc) if pr + rc > 0 else 0)
    return np.mean(f)

def block(pred, yt, m):
    yt, p = yt[m], pred[m]
    core = np.isin(yt, V.CORE4)
    return dict(n=int(m.sum()), sig=int(pd.unique(sig[m]).size),
                acc7=(yt == p).mean(), acc5=(to5(yt) == to5(p)).mean(),
                core4=(yt[core] == p[core]).mean(), mF1_7=mf1(yt, p, V.C7),
                mF1_core4=mf1(yt, p, V.CORE4))

def boot(ok_a, ok_b, m, n_boot=2000, seed=0):
    s = sig[m]; a = ok_a[m].astype(float); b = ok_b[m].astype(float)
    u, inv = np.unique(s, return_inverse=True); n = len(u)
    sa = np.bincount(inv, a, n); sb = np.bincount(inv, b, n); c = np.bincount(inv, minlength=n)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (n_boot, n))
    d = (sb[idx].sum(1) - sa[idx].sum(1)) / c[idx].sum(1)
    return f"{(b.mean()-a.mean())*100:+.2f} [{np.quantile(d,.025)*100:+.2f},{np.quantile(d,.975)*100:+.2f}]"

variants = VARS
preds, seedpreds, have_all = {}, {}, np.ones(len(fr), bool)
for v in variants:
    Ps, hv = [], None
    for s in (0, 1, 2):
        P, h, got = V.load_oof(fr, rd, v, s, range(6))
        if len(got) < 6: continue
        Ps.append(P); hv = h if hv is None else hv & h
    if not Ps: continue
    preds[v] = C7[np.mean(Ps, 0).argmax(1)]
    seedpreds[v] = [C7[P.argmax(1)] for P in Ps]
    have_all &= hv
for v in [x for x in A.extra.split(",") if x]:
    Ps, hv = [], None
    for s in (0, 1, 2):
        P, h, got = V.load_oof(fr, rd_pass, v, s, range(6))
        if len(got) < 6: continue
        Ps.append(P); hv = h if hv is None else hv & h
    if not Ps: continue
    preds[v + "[pass]"] = C7[np.mean(Ps, 0).argmax(1)]
    seedpreds[v + "[pass]"] = [C7[P.argmax(1)] for P in Ps]
    have_all &= hv
for spec in [x for x in A.also.split(",") if x]:
    name, rest = spec.split("=")
    suf, v = rest.split(":")
    Ps, hv = [], None
    for s in (0, 1, 2):
        P, h, got = V.load_oof(fr, rd.parent / (rd.name + suf), v, s, range(6))
        if len(got) < 6: continue
        Ps.append(P); hv = h if hv is None else hv & h
    if not Ps: continue
    preds[name] = C7[np.mean(Ps, 0).argmax(1)]
    seedpreds[name] = [C7[P.argmax(1)] for P in Ps]
    have_all &= hv
# protected-permissive (FYA) phases (note 29, label_check_pplt): detectors whose labelled phase is one
v3p = pd.read_parquet(V.LABELS_V3, columns=["DeviceId", "detector", "phase_target"])
pp = pd.read_parquet(V.DCW / "cabinet" / "label_check_pplt.parquet")
pp = set(zip(pp.loc[pp.pplt.eq(True), "sid"].str.lower(), "P" + pp.loc[pp.pplt.eq(True), "p"].astype(str)))
v3p["fya"] = [(a.lower(), str(b)) in pp for a, b in zip(v3p.DeviceId, v3p.phase_target)]
v3p = v3p.assign(DeviceId=v3p.DeviceId.str.lower(), Detector=v3p.detector.astype(fr.Detector.dtype))
fya = fr[V.KEY].merge(v3p[["DeviceId", "Detector", "fya"]], on=["DeviceId", "Detector"], how="left").fya.eq(True).to_numpy()
# permissive = FYA / permissive left-turn phase (events, timing or behaviour) OR a right-turn-only lane (print lane R)
v3r = pd.read_parquet(V.LABELS_V3, columns=["DeviceId", "detector", "lane_type"])
v3r = v3r.assign(DeviceId=v3r.DeviceId.str.lower(), Detector=v3r.detector.astype(fr.Detector.dtype),
                 rt=v3r.lane_type.astype("string").eq("R").fillna(False))
rt = fr[V.KEY].merge(v3r[["DeviceId", "Detector", "rt"]], on=["DeviceId", "Detector"], how="left").rt.eq(True).to_numpy(dtype=bool, na_value=False)
perm = fya | rt
fixp = esp["FIX"][0]
preds["baseline"] = C7[Pb.argmax(1)]
out = {}
sets = {"FIX": fix, "FIXb": fix & hb, "AGR": agr, "AGRb": agr & hb, "NOV2": nov2, "DIS": dis,
        "FIXp": fixp, "FIXpb": fixp & hb, "FYA": fix & fya, "FYAb": fix & fya & hb, "nFYAb": fix & ~fya & hb,
        "PRM": fix & perm, "PRMb": fix & perm & hb}
for v, p in preds.items():
    r = {}
    for sn, m in sets.items():
        m = m & have_all & (hb if v == "baseline" else True)
        r[sn] = {g: block(p, yp, m & (wg == g if g != "all" else True)) for g in ("all", "m30", "h6", "full")}
    r["DIS_vs_v2"] = block(p, yv2, dis & have_all & (hb if v == "baseline" else True))
    if v in seedpreds:
        for sn2, m2 in (("FIXb", fix & hb), ("AGRb", agr & hb), ("DIS", dis)):
            for g in ("all", "m30", "h6", "full"):
                mm = m2 & have_all & (wg == g if g != "all" else True)
                r[f"sd_{sn2}_{g}"] = np.std([(q[mm] == yp[mm]).mean() for q in seedpreds[v]], ddof=1) * 100
        r["sd_acc7_FIX"] = np.std([(q[fix & have_all] == yp[fix & have_all]).mean() for q in seedpreds[v]], ddof=1) * 100
        r["sd_acc7_AGR"] = np.std([(q[agr & have_all] == yp[agr & have_all]).mean() for q in seedpreds[v]], ddof=1) * 100
    # per-class P/R on FIXb (the rows the baseline has, same for every model), all windows and full
    for c in V.C7:
        for g in ("all", "full"):
            m = fix & have_all & hb & (wg == g if g != "all" else True)
            tp = ((yp[m] == c) & (p[m] == c)).sum()
            r[f"{c}_{g}_PR"] = (tp / max((p[m] == c).sum(), 1), tp / max((yp[m] == c).sum(), 1))
    out[v] = r
# bootstraps (acc7 and core4) vs baseline on rows the baseline has, and vs the v2 control on all rows
okb = preds["baseline"] == yp
okc = preds.get("v2.all.nowi", preds["baseline"]) == yp
bs = {}
core = np.isin(yp, V.CORE4)
for v in preds:
    if v == "baseline": continue
    ok = preds[v] == yp
    bs[v] = {}
    okr = preds[A.ref] == yp if A.ref in preds and v != A.ref else None
    for sn, m in (("FIX", fix), ("AGR", agr), ("DIS", dis), ("FIXp", fixp), ("FYA", fix & fya), ("nFYA", fix & ~fya),
                  ("PRM", fix & perm)):
        if okr is not None:
            for g in ("all", "m30", "h6", "full"):
                bs[v][f"{sn}_{g}_vsRef"] = boot(okr, ok, m & have_all & (wg == g if g != "all" else True))
            bs[v][f"{sn}_core4_vsRef"] = boot(okr, ok, m & have_all & core)
        for g in ("all", "m30", "h6", "full"):
            mm = m & have_all & (wg == g if g != "all" else True)
            bs[v][f"{sn}_{g}_vsBase"] = boot(okb, ok, mm & hb)
            bs[v][f"{sn}_{g}_vsV2ctl"] = boot(okc, ok, mm)
        bs[v][f"{sn}_acc5_vsBase"] = boot(to5(preds["baseline"]) == to5(yp), to5(preds[v]) == to5(yp), m & have_all & hb)
        bs[v][f"{sn}_core4_vsBase"] = boot(okb, ok, m & have_all & hb & core)
        bs[v][f"{sn}_core4_vsV2ctl"] = boot(okc, ok, m & have_all & core)
# WI: share predicted non-PM, all windows and full, per variant (+ baseline where it has rows)
wi = es["WI"][0] & have_all
wires = {}
for v, p in preds.items():
    m = wi & (hb if v == "baseline" else True)
    nonpm = np.isin(p, ["Other", "Mid", "Bike"])
    wires[v] = dict(n=int(m.sum()), sig=int(pd.unique(sig[m]).size), all=nonpm[m].mean(),
                    full=nonpm[m & (wg == "full")].mean(), m30=nonpm[m & (wg == "m30")].mean())
    if v != "baseline":  # same rows as baseline for a fair compare
        mb = wi & hb
        wires[v]["on_base_rows_full"] = nonpm[mb & (wg == "full")].mean()
unu = es["UNU"][0] & have_all
yu = lab.label_print_first.to_numpy(object)
unures = {v: block(p, yu, unu & (hb if v == "baseline" else True)) for v, p in preds.items()}
v2m = es["V2"][0] & have_all
yv2t = lab.func5_v2t.to_numpy(object)
v2res = {v: dict(n=int((v2m & (hb if v == "baseline" else True)).sum()),
                 acc5=(to5(p)[v2m & (hb if v == "baseline" else True)] == yv2t[v2m & (hb if v == "baseline" else True)]).mean(),
                 acc5_b=(to5(p)[v2m & hb] == yv2t[v2m & hb]).mean(),
                 full_b=(to5(p)[v2m & hb & (wg == "full")] == yv2t[v2m & hb & (wg == "full")]).mean())
         for v, p in preds.items()}
res = dict(main=out, boot=bs, WI=wires, UNU=unures, V2=v2res,
           counts={k: int((m & have_all).sum()) for k, m in sets.items()},
           sigs={k: int(pd.unique(sig[m & have_all]).size) for k, m in sets.items()})
json.dump(res, open(rd / f"{A.out}.json", "w"), indent=1, default=lambda x: float(x) if np.isscalar(x) else str(x))

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
for sn in ("FIX", "FIXb", "AGR", "AGRb", "NOV2", "DIS", "FIXp", "FIXpb", "FYA", "FYAb", "nFYAb", "PRM", "PRMb"):
    rows = []
    for v in out:
        b = out[v][sn]
        rows.append(dict(v=v, n=b["all"]["n"], sig=b["all"]["sig"], acc7=b["all"]["acc7"], acc5=b["all"]["acc5"],
                         core4=b["all"]["core4"], mF1_7=b["all"]["mF1_7"], mF1_c4=b["all"]["mF1_core4"],
                         m30=b["m30"]["acc7"], h6=b["h6"]["acc7"], full=b["full"]["acc7"],
                         full_c4=b["full"]["core4"], sd=out[v].get("sd_acc7_" + sn[:3], np.nan)))
    print(f"\n== {sn} (truth = print label)"); print(pd.DataFrame(rows).round(4).to_string(index=False))
print("\n== DIS scored vs v2 label"); print(pd.DataFrame({v: out[v]["DIS_vs_v2"] for v in out}).T.round(4))
print("\n== bootstrap (acc pt, 95% CI, signal-grouped)"); print(pd.DataFrame(bs).T.to_string())
print("\n== WI non-PM share"); print(pd.DataFrame(wires).T.round(4))
print("\n== UNU"); print(pd.DataFrame(unures).T.round(4))
print("\n== V2 set"); print(pd.DataFrame(v2res).T.round(4))
print("\n== per-class P,R on FIXb (rows the baseline has)")
print(pd.DataFrame({v: {k: tuple(round(x, 3) for x in out[v][k]) for k in out[v] if k.endswith("_PR")} for v in out}).T.to_string())
print("\n== seed sd (acc7 pt)"); print(pd.DataFrame({v: {k: out[v][k] for k in out[v] if k.startswith("sd_")} for v in out if v in seedpreds}).T.round(3).to_string())
print(res["counts"], res["sigs"])
