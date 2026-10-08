"""run hv4.assess(keep_all) on every capture -> hres_<tag>.pkl ; `cmp a b` compares"""
import sys, pickle, warnings, time, glob, os
import numpy as np, pandas as pd
def run(pkg, tag, pat="cap/*.pkl"):
    sys.path.insert(0, pkg); warnings.simplefilter("ignore")
    from detector_classifier import health_v4 as hv4
    import inspect as _insp  # CONPATCH
    if "con" in _insp.signature(hv4.assess).parameters:
        from detector_classifier import pipeline as _pl
        _con = _pl._connect(4, "4GB")
        _orig_assess = hv4.assess
        hv4.assess = lambda *x, **kw: _orig_assess(*x, **kw, con=_con)
    res = {}
    for f in sorted(glob.glob(pat)):
        a, k = pickle.load(open(f, "rb"))
        from detector_classifier import health_core as hc
        P0 = a[0]
        a = (hc.prep_arrays(P0.t, P0.eid, P0.par),) + tuple(a[1:])
        t = time.perf_counter()
        r = hv4.assess(*a, **k, keep_all=True)
        res[os.path.basename(f)[:-4]] = (r, time.perf_counter() - t)
    pickle.dump(res, open(f"hres_{tag}.pkl", "wb"))
    print(tag, "total", sum(v[1] for v in res.values()))
def eq(a, b):
    if isinstance(a, pd.DataFrame):
        if list(a.columns) != list(b.columns): return False, "cols"
        bad = [c for c in a.columns if not a[c].reset_index(drop=True).equals(b[c].reset_index(drop=True))]
        return (not bad and len(a) == len(b)), bad
    return a == b, ""
def cmp(t1, t2, internals=True):
    A, B = pickle.load(open(f"hres_{t1}.pkl", "rb")), pickle.load(open(f"hres_{t2}.pkl", "rb"))
    nb = 0
    for k in A:
        (oa, na, *ra), _ = A[k]; (ob, nbb, *rb), _ = B[k]
        e1, d1 = eq(oa, ob)
        e2 = na == nbb
        msg = []
        if not e1: msg.append(f"out {d1}")
        if not e2: msg.append("note")
        if internals and ra:
            Ra, Rb = ra[0], rb[0]
            common = [c for c in Ra.columns if c in Rb.columns]
            bad = [c for c in common if not Ra[c].reset_index(drop=True).equals(Rb[c].reset_index(drop=True))]
            if bad: msg.append(f"R {bad[:12]}")
            DROP = {"n_twin", "chop5", "nb5", "chop15_self", "chop15_sig", "corr_ph", "corr_ph_exp", "beta15", "chop30",
                    "nb30", "chop15_ref", "chop5_ref", "zero15_exp", "max_dev_1h", "mean_on_s", "occ_max_bin", "disp",
                    "disp_z"}
            miss = [c for c in Ra.columns if c not in Rb.columns and c not in DROP]
            if miss: msg.append(f"Rmissing {miss[:8]}")
            Ea, Eb = ra[1], rb[1]
            ce = [c for c in Ea.columns if c in Eb.columns]
            bad = [c for c in ce if not Ea[c].reset_index(drop=True).equals(Eb[c].reset_index(drop=True))]
            if bad or len(Ea) != len(Eb): msg.append(f"E {bad[:8]}")
        if msg:
            nb += 1; print("DIFF", k, "; ".join(msg))
    ta = sum(v[1] for v in A.values()); tb = sum(v[1] for v in B.values())
    print(f"cases {len(A)} differ {nb}   time {t1} {ta:.2f}s  {t2} {tb:.2f}s")
if __name__ == "__main__":
    if sys.argv[1] == "cmp": cmp(sys.argv[2], sys.argv[3], len(sys.argv) < 5)
    else: run(sys.argv[1], sys.argv[2])
