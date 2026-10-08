"""numpy _fast_stats == the DuckDB SQL (with the window medians) on every capture and on random adversarial bins"""
import sys, pickle, warnings, glob
sys.path.insert(0, "../final_v7_next/src"); warnings.simplefilter("ignore")
from detector_classifier import health_core as hc, health_v4_stats as hs
import numpy as np, pandas as pd, duckdb
exec(open("patch/fast_np.py").read(), hs.__dict__)          # candidate _fast_stats into the module namespace
SQL_OLD = open("patch/sql_old.sql").read()
con = duckdb.connect(); con.execute("SET threads=1")
COLS = ["detector", "q5_gy", "q5_all", "fo_all", "fem_all", "n_spk", "fo_c", "fem_c", "n_spk_c"]


def cmp(cb, tag):
    con.register("h_cb", cb)
    A = con.execute(SQL_OLD).df().sort_values("detector").reset_index(drop=True)
    con.unregister("h_cb")
    B = hs._fast_stats(cb.assign(**hs._colour_medians(cb)))
    bad = [c for c in COLS if not (A[c].dtype == B[c].dtype and np.array_equal(A[c].to_numpy(), B[c].to_numpy(),
                                                                                equal_nan=True))]
    if bad:
        print("DIFF", tag, bad, [(A[c].dtype, B[c].dtype) for c in bad][:3])
    return not bad


ok = n = 0
for f in sorted(glob.glob("cap/*.pkl")):
    a, k = pickle.load(open(f, "rb"))
    P = hc.prep_arrays(a[0].t, a[0].eid, a[0].par)
    CB, _ = hs.colour_bins(P, (a[2] - a[1]).total_seconds(), a[3])
    if not len(CB):
        continue
    lanes = a[6]
    cb = CB.assign(ln=np.fmax(pd.to_numeric(CB.detector.map(lanes), errors="coerce").fillna(1).to_numpy(float), 1))
    n += 1
    ok += cmp(cb, f)
print("captures", n, "equal", ok)
rng = np.random.default_rng(7)
ok = n = 0
for t in range(300):
    nd, nb = int(rng.integers(1, 12)), int(rng.integers(1, 300))
    d = {"detector": np.repeat(np.arange(nd) + 1, nb).astype(np.int64), "b": np.tile(np.arange(nb), nd)}
    for c in ("n", "nG", "nY", "nR", "nU", "fGY", "fR", "fU", "cGY", "cR", "cU", "xGY", "xR", "xU"):
        d[c] = rng.integers(0, rng.choice([3, 30, 300]), nd * nb).astype(np.float32)
    sec = rng.random((3, nd * nb)) * 300
    sec = sec / np.maximum(sec.sum(0) / 300, 1) * rng.random(nd * nb)
    d["sGY"], d["sR"], d["sRg"] = (sec[i].astype(np.float32) for i in range(3))
    for c in ("sGY", "sR"):
        m = rng.random(nd * nb) < 0.2
        d[c][m] = 0
    cb = pd.DataFrame(d)
    cb["ln"] = rng.choice([1.0, 2.0, 3.0], nd * nb).repeat(1)
    cb["ln"] = np.repeat(rng.choice([1.0, 2.0, 3.0], nd), nb)
    n += 1
    ok += cmp(cb, f"rand{t}")
print("random", n, "equal", ok)
