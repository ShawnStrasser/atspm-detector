import os as _os
from pathlib import Path as _P
DCW = _os.environ.get("DC_WORK", str(_P.home() / "dc_work"))
import warnings, numpy as np, pandas as pd
warnings.simplefilter("ignore")
from detector_classifier import pipeline as P, funcnet as FN, function_stage as FS, stacker as ST
V = DCW + r"\s115v"
rng = np.random.default_rng([11, 12])
cm = dict(zip(range(1, 65), (rng.permutation(64) + 1).tolist())); pm = dict(zip(range(1, 17), (rng.permutation(16) + 1).tolist()))
ic, ip = {b: a for a, b in cm.items()}, {b: a for a, b in pm.items()}
cap = {}
o_both = FN.FuncNet.both
def both(self, z, dets, pieces_rel, keep=None):
    r = o_both(self, z, dets, pieces_rel, keep)
    cap.setdefault("net", []).append((list(map(int, dets)), [int(c) for c in z["cand"]], r[0].copy(), r[1].copy(), None if keep is None else {int(k): sorted(v) for k, v in keep.items()}))
    return r
FN.FuncNet.both = both
S, E = "2026-09-20 12:00:00", "2026-09-20 12:05:00"
res = {}
for tag, f in (("orig", rf"{V}\ev\v12.parquet"), ("perm", rf"{V}\perm\v12_perm.parquet")):
    cap.clear()
    P.predict(f, start=S, end=E, min_actuations=1, profile="full")
    res[tag] = cap["net"]
    print(tag, "net calls", len(cap["net"]), [ (len(x[0]), len(x[1]), x[4] is None) for x in cap["net"]])
for call in range(len(res["orig"])):
    da, ca, fa, pa, ka = res["orig"][call]; db, cb, fb, pb, kb = res["perm"][call]
    mapd = [ic[d] for d in db]; mapc = [ip[c] for c in cb]
    ia = {d: i for i, d in enumerate(da)}; ja = {c: i for i, c in enumerate(ca)}
    order = [ia[d] for d in mapd]
    print("call", call, "dets same set", sorted(mapd) == sorted(da), "cands same set", sorted(mapc) == sorted(ca))
    fd = np.abs(fa[order] - fb).max()
    pdif = np.abs(pa[order][:, [ja[c] for c in mapc]] - pb).max()
    print("  function probs max diff", fd, " phase probs max diff", pdif)
    if ka is not None:
        kbm = {ic[d]: sorted(ip[c] for c in v) for d, v in kb.items()}
        print("  keep equal", kbm == ka)
