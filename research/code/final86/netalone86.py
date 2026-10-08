"""Note 86 diagnostic: the siba net ALONE (argmax of the 7 function probabilities) vs v4l truth, labelled rows of the
stacker frame, per window pool and fold, for any fpreds net tags given (TAG or TAG:seed).  Descriptive only.
    set F76_ARM=c & python netalone86.py x74_sibaflt:0 x86_siba4l:0 ...
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("F76_ARM", "c")
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
import numpy as np  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "final83"))
import fit83 as F  # noqa: E402

import oof86  # noqa: E402  (label pointers -> v4m, v4m trees: as the oof86 scoring)
oof86.use_trees("v4m")
s74, S, E, Xs, y, trm, names = F.stacker_inputs()
import cand64 as C  # noqa: E402
fr = E["fr"]
tr = fr.truth_v3s.to_numpy(object)
lab = np.array([t in E["C7"] for t in tr])
C7 = np.array(E["C7"])
for tag in sys.argv[1:]:
    P = s74.net_probs(fr, tag)
    ok = (C7[np.nan_to_num(P, nan=-1).argmax(1)] == tr).astype(float)
    row = {p: round(100 * ok[lab & fr.wgroup.isin(f).to_numpy()].mean(), 2) for p, f in C.POOLS.items()}
    g = lab & fr.wgroup.isin(C.GE30).to_numpy()
    row["ge30_folds"] = [round(100 * ok[g & (fr.fold.to_numpy() == k)].mean(), 2) for k in range(6)]
    F.log(f"{tag}: {row}")
