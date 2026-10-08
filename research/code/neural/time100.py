"""Note 100: CPU time of the siba function network per signal as ONNX (onnxruntime, 4 threads, spinning off as in the
package), teacher-size vs distilled students. Inputs per piece exactly as export74.check builds them (candidate filter
tree p >= .01 applied, as in production). Signals = fold-0 held-out stg keys of the function table (non-locked by
construction): median detector count ("typical") and most detectors ("busiest"); windows m30_a and h3_a (4 pieces past
2 h, as tcn69_func infer). Network time only (raster built once beforehand; identical for every variant).
'mean3' = the three teacher members run back to back (what the package does).

    python time100.py --onnx NAME=PATH[,NAME=PATH...] [--mean3 A,B,C] [--reps 5]   -> %DC_WORK%/s100/time100.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tcn69_func as T  # noqa: E402

F53, M, I2 = T.F53, T.M, T.I2


def sess(path, threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def feeds(table, key, wname, kt):
    start, secs = next((s, n) for w, s, n in I2.WINDOWS["stg"] if w == wname)
    pcs = I2.pieces("stg", start, secs, max(M.GRID, 4))
    if secs > M.LONG_S:
        from pieces_infer import subset
        pcs = [pcs[i] for i in subset(len(pcs), 4)]
    plan = [(key, p0, (p1 - p0) // 1000, wname) for p0, p1 in pcs]
    out = []
    for b in DataLoader(F53.DSF(table, plan, 1000), batch_size=1, collate_fn=F53.collateF):
        b["keep"], _ = T.keep_mask(b, kt)
        bi, di, ki, B, D, K = T.T2.pair_index(b["dmask"], b["ncand"])
        mm = b["keep"][bi, di, ki]
        bi, di, ki = bi[mm], di[mm], ki[mm]
        xp = M.assemble53(b, bi, di, ki, T.ADALL)
        act = (b["nact"][0] > 0).float()
        out.append({"xp": xp.numpy(), "pd": di.numpy(), "pk": ki.numpy(), "act": act.numpy(),
                    "dk": np.zeros((D, K), np.float32)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--mean3", default="")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--keep", required=True)
    ap.add_argument("--frows", default="")
    a = ap.parse_args()
    if a.frows:
        F53.FROWS = Path(a.frows)
    torch.set_num_threads(a.threads)
    S = {n: sess(p, a.threads) for n, p in (x.split("=", 1) for x in a.onnx.split(","))}
    arms = {n: [n] for n in S}
    if a.mean3:
        arms["mean3"] = a.mean3.split(",")
    table, sig = F53.func_table(False)
    f0 = set((sig.period + "|" + sig.DeviceId)[sig.fold == 0])
    keys = sorted(k for k in table if k in f0 and table[k]["period"] == "stg")
    nd = np.array([len(table[k]["dets"]) for k in keys])
    pick = {"typical": keys[int(np.argsort(nd)[len(nd) // 2])], "busiest": keys[int(nd.argmax())]}
    kt = T.keep_table(a.keep, 0.01)
    out = {}
    for lab, key in pick.items():
        for wname in ("m30_a", "h3_a"):
            fd = feeds(table, key, wname, kt)
            for arm, members in arms.items():
                ts = []
                for r in range(a.reps + 1):
                    t0 = time.perf_counter()
                    for n in members:
                        for f in fd:
                            S[n].run(None, f)
                    ts.append(time.perf_counter() - t0)
                k = f"{arm}|{lab}|{wname}"
                out[k] = dict(n_det=int(len(table[key]["dets"])), n_cand=int(len(table[key]["cand"])), pieces=len(fd),
                              pairs=int(sum(len(f["pd"]) for f in fd)), cold_s=round(ts[0], 4),
                              warm_s=round(float(np.median(ts[1:])), 4), warm_min_s=round(float(min(ts[1:])), 4),
                              threads=a.threads)
                print(k, out[k], flush=True)
    f = F53.DC_WORK / "s100" / "time100.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    old = json.load(open(f)) if f.exists() else {}
    old.update(out)
    json.dump(old, open(f, "w"), indent=1)


if __name__ == "__main__":
    main()
