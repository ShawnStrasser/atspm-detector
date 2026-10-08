"""Note 74: CPU inference time per signal of tcn69 function nets (torch, fp32, warm; same raster / batching as
`tcn69_func.py infer`), so variants (siba / sibm / GRU backbone / no-sibling base) are compared on equal terms.
Times the network forward for every piece of one window of one signal; raster loading is timed separately (it is the
same for every variant). Signals = fold-0 held-out keys of the function table (non-locked by construction): the one
with the median detector count ("typical") and the one with the most ("busiest"); windows m30_a and h3_a (stg).

    python time74.py --ckpt PATH[,PATH...] [--threads 4] [--reps 3]  -> %DC_WORK%/s74/time74.json (appends)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tcn69_func as T  # noqa: E402

F53, M, I2 = T.F53, T.M, T.I2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--keep", default="", help="tree pair table: time the candidate filter (tree p >= .01) too")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    table, sig = F53.func_table(False)
    f0 = set((sig.period + "|" + sig.DeviceId)[sig.fold == 0])
    keys = sorted(k for k in table if k in f0 and table[k]["period"] == "stg")
    nd = np.array([len(table[k]["dets"]) for k in keys])
    pick = {"typical": keys[int(np.argsort(nd)[len(nd) // 2])], "busiest": keys[int(nd.argmax())]}
    out = {}
    kt = T.keep_table(a.keep, 0.01) if a.keep else None
    for path in a.ckpt.split(","):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        model = T.Net69(cfg).eval()
        model.load_state_dict(ck["state"])
        name = Path(path).stem
        for lab, key in pick.items():
            for wname in ("m30_a", "h3_a"):
                start, secs = next((s, n) for w, s, n in I2.WINDOWS["stg"] if w == wname)
                pcs = I2.pieces("stg", start, secs, max(M.GRID, 4))
                if secs > M.LONG_S:
                    from pieces_infer import subset
                    pcs = [pcs[i] for i in subset(len(pcs), 4)]
                plan = [(key, p0, (p1 - p0) // 1000, wname) for p0, p1 in pcs]
                t0 = time.perf_counter()
                batches = list(DataLoader(F53.DSF(table, plan, cfg["bw"]), batch_sampler=M.batches53(plan, 180, 6),
                                          num_workers=0, collate_fn=F53.collateF))
                t_load = time.perf_counter() - t0
                for flt in ([False, True] if kt is not None else [False]):
                    T.STATS.update(pairs_all=0, pairs_run=0)
                    for b in batches:
                        b.pop("keep", None)
                        if flt:
                            b["keep"], _ = T.keep_mask(b, kt)
                    ts = []
                    with torch.no_grad():
                        for r in range(a.reps + 1):
                            t0 = time.perf_counter()
                            for b in batches:
                                T.forward69(model, b, "cpu", chunk=1024)
                            ts.append(time.perf_counter() - t0)
                    k = f"{name}|{lab}|{wname}" + ("|filter.01" if flt else "")
                    out[k] = dict(n_det=int(len(table[key]["dets"])), n_cand=int(len(table[key]["cand"])),
                                  pieces=len(pcs), load_s=round(t_load, 3), cold_s=round(ts[0], 3),
                                  warm_s=round(float(np.median(ts[1:])), 3), threads=a.threads,
                                  pair_share=round(T.STATS["pairs_run"] / max(T.STATS["pairs_all"], 1), 3))
                    print(k, out[k], flush=True)
    f = F53.DC_WORK / "s74" / "time74.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    old = json.load(open(f)) if f.exists() else {}
    old.update(out)
    json.dump(old, open(f, "w"), indent=1)


if __name__ == "__main__":
    main()
