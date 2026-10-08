"""Note 87b A: CPU inference time of the phase network per signal at 3 h, GRU p3 (package ONNX) vs TCN (tcn53 fold-0
checkpoints exported to ONNX here), onnxruntime, 4 intra-op threads, pair batch 64 (the package setting).

Shape-only timing: these graphs' cost does not depend on input values, so each net is run on random inputs of the exact
production shape: 4 pieces x 30 min (K = 4 of a 32 grid above 120 min), [pairs, channels, T] with pairs = the signal's
candidates with tree p >= .01 at its h3 window (f76 OOF pool; the package's filter). Signals = the pool's median and
maximum kept-pair count at h3 (non-locked, asserted). Raster building is NOT timed (the TCN's 6 extra channels are
O(events) numpy; the package's 'networks: streams' stage is ~0.03-0.1 s).  Warm = median of 5 runs after 1 warm-up.

    python t87_speed.py        -> %DC_WORK%/s87/t87_speed.json (+ s87/onnx/*.onnx)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: F401,E402
from common import DC_WORK  # noqa: E402

OUT = DC_WORK / "s87"
PKG = DC_WORK / "final_v3_candidate_v4b" / "weights" / "gru.onnx"
THREADS, PB, REPS = 4, 64, 5


def export_tcn(tag: str) -> tuple[Path, int, int]:
    import torch
    torch.set_num_threads(THREADS)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
    from neural import tcn53 as T
    ck = torch.load(T.MODELDIR / f"{tag}.pt", map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    m = T.build_model(cfg).eval()
    m.load_state_dict(ck["state"])
    cin, bw = len(cfg["chans"]), int(cfg["bw"])
    T_ = 1800 * 1000 // bw
    d = OUT / "onnx"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{tag}.onnx"
    if not f.exists():
        x = torch.randn(8, cin, T_)
        torch.onnx.export(m, (x,), str(f), input_names=["x"], output_names=["logit"], opset_version=17,
                          dynamic_axes={"x": {0: "n", 2: "t"}, "logit": {0: "n"}}, dynamo=False)
        import onnxruntime as ort
        s = ort.InferenceSession(str(f), providers=["CPUExecutionProvider"])
        with torch.no_grad():
            ref = m(x).numpy()
        got = s.run(None, {"x": x.numpy()})[0].ravel()
        print(f"{tag}: parity max |diff| {np.abs(ref.ravel() - got).max():.2e}", flush=True)
    return f, cin, T_


def session(f: Path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = THREADS
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(f), so, providers=["CPUExecutionProvider"])


def run(s, n: int, cin: int, T_: int, pieces: int = 4) -> float:
    name = s.get_inputs()[0].name
    rng = np.random.default_rng(0)
    xs = [rng.random((n, cin, T_), dtype=np.float32) for _ in range(pieces)]

    def once():
        t0 = time.perf_counter()
        for x in xs:
            for i in range(0, n, PB):
                s.run(None, {name: np.ascontiguousarray(x[i:i + PB])})
        return time.perf_counter() - t0
    once()
    return float(np.median([once() for _ in range(REPS)]))


def main():
    lk = set(pd.read_csv(DC_WORK / "official" / "locked_v2.csv").DeviceId.str.lower())
    q = pd.read_parquet(DC_WORK / "final_v3_work" / "f76" / "phase" / "full" / "oof.parquet",
                        columns=["DeviceId", "Detector", "win", "cand_phase", "p0_bag"])
    q = q[q.win.str.startswith("h3")]
    assert not q.DeviceId.str.replace("@stg", "").str.lower().isin(lk).any()
    kept = q[q.p0_bag >= 0.01].groupby(["DeviceId", "win"]).size()
    allp = q.groupby(["DeviceId", "win"]).size()
    srt = kept.sort_values()
    pick = {"typical": srt.index[len(srt) // 2], "busiest": srt.index[-1]}
    res = {"settings": dict(threads=THREADS, pair_batch=PB, pieces=4, reps=REPS, runtime="onnxruntime CPU"),
           "signals": {k: {"kept_pairs": int(kept[v]), "all_pairs": int(allp[v]),
                           "pct90_kept": int(srt.quantile(.9))} for k, v in pick.items()}}
    nets = {"gru_p3": (PKG, 9, 1800)}
    for tag in ("ad_all_e100_f0", "r05_all_f0"):
        nets[tag] = export_tcn(tag)
    sizes = {k: v["kept_pairs"] for k, v in res["signals"].items()}
    sizes["p90"] = int(srt.quantile(.9))
    for nm, (f, cin, T_) in nets.items():
        s = session(f)
        res.setdefault("seconds_3h_kept", {})[nm] = {k: round(run(s, n, cin, T_), 3) for k, n in sizes.items()}
        print(nm, res["seconds_3h_kept"][nm], flush=True)
    g = res["seconds_3h_kept"]["gru_p3"]
    res["ratio_gru_over"] = {nm: {k: round(g[k] / v[k], 2) for k in v} for nm, v in res["seconds_3h_kept"].items()}
    json.dump(res, open(OUT / "t87_speed.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
