"""Note 86: ONNX export of the full-data PHASE TCN (tcn53 ad_all recipe, `%DC_WORK%/tcn53/models/ad_all76_full.pt`) with a
torch-parity check.  The graph is the pair scorer (Net53) alone: input x [pairs, 15 channels, T bins at 1 s] -> logit
[pairs] (as note 87b's speed export); the raster / channel assembly stays in numpy.

Parity (torch fp32 CPU vs onnxruntime CPU, max |logit diff| <= 1e-4):
  * random inputs of the production shape (64 pairs x 1800 bins, and 7 pairs x 300 bins: dynamic axes);
  * REAL pair inputs: the tcn53 inference path (D2 table, DS53 / collate53 / assemble53) on 6 non-locked pool signals x the
    5-min, 30-min and one 30-min piece of the 3-h window -- every pair, then per detector-window log-softmax over candidates.

    python export86_phase.py [TAG]   -> %DC_WORK%/s86/phase/ad_all76_full.onnx + parity86_phase.json
CPU only (CUDA hidden), 4 threads.  locked_v2 asserted absent.
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: F401,E402
from common import DC_WORK  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "ad_all76_full"   # a fold checkpoint tag = code test only
OUT = DC_WORK / "s86" / "phase"
TOL = 1e-4


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    import torch
    import onnxruntime as ort
    from torch.utils.data import DataLoader
    torch.set_num_threads(4)
    from neural import tcn53 as T
    from neural import data2 as D2
    from neural import infer2 as I2
    from neural import train2 as T2
    from neural import phase_v3_net as P3N
    ck = torch.load(T.MODELDIR / f"{TAG}.pt", map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    assert cfg["bw"] == 1000 and len(cfg["chans"]) == 15 and cfg["feats"] == "none", cfg
    assert cfg["fold"] == "full" or TAG != "ad_all76_full", cfg
    m = T.build_model(cfg).eval()
    m.load_state_dict(ck["state"])
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"{TAG}.onnx"
    x0 = torch.randn(8, 15, 1800)
    torch.onnx.export(m, (x0,), str(f), input_names=["x"], output_names=["logit"], opset_version=17,
                      dynamic_axes={"x": {0: "n", 2: "t"}, "logit": {0: "n"}}, dynamo=False)
    s = ort.InferenceSession(str(f), providers=["CPUExecutionProvider"])
    res = {"tag": TAG, "onnx": str(f), "cfg": {k: v for k, v in cfg.items() if k != "lr_sched"},
           "epochs": cfg.get("epochs"), "tol": TOL, "random": {}, "real": {}}

    def cmp(x):
        with torch.no_grad():
            ref = m(x, None).float().numpy().ravel()
        got = s.run(None, {"x": x.numpy()})[0].ravel()
        return ref, got

    for n, t in ((64, 1800), (7, 300)):
        ref, got = cmp(torch.randn(n, 15, t))
        res["random"][f"{n}x{t}"] = float(np.abs(ref - got).max())
    # real inputs: the tcn53 inference path on a few non-locked pool signals
    sigs = D2.training_signals()
    lk = P3N.locked_ids()
    assert not sigs.DeviceId.isin(lk).any()
    table = D2.load_table(sigs, labelled_only=False)
    keys = [k for k in sigs.key if k in table and table[k]["period"] == "dec"]
    keys = [keys[i] for i in np.linspace(0, len(keys) - 1, 6).astype(int)]
    assert not any(k.split("|", 1)[1] in lk for k in keys)
    mx_l, mx_lp, npairs = 0.0, 0.0, 0
    for name, start, secs in I2.WINDOWS["dec"]:
        if name not in ("m5_a", "m30_b", "h3_a"):
            continue
        pcs = I2.pieces("dec", start, secs, T.GRID)[:1]
        plan = [(k, p0, (p1 - p0) // 1000, name) for k in keys for p0, p1 in pcs]
        dl = DataLoader(T.DS53(table, plan, cfg["bw"]), batch_sampler=T.batches53(plan, 180, 6), num_workers=0,
                        collate_fn=T.collate53)
        for batch in dl:
            bi, di, ki, B, D, K = T2.pair_index(batch["dmask"], batch["ncand"])
            x = T.assemble53(batch, bi, di, ki, cfg["chans"]).float()
            ref, got = cmp(x)
            mx_l = max(mx_l, float(np.abs(ref - got).max()))
            npairs += len(ref)
            for v in (ref, got):
                lg = torch.full((B, D, K), -1e4)
                lg[bi, di, ki] = torch.from_numpy(v)
                v[:] = torch.log_softmax(lg, 2)[bi, di, ki].numpy()
            mx_lp = max(mx_lp, float(np.abs(ref - got).max()))
        log(f"  {name}: pairs so far {npairs:,}, max |dlogit| {mx_l:.2e}, |dlogp| {mx_lp:.2e}")
    res["real"] = {"signals": len(keys), "windows": ["m5_a", "m30_b", "h3_a (1 piece)"], "pairs": npairs,
                   "max_abs_logit": mx_l, "max_abs_logp": mx_lp}
    worst = max(max(res["random"].values()), mx_l, mx_lp)
    res["pass"] = bool(worst <= TOL)
    json.dump(res, open(OUT / "parity86_phase.json", "w"), indent=1, default=str)
    log(f"{TAG}: parity random {res['random']}, real {res['real']} -> {'PASS' if res['pass'] else 'FAIL'}")
    assert res["pass"]


if __name__ == "__main__":
    main()
