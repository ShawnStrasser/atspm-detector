"""Note 90: parity of the package's phase TCN path vs the research tcn53 path, on real signals.

For the bench signals (non-locked; 4 signals x 30 min / 3 h / 24 h, the package's own piece plan) the package builds its
interval streams from the raw events (gru_input.build_streams); on THAT bundle:
  raster   research tcn53.render53 + assemble53 (the training / OOF code) vs package GruPhaseModel.raster
           (funcnet.render_cached + assemble) -> max |diff| over every pair, channel and bin (must be 0)
  logits   torch Net53 (models/ad_all76_full.pt, fp32 CPU) on the research raster vs onnxruntime phase_tcn.onnx on
           the package raster
  probs    research pooling (mean log-softmax over pieces, renormalised over kept candidates) vs package
           gru_blend.phase_probs_kept, every pair and with a random candidate filter
    python parity90.py phase   -> %DC_WORK%/final_v3_work/v3fit90/parity_tcn90.json
CPU only, 2 threads.  locked_v2 asserted absent.
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: F401,E402
from common import DC_WORK  # noqa: E402

PKG = Path(os.environ.get("DC_PKG", str(DC_WORK / "final_v3_candidate_v4f")))
B71 = DC_WORK / "bench71"
OUT = DC_WORK / "final_v3_work" / "v3fit90"
TOL = 1e-4


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def cmd_phase():
    import torch
    torch.set_num_threads(2)
    from neural import tcn53 as M
    ck = torch.load(M.MODELDIR / "ad_all76_full.pt", map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    net = M.build_model(cfg).eval()
    net.load_state_dict(ck["state"])
    pkg_mods = {f.stem for f in PKG.glob("*.py")}
    for name in list(sys.modules):
        if name in pkg_mods:
            del sys.modules[name]
    sys.path.insert(0, str(PKG))
    import gru_input as gi
    import gru_blend as gb
    import predict as P
    assert Path(gb.__file__).resolve().parent == PKG.resolve()
    cfg_g = gb.config(PKG / "weights")
    assert Path(cfg_g["weights_path"]).name == "phase_tcn.onnx"
    from gru_onnx import GruPhaseModel
    model = GruPhaseModel(cfg_g["weights_path"], pair_batch=64, threads=2)
    assert model.n_ch == 15
    lk = set(pd.read_csv(DC_WORK / "official" / "locked_v2.csv").DeviceId.str.lower())
    rng = np.random.default_rng(90)
    res = {"pkg": str(PKG), "cases": [], "max_raster": 0.0, "max_logit": 0.0, "max_prob": 0.0, "max_prob_filtered": 0.0,
           "pairs": 0}
    for sig in ("typical_r8", "typical_r11", "busiest_ev", "busiest_ch"):
        for L in ("m30", "h3", "h24"):
            f = B71 / f"ev_{sig}_{L}.parquet"
            con = P._connect(2)
            w0, w1, _ = P.load_events(con, str(f))
            assert not set(con.sql("select distinct DeviceId from ev").df().DeviceId.str.lower()) & lk
            t0, t1 = int(round(w0 * 1000)), int(round(w1 * 1000))
            streams, rel = gb.streams_for(con, t0, t1, **gb.piece_plan(cfg_g, round((w1 - w0) / 60)))
            con.close()
            dev, z = next(iter(streams.items()))
            dets = [int(c) for c in z["det_ch"]]
            cand = [int(c) for c in z["cand"]]
            D, K = len(dets), len(cand)
            z["_chpos"] = {int(c): i for i, c in enumerate(z["det_ch"])}
            lp_t = np.zeros((D, K))
            mr = ml = 0.0
            for a, b in rel:
                T = max(int((b - a) // 1000), 8)
                det, ph, sg, nact, partner = M.render53(z, int(a), dets, T, 1000)
                bt = dict(det=torch.from_numpy(det[None]), ph=torch.from_numpy(ph[None]), sig=torch.from_numpy(sg[None]),
                          ncand=torch.tensor([K]), partner=torch.from_numpy(partner[None]))
                bi = torch.zeros(D * K, dtype=torch.long)
                di = torch.arange(D).repeat_interleave(K)
                ki = torch.arange(K).repeat(D)
                xr = M.assemble53(bt, bi, di, ki, cfg["chans"]).float()
                xp, _ = model.raster(z, dets, int(a), T)
                mr = max(mr, float(np.abs(xr.numpy() - xp).max()))
                with torch.no_grad():
                    lt = net(xr, None).float().numpy().ravel()
                lo = model.logits(xp)
                ml = max(ml, float(np.abs(lt - lo).max()))
                lp_t += torch.log_softmax(torch.from_numpy(lt.reshape(D, K)).double(), 1).numpy()
            lp_t /= len(rel)
            # package: every pair, then a random candidate filter (research: renormalise the pooled log-probs)
            got = gb.phase_probs_kept(streams, rel, cfg_g["weights_path"], None)
            pt = np.exp(lp_t - lp_t.max(1, keepdims=True))
            pt /= pt.sum(1, keepdims=True)
            ref = pd.DataFrame({"Detector": np.repeat(dets, K), "cand_phase": np.tile(cand, D), "p": pt.ravel()})
            mm = got.merge(ref, on=["Detector", "cand_phase"], how="inner")
            assert len(mm) == D * K
            mp = float(np.abs(mm.p_gru - mm.p).max())
            keep = {}
            for d in dets:
                m = rng.random(K) < 0.4
                m[rng.integers(K)] = True
                keep[(dev, d)] = {c for c, x in zip(cand, m) if x}
            gotf = gb.phase_probs_kept(streams, rel, cfg_g["weights_path"], keep)
            km = np.array([[c in keep[(dev, d)] for c in cand] for d in dets])
            lpk = np.where(km, lp_t, -np.inf)
            pk = np.exp(lpk - lpk.max(1, keepdims=True))
            pk /= pk.sum(1, keepdims=True)
            reff = pd.DataFrame({"Detector": np.repeat(dets, K), "cand_phase": np.tile(cand, D), "p": pk.ravel()})
            mf = gotf.merge(reff, on=["Detector", "cand_phase"], how="inner")
            assert len(mf) == int(km.sum())
            mpf = float(np.abs(mf.p_gru - mf.p).max())
            case = {"signal": sig, "len": L, "dets": D, "cands": K, "pieces": len(rel), "max_raster": mr,
                    "max_logit": ml, "max_prob": mp, "max_prob_filtered": mpf}
            res["cases"].append(case)
            res["pairs"] += D * K * len(rel)
            for k in ("max_raster", "max_logit", "max_prob", "max_prob_filtered"):
                res[k] = max(res[k], case[k])
            log(case)
    res["pass"] = bool(res["max_raster"] == 0.0 and max(res["max_logit"], res["max_prob"], res["max_prob_filtered"]) < TOL)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "parity_tcn90.json", "w"), indent=1)
    log(f"phase TCN parity: raster {res['max_raster']:.1e}, logit {res['max_logit']:.2e}, prob {res['max_prob']:.2e} / "
        f"filtered {res['max_prob_filtered']:.2e} over {res['pairs']:,} pair-pieces -> {'PASS' if res['pass'] else 'FAIL'}")
    assert res["pass"]


if __name__ == "__main__":
    {"phase": cmd_phase}[sys.argv[1]]()
