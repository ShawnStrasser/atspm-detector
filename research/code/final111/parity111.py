"""Note 111: parity of package v6's PHASE network (the siba members' pair phase head, funcnet.FuncNet.both) against the
torch checkpoints (research tcn69_func.forward69, fp32 CPU) on the bench extracts (typical / busiest x 30 min / 3 h):
per piece pair logits (unfiltered), per member and 3-member-average phase probabilities (research infer recipe: softmax
of the piece-mean log-softmax over candidates; filtered pairs logit -1e4), unfiltered AND with a random candidate
filter; the function probabilities of the same pass are compared too.

    python parity111.py     -> %DC_WORK%/s111/siba_phase_parity111.json  (PASS = every max |diff| <= 1e-4)
CPU only (CUDA hidden), 2 threads.  locked_v2 asserted absent from every signal used.
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "final83"))
import export83 as E  # noqa: E402

E.THREADS = 2
PKG = E.W / "final_v3_candidate_v6"
OUT = E.W / "s111"


def main():
    import torch
    torch.set_num_threads(E.THREADS)
    man = json.load(open(PKG / "weights" / "funcnet" / "manifest.json"))
    tags = [m["tag"] for m in man["members"]]
    nets = []
    for m in man["members"]:
        E.CKPT = E.W / "tcn53" / "models" / m["checkpoint"]
        net, cfg, T69 = E.build_net()
        nets.append(net)
    from neural import tcn53 as M
    pkg_mods = {f.stem for f in PKG.glob("*.py")}
    for name in list(sys.modules):
        if name in pkg_mods:
            del sys.modules[name]
    sys.path.insert(0, str(PKG))
    import gru_input as gi
    import gru_blend as gb
    import funcnet as FN
    import predict as P
    assert Path(FN.__file__).resolve().parent == PKG.resolve()
    model = FN.FuncNet(PKG / "weights" / "funcnet", threads=E.THREADS)
    assert model.member_tags == tags
    lk = E.locked()
    res = {"members": tags, "pieces": 0, "max_abs_logit": 0.0, "member_max_abs_pphase": 0.0,
           "avg_max_abs_pphase": 0.0, "filt_avg_max_abs_pphase": 0.0, "avg_max_abs_pfunc": 0.0,
           "filt_avg_max_abs_pfunc": 0.0, "argmax_phase_differs": 0, "cases": []}
    cfg_g = gb.config(PKG / "weights")
    rng = np.random.default_rng(111)
    for sig in ("typical_r8", "typical_r11", "busiest_ch", "busiest_ev"):
        for L in ("m30", "h3"):
            f = E.B71 / f"ev_{sig}_{L}.parquet"
            if not f.exists():
                continue
            con = P._connect(E.THREADS)
            w0, w1, _ = P.load_events(con, str(f))
            assert not set(con.sql("select distinct DeviceId from ev").df().DeviceId.str.lower()) & lk
            t0, t1 = int(round(w0 * 1000)), int(round(w1 * 1000))
            pieces = gi.split_range(t0, t1, gi.CHUNK_MS, **gb.piece_plan(cfg_g, round((w1 - w0) / 60)))
            z = next(iter(gi.build_streams(con, t0, t1, windows=pieces).values()))
            con.close()
            dets = [int(c) for c in z["det_ch"]]
            z["_chpos"] = {int(c): i for i, c in enumerate(z["det_ch"])}
            cand = [int(c) for c in z["cand"]]
            K = len(cand)
            keep = {}
            for d in dets:
                if rng.random() < 0.2:
                    continue
                m = rng.random(K) < 0.4
                m[rng.integers(K)] = True
                keep[d] = {c for c, x in zip(cand, m) if x}
            km = np.array([[c in keep.get(d, set(cand)) for c in cand] for d in dets], bool)
            pr = [(pa - t0, pb - t0) for pa, pb in pieces]
            pf_on, pp_on = model.both(z, dets, pr)
            pf_of, pp_of = model.both(z, dets, pr, keep=keep)
            for filt in (False, True):
                pps, pfs = [], []
                for i, net in enumerate(nets):
                    accq = accf = None
                    for pa, pb in pieces:
                        a_, T = pa - t0, max(int((pb - pa) // 1000), 8)
                        det, ph, sg, nact, partner = M.render53(z, a_, dets, T, 1000)
                        b = dict(det=torch.from_numpy(det[None]), ph=torch.from_numpy(ph[None]),
                                 sig=torch.from_numpy(sg[None]), ncand=torch.tensor([K]),
                                 dmask=torch.ones(1, len(dets), dtype=torch.bool),
                                 partner=torch.from_numpy(partner[None]), nact=torch.from_numpy(nact[None]))
                        if filt:
                            b["keep"] = torch.from_numpy(km[None])
                        with torch.no_grad():
                            lg_t, fl_t = T69.forward69(net, b, "cpu", chunk=1024)
                        lq = torch.log_softmax(lg_t.float(), dim=2)[0].numpy().astype(np.float64)
                        lf = torch.log_softmax(fl_t.float(), dim=2)[0].numpy().astype(np.float64)
                        if not filt:
                            _, lg_o = model.piece_logprobs(model.members[i], z, dets, a_, T, None)
                            res["max_abs_logit"] = max(res["max_abs_logit"], float(np.abs(lg_t[0].numpy() - lg_o).max()))
                        res["pieces"] += 1
                        accq = lq if accq is None else accq + lq
                        accf = lf if accf is None else accf + lf
                    for acc, lst in ((accq, pps), (accf, pfs)):
                        v = acc / len(pieces)
                        p = np.exp(v - v.max(1, keepdims=True))
                        lst.append(p / p.sum(1, keepdims=True))
                    if not filt:
                        one = FN.FuncNet.__new__(FN.FuncNet)
                        one.members, one.pair_batch, one.meta = [model.members[i]], model.pair_batch, model.meta
                        one.member_tags = [tags[i]]
                        _, pq1 = one.both(z, dets, pr)
                        res["member_max_abs_pphase"] = max(res["member_max_abs_pphase"],
                                                           float(np.abs(pps[-1] - pq1).max()))
                avg_q, avg_f = np.mean(pps, 0), np.mean(pfs, 0)
                oq, of_ = (pp_of, pf_of) if filt else (pp_on, pf_on)
                kq = "filt_avg_max_abs_pphase" if filt else "avg_max_abs_pphase"
                kf = "filt_avg_max_abs_pfunc" if filt else "avg_max_abs_pfunc"
                res[kq] = max(res[kq], float(np.abs(avg_q - oq).max()))
                res[kf] = max(res[kf], float(np.abs(avg_f - of_).max()))
                res["argmax_phase_differs"] += int((avg_q.argmax(1) != oq.argmax(1)).sum())
            res["cases"].append(dict(sig=sig, L=L, D=len(dets), K=K, pieces=len(pieces),
                                     filt_pairs_kept=round(float(km.mean()), 3)))
            E.log(f"{sig} {L}: D {len(dets)} K {K}; logit {res['max_abs_logit']:.2e} phase avg {res['avg_max_abs_pphase']:.2e}"
                  f" filt {res['filt_avg_max_abs_pphase']:.2e}; func {res['avg_max_abs_pfunc']:.2e}")
    res["pass_1e-4"] = bool(max(res["max_abs_logit"], res["member_max_abs_pphase"], res["avg_max_abs_pphase"],
                                res["filt_avg_max_abs_pphase"], res["avg_max_abs_pfunc"],
                                res["filt_avg_max_abs_pfunc"]) <= 1e-4)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "siba_phase_parity111.json", "w"), indent=1)
    E.log(json.dumps({k: v for k, v in res.items() if k != "cases"}))


if __name__ == "__main__":
    main()
