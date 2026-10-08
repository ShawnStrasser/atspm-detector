"""Note 84: the three full-data siba v4l members (seeds 0 / 1 / 2: tcn53/models/x74_sibafull4l{,_s1,_s2}_full.pt; their
research ONNX twins are s74/full/x74_sibafull4l{,_s1,_s2}_full.onnx) as package graphs of candidate v4b.

Each checkpoint is exported with export83's pair + head graphs (head masks candidates the pair net did not run), then
checked IN THE PACKAGE: package FuncNet (onnxruntime) vs the torch forward69 (fp32 CPU) on the bench extracts (typical /
busiest x 30 min / 3 h), every piece, each member alone (unfiltered and with a random candidate filter), and the
3-member probability average vs the average of the three torch members.

    python export84.py export   -> v4b weights/funcnet/{tag}_pair.onnx, {tag}_head.onnx (3 members), manifest.json
    python export84.py parity   -> %DC_WORK%/final_v3_work/v3fit84/siba_parity84.json
CPU only (CUDA hidden), 2 threads (other CPU work runs in parallel).  locked_v2 asserted absent from every signal used.
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "final83"))
import export83 as E  # noqa: E402

E.THREADS = 2
PKG = E.W / "final_v3_candidate_v4b"
OUT = E.W / "final_v3_work" / "v3fit84"
TMP = E.W / "final_v3_work" / "f84" / "export_tmp"
MEMBERS = [("x74_sibafull4l", "x74_sibafull4l_full.pt"), ("x74_sibafull4l_s1", "x74_sibafull4l_s1_full.pt"),
           ("x74_sibafull4l_s2", "x74_sibafull4l_s2_full.pt")]


def cmd_export():
    fd = PKG / "weights" / "funcnet"
    man = json.load(open(fd / "manifest.json"))
    for f in fd.glob("*.onnx"):
        f.unlink()
    mem = []
    for tag, ck in MEMBERS:
        (TMP / "weights" / "funcnet").mkdir(parents=True, exist_ok=True)
        E.PKG, E.CKPT, E.TAG = TMP, E.W / "tcn53" / "models" / ck, tag
        E.cmd_export()
        for s in ("pair", "head"):
            shutil.move(str(TMP / "weights" / "funcnet" / f"{tag}_{s}.onnx"), str(fd / f"{tag}_{s}.onnx"))
        mem.append({"tag": tag, "pair": f"{tag}_pair.onnx", "head": f"{tag}_head.onnx", "checkpoint": ck,
                    "seed": int(tag[-1]) if tag[-2:] in ("s1", "s2") else 0})
    man["members"] = mem
    man["averaging"] = ("three full-data members (seeds 0 / 1 / 2, same recipe) averaged in probability space -> stacker "
                        "'mean3'; members whose files are absent are skipped (one left -> stacker 'single')")
    man["trained"] = ("note 74 FULL x74_sibafull4l / _s1 / _s2: all non-locked training signals, v4l labels "
                      "(tcn53/func_rows_v4l.parquet), 43 epochs on the six-fold-picked schedule (s74/full/"
                      "lr_siba_full.json), research tcn53 / tcn69_func (snap74b, content partner tie-break)")
    man["candidate_filter"] = ("ON by default (predict.SIBA_FILTER, note 84): the pair net runs only on (detector, "
                               "candidate) pairs with tree p >= .01; the head masks the others (logit -1e4)")
    man["note"] = "note 84 production net (candidate v4b): 3 members"
    json.dump(man, open(fd / "manifest.json", "w"), indent=1)
    shutil.rmtree(TMP, ignore_errors=True)
    E.log(f"exported {len(mem)} members -> {fd}")


def cmd_parity():
    import torch
    torch.set_num_threads(E.THREADS)
    nets = []
    for tag, ck in MEMBERS:
        E.CKPT = E.W / "tcn53" / "models" / ck
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
    assert model.member_tags == [t for t, _ in MEMBERS], model.member_tags
    lk = E.locked()
    res = {"members": [t for t, _ in MEMBERS], "pieces": 0, "max_abs_flogp": 0.0, "max_abs_logit": 0.0,
           "filt_max_abs_flogp": 0.0, "member_max_abs_prob": 0.0, "avg_max_abs_prob": 0.0,
           "filt_avg_max_abs_prob": 0.0, "member_argmax_differs_share": [], "cases": []}
    cfg_g = gb.config(PKG / "weights")
    rng = np.random.default_rng(84)
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
            P_on, P_onf = model.probs(z, dets, pr), model.probs(z, dets, pr, keep=keep)
            for filt in (False, True):
                pts = []
                for i, net in enumerate(nets):
                    acc = None
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
                        lf_t = torch.log_softmax(fl_t.float(), dim=2)[0].numpy()
                        lf_o, lg_o = model.piece_logprobs(model.members[i], z, dets, a_, T, km if filt else None)
                        d1 = float(np.abs(lf_t - lf_o).max())
                        if filt:
                            res["filt_max_abs_flogp"] = max(res["filt_max_abs_flogp"], d1)
                        else:
                            res["max_abs_flogp"] = max(res["max_abs_flogp"], d1)
                            res["max_abs_logit"] = max(res["max_abs_logit"], float(np.abs(lg_t[0].numpy() - lg_o).max()))
                        res["pieces"] += 1
                        acc = lf_t if acc is None else acc + lf_t
                    p = np.exp(acc / len(pieces))
                    pts.append(p / p.sum(1, keepdims=True))
                    if not filt:
                        po = FN.FuncNet.__new__(FN.FuncNet)
                        po.members, po.pair_batch, po.meta = [model.members[i]], model.pair_batch, model.meta
                        res["member_max_abs_prob"] = max(res["member_max_abs_prob"],
                                                         float(np.abs(pts[-1] - po.probs(z, dets, pr)).max()))
                avg = np.mean(pts, 0)
                k_ = "filt_avg_max_abs_prob" if filt else "avg_max_abs_prob"
                res[k_] = max(res[k_], float(np.abs(avg - (P_onf if filt else P_on)).max()))
                if not filt:
                    am = np.array([q.argmax(1) for q in pts])
                    res["member_argmax_differs_share"].append(round(float((am != am[0]).any(0).mean()), 3))
            res["cases"].append(dict(sig=sig, L=L, D=len(dets), K=K, pieces=len(pieces)))
            E.log(f"{sig} {L}: D {len(dets)} K {K}; |dlogp| {res['max_abs_flogp']:.2e} filt {res['filt_max_abs_flogp']:.2e}"
                  f" avg |dprob| {res['avg_max_abs_prob']:.2e} filt {res['filt_avg_max_abs_prob']:.2e}")
    res["pass_1e-4"] = bool(max(res["max_abs_flogp"], res["filt_max_abs_flogp"], res["member_max_abs_prob"],
                                res["avg_max_abs_prob"], res["filt_avg_max_abs_prob"]) <= 1e-4)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "siba_parity84.json", "w"), indent=1)
    E.log(json.dumps({k: v for k, v in res.items() if k != "cases"}))


if __name__ == "__main__":
    {"export": cmd_export, "parity": cmd_parity}[sys.argv[1]]()
