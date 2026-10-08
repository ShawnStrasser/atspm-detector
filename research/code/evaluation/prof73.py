"""Note 73: per-component timing profile of the current champion pipeline, one signal, one process (groundwork for the
speed step; NO model change).

Package parts = the candidate package `%DC_WORK%/final_v3_candidate_v2` (numpy boosters, GRU ONNX), instrumented by
wrapping its functions (nothing in the package is edited).  Research-only parts are timed as PROXIES on the package's own
answers for the same signal:
  fj          TCN function head `tcn53/models/fj_f0.pt`, exported here to ONNX (one graph: pair net + candidate pooling +
              function head); raster = a torch-free copy of `tcn53.render53` + `assemble53` on the GRU's own interval
              streams (same 30-min pieces, K = 4 above 2 h), so the streams are shared with the GRU.
  D lanes     = the package lane step (A pair model: same tree shape 300 x 15 leaves x 3 seeds as the note-58 D model)
              + the D-only cost: gathering the 229 function features + 7 probabilities of both detectors (min / max).
  stack pick  `ln6_pick.work` + `ln7_stackhealth.work` (span / co-location / track / stack-health inputs; the ln6
              health call is off, health is shared with the package's) + `s67_decider.decode_x` (lane-confidence gate .9).
  stacker     `s67_decider.stack_X` + `ctx_X` + a 3-seed booster of the s67 shape (7 classes, 150 rounds, 7 leaves),
              trained here on random data (timing only) and run with the package's numpy booster.
  night speed `sp1_night_speed.night_speed` (as is: the bench windows are daytime -> early exit; plus a forced-night run).
  twin decode only on 5 / 10-min samples -> not run at 30 min / 3 h.
  function trees: the package's 442-feature 3-seed head stands in for the 229-feature arm (no exported booster).
  setback: the package's note-41 model stands in for sb7 (same model family).

Signals = note-71 bench extracts (`%DC_WORK%/bench71/ev_*.parquet`, non-locked, asserted): typical_r8 (19-22 channels),
busiest_ch (43 channels), busiest_ev (31 channels, most events); windows m30 (07:30-08:00) and h3 (06:00-09:00).
CPU, 4 threads.  Cold = fresh process, first call (model files parsed, sessions created); warm = models / sessions kept,
median of 3 further calls.

    python prof73.py export      # fj ONNX export + parity vs torch on real inputs; synthetic stacker boosters
    python prof73.py pb          # GRU pair batch 64 vs 512: p_gru and full predict() outputs
    python prof73.py profile     # every (signal, length) in a fresh process -> %DC_WORK%/prof73/r_*.json
    python prof73.py summary     # rows of note 73 -> %DC_WORK%/prof73/prof.json
    python prof73.py ideas [--trees-only]   # speed headroom on the 3-h signals: flat / ONNX TreeEnsemble boosters,
                                 # GRU candidate filter by tree probability, int8 quantisation, K = 2 pieces
"""
from __future__ import annotations

import time
_T_START = time.perf_counter()
import os  # noqa: E402
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import functools  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

W = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
PKG = W / "final_v3_candidate_v2"
B71 = W / "bench71"
OUT = W / "prof73"
CODE = Path(__file__).resolve().parents[1]
FJ_PT = W / "tcn53" / "models" / "fj_f0.pt"
FJ_ONNX = OUT / "fj_f0.onnx"
SIGS = {"typical": "typical_r8", "busiest": "busiest_ch", "busiest_ev": "busiest_ev"}
LENS = ["m30", "h3"]
NOMINAL = {"m30": ("07:30:00", 1800), "h3": ("06:00:00", 10800)}
C7 = ["Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other"]
PKG_C = ["Advance", "Presence", "Count", "Yellow_Red", "Other", "Mid", "Bike"]      # package PROB_COLS order
THREADS = 4
FJ_PAIR_BATCH = 64
W_TREE = 0.6
GATE = 0.9


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def ev_file(sig, L):
    return B71 / f"ev_{SIGS[sig]}_{L}.parquet"


def locked() -> set:
    import pandas as pd
    return set(pd.read_csv(W / "official" / "locked_v2.csv").DeviceId.str.lower())


# ======================================================================== torch-free fj raster (copy of tcn53)
def _tent(t, w0, T, bw):
    import numpy as np
    out = np.zeros(T + 1, dtype=np.float64)
    if t.size:
        pos = (t.astype(np.float64) - w0) / bw - 0.5
        i0 = np.floor(pos).astype(np.int64)
        f = pos - i0
        m0 = (i0 >= 0) & (i0 < T)
        np.add.at(out, i0[m0], 1.0 - f[m0])
        m1 = (i0 + 1 >= 0) & (i0 + 1 < T)
        np.add.at(out, i0[m1] + 1, f[m1])
    return np.clip(out[:T], 0.0, 2.0).astype(np.float32)


def _sel(t, w0, w1, bw):
    import numpy as np
    i0 = int(np.searchsorted(t, w0 - bw, "left"))
    i1 = int(np.searchsorted(t, w1 + bw, "left"))
    return t[i0:i1]


def render53_np(z, w0, dets, T, bw=1000):
    """== tcn53.render53 (verified in `export`), with gru_input.cover / onrate (== neural.raster._cover / _onrate)."""
    import numpy as np
    from gru_input import cover, onrate
    w1 = w0 + T * bw
    K = len(z["cand"])
    ph = np.zeros((K, 6, T), dtype=np.float32)
    for tag, row in (("g", 0), ("y", 1), ("r", 2), ("c", 3)):
        ptr, on, off = z[tag + "_ptr"], z[tag + "_on"], z[tag + "_off"]
        for k in range(K):
            s, e = int(ptr[k]), int(ptr[k + 1])
            if e > s:
                ph[k, row] = cover(on[s:e], off[s:e], w0, w1, T, bw)
                if tag == "c":
                    ph[k, 4] = _tent(_sel(on[s:e], w0, w1, bw), w0, T, bw)
                    ph[k, 5] = _tent(_sel(off[s:e], w0, w1, bw), w0, T, bw)
    sig = np.zeros((3, T), dtype=np.float32)
    sig[0] = ph[:, 0].sum(0)
    sig[1] = ph[:, 3].sum(0)
    sig[2] = cover(z["co_on"], z["co_off"], w0, w1, T, bw)
    G = ph[:, 0].astype(np.float64)
    ov = G @ G.T
    np.fill_diagonal(ov, -1.0)
    partner = np.where(ov.max(1) > 0, ov.argmax(1), -1).astype(np.int64) if K > 1 else np.full(K, -1, np.int64)
    chpos = z.get("_chpos")
    if chpos is None:
        chpos = z["_chpos"] = {int(c): i for i, c in enumerate(z["det_ch"])}
    D = len(dets)
    det = np.zeros((D, 4, T), dtype=np.float32)
    nact = np.zeros(D, dtype=np.float32)
    ptr, on, off = z["det_ptr"], z["det_on"], z["det_off"]
    for i, ch in enumerate(dets):
        k = chpos.get(int(ch))
        if k is None:
            continue
        s, e = int(ptr[k]), int(ptr[k + 1])
        if e <= s:
            continue
        det[i, 0] = cover(on[s:e], off[s:e], w0, w1, T, bw)
        det[i, 1] = onrate(on[s:e], w0, w1, T, bw)
        nact[i] = det[i, 1].sum()
        det[i, 2] = _tent(_sel(on[s:e], w0, w1, bw), w0, T, bw)
        det[i, 3] = _tent(_sel(off[s:e], w0, w1, bw), w0, T, bw)
    det[:, 1] = np.clip(det[:, 1], 0, 4) / 2.0
    return det, ph, sig, nact, partner


def assemble15_np(det, ph, sig, partner):
    """-> [D, K, 15, T] in the fj channel order (tcn53_func.ADALL); == tcn53.assemble53 for one signal."""
    import numpy as np
    D, _, T = det.shape
    K = ph.shape[0]
    x = np.empty((D, K, 15, T), dtype=np.float32)
    x[:, :, 0:2] = det[:, None, 0:2]                      # occ, onrate
    x[:, :, 2:6] = ph[None, :, 0:4]                       # g, y, rc, call
    x[:, :, 6] = (sig[0][None, None] - ph[:, 0][None]) / 2.0
    x[:, :, 7] = (sig[1][None, None] - ph[:, 3][None]) / float(max(K, 2) - 1)
    x[:, :, 8] = sig[2][None, None]                       # coord
    x[:, :, 9:11] = det[:, None, 2:4]                     # onE, offE
    x[:, :, 11:13] = ph[None, :, 4:6]                     # cOn, cOff
    ok = (partner >= 0).astype(np.float32)
    pk = np.clip(partner, 0, None)
    x[:, :, 13] = (ph[pk, 0] * ok[:, None])[None]         # pg
    x[:, :, 14] = (ph[pk, 3] * ok[:, None])[None]         # pc
    return x


class FjOnnx:
    def __init__(self, path=FJ_ONNX, threads=THREADS, pair_batch=FJ_PAIR_BATCH):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.pair_batch = pair_batch

    def run(self, x):
        """x [D, K, 15, T] -> function log-probs [D, 7], phase logits [D, K] (detector batches of ~pair_batch pairs)."""
        import numpy as np
        D, K = x.shape[:2]
        bd = max(1, self.pair_batch // max(K, 1))
        lf, lg = [], []
        for i in range(0, D, bd):
            a, b = self.s.run(None, {"x": np.ascontiguousarray(x[i:i + bd])})
            lf.append(a)
            lg.append(b)
        return np.concatenate(lf), np.concatenate(lg)


def fj_signal(model, z, span_ms, pieces_rel, rec=None):
    """fj on one signal: mean log-prob over pieces -> probabilities [D, 7] (C7 order), dets."""
    import numpy as np
    dets = [int(c) for c in z["det_ch"]]
    acc = None
    for a, b in pieces_rel:
        T = max(int((b - a) // 1000), 8)
        t = time.perf_counter()
        det, ph, sig, nact, partner = render53_np(z, int(a), dets, T)
        x = assemble15_np(det, ph, sig, partner)
        t1 = time.perf_counter()
        lf, _ = model.run(x)
        t2 = time.perf_counter()
        if rec is not None:
            rec("fj raster (render + assemble 15 ch)", t1 - t)
            rec("fj ONNX inference", t2 - t1)
        acc = lf if acc is None else acc + lf
    v = acc / len(pieces_rel)
    p = np.exp(v - v.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True), dets


# ======================================================================== export + parity
def cmd_export(a):
    import numpy as np
    import torch
    sys.path.insert(0, str(CODE))
    import rpath  # noqa: F401
    torch.set_num_threads(THREADS)
    from neural import tcn53_func as TF
    from neural import tcn53 as M
    from neural import train2 as T2
    OUT.mkdir(parents=True, exist_ok=True)
    ck = torch.load(FJ_PT, map_location="cpu", weights_only=False)
    net = TF.NetF(len(TF.ADALL), ck.get("arch", "tcn")).eval()
    net.load_state_dict(ck["state"])
    assert TF.ADALL == ["occ", "onrate", "g", "y", "rc", "call", "og", "oc", "coord", "onE", "offE", "cOn", "cOff",
                        "pg", "pc"]

    class FJ(torch.nn.Module):
        def __init__(self, n):
            super().__init__()
            self.n = n

        def forward(self, x):                                      # [D, K, 15, T]
            D, K, C, T = x.shape
            s, z = self.n(x.reshape(D * K, C, T))
            logit = s.reshape(D, K)
            Z = z.reshape(D, K, -1)
            p = torch.softmax(logit, dim=1)
            zd = torch.cat([(p[..., None] * Z).sum(1), Z.amax(1)], dim=-1)
            return torch.log_softmax(self.n.fhead(zd), dim=-1), logit

    fj = FJ(net).eval()
    ex = torch.randn(3, 4, 15, 600)
    kw = dict(input_names=["x"], output_names=["flogp", "logit"], opset_version=17,
              dynamic_axes={"x": {0: "D", 1: "K", 3: "T"}, "flogp": {0: "D"}, "logit": {0: "D", 1: "K"}})
    try:
        torch.onnx.export(fj, (ex,), str(FJ_ONNX), dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(fj, (ex,), str(FJ_ONNX), **kw)
    log(f"exported {FJ_ONNX} ({FJ_ONNX.stat().st_size / 2**20:.1f} MB)")

    # ---- parity on real inputs: the GRU's interval streams of bench signals, every piece
    rc = sys.modules.pop("common")
    sys.path.insert(0, str(PKG))
    import predict as P
    import gru_blend as gb
    import gru_input as gi
    sys.modules["common"] = rc
    assert Path(P.__file__).parent == PKG and Path(gi.__file__).parent == PKG
    model = FjOnnx()
    res = {"pieces": [], "max_abs_flogp": 0.0, "max_abs_logit": 0.0, "max_abs_prob": 0.0,
           "raster_max_abs": 0.0, "assemble_max_abs": 0.0}
    lk = locked()
    cfg = gb.config(PKG / "weights")
    for sig in ("typical", "busiest"):
        for L in LENS:
            con = P._connect(THREADS)
            w0, w1, info = P.load_events(con, str(ev_file(sig, L)))
            devs = con.sql("select distinct DeviceId from ev").df().DeviceId
            assert not set(devs.str.lower()) & lk
            t0, t1 = int(round(w0 * 1000)), int(round(w1 * 1000))
            plan = gb.piece_plan(cfg, round((w1 - w0) / 60))
            pieces = gi.split_range(t0, t1, gi.CHUNK_MS, **plan)
            z = next(iter(gi.build_streams(con, t0, t1, windows=pieces).values()))
            con.close()
            dets = [int(c) for c in z["det_ch"]]
            acc_t = acc_o = None
            for pa, pb in pieces:
                a_, T = pa - t0, max(int((pb - pa) // 1000), 8)
                det, ph, sg, nact, partner = render53_np(z, a_, dets, T)
                dT, pT, sT, nT, prT = M.render53(z, a_, dets, T, 1000)
                res["raster_max_abs"] = max(res["raster_max_abs"], float(max(np.abs(det - dT).max(), np.abs(ph - pT).max(),
                                                                               np.abs(sg - sT).max())))
                assert (partner == prT).all()
                x = assemble15_np(det, ph, sg, partner)
                K = len(z["cand"])
                b = dict(det=torch.from_numpy(det[None]), ph=torch.from_numpy(ph[None]), sig=torch.from_numpy(sg[None]),
                         ncand=torch.tensor([K]), dmask=torch.ones(1, len(dets), dtype=torch.bool),
                         partner=torch.from_numpy(partner[None]))
                bi, di, ki, _, _, _ = T2.pair_index(b["dmask"], b["ncand"])
                xt = M.assemble53(b, bi, di, ki, TF.ADALL).numpy().reshape(len(dets), K, 15, T)
                res["assemble_max_abs"] = max(res["assemble_max_abs"], float(np.abs(x - xt).max()))
                with torch.no_grad():                               # the research forward itself (fp32, CPU)
                    lg_t, fl_t = TF.forwardF(net, b, "cpu", TF.ADALL, chunk=1024)
                lf_t = torch.log_softmax(fl_t.float(), dim=2)[0].numpy()
                lg_t = lg_t[0].numpy()
                lf_o, lg_o = model.run(x)
                d1, d2 = float(np.abs(lf_t - lf_o).max()), float(np.abs(lg_t - lg_o).max())
                res["max_abs_flogp"] = max(res["max_abs_flogp"], d1)
                res["max_abs_logit"] = max(res["max_abs_logit"], d2)
                res["pieces"].append(dict(sig=sig, L=L, D=len(dets), K=K, T=T, flogp=d1, logit=d2,
                                          argmax_same=bool((lf_t.argmax(1) == lf_o.argmax(1)).all())))
                acc_t = lf_t if acc_t is None else acc_t + lf_t
                acc_o = lf_o if acc_o is None else acc_o + lf_o
            pt = np.exp(acc_t / len(pieces)); pt /= pt.sum(1, keepdims=True)
            po = np.exp(acc_o / len(pieces)); po /= po.sum(1, keepdims=True)
            res["max_abs_prob"] = max(res["max_abs_prob"], float(np.abs(pt - po).max()))
            log(f"{sig} {L}: {len(pieces)} pieces, D {len(dets)}; max |dlogp| {res['max_abs_flogp']:.2e}, "
                f"|dlogit| {res['max_abs_logit']:.2e}, |dprob| {res['max_abs_prob']:.2e}")
    res["pass_1e-4"] = bool(max(res["max_abs_flogp"], res["max_abs_logit"], res["max_abs_prob"]) <= 1e-4)
    json.dump(res, open(OUT / "fj_parity.json", "w"), indent=1)
    log(f"parity: {json.dumps({k: v for k, v in res.items() if k != 'pieces'})}")

    # ---- synthetic boosters of the s67 stacker shape (timing only; never used for any score)
    import lightgbm as lgb
    rng = np.random.default_rng(73)
    nX = 20 + 16 + 5 + 3 + 3 + 1 + 4 + 4 + 5 + 1          # stack_X + ctx_X widths (checked at run time)
    X = rng.normal(size=(20000, nX))
    y = rng.integers(0, 7, 20000)
    for s in range(3):
        prm = dict(objective="multiclass", num_class=7, learning_rate=0.05, num_leaves=7, max_depth=3,
                   min_data_in_leaf=300, lambda_l2=10.0, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                   num_threads=THREADS, verbose=-1, seed=s)
        lgb.train(prm, lgb.Dataset(X + 0.3 * y[:, None] * (rng.random(nX) > .7), y), num_boost_round=150) \
            .save_model(str(OUT / f"stacker_shape_s{s}.txt"))
    json.dump({"n_features": nX}, open(OUT / "stacker_shape.json", "w"))
    log("stacker-shaped boosters saved")


# ======================================================================== GRU pair batch 64 vs 512
def cmd_pb(a):
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(PKG))
    import predict as P
    import gru_blend as gb
    P.set_backend("numpy")
    lk = locked()
    cfg = gb.config(PKG / "weights")
    orig = gb.phase_probs
    rows = []
    for sig in SIGS:
        for L in LENS:
            f = str(ev_file(sig, L))
            con = P._connect(THREADS)
            w0, w1, _ = P.load_events(con, f)
            assert not set(con.sql("select distinct DeviceId from ev").df().DeviceId.str.lower()) & lk
            t0, t1 = int(round(w0 * 1000)), int(round(w1 * 1000))
            plan = gb.piece_plan(cfg, round((w1 - w0) / 60))
            g = {pb: orig(con, cfg["weights_path"], t0, t1, pair_batch=pb, **plan) for pb in (512, 64)}
            con.close()
            d = np.abs(g[512].p_gru.to_numpy() - g[64].p_gru.to_numpy())
            outs = {}
            for pb in (512, 64):
                gb.phase_probs = functools.partial(orig, pair_batch=pb)
                outs[pb] = P.predict(f, threads=THREADS)
            gb.phase_probs = orig
            A, B = outs[512], outs[64]
            num = [c for c in A.columns if pd.api.types.is_float_dtype(A[c])]
            dd = max(float(np.nanmax(np.abs(A[c].to_numpy(float) - B[c].to_numpy(float)))) if A[c].notna().any() else 0.0
                     for c in num)
            same_cols = [c for c in A.columns if not A[c].equals(B[c])]
            r = dict(sig=sig, L=L, pairs=len(g[512]), p_gru_max_abs=float(d.max()), p_gru_identical=bool((d == 0).all()),
                     predict_identical=A.equals(B), predict_cols_differing=same_cols, predict_float_max_abs=dd,
                     phase_pred_same=bool(A.phase_guess.equals(B.phase_guess)),
                     function_pred_same=bool(A.function_guess.equals(B.function_guess)))
            rows.append(r)
            log(json.dumps(r))
    json.dump(rows, open(OUT / "pb64.json", "w"), indent=1)


# ======================================================================== profile
REC: dict = {}
BCACHE: dict = {}


def rec(label, dt):
    REC[label] = REC.get(label, 0.0) + dt


def wrap(obj, name, label):
    f = getattr(obj, name)

    @functools.wraps(f)
    def g(*args, **kw):
        t = time.perf_counter()
        try:
            return f(*args, **kw)
        finally:
            rec(label, time.perf_counter() - t)
    setattr(obj, name, g)
    return f


class _TimedBooster:
    def __init__(self, b, label):
        self.b, self.label = b, label

    def predict(self, X, *a, **k):
        if CAPTURE_X:
            CAP[f"X:{self.label}"] = X
        t = time.perf_counter()
        try:
            return self.b.predict(X, *a, **k)
        finally:
            rec(self.label, time.perf_counter() - t)


def _booster_label(path):
    n = Path(path).stem
    if n.startswith("phase_lgbm"):
        return f"phase trees {n[-2:]}"
    if n.startswith("function_lgbm"):
        return f"function trees {n[-2:]}"
    if n.startswith("decode"):
        return "joint decoder trees"
    return n


CAP: dict = {}
CAPTURE_X = False


def instrument(P):
    import gru_blend as gb
    import gru_input as gi
    import gru_onnx as go
    import features as f1
    import features_partner as f2
    import features_yellowred as f3
    import similarity as sim_mod
    import decode as dec
    import features_expert as fx
    import lanes as lane_mod
    import setback as sb_mod
    import health_core as hc
    # event load / DuckDB
    for n, lab in (("_connect", "events: duckdb connect"), ("load_events", "events: load + dedupe"),
                   ("build_chunk_tables", "events: interval / cycle tables"),
                   ("detector_universe", "events: universe / facts / gate"), ("signal_facts", "events: universe / facts / gate"),
                   ("gate_frame", "events: universe / facts / gate")):
        wrap(P, n, lab)
    # GRU
    orig_bs = gi.build_streams

    def bs(*a, **k):
        t = time.perf_counter()
        r = orig_bs(*a, **k)
        rec("GRU: interval streams (SQL)", time.perf_counter() - t)
        CAP["streams"] = r
        CAP["stream_args"] = (a, k)
        return r
    gi.build_streams = bs
    wrap(go, "render", "GRU: raster (render + assemble)")
    wrap(go, "assemble", "GRU: raster (render + assemble)")
    wrap(go.GruPhaseModel, "logits", "GRU: ONNX inference")
    wrap(go.GruPhaseModel, "__init__", "load: GRU ONNX session")
    wrap(gb, "phase_probs", "[incl] GRU total")
    gb._MODEL_CACHE = type("Keep", (dict,), {"pop": lambda self, *a: None})()   # keep the session (warm)
    orig_mix = gb.mix

    def mix(df, col, gru, weight):
        t = time.perf_counter()
        r = orig_mix(df, col, gru, weight)
        rec("phase: GRU mix", time.perf_counter() - t)
        CAP["mix"] = (df[["DeviceId", "Detector", "cand_phase", col]].copy(), gru.copy(), weight, r)
        return r
    gb.mix = mix
    # phase features
    wrap(f1, "apply_window", "phase features: base (f1)")
    wrap(f1, "build_window", "phase features: base (f1)")
    wrap(f1, "finalise", "phase features: base (f1)")
    wrap(f2, "build_window", "phase features: partner (f2)")
    wrap(f2, "add_partner_diffs", "phase features: partner (f2)")
    wrap(sim_mod, "build_window", "phase features: similarity graph")
    wrap(f3, "window_cycles", "phase features: yellow-red / lag (f3)")
    wrap(f3, "build", "phase features: yellow-red / lag (f3)")
    wrap(P, "build_features", "[incl] phase features total")
    # boosters
    orig_load = P._load_booster

    def load(path):
        key = str(path)
        if key not in BCACHE:
            t = time.perf_counter()
            BCACHE[key] = orig_load(path)
            rec("load: tree models", time.perf_counter() - t)
        return _TimedBooster(BCACHE[key], _booster_label(path))
    P._load_booster = load
    wrap(P, "_softmax_by_detector", "phase: softmax / normalise")
    wrap(P, "_normalise_by_detector", "phase: softmax / normalise")
    wrap(dec, "assemble", "joint decoder features")
    wrap(P, "score", "[incl] phase scoring total")
    # function
    wrap(P, "_function_frame", "function features: frame (shape / sibling / lag)")
    orig_fx = fx.build

    def fxb(con, w0, w1, top):
        t = time.perf_counter()
        r = orig_fx(con, w0, w1, top)
        rec("function features: expert (fx)", time.perf_counter() - t)
        return r
    fx.build = fxb
    orig_sf = P.score_function

    def sf(*a, **k):
        t = time.perf_counter()
        r = orig_sf(*a, **k)
        rec("[incl] function total", time.perf_counter() - t)
        CAP["fn"] = r
        return r
    P.score_function = sf
    # lanes / setback / health
    PM = lane_mod.PairModel
    _pm = {}

    def pm_factory(d):
        if str(d) not in _pm:
            t = time.perf_counter()
            _pm[str(d)] = PM(d)
            rec("load: lane / setback models", time.perf_counter() - t)
        return _pm[str(d)]
    lane_mod.PairModel = pm_factory
    orig_pmp = PM.predict

    def pmp(self, Pf):
        t = time.perf_counter()
        r = orig_pmp(self, Pf)
        rec("lanes: pair model (3 seeds)", time.perf_counter() - t)
        CAP["lane_pairs"] = Pf[["DeviceId", "da", "db"]].copy() if "DeviceId" in Pf else Pf[["da", "db"]].copy()
        return r
    PM.predict = pmp
    wrap(lane_mod, "signal_pair_cues", "lanes: pair cues")
    wrap(lane_mod, "add_context", "lanes: pair cues")
    wrap(lane_mod, "decode_groups", "lanes: decode")
    orig_lanes = lane_mod.lanes

    def lanes(on, pr, pm, t0, t1):
        t = time.perf_counter()
        r = orig_lanes(on, pr, pm, t0, t1)
        rec("[incl] lanes total", time.perf_counter() - t)
        CAP["on"], CAP["lanes"], CAP["lane_ph"] = on, r[1], r[0]
        return r
    lane_mod.lanes = lanes
    SM = sb_mod.SetbackModel
    _sm = {}

    def sm_factory(d):
        if str(d) not in _sm:
            t = time.perf_counter()
            _sm[str(d)] = SM(d)
            rec("load: lane / setback models", time.perf_counter() - t)
        return _sm[str(d)]
    sb_mod.SetbackModel = sm_factory
    orig_sb = sb_mod.setback

    def sbk(*a, **k):
        t = time.perf_counter()
        r = orig_sb(*a, **k)
        rec("setback (package note-41 model)", time.perf_counter() - t)
        CAP["setback"] = r
        return r
    sb_mod.setback = sbk
    orig_h = hc.health

    def hh(*a, **k):
        t = time.perf_counter()
        r = orig_h(*a, **k)
        rec("health v5 (health_core)", time.perf_counter() - t)
        CAP["health"] = r
        return r
    hc.health = hh
    orig_po = P.post_outputs

    def po(*a, **k):
        t = time.perf_counter()
        r = orig_po(*a, **k)
        rec("[incl] lanes + setback + health total", time.perf_counter() - t)
        return r
    P.post_outputs = po
    wrap(P, "_assemble", "output assembly")
    wrap(P, "_top_phase", "output assembly")


def research_extras(P, sig, L, fjm_holder, stk_holder, cold):
    """research-only pieces on the package's own answers for the same signal (see module doc)."""
    import numpy as np
    import pandas as pd
    import gru_input as gi
    import gru_blend as gb
    R = {}

    def r(label, dt):
        R[label] = R.get(label, 0.0) + dt
    z_all = CAP["streams"]
    dev = next(iter(z_all))
    z = z_all[dev]
    (con_, t0_ms, t1_ms), kw = CAP["stream_args"]
    pieces = kw["windows"]
    rel = [(a - t0_ms, b - t0_ms) for a, b in pieces]
    # ---- fj
    if fjm_holder.get("m") is None:
        t = time.perf_counter()
        fjm_holder["m"] = FjOnnx()
        r("load: fj ONNX session", time.perf_counter() - t)
    t = time.perf_counter()
    Pn, dets = fj_signal(fjm_holder["m"], z, t1_ms - t0_ms, rel, r)
    r("[incl] fj total", time.perf_counter() - t)
    # ---- the package's answers as the research frame
    fn = CAP["fn"].copy()
    fn["Detector"] = fn.Detector.astype(int)
    Pt = fn[[f"p_{c.lower()}" for c in C7]].to_numpy(float)          # C7 order
    pos = {d: i for i, d in enumerate(dets)}
    ix = np.array([pos.get(int(d), -1) for d in fn.Detector])
    PnF = np.where((ix >= 0)[:, None], Pn[np.clip(ix, 0, None)], Pt)
    t = time.perf_counter()
    Pb = W_TREE * Pt + (1 - W_TREE) * PnF
    r("fj blend", time.perf_counter() - t)
    ln = CAP["lanes"].copy()
    ln["Detector"] = ln.Detector.astype(int)
    on = CAP["on"]
    # ---- D lanes: extra over the package lane step = gather 2 x (229 features + 7 probs) per pair, min / max
    pairs = CAP.get("lane_pairs")
    npair = 0 if pairs is None else len(pairs)
    t = time.perf_counter()
    Ff = np.random.default_rng(0).normal(size=(len(fn), 229 + 7)).astype(np.float32)
    if npair:
        di = {d: i for i, d in enumerate(fn.Detector)}
        ia = np.array([di.get(int(x), 0) for x in pairs.da])
        ib = np.array([di.get(int(x), 0) for x in pairs.db])
        A_, B_ = Ff[ia], Ff[ib]
        _ = np.hstack([np.zeros((npair, 37), np.float32), np.fmin(A_, B_), np.fmax(A_, B_)])
    r("D lanes: extra function-feature block", time.perf_counter() - t)
    R["_n_lane_pairs"] = npair
    # ---- stack pick inputs (ln6 + ln7 on this window) + decode
    import a2_features as A2F
    import ln6_pick as L6
    import ln7_stackhealth as L7
    hh, secs = NOMINAL[L]
    day = pd.Timestamp(CAP["t0"]).normalize()
    t0w = str(day + pd.Timedelta(hh))
    A2F.WINDOWS["bench"] = [("w", t0w, secs)]
    L6.CACHE["bench"] = OUT / f"iv_{sig}_{L}.parquet"
    L6.EVR["bench"] = OUT / "no_events_here"                          # ln6 health off (shared with the package)
    top = CAP["top"].copy()
    top["Detector"] = top.Detector.astype(int)
    k = top[["Detector", "phase_pred"]].rename(columns={"phase_pred": "pred_phase"}).merge(
        fn[["Detector"] + [f"p_{c.lower()}" for c in C7]], on="Detector", how="inner")
    k["func"] = np.array(C7, object)[Pb[[list(fn.Detector).index(d) for d in k.Detector]].argmax(1)]
    for i, c in enumerate(C7):
        k[f"P_{c}"] = k[f"p_{c.lower()}"]
    k = k.merge(ln[["Detector", "lanes", "lane_conf", "phase"]], on="Detector", how="left")
    k["lanes"] = k.lanes.where(k.phase.eq(k.pred_phase), "").fillna("")
    k["win"] = "w"
    t = time.perf_counter()
    K6 = L6.work((dev, "bench", k[["Detector", "win", "pred_phase", "func", "lanes"] + [f"P_{c}" for c in C7]]))
    r("stack pick inputs: ln6 span / co-location / track", time.perf_counter() - t)
    t = time.perf_counter()
    H7 = L7.work((dev, "bench", k[["Detector", "win", "pred_phase", "func", "lanes"]]))
    r("stack pick inputs: ln7 stack health", time.perf_counter() - t)
    t = time.perf_counter()
    fr = k[["Detector", "pred_phase", "lanes", "lane_conf"]].copy()
    fr.insert(0, "DeviceId", dev)
    fr["period"], fr["win"], fr["wgroup"] = "bench", "w", L
    nmap = {d: int(((v[0] >= CAP["w0"]) & (v[0] < CAP["w1"])).sum()) for (dv, d), v in on.items()}
    fr["det_n_on"] = fr.Detector.map(nmap).fillna(0).astype(int)
    fr["lanes5g"] = fr.lanes.where(fr.lanes.ne(""), None)
    k6 = ["Detector"]
    x = fr[k6].merge(K6.astype({"Detector": int})[k6 + ["span", "span_peers", "coloc_peers", "track"]], on=k6, how="left")
    if len(H7) and "chi" in H7:
        H = H7.astype({"Detector": int}).copy()
        g = H.groupby(["clus"]).chi
        lo1 = g.transform("min")
        second = g.transform(lambda v: np.sort(v.dropna().to_numpy())[1] if v.notna().sum() > 1 else np.nan)
        H["chi_p"] = np.where(H.chi.eq(lo1) & H.chi.notna(), second, lo1)
        H["unh2"] = (H.chi >= 10.0) & (H.chi >= 3.0 * (H.chi_p + 0.5))
        y = fr[k6].merge(H[k6 + ["unh2"]].drop_duplicates(k6), on=k6, how="left").unh2.eq(True).to_numpy()
    else:
        y = np.zeros(len(fr), bool)
    fr["pk_unhealthy"] = y
    fr["pk_span"] = x.span.eq(True).to_numpy()
    fr["pk_span_peers"] = x.span_peers.fillna("").to_numpy()
    fr["pk_coloc_peers"] = x.coloc_peers.fillna("").to_numpy()
    fr["pk_track"] = x.track.to_numpy(float)
    fr = fr.reset_index(drop=True)
    r("stack pick inputs: attach", time.perf_counter() - t)
    rows = np.array([list(fn.Detector).index(d) for d in fr.Detector])
    Ptf, Pnf, Pbf = Pt[rows], PnF[rows], Pb[rows]
    # ---- context stacker
    import s67_decider as S67
    sys.modules.setdefault("s59_step6", types.SimpleNamespace())
    hc_ = CAP["health"].rename(columns={"detector": "Detector"}) if CAP.get("health") is not None else None
    Hm = np.full((len(fr), 16), np.nan, np.float32)
    if hc_ is not None and len(hc_):
        hs = dict(zip(hc_.Detector.astype(int), hc_.health_score.astype(float)))
        Hm[:, 0] = fr.Detector.map(hs).to_numpy(float)
    lph = CAP["lane_ph"]
    t = time.perf_counter()
    nl = fr.pred_phase.map(dict(zip(lph.phase, lph.n_lanes))) if len(lph) else pd.Series(np.nan, index=fr.index)
    nlc = fr.pred_phase.map(dict(zip(lph.phase, lph.n_lanes_conf))) if len(lph) else nl
    S67._CTX.clear()
    S67._CTX["H"], S67._CTX["ln"] = Hm, np.column_stack([nl, nlc, fr.lane_conf]).astype(float)
    X = np.hstack([S67.stack_X(fr, Ptf, Pnf), S67.ctx_X({"fr": fr}, Pbf)])
    r("stacker: features (probs + context)", time.perf_counter() - t)
    if stk_holder.get("b") is None:
        from lgbm_numpy import NumpyBooster
        t = time.perf_counter()
        stk_holder["b"] = [NumpyBooster(OUT / f"stacker_shape_s{s}.txt") for s in range(3)]
        r("load: stacker boosters", time.perf_counter() - t)
    nX = stk_holder["b"][0].n_features
    assert X.shape[1] == nX, (X.shape, nX)
    t = time.perf_counter()
    Ps = np.mean([b.predict(X) for b in stk_holder["b"]], 0)
    r("stacker: 3-seed booster", time.perf_counter() - t)
    t = time.perf_counter()
    for i, c in enumerate(C7):
        fr[f"P_{c}"] = Pbf[:, i]          # real blend probabilities (the synthetic stacker output is timing-only)
    pred = S67.decode_x(fr, Pbf, gate=GATE, lconf=fr.lane_conf.to_numpy(float))
    r("decode: stack pick + per-lane decode (gate .9)", time.perf_counter() - t)
    R["_twin_decode"] = "not run (5 / 10-min samples only)"
    # ---- night speed
    import sp1_night_speed as SP
    sb = CAP.get("setback")
    sbd = dict(zip(sb.Detector.astype(int), sb.distance_ft)) if sb is not None and len(sb) else {}
    lanes_of = {int(d): tuple(int(v) for v in s.split(",")) if isinstance(s, str) and s else ()
                for d, s in zip(fr.Detector, fr.lanes)}
    detl = [dict(det=int(d), phase=p, function=C7[int(c)], lanes=lanes_of.get(int(d), ()), setback_ft=sbd.get(int(d), np.nan))
            for d, p, c in zip(fr.Detector, fr.pred_phase, pred)]
    ivs = pd.read_parquet(OUT / f"iv_{sig}_{L}.parquet")
    ivs["t"] = ivs.t_on.astype("datetime64[us]").astype("int64") / 1e6
    ont = {int(d): g.t.to_numpy() for d, g in ivs.groupby("Detector")}
    offt = {int(d): (g.t + g.dur).to_numpy() for d, g in ivs.groupby("Detector")}
    gr = pd.read_parquet(OUT / f"green_{sig}_{L}.parquet")
    greens = {int(p): (g["gs"].to_numpy(), g["ge"].to_numpy()) for p, g in gr.groupby("p")}
    t = time.perf_counter()
    SP.night_speed(CAP["w0"], CAP["w1"], detl, ont, offt, greens)
    r("night speed (as is: daytime sample)", time.perf_counter() - t)
    t = time.perf_counter()
    dr, _ = SP.night_speed(CAP["w0"], CAP["w1"], detl, ont, offt, greens, p={"night_start": 0.0, "night_end": 24.0})
    r("night speed (forced night, worst case)", time.perf_counter() - t)
    R["_n_advance"] = sum(1 for x in detl if x["function"] == "Advance")
    return R


def prep_intervals(sig, L):
    """research-side inputs that production already holds in its DuckDB tables (untimed): ON intervals, greens."""
    import duckdb
    f = OUT / f"iv_{sig}_{L}.parquet"
    if f.exists():
        return
    con = duckdb.connect()
    con.execute(f"SET threads={THREADS}")
    src = ev_file(sig, L)
    con.execute(f"""COPY (WITH e AS (SELECT DISTINCT DeviceId, Timestamp ts, EventId, Parameter ch FROM '{src}'
                                 WHERE EventId IN (81,82) AND Parameter <= 64),
        d AS (SELECT *, LEAD(ts) OVER w nts, LEAD(EventId) OVER w nev FROM e
              WINDOW w AS (PARTITION BY ch ORDER BY ts, CASE WHEN EventId=82 THEN 0 ELSE 1 END))
        SELECT DeviceId, ch::INT AS Detector, ts AS t_on, epoch_ms(nts)/1000.0 - epoch_ms(ts)/1000.0 AS dur
        FROM d WHERE EventId=82 AND nev=81) TO '{f}'""")
    con.execute(f"""COPY (WITH e AS (SELECT DISTINCT Timestamp ts, EventId, Parameter p FROM '{src}'
                                 WHERE EventId IN (1,8,10) AND Parameter BETWEEN 1 AND 16),
        d AS (SELECT *, LEAD(ts) OVER (PARTITION BY p ORDER BY ts) nts FROM e)
        SELECT p::INT p, epoch_ms(ts)/1000.0 gs, epoch_ms(nts)/1000.0 ge FROM d WHERE EventId=1 AND nts IS NOT NULL)
        TO '{OUT / f'green_{sig}_{L}.parquet'}'""")
    con.close()


def cmd_one(sig, L, out):
    t_imp0 = time.perf_counter()
    import numpy as np  # noqa: F401
    import pandas as pd
    import duckdb  # noqa: F401
    import onnxruntime  # noqa: F401
    sys.path.insert(0, str(PKG))
    import predict as P
    t_imp = time.perf_counter() - t_imp0
    P.set_backend("numpy")
    # research modules: their `common` is the research one; the package's own modules are already bound
    pkg_common = sys.modules.pop("common")
    sys.path.insert(1, str(CODE))
    import rpath  # noqa: F401
    import common as _rc  # noqa: F401
    import s67_decider  # noqa: F401
    import ln6_pick  # noqa: F401
    import ln7_stackhealth  # noqa: F401
    import sp1_night_speed  # noqa: F401
    import a2_features  # noqa: F401
    import atspm_decode  # noqa: F401
    sys.modules["common"] = pkg_common
    sys.path.remove(str(PKG))
    sys.path.insert(0, str(PKG))
    instrument(P)
    orig_tp = P._top_phase

    def tp(ph):
        r = orig_tp(ph)
        CAP["top"] = r
        return r
    P._top_phase = tp
    orig_le = P.load_events

    def le(*a, **k):
        r = orig_le(*a, **k)
        CAP["w0"], CAP["w1"], CAP["t0"] = r[0], r[1], r[2]["t0"]
        return r
    P.load_events = le
    f = str(ev_file(sig, L))
    runs = []
    fjm, stk = {}, {}
    for i in range(4):
        REC.clear()
        t = time.perf_counter()
        res = P.predict(f, threads=THREADS)
        wall = time.perf_counter() - t
        pk = dict(REC)
        t = time.perf_counter()
        ex = research_extras(P, sig, L, fjm, stk, i == 0)
        ex["[incl] research extras total"] = time.perf_counter() - t
        runs.append(dict(predict_wall=wall, package=pk, research=ex))
        log(f"{sig} {L} run {i}: predict {wall:.2f}s, research extras {ex['[incl] research extras total']:.2f}s")
    json.dump(dict(sig=sig, L=L, import_s=t_imp, start_to_import_s=time.perf_counter() - _T_START,
                   detectors=int(len(res)), channels=int(res.Detector.nunique()),
                   cand_phases=int(res.n_candidate_phases.max()), runs=runs), open(out, "w"), indent=1)


def cmd_profile(a):
    OUT.mkdir(parents=True, exist_ok=True)
    lk = locked()
    import duckdb
    for sig in SIGS:
        for L in LENS:
            ids = duckdb.sql(f"select distinct lower(DeviceId) d from '{ev_file(sig, L)}'").df().d
            assert not set(ids) & lk
            prep_intervals(sig, L)
    for sig in SIGS:
        for L in LENS:
            o = OUT / f"r_{sig}_{L}.json"
            if not o.exists() or a.force:
                subprocess.run([sys.executable, "-W", "ignore", __file__, "--one", sig, L, str(o)], check=True)



# ======================================================================== summary
GROUPS = [  # (row, [labels]) -- rows do not overlap; "rest" rows = inclusive total minus its timed children
    ("event load / DuckDB tables", ["events: duckdb connect", "events: load + dedupe", "events: interval / cycle tables",
                                    "events: universe / facts / gate"]),
    ("GRU: interval streams (SQL)", ["GRU: interval streams (SQL)"]),
    ("GRU: raster", ["GRU: raster (render + assemble)"]),
    ("GRU: ONNX inference", ["GRU: ONNX inference"]),
    ("phase features (f1 / f2 / sim / f3)", ["[incl] phase features total"]),
    ("phase trees s0", ["phase trees s0"]), ("phase trees s1", ["phase trees s1"]), ("phase trees s2", ["phase trees s2"]),
    ("GRU mix + softmax", ["phase: GRU mix", "phase: softmax / normalise"]),
    ("joint decoder (features + trees)", ["joint decoder features", "joint decoder trees"]),
    ("function features (frame + expert)", ["function features: frame (shape / sibling / lag)",
                                            "function features: expert (fx)"]),
    ("function trees s0", ["function trees s0"]), ("function trees s1", ["function trees s1"]),
    ("function trees s2", ["function trees s2"]),
    ("lanes (package A; D proxy)", ["lanes: pair cues", "lanes: pair model (3 seeds)", "lanes: decode"]),
    ("health v5", ["health v5 (health_core)"]),
    ("setback (package)", ["setback (package note-41 model)"]),
    ("output assembly", ["output assembly"]),
]
RGROUPS = [
    ("fj raster (15 ch)", ["fj raster (render + assemble 15 ch)"]),
    ("fj ONNX inference", ["fj ONNX inference"]),
    ("D lanes extra block", ["D lanes: extra function-feature block"]),
    ("stack pick inputs (ln6 + ln7)", ["stack pick inputs: ln6 span / co-location / track", "stack pick inputs: ln7 stack health",
                                       "stack pick inputs: attach"]),
    ("stacker (features + 3 seeds)", ["stacker: features (probs + context)", "stacker: 3-seed booster"]),
    ("stack pick + lane decode (gate .9)", ["decode: stack pick + per-lane decode (gate .9)"]),
    ("night speed (forced night)", ["night speed (forced night, worst case)"]),
]


def cmd_summary(a):
    import numpy as np
    out = {}
    for sig in SIGS:
        for L in LENS:
            r = json.load(open(OUT / f"r_{sig}_{L}.json"))
            runs = r["runs"]

            def val(run, labels, part):
                return sum(run[part].get(x, 0.0) for x in labels)
            row = {}
            for nm, labs in GROUPS:
                row[nm] = [float(np.median([val(x, labs, "package") for x in runs[1:]])), val(runs[0], labs, "package")]
            pk_w = float(np.median([x["predict_wall"] for x in runs[1:]]))
            pk_c = runs[0]["predict_wall"]
            # function / phase / post 'rest' (merges, frames) = wall minus every timed row (loads excluded)
            for nm, labs in RGROUPS:
                row[nm] = [float(np.median([val(x, labs, "research") for x in runs[1:]])), val(runs[0], labs, "research")]
            ld = ["load: tree models", "load: lane / setback models", "load: GRU ONNX session"]
            row["model loading (cold only)"] = [0.0, val(runs[0], ld, "package") +
                                                val(runs[0], ["load: fj ONNX session", "load: stacker boosters"], "research")]
            timed_w = sum(v[0] for n, v in row.items() if n in dict(GROUPS))
            timed_c = sum(v[1] for n, v in row.items() if n in dict(GROUPS))
            row["package glue (merges, frames, Python)"] = [pk_w - timed_w,
                                                           pk_c - timed_c - val(runs[0], ld, "package")]
            row["import (cold only)"] = [0.0, r["import_s"]]
            row["TOTAL"] = [sum(v[0] for n, v in row.items()), sum(v[1] for n, v in row.items())]
            out[f"{sig}|{L}"] = dict(channels=r["channels"], cand=r["cand_phases"], predict_warm=pk_w, predict_cold=pk_c,
                                     rows={k: [round(v[0], 3), round(v[1], 3)] for k, v in row.items()},
                                     n_lane_pairs=runs[0]["research"]["_n_lane_pairs"])
    json.dump(out, open(OUT / "prof.json", "w"), indent=1)
    keys = list(out)
    print("component".ljust(42) + "".join(k.replace("busiest_ev", "bev").replace("typical", "typ").replace("busiest", "bus")
                                          .ljust(13) for k in keys))
    for nm in out[keys[0]]["rows"]:
        print(nm[:41].ljust(42) + "".join(f"{out[k]['rows'][nm][0]:5.2f}/{out[k]['rows'][nm][1]:5.2f}  " for k in keys))
    print("channels".ljust(42) + "".join(f"{out[k]['channels']:<13}" for k in keys))


# ======================================================================== ideas (speed headroom, measured; no model change)
class FlatBooster:
    """All trees of a LightGBM text model traversed at once (one numpy step per tree level, not per tree)."""

    def __init__(self, nb):
        import numpy as np
        T = nb.trees
        n = len(T)
        mx = max((len(t.feat) if t.feat is not None else 0) for t in T) or 1
        ml = max(len(t.leaf) for t in T)
        self.feat = np.zeros((n, mx), np.int64)
        self.thr = np.full((n, mx), np.inf)
        self.dl = np.zeros((n, mx), bool)
        self.miss = np.zeros((n, mx), np.int64)
        self.left = np.full((n, mx), -1, np.int64)
        self.right = np.full((n, mx), -1, np.int64)
        self.leaf = np.zeros((n, ml))
        self.stump = np.zeros(n, bool)
        for i, t in enumerate(T):
            self.leaf[i, :len(t.leaf)] = t.leaf
            if t.feat is None:
                self.stump[i] = True
                continue
            m = len(t.feat)
            self.feat[i, :m], self.thr[i, :m], self.dl[i, :m] = t.feat, t.thr, t.default_left
            self.miss[i, :m], self.left[i, :m], self.right[i, :m] = t.miss, t.left, t.right
        self.nb, self.n = nb, n

    def predict(self, X):
        import numpy as np
        nb = self.nb
        if hasattr(X, "to_numpy"):
            X = X.to_numpy(dtype=np.float64, na_value=np.nan)
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        R = X.shape[0]
        ti = np.arange(self.n)[None, :]
        node = np.where(self.stump[None, :], -1, 0) * np.ones((R, 1), np.int64)
        rows = np.arange(R)[:, None]
        act = node >= 0
        while act.any():
            nd = np.where(act, node, 0)
            f = self.feat[ti, nd]
            v = X[rows, f]
            mi = self.miss[ti, nd]
            isn = np.isnan(v)
            missing = ((mi == 2) & isn) | ((mi == 1) & (isn | (np.abs(v) <= 1e-35)))
            v = np.where(isn & (mi != 2), 0.0, v)
            gl = np.where(missing, self.dl[ti, nd], v <= self.thr[ti, nd])
            nxt = np.where(gl, self.left[ti, nd], self.right[ti, nd])
            node = np.where(act, nxt, node)
            act = node >= 0
        vals = self.leaf[ti, ~node]                                   # [R, n_trees]
        raw = vals.reshape(R, -1, nb.k).sum(1) if nb.k > 1 else vals.sum(1, keepdims=True)
        if nb.average_output:
            raw /= max(1, self.n // nb.k)
        name = nb.objective[0] if nb.objective else ""
        if name == "binary":
            raw = 1.0 / (1.0 + np.exp(-nb.sigmoid * raw))
        elif name in ("multiclass", "softmax"):
            e = np.exp(raw - raw.max(1, keepdims=True))
            raw = e / e.sum(1, keepdims=True)
        return raw[:, 0] if nb.k == 1 else raw



def lgbm_to_onnx(nb, path):
    """A numpy-booster (LightGBM text model) -> ONNX ai.onnx.ml TreeEnsembleRegressor (raw scores, double thresholds / inputs).
    Missing values: LightGBM 'nan' type -> default direction; 'none' type (NaN read as 0) -> 0 <= threshold."""
    import numpy as np
    import onnx
    from onnx import helper, TensorProto
    a = {k: [] for k in ("tid", "nid", "fid", "val", "mode", "t", "f", "miss")}
    tw = {k: [] for k in ("tid", "nid", "tgt", "w")}
    for ti, t in enumerate(nb.trees):
        k = ti % nb.k
        if t.feat is None:
            a["tid"].append(ti); a["nid"].append(0); a["fid"].append(0); a["val"].append(0.0); a["mode"].append("LEAF")
            a["t"].append(0); a["f"].append(0); a["miss"].append(0)
            tw["tid"].append(ti); tw["nid"].append(0); tw["tgt"].append(k); tw["w"].append(float(t.leaf[0]))
            continue
        m = len(t.feat)
        nid = lambda c: int(c) if c >= 0 else m + int(~c)                       # noqa: E731
        for i in range(m):
            mt = int(t.miss[i])
            tr = bool(t.default_left[i]) if mt == 2 else (bool(t.thr[i] >= 0) if mt == 0 else bool(t.default_left[i]))
            a["tid"].append(ti); a["nid"].append(i); a["fid"].append(int(t.feat[i])); a["val"].append(float(t.thr[i]))
            a["mode"].append("BRANCH_LEQ"); a["t"].append(nid(t.left[i])); a["f"].append(nid(t.right[i]))
            a["miss"].append(int(tr))
        for j, v in enumerate(t.leaf):
            a["tid"].append(ti); a["nid"].append(m + j); a["fid"].append(0); a["val"].append(0.0); a["mode"].append("LEAF")
            a["t"].append(0); a["f"].append(0); a["miss"].append(0)
            tw["tid"].append(ti); tw["nid"].append(m + j); tw["tgt"].append(k); tw["w"].append(float(v))
    node = helper.make_node("TreeEnsembleRegressor", ["X"], ["Y"], domain="ai.onnx.ml", n_targets=nb.k,
                            aggregate_function="SUM", post_transform="NONE",
                            nodes_treeids=a["tid"], nodes_nodeids=a["nid"], nodes_featureids=a["fid"],
                            nodes_values_as_tensor=helper.make_tensor("nv", TensorProto.DOUBLE, [len(a["val"])], a["val"]),
                            nodes_modes=a["mode"], nodes_truenodeids=a["t"],
                            nodes_falsenodeids=a["f"], nodes_missing_value_tracks_true=a["miss"],
                            target_treeids=tw["tid"], target_nodeids=tw["nid"], target_ids=tw["tgt"],
                            target_weights_as_tensor=helper.make_tensor("tw", TensorProto.DOUBLE, [len(tw["w"])], tw["w"]))
    g = helper.make_graph([node], "lgbm", [helper.make_tensor_value_info("X", TensorProto.DOUBLE, [None, nb.n_features])],
                          [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, nb.k])])
    mdl = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("ai.onnx.ml", 3)], ir_version=9)
    onnx.save(mdl, str(path))


class OnnxTrees:
    def __init__(self, nb, path):
        import onnxruntime as ort
        lgbm_to_onnx(nb, path)
        so = ort.SessionOptions()
        so.intra_op_num_threads = THREADS
        self.s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.nb = nb

    def predict(self, X):
        import numpy as np
        nb = self.nb
        if hasattr(X, "to_numpy"):
            X = X.to_numpy(dtype=np.float64, na_value=np.nan)
        raw = self.s.run(None, {"X": np.ascontiguousarray(X, np.float64)})[0].astype(np.float64)
        if nb.average_output:
            raw /= max(1, len(nb.trees) // nb.k)
        name = nb.objective[0] if nb.objective else ""
        if name == "binary":
            raw = 1.0 / (1.0 + np.exp(-nb.sigmoid * raw))
        elif name in ("multiclass", "softmax"):
            e = np.exp(raw - raw.max(1, keepdims=True))
            raw = e / e.sum(1, keepdims=True)
        return raw[:, 0] if nb.k == 1 else raw


def _best_of(f, n=3):
    ts, r = [], None
    for _ in range(n):
        t = time.perf_counter()
        r = f()
        ts.append(time.perf_counter() - t)
    return sorted(ts)[len(ts) // 2], r


def cmd_ideas(a):
    import numpy as np
    import pandas as pd
    global CAPTURE_X
    sys.path.insert(0, str(PKG))
    import predict as P
    import gru_blend as gb
    import gru_input as gi
    import gru_onnx as go
    P.set_backend("numpy")
    instrument(P)
    CAPTURE_X = True
    from onnxruntime.quantization import quantize_dynamic, QuantType
    qg, qf = OUT / "gru_int8.onnx", OUT / "fj_f0_int8.onnx"
    if not qg.exists():
        quantize_dynamic(str(PKG / "weights" / "gru.onnx"), str(qg), weight_type=QuantType.QInt8)
    if not qf.exists():
        quantize_dynamic(str(FJ_ONNX), str(qf), weight_type=QuantType.QInt8)
    res = {}
    for sig in ("typical", "busiest"):
        L = "h3"
        CAP.clear()
        P.predict(str(ev_file(sig, L)), threads=THREADS)
        if a.trees_only:
            R = {}
            for lab in ("phase trees s0", "function trees s0", "joint decoder trees"):
                X = CAP[f"X:{lab}"]
                nb = next(b for k, b in BCACHE.items() if _booster_label(k) == lab)
                t_old, y0 = _best_of(lambda: nb.predict(X))
                ot = OnnxTrees(nb, OUT / f"trees_{lab.replace(' ', '_')}.onnx")
                t_ox, y2 = _best_of(lambda: ot.predict(X))
                y0, y2 = np.asarray(y0), np.asarray(y2)
                R[lab] = dict(rows=int(len(X)), trees=len(nb.trees), numpy_s=round(t_old, 4), onnx_s=round(t_ox, 4),
                              onnx_max_abs=float(np.abs(y0 - y2).max()),
                              onnx_argmax_same=float((y0.argmax(1) == y2.argmax(1)).mean()) if y0.ndim == 2 else None)
            res[f"{sig}|{L}"] = R
            log(f"{sig} {L}: {json.dumps(R)}")
            continue
        R = {}
        # ---- 1. flat (all-trees-at-once) numpy boosters: parity + time
        for lab in ("phase trees s0", "function trees s0", "joint decoder trees"):
            X = CAP[f"X:{lab}"]
            nb = next(b for k, b in BCACHE.items() if _booster_label(k) == lab)
            fb = FlatBooster(nb)
            t_old, y0 = _best_of(lambda: nb.predict(X))
            t_new, y1 = _best_of(lambda: fb.predict(X))
            ot = OnnxTrees(nb, OUT / f"trees_{lab.replace(' ', '_')}.onnx")
            t_ox, y2 = _best_of(lambda: ot.predict(X))
            y0, y2 = np.asarray(y0), np.asarray(y2)
            same = (y0.argmax(1) == y2.argmax(1)).mean() if y0.ndim == 2 else float("nan")
            R[lab] = dict(rows=int(len(X)), trees=fb.n, numpy_s=round(t_old, 4), flat_s=round(t_new, 4),
                          max_abs=float(np.abs(y0 - np.asarray(y1)).max()), onnx_s=round(t_ox, 4),
                          onnx_max_abs=float(np.abs(y0 - y2).max()), onnx_argmax_same=float(same))
        # ---- 2. GRU: candidate pruning by the trees' own probability (pairs kept, time, blended top-phase agreement)
        df, gru, wt, mixed = CAP["mix"]
        z = next(iter(CAP["streams"].values()))
        (_, t0, t1), kw = CAP["stream_args"]
        dets = [int(c) for c in z["det_ch"]]
        cand = np.asarray(z["cand"], int)
        m = gb._model(PKG / "weights" / "gru.onnx", 512)
        pieces = [(x - t0, y - t0) for x, y in kw["windows"]]
        Xs = []
        for pa, pb in pieces:
            T = max(int((pb - pa) // 1000), 8)
            det, ph, sg, na = gi.render(z, int(pa), dets, T)
            Xs.append(gi.assemble(det, ph, sg))
        t_full, lf = _best_of(lambda: [m.logits(x) for x in Xs])
        pm = {(int(d), int(c)): v for d, c, v in zip(df.Detector, df.cand_phase, df.p0)}
        pr = {"full_s": round(t_full, 3), "pairs": int(len(dets) * len(cand))}
        grp = [df.DeviceId.to_numpy(), df.Detector.to_numpy()]
        base_top = pd.Series(mixed, index=df.index).groupby(grp).idxmax()
        allp = [(d, int(c)) for d in dets for c in cand]
        for thr in (0.001, 0.01, 0.05):
            keep = np.array([pm.get(k, 1.0) >= thr for k in allp])
            t_k, _ = _best_of(lambda: [m.logits(x[keep]) for x in Xs])
            kk = {k for k, k_ in zip(allp, keep) if k_}
            g2 = gru[[(int(d), int(c)) in kk for d, c in zip(gru.Detector, gru.cand_phase)]].copy()
            g2["p_gru"] = g2.p_gru / g2.groupby("Detector").p_gru.transform("sum")
            mx2 = gb.mix(df, "p0", g2, wt)
            top2 = pd.Series(mx2, index=df.index).groupby(grp).idxmax()
            pr[f"thr{thr}"] = dict(kept=round(float(keep.mean()), 3), s=round(t_k, 3),
                                   blend_top_same=float((top2 == base_top).mean()),
                                   max_abs_blend=float(np.abs(mx2 - mixed).max()))
        R["gru_prune"] = pr
        # ---- 3. int8 dynamic quantisation (GRU, fj): time + output difference on the same inputs
        mq = go.GruPhaseModel(qg, pair_batch=512)
        t_q, lq = _best_of(lambda: [mq.logits(x) for x in Xs])

        def lp(ls):
            v = np.mean([l.reshape(len(dets), len(cand)) for l in ls], 0)
            v = v - v.max(1, keepdims=True)
            e = np.exp(v)
            return e / e.sum(1, keepdims=True)
        R["gru_int8"] = dict(fp32_s=round(t_full, 3), int8_s=round(t_q, 3),
                             max_abs_prob=float(np.abs(lp(lq) - lp(lf)).max()),
                             top_same=float((lp(lq).argmax(1) == lp(lf).argmax(1)).mean()))
        f32, f8 = FjOnnx(), FjOnnx(qf)
        t32, (pa32, _) = _best_of(lambda: fj_signal(f32, z, t1 - t0, pieces))
        t8, (pa8, _) = _best_of(lambda: fj_signal(f8, z, t1 - t0, pieces))
        R["fj_int8"] = dict(fp32_s=round(t32, 3), int8_s=round(t8, 3), max_abs_prob=float(np.abs(pa32 - pa8).max()),
                            top_same=float((pa32.argmax(1) == pa8.argmax(1)).mean()))
        # ---- 4. fewer pieces at 3 h (K = 2 instead of 4): time only
        t2, _ = _best_of(lambda: [m.logits(x) for x in Xs[::2]])
        R["gru_K2_s"] = round(t2, 3)
        res[f"{sig}|{L}"] = R
        log(f"{sig} {L}: {json.dumps(R)}")
    json.dump(res, open(OUT / ("ideas_trees.json" if a.trees_only else "ideas.json"), "w"), indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        cmd_one(*sys.argv[2:5])
    else:
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd", choices=["export", "pb", "profile", "summary", "ideas"])
        ap.add_argument("--force", action="store_true")
        ap.add_argument("--trees-only", action="store_true")
        a = ap.parse_args()
        {"export": cmd_export, "pb": cmd_pb, "profile": cmd_profile, "summary": cmd_summary, "ideas": cmd_ideas}[a.cmd](a)
