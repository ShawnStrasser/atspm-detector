"""The function half (notes 57 / 58 / 59 / 62 / 67 / 69 / 111 / 114): one row per detector in, its function out.

    trees (229 features, one seed)         -> P_trees      `weights/function/`
    network siba head (ONNX, 3 members)    -> P_net        `weights/funcnet/`   (funcnet.py; same pass as the phase head)
    lanes, model D (cues + function block) -> lanes, lane confidence, n_lanes per phase   (lanes.py; samples >= 30 min)
    pick inputs (span / co-location / track / stack size) -> context columns            (pick.py)
    context stacker (one seed)             -> P_final      `weights/stacker/`  (stacker.py)
    per-lane decode, lane-confidence gate .9, stack pick (stack loser -> best non-ATSPM for Advance / Presence)
    short samples (< 30 min): no lanes; twin decode (Count / Yellow_Red co-actuation twins, threshold .4)

Every input is the hi-res log and the model's own predicted phase; phase numbers are grouping keys only.  No health
output is read here (note 114): health is computed after classification, as a separate output.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import atspm_decode as AD
from . import lanes as lane_mod
from . import pick as PK
from . import trees_onnx
from .stacker import Stacker
from .common import read_json

C7 = ["Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other"]
LANE_MIN_MINUTES = 30            # the lane model and the pick inputs exist from 30 min (training)
GATE = 0.9                       # note 67b: the one-per-lane rule binds only detectors with lane confidence >= .9
TWIN_THR = 0.4                   # note 62
_CACHE: dict = {}


def _cached(key, make):
    if key not in _CACHE:
        _CACHE[key] = make()
    return _CACHE[key]


def models(model_dir: Path) -> dict:
    md = Path(model_dir)

    def make():
        fmeta = read_json(md / "function" / "function229.json")
        return {"fmeta": fmeta,
                "ftrees": trees_onnx.Bag([md / "function" / f for f in fmeta["onnx_files"]]),
                "lanes": lane_mod.PairModel(md / "lanes"),
                "stacker": Stacker(md / "stacker")}
    return _cached(("models", str(md.resolve())), make)


def net_model(model_dir: Path):
    """the function network (every member present), loaded on first use and cached."""
    md = Path(model_dir)

    def make():
        from . import funcnet
        return funcnet.FuncNet(md / "funcnet")
    return _cached(("net", str(md.resolve())), make)


def _on_window(on_iv: dict, dev: str):
    """ON times (seconds, local clock) and their hour of day for every detector of one signal inside the sample window
    (`on_iv` = predict's per-call ON table {DeviceId: {det: (t_on, t_off)}}, fetched once)."""
    on, hour = {}, {}
    for d, (t, _) in on_iv.get(dev, {}).items():
        on[int(d)] = t
        hour[int(d)] = np.floor(np.mod(t, 86400.0) / 3600.0).astype(int)
    return on, hour


def run(con, top: pd.DataFrame, model_dir: Path, b0: float, b1: float, streams: dict | None, pieces_rel,
        log=lambda m: None, inject: dict | None = None, net_keep: dict | None = None, on_iv: dict | None = None):
    """top: the function design frame (one row per detector, all signals; DeviceId, Detector, pred_phase, det_n_on and the
    229 features).  -> (detector table with P_trees / P_net / P_final, lanes, function_pred ..., phase lane table).
    `inject` (research parity harness only): {"Pt" / "Pn": f(dev, dets) -> probs or None, "lane_probs": {dev: {(a, b): p}},
    "lane_decoder": Decoder, "stacker": object with .predict(minutes, Pt, Pn, fr), "capture": dict}.
    `net_keep` (siba candidate filter, ON by default since note 84): {DeviceId: {detector: set of candidate phases}}."""
    inject = inject or {}
    M = models(model_dir)
    # the function network runs on the caller's streams (always passed by predict since note 120); a detector it gives
    # no answer for (or a signal without streams, not seen in practice) carries the trees' probabilities in the net
    # columns, exactly as before -- one stacker for every case
    net = net_model(model_dir) if streams and pieces_rel else None
    fm = M["fmeta"]
    feats = fm["features"]
    for c in feats:
        if c not in top.columns:
            top[c] = np.nan
    top = top.sort_values(["DeviceId", "Detector"]).reset_index(drop=True)
    Pt = M["ftrees"].predict(top[feats])
    secs = max(b1 - b0, 1.0)
    minutes = secs / 60.0
    lane_regime = round(minutes) >= LANE_MIN_MINUTES
    det_parts, ph_parts = [], []
    for dev, g in top.groupby("DeviceId", sort=True):
        ix = g.index.to_numpy()
        dets = g.Detector.astype(int).to_numpy()
        pt = Pt[ix]
        if inject.get("Pt") is not None:
            v = inject["Pt"](dev, dets)
            pt = np.where(np.isnan(v), pt, v) if v is not None else pt
        # ---- function network
        pn = np.full_like(pt, np.nan)
        z = (streams or {}).get(dev) if net is not None else None
        if z is not None and pieces_rel:
            zd = [int(c) for c in z["det_ch"]]
            P = net.probs(z, zd, pieces_rel, keep=None if net_keep is None else net_keep.get(dev, {}))
            pos = {d: i for i, d in enumerate(zd)}
            for j, d in enumerate(dets):
                if d in pos and np.isfinite(P[pos[d]]).all():
                    pn[j] = P[pos[d]]
        if inject.get("Pn") is not None:
            v = inject["Pn"](dev, dets)
            pn = np.where(np.isnan(v), pn, v) if v is not None else pn
        pn = np.where(np.isnan(pn), pt, pn)
        func_t = np.array(C7, object)[pt.argmax(1)]
        pred_ph = g.pred_phase.to_numpy(float)
        n_on_frame = g.det_n_on.to_numpy(float)
        on0, hour0 = _on_window(on_iv or {}, dev)
        e0 = np.zeros(0)                       # the frame's detectors only (as the research windows)
        on = {int(d): on0.get(int(d), e0) for d in dets}
        hour = {int(d): hour0.get(int(d), np.zeros(0, int)) for d in dets}
        groups = {}
        for d, p in zip(dets, pred_ph):
            if np.isfinite(p):
                groups.setdefault(p, []).append(int(d))
        fr = pd.DataFrame({"Detector": dets, "pred_phase": pred_ph, "det_n_on": n_on_frame})
        fr["lanes5g"], fr["lanes_out"] = None, ""
        for c in ("phase_n_lanes", "phase_n_lanes_conf", "lane_conf", "pk_track"):
            fr[c] = np.nan
        fr["n_lanes_spanned"] = 0
        fr["pk_span"], fr["stack_n"] = False, np.nan
        fr["pk_span_peers"], fr["pk_coloc_peers"] = "", ""
        ph_t = pd.DataFrame()
        if lane_regime:
            # ---- lanes D (on the trees' function: the lane model was trained on it)
            Fm = np.hstack([g[feats].to_numpy(np.float64, na_value=np.nan), pt]).astype(np.float32)
            fmat = {(dev, int(d)): Fm[j] for j, d in enumerate(dets)}
            pr = pd.DataFrame({"DeviceId": dev, "Detector": dets, "phase_use": pred_ph, "func_use": func_t})
            pr["phase_use"] = pr.phase_use.where(np.isfinite(pred_ph), None)
            ont = {(dev, int(d)): (on[int(d)], hour[int(d)]) for d in on}
            try:
                ph_t, det_t = lane_mod.lanes(ont, pr, M["lanes"], b0, b1, fmat, inject.get("lane_probs"),
                                             inject.get("lane_decoder"))
            except Exception as exc:                                      # pragma: no cover
                log(f"lanes unavailable for {dev} ({type(exc).__name__}: {exc})")
                ph_t, det_t = pd.DataFrame(), pd.DataFrame()
            if len(det_t):
                dt = det_t.set_index(det_t.Detector.astype(int))
                lt = dt.reindex(dets)
                ok = lt.lanes.fillna("").ne("") & (lt.phase.astype(float).to_numpy() == pred_ph)
                fr["lanes5g"] = np.where(ok, lt.lanes.to_numpy(object), None)
                fr["lanes_out"] = lt.lanes.fillna("").to_numpy(object)
                fr["n_lanes_spanned"] = lt.n_lanes_spanned.fillna(0).astype(int).to_numpy()
                fr["lane_conf"] = lt.lane_conf.astype(float).to_numpy()
                if len(ph_t):
                    pmap = ph_t.set_index(ph_t.phase.astype(float))
                    lp = lt.phase.astype(float)
                    fr["phase_n_lanes"] = lp.map(pmap.n_lanes).astype(float).to_numpy()
                    fr["phase_n_lanes_conf"] = lp.map(pmap.n_lanes_conf).astype(float).to_numpy()
            # ---- pick inputs (ln6 span / co-location / track, ln7 stack size) on the trees' function; one pair matrix
            #      per phase group, shared by span_feats and stack_sizes
            cls = dict(zip(dets.tolist(), func_t))
            lsets = {int(d): (frozenset(int(v) for v in s.split(",")) if isinstance(s, str) and s else frozenset())
                     for d, s in zip(dets, fr.lanes5g)}
            sp, co, tr, sn = {}, {}, {}, {}
            for _, gd in groups.items():
                gg = PK.order_dets(on, [d for d in gd if len(on[d]) >= PK.MIN_ON], b0)
                if len(gg) >= 2:
                    M_, L_ = PK.pair_mats(on, gg, secs)
                    a_, b_ = PK.span_feats(on, gg, secs, M_, L_)
                    sp.update(a_)
                    co.update(b_)
                    tr.update(PK.track_feats(on, gg, cls, lsets, b0, secs))
                    sn.update(PK.stack_sizes(on, gg, M_))
            fr["pk_span"] = [bool(sp.get(int(d), (False,))[0]) for d in dets]
            fr["pk_span_peers"] = [sp.get(int(d), (False, 0.0, ""))[2] for d in dets]
            fr["pk_coloc_peers"] = [co.get(int(d), "") for d in dets]
            fr["pk_track"] = [tr.get(int(d), np.nan) for d in dets]
            fr["stack_n"] = [float(sn.get(int(d), np.nan)) for d in dets]
        # ---- context stacker -> final probabilities
        Ps = (inject["stacker"].predict(minutes, pt, pn, fr) if inject.get("stacker") is not None
              else M["stacker"].predict(minutes, pt, pn, fr))
        lanes_g = [s if (isinstance(s, str) and s and not (c < GATE)) else None
                   for s, c in zip(fr.lanes5g, fr.lane_conf)]
        pred = AD.decode_signal(Ps, list(pred_ph), n_on_frame, lanes_g, dets, fr.pk_span.to_numpy(), fr.pk_span_peers.to_numpy(object),
                                fr.pk_coloc_peers.to_numpy(object), fr.pk_track.to_numpy(float))
        if not lane_regime:
            pairs = PK.twin_pairs(on, groups, secs, b0)
            tw_det = AD.twin_tokens(pairs, TWIN_THR, 1.0)
            pos = {int(d): j for j, d in enumerate(dets)}
            tw = {pos[a]: {pos[b] for b in bs if b in pos} for a, bs in tw_det.items() if a in pos}
            pred = AD.twin_decode(Ps, pred, tw, n_on_frame >= 5, ("Count", "Yellow_Red"), "cy")
        out = pd.DataFrame({"DeviceId": dev, "Detector": dets})
        out["function_pred"] = np.array(C7, object)[pred]
        out["function_prob"] = Ps[np.arange(len(dets)), pred]
        for i, c in enumerate(C7):
            out[f"p_{c.lower()}"] = Ps[:, i]
        out["function_trees_guess"] = func_t
        out["lanes"] = fr.lanes_out.to_numpy(object)
        out["n_lanes_spanned"] = fr.n_lanes_spanned.to_numpy()
        out["lane_conf"] = fr.lane_conf.to_numpy()
        out["lane_phase"] = pred_ph
        if inject.get("capture") is not None:
            inject["capture"][dev] = dict(fr=fr.copy(), Pt=pt, Pn=pn, Ps=Ps, pred=pred.copy(), func_t=func_t,
                                          ph_t=ph_t.copy() if len(ph_t) else ph_t)
        det_parts.append(out)
        if len(ph_t):
            ph_parts.append(ph_t)
    det = pd.concat(det_parts, ignore_index=True) if det_parts else pd.DataFrame()
    ph = pd.concat(ph_parts, ignore_index=True) if ph_parts else pd.DataFrame()
    return det, ph
