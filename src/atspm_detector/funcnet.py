"""The network ("siba", note 69; three width-32 members): a TCN that reads the 1-s trace of every (detector, candidate
phase) pair.  Its pair graph's logit is the PHASE network (blended with the ranker before the decoder, note 111) and its
head, with SIBLING ATTENTION over the detectors that look like they serve the same candidate, gives the 7-class
FUNCTION probabilities read by the stacker.  One pass serves both tasks.  Phase-anonymous throughout.

Runtime: onnxruntime on the CPU, two graphs per member (`weights/funcnet/manifest.json`):
    pair  x [N, 15, T]  -> s [N] (pair phase logit), z [N, 128] (pair embedding)        run in pair batches
    head  logit [D, K], Z [D, K, 128], act [D]  -> function log-probabilities [D, 7]  one call per piece
A sample is read in pieces (<= 30 min each; 4 evenly spaced pieces above 2 h); per member the log-probabilities and the
pair logits are averaged over pieces, then softmax; members are averaged in probability space.  Only the KEPT pairs
(ranker p >= .01) go through the pair graph; the others get logit -1e4 and a zero embedding, which the head ignores.
The kept-pair raster of a piece is built once and shared by the three members.

Inputs = 15 channels on the 1-s raster: detector occupancy, ON rate, ON / OFF edge impulses; the candidate's green /
yellow / red-clearance / call; how many OTHER phases are green / called, coordination; the candidate's call placed /
dropped impulses; the green and call of its most overlapping partner phase.  No phase or channel number.  Class order:
Advance, Presence, Count, Yellow_Red, Mid, Bike, Other.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .streams import cover, onrate
from .trees_onnx import session
from .common import read_json

C7 = ["Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other"]
DEFAULT_THREADS = int(os.environ.get("DC_NET_THREADS", "4"))
PAIR_BATCH = 64


def _tent(t, w0, T, bw):
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
    i0 = int(np.searchsorted(t, w0 - bw, "left"))
    i1 = int(np.searchsorted(t, w1 + bw, "left"))
    return t[i0:i1]


def pick_partner(ph) -> np.ndarray:
    """Partner of each candidate = the other candidate whose green overlaps it most (-1 = none).  A tie on overlap is
    broken by the partner's own green / call traces (content, never candidate order): tied candidates whose traces are
    identical feed identical partner channels, so the choice among them cannot matter (note 76)."""
    K = ph.shape[0]
    if K < 2:
        return np.full(K, -1, np.int64)
    G = ph[:, 0].astype(np.float64)
    ov = G @ G.T
    np.fill_diagonal(ov, -1.0)
    out = np.full(K, -1, np.int64)
    for k in range(K):
        mx = ov[k].max()
        if not mx > 0:
            continue
        tied = np.flatnonzero(ov[k] >= mx - 1e-9 * max(mx, 1.0))
        out[k] = tied[0] if len(tied) == 1 else max(
            tied, key=lambda q: (ph[q, 0].tobytes(), ph[q, 3].tobytes()))
    return out


def render(z, w0, dets, T, bw=1000):
    """-> det [D,4,T], ph [K,6,T], sig [3,T], nact [D], partner [K] for one piece (relative ms w0, T bins)."""
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
    partner = pick_partner(ph)
    chpos = {int(c): i for i, c in enumerate(z["det_ch"])}
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


def render_cached(z, w0, dets, T, bw=1000):
    """`render`, memoised on the interval bundle `z` per (piece, detector list)."""
    key = (int(w0), int(T), int(bw), tuple(int(d) for d in dets))
    cache = z.setdefault("_r15", {})
    r = cache.get(key)
    if r is None:
        r = cache[key] = render(z, w0, dets, T, bw)
    return r


def assemble_kept(det, ph, sig, partner, di, ki):
    """The [n, 15, T] raster of the pairs (di[i], ki[i]) in the channel order of the manifest."""
    K = ph.shape[0]
    x = np.empty((len(di), 15, det.shape[2]), dtype=np.float32)
    x[:, 0:2] = det[di, 0:2]
    x[:, 2:6] = ph[ki, 0:4]
    x[:, 6] = (sig[0][None] - ph[ki, 0]) / 2.0
    x[:, 7] = (sig[1][None] - ph[ki, 3]) / float(max(K, 2) - 1)
    x[:, 8] = sig[2][None]
    x[:, 9:11] = det[di, 2:4]
    x[:, 11:13] = ph[ki, 4:6]
    ok = (partner >= 0).astype(np.float32)
    pk = np.clip(partner, 0, None)
    x[:, 13] = ph[pk, 0][ki] * ok[ki, None]
    x[:, 14] = ph[pk, 3][ki] * ok[ki, None]
    return x


class FuncNet:
    def __init__(self, model_dir, threads: int = DEFAULT_THREADS, pair_batch: int = PAIR_BATCH):
        md = Path(model_dir)
        self.meta = read_json(md / "manifest.json")
        assert self.meta["classes"] == C7
        # every member in manifest order: the stacker was trained on the 3-member mean (note 120: no 'single' fallback)
        mem = self.meta["members"]
        miss = [m[k] for m in mem for k in ("pair", "head") if not (md / m[k]).exists()]
        if miss:
            raise FileNotFoundError(f"network graphs missing in {md}: {miss}")
        self.member_tags = [m["tag"] for m in mem]
        self.members = [(session(md / m["pair"], threads, arena=False), session(md / m["head"], threads, arena=False))
                        for m in mem]
        self.pair_batch = int(pair_batch)

    def _run(self, member, x, idx, D, K, act):
        """one member on one piece's kept-pair raster -> (function log-probs [D, 7], pair phase logits [D, K])."""
        pair, head = member
        s = np.full(D * K, -1e4, np.float32)
        Z = np.zeros((D * K, 128), np.float32)
        bp = max(1, self.pair_batch)
        for i in range(0, len(idx), bp):
            a, b = pair.run(None, {"x": x[i:i + bp]})
            j = idx[i:i + bp]
            s[j], Z[j] = a.reshape(-1), b
        logit = s.reshape(D, K)
        fl = head.run(None, {"logit": logit, "Z": Z.reshape(D, K, 128), "act": act})[0]
        return fl, logit

    def both(self, z, dets, pieces_rel, keep: dict | None = None):
        """ONE pass of every member over the pieces -> (function probabilities [D, 7] in C7 order, phase probabilities
        [D, K] over z["cand"]).  Phase: per member a softmax over the kept candidates of the piece-mean pair logit (the
        per-piece normalisers cancel), members averaged in probability space; dropped candidates get probability 0.
        `keep` {detector: set of candidate phases} (a detector absent from it keeps every candidate).  Memoised on `z`
        per (members, detectors, pieces, filter): the phase and the function stage share one pass."""
        D = len(dets)
        K = len(z["cand"])
        if D == 0 or K < 1 or not pieces_rel:
            return np.full((D, 7), np.nan), np.full((D, max(K, 0)), np.nan)
        km = None
        if keep is not None:
            cand = [int(c) for c in z["cand"]]
            km = np.array([[c in keep[int(d)] for c in cand] if int(d) in keep else [True] * len(cand)
                           for d in dets], bool)
            km[~km.any(1)] = True                      # never a detector with no candidate
        mkey = (tuple(self.member_tags), tuple(int(d) for d in dets), tuple((int(a), int(b)) for a, b in pieces_rel),
                None if km is None else km.tobytes())
        memo = z.setdefault("_siba", {})
        if mkey in memo:
            return memo[mkey]
        idx = np.arange(D * K) if km is None else np.flatnonzero(km.reshape(-1))
        di, ki = idx // K, idx % K
        n_m = len(self.members)
        acc, lacc = [None] * n_m, [None] * n_m
        for a, b in pieces_rel:                        # piece outer: one kept-pair raster in memory at a time
            T = max(int((b - a) // 1000), 8)
            det, ph, sig, nact, partner = render_cached(z, int(a), dets, T)
            x = assemble_kept(det, ph, sig, partner, di, ki)
            act = (nact > 0).astype(np.float32)
            for mi, m in enumerate(self.members):
                fl, lg = self._run(m, x, idx, D, K, act)
                acc[mi] = fl.astype(np.float64) if acc[mi] is None else acc[mi] + fl
                lacc[mi] = lg.astype(np.float64) if lacc[mi] is None else lacc[mi] + lg
            del x
        out = np.zeros((D, 7))
        pph = np.zeros((D, K))
        valid = np.ones((D, K), bool) if km is None else km
        for mi in range(n_m):
            v = acc[mi] / len(pieces_rel)
            p = np.exp(v - v.max(1, keepdims=True))
            out += p / p.sum(1, keepdims=True)
            lv = np.where(valid, lacc[mi] / len(pieces_rel), -np.inf)
            e = np.where(valid, np.exp(lv - lv.max(1, keepdims=True)), 0.0)
            pph += e / e.sum(1, keepdims=True)
        res = (out / n_m, pph / n_m)
        memo[mkey] = res
        return res

    def probs(self, z, dets, pieces_rel, keep: dict | None = None) -> np.ndarray:
        """-> function probabilities [D, 7] (C7 order); the same pass as `both` (memoised)."""
        return self.both(z, dets, pieces_rel, keep)[0]
