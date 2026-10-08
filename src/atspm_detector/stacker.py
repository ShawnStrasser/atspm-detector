"""The context stacker (notes 67 / 69): the final function probabilities from the trees' and the function network's
probabilities plus hi-res context columns of the same sample (the subset in `stacker.json`).  A small LightGBM (7 leaves,
depth 3, 150 rounds; one seed, several are averaged when present), trained on six-fold out-of-fold probabilities of both base models; run here as ONNX tree ensembles.

Columns (46 built, in this order; research `s67_decider.stack_X` + `ctx_X`).  The model reads the subset named in
`stacker.json` "feature_names" (all 46).  No health input of any kind: the 15 health columns went in note 83 (79b), the
stack-relative unhealthy flag pk_unhealthy in note 114; stack size is structure (hi-res co-location clusters):
  probabilities (20)  log P_trees (7), log P_net (7), log(sample minutes), entropy and top-2 margin of each, argmax agree
  stack (1)           stack_n: size of the detector's hi-res stack (pick.stack_sizes; NaN outside a stack)
  pick (4)            spans lanes (flag), track (corr with the phase's Count total), number of covered / co-located peers
  lanes (6)           lanes spanned, lowest lane number, lanes on the phase among its detectors, phase n_lanes and its
                      confidence, own lane confidence (lane model D; samples >= 30 min only)
  phase-mates (9)     detectors on the predicted phase; best OTHER member's P(Advance / Presence / Count / YR) and own
                      rank of each (blend 0.6 trees + 0.4 net)
  lane-mates (5)      number of lane-mates, their best P(Advance / Presence / Count / YR)
  volume (1)          log(1 + actuations)
No label, phase number, channel number, technology or print fact.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import trees_onnx
from .common import read_json

W_TREE = 0.6
C7 = ["Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other"]
ALL_NAMES = ([f"lt_{c}" for c in C7] + [f"ln_{c}" for c in C7] +
             ["log_minutes", "ent_t", "ent_n", "margin_t", "margin_n", "agree", "stack_n",
              "pk_span", "pk_track", "n_span_peers", "n_coloc_peers", "nl_self", "lane_min",
              "n_lanes_phase", "phase_n_lanes", "phase_n_lanes_conf", "lane_conf", "n_ph", "phm_A", "phm_P", "phm_C",
              "phm_Y", "rk_A", "rk_P", "rk_C", "rk_Y", "lm_n", "lm_A", "lm_P", "lm_C", "lm_Y", "log1p_det_n_on"])


def ent(P):
    Q = np.clip(P, 1e-9, 1)
    return -(Q * np.log(Q)).sum(1)


def margin(P):
    s = np.sort(P, 1)
    return s[:, -1] - s[:, -2]


def stack_X(minutes: float, Pt, Pn):
    n = len(Pt)
    lm = np.full(n, np.log(max(float(minutes), 1e-3)))
    lt, ln = np.log(np.clip(Pt, 1e-6, 1)), np.log(np.clip(Pn, 1e-6, 1))
    return np.column_stack([lt, ln, lm, ent(Pt), ent(Pn), margin(Pt), margin(Pn),
                            (Pt.argmax(1) == Pn.argmax(1)).astype(float)])


def grp_second_max(v, g):
    """per element: max of v over the OTHER members of its group g (NaN for singletons)."""
    o = np.lexsort((-v, g))
    gs, vs = g[o], v[o]
    first = np.r_[True, gs[1:] != gs[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(gs)), 0))
    top = vs[start]
    sec_pos = start + 1
    sec = np.where((sec_pos < len(gs)) & (gs[np.minimum(sec_pos, len(gs) - 1)] == gs), vs[np.minimum(sec_pos, len(gs) - 1)],
                   np.nan)
    out_s = np.where(np.arange(len(gs)) == start, sec, top)
    r = np.empty(len(v))
    r[o] = out_s
    return r


def ctx_X(fr: pd.DataFrame, Pb: np.ndarray) -> np.ndarray:
    """fr (one signal-sample, one row per detector): stack_n, pk_span, pk_track, pk_span_peers,
    pk_coloc_peers, lanes5g (comma string / None), phase_n_lanes, phase_n_lanes_conf, lane_conf, pred_phase, det_n_on."""
    n = len(fr)
    H = fr[["stack_n"]].to_numpy(np.float64)
    LN = fr[["phase_n_lanes", "phase_n_lanes_conf", "lane_conf"]].to_numpy(float)
    lanes = fr.lanes5g.to_numpy(object)
    lsets = [[int(x) for x in s.split(",")] if isinstance(s, str) and s else [] for s in lanes]
    nl_self = np.array([len(s) for s in lsets], float)
    lane_min = np.array([min(s) if s else np.nan for s in lsets], float)
    npeer = lambda col: np.array([len(s.split(",")) if isinstance(s, str) and s else 0          # noqa: E731
                                  for s in fr[col].to_numpy(object)], float)
    grp = fr.groupby(["pred_phase"], dropna=False).ngroup().to_numpy()
    n_ph = pd.Series(grp).map(pd.Series(grp).value_counts()).to_numpy(float)
    phm = np.column_stack([grp_second_max(Pb[:, c], grp) for c in range(4)])
    rk = np.column_stack([pd.Series(Pb[:, c]).groupby(grp).rank(ascending=False).to_numpy() for c in range(4)])
    e = pd.DataFrame({"g": grp, "i": np.arange(n), "lane": lsets}).explode("lane").dropna(subset=["lane"])
    n_lanes_phase = pd.Series(grp).map(e.groupby("g").lane.nunique()).fillna(0).to_numpy(float)
    m = e.merge(e, on=["g", "lane"], suffixes=("", "_m"))
    m = m[m.i != m.i_m].drop_duplicates(["i", "i_m"])
    lm = np.full((n, 5), np.nan)
    if len(m):
        for c in range(4):
            m[f"p{c}"] = Pb[m.i_m.to_numpy(int), c]
        agg = m.groupby("i").agg(nm=("i_m", "size"), p0=("p0", "max"), p1=("p1", "max"), p2=("p2", "max"),
                                 p3=("p3", "max"))
        lm[agg.index.to_numpy(int)] = agg[["nm", "p0", "p1", "p2", "p3"]].to_numpy(float)
    lm[:, 0] = np.where(nl_self > 0, np.nan_to_num(lm[:, 0]), np.nan)
    pk = np.column_stack([fr.pk_span.to_numpy(float), fr.pk_track.to_numpy(float),
                          npeer("pk_span_peers"), npeer("pk_coloc_peers")])
    ln_on = np.log1p(fr.det_n_on.to_numpy(float))
    return np.column_stack([H, pk, nl_self, lane_min, n_lanes_phase, LN, n_ph, phm, rk, lm, ln_on]).astype(np.float32)


class Stacker:
    """The 'mean3' stacker (trained on the trees + the 3-member network mean), loaded on first use.  note 120: the
    'single' (1-member network) and 'nonet' (no network) variants of the fast switch are gone."""

    def __init__(self, model_dir):
        self.md = Path(model_dir)
        self.meta = read_json(self.md / "stacker.json")
        self._bag = None
        self.classes = self.meta["classes"]
        names = self.meta.get("feature_names") or ALL_NAMES
        assert all(n in ALL_NAMES for n in names), "unknown stacker column"
        self.cols = np.array([ALL_NAMES.index(n) for n in names], int)

    def bag(self):
        if self._bag is None:
            self._bag = trees_onnx.Bag([self.md / f for f in self.meta["onnx_files"]])
        return self._bag

    def predict(self, minutes: float, Pt: np.ndarray, Pn: np.ndarray, fr: pd.DataFrame) -> np.ndarray:
        Pb = W_TREE * Pt + (1 - W_TREE) * Pn
        X = np.hstack([stack_X(minutes, Pt, Pn), ctx_X(fr, Pb).astype(np.float64)])   # as trained (float64 | float32)
        return self.bag().predict(X[:, self.cols])
