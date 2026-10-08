"""ONNX Runtime sessions for every model of the package, and the tree-ensemble wrapper.

Each tree file under `weights/` is one LightGBM model compiled to an `ai.onnx.ml` TreeEnsemble (double inputs, double
thresholds, double leaf sums -- the same arithmetic as LightGBM).  The graph returns the raw score; the objective's
transform is applied here from the model's own metadata, exactly as LightGBM does:

    binary      sigmoid(sigmoid_coef * raw)
    multiclass  softmax over the classes
    lambdarank / regression / quantile   raw score

Sessions are created once per file and process (lazily, on first use) and reused by every later call.
Session options (`session`): graph optimisation ORT_ENABLE_ALL, sequential execution, `threads` intra-op threads
(default 4; env DC_TREE_THREADS for the trees, DC_NET_THREADS for the network), one inter-op thread, idle pool threads
do not spin (DC_ORT_SPIN=1 restores spinning: same answers, ~5x the CPU time).  The CPU memory arena can be switched off
per session: the network graphs do, so their high-water mark is not kept reserved for the life of the process.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

DEFAULT_THREADS = int(os.environ.get("DC_TREE_THREADS", "4"))
_SESSIONS: dict = {}


def session(path, threads: int = DEFAULT_THREADS, arena: bool = True):
    """A configured onnxruntime CPU session for `path`."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, int(threads))
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if not arena:
        so.enable_cpu_mem_arena = False
    if os.environ.get("DC_ORT_SPIN", "0") != "1":
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


class OnnxTrees:
    def __init__(self, path, threads: int = DEFAULT_THREADS):
        self.path = str(path)
        self.session = session(self.path, threads)
        m = self.session.get_modelmeta().custom_metadata_map
        self.objective = m.get("objective", "").split()
        self.k = int(m["k"])
        self.average_output = m.get("average_output", "0") == "1"
        self.sigmoid = float(m.get("sigmoid", "1.0"))
        self.n_features = int(m["n_features"])
        self.n_trees = int(m["n_trees"])
        self.feature_names = json.loads(m["feature_names"])

    def raw(self, X) -> np.ndarray:
        if hasattr(X, "to_numpy"):
            X = X.to_numpy(dtype=np.float64, na_value=np.nan)
        X = np.ascontiguousarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"{Path(self.path).name}: expected {self.n_features} feature columns, got {X.shape}")
        if X.shape[0] == 0:
            return np.zeros((0, self.k))
        r = np.asarray(self.session.run(None, {"X": X})[0], dtype=np.float64).reshape(X.shape[0], self.k)
        if self.average_output:
            r = r / max(1, self.n_trees // self.k)
        return r

    def predict(self, X) -> np.ndarray:
        raw = self.raw(X)
        name = self.objective[0] if self.objective else ""
        if name == "binary":
            raw = 1.0 / (1.0 + np.exp(-self.sigmoid * raw))
        elif name in ("multiclass", "softmax"):
            e = np.exp(raw - raw.max(1, keepdims=True))
            raw = e / e.sum(1, keepdims=True)
        return raw[:, 0] if self.k == 1 else raw


def load(path, threads: int = DEFAULT_THREADS) -> OnnxTrees:
    """Cached tree model per file (stays loaded for the life of the process: warm calls skip the load)."""
    key = (str(Path(path).resolve()), int(threads))
    s = _SESSIONS.get(key)
    if s is None:
        s = _SESSIONS[key] = OnnxTrees(path, threads)
    return s


class Bag:
    """Mean prediction of several seed models (in each model's output space, as training averaged them)."""

    def __init__(self, paths, threads: int = DEFAULT_THREADS):
        self.models = [load(p, threads) for p in paths]
        self.n_features = self.models[0].n_features
        self.feature_names = self.models[0].feature_names

    def predict(self, X) -> np.ndarray:
        if hasattr(X, "to_numpy"):
            X = X.to_numpy(dtype=np.float64, na_value=np.nan)
        out = None
        for m in self.models:
            p = m.predict(X)
            out = p if out is None else out + p
        return out / len(self.models)
