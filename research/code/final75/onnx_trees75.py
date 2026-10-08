"""Note 75: LightGBM text model -> ONNX tree ensemble (research side; the package only RUNS the .onnx files).

Two encodings (both ai.onnx.ml):
  te5   `TreeEnsemble` (ai.onnx.ml opset 5): double inputs, double thresholds, double leaf weights, DOUBLE output
        -> sums leaves in double like LightGBM / the numpy booster (default; parity ~1e-12)
  ter3  `TreeEnsembleRegressor` (ai.onnx.ml opset 3, note 73's converter): float output (float32 rounding of the raw sum)

The graph returns the RAW score [N, k] (sum of leaves); the objective's transform (sigmoid / softmax / average) is
applied by the package runtime (`trees_onnx.py`), which reads it from the model's metadata (objective, k,
average_output, sigmoid, n_features, feature_names) -- exactly as `lgbm_numpy.NumpyBooster.predict` does.

LightGBM missing handling: missing type 'nan' -> the default direction; 'none' (NaN read as 0) -> 0 <= threshold;
'zero' (zero_as_missing) -> asserted absent (it would need the zero test, which an ONNX tree ensemble cannot express).
"""
from __future__ import annotations

import json

import numpy as np


def _tracks_true(t, i) -> int:
    mt = int(t.miss[i])
    if mt == 2:
        return int(bool(t.default_left[i]))
    if mt == 0:
        return int(bool(t.thr[i] >= 0))
    raise NotImplementedError("zero-as-missing split")


def _meta(nb) -> dict:
    return {"objective": " ".join(nb.objective), "k": str(nb.k), "average_output": str(int(nb.average_output)),
            "sigmoid": repr(nb.sigmoid), "n_features": str(nb.n_features), "n_trees": str(len(nb.trees)),
            "feature_names": json.dumps(nb.feature_names), "format": "raw sum of leaves; transform in trees_onnx.py"}


def to_onnx_te5(nb, path):
    import onnx
    from onnx import helper, TensorProto
    roots, feat, split, modes, tn, tl, fn, fl, miss = [], [], [], [], [], [], [], [], []
    lt, lw = [], []
    for ti, t in enumerate(nb.trees):
        k = ti % nb.k
        base_n, base_l = len(feat), len(lt)
        if t.feat is None:                      # stump: one always-true node to a single leaf
            roots.append(base_n)
            feat.append(0); split.append(0.0); modes.append(0)
            tn.append(base_l); tl.append(1); fn.append(base_l); fl.append(1); miss.append(1)
            lt.append(k); lw.append(float(t.leaf[0]))
            # a NaN input with tracks_true=1 goes 'true'; a number goes v <= 0 or not -> both map to the same leaf
            continue
        roots.append(base_n)
        m = len(t.feat)
        for i in range(m):
            feat.append(int(t.feat[i])); split.append(float(t.thr[i])); modes.append(0)
            miss.append(_tracks_true(t, i))
            for ch, nid, leaf in ((t.left[i], tn, tl), (t.right[i], fn, fl)):
                if ch >= 0:
                    nid.append(base_n + int(ch)); leaf.append(0)
                else:
                    nid.append(base_l + int(~ch)); leaf.append(1)
        for v in t.leaf:
            lt.append(k); lw.append(float(v))
    node = helper.make_node(
        "TreeEnsemble", ["X"], ["Y"], domain="ai.onnx.ml",
        aggregate_function=1, n_targets=nb.k, post_transform=0, tree_roots=roots,
        nodes_featureids=feat, nodes_splits=helper.make_tensor("s", TensorProto.DOUBLE, [len(split)], split),
        nodes_modes=helper.make_tensor("m", TensorProto.UINT8, [len(modes)], modes),
        nodes_truenodeids=tn, nodes_trueleafs=tl, nodes_falsenodeids=fn, nodes_falseleafs=fl,
        nodes_missing_value_tracks_true=miss,
        leaf_targetids=lt, leaf_weights=helper.make_tensor("w", TensorProto.DOUBLE, [len(lw)], lw))
    g = helper.make_graph([node], "lgbm", [helper.make_tensor_value_info("X", TensorProto.DOUBLE, [None, nb.n_features])],
                          [helper.make_tensor_value_info("Y", TensorProto.DOUBLE, [None, nb.k])])
    mdl = helper.make_model(g, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("ai.onnx.ml", 5)],
                            ir_version=10)
    for k_, v in _meta(nb).items():
        e = mdl.metadata_props.add()
        e.key, e.value = k_, v
    onnx.checker.check_model(mdl)
    onnx.save(mdl, str(path))


def raw_numpy(nb, X) -> np.ndarray:
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    raw = np.zeros((X.shape[0], nb.k))
    for i, t in enumerate(nb.trees):
        raw[:, i % nb.k] += t.predict(X)
    return raw
