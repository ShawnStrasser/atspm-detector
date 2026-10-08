"""Note 77 (copy of note 76's diag76.py, channel modes added): where does phase numbering / channel order leak?  Stage-by-stage renumbering diff of a candidate package.

One (signal, length) of the note-71 bench extracts is scored twice by the package's own predict(): as logged and with
every phase number permuted in the raw log (events 1, 7-11, 43, 44, 150).  Every intermediate the package builds is
captured (phase pair features, tree p0, GRU, decoder inputs / probs, function frame, expert features, trees / net /
stacker probabilities, lanes / pick / health context, final outputs) and compared row by row after mapping the phase
keys through the permutation.  Optional --chan permutes detector channel numbers as well (diagnostic only: the decoder's
channel adjacency uses channel differences by design, so channel invariance is not expected there).

    python diag77.py <package dir> <sig> <L> <out.json> [--seed 75] [--chan | --chanrev | --chanshift | --chanrevshift]
      --chanshift     ch -> ch + k (k moves the block to the top / bottom of 1..64): values change, order and differences kept
      --chanrevshift  reversal and shift together
    Peer lists (pk_span_peers / pk_coloc_peers) are compared as SETS after mapping the channels back.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

W = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
B71 = W / "bench71"
TOL = 1e-6


def capture(P):
    import decode as dec
    import features_expert as fx
    import function_stage as fs
    import gru_blend as gb
    C = {}

    def wrap(obj, name, fn):
        f = getattr(obj, name)
        setattr(obj, name, fn(f))

    wrap(P, "build_features", lambda f: (lambda *a, **k: C.__setitem__("feat", f(*a, **k)) or C["feat"]))
    wrap(gb, "phase_probs_kept", lambda f: (lambda *a, **k: C.__setitem__("gru", f(*a, **k)) or C["gru"]))
    wrap(dec, "assemble", lambda f: (lambda *a, **k: C.__setitem__("dec", f(*a, **k)) or C["dec"]))
    wrap(P, "score", lambda f: (lambda *a, **k: C.__setitem__("score", f(*a, **k)) or C["score"]))
    wrap(P, "_function_frame", lambda f: (lambda *a, **k: C.__setitem__("top", f(*a, **k)) or C["top"]))
    wrap(fx, "build", lambda f: (lambda *a, **k: C.__setitem__("expert", f(*a, **k)) or C["expert"]))
    C["fs"] = {}

    def run_wrap(f):
        def g(con, top, model_dir, b0, b1, streams, pieces_rel, log=lambda m: None, inject=None):
            r = f(con, top, model_dir, b0, b1, streams, pieces_rel, log, {"capture": C["fs"]})
            C["fs_out"] = r
            return r
        return g
    wrap(fs, "run", run_wrap)
    return C


def frame_diff(a: pd.DataFrame, b: pd.DataFrame, keys: list, name: str, out: dict):
    a = a.copy()
    b = b.copy()
    for k in keys:
        if k in ("DeviceId", "win"):
            a[k] = a[k].astype(str)
            b[k] = b[k].astype(str)
        else:
            a[k] = a[k].astype(float)
            b[k] = b[k].astype(float)
    m = a.merge(b, on=keys, how="outer", suffixes=("", "__s"), indicator=True)
    res = {"rows_a": int(len(a)), "rows_b": int(len(b)), "unmatched": int((m._merge != "both").sum()), "cols": {}}
    m = m[m._merge == "both"]
    for c in a.columns:
        if c in keys or c + "__s" not in m.columns or c == "partner_set":
            continue
        x, y = m[c], m[c + "__s"]
        if not (pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y)):
            ne = int((x.astype(str) != y.astype(str)).sum())
            if ne:
                res["cols"][c] = {"n_diff": ne}
            continue
        x, y = x.to_numpy(float), y.to_numpy(float)
        nan_mis = np.isnan(x) != np.isnan(y)
        d = np.abs(np.where(np.isnan(x) | np.isnan(y), 0.0, x - y))
        scale = np.maximum(1.0, np.abs(np.nan_to_num(x)))
        bad = nan_mis | (d / scale > TOL)
        if bad.any():
            res["cols"][c] = {"n_diff": int(bad.sum()), "n_nanmis": int(nan_mis.sum()), "max_abs": float(d.max())}
    out[name] = res


def main():
    pkg, sig, L, outp = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 75
    chanshift = "--chanshift" in sys.argv or "--chanrevshift" in sys.argv
    chan = "--chan" in sys.argv or "--chanrev" in sys.argv or chanshift
    chanrev = "--chanrev" in sys.argv or "--chanrevshift" in sys.argv          # channel ch -> max+1-ch: |channel differences| (decoder adjacency) kept,
    #                                            channel ORDER reversed -> isolates channel-order tie-breaks
    sys.path.insert(0, str(pkg))
    import predict as P
    ev = pd.read_parquet(B71 / f"ev_{sig}_{L}.parquet")
    if "--minutes" in sys.argv:          # a shorter sample cut from the start of the extract
        mins = float(sys.argv[sys.argv.index("--minutes") + 1])
        ts = pd.to_datetime(ev.Timestamp)
        ev = ev[ts < ts.min() + pd.Timedelta(minutes=mins)].reset_index(drop=True)
    is_ph = ev.EventId.isin([1, 7, 8, 9, 10, 11, 43, 44, 150])
    phases = sorted(int(x) for x in ev.loc[ev.EventId.isin([1, 7, 8, 9, 10, 11, 43, 44]), "Parameter"].unique())
    rng = np.random.default_rng(seed)
    perm = dict(zip(phases, [int(x) for x in rng.permutation(phases)]))
    if "--nophase" in sys.argv:
        perm = {p: p for p in phases}
    ev2 = ev.copy()
    ev2.loc[is_ph, "Parameter"] = ev2.loc[is_ph, "Parameter"].map(lambda v: perm.get(int(v), int(v)))
    cperm = None
    if chan:
        is_d = ev.EventId.isin([81, 82]) & (ev.Parameter <= 64)
        chs = sorted(int(x) for x in ev.loc[is_d, "Parameter"].unique())
        if chanrev or chanshift:
            tgt = [max(chs) + min(chs) - c for c in chs] if chanrev else list(chs)
            k = 0
            if chanshift:
                k = 64 - max(tgt) if max(tgt) < 64 else -(min(tgt) - 1)
            cperm = dict(zip(chs, [c + k for c in tgt]))
        else:
            cperm = dict(zip(chs, [int(x) for x in rng.permutation(chs)]))
        ev2.loc[is_d, "Parameter"] = ev2.loc[is_d, "Parameter"].map(cperm)
    C = capture(P)
    r1 = P.predict(ev, threads=4, min_actuations=1)
    S1 = dict(C)
    C.clear()
    C["fs"] = {}
    r2 = P.predict(ev2, threads=4, min_actuations=1)
    S2 = dict(C)
    inv = {v: k for k, v in perm.items()}
    cinv = {v: k for k, v in cperm.items()} if cperm else None

    def back(df, cols_ph=("cand_phase", "pred_phase", "phase_pred", "phase_2nd", "phase", "lane_phase", "partner_phase", "phase_guess"),
             cols_ch=("Detector", "other")):
        df = df.copy()
        for c in cols_ph:
            if c in df.columns:
                df[c] = df[c].map(lambda v: inv.get(int(v), v) if pd.notna(v) else v)
        if cinv:
            for c in cols_ch:
                if c in df.columns:
                    df[c] = df[c].map(lambda v: cinv.get(int(v), v) if pd.notna(v) else v)
        return df

    out = {"pkg": pkg.name, "sig": sig, "L": L, "perm": perm, "chan": bool(chan), "chanrev": bool(chanrev),
           "chanshift": bool(chanshift), "cperm": cperm}

    def peers_back(df):
        df = df.copy()
        for c in ("pk_span_peers", "pk_coloc_peers"):
            if c in df.columns:
                df[c] = df[c].map(lambda s: ",".join(str(x) for x in sorted(
                    (cinv.get(int(v), int(v)) if cinv else int(v)) for v in str(s).split(",") if v)) if isinstance(s, str) else s)
        return df

    def peers_sort(df):
        df = df.copy()
        for c in ("pk_span_peers", "pk_coloc_peers"):
            if c in df.columns:
                df[c] = df[c].map(lambda s: ",".join(str(x) for x in sorted(int(v) for v in str(s).split(",") if v))
                                  if isinstance(s, str) else s)
        return df
    K4 = ["DeviceId", "Detector", "cand_phase", "win"]
    frame_diff(S1["feat"][0], back(S2["feat"][0]), K4, "phase_features", out)
    if "score" in S1:
        a = S1["score"][K4 + ["p0", "prob"]]
        frame_diff(a, back(S2["score"][K4 + ["p0", "prob"]]), K4, "phase_probs", out)
    if "gru" in S1 and len(S1["gru"]):
        kk = [c for c in S1["gru"].columns if c in ("DeviceId", "Detector", "cand_phase", "win")]
        frame_diff(S1["gru"], back(S2["gru"]), kk, "gru", out)
    if "dec" in S1:
        frame_diff(S1["dec"], back(S2["dec"]), ["DeviceId", "Detector", "win", "cand_phase"], "decoder_inputs", out)
    if "top" in S1:
        frame_diff(S1["top"], back(S2["top"]), ["DeviceId", "Detector", "win"], "function_frame", out)
    if "expert" in S1:
        frame_diff(S1["expert"], back(S2["expert"]), ["DeviceId", "Detector"], "expert", out)
    for dev, c1 in S1["fs"].items():
        c2 = S2["fs"][dev]
        dets1 = c1["fr"].Detector.to_numpy()
        dets2 = c2["fr"].Detector.to_numpy()
        if cinv:
            dets2 = np.array([cinv[int(d)] for d in dets2])
        o2 = pd.Series(range(len(dets2)), index=dets2).reindex(dets1).to_numpy()
        for k in ("Pt", "Pn", "Ps"):
            d = np.abs(c1[k] - c2[k][o2])
            out.setdefault("function_probs", {})[f"{dev}|{k}"] = float(np.nanmax(d))
        fr2 = peers_back(back(c2["fr"])).set_index("Detector").reindex(dets1).reset_index()
        frame_diff(peers_sort(c1["fr"]), fr2, ["Detector"], f"stack_context|{dev}", out)
    r2b = back(r2)
    if cinv:                                   # text naming a detector ("relative to d12") follows the renumbering
        import re
        for c in ("health_reason", "health_watch", "health_bad_periods", "review_reason"):
            if c in r2b.columns:
                r2b[c] = r2b[c].map(lambda t: re.sub(r"\bd(\d+)\b",
                                                     lambda mm: f"d{cinv.get(int(mm.group(1)), mm.group(1))}", t)
                                    if isinstance(t, str) else t)
    frame_diff(r1, r2b, ["DeviceId", "Detector"], "outputs", out)
    json.dump(out, open(outp, "w"), indent=1, default=str)
    # short print
    for k, v in out.items():
        if isinstance(v, dict) and "cols" in v:
            print(k, "rows", v["rows_a"], v["rows_b"], "unmatched", v["unmatched"],
                  {c: (d["n_diff"], round(d.get("max_abs", 0), 6)) for c, d in v["cols"].items()})
    print("function_probs", out.get("function_probs"))


if __name__ == "__main__":
    main()
