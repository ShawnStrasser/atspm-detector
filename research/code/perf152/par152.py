"""Note 152: parity of the batched health (src) against the captured main outputs (cap152.py).

    python par152.py <set> [--mode alone|group|both] [--n N] [--x]
alone: each signal through health_v4.assess (a batch of one); group: all signals of the set that share a window in
one health_v4.assess_group call.  Compares the per-detector table (every column, floats bit for bit, NaN == NaN) and
the signal note exactly; --x also compares the internal X columns (localises a difference)."""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.environ.get("P152_SRC") or str(REPO / "src"))
from atspm_detector import health_core as hc, health_v4 as hv  # noqa: E402


def same_val(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (a != a and b != b)
    if isinstance(a, (list, dict)) or isinstance(b, (list, dict)):
        return repr(a) == repr(b)
    try:
        if pd.isna(a) and pd.isna(b):
            return True
    except (TypeError, ValueError):
        pass
    return a == b


def cmp_frames(a: pd.DataFrame, b: pd.DataFrame, tag: str, cols=None) -> list:
    bad = []
    if len(a) != len(b):
        return [f"{tag}: rows {len(a)} vs {len(b)}"]
    a = a.reset_index(drop=True)
    b = b.reset_index(drop=True)
    for c in (cols or a.columns):
        if c not in b.columns:
            bad.append(f"{tag}: column {c} missing")
            continue
        x, y = a[c].tolist(), b[c].tolist()
        d = [i for i in range(len(x)) if not same_val(x[i], y[i])]
        if d:
            bad.append(f"{tag}: {c} {len(d)} rows differ, e.g. {x[d[0]]!r} vs {y[d[0]]!r}")
    return bad


def cmp_x(rec, X, s, S):
    """the internal X of signal s (new, keys) vs the captured X (old), on the columns both have."""
    Xo = rec["X"]
    if Xo is None or X is None:
        return []
    kb = s * hc.BASE
    Xn = X[(X.detector.astype(int) // hc.BASE) == s].copy() if S > 1 else X.copy()
    Xn["detector"] = Xn.detector.astype(int) - kb
    Xn = Xn.sort_values("detector").reset_index(drop=True)
    Xo = Xo.sort_values("detector").reset_index(drop=True)
    skip = {"phase", "pred_phase", "sig", "alt_set"}
    cols = [c for c in Xo.columns if c in Xn.columns and c not in skip]
    return cmp_frames(Xo, Xn, "X", cols)


def item(rec):
    return dict(phase=rec["phase"], fn_label=rec["flab"], fn_prob=rec["func"], lanes=rec["span"],
                phase_conf=rec["pconf"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set")
    ap.add_argument("--mode", default="both")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--x", action="store_true")
    a = ap.parse_args()
    recs = pickle.load(open(DCW / "s152" / "cap" / f"{a.set}.pkl", "rb"))
    if a.n:
        recs = recs[:a.n]
    nbad = 0
    if a.mode in ("alone", "both"):
        t0 = time.perf_counter()
        for k, r in enumerate(recs):
            P = hc.prep_arrays(r["t"], r["eid"], r["par"])
            res, X, E = hv.assess_group(P, r["start"], r["end"], [item(r)], keep_all=True)
            out, note = res[0]
            bad = cmp_frames(r["out"], out, "out") + ([] if note == r["note"] else [f"note {r['note']!r} vs {note!r}"])
            if a.x:
                bad += cmp_x(r, X, 0, 1)
            if bad:
                nbad += 1
                print(f"ALONE {k}:", *bad[:12], sep="\n   ")
        print(f"alone: {len(recs)} signals, {nbad} differ, {time.perf_counter() - t0:.1f}s")
    if a.mode in ("group", "both"):
        groups = {}
        for k, r in enumerate(recs):
            groups.setdefault((r["start"], r["end"]), []).append(k)
        nb2 = 0
        t0 = time.perf_counter()
        for (st, en), ks in groups.items():
            G = hc.prep_group([hc.prep_arrays(recs[k]["t"], recs[k]["eid"], recs[k]["par"]) for k in ks],
                              names=list(range(len(ks))))
            res, X, E = hv.assess_group(G, st, en, [item(recs[k]) for k in ks], keep_all=True)
            for s, k in enumerate(ks):
                out, note = res[s]
                r = recs[k]
                bad = cmp_frames(r["out"], out, "out") + ([] if note == r["note"] else
                                                          [f"note {r['note']!r} vs {note!r}"])
                if a.x:
                    bad += cmp_x(r, X, s, len(ks))
                if bad:
                    nb2 += 1
                    print(f"GROUP {k}:", *bad[:12], sep="\n   ")
        print(f"group: {len(recs)} signals in {len(groups)} batches, {nb2} differ, {time.perf_counter() - t0:.1f}s")
        nbad += nb2
    sys.exit(1 if nbad else 0)


if __name__ == "__main__":
    main()
