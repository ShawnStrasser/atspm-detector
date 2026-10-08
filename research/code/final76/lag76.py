"""Note 76 (channel-order part): the detector-pair lag table with the tie-keeping top-K cut, and the function frame's
lag features with tie-averaged 'best' / 'twin' neighbours (phasefree/features_yellowred.py, phasefree/function.py).

    python lag76.py build --src dec|stg     # new lag tables for every frame-v6e signal x window (same caches / windows as
                                            # the pool scan) -> f76/lag_<src>.parquet
    python lag76.py frame                   # (1) old function + stored lag tables must reproduce the frame's lag columns
                                            # (recipe check); (2) new function + new tables -> f76/frame_v6e_c/ (= the
                                            # f76 frame, i.e. excl_partner_diff already fixed, + new lag columns)
CPU, 3 threads; locked_v2 asserted absent.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["build", "frame"])
ap.add_argument("--src", default="dec", choices=["dec", "stg"])
ap.add_argument("--threads", type=int, default=3)
ap.add_argument("--chunk", type=int, default=8)
ap.add_argument("--limit", type=int, default=0)
A = ap.parse_args()
W0 = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
if A.stage == "build" and A.src == "stg":
    os.environ["DC_WORK"] = str(W0 / "official" / "stg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: E402
import build_features as bf  # noqa: E402
import features_yellowred as FY  # noqa: E402
import function as FN  # noqa: E402

assert "rank() OVER" in FY.SQL_LAG and hasattr(FN, "_tied_top"), "research path must resolve phasefree/"
F76 = W0 / "final_v3_work" / "f76"
LAGCOLS_PREFIX = ("lagany_", "lagsib_")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def stage_build():
    from common import connect
    if A.src == "stg":
        import windows_stg  # noqa: F401
    fr = pd.read_parquet(W0 / "trackA" / "v3" / "frame_v6e" / "feat_frame.parquet", columns=["DeviceId", "period"])
    fids = set(fr[fr.period == A.src].DeviceId.str.lower())
    lk = set(pd.read_csv(W0 / "official" / "locked_v2.csv").DeviceId.str.lower())
    assert not fids & lk
    meta = pd.read_parquet(bf.CACHE / "signal_meta.parquet", columns=["DeviceId"]).DeviceId
    ids = sorted(d for d in meta if d.lower() in fids)
    if A.limit:
        ids = ids[:A.limit]
    wins = bf.WINDOW_SETS["mixedb" if A.src == "dec" else "stgall"]
    con = connect(threads=A.threads)
    log(f"{A.src}: {len(ids)} of {len(fids)} frame signals in the cache x {len(wins)} windows")
    parts, t0 = [], time.time()
    for i in range(0, len(ids), A.chunk):
        bf.load_chunk(con, ids[i:i + A.chunk])
        dm = con.sql("SELECT * FROM devmap").df()
        for w in wins:
            w0 = bf._epoch(w)
            con.execute(f"CREATE OR REPLACE TEMP TABLE onev AS SELECT * FROM onev_all "
                        f"WHERE t_on >= {w0} AND t_on < {w0 + w['secs']}")
            r = FY.build(con, w["win"], dm, FY.SQL_LAG)
            if len(r):
                parts.append(r)
        log(f"chunk {i // A.chunk + 1}/{(len(ids) + A.chunk - 1) // A.chunk} elapsed {time.time() - t0:.0f}s")
    d = pd.concat(parts, ignore_index=True).rename(columns={"det": "Detector", "oth": "other"})
    d = bf._shrink(d, int16=("Detector", "other"))
    d.to_parquet(F76 / f"lag_{A.src}.parquet", index=False)
    log(f"wrote {len(d):,} rows ({time.time() - t0:.0f}s)")


def _old_function():
    spec = importlib.util.spec_from_file_location("function_old", rpath.MODEL / "function.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def stage_frame():
    sys.path.insert(0, str(rpath.CODE / "trackA"))
    import v6e_frame as V6E
    import function_v4 as fv4
    FO = _old_function()
    fr = pd.read_parquet(F76 / "frame_v6e" / "feat_frame.parquet")
    lagc = [c for c in fr.columns if c.startswith(LAGCOLS_PREFIX)]
    res = {"lag_cols": len(lagc)}
    newparts = []
    for per in ("dec", "stg"):
        f = fr[fr.period == per]
        keep = set(f.DeviceId)
        probs = pd.read_parquet(V6E.PH / f"phase_pred_{V6E.PROBS}_{per}.parquet")
        probs["DeviceId"] = probs.DeviceId.str.lower()
        probs = probs[probs.DeviceId.isin(keep)]
        i = probs.groupby(["DeviceId", "Detector", "win"], sort=False)["prob"].idxmax()
        top = probs.loc[i, ["DeviceId", "Detector", "win", "cand_phase"]].rename(columns={"cand_phase": "pred_phase"})
        top = top.reset_index(drop=True)
        # (1) recipe check: old function on the stored lag tables
        _, lag_files = V6E.inputs(per)
        lo = fv4._read(list(lag_files), None)
        lo["DeviceId"] = lo.DeviceId.str.lower()
        lo = lo[lo.DeviceId.isin(keep)]
        a = FO.add_lag_features(top.copy(), lo)
        k = ["DeviceId", "Detector", "win"]
        a["Detector"] = a.Detector.astype(f.Detector.dtype)
        m = f[k + lagc].merge(a[k + lagc], on=k, how="left", suffixes=("", "__r"))
        bad = {}
        for c in lagc:
            x, y = m[c].to_numpy(float), m[c + "__r"].to_numpy(float)
            ok = (np.isnan(x) & np.isnan(y)) | (np.abs(x - y) <= 1e-5 * np.maximum(1, np.abs(np.nan_to_num(x))))
            if (~ok).any():
                bad[c] = int((~ok).sum())
        res[f"recipe_check_{per}"] = {"rows": len(f), "cols_with_mismatch": bad}
        log(f"{per}: recipe check {res[f'recipe_check_{per}']}")
        # (2) new function on the new tables
        ln = pd.read_parquet(F76 / f"lag_{per}.parquet")
        ln["DeviceId"] = ln.DeviceId.str.lower()
        ln = ln[ln.DeviceId.isin(keep)]
        b = FN.add_lag_features(top.copy(), ln)
        b["Detector"] = b.Detector.astype(f.Detector.dtype)
        mb = f[k].reset_index().merge(b[k + lagc], on=k, how="left")
        newparts.append(mb)
        # how many values move (old rule vs new rule)
        ch = {}
        for c in lagc:
            x, y = m[c].to_numpy(float), mb[c].to_numpy(float)
            ok = (np.isnan(x) & np.isnan(y)) | (np.abs(np.nan_to_num(x) - np.nan_to_num(y)) <= 1e-6 * np.maximum(1, np.abs(np.nan_to_num(x))))
            ch[c] = int((~ok).sum())
        res[f"changed_{per}"] = ch
        res[f"rows_any_lag_change_{per}"] = float(np.mean(np.column_stack(
            [~((np.isnan(m[c].to_numpy(float)) & np.isnan(mb[c].to_numpy(float))) |
               (np.abs(np.nan_to_num(m[c].to_numpy(float)) - np.nan_to_num(mb[c].to_numpy(float))) <= 1e-6 *
                np.maximum(1, np.abs(np.nan_to_num(m[c].to_numpy(float)))))) for c in lagc]).any(1)))
        log(f"{per}: rows with any lag column changed {res[f'rows_any_lag_change_{per}']:.4f}")
    nb = pd.concat(newparts, ignore_index=True).set_index("index")
    out = F76 / "frame_v6e_c"
    out.mkdir(parents=True, exist_ok=True)
    for c in lagc:
        fr[c] = nb.loc[fr.index, c].to_numpy().astype(fr[c].dtype)
    fr.to_parquet(out / "feat_frame.parquet", index=False)
    shutil.copy(F76 / "frame_v6e" / "feat_meta.json", out / "feat_meta.json")
    json.dump(res, open(F76 / "lag76.json", "w"), indent=1)
    log(f"-> {out}")


if __name__ == "__main__":
    stage_build() if A.stage == "build" else stage_frame()
