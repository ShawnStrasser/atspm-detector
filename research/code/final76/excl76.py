"""Note 76: the phase-order-free partner features on the note-57 / cand64 PHASE POOL.

Re-runs only the mask stage of the feature build (`features.apply_window` -> `phasefree/features._mask_features`
and `phasefree/features_partner._partner_features`) for every pool signal x window, from the same event caches the
pool was built from (DEC = `%DC_WORK%/cache`, windows `mixedb`; STG + REL = `%DC_WORK%/official/stg/cache`, windows
`stgall`).  Writes the NEW `excl_partner_diff` per pair, the tie size behind it, and the partner tie sets of
features_partner (only where more than one partner ties after both keys).

    python excl76.py --src dec|stg     -> %DC_WORK%/final_v3_work/f76/excl_<src>.parquet (+ ptie_<src>.parquet)

CPU, threads <= 3 per process; locked_v2 asserted absent.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, choices=["dec", "stg"])
ap.add_argument("--threads", type=int, default=3)
ap.add_argument("--chunk", type=int, default=8)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--frame-extra", action="store_true", dest="frame_extra")
A = ap.parse_args()
W0 = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
if A.src == "stg":
    os.environ["DC_WORK"] = str(W0 / "official" / "stg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: F401,E402
import build_features as bf  # noqa: E402
import features as F  # noqa: E402
import features_partner as FP  # noqa: E402
from common import connect  # noqa: E402

assert hasattr(F, "tied_best"), "research path must resolve phasefree/features.py"
if A.src == "stg":
    import windows_stg  # noqa: F401,E402
OUT = W0 / "final_v3_work" / "f76"
OUT.mkdir(parents=True, exist_ok=True)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def tie_sizes(con) -> pd.DataFrame:
    """(dev, det, p) -> number of partners tied on co-green time behind excl_partner_diff (1 = no tie)."""
    om = con.sql("SELECT dev, det, mask, n_on FROM onmask").df()
    mt = con.sql("SELECT * FROM masktime").df()
    cand = con.sql("SELECT * FROM cand").df()
    rows = []
    for dev, cd in cand.groupby("dev"):
        phases = sorted(int(x) for x in cd.p.unique())
        mtd, omd = mt[mt.dev == dev], om[om.dev == dev]
        if not len(mtd) or not len(omd):
            continue
        masks = mtd["mask"].to_numpy(dtype=np.int64)
        secs = mtd["secs"].to_numpy(dtype=float)
        if secs.sum() <= 0:
            continue
        bits = {p: ((masks >> (p - 1)) & 1).astype(bool) for p in phases}
        for p in phases:
            cot = []
            for q in phases:
                if q == p:
                    continue
                if secs[bits[p] & ~bits[q]].sum() < 120 or secs[~bits[p] & bits[q]].sum() < 120:
                    continue
                cot.append(secs[bits[p] & bits[q]].sum())
            n = int(F.tied_best(cot).sum()) if cot else 0
            for det in omd.det.unique():
                rows.append((dev, int(det), p, n, float(max(cot)) if cot else np.nan))
    return pd.DataFrame(rows, columns=["dev", "det", "p", "n_tied", "cot_max"])


def main():
    pool = pd.read_parquet(W0 / "trees57" / "phase" / "pool.parquet", columns=["DeviceId", "src"]).drop_duplicates()
    lk = set(pd.read_csv(W0 / "official" / "locked_v2.csv").DeviceId.str.lower())
    srcs = {"dec": ["DEC"], "stg": ["STG", "REL"]}[A.src]
    ids = sorted(pool[pool.src.isin(srcs)].DeviceId.unique())
    plain = [d.replace("@stg", "") for d in ids]
    tag = A.src
    if A.frame_extra:
        # signals of the function frame (v6e) that are not in the phase pool (function labels without phase timing)
        fr = pd.read_parquet(W0 / "trackA" / "v3" / "frame_v6e" / "feat_frame.parquet", columns=["DeviceId", "period"])
        have = {d.lower() for d in plain}
        fids = sorted(set(fr[fr.period == A.src].DeviceId.str.lower()) - have)
        meta = pd.read_parquet(bf.CACHE / "signal_meta.parquet", columns=["DeviceId"]).DeviceId
        cmap = {d.lower(): d for d in meta}
        plain = [cmap[d] for d in fids if d in cmap]
        log(f"frame-only signals: {len(fids)}, in the cache: {len(plain)}")
        tag = A.src + "_fx"
    suffix = "@stg" if A.src == "stg" else ""
    if A.limit:
        plain = plain[:A.limit]
    assert not {d.lower() for d in plain} & lk, "locked signal in the scan"
    wins = bf.WINDOW_SETS["mixedb" if A.src == "dec" else "stgall"]
    con = connect(threads=A.threads)
    log(f"{A.src}: {len(plain)} signals x {len(wins)} windows")
    parts, pt_parts, t0 = [], [], time.time()
    for i in range(0, len(plain), A.chunk):
        ch = plain[i:i + A.chunk]
        bf.load_chunk(con, ch)
        dm = con.sql("SELECT * FROM devmap").df()
        for w in wins:
            w0 = bf._epoch(w)
            F.apply_window(con, w0, w0 + w["secs"])
            m = F._mask_features(con)
            if not len(m):
                continue
            ts = tie_sizes(con)
            m = m[["dev", "det", "p", "excl_partner_diff", "n_excl_pairs"]].merge(ts, on=["dev", "det", "p"], how="left")
            m = m.merge(dm, on="dev").drop(columns="dev")
            m["win"] = w["win"]
            parts.append(m)
            pex, prow = FP._partner_features(con)
            if len(prow):
                multi = prow[prow.partner_set.str.contains(",")]
                if len(multi):
                    pm = pex.merge(multi, on=["dev", "p"], how="inner").merge(dm, on="dev").drop(columns="dev")
                    pm["win"] = w["win"]
                    pt_parts.append(pm)
        log(f"chunk {i // A.chunk + 1}/{(len(plain) + A.chunk - 1) // A.chunk} elapsed {time.time() - t0:.0f}s")
    df = pd.concat(parts, ignore_index=True)
    df["DeviceId"] = df.DeviceId + suffix
    df = df.rename(columns={"det": "Detector", "p": "cand_phase", "excl_partner_diff": "epd_new"})
    df.to_parquet(OUT / f"excl_{tag}.parquet", index=False)
    if pt_parts:
        pt = pd.concat(pt_parts, ignore_index=True)
        pt["DeviceId"] = pt.DeviceId + suffix
        pt.to_parquet(OUT / f"ptie_{tag}.parquet", index=False)
        log(f"partner ties: {len(pt):,} detector-pair rows")
    else:
        log("partner ties: none")
    log(f"wrote {len(df):,} rows, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
