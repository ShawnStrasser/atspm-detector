"""Note 151: compare two predict outputs (detector + phase tables) column by column.
    python cmp151.py <tagA_job> <tagB_job> [--by-signal]   (files %DC_WORK%/s151/runs/<x>_{det,ph}.parquet)
Floats: count of rows not bit-identical (NaN == NaN) and max |diff|; everything else: exact equality."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
R = DCW / "s151" / "runs"


def load(x, kind):
    return pd.read_parquet(R / f"{x}_{kind}.parquet").drop(columns=["item"], errors="ignore")


def compare(a, b, key, name):
    a = a.sort_values(key).reset_index(drop=True)
    b = b.sort_values(key).reset_index(drop=True)
    rep = {"rows": (len(a), len(b))}
    if len(a) != len(b) or not (a[key].astype(str).values == b[key].astype(str).values).all():
        ka = set(map(tuple, a[key].astype(str).values))
        kb = set(map(tuple, b[key].astype(str).values))
        print(f"{name}: KEY MISMATCH only A {len(ka - kb)} only B {len(kb - ka)}")
        a = a[a[key].astype(str).apply(tuple, axis=1).isin(ka & kb)].reset_index(drop=True)
        b = b[b[key].astype(str).apply(tuple, axis=1).isin(ka & kb)].reset_index(drop=True)
    bad = {}
    for c in a.columns:
        if c not in b.columns:
            bad[c] = "missing in B"
            continue
        x, y = a[c], b[c]
        if x.dtype.kind == "f" or y.dtype.kind == "f":
            xv = pd.to_numeric(x, errors="coerce").to_numpy(float)
            yv = pd.to_numeric(y, errors="coerce").to_numpy(float)
            same = (xv == yv) | (np.isnan(xv) & np.isnan(yv))
            if not same.all():
                d = np.abs(xv - yv)
                d = d[~same]
                bad[c] = f"{(~same).sum()} rows differ, max |d| {np.nanmax(d) if np.isfinite(d).any() else 'nan-mismatch'}"
        else:
            na_x, na_y = x.isna().to_numpy(), y.isna().to_numpy()
            xs, ys = x.astype(str).to_numpy(), y.astype(str).to_numpy()
            ne = (na_x != na_y) | (~na_x & ~na_y & (xs != ys))
            if str(x.dtype) != str(y.dtype):
                bad.setdefault("_dtypes", []).append(f"{c}: {x.dtype} vs {y.dtype}")
            if ne.any():
                i = np.flatnonzero(ne)[0]
                bad[c] = f"{ne.sum()} rows differ, e.g. {xs[i]!r} vs {ys[i]!r}"
    extra = [c for c in b.columns if c not in a.columns]
    print(f"{name}: rows {rep['rows']}, {'IDENTICAL' if not bad and not extra else 'DIFFERENT'}")
    dt = bad.pop("_dtypes", None)
    for c, v in bad.items():
        print(f"   {c}: {v}")
    if dt:
        print("   (dtype only, values equal):", "; ".join(dt))
    if extra:
        print("   extra columns in B:", extra)
    return not bad


def main():
    A, B = sys.argv[1], sys.argv[2]
    da, db, pa, pb = load(A, "det"), load(B, "det"), load(A, "ph"), load(B, "ph")
    if "--common" in sys.argv:          # only the signals present in both
        ids = set(da.DeviceId) & set(db.DeviceId)
        da, db = da[da.DeviceId.isin(ids)], db[db.DeviceId.isin(ids)]
        pa, pb = pa[pa.DeviceId.isin(ids)], pb[pb.DeviceId.isin(ids)]
        print(f"{len(ids)} common signals")
    ok1 = compare(da, db, ["DeviceId", "Detector"], "detectors")
    ok2 = compare(pa, pb, ["DeviceId", "phase"], "phases") if len(pa) or len(pb) else True
    sys.exit(0 if ok1 and ok2 else 1)


if __name__ == "__main__":
    main()
