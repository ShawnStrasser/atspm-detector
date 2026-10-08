"""124v: exact cell comparison of two runs: python cmp.py <A> <B> [--json out]. Floats compared with == (NaN == NaN)."""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, duckdb
W = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work")); OUT = W / "s124v" / "out"
HEALTHISH = lambda c: c.startswith("health_") or c in ("status", "review_flag", "review_reason")
KEYS = {"det": ["case", "DeviceId", "Detector"], "cand": ["case", "DeviceId", "Detector", "cand_phase"],
        "ph": ["case", "DeviceId", "phase"]}


def eq(a, b):
    an, bn = pd.isna(a), pd.isna(b)
    try:
        x, y = pd.to_numeric(a, errors="raise").to_numpy(float), pd.to_numeric(b, errors="raise").to_numpy(float)
        return (x == y) | (np.isnan(x) & np.isnan(y))
    except (ValueError, TypeError):
        return ((a.astype(str) == b.astype(str)) & ~an & ~bn).to_numpy() | (an & bn).to_numpy()


def main():
    A, B = sys.argv[1], sys.argv[2]
    res = {"A": A, "B": B}
    for k, key in KEYS.items():
        a = duckdb.sql(f"select * from '{(OUT / f'{A}_{k}.parquet').as_posix()}'").df()
        b = duckdb.sql(f"select * from '{(OUT / f'{B}_{k}.parquet').as_posix()}'").df()
        for x in (a, b):
            for c in key[2:]:
                x[c] = x[c].astype("int64")
        m = a.merge(b, on=key, how="outer", suffixes=("_a", "_b"), indicator=True)
        r = {"rows": [len(a), len(b)], "unmatched": int((m._merge != "both").sum()),
             "only_in_A_cols": sorted(set(a.columns) - set(b.columns)), "only_in_B_cols": sorted(set(b.columns) - set(a.columns))}
        m = m[m._merge == "both"]
        diff, ex = {}, {}
        for c in sorted((set(a.columns) & set(b.columns)) - set(key)):
            ok = eq(m[c + "_a"].reset_index(drop=True), m[c + "_b"].reset_index(drop=True))
            n = int((~ok).sum())
            if n:
                diff[c] = n
                i = np.flatnonzero(~ok)[0]; rr = m.iloc[i]
                ex[c] = [str(rr["case"]), int(rr[key[2]]), str(rr[c + "_a"])[:160], str(rr[c + "_b"])[:160]]
        r["n_common_cols"] = len((set(a.columns) & set(b.columns)) - set(key))
        r["nonhealth_cols_differ"] = {c: n for c, n in diff.items() if not HEALTHISH(c)}
        r["health_cols_differ"] = {c: n for c, n in diff.items() if HEALTHISH(c)}
        r["examples"] = ex
        if k == "det":
            r["n_cases"] = int(m.case.nunique()); r["n_det"] = len(m)
            r["n_nonhealth_cols"] = sum(not HEALTHISH(c) for c in (set(a.columns) & set(b.columns)) - set(key))
        res[k] = r
    s = json.dumps(res, indent=1, default=str); print(s)
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(s)


if __name__ == "__main__":
    main()
