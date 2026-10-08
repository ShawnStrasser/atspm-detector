"""124v: health v4 in the package vs the research scorer (resolved_v4d = score_v4c + note 121) on 5 review signals,
ALL 8 research windows (m30_a-d were not in note 124's comparison).  Uses note 124's harness (cmp124.run) for the
classifier inputs, but its own strict comparison: unmatched rows count as differences, every common column is listed.
    python health5.py research|package|pkg_oof"""
import importlib.util, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
W = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work")); O = W / "s124v"
spec = importlib.util.spec_from_file_location("cmp124", Path(__file__).resolve().parents[1] / "final124" / "cmp124.py")
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
mode = sys.argv[1]
allsig = C.signals()
rng = np.random.default_rng(124)
sigs = sorted(rng.choice(allsig, 5, replace=False).tolist())
lk = set(pd.read_csv(W / "official" / "locked_v2.csv").DeviceId.str.lower()); assert not set(sigs) & lk
wins = ["m30_a", "m30_b", "m30_c", "m30_d", "h3_a", "h3_b", "h24_a", "h24_b"]
A, N, times = C.run(mode, sigs, wins)
Rr = pd.read_parquet(W / "s121" / "resolved_v4d.parquet")
Rr = Rr[Rr.DeviceId.isin(sigs) & Rr.window.isin(wins)]
M = A.merge(Rr, on=C.KEY, how="outer", suffixes=("", "_r"), indicator=True)
res = {"mode": mode, "signals": sigs, "rows_pkg": len(A), "rows_research": len(Rr),
       "merge": M._merge.astype(str).value_counts().to_dict()}
# research rows with actuations but no package row (or vice versa) are listed
un = M[M._merge != "both"]
res["unmatched_examples"] = un[C.KEY + ["_merge"]].astype(str).head(10).values.tolist()
res["unmatched_research_only_by_window"] = un[un._merge == "right_only"].window.value_counts().to_dict()
M = M[M._merge == "both"].copy()
common = sorted(c for c in A.columns if c + "_r" in M.columns and c not in C.KEY)
res["n_common_cols"] = len(common)
diff = {}
for c in common:
    a, b = M[c], M[c + "_r"]
    try:
        x, y = pd.to_numeric(a, errors="raise").to_numpy(float), pd.to_numeric(b, errors="raise").to_numpy(float)
        both = np.isfinite(x) & np.isfinite(y)
        rel = np.where(both, np.abs(x - y) / np.maximum(np.abs(y), 1e-9), 0)
        bad = (np.isfinite(x) != np.isfinite(y)) | (rel > 1e-9)
    except (ValueError, TypeError):
        f = lambda s: s.fillna("").astype(str).replace({"nan": "", "None": "", "<NA>": ""})
        bad = (f(a) != f(b)).to_numpy()
    if bad.any():
        r = M[bad].iloc[0]
        diff[c] = [int(bad.sum()), r.DeviceId[:8], r.window, int(r.detector), str(r[c])[:80], str(r[c + "_r"])[:80]]
res["cols_differ"] = diff
res["status_equal"] = {w: [int((g.st_v4d.astype(str) == g.st_v4d_r.astype(str)).sum()), len(g)] for w, g in M.groupby("window")}
res["status_counts_pkg"] = M.st_v4d.astype(str).value_counts().to_dict()
M.drop(columns=["_merge"]).astype(str).to_parquet(O / f"health5_{mode}_M.parquet")
s = json.dumps(res, indent=1, default=str); print(s)
(O / f"health5_{mode}.json").write_text(s)
