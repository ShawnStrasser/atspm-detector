"""Note 134: package health with the adopted 4-h persistence rule (rule P) = research scorer score_v4d --persist.

Same method as note 124 (cmp124.run: 24 signals behind the 40 rows of review/health_review_v4.xlsx, every detector,
research classifier inputs), windows h3_a/b, h24_a/b and m30_a-d.  Research truth: s121/resolved_v4d.parquet columns
st_v4d_persist / sev_rule_persist.

    python cmp134.py [research|package]   -> %DC_WORK%/s134/cmp134_<mode>.parquet, rows134_<mode>.csv + print
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pandas as pd

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
OUT = DCW / "s134"
CODE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cmp124", CODE / "final124" / "cmp124.py")
c124 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c124)
KEY = c124.KEY


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "research"
    OUT.mkdir(parents=True, exist_ok=True)
    wins = ["h3_a", "h3_b", "h24_a", "h24_b", "m30_a", "m30_b", "m30_c", "m30_d"]
    A, _, _ = c124.run(mode, c124.signals(), wins)
    Rr = pd.read_parquet(DCW / "s121" / "resolved_v4d.parquet",
                         columns=KEY + ["st8", "st_v4d", "sev_rule", "st_v4d_persist", "sev_rule_persist"])
    M = A[KEY + ["st_v4d", "sev_rule", "health_status", "health_categories", "health_reason"]].merge(
        Rr, on=KEY, how="inner", suffixes=("", "_r"))
    M["eq_status"] = M.st_v4d.astype(str) == M.st_v4d_persist.astype(str)
    M["eq_rule"] = M.sev_rule.fillna("").astype(str) == M.sev_rule_persist.fillna("").astype(str)
    M["eq_out"] = M.health_status.astype(str) == M.st_v4d.astype(str)
    print(mode, "rows package", len(A), "matched", len(M))
    for w, g in M.groupby("window"):
        print(f"  {w}: n {len(g)} status = persist {g.eq_status.sum()} rule = {g.eq_rule.sum()} "
              f"P {int((g.sev_rule == 'P').sum())} (research {int((g.sev_rule_persist == 'P').sum())})")
    print("TOTAL status equal", int(M.eq_status.sum()), "/", len(M), "; rule equal", int(M.eq_rule.sum()),
          "; output column = st_v4d", int(M.eq_out.sum()))
    d = M[~M.eq_status | ~M.eq_rule]
    if len(d):
        print(d[KEY + ["st8", "st_v4d", "sev_rule", "st_v4d_persist", "sev_rule_persist"]].to_string(index=False))
    rows = json.loads((DCW / "s118c" / "plot" / "rows.json").read_text())
    k = pd.DataFrame([{c: r[c] for c in ("row", "DeviceId", "window", "detector")} for r in rows])
    k = k.merge(M, on=KEY, how="left")
    r121 = pd.read_csv(DCW / "s121" / "rows121.csv")[["row", "old_status", "new_status", "with_persist_rule"]]
    k = k.merge(r121, on="row", how="left")
    k["pkg_shown"] = [p if p != s8 else o for p, s8, o in zip(k.st_v4d, k.st8, k.old_status)]
    k["rows_equal"] = k.pkg_shown.astype(str) == k.with_persist_rule.astype(str)
    cols = ["row", "window", "detector", "old_status", "new_status", "with_persist_rule", "pkg_shown", "sev_rule",
            "rows_equal", "health_categories"]
    print(k[cols].to_string(index=False))
    print("sheet rows equal:", int(k.rows_equal.sum()), "/", len(k))
    for c in M.columns:
        if M[c].dtype == object:
            M[c] = M[c].astype(str)
    M.to_parquet(OUT / f"cmp134_{mode}.parquet")
    k[cols + ["health_reason"]].to_csv(OUT / f"rows134_{mode}.csv", index=False)


if __name__ == "__main__":
    main()
