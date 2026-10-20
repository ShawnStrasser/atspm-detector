"""Note 153: final health outputs of the SQL (single query or health() script mode) against the captured numpy
outputs of note 152 (out / signal note per signal): python out153.py <set> [n] [query|script]"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dev153 as D, pandas as pd, numpy as np, json
from atspm_detector import health_duckdb as HQ
rec = D.load(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != "0" else None)
mode = sys.argv[3] if len(sys.argv) > 3 else "script"
con = D.connect(rec)
t0 = time.perf_counter()
if mode == "query":
    got = con.sql(HQ.build("events", "detectors", "windows").text(HQ.FINAL)).df()
else:
    got = HQ.health(con, "events", "detectors", windows_table="windows")
print(mode, round(time.perf_counter() - t0, 2), "s", len(got))
E = []
for i, r in enumerate(rec):
    o = r["out"].copy(); o.insert(0, "DeviceId", f"S{i:03d}"); o["health_signal_note"] = r["note"]
    o["health_bad_periods"] = o.health_bad_periods.map(lambda L: "" if not isinstance(L, list) or not L else json.dumps(
        [{"start": str(d["start"]), "end": str(d["end"]), "what": d["what"], "recovered": d.get("recovered")} for d in L]))
    o.loc[o.health_status.eq("not_enough_data"), "health_score"] = np.nan
    E.append(o)
E = pd.concat(E, ignore_index=True).rename(columns={"detector": "Detector"})
E["health_score"] = E.health_score.astype(float)
cols = ["health_status", "health_categories", "health_score", "health_reason", "health_bad_periods", "health_config",
        "health_watch", "health_signal_note"]
bad = D.compare(got, E, cols)
print("rows main", len(E), "sql", len(got), "tables left", con.sql("select count(*) from duckdb_tables() where table_name like 'hq_%'").fetchone())
for c, d in bad.items():
    for _, r in d.head(4).iterrows():
        print(c, r.DeviceId, r.Detector, "\n  main:", r[c], "\n  sql :", r[c + "_sql"])
