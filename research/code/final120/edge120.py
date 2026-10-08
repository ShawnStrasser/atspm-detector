"""Note 120 edge cases (note-115 edge115 set + empty window + 1 / 5 min) with a probe of the models each case loads.
    python edge120.py <name> <dir containing detector_classifier> [profile|-]   -> s120/edge_<name>.pkl + probe json"""
import json, os, sys
from pathlib import Path
import pandas as pd, duckdb
W = Path(os.environ.get("DC_WORK") or Path.home() / "dc_work")
name, pkg = sys.argv[1], sys.argv[2]
profile = sys.argv[3] if len(sys.argv) > 3 else "-"
sys.path.insert(0, pkg)
from detector_classifier import pipeline as P, trees_onnx, stacker, function_stage as fs
probe = {"onnx": set(), "variant": set(), "net": 0}
o_load, o_bag, o_net = trees_onnx.load, stacker.Stacker.bag, fs.net_model
def load(path, *a, **k):
    probe["onnx"].add(Path(path).name); return o_load(path, *a, **k)
def bag(self, variant=None):
    probe["variant"].add(variant or getattr(self, "variant", "mean3")); return o_bag(self, variant) if variant is not None else o_bag(self)
def net_model(*a, **k):
    probe["net"] += 1; return o_net(*a, **k)
trees_onnx.load = P.trees_onnx.load = load
stacker.Stacker.bag = bag
fs.net_model = P.fs.net_model = net_model
q = lambda f: duckdb.sql(f"select * from read_parquet('{(W / 'bench71' / f).as_posix()}')").df()
ev, ev2 = q("ev_typical_r8_h3.parquet"), q("ev_busiest_ch_m30.parquet")
t0 = ev.Timestamp.min()
cases = {
  "df_lower": ev.rename(columns=str.lower),
  "m1": ev[ev.Timestamp < t0 + pd.Timedelta(minutes=1)],
  "m3": ev[ev.Timestamp < t0 + pd.Timedelta(minutes=3)],
  "m5": ev[ev.Timestamp < t0 + pd.Timedelta(minutes=5)],
  "no_calls": ev[~ev.EventId.isin([43, 44])],
  "no_term": ev[~ev.EventId.isin([7, 8, 9, 10])],
  "no_coord": ev[ev.EventId != 131],
  "no_bg": ev[ev.EventId != 1],
  "one_det": ev[~ev.EventId.isin([81, 82]) | (ev.Parameter == ev[ev.EventId == 82].Parameter.mode()[0])],
  "no_det": ev[~ev.EventId.isin([81, 82])],
  "one_bg": ev[(ev.EventId != 1) | (ev.Timestamp == ev[ev.EventId == 1].Timestamp.min())],
  "two_sig": pd.concat([ev[ev.Timestamp < t0 + pd.Timedelta(minutes=40)], ev2]),
}
kw = {} if profile == "-" else {"profile": profile}
out, pr = [], {}
for k, e in list(cases.items()) + [("empty_window", None)]:
    probe["onnx"], probe["variant"], probe["net"] = set(), set(), 0
    try:
        if k == "empty_window":
            r = P.predict(ev, start=str(t0 + pd.Timedelta(hours=5)), end=str(t0 + pd.Timedelta(hours=6)), min_actuations=1, **kw)
        else:
            r = P.predict(e, min_actuations=1, chunk_signals=1 if k == "two_sig" else None, **kw)
        r = r.assign(case=k)
    except Exception as exc:
        r = pd.DataFrame({"case": [k], "status": [f"EXC {type(exc).__name__}: {exc}"]})
    pr[k] = {"onnx": sorted(x for x in probe["onnx"] if "decode" in x), "variant": sorted(probe["variant"]), "net": probe["net"]}
    print(k, len(r), pr[k], flush=True)
    out.append(r)
pd.concat(out, ignore_index=True).to_pickle(W / "s120" / f"edge_{name}.pkl")
json.dump(pr, open(W / "s120" / f"edge_{name}_probe.json", "w"), indent=1)
