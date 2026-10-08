"""124v: predict() -> health wiring on the 5 review signals.  (1) the events predict hands to health_v4.assess equal
cmp124.prep of the raw log; (2) predict's health columns equal assess re-run on the captured inputs; (3) info: predict's
health status vs the research scorer (classifier inputs differ: package answers vs research OOF inputs)."""
import importlib.util, json, os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, duckdb
warnings.simplefilter("ignore")
W = Path.home() / "dc_work"; O = W / "s124v"
os.environ["DC_PKG"] = str(W / "final_v7_prod" / "src")
spec = importlib.util.spec_from_file_location("cmp124", Path(__file__).resolve().parents[1] / "final124" / "cmp124.py")
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
from detector_classifier import pipeline as PL, health_v4 as H4
sigs = json.loads((O / "health5_research.json").read_text())["signals"]
cap = []
orig = PL.hv4.assess
def wrap(*a, **k):
    r = orig(*a, **k); cap.append((a, k, r)); return r
PL.hv4.assess = wrap
Rr = pd.read_parquet(W / "s121" / "resolved_v4d.parquet")
res = []
for dev in sigs:
    e = C.load_events(dev)
    f = O / "tmp_w40.parquet"
    src = next(p for p in C.EVD.iterdir() if p.name.lower() == f"deviceid={dev}")
    duckdb.sql(f"COPY (select '{dev}' as DeviceId, Timestamp, EventId, Parameter from read_parquet('{src.as_posix()}/*.parquet')) TO '{f.as_posix()}' (FORMAT parquet)")
    for w in ("m30_a", "m30_d", "h3_a", "h24_a", "h24_b"):
        s, h = C.WIN[w]; t0 = pd.Timestamp(s); t1 = t0 + pd.Timedelta(hours=h)
        cap.clear()
        out = PL.predict(str(f), start=str(t0), end=str(t1), min_actuations=1)
        (a, k, (ht, note)), = cap
        P = a[0]; Q = C.prep(e, t0, t1)
        same_ev = all(np.array_equal(getattr(P, x), getattr(Q, x)) for x in ("t", "eid", "par", "td", "ed", "pd_", "g0", "g1"))
        ht2, note2 = H4.assess(*a, **k)
        m = out.merge(ht2.rename(columns={"detector": "Detector"}), on="Detector", how="left", suffixes=("", "_re"))
        neq = {}
        for c in ("health_status", "health_score", "health_reason", "health_categories", "health_config", "health_watch"):
            x, y = m[c].astype(str).replace({"nan": "", "None": "", "<NA>": ""}), m[c + "_re"].astype(str).replace({"nan": "", "None": "", "<NA>": ""})
            if c == "health_score":
                x = x.where(m.health_status.ne("not_enough_data"), ""); y = y.where(m.health_status.ne("not_enough_data"), "")
            neq[c] = int((x != y).sum())
        r = Rr[(Rr.DeviceId == dev) & (Rr.window == w)][["detector", "st_v4d"]].rename(columns={"detector": "Detector"})
        mm = out.merge(r, on="Detector", how="inner")
        res.append(dict(dev=dev[:8], window=w, n_det=len(out), events_equal=bool(same_ev), n_ev=len(P.t),
                        predict_vs_assess_differ=neq, note_equal=bool(out.health_signal_note.fillna("").eq(note).all()),
                        vs_research_status_equal=f"{int((mm.health_status == mm.st_v4d).sum())}/{len(mm)}"))
        print(json.dumps(res[-1]), flush=True)
json.dump(res, open(O / "wiring.json", "w"), indent=1)
