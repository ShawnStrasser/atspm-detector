"""Note 128 edge cases (copy of final124/edge124.py; outputs s128; --cmp compares EVERY column): the note-120 battery (1 / 3 / 5 min, no calls, no 7-10, no 131, no Begin Green, one detector,
no detector, two signals, empty window) + health-specific lengths (24 h from 06:00, 30 h, 2 whole days, 7 days) and a
detector held ON for 3 h.  Runs one package; compare old / new with --cmp.
    python edge124.py <name> <dir containing detector_classifier>      -> s124/edge_<name>.pkl
    python edge124.py --cmp old new"""
import os, sys, time
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
W = Path(os.environ.get("DC_WORK") or Path.home() / "dc_work")
OUT = W / "s128"
HEALTH = {"health_status", "health_score", "health_reason", "health_bad_periods", "health_watch", "health_categories",
          "health_config", "health_signal_note", "status", "review_flag", "review_reason"}


def cases():
    q = lambda f: duckdb.sql(f"select * from read_parquet('{(W / 'bench71' / f).as_posix()}')").df()  # noqa: E731
    ev, ev2 = q("ev_typical_r8_h3.parquet"), q("ev_busiest_ch_m30.parquet")
    t0 = ev.Timestamp.min()
    import json
    dev = next(r["DeviceId"] for r in json.loads((W / "s118c" / "plot" / "rows.json").read_text()) if r["row"] == 2)
    lk = set(pd.read_csv(W / "official" / "locked_v2.csv").DeviceId.str.lower())
    assert dev not in lk
    w = ds.dataset(W / "health4" / "w40_events" / f"DeviceId={dev}").to_table().to_pandas().assign(DeviceId=dev)
    wk = duckdb.sql(f"select * from read_parquet('{(W / 's124v' / 'rob' / 'week_n08.parquet').as_posix()}')").df()
    stuck = w[(w.Timestamp >= "2026-09-27") & (w.Timestamp < "2026-09-28")].copy()
    ch = int(stuck[stuck.EventId == 82].Parameter.mode()[0])
    a, b = pd.Timestamp("2026-09-27 09:00"), pd.Timestamp("2026-09-27 12:00")
    m = stuck.EventId.isin([81, 82]) & (stuck.Parameter == ch) & (stuck.Timestamp >= a) & (stuck.Timestamp <= b)
    stuck = pd.concat([stuck[~m], pd.DataFrame({"DeviceId": dev, "Timestamp": [a, b], "EventId": [82, 81],
                                                "Parameter": [ch, ch]}).astype({c: stuck.dtypes[c] for c in
                                                                                  ("Timestamp", "EventId", "Parameter")})])
    C = {
        "df_lower": (ev.rename(columns=str.lower), None, None),
        "m1": (ev[ev.Timestamp < t0 + pd.Timedelta(minutes=1)], None, None),
        "m3": (ev[ev.Timestamp < t0 + pd.Timedelta(minutes=3)], None, None),
        "m5": (ev[ev.Timestamp < t0 + pd.Timedelta(minutes=5)], None, None),
        "no_calls": (ev[~ev.EventId.isin([43, 44])], None, None),
        "no_term": (ev[~ev.EventId.isin([7, 8, 9, 10])], None, None),
        "no_coord": (ev[ev.EventId != 131], None, None),
        "no_bg": (ev[ev.EventId != 1], None, None),
        "one_det": (ev[~ev.EventId.isin([81, 82]) | (ev.Parameter == ev[ev.EventId == 82].Parameter.mode()[0])], None, None),
        "no_det": (ev[~ev.EventId.isin([81, 82])], None, None),
        "one_bg": (ev[(ev.EventId != 1) | (ev.Timestamp == ev[ev.EventId == 1].Timestamp.min())], None, None),
        "two_sig": (pd.concat([ev[ev.Timestamp < t0 + pd.Timedelta(minutes=40)], ev2]), None, None),
        "empty_window": (ev, str(t0 + pd.Timedelta(hours=5)), str(t0 + pd.Timedelta(hours=6))),
        "h24_from_06": (w, "2026-09-26 06:00", "2026-09-27 06:00"),
        "h30": (w, "2026-09-27 00:00", "2026-09-28 06:00"),
        "d2_whole": (w, "2026-09-27 00:00", "2026-09-29 00:00"),
        "stuck_3h": (stuck, "2026-09-27 00:00", "2026-09-28 00:00"),
        "week": (wk, None, None),
    }
    return C


def run(name, pkg):
    sys.path.insert(0, pkg)
    import warnings
    warnings.simplefilter("ignore")
    from detector_classifier import pipeline as P
    out = []
    for k, (e, s, en) in cases().items():
        t = time.perf_counter()
        try:
            r = P.predict(e, start=s, end=en, min_actuations=1, chunk_signals=1 if k == "two_sig" else None)
            r = r.assign(case=k)
        except Exception as exc:                                    # noqa: BLE001
            import traceback
            traceback.print_exc()
            r = pd.DataFrame({"case": [k], "status": [f"EXC {type(exc).__name__}: {exc}"]})
        print(k, len(r), f"{time.perf_counter() - t:.1f}s",
              r.health_status.value_counts().to_dict() if "health_status" in r else "", flush=True)
        out.append(r)
    pd.concat(out, ignore_index=True).to_pickle(OUT / f"edge_{name}.pkl")


def cmp(a, b):
    A, B = pd.read_pickle(OUT / f"edge_{a}.pkl"), pd.read_pickle(OUT / f"edge_{b}.pkl")
    print("exceptions:", A.status.astype(str).str.startswith("EXC").sum(), B.status.astype(str).str.startswith("EXC").sum())
    k = ["case", "DeviceId", "Detector"]
    A, B = A[A.Detector.notna()], B[B.Detector.notna()]
    m = A.merge(B, on=k, suffixes=("_a", "_b"))
    print("rows", len(A), len(B), "matched", len(m))
    bad = {}
    for c in A.columns:
        if c in k or c + "_b" not in m:
            continue
        x, y = m[c + "_a"], m[c + "_b"]
        try:
            xx, yy = x.astype(float).to_numpy(), y.astype(float).to_numpy()
            d = ~((xx == yy) | (np.isnan(xx) & np.isnan(yy)))
        except (TypeError, ValueError):
            d = ~((x.astype(str) == y.astype(str)) | (x.isna() & y.isna()))
        if d.sum():
            bad[c] = int(d.sum())
    print("columns differing (health included):", bad or "none")
    for c, g in m.groupby("case"):
        print(c, "health old", g.health_status_a.value_counts().to_dict(), "new", g.health_status_b.value_counts().to_dict())
    w = B[B.case == "stuck_3h"]
    print(w[w.health_categories.astype(str).str.contains("Stuck")][["Detector", "health_status", "health_reason"]].to_string())


if __name__ == "__main__":
    if sys.argv[1] == "--cmp":
        cmp(sys.argv[2], sys.argv[3])
    else:
        run(sys.argv[1], sys.argv[2])
