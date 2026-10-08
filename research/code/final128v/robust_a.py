"""Robustness tests of the installed v7 wheel."""
import os as _os
from pathlib import Path as _P
DCW = _os.environ.get("DC_WORK", str(_P.home() / "dc_work"))
import sys, json, time, warnings, traceback, os
from pathlib import Path
import numpy as np, pandas as pd, duckdb
warnings.simplefilter("ignore")  # (copy of s115v/robust.py with paths parameterised)
if os.environ.get("PKG"):
    sys.path.insert(0, os.environ["PKG"]); from detector_classifier import predict
else:
    from detector_classifier import predict
W = Path(DCW); V = W / "s115v"; R = W / "s128v" / "rob"; R.mkdir(exist_ok=True)
con = duckdb.connect(); con.execute("set threads=4; set memory_limit='4GB'")
F = os.environ.get("EVF", (V / "ev" / "v08.parquet").as_posix())
S, E = os.environ.get("WIN_S", "2026-09-20 09:00:00"), os.environ.get("WIN_E", "2026-09-20 12:00:00")
base_df = con.sql(f"select * from '{F}' where Timestamp >= '{S}' and Timestamp < '{E}'").df()
print("base rows", len(base_df), base_df.dtypes.to_dict(), flush=True)
res = {}
ALLOWED = [1, 7, 8, 9, 10, 11, 43, 44, 81, 82, 131, 150, 173]


def norm(o):
    o = o.sort_values(["DeviceId", "Detector"]).reset_index(drop=True)
    return o.astype(object).where(o.notna(), None)


def same(a, b, cols=None):
    a, b = norm(a), norm(b)
    if len(a) != len(b) or [int(x) for x in a.Detector] != [int(x) for x in b.Detector]:
        return f"rows {len(a)} vs {len(b)}"
    bad = []
    for c in (cols or a.columns):
        for u, v in zip(a[c], b[c]):
            if u is None and v is None:
                continue
            if isinstance(u, float) and isinstance(v, float):
                if abs(u - v) > 1e-9:
                    bad.append(f"{c} ({u} vs {v})"); break
            elif str(u) != str(v):
                bad.append(f"{c} ({str(u)[:50]!r} vs {str(v)[:50]!r})"); break
    return "identical" if not bad else f"differ: {bad}"


def run(name, fn):
    t = time.perf_counter()
    try:
        res[name] = fn()
    except Exception as e:
        res[name] = f"EXCEPTION {type(e).__name__}: {str(e)[:200]}"
        traceback.print_exc()
    print(f"{name}: {res[name]}  ({time.perf_counter()-t:.1f}s)", flush=True)


ref = predict(base_df, min_actuations=1)
run("df_vs_path_same_window", lambda: same(ref, predict(F, start=S, end=E, min_actuations=1)))
run("empty_df", lambda: (lambda o: f"rows {len(o)} ncols {len(o.columns)}")(predict(base_df.iloc[:0], min_actuations=1)))


def empty_file():
    f = R / "empty.parquet"
    con.execute(f"COPY (select * from '{F}' limit 0) TO '{f.as_posix()}' (FORMAT parquet)")
    o, ph = predict(str(f), return_phases=True)
    return f"rows {len(o)}, phases {len(ph)}"


run("empty_parquet", empty_file)
run("window_without_events", lambda: f"rows {len(predict(F, start='2030-01-01', end='2030-01-02'))}")
run("only_disallowed_events", lambda: f"rows {len(predict(base_df.assign(EventId=4)))}")


def one_det():
    ch = int(base_df[base_df.EventId == 82].Parameter.value_counts().index[0])
    d = base_df[~base_df.EventId.isin([81, 82]) | (base_df.Parameter == ch)]
    o = predict(d, min_actuations=1)
    return o[["Detector", "phase_pred", "phase_prob", "function_pred", "lanes", "status", "health_status"]].astype(str).to_dict("records")


run("one_detector", one_det)
run("no_detector_events", lambda: f"rows {len(predict(base_df[~base_df.EventId.isin([81, 82])], min_actuations=1))}")


def no_green():
    o = predict(base_df[base_df.EventId != 1], min_actuations=1)
    return (f"rows {len(o)} status {o.status.str[:60].value_counts().to_dict()} phase_pred notna "
            f"{int(o.phase_pred.notna().sum())} health {o.health_status.value_counts().to_dict()}")


run("no_begin_green", no_green)
run("no_colour_end_events", lambda: (lambda o: f"rows {len(o)} answered {int(o.phase_pred.notna().sum())} same phase as ref {int((o.phase_pred.values == ref.sort_values('Detector').phase_pred.values).sum())}")(predict(base_df[~base_df.EventId.isin([7, 8, 9, 10, 11])], min_actuations=1).sort_values("Detector")))
run("no_calls_43_44", lambda: (lambda o: f"rows {len(o)} answered {int(o.phase_pred.notna().sum())}")(predict(base_df[~base_df.EventId.isin([43, 44])], min_actuations=1)))
run("all_rows_duplicated", lambda: same(ref, predict(pd.concat([base_df, base_df]), min_actuations=1)))
run("2pct_duplicated", lambda: same(ref, predict(pd.concat([base_df, base_df.sample(frac=.02, random_state=1)]), min_actuations=1)))


def gt64():
    d = base_df[base_df.EventId.isin([81, 82])].copy()
    d["Parameter"] = d.Parameter + 64
    o = predict(pd.concat([base_df, d]), min_actuations=1)
    return same(ref, o) + f"; max Detector {int(o.Detector.max())}"


run("channels_gt64_added", gt64)
run("shuffled_rows", lambda: same(ref, predict(base_df.sample(frac=1, random_state=3).reset_index(drop=True), min_actuations=1)))
run("lowercase_cols", lambda: same(ref, predict(base_df.rename(columns=str.lower), min_actuations=1)))
run("mixed_cols_df", lambda: same(ref, predict(base_df.rename(columns={"DeviceId": "device_id", "Timestamp": "TimeStamp", "EventId": "Event Code", "Parameter": "EventParam"}), min_actuations=1)))


def to_file(d, name, fmt):
    f = R / name
    con.register("b", d)
    con.execute(f"COPY b TO '{f.as_posix()}' " + ("(HEADER, DELIMITER ',')" if fmt == "csv" else "(FORMAT parquet)"))
    con.unregister("b")
    return str(f)


run("mixed_cols_csv", lambda: same(ref, predict(to_file(base_df.rename(columns={"DeviceId": "deviceid", "Timestamp": "time_stamp", "EventId": "event_code", "Parameter": "event_param"}), "mixed.csv", "csv"), min_actuations=1)))
run("mixed_cols_parquet", lambda: same(ref, predict(to_file(base_df.rename(columns={"DeviceId": "deviceID", "Timestamp": "TIMESTAMP", "EventId": "EventID", "Parameter": "param"}), "mixed.parquet", "pq"), min_actuations=1)))
run("string_timestamps", lambda: same(ref, predict(base_df.assign(Timestamp=base_df.Timestamp.dt.strftime("%Y-%m-%d %H:%M:%S.%f")), min_actuations=1)))
run("float_codes_extra_cols", lambda: same(ref, predict(base_df.assign(EventId=base_df.EventId.astype(float), Parameter=base_df.Parameter.astype(float), extra=1), min_actuations=1)))
run("tz_aware_timestamps", lambda: same(ref, predict(base_df.assign(Timestamp=base_df.Timestamp.dt.tz_localize("America/Indiana/Indianapolis")), min_actuations=1)))
run("missing_column", lambda: predict(base_df.drop(columns=["Parameter"])))


def two_sig():
    G = os.environ.get("EVG", (V / "ev" / "v05.parquet").as_posix())
    d2 = con.sql(f"select * from '{G}' where Timestamp >= '{S}' and Timestamp < '{E}'").df()
    r2 = predict(d2, min_actuations=1)
    both = predict(pd.concat([base_df, d2]), min_actuations=1)
    ch = predict(pd.concat([base_df, d2]), min_actuations=1, chunk_signals=1)
    return f"joint vs separate: {same(pd.concat([ref, r2]), both)}; chunked vs separate: {same(pd.concat([ref, r2]), ch)}"


run("two_signals", two_sig)
run("m5", lambda: (lambda o: f"rows {len(o)} answered {int(o.phase_pred.notna().sum())}")(predict(F, start="2026-09-20 12:00:00", end="2026-09-20 12:05:00")))
run("m1", lambda: (lambda o: f"rows {len(o)} answered {int(o.phase_pred.notna().sum())} status0 {o.status.iloc[0][:80]}")(predict(F, start="2026-09-20 12:00:00", end="2026-09-20 12:01:00")))


def stuck():
    ch = int(base_df[base_df.EventId == 82].Parameter.value_counts().index[-1])
    d = base_df[~((base_df.EventId == 81) & (base_df.Parameter == ch))]
    o = predict(d, min_actuations=1)
    r = o[o.Detector == ch].iloc[0]
    return f"det {ch}: status {r.status[:70]!r} health {r.health_status} phase {r.phase_pred}"


run("stuck_on_channel", stuck)


def param0():
    d = base_df[base_df.EventId.isin([81, 82])].head(200).copy()
    d["Parameter"] = 0
    o = predict(pd.concat([base_df, d]), min_actuations=1)
    return f"rows {len(o)} (ref {len(ref)}) detectors {sorted(int(x) for x in o.Detector)[:3]}"


run("channel_0", param0)
json.dump(res, open(R / f"robust_{os.environ.get('TAG','v7')}.json", "w"), indent=1, default=str)
