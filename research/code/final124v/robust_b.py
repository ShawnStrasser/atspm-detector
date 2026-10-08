"""Note 115d: robustness tests of the fixed v7 package (F2-F7).  SRC=<src dir> or the installed package.
    python robust115d.py <tag>"""
import os as _os
from pathlib import Path as _P
DCW = _os.environ.get("DC_WORK", str(_P.home() / "dc_work"))
import os, sys, json, time, warnings, tempfile, shutil, traceback
from pathlib import Path
import numpy as np, pandas as pd, duckdb
if os.environ.get("SRC"):
    sys.path.insert(0, os.environ["SRC"])
from detector_classifier import predict
import detector_classifier
W = Path(DCW); V = W / "s115v"; D = W / "s115dv"; R = W / "s124v" / "rob"; R.mkdir(exist_ok=True)
TAG = sys.argv[1] if len(sys.argv) > 1 else "v7d"
con = duckdb.connect(); con.execute("set threads=4; set memory_limit='4GB'")
F = (V / "ev" / "v08.parquet").as_posix()
S, E = "2026-09-20 09:00:00", "2026-09-20 12:00:00"
base = con.sql(f"select * from '{F}' where Timestamp >= '{S}' and Timestamp < '{E}'").df()
lk = set(pd.read_csv(W / "official" / "locked_v2.csv").DeviceId.str.lower())
assert base.DeviceId.str.lower().iloc[0] not in lk
res = {"module": detector_classifier.__file__}


def norm(o):
    o = o.sort_values(["DeviceId", "Detector"]).reset_index(drop=True)
    return o.astype(object).where(o.notna(), None)


def same(a, b):
    a, b = norm(a), norm(b)
    if len(a) != len(b) or [int(x) for x in a.Detector] != [int(x) for x in b.Detector]:
        return f"rows {len(a)} vs {len(b)}"
    bad = []
    for c in a.columns:
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
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            r = fn()
        except Exception as e:
            r = f"EXCEPTION {type(e).__name__}: {str(e)[:300]}"
            traceback.print_exc()
    ws = sorted({str(x.message)[:120] for x in w if not issubclass(x.category, DeprecationWarning)})
    res[name] = {"result": r, "warnings": ws}
    print(f"{name}: {r}  warnings={ws}  ({time.perf_counter()-t:.1f}s)", flush=True)


def to_file(d, name, fmt):
    f = R / name
    con.register("b", d)
    con.execute(f"COPY b TO '{f.as_posix()}' " + ("(HEADER, DELIMITER ',')" if fmt == "csv" else "(FORMAT parquet)"))
    con.unregister("b")
    return str(f)


ref = predict(base, min_actuations=1)
TZ = "America/Indiana/Indianapolis"
tzdf = base.assign(Timestamp=base.Timestamp.dt.tz_localize(TZ))
# F2
run("F2_tz_df", lambda: same(ref, predict(tzdf, min_actuations=1)))
run("F2_tz_df_utc_view", lambda: same(ref, predict(tzdf.assign(Timestamp=tzdf.Timestamp.dt.tz_convert("UTC").dt.tz_convert(TZ)), min_actuations=1)))
run("F2_tz_fixed_offset_df", lambda: same(ref, predict(base.assign(Timestamp=base.Timestamp.dt.tz_localize("-04:00")), min_actuations=1)))


def tz_parquet_pandas():
    f = R / "tz_pandas.parquet"
    tzdf.to_parquet(f, index=False) if _has_pyarrow() else None
    if not f.exists():
        return "skipped (no pyarrow to write pandas metadata)"
    return same(ref, predict(str(f), min_actuations=1))


def _has_pyarrow():
    try:
        import pyarrow  # noqa
        return True
    except ImportError:
        return False


run("F2_tz_parquet_pandas_meta", tz_parquet_pandas)


def tz_parquet_nometa():
    con.execute("SET TimeZone='UTC'")
    f = to_file(con.sql("select DeviceId, timezone('UTC', Timestamp + INTERVAL 4 HOUR) AS Timestamp, EventId, Parameter from base").df(), "tz_nometa.parquet", "pq")
    o = predict(f, min_actuations=1)
    return same(ref, o)


run("F2_tz_parquet_no_zone_UTC_wallclock_note", tz_parquet_nometa)


def tz_csv():
    d = base.copy()
    d["Timestamp"] = d.Timestamp.dt.strftime("%Y-%m-%d %H:%M:%S.%f") + "-04:00"
    f = R / "tz_offsets.csv"
    d.to_csv(f, index=False)
    return same(ref, predict(str(f), min_actuations=1))


run("F2_csv_with_offsets", tz_csv)
run("F2_start_end_with_offset", lambda: same(predict(F, start=S, end=E, min_actuations=1),
                                             predict(F, start=S + "-04:00", end=E + "-04:00", min_actuations=1)))
# F3


def bad_rows():
    x = base.head(3).copy()
    x["EventId"] = [131, 150, 1]
    x["Parameter"] = [-1, 70000, 99999]
    x2 = pd.DataFrame({"DeviceId": [base.DeviceId.iloc[0]] * 2, "Timestamp": [base.Timestamp.iloc[5], pd.NaT],
                       "EventId": [82, 81], "Parameter": [np.nan, 3]})
    o = predict(pd.concat([base, x, x2], ignore_index=True), min_actuations=1)
    return same(ref, o)


run("F3_bad_values_df", bad_rows)


def bad_csv():
    d = base.copy().astype({"EventId": str, "Parameter": str})
    extra = pd.DataFrame({"DeviceId": [d.DeviceId.iloc[0]] * 3, "Timestamp": [d.Timestamp.iloc[0], "not a time", d.Timestamp.iloc[1]],
                          "EventId": ["abc", "82", "131"], "Parameter": ["1", "4", "-1"]})
    f = R / "bad.csv"
    pd.concat([d, extra]).to_csv(f, index=False)
    return same(ref, predict(str(f), min_actuations=1))


run("F3_bad_values_csv", bad_csv)


# F4
def apostrophe():
    d = R / "o'brien dir"
    d.mkdir(exist_ok=True)
    f = d / "ev's.parquet"
    shutil.copy(W / "s115v" / "ev" / "v08.parquet", f)
    old = tempfile.tempdir
    tempfile.tempdir = str(R / "tmp o'neil")
    Path(tempfile.tempdir).mkdir(exist_ok=True)
    try:
        o = predict(str(f), start=S, end=E, min_actuations=1)
        c = d / "ev's.csv"
        base.to_csv(c, index=False)
        o2 = predict(str(c), min_actuations=1)
    finally:
        tempfile.tempdir = old
    return f"parquet: {same(predict(F, start=S, end=E, min_actuations=1), o)}; csv: {same(ref, o2)}"


run("F4_apostrophe_paths", apostrophe)


# F5
def batch():
    out = []
    for case in ("v05", "v12"):
        G = (V / "ev" / f"{case}.parquet").as_posix()
        d2 = con.sql(f"select * from '{G}' where Timestamp >= '2026-09-20 09:30:00' and Timestamp < '2026-09-20 11:00:00'").df()
        alone = predict(d2, min_actuations=1)
        both = predict(pd.concat([base, d2]), min_actuations=1)
        ch = predict(pd.concat([base, d2]), min_actuations=1, chunk_signals=1)
        out.append(f"{case}: joint vs separate {same(pd.concat([ref, alone]), both)}; chunked {same(pd.concat([ref, alone]), ch)}")
    return " | ".join(out)


run("F5_batched_vs_alone", batch)


def batch_files():
    a = (V / "ev" / "v08.parquet").as_posix(); b = (V / "ev" / "v03.parquet").as_posix()
    f = to_file(con.sql(f"select * from '{a}' union all select * from '{b}'").df(), "two.parquet", "pq")
    o = predict(f, start=S, end=E, min_actuations=1)
    oa = predict(a, start=S, end=E, min_actuations=1)
    ob = predict(b, start=S, end=E, min_actuations=1)
    return same(pd.concat([oa, ob]), o)


run("F5_batched_file_3h", batch_files)


# F6
def f6():
    o = predict(F, start=S, end=E, min_actuations=1)
    r = o[o.Detector.isin([16, 17])]
    return r[["Detector", "health_status", "health_reason"]].astype(str).to_dict("records")


run("F6_v08_dets_16_17", f6)


# F7
def chan0():
    d = base[base.EventId.isin([81, 82])].head(200).copy()
    d["Parameter"] = 0
    o = predict(pd.concat([base, d]), min_actuations=1)
    return f"rows {len(o)} (ref {len(ref)}), min detector {int(o.Detector.min())}; {same(ref, o)}"


run("F7_channel_0", chan0)


def stuck(kind):
    ch = int(base[base.EventId == 82].Parameter.value_counts().index[-1])
    if kind == "no81":       # every 81 of the channel removed (the verifier's case)
        d = base[~((base.EventId == 81) & (base.Parameter == ch))]
    else:                    # one 82 and nothing after it: stuck ON for the last 2 h 45 min
        t = base.Timestamp.min() + pd.Timedelta(minutes=15)
        keep = ~(base.EventId.isin([81, 82]) & (base.Parameter == ch) & (base.Timestamp >= t))
        one = pd.DataFrame({"DeviceId": [base.DeviceId.iloc[0]], "Timestamp": [t], "EventId": [82], "Parameter": [ch]})
        d = pd.concat([base[keep], one], ignore_index=True)
    o = predict(d, min_actuations=1)
    r = o[o.Detector == ch].iloc[0]
    oth = same(ref[ref.Detector != ch][["DeviceId", "Detector", "phase_pred", "function_pred"]],
               o[o.Detector != ch][["DeviceId", "Detector", "phase_pred", "function_pred"]])
    return f"det {ch}: status {r.status[:60]!r} review {r.review_reason!r} health {r.health_status}; others phase/function {oth}"


run("F7_stuck_on_no81", lambda: stuck("no81"))
run("F7_stuck_on_one82", lambda: stuck("one82"))
json.dump(res, open(R / f"robust_{TAG}.json", "w"), indent=1, default=str)
