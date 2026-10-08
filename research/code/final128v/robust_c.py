"""Note 115dv: extra adversarial checks on NEW signals (n05 20 ch, n07 27 ch, n02 12 ch).  python robust_c.py <tag>
SRC=<dir with detector_classifier> or the installed package."""
import os as _os
from pathlib import Path as _P
DCW = _os.environ.get("DC_WORK", str(_P.home() / "dc_work"))
import os, sys, json, time, warnings, traceback
from pathlib import Path
import numpy as np, pandas as pd, duckdb
if os.environ.get("SRC"):
    sys.path.insert(0, os.environ["SRC"])
from detector_classifier import predict
import detector_classifier
W = Path(DCW); D = W / "s115dv"; R = W / "s128v" / "rob"; R.mkdir(exist_ok=True)
TAG = sys.argv[1] if len(sys.argv) > 1 else "c"
con = duckdb.connect(); con.execute("set threads=4; set memory_limit='4GB'")
ev = lambda c: (D / "ev" / f"{c}.parquet").as_posix()
lk = set(pd.read_csv(W / "official" / "locked_v2.csv").DeviceId.str.lower())
S, E = "2026-09-20 15:00:00", "2026-09-20 18:00:00"
def load(c, s=S, e=E):
    d = con.sql(f"select * from '{ev(c)}' where Timestamp >= '{s}' and Timestamp < '{e}'").df()
    assert d.DeviceId.str.lower().iloc[0] not in lk
    return d
A, B, C = load("n05"), load("n07", "2026-09-20 07:00:00", "2026-09-20 08:30:00"), load("n02")
res = {"module": detector_classifier.__file__}


def norm(o):
    o = o.sort_values(["DeviceId", "Detector"]).reset_index(drop=True)
    return o.astype(object).where(o.notna(), None)


def same(a, b, skip=()):
    a, b = norm(a), norm(b)
    if len(a) != len(b) or [int(x) for x in a.Detector] != [int(x) for x in b.Detector]:
        return f"rows {len(a)} vs {len(b)}"
    bad = []
    for c in a.columns:
        if c in skip:
            continue
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


rA, rB, rC = (predict(x, min_actuations=1) for x in (A, B, C))
P = lambda d, **k: predict(d, min_actuations=1, **k)
# batching: different windows, different lengths, plus a signal with no begin-green and one with only color events
def batch_mixed():
    nog = C[C.EventId != 1]; r_nog = P(nog)
    onlyc = load("n00")[lambda d: ~d.EventId.isin([81, 82])]
    both = P(pd.concat([A, B, nog, onlyc]))
    ch = P(pd.concat([A, B, nog, onlyc]), chunk_signals=2)
    alone = pd.concat([rA, rB, r_nog])
    return f"joint {same(alone, both)}; chunked {same(alone, ch)}; rows {len(both)}"
run("batch_mixed_windows_and_broken_signals", batch_mixed)
run("batch_le2h_3sig", lambda: same(pd.concat([P(A), P(B), P(C)]),
                                    P(pd.concat([C, A, B]).sample(frac=1, random_state=5))))
def glob2():
    g = R / "glob"; g.mkdir(exist_ok=True)
    for n, d in (("a", A), ("c", C)):
        con.register("x", d); con.execute(f"COPY x TO '{(g / f'{n}.parquet').as_posix()}' (FORMAT parquet)"); con.unregister("x")
    return same(pd.concat([rA, rC]), predict(str(g / "*.parquet"), min_actuations=1))
run("glob_two_files", glob2)
# ids and odd strings
run("deviceid_apostrophe_unicode", lambda: same(rA, P(A.assign(DeviceId="o'hare #1 é")), skip=("DeviceId",)))
run("deviceid_apostrophe_batched", lambda: same(pd.concat([rA.assign(DeviceId="x'1"), rC.assign(DeviceId="y''2")]),
     P(pd.concat([A.assign(DeviceId="x'1"), C.assign(DeviceId="y''2")]))))
# time zones
run("tz_utc_aware_df", lambda: same(rA, P(A.assign(Timestamp=A.Timestamp.dt.tz_localize("UTC")))))
run("tz_two_signals_two_zones", lambda: same(pd.concat([rA, rC]), P(pd.concat([
    A.assign(Timestamp=A.Timestamp.dt.tz_localize("America/Indiana/Indianapolis")),
    C.assign(Timestamp=C.Timestamp.dt.tz_localize("America/Indiana/Indianapolis"))]))))
run("start_end_tz_timestamps", lambda: same(predict(ev("n05"), start=S, end=E, min_actuations=1),
    predict(ev("n05"), start=pd.Timestamp(S, tz="America/Chicago"), end=pd.Timestamp(E, tz="America/Chicago"), min_actuations=1)))
def iso_t_csv():
    f = R / "iso_t.csv"; A.assign(Timestamp=A.Timestamp.dt.strftime("%Y-%m-%dT%H:%M:%S.%f")).to_csv(f, index=False)
    return same(rA, predict(str(f), min_actuations=1))
run("csv_iso_T_timestamps", iso_t_csv)
run("df_vs_path_window_new", lambda: same(rA, predict(ev("n05"), start=S, end=E, min_actuations=1)))
# disallowed / fault events added: phase / function must not move
def add_codes(codes):
    rng = np.random.default_rng(1)
    x = A.sample(3000, random_state=2).copy()
    x["EventId"] = rng.choice(codes, len(x)); x["Parameter"] = rng.integers(1, 17, len(x))
    return P(pd.concat([A, x]))
run("add_disallowed_4_5_6_13_21_22_23_32_33_45_61_66_89_90_102", lambda: same(rA, add_codes([4, 5, 6, 13, 21, 22, 23, 32, 33, 45, 61, 62, 63, 64, 65, 66, 89, 90, 102, 105])))
run("add_fault_83_88", lambda: same(rA, add_codes([83, 84, 85, 86, 87, 88])))
# gaps, inverted / thin windows
def gap():
    d = A[(A.Timestamp < "2026-09-20 16:00") | (A.Timestamp >= "2026-09-20 17:30")]
    o = P(d); return f"rows {len(o)} answered {int(o.phase_pred.notna().sum())} minutes {o.minutes_of_data.iloc[0]}"
run("hole_90min", gap)
run("start_after_end", lambda: f"rows {len(predict(ev('n05'), start=E, end=S))}")
run("one_event_only", lambda: f"rows {len(P(A.head(1)))}")
def single_on():
    ch = int(A[A.EventId == 82].Parameter.iloc[0])
    d = A[~(A.EventId.isin([81, 82]) & (A.Parameter == ch))]
    one = A[A.EventId.isin([81, 82]) & (A.Parameter == ch)].head(2)
    o = predict(pd.concat([d, one]))
    r = o[o.Detector == ch]
    return f"det {ch}: {r[['phase_pred','function_pred','status','health_status']].astype(str).to_dict('records')}"
run("one_actuation_default_min", single_on)
# whole-signal mirror: channel numbers shifted +10 (bijection): answers must not change (phase-anonymous)
def shift():
    d = A.copy(); m = d.EventId.isin([81, 82]); d.loc[m, "Parameter"] = (d.loc[m, "Parameter"] - 1 + 10) % 64 + 1
    o = P(d); o["Detector"] = (o.Detector - 1 - 10) % 64 + 1
    return same(rA, o)
run("channels_rotated_plus10", shift)
# health sanity on all new-signal parity outputs: any absurd ratio in a reason text?
def health_scan():
    return "skipped (reads old s124v outputs, not this package)"
run("health_ratio_scan_new_signals", health_scan)
json.dump(res, open(R / f"robust_c_{TAG}.json", "w"), indent=1, default=str)
