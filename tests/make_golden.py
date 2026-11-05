"""Regenerate tests/golden (stage 0 of the incremental-health design): the frozen oracles' outputs (tests/legacy) on the
bundled sample and the synthetic sets of tests/synth.py.   python tests/make_golden.py   (PYTHONPATH: the package)

Golden cases (name -> what runs; see CASES): vehicle health with periods on the whole log, on an odd window and with
every option off but occ_strict (1.2.0 without its imputed time ON, removed in 2.0.0; on synth_a the same output); ped health with periods, with the movement model, with every option
off; ped assignments with their candidates; ped_detectors.  Files go through DuckDB's parquet writer (no pyarrow
needed); manifest.json holds a fingerprint of each input, so a changed generator (or numpy stream) shows up as such."""
from __future__ import annotations

import hashlib
import json
import os
import sys

import duckdb
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(HERE, "golden")
sys.path.insert(0, HERE)

import synth  # noqa: E402

SAMPLE_DET = os.path.join(GOLD, "sample_detectors.parquet")
WIN = ("2026-10-05 05:17", "2026-10-06 21:01")


def inputs(name: str):
    """(events, detectors) of an input set: 'sample' (the bundled 30-min log, with its stored detectors table),
    'synth_a' (3 signals x 2 days, faults), 'synth_b' (2 signals x 1 day, no faults)."""
    if name == "sample":
        from atspm_detector.check import SAMPLE, read_parquet
        return read_parquet(SAMPLE), read_pq(SAMPLE_DET)
    if name == "synth_a":
        return synth.make(3, 2, seed=0)
    if name == "synth_b":
        return synth.make(2, 1, seed=1, start="2026-10-10", faults=False)
    raise KeyError(name)


def connect(ev, det) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET threads = 4")
    con.register("ev_df", ev)
    con.register("det_df", det)
    con.execute("CREATE TABLE events AS SELECT * FROM ev_df; CREATE TABLE detectors AS SELECT * FROM det_df")
    con.unregister("ev_df"), con.unregister("det_df")
    return con


def steps(M, con, names, **kw) -> dict:
    """the steps `names` of module M's plan as (DataFrame, [(column, type)]), from one run of its script over the windows
    table wtab (a name the plan does not have is left out)."""
    q = M.build("events", "detectors", windows="wtab", periods=True, **kw)
    have = [n for n in names if n in [c[0] for c in q.ctes]]
    stmts, drops = q.script([f"SELECT * FROM {n}" for n in have])
    out, first = {}, len(stmts) - len(have)
    for i, st in enumerate(stmts):
        r = con.execute(st)
        if i >= first:
            out[have[i - first]] = (r.df(), [(d[0], str(d[1])) for d in r.description])
        for t in drops[i]:
            con.execute(f"DROP TABLE IF EXISTS {t}")
    return out


def fingerprint(df: pd.DataFrame) -> str:
    """a hash of a frame's columns and values (numbers and times by their bytes, text as text): the same input on any
    pandas / numpy version gives the same hash."""
    h = hashlib.sha256(repr(list(df.columns)).encode())
    for c in df.columns:
        x = df[c]
        if pd.api.types.is_datetime64_any_dtype(x):
            h.update(x.astype("datetime64[ns]").astype("int64").to_numpy().tobytes())
        elif pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x):
            h.update(x.astype("float64").to_numpy().tobytes())
        else:
            h.update("\x1f".join("" if v is None or v != v else str(v) for v in x.astype(object)).encode())
    return h.hexdigest()[:16]


def cases(H, P, A) -> dict:
    """case name -> (input set, function(con) -> {output name: DataFrame}) for the modules H / P / A (health_duckdb,
    ped_health_duckdb, ped_assign: the oracle's or the package's)."""
    def veh(**kw):
        def run(con):
            d, p = H.health(con, "events", "detectors", periods=True, **kw)
            return {"det": d, "per": p}
        return run

    def ped(**kw):
        def run(con):
            i, p = P.ped_health(con, "events", periods=True, **kw)
            return {"inputs": i, "periods": p}
        return run

    def assign(con):
        a, c = A.ped_assignments(con, "events", return_candidates=True)
        return {"pa": a, "cand": c, "pd": A.ped_detectors(con, "events")}
    return {
        "sample_veh": ("sample", veh()),
        "sample_ped": ("sample", ped()),
        "synth_a_veh": ("synth_a", veh()),
        "synth_a_veh_win": ("synth_a", veh(start=WIN[0], end=WIN[1])),
        "synth_a_veh_off": ("synth_a", veh(options={**H.OPTIONS_OFF, "occ_strict": True})),   # 2.0.0: never imputed
        "synth_a_ped": ("synth_a", ped()),
        "synth_a_ped_assign": ("synth_a", ped(assign=True)),
        "synth_a_ped_off": ("synth_a", ped(options={k: False for k in P.OPT})),
        "synth_a_assign": ("synth_a", assign),
        "synth_b_veh": ("synth_b", veh()),
        "synth_b_ped": ("synth_b", ped()),
    }


def to_pq(df: pd.DataFrame, path: str):
    con = duckdb.connect()
    con.register("df", df)
    con.execute(f"COPY (SELECT * FROM df) TO '{path}' (FORMAT parquet)")


def read_pq(path: str) -> pd.DataFrame:
    return duckdb.connect().execute(f"SELECT * FROM read_parquet('{path}')").df()


def values(df: pd.DataFrame) -> dict:
    """column -> list of plain values (None for every kind of missing, lists for arrays, Timestamps, Python numbers):
    a frame and its golden file compare equal value by value, whatever types the parquet round trip gives an empty or
    all-missing column."""
    def norm(v):
        if isinstance(v, (list, tuple)) or hasattr(v, "tolist") and not isinstance(v, (str, bytes)) and getattr(v, "ndim", 0):
            return [norm(x) for x in list(v)]
        if v is None or v is pd.NA or v is pd.NaT or (isinstance(v, float) and v != v):
            return None
        if isinstance(v, pd.Timestamp) or hasattr(v, "dtype") and str(getattr(v, "dtype", "")).startswith("datetime"):
            return pd.Timestamp(v)
        return v.item() if hasattr(v, "item") else v
    return {c: [norm(v) for v in df[c].astype(object)] for c in df.columns}


def dtypes(df: pd.DataFrame) -> dict:
    return {c: str(t) for c, t in df.dtypes.items()}


def main():
    sys.path.insert(0, HERE)
    from legacy import health_duckdb_v2 as H, ped_assign_v2 as A, ped_health_duckdb_v2 as P
    os.makedirs(GOLD, exist_ok=True)
    if not os.path.exists(SAMPLE_DET):          # the bundled sample's detectors: predict()'s raw guesses (once)
        from atspm_detector import predict
        from atspm_detector.check import SAMPLE, read_parquet
        out = predict(read_parquet(SAMPLE))
        span = out.lanes.map(lambda s: float(len(s.split(","))) if isinstance(s, str) and s else float("nan"))
        d = pd.DataFrame({"DeviceId": out.DeviceId, "Detector": out.Detector, "phase": out.phase_guess,
                          "function": out.function_guess, "lanes": span})
        for c in H.PROB.values():
            d[c] = out[c]
        to_pq(d, SAMPLE_DET)
    man = {"duckdb": duckdb.__version__, "inputs": {}, "files": {}}
    sets = {}
    for case, (inp, run) in cases(H, P, A).items():
        if inp not in sets:
            ev, det = inputs(inp)
            sets[inp] = connect(ev, det)
            man["inputs"][inp] = {"events": fingerprint(ev), "detectors": fingerprint(det), "rows": len(ev)}
        for k, df in run(sets[inp]).items():
            f = f"{case}__{k}.parquet"
            to_pq(df, os.path.join(GOLD, f))
            man["files"][f] = {"rows": len(df), "dtypes": dtypes(df)}
            print(f, len(df))
    with open(os.path.join(GOLD, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1)


if __name__ == "__main__":
    main()
