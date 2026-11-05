"""Health in DuckDB SQL (note 153): the query text, the run on an existing connection, the same answers as predict."""
import duckdb
import numpy as np
import pandas as pd

import atspm_detector as ad
from atspm_detector import health_duckdb as hq
from atspm_detector.check import SAMPLE

H = ["health_status", "health_score", "health_reason", "health_categories", "health_config",
     "health_watch", "health_signal_note"]


def _detectors(out: pd.DataFrame) -> pd.DataFrame:
    """predict()'s answers as the detectors table health() reads: the raw guesses, the lane span."""
    span = out.lanes.map(lambda s: float(len(s.split(","))) if isinstance(s, str) and s else np.nan)
    d = pd.DataFrame({"DeviceId": out.DeviceId, "Detector": out.Detector, "phase": out.phase_guess,
                      "function": out.function_guess, "lanes": span})
    for c in hq.PROB.values():
        d[c] = out[c]
    return d


def _same(a: pd.DataFrame, b: pd.DataFrame, cols=H):
    m = a.merge(b, on=["DeviceId", "Detector"], suffixes=("", "_b"))
    assert len(m) == len(a) == len(b)
    for c in cols:
        x, y = m[c], m[c + "_b"]
        if c == "health_score":
            assert np.allclose(x.astype(float), y.astype(float), rtol=1e-9, equal_nan=True), c
        else:
            assert (x.fillna("").astype(str) == y.fillna("").astype(str)).all(), c


def test_health_on_a_connection_equals_predict():
    ev = pd.read_parquet(SAMPLE)
    out, found = ad.predict(ev, return_findings=True)
    con = duckdb.connect()
    con.register("ev_df", ev)
    con.execute("CREATE TABLE events AS SELECT * FROM ev_df")
    con.register("det_df", _detectors(out))
    con.execute("CREATE TABLE detectors AS SELECT * FROM det_df")
    got = hq.health(con, "events", "detectors")
    _same(got, out.dropna(subset=["health_status"])[["DeviceId", "Detector", "health_status"]], ["health_status"])
    # predict's findings table holds the same rows as the health on the connection (its window may differ by a second)
    both = ad.health(con, "events", "detectors", ped=False)
    assert (found[["DeviceId", "Detector", "Label", "Detail"]].reset_index(drop=True)
            .equals(both[["DeviceId", "Detector", "Label", "Detail"]].reset_index(drop=True)))
    # the steps are temp tables of their own, all gone afterwards
    left = con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name LIKE 'hq_%'").fetchone()[0]
    assert left == 0
    # a detectors table without class probabilities works too (the function's own limits only)
    con.execute("CREATE TABLE det2 AS SELECT DeviceId, Detector, phase, function, lanes FROM detectors")
    two = hq.health(con, "events", "det2")
    assert len(two) == len(got) and two.health_status.notna().all()


def test_health_sql_text_and_names():
    sql = hq.health_sql("my_events", "my_detectors", start="2026-10-06 12:00", end="2026-10-06 12:30")
    assert sql.startswith("WITH ") and "my_events" in sql and "my_detectors" in sql
    assert "TIMESTAMP '2026-10-06 12:00:00'" in sql
    # every step carries the prefix, so a caller's table called like a step is never shadowed
    for n, _, _ in hq.build("e", "d").ctes:
        assert f"{hq.PREFIX}{n} AS " in sql
    # without class probabilities: no p_* column read
    assert "NULL::DOUBLE AS p_advance" in hq.health_sql("e", "d", probabilities=False)
