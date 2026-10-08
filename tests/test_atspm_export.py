"""to_atspm_config: the detector configuration for the atspm package.

    pytest tests/test_atspm_export.py

The contract and option tests need only this package.  The end-to-end tests run atspm's measures on the bundled sample
with the exported config; they are skipped when atspm is not installed (`pip install atspm`).  The comparison with
atspm's own sample data runs only when ATSPM_SAMPLES points to an atspm source tree (the folder holding src/atspm and
tests/).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import atspm_detector as dc
from atspm_detector.atspm_export import ATSPM_COLUMNS, COVERAGE_COLS, EXTRA_OUT, REPORT_COLS
from atspm_detector.check import SAMPLE, read_parquet

ATSPM_FUNCTIONS = {"Advance", "Presence", "Yellow_Red", "Stopbar Count"}


@pytest.fixture(scope="module")
def sample_pred():
    return dc.predict(SAMPLE)


@pytest.fixture(scope="module")
def atspm():
    return pytest.importorskip("atspm")


# ------------------------------------------------------------------ helpers (atspm side)
def run_atspm(events: pd.DataFrame, config: pd.DataFrame, extra=()):
    from atspm import SignalDataProcessor
    import atspm as _a
    aggs = [{"name": "has_data", "params": {"no_data_min": 5, "min_data_points": 3}},
            {"name": "actuations", "params": {}},
            {"name": "arrival_on_green", "params": {"latency_offset_seconds": 0}},
            {"name": "split_failures", "params": {"red_time": 5, "red_occupancy_threshold": 0.80,
                                                  "green_occupancy_threshold": 0.80, "by_approach": True,
                                                  "by_cycle": False}},
            {"name": "yellow_red", "params": {"latency_offset_seconds": 0}}]
    if (Path(_a.__file__).parent / "queries" / "platoon_ratio.sql").exists():      # atspm 2.5 and newer
        aggs.append({"name": "platoon_ratio", "params": {}})
    aggs += list(extra)
    p = SignalDataProcessor(raw_data=events, detector_config=config, bin_size=15, remove_incomplete=False,
                            verbose=0, aggregations=aggs)
    p.load()
    p.aggregate()
    tabs = {a["name"]: p.conn.query(f"SELECT * FROM {a['name']}").df() for a in aggs}
    p.conn.close()
    return tabs


def atspm_events(path) -> pd.DataFrame:
    ev = read_parquet(path)
    ev.columns = ["TimeStamp" if c.lower() == "timestamp" else c for c in ev.columns]
    return ev[["TimeStamp", "DeviceId", "EventId", "Parameter"]]


def aog_by_hand(ev: pd.DataFrame, config: pd.DataFrame) -> pd.DataFrame:
    """Share of Advance ON events arriving while their phase shows green (last of events 1 / 8 / 10 before it = 1)."""
    ev = ev.assign(DeviceId=ev.DeviceId.astype(str), EventId=ev.EventId.astype("int64"),
                   Parameter=ev.Parameter.astype("int64"))
    adv = config[config.Function.eq("Advance")].assign(DeviceId=lambda x: x.DeviceId.astype(str),
                                                       Parameter=lambda x: x.Parameter.astype("int64"))
    on = ev[ev.EventId.eq(82)].merge(adv, on=["DeviceId", "Parameter"])
    ph = ev[ev.EventId.isin([1, 8, 10])].rename(columns={"Parameter": "Phase", "EventId": "state"})
    ph = ph.assign(Phase=ph.Phase.astype(int))
    rows = []
    for (dv, p), g in on.groupby(["DeviceId", "Phase"]):
        s = ph[(ph.DeviceId == dv) & (ph.Phase == p)].sort_values(["TimeStamp", "state"])
        ts = s.TimeStamp.astype("datetime64[ns]").to_numpy().astype("int64")
        tg = g.TimeStamp.astype("datetime64[ns]").to_numpy().astype("int64")
        k = np.searchsorted(ts, tg, side="right") - 1                     # last phase event at or before the ON
        st = np.where(k >= 0, s.state.to_numpy()[np.clip(k, 0, None)], 0)
        rows.append({"DeviceId": dv, "Phase": int(p), "n": len(g), "green": int((st == 1).sum())})
    return pd.DataFrame(rows, columns=["DeviceId", "Phase", "n", "green"])


# ------------------------------------------------------------------ 1. contract (no atspm needed)
def test_export_contract(sample_pred):
    cfg, rep = dc.to_atspm_config(sample_pred, return_report=True)
    assert list(cfg.columns) == ATSPM_COLUMNS
    assert str(cfg.Phase.dtype) == "int16" and str(cfg.Parameter.dtype) == "int16"
    assert cfg.notna().all().all() and len(cfg) > 0
    assert not cfg.duplicated(["DeviceId", "Parameter"]).any()             # one phase per detector
    assert set(cfg.Function) <= ATSPM_FUNCTIONS
    ev = read_parquet(SAMPLE)
    greens = set(ev.loc[ev.EventId.eq(1), "Parameter"].astype(int))
    assert set(cfg.Phase.astype(int)) <= greens
    assert list(rep.columns) == REPORT_COLS and len(rep) == len(sample_pred)
    assert rep.exported.sum() == len(cfg)
    assert (rep.reason.ne("") | rep.exported).all()                        # every drop has a reason
    assert not (rep.exported & rep.reason.ne("")).any()
    # every exported row is the model's own answer
    m = cfg.merge(sample_pred, left_on=["DeviceId", "Parameter"], right_on=["DeviceId", "Detector"])
    assert len(m) == len(cfg) and (m.Phase.astype(int) == m.phase_pred.astype(int)).all()
    cov = dc.atspm_coverage(cfg)
    assert list(cov.columns) == COVERAGE_COLS and len(cov) == cfg.groupby(["DeviceId", "Phase"]).ngroups
    print("\nbundled sample config:\n" + cfg.to_string())
    print(rep[~rep.exported].to_string())
    print(cov.to_string())


def test_options(sample_pred):
    base = dc.to_atspm_config(sample_pred)
    p = sample_pred.copy()
    i = p.index[p.function_pred.eq("Presence")][0]
    p.loc[i, "health_status"] = "bad"
    assert int(p.Detector[i]) not in set(dc.to_atspm_config(p).Parameter.astype(int))
    assert int(p.Detector[i]) in set(dc.to_atspm_config(p, exclude_health=()).Parameter.astype(int))
    hi = dc.to_atspm_config(sample_pred, min_function_prob=0.9)
    assert (sample_pred.set_index("Detector").function_prob.reindex(hi.Parameter.astype(int)) >= 0.9).all()
    hp = dc.to_atspm_config(sample_pred, min_phase_prob=0.999)
    assert (sample_pred.set_index("Detector").phase_prob.reindex(hp.Parameter.astype(int)) >= 0.999).all()
    allc = dc.to_atspm_config(sample_pred, include=("Advance", "Presence", "Yellow_Red", "Count", "Mid", "Bike",
                                                    "Other"), function_names={"Count": "Count"})
    assert {"Mid", "Bike"} <= set(allc.Function) and "Stopbar Count" not in set(allc.Function)
    with pytest.raises(ValueError):
        dc.to_atspm_config(sample_pred, include=("Advance", "Ped"))
    off = dc.to_atspm_config(sample_pred, one_per_lane=False)
    assert len(off) >= len(base) and set(base.Parameter) <= set(off.Parameter)
    ext = dc.to_atspm_config(sample_pred, extra_columns=True)
    assert list(ext.columns) == ATSPM_COLUMNS + EXTRA_OUT and ext[ATSPM_COLUMNS].equals(base)
    assert dc.to_atspm_config(sample_pred, device_id_dtype="string").DeviceId.dtype == "string"


def test_file_input(sample_pred, tmp_path):
    base = dc.to_atspm_config(sample_pred)
    f = tmp_path / "pred.csv"
    sample_pred.to_csv(f, index=False)                                     # what --out writes
    assert dc.to_atspm_config(f).equals(base)
    import duckdb
    q = tmp_path / "pred.parquet"
    con = duckdb.connect()
    con.register("t", sample_pred)
    con.execute(f"COPY t TO '{q.as_posix()}' (FORMAT parquet)")
    con.close()
    assert dc.to_atspm_config(str(q)).equals(base)
    with pytest.raises(TypeError):
        dc.to_atspm_config(42)


def test_cli_out_atspm(tmp_path):
    from atspm_detector.pipeline import main
    main(["--events", str(SAMPLE), "--out", str(tmp_path / "det.csv"), "--out-atspm", str(tmp_path / "cfg.csv"),
          "--quiet"])
    cfg = pd.read_csv(tmp_path / "cfg.csv", dtype={"DeviceId": str})
    assert list(cfg.columns) == ATSPM_COLUMNS
    ref = dc.to_atspm_config(dc.predict(SAMPLE))
    assert cfg.astype({"Phase": "int16", "Parameter": "int16"}).equals(ref)


def test_one_per_lane_synthetic():
    r = pd.DataFrame({"DeviceId": ["A"] * 6, "Detector": [1, 2, 3, 4, 5, 6],
                      "phase_pred": [2, 2, 2, 2, 2, 4], "function_pred": ["Advance"] * 3 + ["Presence"] * 2 + ["Advance"],
                      "function_prob": [.9, .5, .99, .6, .7, .8], "phase_prob": [.99] * 6,
                      "lanes": ["1", "2", "1,2", "1", None, "1"], "health_status": ["ok"] * 6})
    c, rep = dc.to_atspm_config(r, return_report=True)
    assert sorted(c.Parameter.tolist()) == [1, 2, 4, 5, 6]                  # det 3 spans lanes 1,2 already taken
    assert rep.set_index("Detector").reason[3] == "lane already has Advance: det 1,2"
    c2 = dc.to_atspm_config(r.assign(lanes=["1", "1", "1,2", "1", None, "1"]))   # two Advance on lane 1: higher prob
    assert 2 not in set(c2.Parameter) and 1 in set(c2.Parameter) and 3 not in set(c2.Parameter)
    dup = pd.concat([r, r.iloc[[0]]], ignore_index=True)                    # an exact duplicate row is exported once
    assert dc.to_atspm_config(dup, one_per_lane=False).duplicated(ATSPM_COLUMNS).sum() == 0


def test_empty_and_errors(sample_pred):
    e = dc.to_atspm_config(sample_pred.iloc[0:0])
    assert list(e.columns) == ATSPM_COLUMNS and len(e) == 0
    assert list(dc.atspm_coverage(e).columns) == COVERAGE_COLS
    with pytest.raises(ValueError):
        dc.to_atspm_config(sample_pred.drop(columns=["phase_pred"]))


# ------------------------------------------------------------------ 2. end to end with atspm (skipped without it)
def test_end_to_end_bundled(sample_pred, atspm):
    cfg = dc.to_atspm_config(sample_pred)
    ev = atspm_events(SAMPLE)
    t = run_atspm(ev, cfg)
    aog, sf, act = t["arrival_on_green"], t["split_failures"], t["actuations"]
    assert len(aog) > 0
    assert set(aog.Phase.astype(int)) == set(cfg.Phase[cfg.Function.eq("Advance")].astype(int))
    assert aog.Percent_AOG.between(0, 1).all()
    hand = aog_by_hand(ev, cfg)
    a = (aog.assign(green=aog.Percent_AOG * aog.Total_Actuations).groupby("Phase")
         .agg(n=("Total_Actuations", "sum"), green=("green", "sum")).reset_index())
    m = a.merge(hand, on="Phase", suffixes=("_atspm", "_hand"))
    print("\nAOG atspm vs by hand:\n" + m.round(3).to_string())
    assert len(m) == len(a) and (m.n_atspm == m.n_hand).all()
    assert np.allclose(m.green_atspm / m.n_atspm, m.green_hand / m.n_hand, atol=1e-6)
    assert len(sf) > 0
    assert set(sf.Phase.astype(int)) <= set(cfg.Phase[cfg.Function.eq("Presence")].astype(int))
    for c in ("Green_Occupancy", "Red_Occupancy"):
        assert sf[c].between(0, 1).all()
    assert (sf.Green_Time > 0).all() and (sf.Split_Failure >= 0).all()
    tot = act.groupby("Detector").Total.sum()
    nact = sample_pred.set_index("Detector").n_actuations.astype(int)
    assert (tot.reindex(nact.index).fillna(0).astype(int) == nact).all()
    assert ("Yellow_Red" in set(cfg.Function)) == (len(t["yellow_red"]) > 0)
    if "platoon_ratio" in t:
        assert len(t["platoon_ratio"]) > 0 and (t["platoon_ratio"].Platoon_Ratio > 0).all()


def test_end_to_end_options(sample_pred, atspm):
    ev = atspm_events(SAMPLE)
    t = run_atspm(ev, dc.to_atspm_config(sample_pred, extra_columns=True))       # extra columns are ignored
    assert len(t["arrival_on_green"]) > 0
    t = run_atspm(ev, dc.to_atspm_config(sample_pred, device_id_dtype="string"))
    assert len(t["arrival_on_green"]) > 0
    t = run_atspm(ev, dc.to_atspm_config(sample_pred.iloc[0:0]))                 # empty config: measures empty
    assert len(t["arrival_on_green"]) == 0 and len(t["actuations"]) > 0


SAMPLES = [("src/atspm/data/sample_raw_data.parquet", "src/atspm/data/sample_config.parquet"),
           ("tests/hires_test_data.parquet", "tests/configs_test_data.parquet")]


@pytest.mark.parametrize("raw,cfgp", SAMPLES, ids=["atspm_sample_data", "atspm_test_data"])
def test_atspm_samples(raw, cfgp, atspm):
    root = os.environ.get("ATSPM_SAMPLES")
    if not root or not (Path(root) / raw).exists():
        pytest.skip("set ATSPM_SAMPLES to an atspm source tree to compare with its shipped configs")
    raw, cfgp = Path(root) / raw, Path(root) / cfgp
    ev = atspm_events(raw)
    pred = dc.predict(ev.rename(columns={"TimeStamp": "Timestamp"}))
    ours = dc.to_atspm_config(pred, device_id_dtype=ev.DeviceId.dtype)
    theirs = read_parquet(cfgp)
    t1 = run_atspm(ev, ours)
    for k in ("arrival_on_green", "split_failures", "yellow_red"):
        assert len(t1[k]) > 0, f"{k} empty with the exported config"
    used = theirs[theirs.Function.isin(["Advance", "Presence", "Yellow_Red"])]
    j = used.merge(ours, on=["DeviceId", "Parameter"], how="left", suffixes=("", "_ours"))
    same = (j.Phase == j.Phase_ours) & (j.Function == j.Function_ours)
    print(f"\n{raw.name}: shipped config rows read by atspm {len(used)}, same phase + function {int(same.sum())}")
    assert t1["arrival_on_green"].Percent_AOG.between(0, 1).all()
    assert t1["split_failures"].Red_Occupancy.between(0, 1).all()
