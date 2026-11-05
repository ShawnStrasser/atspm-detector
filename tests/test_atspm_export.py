"""to_atspm_config: the detector configuration for the atspm package.

    pytest tests/test_atspm_export.py

A schema check of the output (columns, types, values) and its options; none needs atspm. The end-to-end run of
atspm's measures on this output lives in atspm's own tests.
"""
from __future__ import annotations

import pandas as pd
import pytest

import atspm_detector as dc
from atspm_detector.atspm_export import ATSPM_COLUMNS, CONFIG_COLUMNS, COVERAGE_COLS, EXTRA_OUT, REPORT_COLS
from atspm_detector.check import SAMPLE, read_parquet

ATSPM_FUNCTIONS = {"Advance", "Presence", "Yellow_Red", "Count"}


@pytest.fixture(scope="module")
def sample_pred():
    return dc.predict(SAMPLE)


# ------------------------------------------------------------------ 1. contract (no atspm needed)
def test_export_contract(sample_pred):
    cfg, rep = dc.to_atspm_config(sample_pred, return_report=True)
    assert list(cfg.columns) == CONFIG_COLUMNS
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
                                                    "Other"), function_names={"Count": "Stopbar Count"})
    assert {"Mid", "Bike"} <= set(allc.Function) and "Count" not in set(allc.Function)
    with pytest.raises(ValueError):
        dc.to_atspm_config(sample_pred, include=("Advance", "Ped"))
    off = dc.to_atspm_config(sample_pred, one_per_lane=False)
    assert len(off) >= len(base) and set(base.Parameter) <= set(off.Parameter)
    ext = dc.to_atspm_config(sample_pred, extra_columns=True)
    assert list(ext.columns) == CONFIG_COLUMNS + EXTRA_OUT and ext[CONFIG_COLUMNS].equals(base)
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
    assert list(cfg.columns) == CONFIG_COLUMNS
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
    assert list(e.columns) == CONFIG_COLUMNS and len(e) == 0
    assert list(dc.atspm_coverage(e).columns) == COVERAGE_COLS
    with pytest.raises(ValueError):
        dc.to_atspm_config(sample_pred.drop(columns=["phase_pred"]))


def test_distance_and_count_columns():
    """Distance is exported as DistanceFromStopBar (NaN where unknown); a Count zone keeps the name Count."""
    pred = pd.DataFrame({"DeviceId": ["A", "A", "A"], "Detector": [1, 2, 3], "phase_pred": [2, 2, 4],
                         "function_pred": ["Advance", "Count", "Advance"], "phase_prob": 0.9, "function_prob": 0.9,
                         "distance_ft": [513.45, 0.0, float("nan")]})
    cfg = dc.to_atspm_config(pred).set_index("Parameter")
    assert list(dc.to_atspm_config(pred).columns) == ["DeviceId", "Phase", "Parameter", "Function", "DistanceFromStopBar"]
    assert cfg.Function.to_dict() == {1: "Advance", 2: "Count", 3: "Advance"}
    assert cfg.DistanceFromStopBar[1] == 513.45 and pd.isna(cfg.DistanceFromStopBar[3])
