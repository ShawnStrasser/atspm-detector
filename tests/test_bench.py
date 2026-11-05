"""The benchmark and its compare script (design section 9): the compare rules here, a small real run under pytest -m bench.

The smoke run is too small for its times to mean anything, so it checks that the benchmark runs, that its equality checks
pass at this size (T8.2) and that compare.py reads its line; timing against the machine's history is compare.py's job on the
full-size runs."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import bench_health  # noqa: E402
import compare as C  # noqa: E402


def _run(update_median=0.1, machine="m", ratio=1.0, equal="same", **kw):
    return {"machine": machine, "duckdb": "2.0.0", "threads": 4, "memory": "4GB", "input": {"synth": "x"}, "impl": "both",
            "seconds": {"package": {"health_w1": 3.0}, "oracle": {"health_w1": 3.0}}, "ratio_package_oracle": {"health_w1": ratio},
            "equal_outputs": {"health_w1": equal}, "peak_mb": 1000.0,
            "updates": {"update_s": {"median": update_median, "p95": 0.1, "max": 0.1}, "judge_s": 2.0, "store_mb": 10.0,
                        "equal_to_health_clock": True}, **kw}


def _levels(lines):
    return {(f["level"], f["metric"]) for f in C.compare(lines)}


def test_compare_steady_runs_flag_nothing():
    assert C.compare([_run() for _ in range(8)]) == []


def test_compare_one_slow_run_is_a_watch_two_fail():
    base = [_run() for _ in range(6)]
    assert _levels(base + [_run(0.13)]) == {("watch", "update median s")}
    assert _levels(base + [_run(0.13), _run(0.13)]) == {("SLOWER", "update median s")}
    assert _levels(base + [_run(0.119), _run(0.119)]) == set()      # within 20 %


def test_compare_only_compares_runs_of_the_same_shape():
    assert C.compare([_run(machine="other") for _ in range(6)] + [_run(9.0), _run(9.0)]) == []


def test_compare_ratio_limit_and_different_outputs():
    base = [_run() for _ in range(3)]
    assert _levels(base + [_run(ratio=1.2), _run(ratio=1.2)]) == {("SLOWER", "ratio package / oracle health_w1")}
    assert _levels(base + [_run(equal="DIFFERENT")]) == {("DIFFERENT", "outputs")}


@pytest.mark.bench
def test_benchmark_smoke(tmp_path):
    out = tmp_path / "results.jsonl"
    args = ["--signals", "3", "--days", "1", "--repeat", "1", "--impl", "both", "--jobs", "health_w1,ped_health,update",
            "--threads", "2", "--memory", "2GB", "--temp", str(tmp_path), "--out", str(out)]
    bench_health.main(args)             # different outputs would stop it here
    line = json.loads(out.read_text(encoding="utf-8"))
    assert line["equal_outputs"]["health_w1"] == "same" and line["equal_outputs"]["ped_health"] == "same"
    assert line["updates"]["equal_to_health_clock"] and line["updates"]["equal_to_ped_health"]
    assert line["updates"]["store_mb"] > 0 and line["updates"]["chunks"] == 96
    assert C.main(["--file", str(out)]) == 0
