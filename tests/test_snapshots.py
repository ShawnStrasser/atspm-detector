"""The rendered SQL for four option sets, kept as files (stage 1 of the incremental-health design, T1.4): a change to a
template shows as a diff of tests/snapshots/*.sql in the pull request.  After a deliberate change, regenerate with
SNAPSHOT_UPDATE=1 pytest tests/test_snapshots.py and commit the files with the template change."""
import os

import pytest

from atspm_detector import health_duckdb as H
from atspm_detector import ped_assign as A
from atspm_detector import ped_health_duckdb as P

SNAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")


def _steps(steps):
    return "\n".join(f"-- {n}\n{s};\n" for n, s in steps)


SETS = {
    "health_defaults": lambda: H.health_sql("events", "detectors"),
    "health_periods_window": lambda: "\n;\n".join(H.health_sql("events", "detectors", start="2026-10-05 05:17",
                                                               end="2026-10-06 21:01", periods=True)),
    "health_all_off": lambda: H.health_sql("events", "detectors", options=H.OPTIONS_OFF, probabilities=False),
    "health_configured_clock": lambda: H.health_sql("events", "detectors", configured=True, bin_origin="clock",
                                                    options={"clean_yardsticks": True}),
    "ped_defaults": lambda: _steps(P._steps("events")),
    "ped_all_on": lambda: _steps(P._steps("events", {k: True for k in P.OPT})),
    "ped_assign_window": lambda: _steps(A._steps("events", "2026-09-20 03:17", "2026-09-21")),
}


@pytest.mark.parametrize("name", list(SETS))
def test_snapshot(name):
    path = os.path.join(SNAP, f"{name}.sql")
    sql = SETS[name]().replace("\r\n", "\n").rstrip("\n") + "\n"
    if os.environ.get("SNAPSHOT_UPDATE") == "1":
        os.makedirs(SNAP, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(sql)
    with open(path, encoding="utf-8") as f:
        assert sql == f.read().replace("\r\n", "\n"), f"{name}: rendered SQL differs from {path}"
