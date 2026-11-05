"""Golden outputs (stage 0 / 1 of the incremental-health design): T0.1 the frozen oracles reproduce tests/golden (guards
the harness itself); stage 1: the package's modules (SQL from the Jinja templates) give the same: every value of the
golden files (floats bit for bit, rows in order), with the oracle's columns and dtypes in the same environment (pandas
versions differ in their default dtypes).  Regenerate with tests/make_golden.py."""
import json
import os

import pytest

import make_golden as G
from atspm_detector import health_duckdb, ped_assign, ped_health_duckdb
from legacy import health_duckdb_v2, ped_assign_v2, ped_health_duckdb_v2

MODS = {"oracle": (health_duckdb_v2, ped_health_duckdb_v2, ped_assign_v2),
        "package": (health_duckdb, ped_health_duckdb, ped_assign)}
MAN = json.load(open(os.path.join(G.GOLD, "manifest.json"), encoding="utf-8"))
_CON = {}


def _con(inp):
    if inp not in _CON:
        ev, det = G.inputs(inp)
        want = MAN["inputs"][inp]
        assert (G.fingerprint(ev), G.fingerprint(det)) == (want["events"], want["detectors"]), \
            f"input set {inp} changed (tests/synth.py or numpy): regenerate the golden files with tests/make_golden.py"
        _CON[inp] = G.connect(ev, det)
    return _CON[inp]


def _values_equal(got, f):
    """every value of output got equals golden file f (row count, columns, floats bit for bit)."""
    assert len(got) == MAN["files"][f]["rows"], f
    gold = G.values(G.read_pq(os.path.join(G.GOLD, f)))
    assert list(gold) == list(got.columns), f
    for c, v in G.values(got).items():
        assert v == gold[c], (f, c)


CASES = list(G.cases(*MODS["oracle"]))
# ped changed on purpose: the movement model knows no number (note 156), the rules are those of the 2026-11-02 review.  Its package output no
# longer equals this frozen oracle; tests/test_ped_assign.py, test_ped_invariance.py and test_ped_incremental.py (oracle v3) guard it
PACKAGE_CASES = [c for c in CASES if "ped" not in c and "assign" not in c]
_ORACLE = {}


def _oracle(case):
    if case not in _ORACLE:
        inp, run = G.cases(*MODS["oracle"])[case]
        _ORACLE[case] = run(_con(inp))
    return _ORACLE[case]


@pytest.mark.parametrize("case", CASES)
def test_oracle_reproduces_golden(case):
    for k, df in _oracle(case).items():
        _values_equal(df, f"{case}__{k}.parquet")


@pytest.mark.parametrize("case", PACKAGE_CASES)
def test_package_equals_golden_and_oracle(case):
    inp, run = G.cases(*MODS["package"])[case]
    out, ref = run(_con(inp)), _oracle(case)
    assert list(out) == list(ref)
    for k, df in out.items():
        assert G.dtypes(df) == G.dtypes(ref[k]), (case, k)
        _values_equal(df, f"{case}__{k}.parquet")
