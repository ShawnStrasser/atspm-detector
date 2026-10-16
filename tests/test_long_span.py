"""1.0.1 regression: samples spanning more than 24.8 days (the network's millisecond offsets overflowed int32 and
predict raised "ValueError: 'list' argument must have no negative elements")."""
import numpy as np
import pandas as pd
import pytest

from atspm_detector import predict
from atspm_detector import streams as st
from atspm_detector.check import SAMPLE, read_parquet      # the installed package's sample, read without pyarrow


@pytest.fixture(scope="module")
def sample():
    return read_parquet(SAMPLE)


@pytest.fixture(scope="module")
def alone(sample):
    return predict(sample, min_actuations=1).set_index("Detector")


def _copies(ev, days):
    """the 30-minute sample repeated at each offset (days): one signal, long gaps between the copies."""
    return pd.concat([ev.assign(Timestamp=ev.Timestamp + pd.Timedelta(days=d)) for d in days], ignore_index=True)


def _check(out, alone, min_agree):
    """every detector answered, health for all; the phases it gives agree with the sample scored alone.  (A whole call
    over weeks of mostly missing data is outside what the model is built for: a detector whose ON is still open when
    the data stops is withheld as 'stuck ON' until its OFF weeks later, so only half the answers are required.)"""
    out = out.set_index("Detector")
    assert set(out.index) == set(alone.index)
    both = out.phase_pred.notna() & alone.phase_pred.reindex(out.index).notna()
    agree = (out.phase_pred[both] == alone.phase_pred.reindex(out.index)[both]).mean()
    assert both.sum() >= 0.5 * alone.phase_pred.notna().sum()
    assert agree >= min_agree, f"phase agreement with the sample alone {agree:.2f}"
    assert out.health_status.notna().all()


def test_offsets_are_64_bit():
    span = 60 * 86_400_000
    v = st._ms(np.array([0.0, 59 * 86_400.0, 60 * 86_400.0]), 0, span)
    assert v.dtype == np.int64 and (v >= 0).all() and v[-1] == span


def test_gap_of_38_days(sample, alone):
    """the integrator's crash case, rebuilt from the bundled sample: one piece, a 38-day gap, one piece."""
    out = predict(_copies(sample, [0, 39]), min_actuations=1)
    _check(out, alone, 0.9)


def test_40_days_with_a_gap(sample, alone):
    out = predict(_copies(sample, [0, 10, 40]), min_actuations=1)
    _check(out, alone, 0.9)


def test_60_days(sample, alone):
    ev = _copies(sample, [0, 20, 40, 60])
    out = predict(ev, min_actuations=1)
    _check(out, alone, 0.9)
    # the first and the last piece scored on their own give the sample's own answers
    t0 = ev.Timestamp.min().floor("min")
    w = pd.Timedelta(minutes=31)
    ref = predict(sample, start=t0, end=t0 + w, min_actuations=1).set_index("Detector")
    first = predict(ev, start=t0, end=t0 + w, min_actuations=1).set_index("Detector")
    d60 = pd.Timedelta(days=60)
    last = predict(ev, start=t0 + d60, end=t0 + d60 + w, min_actuations=1).set_index("Detector")
    for x in (first, last):
        for c in ("phase_pred", "function_pred"):
            pd.testing.assert_series_equal(x[c], ref[c].reindex(x.index), check_names=False)
