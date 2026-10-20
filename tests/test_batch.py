"""Batched predict (several signals in one pass) gives bit for bit the answers each signal gets alone."""
import pandas as pd
import pytest

from atspm_detector import predict
from atspm_detector.check import SAMPLE, read_parquet
from atspm_detector.common import timedelta as _td


def _three_signals() -> pd.DataFrame:
    ev = read_parquet(SAMPLE)
    t = pd.to_datetime(ev.Timestamp)
    a = ev.assign(DeviceId="sig-a")
    b = ev[t < t.min() + _td(minutes=20)].assign(DeviceId="sig-b",                     # shorter, another day
                                                 Timestamp=lambda d: pd.to_datetime(d.Timestamp) + _td(days=1))
    c = ev[t >= t.min() + _td(minutes=5)].assign(DeviceId="sig-c")                      # another window
    return pd.concat([a, b, c], ignore_index=True)


def _eq(x: pd.DataFrame, y: pd.DataFrame) -> None:
    x = x.sort_values(list(x.columns[:2])).reset_index(drop=True)
    y = y.sort_values(list(y.columns[:2])).reset_index(drop=True)
    pd.testing.assert_frame_equal(x, y, check_exact=True, check_dtype=False)


@pytest.mark.parametrize("batch", [None, 2, 1])
def test_batch_equals_alone(batch):
    ev = _three_signals()
    out, ph = predict(ev, min_actuations=1, return_phases=True, batch_signals=batch)
    assert out.DeviceId.nunique() == 3
    alone = [predict(ev[ev.DeviceId == d], min_actuations=1, return_phases=True) for d in ("sig-a", "sig-b", "sig-c")]
    _eq(out, pd.concat([a for a, _ in alone], ignore_index=True))
    _eq(ph, pd.concat([p for _, p in alone if len(p)], ignore_index=True))
