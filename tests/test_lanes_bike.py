"""A final Bike zone covers exactly one lane and never adds a lane to its phase (note 148)."""
import numpy as np
import pandas as pd

from atspm_detector import lanes as L


def _run(funcs, probs, bikes):
    g = pd.DataFrame({"DeviceId": "s", "Detector": list(funcs), "phase_use": 2.0, "func_use": list(funcs.values())})
    n_on = {d: 1000 - 10 * d for d in funcs}
    dec = L.Decoder({"Other": {1: np.log(0.5), 2: np.log(0.5), 3: np.log(0.1), 4: np.log(0.1)}})
    ph, de = L.decode_groups("s", g, n_on, 24.0, probs, dec)
    ctx = dict(g=g, n_on=n_on, hours=24.0, probs=probs, decoder=dec, min_on=L.MIN_ON, keys=None)
    ph2, de2 = L.bike_one_lane("s", ctx, ph, de, bikes)
    return ph, de.set_index("Detector"), ph2, de2.set_index("Detector")


def test_bike_that_made_a_lane_is_dropped_from_the_lane_count():
    # 1 Count + 2 Presence share a lane; 3 was decoded as a Count on a lane of its own but is finally Bike
    funcs = {1: "Count", 2: "Presence", 3: "Count"}
    probs = {(1, 2): 0.95, (1, 3): 0.05, (2, 3): 0.05}
    ph, de, ph2, de2 = _run(funcs, probs, [3])
    assert int(ph.n_lanes.iloc[0]) == 2 and de.at[3, "n_lanes_spanned"] == 1
    assert int(ph2.n_lanes.iloc[0]) == 1
    assert de2.at[3, "lanes"] == "1" and de2.at[3, "n_lanes_spanned"] == 1
    assert de2.loc[[1, 2], "lanes"].tolist() == ["1", "1"]


def test_spanning_bike_gets_one_lane_and_others_are_untouched():
    # two Count lanes; 3 (decoded Other, finally Bike) looks like both, a bit more like the lane of 2
    funcs = {1: "Count", 2: "Count", 3: "Other"}
    probs = {(1, 2): 0.02, (1, 3): 0.90, (2, 3): 0.95}
    ph, de, ph2, de2 = _run(funcs, probs, [3])
    assert de.at[3, "n_lanes_spanned"] == 2
    assert int(ph2.n_lanes.iloc[0]) == int(ph.n_lanes.iloc[0]) == 2
    assert de2.at[3, "n_lanes_spanned"] == 1 and de2.at[3, "lanes"] == de.at[2, "lanes"]
    pd.testing.assert_frame_equal(de2.drop(index=3), de.drop(index=3))
    pd.testing.assert_frame_equal(ph2, ph)


def test_no_bike_no_change():
    funcs = {1: "Count", 2: "Count", 3: "Other"}
    probs = {(1, 2): 0.02, (1, 3): 0.90, (2, 3): 0.95}
    ph, de, ph2, de2 = _run(funcs, probs, [])
    pd.testing.assert_frame_equal(de2, de)
    pd.testing.assert_frame_equal(ph2, ph)
