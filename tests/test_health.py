"""Health 1.1.0: the checks, their limits and measures (notes 144, 147m, 150)."""
import json

import numpy as np
import pandas as pd

from atspm_detector import health_core as hc
from atspm_detector import health_counts as hk
from atspm_detector import health_v4 as hv
from atspm_detector import health_v4_stats as hs

CHECKS = {"Stuck on", "Counts fall", "Misses vehicles at night", "Busy at night", "Too-fast actuations",
          "Too many for the traffic", "Spiky counts"}
REMOVED = ("Chattering", "Erratic time ON", "Unusual daily pattern", "ON longer than its kind", "Goes silent",
           "Count drops", "Count zone held ON", "undercounts vs partner", "Erratic counts")


def test_categories_are_the_checks():
    assert set(hv.CATEGORY.values()) == CHECKS
    assert set(hv.CONFIG.values()) == {"Extension time on a count zone", "Set to pulse but holds ON"}


def test_score_bad_starts_at_the_bad_limit():
    s = hv.score(np.array([9.99, 10.0, 12.0, 13.0, 14.0, 99.0]), 10.0, 13.0)
    assert s[0] == 0.0 and abs(s[1] - 0.35) < 1e-12 and 0.35 < s[2] < 0.75
    assert abs(s[3] - 0.75) < 1e-12 and s[4] > 0.75 and s[5] == 1.0


def _X(wg):
    return pd.DataFrame({"fn": ["Presence", "Count", "Bike", "Presence", "Advance"], "span": ["1", "1", "1", "1", "2+"],
                         "cmode": ["", "pulse", "", "", ""], "band": ["high", "medium", "low", "low", "high"],
                         "dur_p50": [1.0, 0.1, 2.0, 3.0, 0.3], "f_top": [0.9, 0.95, 0.9, 0.5, 0.9],
                         "alt_fns": ["", "", "", "Presence,Other", ""],
                         "alt_set": [set(), set(), set(), {"Presence", "Other"}, set()], "wg": wg})


def test_limits_are_the_reviewed_cut_offs():
    lim = hv._load_limits()
    h = lim["h24"]
    for t in ("stuck_s", "silent_exp"):
        for key, (s, b) in h[t].items():
            assert 0 < s <= b, (t, key)
    # stuck on: bad starts at the user's minutes (sure types)
    for key, mins in (("Bike|1", 10), ("Count|1|pulse", 20), ("Yellow_Red|1", 20), ("Advance|1", 30), ("Mid|1", 35),
                      ("Presence|1", 60), ("Other|2+", 60)):
        assert abs(h["stuck_s"][key][1] - 60 * mins) < 1e-6, key
    tf = lim["too_fast"]["per_1000"]
    assert all(tf[f"{f}|2+"] == tf[f"{f}|1"] for f in hv.C7)            # 2+ lanes: the 1-lane limit
    assert lim["too_many_h24_veh_h_lane"][0] == 2100.0
    refs = hv._load_refs()
    L, B = hv.limits(_X("h24"), refs, lim)
    assert L["stuck"][0] == h["stuck_s"]["Presence|1"][0] and B["stuck"][2] == h["stuck_s"]["Bike|1"][1]
    # unsure Presence / Other: the least strict of the two
    assert L["stuck"][3] == max(h["stuck_s"]["Presence|1"][0], h["stuck_s"]["Other|1"][0])
    assert B["dropout"][3] == max(h["silent_exp"]["Presence|1"][1], h["silent_exp"]["Other|1"][1])
    assert L["rapid"][4] == tf["Advance|1"] and abs(B["rapid"][4] / L["rapid"][4] - lim["too_fast"]["bad_x"]) < 1e-12
    assert (L["level"] == 10.0).all() and (L["night_drop"] == 8.0).all() and (B["volume"] > L["volume"]).all()
    # shorter samples: the 1.0.1 limits
    L3, B3 = hv.limits(_X("h3"), refs, lim)
    assert (L3["dropout"] == lim["current"]["silent_exp"]).all() and (L3["stuck"] >= 300).all()
    assert (B3["stuck"] > L3["stuck"]).all() and (L3["volume"] >= lim["current"]["too_many_veh_h_lane"]).all()


def test_too_fast_counts_every_on_to_on_gap():
    # channel 5 (Presence, 1 lane: gap limit 0.8 s): ONs at 0, 0.5 (fast), 0.7 (fast, logged again without an OFF),
    # 10.0, 10.8 (not fast: 8 ticks); channel 6 (Advance, 2 lanes: 0.3 s): 0, 0.2 (fast), 5, 5.3 (not fast)
    rows = [(0.0, 82, 5), (0.3, 81, 5), (0.5, 82, 5), (0.7, 82, 5), (1.0, 81, 5), (10.0, 82, 5), (10.4, 81, 5),
            (10.8, 82, 5), (11.0, 81, 5),
            (0.0, 82, 6), (0.1, 81, 6), (0.2, 82, 6), (0.25, 81, 6), (5.0, 82, 6), (5.1, 81, 6), (5.3, 82, 6),
            (5.4, 81, 6)]
    t, e, p = map(np.array, zip(*rows))
    P = hc.prep_arrays(t, e, p)
    T = hs.too_fast(P, {5: "Presence", 6: "Advance"}, {6: 2.0})
    r = T.set_index("detector")
    assert r.loc[5, "N"] == 5 and r.loc[5, "n_fast"] == 2
    assert r.loc[6, "N"] == 4 and r.loc[6, "n_fast"] == 1
    assert abs(r.loc[5, "fast_rate"] - 400.0) < 1e-9
    # pairs: channel 6: its short gap is followed by a normal one (a pair); channel 5: one of its two short gaps
    assert r.loc[6, "pair_share"] == 1.0 and r.loc[5, "pair_share"] == 0.5


def test_common_holds():
    # det 1 and 2 held ON together (start and end within 5 s): common; det 3 overlaps det 4 but ends 10 min later
    s = np.array([100.0, 103.0, 5000.0, 5100.0, 9000.0])
    f = np.array([2000.0, 2004.0, 6000.0, 6600.0, 9900.0])
    c = np.array([1, 2, 3, 4, 5])
    anyov, common = hk.common_holds(s, f, c, 5.0, block=2)
    assert anyov.tolist() == [True, True, True, True, False]
    assert common.tolist() == [True, True, False, False, False]
    assert hk.common_mates(s, f, c, 0) == [2]


def test_twin_channels_are_not_a_common_failure():
    # det 1 and 2 = one zone on two inputs (identical counts all day): their shared hold is not a common failure;
    # det 3 held ON at the same time as well, with the same start and end: common against the zone
    T = np.zeros((4, 4), bool)
    T[1, 2] = T[2, 1] = True
    s, f, c = np.array([100.0, 101.0]), np.array([2000.0, 2002.0]), np.array([1, 2])
    assert hk.common_holds(s, f, c, 5.0, twin=T)[1].tolist() == [False, False]
    assert hk.common_holds(s, f, c, 5.0)[1].tolist() == [True, True]
    s3, f3, c3 = np.r_[s, 102.0], np.r_[f, 2003.0], np.r_[c, 3]
    assert hk.common_holds(s3, f3, c3, 5.0, twin=T)[1].tolist() == [True, True, True]
    assert hk.common_mates(s3, f3, c3, 0, T) == [3]
    rng = np.random.default_rng(5)
    a = rng.poisson(10, 288)
    B = _bins([a, a, rng.poisson(10, 288)])
    M = hk.twin_matrix(B)
    assert M[1, 2] and M[2, 1] and not M[1, 3]


def _bins(counts, start="2026-09-28 00:00", bin_s=300):
    """a minimal events_to_bins dict from per-detector 5-min counts (occupancy = 2 s per vehicle)."""
    n = np.asarray(counts, np.float32)
    return {"dets": np.arange(1, len(n) + 1), "n_on": n, "occ": n * 2.0, "cov": np.ones(n.shape[1], bool),
            "start": pd.Timestamp(start), "bin_s": bin_s, "hour": np.zeros(n.shape[1], np.int32)}


def _day(level, rng, nb=288):
    h = (np.arange(nb) * 300 // 3600) % 24
    shape = np.where((h >= 6) & (h < 21), 1.0, 0.25)
    return rng.poisson(level * shape)


def test_hourly_geh_count_drop_and_night():
    rng = np.random.default_rng(150)
    a, b, c = _day(40, rng), _day(40, rng), _day(40, rng)
    d = _day(40, rng)
    d[150:] = rng.poisson(8, 288 - 150)                     # det 4 counts a fifth from 12:30 on
    e = _day(40, rng)
    h = (np.arange(288) * 300 // 3600) % 24
    night = (h >= 21) | (h < 5)
    e[night] = rng.poisson(1, int(night.sum()))              # det 5 misses most vehicles at night
    G = hk.hourly_geh(_bins([a, b, c, d, e]), {k: 2.0 for k in range(1, 6)}).set_index("detector")
    assert G.geh_ok.all()
    assert G.loc[4, "geh_drop"] >= 10 and pd.Timestamp(G.loc[4, "drop_t"]).hour in (12, 13)
    assert not (G.loc[[1, 2, 3], "geh_drop"] >= 10).any()
    assert G.loc[5, "geh_night"] >= 8 and (G.loc[[1, 2, 3], "geh_night"] < 5).all()


def test_spiky_counts():
    rng = np.random.default_rng(147)
    P = hv._load_limits()["spiky"]
    a, b = _day(10, rng), _day(10, rng)
    s = a.copy()
    for k in (40, 100, 200):                                  # three 10-min bursts on det 3
        s[k:k + 2] += 60
    m = _day(10, rng)
    m[100:102] += 60                                          # det 4 bursts once, together with its mate det 2
    mates = b.copy()
    mates[100:102] += 60
    T, runs = hk.spiky(_bins([a, mates, s, m]), {1: 2.0, 2: 2.0, 3: 3.0, 4: 2.0}, 24.0, P)
    T = T.set_index("detector")
    assert T.loc[3, "spk_bursts"] == 3 and T.loc[3, "spk_lvl"] == 2 and len(runs[3]) == 3
    assert T.loc[1, "spk_lvl"] == 0
    assert T.loc[4, "spk_bursts"] == 0                        # its mates rose with it: not a burst
    T2, _ = hk.spiky(_bins([a, mates, s, m]), {1: 2.0, 2: 2.0, 3: 3.0, 4: 6.0}, 24.0, P)
    assert T2.set_index("detector").loc[4, "spk_bursts"] == 1 and T2.set_index("detector").loc[4, "spk_lvl"] == 0
    # low volume: < 200 vehicles a day is never flagged
    low = np.zeros(288, int)
    for k in (40, 100, 200):
        low[k] = 30
    T3, _ = hk.spiky(_bins([a, low]), {1: 2.0, 2: 3.0}, 24.0, P)
    T3 = T3.set_index("detector")
    assert T3.loc[2, "spk_bursts"] == 3 and T3.loc[2, "spk_lvl"] == 0


def test_sample_reports_only_the_checks():
    from atspm_detector import predict
    from atspm_detector.check import SAMPLE
    out = predict(pd.read_parquet(SAMPLE), min_actuations=1)
    for c in out.health_categories.fillna(""):
        for part in [x for x in c.split("; ") if x]:
            assert part.rsplit(" (", 1)[0] in CHECKS, part
    text = " ".join(out.health_reason.fillna("")) + " " + " ".join(out.health_bad_periods.fillna("").astype(str))
    for w in REMOVED:
        assert w not in text, w
    json.loads(out.health_bad_periods.dropna().iloc[0] or "[]")


def _three_windows():
    """the bundled sample as three signals sharing one window: as it is, without two detector channels, and with its
    phases renumbered; with the classifier's answers of the sample."""
    from atspm_detector import predict
    from atspm_detector.check import SAMPLE
    ev = pd.read_parquet(SAMPLE)
    ev["Timestamp"] = pd.to_datetime(ev.Timestamp)
    out = predict(ev, min_actuations=1)
    start = ev.Timestamp.min().floor("min")
    end = ev.Timestamp.max().ceil("min")
    cls = ["Advance", "Presence", "Count", "Yellow_Red", "Other", "Mid", "Bike"]
    pc = ["p_advance", "p_presence", "p_count", "p_yellow_red", "p_other", "p_mid", "p_bike"]
    o = out.dropna(subset=["phase_guess"])
    it = dict(phase={int(r.Detector): float(r.phase_guess) for r in o.itertuples()},
              fn_label={int(r.Detector): r.function_guess for r in o.itertuples() if isinstance(r.function_guess, str)},
              fn_prob={int(r.Detector): {c: float(getattr(r, p)) for c, p in zip(cls, pc)} for r in o.itertuples()},
              lanes={int(r.Detector): 1.0 for r in o.itertuples()})
    drop = sorted(it["phase"])[:2]
    ev_b = ev[~(ev.EventId.isin([81, 82]) & ev.Parameter.isin(drop))]
    it_b = {k: {d: v for d, v in m.items() if d not in drop} for k, m in it.items()}
    ev_c = ev.copy()
    col = ev_c.EventId.isin([1, 8, 9, 10])
    ev_c.loc[col, "Parameter"] = ev_c.loc[col, "Parameter"] + 20
    it_c = dict(it, phase={d: p + 20 for d, p in it["phase"].items()})
    preps = [hc._prep(e[["Timestamp", "EventId", "Parameter"]], start, end) for e in (ev, ev_b, ev_c)]
    return preps, [it, it_b, it_c], start, end


def test_health_batch_equals_alone():
    preps, items, start, end = _three_windows()
    alone = [hv.assess(P, start, end, **it) for P, it in zip(preps, items)]
    batch = hv.assess_group(hc.prep_group(preps), start, end, items)     # one batch: every check for all three
    assert len(batch) == 3 and any(len(o) for o, _ in alone)
    for (a, na), (b, nb) in zip(alone, batch):
        assert na == nb
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_exact=True)
    # the renumbered phases change only the "P<n>" labels
    pa, pc = alone[0][0], alone[2][0]
    assert pa.health_status.tolist() == pc.health_status.tolist()


def test_memory_based_batch():
    from atspm_detector.pipeline import BATCH_RANGE, BATCH_SIGNALS, MB_PER_MEVENT, auto_batch
    assert auto_batch(10**6, 5, free_mb=0) == BATCH_SIGNALS              # unknown free memory
    assert auto_batch(2 * 10**6, 10, free_mb=10**7) == BATCH_RANGE[1]    # plenty: the ceiling
    assert auto_batch(5 * 10**7, 1, free_mb=100) == BATCH_RANGE[0]       # a huge signal: one at a time
    per = 0.2 * MB_PER_MEVENT                                            # 200k events per signal
    assert auto_batch(2 * 10**6, 10, free_mb=2 * 12 * per + 1) == 12     # half the free memory
