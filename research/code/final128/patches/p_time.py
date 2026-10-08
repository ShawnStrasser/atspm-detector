import os
p = os.path.join(os.environ.get("DC_WORK", os.path.expanduser("~/dc_work")), r"final_v7_next\src\detector_classifier\health_core.py")
s = open(p, encoding="utf-8").read()


def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:80], s.count(a))
    s = s.replace(a, b)


rep('''    t15 = B["start"] + pd.to_timedelta(np.arange(len(x)) * B["bin_s"] * AGG, unit="s")
    date = t15.normalize().to_numpy()
    slot = (t15.hour * 4 + t15.minute // 15).to_numpy()''', '''    date, slot, dow = _clock15(B, len(x))''')
rep('''        wk = pd.DatetimeIndex(t15[si]).dayofweek >= 5''', '''        wk = dow[si] >= 5''')
rep('''def spike_stats(B: dict, i: int, ref_rows, on: tuple | None, lt1_lim: float) -> dict:''', '''DAY_NS = 86_400_000_000_000


def _clock15(B: dict, n: int):
    """date (datetime64 midnight), 15-min clock slot (hour * 4 + minute // 15) and day of week (Monday = 0) of the n
    15-min bins of B, in integer nanoseconds (= the Timestamp arithmetic); computed once per bins dict."""
    key = ("_clock15", n)
    if key not in B:
        ns = int(B["start"].value) + np.arange(n, dtype=np.int64) * (int(B["bin_s"]) * AGG * 1_000_000_000)
        days = ns // DAY_NS
        B[key] = ((days * DAY_NS).astype("datetime64[ns]"), (ns - days * DAY_NS) // 900_000_000_000,
                  (days + 3) % 7)
    return B[key]


def spike_stats(B: dict, i: int, ref_rows, on: tuple | None, lt1_lim: float) -> dict:''')
rep('''                    if on_hour is None:            # clock hour of every ON, one conversion for all detectors
                        ks = list(on)
                        hh = (start + pd.to_timedelta(np.concatenate([on[k][0] for k in ks]), unit="s")).hour.to_numpy()
                        cut = np.cumsum([len(on[k][0]) for k in ks])[:-1]
                        on_hour = dict(zip(ks, np.split(hh, cut)))''', '''                    if on_hour is None:            # clock hour of every ON, one conversion for all detectors
                        on_hour = _memo(events_df, ("onhour", start, end), lambda: _on_hours(on, start))''')
rep('''def _add_cols(df: pd.DataFrame, cols: dict) -> pd.DataFrame:''', '''def _on_hours(on: dict, start) -> dict:
    """{detector: clock hour of each ON} (one Timestamp conversion for all detectors)."""
    ks = list(on)
    hh = (start + pd.to_timedelta(np.concatenate([on[k][0] for k in ks]), unit="s")).hour.to_numpy()
    cut = np.cumsum([len(on[k][0]) for k in ks])[:-1]
    return dict(zip(ks, np.split(hh, cut)))


def _add_cols(df: pd.DataFrame, cols: dict) -> pd.DataFrame:''')
open(p, "w", encoding="utf-8").write(s)

p = os.path.join(os.environ.get("DC_WORK", os.path.expanduser("~/dc_work")), r"final_v7_next\src\detector_classifier\health_v4.py")
s = open(p, encoding="utf-8").read()
rep('''    tday = np.array([_day_type(start + pd.Timedelta(seconds=900 * j)) for j in range(nb15)], object)
    lv, ev, dr = [], [], []''', '''    dow = ((int(start.value) + np.arange(nb15, dtype=np.int64) * 900_000_000_000) // hc.DAY_NS + 3) % 7
    tday = np.where(dow >= 5, "h24_a", "h24_b").astype(object)          # = _day_type of each 15-min bin start
    wcache = {}
    lv, ev, dr = [], [], []''')
rep('''        w = np.array([refs.tod_w.get((r.type, r.band, dd, int(hh)),
                                     refs.tod_wb.get((r.band, dd, int(hh)), 1.0)) for dd, hh in zip(tday, b15["hour"])],
                     float)''', '''        wk = (r.type, r.band)
        if wk not in wcache:
            wcache[wk] = np.array([refs.tod_w.get((r.type, r.band, dd, int(hh)), refs.tod_wb.get((r.band, dd, int(hh)), 1.0))
                                   for dd, hh in zip(tday, b15["hour"])], float)
        w = wcache[wk]''')
open(p, "w", encoding="utf-8").write(s)
print("patched")
