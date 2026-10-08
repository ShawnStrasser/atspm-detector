"""Wall time of every pandas row/group loop (iterrows / itertuples / groupby iteration / apply / map with a Python
callable) called from inside the package, on one warm predict call."""
import os as _os
from pathlib import Path as _P
DCW = _os.environ.get("DC_WORK", str(_P.home() / "dc_work"))
import sys, time, warnings, collections
from pathlib import Path
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy, SeriesGroupBy
warnings.simplefilter("ignore")
from detector_classifier import predict
V = Path(DCW + r"\s115v")
case, s, e, prof = sys.argv[1], sys.argv[2] or None, sys.argv[3] or None, sys.argv[4]
f = str(V / "ev" / f"{case}.parquet")
predict(f, start=s, end=e, min_actuations=1, profile=prof)
STAT = collections.defaultdict(lambda: [0, 0, 0.0])        # calls, items, seconds


def caller():
    fr = sys._getframe(2)
    while fr and "detector_classifier" not in fr.f_code.co_filename:
        fr = fr.f_back
    return f"{Path(fr.f_code.co_filename).name}:{fr.f_lineno}" if fr else None


def wrap_iter(cls, name):
    orig = getattr(cls, name)

    def w(self, *a, **k):
        where = caller()
        it = orig(self, *a, **k)
        if where is None:
            return it

        def gen():
            t = time.perf_counter(); n = 0
            for x in it:
                n += 1
                yield x
            st = STAT[(where, name)]; st[0] += 1; st[1] += n; st[2] += time.perf_counter() - t
        return gen()
    setattr(cls, name, w)


def wrap_call(cls, name):
    orig = getattr(cls, name)

    def w(self, *a, **k):
        where = caller()
        t = time.perf_counter()
        r = orig(self, *a, **k)
        if where is not None and (name != "map" or (a and callable(a[0]))):
            st = STAT[(where, name)]; st[0] += 1; st[1] += len(self) if hasattr(self, "__len__") else 0
            st[2] += time.perf_counter() - t
        return r
    setattr(cls, name, w)


wrap_iter(pd.DataFrame, "iterrows"); wrap_iter(pd.DataFrame, "itertuples")
wrap_iter(DataFrameGroupBy, "__iter__"); wrap_iter(SeriesGroupBy, "__iter__")
wrap_call(DataFrameGroupBy, "apply"); wrap_call(SeriesGroupBy, "apply"); wrap_call(pd.Series, "apply")
wrap_call(pd.DataFrame, "apply"); wrap_call(pd.Series, "map")
t = time.perf_counter()
predict(f, start=s, end=e, min_actuations=1, profile=prof)
tot = time.perf_counter() - t
print(f"{case} {s}..{e} {prof}: warm {tot:.3f}s (with wrappers)")
for (where, name), (c, n, sec) in sorted(STAT.items(), key=lambda kv: -kv[1][2]):
    print(f"  {sec*1000:8.2f} ms  {c:4d} calls {n:7d} items  {name:10s} {where}")
print(f"  total {sum(v[2] for v in STAT.values())*1000:.1f} ms of {tot*1000:.0f} ms")
