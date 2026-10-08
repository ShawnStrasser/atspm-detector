"""cProfile of the installed v7 wheel (warm call) -> top functions by own time inside the package."""
import os as _os
from pathlib import Path as _P
DCW = _os.environ.get("DC_WORK", str(_P.home() / "dc_work"))
import cProfile, pstats, io, sys, time, warnings
warnings.simplefilter("ignore")
from detector_classifier import predict
from pathlib import Path
V = Path(DCW + r"\s115v")
case, s, e, prof = sys.argv[1], sys.argv[2] or None, sys.argv[3] or None, sys.argv[4]
f = str(V / "ev" / f"{case}.parquet")
predict(f, start=s, end=e, min_actuations=1, profile=prof)          # cold
t = time.perf_counter(); predict(f, start=s, end=e, min_actuations=1, profile=prof); warm = time.perf_counter() - t
pr = cProfile.Profile(); pr.enable()
predict(f, start=s, end=e, min_actuations=1, profile=prof)
pr.disable()
st = pstats.Stats(pr)
print(f"{case} {s}..{e} {prof}: warm {warm:.3f}s")
rows = []
for (fn, ln, name), (cc, nc, tt, ct, _) in st.stats.items():
    if "detector_classifier" in fn:
        rows.append((tt, ct, nc, f"{Path(fn).name}:{ln} {name}"))
rows.sort(reverse=True)
print("own-time top (package functions): tottime cumtime ncalls")
for r in rows[:25]:
    print(f"  {r[0]:.4f} {r[1]:.4f} {r[2]:>7} {r[3]}")
# package cumulative by function (stage view)
rows.sort(key=lambda r: -r[1])
print("cumtime top:")
for r in rows[:30]:
    print(f"  {r[1]:.4f} {r[3]}")
# python-level builtins / pandas hot spots
s2 = io.StringIO(); pstats.Stats(pr, stream=s2).sort_stats("tottime").print_stats(25); print(s2.getvalue()[-6000:])
