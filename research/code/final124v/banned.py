"""124v: banned imports.  Block torch / lightgbm / scipy / sklearn / pyarrow / matplotlib, run predict on the bundled
sample (whole, 30 min) and on a new 24-h signal, import the charts module, call health_chart (must fail cleanly), and
list which banned modules were imported.  Arg 'noblock' = no blocker, just report sys.modules after predict."""
import importlib.abc, sys, os, warnings
BAN = ("torch", "lightgbm", "scipy", "sklearn", "pyarrow", "matplotlib")
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in BAN:
            raise ImportError(f"blocked {name}")
if "noblock" not in sys.argv:
    sys.meta_path.insert(0, Block())
warnings.simplefilter("ignore" if "noblock" in sys.argv else "error")   # fresh venv: any warning is a failure
import detector_classifier as dc
from detector_classifier import predict
from importlib.resources import files
sample = str(files("detector_classifier") / "data" / "sample_events.parquet")
o = predict(sample, min_actuations=1)
print("sample rows", len(o), "health", o.health_status.value_counts().to_dict(), "cols", len(o.columns))
o2 = predict(sys.argv[1], start="2026-09-20 06:00:00", end="2026-09-21 06:00:00", min_actuations=1)
print("24h rows", len(o2), "health", o2.health_status.value_counts().to_dict())
import detector_classifier.charts as ch
print("charts module imported:", ch.__name__)
try:
    ch.health_chart(sample, o, int(o.Detector.iloc[0]), path=os.devnull)
    print("health_chart: ran")
except ImportError as e:
    print("health_chart without matplotlib -> ImportError:", e)
print("banned modules loaded:", sorted({m.split(".")[0] for m in sys.modules} & set(BAN)) or "none")
