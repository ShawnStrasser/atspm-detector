"""two signals in one call, one starting 7 h after `start`, the other ending 10 h before `end`: each signal's answers
= the old package on that signal alone with the out-of-range bound left out"""
import sys, warnings, pandas as pd
warnings.simplefilter("ignore")
sys.path.insert(0, sys.argv[1])
from detector_classifier import predict
a = pd.read_parquet("clip/s05_m30.parquet"); b = pd.read_parquet("clip/s20_h3.parquet")
if sys.argv[2] == "new":
    r = predict(pd.concat([a, b]), start="2026-09-19 07:30:00", end="2026-09-19 18:00:00", min_actuations=1)
else:
    r = pd.concat([predict(a, start="2026-09-19 07:30:00", end=None, min_actuations=1),
                   predict(b, start=None, end="2026-09-19 18:00:00", min_actuations=1)], ignore_index=True)
r.to_pickle(f"two_sig_{sys.argv[2]}.pkl")
