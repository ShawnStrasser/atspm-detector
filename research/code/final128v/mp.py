import os
import sys, warnings, pickle; sys.path.insert(0, sys.argv[1]); warnings.simplefilter("ignore")
from detector_classifier import predict
o = predict(os.path.join(os.environ.get("DC_WORK", os.path.expanduser("~/dc_work")), r"s128v\ev\x07_m30.parquet"), min_prob=0.99)
pickle.dump(o, open(f"far/mp_{sys.argv[2]}.pkl", "wb"))
