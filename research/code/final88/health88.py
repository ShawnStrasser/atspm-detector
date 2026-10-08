"""Note 88: does the health reference set move once fault events stop shaping it?

hb_data.weak_labels (note 38) defined `presumed_healthy` with dq_core's s_health / dq_score and vd_audit's card
erratic flag, all three of which carried detector fault events 83-88; note 79's evaluation set took its healthy
negatives from it and note 83 calibrated the package's tick-based chatter / rapid limits on those negatives.
hb_data is now fault-free.  This script rebuilds the note-79 set and re-runs the note-83 calibration on it, writing
only into f88 (the original health79 / health83 outputs stay as they are):

    python health88.py   -> %DC_WORK%/final_v3_work/f88/health/{evalset.parquet, calib.json, health88.json}
CPU, pandas only.  Training signals only (h79_evalset asserts locked_v2 absent).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "health"))
import h79_evalset as EV  # noqa: E402
import h83_ticks as T  # noqa: E402

OUT = EV.H.DCW / "final_v3_work" / "f88" / "health"
OUT.mkdir(parents=True, exist_ok=True)
# weak labels recomputed with the fault-free hb_data (h79_evalset reads the cached health/weak_labels.parquet)
EV.H.weak_labels().to_parquet(OUT / "weak_labels.parquet", index=False)
REDIRECT = {EV.H.HB / "weak_labels.parquet": OUT / "weak_labels.parquet",
            EV.H.DCW / "health79" / "evalset.parquet": OUT / "evalset.parquet"}
_rp = pd.read_parquet


def _read(p, *a, **kw):
    return _rp(REDIRECT.get(Path(p), p), *a, **kw)


pd.read_parquet = _read
EV.OUT = OUT
EV.main()
pd.read_parquet = _rp
new = pd.read_parquet(OUT / "evalset.parquet")
old = pd.read_parquet(EV.H.DCW / "health79" / "evalset.parquet")
k = ["DeviceId", "detector"]
cnt = lambda e: e.groupby(["cls", "tier"]).size().to_dict()  # noqa: E731
# h83 calib on the new evalset: stage_calib reads OUT/ticks.parquet + health79/evalset.parquet, writes OUT/calib.json
orig_out = T.OUT
shutil.copy(orig_out / "ticks.parquet", OUT / "ticks.parquet")
T.OUT = OUT
pd.read_parquet = _read


class _A:
    pass


T.stage_calib(_A())
pd.read_parquet = _rp
(OUT / "ticks.parquet").unlink()
c_new = json.load(open(OUT / "calib.json"))
c_old = json.load(open(orig_out / "calib.json"))
res = {"evalset_old": {str(x): v for x, v in cnt(old).items()}, "evalset_new": {str(x): v for x, v in cnt(new).items()},
       "healthy_N_added": int(len(new[(new.cls == "healthy") & (new.tier == "N")].merge(
           old[(old.cls == "healthy") & (old.tier == "N")][k], on=k, how="left", indicator=True).query("_merge == 'left_only'"))),
       "calib_old_chat": c_old.get("chat"), "calib_new_chat": c_new.get("chat"),
       "calib_old_rapid": c_old.get("rapid_lim_new"), "calib_new_rapid": c_new.get("rapid_lim_new")}
json.dump(res, open(OUT / "health88.json", "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
