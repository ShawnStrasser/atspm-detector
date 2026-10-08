"""Note 105: package copy `%DC_WORK%/final_v3_candidate_v5c` = v5b with ONLY the lane model refit on the corrected lane
truth (ln105_lanefix build -> f105/ln8).  Recipe = pkg95 pkglanes (fit83.stage_lanes: lanes D both orientations, 2026-only
OOF function block, Sept-2026 labelled pairs, 3 seeds); lam = mode of the note-105 per-fold picks.  v5b untouched.

    python pkg105.py fit        full-data lane pair models -> %DC_WORK%/final_v3_work/v3fit105/lanes
    python pkg105.py assemble   copy v5b -> v5c (once), lane ONNX + lane_model.json, parity -> v3fit105/lanes_parity105.json
Then: python <v5c>/check.py --freeze ; python <v5c>/check.py
CPU 4 threads.  locked_v2 asserted absent by the loaders.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ["%s" % _v] = "4"
import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "final95"))
sys.path.insert(0, str(CODE / "lanes"))
import pkg95 as P95  # noqa: E402

O88, F = P95.O88, P95.F
DCW = F.DC_WORK
F105 = DCW / "final_v3_work" / "f105"
SRC_LN8 = F105 / "ln8"
FIT = DCW / "final_v3_work" / "v3fit105"
SRC_PKG = DCW / "final_v3_candidate_v5b"
PKG = DCW / "final_v3_candidate_v5c"
F.THREADS = 4
F.F75.THREADS = 4


def stage_fit(a):
    pk = json.load(open(SRC_LN8 / "pick_base.json"))["pick"]
    work = F105 / "pkg" / "ln8"
    work.mkdir(parents=True, exist_ok=True)
    for f in ("cues.parquet", "dets.parquet", "truth_det.parquet", "truth_phase.parquet", "truth_text_pairs.parquet"):
        shutil.copy(SRC_LN8 / f, work / f)
    json.dump({"pick": {"D.func": {str(f): "D.func@" + v.split("@")[1].split("|")[0] for f, v in pk.items()}},
               "note": "note 105 per-fold picks (ln105_lanefix decode, var base)"}, open(work / "decode_pick.json", "w"))
    (FIT / "lanes").mkdir(parents=True, exist_ok=True)
    O88.FIT = FIT
    O88.F88 = F105 / "pkg"
    P95._mask()
    O88._pkg_out()
    O88._v4m_everywhere()
    F.F83 = O88.F88
    import ln8_validate as L8
    L8.OUT = work                                   # _l8_v4l keeps it (files already present)
    F.stage_lanes(a)
    m = json.load(open(FIT / "lanes" / "lane_model.json"))
    m["labels"] = ("note 105: corrected print-lane truth (v4q + user lane answers v1/v2 + TR+R arrow rule + overlap lanes "
                   "not counted; research/labels/lane_truth_corrections_user_v2.csv) + 2026-only OOF function block, "
                   "Sept-2026 rows")
    json.dump(m, open(FIT / "lanes" / "lane_model.json", "w"), indent=1)
    # parity rows: a sample of the training frame
    import of77
    import lane_output as LO
    D, Pf, Pr = of77.sym_frames()
    il = np.flatnonzero(Pf.labelled.to_numpy())[:: 40][:2000]
    c2 = L8.c2_matrix(Pf, D, il)
    np.save(FIT / "lanes" / "X_check.npy", np.hstack([Pf.loc[il, LO.FEATURES].to_numpy(np.float32), c2[0]]))
    print("fit done", m["trained_on"], "lam", m["lam"], flush=True)


def stage_assemble(a):
    if not PKG.exists():
        shutil.copytree(SRC_PKG, PKG, ignore=shutil.ignore_patterns("__pycache__"))
        print(f"copied {SRC_PKG.name} -> {PKG.name}")
    sys.path.insert(0, str(CODE / "final90"))
    sys.path.insert(0, str(CODE / "final75"))
    import assemble90 as A
    A.PKG, A.WTS = PKG, PKG / "weights"
    OT, NumpyBooster, TO = A._onnx_tools()
    X = np.load(FIT / "lanes" / "X_check.npy").astype(np.float64)
    res = {}
    for s in range(3):
        nb = NumpyBooster(FIT / "lanes" / f"lane_pair_D_s{s}.txt")
        dst = PKG / "weights" / "lanes" / f"lane_pair_D_s{s}.onnx"
        OT.to_onnx_te5(nb, dst)
        res[dst.name] = A._check(OT, TO, nb, dst, X)
    old = json.load(open(PKG / "weights" / "lanes" / "lane_model.json"))
    m = json.load(open(FIT / "lanes" / "lane_model.json"))
    assert m["features"] == old["features"] and m["c2_columns"] == old["c2_columns"], "lane feature list changed"
    m["onnx_files"] = [f"lane_pair_D_s{s}.onnx" for s in range(3)]
    json.dump(m, open(PKG / "weights" / "lanes" / "lane_model.json", "w"), indent=1)
    card = PKG / "weights" / "model_card.json"
    c = json.load(open(card))
    c["note105"] = ("v5c = v5b with the lane model refit on the corrected lane truth (note 105: user lane answers, "
                    "TR+R arrow rule, overlap lanes not counted); everything else = v5b")
    json.dump(c, open(card, "w"), indent=1)
    json.dump(res, open(FIT / "lanes_parity105.json", "w"), indent=1)
    mx = max(v["max_abs_out"] for v in res.values())
    print(f"lanes ONNX x3; max |out diff| {mx:.2e}", flush=True)
    assert mx < 1e-9


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    globals()[f"stage_{a.stage}"](a)
