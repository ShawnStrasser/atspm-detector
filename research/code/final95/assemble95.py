"""Note 95: package `%DC_WORK%/final_v3_candidate_v5` = v4f's code (copied; v4f untouched) with EVERY learned part refitted
on 2026-only data (Sept-2026 staging period; Dec-2024 never used) and labels v4q:
  phase     ranker 3 seeds (phase95 fitranker -> v3fit95/phase), TCN ad_all full refit p95_ad_full (export95_phase ->
            s95/phase_tcn), decoder on the 2026 trees + TCN blend (phase95 fitdec -> v3fit95/decoder)
  function  trees 229 x 3, lanes D, setback P50, stackers mean3 + single (pkg95 -> v3fit95)
  siba      the three full-data 2026 members s95_sibafull{,_s1,_s2} (or the agreement-picked three)
  health    output fix: health_score = NaN when health_status = not_enough_data (orchestrator 2026-10-05)
Unchanged (not learned from Dec-2024 data or not learned at all): setback P10 / P90 band and pair HGB (note 41, Sept-2026
66 h), health rules, night speed, decode rules.

    python assemble95.py init     copy v4f -> v5 (once)
    python assemble95.py phase    ranker + TCN + decoder ONNX, blend.json, parity -> v3fit95/parity_phase95.json
    python assemble95.py func     function / lanes / stackers / setback ONNX + metadata, parity -> v3fit95/trees_parity95.json
    python assemble95.py siba     members (SIBA95_TAGS) exported + manifest; then parity
    python assemble95.py health   predict.py: health_score NaN for not_enough_data
    python assemble95.py card     model card entry + sha256
Then: python <pkg>/check.py --freeze ; python <pkg>/check.py
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CODE = Path(__file__).resolve().parents[1]
for _d in ("final90", "final84", "final83", "final75"):
    sys.path.insert(0, str(CODE / _d))
sys.path.insert(0, str(CODE))
import rpath  # noqa: E402,F401
from common import DC_WORK  # noqa: E402
import assemble90 as A  # noqa: E402

FIT = DC_WORK / "final_v3_work" / "v3fit95"
PKG = DC_WORK / "final_v3_candidate_v5"
SRC = DC_WORK / "final_v3_candidate_v4f"
WTS = PKG / "weights"
TCN_ONNX = DC_WORK / "s95" / "phase_tcn" / "p95_ad_full.onnx"
A.FIT, A.PKG, A.WTS, A.TCN_ONNX = FIT, PKG, WTS, TCN_ONNX
TAGS = os.environ.get("SIBA95_TAGS", "s95_sibafull,s95_sibafull_s1,s95_sibafull_s2").split(",")


def cmd_init():
    assert not PKG.exists(), PKG
    shutil.copytree(SRC, PKG, ignore=shutil.ignore_patterns("__pycache__"))
    print(f"copied {SRC.name} -> {PKG.name}")


def cmd_phase():
    OT, NumpyBooster, TO = A._onnx_tools()
    # ranker (3 seeds)
    pm = json.load(open(WTS / "phase_lgbm_v5.json"))
    fit = json.load(open(FIT / "phase" / "phase_fit95.json"))
    assert pm["features"] == fit["features"], "phase feature list changed"
    res = {}
    for s in range(3):
        nb = NumpyBooster(FIT / "phase" / f"phase_lgbm_v5_s{s}.txt")
        OT.to_onnx_te5(nb, WTS / f"phase_lgbm_v5_s{s}.onnx")
        X = pd.read_parquet(DC_WORK / "s95" / "phase" / "pool.parquet", columns=pm["features"]).sample(
            20000, random_state=95).to_numpy(np.float64, na_value=np.nan)
        res[f"phase_lgbm_v5_s{s}.onnx"] = A._check(OT, TO, nb, WTS / f"phase_lgbm_v5_s{s}.onnx", X)
    pm["note95"] = {"n_estimators": fit["n_estimators"], "rule": fit["rule"], "trained_on": fit["trained_on"],
                    "data": "2026-only: Sept-2026 staging period of every non-locked pool signal (Dec-2024 dropped)"}
    json.dump(pm, open(WTS / "phase_lgbm_v5.json", "w"), indent=1)
    # TCN + decoder (assemble90 recipe, then 2026 wording)
    A.cmd_phase()
    par = json.load(open(TCN_ONNX.parent / "parity86_phase.json"))
    b = json.load(open(WTS / "blend.json"))
    b["network"]["trained"] = (f"note 95: full-data refit of the ad_all recipe on the 2026-only pool (760 Sept-2026 signals, "
                               f"{par.get('epochs')} epochs, seed 0, lr replayed from the six 2026 fold runs), locked never used")
    b["network"]["onnx_parity_vs_torch"] = {"random": par["random"], "real_max_abs_logp": par["real"]["max_abs_logp"]}
    json.dump(b, open(WTS / "blend.json", "w"), indent=1)
    res.update(json.load(open(FIT / "parity_phase90.json")))
    json.dump(res, open(FIT / "parity_phase95.json", "w"), indent=1)
    mx = max(v["max_abs_out"] for k, v in res.items() if isinstance(v, dict) and "max_abs_out" in v)
    print(f"phase: ranker x3 + TCN + decoder; max |out diff| {mx:.2e}", flush=True)
    assert mx < 1e-9


def cmd_func():
    """assemble90.cmd_func with the 2026 fit folder and the f95 setback frame."""
    src = Path(A.__file__).read_text(encoding="utf-8")
    assert 'DC_WORK / "final_v3_work" / "f90" / "sb7" / "feat.parquet"' in src
    import types
    code = re.search(r"def cmd_func\(\):.*?(?=\ndef cmd_siba)", src, re.S).group(0)
    code = code.replace('DC_WORK / "final_v3_work" / "f90" / "sb7" / "feat.parquet"',
                        'DC_WORK / "final_v3_work" / "f95" / "sb7" / "feat.parquet"')
    code = code.replace('FIT / "trees_parity90.json"', 'FIT / "trees_parity95.json"')
    code = code.replace("note 90: print-lane truth (unchanged) + v4o OOF function block (final training labels)",
                        "note 95: print-lane truth (v4q) + 2026-only OOF function block, Sept-2026 rows")
    code = code.replace("note 90: sb7 features on the v4l print store and the v4o OOF function",
                        "note 95: sb7 features (Sept-2026 windows) on the 2026-only OOF function")
    code = code.replace("v4l truth (= v4o plain columns); trees = v4o OOF (final training labels, note 89b)",
                        "v4q (note 95); trees = 2026-only OOF (Sept-2026 rows; Dec-2024 never used)")
    code = code.replace("note 90 = note-86 recipe on the v4o trees OOF", "note 95 = note-86 recipe, 2026-only nets + trees")
    ns = dict(vars(A))
    exec(compile(code, "assemble95_cmd_func", "exec"), ns)
    ns["cmd_func"]()


def cmd_siba():
    import siba90 as S9
    S9.TAGS = TAGS
    S9.X.PKG = PKG
    S9.X.OUT = FIT
    S9.X.TMP = DC_WORK / "final_v3_work" / "f95" / "export_tmp"
    S9.X.MEMBERS = [(t, f"{t}_full.pt") for t in TAGS]
    S9.cmd_export()
    fd = WTS / "funcnet"
    man = json.load(open(fd / "manifest.json"))
    lr = json.load(open(DC_WORK / "s95" / "lr_siba95_full.json"))
    man["trained"] = (f"note 95 FULL {' / '.join(TAGS)}: 2026-only (Sept-2026 rows of every non-locked training signal), "
                      f"labels v4q (tcn53/func_rows_v4q_2026.parquet), {len(lr)} epochs on the schedule replayed from the "
                      "18 2026 fold runs (s95/lr_siba95_full.json), research tcn69_func (snap95)")
    man["note"] = "note 95 production net (package v5): 3 full-data members, 2026-only"
    json.dump(man, open(fd / "manifest.json", "w"), indent=1)
    S9.X.cmd_parity()
    shutil.move(str(FIT / "siba_parity84.json"), str(FIT / "siba_parity95.json"))


def cmd_health():
    p = PKG / "predict.py"
    s = p.read_text(encoding="utf-8")
    tag = "# note 95: health_score is NaN when the status is not_enough_data"
    if tag in s:
        print("already patched"); return
    old = '        h = h.rename(columns={"status": "health_status"})\n'
    assert s.count(old) == 1
    new = old + ("        " + tag + " (orchestrator 2026-10-05: a detector too quiet to judge has\n"
                 "        # no score, so filtering on health_score never keeps it by accident)\n"
                 '        h.loc[h.health_status.eq("not_enough_data"), "health_score"] = np.nan\n')
    p.write_text(s.replace(old, new), encoding="utf-8")
    print("predict.py patched")


def cmd_card():
    card = json.load(open(WTS / "model_card.json"))
    ent = json.load(open(FIT / "card95.json"))
    card["final_v5_note95"] = ent
    card["file_sha256"] = {str(q.relative_to(WTS)).replace("\\", "/"): hashlib.sha256(q.read_bytes()).hexdigest()
                           for q in sorted(WTS.rglob("*")) if q.is_file() and q.name != "model_card.json"}
    card["total_model_bytes"] = sum(q.stat().st_size for q in WTS.rglob("*") if q.is_file() and q.name != "model_card.json")
    json.dump(card, open(WTS / "model_card.json", "w"), indent=1, default=str)
    print("model card updated", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd != "init":
        assert PKG.exists(), PKG
    globals()[f"cmd_{cmd}"]()
