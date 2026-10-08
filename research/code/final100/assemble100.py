"""Note 100c: package `%DC_WORK%/final_v3_candidate_v5b` = v5 (copied; v5 untouched) with the function network swapped:
three full-data plain width-32 siba members x100_w32full{,_s1,_s2} (s95_sibafull recipe at --width 32, lr replayed from
the 18 w32 fold runs: s100/lr_w32_full.json) and the stackers mean3 / single refitted on the w32 OOF (pkg100.py ->
final_v3_work/v3fit100). Everything else = v5 (re-exported from v3fit100 = a copy of v3fit95, parity-checked).

    python assemble100.py init     copy v5 -> v5b (once)
    python assemble100.py func     assemble95 func (trees / lanes / stackers / setback ONNX) from v3fit100 into v5b + parity
    python assemble100.py siba     w32 members exported (export83 graphs) + manifest + parity (package FuncNet vs torch)
    python assemble100.py card     model card entry final_v5b_note100 + sha256
Then: python <v5b>/check.py --freeze ; python <v5b>/check.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "final95"))
import assemble95 as A95  # noqa: E402

DCW = A95.DC_WORK
FIT = DCW / "final_v3_work" / "v3fit100"
SRC = DCW / "final_v3_candidate_v5"
PKG = DCW / "final_v3_candidate_v5b"
WTS = PKG / "weights"
TAGS = ["x100_w32full", "x100_w32full_s1", "x100_w32full_s2"]
LR = DCW / "s100" / "lr_w32_full.json"
A95.FIT, A95.PKG, A95.WTS, A95.TAGS = FIT, PKG, WTS, TAGS
A95.A.FIT, A95.A.PKG, A95.A.WTS = FIT, PKG, WTS


def cmd_init():
    assert not PKG.exists(), PKG
    shutil.copytree(SRC, PKG, ignore=shutil.ignore_patterns("__pycache__"))
    print(f"copied {SRC.name} -> {PKG.name}")


def cmd_func():
    assert (FIT / "stacker" / "stacker_mean3.json").exists()
    assert "x100_w32" in json.load(open(FIT / "stacker" / "stacker_mean3.json"))["net"]
    A95.cmd_func()


def cmd_siba():
    import siba90 as S9
    S9.TAGS = TAGS
    S9.X.PKG = PKG
    S9.X.OUT = FIT
    S9.X.TMP = DCW / "final_v3_work" / "f100" / "export_tmp"
    S9.X.MEMBERS = [(t, f"{t}_full.pt") for t in TAGS]
    S9.cmd_export()
    fd = WTS / "funcnet"
    man = json.load(open(fd / "manifest.json"))
    lr = json.load(open(LR))
    man["trained"] = (f"note 100c FULL {' / '.join(TAGS)}: plain siba, TCN width 32 (7 blocks), 2026-only (Sept-2026 rows "
                      f"of every non-locked training signal), labels v4q (tcn53/func_rows_v4q_2026.parquet), {len(lr)} epochs "
                      "on the schedule replayed from the 18 w32 fold runs (s100/lr_w32_full.json), research tcn69_func (snap95)")
    man["note"] = "note 100c production net (package v5b): 3 full-data width-32 members, 2026-only"
    json.dump(man, open(fd / "manifest.json", "w"), indent=1)
    S9.X.cmd_parity()
    shutil.move(str(FIT / "siba_parity84.json"), str(FIT / "siba_parity100.json"))


def cmd_card():
    card = json.load(open(WTS / "model_card.json"))
    card["final_v5b_note100"] = {
        "package": "final_v3_candidate_v5b = v5 with the function network swapped for 3 width-32 siba members (note 100b/c)",
        "function_network": {"members": TAGS, "width": 32, "nblk": 7, "epochs": len(json.load(open(LR))),
                             "recipe": "s95_sibafull at --width 32; plain (no distillation)"},
        "stackers": "mean3 / single refitted on the x100_w32 six-fold OOF (pkg100.py, v3fit100), Sept-2026 rows, v4q",
        "oof_headline_ge30": {"E": 0.9307, "R": 0.9397, "vs_v5_E_pt": [0.023, -0.089, 0.141],
                              "vs_v5_R_pt": [0.021, -0.088, 0.132], "m5_E_vs_v5": [-0.069, -0.298, 0.16],
                              "m10_E_vs_v5": [-0.203, -0.395, -0.007]},
        "unchanged_from_v5": "phase ranker / TCN / decoder, function trees, lanes D, setback, health rules, decode rules",
        "locked": "not used"}
    card["file_sha256"] = {str(q.relative_to(WTS)).replace("\\", "/"): hashlib.sha256(q.read_bytes()).hexdigest()
                           for q in sorted(WTS.rglob("*")) if q.is_file() and q.name != "model_card.json"}
    card["total_model_bytes"] = sum(q.stat().st_size for q in WTS.rglob("*") if q.is_file() and q.name != "model_card.json")
    json.dump(card, open(WTS / "model_card.json", "w"), indent=1, default=str)
    print("model card updated", flush=True)


if __name__ == "__main__":
    {"init": cmd_init, "func": cmd_func, "siba": cmd_siba, "card": cmd_card}[sys.argv[1]]()
