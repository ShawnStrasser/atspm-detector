"""Track B, stage 21: B10 (wider TCN) -> B7 (state-space backbone) -> six-fold baseline.
Unattended and resumable: every step is skipped when its output exists; relaunch after a crash.

Variants fixed before launch:
  B10  --arch tcn --width 192          2x channels (96 -> 192), 2.68 M weights (TCN 0.70 M)
  B7   --arch s4d --blocks 4           bidirectional S4D (pure PyTorch, `trackb_ssm.py`), 0.50 M
Gate (identical to `trackb_b3b4.py`, fold-0 blend before the decoder, m30), each variant
against the PLAIN TCN: screen seed 0 vs tb_tcn_f0_oof >= +0.3 pt; confirm seed 1,
mean(s0, s1) vs three-seed TCN mean (.9750) >= +0.3 pt.
Six folds: the kept variant with the higher two-seed mean, else the plain TCN (fold 0 exists).
Then trackb_eval_full.py for that set AND for the stage-13 GRU files, same harness.

    python research/code/neural/b10b7_chain.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import trackb_b3b4 as C  # noqa: E402  (run / evaluate / step / trail, same gate)

B10 = {"b10": ["--width", "192"]}
B7 = {"b7": ["--arch", "s4d", "--blocks", "4"]}   # later "--arch" overrides the base one
GRU13 = ",".join(f"gru2_oof_f{k}" for k in range(6))


def base_champ() -> dict:
    b2 = json.load(open(C.EVAL / "b2.json"))["candidates"]
    names = {"c0": "base", "c1": "seed1", "c2": "seed2"}
    return dict(name="tcn", args=list(C.BASE_ARGS), seeds=dict(C.TCN),
                scores={c: {m: C.bb(b2, n, m) for m in C.MS} for c, n in names.items()})


def full(tags: list[str], out: str) -> None:
    if not (C.EVAL / f"{out}.json").exists():
        C.say(f"EVAL_FULL {out}: {tags}")
        C.sh([C.NN + "trackb_eval_full.py", "--tags", ",".join(tags), "--out", out],
             "trackB_eval.log")


def main() -> None:
    for d in (C.STATE, C.EVAL, C.RUNS, C.PREDS, C.LOG):
        d.mkdir(parents=True, exist_ok=True)
    kept = []
    for name, cands in (("b10", B10), ("b7", B7)):
        C.say(f"=== {name}, fold 0 ===")
        ch = C.step(name, base_champ(), cands, allow_h1=False)
        if ch["name"] != "tcn":
            kept.append(ch)
    if kept:
        win = max(kept, key=lambda c: sum(s["m30"] for s in c["scores"].values()))
    else:
        win = base_champ()
    C.say(f"six-fold set = {win['name']}: {' '.join(win['args'])}")
    json.dump(win, open(C.STATE / "b10b7_champion.json", "w"), indent=1)
    tags = [next(iter(win["seeds"].values()))]
    for k in range(1, 6):
        tags.append(C.run(f"tb_{win['name']}_f{k}", k, win["args"], 0))
    full(tags, f"final6_{win['name']}")
    full(GRU13.split(","), "final6_gru13")
    C.say("ALLDONE")


if __name__ == "__main__":
    main()
