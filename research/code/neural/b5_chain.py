"""Track B, B5 chain: screen two sibling-context variants on fold 0, confirm, six folds.
Unattended and resumable (every step is skipped when its output exists).

Gate (pre-registered, identical to `trackb_b3b4.py`; fold-0 blend before the decoder, m30):
  screen   best variant seed 0 vs TCN seed 0 (tb_tcn_f0_oof): needs >= +0.3 pt;
  confirm  seed 1 of that variant; mean(s0, s1) vs three-seed TCN mean (.9750) >= +0.3 pt.
Only a kept variant earns folds 1-5 and `trackb_eval_full.py`.

    python research/code/neural/b5_chain.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import rpath  # noqa: F401,E402
import trackb_b3b4 as C  # noqa: E402

B5 = {"b5a": ["--variant", "b5a"], "b5b": ["--variant", "b5b"]}


def run(tag: str, fold: int, args: list[str], seed: int) -> str:
    if not (C.RUNS / f"{tag}.done.json").exists():
        C.say(f"TRAIN {tag} (fold {fold}): {' '.join(args)} --seed {seed}")
        C.sh([C.NN + "b5_train.py", "--tag", tag, "--fold", str(fold), *args,
              "--seed", str(seed), "--epochs", str(C.EP)], f"trackB_{tag}.log")
    if not (C.PREDS / f"{tag}_oof_bywindow.parquet").exists():
        C.say(f"INFER {tag}")
        C.sh([C.NN + "b5_infer.py", "--ckpt", tag, "--fold", str(fold)], f"trackB_{tag}.log")
    return f"{tag}_oof"


def main() -> None:
    C.run = run
    for d in (C.STATE, C.EVAL, C.RUNS, C.PREDS, C.LOG):
        d.mkdir(parents=True, exist_ok=True)
    b2 = json.load(open(C.EVAL / "b2.json"))["candidates"]
    names = {"c0": "base", "c1": "seed1", "c2": "seed2"}
    champ = dict(name="tcn", args=list(C.BASE_ARGS), seeds=dict(C.TCN),
                 scores={c: {m: C.bb(b2, n, m) for m in C.MS} for c, n in names.items()})
    C.say("=== B5: sibling context inside the network, fold 0 ===")
    champ = C.step("b5", champ, B5, allow_h1=False)
    json.dump(champ, open(C.STATE / "b5_champion.json", "w"), indent=1)
    if champ["name"] == "tcn":
        C.say("B5 not kept -- no six-fold run")
        C.say("ALLDONE")
        return
    C.say(f"=== six folds of {champ['name']} ===")
    tags = [next(iter(champ["seeds"].values()))]
    for k in range(1, 6):
        tags.append(run(f"tb_{champ['name']}_f{k}", k, champ["args"], 0))
    out = f"final6_{champ['name']}"
    if not (C.EVAL / f"{out}.json").exists():
        C.sh([C.NN + "trackb_eval_full.py", "--tags", ",".join(tags), "--out", out],
             "trackB_eval.log")
    C.say("ALLDONE")


if __name__ == "__main__":
    main()
