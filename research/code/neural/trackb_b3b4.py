"""Track B (stage 16): B3 (dropout + augmentation) then B4 (mixed sample lengths), fold 0,
TCN backbone -- then all six folds of whatever survived.  Unattended and resumable.

Every step is skipped when its output exists (`runs/<tag>.done.json`, the prediction
parquet, the eval json), so after a crash just launch it again.  Paths are native Windows
paths handed to subprocesses, so nothing depends on Git Bash path conversion.

Gate (pre-registered, fold-0 blend before the decoder, metric m30):
  screen   candidate seed 0 vs champion seed 0: keep going only if >= +0.3 pt;
  confirm  fit seed 1; mean(candidate s0, s1) vs mean(champion seeds) >= +0.3 pt => kept.
  B4 may pass on 1 h instead (>= +0.3 pt at h1) provided 30 min does not drop > 0.2 pt.
Baseline seeds: tb_tcn_f0_oof / tb_b2s1_f0_oof / tb_b2s2_f0_oof, read from eval/b2.json.

    python research/code/neural/trackb_b3b4.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
import rpath  # noqa: F401,E402
from common import DC_WORK  # noqa: E402

PY = sys.executable
W = DC_WORK / "trackB"
LOG = DC_WORK / "logs"
STATE, EVAL, RUNS, PREDS = W / "state", W / "eval", W / "runs", W / "preds"
EP = 80
MARGIN, DROP_TOL = 0.003, 0.002
BASE_ARGS = ["--arch", "tcn"]
TCN = {"c0": "tb_tcn_f0_oof", "c1": "tb_b2s1_f0_oof", "c2": "tb_b2s2_f0_oof"}
B3 = {"b3a": ["--dropout", "0.10", "--chdrop", "0.15", "--jitter", "300"],
      "b3b": ["--dropout", "0.20", "--chdrop", "0.30", "--jitter", "300"]}
B4 = ["--minutes", "5,10,15,30,30,60,120,360", "--det-budget", "21600"]
NN = "research/code/neural/"


def say(m: str) -> None:
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {m}", flush=True)


def sh(args: list[str], logname: str) -> None:
    with open(LOG / logname, "a") as f:
        r = subprocess.run([PY] + args, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise SystemExit(f"FAILED ({r.returncode}): {' '.join(args)}")


def run(tag: str, fold: int, args: list[str], seed: int) -> str:
    if not (RUNS / f"{tag}.done.json").exists():
        say(f"TRAIN {tag} (fold {fold}): {' '.join(args)} --seed {seed}")
        sh([NN + "trackb_train.py", "--tag", tag, "--fold", str(fold), *args,
            "--seed", str(seed), "--epochs", str(EP)], f"trackB_{tag}.log")
    if not (PREDS / f"{tag}_oof_bywindow.parquet").exists():
        say(f"INFER {tag}")
        sh([NN + "trackb_infer.py", "--ckpt", tag, "--fold", str(fold)], f"trackB_{tag}.log")
    return f"{tag}_oof"


def evaluate(out: str, spec: dict) -> dict:
    if not (EVAL / f"{out}.json").exists():
        p = STATE / f"spec_{out}.json"
        json.dump({k: [v] for k, v in spec.items()}, open(p, "w"), indent=1)
        say(f"EVAL {out}: {spec}")
        sh([NN + "trackb_eval.py", "--spec", str(p), "--out", out], "trackB_eval.log")
    return json.load(open(EVAL / f"{out}.json"))["candidates"]


def bb(c: dict, name: str, m: str = "m30") -> float:
    return float(c[name]["blend_before"][m])


def mean(xs):
    return sum(xs) / len(xs)


def passes(cand: dict, base: dict, allow_h1: bool) -> tuple[bool, dict]:
    d = {m: cand[m] - base[m] for m in cand}
    ok = d["m30"] >= MARGIN or (allow_h1 and d["h1"] >= MARGIN and d["m30"] >= -DROP_TOL)
    return ok, {m: round(100 * v, 3) for m, v in d.items()}


def trail(entry: dict) -> None:
    p = STATE / "trail.json"
    t = json.loads(p.read_text()) if p.exists() else []
    t = [e for e in t if e["step"] != entry["step"]] + [entry]
    p.write_text(json.dumps(t, indent=1))
    say(json.dumps(entry))


MS = ("m5", "m10", "m30", "h1", "h3")


def step(name: str, champ: dict, cands: dict[str, list[str]], allow_h1: bool) -> dict:
    """champ = {args, seeds:{name: tag}, scores:{name:{m:acc}}}.  Returns the new champion."""
    c0 = next(iter(champ["seeds"]))
    tags = {c: run(f"tb_{c}_f0", 0, champ["args"] + a, 0) for c, a in cands.items()}
    res = evaluate(name, {**{c0: champ["seeds"][c0]}, **tags})
    s0 = {c: {m: bb(res, c, m) for m in MS} for c in cands}
    best = max(s0, key=lambda c: s0[c]["m30"])
    ok, d = passes(s0[best], champ["scores"][c0], allow_h1)
    entry = dict(step=f"{name}_screen", champion_seed0=champ["seeds"][c0],
                 scores={c: {m: round(v, 5) for m, v in s.items()} for c, s in s0.items()},
                 best=best, delta_pt=d, passed=ok)
    trail(entry)
    if not ok:
        return champ
    t1 = run(f"tb_{best}s1_f0", 0, champ["args"] + cands[best], 1)
    res1 = evaluate(f"{name}c", {f"{best}_s1": t1})
    s1 = {m: bb(res1, f"{best}_s1", m) for m in MS}
    cm = {m: mean([s0[best][m], s1[m]]) for m in MS}
    bm = {m: mean([v[m] for v in champ["scores"].values()]) for m in MS}
    ok, d = passes(cm, bm, allow_h1)
    trail(dict(step=f"{name}_confirm", cand=best, seed1={m: round(v, 5) for m, v in s1.items()},
               cand_mean={m: round(v, 5) for m, v in cm.items()},
               champ_mean={m: round(v, 5) for m, v in bm.items()},
               n_champ_seeds=len(champ["scores"]), delta_pt=d, kept=ok))
    if not ok:
        return champ
    return dict(name=best, args=champ["args"] + cands[best],
                seeds={f"{best}_s0": tags[best], f"{best}_s1": t1},
                scores={f"{best}_s0": s0[best], f"{best}_s1": s1})


def main() -> None:
    for d in (STATE, EVAL, RUNS, PREDS, LOG):
        d.mkdir(parents=True, exist_ok=True)
    b2 = json.load(open(EVAL / "b2.json"))["candidates"]
    names = {"c0": "base", "c1": "seed1", "c2": "seed2"}
    champ = dict(name="tcn", args=list(BASE_ARGS), seeds=dict(TCN),
                 scores={c: {m: bb(b2, n, m) for m in MS} for c, n in names.items()})
    say("=== B3: dropout + augmentation, fold 0 ===")
    champ = step("b3", champ, B3, allow_h1=False)
    say(f"after B3 champion = {champ['name']}: {' '.join(champ['args'])}")
    say("=== B4: mixed sample lengths 5 min .. 6 h, fold 0 ===")
    b4 = "b4" if champ["name"] == "tcn" else f"b4{champ['name'][-1]}"
    champ = step("b4", champ, {b4: B4}, allow_h1=True)
    say(f"after B4 champion = {champ['name']}: {' '.join(champ['args'])}")
    (STATE / "best_args.txt").write_text(" ".join(champ["args"]), encoding="ascii")
    (STATE / "best_seed0.txt").write_text(next(iter(champ["seeds"].values())), encoding="ascii")
    json.dump(champ, open(STATE / "b3b4_champion.json", "w"), indent=1)
    if champ["name"] == "tcn":
        say("neither B3 nor B4 kept -- no six-fold run")
        say("ALLDONE")
        return
    say(f"=== six folds of {champ['name']} ===")
    tags = [next(iter(champ["seeds"].values()))]
    for k in range(1, 6):
        tags.append(run(f"tb_{champ['name']}_f{k}", k, champ["args"], 0))
    out = f"final6_{champ['name']}"
    if not (EVAL / f"{out}.json").exists():
        sh([NN + "trackb_eval_full.py", "--tags", ",".join(tags), "--out", out],
           "trackB_eval.log")
    say("ALLDONE")


if __name__ == "__main__":
    main()
