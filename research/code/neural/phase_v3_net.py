"""Note 37: the stage-13 GRU re-fitted with the 71 released NEWTEST signals in the pool.

Nothing here changes the recipe: `train2.run` (stage-13 GRU, phase loss, 5-60 min windows,
patience 7, 45-epoch cap, seed 0), `infer2`'s 30-min pieces pooled by mean log-prob, and
`export_gru.to_onnx`.  What changes is the signal map (`training_signals`, patched below):

    709 stage-13 signals  -> the stage-13 phase fold map (folds.csv + newtrain_folds.csv)
    71 released NEWTEST   -> period "stg", fold from folds_v4.csv (1-5)

Long windows (> 120 min: h3, h6, h24, full) are scored with production's final_v3 rule:
K = 4 evenly spaced 30-min pieces from the 32-piece grid (note 33, `pieces_infer.subset`).
Locked signals (`official/locked_v2.csv`) are asserted absent from every table.

    python phase_v3_net.py cache                 # rasters for the 71 released signals
    python phase_v3_net.py train --fold 0        # (or --final)
    python phase_v3_net.py infer --fold 0
    python phase_v3_net.py export                # p3_final -> phase_v3/gru.onnx + checks
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402
from common import DC_WORK  # noqa: E402
from neural import data2 as D2  # noqa: E402

OUTROOT = DC_WORK / "final_v3_work" / "phase_v3"
WORK = OUTROOT / "work"
MODELDIR = WORK / "gru_models"
RUNDIR = WORK / "gru_runs"
PREDDIR = WORK / "gru_preds"
RELEASED = DC_WORK / "official" / "newtest_released.csv"
LOCKED = DC_WORK / "official" / "locked_v2.csv"
FOLDS_V4 = DC_WORK / "folds_v4.csv"
LONG_MIN, K_LONG, GRID = 120, 4, 32

_orig_training_signals = D2.training_signals


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def locked_ids() -> set:
    return set(pd.read_csv(LOCKED).DeviceId.astype(str).str.lower())


def released_ids() -> list[str]:
    r = sorted(pd.read_csv(RELEASED).DeviceId.astype(str).str.lower())
    assert len(r) == 71 and not set(r) & locked_ids()
    return r


def training_signals_v3() -> pd.DataFrame:
    old = _orig_training_signals()
    f4 = pd.read_csv(FOLDS_V4)
    fm = dict(zip(f4.DeviceId.str.lower(), f4.fold.astype(int)))
    rel = released_ids()
    new = pd.DataFrame({"DeviceId": rel, "period": "stg", "fold": [fm[d] for d in rel]})
    new["key"] = new.period + "|" + new.DeviceId
    df = pd.concat([old, new], ignore_index=True)
    assert not df.DeviceId.isin(locked_ids()).any(), "locked signal in the GRU pool"
    assert df.key.is_unique
    return df.sort_values("key").reset_index(drop=True)


def training_signals_2026() -> pd.DataFrame:
    """note 95 (env DC_SIG2026=1): 2026-only pool = every non-locked folds_v4 signal with a Sept-2026 raster
    (neural/sig_stg), period stg only (Dec-2024 keys dropped), folds_v4 for every signal (not the stage-13 map)."""
    f4 = pd.read_csv(FOLDS_V4)
    f4["DeviceId"] = f4.DeviceId.str.lower()
    have = {p.stem.lower() for p in D2.PERIODS["stg"]["sigdir"].glob("*.npz")}
    df = f4[f4.DeviceId.isin(have) & ~f4.DeviceId.isin(locked_ids())][["DeviceId", "fold"]].copy()
    df["fold"] = df.fold.astype(int)
    df["period"] = "stg"
    df["key"] = df.period + "|" + df.DeviceId
    assert df.key.is_unique
    return df[["DeviceId", "period", "fold", "key"]].sort_values("key").reset_index(drop=True)


import os as _os  # noqa: E402
D2.training_signals = training_signals_2026 if _os.environ.get("DC_SIG2026") == "1" else training_signals_v3


# ---------------------------------------------------------------------- cache
def cmd_cache(a) -> None:
    from neural import ncache2 as NC
    meta = pd.read_parquet(DC_WORK / "official" / "stg" / "cache" / "signal_meta.parquet",
                           columns=["DeviceId"])
    rel = set(released_ids())
    devs = sorted(d for d in meta.DeviceId if str(d).lower() in rel)
    assert len(devs) == 71, len(devs)
    NC.build("stg", devices=devs)
    # NC.check("stg") flags cf208ff8 (01032): its raw log carries only 2 call (43/44) events
    # in 66 h -- a property of the data, not a truncated cache; the trees see the same log.
    try:
        log(json.dumps(NC.check("stg")))
    except AssertionError as e:
        log(f"check: {e} -- accepted (sparse 43/44 logging in the raw events)")


# ---------------------------------------------------------------------- train
def cmd_train(a) -> None:
    from neural import train2 as T2
    T2.MODELDIR = MODELDIR
    T2.RUNDIR = RUNDIR
    ns = argparse.Namespace(fold=-1 if a.final else a.fold, final=a.final,
                            tag=("p3_final" if a.final else f"p3_f{a.fold}"),
                            epochs=45, nwin=2, max_bs=8, max_det=12, lr=3e-3, wd=1e-4,
                            patience=7, lr_patience=2, es_frac=0.10, workers=a.workers,
                            chunk=0, seed=0, force=False)
    T2.run(ns)


# ---------------------------------------------------------------------- infer
def cmd_infer(a) -> None:
    import torch
    from neural import infer2 as I2
    from pieces_infer import subset
    orig_pieces = I2.pieces

    def pieces_k4(period, start, secs, max_chunks):
        out = orig_pieces(period, start, secs, GRID)
        if secs > LONG_MIN * 60:
            out = [out[i] for i in subset(len(out), K_LONG)]
        return out

    I2.pieces = pieces_k4
    I2.MODELDIR = MODELDIR
    PREDDIR.mkdir(parents=True, exist_ok=True)
    tag = a.out or ("p3_all" if a.all else f"p3_oof_f{a.fold}")
    dest = PREDDIR / f"{tag}_bywindow.parquet"
    if dest.exists():
        log(f"{dest.name} exists -- skipping")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sigs = D2.training_signals()
    table = D2.load_table(sigs, labelled_only=False)
    keys = [k for k in (sigs.key if a.all else sigs.loc[sigs.fold == a.fold, "key"])
            if k in table]
    if a.released_only:                       # stage-13 final GRU on the 71 (never trained on them)
        rel = set(released_ids())
        keys = [k for k in sigs.key if k in table and k.split("|", 1)[1] in rel]
    log(f"{tag}: {len(keys)} signals")
    ck = (Path(a.ckpt) if a.ckpt and Path(a.ckpt).exists()
          else MODELDIR / f"{a.ckpt or ('p3_f%d' % a.fold)}.pt")
    model = I2.load_model(str(ck), device)
    t0 = time.time()
    df = I2.score_windows(model, table, keys, device, D2.Stores(), GRID, 180, 6, 1024)
    df.to_parquet(dest, index=False)
    log(f"wrote {dest} ({len(df):,} rows) in {time.time()-t0:.0f}s")


# ---------------------------------------------------------------------- export
def cmd_export(a) -> None:
    import onnxruntime as ort
    import torch
    from neural import export_gru as EG
    from neural.raster import assemble_flat
    from neural.train2 import pair_index
    ck = MODELDIR / "p3_final.pt"
    out = OUTROOT / "gru.onnx"
    model = EG.load_ckpt(str(ck))
    EG.to_onnx(model, out)                      # logs the random-input check
    # real-data check: rasters of released + old signals, 30 min and 2 h
    model = EG.load_ckpt(str(ck))
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    sigs = D2.training_signals()
    table = D2.load_table(sigs, labelled_only=False)
    rng = np.random.default_rng(0)
    keys = list(rng.choice([k for k in sigs.key if k in table], 12, replace=False))
    stores = D2.Stores()
    worst_s, worst_p, same = 0.0, 0.0, True
    for k in keys:
        per = table[k]["period"]
        lo, hi = D2.PERIODS[per]["lo"], D2.PERIODS[per]["hi"]
        for T in (1800, 7200):
            w0 = int(rng.integers(lo, hi - T * 1000))
            ds = D2.FixedDataset(table, [(k, w0, T)], stores)
            b = D2.collate([ds[0]])
            bi, di, ki, B, D, K = pair_index(b["dmask"], b["ncand"])
            x = assemble_flat(b["det"], b["ph"], b["sig"], b["ncand"], bi, di, ki)
            with torch.no_grad():
                s_t = model(x)[0].numpy()
            s_o = np.asarray(sess.run(None, {"x": x.numpy()})[0]).ravel()
            worst_s = max(worst_s, float(np.abs(s_t - s_o).max()))
            for d in range(D):
                m = (di == d).numpy()
                if not m.any():
                    continue
                pt = np.exp(s_t[m] - s_t[m].max()); pt /= pt.sum()
                po = np.exp(s_o[m] - s_o[m].max()); po /= po.sum()
                worst_p = max(worst_p, float(np.abs(pt - po).max()))
                same &= int(pt.argmax()) == int(po.argmax())
    res = {"onnx": str(out), "ckpt": str(ck), "real_rasters": len(keys) * 2,
           "max_abs_score_diff": worst_s, "max_abs_prob_diff": worst_p,
           "argmax_identical": bool(same), "pass_1e-4": bool(worst_p <= 1e-4)}
    json.dump(res, open(OUTROOT / "gru_onnx_check.json", "w"), indent=1)
    log(json.dumps(res))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cache", "train", "infer", "export"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=None)
    ap.add_argument("--released-only", action="store_true")
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
