"""Stage 33: the GRU on K evenly spaced 30-minute pieces of a long sample (per-piece OOF).

Production (`model/gru_onnx.py`) cuts a sample into 30-minute pieces and pools them by
the mean log-probability; its cost grows with the number of pieces.  This script runs a
stage-13 GRU fold model over the long OOF windows (h3, h6, h24, full) with the SAME
pieces `infer2.score_windows` used (max 32), but keeps each piece's log-probabilities,
then pools every K-piece subset the way `split_range(max_chunks=K)` would choose it
(`linspace(0, n-1, K).round()`).  Pooling: `lp` = mean log-prob (production), `pr` =
mean probability (the alternative asked about).

    python research/code/neural/pieces_infer.py --fold 0

-> %DC_WORK%/trackB/pieces/gru2_pieces_f{fold}.parquet
   DeviceId, Detector, cand_phase, win, variant, prob, n_pieces
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402
from common import DC_WORK  # noqa: E402
from neural import data2 as D2  # noqa: E402
from neural import infer2 as I2  # noqa: E402
from neural.train2 import forward_batch  # noqa: E402

OUT = DC_WORK / "trackB" / "pieces"
LONG = ("h3_", "h6_", "h24_", "full")
KS = (1, 2, 3, 4, 6, 8)


def subset(n: int, k: int) -> np.ndarray:
    if k >= n:
        return np.arange(n)
    return np.linspace(0, n - 1, k).round().astype(int)


@torch.no_grad()
def run(model, table, keys, device, stores, budget=180, max_bs=6, chunk=1024):
    acc: dict = {}     # (key, win, det) -> [cand, {piece_idx: lp}, n_pieces]
    for period in ("dec", "stg"):
        pk = [k for k in keys if table[k]["period"] == period]
        if not pk:
            continue
        for name, start, secs in I2.WINDOWS[period]:
            if not name.startswith(LONG):
                continue
            chunks = I2.pieces(period, start, secs, 32)
            pidx = {a: i for i, (a, _) in enumerate(chunks)}
            plan = [(k, a, (b - a) // 1000) for k in pk for a, b in chunks]
            ds = D2.FixedDataset(table, plan, stores)
            for bidx in D2.batches_of(plan, budget_minutes=budget, max_bs=max_bs):
                batch = D2.collate([ds[i] for i in bidx])
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=(device == "cuda")):
                    logit = forward_batch(model, batch, device, chunk=chunk)
                lp = F.log_softmax(logit, dim=2).float().cpu().numpy()
                for b, (key, dev, w0, dets, cand) in enumerate(batch["meta"]):
                    K = len(cand)
                    for j, d in enumerate(dets):
                        e = acc.setdefault((key, name, int(d)), [cand, {}, len(chunks)])
                        e[1][pidx[int(w0)]] = lp[b, j, :K].copy()
            I2.log(f"  {period} {name}: {len(chunks)} piece(s) x {len(pk)} signals")
    rows = []
    for (key, win, det), (cand, parts, n) in acc.items():
        dev = key.split("|", 1)[1] + ("@stg" if key.startswith("stg|") else "")
        have = np.array(sorted(parts))
        L = np.stack([parts[i] for i in have])            # [pieces, K]
        for k in KS + (0,):                                # 0 = all pieces (reference)
            sel = have if k == 0 else np.array(
                [i for i in subset(n, k) if i in parts])
            if len(sel) == 0:
                continue
            m = np.stack([parts[i] for i in sel])
            v = m.mean(0)
            p_lp = np.exp(v - v.max()); p_lp /= p_lp.sum()
            p_pr = np.exp(m).mean(0); p_pr /= p_pr.sum()
            tagk = "all" if k == 0 else f"k{k}"
            for c, a_, b_ in zip(cand, p_lp, p_pr):
                rows.append((dev, int(det), int(c), win, f"lp_{tagk}", float(a_), len(sel)))
                rows.append((dev, int(det), int(c), win, f"pr_{tagk}", float(b_), len(sel)))
        del L
    return pd.DataFrame(rows, columns=["DeviceId", "Detector", "cand_phase", "win",
                                       "variant", "prob", "n_pieces"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"gru2_pieces_f{a.fold}.parquet"
    if dest.exists():
        I2.log(f"{dest.name} exists -- skipping")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sigs = D2.training_signals()
    table = D2.load_table(sigs, labelled_only=False)
    keys = [k for k in sigs.loc[sigs.fold == a.fold, "key"] if k in table]
    I2.log(f"fold {a.fold}: {len(keys)} signals")
    model = I2.load_model(f"gru2_f{a.fold}", device)
    t0 = time.time()
    df = run(model, table, keys, device, D2.Stores())
    df.to_parquet(dest, index=False)
    I2.log(f"wrote {dest} ({len(df):,} rows) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
