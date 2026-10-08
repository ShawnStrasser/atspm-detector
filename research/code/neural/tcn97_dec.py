"""Note 97 (fast test): siba (tcn69_func, x86_siba4l recipe) given the DECIDED phase as an input.

One extra input channel per (detector, candidate) pair, constant over time: 1 if this candidate is the phase decoder's top
phase for the detector, else 0 (all 0 where no decoder output exists = whole signal-periods outside the phase pool).
Decoder = the v4f phase pipeline (trees + TCN ad_all blend -> joint decoder), six-fold OOF: column p2_tcn_ad76 of
%DC_WORK%/s90/p87_q.parquet (phase-anonymous: a 0/1 flag on a candidate, never a phase number).

Named windows (inference, inner-val) use the decoder top of that window. Random training windows (5-30 min) use the top
of one of the same signal's evaluation windows of the nearest length class (m5 / m10 / m30; variant fixed per
(signal, start)), so the flag is about as reliable in training as at inference.

Everything else = tcn69_func.py (imported, its forward patched).  Same CLI:
    python tcn97_dec.py train --tag x97_dec --fold 0 --seed 0 --sib attn --max_det 32 --accum 2 --frows .../func_rows_v4l.parquet
    python tcn97_dec.py infer --tag x97_dec --fold 0 --keep .../cand64/phase_oof.parquet --keep_thr 0.01 --frows ...
"""
from __future__ import annotations

import os
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: F401,E402
from neural import tcn53 as M  # noqa: E402
from neural import tcn69_func as T  # noqa: E402

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
DEC_FILE = Path(os.environ.get("DC97_DEC") or (DCW / "s90" / "p87_q.parquet"))
DEC_COL = os.environ.get("DC97_COL", "p2_tcn_ad76")
_TOPS: dict | None = None
_BYLEN: dict | None = None
STATS97 = {"det_rows": 0, "flagged": 0}


def _load():
    global _TOPS, _BYLEN
    if _TOPS is not None:
        return
    q = pd.read_parquet(DEC_FILE, columns=["DeviceId", "Detector", "win", "cand_phase", DEC_COL])
    k = ["DeviceId", "Detector", "win"]
    t = q.sort_values(k + [DEC_COL], ascending=[True, True, True, False]).groupby(k, sort=False).first().reset_index()
    stg = t.DeviceId.str.endswith("@stg")
    key = np.where(stg, "stg|" + t.DeviceId.str.replace("@stg", "", regex=False), "dec|" + t.DeviceId)
    _TOPS = {(kk, w, int(d)): int(c) for kk, w, d, c in zip(key, t.win, t.Detector.astype(int), t.cand_phase.astype(int))}
    _BYLEN = {}
    for kk, w in set(zip(key, t.win)):
        cls = w.split("_")[0]
        if cls in ("m5", "m10", "m30"):
            _BYLEN.setdefault((kk, cls), []).append(w)
    for v in _BYLEN.values():
        v.sort()
    T.log(f"note 97: decoder tops {len(_TOPS):,} detector-windows from {DEC_FILE.name}:{DEC_COL}")


def _len_class(secs: float) -> str:
    m = secs / 60.0
    return min(("m5", 5.0), ("m10", 10.0), ("m30", 30.0), key=lambda x: abs(np.log(m / x[1])))[0]


def dec_flags(b: dict, K: int, T_bins: int, bw: int) -> torch.Tensor:
    """[B, D, K] float: 1 on the decoder's top candidate of each detector (0 everywhere if unknown)."""
    _load()
    B, D = b["dmask"].shape
    out = torch.zeros((B, D, K), dtype=torch.float32)
    secs = T_bins * bw / 1000.0
    for i, (key, _dev, w0, dets, cand, win) in enumerate(b["meta"]):
        if win is None:                       # random training window -> an evaluation window of the same length class
            ws = _BYLEN.get((key, _len_class(secs)))
            win = ws[zlib.crc32(f"{key}|{int(w0)}".encode()) % len(ws)] if ws else None
        if win is None:
            continue
        pos = {int(c): j for j, c in enumerate(cand)}
        for j, d in enumerate(dets):
            STATS97["det_rows"] += 1
            c = _TOPS.get((key, win, int(d)))
            if c is not None and c in pos:
                out[i, j, pos[c]] = 1.0
                STATS97["flagged"] += 1
    return out


def forward97(model, batch, dev, train=False, chunk=0):
    """tcn69_func.forward69 + the decided-phase channel (appended last to the pair input)."""
    cfg = model.cfg
    chans = [c for c in cfg["chans"] if c != "dec"]
    b = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
    bi, di, ki, B, D, K = T.T2.pair_index(b["dmask"], b["ncand"])
    T.STATS["pairs_all"] += int(bi.numel())
    if "keep" in b:
        m = b["keep"][bi, di, ki]
        bi, di, ki = bi[m], di[m], ki[m]
    T.STATS["pairs_run"] += int(bi.numel())
    x = T.M.assemble53(b, bi, di, ki, chans)
    flag = dec_flags(batch, K, x.shape[-1], cfg["bw"]).to(dev)[bi, di, ki]
    x = torch.cat([x, flag.to(x.dtype)[:, None, None].expand(-1, 1, x.shape[-1])], dim=1)
    if chunk and x.shape[0] > chunk:
        outs = [model.pair(x[i:i + chunk]) for i in range(0, x.shape[0], chunk)]
        s = torch.cat([o[0] for o in outs]); z = torch.cat([o[1] for o in outs])
    else:
        s, z = model.pair(x)
    logit = torch.full((B, D, K), -1e4, device=dev, dtype=torch.float32)
    logit[bi, di, ki] = s.float()
    Z = torch.zeros((B, D, K, z.shape[-1]), device=dev, dtype=torch.float32)
    Z[bi, di, ki] = z.float()
    p = torch.softmax(logit, dim=2)
    valid = (logit > -1e3)
    zmax = Z.masked_fill(~valid[..., None], -1e4).amax(2)
    parts = [(p[..., None] * Z).sum(2), zmax]
    if cfg["sib"] != "none":
        act = (b["nact"] > 0).float()
        w = p * valid.float() * act[..., None]
        Wk = w.sum(1, keepdim=True)
        ns = (Wk - w).clamp(min=0)
        if cfg["sib"] == "mean":
            S = torch.einsum("bdk,bdke->bke", w, Z)[:, None]
            Sm = (S - w[..., None] * Z) / (ns[..., None] + 1e-3)
        else:
            with torch.autocast("cuda", enabled=False):
                Sm = model.sib(Z, w)
        parts += [(p[..., None] * Sm).sum(2), torch.log1p((p * ns).sum(2, keepdim=True))]
    flog = model.fhead(torch.cat(parts, -1))
    return logit, flog


def main():
    assert T.ADALL is T.F53.ADALL
    T.ADALL = list(T.ADALL) + ["dec"]          # cfg["chans"] -> 16 inputs (Net69 sizes the pair net from it)
    T.forward69 = forward97
    try:
        T.main()
    finally:
        T.log(f"note 97 flags: {STATS97}")


if __name__ == "__main__":
    main()
