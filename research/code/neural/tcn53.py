"""Note 53: the TCN phase network optimised ON ITS OWN (validation plan step 3, phase part).

One configurable pipeline -- the note-15 TCN pair scorer (`trackb_models.TCND`, 7 dilated
blocks, 96 channels, attention pooling, phase loss only, 5-30 min training windows, patience 7)
on the note-37 signal pool (780 signals: 709 stage-13 + 71 released NEWTEST; locked_v2 asserted
absent), with three knobs:

  --bw MS          raster bin: 1000 (shipped) or 500 (0.5 s; the event clock is 0.1 s)
  --chans A,B,..   the per-pair input channels, by name (default = the shipped nine):
        occ     detector occupancy fraction            (shipped ch 0)
        onrate  detector ON onsets per bin (clip 4)/2  (ch 1)
        g y rc  candidate phase green / yellow / red clearance   (ch 2-4)
        call    candidate phase call 43 -> 44 held     (ch 5)
        og      other phases green, count / 2          (ch 6)
        oc      share of the other phases called       (ch 7)
        coord   signal coordinated (131 pattern 1-253) (ch 8)
      new:
        onE offE  detector ON / OFF edge impulses, each event split between the two nearest
                  bin centres by its sub-bin position ("tent"), so 0.1-s timing survives 1-s bins
        cOn cOff  candidate phase call placed (43) / dropped (44) as separate impulses
        pg pc     the PARTNER phase's green / call, partner = the other candidate phase whose
                  green overlaps this one's most inside the window (behaviour only, no number)
  --feats none|all|topk   hybrid: the phase_v3 ranker's per-pair engineered features
                  (`tcn53_prep.py`, standardised) through a small MLP, concatenated to the
                  pooled embedding before the scoring head; needs --windows fixed
  --windows random|fixed  random = the stage-13 sampler; fixed = the 22 evaluation windows
                  of each TRAINING signal (the only windows the tree features exist for),
                  long windows as a random 30-min piece -- `fx0` is the no-feature control

Everything phase-anonymous: candidates are an unordered set, the partner is found from green
overlap, feature rows are looked up by (window, detector, candidate) and never carry a number.

    python tcn53.py train --tag base --fold 0
    python tcn53.py infer --tag base --fold 0    -> %DC_WORK%/tcn53/preds/base_f0_bywindow.parquet

Note 86 knobs (defaults unchanged): `train --full --lr_sched F.json` = one refit on EVERY training signal of the pool (no
fold, no inner-val), fixed epoch count = len(lr list), lr per epoch from the list, final weights -> models/{tag}_full.pt;
`infer --otag NAME` writes preds/{NAME}_f{fold}_bywindow.parquet (same model, new output name).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402
from common import DC_WORK  # noqa: E402
from neural import data2 as D2  # noqa: E402
from neural import phase_v3_net as P3N  # noqa: E402,F401  -- patches D2 to the 780-signal pool
from neural import infer2 as I2  # noqa: E402
from neural.raster import _cover, _onrate  # noqa: E402
from neural.trackb_models import TCND, n_params  # noqa: E402
from neural import train2 as T2  # noqa: E402

ROOT = DC_WORK / "tcn53"
MODELDIR, RUNDIR, PREDDIR = ROOT / "models", ROOT / "runs", ROOT / "preds"
FEATDIR = ROOT / "feats"
BASE9 = ["occ", "onrate", "g", "y", "rc", "call", "og", "oc", "coord"]
ALLCH = BASE9 + ["onE", "offE", "cOn", "cOff", "pg", "pc"]
LONG_S, K_LONG, GRID = 120 * 60, 4, 32


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------ rasterisation
def _tent(t: np.ndarray, w0: int, T: int, bw: int) -> np.ndarray:
    out = np.zeros(T + 1, dtype=np.float64)
    if t.size:
        pos = (t.astype(np.float64) - w0) / bw - 0.5
        i0 = np.floor(pos).astype(np.int64)
        f = pos - i0
        m0 = (i0 >= 0) & (i0 < T)
        np.add.at(out, i0[m0], 1.0 - f[m0])
        m1 = (i0 + 1 >= 0) & (i0 + 1 < T)
        np.add.at(out, i0[m1] + 1, f[m1])
    return np.clip(out[:T], 0.0, 2.0).astype(np.float32)


def _sel(t: np.ndarray, w0: int, w1: int, bw: int) -> np.ndarray:
    i0 = int(np.searchsorted(t, w0 - bw, "left"))
    i1 = int(np.searchsorted(t, w1 + bw, "left"))
    return t[i0:i1]


def pick_partner(ph) -> np.ndarray:
    """Partner of each candidate = the other candidate whose green overlaps it most (-1 = none).  An overlap tie is
    broken by the partner's own green / call traces (content, never candidate order); tied candidates with identical
    traces give identical partner channels, so the choice cannot matter (note 76; same code as the package funcnet)."""
    K = ph.shape[0]
    if K < 2:
        return np.full(K, -1, np.int64)
    G = ph[:, 0].astype(np.float64)
    ov = G @ G.T
    np.fill_diagonal(ov, -1.0)
    out = np.full(K, -1, np.int64)
    for k in range(K):
        mx = ov[k].max()
        if not mx > 0:
            continue
        tied = np.flatnonzero(ov[k] >= mx - 1e-9 * max(mx, 1.0))
        out[k] = tied[0] if len(tied) == 1 else max(
            tied, key=lambda q: (ph[q, 0].tobytes(), ph[q, 3].tobytes()))
    return out


def render53(z: dict, w0: int, dets, T: int, bw: int):
    """-> det [D,4,T] (occ, onrate, onE, offE), ph [K,6,T] (g, y, rc, call, cOn, cOff),
    sig [3,T], nact [D], partner [K] (index into the candidates, -1 = none)."""
    w1 = w0 + T * bw
    K = len(z["cand"])
    ph = np.zeros((K, 6, T), dtype=np.float32)
    for tag, row in (("g", 0), ("y", 1), ("r", 2), ("c", 3)):
        ptr, on, off = z[tag + "_ptr"], z[tag + "_on"], z[tag + "_off"]
        for k in range(K):
            s, e = int(ptr[k]), int(ptr[k + 1])
            if e > s:
                ph[k, row] = _cover(on[s:e], off[s:e], w0, w1, T, bw)
                if tag == "c":
                    ph[k, 4] = _tent(_sel(on[s:e], w0, w1, bw), w0, T, bw)
                    ph[k, 5] = _tent(_sel(off[s:e], w0, w1, bw), w0, T, bw)
    sig = np.zeros((3, T), dtype=np.float32)
    sig[0] = ph[:, 0].sum(0)
    sig[1] = ph[:, 3].sum(0)
    sig[2] = _cover(z["co_on"], z["co_off"], w0, w1, T, bw)
    partner = pick_partner(ph)      # note 76: overlap ties broken by content, never by candidate order

    chpos = z["_chpos"]
    D = len(dets)
    det = np.zeros((D, 4, T), dtype=np.float32)
    nact = np.zeros(D, dtype=np.float32)
    ptr, on, off = z["det_ptr"], z["det_on"], z["det_off"]
    for i, ch in enumerate(dets):
        k = chpos.get(int(ch))
        if k is None:
            continue
        s, e = int(ptr[k]), int(ptr[k + 1])
        if e <= s:
            continue
        det[i, 0] = _cover(on[s:e], off[s:e], w0, w1, T, bw)
        det[i, 1] = _onrate(on[s:e], w0, w1, T, bw)
        nact[i] = det[i, 1].sum()
        det[i, 2] = _tent(_sel(on[s:e], w0, w1, bw), w0, T, bw)
        det[i, 3] = _tent(_sel(off[s:e], w0, w1, bw), w0, T, bw)
    det[:, 1] = np.clip(det[:, 1], 0, 4) / 2.0
    return det, ph, sig, nact, partner


def assemble53(b: dict, bi, di, ki, sel: list[str]) -> torch.Tensor:
    det, ph, sig, ncand = b["det"], b["ph"], b["sig"], b["ncand"]
    d = det[bi, di]
    p = ph[bi, ki]
    s = sig[bi]
    src = {"occ": d[:, 0:1], "onrate": d[:, 1:2], "onE": d[:, 2:3], "offE": d[:, 3:4],
           "g": p[:, 0:1], "y": p[:, 1:2], "rc": p[:, 2:3], "call": p[:, 3:4],
           "cOn": p[:, 4:5], "cOff": p[:, 5:6], "coord": s[:, 2:3]}
    parts = []
    for n in sel:
        if n in src:
            parts.append(src[n])
        elif n == "og":
            parts.append((s[:, 0:1] - p[:, 0:1]) / 2.0)
        elif n == "oc":
            den = (ncand[bi].clamp(min=2) - 1).float().view(-1, 1, 1)
            parts.append((s[:, 1:2] - p[:, 3:4]) / den)
        elif n in ("pg", "pc"):
            pk = b["partner"][bi, ki]
            ok = (pk >= 0).to(p.dtype).view(-1, 1, 1)
            row = 0 if n == "pg" else 3
            parts.append(ph[bi, pk.clamp(min=0), row:row + 1] * ok)
        else:
            raise KeyError(n)
    return torch.cat(parts, dim=1)


# ------------------------------------------------------------------ data
def window_pieces(period: str, name: str, grid: int = 0) -> list[tuple[int, int]]:
    start, secs = next((s, n) for w, s, n in I2.WINDOWS[period] if w == name)
    return I2.pieces(period, start, secs, grid)


def fixed_plan(table, keys, nwin, seed, epoch) -> list[tuple]:
    rng = np.random.default_rng((seed, epoch, 52))
    plan = []
    cache = {}
    for k in keys:
        per = table[k]["period"]
        names = [w for w, _, _ in I2.WINDOWS[per]]
        for _ in range(nwin):
            nm = names[int(rng.integers(len(names)))]
            pcs = cache.get((per, nm))
            if pcs is None:
                pcs = cache[(per, nm)] = window_pieces(per, nm, 0)
            a, b = pcs[int(rng.integers(len(pcs)))]
            plan.append((k, a, (b - a) // 1000, nm))
    rng.shuffle(plan)
    return plan


def es_plan53(table, keys) -> list[tuple]:
    plan = []
    for k in keys:
        per = table[k]["period"]
        for nm in ("m5_a", "m5_c", "m30_a", "m30_c"):
            a, b = window_pieces(per, nm, 0)[0]
            plan.append((k, a, (b - a) // 1000, nm))
    return plan


def batches53(plan, budget_minutes=120, max_bs=8, shuffle_seed=None):
    by = {}
    for i, it in enumerate(plan):
        by.setdefault(it[2], []).append(i)
    out = []
    for secs, idx in by.items():
        bs = int(np.clip(round(budget_minutes / (secs / 60.0)), 1, max_bs))
        for i in range(0, len(idx), bs):
            out.append(idx[i:i + bs])
    if shuffle_seed is not None:
        np.random.default_rng(shuffle_seed).shuffle(out)
    return out


class DS53(Dataset):
    def __init__(self, table, plan, bw: int, max_det: int = 0, seed: int = 0):
        self.table, self.plan, self.bw = table, plan, int(bw)
        self.max_det, self.seed = max_det, seed
        self.stores = None

    def __len__(self):
        return len(self.plan)

    def __getitem__(self, i):
        key, w0, secs, win = self.plan[i]
        rec = self.table[key]
        n = len(rec["dets"])
        if self.max_det and n > self.max_det:
            idx = np.random.default_rng((self.seed, i, w0)).permutation(n)[:self.max_det]
            idx.sort()
        else:
            idx = np.arange(n)
        dets = rec["dets"][idx].tolist()
        if self.stores is None:
            self.stores = D2.Stores()
        z = self.stores.get(rec["period"]).get(rec["dev"])
        T = int(secs) * 1000 // self.bw
        det, ph, sig, nact, partner = render53(z, int(w0), dets, T, self.bw)
        return dict(det=det, ph=ph, sig=sig, nact=nact, partner=partner,
                    y_phase=rec["y_phase"][idx], ncand=len(rec["cand"]), key=key,
                    dev=rec["dev"], w0=int(w0), dets=np.asarray(dets, np.int32),
                    cand=rec["cand"], win=win)


def collate53(batch):
    B = len(batch)
    D = max(len(b["dets"]) for b in batch)
    K = max(b["ncand"] for b in batch)
    T = batch[0]["det"].shape[-1]
    det = np.zeros((B, D, 4, T), np.float32)
    ph = np.zeros((B, K, 6, T), np.float32)
    sig = np.zeros((B, 3, T), np.float32)
    nact = np.zeros((B, D), np.float32)
    yp = np.full((B, D), -1, np.int64)
    part = np.full((B, K), -1, np.int64)
    ncand = np.zeros(B, np.int64)
    dmask = np.zeros((B, D), bool)
    meta = []
    for b, it in enumerate(batch):
        d, k = len(it["dets"]), it["ncand"]
        det[b, :d] = it["det"]; ph[b, :k] = it["ph"]; sig[b] = it["sig"]
        nact[b, :d] = it["nact"]; yp[b, :d] = it["y_phase"]; part[b, :k] = it["partner"]
        ncand[b] = k; dmask[b, :d] = True
        meta.append((it["key"], it["dev"], it["w0"], it["dets"], it["cand"], it["win"]))
    t = torch.from_numpy
    return dict(det=t(det), ph=t(ph), sig=t(sig), nact=t(nact), y_phase=t(yp),
                partner=t(part), ncand=t(ncand), dmask=t(dmask), meta=meta)


# ------------------------------------------------------------------ engineered features
class FeatStore:
    """Standardised ranker features, gathered in the main process (never pickled to workers)."""

    def __init__(self, mode: str, device: str):
        meta = json.load(open(FEATDIR / "meta.json"))
        names = meta["features"] if mode == "all" else meta["topk"]
        cols = [meta["features"].index(n) for n in names]
        X = np.load(FEATDIR / "F.npy", mmap_mode="r")
        X = np.ascontiguousarray(X[:, cols]) if mode != "all" else np.asarray(X)
        self.X = torch.from_numpy(X).to(device)                     # float16 [rows, F]
        self.nf = len(cols)
        rows = pd.read_parquet(FEATDIR / "rows.parquet")
        dev = rows.DeviceId.to_numpy()
        stg = np.char.endswith(dev.astype(str), "@stg")
        plain = np.char.replace(dev.astype(str), "@stg", "")
        rows["key"] = np.where(stg, "stg|", "dec|") + plain
        rows["r"] = np.arange(len(rows))
        self.idx = {}
        for (key, win), g in rows.groupby(["key", "win"], sort=False):
            dets = np.unique(g.Detector.to_numpy())
            mat = np.full((len(dets), 17), -1, np.int64)
            pos = np.searchsorted(dets, g.Detector.to_numpy())
            cp = g.cand_phase.to_numpy().astype(int)
            ok = (cp >= 0) & (cp <= 16)
            mat[pos[ok], cp[ok]] = g.r.to_numpy()[ok]
            self.idx[(key, win)] = (dets, mat)
        log(f"features: {self.nf} columns, {len(rows):,} rows, {len(self.idx):,} windows")

    def gather(self, meta, D: int, K: int, bi, di, ki, device) -> torch.Tensor:
        R = np.full((len(meta), D, K), -1, np.int64)
        for b, (key, _dev, _w0, dets, cand, win) in enumerate(meta):
            e = self.idx.get((key, win))
            if e is None:
                continue
            dl, mat = e
            pos = np.searchsorted(dl, dets)
            pos = np.clip(pos, 0, max(len(dl) - 1, 0))
            hit = (dl[pos] == dets) if len(dl) else np.zeros(len(dets), bool)
            for j in np.nonzero(hit)[0]:
                R[b, j, :len(cand)] = mat[pos[j], cand]
        Rt = torch.from_numpy(R).to(device)[bi, di, ki]
        miss = (Rt < 0)
        f = self.X[Rt.clamp(min=0)].float() * (~miss).float()[:, None]
        return torch.cat([f, miss.float()[:, None]], dim=1)


# ------------------------------------------------------------------ model
class Net53(nn.Module):
    def __init__(self, cin: int, nfeat: int = 0, dropout: float = 0.0):
        super().__init__()
        self.backbone = TCND(cin=cin, dropout=dropout)
        e = self.backbone.out_dim
        self.nfeat = int(nfeat)
        if self.nfeat:
            self.fmlp = nn.Sequential(nn.Linear(self.nfeat + 1, 128), nn.GELU(),
                                      nn.Linear(128, 64), nn.GELU())
            e += 64
        self.proj = nn.Sequential(nn.Linear(e, 128), nn.GELU())
        self.phase = nn.Linear(128, 1)

    def forward(self, x, f=None):
        z = self.backbone(x)
        if self.nfeat:
            z = torch.cat([z, self.fmlp(f).to(z.dtype)], dim=-1)
        return self.phase(self.proj(z)).squeeze(-1)


def forward53(model, batch, dev, cfg, fs: FeatStore | None, chunk: int = 0):
    b = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v)
         for k, v in batch.items()}
    bi, di, ki, B, D, K = T2.pair_index(b["dmask"], b["ncand"])
    x = assemble53(b, bi, di, ki, cfg["chans"])
    f = fs.gather(batch["meta"], D, K, bi.cpu().numpy(), di.cpu().numpy(),
                  ki.cpu().numpy(), dev) if fs is not None else None
    if chunk and x.shape[0] > chunk:
        pl = torch.cat([model(x[i:i + chunk], None if f is None else f[i:i + chunk])
                        for i in range(0, x.shape[0], chunk)])
    else:
        pl = model(x, f)
    logit = torch.full((B, D, K), -1e4, device=dev, dtype=torch.float32)
    logit[bi, di, ki] = pl.float()
    return logit


def build_model(cfg: dict) -> Net53:
    nf = 0
    if cfg["feats"] != "none":
        meta = json.load(open(FEATDIR / "meta.json"))
        nf = len(meta["features"]) if cfg["feats"] == "all" else len(meta["topk"])
    return Net53(len(cfg["chans"]), nf, cfg.get("dropout", 0.0))


@torch.no_grad()
def evaluate53(model, table, plan, dev, cfg, fs, workers) -> dict:
    model.eval()
    ds = DS53(table, plan, cfg["bw"])
    dl = DataLoader(ds, batch_sampler=batches53(plan, 240, 12), num_workers=workers,
                    collate_fn=collate53)
    out = {}
    for batch in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            logit = forward53(model, batch, dev, cfg, fs, chunk=cfg["chunk"])
        pred = logit.argmax(2).cpu().numpy()
        yp, na = batch["y_phase"].numpy(), batch["nact"].numpy()
        T = int(round(batch["det"].shape[-1] * cfg["bw"] / 60000))
        m = (yp >= 0) & (na > 0)
        h, n = out.get(T, (0, 0))
        out[T] = (h + int((pred[m] == yp[m]).sum()), n + int(m.sum()))
    acc = {T: h / max(n, 1) for T, (h, n) in sorted(out.items())}
    acc["score"] = float(np.mean(list(acc.values()))) if acc else 0.0
    return acc


# ------------------------------------------------------------------ train
def lr_schedule_from_runs(done_files, n_epochs: int) -> list[float]:
    """note 86: per-epoch learning rate of fold runs, replayed from their recorded inner-val scores through the same
    ReduceLROnPlateau as cmd_train (lr used DURING epoch e); median over runs per epoch (runs that stopped earlier drop
    out of later epochs) -- as tcn69_func.lr_schedule_from_runs."""
    lrs = []
    for f in done_files:
        h = json.load(open(f))["hist"]
        p = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.SGD([p], lr=3e-3)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2, min_lr=3e-3 / 64)
        r = []
        for e in h:
            r.append(opt.param_groups[0]["lr"])
            sch.step(e["es"])
        lrs.append(r)
    return [float(np.median([r[e] for r in lrs if e < len(r)])) for e in range(n_epochs)]


def cmd_train_full(a, cfg) -> None:
    """note 86: one refit on ALL training signals of the pool (locked asserted absent), no inner-val, fixed epoch count
    and learning-rate schedule (--lr_sched), final weights kept (models/{tag}_full.pt)."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tag = f"{a.tag}_full"
    done = RUNDIR / f"{tag}.done.json"
    if done.exists():
        log(f"{tag}: done -- skipping"); return
    lrs = json.load(open(a.lr_sched))
    cfg = dict(cfg, fold="full", epochs=len(lrs), lr_sched=lrs)
    sigs = D2.training_signals()
    assert not sigs.DeviceId.isin(P3N.locked_ids()).any()
    table = D2.load_table(sigs)
    tr_keys = [k for k in sigs.key if k in table]
    assert not any(k.split("|", 1)[1] in P3N.locked_ids() for k in tr_keys)
    model = build_model(cfg).to(dev)
    log(f"{tag}: {json.dumps({k: v for k, v in cfg.items() if k != 'lr_sched'})}; params {n_params(model):,}; "
        f"train {len(tr_keys)} signals (all); {len(lrs)} epochs; lr {lrs[0]:.1e} -> {lrs[-1]:.1e}")
    opt = torch.optim.AdamW(model.parameters(), lr=lrs[0], weight_decay=1e-4)
    ck_fin, ck_last = MODELDIR / f"{tag}.pt", MODELDIR / f"{tag}.last.pt"
    ep0, hist = 0, []
    if ck_last.exists():
        st = torch.load(ck_last, map_location=dev, weights_only=False)
        model.load_state_dict(st["state"]); opt.load_state_dict(st["opt"])
        ep0, hist = st["epoch"] + 1, st["hist"]
        log(f"resumed at epoch {ep0}")
    for ep in range(ep0, len(lrs)):
        for g in opt.param_groups:
            g["lr"] = lrs[ep]
        plan = [(k, w0, s, None) for k, w0, s in D2.sample_plan(table, tr_keys, 2, a.seed, ep)]
        ds = DS53(table, plan, a.bw, max_det=12, seed=a.seed)
        dl = DataLoader(ds, batch_sampler=batches53(plan, 120, 8, shuffle_seed=ep), num_workers=a.workers,
                        collate_fn=collate53, pin_memory=(dev == "cuda"))
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for batch in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                logit = forward53(model, batch, dev, cfg, None, chunk=a.chunk)
            loss = T2.phase_loss(logit, batch, dev)
            if loss is None:
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            tot += float(loss.detach()); nb += 1
        hist.append(dict(epoch=ep, loss=tot / max(nb, 1), lr=lrs[ep], secs=time.time() - t0))
        log(f"ep {ep}: loss {tot / max(nb, 1):.4f} lr {lrs[ep]:.1e} {time.time() - t0:.0f}s ({nb} steps)")
        torch.save({"state": model.state_dict(), "opt": opt.state_dict(), "epoch": ep, "hist": hist}, ck_last)
    torch.save({"cfg": cfg, "state": model.state_dict(), "epoch": len(lrs) - 1, "es": float("nan")}, ck_fin)
    json.dump(dict(tag=tag, cfg=cfg, params=n_params(model), epochs_run=len(hist), n_signals=len(tr_keys),
                   train_min=sum(h["secs"] for h in hist) / 60, hist=hist), open(done, "w"), indent=1, default=str)
    log(f"{tag}: wrote {ck_fin} (final weights after epoch {len(lrs) - 1})")


def cmd_train(a) -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    if a.full:
        for d in (MODELDIR, RUNDIR):
            d.mkdir(parents=True, exist_ok=True)
        assert a.feats == "none" and a.windows == "random" and a.lr_sched, "--full: plain recipe + --lr_sched"
        cfg = dict(bw=a.bw, chans=a.chans.split(","), feats=a.feats, windows=a.windows,
                   dropout=a.dropout, chunk=a.chunk, seed=a.seed, tag=a.tag)
        for c in cfg["chans"]:
            assert c in ALLCH, c
        return cmd_train_full(a, cfg)
    for d in (MODELDIR, RUNDIR):
        d.mkdir(parents=True, exist_ok=True)
    tag = f"{a.tag}_f{a.fold}" if a.fold >= 0 else f"{a.tag}_final"
    done = RUNDIR / f"{tag}.done.json"
    if done.exists():
        log(f"{tag}: done -- skipping"); return
    cfg = dict(bw=a.bw, chans=a.chans.split(","), feats=a.feats, windows=a.windows,
               dropout=a.dropout, chunk=a.chunk, seed=a.seed, fold=a.fold, tag=a.tag)
    for c in cfg["chans"]:
        assert c in ALLCH, c
    assert cfg["feats"] == "none" or cfg["windows"] == "fixed"
    sigs = D2.training_signals()
    lk = P3N.locked_ids()
    assert not sigs.DeviceId.isin(lk).any()
    tr_sigs = sigs if a.fold < 0 else sigs[sigs.fold != a.fold]
    table = D2.load_table(sigs)
    keys = [k for k in tr_sigs.key if k in table]
    rng = np.random.default_rng(a.seed + 7)
    perm = rng.permutation(len(keys))
    n_es = max(20, int(0.10 * len(keys)))
    es_keys = [keys[i] for i in perm[:n_es]]
    tr_keys = [keys[i] for i in perm[n_es:]]
    fs = FeatStore(a.feats, dev) if a.feats != "none" else None
    model = build_model(cfg).to(dev)
    log(f"{tag}: {json.dumps(cfg)}; params {n_params(model):,}; train {len(tr_keys)} "
        f"signals, inner-val {len(es_keys)}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                       patience=2, min_lr=a.lr / 64)
    ck_best, ck_last = MODELDIR / f"{tag}.pt", MODELDIR / f"{tag}.last.pt"
    best, best_ep, ep0, hist = -1.0, -1, 0, []
    if ck_last.exists():
        st = torch.load(ck_last, map_location=dev, weights_only=False)
        model.load_state_dict(st["state"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        best, best_ep, ep0, hist = st["best"], st["best_ep"], st["epoch"] + 1, st["hist"]
        log(f"resumed at epoch {ep0} (best {best:.4f} @ {best_ep})")
    esplan = es_plan53(table, es_keys)
    stopped = False
    for ep in range(ep0, a.epochs):
        if a.windows == "fixed":
            plan = fixed_plan(table, tr_keys, 2, a.seed, ep)
        else:
            plan = [(k, w0, s, None) for k, w0, s in
                    D2.sample_plan(table, tr_keys, 2, a.seed, ep)]
        ds = DS53(table, plan, a.bw, max_det=12, seed=a.seed)
        dl = DataLoader(ds, batch_sampler=batches53(plan, 120, 8, shuffle_seed=ep),
                        num_workers=a.workers, collate_fn=collate53,
                        pin_memory=(dev == "cuda"))
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for batch in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                logit = forward53(model, batch, dev, cfg, fs, chunk=a.chunk)
            loss = T2.phase_loss(logit, batch, dev)
            if loss is None:
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            tot += float(loss.detach()); nb += 1
        secs = time.time() - t0
        acc = evaluate53(model, table, esplan, dev, cfg, fs, a.workers)
        sched.step(acc["score"])
        hist.append(dict(epoch=ep, loss=tot / max(nb, 1), secs=secs,
                         lr=opt.param_groups[0]["lr"], es=acc["score"],
                         **{f"m{k}": v for k, v in acc.items() if isinstance(k, int)}))
        log(f"ep {ep}: loss {tot/max(nb,1):.4f} es {acc['score']:.4f} "
            + " ".join(f"m{k}={v:.4f}" for k, v in acc.items() if isinstance(k, int))
            + f" lr {opt.param_groups[0]['lr']:.1e} {secs:.0f}s ({nb} steps)")
        if acc["score"] > best + 1e-5:
            best, best_ep = acc["score"], ep
            torch.save({"cfg": cfg, "state": model.state_dict(), "epoch": ep,
                        "es": best}, ck_best)
        torch.save({"state": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep, "best": best,
                    "best_ep": best_ep, "hist": hist}, ck_last)
        if ep - best_ep >= a.patience:
            stopped = True
            break
    info = dict(tag=tag, cfg=cfg, params=n_params(model), best_es=best, best_epoch=best_ep,
                epochs_run=len(hist), plateaued=stopped,
                train_min=sum(h["secs"] for h in hist) / 60, hist=hist)
    json.dump(info, open(done, "w"), indent=1, default=str)
    log(f"{tag}: best es {best:.4f} @ep {best_ep} of {len(hist)}; plateaued={stopped}; "
        f"{info['train_min']:.0f} min")


# ------------------------------------------------------------------ infer
@torch.no_grad()
def cmd_infer(a) -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    PREDDIR.mkdir(parents=True, exist_ok=True)
    tag = f"{a.tag}_f{a.fold}"
    dest = PREDDIR / f"{(a.otag or a.tag)}_f{a.fold}_bywindow.parquet"
    if dest.exists():
        log(f"{dest.name} exists -- skipping"); return
    ck = torch.load(MODELDIR / f"{tag}.pt", map_location=dev, weights_only=False)
    cfg = ck["cfg"]
    cfg["chunk"] = a.chunk
    model = build_model(cfg).to(dev).eval()
    model.load_state_dict(ck["state"])
    fs = FeatStore(cfg["feats"], dev) if cfg["feats"] != "none" else None
    sigs = D2.training_signals()
    table = D2.load_table(sigs, labelled_only=False)
    keys = [k for k in sigs.loc[sigs.fold == a.fold, "key"] if k in table]
    assert not any(k.split("|", 1)[1] in P3N.locked_ids() for k in keys)
    log(f"{tag}: epoch {ck['epoch']} es {ck['es']:.4f}; {len(keys)} signals")
    from pieces_infer import subset
    acc: dict = {}
    t0 = time.time()
    for period in ("dec", "stg"):
        pk = [k for k in keys if table[k]["period"] == period]
        if not pk:
            continue
        for name, start, secs in I2.WINDOWS[period]:
            pcs = I2.pieces(period, start, secs, GRID)
            if secs > LONG_S:
                pcs = [pcs[i] for i in subset(len(pcs), K_LONG)]
            plan = [(k, p0, (p1 - p0) // 1000, name) for k in pk for p0, p1 in pcs]
            ds = DS53(table, plan, cfg["bw"])
            dl = DataLoader(ds, batch_sampler=batches53(plan, 180, 6),
                            num_workers=a.workers, collate_fn=collate53)
            for batch in dl:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                    logit = forward53(model, batch, dev, cfg, fs, chunk=a.chunk)
                lp = F.log_softmax(logit, dim=2).float().cpu().numpy()
                na = batch["nact"].numpy()
                for b, (key, _d, _w, dets, cand, _win) in enumerate(batch["meta"]):
                    K = len(cand)
                    for j, d in enumerate(dets):
                        e = acc.get((key, name, int(d)))
                        if e is None:
                            e = acc[(key, name, int(d))] = [cand, np.zeros(K), 0, 0.0]
                        e[1] += lp[b, j, :K]; e[2] += 1; e[3] += float(na[b, j])
        log(f"  {period} done ({time.time()-t0:.0f}s)")
    rows = []
    for (key, win, det), (cand, s, n, na) in acc.items():
        v = s / max(n, 1)
        p = np.exp(v - v.max()); p /= p.sum()
        dv = key.split("|", 1)[1] + ("@stg" if key.startswith("stg|") else "")
        rows += [(dv, int(det), int(c), win, float(pv), float(na)) for c, pv in zip(cand, p)]
    df = pd.DataFrame(rows, columns=["DeviceId", "Detector", "cand_phase", "win", "prob",
                                     "n_act"])
    df.to_parquet(dest, index=False)
    log(f"wrote {dest} ({len(df):,} rows) in {time.time()-t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "infer"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--bw", type=int, default=1000)
    ap.add_argument("--chans", default=",".join(BASE9))
    ap.add_argument("--feats", default="none", choices=["none", "all", "topk"])
    ap.add_argument("--windows", default="random", choices=["random", "fixed"])
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--full", action="store_true", help="note 86: refit on every training signal (train)")
    ap.add_argument("--lr_sched", default="", help="--full: json list of the learning rate per epoch")
    ap.add_argument("--otag", default="", help="note 86: output name for infer (default --tag)")
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
