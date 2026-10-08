"""Note 53 (function part): a FUNCTION head on the TCN, screened on fold 0 of folds_v4.

Same TCN pair scorer as `tcn53.py` (1-s raster, `ad_all` channels = the shipped nine + ON/OFF edge
impulses, call placed / dropped impulses, partner-phase green / call).  The per-detector function
representation pools the pair embeddings over the candidate phases: [sum_c p(c) z(d,c), max_c z(d,c)],
p = the phase head's softmax -- so the function head reads the detector's behaviour *relative to the
phase it most likely serves* without ever seeing a phase number.  7 classes (frame v6e / labels v3s:
Advance, Presence, Count, Yellow_Red, Mid, Bike, Other), plain cross-entropy.

  --mode func    function loss only (the phase scores are only the pooling weights)
  --mode joint   phase loss + function loss

Labels: frame v6e training target (`first.all.wi`, v3s, the trees' own training rows, `func_rows.parquet`
built by `flab.py`), one label per detector; phase target = official timing (as `tcn53.py`) where known.
Split = folds_v4 (the function folds): fold 0 held out, inner validation 10 % of the training signals.
Inference: the 22 windows of every fold-0 signal (K = 4 pieces past 2 h), mean log-prob over pieces.

    python tcn53_func.py train --tag fj --mode joint
    python tcn53_func.py infer --tag fj      -> %DC_WORK%/tcn53/fpreds/fj_f0.parquet

Note 68: `--arch gru` swaps ONLY the pair backbone for the p3 GRU's (`trackb_models.GRUAttnD`: stride-4
conv stem, 3-layer BiGRU h128, attention pooling; 768-d -> proj 128). Inputs, channels, heads, loss,
sampler, optimiser, early stopping and inference are the fj recipe unchanged ("gj").  The arch is stored
in the checkpoint; infer reads it from there.  The GRU backbone runs in fp32 (autocast off): under bf16
autocast cuDNN's GRU drops to a slow path (~14x slower on the A1000); heads stay in bf16 as for fj.
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402
from common import DC_WORK  # noqa: E402
from neural import data2 as D2  # noqa: E402
from neural import tcn53 as M  # noqa: E402  (patches D2 to the 780-signal pool)
from neural import infer2 as I2  # noqa: E402
from neural import train2 as T2  # noqa: E402
from neural.trackb_models import GRUAttnD  # noqa: E402

ROOT = DC_WORK / "tcn53"
FROWS = ROOT / "func_rows.parquet"
C7 = ["Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other"]
ADALL = M.BASE9 + ["onE", "offE", "cOn", "cOff", "pg", "pc"]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def func_table(train_only: bool) -> tuple[dict, pd.DataFrame]:
    """key -> dets (function-labelled, training rows), y_phase, y_func, cand, period, dev, fold."""
    r = pd.read_parquet(FROWS)
    lk = M.P3N.locked_ids()
    assert not r.DeviceId.isin(lk).any()
    sig = r[["period", "DeviceId", "fold"]].drop_duplicates()
    if train_only:
        r = r[r.ok]
    lab = r.groupby(["period", "DeviceId", "Detector"]).y.first().reset_index()
    if not train_only:                          # inference: every detector of the frame
        lab = r[["period", "DeviceId", "Detector"]].drop_duplicates()
        lab["y"] = None
    ph = D2.official_labels()
    out = {}
    for period, g in lab.groupby("period"):
        cmap = D2._cand_map(period)
        real = D2.PERIODS[period]["real"]
        sigdir = D2.PERIODS[period]["sigdir"]
        p = ph[ph[real]]
        pm = {(d, int(x)): int(q) for d, x, q in zip(p.DeviceId, p.Detector, p.Phase)}
        for dev, gg in g.groupby("DeviceId"):
            cand = cmap.get(dev, np.zeros(0, np.int64))
            if len(cand) < 2 or not (sigdir / f"{dev}.npz").exists():
                continue                        # no raster cache (not in the phase pool)
            pos = {int(c): i for i, c in enumerate(cand)}
            dets = gg.Detector.astype(int).to_numpy()
            yp = np.array([pos.get(pm.get((dev, int(d)), -99), -1) for d in dets], np.int64)
            yf = np.array([C7.index(y) if isinstance(y, str) and y in C7 else -1 for y in gg.y], np.int64)
            out[f"{period}|{dev}"] = dict(dets=dets.astype(np.int32), y_phase=yp, y_func=yf,
                                          cand=np.asarray(cand, np.int64), period=period, dev=dev)
    return out, sig


class DSF(M.DS53):
    def __getitem__(self, i):
        it = super().__getitem__(i)
        key, w0 = it["key"], it["w0"]
        rec = self.table[key]
        n = len(rec["dets"])
        if self.max_det and n > self.max_det:
            idx = np.random.default_rng((self.seed, i, w0)).permutation(n)[:self.max_det]
            idx.sort()
        else:
            idx = np.arange(n)
        it["y_func"] = rec["y_func"][idx]
        return it


def collateF(batch):
    b = M.collate53(batch)
    B, D = b["dmask"].shape
    yf = np.full((B, D), -1, np.int64)
    for j, it in enumerate(batch):
        yf[j, :len(it["dets"])] = it["y_func"]
    b["y_func"] = torch.from_numpy(yf)
    return b


class NetF(nn.Module):
    def __init__(self, cin: int, arch: str = "tcn"):
        super().__init__()
        self.pair = M.Net53(cin)
        self.arch = arch
        if arch == "gru":                       # note 68: p3 GRU backbone, everything else unchanged
            self.pair.backbone = GRUAttnD(cin=cin)
            self.pair.proj = nn.Sequential(nn.Linear(self.pair.backbone.out_dim, 128), nn.GELU())
        self.fhead = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, len(C7)))

    def forward(self, x):
        if self.arch == "gru":                  # note 68: cuDNN GRU under bf16 autocast falls off the fast path
            with torch.autocast("cuda", enabled=False):   # (~14x slower); the GRU backbone runs in fp32
                h = self.pair.backbone(x.float())
        else:
            h = self.pair.backbone(x)
        z = self.pair.proj(h)
        return self.pair.phase(z).squeeze(-1), z


def forwardF(model, batch, dev, chans, chunk=0):
    b = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
    bi, di, ki, B, D, K = T2.pair_index(b["dmask"], b["ncand"])
    x = M.assemble53(b, bi, di, ki, chans)
    if chunk and x.shape[0] > chunk:
        outs = [model(x[i:i + chunk]) for i in range(0, x.shape[0], chunk)]
        s = torch.cat([o[0] for o in outs]); z = torch.cat([o[1] for o in outs])
    else:
        s, z = model(x)
    logit = torch.full((B, D, K), -1e4, device=dev, dtype=torch.float32)
    logit[bi, di, ki] = s.float()
    Z = torch.zeros((B, D, K, z.shape[-1]), device=dev, dtype=torch.float32)
    Z[bi, di, ki] = z.float()
    p = torch.softmax(logit, dim=2)
    valid = (logit > -1e3)
    zmax = Z.masked_fill(~valid[..., None], -1e4).amax(2)
    zd = torch.cat([(p[..., None] * Z).sum(2), zmax], dim=-1)
    flog = model.fhead(zd)
    return logit, flog


def func_loss(flog, batch, dev):
    yf = batch["y_func"].to(dev)
    na = batch["nact"].to(dev)
    ok = (yf >= 0) & (na > 0)
    if not bool(ok.any()):
        return None
    return F.cross_entropy(flog[ok].float(), yf[ok])


@torch.no_grad()
def evaluateF(model, table, plan, dev, chans, workers):
    model.eval()
    dl = DataLoader(DSF(table, plan, 1000), batch_sampler=M.batches53(plan, 240, 12),
                    num_workers=workers, collate_fn=collateF)
    hp = np.zeros(2); hf = np.zeros(2)
    for batch in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            logit, flog = forwardF(model, batch, dev, chans)
        na = batch["nact"].numpy() > 0
        yp, yf = batch["y_phase"].numpy(), batch["y_func"].numpy()
        pp, pf = logit.argmax(2).cpu().numpy(), flog.argmax(2).cpu().numpy()
        m = (yp >= 0) & na; hp += [(pp[m] == yp[m]).sum(), m.sum()]
        m = (yf >= 0) & na; hf += [(pf[m] == yf[m]).sum(), m.sum()]
    return hp[0] / max(hp[1], 1), hf[0] / max(hf[1], 1)


def cmd_train(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    (ROOT / "models").mkdir(parents=True, exist_ok=True)
    (ROOT / "runs").mkdir(parents=True, exist_ok=True)   # note 65 package bug
    tag = f"{a.tag}_f{a.fold}"
    done = ROOT / "runs" / f"{tag}.done.json"
    if done.exists():
        log("done"); return
    table, sig = func_table(True)
    tr = set((sig.period + "|" + sig.DeviceId)[sig.fold != a.fold])
    keys = sorted(k for k in table if k in tr)
    rng = np.random.default_rng(a.seed + 7)
    perm = rng.permutation(len(keys))
    n_es = max(20, int(0.1 * len(keys)))
    es_keys, tr_keys = [keys[i] for i in perm[:n_es]], [keys[i] for i in perm[n_es:]]
    model = NetF(len(ADALL), a.arch).to(dev)
    log(f"{tag}: arch {a.arch}, mode {a.mode}; train {len(tr_keys)} signals, inner-val {len(es_keys)}; "
        f"{sum((table[k]['y_func'] >= 0).sum() for k in tr_keys):,} labelled detectors")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2,
                                                       min_lr=3e-3 / 64)
    ck_best, ck_last = ROOT / "models" / f"{tag}.pt", ROOT / "models" / f"{tag}.last.pt"
    best, best_ep, ep0, hist = -1.0, -1, 0, []
    if ck_last.exists():
        st = torch.load(ck_last, map_location=dev, weights_only=False)
        model.load_state_dict(st["state"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        best, best_ep, ep0, hist = st["best"], st["best_ep"], st["epoch"] + 1, st["hist"]
    esplan = M.es_plan53(table, es_keys)
    stopped = False
    for ep in range(ep0, a.epochs):
        plan = [(k, w0, s, None) for k, w0, s in D2.sample_plan(table, tr_keys, 2, a.seed, ep)]
        dl = DataLoader(DSF(table, plan, 1000, max_det=12, seed=a.seed),
                        batch_sampler=M.batches53(plan, 120, 8, shuffle_seed=ep),
                        num_workers=a.workers, collate_fn=collateF, pin_memory=True)
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for batch in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit, flog = forwardF(model, batch, dev, ADALL)
            lf = func_loss(flog, batch, dev)
            lp = T2.phase_loss(logit, batch, dev) if a.mode == "joint" else None
            parts = [x for x in (lf, lp) if x is not None]
            if not parts:
                continue
            loss = sum(parts)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            tot += float(loss.detach()); nb += 1
        ap_, af = evaluateF(model, table, esplan, dev, ADALL, a.workers)
        score = af if a.mode == "func" else 0.5 * (ap_ + af)
        sched.step(score)
        hist.append(dict(epoch=ep, loss=tot / max(nb, 1), es_phase=ap_, es_func=af, secs=time.time() - t0))
        log(f"ep {ep}: loss {tot/max(nb,1):.4f} es phase {ap_:.4f} func {af:.4f} "
            f"lr {opt.param_groups[0]['lr']:.1e} {time.time()-t0:.0f}s")
        if score > best + 1e-5:
            best, best_ep = score, ep
            torch.save({"state": model.state_dict(), "epoch": ep, "es": score, "mode": a.mode, "arch": a.arch}, ck_best)
        torch.save({"state": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep, "best": best, "best_ep": best_ep, "hist": hist}, ck_last)
        if ep - best_ep >= a.patience:
            stopped = True
            break
    json.dump(dict(tag=tag, mode=a.mode, arch=a.arch, best=best, best_epoch=best_ep, epochs=len(hist),
                   plateaued=stopped, hist=hist), open(done, "w"), indent=1)
    log(f"{tag}: best {best:.4f} @ep {best_ep} of {len(hist)}; plateaued={stopped}")


@torch.no_grad()
def cmd_infer(a):
    from pieces_infer import subset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tag = f"{a.tag}_f{a.fold}"
    out = ROOT / "fpreds" / f"{tag}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        log("exists"); return
    ck = torch.load(ROOT / "models" / f"{tag}.pt", map_location=dev, weights_only=False)
    model = NetF(len(ADALL), ck.get("arch", "tcn")).to(dev).eval()
    model.load_state_dict(ck["state"])
    table, sig = func_table(False)
    f0 = set((sig.period + "|" + sig.DeviceId)[sig.fold == a.fold])
    keys = sorted(k for k in table if k in f0)
    log(f"{tag}: epoch {ck['epoch']}; {len(keys)} held-out signals")
    acc, pacc = {}, {}
    for period in ("dec", "stg"):
        pk = [k for k in keys if table[k]["period"] == period]
        for name, start, secs in I2.WINDOWS[period]:
            pcs = I2.pieces(period, start, secs, M.GRID)
            if secs > M.LONG_S:
                pcs = [pcs[i] for i in subset(len(pcs), M.K_LONG)]
            plan = [(k, p0, (p1 - p0) // 1000, name) for k in pk for p0, p1 in pcs]
            dl = DataLoader(DSF(table, plan, 1000), batch_sampler=M.batches53(plan, 180, 6),
                            num_workers=a.workers, collate_fn=collateF)
            for batch in dl:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logit, flog = forwardF(model, batch, dev, ADALL, chunk=1024)
                lp = F.log_softmax(flog.float(), dim=2).cpu().numpy()
                lq = F.log_softmax(logit.float(), dim=2).cpu().numpy()   # note 68: phase head too
                for b, (key, _d, _w, dets, cand, _n) in enumerate(batch["meta"]):
                    K = len(cand)
                    for j, d in enumerate(dets):
                        e = acc.setdefault((key, name, int(d)), [np.zeros(len(C7)), 0])
                        e[0] += lp[b, j]; e[1] += 1
                        q = pacc.setdefault((key, name, int(d)), [cand, np.zeros(K), 0])
                        q[1] += lq[b, j, :K]; q[2] += 1
        log(f"  {period} done")
    rows = []
    for (key, win, det), (s, n) in acc.items():
        v = s / n
        p = np.exp(v - v.max()); p /= p.sum()
        per, dv = key.split("|", 1)
        rows.append((dv, det, per, win, *p.tolist()))
    df = pd.DataFrame(rows, columns=["DeviceId", "Detector", "period", "win"] + [f"P_{c}" for c in C7])
    df.to_parquet(out, index=False)
    log(f"wrote {out} ({len(df):,} rows)")
    rows = []                                   # phase head, tcn53 `preds` format (DeviceId@stg for Sept)
    for (key, win, det), (cand, s, n) in pacc.items():
        v = s / max(n, 1)
        p = np.exp(v - v.max()); p /= p.sum()
        dv = key.split("|", 1)[1] + ("@stg" if key.startswith("stg|") else "")
        rows += [(dv, int(det), int(c), win, float(pv)) for c, pv in zip(cand, p)]
    pout = ROOT / "ppreds" / f"{tag}.parquet"
    pout.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["DeviceId", "Detector", "cand_phase", "win", "prob"]).to_parquet(pout, index=False)
    log(f"wrote {pout} ({len(rows):,} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "infer"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mode", default="joint", choices=["joint", "func"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--arch", default="tcn", choices=["tcn", "gru"], help="pair backbone (note 68)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0, help="folds_v4 fold held out")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
