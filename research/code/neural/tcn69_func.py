"""Note 69: tuning the function network ("fj" recipe of `tcn53_func.py`) -- one configurable script, defaults = fj.

Knobs (all phase-anonymous; the defaults reproduce fj: 1 s, 15 ad_all channels, TCN 96 x 7, no sibling context,
plain cross-entropy, no augmentation, 5-30 min training windows, <= 12 detectors per training window, K = 4 pieces):

  --bw MS        raster bin (1000 / 500 / 200 / 100); events are on the 0.1-s clock
  --fstem F      learned pre-stem Conv1d(cin -> 32, k 2F+1, stride F) so the TCN body runs at bw*F (e.g. 100 ms x 10 =
                 the 1-s body, same receptive field and memory) while the first layer sees 0.1-s pulses / lags
  --sib none|mean|attn   sibling context for the function head: for detector d and candidate phase c, a summary of the
                 OTHER detectors' pair embeddings on the same candidate, weighted by their own p(c) (mean) or by
                 attention with a log p(c) bias (attn, 4 heads, null key); pooled over c with p(d, c) like the rest
  --cw W         class weight W on the four ATSPM classes (Mid / Bike / Other keep weight 1 = full loss)
  --classes A,B  note 92b: function class list (default the 7 classes); must start with the 7, extra = Other subclasses
                 (labels in --frows y); stored in cfg["classes"] and restored by infer
  --ls E         label smoothing on the function loss
  --dropout P    TCN block dropout;  --cdrop P  drop each context (non-detector) input channel per pair in training
  --width / --nblk   TCN width / dilated blocks (receptive field doubles per block)
  --arch gru     note 74: p3 GRU pair backbone (GRUAttnD) instead of the TCN, run in fp32; everything else unchanged
  --minutes      training window lengths (cycled), e.g. 5,10,20,30,60,60,120
  --max_det      detectors per training window (siblings need more than 12)
  --klong        pieces averaged past 2 h at inference (infer only; --otag names the output)

    python tcn69_func.py train --tag x69_r01s --bw 100 --fstem 10 --fold 0
    python tcn69_func.py infer --tag x69_r01s --fold 0      -> %DC_WORK%/tcn53/fpreds/x69_r01s_f0.parquet
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: F401,E402
from neural import data2 as D2  # noqa: E402
from neural import tcn53 as M  # noqa: E402
from neural import tcn53_func as F53  # noqa: E402
from neural import infer2 as I2  # noqa: E402
from neural import train2 as T2  # noqa: E402
from neural.trackb_models import GRUAttnD, TCND  # noqa: E402

ROOT = F53.ROOT
C7, ADALL = F53.C7, F53.ADALL
DETCH = {"occ", "onrate", "onE", "offE"}
log = F53.log
STATS = {"pairs_all": 0, "pairs_run": 0}


def keep_table(path, thr):
    """note 74 candidate filter: {(period|dev, win, det): set of candidate phases with tree probability >= thr} from a
    pair table (DeviceId[@stg], Detector, win, cand_phase, p0_bag) -- e.g. %DC_WORK%/cand64/phase_oof.parquet (OOF trees).
    Detector-windows absent from the table keep every candidate."""
    q = pd.read_parquet(path, columns=["DeviceId", "Detector", "win", "cand_phase", "p0_bag"])
    q = q[q.p0_bag >= thr]
    stg = q.DeviceId.str.endswith("@stg")
    key = np.where(stg, "stg|" + q.DeviceId.str.replace("@stg", "", regex=False), "dec|" + q.DeviceId)
    out = {}
    for k, w, d, c in zip(key, q.win.to_numpy(), q.Detector.to_numpy(int), q.cand_phase.to_numpy(int)):
        out.setdefault((k, w, int(d)), set()).add(int(c))
    return out


def keep_mask(batch, kt):
    B, D = batch["dmask"].shape
    K = int(batch["ncand"].max())
    m = torch.ones((B, D, K), dtype=torch.bool)
    n_hit = 0
    for b, (key, _d, _w, dets, cand, win) in enumerate(batch["meta"]):
        for j, d in enumerate(dets):
            s = kt.get((key, win, int(d)))
            if s is None:
                continue
            n_hit += 1
            m[b, j, :len(cand)] = torch.as_tensor([int(c) in s for c in cand])
    return m, n_hit


class Pair69(nn.Module):
    def __init__(self, cin, fstem=1, width=96, nblk=7, dropout=0.0, arch="tcn"):
        super().__init__()
        if fstem > 1:
            self.fine = nn.Sequential(nn.Conv1d(cin, 32, 2 * fstem + 1, stride=fstem, padding=fstem),
                                      nn.BatchNorm1d(32), nn.GELU())
            c0 = 32
        else:
            self.fine, c0 = nn.Identity(), cin
        self.arch = arch
        if arch == "gru":       # note 74: p3 GRU pair backbone (as tcn53_func --arch gru, note 68); rest unchanged
            self.backbone = GRUAttnD(cin=c0)
        else:
            self.backbone = TCND(cin=c0, c=width, nblk=nblk, dropout=dropout)
        self.proj = nn.Sequential(nn.Linear(self.backbone.out_dim, 128), nn.GELU())
        self.phase = nn.Linear(128, 1)

    def forward(self, x):
        if self.arch == "gru":  # cuDNN GRU under bf16 autocast takes a ~14x slower path (note 68): backbone in fp32
            with torch.autocast("cuda", enabled=False):
                h = self.backbone(self.fine(x).float())
        else:
            h = self.backbone(self.fine(x))
        z = self.proj(h)
        return self.phase(z).squeeze(-1), z


class SibAttn(nn.Module):
    def __init__(self, e=128, h=4):
        super().__init__()
        self.h, self.dh = h, e // h
        self.q, self.k, self.v = nn.Linear(e, e), nn.Linear(e, e), nn.Linear(e, e)
        self.nk = nn.Parameter(torch.zeros(h, self.dh))      # null key (value 0): nothing to attend to
        self.o = nn.Linear(e, e)

    def forward(self, Z, w):
        """Z [B,D,K,E], w [B,D,K] sibling weights (0 = absent) -> [B,D,K,E] summary of the OTHER detectors per (d,c)."""
        B, D, K, E = Z.shape
        X = Z.permute(0, 2, 1, 3).reshape(B * K, D, E)
        W = w.permute(0, 2, 1).reshape(B * K, D)
        q = self.q(X).view(B * K, D, self.h, self.dh).transpose(1, 2)
        k = self.k(X).view(B * K, D, self.h, self.dh).transpose(1, 2)
        v = self.v(X).view(B * K, D, self.h, self.dh).transpose(1, 2)
        s = q @ k.transpose(-1, -2) / math.sqrt(self.dh)                    # [BK,h,D,D]
        bias = torch.log(W.clamp(min=1e-6))[:, None, None, :].expand_as(s).clone()
        bias = bias.masked_fill((W <= 0)[:, None, None, :], -1e4)
        eye = torch.eye(D, dtype=torch.bool, device=Z.device)
        bias = bias.masked_fill(eye[None, None], -1e4)
        sn = (q * self.nk[None, :, None, :]).sum(-1, keepdim=True) / math.sqrt(self.dh)   # [BK,h,D,1]
        a = torch.softmax(torch.cat([s + bias, sn], -1), -1)[..., :D]
        o = (a @ v).transpose(1, 2).reshape(B * K, D, E)
        return self.o(o).view(B, K, D, E).permute(0, 2, 1, 3)


class Net69(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.pair = Pair69(len(cfg["chans"]), cfg["fstem"], cfg["width"], cfg["nblk"], cfg["dropout"],
                           cfg.get("arch", "tcn"))
        nin = 256
        if cfg["sib"] == "attn":
            self.sib = SibAttn()
        if cfg["sib"] != "none":
            nin += 128 + 1
        self.fhead = nn.Sequential(nn.Linear(nin, 128), nn.GELU(), nn.Linear(128, len(C7)))


def forward69(model, batch, dev, train=False, chunk=0):
    cfg = model.cfg
    b = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
    bi, di, ki, B, D, K = T2.pair_index(b["dmask"], b["ncand"])
    STATS["pairs_all"] += int(bi.numel())
    if "keep" in b:          # note 74 candidate filter: the pair net never sees a dropped (detector, candidate) pair
        m = b["keep"][bi, di, ki]
        bi, di, ki = bi[m], di[m], ki[m]
    STATS["pairs_run"] += int(bi.numel())
    x = M.assemble53(b, bi, di, ki, cfg["chans"])
    if train and cfg["cdrop"] > 0:
        ctx = [i for i, c in enumerate(cfg["chans"]) if c not in DETCH]
        keep = (torch.rand(x.shape[0], len(ctx), device=dev) >= cfg["cdrop"]).to(x.dtype)
        x[:, ctx] = x[:, ctx] * keep[..., None]
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
        w = p * valid.float() * act[..., None]                        # [B,D,K]
        Wk = w.sum(1, keepdim=True)                                   # [B,1,K]
        ns = (Wk - w).clamp(min=0)                                    # expected siblings on c, excluding d
        if cfg["sib"] == "mean":
            S = torch.einsum("bdk,bdke->bke", w, Z)[:, None]           # [B,1,K,E]
            Sm = (S - w[..., None] * Z) / (ns[..., None] + 1e-3)
        else:
            with torch.autocast("cuda", enabled=False):
                Sm = model.sib(Z, w)
        parts += [(p[..., None] * Sm).sum(2), torch.log1p((p * ns).sum(2, keepdim=True))]
    flog = model.fhead(torch.cat(parts, -1))
    return logit, flog


def func_loss69(flog, batch, dev, cfg):
    yf = batch["y_func"].to(dev)
    ok = (yf >= 0) & (batch["nact"].to(dev) > 0)
    if not bool(ok.any()):
        return None
    w = None
    if cfg["cw"] != 1.0:
        w = torch.tensor([cfg["cw"]] * 4 + [1.0] * (len(C7) - 4), device=dev)
    return F.cross_entropy(flog[ok].float(), yf[ok], weight=w, label_smoothing=cfg["ls"])


def train_step_accum(model, batch, dev, cfg, n, opt):
    """note 74, --accum N: the batch's samples (windows) split into N micro-batches, each loss weighted by its share of
    the batch's labelled rows, gradients summed -> the gradient of the full-batch loss (sibling attention never crosses
    a sample). Only BatchNorm batch statistics differ (per micro-batch). Returns the full-batch loss value, or None."""
    B = batch["dmask"].shape[0]
    na = batch["nact"] > 0
    nf_tot = int(((batch["y_func"] >= 0) & na).sum())
    np_tot = int(((batch["y_phase"] >= 0) & na).sum())
    if nf_tot == 0 and np_tot == 0:
        return None
    opt.zero_grad(set_to_none=True)
    tot = 0.0
    for idx in np.array_split(np.arange(B), min(n, B)):
        ii = [int(i) for i in idx]
        ti = torch.as_tensor(ii)
        mb = {k: (v[ti] if torch.is_tensor(v) else [v[i] for i in ii]) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            logit, flog = forward69(model, mb, dev, train=True)
        mna = mb["nact"] > 0
        nf = int(((mb["y_func"] >= 0) & mna).sum())
        npc = int(((mb["y_phase"] >= 0) & mna).sum())
        lf = func_loss69(flog, mb, dev, cfg) if nf else None
        lp = T2.phase_loss(logit, mb, dev) if npc else None
        parts = ([lf * (nf / nf_tot)] if lf is not None else []) + ([lp * (npc / np_tot)] if lp is not None else [])
        if parts:
            loss = sum(parts)
            loss.backward()
            tot += float(loss.detach())
    return tot


@torch.no_grad()
def evaluate69(model, table, plan, dev, workers, echunk=2048):
    model.eval()
    dl = DataLoader(F53.DSF(table, plan, model.cfg["bw"]), batch_sampler=M.batches53(plan, 240, 12),
                    num_workers=workers, collate_fn=F53.collateF)
    hp = np.zeros(2); hf = np.zeros(2)
    for batch in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            logit, flog = forward69(model, batch, dev, chunk=echunk)
        na = batch["nact"].numpy() > 0
        yp, yf = batch["y_phase"].numpy(), batch["y_func"].numpy()
        pp, pf = logit.argmax(2).cpu().numpy(), flog.argmax(2).cpu().numpy()
        m = (yp >= 0) & na; hp += [(pp[m] == yp[m]).sum(), m.sum()]
        m = (yf >= 0) & na; hf += [(pf[m] == yf[m]).sum(), m.sum()]
    return hp[0] / max(hp[1], 1), hf[0] / max(hf[1], 1)


def lr_schedule_from_runs(done_files, n_epochs):
    """note 74: per-epoch learning rate of fold runs, replayed from their recorded inner-val scores through the same
    ReduceLROnPlateau as cmd_train (lr used DURING epoch e); median over runs per epoch (runs that stopped earlier drop
    out of later epochs)."""
    lrs = []
    for f in done_files:
        h = json.load(open(f))["hist"]
        p = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.SGD([p], lr=3e-3)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2, min_lr=3e-3 / 64)
        r = []
        for e in h:
            r.append(opt.param_groups[0]["lr"])
            sch.step(0.5 * (e["es_phase"] + e["es_func"]))
        lrs.append(r)
    return [float(np.median([r[e] for r in lrs if e < len(r)])) for e in range(n_epochs)]


def cmd_train_full(a, cfg):
    """note 74: one refit on ALL training signals of the function table (locked signals are not in it; asserted by
    func_table), no inner-val, fixed epoch count and learning-rate schedule (from --lr_sched), final weights kept."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tag = f"{a.tag}_full"
    done = ROOT / "runs" / f"{tag}.done.json"
    if done.exists():
        log("done"); return
    lrs = json.load(open(a.lr_sched))
    cfg = dict(cfg, fold="full", epochs=len(lrs), lr_sched=lrs)
    table, sig = F53.func_table(True)
    lk = M.P3N.locked_ids()
    assert not any(k.split("|", 1)[1] in lk for k in table)
    tr_keys = sorted(table)
    model = Net69(cfg).to(dev)
    log(f"{tag}: {json.dumps({k: v for k, v in cfg.items() if k != 'lr_sched'})}; train {len(tr_keys)} signals (all); "
        f"{len(lrs)} epochs; lr {lrs[0]:.1e} -> {lrs[-1]:.1e}")
    opt = torch.optim.AdamW(model.parameters(), lr=lrs[0], weight_decay=1e-4)
    ck_fin, ck_last = ROOT / "models" / f"{tag}.pt", ROOT / "models" / f"{tag}.last.pt"
    ep0, hist = 0, []
    if ck_last.exists():
        st = torch.load(ck_last, map_location=dev, weights_only=False)
        model.load_state_dict(st["state"]); opt.load_state_dict(st["opt"])
        ep0, hist = st["epoch"] + 1, st["hist"]
    for ep in range(ep0, len(lrs)):
        for g in opt.param_groups:
            g["lr"] = lrs[ep]
        plan = [(k, w0, s, None) for k, w0, s in D2.sample_plan(table, tr_keys, 2, a.seed, ep,
                                                                 minutes=tuple(cfg["minutes"]))]
        dl = DataLoader(F53.DSF(table, plan, a.bw, max_det=a.max_det, seed=a.seed),
                        batch_sampler=M.batches53(plan, 120, 8, shuffle_seed=ep),
                        num_workers=a.workers, collate_fn=F53.collateF, pin_memory=True, persistent_workers=False)
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for batch in dl:
            loss = train_step_accum(model, batch, dev, cfg, max(a.accum, 1), opt)
            if loss is None:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            tot += loss; nb += 1
            if a.smoke and nb >= a.smoke:
                break
        hist.append(dict(epoch=ep, loss=tot / max(nb, 1), lr=lrs[ep], secs=time.time() - t0))
        log(f"ep {ep}: loss {tot/max(nb,1):.4f} lr {lrs[ep]:.1e} {time.time()-t0:.0f}s ({nb} steps)")
        torch.save({"state": model.state_dict(), "opt": opt.state_dict(), "epoch": ep, "hist": hist}, ck_last)
    torch.save({"state": model.state_dict(), "epoch": len(lrs) - 1, "es": None, "cfg": cfg}, ck_fin)
    json.dump(dict(tag=tag, cfg=cfg, epochs=len(hist), train_min=sum(h["secs"] for h in hist) / 60, hist=hist,
                   n_signals=len(tr_keys)), open(done, "w"), indent=1)
    log(f"{tag}: wrote {ck_fin} (final weights after epoch {len(lrs) - 1})")


def cmd_train(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    for d in ("models", "runs"):
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    tag = f"{a.tag}_f{a.fold}"
    done = ROOT / "runs" / f"{tag}.done.json"
    if done.exists():
        log("done"); return
    cfg = dict(bw=a.bw, fstem=a.fstem, sib=a.sib, cw=a.cw, ls=a.ls, dropout=a.dropout, cdrop=a.cdrop,
               width=a.width, nblk=a.nblk, minutes=[int(x) for x in a.minutes.split(",")], max_det=a.max_det,
               chans=ADALL, seed=a.seed, fold=a.fold, mode="joint")
    if a.arch != "tcn":
        cfg["arch"] = a.arch                     # absent = tcn, so older checkpoints load unchanged
    if len(C7) != 7:
        cfg["classes"] = list(C7)                # note 92b: absent = the 7 classes
    if a.full:
        return cmd_train_full(a, cfg)
    table, sig = F53.func_table(True)
    tr = set((sig.period + "|" + sig.DeviceId)[sig.fold != a.fold])
    keys = sorted(k for k in table if k in tr)
    rng = np.random.default_rng(a.seed + 7)
    perm = rng.permutation(len(keys))
    n_es = max(20, int(0.1 * len(keys)))
    es_keys, tr_keys = [keys[i] for i in perm[:n_es]], [keys[i] for i in perm[n_es:]]
    model = Net69(cfg).to(dev)
    log(f"{tag}: {json.dumps(cfg)}; params {sum(p.numel() for p in model.parameters()):,}; "
        f"train {len(tr_keys)} signals, inner-val {len(es_keys)}")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2, min_lr=3e-3 / 64)
    ck_best, ck_last = ROOT / "models" / f"{tag}.pt", ROOT / "models" / f"{tag}.last.pt"
    best, best_ep, ep0, hist = -1.0, -1, 0, []
    if ck_last.exists():
        st = torch.load(ck_last, map_location=dev, weights_only=False)
        model.load_state_dict(st["state"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        best, best_ep, ep0, hist = st["best"], st["best_ep"], st["epoch"] + 1, st["hist"]
    esplan = M.es_plan53(table, es_keys)
    stopped = False
    for ep in range(ep0, a.epochs):
        plan = [(k, w0, s, None) for k, w0, s in D2.sample_plan(table, tr_keys, 2, a.seed, ep,
                                                                 minutes=tuple(cfg["minutes"]))]
        dl = DataLoader(F53.DSF(table, plan, a.bw, max_det=a.max_det, seed=a.seed),
                        batch_sampler=M.batches53(plan, 120, 8, shuffle_seed=ep),
                        num_workers=a.workers, collate_fn=F53.collateF, pin_memory=True, persistent_workers=False)
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for batch in dl:
            if a.accum > 1:     # note 74: same batch, split by sample; gradient of the same full-batch loss
                loss = train_step_accum(model, batch, dev, cfg, a.accum, opt)
                if loss is None:
                    continue
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
                tot += loss; nb += 1
                if a.smoke and nb >= a.smoke:
                    break
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                logit, flog = forward69(model, batch, dev, train=True)
            lf = func_loss69(flog, batch, dev, cfg)
            lp = T2.phase_loss(logit, batch, dev)
            parts = [x for x in (lf, lp) if x is not None]
            if not parts:
                continue
            loss = sum(parts)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            tot += float(loss.detach()); nb += 1
            if a.smoke and nb >= a.smoke:
                break
        ap_, af = evaluate69(model, table, esplan[:40] if a.smoke else esplan, dev, a.workers, a.echunk)
        score = 0.5 * (ap_ + af)
        sched.step(score)
        hist.append(dict(epoch=ep, loss=tot / max(nb, 1), es_phase=ap_, es_func=af, secs=time.time() - t0))
        log(f"ep {ep}: loss {tot/max(nb,1):.4f} es phase {ap_:.4f} func {af:.4f} "
            f"lr {opt.param_groups[0]['lr']:.1e} {time.time()-t0:.0f}s ({nb} steps)")
        if score > best + 1e-5:
            best, best_ep = score, ep
            torch.save({"state": model.state_dict(), "epoch": ep, "es": score, "cfg": cfg}, ck_best)
        torch.save({"state": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep, "best": best, "best_ep": best_ep, "hist": hist}, ck_last)
        if a.swa:               # note 95: keep every epoch's weights for stochastic weight averaging at the end
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                       ROOT / "models" / f"{tag}.ep{ep}.pt")
        if ep - best_ep >= a.patience:
            stopped = True
            break
    if a.swa:
        # note 95 SWA: average the weights of the last 25 % of the epochs run (BatchNorm running statistics averaged
        # too -- an approximation of torch's update_bn pass); the average replaces the best checkpoint for inference
        n = len(hist)
        e0 = int(np.floor(0.75 * n))
        eps = list(range(e0, n))
        sts = [torch.load(ROOT / "models" / f"{tag}.ep{e}.pt", map_location="cpu") for e in eps]
        avg = {}
        for k in sts[0]:
            if sts[0][k].is_floating_point():
                avg[k] = sum(st[k].float() for st in sts) / len(sts)
            else:
                avg[k] = sts[-1][k]
        torch.save({"state": avg, "epoch": n - 1, "es": float("nan"), "cfg": cfg, "swa_epochs": eps}, ck_best)
        for e in range(n):
            (ROOT / "models" / f"{tag}.ep{e}.pt").unlink(missing_ok=True)
        log(f"{tag}: SWA over epochs {eps[0]}..{eps[-1]} ({len(eps)} of {n}) -> {ck_best}")
    json.dump(dict(tag=tag, cfg=cfg, best=best, best_epoch=best_ep, epochs=len(hist), plateaued=stopped,
                   train_min=sum(h["secs"] for h in hist) / 60, hist=hist), open(done, "w"), indent=1)
    log(f"{tag}: best {best:.4f} @ep {best_ep} of {len(hist)}; plateaued={stopped}")


@torch.no_grad()
def cmd_infer(a):
    from pieces_infer import subset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tag = f"{a.tag}_f{a.fold}"
    otag = f"{a.otag or a.tag}_f{a.fold}"
    out = ROOT / "fpreds" / f"{otag}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        log("exists"); return
    ck = torch.load((Path(a.ckdir) if a.ckdir else ROOT / "models") / f"{tag}.pt", map_location=dev,
                    weights_only=False)
    cfg = ck["cfg"]
    if "classes" in cfg:                         # note 92b: K-class function head
        C7[:] = cfg["classes"]
    model = Net69(cfg).to(dev).eval()
    kt = keep_table(a.keep, a.keep_thr) if a.keep else None
    n_hit = n_det = 0
    model.load_state_dict(ck["state"])
    table, sig = F53.func_table(False)
    f0 = set((sig.period + "|" + sig.DeviceId)[sig.fold == a.fold])
    keys = sorted(k for k in table if k in f0)
    if a.smoke:
        keys = keys[:3]
    log(f"{otag}: epoch {ck['epoch']}; {len(keys)} held-out signals; klong {a.klong}")
    acc, pacc = {}, {}
    for period in ("dec", "stg"):
        pk = [k for k in keys if table[k]["period"] == period]
        for name, start, secs in I2.WINDOWS[period]:
            pcs = I2.pieces(period, start, secs, max(M.GRID, a.klong))
            if secs > M.LONG_S:
                pcs = [pcs[i] for i in subset(len(pcs), a.klong)]
            plan = [(k, p0, (p1 - p0) // 1000, name) for k in pk for p0, p1 in pcs]
            dl = DataLoader(F53.DSF(table, plan, cfg["bw"]), batch_sampler=M.batches53(plan, 180, 6),
                            num_workers=a.workers, collate_fn=F53.collateF)
            for batch in dl:
                if kt is not None:
                    batch["keep"], h = keep_mask(batch, kt)
                    n_hit += h; n_det += int(batch["dmask"].sum())
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                    logit, flog = forward69(model, batch, dev, chunk=a.ichunk)
                lp = F.log_softmax(flog.float(), dim=2).cpu().numpy()
                lq = F.log_softmax(logit.float(), dim=2).cpu().numpy()
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
    rows = []
    for (key, win, det), (cand, s, n) in pacc.items():
        v = s / max(n, 1)
        p = np.exp(v - v.max()); p /= p.sum()
        dv = key.split("|", 1)[1] + ("@stg" if key.startswith("stg|") else "")
        rows += [(dv, int(det), int(c), win, float(pv)) for c, pv in zip(cand, p)]
    pout = ROOT / "ppreds" / f"{otag}.parquet"
    pout.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["DeviceId", "Detector", "cand_phase", "win", "prob"]).to_parquet(pout, index=False)
    log(f"wrote {pout} ({len(rows):,} rows)")
    st = dict(STATS, det_pieces=n_det, det_pieces_filtered=n_hit, keep=a.keep, keep_thr=a.keep_thr)
    json.dump(st, open(ROOT / "fpreds" / f"{otag}.pairs.json", "w"))
    log(f"pairs run {STATS['pairs_run']:,} of {STATS['pairs_all']:,}; filtered det-pieces {n_hit:,} of {n_det:,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "infer"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--otag", default="")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bw", type=int, default=1000)
    ap.add_argument("--fstem", type=int, default=1)
    ap.add_argument("--arch", default="tcn", choices=["tcn", "gru"], help="pair backbone (note 74; gru runs in fp32)")
    ap.add_argument("--sib", default="none", choices=["none", "mean", "attn"])
    ap.add_argument("--cw", type=float, default=1.0)
    ap.add_argument("--ls", type=float, default=0.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--cdrop", type=float, default=0.0)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--nblk", type=int, default=7)
    ap.add_argument("--minutes", default="5,10,15,20,30,30,30")
    ap.add_argument("--max_det", type=int, default=12)
    ap.add_argument("--klong", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smoke", type=int, default=0, help="steps per epoch (smoke test)")
    ap.add_argument("--accum", type=int, default=1, help="note 74: micro-batches per batch (gradient accumulation)")
    ap.add_argument("--ichunk", type=int, default=1024, help="pairs per forward chunk at inference")
    ap.add_argument("--ckdir", default="", help="checkpoint folder for infer (default %%DC_WORK%%/tcn53/models)")
    ap.add_argument("--keep", default="", help="note 74: tree pair table for the candidate filter (infer)")
    ap.add_argument("--keep_thr", type=float, default=0.01)
    ap.add_argument("--frows", default="", help="note 74: function label rows (default tcn53/func_rows.parquet)")
    ap.add_argument("--full", action="store_true", help="note 74: refit on every training signal (no fold held out)")
    ap.add_argument("--lr_sched", default="", help="--full: json list of the learning rate per epoch")
    ap.add_argument("--echunk", type=int, default=2048, help="pairs per forward chunk in inner-val evaluation")
    ap.add_argument("--swa", action="store_true", help="note 95: average the last 25 %% of epochs (SWA) as the model")
    ap.add_argument("--classes", default="", help="note 92b: function classes (the 7 first, then Other subclasses)")
    a = ap.parse_args()
    if a.classes:
        cl = a.classes.split(",")
        assert cl[:7] == C7[:7] and len(set(cl)) == len(cl), cl
        C7[:] = cl
        log(f"function classes: {C7}")
    if a.frows:                  # note 74: e.g. %DC_WORK%/tcn53/func_rows_v4l.parquet (v4l labels, note 81)
        F53.FROWS = Path(a.frows)
        log(f"function labels from {F53.FROWS}")
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
