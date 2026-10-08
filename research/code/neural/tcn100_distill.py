"""Note 100: distil the 3-member siba ensemble (x86_siba4l seeds 0/1/2, one fold) into ONE smaller siba TCN.

Teacher = mean of the three fold-k teachers' tempered probabilities, computed ONLINE on every training batch (the same
random 5-30 min windows / detector subsets the student sees; teachers in eval mode, bf16, no grad), for both heads:
  function  p_T(class) = mean_m softmax(flog_m / T)
  phase     q_T(cand)  = mean_m softmax(logit_m / T)   (over the window's candidates, as the listwise phase loss)
Student = tcn69_func.Net69 with the note-69 siba recipe (sib attn, max_det 32) and a smaller TCN (--width / --nblk).
Loss = alpha * (CE_func(true) + CE_phase(true))  +  (1 - alpha) * T^2 * (KL(p_T || p_s,T) + KL(q_T || q_s,T)),
CE on labelled actuated detectors (as tcn69), KL on every actuated detector of the window. Inner-val early stopping on
the student's TRUE-label accuracy, exactly as tcn69_func.cmd_train. The checkpoint is a plain Net69 checkpoint (cfg has
the student width), so `tcn69_func.py infer`, `export74.py` and `time74.py` take it unchanged.
Locked signals: never in the function table (func_table asserts it; re-asserted here).

    python tcn100_distill.py --tag x100_kdw56 --fold 0 --seed 0 --width 56 \
        --teachers x86_siba4l_f0,x86_siba4l_s1_f0,x86_siba4l_s2_f0 --T 2 --alpha 0.5 --frows .../func_rows_v4l.parquet
    python tcn69_func.py infer --tag x100_kdw56 --fold 0 --keep .../cand64/phase_oof.parquet --keep_thr 0.01
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tcn69_func as T  # noqa: E402

F53, M, D2, T2 = T.F53, T.M, T.D2, T.T2
ROOT, ADALL, log = T.ROOT, T.ADALL, T.log


def load_teachers(tags, dev, ckdir):
    out = []
    for t in tags:
        ck = torch.load(Path(ckdir) / f"{t}.pt", map_location=dev, weights_only=False)
        net = T.Net69(ck["cfg"]).to(dev).eval()
        net.load_state_dict(ck["state"])
        for p in net.parameters():
            p.requires_grad_(False)
        out.append(net)
        log(f"teacher {t}: epoch {ck['epoch']}, width {ck['cfg']['width']} x {ck['cfg']['nblk']}, sib {ck['cfg']['sib']}")
    return out


@torch.no_grad()
def teacher_targets(teachers, batch, dev, temp):
    pf = pp = None
    for net in teachers:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            logit, flog = T.forward69(net, batch, dev)
        f = torch.softmax(flog.float() / temp, -1)
        p = torch.softmax(logit.float() / temp, -1)
        pf = f if pf is None else pf + f
        pp = p if pp is None else pp + p
    return pf / len(teachers), pp / len(teachers)


def kd_terms(logit, flog, pf, pp, batch, dev, temp):
    act = (batch["nact"].to(dev) > 0)
    if not bool(act.any()):
        return None, None
    lsf = F.log_softmax(flog.float()[act] / temp, -1)
    kf = F.kl_div(lsf, pf[act], reduction="none").sum(-1).mean()
    ok = act & (batch["ncand"].to(dev)[:, None] >= 2)
    kp = None
    if bool(ok.any()):
        lsp = F.log_softmax(logit.float()[ok] / temp, -1)
        kp = F.kl_div(lsp, pp[ok], reduction="none").sum(-1).mean()
    return kf, kp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=56)
    ap.add_argument("--nblk", type=int, default=7)
    ap.add_argument("--teachers", default="x86_siba4l_f0,x86_siba4l_s1_f0,x86_siba4l_s2_f0")
    ap.add_argument("--tckdir", default="", help="teacher checkpoint folder (default %%DC_WORK%%/tcn53/models)")
    ap.add_argument("--T", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.5, help="weight of the true-label loss")
    ap.add_argument("--max_det", type=int, default=32)
    ap.add_argument("--minutes", default="5,10,15,20,30,30,30")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--frows", default="")
    a = ap.parse_args()
    if a.frows:
        F53.FROWS = Path(a.frows)
        log(f"function labels from {F53.FROWS}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    for d in ("models", "runs"):
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    tag = f"{a.tag}_f{a.fold}"
    done = ROOT / "runs" / f"{tag}.done.json"
    if done.exists():
        log("done"); return
    teachers = load_teachers(a.teachers.split(","), dev, a.tckdir or ROOT / "models")
    assert all(t.cfg["fold"] == a.fold for t in teachers), "teachers must share the student's held-out fold"
    cfg = dict(bw=1000, fstem=1, sib="attn", cw=1.0, ls=0.0, dropout=0.0, cdrop=0.0, width=a.width, nblk=a.nblk,
               minutes=[int(x) for x in a.minutes.split(",")], max_det=a.max_det, chans=ADALL, seed=a.seed,
               fold=a.fold, mode="joint", distill=dict(teachers=a.teachers, T=a.T, alpha=a.alpha))
    table, sig = F53.func_table(True)
    lk = M.P3N.locked_ids()
    assert not any(k.split("|", 1)[1] in lk for k in table)
    tr = set((sig.period + "|" + sig.DeviceId)[sig.fold != a.fold])
    keys = sorted(k for k in table if k in tr)
    rng = np.random.default_rng(a.seed + 7)           # same inner-val split as tcn69_func.cmd_train
    perm = rng.permutation(len(keys))
    n_es = max(20, int(0.1 * len(keys)))
    es_keys, tr_keys = [keys[i] for i in perm[:n_es]], [keys[i] for i in perm[n_es:]]
    model = T.Net69(cfg).to(dev)
    npar = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    log(f"{tag}: {json.dumps(cfg)}; params {npar(model):,} (teacher {npar(teachers[0]):,}); "
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
        dl = DataLoader(F53.DSF(table, plan, 1000, max_det=a.max_det, seed=a.seed),
                        batch_sampler=M.batches53(plan, 120, 8, shuffle_seed=ep),
                        num_workers=a.workers, collate_fn=F53.collateF, pin_memory=True, persistent_workers=False)
        model.train()
        t0, nb = time.time(), 0
        sums = np.zeros(4)
        for batch in dl:
            pf, pp = teacher_targets(teachers, batch, dev, a.T)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                logit, flog = T.forward69(model, batch, dev, train=True)
            lf = T.func_loss69(flog, batch, dev, cfg)
            lp = T2.phase_loss(logit, batch, dev)
            kf, kp = kd_terms(logit, flog, pf, pp, batch, dev, a.T)
            hard = [x for x in (lf, lp) if x is not None]
            soft = [x for x in (kf, kp) if x is not None]
            if not hard and not soft:
                continue
            loss = a.alpha * sum(hard) + (1 - a.alpha) * a.T ** 2 * sum(soft)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sums += [float(loss.detach()), float(sum(hard).detach()) if hard else 0.0,
                     float(kf.detach()) if kf is not None else 0.0, float(kp.detach()) if kp is not None else 0.0]
            nb += 1
            if a.smoke and nb >= a.smoke:
                break
        ap_, af = T.evaluate69(model, table, esplan[:40] if a.smoke else esplan, dev, a.workers)
        score = 0.5 * (ap_ + af)
        sched.step(score)
        m = sums / max(nb, 1)
        hist.append(dict(epoch=ep, loss=m[0], hard=m[1], kl_func=m[2], kl_phase=m[3], es_phase=ap_, es_func=af,
                         secs=time.time() - t0))
        log(f"ep {ep}: loss {m[0]:.4f} (hard {m[1]:.4f} klf {m[2]:.4f} klp {m[3]:.4f}) es phase {ap_:.4f} "
            f"func {af:.4f} lr {opt.param_groups[0]['lr']:.1e} {time.time()-t0:.0f}s ({nb} steps)")
        if score > best + 1e-5:
            best, best_ep = score, ep
            torch.save({"state": model.state_dict(), "epoch": ep, "es": score, "cfg": cfg}, ck_best)
        torch.save({"state": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep, "best": best, "best_ep": best_ep, "hist": hist}, ck_last)
        if ep - best_ep >= a.patience:
            stopped = True
            break
    json.dump(dict(tag=tag, cfg=cfg, best=best, best_epoch=best_ep, epochs=len(hist), plateaued=stopped,
                   train_min=sum(h["secs"] for h in hist) / 60, hist=hist), open(done, "w"), indent=1)
    log(f"{tag}: best {best:.4f} @ep {best_ep} of {len(hist)}; plateaued={stopped}")


if __name__ == "__main__":
    main()
