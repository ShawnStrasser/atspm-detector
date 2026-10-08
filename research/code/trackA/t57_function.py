"""Note 57 (validation plan step 2): the FUNCTION trees on their own -- feature groups, phase input (trees-only vs
blend), 7-class vs ATSPM-focused weighting, seed averaging.  Six folds (folds_v4), v3s labels, the note-55 recipe
(`v3_retrain.py fit --frame v6e --labels v3s --min-on 5 --clean --require-validated --allow-not-checkable-high
--health3 --dec-role --dec-role-rule fp`, variant first.all.wi, same LightGBM params).

    python t57_function.py fit --cfg drop:px --seeds 0          # leave a feature group out
    python t57_function.py fit --cfg full --frame v6t --seeds 0,1,2   # trees-only phase input
    python t57_function.py fit --cfg wA2 --seeds 0,1,2          # ATSPM truth rows weight 2 (Other/Mid/Bike keep 1)
    python t57_function.py fit --cfg shuf:px --seeds 0          # px columns permuted across rows (control)
    python t57_function.py score --cfgs v6e/full,v6e/drop:px@s0 --base v6e/full@s0 --tag x

`full` on frame v6e IS the note-55 run (`frame_v6e/run_51ba131222_..._h3_drfp`), loaded from there.
Scoring = note 55's: ATSPM-only stack-aware score on note-54 step-4 rows (v3s truth, Dec fp rule), per-lane decode
with the stack-scoped PICK rule (gated >= 30 min), everything / realistic, paired signal bootstrap.  Lanes
(`ln5_lanes_v3s`) and pick inputs (`ln6_pick_v3s`) are those of the v3s baseline for every config (they are
computed from a model's predictions; re-computing them per config is ~35 min each) -- an approximation that can only
favour the baseline slightly.  `cfg@sN` = seed N alone; `cfg` = mean of its seeds.  `@pool` suffix = decision rule
with the non-ATSPM classes' probability pooled (no retrain).  Locked_v2 asserted absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: F401,E402
import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import v3_retrain as V  # noqa: E402
import a2_model as A2  # noqa: E402

OUT = V.DCW / "trees57" / "function"
BASE_RUN = "run_51ba131222_exclude_min5_clean_valnc_h3_drfp"
VAR = "first.all.wi"
ATS = ["Advance", "Presence", "Count", "Yellow_Red"]
PH_FEATS = json.load(open(V.DCW / "final_v3_work" / "phase_v3" / "phase_lgbm_v5.json"))["features"]
PH_V2 = ("pex_", "dtg_h", "tog_h", "dur_g2", "burst_g2", "first_on", "cyc_hit_frac", "release_lag", "queue_end_frac",
         "n_long", "call43_b", "call43_red_035", "cogreen", "excl_secs", "partner_lead", "call43_red_lift")
COND = ("on_lift_green_coord", "on_lift_green_free", "px_disp_coord", "px_disp_free", "px_disp_coord_over_free")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def group_of(c: str) -> set:
    g = set()
    if c.split("__")[0] in COND:
        g.add("cond")
    if c in PH_FEATS:
        if "__" in c:
            g.add("pp_pdiff" if c.endswith("__pdiff") else "pp_xcand")
        elif c.startswith(PH_V2):
            g.add("pp_v2")
        else:
            g.add("pp_core")
    elif c.startswith("px_"):
        g.add("px")
        g.add("px_" + A2.family_of(c))
    elif c.startswith("yr_"):
        g.add("yr")
    elif "__sib" in c or c == "sib_n":
        g.add("sib")
    elif c.startswith(("lagany", "lagsib")):
        g.add("lag")
    elif c == "top_prob":
        g.add("phctx")
    else:
        g.add("ratio")
    return g


def setup(frame: str):
    V.set_frame(frame)
    V.LABELS_V3, V.DEC_ROLE_FILE = V.LABEL_SETS["v3s"]
    V.MIN_ON, V.CLEAN, V.REQ_VAL, V.NC_HIGH = 5, True, True, True
    V.HEALTH3, V.DEC_ROLE, V.DEC_RULE = True, True, "fp"


def cfg_dir(frame: str, cfg: str) -> Path:
    setup(frame)
    if cfg == "full" and frame == "v6e":
        return V.OUT / BASE_RUN
    return OUT / frame / cfg.replace(":", "_").replace("+", "-")


def stage_fit(a):
    import lightgbm as lgb
    setup(a.frame)
    # note 81: LABEL_SETS["v3s"] is now the v4l table, whose fingerprint differs from BASE_RUN (the v3s-trained note-55
    # run). A fit on another label table must write to its own folder (s80.stage_fit sets OUT and BASE_RUN), never
    # into the v3s trees folders.
    if V.run_dir("exclude").name != BASE_RUN:
        assert OUT != V.DCW / "trees57" / "function", (
            f"label table {V.LABELS_V3.name} is not BASE_RUN's; set t57_function.OUT / BASE_RUN first (see s80.stage_fit)")
    fr, cols = V.load_feats()
    lab = V.load_labels(fr, "exclude")
    y, ok = V.variant_target(lab, fr, VAR)
    yi = pd.Series(y).map({c: i for i, c in enumerate(V.C7)}).fillna(-1).astype(int).to_numpy()
    cfg = a.cfg
    w = np.ones(len(fr))
    if cfg.startswith("drop:"):
        gs = set(cfg[5:].split("+"))
        cols = [c for c in cols if not (group_of(c) & gs)]
    elif cfg.startswith("shuf:"):
        gs = set(cfg[5:].split("+"))
        rng = np.random.default_rng(57)
        sh = [c for c in cols if group_of(c) & gs]
        for c in sh:
            fr[c] = fr[c].to_numpy()[rng.permutation(len(fr))]
        log(f"shuffled {len(sh)} columns")
    elif cfg.startswith("wA"):
        w = np.where(np.isin(y.astype(str), ATS), float(cfg[2:]), 1.0)
    else:
        assert cfg == "full", cfg
    d = cfg_dir(a.frame, cfg)
    d.mkdir(parents=True, exist_ok=True)
    json.dump({"cfg": cfg, "frame": a.frame, "n_cols": len(cols), "cols": cols}, open(d / "cols.json", "w"), indent=1)
    log(f"{a.frame}/{cfg}: {len(cols)} features, {int(ok.sum()):,} training rows -> {d}")
    folds = fr.fold.to_numpy()
    for s in [int(x) for x in a.seeds.split(",")]:
        for k in range(A2.N_FOLDS):
            f = d / f"P_{VAR}_s{s}_f{k}.npy"
            if f.exists():
                continue
            t0 = time.time()
            inner = (k + 1) % A2.N_FOLDS
            trm = ok & (folds != k) & (folds != inner)
            vam = ok & (folds == inner)
            prm = dict(A2.FUNC_PARAMS, n_jobs=a.threads, num_class=len(V.C7), seed=s, bagging_seed=s + 1,
                       feature_fraction_seed=s + 2, data_random_seed=s + 3)
            n = prm.pop("n_estimators")
            m = lgb.LGBMClassifier(n_estimators=n, **prm)
            m.fit(fr.loc[trm, cols], yi[trm], sample_weight=w[trm], eval_set=[(fr.loc[vam, cols], yi[vam])],
                  eval_sample_weight=[w[vam]], eval_metric="multi_logloss",
                  callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
            np.save(f, m.predict_proba(fr.loc[folds == k, cols]).astype(np.float32))
            with open(d / "timing.jsonl", "a") as fh:
                fh.write(json.dumps({"seed": s, "fold": k, "trees": int(m.best_iteration_ or 0),
                                     "secs": round(time.time() - t0, 1)}) + "\n")
            log(f"  s{s} f{k}: {m.best_iteration_} trees, {time.time()-t0:.0f}s")


# ------------------------------------------------------------------ scoring
def load_P(frame: str, cfg: str, seeds, keys: pd.DataFrame) -> np.ndarray:
    """Seed-mean 7-class OOF of a config, aligned to `keys` (the v6e scoring frame's rows)."""
    setup(frame)
    fk = pd.read_parquet(V.FEATS, columns=V.KEY + ["fold"])
    fk["DeviceId"] = fk.DeviceId.str.lower()
    d = cfg_dir(frame, cfg)
    Ps = []
    for s in seeds:
        P = np.zeros((len(fk), 7), np.float32)
        fo = fk.fold.to_numpy()
        for k in range(6):
            P[fo == k] = np.load(d / f"P_{VAR}_s{s}_f{k}.npy")
        Ps.append(P)
    P = np.mean(Ps, 0)
    fk["_i"] = np.arange(len(fk))
    fk["Detector"] = fk.Detector.astype(keys.Detector.dtype)
    idx = keys[V.KEY].merge(fk[V.KEY + ["_i"]], on=V.KEY, how="left")._i.to_numpy()
    assert not np.isnan(idx).any()
    return P[idx.astype(int)]


def seeds_of(frame, cfg):
    d = cfg_dir(frame, cfg)
    return sorted({int(p.name.split("_s")[1].split("_")[0]) for p in d.glob(f"P_{VAR}_s*_f5.npy")})


def pooled(P):
    """Decision rule: non-ATSPM classes compete with their summed probability (held by their best member)."""
    P = P.copy()
    na = [V.C7.index(c) for c in ("Mid", "Bike", "Other")]
    tot = P[:, na].sum(1)
    best = np.array(na)[P[:, na].argmax(1)]
    P[:, na] = 0
    P[np.arange(len(P)), best] = tot
    return P


_FR = {}


def scoring_frame(frame: str):
    """note-55 scoring frame (labels, lanes, pick inputs, groups); pred_phase from the config's own frame."""
    if frame in _FR:
        return _FR[frame]
    import atspm_score as S
    import atspm_pick55 as AP
    base = S.load(BASE_RUN)                      # v6e keys + labels (+ the baseline P, replaced below)
    if frame != "v6e":
        setup(frame)
        pp = pd.read_parquet(V.FEATS, columns=V.KEY + ["pred_phase"])
        pp["DeviceId"] = pp.DeviceId.str.lower()
        pp = pp.astype({"Detector": base.Detector.dtype})
        x = base[V.KEY].merge(pp, on=V.KEY, how="left")
        assert x.pred_phase.notna().all()
        log(f"{frame}: predicted phase differs from v6e on {(x.pred_phase.to_numpy() != base.pred_phase.to_numpy()).mean():.2%}")
        base["pred_phase"] = x.pred_phase.to_numpy()
    fr = S.attach_lanes(base, "ln5_lanes_v3s")
    fr = AP.attach_pick(fr, "ln6_pick_v3s")
    G = AP.groups(fr)
    rows = AP.score_rows(fr)
    _FR[frame] = (fr, G, rows)
    return _FR[frame]


def evaluate(spec: str):
    """spec = frame/cfg[@sN][@pool][@greedy|@argmax] -> per-row credit frames for everything / realistic."""
    import atspm_score as S
    import atspm_pick55 as AP
    parts = spec.split("@")
    frame, cfg = parts[0].split("/", 1)
    opts = parts[1:]
    seeds = [int(o[1:]) for o in opts if o.startswith("s") and o[1:].isdigit()] or seeds_of(frame, cfg)
    fr, G, rows = scoring_frame(frame)
    P = load_P(frame, cfg, seeds, fr)
    if "pool" in opts:
        P = pooled(P)
    dec = "argmax" if "argmax" in opts else "greedy" if "greedy" in opts else "pick"
    pred = P.argmax(1) if dec == "argmax" else AP.decode_all(fr, P, G, dec)
    return {s: S.credit(fr, pred, "truth_v3s", r, True) for s, r in rows.items()}, seeds


def stage_score(a):
    import atspm_score as S
    WG = ["m5", "m10", "m30", "h1", "h3", "h6", "h24", "full"]
    base, _ = evaluate(a.base)
    res = {"base": a.base, "results": {}}
    for spec in [a.base] + [s for s in a.cfgs.split(",") if s and s != a.base]:
        D, seeds = evaluate(spec) if spec != a.base else (base, None)
        r = {"seeds": seeds}
        for sname in ("everything", "realistic"):
            s = S.summ(D[sname])
            r[sname] = {"n": s["n"], "signals": s["signals"], "atspm": s["atspm"], "acc7": s["acc7"],
                        "secondary": s["secondary_acc"], "stack_extra": s["stack_extra"], "err": s["err"],
                        "by_window": {g: S.summ(D[sname][D[sname].wgroup == g])["atspm"] for g in WG}}
            if spec != a.base:
                r[sname]["ci_atspm_pt"] = S.boot(base[sname], D[sname])
                r[sname]["ci_acc7_pt"] = S.boot(base[sname], D[sname], "ok7")
                r[sname]["d_atspm_pt"] = round(100 * (D[sname].ok_a.mean() - base[sname].ok_a.mean()), 3)
                bw = {}
                for g in ("m5", "m30", "h6", "full"):
                    b0, b1 = base[sname][base[sname].wgroup == g], D[sname][D[sname].wgroup == g]
                    bw[g] = [round(100 * (b1.ok_a.mean() - b0.ok_a.mean()), 3)] + S.boot(b0, b1)
                r[sname]["d_by_window_pt_ci"] = bw
        res["results"][spec] = r
        e, rr = r["everything"], r["realistic"]
        log(f"{spec:36s} ATSPM E {e['atspm']:.4f} R {rr['atspm']:.4f} acc7 E {e['acc7']:.4f} "
            + (f"| dE {e['d_atspm_pt']:+.3f} {e['ci_atspm_pt']} dR {rr['d_atspm_pt']:+.3f} {rr['ci_atspm_pt']}"
               if spec != a.base else "(base)"))
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / f"score_{a.tag}.json", "w"), indent=1, default=str)
    log(f"-> {OUT / f'score_{a.tag}.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fit", "score"])
    ap.add_argument("--cfg", default="full")
    ap.add_argument("--frame", default="v6e", choices=["v6e", "v6t"])
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--cfgs", default="")
    ap.add_argument("--base", default="v6e/full")
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    globals()[f"stage_{a.stage}"](a)
