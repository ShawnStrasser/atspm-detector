"""Note 83: full-data refits for final_v3 candidate v4 -- every model that reads function labels refitted on v4l.
-> %DC_WORK%/final_v3_work/v3fit83/ (starts as a copy of v3fit77; phase ranker + decoder + GRU unchanged: phase truth =
timing, which v4l does not touch).

    python fit83.py func       # 229 function trees, 3 seeds, v4l labels (t57 setup -> LABEL_SETS["v3s"] = v4l),
                               # n_estimators = mean best iteration of the 18 note-80 v4l fold fits (f76/function_c_v4l)
    python fit83.py lanetruth  # ln8 print-lane truth rebuilt from v4l -> f83/ln8/truth_*.parquet (+ change counts)
    python fit83.py lanes      # lanes D both orientations (fit77 recipe) on the v4l lane truth + v4l OOF function block
    python fit83.py stacker    # context stacker 'single' (one function-net member): v4l trees OOF + v4l truth, the
                               # three single-seed siba OOF versions stacked (3 x rows), WITHOUT the 15 health columns
    python fit83.py sbfeats    # setback sb7 features on v4l (print store cabinet_v4l, v4l OOF function) -> f83/sb7
    python fit83.py setback    # setback P50 per length group on f83/sb7 (fit75 recipe)
One stage per process.  Run with F76_ARM=c.  locked_v2 asserted absent (loaders + here).  CPU, 4 threads.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("F76_ARM", "c")
import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: E402
from common import DC_WORK  # noqa: E402

for _p in ("final75", "final76", "final77", "evaluation", "trackA", "lanes"):
    sys.path.insert(0, str(rpath.CODE / _p))
assert os.environ.get("F76_ARM") == "c", "run with F76_ARM=c"
import fit75 as F75  # noqa: E402
import fit76 as F76  # noqa: E402

THREADS = 4
F75.THREADS = THREADS
OUT = DC_WORK / "final_v3_work" / "v3fit83"
if not OUT.exists():
    shutil.copytree(DC_WORK / "final_v3_work" / "v3fit77", OUT)
F75.OUT = OUT
F76.OUT = OUT
F83 = DC_WORK / "final_v3_work" / "f83"
TREES_V4L = DC_WORK / "final_v3_work" / "f76" / "function_c_v4l"
CFGD = TREES_V4L / "v6e" / "drop_pp_xcand-pp_pdiff-pp_v2-yr-ratio-phctx"
LAB_V4L = rpath.REPO / "research" / "labels" / "function_labels_v4l.parquet"
N_STACKX = 20
HF = ["hf_score", "hf_status", "hf_nfam", "hf_s_dropout", "hf_s_stuck", "hf_s_chatter", "hf_s_rapid", "hf_s_volume",
      "hf_s_level", "hf_s_choppy", "hf_chi", "hf_surge", "hf_drop", "hf_chat", "hf_rel_chi", "hf_clus_n"]
HEALTH15 = [N_STACKX + i for i, c in enumerate(HF) if c != "hf_clus_n"]     # note 79b: dropped; hf_clus_n kept
log = F75.log


def v4l_trees():
    """function paths: arm-c frame, trees OOF / timing = the note-80 v4l fold fits."""
    import t57_function as T57F
    import s59_step6 as S59
    import v3_retrain as V
    assert V.LABEL_SETS["v3s"][0].resolve() == LAB_V4L.resolve(), "LABEL_SETS['v3s'] is not v4l"
    import atspm_score as AS
    assert Path(AS.V3S).resolve() == LAB_V4L.resolve(), "atspm_score.V3S is not v4l"
    F76._func_paths(True)                                  # arm-c frame patch (V.set_frame)
    T57F.OUT = TREES_V4L
    S59.FUNC_DIR = CFGD
    assert (CFGD / "timing.jsonl").exists()
    return CFGD


# ================================================================================================ trees
def stage_func(a):
    v4l_trees()
    F75.stage_func(a)
    m = json.load(open(OUT / "function" / "function229.json"))
    m["recipe"] = m["recipe"].replace("v3s labels", "v4l labels (note 80/81)")
    m["n_estimators_rule"] = "mean best iteration of the 18 note-80 v4l six-fold fits (3 seeds); no early stopping"
    F75.save_json(m, OUT / "function" / "function229.json")


# ================================================================================================ lanes
def _l8_v4l():
    import ln8_validate as L8
    d = F83 / "ln8"
    d.mkdir(parents=True, exist_ok=True)
    for f in ("cues.parquet", "dets.parquet", "decode_pick.json"):
        if not (d / f).exists():
            shutil.copy(L8.OUT / f, d / f)
    L8.OUT = d
    L8.V3S = LAB_V4L
    L8.FUNC_DIR = CFGD
    return L8, d


def stage_lanetruth(a):
    L8, d = _l8_v4l()
    old = DC_WORK / "lanes" / "ln8"
    L8.stage_truth(argparse.Namespace(workers=THREADS))
    import ln2_pairmodel as L2
    T0 = L2.truth_pairs(pd.read_parquet(old / "truth_det.parquet"), pd.read_parquet(old / "truth_phase.parquet"))
    T1 = L2.truth_pairs(pd.read_parquet(d / "truth_det.parquet"), pd.read_parquet(d / "truth_phase.parquet"))
    k = ["DeviceId", "da", "db"]
    m = T0[k + ["same_lane"]].merge(T1[k + ["same_lane"]], on=k, how="outer", suffixes=("_o", "_n"), indicator=True)
    r = {"pairs_old": len(T0), "pairs_new": len(T1), "only_old": int((m._merge == "left_only").sum()),
         "only_new": int((m._merge == "right_only").sum()),
         "same_lane_changed": int(((m._merge == "both") & (m.same_lane_o != m.same_lane_n)).sum()),
         "signals_old": int(T0.DeviceId.nunique()), "signals_new": int(T1.DeviceId.nunique())}
    json.dump(r, open(d / "truth_change.json", "w"), indent=1)
    log(f"lane truth v3s -> v4l: {r}")


def stage_lanes(a):
    import lightgbm as lgb
    import lane_output as LO
    v4l_trees()
    L8, d8 = _l8_v4l()
    import of77
    dd = OUT / "lanes"
    D, Pf, Pr = of77.sym_frames()
    assert not Pf.DeviceId.isin(F75.locked()).any()
    il = np.flatnonzero(Pf.labelled.to_numpy())
    c2 = L8.c2_matrix(Pf, D, il)
    X = np.vstack([np.hstack([Pf.loc[il, LO.FEATURES].to_numpy(np.float32), c2[0]]),
                   np.hstack([Pr.loc[il, LO.FEATURES].to_numpy(np.float32), c2[0]])])
    names = LO.FEATURES + c2[1]
    y = np.concatenate([Pf.same_lane.to_numpy()[il]] * 2).astype(int)
    log(f"lanes D (both orientations, v4l): {len(il):,} labelled pair-windows x 2, {X.shape[1]} features")
    files = []
    for s in L8.SEEDS:
        f = dd / f"lane_pair_D_s{s}.txt"
        files.append(f.name)
        lgb.LGBMClassifier(random_state=s, **dict(L8.PARAMS, n_jobs=THREADS)).fit(pd.DataFrame(X, columns=names), y) \
            .booster_.save_model(str(f))
        log(f"  seed {s} done")
    det, ph, T = L8.truth_tabs()
    D9 = D[(D.period == "stg") & D.win.isin(L8.TRAIN_WINS) & D.DeviceId.isin(set(ph.DeviceId))]
    pri = L8.span_priors(det, ph, D9.rename(columns={"Detector": "det"}), set(D9.DeviceId))[0]
    pk = json.load(open(of77.O8 / "decode_pick.json"))["pick"]["D.func"]
    lams = [float(v.split("@")[1]) for v in pk.values()]
    lam = float(pd.Series(lams).mode().iloc[0])
    F75.save_json({"pair_models": files, "features": names, "n_cue_context_features": len(LO.FEATURES),
                   "c2_columns": json.load(open(L8.FUNC_DIR / "cols.json"))["cols"] + [f"P_{c}" for c in LO.C7],
                   "span_prior": {fn: {str(k): v for k, v in x.items()} for fn, x in pri.items()},
                   "lam": lam, "lam_fold_picks": lams, "beta": 0.0, "role_pen": 4.0, "min_on": LO.MIN_ON,
                   "params": {k: v for k, v in L8.PARAMS.items() if k != "n_jobs"},
                   "orientation": "both (note 77): trained on (a, b) and (b, a) of every labelled pair; at inference "
                                  "P(same) = mean of the two orientations; decode in canonical behavioural order",
                   "labels": "note 83: print-lane truth from function_labels_v4l; function block = v4l OOF trees",
                   "note": "note 58 joint pair model D: cues + context + predicted-function pair type + function model "
                           "block (229 features + 7 probabilities of both detectors, min / max); decode = note-42 func "
                           "decoder; lam = mode of the note-77 per-fold picks",
                   "trained_on": {"labelled_pairs": int(len(il)), "rows": int(len(y)),
                                  "signals": int(Pf.loc[il, "DeviceId"].nunique()), "same_share": float(y.mean())}},
                  dd / "lane_model.json")
    log(f"lam {lam}")


# ================================================================================================ stacker
def stacker_inputs():
    """(E, X_seed[s] for s in 0..2 with the 15 health columns removed, y, trm, names) -- the v4 recipe's stacker rows.
    Paths as s80.stage_stack: only T57F.OUT moves (no frame patch: the scoring frame / OOF runs use the v6e paths)."""
    import t57_function as T57F
    import v3_retrain as V
    import atspm_score as AS
    import f76_function  # noqa: F401  (its import sets T57F.OUT = f76/function_c; must come BEFORE the override)
    assert V.LABEL_SETS["v3s"][0].resolve() == LAB_V4L.resolve() and Path(AS.V3S).resolve() == LAB_V4L.resolve()
    T57F.OUT = TREES_V4L
    import cand64 as C
    import of77
    s74, S, E = of77.setup_new()
    fr, C7 = E["fr"], E["C7"]
    assert not fr.DeviceId.str.lower().isin(F75.locked()).any()
    import s59_step6 as S59
    assert Path(S59.FUNC_DIR).resolve() == CFGD.resolve(), f"stacker tree input dir {S59.FUNC_DIR} is not the v4l OOF"
    Pt = E["Pt"]
    y = np.array([C7.index(t) if isinstance(t, str) and t in C7 else -1 for t in fr.truth_v3s.to_numpy(object)])
    trm = (y >= 0) & ~fr.exclude_train_score.fillna(False).to_numpy(bool) & (fr.det_n_on >= 5).to_numpy()
    Xs = {}
    for s in (0, 1, 2):
        P1 = s74.net_probs(fr, f"x69_siba:{s}")
        P1 = np.where(np.isnan(P1[:, :1]), Pt, P1)
        X = np.hstack([S.stack_X(fr, Pt, P1), S.ctx_X(E, C.W_TREE * Pt + (1 - C.W_TREE) * P1)])
        assert X.shape[1] == 62, X.shape
        assert np.array_equal(X[:, N_STACKX + len(HF) + 1], fr.pk_unhealthy.to_numpy(float))
        Xs[s] = np.delete(X, HEALTH15, axis=1)
    names = ([f"lt_{c}" for c in C7] + [f"ln_{c}" for c in C7] + ["log_minutes", "ent_t", "ent_n", "margin_t", "margin_n",
                                                                  "agree"] +
             HF + ["pk_span", "pk_unhealthy", "pk_track", "n_span_peers", "n_coloc_peers", "nl_self", "lane_min",
                   "n_lanes_phase", "phase_n_lanes", "phase_n_lanes_conf", "lane_conf", "n_ph", "phm_A", "phm_P", "phm_C",
                   "phm_Y", "rk_A", "rk_P", "rk_C", "rk_Y", "lm_n", "lm_A", "lm_P", "lm_C", "lm_Y", "log1p_det_n_on"])
    assert len(names) == 62
    names = [n for i, n in enumerate(names) if i not in HEALTH15]
    return s74, S, E, Xs, y, trm, names


PRM0 = dict(objective="multiclass", num_class=7, learning_rate=0.05, num_leaves=7, max_depth=3, min_data_in_leaf=300,
            lambda_l2=10.0, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1)


def stage_stacker(a):
    import lightgbm as lgb
    d = OUT / "stacker"
    for f in d.glob("stacker*"):
        f.unlink()
    s74, S, E, Xs, y, trm, names = stacker_inputs()
    X = np.vstack([Xs[s][trm] for s in (0, 1, 2)])
    yy = np.concatenate([y[trm]] * 3)
    log(f"stacker single (v4l, no health): X {X.shape}, {int(trm.sum()):,} rows x 3 seeds' net inputs")
    np.save(d / "X_check.npy", Xs[0][:2000].astype(np.float32))
    files = []
    for s in (0, 1, 2):
        f = d / f"stacker_single_s{s}.txt"
        lgb.train(dict(PRM0, num_threads=THREADS, seed=s), lgb.Dataset(X, yy), num_boost_round=150).save_model(str(f))
        files.append(f.name)
        log(f"  seed {s} done")
    fr = E["fr"]
    F75.save_json({"classes": E["C7"], "n_features": len(names), "feature_names": names, "files": files,
                   "params": PRM0, "num_boost_round": 150,
                   "variant": "single: trained on the three single-seed versions of every OOF row (3 x rows, note 75)",
                   "net": "siba x69_siba single-seed OOF fold models (v3s-trained); production net = full-data siba on v4l",
                   "dropped": "note 79b / 83: the 15 health columns (health_core score / status / families / 7 rule "
                              "scores; stack-relative chi / surge / drop / chat / rel_chi); hf_clus_n and pk_unhealthy kept",
                   "labels": "v4l (note 80/81); trees = note-80 v4l OOF",
                   "trained_on": {"rows": int(trm.sum()), "signals": int(fr.DeviceId[trm].nunique())}},
                  d / "stacker.json")


# ================================================================================================ setback
def stage_sbfeats(a):
    import sb7_validate as S7
    import sb5_eval as SB
    S7.OUT = F83 / "sb7"
    S7.FUNC_DIR = CFGD
    S7.V3S = LAB_V4L
    real = S7.truth_all

    def truth_all():
        """sb7 truth from the v4l store (cabinet_v4l = training store + the 71 released records, re-reads folded in)."""
        con = SB.connect()
        cols = "DeviceId, detector, function, technology, distance_ft, unusual_layout"
        p = con.sql(f"SELECT {cols}, 'v4l' AS src FROM read_parquet("
                    f"'{(DC_WORK / 'cabinet_v4l' / 'print_labels.parquet').as_posix()}')").df()
        p["dev"] = p.DeviceId.str.lower()
        assert not p.dev.isin(F75.locked()).any()
        d = p.distance_ft.map(SB.parse_dist)
        p["dist"], p["dist_kind"] = [x[0] for x in d], [x[1] for x in d]
        p = p.rename(columns={"detector": "det", "function": "print_fn_raw"})
        p["det"] = p.det.astype("int16")
        v = pd.read_parquet(LAB_V4L, columns=["DeviceId", "detector", "print_function", "unusual_layout",
                                              "exclude_train_score"])
        v["dev"] = v.DeviceId.str.lower()
        v["det"] = v.detector.astype("int16")
        p = p.merge(v[["dev", "det", "print_function", "unusual_layout", "exclude_train_score"]].rename(
            columns={"unusual_layout": "unusual_v3s"}), on=["dev", "det"], how="left")
        p["print_fn"] = p.print_function.fillna(p.print_fn_raw)
        unus = p.unusual_v3s.where(p.unusual_v3s.notna(), p.unusual_layout)
        p = p[~unus.eq(True) & ~p.exclude_train_score.eq(True)]
        return p[["dev", "det", "print_fn", "technology", "dist", "dist_kind", "src"]].drop_duplicates(["dev", "det"])
    assert real is not None
    S7.truth_all = truth_all

    class _Con:                       # sb7 sets DuckDB threads=6; this machine's cap is 4 (AGENTS.md / brief)
        def __init__(self, c):
            self._c = c

        def execute(self, q, *a, **k):
            if q.strip().upper().startswith("SET THREADS"):
                q = f"SET threads={THREADS}"
            return self._c.execute(q, *a, **k)

        def __getattr__(self, n):
            return getattr(self._c, n)
    _orig_connect = SB.connect

    def _connect():
        c = _orig_connect()
        c.execute(f"SET threads={THREADS}")
        c.execute("SET memory_limit='10GB'")
        return _Con(c)
    SB.connect = _connect
    v4l_trees()
    S7.OUT.mkdir(parents=True, exist_ok=True)
    S7.stage_feats()


def stage_setback(a):
    import sb7_validate as S7
    S7.OUT = F83 / "sb7"
    assert (S7.OUT / "feat.parquet").exists()
    for f in (OUT / "setback").glob("setback_*_q50_s*.txt"):
        f.unlink()
    F75.stage_setback(a)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["func", "lanetruth", "lanes", "stacker", "sbfeats", "setback"])
    a = ap.parse_args()
    F83.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    globals()[f"stage_{a.stage}"](a)
    log(f"== {a.stage} done ({time.time() - t0:.0f}s)")
