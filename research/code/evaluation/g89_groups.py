"""Note 89: decide the note-87 label-exclusion GROUPS with data, not by row review and never by model agreement.

For each group, the function trees (229-feature arm c, note-57/76 recipe, v4l training labels) are refit with that
group's TRAINING rows restored (labels as originally given; every other exclusion still applies), then the note-67/69
context stacker (main net siba x69_siba, 3 stacker seeds) and the gate-.9 lane decode exactly as s80.py. Scoring truth
is FIXED for every arm (v4l truth_v3s, everything / realistic) -- only the trees' training rows change.

    F76_ARM=c python g89_groups.py fit   --arm base --seeds 0         # trees -> %DC_WORK%/x89/trees/<arm>/
    F76_ARM=c python g89_groups.py stack --arm base                   # stacker -> %DC_WORK%/x89/stack/<arm>/
    F76_ARM=c python g89_groups.py score --arm base                   # ok arrays -> %DC_WORK%/x89/ok_<arm>.npz
    python g89_groups.py report                                       # paired deltas vs base -> x89/g89.json
    python g89_groups.py labels --restore a,b                         # final training labels -> research/labels/...

Arms (membership = %DC_WORK%/rev87/items_all.parquet, note 87):
  base            current recipe on note 88's labels v4m (dq_suspect recomputed without fault events 83-88, card
                  rule actuation-only) -- every arm starts from v4m
  repro           = base (kept for the reproduction check against note 88's f76/function_c_v4m seed 0)
  dec_role        Dec-2024 samples of Dec-role channels trained on (dec_role_changed_train cleared)
  check_fail / unhealthy / misconfigured   label-check verdict treated as pass for those detectors
  dq_check        every dq_suspect row left in v4m trained on (the --clean data-quality exclusion)
  beh_tied_low    R5 rows back (exclude_train_score off, train_use on)
  unusual_signal  the 21 unusual_layout signals' labels trained on
  small           R6 noise inputs + explained-other sweep (old Other label) + print re-read flips (old reading;
                  the user_ruling flip excluded)
  all:<a+b+..>    several groups at once
Not tested (user rules, not agent discretion): R1 loop config Count, YR identical to Count, stacked-group relabel,
R2 / R4. Seeds: --seeds 0 screens; the stacker gets single-seed trees then (spec_probs seed check relaxed, same for
every arm). locked_v2 asserted absent from every label / dec table written here.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CODE = next(p for p in Path(__file__).resolve().parents if p.name == "code")
for p in (CODE, CODE / "evaluation", CODE / "final76", CODE / "final77", CODE / "trackA"):
    sys.path.insert(0, str(p))
import rpath  # noqa: E402,F401

REPO = CODE.parents[1]
DCW = Path(os.environ.get("DC_WORK", Path.home() / "dc_work"))
X89 = DCW / "x89"
SFX = os.environ.get("G89_SFX", "")   # "" = seed-0 screen; "3" = 3-seed trees (stack / ok files apart)
SRC = os.environ.get("G89_SRC", "v4m")     # base label table of the experiment: v4m (note 89 round 1) | v4n (round 2)
V4L = REPO / f"research/labels/function_labels_{SRC}.parquet"   # v4m = v4l with dq recomputed without fault events
DEC4L = DCW / "cabinet_v4l" / "dec_role_changed_v4l.parquet" if SRC == "v4m" else DCW / "x89" / f"dec_role_changed_{SRC}.parquet"
if SRC != "v4m":                           # round 2 (base = v4n): own folder
    X89 = X89 / SRC
TRAIN_COLS = ("validated", "dq_suspect", "exclude_train_score", "train_use", "unusual_layout", "label_print_first", "tier")
ITEMS = DCW / "rev87" / "items_all.parquet"
CHANGES = REPO / "research/labels/function_label_changes_v4l.csv"
GE30 = ["m30", "h1", "h3", "h6", "h24", "full"]
GROUPS = ["dec_role", "check_fail", "unhealthy", "misconfigured", "dq_check", "beh_tied_low", "unusual_signal", "small",
          "not_checkable"]
VALKEY = {"check_fail": "fail", "unhealthy": "unhealthy", "misconfigured": "misconfigured"}


def _locked() -> set[str]:
    return set(pd.read_csv(DCW / "official/locked_v2.csv").DeviceId.astype(str).str.lower())


def groups_of(arm: str) -> list[str]:
    if arm in ("base", "repro"):
        return []
    g = arm[4:].split("+") if arm.startswith("all:") else [arm]
    assert set(g) <= set(GROUPS), g
    return g


def build_tables(arm: str) -> tuple[Path, Path, dict]:
    """Modified label table + Dec-role table for an arm (labels as originally given for the restored rows)."""
    lk = _locked()
    L = pd.read_parquet(V4L)
    if "g89_restored" in L:            # v4n source: start from its TRAINING values
        for c in TRAIN_COLS:
            L[c] = L[f"{c}_train"]
    L["DeviceId_orig"] = L.DeviceId
    L["DeviceId"] = L.DeviceId.str.lower()
    assert not L.DeviceId.isin(lk).any()
    L["g89_restored"] = (L.g89_restored.astype("string") if "g89_restored" in L
                         else pd.Series(pd.NA, index=L.index, dtype="string"))
    D = pd.read_parquet(DEC4L)
    D["_id"] = D.DeviceId.str.lower()
    I = pd.read_parquet(ITEMS)
    key = lambda df, d="detector": list(zip(df.DeviceId.str.lower(), df[d].astype(int)))
    Lk = pd.Series(key(L), index=L.index)
    info = {}

    def members(k):
        return set(key(I[I.key == k], "Detector"))
    for g in groups_of(arm):
        if g in VALKEY:
            m = Lk.isin(members(g)) & L.validated.eq(VALKEY[g]).fillna(False)
            L.loc[m, "validated"] = "pass"
        elif g == "not_checkable":     # round 2: no behaviour check possible -> trained like a pass
            m = L.validated.eq("not_checkable").fillna(False) & L.function.notna()
            L.loc[m, "validated"] = "pass"
        elif g == "dq_check":          # every dq_suspect row left after note 88's fault-free recompute
            m = L.dq_suspect.eq(True).fillna(False)
            L.loc[m, "dq_suspect"] = False
        elif g == "dec_role":
            mem = members(g)
            m = pd.Series(list(zip(D._id, D.detector.astype(int))), index=D.index).isin(mem)
            D.loc[m, "dec_role_changed_train"] = False
            D.loc[m, "dec_status_train"] = "same_restored"
            info["dec_rows_cleared"] = int(m.sum())
            L.loc[Lk.isin(mem), "g89_restored"] = L.loc[Lk.isin(mem), "g89_restored"].fillna("") + "dec_role;"
            info[g] = int(m.sum())
            continue
        elif g == "beh_tied_low":
            m = L.rule_v4l.eq("R5_behaviour_tied_low").fillna(False)
            L.loc[m, ["exclude_train_score", "train_use"]] = [False, True]
        elif g == "unusual_signal":
            m = L.unusual_layout.eq(True).fillna(False) & L.function.notna()
            L.loc[m, ["unusual_layout", "train_use"]] = [False, True]
        elif g == "small":
            m1 = L.rule_v4l.eq("R6_noise_input").fillna(False)
            L.loc[m1, ["exclude_train_score", "train_use"]] = [False, True]
            C = pd.read_csv(CHANGES, dtype={"DeviceName": str})
            ck = {(a, int(b)): (t, v) for a, b, t, v in zip(C.DeviceName, C.detector, C.train_label_v3s, C.validated_v3s)}
            nm = L.DeviceName.astype(str)
            m2 = pd.Series(False, index=L.index)
            for r in I[I.key.isin(["expl_sweep", "reread_flip"])].itertuples():
                i = L.index[(L.DeviceId == r.DeviceId) & (L.detector.astype(int) == r.Detector)]
                if len(i) == 0 or L.at[i[0], "source"] == "user_ruling":
                    continue
                t, v = ck.get((nm.at[i[0]], int(r.Detector)), (None, None))
                if not isinstance(t, str) or t == L.at[i[0], "label_print_first"]:
                    continue      # only the training label is restored (a truth-only flip changes nothing here)
                L.loc[i[0], ["label_print_first", "train_use", "exclude_train_score"]] = [t, True, False]
                if pd.isna(L.at[i[0], "validated"]) and isinstance(v, str):
                    L.loc[i[0], "validated"] = v
                if pd.isna(L.at[i[0], "tier"]):
                    L.loc[i[0], "tier"] = "incomplete"
                m2.at[i[0]] = True
            m = m1 | m2
        info[g] = int(m.sum())
        L.loc[m, "g89_restored"] = L.loc[m, "g89_restored"].fillna("") + g + ";"
    d = X89 / "tables"
    d.mkdir(parents=True, exist_ok=True)
    tag = arm.replace(":", "_").replace("+", "-")
    lp, dp = d / f"labels_{SRC}_{tag}.parquet", d / f"dec_{SRC}_{tag}.parquet"
    for c in TRAIN_COLS:               # v3_retrain reads <col>_train whenever g89_restored exists
        L[f"{c}_train"] = L[c]
    L.to_parquet(lp, index=False)
    D.drop(columns="_id").to_parquet(dp, index=False)
    return lp, dp, info


def stage_labels(a):
    """Final training labels = the source table (G89_SRC) with the RESTORED groups' training values in <col>_train
    (plain columns keep the source's scoring-side values, so every scoring set is unchanged); v3_retrain trains on the
    _train values whenever g89_restored exists. -> research/labels/function_labels_<out>.parquet + %DC_WORK%/x89/
    dec_role_changed_<out>.parquet. Then the training-row census and the 20-row sanity sample of what stays excluded."""
    rest = [g for g in a.restore.split(",") if g]
    arm = ("all:" + "+".join(rest)) if len(rest) > 1 else (rest[0] if rest else "base")
    lp, dp, info = build_tables(arm)
    L, M = pd.read_parquet(lp), pd.read_parquet(V4L)
    O = M.copy()
    for c in TRAIN_COLS:
        O[f"{c}_train"] = L[f"{c}_train"].to_numpy()
    O["g89_restored"] = L.g89_restored.to_numpy()
    O["label_version"] = a.out
    assert (O.truth_v3s.astype("string").fillna("-") == M.truth_v3s.astype("string").fillna("-")).all(), "truth moved"
    out = REPO / f"research/labels/function_labels_{a.out}.parquet"
    O.to_parquet(out, index=False)
    dout = X89 / f"dec_role_changed_{a.out}.parquet"
    pd.read_parquet(dp).to_parquet(dout, index=False)
    L["DeviceId"] = L.pop("DeviceId_orig")
    # census: labelled detectors trained on (any frame row) under the recipe, source vs output
    import f76_function as F  # noqa: F401
    import t57_function as T57F
    import v3_retrain as V
    res = {"source": SRC, "out": a.out, "restored": rest, "info": info}
    for tag, (l, d) in {SRC: (V4L, DEC4L), a.out: (out, dout)}.items():
        V.LABEL_SETS["v3s"] = (l, d)
        T57F.setup("v6e")
        fr, _ = V.load_feats()
        lab = V.load_labels(fr, "exclude")
        y, ok = V.variant_target(lab, fr, T57F.VAR)
        k = lab[["DeviceId", "Detector"]].assign(ok=ok, lab=pd.notna(lab.label_print_first.to_numpy(object)))
        g = k.groupby(["DeviceId", "Detector"]).agg(ok=("ok", "any"), lab=("lab", "any"))
        Lt = pd.read_parquet(l)
        lpf = Lt["label_print_first_train"] if "label_print_first_train" in Lt else Lt.label_print_first
        res[tag] = dict(labelled_detectors=int(lpf.notna().sum()),
                        labelled_with_frame_rows=int(g.lab.sum()), trained=int(g.ok.sum()),
                        excluded_with_data=int((g.lab & ~g.ok).sum()), train_rows=int(ok.sum()))
    # what stays excluded (v4n): labelled detectors with frame rows but no training row, by first blocking reason;
    # sanity sample = 20 random ones of them, the user's own rulings (R2 / R4 / user_ruling) left out
    Ln = L.copy()
    Ln["DeviceId"] = Ln.DeviceId.str.lower()
    Ln = Ln.rename(columns={"detector": "Detector"}).astype({"Detector": g.index.get_level_values(1).dtype})
    gx = g[g.lab & ~g.ok].reset_index().merge(Ln, on=["DeviceId", "Detector"], how="left")
    b = lambda c: gx[c].eq(True).fillna(False)
    rule = gx.rule_v4l.astype("string").fillna("")
    vnc = gx.validated.astype("string").fillna("none")
    xu = gx.DeviceName.astype(str).str.upper().isin(V.EXTRA_UNUSUAL) & ~gx.readmit_radar_over_loops.eq(True).fillna(False)
    gx["reason"] = np.select(
        [xu & ~gx.g89_restored.astype("string").fillna("").str.contains("unusual_signal"),
         rule.str.startswith(("R2", "R4")) | gx.source.eq("user_ruling") | gx.user_review.notna(),
         rule.str.startswith("R1"), b("yr_count_identical"), gx.pending_user.notna() | b("exclude_train_score"),
         b("dead"), b("unusual_layout") | ~b("train_use"), vnc.eq("not_checkable"), vnc.isin(["no_data", "none"]),
         ~vnc.eq("pass")],
        ["clean_extra_unusual_signal", "user_ruling", "R1_loop_config_count", "yr_identical_to_count", "excluded_stack_or_rule", "dead",
         "not_train_use", "not_checkable_not_readmitted", "no_validation_data", "validated_" + vnc],
        "window_level_only (min-on / health3 / health_flag)")
    res["excluded_by_reason"] = gx.reason.value_counts().to_dict()
    keepc = ["DeviceId", "Detector", "DeviceName", "function", "label_print_first", "phase_target", "technology", "source",
             "print_confidence", "validated", "failed_checks", "validation_reason", "dead", "n_on_new", "reason"]
    gx[[c for c in keepc if c in gx]].to_parquet(X89 / f"excluded_{a.out}.parquet", index=False)
    pool = gx[gx.reason != "user_ruling"].copy()
    pool["DeviceName"] = pool.DeviceName.where(pool.DeviceName.notna(), pool.DeviceId.str[:8])
    smp = pool.sample(20, random_state=89)[["DeviceName", "Detector", "function", "label_print_first", "reason",
                                             "source", "validated"]]
    res["sample_pool"] = int(len(pool))
    res["sample"] = smp.to_dict("records")
    json.dump(res, open(X89 / f"labels_{a.out}.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in res.items() if k != "sample"}, indent=1, default=str))
    print(smp.to_string())


def arm_dir(arm: str) -> Path:
    return X89 / "trees" / arm.replace(":", "_").replace("+", "-")


def _patch(arm: str):
    """v3_retrain / t57 / f76 pointed at the arm's tables and folder (one process per arm)."""
    assert os.environ.get("F76_ARM") == "c"
    lp, dp, info = build_tables(arm)
    import f76_function as F      # noqa: F401  (patches V.set_frame -> frame_v6e_c, resets T57F.OUT)
    import t57_function as T57F
    import v3_retrain as V
    V.LABEL_SETS["v3s"] = (lp, dp)
    if "unusual_signal" in groups_of(arm):
        L = pd.read_parquet(V4L, columns=["DeviceName", "unusual_layout"])
        u = set(L.loc[L.unusual_layout.eq(True).fillna(False), "DeviceName"].astype(str).str.upper())
        V.EXTRA_UNUSUAL = V.EXTRA_UNUSUAL - u      # --clean re-flags these; restored with the group
    T57F.OUT = arm_dir(arm)
    return T57F, F, V, info


def stage_fit(a):
    T57F, F, V, info = _patch(a.arm)
    T57F.setup("v6e")
    fr, cols = V.load_feats()
    lab = V.load_labels(fr, "exclude")
    y, ok = V.variant_target(lab, fr, T57F.VAR)
    info.update(train_rows=int(ok.sum()), train_detectors=int(lab.loc[ok, ["DeviceId", "Detector"]].drop_duplicates().shape[0]),
                classes=pd.Series(y[ok]).value_counts().to_dict())
    print(a.arm, json.dumps(info, default=str), flush=True)
    d = arm_dir(a.arm)
    d.mkdir(parents=True, exist_ok=True)
    json.dump(info, open(d / "info.json", "w"), indent=1, default=str)
    if a.info_only:
        return
    T57F.stage_fit(argparse.Namespace(frame="v6e", cfg=F.CFG, seeds=a.seeds, threads=a.threads))
    old = json.load(open(DCW / "trees57" / "function" / "v6e" / "drop_pp_xcand-pp_pdiff-pp_v2-yr-ratio-phctx" /
                         "cols.json"))["cols"]
    assert json.load(open(F.new_dir() / "cols.json"))["cols"] == old, "feature list differs from the 229-feature arm"


def _setup_stack(arm: str):
    T57F, F, V, _ = _patch(arm)
    import s59_step6 as S59
    orig = S59.spec_probs

    def spec_probs(spec, fr):          # seed-0 screen: single-seed trees accepted (same for every arm)
        if SFX == "":
            spec = spec + "@s0"          # the screen always uses seed 0, even once more seeds exist
        P, seeds = orig(spec, fr)
        return P, ([0, 1, 2] if seeds in ([0],) else seeds)
    S59.spec_probs = spec_probs
    import of77
    s74, S, E = of77.setup_new()
    assert str(S59.FUNC_DIR).startswith(str(arm_dir(arm))), S59.FUNC_DIR
    assert not E["fr"].DeviceId.str.lower().isin(_locked()).any()
    s74.OUT = X89 / ("stack" + SFX)
    return s74, S, E, of77


def stage_stack(a):
    s74, S, E, _ = _setup_stack(a.arm)
    s74.cmd_stack(argparse.Namespace(name=a.arm.replace(":", "_").replace("+", "-"), main="x69_siba", extra=""))


def stage_score(a):
    s74, S, E, of77 = _setup_stack(a.arm)
    fr = E["fr"]
    lc = of77.lane_ctx(fr, of77.O8 / "lanes_D.func.parquet")[:, 2]
    d = X89 / ("stack" + SFX) / a.arm.replace(":", "_").replace("+", "-")
    Ps = [np.load(d / f"stack_ctx_s{s}.npy").astype(float) for s in (0, 1, 2)]
    ok = s74.gate_ok(E, np.mean(Ps, 0), lc)
    np.savez_compressed(X89 / f"ok{SFX}_{a.arm.replace(':', '_').replace('+', '-')}.npz", E=ok["E"], R=ok["R"],
                        sig=fr.DeviceId.to_numpy(str), wgroup=fr.wgroup.to_numpy(str), det=fr.Detector.to_numpy(int),
                        win=fr.win.to_numpy(str), period=fr.period.to_numpy(str),
                        truth=fr.truth_v3s.astype("string").fillna("").to_numpy(str))
    m = fr.wgroup.isin(GE30).to_numpy()
    print(a.arm, {s: round(float(np.nanmean(ok[s][m])), 4) for s in ("E", "R")}, flush=True)


def stage_report(a):
    import s80
    arms = [p.stem[3 + len(SFX):] for p in sorted(X89.glob(f"ok{SFX}_*.npz"))]
    Z = {k: np.load(X89 / f"ok{SFX}_{k}.npz") for k in arms}
    base = Z[a.base]
    sig, wg = base["sig"], base["wgroup"]
    pools = {"ge30": np.isin(wg, GE30), "m5": wg == "m5", "m10": wg == "m10"}
    res = {"base": a.base}
    for k, z in Z.items():
        assert (z["sig"] == sig).all() and (z["win"] == base["win"]).all() and (z["det"] == base["det"]).all()
        assert (z["truth"] == base["truth"]).all(), "scoring truth must be identical across arms"
        r = {f"{s}_{p}": s80._acc_ci(np.where(pm, z[s], np.nan), sig) for s in ("E", "R") for p, pm in pools.items()}
        if k != a.base:
            for s in ("E", "R"):
                for p, pm in pools.items():
                    r[f"d{s}_{p}"] = s80._delta(np.where(pm, base[s], np.nan), np.where(pm, z[s], np.nan), sig, True)
        info = arm_dir(k.replace("_", ":", 1) if k.startswith("all_") else k) / "info.json"
        if not info.exists():
            info = X89 / "trees" / k / "info.json"
        if info.exists():
            r["info"] = json.load(open(info))
        res[k] = r
        print(k, r["E_ge30"]["acc"], r["R_ge30"]["acc"], r.get("dE_ge30"), r.get("dR_ge30"), r.get("dE_m5"), flush=True)
    json.dump(res, open(X89 / f"g89_{a.tag}.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fit", "stack", "score", "report", "labels"])
    ap.add_argument("--restore", default="")
    ap.add_argument("--out", default="v4n")
    ap.add_argument("--arm", default="base")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--info-only", action="store_true", dest="info_only")
    ap.add_argument("--base", default="base")
    ap.add_argument("--tag", default="s0")
    a = ap.parse_args()
    {"fit": stage_fit, "stack": stage_stack, "score": stage_score, "report": stage_report,
     "labels": stage_labels}[a.stage](a)
