"""Note 54: per-lane ATSPM decode + ATSPM-only, stack-aware function scoring (user rule 2026-09-30, AGENTS.md
"ATSPM classes, one per lane, and scoring"). Re-scores an existing six-fold OOF; trains nothing.

    python atspm_score.py                                   # function_v3e OOF (frame v6e, run ..._h3, 3 seeds)
    python atspm_score.py --run <run dir name> --tag v3s    # after the v3s retrain (same frame)

-> %DC_WORK%/trackA/atspm54/<tag>/{rows.parquet, score.json, score.txt}

Rows: every frame detector-window with >= 5 actuations takes part in the decode (labelled or not); scoring rows are
note 49's sets (everything = honest truth, >= 5 actuations; realistic = minus validated fail / misconfigured).
Decode (lanes/atspm_decode.py): groups = (signal, period, window, PREDICTED phase); lanes = note-42 OOF lane output
(ln5_extend.py: every signal, both periods, all 14 windows; its six fold pair models reproduce note 42's OOF exactly);
note 42's own ln3 decode (Sept m30 / h6 / h24 / full, 474 print-lane signals) is kept as a comparison ("lane" scope). A detector's lanes are used only when the lane decode put it on the same
predicted phase; otherwise (and for < 10 actuations, Bike) it is unconstrained. Scoring:
  acc7          old 7-class accuracy (continuity with notes 45 / 49), no stack credit.
  ATSPM score   1 - ATSPM errors / rows; errors = ATSPM truth called non-ATSPM or another ATSPM class, non-ATSPM truth
                called ATSPM. Other / Mid / Bike mix-ups are not errors here (reported as `secondary`).
  stack credit  a stacked group (stack_labels_v3s: same phase, same role, every member's truth = the role): in a window,
                the member with the highest P(role) among those called the role is right; further members called the
                role are wrong (`stack_extra` = the extra lane the stack causes, reported apart); members called non-ATSPM
                are right when some member got the role; if none did, the member with the highest P(role) is a miss.
locked_v2 asserted absent.
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
import atspm_decode as AD  # noqa: E402

RUN = "run_ad6ea6959f_exclude_min5_clean_valnc_h3"      # function_v3e (note 49)
LANES = V.DCW / "lanes" / "ln3_grid_det.parquet"
LANE_CFG = 6                                           # grid index of (lam 3, beta 0), picked on all six folds
V3S = rpath.LABELS_CURRENT                              # note 81: v4l is the scoring truth (was function_labels_v3s)
OUTD = V.DCW / "trackA" / "atspm54"
C7 = np.array(V.C7, object)
ATS = {"Advance", "Presence", "Count", "Yellow_Red"}
LANE_WG = ["m30", "h6", "h24", "full"]
WG = ["m5", "m10", "m30", "h1", "h3", "h6", "h24", "full"]
# (mode, span, lane column). lanes5g = lanes5 with the 5- and 10-min windows blanked: the pair model was trained on
# >= 30-min windows only, and on 5 / 10 min the decode costs 4 / 1.3 pt (first run of this script) -> gate at 30 min.
CONFIGS = {"greedy_strict": ("greedy", "strict", "lanes5g"), "nonatspm_strict": ("nonatspm", "strict", "lanes5g"),
           "exact_strict": ("exact", "strict", "lanes5g"), "greedy_single": ("greedy", "single", "lanes5g"),
           "greedy_strict_ungated": ("greedy", "strict", "lanes5")}
PRIMARY = "greedy_strict"                               # the user's "most probable one gets it", gated at 30 min
# note 56 (orchestrator): the stack-scoped pick rule is the default decode. decode(..., pick=True) uses the hi-res
# inputs attached by attach_pick_inputs(); main() makes it PRIMARY when the input files exist (--no-pick = note 54).
PICK_IN = ("ln6_pick_v3s", "ln7_stackhealth_v3s")
CHI_MIN, CHI_X = 10.0, 3.0                              # note 56 stack-relative health flag


def attach_pick_inputs(fr: pd.DataFrame, ln6: str = PICK_IN[0], ln7: str = PICK_IN[1]) -> pd.DataFrame:
    """pick inputs (hi-res only): span / span_peers / coloc_peers / track from ln6_pick.py (note 55), the health flag
    from ln7_stackhealth.py (note 56: member's chi >= CHI_MIN and >= CHI_X x (best partner's chi + 0.5))."""
    k6 = ["DeviceId", "Detector", "period", "win"]
    K = pd.read_parquet(V.DCW / "lanes" / f"{ln6}.parquet").astype({"Detector": fr.Detector.dtype})
    H = pd.read_parquet(V.DCW / "lanes" / f"{ln7}.parquet").astype({"Detector": fr.Detector.dtype})
    lk = V.locked_signals()
    assert not K.DeviceId.isin(lk).any() and not H.DeviceId.isin(lk).any()
    g = H.groupby(["DeviceId", "period", "win", "clus"]).chi
    lo1 = g.transform("min")
    n_min = H.chi.eq(lo1) & H.chi.notna()
    second = g.transform(lambda v: np.sort(v.dropna().to_numpy())[1] if v.notna().sum() > 1 else np.nan)
    H["chi_p"] = np.where(n_min, second, lo1)              # best partner = the smallest chi among the OTHER members
    H["unh2"] = (H.chi >= CHI_MIN) & (H.chi >= CHI_X * (H.chi_p + 0.5))
    x = fr[k6].merge(K[k6 + ["span", "span_peers", "coloc_peers", "track"]], on=k6, how="left")
    y = fr[k6].merge(H[k6 + ["unh2"]], on=k6, how="left")
    assert len(x) == len(fr) == len(y)
    fr["pk_unhealthy"] = y.unh2.eq(True).to_numpy()
    fr["pk_span"] = x.span.eq(True).to_numpy()
    fr["pk_span_peers"] = x.span_peers.fillna("").to_numpy()
    fr["pk_coloc_peers"] = x.coloc_peers.fillna("").to_numpy()
    fr["pk_track"] = x.track.to_numpy(float)
    return fr


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load(run: str) -> pd.DataFrame:
    V.set_frame("v6e")
    fr = pd.read_parquet(V.FEATS, columns=V.KEY + ["wgroup", "fold", "pred_phase", "det_n_on"])
    fr["DeviceId"] = fr.DeviceId.str.lower()
    lk = V.locked_signals()
    assert not fr.DeviceId.isin(lk).any()
    Ps = []
    for s in (0, 1, 2):
        P, hv, got = V.load_oof(fr, V.OUT / run, "first.all.wi", s, range(6))
        assert len(got) == 6 and hv.all()
        Ps.append(P)
    P = np.mean(Ps, 0)
    for i, c in enumerate(V.C7):
        fr[f"P_{c}"] = P[:, i]
    lab = pd.read_parquet(V3S)
    lab["DeviceId"] = lab.DeviceId.str.lower()
    assert not lab.DeviceId.isin(lk).any()
    lab = lab.rename(columns={"detector": "Detector"}).astype({"Detector": fr.Detector.dtype})
    cols = ["truth_v3", "truth_v3s", "validated", "stack_group", "stack_role", "exclude_train_score",
            "yr_count_identical", "dec_fp_truth", "unusual_layout_v3", "readmit_radar_over_loops", "stack_relabel"]
    fr = fr.merge(lab[["DeviceId", "Detector"] + cols], on=["DeviceId", "Detector"], how="left")
    # note 52's full Dec-role rule (function / phase / absent), for the comparison row
    d = pd.read_parquet(V.DCW / "cabinet" / "dec_role_changed.parquet",
                        columns=["DeviceId", "detector", "dec_role_changed_truth"])
    d = d.assign(DeviceId=d.DeviceId.str.lower(), Detector=d.detector.astype(fr.Detector.dtype))
    fr = fr.merge(d[["DeviceId", "Detector", "dec_role_changed_truth"]], on=["DeviceId", "Detector"], how="left")
    for c in ("exclude_train_score", "yr_count_identical", "dec_fp_truth", "dec_role_changed_truth", "stack_relabel",
              "readmit_radar_over_loops"):
        fr[c] = fr[c].eq(True)
    fr["dec_fp_truth"] &= fr.period.eq("dec")
    fr["dec_all_truth"] = fr.dec_role_changed_truth & fr.period.eq("dec")
    y = fr.truth_v3.to_numpy(object)
    fr["setA"] = pd.notna(y) & np.isin(y, V.C7) & (fr.det_n_on >= 5).to_numpy()
    fr["setR"] = fr.setA & ~fr.validated.isin(["fail", "misconfigured"])
    y2 = fr.truth_v3s.to_numpy(object)
    fr["setA_s"] = pd.notna(y2) & np.isin(y2, V.C7) & (fr.det_n_on >= 5).to_numpy() & ~fr.exclude_train_score
    fr["setR_s"] = fr.setA_s & ~fr.validated.isin(["fail", "misconfigured"])
    return fr


def attach_lanes(fr: pd.DataFrame, ln5: str) -> pd.DataFrame:
    """lanes5 = ln5 (every signal, both periods, all 14 windows; fold models); lanes_ln3 = note 42's own OOF decode
    (print-lane signals, Sept m30 / h6 / h24 / full) for comparison; lane_sig marks ln3's scope."""
    L5 = pd.read_parquet(V.DCW / "lanes" / f"{ln5}.parquet", columns=["DeviceId", "Detector", "period", "win", "phase", "lanes"])
    L5 = L5[L5.lanes != ""].astype({"Detector": fr.Detector.dtype})
    x = fr[["DeviceId", "Detector", "period", "win"]].merge(L5, on=["DeviceId", "Detector", "period", "win"], how="left")
    assert len(x) == len(fr)
    fr["lanes5"] = x.lanes.where(x.phase.eq(fr.pred_phase.to_numpy()), None).to_numpy()
    fr["lanes5g"] = fr.lanes5.where(~fr.wgroup.isin(["m5", "m10"]), None)
    L = pd.read_parquet(LANES, columns=["DeviceId", "Detector", "phase", "lanes", "win", "cfg"])
    L = L[(L.cfg == LANE_CFG) & (L.lanes != "")].drop(columns="cfg")
    L["DeviceId"] = L.DeviceId.str.lower()
    L["Detector"] = L.Detector.astype(fr.Detector.dtype)
    L = L.rename(columns={"phase": "lane_phase"})
    x = fr[["DeviceId", "Detector", "period", "win"]].merge(L.assign(period="stg"), on=["DeviceId", "Detector", "period", "win"],
                                                              how="left")
    fr["lanes_ln3"] = x.lanes.where(x.lane_phase.eq(fr.pred_phase.to_numpy()), None).to_numpy()
    fr["lane_sig"] = fr.DeviceId.isin(set(L.DeviceId)) & fr.period.eq("stg") & fr.wgroup.isin(LANE_WG)
    return fr


def decode(fr: pd.DataFrame, lane_col: str, mode: str, span: str, pick: bool = False) -> np.ndarray:
    P = fr[[f"P_{c}" for c in V.C7]].to_numpy()
    pred = P.argmax(1)
    sub = fr.loc[(fr.det_n_on >= 5), ["DeviceId", "period", "win", "pred_phase", lane_col]]
    has = sub[lane_col].notna().groupby([sub.DeviceId, sub.period, sub.win, sub.pred_phase]).transform("sum")
    sub = sub[has.to_numpy() >= 2]
    lanes_all = fr[lane_col].to_numpy(object)
    changed = 0
    for _, g in sub.groupby(["DeviceId", "period", "win", "pred_phase"]).indices.items():
        ix = sub.index.to_numpy()[g]
        ls = [frozenset(int(v) for v in s.split(",")) if isinstance(s, str) and s else frozenset() for s in lanes_all[ix]]
        pk = None
        if pick:
            pk = AD.stack_pick(fr.Detector.to_numpy()[ix], ls, fr.pk_unhealthy.to_numpy()[ix], fr.pk_span.to_numpy()[ix],
                               fr.pk_span_peers.to_numpy(object)[ix], fr.pk_coloc_peers.to_numpy(object)[ix],
                               fr.pk_track.to_numpy(float)[ix])
        out = AD.decode_group(P[ix], ls, mode, span, pk)
        changed += int((out != pred[ix]).sum())
        pred[ix] = out
    log(f"decode {lane_col} {mode}/{span}: {changed:,} detector-windows changed")
    return pred


def credit(fr: pd.DataFrame, pred: np.ndarray, truth: str, rows: np.ndarray, stack: bool) -> pd.DataFrame:
    """Per scoring row: ok7, ok_atspm, err type, stack tags."""
    t = fr[truth].to_numpy(object)[rows]
    p = C7[pred[rows]]
    ta, pa = np.isin(t, list(ATS)), np.isin(p, list(ATS))
    ok_a = np.where(ta, p == t, ~pa)
    err = np.where(ok_a, "", np.where(ta & pa, "A->wrongA", np.where(ta, "A->nonA", "nonA->A")))
    d = pd.DataFrame({"DeviceId": fr.DeviceId.to_numpy()[rows], "period": fr.period.to_numpy()[rows],
                      "win": fr.win.to_numpy()[rows], "wgroup": fr.wgroup.to_numpy()[rows], "t": t, "p": p,
                      "ok7": p == t, "ok_a": ok_a, "err": err, "stack_tag": ""})
    if not stack:
        return d
    sg = fr.stack_group.to_numpy(object)[rows]
    role = fr.stack_role.to_numpy(object)[rows]
    # a group earns stack credit only if EVERY member's truth (label table, this truth column) is the role
    g_ok = fr[fr.stack_group.notna()].groupby("stack_group").apply(
        lambda g: bool((g[truth] == g.stack_role).all()), include_groups=False)
    valid = set(g_ok.index[g_ok])
    P = fr[[f"P_{c}" for c in V.C7]].to_numpy()[rows]
    m = pd.notna(sg) & np.isin(sg, list(valid))
    if not m.any():
        return d
    idx = np.flatnonzero(m)
    k = pd.DataFrame({"i": idx, "g": sg[idx], "w": d.period.to_numpy()[idx] + "|" + d.win.to_numpy()[idx]})
    for (_, _), gg in k.groupby(["g", "w"]):
        if len(gg) < 2:
            continue
        ii = gg.i.to_numpy()
        r = role[ii[0]]
        ci = V.C7.index(r)
        pr = P[ii, ci]
        called = p[ii] == r
        nona = ~np.isin(p[ii], list(ATS))
        if called.any():
            win_i = ii[called][np.argmax(pr[called])]
            for j in ii[called]:
                if j != win_i:
                    d.loc[j, ["ok_a", "err", "stack_tag"]] = [False, "stack_extra", "extra"]
                else:
                    d.loc[j, ["ok_a", "err"]] = [True, ""]
            for j in ii[nona]:
                d.loc[j, ["ok_a", "err", "stack_tag"]] = [True, "", "credited"]
        elif nona.any():
            miss = ii[nona][np.argmax(pr[nona])]
            for j in ii[nona]:
                if j == miss:
                    d.loc[j, ["ok_a", "err", "stack_tag"]] = [False, "A->nonA", "miss"]
                else:
                    d.loc[j, ["ok_a", "err", "stack_tag"]] = [True, "", "credited"]
    return d


def summ(d: pd.DataFrame) -> dict:
    if not len(d):
        return {}
    na = ~np.isin(d.t, list(ATS)) & ~np.isin(d.p, list(ATS))
    e = d.err.value_counts().to_dict()
    return {"n": int(len(d)), "signals": int(d.DeviceId.nunique()), "acc7": round(float(d.ok7.mean()), 4),
            "atspm": round(float(d.ok_a.mean()), 4),
            "err": {k: int(v) for k, v in e.items() if k},
            "stack_extra": int((d.stack_tag == "extra").sum()), "stack_credited": int((d.stack_tag == "credited").sum()),
            "atspm_rows": int(np.isin(d.t, list(ATS)).sum()),
            "secondary_acc": round(float((d.t[na] == d.p[na]).mean()), 4) if na.any() else None,
            "secondary_n": int(na.sum())}


def boot(d0: pd.DataFrame, d1: pd.DataFrame, col: str = "ok_a", reps: int = 1000, seed: int = 54) -> list:
    """95 % signal-bootstrap CI of the difference in mean `col` (d1 - d0), both over their own rows."""
    a = d0.groupby("DeviceId")[col].agg(["sum", "size"])
    b = d1.groupby("DeviceId")[col].agg(["sum", "size"])
    s = a.index.union(b.index)
    a, b = a.reindex(s, fill_value=0).to_numpy(float), b.reindex(s, fill_value=0).to_numpy(float)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(s), np.full(len(s), 1 / len(s)), size=reps).astype(float)
    diff = (w @ b[:, 0]) / (w @ b[:, 1]) - (w @ a[:, 0]) / (w @ a[:, 1])
    return [round(float(np.percentile(diff, 2.5)) * 100, 2), round(float(np.percentile(diff, 97.5)) * 100, 2)]


def main(run: str, tag: str, ln5: str, base: str | None, base_ln5: str | None, use_pick: bool = True):
    global PRIMARY
    out = OUTD / tag
    out.mkdir(parents=True, exist_ok=True)
    fr = attach_lanes(load(run), ln5)
    if use_pick and all((V.DCW / "lanes" / f"{f}.parquet").exists() for f in PICK_IN):
        fr = attach_pick_inputs(fr)
        CONFIGS["pick_stack"] = ("greedy", "strict", "lanes5g")
        PRIMARY = "pick_stack"                           # note 56 default
    Pm = fr[[f"P_{c}" for c in V.C7]].to_numpy()
    argmax = Pm.argmax(1)
    preds = {"argmax": argmax}
    for name, (mode, span, col) in CONFIGS.items():
        preds[name] = decode(fr, col, mode, span, name == "pick_stack")
    preds["ln3_" + PRIMARY] = decode(fr, "lanes_ln3", *CONFIGS[PRIMARY][:2], PRIMARY == "pick_stack")
    res = {"run": run, "rows_frame": int(len(fr))}
    lines = []
    # steps of the waterfall: (name, pred, truth column, set columns, stack credit, extra exclusion)
    steps = [("0 argmax, v3 truth (note 49)", "argmax", "truth_v3", ("setA", "setR"), False, None),
             ("1 + per-lane decode", PRIMARY, "truth_v3", ("setA", "setR"), False, None),
             ("2 + stack credit (v3 truth)", PRIMARY, "truth_v3", ("setA", "setR"), True, None),
             ("3 + v3s truth (stack relabel, YR==Count out)", PRIMARY, "truth_v3s", ("setA_s", "setR_s"), True, None),
             ("4 + Dec-role filter, function/phase rule", PRIMARY, "truth_v3s", ("setA_s", "setR_s"), True, "dec_fp_truth"),
             ("3a v3s truth, argmax (rule without decode)", "argmax", "truth_v3s", ("setA_s", "setR_s"), True, None),
             ("4a v3s + Dec fp, argmax", "argmax", "truth_v3s", ("setA_s", "setR_s"), True, "dec_fp_truth"),
             ("4b v3s + Dec full rule (note 52)", PRIMARY, "truth_v3s", ("setA_s", "setR_s"), True, "dec_all_truth")]
    D = {}
    for scope in ("all", "lane"):
        for name, pk, tcol, sets, stack, excl in steps:
            for sname, scol in zip(("everything", "realistic"), sets):
                m = fr[scol].to_numpy().copy()
                if excl:
                    m &= ~fr[excl].to_numpy()
                if scope == "lane":
                    m &= fr.lane_sig.to_numpy()
                rows = np.flatnonzero(m)
                d = credit(fr, preds[pk], tcol, rows, stack)
                D[(scope, name, sname)] = d
                s = summ(d)
                s["by_window"] = {g: summ(d[d.wgroup == g]) for g in WG if (d.wgroup == g).any()}
                res.setdefault(scope, {}).setdefault(name, {})[sname] = s
    # decode variants on the step-4 set; note-42 lanes (ln3) vs ln5 lanes on ln3's scope
    var = {}
    for sname, scol in (("everything", "setA_s"), ("realistic", "setR_s")):
        m = fr[scol].to_numpy() & ~fr.dec_fp_truth.to_numpy()
        for pk in ["argmax"] + list(CONFIGS):
            for scope, sm in (("lane", fr.lane_sig.to_numpy()), ("all", np.ones(len(fr), bool))):
                d = credit(fr, preds[pk], "truth_v3s", np.flatnonzero(m & sm), True)
                var[f"{scope}|{sname}|{pk}"] = summ(d)
        for pk in ("ln3_" + PRIMARY,):
            d = credit(fr, preds[pk], "truth_v3s", np.flatnonzero(m & fr.lane_sig.to_numpy()), True)
            var[f"lane|{sname}|{pk}"] = summ(d)
    res["variants"] = var
    # CIs: decode effect (1 - 0), stack credit (2 - 1), whole rule on v3s truth (4 - 4a), each scope / set
    ci = {}
    for scope in ("all", "lane"):
        for sname in ("everything", "realistic"):
            g = lambda n: D[(scope, n, sname)]  # noqa: E731
            ci[f"{scope}|{sname}"] = {
                "decode (1-0)": boot(g(steps[0][0]), g(steps[1][0])),
                "stack credit (2-1)": boot(g(steps[1][0]), g(steps[2][0])),
                "decode on v3s+Dec fp (4-4a)": boot(g(steps[6][0]), g(steps[4][0])),
                "acc7 decode (1-0)": boot(g(steps[0][0]), g(steps[1][0]), "ok7")}
    res["ci_pt"] = ci
    # which decisions changed (lane scope, step-4 set): changed rows by truth / old / new class
    m = fr.setA_s.to_numpy() & ~fr.dec_fp_truth.to_numpy() & fr.lane_sig.to_numpy()
    ch = m & (preds[PRIMARY] != argmax)
    tt = fr.truth_v3s.to_numpy(object)[ch]
    res["changed_lane_everything"] = {
        "n": int(ch.sum()), "atspm_fixed": int(sum((C7[preds[PRIMARY][ch]] == tt) | (~np.isin(C7[preds[PRIMARY][ch]], list(ATS)) & ~np.isin(tt, list(ATS))))),
        "by_move": pd.Series([f"{a}->{b} (truth {t})" for a, b, t in zip(C7[argmax[ch]], C7[preds[PRIMARY][ch]], tt)])
        .value_counts().head(15).to_dict()}
    # the rule's effect per window, everything, argmax vs decode on the step-4 set (v3s truth, Dec fp, stack credit)
    m = fr.setA_s.to_numpy() & ~fr.dec_fp_truth.to_numpy()
    res["by_window_rule"] = {}
    for g in WG:
        mm = np.flatnonzero(m & fr.wgroup.eq(g).to_numpy())
        d0, d1 = credit(fr, argmax, "truth_v3s", mm, True), credit(fr, preds[PRIMARY], "truth_v3s", mm, True)
        res["by_window_rule"][g] = {"argmax": summ(d0)["atspm"], "decode": summ(d1)["atspm"], "ci_pt": boot(d0, d1)}
    d = D[("all", steps[4][0], "everything")]
    res["errors_by_truth"] = d[~d.ok_a].groupby(["t", "err"]).size().unstack(fill_value=0).to_dict("index")
    if base:
        fb = attach_lanes(load(base), base_ln5 or ln5)
        assert (fb[V.KEY].to_numpy() == fr[V.KEY].to_numpy()).all()
        mode, span, col = CONFIGS[PRIMARY]
        if PRIMARY == "pick_stack":
            fb = attach_pick_inputs(fb)
        pb = decode(fb, col, mode, span, PRIMARY == "pick_stack")
        res["vs_base"] = {"base": base}
        for sname, scol in (("everything", "setA_s"), ("realistic", "setR_s")):
            mm = np.flatnonzero(fr[scol].to_numpy() & ~fr.dec_fp_truth.to_numpy())
            d0, d1 = credit(fb, pb, "truth_v3s", mm, True), credit(fr, preds[PRIMARY], "truth_v3s", mm, True)
            res["vs_base"][sname] = {"base": summ(d0), "new": summ(d1), "ci_atspm_pt": boot(d0, d1),
                                     "ci_acc7_pt": boot(d0, d1, "ok7"),
                                     "by_window": {g: [summ(d0[d0.wgroup == g])["atspm"], summ(d1[d1.wgroup == g])["atspm"]]
                                                   for g in WG}}
    json.dump(res, open(out / "score.json", "w"), indent=1, default=str)
    # text summary
    for scope in ("all", "lane"):
        lines.append(f"== scope {scope} ==")
        lines.append(f"{'step':52s} {'n E':>8s} {'acc7 E':>7s} {'ATSPM E':>8s} {'n R':>8s} {'acc7 R':>7s} {'ATSPM R':>8s}"
                     f" {'extra E':>7s} {'sec E':>6s}")
        for name, *_ in steps:
            e, r = res[scope][name]["everything"], res[scope][name]["realistic"]
            lines.append(f"{name:52s} {e['n']:8d} {e['acc7']:7.4f} {e['atspm']:8.4f} {r['n']:8d} {r['acc7']:7.4f} "
                         f"{r['atspm']:8.4f} {e['stack_extra']:7d} {e['secondary_acc']:6.4f}")
    lines.append("CI (pt): " + json.dumps(ci))
    for k, v in var.items():
        lines.append(f"{k:50s} n {v['n']:7d} acc7 {v['acc7']:.4f} ATSPM {v['atspm']:.4f} err {v['err']} extra {v['stack_extra']}"
                     + (f" byw {v.get('by_window')}" if 'by_window' in v else ""))
    lines.append("changed: " + json.dumps(res["changed_lane_everything"]))
    for scope in ("all", "lane"):
        s = res[scope][steps[4][0]]["everything"]["by_window"]
        s0 = res[scope][steps[0][0]]["everything"]["by_window"]
        lines.append(f"by window {scope}, step 0 -> step 4 ATSPM (everything): "
                     + ", ".join(f"{g} {s0[g]['atspm']:.4f}->{s[g]['atspm']:.4f}" for g in s))
    lines.append("rule by window (step-4 set, everything; argmax -> decode [CI pt]): " + json.dumps(res["by_window_rule"]))
    lines.append("errors by truth (step 4, everything): " + json.dumps(res["errors_by_truth"]))
    if base:
        lines.append("vs base: " + json.dumps(res["vs_base"], default=str))
    open(out / "score.txt", "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--tag", default="v3e")
    ap.add_argument("--lanes", default="ln5_lanes_v3e", help="ln5_extend.py output (lanes from this run's predictions)")
    ap.add_argument("--base", default=None, help="run to compare against on the same rows (e.g. the v3e run)")
    ap.add_argument("--base-lanes", default=None, dest="base_lanes")
    ap.add_argument("--no-pick", action="store_true", dest="no_pick", help="note 54 greedy decode as PRIMARY")
    a = ap.parse_args()
    main(a.run, a.tag, a.lanes, a.base, a.base_lanes, not a.no_pick)
